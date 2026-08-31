"""Build the sealed Scale-0 development dataset.

The dataset is built once and shared by every arm. That is a matching rule, not
an optimisation: "all arms receive the same raw transition IDs, branch groups,
split manifest, actions, outcomes, and order" is one of the eight quantities the
matrix holds fixed, and the only way to be sure of it is to derive the arms'
inputs from one collection and check the digests agree.

Each encoder slot re-runs collection rather than reusing the first slot's
records. It costs a second pass and it buys a real check: if the two passes
produced different transition IDs, something in collection depended on the
encoder, and every cross-encoder comparison downstream would be confounded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sentinel.env.adapters.procedural_visual import ProceduralVisualAdapter  # noqa: E402
from sentinel.env.adapters.synthetic_control import SyntheticControlAdapter  # noqa: E402
from sentinel.wm.cache import LatentCache  # noqa: E402
from sentinel.wm.collect import (  # noqa: E402
    CollectionPlan,
    CollectionResult,
    FeatureTable,
    collect,
)
from sentinel.wm.dataset import (  # noqa: E402
    CollectorPolicy,
    Split,
    SplitManifest,
    TransitionRecord,
    assert_no_transition_overlap,
    assert_trainable,
    audit_splits,
    branch_coverage_report,
    mixture_report,
)
from sentinel.wm.encoder import CachedEncoder, DeterministicControlEncoder  # noqa: E402
from sentinel.wm.latent_contract import ContractViolation  # noqa: E402
from sentinel.wm.resource import ResourceReport, measure  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402

ADAPTERS: Mapping[str, Callable] = {
    "synthetic_control": lambda gate: SyntheticControlAdapter(gate=gate),
    "procedural_visual": lambda gate: ProceduralVisualAdapter(gate=gate),
}

GENERATIVE_FAMILIES = frozenset({"procedural_visual"})
"""Families where complete transition disjointness across splits is achievable.

The controlled family reaches a small finite set of observations, so some
overlap is arithmetic rather than leakage; it is measured instead of forbidden.
"""


@dataclass
class EncoderDataset:
    encoder_slot: str
    records: list[TransitionRecord]
    manifest: SplitManifest
    table: FeatureTable
    cache: LatentCache
    audits: dict[str, Any]
    collection: dict[str, Any]
    resource: ResourceReport

    @property
    def transition_ids_digest(self) -> str:
        return digest_of([r.transition_id for r in self.records])

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "encoder_slot": self.encoder_slot,
            "transitions": len(self.records),
            "transition_ids_digest": self.transition_ids_digest,
            "split_manifest_digest": self.manifest.digest,
            "feature_table": self.table.size_report(),
            "cache": self.cache.size_report(),
            "cache_stats": self.cache.stats.canonical_dict(),
            "audits": self.audits,
            "collection": self.collection,
            "resource": self.resource.canonical_dict(),
        }


def build_plans(config: Mapping[str, Any]) -> dict[str, CollectionPlan]:
    plans: dict[str, CollectionPlan] = {}
    for environment, count in config["per_environment"].items():
        mixture = {
            CollectorPolicy(name): int(round(count * share))
            for name, share in config["mixture"].items()
        }
        drift = count - sum(mixture.values())
        if drift:
            # Assign the rounding remainder to the largest stratum, and say so,
            # rather than letting the mixture silently miss its target count.
            largest = max(mixture, key=lambda p: mixture[p])
            mixture[largest] += drift
        plans[environment] = CollectionPlan(
            environment=environment,
            transitions=int(count),
            mixture=mixture,
            episode_length=int(config["episode_length"]),
            branch_every=int(config["branch_every"]),
            branch_actions=int(config["branch_actions"]),
        )
    return plans


def build_encoder_dataset(
    encoder_slot: str,
    variant: str,
    config: Mapping[str, Any],
    output_root: Path,
    feature_dimension: int,
) -> EncoderDataset:
    plans = build_plans(config["data"])
    manifest = SplitManifest(
        salt=config["data"]["split_salt"],
        weights={Split(name): float(w) for name, w in config["data"]["split_weights"].items()},
    )
    cache = LatentCache(output_root / "cache" / encoder_slot)
    encoder = CachedEncoder(
        DeterministicControlEncoder(feature_dimension=feature_dimension, variant=variant),
        cache,
        digest_of({"projector": "scale-0-identity", "width": feature_dimension}),
    )

    report = ResourceReport(label=f"dataset:{encoder_slot}")
    records: list[TransitionRecord] = []
    features: dict[str, Any] = {}
    collection_summary: dict[str, Any] = {}
    audits: dict[str, Any] = {}

    with measure(f"dataset:{encoder_slot}", report):
        for family, plan in plans.items():
            result: CollectionResult = collect(
                ADAPTERS[family], plan, manifest, encoder, family=family
            )
            records.extend(result.records)
            features.update(result.features)
            collection_summary[family] = result.canonical_dict()

            family_records = result.records
            family_audit = audit_splits(family_records, manifest)
            family_audit["mixture"] = mixture_report(family_records)
            family_audit["branch_coverage"] = branch_coverage_report(family_records)
            if family in GENERATIVE_FAMILIES:
                assert_no_transition_overlap(family_audit, family)
                family_audit["transition_disjointness"] = "asserted"
            else:
                family_audit["transition_disjointness"] = (
                    "measured; a finite reachable observation set makes disjointness "
                    "arithmetically unattainable in this family"
                )
            audits[family] = family_audit

        assert_trainable(records)
        audits["combined"] = audit_splits(records, manifest)
        manifest.seal()
        cache.flush()
        table = FeatureTable.from_mapping(features)

    total = int(config["data"]["total_transitions"])
    if len(records) != total:
        raise ContractViolation(
            f"collected {len(records)} transitions, the matrix fixes {total}"
        )
    report.cache_report = cache.size_report()
    report.throughput = {
        "transitions_per_second": len(records) / max(report.wall_seconds, 1e-9),
    }
    return EncoderDataset(
        encoder_slot=encoder_slot,
        records=records,
        manifest=manifest,
        table=table,
        cache=cache,
        audits=audits,
        collection=collection_summary,
        resource=report,
    )


def build_all(
    config: Mapping[str, Any], output_root: Path
) -> tuple[dict[str, EncoderDataset], dict[str, Any]]:
    """Build one dataset per encoder slot and prove the raw data is shared."""
    feature_dimension = int(config["encoder"]["feature_dimension"])
    datasets: dict[str, EncoderDataset] = {}
    for slot in config["matrix"]["encoders"]:
        variant = config["encoder"]["control_variants"][slot]
        datasets[slot] = build_encoder_dataset(
            slot, variant, config, output_root, feature_dimension
        )

    digests = {slot: dataset.transition_ids_digest for slot, dataset in datasets.items()}
    manifests = {slot: dataset.manifest.digest for slot, dataset in datasets.items()}
    if len(set(digests.values())) != 1:
        raise ContractViolation(
            f"encoder slots saw different raw transitions: {digests}; collection "
            "must not depend on the encoder"
        )
    if len(set(manifests.values())) != 1:
        raise ContractViolation(f"encoder slots used different split manifests: {manifests}")

    feature_digests = {slot: d.table.digest for slot, d in datasets.items()}
    if len(set(feature_digests.values())) != len(feature_digests):
        raise ContractViolation(
            "two encoder slots produced identical features; the slots are not independent"
        )

    summary = {
        "transition_ids_digest": next(iter(digests.values())),
        "split_manifest_digest": next(iter(manifests.values())),
        "per_slot": {slot: dataset.canonical_dict() for slot, dataset in datasets.items()},
        "feature_digests": feature_digests,
        "raw_data_shared": True,
        "feature_paths_independent": True,
    }
    return datasets, summary
