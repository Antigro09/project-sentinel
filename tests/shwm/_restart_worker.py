"""Worker for the cross-process restart gate. Not a test module.

Run as a subprocess so that "no undeclared process state" is tested against an
actual fresh interpreter rather than against a fresh object in a warm process.
Anything the run depends on that lives in a module global, a cached import, or
the MLX global random stream will diverge here and nowhere else.

    python _restart_worker.py full   <workdir> <updates>
    python _restart_worker.py first  <workdir> <updates>   # then checkpoint
    python _restart_worker.py second <workdir> <updates>   # resume and finish
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sentinel.env.adapters.synthetic_control import SyntheticControlAdapter  # noqa: E402
from sentinel.wm.cache import LatentCache  # noqa: E402
from sentinel.wm.collect import (  # noqa: E402
    CollectionPlan,
    FeatureTable,
    SequenceSampler,
    collect,
)
from sentinel.wm.dataset import CollectorPolicy, Split, SplitManifest  # noqa: E402
from sentinel.wm.encoder import CachedEncoder, DeterministicControlEncoder  # noqa: E402
from sentinel.wm.latent_contract import RepresentationKind  # noqa: E402
from sentinel.wm.models import build_model  # noqa: E402
from sentinel.wm.objective import ObjectiveConfig  # noqa: E402
from sentinel.wm.sizing import solve_config  # noqa: E402
from sentinel.wm.trainer import Trainer, build_optimizer, parameter_digest  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402

TRANSITIONS = 1_200
WIDTH = 64
TARGET = 50_000_000
SEED = 6600


def build(workdir: Path):
    plan = CollectionPlan(
        environment="synthetic_control",
        transitions=TRANSITIONS,
        mixture={
            CollectorPolicy.RANDOM: 360,
            CollectorPolicy.SCRIPTED_ORACLE: 300,
            CollectorPolicy.SENTINEL: 300,
            CollectorPolicy.UNCERTAINTY_SEEKING: 240,
        },
        episode_length=40,
    )
    manifest = SplitManifest(salt="restart", weights={Split.TRAIN: 0.8, Split.DEV_HELD_OUT: 0.2})
    encoder = CachedEncoder(
        DeterministicControlEncoder(feature_dimension=WIDTH),
        LatentCache(workdir / "cache"),
        digest_of("projector"),
    )
    collected = collect(
        lambda gate: SyntheticControlAdapter(gate=gate),
        plan,
        manifest,
        encoder,
        family="synthetic_control",
    )
    table = FeatureTable.from_mapping(collected.features)
    sized = solve_config(
        RepresentationKind.HYBRID,
        TARGET,
        encoder_dimension=WIDTH,
        latent_width=256,
        action_count=4,
    )
    model = build_model(sized.config, seed=SEED)
    sampler = SequenceSampler.from_records(
        collected.records,
        manifest,
        split=Split.TRAIN,
        sequence_length=8,
        batch_size=4,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        optimizer=build_optimizer(),
        sampler=sampler,
        table=table,
        objective=ObjectiveConfig(),
        seed=SEED,
        data_digest=collected.transition_ids_digest,
        split_manifest_digest=manifest.digest,
    )
    return trainer, collected


def main() -> None:
    phase, workdir, updates = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    trainer, collected = build(workdir)
    checkpoint = workdir / "checkpoint"

    if phase == "second":
        trainer.restore(checkpoint)

    outcome = trainer.run(updates, diagnose_every=0)

    if phase == "first":
        trainer.save(checkpoint)

    print(
        json.dumps(
            {
                "phase": phase,
                "updates": trainer.update_index,
                "parameters_digest": parameter_digest(trainer.model),
                "loss_history": [round(v, 6) for v in trainer.loss_history],
                "data_digest": collected.transition_ids_digest,
                "prng_key": list(trainer.declared_state({}, {}, {}).prng_key),
            }
        )
    )


if __name__ == "__main__":
    main()
