"""G. Feature qualification across interfaces and strata.

The question is narrower than "which representation is best". It is whether any
admissible interface carries (a) enough to predict what a *counterfactual* action
would do, and (b) enough history to recover a hidden variable that no single
frame reveals. Those are the two things a Scale-1 world model would be asked to
supply, and an interface that cannot support them makes the training run a
measurement of the interface.

Every readout is identical across interfaces: the same flattening, the same fixed
projection to a common width, the same probe family, the same penalty grid. So a
difference between columns is a difference in features and not in capacity.

Two controls bracket every number. The oracle interface is evaluator-only and
must win, or the probe cannot read the variable and the column means nothing. A
shuffled-label arm must score at baseline, or the protocol is leaking.

    .venv-shwm/bin/python experiments/shwm/feature_qualification.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS,
    GOAL_PHRASES,
    GRID,
    ProceduralVisualV2Adapter,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder  # noqa: E402
from sentinel.wm.backbones import FROZEN_CANDIDATES  # noqa: E402
from sentinel.wm.interfaces import (  # noqa: E402
    InterfaceContext,
    build_interfaces,
    interface_report,
)
from sentinel.wm.packet import SLOT_COUNT, SLOT_WIDTH  # noqa: E402
from sentinel.wm.splits_v2 import (  # noqa: E402
    CANONICAL_APPEARANCE_SEED,
    EpisodeDescriptor,
    Stratum,
    StratifiedSplitManifest,
)
from sentinel.wm.versioning import digest_of  # noqa: E402

from feature_sufficiency import PENALTIES, RandomFourier, ridge_fit, ridge_predict  # noqa: E402

READOUT_WIDTH = 512
HISTORY_STEPS = 3
RFF_WIDTH = 1024
RFF_BANDWIDTHS = (0.05, 0.2)


@dataclass(frozen=True, slots=True)
class QualTarget:
    name: str
    kind: str
    classes: int = 0
    history: bool = False
    group: str = ""


TARGETS: tuple[QualTarget, ...] = (
    QualTarget("agent_row", "classification", GRID, group="position"),
    QualTarget("agent_col", "classification", GRID, group="position"),
    QualTarget("delta_row", "regression", group="relation"),
    QualTarget("delta_col", "regression", group="relation"),
    QualTarget("blocked_0", "classification", 2, group="legal_actions"),
    QualTarget("blocked_1", "classification", 2, group="legal_actions"),
    QualTarget("blocked_2", "classification", 2, group="legal_actions"),
    QualTarget("blocked_3", "classification", 2, group="legal_actions"),
    QualTarget("successor_0", "regression", group="intervention"),
    QualTarget("successor_1", "regression", group="intervention"),
    QualTarget("successor_2", "regression", group="intervention"),
    QualTarget("successor_3", "regression", group="intervention"),
    QualTarget("moved_class", "classification", 2, group="action_effect"),
    QualTarget("polarity", "classification", 2, history=True, group="hidden_phase"),
    QualTarget("goal_progress", "regression", group="progress"),
    QualTarget("terminated", "classification", 2, group="termination"),
)


def collect(descriptors: Sequence[EpisodeDescriptor], steps: int) -> list[dict[str, Any]]:
    """Trajectories plus, at every step, the outcome of every legal action.

    The counterfactual branches come from restoring the exact state, so the
    intervention targets are what those actions really do rather than what a
    model guessed they would.
    """
    gate = AuthorityGate(gate_id="qualification")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    samples: list[dict[str, Any]] = []
    for descriptor in descriptors:
        dynamic = "base"
        if descriptor.appearance_seed != CANONICAL_APPEARANCE_SEED:
            dynamic = f"appearance:{descriptor.appearance_seed}"
        result = adapter.reset(descriptor.layout_seed, dynamic)
        level = adapter._require()
        goal = level.markers[adapter._goal_marker]
        previous = None
        for step in range(steps):
            snapshot = adapter.snapshot()
            truth_raw = snapshot.reveal("evaluator")
            position = tuple(int(v) for v in truth_raw["position"])

            successors, blocked = {}, {}
            for action in ACTIONS:
                adapter.restore(snapshot)
                token = gate.authorize_evaluator(action, "intervention")
                after = adapter.step(action, token)
                successors[action] = float(after.probes.values["observable_signature"])
                blocked[action] = int(not after.probes.values["action_succeeded"])
            adapter.restore(snapshot)

            probes = adapter.probes()
            samples.append(
                {
                    "descriptor": descriptor,
                    "step": step,
                    "observation": adapter._observation(),
                    "frame": adapter.frame().copy(),
                    "goal_text": adapter.goal_text(),
                    "truth": {
                        "agent_row": position[0],
                        "agent_col": position[1],
                        "delta_row": float(goal[0] - position[0]),
                        "delta_col": float(goal[1] - position[1]),
                        "blocked_0": blocked[0],
                        "blocked_1": blocked[1],
                        "blocked_2": blocked[2],
                        "blocked_3": blocked[3],
                        "successor_0": successors[0],
                        "successor_1": successors[1],
                        "successor_2": successors[2],
                        "successor_3": successors[3],
                        "moved_class": int(previous is not None and position != previous),
                        "polarity": int(truth_raw["polarity"]),
                        "goal_progress": float(probes.values["goal_progress"]),
                        "terminated": int(probes.values["termination"]),
                    },
                }
            )
            previous = position
            action = ACTIONS[(step * 3 + descriptor.layout_seed) % len(ACTIONS)]
            result = adapter.step(action, gate.authorize_evaluator(action, "rollout"))
            if result.terminated:
                break
    return samples


def build_slot_matrices(samples, config, encoder_ids) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run each backbone once per observation and derive every interface from it."""
    tokens_by_encoder: dict[str, list[np.ndarray]] = {}
    timings: dict[str, Any] = {}
    root = REPO / config["encoder"]["weights_root"]
    for encoder_id in encoder_ids:
        candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
        encoder = MlxVlmBackboneEncoder(
            BackboneSpec(
                encoder_id,
                candidate.repository,
                config["encoder"]["revisions"][encoder_id],
                config["encoder"]["licences"][encoder_id],
                root / encoder_id,
            )
        )
        started = time.perf_counter()
        rows = [
            encoder.encode_visual_tokens(s["observation"], s["frame"]) for s in samples
        ]
        elapsed = time.perf_counter() - started
        timings[encoder_id] = {
            "observations": len(samples),
            "seconds": elapsed,
            "observations_per_second": len(samples) / elapsed,
            "visual_tokens": int(rows[0].shape[0]),
        }
        tokens_by_encoder[encoder_id] = rows
        encoder.release()

    interfaces = build_interfaces(tuple(encoder_ids))
    matrices: dict[str, np.ndarray] = {}
    for interface in interfaces:
        stack = np.zeros((len(samples), SLOT_COUNT, SLOT_WIDTH), dtype=np.float32)
        for index, sample in enumerate(samples):
            context = InterfaceContext(
                observation=sample["observation"],
                frame=sample["frame"],
                visual_tokens={k: tokens_by_encoder[k][index] for k in encoder_ids},
                truth=sample["truth"],
            )
            stack[index] = interface.slots(context)
        matrices[interface.name] = stack.reshape(len(samples), -1)
    return matrices, {"encode_timings": timings, "interfaces": interface_report(interfaces)}


def readout(matrix: np.ndarray, tag: str) -> np.ndarray:
    """One fixed projection to a common width, identical for every interface."""
    seed = int(digest_of({"readout": tag, "dim": int(matrix.shape[1])})[7:15], 16)
    generator = np.random.default_rng(seed)
    projection = (
        generator.normal(size=(matrix.shape[1], READOUT_WIDTH)) / np.sqrt(matrix.shape[1])
    ).astype(np.float32)
    return matrix @ projection


def with_history(reduced: np.ndarray, episode_index: np.ndarray, steps: int) -> np.ndarray:
    """Concatenate the current readout with the previous `steps-1`, zero-padded
    at an episode start so history never crosses an episode boundary."""
    blocks = [reduced]
    for lag in range(1, steps):
        lagged = np.zeros_like(reduced)
        for i in range(len(reduced)):
            j = i - lag
            if j >= 0 and episode_index[j] == episode_index[i]:
                lagged[i] = reduced[j]
        blocks.append(lagged)
    return np.concatenate(blocks, axis=1)


def probe_target(train_x, train_y, val_x, val_y, test_x, test_y, target, shuffled=False):
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    tr, va, te = (train_x - mean) / scale, (val_x - mean) / scale, (test_x - mean) / scale
    if shuffled:
        generator = np.random.default_rng(11)
        train_y = generator.permutation(train_y)

    if target.kind == "classification":
        counts = np.bincount(test_y.astype(int), minlength=max(target.classes, 1))
        degenerate = float(counts.max() / max(counts.sum(), 1)) > 0.95
        encode = lambda y: np.eye(target.classes, dtype=np.float32)[y.astype(int)]
        score = lambda p, y: float((p.argmax(axis=1) == y.astype(int)).mean())
        train_t, val_t = encode(train_y), val_y
        baseline = float(
            (np.bincount(train_y.astype(int), minlength=target.classes).argmax()
             == test_y.astype(int)).mean()
        )
    else:
        degenerate = float(test_y.var()) < 1e-3
        variance = float(((test_y - train_y.mean()) ** 2).mean())
        train_t, val_t = train_y.reshape(-1, 1), val_y
        score = lambda p, y: float(1.0 - ((y - p.reshape(-1)) ** 2).mean() / (variance + 1e-12))
        baseline = 0.0

    best = None
    for bandwidth in RFF_BANDWIDTHS:
        expansion = RandomFourier(RFF_WIDTH, bandwidth)
        expansion.fit_shape(tr.shape[1], seed=6600)
        tr_e, va_e, te_e = expansion(tr), expansion(va), expansion(te)
        for penalty in PENALTIES:
            weights = ridge_fit(tr_e, train_t, penalty)
            value = score(ridge_predict(va_e, weights), val_t)
            if best is None or value > best[0]:
                best = (value, ridge_predict(te_e, weights))
    return {
        "target": target.name,
        "group": target.group,
        "history": target.history,
        "held_out_score": score(best[1], test_y),
        "baseline": baseline,
        "margin": score(best[1], test_y) - baseline,
        "degenerate": degenerate,
        "shuffled_control": shuffled,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-levels", type=int, default=150)
    parser.add_argument("--shift-levels", type=int, default=60)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--encoders", nargs="*", default=["qwen3_vl_4b", "gemma3_4b"])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/feature-qualification.json")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--history-steps", type=int, default=None)
    arguments = parser.parse_args()

    if arguments.history_steps:
        arguments.steps = max(arguments.steps, arguments.history_steps)

    import yaml

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())

    train_layouts = list(range(10_000, 10_000 + arguments.clean_levels))
    held_layouts = list(range(20_000, 20_000 + arguments.clean_levels // 2))
    held_appearances = list(range(30_000, 30_000 + arguments.shift_levels))

    manifest = StratifiedSplitManifest(
        train_layouts=frozenset(train_layouts),
        held_layouts=frozenset(held_layouts),
        train_appearances=frozenset({CANONICAL_APPEARANCE_SEED}),
        held_appearances=frozenset(held_appearances),
    )

    def descriptors(layouts, appearance_of):
        out = []
        for layout in layouts:
            appearance = appearance_of(layout)
            descriptor = EpisodeDescriptor(
                "procedural_visual_v2", layout, appearance, layout, layout
            )
            manifest.assign(descriptor)
            out.append(descriptor)
        return out

    train_set = descriptors(train_layouts, lambda _: CANONICAL_APPEARANCE_SEED)
    clean_set = descriptors(held_layouts, lambda _: CANONICAL_APPEARANCE_SEED)
    shift_set = descriptors(
        train_layouts[: arguments.shift_levels],
        lambda layout: held_appearances[layout % len(held_appearances)],
    )
    crossed_set = descriptors(
        held_layouts[: arguments.shift_levels],
        lambda layout: held_appearances[layout % len(held_appearances)],
    )
    manifest.seal()

    groups = {
        "train": train_set,
        Stratum.DYNAMICS_CLEAN.value: clean_set,
        Stratum.APPEARANCE_SHIFT.value: shift_set,
        Stratum.CROSSED_SHIFT.value: crossed_set,
    }
    print("collecting v2 trajectories with counterfactual branches")
    collected = {name: collect(items, arguments.steps) for name, items in groups.items()}
    for name, samples in collected.items():
        print(f"  {name:18s} {len(samples):5d} observations from {len(groups[name])} levels")

    ordered = [(name, s) for name in groups for s in collected[name]]
    samples = [s for _, s in ordered]
    boundaries: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in groups:
        boundaries[name] = (cursor, cursor + len(collected[name]))
        cursor += len(collected[name])

    cache = arguments.out.parent / "qualification-slots.npz"
    signature = digest_of(
        {
            "clean": arguments.clean_levels,
            "shift": arguments.shift_levels,
            "steps": arguments.steps,
            "encoders": sorted(arguments.encoders),
        }
    )
    matrices = meta = None
    if cache.exists() and not arguments.rebuild:
        stored = np.load(cache, allow_pickle=True)
        if str(stored["signature"]) == signature:
            matrices = {k[5:]: stored[k] for k in stored.files if k.startswith("mat::")}
            meta = json.loads(str(stored["meta"]))
            print(f"  reusing cached slot matrices from {cache.name}")
    if matrices is None:
        matrices, meta = build_slot_matrices(samples, config, arguments.encoders)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, signature=signature, meta=json.dumps(meta, default=str),
            **{f"mat::{k}": v for k, v in matrices.items()},
        )

    episode_index = np.array(
        [hash((s["descriptor"].layout_seed, s["descriptor"].appearance_seed)) for s in samples]
    )
    train_lo, train_hi = boundaries["train"]
    validation_mask = np.zeros(len(samples), dtype=bool)
    validation_mask[train_lo : train_lo + (train_hi - train_lo) // 5] = True

    results: dict[str, Any] = {}
    for name, matrix in matrices.items():
        reduced = readout(matrix, name)
        # Full-episode history. Polarity is the initial value XOR the switch
        # crossings since, and the initial value is shown only on the reset
        # frame -- so a window that cannot reach step 0 makes the hidden variable
        # unrecoverable in principle and the probe would be measuring that
        # instead of the interface.
        historical = with_history(reduced, episode_index, arguments.steps)
        per_stratum: dict[str, Any] = {}
        for stratum in (
            Stratum.DYNAMICS_CLEAN.value,
            Stratum.APPEARANCE_SHIFT.value,
            Stratum.CROSSED_SHIFT.value,
        ):
            lo, hi = boundaries[stratum]
            rows = []
            for target in TARGETS:
                features = historical if target.history else reduced
                y = np.array([s["truth"][target.name] for s in samples], dtype=np.float32)
                train_rows = np.zeros(len(samples), dtype=bool)
                train_rows[train_lo:train_hi] = True
                fit_rows = train_rows & ~validation_mask
                test_rows = np.zeros(len(samples), dtype=bool)
                test_rows[lo:hi] = True
                row = probe_target(
                    features[fit_rows], y[fit_rows],
                    features[validation_mask], y[validation_mask],
                    features[test_rows], y[test_rows], target,
                )
                if target.group in ("intervention", "hidden_phase"):
                    row["shuffled_margin"] = probe_target(
                        features[fit_rows], y[fit_rows],
                        features[validation_mask], y[validation_mask],
                        features[test_rows], y[test_rows], target, shuffled=True,
                    )["margin"]
                rows.append(row)
            per_stratum[stratum] = rows
        results[name] = per_stratum

    def group_margin(name, stratum, group):
        rows = [r for r in results[name][stratum] if r["group"] == group and not r["degenerate"]]
        return float(np.mean([r["margin"] for r in rows])) if rows else float("nan")

    clean = Stratum.DYNAMICS_CLEAN.value
    admissible = [n for n in results if n != "oracle_structured_state"]
    gate = {
        "oracle_calibrated": group_margin("oracle_structured_state", clean, "intervention") > 0.2,
        "interfaces_supporting_intervention": [
            n for n in admissible if group_margin(n, clean, "intervention") > 0.05
        ],
        "interfaces_recovering_hidden_phase": [
            n for n in admissible if group_margin(n, clean, "hidden_phase") > 0.05
        ],
    }
    gate["passed"] = bool(
        gate["oracle_calibrated"]
        and gate["interfaces_supporting_intervention"]
        and gate["interfaces_recovering_hidden_phase"]
    )

    document = {
        "gate": "S1.2 v2 feature qualification",
        "split_manifest_digest": manifest.digest,
        "observations": {k: len(v) for k, v in collected.items()},
        "meta": meta,
        "results": results,
        "gate_result": gate,
        "note": (
            "appearance_shift is a perception diagnostic and does not veto the screen"
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    for stratum in (Stratum.DYNAMICS_CLEAN.value, Stratum.APPEARANCE_SHIFT.value):
        print(f"\n=== {stratum}: mean margin over baseline, by group ===")
        groups_seen = ["position", "relation", "legal_actions", "intervention",
                       "action_effect", "hidden_phase", "progress"]
        print(f"{'interface':34s} " + " ".join(f"{g[:11]:>12s}" for g in groups_seen))
        for name in results:
            cells = [f"{group_margin(name, stratum, g):+.3f}" for g in groups_seen]
            print(f"{name:34s} " + " ".join(f"{c:>12s}" for c in cells))

    print(f"\ngate passed: {gate['passed']}")
    print(f"  oracle calibrated              : {gate['oracle_calibrated']}")
    print(f"  intervention-capable interfaces: {gate['interfaces_supporting_intervention']}")
    print(f"  hidden-phase-capable interfaces: {gate['interfaces_recovering_hidden_phase']}")
    print(f"written: {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
