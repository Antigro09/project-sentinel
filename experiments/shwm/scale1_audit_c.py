"""C. Is hidden-phase recovery real, or is it reading the reset indicator?

This audit exists because the answer was once "the indicator". At three switches
only 27% of episodes ever changed polarity, so in the rest the hidden value
equalled the one drawn on the reset frame and a probe scoring well on it had
learned nothing about state.

Density was raised and the rate is now 76%, but that fixes the environment, not
the evidence. What settles it is conditioning: on **post-switch** states the
initial value is provably the wrong answer, so an interface that still recovers
polarity there is tracking something. A `reset_frame_only` feature is included
precisely so it can fail that subset -- if it does not, the subset is not doing
its job.

    .venv-shwm/bin/python experiments/shwm/scale1_audit_c.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS,
    ProceduralVisualV2Adapter,
    build_hidden_state_certificate,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402

from feature_qualification import build_slot_matrices, readout  # noqa: E402
from feature_sufficiency import PENALTIES, RandomFourier, ridge_fit, ridge_predict  # noqa: E402

RFF_WIDTH = 1024
BANDWIDTHS = (0.05, 0.2)


def collect_phase_set(layouts, steps: int) -> list[dict[str, Any]]:
    """Trajectories annotated with everything the conditioning needs."""
    gate = AuthorityGate(gate_id="phase-audit")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    samples: list[dict[str, Any]] = []
    for layout in layouts:
        result = adapter.reset(layout)
        initial = adapter._polarity
        previous_action = -1
        for step in range(steps):
            snapshot = adapter.snapshot().reveal("evaluator")
            samples.append(
                {
                    "layout": layout,
                    "step": step,
                    "observation": adapter._observation(),
                    "frame": adapter.frame().copy(),
                    "truth": {"polarity": int(snapshot["polarity"])},
                    "initial_polarity": int(initial),
                    "crossings": int(snapshot["switch_crossings"]),
                    "previous_action": previous_action,
                    "action_result": int(not snapshot["last_blocked"]),
                    "scalar_sensors": {
                        "action_result": float(not snapshot["last_blocked"]),
                    },
                }
            )
            action = ACTIONS[(step * 3 + layout) % len(ACTIONS)]
            previous_action = action
            result = adapter.step(action, gate.authorize_evaluator(action, "phase-audit"))
            if result.terminated:
                break
    for sample in samples:
        sample["episode_crossings"] = max(
            s["crossings"] for s in samples if s["layout"] == sample["layout"]
        )
    return samples


def probe(train_x, train_y, val_x, val_y, test_x, test_y) -> dict[str, Any]:
    if len(np.unique(test_y)) < 2 or len(test_y) < 20:
        return {"score": float("nan"), "baseline": float("nan"), "margin": float("nan"),
                "observations": int(len(test_y)), "insufficient": True}
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    tr, va, te = (train_x - mean) / scale, (val_x - mean) / scale, (test_x - mean) / scale
    encode = lambda y: np.eye(2, dtype=np.float32)[y.astype(int)]
    score = lambda p, y: float((p.argmax(axis=1) == y.astype(int)).mean())
    best = None
    for bandwidth in BANDWIDTHS:
        expansion = RandomFourier(RFF_WIDTH, bandwidth)
        expansion.fit_shape(tr.shape[1], seed=6600)
        tr_e, va_e, te_e = expansion(tr), expansion(va), expansion(te)
        for penalty in PENALTIES:
            weights = ridge_fit(tr_e, encode(train_y), penalty)
            value = score(ridge_predict(va_e, weights), val_y)
            if best is None or value > best[0]:
                best = (value, ridge_predict(te_e, weights))
    held = score(best[1], test_y)
    majority = int(np.bincount(train_y.astype(int), minlength=2).argmax())
    baseline = float((np.full_like(test_y, majority) == test_y.astype(int)).mean())
    return {"score": held, "baseline": baseline, "margin": held - baseline,
            "observations": int(len(test_y)), "insufficient": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-levels", type=int, default=200)
    parser.add_argument("--test-levels", type=int, default=100)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--encoders", nargs="*", default=["qwen3_vl_4b"])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/audit-c.json")
    arguments = parser.parse_args()

    import yaml

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    train_layouts = list(range(40_000, 40_000 + arguments.train_levels))
    test_layouts = list(range(50_000, 50_000 + arguments.test_levels))

    print("collecting phase-audit trajectories")
    train_samples = collect_phase_set(train_layouts, arguments.steps)
    test_samples = collect_phase_set(test_layouts, arguments.steps)
    samples = train_samples + test_samples
    split = len(train_samples)
    print(f"  train {len(train_samples)}  test {len(test_samples)}")

    cache = arguments.out.parent / "phase-slots.npz"
    signature = digest_of({"train": arguments.train_levels, "test": arguments.test_levels,
                           "steps": arguments.steps, "encoders": sorted(arguments.encoders)})
    matrices = None
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if str(stored["signature"]) == signature:
            matrices = {k[5:]: stored[k] for k in stored.files if k.startswith("mat::")}
            print("  reusing cached slots")
    if matrices is None:
        matrices, _ = build_slot_matrices(samples, config, arguments.encoders)
        np.savez_compressed(cache, signature=signature, meta="{}",
                            **{f"mat::{k}": v for k, v in matrices.items()})

    interface = f"{arguments.encoders[0]}_spatial_slots"
    reduced = readout(matrices[interface], interface)
    oracle_reduced = readout(matrices["oracle_structured_state"], "oracle_structured_state")

    layouts = np.array([s["layout"] for s in samples])
    steps = np.array([s["step"] for s in samples])
    crossings = np.array([s["crossings"] for s in samples])
    episode_crossings = np.array([s["episode_crossings"] for s in samples])
    actions = np.array([s["previous_action"] for s in samples])
    action_onehot = np.eye(len(ACTIONS) + 1, dtype=np.float32)[actions + 1]

    def lagged(source, lag):
        out = np.zeros_like(source)
        for i in range(len(source)):
            j = i - lag
            if j >= 0 and layouts[j] == layouts[i]:
                out[i] = source[j]
        return out

    reset_frame = np.zeros_like(reduced)
    for i in range(len(samples)):
        start = i - steps[i]
        reset_frame[i] = reduced[start] if layouts[start] == layouts[i] else reduced[i]

    history = np.concatenate([reduced] + [lagged(reduced, k) for k in range(1, arguments.steps)], axis=1)
    generator = np.random.default_rng(5)
    shuffled_rows = generator.permutation(len(samples))
    history_shuffled = np.concatenate(
        [reduced] + [lagged(reduced, k)[shuffled_rows] for k in range(1, arguments.steps)], axis=1
    )
    action_history = np.concatenate(
        [history] + [lagged(action_onehot, k) for k in range(arguments.steps)], axis=1
    )
    action_history_shuffled = np.concatenate(
        [history] + [lagged(action_onehot, k)[shuffled_rows] for k in range(arguments.steps)], axis=1
    )

    variants = {
        "current_frame_only": reduced,
        "reset_frame_only": reset_frame,
        "correct_history": history,
        "shuffled_history": history_shuffled,
        "correct_actions": action_history,
        "shuffled_actions": action_history_shuffled,
        "structured_oracle": oracle_reduced,
    }

    y = np.array([s["truth"]["polarity"] for s in samples], dtype=np.float32)
    is_train = np.zeros(len(samples), dtype=bool)
    is_train[:split] = True
    validation = is_train & (np.arange(len(samples)) % 5 == 0)
    fit = is_train & ~validation
    test = ~is_train

    conditions = {
        "all_episodes": test,
        "switch_crossing_episodes": test & (episode_crossings > 0),
        "post_first_switch_states": test & (crossings >= 1),
        "after_two_or_more_changes": test & (crossings >= 2),
    }
    for action in ACTIONS:
        conditions[f"previous_action_{action}"] = test & (actions == action)

    results: dict[str, Any] = {}
    for variant, features in variants.items():
        rows = {}
        for condition, mask in conditions.items():
            rows[condition] = probe(
                features[fit], y[fit], features[validation], y[validation],
                features[mask], y[mask],
            )
        results[variant] = rows

    certificate = build_hidden_state_certificate(9000, max_depth=8)
    pins = {
        "reset_indicator_cannot_solve_post_switch": {
            "reset_frame_only_margin": results["reset_frame_only"]["post_first_switch_states"]["margin"],
            "passes": results["reset_frame_only"]["post_first_switch_states"]["margin"] < 0.10,
            "meaning": (
                "on post-switch states the initial value is the wrong answer, so a feature "
                "carrying only the reset frame must fail there"
            ),
        },
        "hidden_state_not_in_scalar_sensors": {
            "sensor_keys": sorted(samples[0]["scalar_sensors"]),
            "passes": "polarity" not in samples[0]["scalar_sensors"],
        },
        "hidden_state_not_in_action_result": {
            "action_result_is_binary_success": True,
            "correlation_with_polarity": (
                correlation := float(
                    np.corrcoef(
                        np.array([s["action_result"] for s in samples], dtype=float), y
                    )[0, 1]
                )
            ),
            "passes": abs(correlation) < 0.2,
            "meaning": (
                "action_result reports whether a move was blocked; a strong correlation "
                "with polarity would make the hidden variable readable from metadata"
            ),
        },
        "every_invariance_pair_reachable": {
            "certificate_histories": [list(certificate.history_a), list(certificate.history_b)],
            "both_executed": True,
            "polarities": [certificate.polarity_a, certificate.polarity_b],
            "passes": (
                certificate.polarity_a != certificate.polarity_b
                and len(certificate.history_a) == len(certificate.history_b)
            ),
        },
        "construct_is_nonvacuous_post_switch": {
            "oracle_margin_post_switch": results["structured_oracle"]["post_first_switch_states"]["margin"],
            "passes": results["structured_oracle"]["post_first_switch_states"]["margin"] > 0.2,
            "meaning": (
                "the oracle recovers polarity post-switch, so the information exists and "
                "is readable; any interface that fails there is failing to carry it, not "
                "being asked something impossible"
            ),
        },
        "no_admissible_variant_recovers_post_switch": {
            "best_non_oracle_margin_post_switch": max(
                results[v]["post_first_switch_states"]["margin"]
                for v in results
                if v != "structured_oracle"
            ),
            "meaning": (
                "if this is at or below zero while the oracle is high, the hidden-state "
                "channel is absent from every admissible feature variant"
            ),
        },
    }

    document = {
        "interface": interface,
        "observations": {"train": len(train_samples), "test": len(test_samples)},
        "conditions": {k: int(v.sum()) for k, v in conditions.items()},
        "results": results,
        "pins": pins,
        "note": (
            "an interface that recovers polarity on post-switch states is tracking "
            "something; one that only recovers it overall may be reading the indicator"
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    order = ["all_episodes", "switch_crossing_episodes", "post_first_switch_states",
             "after_two_or_more_changes"] + [f"previous_action_{a}" for a in ACTIONS]
    print(f"\n=== C: polarity margin over baseline, interface {interface} ===")
    print(f"{'feature variant':24s} " + " ".join(f"{c[:14]:>15s}" for c in order))
    for variant in variants:
        cells = []
        for condition in order:
            row = results[variant][condition]
            cells.append("  n/a" if row["insufficient"] else f"{row['margin']:+.3f}")
        print(f"{variant:24s} " + " ".join(f"{c:>15s}" for c in cells))
    print(f"\nobservations per condition: " +
          ", ".join(f"{k}={int(v.sum())}" for k, v in conditions.items()))
    print("\npins:")
    for name, pin in pins.items():
        verdict = pin.get("passes")
        print(f"  {name:44s} {verdict}")
    print(f"\nwritten: {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
