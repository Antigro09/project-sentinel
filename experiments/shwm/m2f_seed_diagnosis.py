"""C. Why did M2E seed 14017 fail, and would more restarts have saved it?

Diagnostic only. Nothing here may enter an M2F threshold or an M2E score, and the
module writes to its own artifact so it cannot be confused with either.

One structural point settles a candidate cause before any compute is spent. The seed
controls the initialisation and the minibatch order and NOTHING else: the training
trajectories, the alias population and the event channel are identical for every seed
in the arm. So "a data or identifiability defect in that seed" is not a thing this
design can produce, and the honest way to say that is to show the dataset digest is
seed-independent rather than to argue it.

Restarts 0-7 reproduce M2E's exactly, which is the positive control on this module: if
they do not match the recorded training likelihoods, the diagnosis is of some other
computation.

    .venv-shwm/bin/python experiments/shwm/m2f_seed_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2e_core as m2e
import m2f_core as core
from m2d_core import ARTIFACTS, write
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

FAILED_SEED = 14_017
CONTROL_SEED = 14_003
K = 64
PROBE_EVERY = 128


def trajectory_of_fit(train, seed: int, restart: int, updates: int = m2d.UPDATES):
    """Train while recording the training likelihood, to separate optimizer instability
    and checkpoint selection from a bad basin."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    spec = core.generic_spec(seed, restart)
    x, y, e, m, reset = m2d.pad(train)
    model = m2d.make_model(spec, x.shape[2], seed * 1_000 + restart)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed * 1_000 + restart)
    curve = []
    for step in range(updates):
        pick = rng.integers(0, len(x), min(32, len(x)))
        xb, yb, eb = mx.array(x[pick]), mx.array(y[pick]), mx.array(e[pick])
        mb, rb = mx.array(m[pick]), mx.array(reset[pick])

        def loss_fn(mo):
            logits, _ = mo(xb, rb, eb)
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, m2d.CLASSES), yb.reshape(-1), reduction="none")
            return (losses * mb.reshape(-1)).sum() / mb.sum()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if (step + 1) % PROBE_EVERY == 0:
            curve.append(m2e.training_log_likelihood(model, train))
    return model, curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2f-seed-diagnosis.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train_t = collect(list(m2d.TRAIN_LAYOUTS), 3, 9, CANONICAL_APPEARANCE_SEED, 11)
    train = build_dataset(train_t, 5)
    population = m2d.build_population(m2d.ALIAS_LAYOUTS)
    tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))

    m2e_report = json.loads((ARTIFACTS / "m2e-transition.json").read_text())
    recorded = next(r for r in m2e_report["validation"]["E_generic_restarts"]["records"]
                    if r["seed"] == FAILED_SEED)["compute"]["restart_scores"]

    report: dict[str, Any] = {
        "failed_seed": FAILED_SEED, "control_seed": CONTROL_SEED, "restarts": K,
        "diagnostic_only": True,
        "may_contribute_to_m2f_thresholds": False,
        "may_contribute_to_m2e_score": False,
        "dataset_is_seed_independent": {
            "note": "the seed drives initialisation and minibatch order only; the "
                    "trajectories, alias population and event channel are identical "
                    "across seeds",
            "train_episode_digest": m2e.episode_manifest(
                train, "train", m2d.TRAIN_LAYOUTS, 11)["episode_digest"],
            "alias_member_digest": m2e.population_manifest(
                population, "alias", m2d.ALIAS_LAYOUTS)["member_digest"]},
    }

    print(f"seed {FAILED_SEED}: {K} generic restarts, every one retained\n")
    print(f"{'restart':>7s} {'train LL':>10s} {'alias':>7s} {'phase':>7s} "
          f"{'T-ent':>6s} {'stay-flip':>10s} {'occ':>5s} {'collapsed':>9s}")
    rows = []
    for restart in range(K):
        row, _ = core.run_restart(train, tensors, population, FAILED_SEED, restart)
        rows.append(row)
        if restart < 10 or row.alias_accuracy > 0.9 or restart % 16 == 15:
            print(f"{restart:7d} {row.training_log_likelihood:10.5f} "
                  f"{row.alias_accuracy:7.4f} {row.phase_accuracy:7.4f} "
                  f"{row.transition_entropy:6.3f} {row.stay_minus_flip:+10.4f} "
                  f"{row.state_occupancy:5.2f} {str(row.collapsed):>9s}", flush=True)
    report["restart_table"] = [r.to_dict() for r in rows]

    reproduced = [round(r.training_log_likelihood, 4) for r in rows[:8]]
    expected = [round(v, 4) for v in recorded]
    report["reproduces_m2e_first_eight"] = bool(reproduced == expected)
    print(f"\nrestarts 0-7 reproduce the M2E record: "
          f"{report['reproduces_m2e_first_eight']}")
    if not report["reproduces_m2e_first_eight"]:
        print(f"  m2f {reproduced}\n  m2e {expected}")

    solved = [r for r in rows if r.alias_accuracy > 0.9]
    report["solved_restarts"] = [r.restart for r in solved]
    report["first_solved_restart"] = solved[0].restart if solved else None
    best = max(rows, key=lambda r: r.training_log_likelihood)
    report["best_by_training_likelihood"] = best.to_dict()
    report["training_likelihood_would_have_found_it"] = bool(
        solved and best.alias_accuracy > 0.9)

    # Control seed, for contrast.
    control = [core.run_restart(train, tensors, population, CONTROL_SEED, r)[0]
               for r in range(8)]
    report["control_restart_table"] = [r.to_dict() for r in control]
    report["control_solved_restarts"] = [r.restart for r in control
                                         if r.alias_accuracy > 0.9]

    # Optimizer instability and checkpoint selection, on the best failed restart.
    model, curve = trajectory_of_fit(train, FAILED_SEED, best.restart)
    report["fit_curve_best_failed_restart"] = [round(v, 6) for v in curve]
    report["fit_curve_is_monotone"] = bool(
        all(curve[i + 1] >= curve[i] - 1e-3 for i in range(len(curve) - 1)))
    report["best_intermediate_beats_final"] = bool(max(curve) > curve[-1] + 1e-3)

    causes = {
        "1_unlucky_initialisation": bool(solved),
        "2_data_or_identifiability_defect_in_that_seed": False,
        "3_optimizer_instability": bool(not report["fit_curve_is_monotone"]),
        "4_checkpoint_selection": bool(report["best_intermediate_beats_final"]),
        "5_insufficient_restart_count": bool(solved and min(
            r.restart for r in solved) >= 8),
    }
    report["causes"] = causes
    report["diagnosis"] = ([k for k, v in causes.items() if v]
                           or ["none_of_the_five_candidates"])

    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nsolved restarts (alias > 0.9): {report['solved_restarts'] or 'NONE'}")
    print(f"control seed {CONTROL_SEED} solved restarts in its first 8: "
          f"{report['control_solved_restarts']}")
    print(f"fit curve monotone: {report['fit_curve_is_monotone']}; "
          f"an intermediate checkpoint beats the final: "
          f"{report['best_intermediate_beats_final']}")
    print(f"diagnosis: {report['diagnosis']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
