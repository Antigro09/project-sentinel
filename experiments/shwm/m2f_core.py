"""M2F. Fresh seeds, a per-restart table, and a training-only fit certificate.

Everything downstream of section D is derived from ONE table with a row per
(seed, restart): training log-likelihood, alias accuracy, phase accuracy, transition
statistics and collapse state. Fixed K=8/16/32 and every adaptive policy are prefix
operations over that table rather than separate experiments, which is what makes a
hundred seeds affordable and, more importantly, guarantees that the procedures are
compared on identical restarts rather than on independently drawn ones.

The certificate is the point of the phase: a rule that reads only training-side
quantities and says CERTIFIED or UNRESOLVED_TRANSITION. It never sees phase accuracy,
distance to XOR, or any validation number -- `certify()` takes a training log-likelihood
and a threshold, and there is nothing else in scope for it to read.

    imported by m2f_transition.py, m2f_gauge.py, m2f_events.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m2d_core as m2d  # noqa: E402
import m2e_core as m2e  # noqa: E402
from m2d_core import ARTIFACTS, FilterSpec, checkpoint_hash, write  # noqa: E402

# Fresh ranges. M2C/M2D used 6600-6602, 7000-7019, 8000-8019, 9000-9009; M2E used
# 13000-13019 and 14000-14019. None of these overlap any of those.
DEV_SEEDS = tuple(range(21_000, 21_100))
VALIDATION_SEEDS = tuple(range(22_000, 22_100))
K_MAX = 32
BLOCK = 8

DEV_ALIAS = tuple(range(91_000, 91_010))
VALIDATION_ALIAS = tuple(range(90_000, 90_010))
HELD_OUT_ALIAS = tuple(range(95_000, 95_010))
HELD_OUT_ALIAS_2 = tuple(range(92_000, 92_010))


@dataclass
class RestartRow:
    seed: int
    restart: int
    training_log_likelihood: float
    alias_accuracy: float
    phase_accuracy: float
    transition_entropy: float
    stay_minus_flip: float
    state_occupancy: float
    belief_entropy: float
    collapsed: bool
    checkpoint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generic_spec(seed: int, restart: int, states: int = 2,
                 gauge: str = "reset_onehot") -> FilterSpec:
    """A seed-and-restart-derived, orientation-free perturbation. No environment
    quantity enters except the number of latent states."""
    return FilterSpec("m2f", "filter", states, "generic", gauge=gauge,
                      perturbation=m2e.generic_antisymmetric(seed * 1_000 + restart,
                                                             states))


def transition_statistics(model, train) -> dict[str, float]:
    import mlx.core as mx

    if not hasattr(model, "logits"):
        return {"transition_entropy": float("nan"), "stay_minus_flip": float("nan"),
                "state_occupancy": float("nan"), "belief_entropy": float("nan"),
                "collapsed": False}
    transition = np.asarray(mx.softmax(model.logits, axis=-1))
    states = transition.shape[-1]
    row_entropy = -(transition * np.log(np.maximum(transition, 1e-12))).sum(axis=-1)
    x, y, e, m, reset = m2d.pad(train)
    _, belief = model(mx.array(x), mx.array(reset), mx.array(e))
    mx.eval(belief)
    belief = np.asarray(belief)
    mask = m.astype(bool)
    entropy = -(belief * np.log(np.maximum(belief, 1e-12))).sum(axis=-1)
    normalised = float(entropy[mask].mean() / np.log(states))
    return {"transition_entropy": float(row_entropy.mean() / np.log(states)),
            "stay_minus_flip": float(transition[0].trace() - transition[1].trace())
            if states == 2 else float("nan"),
            "state_occupancy": float(
                len(np.unique(belief.argmax(axis=-1)[mask])) / states),
            "belief_entropy": normalised,
            "collapsed": bool(normalised > 0.9)}


def score_on(model, train, tensors, population, events=None):
    scored = m2d.score_population(model, tensors, events)
    assignment = m2d.fit_state_assignment(model, train)
    phase = float("nan")
    if assignment is not None and scored["belief"].shape[1] > 1:
        predicted = assignment[scored["belief"].argmax(axis=1)]
        truth = np.array([population.states[r.self_index].polarity
                          for r in population.rows])
        phase = float((predicted == truth).mean())
    return scored, phase


def run_restart(train, tensors, population, seed: int, restart: int,
                spec_for: Callable[[int, int], FilterSpec] | None = None,
                events: np.ndarray | None = None,
                extra: dict[str, tuple] | None = None,
                model_override=None) -> tuple[RestartRow, dict[str, np.ndarray]]:
    """Train one restart and record everything about it. Nothing is discarded.

    `extra` maps a label to (tensors, population) so a single trained model can be
    scored on held-out alias sets in the same pass. Retraining to score elsewhere would
    have cost as much again and, worse, would have made the held-out number come from a
    different model than the validation number.
    """
    if model_override is None:
        spec = (spec_for or generic_spec)(seed, restart)
        model, _ = m2d.train_model(spec, train, seed * 1_000 + restart)
    else:
        model = model_override
    likelihood = m2e.training_log_likelihood(model, train)
    scored, phase = score_on(model, train, tensors, population, events)
    statistics = transition_statistics(model, train)
    row = RestartRow(seed=seed, restart=restart,
                     training_log_likelihood=likelihood,
                     alias_accuracy=float(scored["hit"].mean()),
                     phase_accuracy=phase,
                     transition_entropy=statistics["transition_entropy"],
                     stay_minus_flip=statistics["stay_minus_flip"],
                     state_occupancy=statistics["state_occupancy"],
                     belief_entropy=statistics["belief_entropy"],
                     collapsed=bool(statistics["collapsed"]),
                     checkpoint=checkpoint_hash(model))
    hits = {"primary": scored["hit"]}
    for label, (other_tensors, other_population) in (extra or {}).items():
        other, _ = score_on(model, train, other_tensors, other_population)
        hits[label] = other["hit"]
    return row, hits


# ---- the fit certificate ------------------------------------------------------------------


def certify(best_training_log_likelihood: float, tau: float) -> str:
    """CERTIFIED or UNRESOLVED_TRANSITION, from a training quantity and a threshold.

    The whole argument for this being answer-free is the signature: there is no phase
    accuracy, no transition matrix, no validation score and no event label in scope. A
    model that cannot fit the training outcomes it was given must not act as though its
    transition is known.
    """
    return "CERTIFIED" if best_training_log_likelihood >= tau else "UNRESOLVED_TRANSITION"


def choose_tau(rows: Sequence[RestartRow], solved_threshold: float = 0.9) -> dict[str, Any]:
    """Freeze tau on DEVELOPMENT by separating solved from unsolved restarts.

    `solved_threshold` is an alias-accuracy cut used only here, on development, to label
    which restarts are in fact good; tau itself is then read off the training-side
    distribution and is the only thing that travels to validation.
    """
    likelihood = np.array([r.training_log_likelihood for r in rows])
    solved = np.array([r.alias_accuracy >= solved_threshold for r in rows])
    if not solved.any() or solved.all():
        return {"tau": float(np.median(likelihood)), "separable": False,
                "solved_fraction": float(solved.mean())}
    # The largest gap between the worst solved restart and the best unsolved one.
    worst_solved = float(likelihood[solved].min())
    best_unsolved = float(likelihood[~solved].max())
    tau = float((worst_solved + best_unsolved) / 2.0)
    return {"tau": tau,
            "separable": bool(worst_solved > best_unsolved),
            "worst_solved_log_likelihood": worst_solved,
            "best_unsolved_log_likelihood": best_unsolved,
            "gap": float(worst_solved - best_unsolved),
            "solved_fraction": float(solved.mean()),
            "development_solved_threshold": solved_threshold}


# ---- procedures, all derived from the same restart table -----------------------------------


def table_by_seed(rows: Sequence[RestartRow]) -> dict[int, list[RestartRow]]:
    out: dict[int, list[RestartRow]] = {}
    for row in rows:
        out.setdefault(row.seed, []).append(row)
    for seed in out:
        out[seed].sort(key=lambda r: r.restart)
    return out


def fixed_k(rows: Sequence[RestartRow], k: int) -> RestartRow:
    prefix = list(rows)[:k]
    return max(prefix, key=lambda r: r.training_log_likelihood)


def adaptive(rows: Sequence[RestartRow], tau: float, start: int = BLOCK,
             block: int = BLOCK, k_max: int = K_MAX) -> tuple[RestartRow, int, str]:
    """Draw a block, keep the best by TRAINING likelihood, stop when it certifies.

    Reads training likelihood and the compute already consumed, and nothing else.
    """
    used = 0
    best: RestartRow | None = None
    while used < k_max:
        used = min(used + (start if used == 0 else block), k_max)
        candidate = fixed_k(rows, used)
        best = candidate if best is None or (
            candidate.training_log_likelihood > best.training_log_likelihood) else best
        if certify(best.training_log_likelihood, tau) == "CERTIFIED":
            return best, used, "CERTIFIED"
    return best, used, "UNRESOLVED_TRANSITION"


def digest_rows(rows: Sequence[RestartRow]) -> str:
    payload = "".join(f"{r.seed}:{r.restart}:{r.checkpoint};" for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
