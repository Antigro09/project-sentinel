"""M2E. Generic transition induction: shared arms, populations and compute accounting.

M2D established that the filter which reached the ceiling was initialised at the answer
-- event 0 at a stay matrix, event 1 at a flip matrix. So every initialisation here is
seed-derived and orientation-free, and the one arm that still carries the answer-oriented
matrix is present only as a calibration ceiling and is marked ineligible in code, not in
prose.

Two things this module exists to make structural rather than claimed:

    a population writes its MEMBER IDS, not just its size;
    a restart procedure reports the compute it consumed, and its baselines get the same.

    imported by m2e_genericity.py, m2e_transition.py, m2e_coupling.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m2d_core as m2d  # noqa: E402
from m2d_core import (ARTIFACTS, CLASSES, FilterSpec, PARAMETER_CEILING,  # noqa: E402
                      antisymmetric_two_state, build_tensors, checkpoint_hash,
                      digest_file, parameter_count, save_predictions, score_population,
                      stratify, write)

# Seeds never exposed to M2C or M2D. M2C/M2D used 6600-6602, 7000-7019, 8000-8019,
# 9000-9009; these ranges are disjoint from all of them.
DEV_SEEDS = tuple(range(13_000, 13_020))
VALIDATION_SEEDS = tuple(range(14_000, 14_020))

UPDATES = m2d.UPDATES          # 1024, unchanged, so M2D numbers stay comparable
NORM = float(np.linalg.norm(antisymmetric_two_state()))


# ---- generic, orientation-free initialisation -------------------------------------------


def generic_antisymmetric(seed: int, states: int) -> np.ndarray:
    """Seed-derived, matched in magnitude to the M2D perturbation, random in orientation.

    Antisymmetric ACROSS THE EVENT AXIS -- P[1] = -P[0] -- so the two event maps do not
    start interchangeable, which is the only property the M2D rule was supposed to have.
    P[0] itself is an ordinary normal draw: it names no stay, no flip, no phase and no
    event. Nothing about the environment enters except the number of latent states.
    """
    rng = np.random.default_rng(np.uint64(0xA53E) ^ np.uint64(seed))
    first = rng.normal(size=(states, states))
    block = np.stack([first, -first])
    return (block / np.linalg.norm(block) * NORM).astype(np.float32)


def initialisation_digest(perturbation: np.ndarray | None) -> str:
    if perturbation is None:
        return "none"
    return hashlib.sha256(
        np.ascontiguousarray(perturbation, dtype=np.float32).tobytes()).hexdigest()[:16]


IDENTITY = np.stack([np.eye(2), np.eye(2)]).astype(np.float32)
SWAP = np.stack([np.array([[0.0, 1.0], [1.0, 0.0]])] * 2).astype(np.float32)
ANSWER = antisymmetric_two_state()


def distances(perturbation: np.ndarray) -> dict[str, float]:
    """Distance from the three reference orientations, on the softmaxed transitions."""

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    here = softmax(np.asarray(perturbation, dtype=np.float64))
    return {
        "to_identity_pair": float(np.linalg.norm(here - softmax(IDENTITY.astype(float)))),
        "to_swap_pair": float(np.linalg.norm(here - softmax(SWAP.astype(float)))),
        "to_answer_orientation": float(np.linalg.norm(here - softmax(ANSWER.astype(float)))),
        "stay_minus_flip_diagonal": float(
            here[0].trace() - here[1].trace()),   # >0 means "event 0 stays, event 1 flips"
    }


# ---- preregistered arms -------------------------------------------------------------------


@dataclass
class Arm:
    key: str
    label: str
    spec_for: Callable[[int], FilterSpec]
    eligible: bool
    restarts: int = 1
    note: str = ""


def two_state(init: str, perturbation) -> Callable[[int], FilterSpec]:
    return lambda seed: FilterSpec("m2e", "filter", 2, init, perturbation=perturbation)


def build_arms(restarts_k: int) -> dict[str, Arm]:
    return {
        "A_exact_xor_accumulator": Arm(
            "A_exact_xor_accumulator", "exact XOR accumulator",
            lambda seed: FilterSpec("m2e", "accumulator"), eligible=False,
            note="diagnostic ceiling only"),
        "B_answer_oriented_init": Arm(
            "B_answer_oriented_init", "M2D answer-oriented initialisation",
            two_state("symmetry_broken", ANSWER), eligible=False,
            note="calibration arm: how much an encoded transition buys; never selectable"),
        "C_zero_symmetric_init": Arm(
            "C_zero_symmetric_init", "zero / symmetric 2-state initialisation",
            two_state("zero", None), eligible=True),
        "D_generic_random_single": Arm(
            "D_generic_random_single", "single generic random antisymmetric",
            lambda seed: FilterSpec("m2e", "filter", 2, "generic",
                                    perturbation=generic_antisymmetric(seed, 2)),
            eligible=True),
        "E_generic_restarts": Arm(
            "E_generic_restarts", f"fixed-K generic restarts (K={restarts_k})",
            lambda seed: FilterSpec("m2e", "filter", 2, "generic",
                                    perturbation=generic_antisymmetric(seed, 2)),
            eligible=True, restarts=restarts_k,
            note="selection by training outcome likelihood only"),
        "F_eight_state_generic": Arm(
            "F_eight_state_generic", "8-state categorical, generic random init",
            lambda seed: FilterSpec("m2e", "filter", 8, "generic",
                                    perturbation=generic_antisymmetric(seed, 8)),
            eligible=True),
        "G_generic_gru": Arm(
            "G_generic_gru", "generic GRU",
            lambda seed: FilterSpec("m2e", "gru"), eligible=True),
        "H_trained_memoryless": Arm(
            "H_trained_memoryless", "trained memoryless current-state model",
            lambda seed: FilterSpec("m2e", "memoryless"), eligible=True),
    }


# ---- training with an explicit compute ledger ---------------------------------------------


@dataclass
class ComputeLedger:
    training_runs: int = 0
    optimizer_updates: int = 0
    likelihood_evaluations: int = 0
    selection_seconds: float = 0.0
    wall_seconds: float = 0.0
    peak_parameters: int = 0
    collapsed_restarts: int = 0
    # The INDEX of the winning restart (0..K-1), not its rank -- the winner's
    # rank is always 0 and would carry no information. A spread of indices is
    # what says the later restarts are doing work.
    selected_rank: int = -1
    restart_scores: list[float] = field(default_factory=list)

    def merge(self, other: "ComputeLedger") -> None:
        self.training_runs += other.training_runs
        self.optimizer_updates += other.optimizer_updates
        self.likelihood_evaluations += other.likelihood_evaluations
        self.selection_seconds += other.selection_seconds
        self.wall_seconds += other.wall_seconds
        self.peak_parameters = max(self.peak_parameters, other.peak_parameters)
        self.collapsed_restarts += other.collapsed_restarts

    def to_dict(self) -> dict[str, Any]:
        return {"training_runs": self.training_runs,
                "optimizer_updates": self.optimizer_updates,
                "likelihood_evaluations": self.likelihood_evaluations,
                "selection_seconds": self.selection_seconds,
                "wall_seconds": self.wall_seconds,
                "peak_parameters": self.peak_parameters,
                "collapsed_restarts": self.collapsed_restarts,
                "selected_rank": self.selected_rank,
                "restart_scores": self.restart_scores}


def training_log_likelihood(model, train) -> float:
    """Mean per-step outcome log-likelihood on the TRAINING set.

    This is the only quantity restart selection may see. It contains no validation
    score, no phase label, no event semantics and no transition target.
    """
    import mlx.core as mx
    import mlx.nn as nn

    x, y, e, m, reset = m2d.pad(train)
    logits, _ = model(mx.array(x), mx.array(reset), mx.array(e))
    losses = nn.losses.cross_entropy(logits.reshape(-1, CLASSES),
                                     mx.array(y).reshape(-1), reduction="none")
    mask = mx.array(m).reshape(-1)
    value = -float((losses * mask).sum() / mask.sum())
    mx.eval(logits)
    return value


def belief_collapsed(model, train) -> bool:
    import mlx.core as mx
    x, y, e, m, reset = m2d.pad(train)
    _, belief = model(mx.array(x), mx.array(reset), mx.array(e))
    if belief is None:
        return False
    mx.eval(belief)
    belief = np.asarray(belief)
    entropy = -(belief * np.log(np.maximum(belief, 1e-12))).sum(axis=-1)
    return bool(entropy[m.astype(bool)].mean() / np.log(belief.shape[-1]) > 0.9)


def train_arm(arm: Arm, train, seed: int, updates: int = UPDATES,
              event_transform=None) -> tuple[Any, ComputeLedger, dict[str, Any]]:
    """One arm, one seed. For K > 1 every restart is trained and every restart is charged."""
    started = time.perf_counter()
    ledger = ComputeLedger()
    candidates = []
    for restart in range(arm.restarts):
        spec = arm.spec_for(seed if arm.restarts == 1 else seed * 1_000 + restart)
        model, count = m2d.train_model(spec, train, seed * 1_000 + restart,
                                       event_transform=event_transform)
        ledger.training_runs += 1
        ledger.optimizer_updates += updates
        ledger.peak_parameters = max(ledger.peak_parameters, count)
        selection_started = time.perf_counter()
        score = training_log_likelihood(model, train)
        ledger.likelihood_evaluations += 1
        ledger.selection_seconds += time.perf_counter() - selection_started
        if belief_collapsed(model, train):
            ledger.collapsed_restarts += 1
        candidates.append((score, restart, model, count, spec))
    ranked = sorted(candidates, key=lambda c: -c[0])
    ledger.restart_scores = [round(c[0], 6) for c in candidates]
    ledger.selected_rank = int(ranked[0][1])
    ledger.wall_seconds = time.perf_counter() - started
    score, restart, model, count, spec = ranked[0]
    identity = {"model_class": type(model).__name__,
                "temporal_mechanism": m2d.MECHANISM[spec.kind],
                "checkpoint_hash": checkpoint_hash(model),
                "initialisation_rule": spec.initialization_rule,
                "initialisation_digest": initialisation_digest(spec.perturbation),
                "trainable_parameters": count,
                "selected_restart": restart,
                "training_log_likelihood": score,
                "eligible": arm.eligible}
    if spec.perturbation is not None and spec.states == 2:
        identity["initial_transition_distances"] = distances(spec.perturbation)
    return model, ledger, identity


# ---- populations that record their members ------------------------------------------------


def population_manifest(population, label: str, layouts) -> dict[str, Any]:
    """Layouts, per-layout counts, the member table's digest -- and the table itself is
    written to the npz beside it. A size without member IDs is not reproducible."""
    members = np.array([[population.states[r.self_index].layout,
                         len(population.states[r.self_index].route),
                         r.action, r.pair_id] for r in population.rows], dtype=np.int32)
    routes = ["|".join(str(a) for a in population.states[r.self_index].route)
              for r in population.rows]
    digest = hashlib.sha256(
        members.tobytes() + "\n".join(routes).encode()).hexdigest()[:16]
    per_layout = {int(l): int((members[:, 0] == l).sum())
                  for l in sorted(set(members[:, 0].tolist()))}
    return {"label": label, "layouts": [int(l) for l in layouts],
            "rows_per_layout": per_layout, "rows": int(len(members)),
            "pairs": len({r.pair_id for r in population.rows}),
            "alias_classes": len({r.alias_class for r in population.rows}),
            "states": len(population.states), "member_digest": digest,
            "member_table": members, "member_routes": routes}


def episode_manifest(trajectories, label: str, layouts, seed: int) -> dict[str, Any]:
    """Every episode's identity: layout, index within layout, length."""
    counts: dict[int, int] = {}
    rows = []
    for item in trajectories:
        layout = int(item["layout"])
        index = counts.get(layout, 0)
        counts[layout] = index + 1
        rows.append((layout, index, len(item["y"])))
    table = np.array(rows, dtype=np.int32)
    return {"label": label, "layouts": [int(l) for l in layouts],
            "collection_seed": seed, "episodes": len(rows),
            "episode_table": table,
            "episode_digest": hashlib.sha256(table.tobytes()).hexdigest()[:16]}
