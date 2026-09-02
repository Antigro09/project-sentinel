"""M2D. Shared machinery: arm identity, the alias population, filters, coupling.

The M2C phase reported a row as "learned event + selected learned filter" while the
code ran an exact XOR accumulator; the selected filter was imported and never called,
and its name reached the artifact as a string field. A string cannot be wrong in a way
the runner notices, so the fix here is structural: every arm is an object that reports
its own identity, and the identity fields are read off the object that actually
computes. `ArmIdentity.temporal_mechanism` comes from the class of the thing holding
the recurrence, not from a literal at the top of the file.

Everything else in this module exists to let the same population and the same head be
reused across sections C to H, so that "matched populations, supervision, evidence and
compute" is a property of the code rather than a claim in a report.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS, GRID  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from structured_calibration import CLASSES, DISPLACEMENTS, collect  # noqa: E402
from belief_factorization import build_dataset, public_event, sequence_features  # noqa: E402

ARTIFACTS = REPO / "artifacts/shwm/scale1"

# Column layout of `sequence_features`, asserted rather than assumed. An earlier guard
# in the T2 suite used 12:16 for the query one-hot -- which is goal direction and
# previous action -- and fired on the honest pipeline for a fortnight.
NEIGHBOUR_BLOCKED = slice(0, 4)
NEIGHBOUR_SWITCH = slice(4, 8)
GOAL_DIRECTION = slice(8, 12)
PREVIOUS_ACTION = slice(12, 17)
QUERY_ACTION = slice(17, 21)
BLOCKED_SCALAR = 21
POSITION = slice(22, 24)
RESET_VALUE = 24
RESET_FLAG = 25
FEATURE_WIDTH = 26

HIDDEN = 128
PARAMETER_CEILING = 250_000
UPDATES = 1024

TRAIN_LAYOUTS = tuple(range(61_000, 61_040))
DETECTOR_TEST_LAYOUTS = tuple(range(81_000, 81_020))
ALIAS_LAYOUTS = tuple(range(90_000, 90_010))
HELD_OUT_LAYOUTS = tuple(range(95_000, 95_020))


def check_feature_layout() -> None:
    """A positive control on the column constants above, run at import of any driver."""
    trajectories = collect([TRAIN_LAYOUTS[0]], 1, 6, CANONICAL_APPEARANCE_SEED, 11)
    x = sequence_features(trajectories[0], [2] * len(trajectories[0]["rows"]))
    assert x.shape[1] == FEATURE_WIDTH, x.shape
    assert np.allclose(x[0, QUERY_ACTION], [0, 0, 1, 0]), x[0, QUERY_ACTION]
    assert x[0, RESET_FLAG] == 1.0 and x[1:, RESET_FLAG].sum() == 0.0


# ---- arm identity ---------------------------------------------------------------------


@dataclass
class ArmIdentity:
    """What actually ran. Filled in from the objects that compute, never from a label."""
    arm_id: str
    event_source: str            # true | learned_hard | learned_soft | constant | none
    temporal_mechanism: str      # exact_accumulator | learned_filter | gru | none
    model_class: str             # the Python class of the recurrence holder
    checkpoint_hash: str
    initialization_rule: str
    trainable_parameters: int
    supervision: str
    input_fields: tuple[str, ...]
    seed: int
    population: str
    metric: str
    query_budget: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["input_fields"] = list(self.input_fields)
        return out


def checkpoint_hash(model) -> str:
    """Content hash of the trained parameters, so two arms cannot share an identity."""
    from mlx.utils import tree_flatten
    digest = hashlib.sha256()
    for name, value in sorted(tree_flatten(model.parameters()), key=lambda kv: kv[0]):
        digest.update(name.encode())
        digest.update(np.asarray(value, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def parameter_count(model) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


# ---- population -----------------------------------------------------------------------


@dataclass
class AliasRow:
    """One directed scoring row: a member of an alias pair under one proposed action."""
    pair_id: int
    alias_class: str
    layout: int
    action: int
    self_index: int            # index into `states`
    partner_index: int
    target_class: int
    other_class: int
    crossings_self: int
    crossings_partner: int
    step: int
    steps_since_change_self: int
    polarity_self: int


@dataclass
class Population:
    states: list[Any]                       # VisibleState, plus route metadata
    rows: list[AliasRow]
    route_rows: list[list[dict[str, Any]]]  # per state, the public rows along its route
    crossing_steps: list[tuple[int, ...]]   # per state, the steps at which polarity flipped

    def summary(self) -> dict[str, Any]:
        return {"states": len(self.states), "rows": len(self.rows),
                "pairs": len({r.pair_id for r in self.rows}),
                "alias_classes": len({r.alias_class for r in self.rows}),
                "layouts": sorted({r.layout for r in self.rows})}


def to_class(signature: float, origin: tuple[int, int]) -> int:
    cell = int(signature)
    delta = (cell // GRID - origin[0], cell % GRID - origin[1])
    return DISPLACEMENTS.index(delta) if delta in DISPLACEMENTS else CLASSES - 1


def replay_route(layout: int, route: Sequence[int]) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Rebuild the PUBLIC rows along a route, plus the steps at which polarity flipped.

    Polarity is read only to record when it changed, which is evaluator metadata used
    for stratification; it never enters a feature vector built from these rows.
    """
    from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="m2d")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    adapter.reset(layout, f"appearance:{CANONICAL_APPEARANCE_SEED}")
    level = adapter._require()
    switches = {tuple(int(v) for v in c) for c in level.switches}
    walls = np.asarray(level.walls, dtype=bool)
    goal = tuple(int(v) for v in level.markers[adapter._goal_marker])

    rows: list[dict[str, Any]] = []
    changes: list[int] = []
    previous_action = -1
    previous_polarity = None
    for step_index in range(len(route) + 1):
        truth = adapter.snapshot().reveal("evaluator")
        polarity = int(truth["polarity"])
        if previous_polarity is not None and polarity != previous_polarity:
            changes.append(step_index)
        previous_polarity = polarity
        rows.append({"position": tuple(int(v) for v in truth["position"]),
                     "switches": switches, "walls": walls, "goal": goal,
                     "previous_action": previous_action,
                     "blocked": float(truth["last_blocked"]),
                     "polarity": polarity})
        if step_index == len(route):
            break
        previous_action = int(route[step_index])
        adapter.step(previous_action, gate.authorize_evaluator(previous_action, "m2d"))
    return rows, tuple(changes)


def build_population(layouts: Sequence[int] = ALIAS_LAYOUTS, depth: int = 6) -> Population:
    """Exact complete-public-packet alias pairs: same packet, same proposed action,
    different history, different hidden phase, different public next outcome."""
    from collections import defaultdict
    from alias_audit import enumerate_states

    states = enumerate_states(list(layouts), depth)
    classes: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        classes[state.key("V2_agent_visible")].append(index)

    route_rows: list[list[dict[str, Any]]] = [None] * len(states)   # type: ignore[list-item]
    crossing_steps: list[tuple[int, ...]] = [()] * len(states)
    rows: list[AliasRow] = []
    pair_id = 0
    needed: set[int] = set()
    for key, members in classes.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = states[members[i]], states[members[j]]
                if a.polarity == b.polarity:
                    continue
                used = False
                for action in ACTIONS:
                    if a.successors[action] == b.successors[action]:
                        continue
                    for me_i, other_i in ((members[i], members[j]), (members[j], members[i])):
                        me, other = states[me_i], states[other_i]
                        target = to_class(me.successors[action], me.position)
                        alternative = to_class(other.successors[action], me.position)
                        if target == alternative:
                            continue
                        rows.append(AliasRow(
                            pair_id=pair_id, alias_class=key, layout=me.layout,
                            action=action, self_index=me_i, partner_index=other_i,
                            target_class=target, other_class=alternative,
                            crossings_self=me.crossings, crossings_partner=other.crossings,
                            step=me.step, steps_since_change_self=-1,
                            polarity_self=me.polarity))
                        needed.update((me_i, other_i))
                        used = True
                if used:
                    pair_id += 1

    for index in sorted(needed):
        route_rows[index], crossing_steps[index] = replay_route(
            states[index].layout, states[index].route)

    for row in rows:
        changes = crossing_steps[row.self_index]
        row.steps_since_change_self = (row.step - changes[-1]) if changes else -1
    return Population(states, rows, route_rows, crossing_steps)


# ---- features over an alias route -------------------------------------------------------


class RouteFeatures:
    """Per-state sequence features with the query one-hot left empty, filled per action.

    The query action is a property of the question being asked at the final step, not of
    the history, so it is written into every step identically. That matches training,
    where the query column is drawn independently of the action actually taken.
    """

    def __init__(self, population: Population) -> None:
        self.population = population
        self.base: dict[int, np.ndarray] = {}
        self.events: dict[int, np.ndarray] = {}
        for index, rows in enumerate(population.route_rows):
            if rows is None:
                continue
            block = sequence_features({"rows": rows}, [0] * len(rows)).copy()
            block[:, QUERY_ACTION] = 0.0
            self.base[index] = block
            self.events[index] = np.array(
                [public_event(rows[t - 1] if t else None, rows[t]) for t in range(len(rows))],
                dtype=np.float32)

    def sequence(self, state_index: int, action: int) -> np.ndarray:
        block = self.base[state_index].copy()
        block[:, QUERY_ACTION.start + action] = 1.0
        return block

    def reset_bit(self, state_index: int) -> float:
        return float(self.base[state_index][0, RESET_VALUE])


# ---- models ------------------------------------------------------------------------------


def antisymmetric_two_state() -> np.ndarray:
    """The frozen symmetry-breaking perturbation, selected on M2C development seeds.

    It is antisymmetric in the state indices and identical for the two event values up
    to sign, so it names no phase, no stay/flip assignment and no event mapping. What it
    does is make the two event maps non-interchangeable at initialization, which is the
    basin the collapsed seed settled in.
    """
    return np.array([[[0.5, -0.5], [-0.5, 0.5]],
                     [[-0.5, 0.5], [0.5, -0.5]]], dtype=np.float32)


@dataclass
class FilterSpec:
    arm_id: str
    kind: str                     # filter | gru | memoryless | accumulator
    states: int = 2
    init: str = "symmetry_broken"  # default | symmetry_broken | permuted | sign_reversed
                                   # | random_antisymmetric | zero | event_permuted
    gauge: str = "reset_onehot"    # reset_onehot | learned
    perturbation: np.ndarray | None = None

    @property
    def initialization_rule(self) -> str:
        if self.kind != "filter":
            return "standard mlx defaults"
        return f"normal(0,0.05) + {self.init} perturbation; initial belief gauge={self.gauge}"


def make_model(spec: FilterSpec, width: int, seed: int):
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)
    states = spec.states

    class LearnedFilter(nn.Module):
        """belief_t = normalise(belief_{t-1} @ T_e). `event` may be a probability, in
        which case the propagated matrix is p(C=0) T_0 + p(C=1) T_1 exactly."""

        def __init__(self) -> None:
            super().__init__()
            if spec.init == "default":
                self.logits = mx.random.normal((2, states, states)) * 0.5
            elif spec.init == "zero":
                self.logits = mx.zeros((2, states, states))
            else:
                base = mx.random.normal((2, states, states)) * 0.05
                perturbation = spec.perturbation
                if perturbation is None:
                    perturbation = np.zeros((2, states, states), dtype=np.float32)
                self.logits = base + mx.array(
                    np.ascontiguousarray(perturbation, dtype=np.float32))
            self.initial = nn.Linear(1, states)
            self.head = nn.Sequential(nn.Linear(width + states, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, CLASSES))

        def initial_belief(self, reset):
            """The gauge: how the public reset stripe becomes an initial belief.

            `reset_onehot` fixes state 0 to polarity 0. That is a real modelling
            choice and not a neutral one -- it pre-aligns the latent state space with
            the quantity being recovered -- so it is a control axis here rather than
            an assumption."""
            if spec.gauge == "learned" or states != 2:
                return mx.softmax(self.initial(reset), axis=-1)
            if spec.gauge == "reset_onehot_swapped":
                return mx.concatenate([reset, 1.0 - reset], axis=-1)
            return mx.concatenate([1.0 - reset, reset], axis=-1)

        def __call__(self, z, reset, event):
            transition = mx.softmax(self.logits, axis=-1)
            belief = self.initial_belief(reset)
            beliefs = []
            for t in range(z.shape[1]):
                ev = event[:, t][:, None, None]
                matrix = transition[0][None] * (1.0 - ev) + transition[1][None] * ev
                belief = mx.sum(belief[:, :, None] * matrix, axis=1)
                belief = belief / mx.maximum(mx.sum(belief, axis=-1, keepdims=True), 1e-9)
                beliefs.append(belief)
            stacked = mx.stack(beliefs, axis=1)
            return self.head(mx.concatenate([z, stacked], axis=-1)), stacked

    class Gru(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(width + 1, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, CLASSES)

        def __call__(self, z, reset, event):
            h = self.gru(nn.relu(self.project(
                mx.concatenate([z, event[:, :, None]], axis=-1))))
            return self.head(h), h

    class Memoryless(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, CLASSES))

        def __call__(self, z, reset, event):
            return self.net(z), None

    class ExactAccumulator(nn.Module):
        """The ceiling: parity of the supplied events, XOR the public reset stripe."""

        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Sequential(nn.Linear(width + 2, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, CLASSES))

        def __call__(self, z, reset, event):
            """Written as the parity RECURSION rather than cumsum-mod-2.

            For hard events the two are identical. For a probability the recursion is
            the correct soft parity -- p_t = p_{t-1}(1-e_t) + (1-p_{t-1})e_t, which is
            the fixed XOR automaton propagated by the posterior mixture -- whereas
            remainder(cumsum(p), 2) is not a probability of anything.
            """
            batch, length, _ = z.shape
            phase = reset[:, 0]
            columns = []
            for t in range(length):
                if t:
                    e = event[:, t]
                    phase = phase * (1.0 - e) + (1.0 - phase) * e
                columns.append(phase)
            stacked = mx.stack(columns, axis=1)
            onehot = mx.stack([1.0 - stacked, stacked], axis=-1)
            return self.head(mx.concatenate([z, onehot], axis=-1)), onehot

    return {"filter": LearnedFilter, "gru": Gru,
            "memoryless": Memoryless, "accumulator": ExactAccumulator}[spec.kind]()


MECHANISM = {"filter": "learned_filter", "gru": "gru",
             "memoryless": "none", "accumulator": "exact_accumulator"}


def pad(items):
    length = max(len(i["y"]) for i in items)
    width = items[0]["x"].shape[1]
    x = np.zeros((len(items), length, width), dtype=np.float32)
    y = np.zeros((len(items), length), dtype=np.int32)
    e = np.zeros((len(items), length), dtype=np.float32)
    m = np.zeros((len(items), length), dtype=np.float32)
    reset = np.zeros((len(items), 1), dtype=np.float32)
    for i, item in enumerate(items):
        n = len(item["y"])
        x[i, :n], y[i, :n], e[i, :n], m[i, :n] = item["x"], item["y"], item["events"], 1.0
        reset[i, 0] = float(item["phases"][0])
    return x, y, e, m, reset


def train_model(spec: FilterSpec, train, seed: int, event_transform=None):
    """Train one arm. `event_transform` rewrites the TRAINING events only (section C
    control 4 permutes labels and must relabel evaluation to match)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    x, y, e, m, reset = pad(train)
    if event_transform is not None:
        e = event_transform(e)
    model = make_model(spec, x.shape[2], seed)
    mx.eval(model.parameters())
    count = parameter_count(model)
    assert count <= PARAMETER_CEILING, (spec.arm_id, count)
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)
    for _ in range(UPDATES):
        pick = rng.integers(0, len(x), min(32, len(x)))
        xb, yb, eb = mx.array(x[pick]), mx.array(y[pick]), mx.array(e[pick])
        mb, rb = mx.array(m[pick]), mx.array(reset[pick])

        def loss_fn(mo):
            logits, _ = mo(xb, rb, eb)
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, CLASSES), yb.reshape(-1), reduction="none")
            return (losses * mb.reshape(-1)).sum() / mb.sum()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    return model, count


def held_out_accuracy(model, test) -> float:
    import mlx.core as mx
    xt, yt, et, mt, rt = pad(test)
    logits, _ = model(mx.array(xt), mx.array(rt), mx.array(et))
    mx.eval(logits)
    predicted = np.asarray(logits).argmax(axis=-1)
    valid = mt.astype(bool)
    return float((predicted[valid] == yt[valid]).mean())


# ---- evaluation on the alias population ---------------------------------------------------


@dataclass
class AliasTensors:
    """The alias population padded once, so every arm is scored on identical tensors.

    Rebuilding the sequences per arm was the obvious way to write this and it is the
    wrong one: a hundred and twenty arm-seeds would each re-derive the population, and
    any drift in that derivation would show up as a difference between arms.
    """
    keys: list[tuple[int, int]]
    z: np.ndarray            # (K, L, W)
    reset: np.ndarray        # (K, 1)
    final: np.ndarray        # (K,)
    lengths: np.ndarray      # (K,)
    events_true: np.ndarray  # (K, L)
    row_key: np.ndarray      # (R,) -> index into keys
    target: np.ndarray       # (R,)
    other: np.ndarray        # (R,)


def build_tensors(population: Population, features: RouteFeatures) -> AliasTensors:
    keys = sorted({(r.self_index, r.action) for r in population.rows})
    position = {key: i for i, key in enumerate(keys)}
    length = max(len(features.base[i]) for i, _ in keys)
    z = np.zeros((len(keys), length, FEATURE_WIDTH), dtype=np.float32)
    events = np.zeros((len(keys), length), dtype=np.float32)
    reset = np.zeros((len(keys), 1), dtype=np.float32)
    final = np.zeros(len(keys), dtype=int)
    lengths = np.zeros(len(keys), dtype=int)
    for k, (index, action) in enumerate(keys):
        sequence = features.sequence(index, action)
        n = len(sequence)
        z[k, :n] = sequence
        events[k, :n] = features.events[index]
        reset[k, 0] = sequence[0, RESET_VALUE]
        final[k] = n - 1
        lengths[k] = n
    return AliasTensors(
        keys=keys, z=z, reset=reset, final=final, lengths=lengths, events_true=events,
        row_key=np.array([position[(r.self_index, r.action)] for r in population.rows]),
        target=np.array([r.target_class for r in population.rows]),
        other=np.array([r.other_class for r in population.rows]))


def score_population(model, tensors: AliasTensors, events: np.ndarray | None = None,
                     batch: int = 1024) -> dict[str, np.ndarray]:
    """Does the model rank its own outcome above its partner's?

    Identical packets and identical proposed action, so a model with no temporal state
    receives identical input for the two directions of a pair and must tie at exactly
    0.5 over them. That is the property that makes the baseline a construction rather
    than an estimate.
    """
    import mlx.core as mx

    if events is None:
        events = tensors.events_true
    out = np.zeros((len(tensors.keys), CLASSES), dtype=np.float32)
    belief_final: np.ndarray | None = None
    for start in range(0, len(tensors.keys), batch):
        stop = min(start + batch, len(tensors.keys))
        logits, belief = model(mx.array(tensors.z[start:stop]),
                               mx.array(tensors.reset[start:stop]),
                               mx.array(events[start:stop]))
        mx.eval(logits)
        logits = np.asarray(logits)
        index = tensors.final[start:stop]
        out[start:stop] = logits[np.arange(stop - start), index]
        if belief is not None:
            mx.eval(belief)
            belief = np.asarray(belief)
            if belief_final is None:
                belief_final = np.zeros((len(tensors.keys), belief.shape[-1]),
                                        dtype=np.float32)
            belief_final[start:stop] = belief[np.arange(stop - start), index]

    per_row = out[tensors.row_key]
    margin = (per_row[np.arange(len(per_row)), tensors.target]
              - per_row[np.arange(len(per_row)), tensors.other])
    probability = 1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0)))
    return {"hit": np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0)),
            "margin": margin.astype(np.float64),
            "nll": -np.log(np.maximum(probability, 1e-12)),
            "brier": (1.0 - probability) ** 2,
            "belief": (belief_final[tensors.row_key] if belief_final is not None
                       else np.zeros((len(per_row), 1), dtype=np.float32))}


def stratify(population: Population) -> dict[str, np.ndarray]:
    """Row-level strata. `changes` is the PAIR minimum, deliberately.

    Stratifying on a row's own crossing count would split the two directions of every
    pair into different strata: aliases share a layout, so they share an initial
    polarity, so their crossing counts always differ in parity. That would destroy the
    construction that pins the memoryless arm at exactly 0.5 and the gate would be
    comparing against a moving baseline. The pair minimum keeps both directions
    together and still means "both histories have had at least n changes".
    """
    rows = population.rows
    return {
        "changes": np.array([min(r.crossings_self, r.crossings_partner) for r in rows]),
        "changes_self": np.array([r.crossings_self for r in rows]),
        "step": np.array([r.step for r in rows]),
        "since_change": np.array([r.steps_since_change_self for r in rows]),
        "action": np.array([r.action for r in rows]),
        "layout": np.array([r.layout for r in rows]),
        "pair": np.array([r.pair_id for r in rows]),
        # A stable digest, not `hash`: Python string hashing is salted per process
        # and the bootstrap grouping must be reproducible across runs.
        "alias_class": np.array([int(hashlib.sha256(r.alias_class.encode())
                                     .hexdigest()[:15], 16) for r in rows]),
    }


def summarise_metric(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "sd": float(values.std()), "minimum": float(values.min()),
            "p10": float(np.percentile(values, 10))}


# ---- statistics ---------------------------------------------------------------------------


def hierarchical_paired_interval(a: np.ndarray, b: np.ndarray, seeds: np.ndarray,
                                 layouts: np.ndarray, classes: np.ndarray,
                                 resamples: int = 4000, seed: int = 99,
                                 mask: np.ndarray | None = None,
                                 chunk: int = 500) -> dict[str, float]:
    """Paired resampling by validation seed -> layout -> alias class.

    Rows inside one alias class are not independent, and classes inside a layout are
    not either, so the resample is nested. Because `a` and `b` are aligned row-wise the
    statistic is the mean of the paired differences, which means a resample only needs
    per-group sums and counts -- no row indices are ever gathered, and the whole thing
    is multinomial weights times two precomputed tables.
    """
    if mask is not None:
        a, b, seeds, layouts, classes = (v[mask] for v in (a, b, seeds, layouts, classes))
    if len(a) == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "excludes_zero": False, "rows": 0}
    difference = a - b
    seed_values = np.unique(seeds)
    layout_values = np.unique(layouts)
    class_values = np.unique(classes)
    class_position = {c: i for i, c in enumerate(class_values)}
    sums = np.zeros((len(seed_values), len(layout_values), len(class_values)))
    counts = np.zeros_like(sums)
    seed_position = {v: i for i, v in enumerate(seed_values)}
    layout_position = {v: i for i, v in enumerate(layout_values)}
    np.add.at(sums, (np.array([seed_position[v] for v in seeds]),
                     np.array([layout_position[v] for v in layouts]),
                     np.array([class_position[v] for v in classes])), difference)
    np.add.at(counts, (np.array([seed_position[v] for v in seeds]),
                       np.array([layout_position[v] for v in layouts]),
                       np.array([class_position[v] for v in classes])), 1.0)

    # Which classes actually occur in each layout; a class absent from a layout must not
    # be resampled into it, which is why the multinomial is drawn per layout.
    present = [np.flatnonzero(counts.sum(axis=0)[l] > 0) for l in range(len(layout_values))]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    S, L = len(seed_values), len(layout_values)
    done = 0
    while done < resamples:
        block = min(chunk, resamples - done)
        ms = rng.multinomial(S, np.full(S, 1.0 / S), size=block).astype(np.float32)
        ml = rng.multinomial(L, np.full(L, 1.0 / L), size=(block, S)).astype(np.float32)
        weight = np.zeros((block, S, L, len(class_values)), dtype=np.float32)
        for l in range(L):
            members = present[l]
            if len(members) == 0:
                continue
            mc = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)),
                                 size=(block, S)).astype(np.float32)
            weight[:, :, l, members] = mc
        weight *= (ms[:, :, None, None] * ml[:, :, :, None])
        numerator = (weight * sums[None]).sum(axis=(1, 2, 3))
        denominator = (weight * counts[None]).sum(axis=(1, 2, 3))
        draws[done:done + block] = numerator / np.maximum(denominator, 1e-9)
        done += block
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"delta": float(difference.mean()), "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0), "rows": int(len(a))}


def fit_state_assignment(model, train) -> np.ndarray | None:
    """Map anonymous latent states to polarity, fitted on TRAINING beliefs only."""
    import mlx.core as mx
    x, y, e, m, reset = pad(train)
    _, belief = model(mx.array(x), mx.array(reset), mx.array(e))
    if belief is None:
        return None
    mx.eval(belief)
    belief = np.asarray(belief)
    phases = np.zeros_like(y)
    for i, item in enumerate(train):
        phases[i, :len(item["phases"])] = item["phases"]
    mask = m.astype(bool)
    argmax = belief.argmax(axis=-1)
    assignment = np.zeros(belief.shape[-1], dtype=int)
    for state in range(belief.shape[-1]):
        picked = mask & (argmax == state)
        assignment[state] = int(round(float(phases[picked].mean()))) if picked.any() else 0
    return assignment


def save_predictions(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Freeze every arm's per-row scores so a metric can be recomputed without retraining.

    Section B of the M2D specification asks for exactly this and the M2C artifact could
    not supply it: there were no stored predictions, so the only way to find out which
    arm had run was to read the source. A hit vector on disk settles that question
    without anyone having to trust a label.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def digest_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
