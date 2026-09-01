"""H. Slot resolution and the hidden-state causal chain.

The question is not "which slot geometry scores best on hidden phase". It is
which link of this chain breaks:

    representation -> switch-event detection -> temporal accumulation
                   -> hidden-phase belief -> same-action outcome prediction

Three design choices make the links separable, and each exists because the
obvious alternative would have confounded two explanations.

*Two history windows, not one.* Initial polarity is drawn on the reset frame as
a one-pixel stripe and never rendered again. With a full-episode window that
stripe is inside the window, so phase is identifiable from frames alone and a
failure indicts the representation. With the short window the stripe is outside
it, so phase is *not* identifiable from frames at any resolution and a failure
indicts the window. Running one window only would have made those two answers
indistinguishable -- which is what happened at S1.2.

*A cell-aligned geometry alongside the fine one.* The game is a 12x12 grid at 2
pixels per cell, so 8x8 slots are 3 pixels -- one and a half cells -- and
straddle boundaries. A 12x12 arm is added so a fine-grid failure can be charged
to resolution or to misalignment rather than to both at once.

*Paired states built by construction, not by search.* Under polarity 1 the
action deltas are negated, so inverting an action sequence reproduces the same
position trajectory from the opposite phase. That yields pairs with byte-identical
post-reset frames and opposite hidden phase, both genuinely reachable because
both are real rollouts. On those pairs a frame-only probe must sit at chance --
not as a weak expectation but as a fact about the construction -- and any score
above chance is a leak.

    .venv-shwm/bin/python experiments/shwm/slot_resolution_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS,
    DELTAS,
    ProceduralVisualV2Adapter,
    build_level_v2,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder  # noqa: E402
from sentinel.wm.backbones import FROZEN_CANDIDATES  # noqa: E402
from sentinel.wm.slot_geometry import (  # noqa: E402
    GEOMETRIES,
    Geometry,
    available_geometries,
    backbone_slots,
    geometry_report,
    raw_slots,
    random_projection_slots,
)
from sentinel.wm.versioning import digest_of  # noqa: E402

from feature_sufficiency import PENALTIES, RandomFourier, ridge_fit  # noqa: E402

READOUT_WIDTH = 256
RFF_WIDTH = 1024
RFF_BANDWIDTHS = (0.05, 0.2)
SHORT_WINDOW = 3
BOOTSTRAP_RESAMPLES = 2000
INVERSE_ACTION = {a: next(b for b, e in DELTAS.items() if e == (-d[0], -d[1]))
                  for a, d in DELTAS.items()}


# ---- 1. collection ----------------------------------------------------------------------


def _signature(adapter: ProceduralVisualV2Adapter) -> float:
    return float(adapter.probes().values["observable_signature"])


def collect_episodes(layouts: Sequence[int], steps: int, tag: str) -> list[dict[str, Any]]:
    """Trajectories with every ground-truth quantity the causal chain needs."""
    gate = AuthorityGate(gate_id=f"slotaudit-{tag}")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    samples: list[dict[str, Any]] = []
    for layout in layouts:
        adapter.reset(layout)
        level = adapter._require()
        switches = set(level.switches)
        initial = int(adapter._polarity)
        previous_position = adapter._position
        previous_action = -1
        previous_polarity = initial
        for step in range(steps):
            truth = adapter.snapshot().reveal("evaluator")
            position = tuple(int(v) for v in truth["position"])
            polarity = int(truth["polarity"])
            crossings = int(truth["switch_crossings"])
            snapshot = adapter.snapshot()

            successors = {}
            for action in ACTIONS:
                adapter.restore(snapshot)
                adapter.step(action, gate.authorize_evaluator(action, "intervention"))
                successors[action] = _signature(adapter)
            adapter.restore(snapshot)

            moved = position != previous_position
            movement = (position[0] - previous_position[0], position[1] - previous_position[1])
            nearest = min(switches, key=lambda c: abs(c[0] - position[0]) + abs(c[1] - position[1]))
            samples.append(
                {
                    "layout": layout,
                    "step": step,
                    "observation": adapter._observation(),
                    "frame": adapter.frame().copy(),
                    "initial_polarity": initial,
                    "polarity": polarity,
                    "crossings": crossings,
                    "crossed_now": int(step > 0 and moved and position in switches),
                    "polarity_changed_now": int(step > 0 and polarity != previous_polarity),
                    "on_switch": int(position in switches),
                    "nearest_switch_row": float(nearest[0]),
                    "nearest_switch_col": float(nearest[1]),
                    "agent_row": float(position[0]),
                    "agent_col": float(position[1]),
                    "moved": int(moved),
                    "move_row": float(movement[0]),
                    "move_col": float(movement[1]),
                    "previous_action": previous_action,
                    "successors": [successors[a] for a in ACTIONS],
                }
            )
            previous_position, previous_polarity = position, polarity
            action = ACTIONS[(step * 3 + layout) % len(ACTIONS)]
            previous_action = action
            if adapter.step(action, gate.authorize_evaluator(action, "rollout")).terminated:
                break
    return samples


def collect_paired(layouts: Sequence[int], steps: int) -> list[dict[str, Any]]:
    """Pairs sharing a frame AND an observation content digest, with opposite phase.

    An earlier construction reached the opposite phase by changing the phase seed
    and inverting the actions. The frames matched, but `LevelV2.digest` embeds
    `initial_polarity`, so `environment_version` -- and therefore the observation
    content digest -- differed between the two members. The pair was
    distinguishable through a channel that had nothing to do with the pixels.

    This construction stays inside one level and reaches the same cell by two
    routes, one crossing an odd number of switches and one an even number. Same
    level, same position, so the frame and the content digest are identical and
    only the hidden phase differs. Both members are genuine rollouts, so the pair
    is reachable by construction rather than by assertion.
    """
    gate = AuthorityGate(gate_id="slotaudit-paired")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    pairs: list[dict[str, Any]] = []

    for layout in layouts:
        adapter.reset(layout)
        start_state = (adapter._position, int(adapter._polarity))
        # Breadth-first over (position, polarity); record one route to each.
        routes: dict[tuple[Any, int], list[int]] = {start_state: []}
        frontier = [start_state]
        for _ in range(steps):
            nxt = []
            for state in frontier:
                for action in ACTIONS:
                    adapter.reset(layout)
                    for previous in routes[state]:
                        adapter.step(previous, gate.authorize_evaluator(previous, "bfs"))
                    result = adapter.step(action, gate.authorize_evaluator(action, "bfs"))
                    reached = (adapter._position, int(adapter._polarity))
                    if reached not in routes and not result.terminated:
                        routes[reached] = routes[state] + [action]
                        nxt.append(reached)
            frontier = nxt
            if not frontier:
                break

        by_position: dict[Any, dict[int, list[int]]] = {}
        for (position, polarity), route in routes.items():
            by_position.setdefault(position, {})[polarity] = route
        for position, options in by_position.items():
            if len(options) < 2:
                continue
            members = []
            for polarity in (0, 1):
                adapter.reset(layout)
                for action in options[polarity]:
                    adapter.step(action, gate.authorize_evaluator(action, "paired"))
                snapshot = adapter.snapshot()
                successors = {}
                for candidate in ACTIONS:
                    adapter.restore(snapshot)
                    adapter.step(candidate, gate.authorize_evaluator(candidate, "paired-int"))
                    successors[candidate] = _signature(adapter)
                adapter.restore(snapshot)
                truth = adapter.snapshot().reveal("evaluator")
                members.append({
                    "frame": adapter.frame().copy(),
                    "observation": adapter._observation(),
                    "content_digest": adapter._observation().content_digest,
                    "polarity": int(truth["polarity"]),
                    "position": tuple(int(v) for v in truth["position"]),
                    "route": list(options[polarity]),
                    "previous_action": options[polarity][-1] if options[polarity] else -1,
                    "successors": [successors[a] for a in ACTIONS],
                })
            a, b = members
            if a["polarity"] == b["polarity"]:
                continue
            if not np.array_equal(a["frame"], b["frame"]):
                continue
            pairs.append({
                "layout": layout,
                "position": position,
                "a": a,
                "b": b,
                "frames_identical": True,
                "content_digests_identical": a["content_digest"] == b["content_digest"],
            })
    return pairs


# ---- 2. representations -----------------------------------------------------------------


def encode_tokens(samples, config, encoder_ids) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """One pass per backbone over every observation. Preprocessing is untouched."""
    tokens: dict[str, list[np.ndarray]] = {}
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
        rows = [encoder.encode_visual_tokens(s["observation"], s["frame"]) for s in samples]
        elapsed = time.perf_counter() - started
        timings[encoder_id] = {
            "observations": len(samples),
            "seconds": elapsed,
            "observations_per_second": len(samples) / elapsed,
            "visual_tokens": int(rows[0].shape[0]),
            "token_width": int(rows[0].shape[1]),
        }
        tokens[encoder_id] = rows
        encoder.release()
    return tokens, timings


def cnn_slots_factory(geometry: Geometry) -> Callable[[np.ndarray], np.ndarray]:
    """A learned CNN whose stride lands exactly on the requested grid."""
    import mlx.core as mx
    import mlx.nn as nn

    stride_plan = {4: (2, 3), 8: (3, 1), 12: (2, 1)}
    if geometry.grid not in stride_plan:
        raise ValueError(f"no stride plan for a {geometry.grid}x{geometry.grid} grid")
    first, second = stride_plan[geometry.grid]

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Conv2d(3, 32, 3, stride=first, padding=1)
            self.b = nn.Conv2d(32, 64, 3, stride=second, padding=1)
            self.c = nn.Conv2d(64, geometry.width, 3, stride=1, padding=1)

        def __call__(self, x: mx.array) -> mx.array:
            x = nn.gelu(self.a(x))
            x = nn.gelu(self.b(x))
            return self.c(x)

    mx.random.seed(6600)
    model = Encoder()
    mx.eval(model.parameters())

    def run(frame: np.ndarray) -> np.ndarray:
        features = model(mx.array(frame.astype(np.float32) / 255.0)[None])
        mx.eval(features)
        grid = np.asarray(features, dtype=np.float32)[0]
        if grid.shape[0] != geometry.grid:
            raise ValueError(f"cnn emitted {grid.shape[0]}, expected {geometry.grid}")
        return grid.reshape(geometry.slot_count, geometry.width)

    return run


def oracle_slots(sample: dict[str, Any], geometry: Geometry) -> np.ndarray:
    """Evaluator-only structured state, including the hidden phase itself."""
    values = [
        sample["agent_row"], sample["agent_col"], float(sample["polarity"]),
        float(sample["crossings"]), float(sample["initial_polarity"]),
        float(sample["on_switch"]), sample["nearest_switch_row"], sample["nearest_switch_col"],
    ]
    out = np.zeros((geometry.slot_count, geometry.width), dtype=np.float32)
    for index, value in enumerate(values):
        out[index % geometry.slot_count, index % geometry.width] = value
    out[:, 0] = float(sample["polarity"])
    return out


def build_representations(samples, tokens, encoder_ids) -> dict[tuple[str, str], np.ndarray]:
    """Every (source, geometry) pair, all derived from one encoding pass."""
    built: dict[tuple[str, str], np.ndarray] = {}
    count = len(samples)
    for encoder_id in encoder_ids:
        for geometry in available_geometries(encoder_id):
            stack = np.zeros((count, geometry.scalars), dtype=np.float32)
            for index in range(count):
                stack[index] = backbone_slots(
                    tokens[encoder_id][index], encoder_id, geometry
                ).reshape(-1)
            built[(f"{encoder_id}_spatial_slots", geometry.name)] = stack
    for geometry in available_geometries("raw"):
        stack = np.zeros((count, geometry.scalars), dtype=np.float32)
        proj = np.zeros((count, geometry.scalars), dtype=np.float32)
        for index, sample in enumerate(samples):
            stack[index] = raw_slots(sample["frame"], geometry).reshape(-1)
            proj[index] = random_projection_slots(sample["frame"], geometry).reshape(-1)
        built[("raw_lowres_spatial", geometry.name)] = stack
        built[("fixed_random_spatial_projection", geometry.name)] = proj
    for geometry in available_geometries("cnn"):
        runner = cnn_slots_factory(geometry)
        stack = np.zeros((count, geometry.scalars), dtype=np.float32)
        for index, sample in enumerate(samples):
            stack[index] = runner(sample["frame"]).reshape(-1)
        built[("learned_cnn_spatial_slots", geometry.name)] = stack
    oracle_geometry = GEOMETRIES[0]
    stack = np.zeros((count, oracle_geometry.scalars), dtype=np.float32)
    for index, sample in enumerate(samples):
        stack[index] = oracle_slots(sample, oracle_geometry).reshape(-1)
    built[("oracle_structured_state", oracle_geometry.name)] = stack
    return built


def reduce_to_common_width(matrix: np.ndarray, tag: str) -> np.ndarray:
    """One fixed projection to a shared width, so no arm gets more readout capacity."""
    seed = int(digest_of({"readout": tag, "dim": int(matrix.shape[1])})[7:15], 16)
    generator = np.random.default_rng(seed)
    projection = (
        generator.normal(size=(matrix.shape[1], READOUT_WIDTH)) / np.sqrt(matrix.shape[1])
    ).astype(np.float32)
    return matrix @ projection


# ---- 3. history conditions ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    role: str
    reachable: bool


CONDITIONS: tuple[Condition, ...] = (
    Condition("current_frame_only", "no history", True),
    Condition("correct_history", "the reference condition", True),
    Condition("reversed_history", "wrong temporal direction", False),
    Condition("shuffled_history", "history from other episodes", False),
    Condition("correct_action_sequence", "history plus aligned actions", True),
    Condition("shuffled_action_sequence", "history plus misaligned actions", False),
    Condition("exact_switch_event_history", "history plus true crossing events", True),
    Condition("structured_hidden_phase_oracle", "evaluator-only upper bound", True),
)
"""`reachable` records whether the condition corresponds to a state of affairs
the environment can actually produce.

Reversed and shuffled histories deliberately do not: they are order-destroying
controls, and pretending otherwise would be the dishonest reading of "pin that
all tested histories are legally reachable". What that pin can and does check is
that every *correct* history is a real rollout, which is verified separately.
"""


def episode_index(layouts: np.ndarray) -> np.ndarray:
    """Contiguous episode ids, so lags never cross an episode boundary."""
    ids = np.zeros(len(layouts), dtype=np.int64)
    current = -1
    seen: dict[int, int] = {}
    for i, layout in enumerate(layouts):
        if layout not in seen:
            current += 1
            seen[layout] = current
        ids[i] = seen[layout]
    return ids


def lag_block(source: np.ndarray, lag: int, episodes: np.ndarray, forward: bool = False) -> np.ndarray:
    """The row `lag` steps back (or forward), zero at an episode boundary."""
    out = np.zeros_like(source)
    step = lag if forward else -lag
    for i in range(len(source)):
        j = i + step
        if 0 <= j < len(source) and episodes[j] == episodes[i]:
            out[i] = source[j]
    return out


def build_condition(
    name: str,
    reduced: np.ndarray,
    oracle: np.ndarray,
    actions_onehot: np.ndarray,
    events: np.ndarray,
    episodes: np.ndarray,
    window: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Feature matrix for one condition. Identical construction for every arm.

    Note on `reversed_history`: reversing the *order of the concatenated blocks*
    would be a fixed column permutation, and a ridge probe is invariant to those,
    so that control would measure nothing at all. Reversal here is temporal --
    lag k reads the future rather than the past -- which changes the content.
    """
    if name == "structured_hidden_phase_oracle":
        return oracle
    if name == "current_frame_only":
        return reduced
    lags = range(1, window)
    if name == "correct_history":
        return np.concatenate([reduced] + [lag_block(reduced, k, episodes) for k in lags], axis=1)
    if name == "reversed_history":
        return np.concatenate(
            [reduced] + [lag_block(reduced, k, episodes, forward=True) for k in lags], axis=1
        )
    if name == "shuffled_history":
        order = rng.permutation(len(reduced))
        return np.concatenate(
            [reduced] + [lag_block(reduced, k, episodes)[order] for k in lags], axis=1
        )
    history = np.concatenate([reduced] + [lag_block(reduced, k, episodes) for k in lags], axis=1)
    if name == "correct_action_sequence":
        stack = [actions_onehot] + [lag_block(actions_onehot, k, episodes) for k in lags]
        return np.concatenate([history] + stack, axis=1)
    if name == "shuffled_action_sequence":
        order = rng.permutation(len(reduced))
        shuffled = actions_onehot[order]
        stack = [shuffled] + [lag_block(shuffled, k, episodes) for k in lags]
        return np.concatenate([history] + stack, axis=1)
    if name == "exact_switch_event_history":
        stack = [events] + [lag_block(events, k, episodes) for k in lags]
        return np.concatenate([history] + stack, axis=1)
    raise ValueError(f"unknown condition {name!r}")


# ---- 4. targets and the shared ridge solve ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    kind: str
    link: str
    classes: int = 0


TARGETS: tuple[Target, ...] = (
    Target("on_switch", "classification", "1_switch_presence", 2),
    Target("nearest_switch_row", "regression", "1_switch_presence"),
    Target("nearest_switch_col", "regression", "1_switch_presence"),
    Target("crossed_now", "classification", "2_crossed_switch", 2),
    Target("polarity_changed_now", "classification", "3_polarity_changed", 2),
    Target("crossing_parity", "classification", "4_accumulation", 2),
    Target("crossing_count", "regression", "4_accumulation"),
    Target("polarity", "classification", "5_hidden_phase", 2),
    Target("initial_polarity", "classification", "5a_reset_stripe", 2),
    Target("agent_row", "regression", "0_position"),
    Target("agent_col", "regression", "0_position"),
    Target("move_row", "regression", "0_movement"),
    Target("move_col", "regression", "0_movement"),
    Target("successor_0", "regression", "9_intervention"),
    Target("successor_1", "regression", "9_intervention"),
    Target("successor_2", "regression", "9_intervention"),
    Target("successor_3", "regression", "9_intervention"),
)


def stack_targets(values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, slice]]:
    """One target matrix so every target shares a single ridge factorization."""
    columns: list[np.ndarray] = []
    spans: dict[str, slice] = {}
    cursor = 0
    for target in TARGETS:
        raw = values[target.name]
        if target.kind == "classification":
            block = np.eye(target.classes, dtype=np.float32)[raw.astype(int)]
        else:
            block = raw.astype(np.float32).reshape(-1, 1)
        spans[target.name] = slice(cursor, cursor + block.shape[1])
        cursor += block.shape[1]
        columns.append(block)
    return np.concatenate(columns, axis=1), spans


def score_target(target: Target, prediction: np.ndarray, truth: np.ndarray,
                 train_truth: np.ndarray) -> tuple[float, float]:
    if target.kind == "classification":
        got = float((prediction.argmax(axis=1) == truth.astype(int)).mean())
        majority = int(np.bincount(train_truth.astype(int), minlength=target.classes).argmax())
        base = float((np.full_like(truth, majority) == truth.astype(int)).mean())
    else:
        variance = float(((truth - train_truth.mean()) ** 2).mean())
        got = float(1.0 - ((truth - prediction.reshape(-1)) ** 2).mean() / (variance + 1e-12))
        base = 0.0
    return got, base


def ridge_sweep(train_x, train_y, val_x, val_y, test_x, spans, val_truth, train_truth):
    """Fit once per (bandwidth, penalty) for ALL targets, pick per target on validation.

    Sharing the factorization across targets is what makes a sweep this wide
    affordable; the targets differ only in the right-hand side.
    """
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    tr, va, te = (train_x - mean) / scale, (val_x - mean) / scale, (test_x - mean) / scale
    best: dict[str, tuple[float, np.ndarray]] = {}
    for bandwidth in RFF_BANDWIDTHS:
        expansion = RandomFourier(RFF_WIDTH, bandwidth)
        expansion.fit_shape(tr.shape[1], seed=6600)
        tr_e, va_e, te_e = expansion(tr), expansion(va), expansion(te)
        design_va = np.hstack([va_e, np.ones((len(va_e), 1), dtype=np.float32)])
        design_te = np.hstack([te_e, np.ones((len(te_e), 1), dtype=np.float32)])
        for penalty in PENALTIES:
            weights = ridge_fit(tr_e, train_y, penalty)
            pv, pt = design_va @ weights, design_te @ weights
            for target in TARGETS:
                span = spans[target.name]
                value, _ = score_target(target, pv[:, span], val_truth[target.name],
                                        train_truth[target.name])
                if target.name not in best or value > best[target.name][0]:
                    best[target.name] = (value, pt[:, span])
    return {name: prediction for name, (_, prediction) in best.items()}


# ---- 5. the recurrent readout and its positive controls ----------------------------------

SEQ_ACTION_CHANNELS = len(ACTIONS) + 1
SEQ_EVENT_CHANNELS = 1
SEQ_WIDTH = READOUT_WIDTH + SEQ_ACTION_CHANNELS + SEQ_EVENT_CHANNELS
GRU_HIDDEN = 64
GRU_EPOCHS = 60
GRU_LR = 3e-3


def build_sequences(reduced, actions_onehot, events, episodes, condition, rng):
    """Per-episode (T, SEQ_WIDTH) tensors. Width is fixed so every condition and
    every representation trains the identical number of parameters -- channels a
    condition does not supply are zeroed rather than removed."""
    order = np.arange(len(reduced))
    if condition == "shuffled_history":
        order = rng.permutation(len(reduced))
    action_source = actions_onehot
    if condition == "shuffled_action_sequence":
        action_source = actions_onehot[rng.permutation(len(reduced))]
    use_actions = condition in ("correct_action_sequence", "shuffled_action_sequence")
    use_events = condition == "exact_switch_event_history"

    groups: dict[int, list[int]] = {}
    for index, episode in enumerate(episodes):
        groups.setdefault(int(episode), []).append(index)
    sequences, indices = [], []
    for episode in sorted(groups):
        rows = groups[episode]
        if condition == "reversed_history":
            feature_rows = rows[::-1]
        else:
            feature_rows = [order[r] for r in rows] if condition == "shuffled_history" else rows
        block = np.zeros((len(rows), SEQ_WIDTH), dtype=np.float32)
        block[:, :READOUT_WIDTH] = reduced[feature_rows]
        if use_actions:
            block[:, READOUT_WIDTH : READOUT_WIDTH + SEQ_ACTION_CHANNELS] = action_source[rows]
        if use_events:
            block[:, -1] = events[rows, 0]
        sequences.append(block)
        indices.append(rows)
    return sequences, indices


def train_recurrent(train_seq, train_lab, test_seq, test_lab, recurrent: bool, seed: int):
    """A small GRU, identical in size for every arm. `recurrent=False` runs the
    same parameters one step at a time from a zero state, so the ablation removes
    memory and nothing else."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)

    class Readout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(SEQ_WIDTH, GRU_HIDDEN)
            self.head = nn.Linear(GRU_HIDDEN, 2)

        def __call__(self, x: mx.array, recurrent: bool = True) -> mx.array:
            if recurrent:
                hidden = self.gru(x)
            else:
                steps = [self.gru(x[:, t : t + 1]) for t in range(x.shape[1])]
                hidden = mx.concatenate(steps, axis=1)
            return self.head(hidden)

    model = Readout()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=GRU_LR)

    def pad(batch):
        length = max(len(b) for b in batch)
        out = np.zeros((len(batch), length, batch[0].shape[1]), dtype=np.float32)
        mask = np.zeros((len(batch), length), dtype=np.float32)
        for i, b in enumerate(batch):
            out[i, : len(b)] = b
            mask[i, : len(b)] = 1.0
        return mx.array(out), mx.array(mask)

    def pad_labels(batch):
        length = max(len(b) for b in batch)
        out = np.zeros((len(batch), length), dtype=np.int32)
        for i, b in enumerate(batch):
            out[i, : len(b)] = b
        return mx.array(out)

    x, mask = pad(train_seq)
    y = pad_labels(train_lab)

    def loss_fn(m):
        logits = m(x, recurrent)
        losses = nn.losses.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction="none")
        return (losses * mask.reshape(-1)).sum() / mask.sum()

    grad_fn = nn.value_and_grad(model, loss_fn)
    for _ in range(GRU_EPOCHS):
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    tx, tmask = pad(test_seq)
    ty = pad_labels(test_lab)
    logits = model(tx, recurrent)
    mx.eval(logits)
    predicted = np.asarray(logits.argmax(axis=-1))
    truth = np.asarray(ty)
    valid = np.asarray(tmask).astype(bool)
    flat_pred = predicted[valid]
    flat_truth = truth[valid]
    return flat_pred, flat_truth


def exact_parity_accumulator(initial_polarity: np.ndarray, events: np.ndarray,
                             episodes: np.ndarray) -> np.ndarray:
    """Closed form, nothing learned: phase = initial XOR parity(events so far).

    This is the label validator. If it does not reproduce the recorded polarity
    exactly, the testbed's own bookkeeping disagrees with itself and every score
    downstream is meaningless.
    """
    out = np.zeros(len(events), dtype=np.int64)
    parity = 0
    current = -1
    for i in range(len(events)):
        if episodes[i] != current:
            current = int(episodes[i])
            parity = 0
        else:
            parity ^= int(events[i, 0])
        out[i] = int(initial_polarity[i]) ^ parity
    return out


# ---- 6. paired same-frame measurements ---------------------------------------------------


def paired_scores(features_a, features_b, truth_a, truth_b, predictor) -> dict[str, Any]:
    """On pairs whose frames are byte-identical, can the arm tell them apart?

    Identical features force identical predictions and therefore an exact tie,
    which scores 0.5. That is the point: a frame-only arm cannot exceed chance
    here by construction, so anything above chance is a leak rather than a win.
    """
    pa, pb = predictor(features_a), predictor(features_b)
    identical_features = bool(np.array_equal(features_a, features_b))
    delta = pa - pb
    label = np.sign(truth_a - truth_b)
    correct = np.where(delta * label > 0, 1.0, np.where(delta == 0, 0.5, 0.0))
    return {
        "score": float(correct.mean()),
        "chance": 0.5,
        "pairs": int(len(correct)),
        "features_identical": identical_features,
        "ties": float((delta == 0).mean()),
    }


def paired_bootstrap(values: np.ndarray, groups: np.ndarray, resamples: int, seed: int):
    """Interval over episodes, not rows: rows inside an episode are not independent."""
    unique = np.unique(groups)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    for r in range(resamples):
        picked = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_group[g] for g in picked])
        draws[r] = values[rows].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))



# ---- 6b. parity capability and the paired evaluation -------------------------------------


def gru_parity_capability(episodes: int = 200, length: int = 8, seed: int = 6600) -> dict[str, Any]:
    """Can the recurrent readout accumulate parity at all?

    Section 3 requires a temporal readout "capable of parity accumulation". That
    is a claim about the readout, and an untested claim would let a failure on
    real features be blamed on the representation when the readout was never able
    to do the job. Here it is given the event stream and the initial value
    directly: if it cannot reach near-perfect parity on that, no score it
    produces downstream means anything.
    """
    generator = np.random.default_rng(seed)
    sequences, labels = [], []
    for _ in range(episodes):
        events = generator.integers(0, 2, size=length)
        initial = int(generator.integers(0, 2))
        block = np.zeros((length, SEQ_WIDTH), dtype=np.float32)
        block[:, -1] = events
        block[0, 0] = float(initial)  # the initial value is visible only at step 0
        phase = initial ^ np.cumsum(np.concatenate([[0], events[1:]])) % 2
        sequences.append(block)
        labels.append(phase.astype(np.int64))
    split = int(0.7 * episodes)
    predicted, actual = train_recurrent(
        sequences[:split], labels[:split], sequences[split:], labels[split:],
        recurrent=True, seed=seed,
    )
    accuracy = float((predicted == actual).mean())
    return {
        "accuracy": accuracy,
        "chance": 0.5,
        "capable": accuracy > 0.9,
        "note": "event stream plus initial value, supplied directly",
    }


def evaluate_pairs(pairs, encoder_tokens=None) -> list[dict[str, Any]]:
    """Items 7 and 8, on states whose observations are byte-identical.

    The first version of this function took the pair features as an argument and
    was called with None, so its identity check compared None to None, passed,
    and verified nothing. Identity is now *computed*: every geometry is built
    from both members' frames and compared elementwise.

    The argument the measurement rests on is short. The two members share a
    byte-identical frame and an identical observation content digest, so they
    share every input any interface reads; any deterministic function of those
    inputs returns the same value for both; so a current-frame probe emits the
    same prediction for a pair whose phases are opposite, ties, and scores
    exactly 0.5. That is a fact about the construction, not a weak result -- and
    anything above 0.5 would be a leak rather than a win.
    """
    rows: list[dict[str, Any]] = []
    if not pairs:
        return rows

    # A prefix of `pairs` is not a sample: pairs are generated layout by layout, so
    # pairs[:200] covers a handful of layouts. Sample across the whole set instead.
    rng = np.random.default_rng(20250901)
    subset = [pairs[i] for i in rng.choice(len(pairs), size=min(200, len(pairs)), replace=False)]

    mismatched: dict[str, int] = {}
    for geometry in GEOMETRIES:
        differences = 0
        for pair in pairs:
            fa = raw_slots(pair["a"]["frame"], geometry)
            fb = raw_slots(pair["b"]["frame"], geometry)
            if not np.array_equal(fa, fb):
                differences += 1
        mismatched[f"raw@{geometry.name}"] = differences
    for geometry in GEOMETRIES:
        runner = cnn_slots_factory(geometry)
        differences = sum(
            0 if np.array_equal(runner(p["a"]["frame"]), runner(p["b"]["frame"])) else 1
            for p in subset
        )
        mismatched[f"cnn@{geometry.name}"] = differences

    token_differences = None
    if encoder_tokens is not None:
        token_differences = sum(
            0 if np.array_equal(a, b) else 1 for a, b in encoder_tokens
        )

    features_identical = all(v == 0 for v in mismatched.values()) and (
        token_differences in (None, 0)
    )

    polarity_a = np.array([p["a"]["polarity"] for p in pairs])
    polarity_b = np.array([p["b"]["polarity"] for p in pairs])
    label = np.sign(polarity_a - polarity_b)
    delta = np.zeros(len(pairs))  # identical features -> identical predictions
    correct = np.where(delta * label > 0, 1.0, np.where(delta == 0, 0.5, 0.0))
    rows.append({
        "measurement": "7_phase_discrimination",
        "condition": "current_frame_only",
        "kind": "identity proof, not a probe result",
        "pairs": len(pairs),
        "implied_score": float(correct.mean()),
        "chance": 0.5,
        "features_identical": features_identical,
        "per_geometry_feature_mismatches": mismatched,
        "backbone_token_mismatches": token_differences,
        "scope": "post-reset observations only",
        "note": (
            "0.5 here is arithmetic, not a measurement: identical features force "
            "identical predictions, hence an exact tie. No probe is run. The scope "
            "matters -- byte-identity can never pair a reset frame against a "
            "post-reset one, so this set excludes precisely the observations where "
            "the polarity stripe makes phase readable from one frame."
        ),
    })

    # content_digest deliberately excludes step, so "identical content digest" is a
    # weaker statement than "identical envelope". Record the gap rather than paper over it.
    differing_step = sum(
        1 for p in pairs if len(p["a"]["route"]) != len(p["b"]["route"])
    )

    successors_a = np.array([p["a"]["successors"] for p in pairs])
    successors_b = np.array([p["b"]["successors"] for p in pairs])
    differing = (successors_a != successors_b).any(axis=1)
    per_action = [
        float((successors_a[:, a] != successors_b[:, a]).mean()) for a in range(len(ACTIONS))
    ]
    equal_length = sum(1 for p in pairs if len(p["a"]["route"]) == len(p["b"]["route"]))
    rows.append({
        "measurement": "8_same_action_outcome_ranking",
        "condition": "current_frame_only",
        "pairs": len(pairs),
        "kind": "identity proof, not a probe result",
        "implied_score": 0.5 if features_identical else float("nan"),
        "chance": 0.5,
        "features_identical": features_identical,
        "scope": "post-reset observations only",
        "fraction_of_pairs_where_some_action_differs": float(differing.mean()),
        "fraction_differing_per_action": per_action,
        "equal_length_route_pairs": equal_length,
        "pairs_whose_members_sit_at_different_steps": differing_step,
        "note": (
            "the same action reaches a different successor while the observation is "
            "identical, so a current-frame predictor is necessarily wrong on one "
            "member of every such pair"
        ),
    })
    return rows


# ---- 7. pins -----------------------------------------------------------------------------


def pin_hidden_state_absent_from_features(samples, pairs) -> dict[str, Any]:
    """Does anything public carry the hidden phase?

    The first version of this pin looked for fields named `features`, `metadata`
    and `scalar_sensors`. The envelope has none of those, so it inspected nothing,
    found nothing, and reported clean -- a pin that passes because it is blind.
    It now reads the fields that exist, and carries a calibration arm: a planted
    forbidden key must be caught, or the check is not doing its job.

    The stronger half is content-based rather than name-based. On pairs that
    differ only in hidden phase, the observation content digest must be identical;
    if it is not, the phase is reachable through some channel regardless of what
    the field names say.
    """
    observation = samples[0]["observation"]
    public = set(observation.structured_observation) | set(observation.modality_payloads)
    forbidden = {"polarity", "switch_crossings", "initial_polarity", "phase", "hidden_phase"}

    planted = dict(observation.structured_observation) | {"polarity": 1}
    calibration_caught = bool(set(planted) & forbidden)

    digest_mismatches = sum(1 for p in pairs if not p["content_digests_identical"])
    return {
        "public_field_names": sorted(public),
        "forbidden_names_present": sorted(public & forbidden),
        "calibration_planted_key_detected": calibration_caught,
        "pairs_checked": len(pairs),
        "pairs_with_differing_content_digest": digest_mismatches,
        "clean": (
            not (public & forbidden)
            and calibration_caught
            and digest_mismatches == 0
            and len(pairs) > 0
        ),
    }


def pin_correct_histories_reachable(samples) -> dict[str, Any]:
    """Every correct-history row came from a real rollout, replayed and compared."""
    gate = AuthorityGate(gate_id="slotaudit-reach")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    checked = mismatched = 0
    by_layout: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_layout.setdefault(sample["layout"], []).append(sample)
    for layout, rows in list(by_layout.items())[:40]:
        adapter.reset(layout)
        for row in sorted(rows, key=lambda r: r["step"]):
            if not np.array_equal(adapter.frame(), row["frame"]):
                mismatched += 1
            checked += 1
            action = ACTIONS[(row["step"] * 3 + layout) % len(ACTIONS)]
            if adapter.step(action, gate.authorize_evaluator(action, "reach")).terminated:
                break
    return {"rows_replayed": checked, "mismatches": mismatched, "clean": mismatched == 0}


# ---- 8. decision rule --------------------------------------------------------------------


def apply_decision_rule(findings: dict[str, Any]) -> dict[str, Any]:
    """The specification's rule, evaluated rather than narrated.

    One clause runs before all the others. A readout that cannot recover a
    prerequisite from a representation that provably contains it has not
    measured the representation, and every downstream comparison inherits that
    failure. Raw pixel blocks are a lossless partition of the frame -- the agent
    occupies exactly four pixels and is uniquely identifiable in every layout --
    so failing to read position from them is a fact about the probe. The
    specification anticipates this: "If raw/CNN also fail: stop and classify the
    audit as testbed/readout invalid."
    """
    if not findings.get("readout_recovers_prerequisites", True):
        return {
            "inputs": findings,
            "outcome": "stop_testbed_or_readout_invalid",
            "selected_geometry": None,
            "screen_unblocked": False,
            "reason": (
                "the shared readout does not recover agent position from a lossless "
                "pixel representation, so the chain past link 1 measures the probe"
            ),
        }

    fine = findings["fine_grid_improves_switch_detection"]
    phase = findings["fine_grid_improves_phase_or_outcome"]
    non_inferior = findings["intervention_non_inferior"]
    pretrained = findings["effect_in_a_pretrained_package"]
    high_capacity_only = findings["high_capacity_works_matched_does_not"]
    coarse_recovers_events = findings["coarse_recovers_switch_events"]
    any_non_oracle_events = findings["any_non_oracle_recovers_events"]
    pixel_recovers_events = findings["pixel_sources_recover_events"]

    if fine and phase and non_inferior and pretrained:
        outcome, selected = "adopt_8x8x64", "g8x8x64"
    elif high_capacity_only:
        outcome, selected = "capacity_requirement_reported_not_adopted", "g4x4x256"
    elif not pixel_recovers_events:
        outcome, selected = "stop_testbed_or_readout_invalid", None
    elif not any_non_oracle_events:
        outcome, selected = "stop_slot_resampling_loss", None
    elif coarse_recovers_events and not phase:
        outcome, selected = "keep_4x4_remaining_question_is_temporal_capacity", "g4x4x256"
    else:
        outcome, selected = "keep_4x4_remaining_question_is_temporal_capacity", "g4x4x256"

    return {
        "inputs": findings,
        "outcome": outcome,
        "selected_geometry": selected,
        "screen_unblocked": outcome.startswith(("adopt", "keep", "capacity")),
    }


# ---- 9. driver ---------------------------------------------------------------------------


def iter_reduced_arms(samples, tokens, encoder_ids):
    """Build one (source, geometry) arm at a time and reduce it immediately.

    The high-capacity diagnostic is 16,384 scalars per row; holding every arm at
    full width at once would cost gigabytes for no benefit, since every readout
    sees the common-width projection anyway.
    """
    count = len(samples)
    for encoder_id in encoder_ids:
        for geometry in available_geometries(encoder_id):
            stack = np.zeros((count, geometry.scalars), dtype=np.float32)
            for index in range(count):
                stack[index] = backbone_slots(tokens[encoder_id][index], encoder_id, geometry).reshape(-1)
            name = f"{encoder_id}_spatial_slots"
            yield name, geometry, reduce_to_common_width(stack, f"{name}|{geometry.name}")
            del stack
    for geometry in available_geometries("raw"):
        for name, builder in (
            ("raw_lowres_spatial", raw_slots),
            ("fixed_random_spatial_projection", random_projection_slots),
        ):
            stack = np.zeros((count, geometry.scalars), dtype=np.float32)
            for index, sample in enumerate(samples):
                stack[index] = builder(sample["frame"], geometry).reshape(-1)
            yield name, geometry, reduce_to_common_width(stack, f"{name}|{geometry.name}")
            del stack
    for geometry in available_geometries("cnn"):
        runner = cnn_slots_factory(geometry)
        stack = np.zeros((count, geometry.scalars), dtype=np.float32)
        for index, sample in enumerate(samples):
            stack[index] = runner(sample["frame"]).reshape(-1)
        yield "learned_cnn_spatial_slots", geometry, reduce_to_common_width(
            stack, f"learned_cnn_spatial_slots|{geometry.name}"
        )
        del stack
    geometry = GEOMETRIES[0]
    stack = np.zeros((count, geometry.scalars), dtype=np.float32)
    for index, sample in enumerate(samples):
        stack[index] = oracle_slots(sample, geometry).reshape(-1)
    yield "oracle_structured_state", geometry, reduce_to_common_width(
        stack, f"oracle_structured_state|{geometry.name}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-levels", type=int, default=180)
    parser.add_argument("--val-levels", type=int, default=60)
    parser.add_argument("--test-levels", type=int, default=120)
    parser.add_argument("--paired-levels", type=int, default=90)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--encoders", nargs="*", default=["qwen3_vl_4b", "gemma3_4b"])
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/slot-resolution-audit.json")
    arguments = parser.parse_args()

    import yaml

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    started = time.perf_counter()

    train_layouts = list(range(60_000, 60_000 + arguments.train_levels))
    val_layouts = list(range(70_000, 70_000 + arguments.val_levels))
    test_layouts = list(range(80_000, 80_000 + arguments.test_levels))
    paired_layouts = list(range(90_000, 90_000 + arguments.paired_levels))

    print("collecting trajectories", flush=True)
    train = collect_episodes(train_layouts, arguments.steps, "train")
    val = collect_episodes(val_layouts, arguments.steps, "val")
    test = collect_episodes(test_layouts, arguments.steps, "test")
    samples = train + val + test
    n_train, n_val = len(train), len(val)
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}")

    print("building same-frame / different-phase pairs", flush=True)
    pairs = collect_paired(paired_layouts, arguments.steps)
    print(f"  {len(pairs)} pairs from {arguments.paired_levels} layouts")

    layouts = np.array([s["layout"] for s in samples])
    episodes = episode_index(layouts)
    steps_array = np.array([s["step"] for s in samples])
    actions = np.array([s["previous_action"] for s in samples])
    actions_onehot = np.eye(SEQ_ACTION_CHANNELS, dtype=np.float32)[actions + 1]
    events = np.array([[float(s["crossed_now"])] for s in samples], dtype=np.float32)
    crossings = np.array([s["crossings"] for s in samples])

    truth_values = {
        "on_switch": np.array([s["on_switch"] for s in samples]),
        "nearest_switch_row": np.array([s["nearest_switch_row"] for s in samples]),
        "nearest_switch_col": np.array([s["nearest_switch_col"] for s in samples]),
        "crossed_now": np.array([s["crossed_now"] for s in samples]),
        "polarity_changed_now": np.array([s["polarity_changed_now"] for s in samples]),
        "crossing_parity": crossings % 2,
        "crossing_count": crossings.astype(np.float32),
        "polarity": np.array([s["polarity"] for s in samples]),
        "initial_polarity": np.array([s["initial_polarity"] for s in samples]),
        "agent_row": np.array([s["agent_row"] for s in samples]),
        "agent_col": np.array([s["agent_col"] for s in samples]),
        "move_row": np.array([s["move_row"] for s in samples]),
        "move_col": np.array([s["move_col"] for s in samples]),
        **{f"successor_{a}": np.array([s["successors"][a] for s in samples]) for a in ACTIONS},
    }
    stacked, spans = stack_targets(truth_values)

    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, len(samples))
    train_truth = {k: v[tr] for k, v in truth_values.items()}
    val_truth = {k: v[va] for k, v in truth_values.items()}
    test_truth = {k: v[te] for k, v in truth_values.items()}
    test_layout_ids = layouts[te]
    test_crossings = crossings[te]

    strata = {
        "all": np.ones(len(test_layout_ids), dtype=bool),
        "post_first_switch": test_crossings >= 1,
        "post_two_changes": test_crossings >= 2,
    }

    print("validating labels with the exact parity accumulator", flush=True)
    reconstructed = exact_parity_accumulator(
        truth_values["initial_polarity"], events, episodes
    )
    accumulator_accuracy = float((reconstructed == truth_values["polarity"]).mean())
    print(f"  exact accumulator reproduces polarity: {accumulator_accuracy:.4f}")

    print("encoding visual tokens (frozen backbones, preprocessing untouched)", flush=True)
    cache_path = arguments.out.parent / "slot-audit-tokens.npz"
    cache_key = digest_of({
        "train": arguments.train_levels, "val": arguments.val_levels,
        "test": arguments.test_levels, "steps": arguments.steps,
        "encoders": sorted(arguments.encoders),
    })
    tokens, timings = None, {}
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=True)
        if str(stored["key"]) == cache_key:
            tokens = {e: list(stored[f"tok::{e}"]) for e in arguments.encoders}
            timings = json.loads(str(stored["timings"]))
            print("  reusing the cached token pass (encoding is unchanged)", flush=True)
    if tokens is None:
        tokens, timings = encode_tokens(samples, config, arguments.encoders)
        np.savez(
            cache_path, key=cache_key, timings=json.dumps(timings),
            **{f"tok::{e}": np.stack(tokens[e]) for e in arguments.encoders},
        )
    for encoder_id, record in timings.items():
        print(f"  {encoder_id}: {record['observations_per_second']:.1f} obs/s, "
              f"{record['visual_tokens']} tokens of width {record['token_width']}")

    windows = {"short": SHORT_WINDOW, "full": arguments.steps}
    results: list[dict[str, Any]] = []
    recurrent_results: list[dict[str, Any]] = []
    correctness: dict[tuple, np.ndarray] = {}
    strata_rows: dict[str, np.ndarray] = {}
    rng_master = np.random.default_rng(31337)

    for source, geometry, reduced in iter_reduced_arms(samples, tokens, arguments.encoders):
        oracle_block = reduced if source == "oracle_structured_state" else None
        print(f"  probing {source} @ {geometry.name}", flush=True)
        for condition in CONDITIONS:
            if condition.name == "structured_hidden_phase_oracle" and source != "oracle_structured_state":
                continue
            if condition.name != "structured_hidden_phase_oracle" and source == "oracle_structured_state":
                continue
            for window_name, window in windows.items():
                if condition.name in ("current_frame_only", "structured_hidden_phase_oracle") and window_name == "short":
                    continue
                rng = np.random.default_rng(int(digest_of({"c": condition.name, "w": window_name})[7:14], 16))
                features = build_condition(
                    condition.name, reduced, reduced if oracle_block is None else oracle_block,
                    actions_onehot, events, episodes, window, rng,
                )
                predictions = ridge_sweep(
                    features[tr], stacked[tr], features[va], stacked[va], features[te],
                    spans, val_truth, train_truth,
                )
                for target in TARGETS:
                    for stratum, mask in strata.items():
                        if stratum != "all" and target.link not in (
                            "5_hidden_phase", "9_intervention", "4_accumulation"
                        ):
                            continue
                        if mask.sum() < 30:
                            continue
                        got, base = score_target(
                            target, predictions[target.name][mask], test_truth[target.name][mask],
                            train_truth[target.name],
                        )
                        if target.kind == "classification":
                            correct = (predictions[target.name][mask].argmax(axis=1)
                                       == test_truth[target.name][mask].astype(int)).astype(float)
                        else:
                            variance = float(((test_truth[target.name][mask]
                                               - train_truth[target.name].mean()) ** 2).mean())
                            correct = 1.0 - ((test_truth[target.name][mask]
                                              - predictions[target.name][mask].reshape(-1)) ** 2) / (variance + 1e-12)
                        low, high = paired_bootstrap(
                            correct, test_layout_ids[mask], BOOTSTRAP_RESAMPLES, 99
                        )
                        correctness[(source, geometry.name, condition.name, window_name,
                                     target.name, stratum)] = correct
                        strata_rows[stratum] = test_layout_ids[mask]
                        results.append({
                            "source": source, "geometry": geometry.name, "condition": condition.name,
                            "window": window_name, "target": target.name, "link": target.link,
                            "stratum": stratum, "score": got, "baseline": base, "margin": got - base,
                            "ci_low": low - base, "ci_high": high - base,
                            "observations": int(mask.sum()),
                        })

        for condition_name in ("current_frame_only", "correct_history", "correct_action_sequence",
                               "shuffled_action_sequence", "exact_switch_event_history"):
            if source == "oracle_structured_state":
                continue
            rng = np.random.default_rng(int(digest_of({"gru": condition_name})[7:14], 16))
            sequences, indices = build_sequences(
                reduced, actions_onehot, events, episodes, condition_name, rng
            )
            labels = [truth_values["polarity"][rows] for rows in indices]
            train_mask = [i for i, rows in enumerate(indices) if rows[0] < n_train]
            test_mask = [i for i, rows in enumerate(indices) if rows[0] >= n_train + n_val]
            predicted, actual = train_recurrent(
                [sequences[i] for i in train_mask], [labels[i] for i in train_mask],
                [sequences[i] for i in test_mask], [labels[i] for i in test_mask],
                recurrent=condition_name != "current_frame_only", seed=6600,
            )
            majority = int(np.bincount(np.concatenate([labels[i] for i in train_mask]).astype(int),
                                       minlength=2).argmax())
            base = float((actual == majority).mean())
            recurrent_results.append({
                "source": source, "geometry": geometry.name, "condition": condition_name,
                "readout": "gru", "score": float((predicted == actual).mean()),
                "baseline": base, "margin": float((predicted == actual).mean()) - base,
                "observations": int(len(actual)),
            })
        del reduced

    print("computing paired geometry differences against the 4x4 reference", flush=True)
    geometry_deltas: list[dict[str, Any]] = []
    reference = "g4x4x256"
    for key, values in correctness.items():
        source, geometry, condition, window, target, stratum = key
        if geometry == reference:
            continue
        base_key = (source, reference, condition, window, target, stratum)
        if base_key not in correctness:
            continue
        base_values = correctness[base_key]
        if len(base_values) != len(values):
            continue
        difference = values - base_values
        groups = strata_rows[stratum]
        low, high = paired_bootstrap(difference, groups, BOOTSTRAP_RESAMPLES, 4242)
        geometry_deltas.append({
            "source": source, "geometry": geometry, "reference": reference,
            "condition": condition, "window": window, "target": target,
            "stratum": stratum, "delta": float(difference.mean()),
            "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0.0 or high < 0.0),
            "improves": bool(low > 0.0),
            "observations": int(len(difference)),
        })

    print("checking the recurrent readout can accumulate parity at all", flush=True)
    parity_capability = gru_parity_capability()
    print(f"  parity accumulation accuracy: {parity_capability['accuracy']:.4f} "
          f"(capable={parity_capability['capable']})")

    paired_token_check = None
    if pairs and arguments.encoders:
        sample_pairs = pairs[:24]
        checker = arguments.encoders[0]
        candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == checker)
        probe_encoder = MlxVlmBackboneEncoder(
            BackboneSpec(
                checker, candidate.repository, config["encoder"]["revisions"][checker],
                config["encoder"]["licences"][checker], REPO / config["encoder"]["weights_root"] / checker,
            )
        )
        paired_token_check = [
            (
                probe_encoder.encode_visual_tokens(p["a"]["observation"], p["a"]["frame"]),
                probe_encoder.encode_visual_tokens(p["b"]["observation"], p["b"]["frame"]),
            )
            for p in sample_pairs
        ]
        probe_encoder.release()
    paired_rows = evaluate_pairs(pairs, paired_token_check)

    elapsed = time.perf_counter() - started
    report = {
        "geometry": geometry_report(),
        "exact_parity_accumulator": {
            "reproduces_recorded_polarity": accumulator_accuracy,
            "valid": accumulator_accuracy > 0.999,
        },
        "encode_timings": timings,
        "conditions": [{"name": c.name, "role": c.role, "reachable": c.reachable} for c in CONDITIONS],
        "ridge_results": results,
        "geometry_deltas": geometry_deltas,
        "recurrent_results": recurrent_results,
        "pins": {
            "hidden_state_absent": pin_hidden_state_absent_from_features(samples, pairs),
            "correct_histories_reachable": pin_correct_histories_reachable(test),
        },
        "parity_capability": parity_capability,
        "paired_measurements": paired_rows,
        "pairs": {
            "count": len(pairs),
            "frames_identical": int(sum(p["frames_identical"] for p in pairs)),
            "content_digests_identical": int(sum(p["content_digests_identical"] for p in pairs)),
            "equal_length_routes": int(sum(
                len(p["a"]["route"]) == len(p["b"]["route"]) for p in pairs
            )),
        },
        "counts": {"train": n_train, "val": n_val, "test": len(samples) - n_train - n_val},
        "wall_clock_seconds": elapsed,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}  ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
