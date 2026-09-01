"""J6/J8. Do the slot interfaces support event detection, once the readout is qualified?

Only reached because the raw-pixel readout passed J3 and J4: an object-relation
decoder recovers agent position at exact-cell accuracy 1.0000 and switch crossing
at balanced accuracy 1.0000 from raw frames. The same decoder structure and the
same frozen relation head are now applied to the slot interfaces, so a difference
between arms is a difference in what the interface preserved rather than a
difference in probe strength. That ordering is the whole point of the rerun.

The geometry question has a concrete shape here that R-squared obscured. Agent
position must be resolved to one of 144 game cells. A 4x4 slot grid has 16 slots
covering nine cells each, an 8x8 grid has 64 slots covering 2.25 cells each, and
only a 12x12 grid is one slot per cell. So finer geometry should help *if* the
limiting factor is spatial resolution -- and the matched-capacity arm pays for
that resolution in channel width, which is exactly the trade to measure.

Every frame is encoded once and indexed twice, since the frame at step t is the
previous frame at step t+1; encoding both would double the backbone cost for no
new information.

    .venv-shwm/bin/python experiments/shwm/slot_event_qualification.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS, GRID, ProceduralVisualV2Adapter,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder  # noqa: E402
from sentinel.wm.backbones import FROZEN_CANDIDATES  # noqa: E402
from sentinel.wm.slot_geometry import (  # noqa: E402
    GEOMETRY_A, GEOMETRY_B, GEOMETRY_C, GEOMETRY_D, available_geometries,
    backbone_slots, raw_slots,
)
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402

from readout_qualification import (  # noqa: E402
    PARAMETER_CAP, balanced_accuracy, bootstrap_difference, brier, count_parameters,
    f1, mask_f1, mask_iou,
)

GEOMETRIES = (GEOMETRY_A, GEOMETRY_B, GEOMETRY_C, GEOMETRY_D)
"""D is the cell-aligned diagnostic, and dropping it was a real defect.

An earlier version of this file ran only A, B and C. That silently removed the one
arm built to separate "the slot grid is coarser than a game cell" from "the slot
readout cannot localise", and led to a classification -- readout architecture
failure -- that the missing arm refutes: at 12x12, one slot per cell, the identical
readout scores exact-cell 1.0000, switch F1 1.0000 and event balanced accuracy
1.0000 on raw slots. Neither backbone can supply a 12x12 grid, so D runs for the
pixel sources only; `available_geometries` enforces that rather than a comment."""


def collect_trajectories(layouts, trajectories, steps, appearance, seed):
    """Per-trajectory frame sequences, so each frame is encoded exactly once."""
    gate = AuthorityGate(gate_id="slot-event")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out = []
    for layout in layouts:
        for trajectory in range(trajectories):
            adapter.reset(layout, f"appearance:{appearance}")
            level = adapter._require()
            switches = tuple(sorted(tuple(int(v) for v in c) for c in level.switches))
            frames, observations, positions, crossed = [], [], [], []
            previous_position = tuple(int(v) for v in adapter._position)
            for step in range(steps):
                truth = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in truth["position"])
                frames.append(adapter.frame().copy())
                observations.append(adapter._observation())
                positions.append(position)
                crossed.append(int(step > 0 and position != previous_position
                                   and position in set(switches)))
                previous_position = position
                action = int(generator.integers(0, len(ACTIONS)))
                if adapter.step(action, gate.authorize_evaluator(action, "roll")).terminated:
                    break
            if len(frames) >= 2:
                out.append({"layout": layout, "frames": frames, "observations": observations,
                            "positions": positions, "crossed": crossed, "switches": switches})
    return out


def encode_all(trajectories, config, encoder_ids):
    """One backbone pass over every distinct frame."""
    flat_obs, flat_frames, index = [], [], []
    for t_index, trajectory in enumerate(trajectories):
        row = []
        for s_index in range(len(trajectory["frames"])):
            row.append(len(flat_frames))
            flat_frames.append(trajectory["frames"][s_index])
            flat_obs.append(trajectory["observations"][s_index])
        index.append(row)
    tokens, timings = {}, {}
    root = REPO / config["encoder"]["weights_root"]
    for encoder_id in encoder_ids:
        candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
        encoder = MlxVlmBackboneEncoder(BackboneSpec(
            encoder_id, candidate.repository, config["encoder"]["revisions"][encoder_id],
            config["encoder"]["licences"][encoder_id], root / encoder_id))
        started = time.perf_counter()
        tokens[encoder_id] = [
            encoder.encode_visual_tokens(o, f) for o, f in zip(flat_obs, flat_frames)]
        timings[encoder_id] = {
            "frames": len(flat_frames),
            "seconds": time.perf_counter() - started,
            "frames_per_second": len(flat_frames) / (time.perf_counter() - started)}
        encoder.release()
        print(f"    {encoder_id}: {timings[encoder_id]['frames_per_second']:.1f} frames/s",
              flush=True)
    return tokens, index, flat_frames, timings


def slot_tensor(source, geometry, tokens, index, flat_frames):
    """(N, g, g, width) for one arm, over every encoded frame."""
    out = np.zeros((len(flat_frames), geometry.grid, geometry.grid, geometry.width),
                   dtype=np.float32)
    for i in range(len(flat_frames)):
        if source == "raw":
            out[i] = raw_slots(flat_frames[i], geometry).reshape(
                geometry.grid, geometry.grid, geometry.width)
        else:
            out[i] = backbone_slots(tokens[source][i], source, geometry).reshape(
                geometry.grid, geometry.grid, geometry.width)
    return out


def train_slot_relation(train_idx, test_sets, slots, positions, switch_masks,
                        geometry, epochs=40, seed=6600):
    """Slot decoder plus the same frozen relation head used on raw pixels.

    The decoder is 1x1 convolutions over the slot grid, then nearest upsampling to
    the 12x12 cell grid. 1x1 keeps it slotwise and shares one set of weights across
    slots, so no arm can win by memorising slot identity, and the upsample is what
    makes the geometry's resolution limit visible: a 4x4 arm must spread one slot's
    logit across nine cells.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    factor = GRID // geometry.grid if GRID % geometry.grid == 0 else None

    class SlotDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = max(32, min(96, PARAMETER_CAP // (geometry.width * 4)))
            self.a = nn.Linear(geometry.width, hidden)
            self.b = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, 2)

        def __call__(self, x: mx.array) -> mx.array:
            z = nn.relu(self.a(x))
            z = nn.relu(self.b(z))
            return self.head(z)                       # (B, g, g, 2)

    model = SlotDecoder()
    mx.eval(model.parameters())
    parameters = count_parameters(model)
    if parameters > PARAMETER_CAP:
        raise ValueError(f"{parameters} parameters exceeds cap {PARAMETER_CAP}")
    optimizer = optim.AdamW(learning_rate=2e-3)

    def to_cells(grid_logits):
        """Nearest-upsample the slot grid to the 12x12 cell grid."""
        g = geometry.grid
        if GRID % g == 0:
            k = GRID // g
            return mx.repeat(mx.repeat(grid_logits, k, axis=1), k, axis=2)
        # 8 does not divide 12: map each cell to its containing slot by index.
        rows = (mx.arange(GRID) * g) // GRID
        out = mx.take(grid_logits, rows, axis=1)
        return mx.take(out, rows, axis=2)

    rng = np.random.default_rng(3)
    n = len(train_idx)
    for _ in range(epochs):
        for _ in range(max(1, n // 64)):
            pick = train_idx[rng.integers(0, n, 64)]
            xb = mx.array(slots[pick])
            ab = mx.array(positions[pick].astype(np.int32))
            sb = mx.array(switch_masks[pick])
            vb = mx.array((switch_masks[pick] >= 0).astype(np.float32))

            def loss_fn(m):
                cells = to_cells(m(xb))
                flat = cells.reshape(len(pick), GRID * GRID, 2)
                agent_loss = nn.losses.cross_entropy(flat[:, :, 0], ab, reduction="mean")
                per_cell = nn.losses.binary_cross_entropy(
                    flat[:, :, 1], mx.maximum(sb, 0.0), with_logits=True, reduction="none")
                switch_loss = (per_cell * vb).sum() / mx.maximum(vb.sum(), 1.0)
                return agent_loss + switch_loss

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    def decode(idx):
        agent, switch = [], []
        for k in range(0, len(idx), 256):
            cells = to_cells(model(mx.array(slots[idx[k:k + 256]])))
            flat = cells.reshape(len(idx[k:k + 256]), GRID * GRID, 2)
            mx.eval(flat)
            values = np.asarray(flat)
            agent.append(np.exp(values[:, :, 0] - values[:, :, 0].max(axis=1, keepdims=True)))
            agent[-1] /= agent[-1].sum(axis=1, keepdims=True)
            switch.append(1.0 / (1.0 + np.exp(-values[:, :, 1])))
        return np.concatenate(agent), np.concatenate(switch)

    results = {}
    for name, (now_idx, prev_idx, truth_positions, truth_crossed, groups,
               truth_switch) in test_sets.items():
        agent_now, _ = decode(now_idx)
        agent_prev, switch_prev = decode(prev_idx)
        p_moved = 1.0 - (agent_now * agent_prev).sum(axis=1)
        p_on_switch = (agent_now * switch_prev).sum(axis=1)
        p_crossed = p_moved * p_on_switch
        predicted = (p_crossed > 0.5).astype(int)
        majority = float(max(np.bincount(truth_crossed, minlength=2)) / len(truth_crossed))
        low, high = bootstrap_difference(
            (predicted == truth_crossed).astype(float), groups, majority)
        results[name] = {
            "parameters": parameters,
            "agent_exact_cell_accuracy": float((agent_now.argmax(axis=1) == truth_positions).mean()),
            "switch_mask_f1": mask_f1(switch_prev > 0.5, truth_switch > 0.5),
            "switch_mask_iou": mask_iou(switch_prev > 0.5, truth_switch > 0.5),
            "event_balanced_accuracy": balanced_accuracy(truth_crossed, predicted),
            "event_f1": f1(truth_crossed, predicted),
            "event_brier": brier(truth_crossed, p_crossed),
            "majority_baseline": majority,
            "ci_low_vs_majority": low, "ci_high_vs_majority": high,
        }
    return results


# ---- driver ---------------------------------------------------------------------------------


def build_arrays(trajectories, index, flat_count):
    """Per-frame targets, with the agent's own cell marked unobservable in the
    switch mask because the renderer paints over it."""
    positions = np.zeros(flat_count, dtype=np.int64)
    switch_masks = np.zeros((flat_count, GRID * GRID), dtype=np.float32)
    for t_index, trajectory in enumerate(trajectories):
        for s_index, flat in enumerate(index[t_index]):
            row, col = trajectory["positions"][s_index]
            positions[flat] = row * GRID + col
            for cell in trajectory["switches"]:
                switch_masks[flat, cell[0] * GRID + cell[1]] = 1.0
            switch_masks[flat, row * GRID + col] = -1.0   # occluded, excluded from loss
    return positions, switch_masks


def pair_indices(trajectories, index):
    now, prev, crossed, groups = [], [], [], []
    for t_index, trajectory in enumerate(trajectories):
        for s_index in range(1, len(index[t_index])):
            now.append(index[t_index][s_index])
            prev.append(index[t_index][s_index - 1])
            crossed.append(trajectory["crossed"][s_index])
            groups.append(trajectory["layout"])
    return (np.array(now), np.array(prev), np.array(crossed), np.array(groups))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=45)
    parser.add_argument("--test-layouts", type=int, default=25)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--encoders", nargs="*", default=["qwen3_vl_4b", "gemma3_4b"])
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/slot-event-qualification.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    import yaml
    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    appearance = CANONICAL_APPEARANCE_SEED

    train_layouts = list(range(61_000, 61_000 + arguments.train_layouts))
    held_layouts = list(range(81_000, 81_000 + arguments.test_layouts))
    print("collecting", flush=True)
    train_t = collect_trajectories(train_layouts, arguments.trajectories, arguments.steps,
                                   appearance, 11)
    a_test_t = collect_trajectories(train_layouts, 1, arguments.steps, appearance, 999)
    b_test_t = collect_trajectories(held_layouts, 2, arguments.steps, appearance, 777)
    everything = train_t + a_test_t + b_test_t
    n_train_t, n_a_t = len(train_t), len(a_test_t)
    print(f"  trajectories: train {n_train_t}  A test {n_a_t}  B test {len(b_test_t)}", flush=True)

    print("  encoding (each frame once)", flush=True)
    tokens, index, flat_frames, timings = encode_all(everything, config, arguments.encoders)
    positions, switch_masks = build_arrays(everything, index, len(flat_frames))

    def subset(lo, hi):
        return pair_indices(everything[lo:hi], index[lo:hi])

    train_now, _, _, _ = subset(0, n_train_t)
    a_now, a_prev, a_crossed, a_groups = subset(n_train_t, n_train_t + n_a_t)
    b_now, b_prev, b_crossed, b_groups = subset(n_train_t + n_a_t, len(everything))
    print(f"  transitions: train {len(train_now)}  A {len(a_now)}  B {len(b_now)}", flush=True)

    report: dict[str, Any] = {
        "appearance_seed": appearance, "parameter_cap": PARAMETER_CAP,
        "encode_timings": timings, "arms": {},
        "counts": {"train_transitions": int(len(train_now)),
                   "a_transitions": int(len(a_now)), "b_transitions": int(len(b_now))},
    }

    sources = list(arguments.encoders) + ["raw"]
    for source in sources:
        for geometry in available_geometries(
                "raw" if source == "raw" else source):
            label = f"{source}@{geometry.name}"
            slots = slot_tensor(source, geometry, tokens, index, flat_frames)
            test_sets = {
                "A_information": (a_now, a_prev, positions[a_now], a_crossed, a_groups,
                                  switch_masks[a_prev]),
                "B_generalisation": (b_now, b_prev, positions[b_now], b_crossed, b_groups,
                                     switch_masks[b_prev]),
            }
            results = train_slot_relation(train_now, test_sets, slots, positions,
                                          switch_masks, geometry)
            report["arms"][label] = results
            for split, record in results.items():
                print(f"  {label:28s} {split:16s} agent-exact "
                      f"{record['agent_exact_cell_accuracy']:.4f}  switch F1 "
                      f"{record['switch_mask_f1']:.4f}  event bal-acc "
                      f"{record['event_balanced_accuracy']:.4f}  F1 {record['event_f1']:.4f}  "
                      f"CI[{record['ci_low_vs_majority']:+.3f},"
                      f"{record['ci_high_vs_majority']:+.3f}]", flush=True)
            del slots

    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
