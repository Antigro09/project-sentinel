"""K3-K5, K9. Three within-slot decoders across every interface, plus a main arm
that is given no privileged event bit.

Structure follows the specification's separation of mechanisms, because conflating
them is what made the J-phase headline overstated:

1. learned visual state  -- the decoders predict agent and switch masks, and that
   is the only thing any of them is trained on;
2. hand-derived public event -- switch crossing is computed from those masks by a
   fixed relation with no parameters, and is reported as DERIVED, never as a
   learned detection;
3. recurrent phase accumulator -- parity over derived events;
4. the MAIN ARM -- a recurrent model over agent-visible packets only, with no
   event bit, no mask supervision and no hidden value, which has to infer the
   dynamics itself. Gate K9 is about this arm existing and being reported
   separately, not about the derived pipeline scoring well.

    .venv-shwm/bin/python experiments/shwm/k_qualification.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS, GRID  # noqa: E402
from sentinel.wm.slot_geometry import (  # noqa: E402
    GEOMETRY_A, GEOMETRY_B, GEOMETRY_C, GEOMETRY_D, backbone_slots,
    random_projection_slots, raw_slots,
)
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from readout_qualification import (  # noqa: E402
    balanced_accuracy, bootstrap_difference, brier, f1, mask_f1, mask_iou,
)
from slot_event_qualification import (  # noqa: E402
    build_arrays, collect_trajectories, encode_all, pair_indices,
)
from within_slot_readouts import DECODERS, PARAMETER_CEILING, train_decoder  # noqa: E402

INTERFACES: tuple[tuple[str, Any, str], ...] = (
    ("qwen3_vl_4b", GEOMETRY_A, "pretrained"),
    ("qwen3_vl_4b", GEOMETRY_B, "pretrained"),
    ("qwen3_vl_4b", GEOMETRY_C, "capacity_diagnostic"),
    ("gemma3_4b", GEOMETRY_A, "pretrained"),
    ("gemma3_4b", GEOMETRY_B, "pretrained"),
    ("gemma3_4b", GEOMETRY_C, "capacity_diagnostic"),
    ("raw", GEOMETRY_A, "pixel_control"),
    ("raw", GEOMETRY_B, "pixel_control"),
    ("raw", GEOMETRY_D, "environment_aligned_diagnostic"),
    ("randproj", GEOMETRY_B, "fixed_random_control"),
)


def build_slots(source, geometry, tokens, frames):
    out = np.zeros((len(frames), geometry.grid, geometry.grid, geometry.width),
                   dtype=np.float32)
    for i in range(len(frames)):
        if source == "raw":
            out[i] = raw_slots(frames[i], geometry).reshape(
                geometry.grid, geometry.grid, geometry.width)
        elif source == "randproj":
            out[i] = random_projection_slots(frames[i], geometry).reshape(
                geometry.grid, geometry.grid, geometry.width)
        else:
            out[i] = backbone_slots(tokens[source][i], source, geometry).reshape(
                geometry.grid, geometry.grid, geometry.width)
    return out


def derived_event(agent_now, agent_prev, switch_prev):
    """The hand-derived public event, stated exactly as the specification asks.

        p_moved     = 1 - sum_c agent_t(c) * agent_{t-1}(c)
        p_on_switch = sum_c     agent_t(c) * switch_{t-1}(c)
        p_crossed   = p_moved * p_on_switch

    No parameters, no fitting, no event label anywhere in training. The switch mask
    is read at t-1 because the renderer paints the agent over the switch beneath it,
    so the destination cell's switch state is not present in the current frame.
    """
    p_moved = 1.0 - (agent_now * agent_prev).sum(axis=1)
    p_on_switch = (agent_now * switch_prev).sum(axis=1)
    return p_moved * p_on_switch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=45)
    parser.add_argument("--test-layouts", type=int, default=25)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--encoders", nargs="*", default=["qwen3_vl_4b", "gemma3_4b"])
    parser.add_argument("--decoders", nargs="*", default=list(DECODERS))
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/k-qualification.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    import yaml
    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    appearance = CANONICAL_APPEARANCE_SEED

    train_t = collect_trajectories(list(range(61_000, 61_000 + arguments.train_layouts)),
                                   arguments.trajectories, arguments.steps, appearance, 11)
    b_t = collect_trajectories(list(range(81_000, 81_000 + arguments.test_layouts)),
                               2, arguments.steps, appearance, 777)
    everything = train_t + b_t
    print(f"trajectories: train {len(train_t)}  held-out {len(b_t)}", flush=True)

    tokens, index, frames, timings = encode_all(everything, config, arguments.encoders)
    positions, switch_masks = build_arrays(everything, index, len(frames))
    visible_mask = (switch_masks >= 0).astype(np.float32)
    switch_target = np.maximum(switch_masks, 0.0)

    train_rows = np.array([i for row in index[:len(train_t)] for i in row])
    b_now, b_prev, b_crossed, b_groups = pair_indices(b_t, index[len(train_t):])
    print(f"frames: train {len(train_rows)}  held-out transitions {len(b_now)}", flush=True)

    report: dict[str, Any] = {
        "parameter_ceiling": PARAMETER_CEILING, "appearance_seed": appearance,
        "encode_timings": timings, "arms": {},
        "event_mechanism": "DERIVED from predicted masks by a parameterless relation",
        "counts": {"train_frames": int(len(train_rows)),
                   "held_out_transitions": int(len(b_now))},
    }

    for source, geometry, role in INTERFACES:
        if source in ("qwen3_vl_4b", "gemma3_4b") and source not in arguments.encoders:
            continue
        slots = build_slots(source, geometry, tokens, frames)
        for decoder in arguments.decoders:
            label = f"{source}@{geometry.name}::{decoder}"
            try:
                fitted = train_decoder(
                    decoder, geometry.grid, geometry.width,
                    slots[train_rows], positions[train_rows],
                    switch_target[train_rows], visible_mask[train_rows], {})
            except ValueError as error:
                report["arms"][label] = {"skipped": str(error), "role": role}
                print(f"  {label:58s} SKIPPED ({error})", flush=True)
                continue
            agent_now, _ = fitted["decode"](slots[b_now])
            agent_prev, switch_prev = fitted["decode"](slots[b_prev])
            exact = float((agent_now.argmax(axis=1) == positions[b_now]).mean())
            actual_switch = switch_target[b_prev] > 0.5
            p_crossed = derived_event(agent_now, agent_prev, switch_prev)
            predicted = (p_crossed > 0.5).astype(int)
            majority = float(max(np.bincount(b_crossed, minlength=2)) / len(b_crossed))
            low, high = bootstrap_difference(
                (predicted == b_crossed).astype(float), b_groups, majority)
            record = {
                "role": role, "decoder": decoder, "geometry": geometry.name,
                "source": source, "parameters": fitted["parameters"],
                "agent_exact_cell_accuracy": exact,
                "agent_mask_f1": mask_f1(
                    agent_now.argmax(axis=1)[:, None] == np.arange(GRID * GRID)[None, :],
                    positions[b_now][:, None] == np.arange(GRID * GRID)[None, :]),
                "switch_mask_f1": mask_f1(switch_prev > 0.5, actual_switch),
                "switch_mask_iou": mask_iou(switch_prev > 0.5, actual_switch),
                "derived_event_balanced_accuracy": balanced_accuracy(b_crossed, predicted),
                "derived_event_f1": f1(b_crossed, predicted),
                "derived_event_brier": brier(b_crossed, p_crossed),
                "majority_baseline": majority,
                "ci_low_vs_majority": low, "ci_high_vs_majority": high,
            }
            report["arms"][label] = record
            print(f"  {label:58s} exact {exact:.4f}  swF1 {record['switch_mask_f1']:.4f}  "
                  f"event {record['derived_event_balanced_accuracy']:.4f}  "
                  f"CI[{low:+.3f},{high:+.3f}]  p={fitted['parameters']}", flush=True)
        del slots

    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
