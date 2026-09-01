"""J5/J7. Run the whole chain with the qualified readout, and test history order.

Every link has now been measured separately. This composes them and checks the
composition, which is not the same thing: an event detector at balanced accuracy
1.000 still has to survive being accumulated over a trajectory, and the
independent-error model that predicted it would is an approximation.

    phase(t) = initial_polarity  XOR  parity(decoded events up to t)

Initial polarity comes from the reset frame, where the renderer draws it as a
one-pixel top stripe -- the reset-observability statement, and the reason the
global "unidentifiable" claim was withdrawn. Events come from the object-relation
decoder. Nothing here is given a hidden value.

J7 asks whether correct order actually matters, and the controls need care because
parity is order-invariant over a fixed multiset.

Both controls preserve the event multiset -- an earlier version of this docstring
claimed they change it, which is false: reversing visits the same (t, t-1) pairs in
the opposite order, and the within-trajectory shuffle permutes the same events. So
the FINAL-step parity is identical under all three modes, and the controls cannot
fail there. What they do change is the alignment between the running parity and the
step it is attributed to, so every intermediate phase is wrong while the endpoint is
right. That is why they score 0.55 and 0.63 against 1.00 rather than 0.00, and it is
a weaker control than a multiset-changing one would be. A control that resampled
which events occur -- across trajectories rather than within -- would be stronger and
is not implemented here.

    .venv-shwm/bin/python experiments/shwm/chain_end_to_end.py
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

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS, GRID, ProceduralVisualV2Adapter,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from readout_qualification import (  # noqa: E402
    GRID as RQ_GRID, PARAMETER_CAP, balanced_accuracy, bootstrap_difference,
    build_heatmap_cnn, count_parameters, f1, stack_frames,
)


def collect(layouts, trajectories, steps, appearance, seed):
    gate = AuthorityGate(gate_id="chain-e2e")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out = []
    for layout in layouts:
        for _ in range(trajectories):
            adapter.reset(layout, f"appearance:{appearance}")
            level = adapter._require()
            switches = set(tuple(int(v) for v in c) for c in level.switches)
            rows = []
            previous = tuple(int(v) for v in adapter._position)
            for step in range(steps):
                truth = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in truth["position"])
                rows.append({
                    "frame": adapter.frame().copy(), "position": position,
                    "polarity": int(truth["polarity"]),
                    "crossings": int(truth["switch_crossings"]),
                    "crossed": int(step > 0 and position != previous and position in switches),
                    "switches": switches, "step": step,
                })
                previous = position
                action = int(generator.integers(0, len(ACTIONS)))
                if adapter.step(action, gate.authorize_evaluator(action, "roll")).terminated:
                    break
            if len(rows) >= 3:
                out.append({"layout": layout, "rows": rows})
    return out


def read_initial_polarity(frame: np.ndarray) -> int:
    """The reset stripe. Hand-coded, and exact: the renderer paints row 0 white
    for polarity 1 and black for polarity 0."""
    return int(frame[0].mean() > frame[1].mean())


def train_object_decoder(trajectories, epochs=40, seed=6600):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    Base = build_heatmap_cnn(2)

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = Base(3)

        def __call__(self, x):
            grid = self.body(x)
            flat = grid.reshape(grid.shape[0], GRID * GRID, 2)
            return flat[:, :, 0], flat[:, :, 1]

    model = Decoder()
    mx.eval(model.parameters())
    parameters = count_parameters(model)
    assert parameters <= PARAMETER_CAP, parameters

    rows = [r for t in trajectories for r in t["rows"]]
    frames = np.stack([r["frame"].astype(np.float32) / 255.0 for r in rows])
    agent = np.array([r["position"][0] * GRID + r["position"][1] for r in rows])
    switch = np.zeros((len(rows), GRID * GRID), dtype=np.float32)
    visible = np.ones((len(rows), GRID * GRID), dtype=np.float32)
    for i, r in enumerate(rows):
        for cell in r["switches"]:
            switch[i, cell[0] * GRID + cell[1]] = 1.0
        visible[i, r["position"][0] * GRID + r["position"][1]] = 0.0

    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(4)
    for _ in range(epochs):
        for _ in range(max(1, len(rows) // 64)):
            idx = rng.integers(0, len(rows), 64)
            xb, ab = mx.array(frames[idx]), mx.array(agent[idx].astype(np.int32))
            sb, vb = mx.array(switch[idx]), mx.array(visible[idx])

            def loss_fn(m):
                al, sl = m(xb)
                per_cell = nn.losses.binary_cross_entropy(
                    sl, sb, with_logits=True, reduction="none")
                return (nn.losses.cross_entropy(al, ab, reduction="mean")
                        + (per_cell * vb).sum() / vb.sum())

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    def decode(frames_array):
        agent_out, switch_out = [], []
        for k in range(0, len(frames_array), 256):
            al, sl = model(mx.array(frames_array[k:k + 256]))
            mx.eval(al, sl)
            agent_out.append(np.asarray(mx.softmax(al, axis=-1)))
            switch_out.append(np.asarray(mx.sigmoid(sl)))
        return np.concatenate(agent_out), np.concatenate(switch_out)

    return decode, parameters


def run_chain(trajectories, decode, mode: str, rng) -> dict[str, Any]:
    """Decode events along each trajectory, accumulate parity, predict phase."""
    predicted_all, truth_all, groups, crossings_all = [], [], [], []
    for trajectory in trajectories:
        rows = trajectory["rows"]
        frames = np.stack([r["frame"].astype(np.float32) / 255.0 for r in rows])
        agent, switch = decode(frames)
        initial = read_initial_polarity(rows[0]["frame"])
        order = list(range(1, len(rows)))
        if mode == "reversed_history":
            order = order[::-1]
        events = []
        for s in order:
            moved = 1.0 - float((agent[s] * agent[s - 1]).sum())
            on_switch = float((agent[s] * switch[s - 1]).sum())
            events.append(1 if moved * on_switch > 0.5 else 0)
        if mode == "shuffled_events":
            events = list(rng.permutation(events))
        parity, predicted = initial, []
        for e in events:
            parity ^= int(e)
            predicted.append(parity)
        for position, s in enumerate(range(1, len(rows))):
            predicted_all.append(predicted[position])
            truth_all.append(rows[s]["polarity"])
            groups.append(trajectory["layout"])
            crossings_all.append(rows[s]["crossings"])
    predicted_all = np.array(predicted_all)
    truth_all = np.array(truth_all)
    groups = np.array(groups)
    crossings_all = np.array(crossings_all)

    out = {"mode": mode, "observations": int(len(truth_all))}
    for name, mask in (("all", np.ones(len(truth_all), bool)),
                       ("post_first_switch", crossings_all >= 1),
                       ("post_two_changes", crossings_all >= 2)):
        if mask.sum() < 20:
            continue
        correct = (predicted_all[mask] == truth_all[mask]).astype(float)
        majority = float(max(np.bincount(truth_all[mask], minlength=2)) / mask.sum())
        low, high = bootstrap_difference(correct, groups[mask], majority)
        out[name] = {
            "phase_accuracy": float(correct.mean()),
            "balanced_accuracy": balanced_accuracy(truth_all[mask], predicted_all[mask]),
            "majority_baseline": majority,
            "margin": float(correct.mean()) - majority,
            "ci_low": low, "ci_high": high,
            "observations": int(mask.sum()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=90)
    parser.add_argument("--test-layouts", type=int, default=60)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/chain-end-to-end.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train = collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                    arguments.trajectories, arguments.steps, appearance, 11)
    test = collect(list(range(81_000, 81_000 + arguments.test_layouts)),
                   2, arguments.steps, appearance, 777)
    print(f"train trajectories {len(train)}  test {len(test)}", flush=True)

    print("training the object decoder (masks only, no phase, no event labels)", flush=True)
    decode, parameters = train_object_decoder(train)
    print(f"  {parameters} parameters (cap {PARAMETER_CAP})", flush=True)

    rng = np.random.default_rng(8)
    report: dict[str, Any] = {"parameters": parameters, "appearance_seed": appearance,
                              "modes": {}}
    print("\nJ5 / J7: end-to-end phase from decoded events", flush=True)
    for mode in ("correct_history", "reversed_history", "shuffled_events"):
        result = run_chain(test, decode, mode, rng)
        report["modes"][mode] = result
        line = "  ".join(
            f"{k} {result[k]['phase_accuracy']:.4f} (base {result[k]['majority_baseline']:.3f}, "
            f"CI[{result[k]['ci_low']:+.3f},{result[k]['ci_high']:+.3f}])"
            for k in ("all", "post_first_switch", "post_two_changes") if k in result)
        print(f"  {mode:18s} {line}", flush=True)

    correct = report["modes"]["correct_history"]
    report["j5_near_oracle_phase"] = correct["all"]["phase_accuracy"] >= 0.90
    report["j7_correct_beats_controls"] = all(
        correct["all"]["phase_accuracy"] > report["modes"][m]["all"]["phase_accuracy"]
        for m in ("reversed_history", "shuffled_events"))
    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nJ5 (phase >= 0.90): {report['j5_near_oracle_phase']}")
    print(f"J7 (correct beats controls): {report['j7_correct_beats_controls']}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
