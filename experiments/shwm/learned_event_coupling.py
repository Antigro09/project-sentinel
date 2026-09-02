"""H / J / U7. Couple the learned event detector to the selected filter, on alias pairs.

With TRUE events, the symmetry-broken two-state filter reaches the exact-accumulator
ceiling on all twenty untouched validation seeds. So whatever remains is event
extraction and its coupling, and this measures exactly that -- on the population the
gate names, which is the exact public-packet alias set rather than general transitions.

The primary population: identical complete current public packet, identical proposed
action, different legal histories, different hidden phase, different public next
outcome. A memoryless model is at exactly chance there by construction, so any lift
comes from history.

    .venv-shwm/bin/python experiments/shwm/learned_event_coupling.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS, GRID  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from structured_calibration import DISPLACEMENTS, CLASSES, collect  # noqa: E402
from belief_factorization import (  # noqa: E402
    build_dataset, public_event, sequence_features, train_event_extractor,
)
from structured_calibration import row_from_state  # noqa: E402
from filter_stability import pad, run as run_filter  # noqa: E402

SEEDS = tuple(range(9000, 9010))
SELECTED_FILTER = "3_two_state_symmetry_broken"


def alias_examples(layouts, depth=6):
    """Exact alias pairs, as (state, partner, action) with differing outcomes."""
    from alias_audit import enumerate_states
    states = enumerate_states(layouts, depth)
    classes = defaultdict(list)
    for state in states:
        classes[state.key("V2_agent_visible")].append(state)
    out = []
    for members in classes.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.polarity == b.polarity:
                    continue
                for action in ACTIONS:
                    if a.successors[action] != b.successors[action]:
                        out.append((a, b, action))
    return out


def to_class(signature, origin):
    cell = int(signature)
    delta = (cell // GRID - origin[0], cell % GRID - origin[1])
    return DISPLACEMENTS.index(delta) if delta in DISPLACEMENTS else CLASSES - 1


def evaluate_on_aliases(predict_fn, examples, phase_of) -> dict[str, Any]:
    """Pairwise: does the model rank its own outcome above its partner's?

    Identical packets, so a memoryless model must tie at exactly 0.5.
    """
    hits, groups, margins = [], [], []
    cache: dict[Any, Any] = {}
    for a, b, action in examples:
        for me, partner in ((a, b), (b, a)):
            key = (me.layout, me.position, me.polarity, me.previous_action, me.blocked)
            if key not in cache:
                cache[key] = row_from_state(me)
            row = cache[key]
            logits = predict_fn(row, action, phase_of(me))
            t = to_class(me.successors[action], me.position)
            o = to_class(partner.successors[action], me.position)
            if t == o:
                continue
            margin = float(logits[t] - logits[o])
            margins.append(margin)
            hits.append(1.0 if margin > 0 else (0.5 if margin == 0 else 0.0))
            groups.append(me.layout)
    hits = np.array(hits)
    groups = np.array(groups)
    return {"pairs": int(len(hits)), "pairwise_accuracy": float(hits.mean()),
            "mean_margin": float(np.mean(margins)), "hits": hits, "groups": groups}


def paired_interval(a, b, groups, resamples=4000, seed=99):
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for r in range(resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in picked])
        draws[r] = a[rows].mean() - b[rows].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=40)
    parser.add_argument("--alias-layouts", type=int, default=10)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/learned-event-coupling.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train = build_dataset(collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                                  arguments.trajectories, arguments.steps, appearance, 11), 5)
    test = build_dataset(collect(list(range(81_000, 81_020)), 2, arguments.steps,
                                 appearance, 777), 6)
    examples = alias_examples(list(range(90_000, 90_000 + arguments.alias_layouts)))
    print(f"train {len(train)} trajectories; {len(examples)} exact alias (pair, action) "
          f"examples where the outcome differs\n", flush=True)

    extractor, event_metrics = train_event_extractor(train, test, 1024, 6600)
    print(f"learned event detector: balanced accuracy "
          f"{event_metrics['balanced_accuracy']:.4f}  F1 {event_metrics['f1']:.4f}",
          flush=True)

    report: dict[str, Any] = {"event_extractor": event_metrics,
                              "selected_filter": SELECTED_FILTER,
                              "alias_examples": len(examples), "arms": {}}

    # Phase estimators over an alias state's own route.
    def phase_true(state):
        return state.polarity

    def phase_from_true_events(state):
        h = 0
        # reconstruct along the route: the reset stripe is public at step 0
        from sentinel.env.adapters.procedural_visual_v2 import build_level_v2
        level = build_level_v2(state.layout, state.layout, state.layout)
        h = int(level.initial_polarity)
        from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
        return (h + state.crossings) % 2

    def phase_from_learned_events(state):
        """Run the learned detector along the state's own legal route."""
        from sentinel.env.adapters.procedural_visual_v2 import (
            ProceduralVisualV2Adapter, build_level_v2)
        from sentinel.wm.authority import AuthorityGate
        gate = AuthorityGate(gate_id="couple")
        adapter = ProceduralVisualV2Adapter(gate=gate)
        adapter.reset(state.layout)
        rows = []
        previous = None
        level = adapter._require()
        switches = {tuple(int(v) for v in c) for c in level.switches}
        walls = np.asarray(level.walls, dtype=bool)
        goal = tuple(int(v) for v in level.markers[adapter._goal_marker])
        previous_action = -1
        for step_index in range(len(state.route) + 1):
            truth = adapter.snapshot().reveal("evaluator")
            row = {"position": tuple(int(v) for v in truth["position"]),
                   "switches": switches, "walls": walls, "goal": goal,
                   "previous_action": previous_action,
                   "blocked": float(truth["last_blocked"]),
                   "polarity": int(truth["polarity"])}
            rows.append(row)
            if step_index == len(state.route):
                break
            previous_action = state.route[step_index]
            adapter.step(previous_action,
                         gate.authorize_evaluator(previous_action, "couple"))
        item = {"x": sequence_features({"rows": rows}, [0] * len(rows)),
                "events": np.array([public_event(rows[i - 1] if i else None, rows[i])
                                    for i in range(len(rows))]),
                "phases": np.array([r["polarity"] for r in rows]),
                "y": np.zeros(len(rows), dtype=int), "layout": state.layout}
        predicted = extractor(item)
        h = int(rows[0]["polarity"])
        for index in range(1, len(predicted)):
            h ^= int(predicted[index])
        return h

    # A head that maps (public row, action, phase estimate) -> displacement logits,
    # trained once with TRUE phase so the head itself is not the variable under test.
    from belief_factorization import head_with_phase
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from structured_calibration import encode as encode_row

    def build_head(seed):
        mx.random.seed(seed)

        class Head(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.a = nn.Linear(width, 128)
                self.b = nn.Linear(128, 128)
                self.head = nn.Linear(128, CLASSES)

            def __call__(self, z):
                return self.head(nn.relu(self.b(nn.relu(self.a(z)))))

        xs, ys = [], []
        for item in train:
            for t in range(len(item["y"])):
                xs.append(np.concatenate([item["x"][t], [float(item["phases"][t])]]))
                ys.append(item["y"][t])
        x, y = np.stack(xs).astype(np.float32), np.array(ys)
        model = Head(x.shape[1])
        mx.eval(model.parameters())
        optimizer = optim.AdamW(learning_rate=2e-3)
        rng = np.random.default_rng(seed)
        for _ in range(1024):
            pick = rng.integers(0, len(x), 128)
            xb, yb = mx.array(x[pick]), mx.array(y[pick].astype(np.int32))

            def loss_fn(m):
                return nn.losses.cross_entropy(m(xb), yb, reduction="mean")

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        def predict(row, action, phase):
            features = np.concatenate([
                encode_row(row, action, with_phase=False), [0.0, 0.0], [float(phase)]])
            out = model(mx.array(features[None].astype(np.float32)))
            mx.eval(out)
            return np.asarray(out)[0]

        return predict

    results = {}
    for name, phase_of in (("true_phase_oracle", phase_true),
                           ("true_event_accumulator", phase_from_true_events),
                           ("learned_event_accumulator", phase_from_learned_events),
                           ("memoryless_constant", lambda s: 0)):
        per_seed = []
        for seed in SEEDS[:3]:
            predict = build_head(seed)
            per_seed.append(evaluate_on_aliases(predict, examples, phase_of))
        results[name] = per_seed
        mean = float(np.mean([r["pairwise_accuracy"] for r in per_seed]))
        report["arms"][name] = {
            "pairwise_accuracy_mean": mean,
            "per_seed": [r["pairwise_accuracy"] for r in per_seed],
            "pairs": per_seed[0]["pairs"],
        }
        print(f"  {name:28s} alias pairwise {mean:.4f}  "
              f"per-seed {[round(r['pairwise_accuracy'], 4) for r in per_seed]}", flush=True)

    print("\npaired intervals by layout, against the memoryless baseline:")
    base = np.mean([r["hits"] for r in results["memoryless_constant"]], axis=0)
    groups = results["memoryless_constant"][0]["groups"]
    for name in ("true_phase_oracle", "true_event_accumulator", "learned_event_accumulator"):
        arm = np.mean([r["hits"] for r in results[name]], axis=0)
        low, high = paired_interval(arm, base, groups)
        report["arms"][name]["vs_memoryless"] = {
            "delta": float(arm.mean() - base.mean()), "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0)}
        print(f"  {name:28s} {arm.mean() - base.mean():+.4f}  [{low:+.4f}, {high:+.4f}]"
              f"{'  *' if low > 0 else ''}")

    learned = report["arms"]["learned_event_accumulator"]["vs_memoryless"]
    report["u7_learned_pipeline_beats_memoryless"] = bool(learned["ci_low"] > 0)
    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"\nU7 (learned event + filter beats memoryless on exact alias pairs): "
          f"{report['u7_learned_pipeline_beats_memoryless']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
