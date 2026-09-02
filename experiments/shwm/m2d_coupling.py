"""E / G / H / I. Learned event coupled to a learned filter, and what removes the gain.

This is the arm M2C claimed and did not run. The M2C coupling imported the selected
filter, never called it, and XOR-accumulated the detector's output instead; the report
then described the row as "learned event + selected learned filter". So U7 is run here
for the first time, against a filter object whose identity is read off the trained model.

Two couplings are preregistered and the choice between them is made on a DEVELOPMENT
alias population drawn from different layouts than the one that is reported:

    hard        feed argmax C_hat_t
    posterior   propagate p(C_t=0) T_0 + p(C_t=1) T_1

The detector is retrained with the query action masked out of its input. The M2D
dataflow audit shows the M2C detector reads A_t, which is not in the legal input set
for C_t -- an event estimate that moves when you ask about a different action is not an
estimate of what happened.

    .venv-shwm/bin/python experiments/shwm/m2d_coupling.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import m2d_core as core
from m2d_core import (ARTIFACTS, ArmIdentity, FilterSpec, MECHANISM, QUERY_ACTION,
                      RouteFeatures, antisymmetric_two_state, build_population,
                      build_tensors, checkpoint_hash, score_population, stratify,
                      summarise_metric, train_model, write)
from m2d_filters import INPUTS_FILTER, INPUTS_MEMORYLESS, population_label
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

DEV_SEEDS = tuple(range(7000, 7005))
VALIDATION_SEEDS = tuple(range(8000, 8020))
DEV_ALIAS_LAYOUTS = tuple(range(91_000, 91_010))
HELD_OUT_ALIAS_LAYOUTS = tuple(range(95_000, 95_010))
SECOND_HELD_OUT_ALIAS_LAYOUTS = tuple(range(92_000, 92_010))

# Misalignment and information-destroying controls. Section H is explicit that
# U8 is judged on THESE: a restricted-input predictor is a weaker detector, not
# a corrupted event, and it is expected to keep whatever information it still
# sees. Scoring it as a corruption would fail the gate for the wrong reason.
DESTROYING = ("2_shift_forward", "3_shift_backward", "4_drop_one_event",
              "5_flip_one_event", "6_cross_episode_shuffle",
              "7_positionwise_permutation", "8_constant")


def mask_query(x: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    out[..., QUERY_ACTION] = 0.0
    return out


# ---- the event detector -----------------------------------------------------------------


class EventDetector:
    """p(C_t | X_{t-1}, A_{t-1}, X_t). Retrospective, and named that way on purpose.

    It answers "did a crossing just happen", not "will one happen next". A prospective
    predictor p(C_{t+1} | B_t, X_t, A_t) is a different model and was not trained.
    """

    RETROSPECTIVE = "p(C_t | X_{t-1}, A_{t-1}, X_t)"

    def __init__(self, mode: str = "full") -> None:
        self.mode = mode          # full | action_only | state_only
        self.model = None
        self.width = 0

    def featurise(self, sequences: np.ndarray) -> np.ndarray:
        """(N, L, W) -> (N, L, width). Query action masked out of both rows."""
        current = mask_query(sequences)
        previous = np.concatenate(
            [np.zeros_like(current[:, :1]), current[:, :-1]], axis=1)
        first = np.zeros(current.shape[:2] + (1,), dtype=np.float32)
        first[:, 1:, 0] = 1.0
        if self.mode == "action_only":
            keep = np.zeros_like(current)
            keep[..., core.PREVIOUS_ACTION] = current[..., core.PREVIOUS_ACTION]
            return np.concatenate([np.zeros_like(previous), keep, first], axis=-1)
        if self.mode == "state_only":
            return np.concatenate([np.zeros_like(previous), current, first], axis=-1)
        return np.concatenate([previous, current, first], axis=-1)

    def fit(self, items, seed: int = 6600, updates: int = 1024) -> "EventDetector":
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        x, y, e, m, _ = core.pad(items)
        design = self.featurise(x)
        valid = m.astype(bool)
        features, labels = design[valid], e[valid].astype(np.int32)
        self.width = features.shape[1]
        mx.random.seed(seed)

        class Head(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.a = nn.Linear(width, core.HIDDEN)
                self.head = nn.Linear(core.HIDDEN, 2)

            def __call__(self, z):
                return self.head(nn.relu(self.a(z)))

        model = Head(self.width)
        mx.eval(model.parameters())
        optimizer = optim.AdamW(learning_rate=2e-3)
        rng = np.random.default_rng(seed)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        for _ in range(updates):
            pick = np.concatenate([rng.choice(positive, 64), rng.choice(negative, 64)])
            xb, yb = mx.array(features[pick]), mx.array(labels[pick])

            def loss_fn(mo):
                return nn.losses.cross_entropy(mo(xb), yb, reduction="mean")

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
        self.model = model
        return self

    def probabilities(self, sequences: np.ndarray, batch: int = 2048) -> np.ndarray:
        import mlx.core as mx
        design = self.featurise(sequences)
        flat = design.reshape(-1, design.shape[-1])
        out = np.zeros(len(flat), dtype=np.float32)
        for start in range(0, len(flat), batch):
            logits = self.model(mx.array(flat[start:start + batch]))
            mx.eval(logits)
            logits = np.asarray(logits)
            shifted = logits - logits.max(axis=1, keepdims=True)
            exponent = np.exp(shifted)
            out[start:start + batch] = exponent[:, 1] / exponent.sum(axis=1)
        probability = out.reshape(design.shape[:2])
        probability[:, 0] = 0.0        # no transition precedes step 0
        return probability


def classification_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    truth = truth.astype(int)
    positive, negative = truth == 1, truth == 0
    recall = float(predicted[positive].mean()) if positive.any() else float("nan")
    specificity = float(1 - predicted[negative].mean()) if negative.any() else float("nan")
    chosen = predicted == 1
    precision = float(truth[chosen].mean()) if chosen.any() else 0.0
    denominator = precision + recall
    bins = np.clip((probability * 10).astype(int), 0, 9)
    calibration = 0.0
    for b in range(10):
        picked = bins == b
        if picked.any():
            calibration += picked.mean() * abs(
                float(probability[picked].mean()) - float(truth[picked].mean()))
    return {"balanced_accuracy": float((recall + specificity) / 2),
            "precision": precision, "recall": recall,
            "f1": float(2 * precision * recall / denominator) if denominator else 0.0,
            "brier": float(((probability - truth) ** 2).mean()),
            "expected_calibration_error": float(calibration),
            "positive_rate": float(truth.mean()), "n": int(len(truth))}


# ---- event corruption controls -----------------------------------------------------------


def corrupt(name: str, base: np.ndarray, lengths: np.ndarray, seed: int) -> np.ndarray:
    """Every control is derived from the SAME frozen base event array.

    Regenerating with a changed corruption mode shifts the RNG call path, which is how
    an earlier control ended up scored against events from different sequences.
    """
    rng = np.random.default_rng(seed)
    out = base.copy()
    if name == "1_correct":
        return out
    if name == "2_shift_forward":
        out[:, 1:] = base[:, :-1]
        out[:, 0] = 0.0
        return out
    if name == "3_shift_backward":
        out[:, :-1] = base[:, 1:]
        out[:, -1] = 0.0
        out[:, 0] = 0.0
        return out
    if name in ("4_drop_one_event", "5_flip_one_event"):
        for k, n in enumerate(lengths):
            positions = np.flatnonzero(base[k, 1:n] >= 0.5) + 1
            if name == "4_drop_one_event":
                if len(positions):
                    out[k, rng.choice(positions)] = 0.0
            else:
                out[k, rng.integers(1, max(n, 2))] = 1.0 - out[k, rng.integers(1, max(n, 2))]
        return out
    if name == "6_cross_episode_shuffle":
        for n in np.unique(lengths):
            members = np.flatnonzero(lengths == n)
            out[members] = base[rng.permutation(members)]
        return out
    if name == "7_positionwise_permutation":
        mask = np.zeros_like(base, dtype=bool)
        for k, n in enumerate(lengths):
            mask[k, 1:n] = True
        values = base[mask]
        out[mask] = rng.permutation(values)
        out[:, 0] = 0.0
        return out
    if name == "8_constant":
        return np.zeros_like(base)
    raise KeyError(name)


CORRUPTIONS = ("1_correct", "2_shift_forward", "3_shift_backward", "4_drop_one_event",
               "5_flip_one_event", "6_cross_episode_shuffle", "7_positionwise_permutation",
               "8_constant")


# ---- arms ---------------------------------------------------------------------------------


def coupling_arms() -> dict[str, FilterSpec]:
    anti = antisymmetric_two_state()
    return {
        "1_learned_event_exact_accumulator": FilterSpec("1", "accumulator"),
        "2_learned_event_learned_filter_2state": FilterSpec("2", "filter", 2,
                                                            "symmetry_broken",
                                                            perturbation=anti),
        "3_learned_event_learned_filter_8state": FilterSpec("3", "filter", 8, "default"),
        "4_learned_event_generic_gru": FilterSpec("4", "gru"),
        "5_learned_event_no_temporal_state": FilterSpec("5", "memoryless"),
        "6_memoryless": FilterSpec("6", "memoryless"),
    }


U7_ARM = "2_learned_event_learned_filter_2state"


def events_for(tensors, detector: EventDetector, coupling: str) -> np.ndarray:
    probability = detector.probabilities(tensors.z)
    return (probability >= 0.5).astype(np.float32) if coupling == "hard" else probability


def parity_fidelity(tensors, detector: EventDetector) -> dict[str, float]:
    """Per-step event accuracy, and the parity of the whole route.

    Parity is the quantity the belief actually depends on, and it is the product of the
    per-step accuracies along a route rather than their average: a detector at 0.86 per
    step over six steps is nowhere near 0.86 on the parity. Reporting only the per-step
    figure would make an arm that cannot work look nearly correct.
    """
    hard = (detector.probabilities(tensors.z) >= 0.5).astype(np.float32)
    valid = np.zeros_like(hard, dtype=bool)
    for k, n in enumerate(tensors.lengths):
        valid[k, 1:n] = True
    estimated = np.array([hard[k, 1:tensors.lengths[k]].sum() % 2
                          for k in range(len(tensors.lengths))])
    truth = np.array([tensors.events_true[k, 1:tensors.lengths[k]].sum() % 2
                      for k in range(len(tensors.lengths))])
    return {"per_step_accuracy": float((hard[valid]
                                        == tensors.events_true[valid]).mean()),
            "final_parity_accuracy": float((estimated == truth).mean()),
            "keys": int(len(tensors.keys))}


def collect_goal_directed(layouts, trajectories, steps, appearance, seed, epsilon=0.25):
    """Same environment, different visitation policy.

    v2 has ONE transition function -- SWITCH_COUNT is a constant and the flip rule never
    varies -- so there is no held-out-dynamics split to make, and inventing one would be
    a fiction. What can be held out is how states are reached, so this drives a
    goal-directed policy instead of the uniform-random one training saw. It is reported
    as a held-out visitation policy and not as held-out dynamics.
    """
    from sentinel.env.adapters.procedural_visual_v2 import (
        ACTIONS, ProceduralVisualV2Adapter)
    from sentinel.wm.authority import AuthorityGate
    from structured_calibration import DELTAS_BY_INDEX

    gate = AuthorityGate(gate_id="policy")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out = []
    for layout in layouts:
        for _ in range(trajectories):
            adapter.reset(layout, f"appearance:{appearance}")
            level = adapter._require()
            switches = {tuple(int(v) for v in c) for c in level.switches}
            walls = np.asarray(level.walls, dtype=bool)
            goal = tuple(int(v) for v in level.markers[adapter._goal_marker])
            rows, previous_action = [], -1
            for _step in range(steps):
                truth = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in truth["position"])
                snapshot = adapter.snapshot()
                successors = []
                for candidate in ACTIONS:
                    adapter.restore(snapshot)
                    adapter.step(candidate, gate.authorize_evaluator(candidate, "s"))
                    successors.append(tuple(int(v) for v in adapter.snapshot()
                                            .reveal("evaluator")["position"]))
                adapter.restore(snapshot)
                rows.append({"position": position, "switches": switches, "walls": walls,
                             "goal": goal, "previous_action": previous_action,
                             "blocked": float(truth["last_blocked"]),
                             "successors": successors,
                             "polarity": int(truth["polarity"]),
                             "crossings": int(truth["switch_crossings"])})
                if generator.random() < epsilon:
                    action = int(generator.integers(0, len(ACTIONS)))
                else:
                    scores = [abs(goal[0] - (position[0] + dr))
                              + abs(goal[1] - (position[1] + dc))
                              for dr, dc in DELTAS_BY_INDEX]
                    action = int(np.argmin(scores))
                previous_action = action
                if adapter.step(action, gate.authorize_evaluator(action, "r")).terminated:
                    break
            if len(rows) >= 3:
                out.append({"layout": layout, "rows": rows})
    return out


def detector_report(detector: EventDetector, splits: dict[str, list]) -> dict[str, Any]:
    """Section I. Everything is reported per split, per action and per phase-change count."""
    out: dict[str, Any] = {"scope": EventDetector.RETROSPECTIVE,
                           "is_prospective_predictor": False}
    for name, items in splits.items():
        x, y, e, m, _ = core.pad(items)
        probability = detector.probabilities(x)
        valid = m.astype(bool)
        valid[:, 0] = False        # step 0 has no preceding transition to detect
        block = {"overall": classification_metrics(e[valid], probability[valid])}
        action = np.argmax(x[..., core.PREVIOUS_ACTION], axis=-1) - 1
        for a in range(4):
            picked = valid & (action == a)
            if picked.sum() > 20:
                block[f"action_{a}"] = classification_metrics(
                    e[picked], probability[picked])
        crossings = np.cumsum(e, axis=1)
        for label, low, high in (("changes_0", 0, 1), ("changes_1", 1, 2),
                                 ("changes_2", 2, 3), ("changes_3", 3, 4),
                                 ("changes_4plus", 4, 99)):
            picked = valid & (crossings >= low) & (crossings < high)
            if picked.sum() > 20:
                block[label] = classification_metrics(e[picked], probability[picked])
        out[name] = block
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(VALIDATION_SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2d-coupling.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    core.check_feature_layout()
    appearance = CANONICAL_APPEARANCE_SEED

    train = build_dataset(collect(list(core.TRAIN_LAYOUTS), 3, 9, appearance, 11), 5)
    held_out_layouts = build_dataset(
        collect(list(core.DETECTOR_TEST_LAYOUTS), 2, 9, appearance, 777), 6)
    far_layouts = build_dataset(
        collect(list(core.HELD_OUT_LAYOUTS), 2, 9, appearance, 313), 7)
    policy_shift = build_dataset(
        collect_goal_directed(list(core.HELD_OUT_LAYOUTS), 2, 9, appearance, 404), 8)

    print(f"train {len(train)} trajectories; held-out layouts {len(held_out_layouts)}; "
          f"far layouts {len(far_layouts)}; policy shift {len(policy_shift)}", flush=True)

    detector = EventDetector("full").fit(train)
    action_only = EventDetector("action_only").fit(train)
    state_only = EventDetector("state_only").fit(train)
    splits = {"development_layouts": train, "held_out_layouts": held_out_layouts,
              "far_held_out_layouts": far_layouts,
              "held_out_visitation_policy": policy_shift}
    detection = detector_report(detector, splits)
    detection["action_only_predictor"] = detector_report(action_only, splits)
    detection["state_only_predictor"] = detector_report(state_only, splits)
    print(f"\ndetector {EventDetector.RETROSPECTIVE}")
    for name in splits:
        block = detection[name]["overall"]
        print(f"  {name:32s} balanced {block['balanced_accuracy']:.4f}  "
              f"F1 {block['f1']:.4f}  Brier {block['brier']:.4f}  "
              f"ECE {block['expected_calibration_error']:.4f}", flush=True)

    populations = {}
    for label, layouts in (("development", DEV_ALIAS_LAYOUTS),
                           ("validation", core.ALIAS_LAYOUTS),
                           ("held_out", HELD_OUT_ALIAS_LAYOUTS),
                           ("held_out_2", SECOND_HELD_OUT_ALIAS_LAYOUTS)):
        population = build_population(layouts)
        features = RouteFeatures(population)
        populations[label] = (population, build_tensors(population, features))
        print(f"{label:12s} alias population: {population.summary()}", flush=True)

    report: dict[str, Any] = {
        "detector": detection, "arms": {}, "identities": {},
        "dev_seeds": list(DEV_SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
        "populations": {k: v[0].summary() for k, v in populations.items()},
        "alias_layouts": {"development": list(DEV_ALIAS_LAYOUTS),
                          "validation": list(core.ALIAS_LAYOUTS),
                          "held_out": list(HELD_OUT_ALIAS_LAYOUTS),
                          "held_out_2": list(SECOND_HELD_OUT_ALIAS_LAYOUTS)}}

    report["parity_fidelity"] = {
        label: parity_fidelity(value[1], detector) for label, value in populations.items()}
    print("\nevent fidelity on each alias population (parity is what the belief needs)")
    for label, block in report["parity_fidelity"].items():
        print(f"  {label:14s} per-step {block['per_step_accuracy']:.4f}  "
              f"final-parity {block['final_parity_accuracy']:.4f}", flush=True)

    # ---- coupling selection on DEVELOPMENT layouts and DEVELOPMENT seeds ----------------
    development, dev_tensors = populations["development"]
    print("\ncoupling selection on the development alias population "
          f"(layouts {DEV_ALIAS_LAYOUTS[0]}-{DEV_ALIAS_LAYOUTS[-1]}, "
          f"{len(DEV_SEEDS)} development seeds)")
    selection: dict[str, float] = {}
    for coupling in ("hard", "posterior"):
        events = events_for(dev_tensors, detector, coupling)
        scores = []
        for seed in DEV_SEEDS:
            model, _ = train_model(coupling_arms()[U7_ARM], train, seed)
            scores.append(float(score_population(model, dev_tensors, events)["hit"].mean()))
        selection[coupling] = float(np.mean(scores))
        print(f"  {coupling:10s} {selection[coupling]:.4f}  "
              f"per-seed {[round(s, 4) for s in scores]}", flush=True)
    selected = max(selection, key=lambda k: selection[k])
    report["coupling_selection"] = {"scores": selection, "selected": selected,
                                    "population": "development alias layouts",
                                    "seeds": list(DEV_SEEDS)}
    print(f"  selected coupling: {selected}")

    # ---- validation ---------------------------------------------------------------------
    seeds = VALIDATION_SEEDS[:arguments.seeds]
    population, tensors = populations["validation"]
    strata = stratify(population)
    hits: dict[str, dict[str, np.ndarray]] = {}
    print(f"\nvalidation alias population, {len(seeds)} seeds, coupling={selected}")
    print(f"{'arm':44s} {'alias acc':>9s} {'p10':>7s} {'min':>7s} {'NLL':>7s} "
          f"{'Brier':>7s} {'margin':>8s}")
    print("-" * 96)
    for name, spec in coupling_arms().items():
        for coupling in ("hard", "posterior"):
            if name == "6_memoryless" and coupling == "posterior":
                continue
            events = (None if name == "6_memoryless"
                      else events_for(tensors, detector, coupling))
            per_seed, records = [], []
            for seed in seeds:
                model, count = train_model(spec, train, seed)
                scored = score_population(model, tensors, events)
                per_seed.append(scored["hit"])
                records.append({"seed": seed, "alias_accuracy": float(scored["hit"].mean()),
                                "nll": float(scored["nll"].mean()),
                                "brier": float(scored["brier"].mean()),
                                "margin": float(scored["margin"].mean()),
                                "parameters": count,
                                "checkpoint": checkpoint_hash(model)})
                if seed == seeds[0]:
                    report["identities"][f"{name}::{coupling}"] = ArmIdentity(
                        arm_id=f"{name}::{coupling}",
                        event_source=("none" if name == "6_memoryless"
                                      else f"learned_{coupling}"),
                        temporal_mechanism=MECHANISM[spec.kind],
                        model_class=type(model).__name__,
                        checkpoint_hash=checkpoint_hash(model),
                        initialization_rule=spec.initialization_rule,
                        trainable_parameters=count,
                        supervision="displacement class; event detector trained on "
                                    "evaluator-derived labels for a PUBLIC quantity",
                        input_fields=(INPUTS_MEMORYLESS if spec.kind == "memoryless"
                                      else INPUTS_FILTER),
                        seed=seed, population=population_label(population),
                        metric="pairwise outcome accuracy on differing successors",
                        query_budget="one forward pass per (state, action)").to_dict()
            key = f"{name}::{coupling}"
            hits[key] = {"hit": np.concatenate(per_seed),
                         "per_seed": np.stack(per_seed)}
            stats = summarise_metric(np.array([r["alias_accuracy"] for r in records]))
            stats.update({"nll": float(np.mean([r["nll"] for r in records])),
                          "brier": float(np.mean([r["brier"] for r in records])),
                          "margin": float(np.mean([r["margin"] for r in records]))})
            report["arms"][key] = {"stats": stats, "records": records}
            marker = "  <- selected" if coupling == selected else ""
            print(f"{key:44s} {stats['mean']:9.4f} {stats['p10']:7.4f} "
                  f"{stats['minimum']:7.4f} {stats['nll']:7.4f} {stats['brier']:7.4f} "
                  f"{stats['margin']:8.4f}{marker}", flush=True)

    baseline = "6_memoryless::hard"
    rows = len(population.rows)
    seed_column = np.repeat(np.array(seeds), rows)
    layout_column = np.tile(strata["layout"], len(seeds))
    class_column = np.tile(strata["alias_class"], len(seeds))
    print("\npaired hierarchical intervals against the trained memoryless model")
    for key in report["arms"]:
        if key == baseline:
            continue
        interval = core.hierarchical_paired_interval(
            hits[key]["hit"], hits[baseline]["hit"], seed_column, layout_column,
            class_column)
        report["arms"][key].setdefault("intervals", {})["vs_memoryless"] = interval
        print(f"  {key:44s} {interval['delta']:+.4f}  "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{'  *' if interval['excludes_zero'] else ''}", flush=True)

    u7_key = f"{U7_ARM}::{selected}"
    gru_key = f"4_learned_event_generic_gru::{selected}"
    accumulator_key = f"1_learned_event_exact_accumulator::{selected}"
    for label, other in (("vs_generic_gru", gru_key),
                         ("vs_exact_accumulator", accumulator_key),
                         ("hard_vs_posterior", f"{U7_ARM}::hard")):
        if other == u7_key:
            continue
        report["arms"][u7_key]["intervals"][label] = core.hierarchical_paired_interval(
            hits[u7_key]["hit"], hits[other]["hit"], seed_column, layout_column,
            class_column)

    # ---- section G: survival after repeated phase changes -------------------------------
    print("\nby number of phase changes (pair minimum, so both directions stay together)")
    changes = np.tile(strata["changes"], len(seeds))
    step = np.tile(strata["step"], len(seeds))
    since = np.tile(strata["since_change"], len(seeds))
    action_column = np.tile(strata["action"], len(seeds))
    survival: dict[str, Any] = {}
    print(f"{'stratum':16s} {'rows':>7s} {'memoryless':>11s} "
          f"{'learned+filter':>15s} {'delta':>8s}  interval")
    for label, mask in (("0", changes == 0), ("1", changes == 1), ("2", changes == 2),
                        ("3", changes == 3), ("4plus", changes >= 4),
                        ("2plus", changes >= 2)):
        if mask.sum() < 50:
            continue
        interval = core.hierarchical_paired_interval(
            hits[u7_key]["hit"], hits[baseline]["hit"], seed_column, layout_column,
            class_column, mask=mask)
        survival[f"changes_{label}"] = {
            "memoryless": float(hits[baseline]["hit"][mask].mean()),
            "learned_event_learned_filter": float(hits[u7_key]["hit"][mask].mean()),
            **interval}
        print(f"changes={label:9s} {int(mask.sum()):7d} "
              f"{hits[baseline]['hit'][mask].mean():11.4f} "
              f"{hits[u7_key]['hit'][mask].mean():15.4f} {interval['delta']:+8.4f}  "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{'  *' if interval['excludes_zero'] else ''}", flush=True)
    for label, column, values in (("step", step, sorted(set(strata["step"].tolist()))),
                                  ("since_change", since,
                                   sorted(set(strata["since_change"].tolist()))),
                                  ("action", action_column, [0, 1, 2, 3])):
        for value in values:
            mask = column == value
            if mask.sum() < 50:
                continue
            survival[f"{label}_{value}"] = {
                "memoryless": float(hits[baseline]["hit"][mask].mean()),
                "learned_event_learned_filter": float(hits[u7_key]["hit"][mask].mean()),
                "rows": int(mask.sum())}
    report["survival"] = survival

    # ---- held-out alias layouts ----------------------------------------------------------
    print("\ntransfer to other alias layout sets (true events are the positive control)")
    transfer: dict[str, Any] = {}
    for label in ("development", "validation", "held_out", "held_out_2"):
        other, other_tensors = populations[label]
        other_events = events_for(other_tensors, detector, selected)
        block: dict[str, Any] = {}
        for name, spec, events in (
                ("true_event_filter", coupling_arms()[U7_ARM], None),
                ("learned_event_filter", coupling_arms()[U7_ARM], other_events),
                ("memoryless", coupling_arms()["6_memoryless"], None)):
            scores = []
            for seed in seeds[:5]:
                model, _ = train_model(spec, train, seed)
                scores.append(float(score_population(
                    model, other_tensors, events)["hit"].mean()))
            block[name] = {"mean": float(np.mean(scores)), "per_seed": scores}
        block["final_parity_accuracy"] = report["parity_fidelity"][label][
            "final_parity_accuracy"]
        transfer[label] = block
        print(f"  {label:14s} memoryless {block['memoryless']['mean']:.4f}  "
              f"true-event {block['true_event_filter']['mean']:.4f}  "
              f"learned-event {block['learned_event_filter']['mean']:.4f}  "
              f"(parity {block['final_parity_accuracy']:.4f})", flush=True)
    report["transfer"] = transfer

    # ---- section H: event corruption -----------------------------------------------------
    print("\nevent corruption controls, all derived from one frozen base sequence")
    base = events_for(tensors, detector, selected)
    lengths = tensors.lengths
    corruption: dict[str, Any] = {}
    print(f"{'control':32s} {'alias acc':>9s} {'delta vs memoryless':>20s}  interval")
    for control in CORRUPTIONS:
        events = corrupt(control, base, lengths, seed=515)
        per_seed = []
        for seed in seeds:
            model, _ = train_model(coupling_arms()[U7_ARM], train, seed)
            per_seed.append(score_population(model, tensors, events)["hit"])
        stacked = np.concatenate(per_seed)
        interval = core.hierarchical_paired_interval(
            stacked, hits[baseline]["hit"], seed_column, layout_column, class_column)
        corruption[control] = {"alias_accuracy": float(stacked.mean()), **interval}
        print(f"{control:32s} {stacked.mean():9.4f} {interval['delta']:+20.4f}  "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{'  *' if interval['excludes_zero'] else ''}", flush=True)
    for name, alternative in (("9_action_only_predictor", action_only),
                              ("10_state_only_predictor", state_only)):
        events = events_for(tensors, alternative, selected)
        per_seed = []
        for seed in seeds:
            model, _ = train_model(coupling_arms()[U7_ARM], train, seed)
            per_seed.append(score_population(model, tensors, events)["hit"])
        stacked = np.concatenate(per_seed)
        interval = core.hierarchical_paired_interval(
            stacked, hits[baseline]["hit"], seed_column, layout_column, class_column)
        corruption[name] = {"alias_accuracy": float(stacked.mean()), **interval}
        print(f"{name:32s} {stacked.mean():9.4f} {interval['delta']:+20.4f}  "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{'  *' if interval['excludes_zero'] else ''}", flush=True)
    report["corruption"] = corruption

    correct = corruption["1_correct"]["delta"]
    report["c8_corruptions_remove_the_advantage"] = bool(
        correct > 0 and all(not corruption[k]["excludes_zero"]
                            or corruption[k]["delta"] < 0 for k in DESTROYING))
    report["c8_judged_on"] = list(DESTROYING)
    frozen = ARTIFACTS / "m2d-coupling-predictions.npz"
    report["frozen_predictions"] = {
        "path": str(frozen.relative_to(core.REPO)),
        "sha256_16": core.save_predictions(
            frozen, {**{f"hit::{k}": v["per_seed"] for k, v in hits.items()},
                     "seeds": np.array(seeds),
                     "row_layout": strata["layout"], "row_pair": strata["pair"],
                     "row_changes": strata["changes"], "row_step": strata["step"],
                     "row_action": strata["action"],
                     "row_alias_class": strata["alias_class"],
                     "events_true": tensors.events_true,
                     "events_learned": base}),
        "contents": "per-arm hit matrices (seeds x rows), row strata, event arrays"}
    report["detector_input_ablations"] = {
        k: corruption[k] for k in ("9_action_only_predictor", "10_state_only_predictor")
        if k in corruption}
    report["u7_learned_event_learned_filter_beats_memoryless"] = bool(
        report["arms"][u7_key]["intervals"]["vs_memoryless"]["ci_low"] > 0)
    report["u7_arm_key"] = u7_key
    report["c7_survives_two_changes"] = bool(
        survival.get("changes_2plus", {}).get("ci_low", -1) > 0)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nU7 ({u7_key} beats memoryless on exact alias pairs): "
          f"{report['u7_learned_event_learned_filter_beats_memoryless']}")
    print(f"C7 (survives two or more phase changes): {report['c7_survives_two_changes']}")
    print(f"C8 (every corruption removes the advantage): "
          f"{report['c8_corruptions_remove_the_advantage']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
