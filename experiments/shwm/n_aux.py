"""D / G / H driver. Public auxiliary heads per interface, per split, with controls.

A trainable interface is trained JOINTLY with its head. An earlier draft encoded with an
untrained CNN and fitted only the head, which would have entered the table as a
"convolutional interface" while actually being a random projection with extra steps.
Frozen interfaces precompute their slots once and share the identical head.

    .venv-shwm/bin/python experiments/shwm/n_aux.py [--interfaces 1,2,3,8]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import n_core as core
import n_heads as heads
import n_interfaces as ifaces
from m2d_core import ARTIFACTS, write
from n_heads import TARGETS, CONTROLS, HEADLINE

SEEDS = (31_000, 31_001, 31_002)


def joint_module(interface, out_dim: int, kind: str, seed: int):
    """Encoder + head as one trainable module, for interfaces whose encoder learns.

    The head half is byte-identical in shape to `n_heads.build_head`, so a trainable
    interface differs from a frozen one only by the convolutions in front.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)
    channels = interface.channels
    spatial = kind in ("multilabel", "spatial_scalar")

    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(6, channels, 3, padding=1)
            self.c2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            self.c3 = nn.Conv2d(channels, ifaces.SLOT_WIDTH, 3, padding=1)
            self.reduce = nn.Conv2d(ifaces.SLOT_WIDTH, heads.REDUCED, 1)
            self.mix = nn.Conv2d(heads.REDUCED + 4, heads.HEAD_HIDDEN, 3, padding=1)
            self.out = nn.Conv2d(heads.HEAD_HIDDEN,
                                 1 if spatial else heads.HEAD_HIDDEN, 1)
            self.pooled = None if spatial else nn.Linear(2 * heads.HEAD_HIDDEN, out_dim)

        def slots(self, frames):
            h = nn.relu(self.c1(frames))
            h = nn.relu(self.c2(h))               # 24 -> 12, cell-aligned
            return self.c3(h)

        def __call__(self, frames, action):
            s = self.slots(frames)
            h = nn.relu(self.reduce(s))
            n, height, width, _ = h.shape
            broadcast = mx.broadcast_to(
                action.reshape(n, 1, 1, 4), (n, height, width, 4))
            h = nn.relu(self.mix(mx.concatenate([h, broadcast], axis=-1)))
            z = self.out(h)
            if spatial:
                rows = mx.array(heads.nearest_index(height, core.GRID))
                columns = mx.array(heads.nearest_index(width, core.GRID))
                z = mx.take(mx.take(z, rows, axis=1), columns, axis=2)
                return z.reshape(n, core.GRID * core.GRID)
            # Max over space, plus the spatial mean as an additive term. A bare max
            # is an existential -- right for "a switch was entered somewhere", blind to
            # "nothing happened anywhere", which is the no-move class. Adding the mean
            # restores that without a new layer: an extra Linear on top of an unbounded
            # post-max activation collapsed every head to a constant, and collapsed it
            # to the SAME constant for three very different interfaces, which is what
            # gave the bug away.
            flat = z.reshape(n, height * width, z.shape[-1])
            return mx.max(flat, axis=1) + mx.mean(flat, axis=1)

    model = Joint()
    mx.eval(model.parameters())
    return model


def train_joint(interface, pairs, y, kind, out_dim, seed, updates=ifaces.UPDATES):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    model = joint_module(interface, out_dim, kind, seed)
    count = heads.parameter_count(model)
    loss_fn = heads._loss(kind)
    optimizer = optim.AdamW(learning_rate=ifaces.LEARNING_RATE)
    rng = np.random.default_rng(seed)
    frames = np.concatenate([pairs.before, pairs.after], axis=-1).astype(np.float32)
    action = pairs.action
    target = y.astype(np.int32) if kind == "categorical" else y.astype(np.float32)
    for _ in range(updates):
        pick = rng.integers(0, len(frames), min(ifaces.BATCH, len(frames)))
        xb, ab = mx.array(frames[pick]), mx.array(action[pick])
        yb = mx.array(target[pick])

        def objective(m):
            return loss_fn(m(xb, ab), yb)

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)
    return model, count


def predict_joint(model, pairs, batch=1024) -> np.ndarray:
    import mlx.core as mx
    frames = np.concatenate([pairs.before, pairs.after], axis=-1).astype(np.float32)
    out = []
    for start in range(0, len(frames), batch):
        block = model(mx.array(frames[start:start + batch]),
                      mx.array(pairs.action[start:start + batch]))
        mx.eval(block)
        out.append(np.asarray(block))
    return np.concatenate(out)


def build_interfaces(selected: set[str]) -> dict[str, Any]:
    catalogue = {
        "1": ifaces.EquivariantCNN(),
        "2": ifaces.RandomProjectionInterface(),
        "3": ifaces.LearnedSlotCNN(),
        "4": ifaces.BackboneSlots("qwen3_vl_4b", pooled=False),
        "5": ifaces.BackboneSlots("gemma3_4b", pooled=False),
        "6": ifaces.BackboneSlots("qwen3_vl_4b", pooled=True),
        "7": ifaces.BackboneSlots("gemma3_4b", pooled=True),
        "8": ifaces.CellAlignedDiagnostic(),
    }
    return {k: v for k, v in catalogue.items() if k in selected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interfaces", default="1,2,3,8")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--updates", type=int, default=0,
                        help="override the shared update count for every interface")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "n-auxiliary.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    if arguments.updates:
        ifaces.UPDATES = arguments.updates

    print("building frozen frame-pair splits", flush=True)
    episodes = core.splits(trajectories=arguments.trajectories)
    pairs = {k: core.to_pairs(v) for k, v in episodes.items()}
    for name, block in pairs.items():
        print(f"  {name:32s} {len(block):5d} pairs  {len(episodes[name]):4d} episodes  "
              f"event rate {block.event.mean():.4f}  digest {block.digest()}", flush=True)
    train = pairs["train"]
    evaluation = {k: v for k, v in pairs.items() if k != "train"}

    selected = set(arguments.interfaces.split(","))
    catalogue = build_interfaces(selected)
    report: dict[str, Any] = {
        "splits": {k: {"pairs": len(v), "episodes": len(episodes[k]),
                       "event_rate": float(v.event.mean()), "digest": v.digest(),
                       "layouts": sorted(set(int(x) for x in v.layout))}
                   for k, v in pairs.items()},
        "seeds": list(SEEDS[:arguments.seeds]),
        "no_dynamics_split": ("v2 has one transition function: SWITCH_COUNT is constant "
                              "and the flip rule never varies, so no dynamics "
                              "generalisation is measured or claimed"),
        "interfaces": {}}

    for key, interface in sorted(catalogue.items()):
        trainable = isinstance(interface, ifaces.EquivariantCNN)
        print(f"\n=== interface {interface.name} "
              f"({'trainable encoder' if trainable else 'frozen encoder'}) ===",
              flush=True)
        frozen_slots = None
        if not trainable:
            encoded = interface.encode(train)
            frozen_slots = encoded.slots
            print(f"  slots {frozen_slots.shape}  trainable {encoded.trainable_parameters}"
                  f"  frozen {encoded.frozen_parameters}  [{encoded.note}]", flush=True)

        block: dict[str, Any] = {"name": interface.name, "eligible": interface.eligible,
                                 "trainable_encoder": trainable, "targets": {}}
        stored: dict[str, dict[str, np.ndarray]] = {}
        for target, (field, kind, out_dim, identifiable) in TARGETS.items():
            per_seed: dict[str, list] = {}
            parameters = 0
            fit_where = heads.identifiable_mask(identifiable, train)
            fit_pairs = train if identifiable is None else train.subset(fit_where)
            for seed in SEEDS[:arguments.seeds]:
                y = getattr(fit_pairs, field)
                if trainable:
                    model, parameters = train_joint(interface, fit_pairs, y, kind,
                                                    out_dim, seed)
                    infer = lambda p, m=model: predict_joint(m, p)
                else:
                    fit_slots = (frozen_slots if identifiable is None
                                 else interface.encode(fit_pairs).slots)
                    model, parameters = heads.train_target(
                        fit_slots, fit_pairs.action, y, kind, out_dim, seed)
                    infer = lambda p, m=model, i=interface: heads.predict(
                        m, i.encode(p).slots, p.action)
                for split, block_pairs in evaluation.items():
                    logits = infer(block_pairs)
                    truth = getattr(block_pairs, field)
                    per_seed.setdefault(split, []).append(
                        heads.score(kind, logits, truth))
                    where = heads.identifiable_mask(identifiable, block_pairs)
                    if identifiable is not None and where.sum() > 20:
                        per_seed.setdefault(f"{split}::identifiable", []).append(
                            heads.score(kind, logits[where], truth[where]))
                    if target in ("2_agent_mask_after", "4_switch_mask_before",
                                  "1_agent_mask_before", "6_retrospective_event"):
                        stored.setdefault(split, {})[f"{target}::{seed}"] = logits
            block["targets"][target] = {
                "kind": kind, "parameters": parameters,
                "by_split": {s: {m: float(np.mean([r[m] for r in rs]))
                                 for m in rs[0]} for s, rs in per_seed.items()}}
            by_split = block["targets"][target]["by_split"]
            key_metric = HEADLINE[kind]
            headline = by_split["held_out_layouts"]
            extra = by_split.get("held_out_layouts::identifiable")
            suffix = (f"   identifiable-subset {extra[key_metric]:.4f}" if extra else "")
            block["targets"][target]["identifiable_subset"] = identifiable
            print(f"  {target:26s} params {parameters:7d}  held-out "
                  f"{key_metric} {headline.get(key_metric, float('nan')):.4f}{suffix}",
                  flush=True)

        # Section D: the event DERIVED from predicted masks and displacement.
        derived: dict[str, Any] = {}
        for split in evaluation:
            values = []
            for seed in SEEDS[:arguments.seeds]:
                keys = stored.get(split, {})
                if all(f"{t}::{seed}" in keys for t in
                       ("2_agent_mask_after", "4_switch_mask_before",
                        "1_agent_mask_before")):
                    probability = heads.derived_event(
                        keys[f"2_agent_mask_after::{seed}"],
                        keys[f"4_switch_mask_before::{seed}"],
                        keys[f"1_agent_mask_before::{seed}"])
                    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6)
                                    / (1 - np.clip(probability, 1e-6, 1 - 1e-6)))
                    values.append(heads.binary_metrics(logits[:, None],
                                                       evaluation[split].event))
            if values:
                derived[split] = {m: float(np.mean([v[m] for v in values]))
                                  for m in values[0]}
        block["event_derived_from_masks"] = derived
        if "held_out_layouts" in derived:
            print(f"  {'event via masks+displacement':26s} held-out balanced "
                  f"{derived['held_out_layouts']['balanced_accuracy']:.4f}", flush=True)

        report["interfaces"][key] = block

    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
