"""Training the core, and the gate that judges it.

The gate is the point of this module. Training a network until a loss
number goes down proves nothing; the plan's Phase 3 exit is specific, and
it is written to be *failable*:

    the core must infer mechanics on held-out mechanic COMBINATIONS far
    better than the majority-class baseline, and in particular must detect
    `charge_period` -- a counter that appears nowhere in any frame.

Both halves matter. Beating the baseline on unseen seeds only shows the
core can interpolate over rules it was taught. Detecting `charge_period` is
the sharp test: it is invisible in any single observation and recoverable
only from a pattern across a sequence, so a model that gets it right has
necessarily posited structure it could not see. That is claim 1 of the plan
in its smallest testable form.

If it fails, the honest conclusion is that architecture-over-scale does not
carry this task at this size, and the plan says to reassess rather than
grind.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import Dataset, iterate_batches, majority_baseline
from .encoding import HEADS
from .model import CoreConfig, TinyRecursiveCore, loss_fn


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    eval_every: int = 1
    patience: int = 8
    """Stop when held-out accuracy has not improved for this many epochs.
    Guards against reading a late overfitting spike as success."""


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    accuracies: dict[str, float] = field(default_factory=dict)
    mean_accuracy: float = 0.0
    seconds: float = 0.0

    def summary(self) -> str:
        per_head = " ".join(f"{k[:6]}={v:.2f}" for k, v in self.accuracies.items())
        return (
            f"epoch {self.epoch:3} loss={self.train_loss:.4f} "
            f"mean_acc={self.mean_accuracy:.3f}  {per_head}  ({self.seconds:.0f}s)"
        )


def evaluate(model: TinyRecursiveCore, dataset: Dataset, batch_size: int = 64) -> dict[str, float]:
    """Per-head accuracy over a dataset."""
    correct = np.zeros(len(HEADS))
    total = 0
    rng = np.random.default_rng(0)
    for grids, actions, labels in iterate_batches(dataset, batch_size, rng, shuffle=False):
        logits = model(mx.array(grids), mx.array(actions))
        mx.eval(logits)
        for i, head_logits in enumerate(logits):
            pred = np.array(mx.argmax(head_logits, axis=-1))
            correct[i] += float(np.sum(pred == labels[:, i]))
        total += len(labels)
    return {name: float(correct[i] / total) for i, (name, _) in enumerate(HEADS)}


def train(
    train_set: Dataset,
    eval_set: Dataset,
    core_config: CoreConfig | None = None,
    train_config: TrainConfig | None = None,
    verbose: bool = True,
) -> tuple[TinyRecursiveCore, list[EpochResult]]:
    cfg = train_config or TrainConfig()
    model = TinyRecursiveCore(core_config or CoreConfig())
    mx.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    optimizer = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    if verbose:
        print(f"core: {model.cfg.describe()}  {model.parameter_count():,} parameters")
        print(f"train: {train_set.summary()}")
        print(f"eval : {eval_set.summary()}")

    history: list[EpochResult] = []
    best_mean = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        losses: list[float] = []

        for grids, actions, labels in iterate_batches(train_set, cfg.batch_size, rng):
            loss, grads = loss_and_grad(
                model, mx.array(grids), mx.array(actions), mx.array(labels)
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss))

        result = EpochResult(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            seconds=time.perf_counter() - started,
        )

        if epoch % cfg.eval_every == 0:
            result.accuracies = evaluate(model, eval_set)
            # Average only over heads that can actually move. Including
            # single-class heads made the early-stopping signal mostly
            # constant noise, which fired the patience counter at epoch 9
            # while the model was still at the class prior — stopping the
            # run long before it had begun to learn anything.
            scorable = [
                result.accuracies[name]
                for i, (name, _) in enumerate(HEADS)
                if len(np.unique(eval_set.labels[:, i])) > 1
            ]
            result.mean_accuracy = float(
                np.mean(scorable if scorable else list(result.accuracies.values()))
            )
            if result.mean_accuracy > best_mean + 1e-4:
                best_mean, best_epoch = result.mean_accuracy, epoch

        history.append(result)
        if verbose:
            print(result.summary(), flush=True)

        if epoch - best_epoch >= cfg.patience:
            if verbose:
                print(f"early stop: no improvement in {cfg.patience} epochs")
            break

    return model, history


@dataclass
class GateResult:
    """The Phase 3 verdict."""

    accuracies: dict[str, float]
    baseline: dict[str, float]
    lift: dict[str, float]
    mean_accuracy: float
    mean_baseline: float
    charge_accuracy: float
    charge_baseline: float
    passed: bool
    scored_heads: list[str] = field(default_factory=list)
    unscored_heads: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "PHASE 3 GATE — held-out MECHANIC COMBINATIONS (never seen in training)",
            "",
            f"{'label':18} {'core':>8} {'baseline':>9} {'lift':>8}",
        ]
        for name in self.accuracies:
            mark = "" if name in self.scored_heads else "   (not scorable)"
            lines.append(
                f"{name:18} {self.accuracies[name]:8.3f} "
                f"{self.baseline[name]:9.3f} {self.lift[name]:+8.3f}{mark}"
            )
        if self.unscored_heads:
            lines += [
                "",
                "  not scorable: " + ", ".join(self.unscored_heads),
                "  (single-class in holdout, or absent from training — no lift is possible)",
            ]
        lines += [
            "",
            f"{'mean (scorable)':18} {self.mean_accuracy:8.3f} {self.mean_baseline:9.3f} "
            f"{self.mean_accuracy - self.mean_baseline:+8.3f}",
            "",
            f"charge_period (the hidden-state test): "
            f"{self.charge_accuracy:.3f} vs baseline {self.charge_baseline:.3f}",
            "",
            "VERDICT: " + ("PASS" if self.passed else "FAIL"),
        ]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def run_gate(
    model: TinyRecursiveCore,
    train_set: Dataset,
    holdout: Dataset,
    min_mean_lift: float = 0.10,
    min_charge_lift: float = 0.10,
) -> GateResult:
    """Judge the core on unseen mechanic combinations.

    Thresholds are modest on purpose. The question at this stage is whether
    the architecture learns *anything* that transfers to rules it was never
    shown — not whether it is good. A weak but real signal justifies
    continuing; no signal is the kill criterion.
    """
    accuracies = evaluate(model, holdout)
    baseline = majority_baseline(train_set, holdout)
    lift = {k: accuracies[k] - baseline[k] for k in accuracies}

    # Average only over heads where lift is *possible*. A head whose holdout
    # contains one class is scored 100% by the core and 100% by the
    # baseline, and a head absent from training cannot be learned at all.
    # Including either in the mean measures nothing while diluting the heads
    # that do measure something — it would make the gate look harder while
    # actually making it less informative.
    informative = [
        name
        for i, (name, _) in enumerate(HEADS)
        if len(np.unique(holdout.labels[:, i])) > 1
        and len(np.unique(train_set.labels[:, i])) > 1
    ]
    uninformative = [name for name, _ in HEADS if name not in informative]

    scored = informative or list(accuracies)
    mean_acc = float(np.mean([accuracies[k] for k in scored]))
    mean_base = float(np.mean([baseline[k] for k in scored]))
    charge_acc = accuracies.get("charge_period", 0.0)
    charge_base = baseline.get("charge_period", 0.0)

    reasons: list[str] = []
    if mean_acc - mean_base < min_mean_lift:
        reasons.append(
            f"mean lift {mean_acc - mean_base:+.3f} below required {min_mean_lift:+.3f}"
        )
    if charge_acc - charge_base < min_charge_lift:
        reasons.append(
            f"charge_period lift {charge_acc - charge_base:+.3f} below required "
            f"{min_charge_lift:+.3f} — no evidence of inferring hidden state"
        )
    if not reasons:
        reasons.append("core generalizes to mechanic combinations it never saw")

    return GateResult(
        accuracies=accuracies,
        baseline=baseline,
        lift=lift,
        mean_accuracy=mean_acc,
        mean_baseline=mean_base,
        charge_accuracy=charge_acc,
        charge_baseline=charge_base,
        scored_heads=scored,
        unscored_heads=uninformative,
        passed=not (mean_acc - mean_base < min_mean_lift or charge_acc - charge_base < min_charge_lift),
        reasons=reasons,
    )
