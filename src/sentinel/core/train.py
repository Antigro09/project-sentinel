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
from mlx.utils import tree_map
import numpy as np

from .data import Dataset, iterate_batches, majority_baseline
from .encoding import HEADS, defined_mask
from .model import CoreConfig, TinyRecursiveCore, loss_fn


@dataclass
class TrainConfig:
    epochs: int = 250
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    eval_every: int = 1
    patience: int = 60
    """Epochs without improvement in the WORST scorable head before stopping.

    Generous because the labels learn at very different rates, and the
    interesting one is the slowest."""


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


def evaluate(
    model: TinyRecursiveCore,
    dataset: Dataset,
    batch_size: int = 64,
    only_defined: bool = True,
) -> dict[str, float]:
    """Per-head accuracy, scored only where the label means anything.

    `only_defined` skips rows whose label is unanswerable for structural
    reasons -- whether waiting ticks a counter that does not exist, whether
    gates start open in a world with no gates. Scoring those rewards a lucky
    guess and punishes an honest one, and it inflated `gates_start_open`
    across 46% of the held-out episodes. See `encoding.defined_mask`.
    """
    correct = np.zeros(len(HEADS))
    counted = np.zeros(len(HEADS))
    rng = np.random.default_rng(0)
    for grids, actions, labels in iterate_batches(dataset, batch_size, rng, shuffle=False):
        logits = model(mx.array(grids), mx.array(actions))
        mx.eval(logits)
        for i, head_logits in enumerate(logits):
            pred = np.array(mx.argmax(head_logits, axis=-1))
            mask = defined_mask(labels, i) if only_defined else np.ones(len(labels), dtype=bool)
            correct[i] += float(np.sum((pred == labels[:, i]) & mask))
            counted[i] += float(mask.sum())
    return {
        name: float(correct[i] / max(1.0, counted[i]))
        for i, (name, _) in enumerate(HEADS)
    }


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

    # Per-head majority-class baselines, on the rows where each label is
    # defined. A head sitting at its prior has learned nothing, and using it
    # as a stopping signal means waiting on noise.
    priors: dict[str, float] = {}
    scorable_names: set[str] = set()
    for i, (name, _) in enumerate(HEADS):
        mask = defined_mask(eval_set.labels, i)
        values = eval_set.labels[mask][:, i]
        if len(np.unique(values)) > 1:
            scorable_names.add(name)
        _, counts = np.unique(values, return_counts=True)
        priors[name] = float(counts.max() / max(1, counts.sum()))

    optimizer = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    if verbose:
        print(f"core: {model.cfg.describe()}  {model.parameter_count():,} parameters")
        print(f"train: {train_set.summary()}")
        print(f"eval : {eval_set.summary()}")

    history: list[EpochResult] = []
    best_watched = -1.0
    best_selection = -1.0
    best_weights = None
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

            # Watch the worst head, scored WITHOUT the defined-mask.
            #
            # Four criteria were measured and three failed, each in a
            # different direction:
            #
            #   mean            saturated heads hold it flat; charge_period
            #                   0.409 +/- 0.022, pinned to its prior,
            #                   against 0.726 for training on
            #   minimum, masked a permanently stuck head vetoes everything.
            #                   `ordered_targets` is under-determined by the
            #                   evidence -- 6% of worlds under random play --
            #                   so it never improves and the run ends:
            #                   charge_period 0.559 -> 0.317
            #   any head        noise in a stuck head keeps training alive
            #                   indefinitely: step_distance 0.975 +/- 0.001
            #                   -> 0.658 +/- 0.306, seeds ending 1483s and
            #                   751s in different places
            #   min over heads  empty early in training, when nothing has
            #   above prior     beaten its prior yet, so it degenerates to
            #                   the mean: charge_period 0.263, below prior
            #
            # What works is the minimum over UNMASKED accuracies, and the
            # reason is mechanical rather than mysterious. Masking sends
            # `gates_start_open` to ~0.997, which makes the minimum
            # `ordered_targets` -- a head that cannot move. Unmasked, the
            # minimum is `wait_advances_charge` at ~0.55, a head that is
            # partly learnable and creeps upward for hundreds of epochs,
            # which is exactly the slow clock the hidden counter needs.
            #
            # So the stopping signal and the report are different
            # instruments and are computed differently on purpose: this one
            # is chosen because it keeps training alive while a slow head is
            # still moving, and `evaluate` masks because a score should not
            # reward guessing at an unanswerable label.
            # Keep the BEST model, not the last one.
            #
            # Three stopping criteria were tuned before noticing that the
            # tuning was mostly beside the point: the run returns whichever
            # weights the final epoch happened to leave behind, so every
            # criterion was really choosing a model by choosing when to
            # stop. Snapshotting the best-scoring epoch decouples the two --
            # stopping late now costs time and nothing else, and the
            # criterion only has to decide when to give up.
            #
            # Selection is on the DEFINED-ONLY mean over scorable heads,
            # which is the honest overall number; a model that guesses well
            # at unanswerable labels should not win on that basis.
            selection = float(
                np.mean([result.accuracies[n] for n in scorable_names])
                if scorable_names
                else result.mean_accuracy
            )
            if selection > best_selection + 1e-5:
                best_selection = selection
                best_weights = tree_map(
                    lambda a: mx.array(a) if isinstance(a, mx.array) else a,
                    model.parameters(),
                )

            watch = evaluate(model, eval_set, only_defined=False)
            watched_heads = [
                watch[name]
                for i, (name, _) in enumerate(HEADS)
                if len(np.unique(eval_set.labels[:, i])) > 1
            ]
            watched = min(watched_heads) if watched_heads else result.mean_accuracy
            if watched > best_watched + 1e-4:
                best_watched, best_epoch = watched, epoch

        history.append(result)
        if verbose:
            print(result.summary(), flush=True)

        if epoch - best_epoch >= cfg.patience:
            if verbose:
                print(f"early stop: no improvement in {cfg.patience} epochs")
            break

    if best_weights is not None:
        model.update(best_weights)
        mx.eval(model.parameters())
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


def save_core(model: TinyRecursiveCore, path: str | Path) -> Path:
    """Persist a trained core.

    Training costs ~20 minutes on this machine, and every measurement that
    silently retrains is 20 minutes not spent measuring something else.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(p))
    return p


def load_core(path: str | Path, cfg: CoreConfig | None = None) -> TinyRecursiveCore:
    """Rebuild a core from saved weights.

    `cfg` must match the configuration it was trained with; the weights
    carry shapes but not hyperparameters, so a mismatch fails loudly at
    load rather than quietly at inference.
    """
    model = TinyRecursiveCore(cfg or CoreConfig())
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model
