"""Re-score a corpus against the current verifier.

Scores are only comparable if they were produced by the same metric. When
the verifier changes, stored fitness values become a mix of old and new
semantics — and a training set whose labels mean different things in
different rows is worse than one that is uniformly wrong, because nothing
downstream can tell which is which.

Re-scoring is cheap: it replays stored programs against regenerated
histories with no LLM involved. Worlds are reproducible from their spec,
so nothing needs to have been kept.

    uv run scripts/rescore_corpus.py corpus/bootstrap.jsonl
    uv run scripts/rescore_corpus.py corpus/bootstrap.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from sentinel.bootstrap import CorpusRecord, CorpusWriter, make_training_history, read_corpus
from sentinel.bootstrap.loader import load_model
from sentinel.verify import Verifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    path = Path(args.path)
    records = read_corpus(path)
    if not records:
        print(f"{path} is empty or unreadable", file=sys.stderr)
        return 1

    print(f"Re-scoring {len(records)} records from {path}")
    verifier = Verifier()
    updated: list[CorpusRecord] = []
    old_scores: list[float] = []
    new_scores: list[float] = []
    reload_failures = 0
    started = time.perf_counter()

    for i, record in enumerate(records, start=1):
        if not record.source:
            updated.append(record)
            continue
        try:
            history = make_training_history(record.spec)
            model = load_model(
                record.source, context={"INITIAL_GRID": history.initial.grid}
            )
            report = verifier.verify(model, history)
        except Exception as exc:  # noqa: BLE001 - a program that no longer loads is data too
            reload_failures += 1
            updated.append(
                CorpusRecord(
                    **{
                        **record.__dict__,
                        "fitness": 0.0,
                        "error": f"rescore failed: {type(exc).__name__}: {exc}",
                    }
                )
            )
            continue

        old_scores.append(record.fitness)
        new_scores.append(report.fitness)
        updated.append(
            CorpusRecord(
                world_id=record.world_id,
                split=record.split,
                spec=record.spec,
                source=record.source,
                fitness=report.fitness,
                solved=report.is_perfect,
                transition_match=report.transition_match,
                coverage=report.coverage,
                outcome_accuracy=report.outcome_accuracy,
                rounds=record.rounds,
                prompt_tokens=record.prompt_tokens,
                output_tokens=record.output_tokens,
                seconds=record.seconds,
                error=record.error,
            )
        )
        if i % 25 == 0:
            print(f"  {i}/{len(records)}")

    elapsed = time.perf_counter() - started
    print(f"\nre-scored {len(new_scores)} in {elapsed:.0f}s ({reload_failures} failed to reload)")
    if new_scores:
        print(f"  old mean {st.mean(old_scores):.3f}  ->  new mean {st.mean(new_scores):.3f}")
        print(f"  solved exactly: {sum(r.solved for r in updated)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    backup = path.with_suffix(path.suffix + ".pre-rescore")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  original preserved at {backup}")

    path.unlink()
    with CorpusWriter(path) as writer:
        for record in updated:
            writer.append(record)
    print(f"  rewrote {len(updated)} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
