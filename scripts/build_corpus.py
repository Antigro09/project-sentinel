"""Build the bootstrap corpus.

Generates solvable worlds, has the local teacher induce a world model for
each, scores every model automatically with the verifier, and writes one
JSONL record per world.

Resumable by design: a build is a long unattended job, and re-running the
same command skips worlds already in the output file. Kill it, restart it,
lose nothing.

    # Validate the pipeline cheaply before committing a night to it
    uv run scripts/build_corpus.py --smoke

    # A real overnight run
    uv run scripts/build_corpus.py --n-train 5000 --n-holdout-seed 300 \\
        --n-holdout-mechanics 300

Start with --smoke. It is the Phase 2 kill criterion: if the teacher cannot
produce verified models for a handful of generated worlds, the corpus
cannot exist and the core cannot be trained, and that is worth knowing in
five minutes rather than at hour six.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sentinel.bootstrap import (
    TEACHER_MODEL,
    CorpusRecord,
    CorpusWriter,
    OllamaClient,
    Sandbox,
    Teacher,
    completed_ids,
    corpus_stats,
    read_corpus,
)
from sentinel.gen import make_split
from sentinel.gen.spec import WorldSpec


def _split_signature(args: argparse.Namespace) -> list[int]:
    """Identity of a generated split.

    A cache built for a different shape must be rejected rather than reused,
    or a resume would silently work on worlds the corpus was not built from.
    """
    return [
        args.n_train,
        args.n_holdout_seed,
        args.n_holdout_mechanics,
        args.withhold,
        args.seed,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-holdout-seed", type=int, default=25)
    parser.add_argument("--n-holdout-mechanics", type=int, default=25)
    parser.add_argument("--withhold", type=int, default=4)
    parser.add_argument("--model", default=TEACHER_MODEL)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", default="corpus/bootstrap.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=2.0,
        help="wall-clock budget per generated function call",
    )
    parser.add_argument(
        "--sandbox",
        choices=("auto", "docker", "inprocess"),
        default="auto",
        help=(
            "how to execute generated code. 'auto' uses a container runtime "
            "when one is present and runs in-process otherwise; 'docker' "
            "refuses to run without isolation"
        ),
    )
    parser.add_argument(
        "--split-cache",
        default="corpus/split.json",
        help="cache generated world specs here so resumes skip regeneration",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help=(
            "stop cleanly after this many hours. The corpus is designed to "
            "accumulate across sessions, so this bounds one night's work "
            "rather than the whole target -- re-run to continue"
        ),
    )
    parser.add_argument(
        "--abort-after",
        type=int,
        default=8,
        help=(
            "stop after this many consecutive worlds produce nothing usable "
            "(guards against a dead model silently burning the whole run)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run to validate the pipeline (8 train, 2+2 holdout)",
    )
    args = parser.parse_args()
    session_started = time.perf_counter()

    if args.smoke:
        args.n_train, args.n_holdout_seed, args.n_holdout_mechanics = 8, 2, 2
        args.out = "corpus/smoke.jsonl"
        args.split_cache = "corpus/smoke_split.json"

    client = OllamaClient(model=args.model)
    if not client.available():
        print(
            f"Ollama is not reachable at {client.host}.\n"
            "Start it with:  ollama serve",
            file=sys.stderr,
        )
        return 1

    installed = client.installed_models()
    if args.model not in installed:
        print(f"Model {args.model!r} is not installed.", file=sys.stderr)
        print(f"Available: {', '.join(installed)}", file=sys.stderr)
        return 1

    print(f"Teacher: {args.model}")

    # Generation is deterministic, so a resume would rebuild exactly the same
    # worlds -- at ~0.4s each that is over half an hour of pure recomputation
    # before a 5000-world run resumes any actual work. Cache it.
    cache_path = Path(args.split_cache) if args.split_cache else None
    work: list[tuple[str, WorldSpec]] | None = None
    withheld_summaries: list[str] = []

    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == _split_signature(args):
                work = [
                    (entry["split"], WorldSpec.from_json(entry["spec"]))
                    for entry in cached["worlds"]
                ]
                withheld_summaries = cached.get("withheld", [])
                print(f"Loaded {len(work)} worlds from {cache_path}")
            else:
                print(f"{cache_path} was built with different settings; regenerating")
        except Exception as exc:  # noqa: BLE001 - a bad cache must never be fatal
            print(f"Could not read {cache_path} ({exc}); regenerating")

    if work is None:
        print("Generating worlds (each is BFS-verified solvable before use)...")
        started = time.perf_counter()
        split = make_split(
            n_train=args.n_train,
            n_holdout_seed=args.n_holdout_seed,
            n_holdout_mechanics=args.n_holdout_mechanics,
            withhold=args.withhold,
            seed=args.seed,
        )
        print(f"  {split.summary()}  ({time.perf_counter() - started:.1f}s)")
        work = (
            [("train", s) for s in split.train]
            + [("holdout_seed", s) for s in split.holdout_seed]
            + [("holdout_mechanics", s) for s in split.holdout_mechanics]
        )
        withheld_summaries = [m.summary() for m in split.withheld]

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "signature": _split_signature(args),
                        "withheld": withheld_summaries,
                        "worlds": [
                            {"split": name, "spec": spec.to_json()} for name, spec in work
                        ],
                    }
                ),
                encoding="utf-8",
            )
            print(f"  cached to {cache_path}")

    print("  withheld mechanic combinations (never seen in training):")
    for summary in withheld_summaries:
        print(f"    - {summary}")
    print()

    already = completed_ids(args.out)
    if already:
        print(f"Resuming: {len(already)} worlds already recorded in {args.out}")
    todo = [(sp, sc) for sp, sc in work if sc.world_id not in already]
    print(f"{len(todo)} worlds to process\n")

    try:
        sandbox = Sandbox(mode=args.sandbox, model_timeout=args.model_timeout)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Execution: {sandbox.describe()}")
    if sandbox.isolated and not sandbox.image_present():
        print(f"  pulling {sandbox.image} (one time; runs are offline after this)...")
        ok, detail = sandbox.pull_image()
        if not ok:
            print(f"  image pull failed: {detail}", file=sys.stderr)
            return 1
        print("  image ready")
    elif not sandbox.isolated:
        print("  generated code will run with full privileges on this machine.")
        print("  install a container runtime and re-run to isolate it.")
    print()

    teacher = Teacher(
        client=client,
        max_rounds=args.rounds,
        model_timeout=args.model_timeout,
        sandbox=sandbox,
    )

    solved = 0
    usable = 0
    consecutive_failures = 0
    aborted = False
    stopped_on_time = False
    run_started = time.perf_counter()

    # Based on session start, not loop start. World generation runs first and
    # can take tens of minutes on a large split; anchoring the deadline after
    # it would quietly overrun the budget the caller actually asked for.
    deadline = (
        session_started + args.max_hours * 3600 if args.max_hours else None
    )
    if deadline:
        print(
            f"Will stop cleanly after {args.max_hours:g}h; "
            f"re-run the same command to continue.\n",
            flush=True,
        )

    with CorpusWriter(args.out) as writer:
        for i, (split_name, spec) in enumerate(todo, start=1):
            # Checked before starting a world, not during. A world takes ~3
            # minutes, so stopping between them costs almost nothing and
            # avoids discarding partial work or writing a torn record.
            if deadline and time.perf_counter() >= deadline:
                print(
                    f"\nTime limit reached after {i - 1} worlds this session.\n"
                    f"Progress saved to {args.out}; re-run to continue where "
                    "this left off.",
                    flush=True,
                )
                stopped_on_time = True
                break

            result = teacher.induce_world(spec, seed=args.seed)
            record = CorpusRecord.from_result(spec, split_name, result)
            writer.append(record)

            solved += int(record.solved)
            usable += int(record.usable)

            # An unreachable model fails instantly, so without this a dead
            # Ollama would blow through every remaining world in minutes and
            # report a "completed" run full of empty records. Failing loudly
            # and early is the whole point of an unattended job.
            if record.usable:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= args.abort_after:
                    print(
                        f"\nABORTING: {consecutive_failures} worlds in a row produced "
                        f"nothing usable.\nLast error: {result.error or 'none reported'}\n"
                        f"Progress is saved in {args.out}; fix the cause and re-run the "
                        "same command to resume.",
                        file=sys.stderr,
                        flush=True,
                    )
                    aborted = True
                    break

            elapsed = time.perf_counter() - run_started
            rate = elapsed / i
            remaining = rate * (len(todo) - i)

            # flush explicitly: stdout is block-buffered when redirected to a
            # log file, so an unattended run would show no progress at all
            # until it finished — indistinguishable from being hung.
            print(
                f"[{i}/{len(todo)}] {split_name:18} {spec.summary()}\n"
                f"    {result.summary()}  "
                f"| solved {solved}/{i} usable {usable}/{i} "
                f"| {rate:.1f}s/world, ~{remaining / 60:.0f} min left",
                flush=True,
            )

    print()
    print("=" * 70)
    all_records = read_corpus(args.out)
    print(corpus_stats(all_records))
    print(client.usage.summary())
    print(f"written to {args.out}")

    remaining = len(work) - len(all_records)
    if remaining > 0:
        elapsed_hours = (time.perf_counter() - run_started) / 3600
        done_this_session = len(all_records) - len(already)
        if done_this_session > 0 and elapsed_hours > 0:
            rate = done_this_session / elapsed_hours
            print(
                f"\n{remaining} worlds remain "
                f"(~{remaining / rate:.1f}h at {rate:.0f} worlds/hour). "
                "Re-run the same command to continue."
            )
        else:
            print(f"\n{remaining} worlds remain. Re-run to continue.")

    if aborted:
        return 3
    if stopped_on_time:
        return 0
    if usable == 0 and todo:
        print(
            "\nKILL CRITERION HIT: the teacher produced no usable model for any "
            "world.\nThe corpus cannot be built and the core cannot be trained "
            "from it.\nInvestigate before scaling up.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
