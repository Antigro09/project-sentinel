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
import sys
import time

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
        "--smoke",
        action="store_true",
        help="tiny run to validate the pipeline (8 train, 2+2 holdout)",
    )
    args = parser.parse_args()

    if args.smoke:
        args.n_train, args.n_holdout_seed, args.n_holdout_mechanics = 8, 2, 2
        args.out = "corpus/smoke.jsonl"

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
    print("  withheld mechanic combinations (never seen in training):")
    for mech in split.withheld:
        print(f"    - {mech.summary()}")
    print()

    work: list[tuple[str, object]] = (
        [("train", s) for s in split.train]
        + [("holdout_seed", s) for s in split.holdout_seed]
        + [("holdout_mechanics", s) for s in split.holdout_mechanics]
    )

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
    run_started = time.perf_counter()

    with CorpusWriter(args.out) as writer:
        for i, (split_name, spec) in enumerate(todo, start=1):
            result = teacher.induce_world(spec, seed=args.seed)
            record = CorpusRecord.from_result(spec, split_name, result)
            writer.append(record)

            solved += int(record.solved)
            usable += int(record.usable)
            elapsed = time.perf_counter() - run_started
            rate = elapsed / i
            remaining = rate * (len(todo) - i)

            print(
                f"[{i}/{len(todo)}] {split_name:18} {spec.summary()}\n"
                f"    {result.summary()}  "
                f"| solved {solved}/{i} usable {usable}/{i} "
                f"| {rate:.1f}s/world, ~{remaining / 60:.0f} min left"
            )

    print()
    print("=" * 70)
    print(corpus_stats(read_corpus(args.out)))
    print(client.usage.summary())
    print(f"written to {args.out}")

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
