"""Pre-run checks for an unattended corpus build.

Run this before committing a night to `build_corpus.py`. Every check here
exists because its failure mode is *silent* — the run appears healthy and
produces nothing, or dies at 1am with no indication why.

    uv run scripts/preflight.py
    uv run scripts/preflight.py --full     # also does one real LLM induction

Exit code is 0 only if every required check passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    symbol = {PASS: "  ok ", WARN: " warn", FAIL: "FAIL "}[status]
    print(f"{symbol} {name}" + (f"\n       {detail}" if detail else ""), flush=True)


def check_disk() -> None:
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / 1e9
    if free_gb < 5:
        record(FAIL, "disk space", f"only {free_gb:.1f} GB free")
    elif free_gb < 20:
        record(WARN, "disk space", f"{free_gb:.1f} GB free")
    else:
        record(PASS, "disk space", f"{free_gb:.0f} GB free")


def check_ollama(model: str) -> bool:
    from sentinel.bootstrap import OllamaClient

    client = OllamaClient(model=model)
    if not client.available():
        record(FAIL, "ollama reachable", "start it with: ollama serve")
        return False
    try:
        installed = client.installed_models()
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "ollama model list", str(exc))
        return False
    if model not in installed:
        record(FAIL, f"model {model} installed", f"available: {', '.join(installed)}")
        return False
    record(PASS, f"model {model} installed")
    return True


def check_generation() -> bool:
    from sentinel.bootstrap import make_training_history
    from sentinel.gen import generate
    from sentinel.verify import evidence_coverage

    started = time.perf_counter()
    weak = 0
    made = 0
    for seed in range(8):
        spec = generate(seed)
        if spec is None:
            continue
        history = make_training_history(spec)
        if history is None:
            continue
        made += 1
        if evidence_coverage(history).unexercised():
            weak += 1
    elapsed = time.perf_counter() - started

    if made == 0:
        record(FAIL, "world generation", "no solvable worlds produced")
        return False
    if weak:
        record(WARN, "evidence quality", f"{weak}/{made} histories cannot falsify")
    else:
        record(PASS, "evidence quality", f"{made}/{made} histories exercise all channels")
    record(PASS, "generation speed", f"{elapsed / made:.2f}s per world")
    return True


def check_sandbox(mode: str) -> bool:
    from sentinel.bootstrap import Sandbox, detect_runtime, make_training_history
    from sentinel.gen import generate

    runtime = detect_runtime()
    if runtime is None:
        if mode == "docker":
            record(FAIL, "container runtime", "none found, but --sandbox docker was asked for")
            return False
        record(
            WARN,
            "container runtime",
            "none found; generated code will run with full privileges",
        )
    else:
        record(PASS, "container runtime", runtime)

    try:
        box = Sandbox(mode=mode)
    except RuntimeError as exc:
        record(FAIL, "sandbox construction", str(exc))
        return False

    if box.isolated and not box.image_present():
        print(f"       pulling {box.image} (one time)...", flush=True)
        ok, detail = box.pull_image()
        if not ok:
            record(FAIL, "image pull", detail)
            return False
    if box.isolated:
        record(PASS, "sandbox image", box.image)

    # A real verification through the real path — the check that matters.
    spec = generate(0)
    history = make_training_history(spec)
    source = (
        "def init_state():\n    return 0\n"
        "def transition(state, action):\n    return state + 1\n"
        "def render(state):\n    return [list(r) for r in INITIAL_GRID]\n"
        "def outcome(state):\n    return 'ongoing'\n"
    )
    started = time.perf_counter()
    result = box.verify(source, history, name="preflight")
    elapsed = time.perf_counter() - started

    if not result.ok or result.report is None:
        record(FAIL, "sandbox verify", f"{result.kind}: {result.error}")
        return False
    if result.report.coverage <= 0.0:
        record(
            FAIL,
            "sandbox metrics",
            "report came back with zero coverage — metrics were lost in transport",
        )
        return False
    record(
        PASS,
        "sandbox verify",
        f"{elapsed:.1f}s per call, coverage={result.report.coverage:.0%}, "
        f"isolated={result.isolated}",
    )

    # A model that never returns must be killed, not allowed to hang the run.
    hostile = (
        "def init_state():\n    return 0\n"
        "def transition(state, action):\n    while True:\n        pass\n"
        "def render(state):\n    return [list(r) for r in INITIAL_GRID]\n"
        "def outcome(state):\n    return 'ongoing'\n"
    )
    started = time.perf_counter()
    hostile_result = box.verify(hostile, history, name="preflight-hostile")
    hostile_elapsed = time.perf_counter() - started
    if hostile_elapsed > box.timeout_for(history) + 30:
        record(FAIL, "infinite loop contained", f"took {hostile_elapsed:.0f}s")
        return False
    record(
        PASS,
        "infinite loop contained",
        f"stopped in {hostile_elapsed:.1f}s "
        f"({'crash recorded' if hostile_result.report and hostile_result.report.crashed else hostile_result.kind or 'handled'})",
    )
    return True


def check_resume(out: str) -> None:
    from sentinel.bootstrap import completed_ids, read_corpus

    path = Path(out)
    if not path.exists():
        record(PASS, "resume state", f"{out} does not exist yet; will start fresh")
        return
    records = read_corpus(path)
    done = completed_ids(path)
    record(
        PASS,
        "resume state",
        f"{len(records)} records readable, {len(done)} worlds will be skipped",
    )


def check_sleep() -> None:
    """macOS idle sleep pauses an unattended run without any error."""
    if sys.platform != "darwin":
        return
    try:
        out = subprocess.run(
            ["pmset", "-g"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except Exception:  # noqa: BLE001
        return
    sleep_line = next(
        (ln for ln in out.splitlines() if ln.strip().startswith("sleep")), ""
    )
    value = sleep_line.split()[1] if len(sleep_line.split()) > 1 else "?"
    if value == "0":
        record(PASS, "system sleep", "disabled")
    else:
        record(
            WARN,
            "system sleep",
            f"set to {value} min — wrap the run in `caffeinate -is` or it will pause",
        )


def check_full_induction(model: str, mode: str) -> bool:
    from sentinel.bootstrap import OllamaClient, Sandbox, Teacher
    from sentinel.gen import generate

    print("       running one real induction (this takes a few minutes)...", flush=True)
    client = OllamaClient(model=model)
    teacher = Teacher(client=client, max_rounds=2, sandbox=Sandbox(mode=mode))
    started = time.perf_counter()
    result = teacher.induce_world(generate(7))
    elapsed = time.perf_counter() - started

    if not result.usable:
        record(FAIL, "end-to-end induction", result.error or "no usable model produced")
        return False
    record(
        PASS,
        "end-to-end induction",
        f"fitness={result.best_fitness:.3f} in {elapsed:.0f}s "
        f"({len(result.attempts)} rounds, {client.usage.output_tokens} output tokens)",
    )
    per_world = elapsed / 2 * 3  # 2 rounds measured, 3 configured
    record(
        PASS,
        "projected throughput",
        f"~{per_world:.0f}s/world at 3 rounds → "
        f"~{8 * 3600 / per_world:.0f} worlds per 8-hour night",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--sandbox", choices=("auto", "docker", "inprocess"), default="auto")
    parser.add_argument("--out", default="corpus/bootstrap.jsonl")
    parser.add_argument(
        "--full", action="store_true", help="also run one real LLM induction"
    )
    args = parser.parse_args()

    print("Preflight checks for an unattended corpus build\n")

    check_disk()
    check_sleep()
    ollama_ok = check_ollama(args.model)
    check_generation()
    sandbox_ok = check_sandbox(args.sandbox)
    check_resume(args.out)

    if args.full and ollama_ok and sandbox_ok:
        check_full_induction(args.model, args.sandbox)

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]

    print("\n" + "=" * 66)
    print(f"{len(results) - len(failures) - len(warnings)} passed, "
          f"{len(warnings)} warnings, {len(failures)} failures")

    if failures:
        print("\nDo NOT start an overnight run. Fix first:")
        for _, name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1

    if warnings:
        print("\nSafe to run, but note:")
        for _, name, detail in warnings:
            print(f"  - {name}: {detail}")

    print("\nReady.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
