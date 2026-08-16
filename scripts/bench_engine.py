"""Measure real engine throughput.

Simulation speed is the resource this architecture spends most: every
world-model verification and every planning rollout is paid for in engine
steps. So it is worth knowing the true number rather than trusting the
documented one.

Two figures are reported per game. Raw steps/sec is the headline. The
"effective" column counts only steps that actually changed the settled
grid — a game that rejects most actions cheaply can post a huge raw rate
while doing almost no work, and that distinction matters when budgeting
search.

    uv run scripts/bench_engine.py
    uv run scripts/bench_engine.py --steps 5000 --games ls20 ft09
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from sentinel.env import Action, Runner, available_games
from sentinel.env.types import GRID_SIZE


def bench_game(game_id: str, steps: int, seed: int) -> dict[str, float] | None:
    runner = Runner(game_id, seed=0)
    obs = runner.reset()

    rng = random.Random(seed)
    previous = obs.grid
    changed = 0
    executed = 0
    resets = 0

    started = time.perf_counter()
    for _ in range(steps):
        legal = list(runner.last.available_actions)
        if not legal:
            break
        choice = rng.choice(legal)
        action = (
            Action(6, rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
            if choice == 6
            else Action(choice)
        )
        settled = runner.step(action)
        executed += 1
        if settled.grid != previous:
            changed += 1
        previous = settled.grid
        if runner.done:
            runner.reset()
            previous = runner.last.grid
            resets += 1
    elapsed = time.perf_counter() - started

    if not executed or elapsed <= 0:
        return None

    return {
        "steps": executed,
        "seconds": elapsed,
        "fps": executed / elapsed,
        "changed_frac": changed / executed,
        "effective_fps": (changed / executed) * (executed / elapsed),
        "resets": resets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--games", nargs="*", default=None)
    args = parser.parse_args()

    games = args.games or available_games()
    if not games:
        print("No games on disk. Run: uv run scripts/fetch_games.py")
        return 1

    print(f"{'game':8} {'steps':>7} {'raw FPS':>12} {'changed':>9} {'eff FPS':>12}")
    print("-" * 52)

    rates: list[float] = []
    for game_id in games:
        try:
            result = bench_game(game_id, args.steps, args.seed)
        except Exception as exc:  # noqa: BLE001 - one bad game shouldn't stop the run
            print(f"{game_id:8} {'ERROR':>7}  {type(exc).__name__}: {exc}")
            continue
        if result is None:
            print(f"{game_id:8} {'skipped':>7}")
            continue
        rates.append(result["fps"])
        print(
            f"{game_id:8} {result['steps']:>7,} {result['fps']:>12,.0f} "
            f"{result['changed_frac']:>8.0%} {result['effective_fps']:>12,.0f}"
        )

    if rates:
        print("-" * 52)
        print(
            f"median {statistics.median(rates):,.0f} FPS   "
            f"min {min(rates):,.0f}   max {max(rates):,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
