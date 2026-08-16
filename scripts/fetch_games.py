"""Download the public ARC-AGI-3 environments for offline use.

This is the only script in the project that touches the network, and it
only needs to run once. Everything afterwards — training, verification,
search, evaluation — runs with no connection at all.

The engine downloads each game as executable Python into
`environment_files/`, so the directory is gitignored: it is third-party
code, reproducible from here on demand.

    uv run scripts/fetch_games.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environments-dir",
        default="environment_files",
        help="where to store downloaded games (default: environment_files)",
    )
    parser.add_argument(
        "--game",
        action="append",
        dest="games",
        help="download only this game id; repeatable. Default: all available.",
    )
    args = parser.parse_args()

    import arc_agi

    from sentinel.env.runner import silence_engine_logging

    logging.basicConfig(level=logging.WARNING)
    silence_engine_logging()

    # NORMAL mode: acquires an anonymous key and lists what is available.
    arcade = arc_agi.Arcade(environments_dir=args.environments_dir)
    listed = sorted({e.game_id.split("-")[0] for e in arcade.get_environments()})
    if not listed:
        print("No environments listed by the API.", file=sys.stderr)
        return 1

    wanted = args.games or listed
    unknown = sorted(set(wanted) - set(listed))
    if unknown:
        print(f"Unknown game ids: {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(listed)}", file=sys.stderr)
        return 1

    print(f"Fetching {len(wanted)} environment(s) into {args.environments_dir}/")
    started = time.perf_counter()
    failures: list[tuple[str, str]] = []

    for game_id in wanted:
        try:
            if arcade.make(game_id) is None:
                failures.append((game_id, "make() returned None"))
            else:
                print(f"  ok  {game_id}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((game_id, f"{type(exc).__name__}: {exc}"))

    elapsed = time.perf_counter() - started
    ok = len(wanted) - len(failures)
    print(f"\n{ok}/{len(wanted)} downloaded in {elapsed:.1f}s")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for game_id, reason in failures:
            print(f"  {game_id}: {reason}", file=sys.stderr)
        return 1

    print("\nAll games are now playable offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
