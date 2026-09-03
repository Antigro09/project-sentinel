"""E / O5. The exact palette posterior p(phi | public history).

Because identical pixels means the semantic trajectories differ by a role permutation,
the likelihood is an indicator: a mapping is either consistent with the public history or
it is not. The posterior is therefore uniform over the surviving permutations, and its
entropy is exactly log2 of the equivalence-class size -- computed, not estimated.

The reference may consult the authored semantic simulator to test a candidate mapping.
That is what makes it a reference; semantic state stays evaluator-only and never reaches
a learned arm.

    .venv-shwm/bin/python experiments/shwm/o_posterior.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

import o_core as O
import o_identifiability as ident
from m2d_core import ARTIFACTS, write
from o_core import N_ROLES, ROLES, ROLE_INDEX


def survivors_after(episode: O.Episode, steps: int) -> list[tuple[int, ...]]:
    """Permutations consistent with the first `steps` transitions of public history."""
    roles, actions = episode.roles, episode.actions
    out = []
    for pi in itertools.permutations(range(N_ROLES)):
        grids = ident.permuted(roles, pi)
        if not ident.cardinality_legal(grids[0]):
            continue
        ok = True
        for t in range(1, min(steps, len(grids) - 1) + 1):
            if not (ident.cardinality_legal(grids[t])
                    and ident.static_scene_legal(grids[t - 1], grids[t])
                    and ident.motion_legal(grids[t - 1], grids[t], int(actions[t - 1]))):
                ok = False
                break
        if ok and steps >= 1:
            ok = ident.switch_consistency_legal(grids[:steps + 1], actions[:steps + 1])
        if ok:
            out.append(pi)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o-posterior.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    episodes = O.collect_appearance(
        list(range(110_000, 110_000 + arguments.episodes)),
        "HIDDEN_PALETTE_CONVENTION", [7_101], 1, 9, seed=11, policy="goal_directed")
    family = 720
    prior_entropy = float(np.log2(family))
    identity = tuple(range(N_ROLES))
    print(f"{len(episodes)} episodes; family {family} permutations; "
          f"prior entropy {prior_entropy:.4f} bits\n", flush=True)

    max_steps = max(e.length for e in episodes) - 1
    curve: dict[int, list[int]] = {k: [] for k in range(max_steps + 1)}
    event_pinned: dict[int, list[bool]] = {k: [] for k in range(max_steps + 1)}
    for episode in episodes:
        for k in range(max_steps + 1):
            keep = survivors_after(episode, min(k, episode.length - 1))
            assert identity in keep
            curve[k].append(len(keep))
            # The event only needs SWITCH and AGENT pinned; the goal markers may stay
            # ambiguous forever and the event is unaffected.
            event_pinned[k].append(all(
                pi[ROLE_INDEX["SWITCH"]] == ROLE_INDEX["SWITCH"]
                and pi[ROLE_INDEX["AGENT"]] == ROLE_INDEX["AGENT"] for pi in keep))

    report: dict[str, Any] = {
        "family_size": family, "prior_entropy_bits": prior_entropy,
        "episodes": len(episodes), "roles": list(ROLES),
        "likelihood": ("an indicator: identical pixels means a role permutation, so a "
                       "mapping is consistent or it is not, and the posterior is uniform "
                       "over survivors"),
        "curve": {}}
    print(f"{'calibration steps':>18s} {'class size':>11s} {'entropy bits':>13s} "
          f"{'true mass':>10s} {'event pinned':>13s}")
    print("-" * 72)
    for k in range(max_steps + 1):
        sizes = np.array(curve[k], dtype=float)
        block = {"mean_class_size": float(sizes.mean()),
                 "median_class_size": float(np.median(sizes)),
                 "posterior_entropy_bits": float(np.log2(sizes).mean()),
                 "true_class_mass": float((1.0 / sizes).mean()),
                 "fully_identified_fraction": float((sizes == 1).mean()),
                 "event_identified_fraction": float(np.mean(event_pinned[k]))}
        report["curve"][str(k)] = block
        print(f"{k:18d} {block['mean_class_size']:11.2f} "
              f"{block['posterior_entropy_bits']:13.4f} {block['true_class_mass']:10.4f} "
              f"{block['event_identified_fraction']:13.3f}", flush=True)

    pinned = [k for k in range(max_steps + 1)
              if report["curve"][str(k)]["event_identified_fraction"] >= 0.99]
    resolved = [k for k in range(max_steps + 1)
                if report["curve"][str(k)]["fully_identified_fraction"] >= 0.99]
    report["minimum_calibration_for_event"] = pinned[0] if pinned else None
    report["minimum_calibration_for_full_mapping"] = resolved[0] if resolved else None
    report["residual_unresolved"] = ("GOAL_ALPHA <-> GOAL_BETA: no visual or behavioural "
                                     "evidence in the permitted set separates the two "
                                     "markers; only the language channel names them")
    zero, last = report["curve"]["0"], report["curve"][str(max_steps)]
    report["concentrates"] = bool(
        last["posterior_entropy_bits"] < zero["posterior_entropy_bits"] - 2.0
        and last["true_class_mass"] > 3.0 * zero["true_class_mass"])
    # Two criteria, both reported. The strict one -- the event identified in >=99% of
    # episodes -- is the one that was coded before the curve was seen, and it FAILS at
    # 0.938. Loosening it after the fact would be the move this project keeps correcting,
    # so O5 is recorded PARTIAL with both numbers rather than passed on the softer test.
    report["o5_strict_event_identified_in_99pc"] = bool(pinned)
    report["o5_status"] = ("PASS" if pinned else
                           "PARTIAL" if report["concentrates"] else "FAIL")
    report["o5_basis"] = (
        f"posterior entropy {zero['posterior_entropy_bits']:.4f} -> "
        f"{last['posterior_entropy_bits']:.4f} bits, true-class mass "
        f"{zero['true_class_mass']:.4f} -> {last['true_class_mass']:.4f}, event "
        f"identified in {last['event_identified_fraction']:.3f} of episodes against a "
        f"pre-stated 0.99")
    report["single_frame_note"] = (
        "SWITCH is pinned from ONE frame by cardinality alone -- seven switch cells is a "
        "generator constant -- but AGENT is not, because agent and both goal markers are "
        "all singletons. The event needs both, so it is NOT single-frame identifiable; "
        "an earlier statement that it was is withdrawn here.")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nminimum calibration steps to pin the EVENT: "
          f"{report['minimum_calibration_for_event']}")
    print(f"minimum to resolve the FULL mapping: "
          f"{report['minimum_calibration_for_full_mapping'] or 'never, within an episode'}")
    print(f"O5: {report['o5_status']} -- {report['o5_basis']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
