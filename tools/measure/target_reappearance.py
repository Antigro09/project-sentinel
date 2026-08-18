"""Split target-mass rises into within-level and across-level.

`encode_history` pairs consecutive steps without consulting `level_index` or
`reset_points`, so a level change is presented to the network as the effect of
an action. If the rises in UNORDERED worlds are all level changes, then the
within-level rise is the clean `ordered_targets` signal and the encoder is
burying it under noise it should never have emitted.
"""
import numpy as np
from sentinel.core.data import load_split
from sentinel.core.encoding import MAX_TRANSITIONS
from sentinel.bootstrap.teacher import make_training_history
from sentinel.gen.grid import TARGET

split = load_split("corpus/split.json")
specs = split["holdout_mechanics"][:120]

agg = {0: {"within": [], "across": []}, 1: {"within": [], "across": []}}
for si, spec in enumerate(specs):
    lab = int(spec.mechanics.ordered_targets)
    for seed in (0, 1):
        try:
            hist = make_training_history(spec, seed=seed)
        except Exception:
            continue
        resets = set(hist.reset_points)
        prev = hist.initial
        prev_level = hist.steps[0].level_index if hist.steps else 0
        within = across = 0
        for i, step in enumerate(hist.steps[:MAX_TRANSITIONS]):
            cb = np.bincount(np.array(prev.grid).reshape(-1), minlength=16)
            ca = np.bincount(np.array(step.settled.grid).reshape(-1), minlength=16)
            rose = ca[TARGET] > cb[TARGET]
            boundary = (step.index in resets) or (step.level_index != prev_level)
            if rose:
                if boundary:
                    across += 1
                else:
                    within += 1
            prev, prev_level = step.settled, step.level_index
        agg[lab]["within"].append(within)
        agg[lab]["across"].append(across)

for lab in (0, 1):
    tag = "ordered  " if lab else "unordered"
    w = np.array(agg[lab]["within"]); a = np.array(agg[lab]["across"])
    print(f"{tag} n={len(w):4d}")
    print(f"    WITHIN-level target rises: {(w>0).mean():6.1%} of episodes, mean {w.mean():.2f}")
    print(f"    ACROSS-level/reset rises : {(a>0).mean():6.1%} of episodes, mean {a.mean():.2f}")
