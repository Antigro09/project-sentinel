"""X2: can ONE core rank well in structurally unrelated domains?

Phase 6 showed the verifier and hypothesis search transfer unchanged to a
domain with no space in it (24/24 on dials). The core does not: its
features are per-value spatial moments -- centroids and displacements --
which describe a board with things on it, and a dial has neither.

That is the gap between "our machinery is domain agnostic" and "our learned
component is". The second is what "pick up a new skill without a custom
model" actually requires.

Design sketch:
  - a domain-agnostic feature set: per-value mass before/after, the change
    histogram, the action taken, and per-lag repetition -- statistics of
    WHAT CHANGED rather than WHERE
  - train one core on grids + dials + functions together
  - measure rank-of-truth in each domain, against a core trained on that
    domain alone

Note the risk, already measured once: the lag features that seemed obviously
useful destroyed step_distance (0.952 -> 0.467) because per-episode
constants broadcast across timesteps swamped the per-transition signal. A
domain-agnostic feature set could fail the same way, so the single-domain
cores are the control that has to be beaten.

Staged: needs X3's function domain to have three domains to train across.
"""

raise SystemExit("staged: needs a third domain (X3)")
