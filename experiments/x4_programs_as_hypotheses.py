"""X4: replace the 8-label hypothesis space with a small DSL.

The scaling plan needs this. Labels cannot express "reverse the string" or
"this endpoint paginates", so a label-based system can never leave the
domain its labels were written for. A grammar can.

The measurement that matters is whether the core still RANKS usefully when
the space stops being enumerable. Today it ranks the truth at median 173 of
5,760; over a program space the equivalent question is whether the true
program sits in the first few hundred derivations by prior probability.

Design sketch:
  - a grammar over transition rules: move(dir, dist), on_hit(effect),
    every(n, effect), when(cell_type, effect), ...
  - the current 5,760 rule sets must be EXPRESSIBLE in it, so the existing
    benchmark stays a valid regression test
  - the core predicts a distribution over productions rather than classes
  - search enumerates derivations in prior order; `first_divergence`
    localises which production to repair

Staged. Prerequisite: X1 must show the core's advantage does not vanish as
the space grows, or a bigger space is the wrong direction entirely.
"""

raise SystemExit("staged: gated on X1")
