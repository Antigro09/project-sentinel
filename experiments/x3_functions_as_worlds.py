"""X3: a hidden Python function is a world.

Actions are calls, observations are outputs, and the rule to infer is the
function itself. Nothing about the architecture changes -- the same
verifier replays the same history through the same contract -- but the
domain stops being a grid and starts being a TOOL.

That is the point. "Figure out an unfamiliar tool by using it" is the
capability an agent needs to do digital work, and it is the one frontier
models most conspicuously lack: they guess what an API does and have no
mechanism to check. Here the check is the whole system.

Staged, not run: needs a small function DSL (X4) to be worth much, since
with a fixed menu of candidate functions this is just dials again with
different labels.
"""

from __future__ import annotations

# --- sketch of the domain -------------------------------------------------
#
# class FunctionWorld:
#     hidden: Callable[[int], int]
#     def step(self, probe: int) -> Observation:   # observation encodes the output
#
# class FunctionModel(WorldModel):
#     candidate: Callable[[int], int]
#     def transition(self, state, action): ...     # state = probes seen so far
#     def render(self, state): ...                 # outputs, drawn into the grid
#
# The interesting measurement is not accuracy but PROBES-TO-CERTAINTY: how
# many calls the system needs before one candidate survives. Humans probing
# an unfamiliar API do roughly this, and it is directly comparable to the
# `replays` numbers we already report for grid worlds.

raise SystemExit("staged: implement after X4 (programs as hypotheses)")
