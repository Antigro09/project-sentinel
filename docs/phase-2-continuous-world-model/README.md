# Phase 2 Continuous World Model

This directory is the implementation-facing plan for the
`phase-2-continuous-world-model` branch. It does not claim that a learned world
model has been implemented or has passed a capability gate.

## Current state

- `phase-1-exact-reference` is frozen at
  `5205543b110ba6da2e3f6da30630809941f821c4`.
- Phase 2 was forked from the same commit in an isolated worktree.
- The formal and numerical setup audit is complete for finite helper claims.
- The Scale-0 implementation is built and tested; the Scale-0 **gate is stopped**
  at the S0.2 encoder preflight because `google/gemma-3-4b-it` is licence-gated
  and this machine holds no access token. No matrix cell has run, so nothing is
  invalidated and no restart is owed. See the
  [Scale-0 handoff](SCALE-0-HANDOFF.md).
- One backbone has been fetched for a runtime feasibility probe only:
  `Qwen/Qwen3-VL-4B-Instruct` at revision `ebb281ec`, Apache-2.0. It is not
  training anything and it does not constitute a matrix run.
- No neural world model has been trained.
- No Phase-2/SHWM final evaluation seeds have been sampled. Historical X64H
  seed artifacts remain part of the inherited exact line and are not Phase-2
  seeds.
- X65A/X65B remain prerequisites for continual-memory claims at Scale 6.
- X66 repository work enters only at Scale 7 after its dependencies pass.

## Document map

| File | Purpose |
|---|---|
| [Architecture](ARCHITECTURE.md) | Component boundaries, interfaces, data contracts, uncertainty, verification, and memory bridge |
| [Scale-0 implementation plan](SCALE-0-IMPLEMENTATION-PLAN.md) | Narrow build order, files, tests, resource preflight, and stop conditions |
| [Scale-0 frozen run matrix](SCALE-0-RUN-MATRIX.md) | Singular cells, seeds, dimensions, data, optimizer, planner, matching tolerances, and hard resource ceilings |
| [Experiment gates](EXPERIMENT-GATES.md) | Scale 0–8 arms, metrics, controls, statistics, and falsifiers |
| [Branch and freeze protocol](BRANCH-AND-FREEZE-PROTOCOL.md) | Branch roles, immutable reference, seed freeze, merge rules, and evidence flow |
| [AGI dependency roadmap](AGI-DEPENDENCY-ROADMAP.md) | X65B/X66 dependency, Phase-2 joins, and later integrated acceptance path |
| [Claude Code Scale-0 handoff](CLAUDE-CODE-PROMPT-SCALE-0.md) | Executable implementation prompt with strict non-goals |
| [Scale-0 handoff](SCALE-0-HANDOFF.md) | What was built, the stop condition, negative findings, evidence labels, and the gate verdict |
| [Scale-0 resource report](SCALE-0-RESOURCE-REPORT.md) | Measured parameters, memory, throughput, cache, planner, and verifier figures, generated from the run artefact |
| [Decision log](DECISION-LOG.md) | Architecture decisions, rejected alternatives, evidence labels, and revisit triggers |

The root strategy is
[SENTINEL-HYBRID-WORLD-MODEL-STRATEGY.md](../../SENTINEL-HYBRID-WORLD-MODEL-STRATEGY.md).
The mathematical cycle and reproducibility artifacts are under
`math-findings/01 Long-Horizon Credit Assignment + Compositional World Modeling/SHWM Action-Conditioned Hybrid World Model/`.

## Definition of setup complete

Setup is complete when:

1. both branches exist at the recorded baseline;
2. the Phase-2 strategy and roadmap amendment are committed;
3. the A–I mathematical cycle has a bounded primary-source audit;
4. the Lean file compiles without unfinished proofs;
5. symbolic, exact, property, and resource checks pass;
6. generated results and figures have checksums;
7. the implementation handoff explicitly forbids final-test sampling;
8. the original dirty Phase-1 checkout remains unchanged.

Setup does not unblock Scale 1 and does not establish any neural capability.
