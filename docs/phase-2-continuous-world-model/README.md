# Phase 2 Continuous World Model

This directory is the implementation-facing plan for the
`phase-2-continuous-world-model` branch. It does not claim that a learned world
model has been implemented or has passed a capability gate.

## Canonical project status

The single place to read current status. Narrative sections elsewhere record how
a line developed and may describe intermediate states that are no longer current.

| Item | Status |
|---|---|
| X64 | **Closed** at the controlled hidden-convention level, 2026-08-27. Narrow: not language understanding in open vocabulary or on real text. |
| Phase-1 exact reference | **Frozen** at `5205543b110ba6da2e3f6da30630809941f821c4`. |
| X65A | Latent-identity development work reached; **X65A as a whole is not closed**. X65A-P and X65B-core remain pending. |
| SHWM Scale 0 | **Passed** at commit `f694c23`. Infrastructure only. |
| SHWM Scale 1 | **No capability result exists.** |

Scale 0's pass establishes that the pipeline runs, is matched, restarts, and fits
the machine. It establishes no world model and no representation winner.

## Current state

- `phase-1-exact-reference` is frozen at
  `5205543b110ba6da2e3f6da30630809941f821c4`.
- Phase 2 was forked from the same commit in an isolated worktree.
- The formal and numerical setup audit is complete for finite helper claims.
- **Scale 0 has passed** at `f694c23`. All 48 mandatory workloads ran against
  both named frozen backbones under the frozen matching contract, and all ten
  gate clauses hold. This unblocks a preregistered Scale-1 design and nothing
  else; no capability was measured and no representation arm won. See the
  [Scale-0 handoff](SCALE-0-HANDOFF.md).
- Both frozen backbones are present locally at their pinned revisions:
  `Qwen/Qwen3-VL-4B-Instruct` at `ebb281ec` (Apache-2.0) and
  `google/gemma-3-4b-it` at `093f9f38` (Gemma terms). They are frozen throughout
  and their parameters are inherited capability, reported apart from the
  trainable budget.
- Trainable models have been built and run for 200 optimizer updates each at
  plumbing weights. That is a throughput measurement, not training: no neural
  world model has been trained to do anything.
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
