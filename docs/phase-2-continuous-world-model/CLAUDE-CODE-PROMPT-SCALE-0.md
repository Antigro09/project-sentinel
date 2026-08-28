# Claude Code Handoff: SHWM Scale 0

Use this as the first implementation prompt after Anthony authorizes Scale 0.

---

You are implementing **SHWM Scale 0: premise, interfaces, provenance, and
throughput preflight** on branch `phase-2-continuous-world-model`.

Read first:

1. `SENTINEL-HYBRID-WORLD-MODEL-STRATEGY.md`
2. `docs/phase-2-continuous-world-model/ARCHITECTURE.md`
3. `docs/phase-2-continuous-world-model/SCALE-0-IMPLEMENTATION-PLAN.md`
4. `docs/phase-2-continuous-world-model/SCALE-0-RUN-MATRIX.md`
5. `docs/phase-2-continuous-world-model/EXPERIMENT-GATES.md`
6. `docs/phase-2-continuous-world-model/BRANCH-AND-FREEZE-PROTOCOL.md`
7. the existing `src/sentinel/wm/contract.py` and verifier tests

## Objective

Build the smallest audited pipeline that can later compare continuous,
discrete, and hybrid action-conditioned latent world models. This task ends
with contracts, data/cache/adapters, fake-model dry runs, parameter/resource
validation, restart, and a three-seed development throughput report. It does
not train or evaluate a final capability model.

## Immutable baseline

- Base commit: `5205543b110ba6da2e3f6da30630809941f821c4`.
- `phase-1-exact-reference` is the trusted exact reference.
- Do not modify exact `WorldModel` semantics or weaken existing tests.
- The separate `phase-1-verifier` checkout had unfinished X65A-L1 work when
  this branch was created. Do not copy or infer completion from it.

## Build

Implement, in narrow commits:

1. typed latent/belief/dynamics/event/uncertainty/verifier records and Protocols;
2. canonical serialization and content hashes;
3. continuous/discrete/hybrid deterministic fake implementations;
4. frozen-encoder adapter and complete encoder identity;
5. content-addressed latent cache with stale-cache rejection;
6. transition/sequence schemas with branch groups, collection propensities,
   split assignment, and taint;
7. one deterministic adapter with separate action-intervention and
   hidden-history-alias fixtures, plus one procedural visual adapter;
8. 50M and 200M model configuration/actual-parameter-count validators;
9. loss-component and metric plumbing;
10. planner/verifier dry run using fake dynamics;
11. full restart with no undeclared process state;
12. Hydra, local MLflow, DVC, checksums, and resource report.

Preflight the two frozen backbone families behind the same interface:
Qwen3-VL 4B and Gemma 3 4B, only after verifying official license, exact model
revision, local runtime support, preprocessing, and precision. If either is
not faithfully runnable, stop and report the incompatibility. A replacement
requires a reviewed pre-run matrix amendment; after any matrix cell runs, all
cells restart. Do not download a larger model as a workaround.

## Required tests

- exact old suite unchanged;
- schema/serialization round trips and malformed-state rejection;
- hash identity changes with weights, preprocessing, precision, projector, or
  modality mask;
- cache hit/miss and stale cache rejection;
- branch groups cannot cross splits;
- hidden simulator state never enters model input;
- target/evaluator fields cannot enter train records;
- action-blind model fails the same-state/different-action fixture;
- observation-only model fails the same-observation/same-action hidden-history
  fixture, while recurrent belief can represent the distinction;
- actual trainable parameter counts are matched across representation arms;
- data, optimizer updates, interactions, planner calls, seeds, and required
  probes match the frozen run matrix;
- every path to action crosses verifier/authority gate;
- continuous versus restarted fake runs match;
- corruption and forbidden global-state channel are detected;
- final seeds cannot be loaded before final freeze.

## Resource reporting

For every matrix workload, and separately for any fake-model dry run, report:

- cold load time;
- peak unified memory;
- throughput;
- cache payload/index/metadata size;
- trainable and frozen parameters separately;
- model/optimizer/activation estimates versus measured process memory;
- rollout/planner calls per second;
- wall time;
- all failures and retries.

Execute all 48 workloads in `SCALE-0-RUN-MATRIX.md`: 36 primary
encoder/representation/size/seed runs plus 12 dimension-sensitivity runs. Use
its exact 100,000 transitions, 200 updates, batch/sequence shape, optimizer,
planner workload, and 112 GiB/200 GiB/time ceilings. Tiny unit-test models and
one-batch checks are not matrix runs.

Do not call raw four-bit weight size a training-memory requirement. Do not call
nominal hidden width an actual parameter count.

## Forbidden in this task

- final seed sampling;
- final test execution;
- OSWorld, SWE-bench, or 15B/70B training;
- changing evaluation after numbers appear;
- claiming planning, transfer, causality, continual learning, or AGI;
- treating frozen-backbone knowledge as Sentinel-learned;
- editing `phase-1-exact-reference`;
- staging unrelated worktrees or artifacts.

## Stop conditions

Stop and report if exact tests regress, evidence/resources cannot be matched,
cache identity is unstable, a split leaks, restart needs undeclared state, a
backbone exhausts the machine, or the environment cannot expose an
action intervention and a separate hidden-history alias without leaking hidden
state.

## Handoff format

Return:

1. commit hash and branch;
2. focused and full-suite results with counts and runtime;
3. clean/dirty status and exact untracked files;
4. implemented file map;
5. actual parameters, memory, and throughput table;
6. negative findings and bugs corrected;
7. evidence labels;
8. exact Scale-0 gate result;
9. explicit `Scale 1 unblocked` or `Scale 1 not unblocked`;
10. cheapest next falsifier.

Do not begin Scale 1 in the same run.

---
