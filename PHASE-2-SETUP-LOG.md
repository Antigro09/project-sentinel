# Phase-2 Continuous World Model Setup Log

Date: 2026-08-28

## Outcome

The continuous action-conditioned world-model direction is now a separate,
committed Sentinel research track. The exact system remains an immutable
reference, while Phase 2 has a detailed architecture, frozen development
matrix, strict experiment gates, mathematical cycle, formal proofs,
reproducibility pipeline, implementation handoff, and updated project roadmap.

Primary content commit:

```text
3eefdf844c9a683bb6758368c6368b0da8b9e112
Phase 2: establish SHWM research and execution plan
```

The commit contains 45 scoped files, 5,913 inserted lines, and five changed
legacy roadmap lines. No production neural implementation is claimed.

## Branch topology

```text
phase-1-exact-reference
  5205543b110ba6da2e3f6da30630809941f821c4
  immutable exact reference; the ref must never advance

phase-2-continuous-world-model
  forked from 5205543b110ba6da2e3f6da30630809941f821c4
  content commit 3eefdf844c9a683bb6758368c6368b0da8b9e112

phase-1-verifier
  continued independently and was observed at
  799fd1d6e557d767911d39c49985f6008f87eaf4 after this setup
```

The Phase-2 worktree is:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel-phase-2-continuous-world-model
```

The primary Phase-1 checkout was never staged, reset, copied, cleaned, or
committed by this setup. It advanced independently while Phase 2 was isolated.
Untracked Phase-1 research/worktree directories remain user-owned.

No branch was pushed and no external repository or service received private
Sentinel content.

## Strategic decision recorded

The setup records the following narrow verdict:

1. Continuous action-conditioned world modeling is a justified experiment.
2. No accepted 15B–70B AGI parameter lower-bound theorem was found in the
   bounded primary-source audit; global nonexistence remains `UNKNOWN`.
3. No accepted requirement for a continuous 1,024-dimensional open-world
   latent was found; 256, 512, and 1,024 are experimental values.
4. Continuous, discrete, and hybrid state remain matched arms.
5. Frozen pretrained grounding is inherited capability, not Sentinel-learned
   knowledge.
6. The learned model proposes; exact observable tests, probes, constraints, and
   authority decide.
7. X65B-core is operationally mandatory before X66 may claim context-scoped
   continual transfer.
8. Scale 0–5 may proceed in parallel with the exact X65 line; Scale 6 must
   consume a committed memory contract, and Scale 7/X66 requires both lines.

## Root and implementation documents

The main parent-folder strategy is:

- `SENTINEL-HYBRID-WORLD-MODEL-STRATEGY.md`

It records the verdict, branch split, SHWM architecture, equations, uncertainty
decomposition, verifier and VDFM bridges, training objective, causal warning,
resource ladder, environments, data plan, Scale 0–8 gates, X65B/X66 dependency,
post-world-model AGI path, and honest evidence status.

The implementation package under `docs/phase-2-continuous-world-model/`
contains:

- `README.md`;
- `ARCHITECTURE.md`;
- `SCALE-0-IMPLEMENTATION-PLAN.md`;
- `SCALE-0-RUN-MATRIX.md`;
- `EXPERIMENT-GATES.md`;
- `BRANCH-AND-FREEZE-PROTOCOL.md`;
- `AGI-DEPENDENCY-ROADMAP.md`;
- `CLAUDE-CODE-PROMPT-SCALE-0.md`;
- `DECISION-LOG.md`.

`ROADMAP.md` now contains the Phase-2 scale track, the corrected X65A state,
the mandatory X65B-core gate, the X66 entry contract, and the later integrated
multimodal/causal/abstraction/memory/metacognition/agency/cross-domain path.

## Frozen Scale-0 matrix

The reader audit found that “50M–100M,” “50M and 200M,” “prototype,” and “equal
budget” had initially been used inconsistently. Those formulations were
replaced by one canonical contract:

```text
2 frozen encoder families
× 3 representation arms
× 2 trainable sizes
= 12 primary cells

12 cells × 3 development seeds
= 36 primary workloads

2 encoders × hybrid 50M × 2 added widths × 3 seeds
= 12 dimension-control workloads

total = 48 mandatory Scale-0 workloads
```

Frozen development factors include:

- Qwen3-VL 4B and Gemma 3 4B encoder families;
- continuous, discrete, and hybrid representations;
- 50M and 200M trainable targets within ±1%;
- primary width 512;
- 256 and 1,024 width controls for the 50M hybrid arm;
- development seeds 6600, 6601, and 6602;
- exactly 100,000 sealed development transitions;
- 200 AdamW updates, batch 32, sequence length 32;
- zero per-run online interactions;
- fixed planner invocations/candidates and evaluator-required probes;
- 112 GiB peak-process, 200 GiB artifact, per-run, cache, and total wall-time
  ceilings.

These are throughput workloads, not capability runs. They have not been
executed. If one encoder is incompatible, the gate stops; replacement requires
a committed pre-run amendment and complete matrix restart.

## Corrected causal attribution fixtures

The independent audit also caught a conflation between action conditioning and
hidden-state aliasing. The setup now requires two independent fixtures:

1. Restore the identical complete simulator state, force different actions,
   and require different observable successors.
2. Hold current observation and action fixed, vary hidden history, and require
   different observable successors.

The first tests whether action enters dynamics. The second tests whether the
recurrent belief state retains sufficient history. Hidden simulator state is
evaluator-only in both.

## Math archive

The committed `math-findings/` archive contains the requested ten topics:

1. Long-Horizon Credit Assignment + Compositional World Modeling
2. Systematic Compositional Reasoning
3. Continual Learning Without Catastrophic Forgetting
4. Causal Abstraction and Intervention-Robust Reasoning
5. Generalization Under Distribution Shift
6. Planning Depth and Tool-Using Agency
7. Meta-Learning + Self-Reflection
8. Memory Architecture for Persistent Knowledge
9. Abstraction and Concept Formation
10. Multi-Objective Alignment of Capabilities

The canonical protocol is at `math-findings/RESEARCH-PROTOCOL.md`. The SHWM
cycle is in topic 1 under:

```text
SHWM Action-Conditioned Hybrid World Model/
```

That cycle contains:

- the strict A–I research report;
- bounded primary-source prior-art audit;
- Claude Code implementation specification;
- Lean 4/Mathlib proof file;
- SymPy, exact-enumeration, Hypothesis, JAX, SciPy, CVXPy, NetworkX, NumPyro,
  Matplotlib, Hydra, MLflow, and DVC checks;
- verification report;
- DVC config/lock;
- SHA-256 artifact ledger.

## Formal and numerical results

### Reproduced, mechanically checked in Lean

- finite machine-code cardinality;
- action-word cardinality `B^H`;
- finite-horizon verifier-quotient trace-score sufficiency;
- passive-policy non-identifiability construction;
- nonnegative weighted loss and ensemble disagreement;
- finite geometric rollout recurrence;
- exact observable mismatch rejection;
- non-injective-probe coverage counterexample.

Lean compiled with Mathlib without `sorry`, `admit`, or unfinished proof.

### Measured finite diagnostics

- SymPy residuals and posterior normalization passed.
- Exact enumeration preserves posterior `(0.5, 0.5)` through 64 passive
  observations in the constructed pair and separates under intervention.
- Six Hypothesis properties ran 200 derandomized examples each: 1,200 total.
- JAX action fixture: action-conditioned MSE `0`, best action-blind MSE `1`.
- JAX history fixture: history-conditioned MSE `0`, best observation-only MSE
  `1`.
- Scale-0 arithmetic: 12 cells, 36 primary runs, 12 dimension runs, 48 total;
  204,800 transition positions and 19,200 planner candidates per run.
- SciPy recovered the planted rollout sensitivity `0.9` to numerical precision.
- CVXPy returned an optimal 200M accounting allocation scaffold.
- NetworkX verified every declared proposal-to-action path crosses the verifier.
- NumPyro normalized the finite intervention posterior.
- Five Matplotlib figures were generated and visually inspected.

All of these are finite/toy/tooling results. None is evidence that SHWM improves
planning, transfer, causal reasoning, continual learning, or AGI capability.

## Reproducibility corrections retained

The log preserves setup failures rather than hiding them:

1. DVC rejected incompatible `--subdir --no-scm` initialization flags; the
   corrected repository-backed command was used.
2. MLflow 3.15 rejected the legacy file store; local SQLite tracking passed.
3. A clean-shell `dvc repro` exposed bare `python` as non-reproducible; all
   stages now pin the dedicated math interpreter and rerun successfully.
4. The first isolated repository suite lacked ignored offline environment
   assets. A temporary link to the existing asset bundle allowed the affected
   tests to run; the link was removed afterward and source assets were not
   changed.
5. The pre-commit whitespace audit rejected Markdown hard-break whitespace and
   DVC lock formatting; the files were normalized before commit.

No mathematical conclusion depended on a failed setup attempt.

## Repository verification

The first full isolated run reported:

```text
486 passed, 5 skipped, 7 failed, 9 errors in 967.16s
```

Every non-pass traced to the absent ignored offline asset bundle. After the
temporary local link, the affected tests reported:

```text
58 passed in 21.09s
```

The corrected full rerun reported:

```text
521 passed, 1 skipped in 940.11s (15:40)
```

Additional final checks:

- four result JSON files parse;
- DVC reports the pipeline up to date;
- all Python research scripts byte-compile;
- Lean recompiles successfully;
- all 28 SHA-256 entries verify;
- the strict report has exactly one section A through I;
- 31 Markdown files have zero broken local links;
- `git diff --cached --check` passes;
- the Phase-2 worktree was clean after the primary content commit.

## Independent reader audit

The first reader pass found six substantive protocol issues:

1. an illegal documentation-commit exception on the immutable exact branch;
2. missing checksum/completion artifacts;
3. contradictory model sizes and ambiguous budget matching;
4. conflated action and hidden-history tests;
5. missing boundary loss and stale X65 roadmap text;
6. noncanonical evidence labels and model-controlled probe risk.

All were corrected. The second pass reduced the package to one wording defect:
`RETRACTED` was missing from two canonical label lists. That was corrected.
The final targeted reader verdict was:

```text
PASS — no content blocker remains from the audit.
```

## Deliberately not done

- no pretrained backbone downloaded;
- no 50M or 200M model implemented or trained;
- no Scale-0 workload executed;
- no Phase-2 validation or final seed sampled;
- no final benchmark run;
- no OSWorld or SWE-bench development;
- no claim that continuous, discrete, or hybrid representation wins;
- no claim that the verifier, memory, or world model has earned its place;
- no claim of AGI, causal understanding, common sense, or human-level ability;
- no external push or publication.

## Handoff boundary

The next implementation task is the committed Scale-0 Claude Code prompt only.
It may begin when Anthony explicitly authorizes it. Scale 1 is **not unblocked**
by this setup.

X65B-core remains required before X66 can support the claimed cross-repository
continual-transfer gate. Full sheaf/category machinery remains optional; the
operational scoped-transfer evidence is not.

The acceleration mechanism is parallel work with a mandatory rejoin, not gate
skipping:

```text
exact X65A-P/X65B evidence line --------┐
                                       ├─ Scale 6 continual integration
SHWM Scale 0–5 evidence line ----------┘
                                              │
                                              └─ Scale 7 / X66
```
