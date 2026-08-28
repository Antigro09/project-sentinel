# Sentinel Phase 2: Hybrid Action-Conditioned World-Model Strategy

Status: architecture and experiment plan; no scaled model has been trained
Date: 2026-08-28
Branch: `phase-2-continuous-world-model`
Exact reference: `phase-1-exact-reference` at `5205543b110ba6da2e3f6da30630809941f821c4`

## Executive verdict

The move toward learned, action-conditioned world models is justified. The
claim that AGI has a known 15B–70B parameter lower bound is not justified, and
neither is a requirement for a continuous 1,024-dimensional latent.

A bounded primary-source audit found no accepted theorem that turns parameter
count into an AGI threshold. Hoffmann et al. derive empirical compute-optimal
relationships among transformer parameters, data, and training compute; their
result is not an AGI lower bound. Current official open multimodal families
also include models below 15B: Qwen3-VL includes dense 2B, 4B, 8B, and 32B
variants, and Gemma 3's technical-report family includes 1B, 4B, 12B, and 27B
variants. This does not prove that a small model can become AGI. It does show
that 15B is not an established theoretical floor.

The representation question is also empirical. V-JEPA 2 supports continuous
predictive embeddings and action-conditioned latent planning. DreamerV3 uses a
recurrent state-space world model with categorical stochastic state. Neither
result establishes a universal representation requirement. Sentinel will
therefore compare continuous, discrete, and hybrid state under matched
trainable-parameter, interaction, and compute budgets.

The strategic core is:

> Give Sentinel pretrained perceptual grounding and learned
> action-conditioned dynamics, while retaining exact execution, uncertainty,
> counterexamples, provenance, revision, and abstention as the control and
> audit layer.

This is a new architecture hypothesis, not evidence of a capability.

## What is frozen and what is allowed to change

### Phase 1 exact reference

`phase-1-exact-reference` is pinned to commit
`5205543b110ba6da2e3f6da30630809941f821c4`. It preserves the last committed
exact reference before Phase 2 was created. It remains:

- the trusted deterministic execution path;
- the behavioral-equivalence oracle where finite enumeration is possible;
- the source of exact counterexamples;
- the memory, provenance, and restart reference;
- the abstention and open-world control;
- the baseline against which neural components must earn their place.

The original `phase-1-verifier` checkout contained unfinished X65A-L1 work at
branch creation. That work was deliberately not copied into either freeze
point or the Phase-2 setup commit. It must obtain its own complete suite,
commit, and phase verdict before it changes the canonical X65 state.

### Phase 2 continuous world model

`phase-2-continuous-world-model` starts from the same committed baseline. It
may add:

- frozen perceptual backbone adapters;
- latent caches and transition datasets;
- stochastic belief-state models;
- action-conditioned dynamics;
- event, reward, termination, and uncertainty heads;
- latent planning;
- observable verifier bridges;
- continuous/discrete/hybrid experimental arms;
- high-dimensional environment adapters.

It may not silently alter Phase-1 evaluators, weaken exact gates, import final
answers, sample final test seeds before freeze, or count inherited backbone
knowledge as Sentinel-learned knowledge.

## The architecture hypothesis

Call the first architecture **Sentinel-Hybrid World Model (SHWM)**.

```text
raw text / image / video / audio / GUI / sensor observation
                            │
                            ▼
                  frozen pretrained encoders
                            │
                            ▼
                 small trainable projectors
                            │
                            ▼
                 stochastic latent belief state
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      fast proposal policy      action-conditioned world model
                                         │
                                         ▼
                               latent imagination/planning
               └───────────────┬─────────┘
                               ▼
                     Sentinel act/ask/abstain gate
                               │
                               ▼
                  exact observable verifier bridge
                               │
                               ▼
                         external action
                               │
                               ▼
                         real environment
                               │
                observation / test / counterexample
                               │
                               ▼
       episodic + semantic + procedural + negative/revision memory
```

The falsifiable claim is that the combination

\[
\text{pretrained grounding}
+\text{learned action-conditioned dynamics}
+\text{verified memory and control}
\]

improves planning and transfer over reactive, model-free, symbolic-only, and
world-model-only controls under matched resources.

## Formal system definition

Let the environment be a partially observed controlled process with hidden
state \(s_t\), multimodal observation \(o_t\), action \(a_t\), progress signal
\(r_t\), and termination indicator \(d_t\). Let \(g_t\) be the current goal,
and let \(M_t\) be persistent Sentinel memory.

Frozen encoders and trainable projectors produce:

\[
x_t=P_\psi(E(o_t),E_g(g_t)).
\]

The recurrent stochastic belief update is:

\[
q_\theta(b_t\mid b_{t-1},x_t,a_{t-1},r_{t-1},R(M_t)).
\]

The action-conditioned world model predicts:

\[
p_\theta(x_{t+1},e_{t+1},r_t,d_t
\mid b_t,a_t),
\]

where \(e_{t+1}\) is a structured verifier-facing event. The event vocabulary
is initially limited to observable changes such as inventory changes, object
appearance/disappearance, focus movement, file or UI changes, action success,
constraint violation, and goal progress. It is a probe interface, not a claim
to have discovered the complete ontology.

The planner selects a candidate action sequence:

\[
a^*_{t:t+H}
=\arg\max_{a_{t:t+H}}
\mathbb E\!\left[\sum_{k=0}^{H}\gamma^k r_{t+k}\right]
-\lambda_U U(b_t,a_{t:t+H})
-\lambda_C C(b_t,a_{t:t+H}),
\]

where \(U\) is epistemic/model-inadequacy cost and \(C\) is predicted
constraint or safety cost. This objective does not make long-horizon search
cheap: with branching factor \(B\) and horizon \(H\), exhaustive open-loop
enumeration contains \(B^H\) sequences. Proposal guidance, options, and
hierarchy are required engineering responses, not eliminations of worst-case
combinatorics.

## Representation contest

The first comparison has three resource-matched arms.

| Arm | State | Strength to test | Primary failure to test |
|---|---|---|---|
| Continuous | predictive real-valued embeddings | smooth prediction and optimization | representation collapse, drift, ungrounded geometry |
| Discrete | categorical stochastic state | explicit uncertainty and stable symbols | codebook collapse, brittle quantization |
| Hybrid | continuous perceptual state plus discrete event/mechanism variables | verifier-facing structure with perceptual flexibility | duplicated state, hard optimization, interface mismatch |

Latent dimensions 256, 512, and 1,024 are experimental values, not a
capability ladder. Equal trainable model size and equal data are required.
Selection depends on planning success, interventional prediction,
calibration, transfer, and rollout stability—not reconstruction loss alone.

At finite precision, a \(d\)-coordinate latent with \(p\) bits per coordinate
has at most \(2^{pd}\) machine codes. This does not imply a one-to-one concept
capacity, because meanings can be context dependent, but it rules out the
literal claim that a finite machine vector operationally stores infinitely
many distinct states.

## Uncertainty and safe control

SHWM must keep three failure sources separate:

1. **Aleatoric uncertainty:** several outcomes are genuinely possible.
2. **Epistemic uncertainty:** the model lacks enough identifying experience.
3. **Model inadequacy:** no in-class transition explains an observation.

An ensemble diagnostic can be written as:

\[
U(b,a)=\operatorname{Disagreement}
\{f_{\theta_k}(b,a)\}_{k=1}^K.
\]

The controller chooses among `ACT`, `ASK`, `OBSERVE`, `RUN_TEST`, `REPLAN`,
`SWITCH_MODEL`, `ABSTAIN`, and `EXPAND_REPRESENTATION` by minimizing expected
decision loss plus action/query/resource cost. Exact thresholds are frozen per
experiment and calibrated against always-act and always-abstain controls.

No ensemble disagreement score by itself proves open-world detection. Model
inadequacy must be tested using held-out mechanics and constructions whose
correct response is clarification, expansion, or abstention.

## Verifier bridge

The exact verifier does not compare latent vectors for equality. It evaluates:

- externally observable outcomes;
- structured event predictions;
- unit tests and compiler results;
- tool outputs;
- reward/progress and terminal state;
- decoded constraints;
- temporal invariants;
- permitted state probes.

Every real transition is evidence against rollout hypotheses that predicted a
different observable. The mechanically checked finite theorem is exact but
limited: if predicted and observed values differ, an equality verifier rejects
the prediction. A second checked counterexample shows the boundary: a
non-injective probe can hide different latent states. Therefore verifier
coverage and event/probe sufficiency must be reported separately from
verifier correctness.

## Memory integration

SHWM retains the four X65 memory classes:

- episodic evidence;
- semantic beliefs;
- procedural skills;
- negative, stale, contradicted, and revision evidence.

Continuous embeddings can propose retrieval candidates. They cannot create
truth, provenance, verification scope, dependency edges, or universal skill
validity. A stored entry must include:

```text
embedding
structured claim, event, or skill
confidence and calibration record
verification domain and validity context
world-model and encoder version
source observations and action trace
counterexamples and superseded state
continuation-relevant side effects
taint / leakage status
```

Scale 6 cannot begin until the X65A/X65B evidence line has a committed phase
verdict. The Phase-2 model must consume the verified memory contract; it must
not redefine continual learning as embedding retrieval.

## Training objective

The initial objective is:

\[
\begin{aligned}
\mathcal L ={}&
\mathcal L_{\text{latent-next}}
+\alpha\mathcal L_{\text{multi-step}}
+\beta\mathcal L_{\text{reward}}
+\gamma\mathcal L_{\text{termination}}\\
&+\delta\mathcal L_{\text{inverse-action}}
+\epsilon\mathcal L_{\text{event}}
+\zeta\mathcal L_{\text{uncertainty}}
+\eta\mathcal L_{\text{consistency}}
+\xi\mathcal L_{\text{boundary}}.
\end{aligned}
\]

Every component and coefficient is ablated. The weighted objective is
nonnegative when each loss and weight is nonnegative, but this elementary
fact is not a convergence guarantee. The components have distinct roles:

- next-latent prediction supplies one-step predictive state;
- multi-step loss exposes compounding rollout error;
- reward and terminal heads make prediction planning-relevant;
- inverse action preserves controllable distinctions;
- event prediction creates a verifier-facing channel;
- uncertainty calibration penalizes confident model exploitation;
- consistency aligns observed and imagined updates;
- boundary separation penalizes collapse of branch-group states whose admitted
  action effects are verifier-distinct.

Pixel reconstruction is an optional diagnostic, never the sole objective.

## Why action interventions are mandatory

Passive prediction does not generally identify action effects. The finite
checked counterexample contains two transition models that agree after the
only action selected by the behavior policy and disagree after an unselected
action. Any number of repeated passive observations leaves posterior mass
split equally. One distinguishing action resolves the deterministic pair.

Consequences for data collection:

- branch several actions from matched initial states;
- log action propensities and collection policy versions;
- hold out action combinations and changed mechanics;
- measure interventional, not only observational, prediction;
- keep `insufficient action coverage` distinct from `wrong dynamics class`.

The result is a non-identifiability warning, not a claim that intervention
data always identifies a high-dimensional causal model.

## Practical resource ladder

Do not begin with a fully trainable 15B model. Under common Adam-style
accounting, a 15B model requires about 180 GB for bf16 weights, bf16 gradients,
and fp32 first/second moments, or about 240 GB when fp32 master weights are
also retained. Both figures exclude activations, allocator overhead, and
framework buffers. A raw 70B four-bit weight array is 35 decimal GB, but
scales, metadata, KV cache, activations, adapters, and runtime overhead are
additional. Fitting quantized weights for inference is not the same as full
training feasibility.

| Tier | Frozen backbone | Trainable world model | Decision purpose |
|---|---:|---:|---|
| S0 | two frozen 4B controls | 50M and 200M | pipeline, attribution, throughput |
| S1 | 4B–8B | 200M–400M | real action-conditioned prediction |
| S2 | 8B–12B | 700M–1.5B | long horizon and cross-environment tests |
| S3 | 12B–32B or sparse model | 1B–3B plus adapters | only after a positive capability curve |
| S4 | 70B class | frozen teacher/oracle | distillation and ceiling only |

For one million cached 512-dimensional fp16 frames, raw latent storage is
1.024 decimal GB (0.954 GiB) before metadata, indexing, actions, and sequence
boundaries.

Rules:

1. Scale one axis at a time.
2. Use at least three seeds at decisive tiers.
3. Fit capability curves, not only training-loss curves.
4. Stop scaling when confidence intervals do not support a capability gain.
5. Record peak resident memory, wall time, energy proxy, and real interactions.

## Environment ladder

### Stratum A: procedural visual control

Use Procgen or a small generated equivalent for held-out levels, appearances,
layouts, and dynamics. It is suitable for pipeline and generalization
experiments, not evidence of open-ended intelligence.

### Stratum B: open-world long-horizon control

Use Crafter or an equivalent survival environment for partial observation,
inventory, exploration, achievement dependencies, and modified mechanics.

### Stratum C: computer and tool environments

Only after the simulator gates pass, add sandboxed screenshots, structured
tool results, files, and reversible actions. OSWorld is a later external test,
not the initial training environment.

### Stratum D: software engineering

Repository tasks enter at Scale 7/X66 after memory, causal, verifier, and
transfer prerequisites. Tests remain the authority. SWE-bench is an eventual
contamination-controlled external gate, not a development set.

## Data plan

Use pretrained encoders; do not attempt internet-scale perceptual pretraining
locally. Cache frozen features with immutable encoder hashes. The first
million-transition target is a planning estimate, not a required quota.

Initial collection mixture:

```text
30% random exploration
25% scripted or oracle-assisted trajectories
25% current Sentinel exploration
20% uncertainty-seeking or adversarial exploration
```

The percentages are preregistered starting values and must be ablated. For
selected initial states, collect paired actions to supervise
\(p(x_{t+1}\mid x_t,a_t)\). Independently hold out environment seeds, visual
appearance, mechanics, goals, action combinations, and complete task families.

No Phase-2 final test seed, hidden dynamic, or evaluator answer is sampled or
exposed before the freeze manifest is committed. Historical X64H seed
artifacts belong to the inherited exact line and are not Phase-2 seeds.

## Scale sequence and gates

### Scale 0: premise and throughput preflight

Build contracts, frozen encoder adapters, latent cache, two environment
adapters, an action-conditioned sequence schema, 50M and 200M configuration
implementations, and matched representation arms. The canonical development
matrix is frozen in
[`SCALE-0-RUN-MATRIX.md`](docs/phase-2-continuous-world-model/SCALE-0-RUN-MATRIX.md):
two named frozen encoders × three representations × two trainable sizes at
width 512, three development seeds, plus a matched 256/1,024 width sensitivity
control. Produce throughput and memory reports only.

Gate: all 48 frozen development workloads complete under one unchanged
evaluator, the exact matching rules, and the preregistered 112 GiB peak-memory,
200 GiB artifact, per-run, and total wall-time ceilings. No capability claim.

### Scale 1: representation contest

Train continuous, discrete, and hybrid arms with the same backbone, data,
trainable parameters, optimizer budget, and planner.

Measure one-step prediction; 5-, 10-, and 25-step rollout consistency;
action-effect discrimination; reward/terminal prediction; calibration; and
real control success.

Falsifier: if representation changes prediction metrics but not planning,
transfer, or calibration, do not describe it as a cognitive advantage.

### Scale 2: world model earns its place

Compare reactive frozen-feature policy, sequence model without action
conditioning, model-free baseline, DreamerV3 baseline, SHWM, and oracle
simulator under equal real interactions.

Gate: SHWM improves task success at equal interactions or reduces interactions
to matched success, with a paired interval excluding zero. Prediction accuracy
alone does not pass.

### Scale 3: verifier contribution

Compare world model only, symbolic Sentinel only, hybrid without verifier,
hybrid with observable verification, and hybrid with verification plus memory.

Gate: verification reduces real rollout failures or improves calibration
without erasing the planning gain.

### Scale 4: counterfactual and causal audit

Change correlations, mechanics, policies, irrelevant visual features, and
forced interventions.

Gate: the selected model beats an equal-size correlational/passive predictor
on intervention outcomes. Do not infer causal understanding from video
prediction.

### Scale 5: cross-environment transfer

Freeze the shared belief/dynamics core and allow only thin observation/action
adapters and bounded adaptation in a held-out environment.

Gate: prior training reduces interactions or optimization steps; resetting the
shared core and memory removes the gain.

### Scale 6: continual learning at continuous scale

Integrate the committed X65A/X65B memory contract. Test recurring environments,
mechanic changes, distractor gaps, false memories, stale skills, byte/retrieval
budgets, and real process restart.

Gate: forward transfer, retention, revision, negative-transfer resistance,
bounded active growth, and restart persistence survive matched controls.

### Scale 7: computer and code domains

Add tool/GUI environments and then X66 repository tasks using the same memory,
uncertainty states, planning interface, and verifier contract.

Gate: no separately trained cognitive core per domain, and repository transfer
must beat matched no-memory and shuffled-memory controls without harming
unrelated repositories.

### Scale 8: parameter and data scaling study

Only after Scales 2–5 pass, vary at least three model sizes and three data
sizes while holding evaluation fixed. Estimate capability as a function of
parameters, transitions, and horizon, with uncertainty. Decide whether the
bottleneck is size, data diversity, state, planner, uncertainty, or memory.

## Required controls

Every major claim requires, where applicable:

- frozen pretrained backbone only;
- random frozen encoder;
- no action conditioning;
- no recurrence;
- no world model;
- no verifier;
- no memory;
- continuous-only, discrete-only, and hybrid;
- symbolic-only;
- oracle simulator;
- larger-budget baseline;
- equal cumulative training and planning compute;
- reset and shuffled-memory controls.

Inherited backbone capability is separately reported. If the frozen VLM alone
causes the result, the Sentinel addition has not earned its place.

## X65B and X66 dependency decision

X65B-core is mandatory before X66 can support a continual-transfer claim. The
minimal X65B scope is context-scoped reusable components:

- validity context and repository/domain scope;
- dependency and provenance edges;
- transfer to structurally related but nonidentical tasks;
- explicit context-switch detection;
- negative transfer and poisoned/stale memory controls;
- composition of prior procedures without storing the final target.

Full sheaf-theoretic or category-theoretic machinery is optional. The
operational capability—scoped transfer without destructive reuse—is not.

Phase 2 does not need to wait for all of X65B to begin Scale 0–5 because those
scales test representation, dynamics, verification, and environment transfer.
Scale 6 must rejoin the X65 evidence line. Scale 7/X66 is blocked until both
the continuous world model and context-scoped memory have passed their gates.

## Eventual path toward an integrated AGI candidate

A successful world model is one subsystem. The dependency path after early
SHWM success is:

1. **Multimodal belief state:** text, vision/video, audio, GUI/tool state,
   sensors, and actions contribute causally under ablation.
2. **Causal and counterfactual modeling:** mechanism changes and interventions,
   not only next-state prediction.
3. **Abstraction and concept formation:** induced objects, events, relations,
   affordances, mechanisms, and procedures improve grounded behavior.
4. **Lifelong multimodal learning:** fast belief updates, persistent structured
   memory, controlled slow parameter adaptation, revision, and restart.
5. **Metacognitive control:** uncertainty changes act/ask/test/replan/abstain
   decisions and reduces catastrophic confidence without universal abstention.
6. **Hierarchical bounded agency:** subgoals, options, monitoring, recovery,
   resource budgets, authority checks, and correct stopping over hundreds to
   thousands of actions.
7. **Cross-domain transfer:** physical, digital, code, and social domains use
   one shared core and memory with thin adapters.
8. **Controlled self-improvement:** sandboxed proposals, frozen evaluation,
   attribution, rollback, and rejection of evaluator gaming.
9. **Frozen integrated evaluation:** all five Sentinel acceptance criteria in
   one unchanged accumulated system.

Parameter count remains an independent variable throughout; it is not a phase
label.

## Final integrated acceptance criteria

| Capability | Operational gate |
|---|---|
| Cross-domain generalization | Unrelated held-out domains, frozen shared core, thin adapters, gain over from-scratch learning, reset removes gain |
| Continual autonomous learning | Positive transfer, retention, revision, bounded active memory, restart, and no contamination in long streams |
| Abstract/causal/common-sense reasoning | Interventions, counterfactuals, hidden mechanisms, physical and social assumptions, incomplete/noisy evidence |
| Goal-directed agency | Autonomous subgoals, tools, monitoring, recovery, authority, resources, and correct stopping on long tasks |
| Multimodal integration | Text, vision/video, audio, sensor/tool state, and actions in one belief model with cross-modal ablations and corruption tests |

The architecture, evaluators, thresholds, domain adapters, and test generation
procedures are frozen before final domains and seeds are revealed. Report
per-domain floors, calibration, catastrophic errors, interaction cost, compute,
and retention. High average performance cannot hide failure of one pillar.

## Evidence status at setup

Canonical project evidence labels remain `MEASURED`, `REPRODUCED`, `INFERRED`,
`HYPOTHESIS`, `SPECULATIVE`, `RETRACTED`, and `UNKNOWN`. “Mechanically checked
in Lean” and “numerically checked” below are verification qualifiers, not
replacement evidence labels.

### REPRODUCED — mechanically checked in Lean

- Lean 4 + Mathlib checks finite latent-code cardinality.
- Lean checks \(B^H\) open-loop sequence cardinality.
- Lean checks a passive-policy non-identifiability construction.
- Lean checks nonnegative weighted loss and squared disagreement.
- Lean checks the finite rollout-error recurrence.
- Lean checks exact observable mismatch rejection and a non-injective-probe
  boundary case.

### MEASURED — finite numerical and symbolic diagnostics

- SymPy confirms recurrence, posterior, and memory arithmetic.
- Exact enumeration confirms the passive/interventional two-model result.
- Hypothesis checks 1,200 finite cases across six helper properties.
- JAX confirms action conditioning is necessary in a two-transition toy where
  an action-blind affine predictor has MSE 1 and an action-conditioned one has
  MSE 0.
- A separate JAX toy holds current observation and action fixed while changing
  history; a history-conditioned predictor has MSE 0 and the best
  observation-only constant has MSE 1.
- Hydra/resource checks validate the frozen matrix arithmetic: 12 primary
  cells, 36 primary runs, 12 dimension controls, and 48 workloads.
- SciPy, CVXPy, NetworkX, NumPyro, Matplotlib, Hydra, and MLflow checks execute
  as reproducibility diagnostics.

### HYPOTHESIS

- learned high-dimensional dynamics will improve control;
- the verifier will reduce model exploitation without erasing gains;
- hybrid latent state will outperform matched alternatives;
- shared dynamics will transfer across environments;
- continuous-scale VDFM will retain its finite controlled benefits.

### UNKNOWN

- the best backbone, state dimension, representation type, model size, planner,
  or data scale;
- whether the M5 Max can train the proposed tiers at useful throughput;
- whether any SHWM component improves real planning or transfer;
- whether the integration is novel beyond the narrow bounded audit;
- how far any successful result would generalize toward human-level AGI.

## Primary references used for the setup decision

- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- Assran et al., [V-JEPA 2](https://arxiv.org/abs/2506.09985).
- Hafner et al., [DreamerV3](https://arxiv.org/abs/2301.04104).
- Qwen Team, [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL) and [technical report](https://arxiv.org/abs/2511.21631).
- Gemma Team, [Gemma 3 technical report](https://arxiv.org/abs/2503.19786) and [official model card](https://ai.google.dev/gemma/docs/core/model_card_3).
- Cobbe et al., [Procgen](https://proceedings.mlr.press/v119/cobbe20a.html).
- Hafner, [Crafter](https://arxiv.org/abs/2109.06780).
- Xie et al., [OSWorld](https://arxiv.org/abs/2404.07972).
- Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770).
- Reed et al., [Gato](https://arxiv.org/abs/2205.06175).
- Apple MLX, [official examples](https://github.com/ml-explore/mlx-examples).

The literature audit is bounded and non-exhaustive. Absence of a found theorem
or prior integration is not a proof of absence.
