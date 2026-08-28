# SHWM Research Cycle: Verifier-Quotiented Interventional Belief Dynamics

Status: finite theory and reproducibility checks complete; scaled architecture
untested.
Canonical evidence labels are `MEASURED`, `REPRODUCED`, `INFERRED`,
`HYPOTHESIS`, `SPECULATIVE`, `RETRACTED`, and `UNKNOWN`. “Mechanically checked
in Lean” is a proof qualifier attached only to the listed `REPRODUCED` theorem
artifacts; `MEASURED` applies only to executed toy/resource checks. All
architectural capability claims are `HYPOTHESIS` or `UNKNOWN`.

### A. Problem Target

#### Capability gap

Phase-1 Sentinel represents worlds as explicit executable hypotheses over
human-designed observations, actions, symbols, and finite candidate pools. This
supports exact refutation, ambiguity, abstention, counterexamples, provenance,
and restart. It does not establish that Sentinel can induce useful predictive
state from high-dimensional raw observations or plan under learned dynamics.

A direct replacement with a monolithic multimodal policy creates a different
gap: it weakens attribution and makes it difficult to distinguish a useful
world model from inherited backbone knowledge, reactive pattern matching,
planner compute, or evaluator leakage.

The target is therefore:

> Learn an action-conditioned stochastic belief model from high-dimensional
> observations, use it for finite-horizon planning, and quotient only those
> latent distinctions that are irrelevant to the current exact verifier probes
> and action horizon.

The construct must support at least four AGI-relevant directions:
distribution-shift generalization, causal/interventional world modeling,
continual revision, and long-horizon planning. This mapping is motivation, not
evidence that any direction has been achieved.

#### Formal task family and notation

Let an environment be a partially observed controlled process

\[
\mathcal E=(\mathcal S,\mathcal O,\mathcal A,T,O,R,D),
\]

where:

- \(s_t\in\mathcal S\) is hidden state;
- \(o_t\in\mathcal O=\prod_{m\in\mathcal M}\mathcal O_m\) is a possibly
  asynchronous multimodal observation;
- \(a_t\in\mathcal A\) is an action;
- \(T(s_{t+1}\mid s_t,a_t)\) is the transition kernel;
- \(O(o_t\mid s_t)\) is the observation kernel;
- \(R\) is task progress or reward;
- \(D\) is termination/failure.

Let \(E\) be a frozen encoder, \(P_\psi\) a trainable projector,
\(x_t=P_\psi(E(o_t),E_g(g_t))\), and \(M_t\) structured persistent memory. A
belief updater defines

\[
q_\theta(b_t\mid b_{t-1},x_t,a_{t-1},r_{t-1},R(M_t)).
\]

The dynamics model predicts

\[
p_\theta(x_{t+1},e_{t+1},r_t,d_t\mid b_t,a_t),
\]

where \(e_{t+1}\in\mathcal E_v\) is a structured verifier-facing event.

Let \(\mathcal P=\{p_i:\mathcal S\to\mathcal Y_i\}\) be exact observable
probes, including task reward, terminal/failure, constraints, and permitted
state or tool/test observations. Define the probe vector

\[
P(s)=(p_i(s))_{i\in I}.
\]

For transition function \(F\), action word
\(\alpha=(a_1,\ldots,a_k)\), and horizon \(H\), define the probe trace

\[
\operatorname{tr}_{P,F}(s,\alpha)
=\bigl(P(s),P(F^{a_1}s),\ldots,P(F^{a_k}\cdots F^{a_1}s)\bigr).
\]

The **finite-horizon verifier equivalence** is

\[
s\equiv_{P,H}s'
\iff
\forall \alpha\in\mathcal A^{\le H},
\operatorname{tr}_{P,F}(s,\alpha)
=\operatorname{tr}_{P,F}(s',\alpha).
\]

The OOD task family holds out environment seeds, visual appearance, action
combinations, mechanics, goals, and complete task families independently. A
successful model must improve real control or interaction efficiency, not only
latent prediction loss.

### B. New Mathematical Construct

#### Candidate formalisms

**Candidate 1 — Monolithic Unified Latent Agent (MULA).** A single frozen or
fine-tuned multimodal sequence model maps history directly to actions and
optionally predicts future tokens. This is expressive and provides a strong
baseline, but it does not expose a clean action-conditioned transition object,
gives weak model-versus-policy attribution, and makes exact counterexample
localization difficult. Rejected as the primary mechanism; retained as a
reactive/generalist control.

**Candidate 2 — Pure Latent World Model (P-LWM).** A stochastic recurrent world
model and latent planner act without an exact observable verifier or structured
revision ledger. This directly tests model-based control, but shared
misspecification can produce low-disagreement, high-confidence model
exploitation. Rejected as the primary mechanism; retained as an ablation.

**Candidate 3 — Verifier-Quotiented Interventional Belief Dynamics (VQ-IBD).**
The surviving formalism stores a posterior over latent state/dynamics while
using \(\equiv_{P,H}\) as the finite planning quotient. It tracks posterior mass
within a quotient class whenever different hidden candidates may split under a
future action. Interventions refine the class; exact observations create
counterexamples and, when needed, probe or representation expansion requests.

#### Definitions

Let \(\Theta\) index candidate learned dynamics and \(Z_t\) index latent state.
The maintained belief is

\[
\mu_t(\theta,z)
=p(\theta,z_t=z\mid o_{\le t},a_{<t},M_t).
\]

Let \([z]_{P,H,\theta}\) be the verifier-equivalence class under candidate
\(\theta\). The planning state is not only a vector; it is

\[
\mathfrak b_t=
\left(
\mu_t,
\Pi_{P,H},
\mathcal C_t,
\mathcal R_t
\right),
\]

where:

- \(\Pi_{P,H}\) is the current quotient partition;
- \(\mathcal C_t\) is the counterexample ledger;
- \(\mathcal R_t\) is the set of unresolved representation/probe obligations.

For candidate action \(a\), define **boundary-split risk**

\[
\rho_t(a)=
\Pr_{(\theta,z),(\theta',z')\sim\mu_t}
\left[
z\equiv_{P,0}z'
\land
\operatorname{tr}_{P,F_\theta}(z,(a))
\ne
\operatorname{tr}_{P,F_{\theta'}}(z',(a))
\right].
\]

This is posterior mass on currently observationally equivalent candidates that
the action may separate. High \(\rho_t(a)\) can mean either danger or
information value. The decision controller therefore evaluates both task and
information utility:

\[
Q_t(a)=
\mathbb E[U_{\mathrm{task}}(a)]
-\lambda_C\Pr(\mathrm{constraint\ violation}\mid a)
-\lambda_E U_{\mathrm{epistemic}}(a)
+\lambda_I I((\Theta,Z);Y_a\mid h_t)
-c(a).
\]

Unsafe distinguishing actions are replaced by questions, observations, or
sandboxed tests. Safe distinguishing actions may be selected for information.

The model objective is

\[
\begin{aligned}
\mathcal L={}&
\mathcal L_{\mathrm{next}}
+\alpha\mathcal L_{\mathrm{multistep}}
+\beta\mathcal L_{\mathrm{reward}}
+\gamma\mathcal L_{\mathrm{terminal}}
+\delta\mathcal L_{\mathrm{inverse}}\\
&+\epsilon\mathcal L_{\mathrm{event}}
+\zeta\mathcal L_{\mathrm{calibration}}
+\eta\mathcal L_{\mathrm{consistency}}
+\xi\mathcal L_{\mathrm{boundary}},
\end{aligned}
\]

where \(\mathcal L_{\mathrm{boundary}}\) penalizes a model that collapses
candidates which are observationally identical now but have verifier-distinct
action effects in branch data. One finite surrogate is a contrastive margin on
branch groups:

\[
\mathcal L_{\mathrm{boundary}}
=\mathbb E_{(h,a,a')}
\left[
\mathbf 1\{P(s_{t+1}^{a})\ne P(s_{t+1}^{a'})\}
\max(0,m-d(\hat z_{t+1}^{a},\hat z_{t+1}^{a'}))
\right].
\]

This surrogate is `HYPOTHESIS`; it does not identify a causal representation
without action coverage and adequate probes.

#### Intuition

- A latent distinction matters only relative to possible action consequences
  and trusted observable probes, not because two vectors differ numerically.
- Quotienting reduces planning state only when it preserves every admitted
  finite-horizon verifier trace.
- Posterior mass inside a quotient class is retained when future actions can
  split the class; current observational equality is not treated as ontological
  identity.
- Branch interventions turn action effects into identifying evidence instead
  of relying on passive temporal correlation.
- The learned model proposes and compresses; exact probes decide, revise, and
  expose missing representation.
- Continuous, discrete, and hybrid parameterizations are implementations of
  the same contract, not assumed cognitive ranks.

### C. Theoretical Results

#### Theorem 1 — Passive-policy intervention non-identifiability

Let \(\mathcal A=\{0,1\}\), \(\mathcal Y=\{0,1\}\), and consider deterministic
models

\[
K_A(y\mid a)=\mathbf 1[y=0],
\qquad
K_B(y\mid a)=\mathbf 1[y=a].
\]

If behavior policy \(\pi\) always chooses \(a=0\), then for every finite
dataset \(D_n=((0,0),\ldots,(0,0))\),

\[
p(D_n\mid K_A)=p(D_n\mid K_B)=1.
\]

Thus any prior with positive mass on both models remains unchanged after any
number of passive observations. Yet under intervention \(do(a=1)\), \(K_A\)
predicts 0 and \(K_B\) predicts 1. The action effect is not identifiable from
the passive policy's support.

**Proof.** For every observed pair, both models assign probability one because
\(K_A(0\mid0)=K_B(0\mid0)=1\). Conditional independence gives likelihood one
for either model and every \(n\); Bayes' rule therefore returns the prior. At
\(a=1\), the deterministic predictions differ by construction. ∎

**Status:** `REPRODUCED — mechanically checked in Lean` for the finite
construction
(`passiveModels_agree_on_observed_action`,
`passiveModels_disagree_on_intervention`, and
`passiveTrace_indistinguishable`). Exact enumeration confirms posterior
0.5/0.5 through 64 passive repeats and 1/0 after one deterministic identifying
intervention.

**Noisy corollary.** Suppose an identifying intervention reports the true
model's distinct outcome independently with reliability \(r>1/2\) and reports
the rival outcome with probability \(1-r\). With equal priors and \(n\)
consistent identifying observations,

\[
p(K_\star\mid D_n)=
\frac{r^n}{r^n+(1-r)^n}.
\]

For \(r=0.9\), two observations exceed posterior 0.95. This concentration
claim depends on correct model class, independence, stationary reliability,
and genuinely identifying interventions.

#### Theorem 2 — Finite-horizon verifier quotient sufficiency

Let \(s\equiv_{P,H}s'\). For any action sequence
\(\alpha\in\mathcal A^{\le H}\) and any plan score
\(J:(\prod_i\mathcal Y_i)^{\le H+1}\to\mathcal V\) depending only on the probe
trace,

\[
J(\operatorname{tr}_{P,F}(s,\alpha))
=J(\operatorname{tr}_{P,F}(s',\alpha)).
\]

**Proof.** Finite-horizon verifier equivalence states that the two trace
arguments are equal for every admissible \(\alpha\). Applying the same function
\(J\) to equal arguments gives equal scores. ∎

**Status:** `REPRODUCED — mechanically checked in Lean` as
`verifierQuotient_sufficient_for_traceScore`.

**Boundary.** The theorem is conditional on \(P\), \(F\), the action set, and
\(H\). It says nothing about longer horizons, missing probes, incorrect learned
dynamics, unmodeled actions, or objectives not measurable from the trace.

#### Lemma 1 — Finite machine latent capacity

If each of \(d\) coordinates has \(2^p\) finite machine values, the latent code
space has

\[
(2^p)^d=2^{pd}
\]

states. Proof is the product rule and exponent law. `REPRODUCED — mechanically
checked in Lean`
as `finitePrecisionLatent_cardinality` and checked by SymPy.

This is not a semantic capacity theorem: context-dependent computations may use
the same code differently. It only rejects literal infinite-state claims for a
fixed finite machine representation.

#### Lemma 2 — Open-loop sequence count

With \(B\) actions at each of \(H\) positions, the number of action words is
\(B^H\). `REPRODUCED — mechanically checked in Lean` as
`actionSequence_cardinality` and checked
by exact enumeration for small cases. For \(B=4,H=25\), the count is
1,125,899,906,842,624.

#### Lemma 3 — Rollout-error accumulation

Let \(e_0=0\) and

\[
e_{h+1}\le\epsilon+L e_h.
\]

Then

\[
e_H\le\epsilon\sum_{i=0}^{H-1}L^i.
\]

For equality recurrence, the formula is mechanically checked in Lean as
`rolloutError_eq_geometric_sum` and symbolically checked. If \(0\le L<1\), the
bound is at most \(\epsilon/(1-L)\); if \(L=1\), it is \(H\epsilon\); if
\(L>1\), it can grow exponentially. A low one-step loss therefore does not by
itself imply stable long rollouts.

#### Lemma 4 — Observable verifier correctness and coverage boundary

An equality verifier rejects every predicted observable unequal to the actual
observable. `REPRODUCED — mechanically checked in Lean` as
`observableMismatch_rejected`. However, two
distinct latent states can have the same value under a non-injective probe;
this is `REPRODUCED — mechanically checked in Lean` by
`noninjectiveProbe_hides_latent_mismatch`. Verifier accuracy and verifier probe
coverage must therefore be separate metrics.

#### Lemma 5 — Objective nonnegativity

When every component loss and coefficient is nonnegative, their finite weighted
sum and finite sum of squared ensemble deviations are nonnegative. Both are
`REPRODUCED — mechanically checked in Lean`. This is an algebraic sanity
condition, not convexity,
identifiability, optimization convergence, calibration, or usefulness.

#### Assumption stress test

| Assumption | If violated | Required diagnostic |
|---|---|---|
| behavior data covers distinguishing actions | causal effect remains non-identifiable | branch/intervention coverage and propensity report |
| verifier probes include task-relevant consequences | quotient merges harmful states | planted unprobed mismatch and coverage metric |
| frozen encoder preserves controllable distinctions | downstream model cannot recover them | random/alternative encoder and inverse-action control |
| candidate dynamics class contains useful mechanism | low disagreement can be confidently wrong | held-out mechanics and explicit inadequacy state |
| rollout sensitivity remains controlled | errors compound with horizon | horizon-conditioned real-versus-imagined divergence |
| frozen matrix matching holds | representation comparison is confounded | actual parameters, data IDs, updates, interactions, probes, seeds, planner calls, wall time |
| persistent memory context is valid | stale mechanics create negative transfer | switch, stale, poisoned, shuffled, reset controls |

### D. Formal Verification Plan

#### Claims sent to Lean first

| Lean theorem | Status | Dependency |
|---|---|---|
| `finitePrecisionLatent_cardinality` | mechanically checked | finite function/cardinality simplification |
| `actionSequence_cardinality` | mechanically checked | finite function/cardinality simplification |
| `verifierQuotient_sufficient_for_traceScore` | mechanically checked | definition of probe trace and equivalence |
| `passiveModels_agree_on_observed_action` | mechanically checked | explicit finite kernels |
| `passiveModels_disagree_on_intervention` | mechanically checked | Boolean decision procedure |
| `passiveTrace_indistinguishable` | mechanically checked | observed-action agreement |
| `weightedLoss_nonnegative` | mechanically checked | nonnegative products and finite sum |
| `squaredDisagreement_nonnegative` | mechanically checked | square nonnegativity and finite sum |
| `rolloutError_eq_geometric_sum` | mechanically checked | induction, finite sums, ring normalization |
| `observableMismatch_rejected` | mechanically checked | definition of exact acceptance |
| `noninjectiveProbe_hides_latent_mismatch` | mechanically checked | explicit Boolean/Unit counterexample |

The file compiles with Lean 4.34.0-rc2 and Mathlib without `sorry` or admitted
proofs.

#### Lean signatures

```lean
theorem finitePrecisionLatent_cardinality (dimension bits : ℕ) :
  Fintype.card (Fin dimension → Fin (2 ^ bits)) = 2 ^ (bits * dimension)

def VerifierEquivalent ... (horizon : ℕ) (left right : State) : Prop := ...

theorem verifierQuotient_sufficient_for_traceScore ... :
  score (probeTrace transition probe left actions) =
  score (probeTrace transition probe right actions)

theorem passiveModels_disagree_on_intervention :
  passiveKernelA true ≠ passiveKernelB true

theorem rolloutError_eq_geometric_sum ... :
  rolloutError lipschitz epsilon horizon =
  epsilon * ∑ i ∈ Finset.range horizon, lipschitz ^ i
```

#### Proof dependency graph

```text
finite product cardinality -> finite latent cardinality
finite function cardinality -> B^H action words

probeTrace -> VerifierEquivalent -> trace-score sufficiency

explicit kernels -> observed agreement -> passive-trace equality
                 -> intervention disagreement

nonnegative square/product -> finite sum -> loss/disagreement nonnegativity

rollout recurrence -> induction -> geometric finite-sum identity

observable equality -> mismatch rejection
constant probe + unequal Bool states -> hidden latent mismatch counterexample
```

#### Deferred formalization

- Bayesian posterior concentration under bounded noise;
- quotient refinement monotonicity when probes are added;
- calibrated safe-commitment risk bounds;
- planner correctness under approximate dynamics;
- relationship to probabilistic bisimulation metrics;
- locality of world-model and memory revision.

Coq/Rocq and Isabelle were unavailable. No secondary proof-assistant result is
claimed.

### E. Mechanism/Architecture Instantiation

#### Computational graph

```text
o_t, g_t
  -> frozen E and E_g
  -> trainable projector P_psi
  -> latent observation x_t
  -> recurrent stochastic belief mu_t(theta,z)
  -> action-conditioned predictions for candidate actions
  -> probe-trace distributions and boundary-split risk
  -> planner / information-action controller
  -> exact authority and verifier bridge
  -> real action
  -> observed probes and counterexamples
  -> posterior, representation-obligation, and structured-memory update
```

#### Pseudocode

```text
initialize persistent structured memory M
initialize belief over latent state and dynamics mu

for each environment step t:
    observation <- environment.observe()
    encoded <- frozen_encoder(observation, encoder_identity)
    x <- projector(encoded)
    memories <- retrieve_candidates(M, x, budget)
    admissible <- scope_provenance_filter(memories)
    mu <- belief_update(mu, x, previous_action, previous_outcome, admissible)

    for action in available_actions:
        prediction[action] <- dynamics(mu, action)
        split_risk[action] <- verifier_boundary_split(mu, action, probes, H)
        task_value[action] <- rollout_value(prediction[action], H)
        info_value[action] <- expected_partition_refinement(mu, action)
        safe_value[action] <- task_value + info_value
                              - uncertainty_cost - constraint_cost

    decision <- act_ask_test_abstain(safe_value, costs, authority)

    if decision is external action:
        verifier.authorize(decision, prediction[decision])
        actual <- environment.step(decision)
        result <- verifier.compare(prediction[decision], actual)
        counterexamples <- result.counterexamples
        mu <- posterior_or_model_class_update(mu, counterexamples)
        M <- provenance_preserving_memory_update(M, actual, counterexamples)
        if mismatch not expressible by current probes/representation:
            add MISSING_REPRESENTATION obligation
    else:
        execute clarification, observation, sandboxed test, or abstention
```

#### Complexity

Let \(C_E\) be encoder cost, \(C_B\) belief-update cost, \(C_F\) one dynamics
call, \(C_V\) verifier cost, \(A=|\mathcal A|\), rollout horizon \(H\), and
planner expansion count \(N\).

- one real step encoding/update: \(O(C_E+C_B)\);
- exhaustive open-loop rollout: \(O(A^H H C_F)\);
- beam width \(W\): \(O(WAHC_F)\) before deduplication details;
- CEM with \(K\) samples and \(I\) iterations: \(O(IKHC_F)\);
- exact probe verification: \(O(C_V)\), domain dependent;
- ensemble of \(K_e\) models multiplies dynamics cost by \(K_e\);
- latent cache payload for \(N_o\) fp16 vectors of dimension \(d\):
  \(2N_od\) bytes before metadata;
- exhaustive quotient construction is generally intractable; the
  implementation approximates relevant splits using posterior samples,
  branch data, and admitted action proposals.

Training complexity is architecture dependent and is reported in actual model
calls, optimizer steps, transitions, wall time, and measured memory. Nominal
parameter count is insufficient.

### F. Empirical Falsification Plan

#### Executed minimal checks

1. **Passive versus intervention enumeration.** Two deterministic kernels stay
   at posterior 0.5/0.5 through 64 passive observations; one distinguishing
   intervention gives posterior 1/0. `MEASURED`, exact toy.
2. **Bounded-noise concentration.** At reliability 0.9, exact enumeration gives
   posterior 0.9878 after two consistent identifying interventions. `MEASURED`,
   model-correct independent-noise toy.
3. **Action-conditioned collision.** The same zero state with actions -1 and +1
   has targets -1 and +1. A JAX affine action-conditioned predictor has MSE 0;
   the best action-blind constant has MSE 1. `MEASURED`, two-transition toy.
4. **Belief-history alias.** The same current observation and action arise from
   histories -1 and +1 and require successors -1 and +1. A history-conditioned
   affine predictor has MSE 0; the best observation-only constant has MSE 1.
   `MEASURED`, separate two-transition toy.
5. **Property checks.** Six helper properties execute 200 derandomized Hypothesis
   cases each. `MEASURED`, finite code properties.
6. **Resource diagnostics.** The frozen matrix arithmetic resolves to 12 cells,
   36 primary runs, 12 dimension-control runs, and 48 workloads. SciPy recovers
   a synthetic rollout sensitivity;
   CVXPy checks a fixed 200M allocation scaffold; NetworkX checks all proposal
   paths cross the verifier; NumPyro represents the finite posterior; MLflow
   logs locally. `MEASURED`, tooling only.

None is a world-model capability experiment.

#### Minimal synthetic tasks for Scale 0–1

**T1a — Action intervention.** Restore the identical full simulator state,
force two different actions, and require different verifier-visible successors.
Tests action conditioning without hidden-state aliasing.

**T1b — Belief aliasing.** Produce the same current observation from different
hidden histories, apply the same action, and require different
verifier-visible successors. Tests recurrent belief/history sufficiency without
changing the action.

**T2 — Passive confounding.** Collection policy omits the distinguishing
action. An intervention split adds it. Tests posterior non-identifiability and
safe information seeking.

**T3 — Probe insufficiency.** Two hidden states match every current probe but
diverge under a held-out safety-relevant event. Tests coverage reporting and
`MISSING_REPRESENTATION` rather than confident merge.

**T4 — Rollout instability.** One-step errors are matched while transition
sensitivity changes across families. Tests horizon-conditioned divergence and
planner exploitation.

**T5 — Surface/mechanism swap.** Appearance changes with mechanics fixed, then
mechanics change with appearance fixed. Tests representation and causal claims.

**T6 — Memory context switch.** A previously valid mechanic becomes invalid
after an explicit boundary. Tests stale-memory containment and revision.

#### Metrics tied to theory

| Prediction | Metric |
|---|---|
| passive support cannot identify untried effects | posterior entropy / equivalence-class mass before and after intervention |
| action conditioning preserves controllable distinctions | action-effect classification and paired successor prediction |
| quotient is sufficient only for admitted probes/horizon | task value equality inside quotient plus planted outside-probe divergence |
| rollout error accumulates with sensitivity | real-versus-imagined divergence by horizon and fitted \(L\) |
| exact verifier detects observable mismatch | detection precision/recall and independent probe coverage |
| large latent is not automatically better | planning/transfer/calibration versus dimension at equal trainable budget |

#### Exact plots generated

- `training-state-memory-lower-bound.png`;
- `latent-cache-footprint.png`;
- `planning-sequence-growth.png`;
- `rollout-error-stability-map.png`;
- `intervention-identifiability.png`.

#### Conditions expected to fail

- no-action model on T1;
- passive-only collection on T2;
- equality verifier with missing probes on T3;
- low one-step-loss model with \(L>1\) on long T4 rollouts;
- correlational model on mechanics interventions in T5;
- unscoped memory on T6;
- monolithic/backbone-only control may match SHWM on tasks already encoded by
  pretraining;
- hybrid representation may lose to simpler arms because its optimization and
  duplicated-state cost are real.

### G. Comparison to Existing Methods

#### Closest prior methods

- Dreamer/RSSM: recurrent stochastic latent dynamics and imagined control.
- V-JEPA 2: continuous predictive video representation plus a smaller
  action-conditioned post-trained model for robot planning.
- world models: recurrent latent environment simulation.
- Gato and multimodal generalist policies: one network across heterogeneous
  modalities and actions.
- causal representation and model-based RL: interventions, mechanism shift,
  and policy-shift robustness.
- bisimulation and state-abstraction methods: merge states that preserve
  reward/transition behavior.
- active learning and experimental design: information-gain action/query
  selection.
- external/episodic/semantic/procedural memory and program libraries.
- exact program verification, runtime monitoring, and CEGIS.

The detailed primary-source audit is in `prior-art-audit.md`.

#### Formal comparison

| Method | State | Action-conditioned | Exact external verifier | Explicit inadequacy | Structured persistent revision | Finite-horizon probe quotient |
|---|---|---:|---:|---:|---:|---:|
| reactive VLM/policy | implicit context | sometimes | no | usually no | usually no | no |
| Dreamer-style | recurrent stochastic latent | yes | no | uncertainty-dependent | no | no |
| V-JEPA 2-AC | continuous predictive latent | yes | task success feedback | not as Sentinel state | no | no |
| symbolic Sentinel | executable hypotheses | yes, explicit | yes | yes | yes | behavioral equivalence where finite |
| SHWM / VQ-IBD | continuous/discrete/hybrid posterior | yes | yes, on observables | separate state | VDFM contract | yes, conditional/approximate in implementation |

#### Expressivity, efficiency, and robustness deltas

**Expressivity hypothesis:** frozen encoders plus learned latent dynamics extend
the observation space beyond hand-authored symbols. This has not been measured.

**Efficiency hypothesis:** verifier quotienting and reusable memory reduce
planning/relearning cost. Exact quotient construction can itself be
intractable; approximation cost must be measured.

**Robustness hypothesis:** observable counterexamples and explicit inadequacy
reduce model exploitation. Missing probes and correlated ensembles remain
strong counterexamples.

#### Novelty delta

The component ideas—latent world models, frozen encoders, discrete or continuous
state, active interventions, verification, structured memory, and planning—are
not new. The bounded possible contribution is the measured integration of:

1. posterior latent action dynamics;
2. finite-horizon equivalence defined by exact verifier probe traces;
3. boundary-split risk for act/test/ask selection;
4. exact counterexamples that update both model belief and representation
   obligations;
5. provenance-scoped continual memory;
6. a frozen ablation protocol separating inherited perception, prediction,
   planning, verification, and memory.

This is `HYPOTHESIS: potentially distinctive integration`, not established
novelty. Bisimulation, shielded model-based RL, runtime assurance, active world
model learning, and neuro-symbolic agents are the strongest collision areas.

### H. Failure Modes & Boundary Conditions

#### Strongest objections

1. **Integration may add no capability.** A strong frozen VLM or Dreamer-style
   baseline may match SHWM. Required response: backbone-only and matched world
   model ablations.
2. **The quotient may be vacuous or expensive.** Full action/probe trace
   equivalence is exponential. Required response: measure approximate split
   recall, cost, and missed harmful merges.
3. **Verifier coverage can be inadequate.** Exactness on the wrong probes is
   not safety. Required response: planted unprobed consequences and coverage.
4. **Interventions may still not identify mechanisms.** Hidden confounding,
   model symmetry, and insufficient actions remain. Required response: report
   equivalence classes and unknown state.
5. **Pretraining can dominate.** The architecture may receive most capability
   from inherited data. Required response: frozen-backbone controls, random
   encoder, contamination disclosure.

#### Adversarial cases

- model exploitation where imagined reward rises in unsupported state regions;
- all ensemble members share the same misspecification;
- event decoder leaks reward/evaluator labels;
- branch sibling enters another split;
- delayed consequence lies beyond \(H\);
- irreversible action is informative but unsafe;
- adversary creates visually similar stale memory;
- backbone precision/conversion silently changes cached features;
- planner receives more calls in one representation arm;
- abstention maximizes apparent accuracy at zero coverage.

#### Identifiability boundaries

- dynamics are identifiable only up to observational/interventional symmetry
  under the admitted actions and probes;
- state coordinates are not semantically unique;
- a finite quotient is horizon and task dependent;
- omitted actions cannot identify their consequences without structural priors;
- open-world inadequacy cannot be calibrated solely from in-class posterior;
- causal direction may remain unidentified from action/outcome data with hidden
  variables and policy selection.

#### Optimization pathologies

- posterior collapse or deterministic belief collapse;
- categorical codebook collapse;
- hybrid continuous/discrete disagreement;
- multi-objective gradient conflict;
- compounding rollout error;
- uncertainty head trained to mirror error only in-distribution;
- planner overfits model defects;
- inverse-action objective preserves nuisance action cues;
- event head becomes a shortcut rather than grounded abstraction;
- memory retrieval changes the belief distribution discontinuously.

#### Disconfirmation criteria

Reject or materially revise SHWM if matched experiments show no control or
interaction benefit, action interventions do not improve held-out mechanism
prediction, verification fails to reduce exploitation, the quotient misses
harmful distinctions at unacceptable rate/cost, or shared-core transfer
disappears under new task families.

### I. Iteration Step

#### Weakest assumption

The weakest assumption is that a small authored probe/event set is sufficient
to define planning-relevant equivalence in high-dimensional worlds. If the
probe set misses a consequential variable, exact quotienting can certify the
wrong merge. Expanding probes manually recreates the representation-authoring
problem Phase 2 is meant to reduce.

#### Next-generation variant

The next theoretical variant is **Counterexample-Induced Probe Expansion
(CIPE)**:

1. retain pairs of latent histories that current probes merge;
2. detect when a later trusted outcome makes their continuation values differ;
3. search for the smallest observable event/probe program separating the
   histories;
4. validate the probe on held-out branch groups and negative controls;
5. add it provisionally to \(P\), recompute affected quotient regions, and
   preserve provenance;
6. promote only if planning or calibration improves without leakage.

A first CIPE theorem target is monotone partition refinement:
if \(P\subseteq P'\), every \(\equiv_{P',H}\) class is contained in an
\(\equiv_{P,H}\) class. This is mathematically straightforward; the hard and
empirical question is whether a generated probe is observable, non-leaking,
compact, and useful.

Do not implement CIPE before Scale 0 establishes the static probe contract and
before planted missing-probe cases demonstrate that expansion is actually
needed.
