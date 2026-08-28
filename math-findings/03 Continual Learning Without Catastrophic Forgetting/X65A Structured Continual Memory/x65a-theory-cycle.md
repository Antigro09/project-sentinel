Sentinel X65 theory cycle: Verified Dependency-Factored Memory (VDFM)

Status: theory and finite falsification package; production implementation is blocked until final X64H passes.  
Date: 2026-08-27  
Evidence boundary: no result in this document demonstrates continual learning in Sentinel. `MEASURED` labels refer only to scripts executed on the deliberately finite model in this package.

### A. Problem Target

#### Direct verdict

The X65 framing is mathematically coherent **provided that task histories share a finite or otherwise compressible latent structure**. The proposed causal chain,

\[
\text{verified prior experience}
\longrightarrow M_t
\longrightarrow \text{better performance on a new task},
\]

is testable with order counterfactuals, reset and shuffled-memory controls, equal compute budgets, literal-target leakage audits, and a real process restart. It is not coherent to demand perfect retention of arbitrary histories under bounded memory; Theorem 1 below proves the finite obstruction. X65 must therefore test whether the *chosen reusable structure* is correct, not whether memory can evade information theory.

The strongest current objection is empirical, not algebraic: a structured store can still be an expensive replay system whose apparent transfer comes from target leakage, task repetition, or extra search compute. The X65A protocol below is designed to make those explanations fail before a continual-learning claim is accepted.

`HYPOTHESIS`: VDFM can provide positive semantic and procedural transfer with bounded growth.  
`FORMALLY PROVED`: finite normalization, predictive sufficiency, a bounded-memory collision result, direct no-target-storage, factorized revision locality, linear raw-replay growth, and a two-part MDL inequality.  
`MEASURED`: exact toy enumeration, symbolic checks, optimization checks, property tests, figures, and a distinct-process reload passed.  
`UNKNOWN`: every X65A capability gate in the real Sentinel substrate.

Final X64H is a hard prerequisite. A failed or unfrozen X64H invalidates the semantic-memory component and blocks X65 implementation.

#### Capability gap

Sentinel already represents hypotheses, programs, uncertainty, clarification, and provenance. What remains absent is evidence that prior *different* tasks improve a later new task while preserving old competence, revising false claims, controlling memory growth, and surviving restart. Saving or retrieving prior records is neither necessary nor sufficient evidence of that causal effect.

This cycle targets five separations:

1. reusable structure versus repeated-answer memorization;
2. capability under a fixed search budget versus an unqualified expressivity claim;
3. useful consolidation versus compression alone;
4. local justified revision versus global catastrophic change;
5. persisted memory versus accidental hidden process state.

#### Formal task family and notation

Tasks arrive sequentially, \(\tau_{1:T}\). Define

\[
\tau_t=(x_t,z_t,b_t,E_t,y_t),\qquad
x_t=(u_t,D_t,e_t,q_t,a_t),
\]

where \(u_t\) is an instruction, \(D_t\) demonstrations, \(e_t\) environment state, \((q_t,a_t)\) clarification events, \(z_t\) a latent typed meaning, \(b_t\) an executable program, \(E_t\) verified evidence, and \(y_t\) an observed outcome. A future target may be inferable from old reusable components, but its literal logical form, program, outputs, convention, and clarification answers may not be stored before observation.

The persistent state is

\[
M_t=(M_t^{\mathrm{epi}},M_t^{\mathrm{sem}},M_t^{\mathrm{proc}},
M_t^{\mathrm{neg}},G_t,Q_t),
\]

where \(G_t\) is a typed provenance/dependency graph and \(Q_t\) is an exact finite posterior or its declared sufficient statistics. The update and retrieval maps are

\[
M_{t+1}=U(M_t,E_t),
\qquad
S_t=R(M_t,x_t,B_R),
\qquad
\operatorname{cost}(M_t)\le B_M.
\]

The task policy may act, clarify, or abstain:

\[
\pi_t\in\mathcal A\cup\{\operatorname{ask},\operatorname{abstain}\}.
\]

The global evaluation functional is

\[
\mathcal J_T=
\sum_{t=1}^T
\left[
\ell_t+
\lambda_M\operatorname{bytes}(M_t)+
\lambda_R\operatorname{cost}(S_t)+
\lambda_Q\operatorname{queries}_t+
\lambda_I\operatorname{interference}_t
\right].
\]

This full functional is **not** optimized online because future task distributions and true interference are unavailable. Online decisions use posterior expected current loss, estimated value of information, declared storage cost, and verifier constraints. \(\mathcal J_T\) is used for preregistered evaluation and hyperparameter selection on development streams only.

### B. New Mathematical Construct

#### Candidate formalisms

| Candidate | State | Main strength | Decisive defect | Decision |
|---|---|---|---|---|
| Posterior Replay Reservoir (PRR) | weighted episodic samples plus a global Bayesian score | simplest exact baseline; clear byte budget | revision locality and procedure composition are not represented; under pressure it degenerates into replay or lossy deletion | rejected as main model; retained as a baseline |
| Contextual Sheaf Revision Memory (CSRM) | local belief sections over overlapping contexts with consistency maps | natural representation of context-dependent truth and convention switches | exact gluing and revision are unnecessarily expensive for the first pilot; operational provenance and executable effects remain underspecified | rejected for X65A; candidate X65B extension |
| **Verified Dependency-Factored Memory (VDFM)** | typed evidence, semantic factors, executable contracts, negative clauses, and a dependency graph | exact finite inference, explicit local revision, compositional procedures, direct leakage audit, and constrained consolidation | guarantees depend on correct factorization, contexts, contracts, and graph edges | surviving candidate |

PRR is rejected because it can pass a retrieval benchmark without establishing abstraction or compounding. CSRM is rejected because its additional mathematical machinery does not buy a more decisive first experiment. VDFM is selected because every claimed benefit has a corresponding control or falsifier in a finite exact model.

#### VDFM memory types

Every memory node has a stable identifier, schema version, validity interval/context, provenance references, byte cost, creation time, supersession state, and cryptographic content hash.

**Episodic evidence** is immutable:

\[
m^{\mathrm{epi}}=(u,D,q,a,b,y,c,p,t,\rho),
\]

where \(c\) contains counterexamples, \(p\) provenance, and \(\rho\) source/replay reliability metadata. An episode is evidence, never truth by declaration.

**Semantic memory** stores contextual probabilistic claims:

\[
m^{\mathrm{sem}}=(h,\theta,\kappa,\mathcal C,\mathcal P,v),
\]

where \(h\) is a typed proposition, \(\theta\) its posterior, \(\kappa\) calibration state, \(\mathcal C\) a validity context, \(\mathcal P\) provenance, and \(v\) a version/supersession chain. The final X64H posterior \(p(\phi\mid H)\), if it passes, is one semantic factor rather than the entire memory system.

**Procedural memory** stores verified contracts:

\[
m^{\mathrm{proc}}=(p,\operatorname{pre},\operatorname{post},
\operatorname{effects},\operatorname{failures},r,\mathcal P,v).
\]

`effects` includes store, stack, register, pointer, memory, and external-state changes. Two programs with equal visible output are not merged unless all continuation-relevant effects are equivalent over the finite verification domain.

**Negative/revision memory** stores scoped defeat information:

\[
m^{\mathrm{neg}}=(h,\neg_{\mathcal C}h,w,\operatorname{reason},
\operatorname{counterexample},\mathcal P,v).
\]

It marks claims rejected, contradicted, stale, or superseded in context \(\mathcal C\). It does not globally ban hypotheses that differ outside that context.

#### Dependency graph

\(G=(V,E)\) is a typed directed graph. Edge labels are

\[
\{\textsf{supports},\textsf{contradicts},\textsf{derives},
\textsf{composes},\textsf{context-gates},\textsf{invalidates}\}.
\]

An edge \(v\to w\) means that changing \(v\) may require recomputing \(w\); it does not mean the two nodes are merely similar. A revision frontier is

\[
\mathcal R(v,\mathcal C)=
\{v\}\cup
\{w:w\text{ is reachable from }v
\text{ through context-compatible dependency edges}\}.
\]

Graph locality is a conditional guarantee: a missing dependency edge can make a revision incorrectly local, while an overly dense graph recreates global instability.

#### Hierarchical finite model

Let the reusable latent state be

\[
\Lambda=(\Phi,\Sigma,\Pi,\Gamma),
\]

where \(\Phi\) contains communication conventions, \(\Sigma\) semantic atoms and schemas, \(\Pi\) procedural contracts, and \(\Gamma\) context and source-reliability variables. For the exact pilot, \(\Lambda\) is finite.

For each task,

\[
c_t\sim p(c_t\mid\Lambda),
\qquad
(z_t,b_t)\sim p(z_t,b_t\mid c_t,\Lambda),
\qquad
\tau_t\sim p(\tau_t\mid z_t,b_t,c_t,\Lambda).
\]

Marginalizing the task composition gives

\[
L_t(\lambda)
=p(\tau_t\mid\lambda)
=\sum_{c,z,b}
p(c\mid\lambda)
p(z,b\mid c,\lambda)
p(\tau_t\mid z,b,c,\lambda).
\]

The exact update is

\[
Q_{t+1}(\lambda)
=
\frac{Q_t(\lambda)L_t(\lambda)}
{\sum_{\lambda'}Q_t(\lambda')L_t(\lambda')}.
\]

For conditionally independent observations in an exponential family, counts or conjugate parameters replace replay. Otherwise the complete finite posterior vector is retained. Neither representation licenses discarding provenance or rare counterexamples needed for revision.

The posterior predictive is

\[
p(\tau_{t+1}\mid H_t)
=\sum_{\lambda}
p(\tau_{t+1}\mid\lambda)Q_t(\lambda).
\]

#### Source reliability and revision

Let \(H\in\{0,1\}\) be a claim, \(R_s\in[0,1]\) source reliability, \(C\) context, and \(O\in\{\textsf{support},\textsf{refute}\}\). For a refuting observation,

\[
p(O=\textsf{refute}\mid H=1,R_s=r)=1-r,
\qquad
p(O=\textsf{refute}\mid H=0,R_s=r)=r.
\]

Then

\[
p(H,R_s,C\mid E)
\propto
p(E\mid H,R_s,C)p(H\mid C)p(R_s)p(C).
\]

A richer fault variable

\[
F\in\{\textsf{old-wrong},\textsf{new-corrupt},
\textsf{context-shift},\textsf{parse-error}\}
\]

prevents every conflict from being interpreted as “the old memory was false.” The exact pilot enumerates \(F\); production approximation is deferred.

Revision is a constrained information projection:

\[
Q'=
\arg\min_{\widetilde Q}
D_{\mathrm{KL}}(\widetilde Q\Vert Q)
+\beta\sum_{v\notin\mathcal R}\mathbf 1[widetilde Q_v\ne Q_v]
\]

subject to the new likelihood, normalization, context constraints, and executable verifier evidence. The immutable evidence is appended; the prior claim becomes superseded rather than erased; affected descendants are invalidated/recomputed; unrelated factors remain unchanged when the factorization assumptions hold.

#### Consolidation

For a two-part computable code,

\[
\mathcal L(M;\tau_{1:T})
=L(M)+\sum_{t=1}^{T}L(\tau_t\mid M).
\]

A proposed consolidation \(M\to M'\) is accepted only if all conditions hold:

\[
\mathcal L(M';H)<\mathcal L(M;H),
\]

\[
\operatorname{VerifierReplay}(M',\mathcal T_{\mathrm{frozen}})
=
\operatorname{VerifierReplay}(M,\mathcal T_{\mathrm{frozen}}),
\]

and required counterexamples, provenance links, contexts, and continuation-relevant effects survive. A short code without preserved or improved future performance is merely compression and fails A6.

#### Retrieval

Let \(\mathcal F_t\) be the dependency-compatible candidate frontier. The main selector solves

\[
S_t^*=\arg\max_{S\subseteq\mathcal F_t,
\operatorname{cost}(S)\le B_R}
\mathbb E[
\Delta\operatorname{taskUtility}
-\lambda_I\operatorname{interference}
-\lambda_C\operatorname{compute}
\mid x_t,S,Q_t].
\]

Because procedures can be complementary and stale claims harmful, this utility is generally neither monotone nor submodular. The exact pilot enumerates all subsets of a small frontier; it does not claim a generic greedy approximation.

#### Open-world status

Retrieval returns both a set and a status

\[
\omega_t\in
\{\textsf{relevant},\textsf{none},\textsf{uncertain},
\textsf{contradicted},\textsf{missing-representation},
\textsf{unknown-task}\}.
\]

`none` means no stored dependency is relevant; `uncertain` means relevant candidates exist with diffuse posterior; `contradicted` means a candidate is defeated in context; `missing-representation` means evidence cannot be expressed by the current semantic/procedural grammar; `unknown-task` means no adequate behavioral candidate exists. These states trigger clarification, expansion, or abstention rather than forced reuse.

#### Intuition

- Episodes are immutable evidence; semantic and procedural nodes are revisable conclusions.
- Reuse occurs through latent factors and executable contracts, not copied final answers.
- Revision follows declared causal dependencies, not vector similarity.
- Consolidation must shorten a computable code *and* preserve frozen behavior and counterexamples.
- Retrieval spends a budget on expected decision value while explicitly charging interference.
- Process persistence contains only serialized, schema-checked memory; runtime caches are treated as adversarial channels.

### C. Theoretical Results

#### Theorem 1 — bounded-memory no-free-lunch

**Statement.** Let arbitrary histories be binary strings \(H\in\{0,1\}^T\). A learner stores \(M=f(H)\) in at most \(B\) bits. A future query may request any coordinate \(H_i\). If every query must be answered correctly for every history, then \(B\ge T\).

**Proof.** If \(B<T\), there are \(2^T\) histories but at most \(2^B\) memory states. By the pigeonhole principle, distinct histories \(h\ne h'\) share a state. They differ at some coordinate \(i\). The learner receives the same state and query \(i\) for both histories, so it returns the same answer and is wrong on at least one. Contradiction. \(\square\)

**Average-error refinement.** For uniform independent history bits and a uniform index query, let \(e\le 1/2\) be optimal average bit error. Since \(I(H;M)\le B\),

\[
H(H\mid M)\ge T-B.
\]

Also \(H(H\mid M)\le\sum_iH(H_i\mid M)\), and binary Fano bounds each term by \(h_2(e_i)\). Concavity of binary entropy gives

\[
h_2(e)\ge 1-B/T,
\qquad
e\ge h_2^{-1}(1-B/T).
\]

This is `MATHEMATICALLY DERIVED`, not mechanically checked here. The finite non-injectivity core is `FORMALLY PROVED` in Lean.

**Consequence.** Bounded continual learning requires assumptions such as a finite latent component model, repeated sufficient statistics, lossy task tolerance, or a restricted future query family. VDFM explicitly chooses the first two.

#### Theorem 2 — finite posterior sufficiency

**Statement.** Suppose \(\tau_1,\ldots,\tau_t,\tau_{t+1}\) are conditionally independent given finite \(\Lambda\), with a fixed future channel \(p(\tau_{t+1}\mid\Lambda)\). Then \(Q_t(\lambda)=p(\lambda\mid\tau_{1:t})\) is a predictive sufficient statistic: histories inducing the same \(Q_t\) induce the same distribution for every future task.

**Proof.** By the law of total probability and conditional independence,

\[
p(\tau_{t+1}\mid\tau_{1:t})
=\sum_\lambda
p(\tau_{t+1}\mid\lambda)
p(\lambda\mid\tau_{1:t}).
\]

The right side depends on the history only through \(Q_t\). Equal posteriors therefore give equal future predictions. \(\square\)

This does not say the selected latent family is adequate. A misspecified finite posterior can be a sufficient statistic for the wrong model.

#### Lemma 2.1 — exact Bayesian normalization

For finite \(\Lambda\), normalized prior \(Q\), nonnegative likelihood \(L\), and evidence mass \(Z=\sum_\lambda Q(\lambda)L(\lambda)>0\),

\[
Q'(\lambda)=Q(\lambda)L(\lambda)/Z
\quad\Longrightarrow\quad
\sum_\lambda Q'(\lambda)=1.
\]

The proof is direct division by \(Z\). `FORMALLY PROVED` in Lean and symbolically checked in SymPy.

#### Theorem 3 — revision locality under factorization

**Statement.** Partition latent memory into \((A,B)\). Assume

\[
p(A,B)=p(A)p(B),
\qquad
p(E\mid A,B)=p(E\mid A),
\]

and \(p(E)>0\). Then revision by \(E\) leaves the unrelated marginal unchanged:

\[
p'(B)=p(B).
\]

**Proof.**

\[
\begin{aligned}
p'(a,b)
&=\frac{p(E\mid a)p(a)p(b)}{p(E)},\\
p'(b)
&=p(b)\sum_a\frac{p(E\mid a)p(a)}{p(E)}
=p(b).
\end{aligned}
\]

\(\square\)

`FORMALLY PROVED` in Lean for finite factors. The graph-update function also mechanically preserves the untouched component. If \(A\) and \(B\) are correlated or evidence depends on both, propagation to \(B\) may be correct; locality cannot be demanded unconditionally.

#### Theorem 4 — raw replay growth and an MDL consolidation threshold

**Statement A.** If each of \(T\) stored episodes has serialized size at least \(s_{\min}>0\), raw replay size satisfies

\[
G_{\mathrm{raw}}(T)=\sum_{t=1}^Ts_t\ge Ts_{\min}.
\]

This is `FORMALLY PROVED` in Lean for \(s_{\min}=1\); scaling gives the general result.

**Statement B.** Suppose each of \(n\) episodes costs \(L_r\) raw units. A shared component costs \(L_c\) once and leaves residual cost \(L_\epsilon\) per episode. Consolidation shortens the two-part code exactly when

\[
L_c<n(L_r-L_\epsilon).
\]

**Proof.** Compare \(nL_r\) with \(L_c+nL_\epsilon\) and rearrange. \(\square\)

The inequality is `FORMALLY PROVED` in Lean and checked by SymPy. It guarantees only compression. Behavioral preservation and future transfer are separate empirical constraints.

#### Theorem 5 — when retrieval utility is submodular, and why VDFM is not generally covered

**Restricted statement.** Let memories independently cover useful features \(j\), with weight \(w_j\ge0\) and coverage probability \(p_{mj}\in[0,1]\):

\[
U_{\mathrm{cov}}(S)=
\sum_jw_j\left(1-\prod_{m\in S}(1-p_{mj})\right).
\]

Then \(U_{\mathrm{cov}}\) is nonnegative, monotone, and submodular.

**Proof.** The marginal value of adding \(e\notin S\) is

\[
\Delta_e(S)=
\sum_jw_jp_{ej}\prod_{m\in S}(1-p_{mj})\ge0.
\]

If \(A\subseteq B\), every product for \(B\) is no larger than its product for \(A\), so \(\Delta_e(A)\ge\Delta_e(B)\). \(\square\)

Under a cardinality budget, standard greedy therefore has the \(1-1/e\) guarantee. Under arbitrary costs, the corresponding \(1-1/e\) result requires a modified partial-enumeration/knapsack algorithm; plain value-per-cost greedy is not granted that guarantee.

**Counterexample for procedural synergy.** Let \(U(S)=1\) iff \(\{a,b\}\subseteq S\), otherwise \(0\). Then

\[
\Delta_b(\varnothing)=0<1=\Delta_b(\{a\}),
\]

which violates diminishing returns. **Counterexample for monotonicity.** Add stale memory \(s\) with penalty \(3/5\); then \(U(\{a,b,s\})=2/5<U(\{a,b\})=1\). Both were `NUMERICALLY VERIFIED` by exhaustive subset enumeration. Main VDFM retrieval therefore uses exact bounded-frontier search in X65A and claims no generic greedy bound.

#### Theorem 6 — procedural compounding under a fixed resource bound

Let a complete enumerator have effective branching factor \(d>1\). Let \(N_d(L)=\sum_{i=0}^{L}d^i\) be candidates through raw description length \(L\). Verified procedures compress a new, unstored composite target to macro length \(L'<L\). If

\[
N_d(L')\le B<N_d(L),
\]

then an exhaustive enumerator ordered by description length reaches the macro representation within budget \(B\) but does not reach the raw representation. A larger-budget memoryless enumerator still can, so this is **capability under a resource bound**, not increased computability or absolute expressivity.

The finite script checks the depth-layer analogue with \(d=6\), \(L=8\), \(L'=3\), and \(B=1000\): \(6^3=216\le1000<6^8=1{,}679{,}616\), while the later composite program is absent from memory.

#### Corollary — stability–plasticity separation

Under Theorem 3's factorization, arbitrarily strong evidence about \(A\) can produce plasticity in \(A\) with zero marginal change in \(B\). When factors are coupled, the Pareto frontier depends on coupling strength. Therefore stability and plasticity should be reported as a vector, not collapsed into a single average.

#### Assumption stress test

| Assumption | If false | Required diagnostic |
|---|---|---|
| finite/compressible reusable \(\Lambda\) | posterior grows with history or discards predictive detail | raw replay and oracle-latent controls; posterior predictive residuals |
| correct conditional independences | “local” revision leaves stale dependents or changes unrelated nodes | planted cross-factor dependencies and graph-edge ablations |
| calibrated source reliability | poison is accepted or true correction is rejected | reliability swaps, corrupt-source controls, calibration curves |
| complete procedural contracts | output-equivalent skills are merged despite hidden-state differences | continuation tests over stack/store/register/pointer effects |
| sound verifier on the finite domain | bad abstractions pass consolidation | trusted-interpreter replay and adversarial counterexamples |
| budget equality across arms | memory advantage is extra compute | instruction-count, candidate-expansion, wall-time, and byte accounting |
| no future leakage | transfer is memorized answer recovery | pre-task memory snapshots and literal/structural canaries |
| task dependencies are real | order effect is manufactured or absent | reverse, random, no-reuse, and dependency-shuffled streams |

### D. Formal Verification Plan

#### Checked claims

The executable Lean file is `formal/X65A.lean`. Lean 4.34.0-rc2 with Mathlib compiled it successfully. The following claims are `FORMALLY PROVED`:

1. `boundedMemory_not_injective`: smaller finite memory cannot injectively encode all histories.
2. `noDirectFutureTarget`: a past-only store does not literally contain a current/future indexed target.
3. `posterior_sum_one`: finite Bayesian normalization.
4. `independentRevisionLeavesRightMarginal`: factorized left-only evidence preserves the right marginal.
5. `reviseLeft_preserves_right`: executable component-local update leaves the other component unchanged.
6. `finitePosterior_isSufficient`: equal finite posteriors induce equal posterior predictives.
7. `rawReplay_linearGrowth`: nonempty raw episodes require linear storage.
8. `reusableComponent_reducesDescriptionLength`: the two-part MDL threshold implies a shorter code.

The direct no-leakage theorem proves **absence of literal storage**, not statistical independence of a future answer from memory. Legitimate compositional transfer should make some future answers inferable; claiming otherwise would contradict the goal.

#### Lean-compatible signatures

```lean
theorem boundedMemory_not_injective
    [Fintype History] [Fintype Memory]
    (hcard : Fintype.card Memory < Fintype.card History)
    (encode : History → Memory) :
    ¬ Function.Injective encode

theorem noDirectFutureTarget
    (hclean : NoFutureTargets memory t) (hfuture : t ≤ t') :
    (t', answer) ∉ memory

theorem posterior_sum_one
    (hpositive : evidenceMass prior likelihood ≠ 0) :
    ∑ latent, posterior prior likelihood latent = 1

theorem independentRevisionLeavesRightMarginal
    (hpositive : evidenceMass leftPrior leftEvidence ≠ 0) :
    ∑ left, factorizedRevision leftPrior leftEvidence rightBelief (left, right)
      = rightBelief right

theorem finitePosterior_isSufficient
    (hsame : first = second) :
    posteriorPredictive first channel = posteriorPredictive second channel

theorem rawReplay_linearGrowth
    (hnonempty : ∀ task, 1 ≤ episodeSize task) :
    taskCount ≤ ∑ task, episodeSize task

theorem reusableComponent_reducesDescriptionLength
    (hsaving : componentCost < repetitions * (rawCost - residualCost)) :
    componentCost + repetitions * residualCost < repetitions * rawCost
```

#### Proof dependency graph

```text
finite sums + nonzero evidence
        └── posterior_sum_one
              ├── independentRevisionLeavesRightMarginal
              └── posterior predictive definition
                         └── finitePosterior_isSufficient

finite cardinality inequality
        └── boundedMemory_not_injective

past-only invariant
        └── noDirectFutureTarget

per-episode positive size
        └── rawReplay_linearGrowth

two-part code algebra
        └── reusableComponent_reducesDescriptionLength
```

#### Remaining formal targets

- Prove a typed dependency-closure update preserves every node outside the closure.
- Formalize finite-context procedural contract composition and continuation-effect preservation.
- Connect the history-level conditional-independence model to the posterior-sufficiency theorem, rather than assuming equal posterior vectors at the theorem boundary.
- Formalize the independent-coverage submodularity theorem and the complementary-memory counterexample.
- Formalize serialization round-trip invariance over a schema-defined memory record.

Coq and Isabelle/HOL were not installed and were not needed to resolve a disputed proof. This is an explicit limitation, not a secondary verification result.

#### Evidence classes

| Class | Claims |
|---|---|
| `FORMALLY PROVED` | eight Lean claims listed above |
| `MATHEMATICALLY DERIVED, UNCHECKED` | entropy error bound; KL minimal-change characterization; standard greedy corollaries |
| `MEASURED/NUMERICALLY VERIFIED` | exact finite posterior curve; subset counterexamples; graph locality construction; search-budget phase; byte-growth code; process restart; 1,500 Hypothesis examples |
| `HYPOTHESIS` | VDFM improves Sentinel transfer, retention, revision, retrieval, or growth |
| `UNKNOWN` | all X65A gates and broader-domain behavior |

### E. Mechanism/Architecture Instantiation

#### Computational mechanism

```text
current task x_t
   │
   ├── open-world/type check ──> missing/unknown? ──> clarify or abstain
   │
   └── dependency frontier from G_t
           │
           └── exact budgeted subset valuation
                    │
                    └── semantic posterior + procedural search
                              │
                              └── act / ask / abstain
                                       │
                               trusted execution verifier
                                       │
                         immutable episodic evidence append
                                       │
                  reliability update + graph-local belief revision
                                       │
                    verifier-constrained MDL consolidation
                                       │
                        schema-check, hash, atomic persist
```

The proposal does not replace Sentinel's executable hypotheses, exact replay/refutation, uncertainty states, controlled semantics, or evidence-authoritative execution policy. It adds a typed persistent memory layer and the causal evaluation needed to establish whether that layer changes later competence.

#### Exact online pseudocode

```text
INPUT: frozen model family, memory M_t, task x_t, byte budget B_M,
       retrieval budget B_R, compute budget B_C

1  validate_schema_and_hashes(M_t)
2  omega, frontier <- dependency_frontier(M_t.G, x_t)
3  if omega in {missing-representation, unknown-task}:
       return EXPAND_OR_ABSTAIN without an in-class memory write
4  for each S subset frontier satisfying cost(S) <= B_R:
       Q_S <- exact_component_posterior(M_t.Q, x_t, S)
       value[S] <- expected_decision_value(Q_S, x_t)
                   - interference_risk(S) - compute_charge(S)
5  S_star <- deterministic_argmax(value, tie_break_by_content_hash)
6  candidates <- bounded_program_search(x_t, S_star.procedures, B_C)
7  decision <- Bayes_action_or_clarification_or_abstention(candidates, Q_S_star)
8  outcome, trace <- trusted_execute_or_query(decision)
9  evidence <- verify_and_type(x_t, decision, outcome, trace)
10 append immutable episodic evidence and provenance
11 infer fault/source reliability variables exactly
12 changed <- revise_semantic_and_negative_factors(evidence)
13 affected <- context_compatible_descendant_closure(M_t.G, changed)
14 invalidate and recompute only affected nodes; preserve old versions
15 if a reusable procedure is proposed:
       require typed pre/post/effects/failure contract and verifier evidence
16 for each deterministic consolidation proposal in canonical order:
       accept only if MDL decreases, frozen replay is unchanged,
       held-out development transfer is not degraded, and provenance survives
17 evict only after value-per-byte audit; never evict unique counterexamples
18 atomically serialize schema-defined M_{t+1}; fsync; reopen and hash-check
19 emit an audit event containing budgets, retrieved IDs, posterior hash,
       candidate count, decision, evidence IDs, revisions, and memory bytes
```

Online step 4 is exact in X65A. Learned or approximate retrieval is not introduced until exact inference provides an oracle diagnostic.

#### Decision rule

For action \(a\), ask action \(q\), and abstention \(\bot\), choose

\[
d_t^*=\arg\min_{d\in\mathcal A\cup\mathcal Q\cup\{\bot\}}
\mathbb E[\ell(d,Y_t)\mid x_t,S_t,Q_t]+c(d).
\]

A memory can influence execution only through this auditable posterior/decision path. Retrieved text cannot bypass the evidence-authoritative verifier.

#### Complexity

Let \(K=|\Lambda|\), \(C_t\) the number of task compositions, \(r=|\mathcal F_t|\), \(k\) the maximum retrieved entries, \(V,E\) graph sizes, \(a=|\mathcal R|\), and \(P_B\) bounded program candidates.

| Operation | X65A time | Space | Boundary |
|---|---:|---:|---|
| exact posterior update | \(O(KC_t)\) | \(O(K)\) | exponential if latent family is represented naively; keep \(K\le256\) in pilot |
| exact retrieval subsets | \(O(\sum_{i=0}^k {r\choose i}\,V_{\text{eval}})\) | \(O(r+K)\) | use \(r\le20,k\le4\) in pilot |
| dependency frontier/closure | \(O(V+E)\) | \(O(V)\) | graph edges must be semantically typed |
| local revision | \(O(aK_a)\) | \(O(aK_a)\) | can approach global cost if graph is dense |
| bounded program search | \(O(P_B)\) | implementation-dependent | identical \(B_C\) across arms |
| one consolidation proposal | replay cost plus code computation | proposal plus audit log | program equivalence is only finite-domain checked |
| serialization | \(O(\operatorname{bytes}(M))\) | same | atomic write plus reopen/hash check |

Exact inference is deliberately preferred for the first pilot. If it fails, the model or stream is wrong; approximate inference cannot be blamed.

### F. Empirical Falsification Plan

#### X65A — Structured Continual Memory

**Prerequisite and freeze.** Do not implement or run X65A until final X64H passes all frozen gates. Then freeze and hash X64H inference, program substrate, task grammar, candidate pools, evaluators, VDFM schemas, exact posterior, retrieval objective, consolidation rules, revision model, budgets, stream generator, metrics, gates, and train/validation generation. Final stream seeds are sampled only after the freeze commit and must never cause edits to these components.

#### Smallest decisive exact pilot

Use a finite latent family with at most \(K=256\) joint states:

- two persistent convention bits inherited from frozen X64H;
- two reusable semantic/context bits;
- four verified primitive procedures with complete finite-state contracts;
- one source-reliability variable and one explicit context-switch variable;
- a typed DSL with six primitive productions and two permitted macro slots;
- 28 tasks per stream: 4 semantic grounding, 4 procedure learning, 6 novel semantic compositions, 6 new procedural compositions, 2 retention probes, 2 revision events, 2 interference/no-reuse controls, and 2 convention-switch/unknown cases;
- the later composite targets are generated post-freeze and never appear as stored programs;
- main budget: 4 KiB serialized memory, at most four retrieved nodes/512 bytes, and 1,000 candidate expansions per task;
- 40 untouched final stream seeds under each of four frozen meta-seeds, paired across all arms.

The exact sizes may be reduced only if runtime/memory profiling shows exact inference is the measured bottleneck; reductions and reasons must be recorded before final seeds are opened.

#### Stream generator

Each stream carries an explicit generator-internal dependency DAG, hidden from the learner and used only for evaluation/leakage checks.

| Stream | Early tasks | Later task | Causal test |
|---|---|---|---|
| A — semantic transfer | ground convention/context factors | unseen semantic compositions | relevant history versus reset/shuffle |
| B — procedural transfer | verify subprogram contracts | new, unstored macro composition | bounded-search reachability |
| C — retention | learn old families | probe after long interference gap | degradation after stream |
| D — revision | plant plausible false contextual claim | trusted counterexample and later reversal evidence | correction, provenance, collateral damage |
| E — interference | insert similar irrelevant/stale claims | target requiring different factor | negative transfer and poison resistance |
| F — no reuse | independent latent components | unrelated target | memory should have no advantage |
| G — convention switch | one persistent convention | explicit boundary, then new convention | context gating and stale-memory retirement |

Run A→B, B→A, dependency-respecting, random, reverse-dependency, and shuffled-memory orders. A transfer effect that survives reverse-dependency order is suspicious; an effect that vanishes under reset/shuffle but appears only after prerequisite tasks supports the intended causal mechanism.

#### Required arms

All arms receive identical current-task evidence, candidate pools, action/query/search budgets, byte budgets where applicable, stopping rules, and counted compute.

1. no memory;
2. raw full replay;
3. random retrieval;
4. most-recent retrieval;
5. surface-similarity retrieval;
6. exact dependency-oracle retrieval;
7. episodic only;
8. semantic only;
9. procedural only;
10. negative/revision only;
11. semantic plus procedural;
12. all memory types without consolidation;
13. main VDFM with consolidation;
14. VDFM without revision;
15. VDFM without provenance;
16. shuffled memory;
17. stale memory;
18. poisoned memory;
19. oracle relevance;
20. unlimited-memory diagnostic.

Raw full replay is subject to the same retrieval and compute budget when it is used as a competitive arm; arm 20 separately diagnoses whether the hard memory budget itself is the bottleneck.

#### Hard budget and restart

At the midpoint:

1. atomically serialize only schema-permitted memory;
2. record its hash and byte count;
3. terminate the process and verify the PID exits;
4. start a clean process with a scrubbed environment and empty runtime caches;
5. reload only the serialized file;
6. verify schema, provenance references, content hashes, and posterior normalization;
7. continue from an externally supplied next-task seed.

Plant a `future_target_answer` value in a runtime-only parent cache and environment variable. The child must not receive it; a canary test must fail if any forbidden key, global random state, full history, future candidate pool, or expected answer survives.

#### Metrics

For paired streams, define

\[
FT=\mathbb E[P(\tau_t\mid H_{\mathrm{relevant}})-P(\tau_t\mid H_{\mathrm{none}})],
\]

\[
RT=\mathbb E[P(\tau_i\text{ after stream})-P(\tau_i\text{ immediately after learning})],
\]

\[
NT=\mathbb E[P(\tau_t\mid H_{\mathrm{irrelevant}})-P(\tau_t\mid H_{\mathrm{none}})].
\]

Also report backward transfer, task accuracy, exact-program/denotation accuracy, log loss, clarification count, candidate expansions, wall time, peak memory, calibration, false-confident acts, abstentions, retrieval precision/recall, irrelevant/stale/poisoned retrieval rate, correction latency, false corrections, collateral posterior total variation, component reuse, program target absence, and

\[
G(T)=\operatorname{bytes}(M_T),
\quad
\frac{G(T)}{T},
\quad
\frac{\Delta\text{performance}}{\operatorname{bytes}(M_T)}.
\]

Report retention and forward transfer as a Pareto plot; do not hide one with their average.

#### Gates

- **A1 — positive semantic transfer.** VDFM's paired semantic log-loss improvement over both no memory and budget-matched raw replay has a 95% hierarchical-bootstrap lower bound above zero on unseen compositions; accuracy/query effects are secondary corroboration.
- **A2 — positive procedural transfer.** Prior verified procedures improve new composite targets absent from memory; target absence and contract composition are audited.
- **A3 — retention.** The lower confidence bound for \(RT\) remains above a validation-frozen degradation margin after the full stream.
- **A4 — revision.** The planted false claim crosses the frozen rejection threshold after trusted counterevidence, its counterexample and supersession chain remain, and unrelated claims stay within a frozen total-variation margin.
- **A5 — interference resistance.** Irrelevant and similar memories do not produce confidence-significant negative transfer or excess false-confident acts relative to no memory.
- **A6 — bounded growth.** The fitted byte-growth slope is materially below raw replay while at least a frozen fraction of raw replay's transfer benefit is retained. Both thresholds are fixed on validation.
- **A7 — retrieval earns its place.** Main retrieval beats random, recency, surface similarity, and shuffled memory under equal budgets.
- **A8 — restart persistence.** Serialized state hashes validate; paired predictive distributions and central effects survive a genuine process restart within deterministic/numerical tolerances.
- **A9 — compounding.** At least one class of new composites is solved with verified macros under the fixed search budget, fails without memory at that budget, and becomes solvable for the larger-budget memoryless control.
- **A10 — no answer leakage.** No future target program, logical form, outputs, convention, or answer is present before observation; automated snapshots and canaries pass.
- **A11 — revision locality.** Correcting one planted claim leaves graph-independent semantic and procedural probes inside the frozen collateral-damage bound.
- **A12 — calibration and defect detection.** Stale, poison, shuffle, irrelevant, context-switch, and hidden-state defects are detected at validation-frozen thresholds.
- **A13 — replication.** A1, A2, A3, A4, A8, and A9 replicate across all four untouched meta-seeds with seed-level effects reported, not pooled away.

No gate threshold may be tuned after final seeds are opened.

#### Statistical protocol

- Unit of randomization and primary resampling: complete stream seed; task-level pairs remain nested inside streams.
- Primary intervals: 95% hierarchical bootstrap over meta-seed, stream seed, then paired target task; at least 10,000 bootstrap replicates.
- Exact binomial intervals for finite success/failure gates and Wilson intervals for rare false-confidence events.
- Calibration: Brier score, negative log likelihood, reliability diagrams, and adaptive ECE with bins frozen on validation.
- Growth: robust slope with stream-level confidence interval; report the entire curve because fixed overhead can reverse early comparisons.
- Multiple primary comparisons: preregister A1/A2 contrasts and use Holm correction; all other arm comparisons are diagnostic unless separately preregistered.
- Report every seed, failures, timeouts, and abstentions. Never condition analysis on successful runs only.
- A task or stream cannot be regenerated after seeing an unfavorable final result.

#### Executed finite checks

The package's DVC pipeline produced these `MEASURED` mechanism checks:

- 16 four-bit histories mapped into eight three-bit states expose a collision separated by an index query; Lean proves this for every smaller finite memory encoder.
- 32 length-five evidence histories collapse to six exact count statistics with identical posteriors within each class.
- Under reliability \(4/5\), exact expected convention-MAP accuracy rises from \(1/2\) with reset memory to \(15104/15625\approx0.9667\) after seven/eight observations; even counts plateau because symmetric ties are possible.
- Complementary retrieval is non-submodular and stale retrieval is nonmonotone; independent weighted coverage passes exhaustive submodularity checks.
- Trusted counterevidence changes a planted claim from \(4/5\) to \(4/13\), later support can reverse it to \(76/85\), and an unrelated factor stays exactly \(9/10\).
- In the six-task toy DAG, both composites solve in dependency order and neither solves when queried before prerequisites.
- The specified toy two-part code has fitted slope 22.38 bytes/task versus 128 for raw replay and is shorter by task five; this says nothing about future utility.
- A distinct child process recovers the exact posterior \((1/5,4/5)\), while the planted forbidden answer channel is absent.
- 1,500 generated Hypothesis cases passed five finite invariants.
- SciPy solved a binary retrieval diagnostic, CVXPy gave its relaxation bound, JAX differentiated the restricted coverage surrogate, NumPyro represented the normalized finite posterior, Hydra loaded the frozen config, MLflow logged the run, and DVC reproduced all stages.

These are not task-level X65A measurements and cannot pass A1–A13.

#### Exact plots

The Matplotlib pipeline generates:

1. `semantic-transfer-curve.png` — exact posterior-predictive gain versus retained observations;
2. `memory-growth.png` — raw replay versus the specified two-part code;
3. `stability-plasticity-frontier.png` — collateral retention versus update strength at different dependency coupling;
4. `retrieval-counterexample.png` — complementary gain and stale-memory harm;
5. `revision-dependency-region.png` — affected versus independent graph nodes;
6. `compounding-reachability-phase.png` — neither/macros-only/both reachability across program length and search budget.

#### Preregistered falsifiers

X65A fails as continual learning if gains come from exact old-answer replay; only repeated tasks improve; budget-matched raw replay ties the main system; old capabilities exceed the degradation margin; false beliefs resist revision; revision damages independent memory; restart removes transfer; memory growth remains approximately one full episode per task; retrieval cannot beat random/recency/surface similarity; relevant memory causes material negative transfer; any arm gets more compute; target/dependency metadata leaks; a final-test failure causes model or generator edits; or X64H did not pass before implementation.

Diagnose failures separately as representation, posterior/model misspecification, consolidation, retrieval, revision, interference, stream construction, budget accounting, restart persistence, or leakage. A failed gate is a useful negative result, not permission to weaken the gate post hoc.

### G. Comparison to Existing Methods

#### Bounded prior-art verdict

No individual VDFM ingredient is new: external memory, episodic/semantic distinctions, replay, Bayesian continual learning, truth maintenance, MDL, program libraries, skill composition, submodular selection, provenance, and restart persistence all have substantial prior art. The bounded audit identifies PlugMem as the closest recent architecture-level collision for typed episodic/semantic/procedural graphs, provenance, consolidation, and budgeted retrieval; Rosenbloom and Soar as older multi-memory integration precedents; DreamCoder as the central procedural-library/MDL collision; and TRUSTMEM/MemGuard as direct threats to verifier-governed consolidation claims. Recent AgentCL and AgentMemoryBench work also overlaps the controlled-stream evaluation. Accordingly, the only defensible novelty hypothesis is a **specific, measured integration** of executable version spaces, exact verifier evidence, latent convention semantics, four typed memory ledgers, graph-local probabilistic revision, verifier-constrained consolidation, procedural compounding, hard restart/leakage tests, and bounded causal evaluation. This audit is bounded and cannot establish novelty.

#### Closest primary sources

- [McClelland, McNaughton & O'Reilly (1995), complementary learning systems](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf): fast episodic acquisition plus slower consolidation; VDFM operationalizes a symbolic finite analogue but does not inherit biological validity.
- [Kirkpatrick et al. (2017), Elastic Weight Consolidation](https://doi.org/10.1073/pnas.1611835114): parameter-level stability through importance-weighted constraints; unlike VDFM's explicit claims, contracts, and provenance.
- [Nguyen et al. (2018), Variational Continual Learning](https://openreview.net/pdf?id=BkQqq0gRb): sequential Bayesian posterior approximation; VDFM's first pilot instead uses exact finite inference and typed revision evidence.
- [Chen, Papadimitriou & Peng (2022), Memory Bounds for Continual Learning](https://arxiv.org/abs/2204.10830): formal lower bounds showing memory can scale linearly with tasks in broad continual-learning settings; supports the need for declared structure rather than contradicting it.
- [Graves, Wayne & Danihelka (2014), Neural Turing Machines](https://arxiv.org/abs/1410.5401) and [Graves et al. (2016), Differentiable Neural Computer](https://www.nature.com/articles/nature20101): differentiable external read/write memory; they do not by themselves establish revision locality, provenance, or causal transfer under a byte budget.
- [Tulving (1972), episodic and semantic memory](https://cir.nii.ac.jp/crid/1574231874408386176?lang=en): cognitive taxonomy precedent; VDFM's engineering types are inspired labels, not a claim of cognitive equivalence.
- [Rosenbloom (2010), Combining Procedural and Declarative Knowledge in a Graphical Architecture](https://ict.usc.edu/pubs/Combining%20Procedural%20and%20Declarative%20Knowledge%20in%20a%20Graphical%20Architecture.pdf), and [Laird & Mohan (2013), knowledge integration across Soar memories](https://cdn.aaai.org/ocs/7606/7606-32587-1-PB.pdf): strong precedents for coordinating procedural, semantic, and episodic memory, including a graphical probabilistic substrate in Rosenbloom.
- [PlugMem (2026)](https://arxiv.org/abs/2603.03296): the closest located recent collision for typed episodic/semantic/procedural graphs, source links, abstraction-aware retrieval, evolution, and compression; it must be treated as a principal component-matched baseline.
- [Doyle (1979), Truth Maintenance System](https://doi.org/10.1016/0004-3702(79)90008-0): dependency-directed belief maintenance and recorded reasons are close precedents for VDFM's graph and supersession logic.
- [de Kleer (1986), Assumption-Based TMS](https://doi.org/10.1016/0004-3702(86)90080-9): assumption sets, contexts, and nogoods closely precede scoped negative memory and dependency-aware invalidation.
- Alchourrón, Gärdenfors & Makinson (1985), *On the Logic of Theory Change*: minimal-change qualitative revision; VDFM changes this to contextual probabilistic factors, source reliability, executable contracts, and immutable provenance.
- [Sutton, Precup & Singh (1999), options](https://doi.org/10.1016/S0004-3702(99)00052-1): formal temporal abstraction and reusable macro-actions; VDFM's procedural compounding is a finite program-synthesis analogue, not a new discovery of skills.
- [Ellis et al. (2020/2023), DreamCoder](https://arxiv.org/abs/2006.08381): Bayesian program induction and learned reusable libraries are the closest precedent for MDL-like procedural abstraction and compounding.
- [Nemhauser & Wolsey (1978)](https://doi.org/10.1287/moor.3.3.177) and [Sviridenko (2004)](https://doi.org/10.1016/S0167-6377(03)00062-2): classical monotone-submodular guarantees; VDFM explicitly demonstrates why its general retrieval utility falls outside their assumptions.
- [AgentCL (2026)](https://arxiv.org/abs/2606.02461): controlled streams with reusable subsolutions/workflows, transfer metrics, unreliable-experience filtering, and memory ablations directly overlap X65A's causal evaluation concept.
- [AgentMemoryBench (2026)](https://openreview.net/pdf?id=MSXbrNExax): joint system/personal memory, online transfer, forgetting, and repair directly threaten any broad claim that multi-memory continual-agent evaluation is new.
- [PLACEMEM (2026)](https://arxiv.org/abs/2607.04089): versioned, provenance-bearing, correction-aware persistent capsules overlap the persistence/revision systems layer, though not VDFM's exact finite theory and executable program-version-space evaluation.
- [TRUSTMEM (2026)](https://arxiv.org/abs/2606.25161) and [MemGuard (2026)](https://arxiv.org/abs/2608.21867): recent verifier-governed consolidation and persisted verifier-signal systems sharply limit claims around trustworthy write/revise/delete operations or durable trust metadata.

The separate `prior-art-audit.md` records a larger source-by-source search and its limitations.

#### Formal comparison

| Method family | Persistent reusable structure | Explicit revision/provenance | Procedural composition | Exact finite posterior/verifier | Hard byte/retrieval budget and clean restart | Main difference from VDFM |
|---|---|---|---|---|---|---|
| replay/reservoir | episodes | usually weak | no | no | sometimes memory budget | VDFM treats episodes as evidence and tests structured compression |
| EWC/regularization | parameter importance | no claim graph | implicit | no | parameter capacity, not retrieval | VDFM is nonparametric and auditable |
| VCL/Bayesian CL | posterior parameters | Bayesian update, usually no provenance graph | implicit | approximate in cited work | not central | VDFM uses exact finite factors and typed evidence |
| CLS | fast/slow systems | consolidation concept | not executable program contracts | no | no | biological/computational principle rather than Sentinel mechanism |
| TMS/AGM | beliefs and reasons | strong qualitative precedent | limited executable effects | logical, not selected Bayesian program model | no restart causal protocol | VDFM adds probabilistic reliability, contexts, contracts, and experiments |
| DreamCoder | learned program library | limited negative/revision memory | strong | Bayesian program search, approximate learning | not the X65 restart/leakage design | closest procedural-consolidation precedent |
| Rosenbloom/Soar | coordinated procedural, semantic, episodic memory | support/integration mechanisms | strong procedural knowledge | graphical or production-system inference | not the X65 causal restart design | blocks any “first integrated multi-memory” claim |
| PlugMem | typed episodic/semantic/procedural memory graphs | source links and memory evolution | procedural abstractions | learned rather than exact finite inference | cross-session, not the complete X65 audit | closest current architecture-level baseline |
| NTM/DNC | differentiable addressable memory | implicit | learned | no exact verifier | capacity yes, causal byte comparison no | storage architecture rather than evidence ledger |
| AgentCL | interactions, insights, skills | reliability filtering | reusable workflows | no Sentinel exact version space | controlled streams, not identical hard protocol | strongest benchmark-level overlap |
| AgentMemoryBench | system/personal stores | repair | procedural system memory | no Sentinel exact verifier | continual benchmark | strongest multi-memory evaluation overlap |
| PLACEMEM | versioned capsules | strong correction/invalidation | runtime reuse | systems prototype | persistence emphasized | strongest correction-aware systems overlap |
| TRUSTMEM/MemGuard | consolidation transitions and verifier metadata | explicit trust/revision signals | limited | verifier-driven, not VDFM finite Bayes | persistence is central | blocks standalone verifier-governance novelty claims |
| **VDFM hypothesis** | four typed ledgers + latent factors | graph-local probabilistic supersession | verified program contracts/macros | yes in finite pilot | both, plus hidden-state canary | unproven integration rather than a new primitive |

#### Claimed deltas, all provisional

- **Expressivity:** explicit context, negative evidence, source reliability, and continuation-relevant procedural effects exceed a simple replay reservoir. This is a representational statement, not measured usefulness.
- **Efficiency:** sufficient statistics and reusable programs can be smaller than raw replay under Theorems 2 and 4, but only if the finite latent model is adequate and repeated structure exists.
- **Robustness:** provenance, supersession, open-world states, and local revision make specific failures detectable; none protect against a wrong graph or verifier.
- **Causal evidence:** dependency order, reset, shuffle, no-reuse, budget equality, and restart make the transfer claim harder to fake. AgentCL shows this direction is not uniquely Sentinel's.

### H. Failure Modes & Boundary Conditions

#### Strongest objections first

1. **The latent family may encode the answer.** A hand-authored component grammar can move human supervision from targets into reusable factors. Audit description length, generator access, and future-target mutual information; call such inputs human-authored.
2. **The dependency DAG may leak the curriculum.** The learner must not receive evaluator-only future edges. Learned dependencies must come from observed provenance; oracle retrieval is a diagnostic arm only.
3. **Raw replay may match VDFM.** If so, the structured architecture has not earned its complexity under that budget, even if both beat no memory.
4. **Exact inference may succeed only because the pilot is tiny.** That establishes a model result, not scalable continual learning. Approximation comes only after the exact oracle separates modeling from inference failures.
5. **Consolidation may preserve old tests while destroying untested behavior.** Finite verifier equivalence does not imply universal program equivalence. Keep failure cases and adversarially expand frozen probes.
6. **Locality can be wrong.** Missing edges cause under-revision; spurious edges cause collateral damage. Plant both and report sensitivity.
7. **Memory selection can require synergy.** Additive relevance scores and ordinary greedy retrieval can miss pairs whose utility appears only jointly.
8. **Stale memories can be actively harmful.** Similarity retrieval is especially vulnerable after context switches; context posteriors and negative memory must gate it.

#### Adversarial cases

- Poison an otherwise reliable source after it earns high \(R_s\).
- Create two procedures with equal visible outputs but different stack/register effects.
- Present an old lexical convention immediately after an explicit switch.
- Make a rare counterexample the only evidence distinguishing two consolidated schemas.
- Split a useful macro across two memories so neither has standalone value.
- Insert a highly similar stale rule with lower description length than the true contextual rule.
- Corrupt a provenance pointer while preserving the content hash of an unrelated node.
- Trigger serialization during a partial transaction or kill between write and rename.
- Supply a task outside the semantic grammar and a program outside the behavioral pool simultaneously.
- Arrange correlated “independent” components to expose false revision locality.

#### Identifiability

Task meaning, convention, semantic schema, and procedure can remain jointly non-identifiable. If two latent states induce the same distribution over every permitted task, action, clarification, and verifier trace, only their observational equivalence class is identifiable. VDFM must retain that class rather than commit to one label. No amount of memory resolves a symmetry that no stream or query separates.

Source reliability and claim truth are also confounded without repeated sources, gold anchors, or differing contexts. A single contradiction cannot identify whether the old claim, new source, context, or parse is at fault.

#### Optimization and systems pathologies

- posterior underflow or zero evidence from model misspecification;
- exact frontier explosion as \(r\) or \(k\) grows;
- cyclic dependency invalidation;
- MDL oscillation between nearly equivalent libraries;
- eviction feedback: low retrieval causes low estimated utility, causing further eviction;
- provenance growth dominating compressed content;
- source-reliability lock-in;
- query policies repeatedly asking easy questions that confirm current memory;
- nondeterministic tie-breaking breaking restart equivalence;
- MLflow/DVC metadata contaminating scientific output hashes;
- process caches, RNG state, or candidate enumeration order surviving restart;
- serialized schema migration silently changing posterior semantics.

#### Boundary conditions

- The no-free-lunch result applies to arbitrary histories and exact future index queries; it does not rule out distributional compression.
- Posterior sufficiency holds only relative to the chosen generative model.
- Revision locality holds only under the declared factorization/evidence assumptions.
- The MDL inequality says nothing about transfer without behavioral constraints.
- Greedy retrieval guarantees apply only to nonnegative monotone submodular utility under the specified budget algorithm.
- Procedural compounding is a resource-bounded search result, not a new computable function.
- Clean restart proves state persistence for the tested schema, not robustness to every crash mode.
- Toy exact enumeration is mechanism evidence, not an AGI-relevant capability result.

### I. Iteration Step

#### Weakest assumption

The weakest assumption is that a hand-specified dependency factorization correctly captures which memories should change together. It is simultaneously responsible for revision locality, efficient retrieval, and bounded consolidation; if it is wrong, all three can fail while the finite algebra remains valid.

The first refinement should not add a neural retriever. It should learn candidate dependency edges from verifier-supported conditional changes, maintain posterior uncertainty over those edges, and test edge interventions against held-out revision events. Until then, graph locality is a guarded hypothesis.

#### Next-generation variant

If X65A passes, X65B can combine VDFM with the rejected contextual-sheaf idea: maintain local factor graphs per validity context and probabilistic consistency maps between overlapping contexts. This could represent gradual context drift without globally switching conventions. It should be attempted only after a finite identifiability theorem and exact small-context oracle are available.

#### Smallest exact recommendation

The smallest pilot that can genuinely distinguish continual autonomous learning from storage/replay/retrieval is:

1. final X64H passes and is frozen;
2. an exact \(K\le256\) latent-component posterior;
3. a 28-task dependency-structured stream containing unseen semantic compositions, new program compositions, delayed retention, planted revision, irrelevant interference, no-reuse, and a context switch;
4. a 4 KiB store, four-node/512-byte retrieval budget, and 1,000-expansion search budget shared across arms;
5. no-memory, budget-matched raw replay, shuffled/reset, typed-memory ablations, oracle relevance, and larger-budget memoryless controls at minimum, followed by the full 20 arms for the frozen run;
6. midpoint process termination and permitted-state-only reload with a forbidden-answer canary;
7. success only if paired transfer, retention, local revision, sublinear-relative growth, retrieval advantage, compounding under budget, restart persistence, and no-leakage gates all pass across untouched seeds.

This pilot is small enough for exact inference on the M5 Max and strong enough to return a decisive negative result. If budget-matched raw replay ties VDFM, or transfer vanishes under target-absence and restart audits, X65A has not demonstrated continual learning.

#### AGI relevance filter

- **Continual learning:** direct target through transfer, retention, revision, and bounded growth.
- **Systematic compositional reasoning:** later semantic and procedural targets are unseen compositions of verified components.
- **Distribution shift:** final streams hold out compositions, orders, contexts, and convention switches.
- **Causal/world-model reasoning:** typed dependency and provenance edges support intervention-style revision tests.
- **Planning/tool use:** verified procedures become bounded-search macros with explicit effects.

These are design mappings only. They are not evidence that Sentinel has acquired any of the broader AGI capabilities.

#### Final scientific status

VDFM is a coherent and falsifiable architecture hypothesis. The finite mathematical core is stronger than a generic “use memory/RAG” proposal, but the empirical claim remains completely open. PlugMem, DreamCoder, Rosenbloom/Soar, TMS/ATMS, TRUSTMEM/MemGuard, AgentCL, AgentMemoryBench, and PLACEMEM collectively make the remaining novelty space narrow. What may remain distinctive is the exact conjunction and its unusually strict causal gates; only implementation after X64H, a frozen audit, and successful component-matched ablations could establish that the combination is useful.
