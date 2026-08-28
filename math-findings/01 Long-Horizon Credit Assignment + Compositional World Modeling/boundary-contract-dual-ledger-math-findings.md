# Boundary Contract Dual Ledger (BCDL)

## Mathematical findings for long-horizon causal credit and safe model revision

**Project:** Sentinel  
**Category:** Long-horizon memory — supporting mechanism  
**Documented:** 2026-08-26  
**Source:** User-provided research report  
**Overall status:** Mathematically coherent under restrictive assumptions; integrated novelty and practical usefulness remain **UNKNOWN**.

## Verdict

The surviving proposal is **Boundary Contract Dual Ledger (BCDL)**. It separates two questions that long-horizon systems often blur:

1. Which intervention changed the eventual outcome?
2. Which part of the world model contradicts verified evidence and can be revised safely?

BCDL gives these questions different mathematical representations. Its causal ledger transports a signed interventional boundary difference and a backward-pulled utility function. Its epistemic ledger uses relational residuals to calculate the greatest local model relation that remains globally consistent with a verified end-to-end contract.

The work establishes two useful paper-level results and passes exhaustive finite checks. It also exposes its own central failure: exact safe diagnoses do not imply unique identification. In the reported 4,000-trial experiment, ambiguity increased sharply when positive evidence was incomplete. BCDL therefore remains a research hypothesis, not a demonstrated long-horizon memory solution.

## Why this belongs under long-horizon memory

BCDL is memory-relevant in three ways:

- It stores a boundary-level causal trace that can be moved across a long modular chain without relying on backpropagated parameter derivatives.
- It protects verified positive evidence while revising a persistent world model.
- It keeps multiple still-viable diagnoses explicit until a new intervention distinguishes them.

The classification must remain narrow. The report does **not** test retention duration, catastrophic forgetting, memory capacity, retrieval, consolidation, manageable memory growth, or transfer of remembered contracts across unrelated domains. BCDL should therefore be treated as a candidate mathematical substrate for long-horizon credit and correction, not as a complete memory system.

## 1. Formal problem target

The formal setting is a finite, acyclic, typed causal program composed of modules

\[
M_i=(I_i,O_i,R_i,\{K_i^\iota\}),
\]

where

\[
R_i\subseteq X_{I_i}\times X_{O_i},\qquad
K_i^\iota:X_{I_i}\rightsquigarrow X_{O_i}.
\]

Persistent state must be explicit at module boundaries. Let (A) be an intervention variable, (a) an intervention, (a_0) a reference intervention, (Y) the outcome, and (u:Y\to[-U,U]) a bounded utility. A verified end-to-end behavioral contract is

\[
G\subseteq X_0\times X_H.
\]

The intended use is to preserve causal influence across a long computation while permitting local, evidence-preserving revision of the world model.

### Required assumptions

The strongest reported results depend on:

- interventional boundary sufficiency;
- action-invariant downstream kernels;
- an acyclic modular decomposition;
- correct protected positive evidence;
- a single faulty module for the localization study;
- bounded interface width or manageable factor-graph treewidth;
- stationary, noiseless, truth-preserving constraints for monotone refinement.

These assumptions are substantive. In particular, discovering a compact sufficient boundary may be as difficult as learning the relevant world model.

## 2. Candidate audit

### Candidate 1 — Counterfactual Cutset Dividends

**Status: REJECTED as a novel construct.**

The candidate defined a coalition value

\[
V(S)=\mathbb E[u(Y)\mid do(a_S,a_{0,\bar S})]
\]

and Möbius interaction dividends

\[
\delta(T)=\sum_{S\subseteq T}(-1)^{|T|-|S|}V(S).
\]

This reduces to the established Harsanyi/Shapley family of attribution methods. The source report also identifies direct overlap with counterfactual Shapley credit assignment and model-based diagnosis. Renaming the coalition structure as a causal cutset does not create an independent formalism.

### Candidate 2 — Boundary Contract Dual Ledger

**Status: RETAINED AS HYPOTHESIS.**

The proposed novelty is not either ledger alone. It is the requirement that instrumental causal credit and epistemic model correction remain algebraically distinct while operating over the same typed causal contracts.

## 3. The dual-ledger construct

### 3.1 Causal-effect ledger

A boundary (C) is interventionally sufficient when an action-invariant downstream kernel (Q_{C\to Y}) satisfies

\[
P_a^Y=P_a^C Q_{C\to Y}
\]

for every admissible intervention. Define

\[
\zeta_C^{a:a_0}=P_a^C-P_{a_0}^C,\qquad
v_C=Q_{C\to Y}u.
\]

The ledger entry is

\[
\mathcal L_C(C;a,a_0,u)=(\zeta_C^{a:a_0},v_C),
\qquad
\kappa_C=\langle\zeta_C^{a:a_0},v_C\rangle.
\]

The important stored object is the signed boundary measure (\zeta_C), not only the scalar credit (\kappa_C). Signed causal measures move forward through the modular system, while utility covectors move backward.

This is different from claiming that gradients are impossible or that a transformer cannot simulate the computation. It is a different bookkeeping semantics, not a computational non-simulability result.

### 3.2 Epistemic-revision ledger

For relations (P:X\to B), (S:C\to Y), and (G:X\to Y), define the left and right residuals

\[
(P\backslash G)(b,y)
\iff
\forall x\,[P(x,b)\Rightarrow G(x,y)],
\]

\[
(G/S)(x,c)
\iff
\forall y\,[S(c,y)\Rightarrow G(x,y)].
\]

For module (R_i:B\to C), prefix (P_i), and suffix (S_i), the greatest globally safe local contract is

\[
A_i^\star=P_i\backslash(G/S_i).
\]

The tuples that conflict with the global contract and the largest safe retained relation are

\[
D_i=R_i\setminus A_i^\star,
\qquad
R_i^{\mathrm{safe}}=R_i\cap A_i^\star.
\]

If (E_i\subseteq R_i) is protected positive evidence, revision is admissible only when

\[
E_i\cap D_i=\varnothing.
\]

An identified version-space update is

\[
\mathcal H_i\leftarrow
\{h\in\mathcal H_i:E_i\subseteq R(h)\subseteq A_i^\star\}.
\]

This ledger does not treat failed reward as sufficient evidence for revising dynamics. It requires a verified model contradiction, preserves positive evidence, and keeps multiple admissible diagnoses unresolved when the evidence does not identify one.

## 4. Theoretical findings

### Theorem 1 — Cut-invariant causal-credit conservation

**Evidence label: PROVEN on paper; mechanical verification UNKNOWN.**

Let (C,D) be successive sufficient cuts. Suppose action-invariant kernels (K:C\rightsquigarrow D) and (Q:D\rightsquigarrow Y) satisfy

\[
P_a^D=P_a^C K,\qquad P_a^Y=P_a^D Q.
\]

Define

\[
\zeta_D=\zeta_CK,\qquad v_D=Qu,\qquad v_C=Kv_D.
\]

Then

\[
\boxed{
\langle\zeta_C,v_C\rangle
=
\langle\zeta_D,v_D\rangle
=
\mathbb E[u(Y)\mid do(A=a)]
-
\mathbb E[u(Y)\mid do(A=a_0)]
}.
\]

The equality follows by exchanging the order of integration and applying the downstream factorization. It shows that the scalar intervention contrast is conserved across sufficient cuts.

**What this establishes:** the ledger avoids computational attenuation caused solely by transporting a derivative through many modules.

**What it does not establish:** a physically mixing environment can genuinely erase causal influence. The theorem cannot preserve an effect that the environment itself destroys.

### Theorem 2 — Greatest-safe local revision

**Evidence label: PROVEN on paper; mechanical verification UNKNOWN.**

For every local replacement (X':B\to C),

\[
\boxed{
P;X';S\subseteq G
\iff
X'\subseteq P\backslash(G/S)
}.
\]

Therefore

\[
R^{\mathrm{safe}}
=
R\cap\bigl(P\backslash(G/S)\bigr)
\]

is the unique greatest safe subrelation of (R).

The result follows from the left- and right-residual adjunctions for relational composition. It supplies a compositional certificate that a local deletion cannot violate the verified global contract.

**What this establishes:** exact safe repair alternatives can be computed in the stated finite relational setting.

**What it does not establish:** the theorem does not identify which module is actually faulty when several repairs are consistent with the evidence.

### Additional paper-level consequences

- **Substitution invariance:** modules with the same typed boundary relation and interventional kernels have identical external ledger values.
- **Approximation bound:** for (n) approximate substitutions with per-module error (\varepsilon),

  \[
  d_{\mathrm{TV}}(P_{\mathrm{out}},P'_{\mathrm{out}})
  \leq\min(1,n\varepsilon),
  \]

  and the two-arm utility-credit error is at most

  \[
  2U\min(1,n\varepsilon).
  \]

- **Finite monotone refinement:** under stationary, noiseless, truth-preserving constraints, at most

  \[
  \sum_i |R_i^{(0)}|
  \]

  strict tuple deletions can occur.
- **Conditional active-identification bound:** if each query leaves at most a fixed fraction (\rho<1) of the current version space, then

  \[
  N_{\mathrm{queries}}
  \leq
  \left\lceil\frac{\log N}{\log(1/\rho)}\right\rceil.
  \]

The active-identification bound is conditional on a strong contraction premise; it is not evidence that such queries are always available.

## 5. Formal verification status

**Status: UNKNOWN.** Lean 4/Mathlib and Coq were unavailable in the reported session. The report supplies candidate signatures, but no theorem should be described as mechanically checked.

The proposed verification order is:

1. left- and right-residual adjunctions;
2. greatest-safe local contract;
3. greatest-safe subrelation;
4. finite push/pull identity;
5. cut conservation;
6. total-variation hybrid bound;
7. finite-refinement termination.

The trusted sequence should prove the residual algebra first, then the safe-repair theorem, and separately prove finite push/pull before cut conservation.

## 6. Computational characteristics

For horizon (H) and dense boundary carrier size (q), the report gives:

| Operation | Time | Memory |
|---|---:|---:|
| Initial balanced cache | (O(Hq^3)) | (O(Hq^2)) |
| One local replacement | (O(q^3\log H)) | (O(Hq^2)) |
| All module residuals | (O(Hq^3)) | (O(Hq^2)) |
| Kernel credit transport | (O(Hq^2)) | (O(q)) streaming |

For factor-graph width (w),

\[
T=O(Nq^{w+1}),\qquad M=O(Nq^w).
\]

Only cache-update depth is logarithmic. General inference still faces interface-width and treewidth explosion. This is an efficiency qualification, not a minor implementation detail.

## 7. Empirical findings

### 4,000-trial localization study

**Evidence label: MEASURED as reported; independent reproduction UNKNOWN.**

The deterministic NumPy experiment used:

- horizons (H\in\{4,8,16,32,64\});
- positive-evidence coverage (p\in\{0,.25,.5,.75,1\});
- boundary size (q=6);
- one corrupted module per chain;
- 4,000 total trials.

Reported outcomes:

- At (p=0), unique localization was **0%**; every module remained a candidate.
- At (H=64,p=.75), unique localization was **29/160 = 18.1%**, exact 95% CI **[12.5%, 25.0%]**, with **3.525 candidates** on average.
- At (p=1), localization was **160/160** for every horizon.
- The true faulty module remained in the candidate set in every trial.

**Interpretation:** the experiment disconfirms the strong claim that residuation alone solves long-horizon identification. It computes safe repair alternatives but usually cannot determine which alternative is true without sufficiently complete positive evidence or additional interventions.

### Exhaustive and symbolic checks

**Evidence label: MEASURED as reported; independent reproduction UNKNOWN.**

The report states that SymPy and exhaustive finite enumeration verified:

- 4,096 associativity cases;
- 4,096 left-residual adjunction cases;
- 4,096 right-residual adjunction cases;
- 65,536 greatest-contract cases;
- the recurrence (T(d)=T(d-1)+c\Rightarrow T(d)=T_0+cd);
- exact cut-credit equality (-1/140) across rational kernels.

These checks support the finite constructions but do not substitute for general proof or trusted-path reproduction from preserved code and data.

## 8. Architecture interpretation

A BCDL cycle would:

1. update a balanced cache of typed contracts;
2. select an interventionally sufficient boundary;
3. store the signed intervention difference, backward-pulled utility, and their pairing;
4. refuse dynamics revision when there is no verified model contradiction;
5. compute each candidate module's greatest safe local contract and deletion set;
6. reject any repair that conflicts with protected positive evidence;
7. update a version space only when one diagnosis is identified;
8. otherwise request an intervention that separates the remaining diagnoses.

This is operationally distinct from BPTT, attention, and return redistribution. It is not shown to be outside the expressive power of those systems, and finite BCDL instances could be simulated by an augmented transformer or model-based reinforcement-learning system.

## 9. Novelty and prior-art boundary

**Overall novelty status: UNKNOWN.**

The individual ingredients have close prior art:

- BPTT and synthetic gradients transport parameter derivatives.
- RUDDER redistributes return.
- Counterfactual Shapley methods allocate causal credit over coalitions.
- Reiter-style diagnosis uses conflicts and consistency repair.
- Dijkstra-style predicate transformers and relational residuation supply weakest-condition mathematics.
- Open causal models already compose typed interventional kernels.
- Predictive-state representations may capture learned sufficient boundaries.

The proposed novelty delta is the integrated invariant:

> Instrumental action credit and epistemic model correction must inhabit separate algebraic ledgers, even though both operate over the same typed causal contracts.

That claim is falsified if the complete mechanism routinely reduces to standard model-based diagnosis plus open-causal composition or predictive-state sufficiency. The source's literature search cannot establish absence of prior art.

## 10. Failure modes and rejection criteria

The most important unresolved failure modes are:

1. **Boundary discovery:** compact interventional sufficiency may be as hard to learn as the full model.
2. **Diagnosis versus identification:** many safe repairs can remain observationally equivalent.
3. **Established mathematics:** the integration may collapse to known diagnosis and causal-composition machinery.
4. **Noise and nonstationarity:** irreversible monotone deletion is brittle when evidence can be wrong or the world changes.
5. **Interacting faults:** multiple-module failure can destroy unique localization and simple residual reasoning.
6. **Interface-width explosion:** treewidth may dominate the logarithmic cache-update depth.
7. **Physical effect decay:** a mixing environment can erase real causal influence.
8. **Feedback:** cyclic systems require equilibrium, trace, or another feedback semantics.
9. **Aliasing:** hidden state can make boundary states indistinguishable despite different causal futures.

Reject BCDL as a useful Sentinel direction if any of the following persists under controlled testing:

- sufficient boundary size grows essentially linearly with horizon;
- active interventions fail to reduce diagnosis sets faster than unstructured search;
- typed contract reuse fails under genuine domain shift;
- separating the two ledgers yields no repair, retention, or transfer advantage;
- a formal reduction shows that the complete mechanism is only renamed diagnosis plus open-causal composition.

## 11. Cheapest decisive next experiment

**Evidence label: INFERRED recommendation from the reported failure mode.**

Run a controlled **hidden-state aliasing and active boundary-refinement** experiment before integrating BCDL into Sentinel.

### Experimental arms

- **Fixed-boundary BCDL:** exact residual diagnosis with no state split.
- **Query-preserving boundary refinement:** when a proposed deletion conflicts with protected evidence, split the boundary with the cheapest witness-separating predicate and select an intervention designed to contract the diagnosis set.
- **Controls:** random intervention selection, oracle sufficient boundary, and no-revision baseline.

### Sweep

Vary horizon, hidden-state aliasing rate, positive-evidence coverage, number of interacting faults, and intervention budget. Use the same stopping rule and query budget in every arm.

### Primary measures

- probability that the true faulty module remains in the candidate set;
- unique-identification rate and diagnosis-set size;
- false deletion of protected evidence;
- boundary width and memory growth versus horizon;
- queries required to reach a fixed posterior/version-space size;
- retained performance on unaffected contracts after repair.

### Preregistered falsifier

The refinement hypothesis fails if, at matched intervention budget, it does not reduce diagnosis-set size faster than random querying, introduces any protected-evidence deletion, or requires boundary width that grows essentially linearly with horizon. A secondary falsifier is no measurable retention advantage over the no-revision or single-ledger control.

## 12. Canonical evidence ledger

| Claim | Status | Basis | Scope limit |
|---|---|---|---|
| Cut-invariant scalar causal credit across sufficient cuts | **PROVEN** | Paper derivation; finite rational check reported | Requires sufficient cuts and action-invariant kernels; not mechanically checked |
| Greatest-safe local subrelation by residuals | **PROVEN** | Paper derivation; exhaustive finite checks reported | Safety is not unique fault identification; not mechanically checked |
| Residuation alone identifies long-horizon faults | **RETRACTED** | 4,000-trial experiment shows severe ambiguity | Complete positive evidence recovers the single-fault toy setting |
| Counterfactual Cutset Dividends are a novel formalism | **RETRACTED** | Reduces to Harsanyi/Shapley-style attribution | None |
| True fault remained among candidates in the reported study | **MEASURED** | Reported in all 4,000 trials | Single-fault, finite synthetic setting; reproduction not supplied here |
| Dual ledgers improve continual correction and transfer | **HYPOTHESIS** | Architectural argument only | Requires ablations and held-out transfer evaluation |
| Compact sufficient boundaries can be learned | **UNKNOWN** | No boundary-learning experiment | Central feasibility question |
| Integrated BCDL mechanism is novel | **UNKNOWN** | Partial literature audit only | Close precursors exist for both ledgers |
| Query-preserving boundary refinement is a general abstraction mechanism | **SPECULATIVE** | Proposed next variant | No theorem or experiment yet |
| BCDL establishes AGI or complete long-term memory | **NOT ESTABLISHED** | No relevant acceptance test | Must not be claimed |

## Bottom line

BCDL is a coherent mathematical proposal for keeping long-horizon causal credit separate from evidence-certified world-model repair. Its conservation and greatest-safe-repair results are meaningful within their assumptions. The central empirical result is negative but useful: safe relational diagnosis does not by itself identify the true cause over long horizons. The next decision should depend on whether active boundary refinement controls ambiguity and memory growth under hidden-state aliasing without damaging protected knowledge.

## 13. References identified in the source report

These links are preserved from the supplied report and have not been independently re-audited in this documentation pass.

- [Counterfactual Shapley Credit Assignment](https://arxiv.org/abs/2607.16999)
- [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343)
- [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857)
- [A Theory of Diagnosis from First Principles](https://cse.sc.edu/~mgv/csce580f11/gradPres/reiter-diagnosis.pdf)
- [Guarded Commands, Nondeterminacy and Formal Derivation of Programs](https://dl.acm.org/doi/10.1145/360933.360975)
- [Open Causal Models](https://arxiv.org/abs/2304.07638)
- [Predictive Representations of State](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)
