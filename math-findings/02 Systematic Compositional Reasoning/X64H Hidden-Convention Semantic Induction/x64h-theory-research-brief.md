# X64H — Hidden-Convention Semantic Induction

Status: theory and falsification package; no X64H architecture result has been measured  
Date: 2026-08-26  
Scope: controlled language, executable task induction, persistent latent conventions, behavioral evidence, and clarification

Evidence vocabulary in this report:

- **FORMALLY PROVED / REPRODUCED:** Lean 4 + Mathlib accepted the theorem with no `sorry`; the tracked proof file was checked again directly.
- **MATHEMATICALLY DERIVED / INFERRED:** a paper derivation is supplied, but it has not been fully mechanized.
- **NUMERICALLY OBSERVED / MEASURED:** the included finite enumeration or symbolic script was actually executed.
- **HYPOTHESIS:** a falsifiable prediction for X64H, not a result.
- **UNKNOWN:** the current package does not settle the claim.

## A. Problem Target

### [1] Direct verdict

**Verdict:** the hidden-convention direction is mathematically sound as a falsifiable replacement for the X64G testbed. It removes the authored-inverse ceiling by withholding the realized convention from every non-oracle arm, while preserving an oracle-convention ceiling. The deterministic authored-inverse limitation is exact; the finite separating-family criterion is exact; the joint posterior, conflict Bayes factor, information-gain objective, and calibrated commitment bound are coherent under their stated assumptions.

This is not evidence that Sentinel can yet induce conventions. The full grammar can remain non-identifiable, exact inference can become factorial or exponential, an `OTHER` likelihood can be misspecified, and synthetic convention families can reward family decoding without transferring to natural language. Those are central falsifiers, not implementation details.

### Capability gap

X64D–X64G asked a learned parser to beat an authored parser that knew the designer's realizer. When that realizer was deterministic and injective, its inverse had zero exact-form loss. When it was stochastic and known, an authored Bayes decoder could attain the Bayes risk. The meaningful capability gap is therefore not “learn a known code better than its inverse.” It is:

> infer a persistent, hidden communication convention from a sequence of instructions, behavioral demonstrations, and clarification answers; simultaneously infer each current task; retain convention evidence across tasks; and abstain when neither the task nor convention is adequately represented.

This maps to four Sentinel research targets without claiming they have been achieved:

- distribution-shift generalization: the test convention is sampled after freeze;
- systematic compositional reasoning: semantic tree compositions are independently held out;
- continual adaptation: the convention posterior persists across tasks and restart;
- meta-learning and tool use: a shared convention prior transfers across episodes, and the agent chooses clarification actions.

### [2] Formal task family and notation

Let:

- \(\mathcal T\) be a finite type set;
- \(G_Z\) be a finite typed semantic tree grammar;
- \(\mathcal Z_d\) be its executable logical forms of depth at most \(d\);
- \(\mathcal U_{\le n}=\Sigma^{\le n}\) be bounded token strings;
- \(\Phi\) be a finite family of persistent conventions for the exact arm;
- \(\phi\sim p_\eta(\phi)\) be one hidden convention sampled per episode;
- \(z_t\sim p_G(z)\) be the intended semantic form at task \(t\);
- \(u_t\sim p_\phi(u\mid z_t)\) be its instruction;
- \(D_t=\{(x_{tj},y_{tj})\}_{j=1}^{m_t}\) be behavioral evidence;
- \(b\in\mathcal B\) be an executable candidate program;
- \(K_B(b\mid z)\) be the semantic-to-program kernel;
- \(q_t\) be a clarification question and \(A_t\) its answer;
- \(H_t=(u_{1:t-1},D_{1:t-1},q_{1:t-1},A_{1:t-1})\) be the observed history under the current convention;
- \(\bot_U,\bot_Z,\bot_B\) denote unknown realization, missing semantic meaning, and missing behavioral program respectively.

For a frozen behavioral input universe \(\mathcal X_B\), write

\[
z\equiv_B z'\iff
\forall x\in\mathcal X_B,\;
\operatorname{exec}(z,x)=\operatorname{exec}(z',x).
\]

If \(\mathcal X_B\) is only a finite probe set, this is probe-relative equivalence and must not be described as global program equivalence.

The behavioral likelihood may be exact,

\[
p(D\mid z)=\prod_j \mathbf 1\!\left[y_j=\operatorname{exec}(z,x_j)\right],
\]

or use a frozen observation-error channel,

\[
p(D\mid z)=\prod_j\left[(1-\rho)\mathbf 1[y_j=\operatorname{exec}(z,x_j)]
+\rho Q(y_j\mid x_j)\right].
\]

The exact channel is appropriate only when the trusted executor and demonstrations are deterministic and error-free. A fast evaluator disagreement is an evaluator bug until equivalence with the trusted path is established.

The inference target for task \(t\) is the posterior over \((Z_t,\Phi,B_t,O_t,M_t)\), where \(O_t\) is the open-world component and \(M_t\) is the matched-versus-conflicting evidence hypothesis. The action target is not necessarily an exact logical form: Sentinel may execute, ask, abstain, or request model expansion.

## B. New Mathematical Construct

### Candidate formalisms and rejection

Five candidates were considered.

| Candidate | Expressivity for required variation | Identifiability profile | Exact-inference profile | Principal problem |
|---|---|---|---|---|
| Typed PCFG with latent lexical emissions | synonyms, some polysemy, optional words | latent-state and atom label switching; needs anchors/separating contexts | polynomial inside sums for a fixed grammar; marginal-MAP can be hard | child permutation, deletion, phrase alignment, and attachment are indirect |
| Probabilistic CCG lexicon | strong typing, ambiguity, flexible composition | lexical entries and derivations can be observationally equivalent | restriction-dependent; general recognition is not cheaply exact | joint lexical/category induction creates a large search space before convention inference |
| General probabilistic synchronous CFG | phrases, reordering, paired meaning/surface structure | synchronous rule and semantic automorphisms | polynomial for fixed binarized subfamilies; grammar induction is hard | unrestricted variants are more expressive than the pilot needs and complicate exactness |
| Adaptor grammar | reusable phrases and open-ended construction discovery | cached fragments, categories, and convention effects can confound | foundational inference is MCMC/approximate | nonparametric inference obscures the first identifiability test |
| Typed Bayesian tree transducer | direct contextual tree-to-string realization | state/rule and semantic-tree automorphisms | dynamic programs exist for restricted fixed models; Bayesian rule induction is approximate | unrestricted latent transducer inference is too expensive for the required exact arm |

The survivor is a deliberately restricted member of the synchronous/tree-transducer family. The rejected first implementation is the adaptor-grammar variant: it is attractive for phrase discovery, but it makes it harder to determine whether X64H failed because conventions are non-identifiable or because nonparametric inference failed.

### [3] Selected model: Finite Typed Synchronous PCFG with Persistent Convention State

Call the selected pilot model **FT-SPCFG**. It is a finite typed synchronous PCFG, equivalently a deliberately restricted probabilistic semantic-tree-to-string grammar, whose structural and lexical parameters are persistent within an episode.

For each typed semantic constructor

\[
g:\tau_1\times\cdots\times\tau_r\to\tau,
\]

a transduction rule has the form

\[
(q,g,\kappa)\Rightarrow
s_0\;q_1[x_{i_1}]\;s_1\cdots q_m[x_{i_m}]\;s_m,
\tag{1}
\]

where:

- \(q,q_j\in Q\) are finite transducer states;
- \(\kappa\) is a bounded context such as parent constructor, semantic role, or attachment state;
- \((i_1,\ldots,i_m)\) is an ordered, non-repeating subset of \(\{1,\ldots,r\}\);
- each \(s_j\) is a fixed token phrase or a typed lexical-emission slot;
- omitted indices implement argument omission;
- a non-identity index order implements a word-order convention;
- alternative rules implement optional function words and phrase variants;
- state transitions \(q_j\) implement local attachment choices;
- bounded-depth left-hand fragments implement phrase-level realization spanning more than one semantic node.

All child slots are type checked. Ill-typed derivations have probability zero. Rules are linear: a semantic child cannot be duplicated. Deletion is permitted only in explicitly marked rule families and is separately ablated because it can destroy identifiability.

A convention is

\[
\phi=(\theta^{\mathrm{rule}},\theta^{\mathrm{lex}},
      \theta^{\mathrm{omit}},\theta^{\mathrm{attach}})\in\Phi.
\]

For the exact pilot, \(\Phi\) is finite and generated by a frozen meta-grammar. It contains structural choices, context-sensitive lexical tables, and a finite set of categorical probability vectors. A later approximate variant may place hierarchical Dirichlet priors over these vectors, but that is not needed to test the core claim.

Given semantic tree \(z\), let \(\mathcal D_\phi(z,u)\) be its valid transducer derivations with surface yield \(u\). Then

\[
p_\phi(u\mid z)
=\sum_{d\in\mathcal D_\phi(z,u)}
\prod_{r\in d}\theta^{\mathrm{rule}}_{\phi,r}
\prod_{e\in d}\theta^{\mathrm{lex}}_{\phi,e}.
\tag{2}
\]

The joint model is

\[
p(\phi,z,u,D)=p_\eta(\phi)p_G(z)p_\phi(u\mid z)p(D\mid z).
\tag{3}
\]

The convention, not independent utterance corruption, is the main source of variation. Conditional surface variation may remain inside a convention, but its probabilities are persistent parameters of \(\phi\).

FT-SPCFG directly supports:

- synonymy through multiple phrases emitted for one typed atom;
- contextual polysemy because the same phrase may be emitted by different atoms under different \((q,\kappa)\);
- word-order conventions through the child-slot permutation;
- phrase realization through multi-token terminals and bounded tree fragments;
- optional function words through rule alternatives;
- omitted arguments through marked deleting rules;
- attachment through transducer state and parent-role context;
- systematic speaker variation because all of these choices persist in \(\phi\).

### [4] Joint posterior and behavioral pushforward

For current evidence \(E=(u,D,H)\), the matched-model posterior is

\[
p(z,\phi\mid E,M=0)
=\frac{p(D\mid z)p_\phi(u\mid z)p(\phi\mid H)p_G(z)}
{\sum_{\phi',z'}p(D\mid z')p_{\phi'}(u\mid z')p(\phi'\mid H)p_G(z')}.
\tag{4}
\]

The persistent convention posterior after \(t\) completed interactions is

\[
p(\phi\mid H_{t+1})\propto p_\eta(\phi)
\prod_{s=1}^{t}
\sum_{z_s}p_G(z_s)p_\phi(u_s\mid z_s)p(D_s\mid z_s)
p(A_s\mid q_s,z_s,\phi).
\tag{5}
\]

Equation (5) prevents answer leakage: only an observed answer enters the product. A future target, future question answer, or the sampled convention identifier must never be serialized into history.

Inside the in-class matched component, the behavioral posterior is the pushforward

\[
p(b\mid u,D,H)=
\sum_{z,\phi}K_B(b\mid z)p(z,\phi\mid u,D,H).
\tag{6}
\]

Forms may be merged for the immediate task action only if they induce the same task-output behavior. Stored programs must retain a continuation signature. Define

\[
z\equiv_{\mathrm{cont}}z'
\iff
\forall c\in\mathcal C_{\mathrm{cont}},\;
\operatorname{exec}(c[z])=\operatorname{exec}(c[z']).
\tag{7}
\]

Current-output equivalence can be strictly coarser than \(\equiv_{\mathrm{cont}}\). Merging under the coarser relation before later composition is unsound.

### [5] Conflict as model comparison

Let \(M=0\) mean the instruction and demonstrations share one intended behavioral meaning. Its marginal likelihood is

\[
L_0=p(u,D\mid M=0,H)
=\sum_{\phi,z}p(\phi\mid H)p_G(z)p_\phi(u\mid z)p(D\mid z).
\tag{8}
\]

Let \(M=1\) mean the instruction meaning \(z_u\) and demonstration meaning \(z_D\) are behaviorally incompatible. Assume the semantic prior gives positive probability to at least two behavioral classes, so \(\kappa>0\). With

\[
\kappa=\sum_{z_u,z_D}p_G(z_u)p_G(z_D)
\mathbf 1[z_u\not\equiv_B z_D],
\]

define the normalized mismatch likelihood

\[
L_1=\frac{1}{\kappa}
\sum_{\phi,z_u,z_D}
p(\phi\mid H)p_G(z_u)p_G(z_D)
\mathbf 1[z_u\not\equiv_B z_D]
p_\phi(u\mid z_u)p(D\mid z_D).
\tag{9}
\]

For frozen prior odds \(\rho_1/\rho_0\),

\[
p(M=1\mid u,D,H)
=\frac{\rho_1L_1}{\rho_0L_0+\rho_1L_1},
\qquad
\frac{p(M=1\mid E)}{p(M=0\mid E)}
=\frac{\rho_1}{\rho_0}\frac{L_1}{L_0}.
\tag{10}
\]

Let \(m\) be X64E's language-posterior mass on forms exactly consistent with \(D\). If hard demonstrations make \(L_0=c_0m\) and the mismatch model makes \(L_1=c_1(1-m)\), then

\[
p(M=1\mid E)=
\frac{\rho_1c_1(1-m)}{\rho_0c_0m+\rho_1c_1(1-m)}.
\tag{11}
\]

This equals X64E's score \(1-m\) for every \(m\) exactly when \(\rho_0c_0=\rho_1c_1\). Otherwise X64E is at most a monotone surrogate. The Bayes-factor form is superior when behavioral evidence is noisy, conflict priors are unequal, the convention is uncertain, or different incompatible meanings have different likelihoods.

The states are distinct:

- **ambiguity:** posterior mass is spread over several compatible \((z,\phi)\) under \(M=0\);
- **conflict:** \(p(M=1\mid E)\) is high;
- **unknown language:** \(p(O=\bot_U\mid E)\) is high;
- **model misspecification:** both match and mismatch predictive likelihoods are low relative to frozen `OTHER` components.

### [6] Active clarification

Let \(X=(Z,\Phi,O,M)\). For a candidate clarification \(q\) with answer \(A_q\), use

\[
q^*=\arg\max_{q\in\mathcal Q}
I(X;A_q\mid E)-\lambda c(q),
\tag{12}
\]

where

\[
I(X;A_q\mid E)=H(A_q\mid E)
-\mathbb E_{X\mid E}H(A_q\mid X,E).
\tag{13}
\]

Behavioral questions request \(\operatorname{exec}(b,x_q)\) for a selected input. Semantic questions ask for a sense, attachment, argument, or paraphrase choice. Both are scored against the same posterior, so a question may primarily reduce task uncertainty, convention uncertainty, or open-world uncertainty.

For a fixed batch \(S\), \(F(S)=I(X;A_S\mid E)\) is monotone submodular if answers are conditionally independent given \(X\). Under that assumption, greedy maximization under a cardinality budget has the standard \(1-1/e\) guarantee. Information gain is not submodular in general; correlated answer channels can create complementarity. Adaptive submodularity of posterior entropy reduction has not been established here and must not be claimed. A version-space mass-reduction objective can be tested as a separate control under the stronger assumptions used in adaptive-submodular active learning.

### [7] Safe commitment

Let the available actions be execution of \(b\), clarification \(q\), abstention, and model expansion. Their Bayes risks are

\[
R_{\mathrm{exec}}(b)=\sum_s p(s\mid E)L(\operatorname{act}(b),s),
\]

\[
R_{\mathrm{ask}}(q)=c(q)+\mathbb E_{A_q\mid E}
\min_a R(a\mid E,A_q),
\]

with analogous frozen costs for abstention and expansion. Sentinel chooses the minimum-risk action; a probability threshold alone does not override a cheaper question or a high consequence of error.

A simple sufficient execution gate is

\[
p(M=1\mid E)\le\gamma,
\qquad
p(O\ne\mathrm{IN}\mid E,M=0)\le\epsilon,
\qquad
p(B=b^*\mid E,M=0,O=\mathrm{IN})\ge1-\delta,
\tag{14}
\]

plus

\[
R_{\mathrm{exec}}(b^*)\le
\min\{R_{\mathrm{ask}}(q^*),R_{\mathrm{abstain}},R_{\mathrm{expand}}\}.
\tag{15}
\]

Under calibration and model adequacy, Equation (14) gives

\[
p(B\ne b^*\mid E)
\le 1-(1-\gamma)(1-\epsilon)(1-\delta)
=\gamma+(1-\gamma)(\epsilon+\delta-\epsilon\delta)
\le\gamma+\epsilon+\delta.
\tag{16}
\]

This bound is conditional on the `OTHER` mixture and in-class posterior being calibrated for the true data process. It does not protect against a misspecified hypothesis class or a badly chosen base likelihood.

### [8] Explicit finite open world

For the first experiment use a finite mixture, not a nonparametric process. Let

\[
O\in\{\mathrm{IN},\bot_U,\bot_Z,\bot_B\},
\qquad p(O=o)=\alpha_o.
\]

Treat `M=1` as a separate top-level mismatch state and evaluate this `O` mixture conditional on `M=0`; Equations (17)–(20) suppress that condition for readability. The full decision state is therefore mutually exclusive mismatch versus matched-and-`O`, which matches the conditional commitment chain in Equation (14).

Freeze normalized base distributions \(m_U(u)\) over bounded strings, \(m_D(D)\) over bounded demonstrations, and \(m_B(D\mid z)\) over behavioral signatures excluded from the current program pool. Define

\[
L_{\bot_U}=m_U(u)\sum_zp_G(z)p(D\mid z),
\tag{17}
\]

\[
L_{\bot_Z}=m_U(u)m_D(D),
\tag{18}
\]

\[
L_{\bot_B}=\sum_{\phi,z}p(\phi\mid H)p_G(z)
p_\phi(u\mid z)m_B(D\mid z),
\tag{19}
\]

and use Equation (8) for \(L_{\mathrm{IN}}\). Then

\[
p(O=o\mid E)=\frac{\alpha_oL_o(E)}{\sum_{o'}\alpha_{o'}L_{o'}(E)}.
\tag{20}
\]

Local unknown phrases may additionally invoke a lexical `OTHER` emission inside FT-SPCFG. High `OTHER` mass triggers clarification, expansion, or abstention. It must never be renormalized away to manufacture a confident in-class singleton. Hierarchical Dirichlet processes, Pitman–Yor adaptors, or grammar-expansion priors are next-generation variants only after the finite mixture passes its falsification tests.

### Intuition

- A task history teaches both “what this task means” and “how this speaker encodes meanings.”
- Behavioral evidence remains authoritative for execution, while language controls posterior allocation among behaviorally viable meanings.
- Persistent systematic variation supplies learnable information; independent corruption does not.
- Identification is possible only through contexts that separate convention components modulo real symmetries.
- Clarification is an information action, not a fallback after decoding fails.
- `OTHER` probability remains inside the decision calculation, so closed-world normalization cannot create false certainty.

## C. Theoretical Results

### [3] Theorem 1 — Deterministic Realizer Dominance

**Theorem 1 (FORMALLY PROVED / REPRODUCED).** Let \(\mathcal Z\) and \(\mathcal U\) be finite, let \(p\) be any distribution on \(\mathcal Z\), and let \(R:\mathcal Z\to\mathcal U\) be deterministic and injective. If a parser may know \(R\), then \(f=R^{-1}\) on \(R(\mathcal Z)\) has zero exact semantic-recovery risk under zero-one loss. For every parser \(g:\mathcal U\to\mathcal Z\),

\[
\operatorname{Acc}(g)=\sum_zp(z)\mathbf 1[g(R(z))=z]
\le1=\operatorname{Acc}(f).
\]

No learned parser can strictly improve expected accuracy.

**Proof.** Injectivity gives \(R^{-1}(R(z))=z\) for every \(z\). Hence every generated item is recovered and the zero-one risk of \(f\) is zero. Zero-one risk is nonnegative, so no parser has lower risk. Equivalently, each indicator in any parser's accuracy is at most one, while every indicator for \(f\) equals one. \(\square\)

The mechanically checked version proves recovery, perfect weighted accuracy, and domination for arbitrary nonnegative finite weights. It is in `formal/X64H.lean`.

### Extension — non-injective and stochastic known realizers

Let \(K(u\mid z)\) be any known channel. For any decoder \(f\),

\[
\operatorname{Acc}(f)=
\sum_u p(f(u))K(u\mid f(u))
\le\sum_u\max_z p(z)K(u\mid z).
\tag{21}
\]

The authored MAP decoder

\[
f^*(u)\in\arg\max_zp(z)K(u\mid z)
\tag{22}
\]

attains equality. Thus the Bayes risk is

\[
R^*=1-\sum_u\max_zp(z)K(u\mid z).
\tag{23}
\]

For deterministic non-injective \(R\), substitute \(K(u\mid z)=\mathbf 1[R(z)=u]\):

\[
R^*=1-\sum_{u\in R(\mathcal Z)}
\max_{z:R(z)=u}p(z).
\tag{24}
\]

For denotational evaluation, quotient \(\mathcal Z\) by \(z\sim_Ez'\) when they have the same evaluated denotation. Replace the point weight in (21) by

\[
w([z],u)=\sum_{z'\in[z]}p(z')K(u\mid z').
\]

The authored Bayes decoder over equivalence classes remains optimal. Its denotational risk is zero exactly when every positive-probability utterance is supported by at most one denotational class. Exact-form ambiguity can therefore coexist with zero denotational error. For composition, use the finer continuation equivalence in Equation (7).

What a learned system may beat is a static parser that does **not** know realized \(\phi\), repeated human specification cost, a computationally expensive authored decoder under a fixed resource budget, robustness when conventions change, cumulative adaptation regret, or total two-part description length. It cannot beat an unconstrained oracle that knows the true channel under the same loss.

### [5] Theorem 2 — Separating-family identifiability

Consider \(k\) semantic atoms \(A=\{1,\ldots,k\}\), \(k\) surface symbols \(W\), and an unknown bijection \(\pi:A\to W\). Let the behavioral channel identify semantic subsets \(C_1,\ldots,C_m\subseteq A\), and observe surface sets \(U_i=\pi(C_i)\). Define

\[
v_a=(\mathbf 1[a\in C_i])_{i=1}^m,
\qquad
s_w=(\mathbf 1[w\in U_i])_{i=1}^m.
\]

**Theorem 2 (FORMALLY PROVED / REPRODUCED).** The convention \(\pi\) is uniquely identifiable from \((C_i,U_i)_{i=1}^m\) if and only if all atom incidence signatures \(v_a\) are distinct.

**Proof.** If \(w=\pi(a)\), then

\[
s_w(i)=\mathbf 1[\pi(a)\in\pi(C_i)]
=\mathbf 1[a\in C_i]=v_a(i).
\]

If all \(v_a\) are distinct, each word's observed signature names one unique atom, determining \(\pi^{-1}\). Conversely, if \(v_a=v_b\) for distinct \(a,b\), swapping \(a\) and \(b\) preserves every \(C_i\), so \(\pi\) and the swapped convention produce identical observations. \(\square\)

Lean verifies the sufficient direction, constructs the duplicate-signature swap, and proves that signature injectivity is equivalent to the absence of a nontrivial signature-preserving permutation.

### Corollary 2.1 — observations required

With \(m\) binary contexts there are at most \(2^m\) signatures. Therefore

\[
m\ge\lceil\log_2 k\rceil
\tag{25}
\]

is necessary. If arbitrary subsets are allowed, it is sufficient: assign every atom a distinct \(m\)-bit code and let context \(i\) contain exactly the atoms whose \(i\)-th code bit is one. Lean verifies \(k\le2^m\) for every injective binary signature map.

For uniformly random subset contexts, atom signatures are iid uniform in \(\{0,1\}^m\), so

\[
P(\text{separating})=
\begin{cases}
\dfrac{(2^m)_k}{(2^m)^k},&2^m\ge k,\\[4pt]
0,&2^m<k,
\end{cases}
\tag{26}
\]

where \((n)_k=n(n-1)\cdots(n-k+1)\). Exact enumeration for \(k=4\) matched (26) at \(m=1,2,3,4\): \(0,0.09375,0.41015625,0.66650390625\).

### Corollary 2.2 — bounded observation noise

Suppose each signature bit is independently flipped with probability \(p<1/2\) and each context is repeated an odd \(r\) times. Majority decoding and Hoeffding's inequality give

\[
P(\widehat v_a(i)\ne v_a(i))
\le e^{-2r(1/2-p)^2}.
\]

A union bound over \(km\) bits gives

\[
P(\text{any signature error})
\le km e^{-2r(1/2-p)^2}.
\tag{27}
\]

Thus it suffices that

\[
r\ge
\frac{\log(km/\delta)}{2(1/2-p)^2}.
\tag{28}
\]

For \(k=8,m=3,p=0.1,\delta=0.05\), the smallest odd integer above this bound is \(21\). Exact binomial enumeration at \(r=21\) produced all-signature recovery probability \(0.9999675\); the Hoeffding-plus-union lower bound was \(0.9710431\). This is a toy finite check, not a Sentinel result.

### Grammar symmetry and fundamental non-identifiability

Let \(\Gamma\) be the group of semantic-grammar and behavioral automorphisms that preserve the task prior, executor observations, and allowed contexts. Then \(\phi\) can at best be identified up to its orbit \([\phi]_\Gamma\). If \(\sigma\in\Gamma\), jointly relabeling meanings by \(\sigma\) and composing the convention with \(\sigma^{-1}\) leaves observations invariant.

Non-identifiability is fundamental when:

- two atoms have duplicate behavioral incidence signatures;
- a deleted argument never affects any demonstration or query;
- semantic and surface labels can be jointly permuted under a grammar automorphism;
- different attachment rules induce the same executable behavior on all allowed inputs;
- the candidate grammar lacks the true distinction;
- every task repeats one semantic structure, so task repetition and convention learning are confounded.

H2 must therefore score posterior mass on the true observational equivalence class, not necessarily the generator's arbitrary convention identifier.

### Theorem 3 — clarification lower bound

Let \(X\) range over \(K\) posterior hypotheses or equivalence classes. Suppose each query contributes at most \(C_{\max}\) bits of conditional mutual information. Any adaptive policy terminating with error probability at most \(\delta\) obeys the Fano-style lower bound

\[
\mathbb E[N]\ge
\frac{H(X\mid E_0)-h_2(\delta)-\delta\log_2(K-1)}{C_{\max}}.
\tag{29}
\]

If each answer alphabet has size at most \(A_{\max}\), then \(C_{\max}\le\log_2A_{\max}\). For exact identification \((\delta=0)\),

\[
\mathbb E[N]\ge\frac{H(X\mid E_0)}{\log_2A_{\max}}.
\tag{30}
\]

**Proof sketch.** By the chain rule, the transcript carries at most \(C_{\max}\mathbb E[N]\) bits about \(X\), under the stated bounded-information and stopping assumptions. Fano bounds residual entropy by \(h_2(\delta)+\delta\log_2(K-1)\). Subtract residual from initial entropy and rearrange. \(\square\)

In exact enumeration of all \(4!=24\) lexical bijections, posterior entropy was \(4.58496\) bits, the largest answer alphabet had size six, and (30) gave \(1.77371\) questions. The optimal and greedy information-gain policies both required exactly \(2.0\) questions in expectation; random disagreement required \(2.85714\). Greedy optimality here is a measured property of this finite instance, not a general theorem.

### Theorem 4 — safe commitment error

**Theorem 4 (MATHEMATICALLY DERIVED / INFERRED).** If the calibrated conflict posterior is at most \(\gamma\), the calibrated conditional open-world posterior is at most \(\epsilon\), and the calibrated conditional in-class posterior for \(b^*\) is at least \(1-\delta\), as ordered in Equation (14), then the posterior probability that execution of \(b^*\) is wrong is at most \(1-(1-\gamma)(1-\epsilon)(1-\delta)\).

**Proof.** Correctness requires matched evidence, in-class adequacy conditional on a match, and correctness inside that class. By the probability chain rule their joint posterior probability is at least \((1-\gamma)(1-\epsilon)(1-\delta)\). Take the complement. \(\square\)

### Finite-convention posterior concentration

**Proposition (MATHEMATICALLY DERIVED / INFERRED).** Let \(\Phi\) be finite, give the true convention class positive prior mass, and let completed task observations be conditionally iid under a persistent \(\phi_*\). If every non-equivalent \(\phi\) has positive KL separation

\[
D_{\mathrm{KL}}(P_{\phi_*}\Vert P_\phi)>0,
\]

then posterior mass on \([\phi_*]_\Gamma\) converges to one almost surely.

**Proof sketch.** For each false class, the normalized log posterior odds are a prior term divided by \(t\) plus an empirical mean log likelihood ratio. The strong law sends the latter to the negative KL divergence. Every false-class odds ratio therefore tends to zero; finiteness permits summation. \(\square\)

This proposition does not apply unchanged to adaptively selected, nonstationary tasks. X64H must either log the policy and use a controlled adaptive-design argument or maintain an iid identification subset.

### Assumption stress test

| Assumption | If violated | Required diagnostic |
|---|---|---|
| Realizer known to authored parser | inverse theorem applies | separate static-family and oracle-convention arms |
| Persistent convention | no cross-task signal | shuffled-history and no-memory arms tie joint inference |
| Separating contexts | posterior cannot concentrate uniquely | compute automorphism/equivalence class before evaluation |
| Trusted behavior channel | language may “correct” executor evidence | differential test trusted and fast evaluators |
| Positive true-model prior | Bayes cannot recover truth | explicit `OTHER` and support audit |
| Calibrated `OTHER` | commitment bound is meaningless | reliability curves and unknown false-confidence rate |
| Conditional answer independence | batch IG need not be submodular | report complementarity counterexamples; no greedy theorem claim |
| Held-out semantics independent of convention | task repetition can mimic adaptation | crossed split and reset/shuffle controls |

## D. Formal Verification Plan

### [3, 5] Claims formalized first

The tracked Lean file `formal/X64H.lean` contains five theorem-level checks and no `sorry`:

1. an injective realizer's `invFun` recovers every generated meaning;
2. the authored inverse receives all nonnegative finite prior weight;
3. no parser has greater weighted exact-recovery accuracy;
4. separating signatures identify a finite convention, while duplicate signatures produce a nontrivial observational automorphism;
5. \(m\) binary contexts can distinguish at most \(2^m\) atoms.

Representative executable signatures are:

```lean
theorem noParserStrictlyImprovesKnownInjectiveRealizer
    [Fintype Z] [Nonempty Z] [DecidableEq Z]
    (weight : Z → ℝ) (hweight : ∀ z, 0 ≤ weight z)
    (realizer : Z → U) (hinjective : Function.Injective realizer)
    (parser : U → Z) :
    weightedAccuracy weight realizer parser ≤
      weightedAccuracy weight realizer (Function.invFun realizer)

theorem separatingSignaturesIdentifyConvention
    (signature : A → S) (hseparating : Function.Injective signature)
    (first second : A ≃ W)
    (hobservation : ∀ word,
      surfaceSignature first signature word =
      surfaceSignature second signature word) :
    first = second

theorem signaturesSeparateIffNoNontrivialAutomorphism
    (signature : A → S) :
    Function.Injective signature ↔
      ∀ permutation : A ≃ A,
        (∀ x, signature (permutation x) = signature x) →
        permutation = Equiv.refl A

theorem binarySignatureCardinalityBound
    [Fintype A] (m : ℕ) (signature : A → (Fin m → Bool))
    (hseparating : Function.Injective signature) :
    Fintype.card A ≤ 2 ^ m
```

### Mechanical-check record

The file was checked with Lean 4.34.0-rc2 and Mathlib revision `85e3a25e006c35636f0e53b0e9296caca2685bc0`. The Mathlib workspace build passed, and the tracked file was then checked directly with `lake env lean`; both exited successfully. This is **FORMALLY PROVED / REPRODUCED** for the exact finite statements in the file. It is not a proof of the stochastic extensions, FT-SPCFG inference, posterior concentration, or X64H's empirical gates.

Coq/Isabelle was not used. The current key finite statements already have direct, small Lean proofs and the task specifically requires Lean as primary. A secondary prover becomes relevant when the stochastic Bayes-risk theorem or a measure-theoretic posterior-concentration theorem is formalized; installing a second compiler before selecting such a theorem would not add an independent mathematical result.

### Symbolic derivation checks

`x64h_symbolic_checks.py` was executed with SymPy 1.14.0. It verified:

- posterior odds equal prior odds times the Bayes factor;
- Equation (11) reduces to \(1-m\) under balanced effective constants;
- conditional-on-match commitment error is \(\epsilon+\delta-\epsilon\delta\), and total conflict/open-world/in-class error is \(1-(1-\gamma)(1-\epsilon)(1-\delta)\);
- for wrong-action loss \(L\) and query cost \(c\), executing is no more expensive than asking only when \(p_{\max}\ge1-c/L\).

The machine-readable result is `results/symbolic-checks.json`. These are **NUMERICALLY OBSERVED / MEASURED symbolic checks**, not independent proofs of model calibration.

### Resource-use audit

- **Lean 4 + Mathlib:** used for the authored-inverse and finite identifiability theorems; checked twice.
- **SymPy:** used for posterior-odds, X64E-equivalence, conflict/open-world/commitment, and act-versus-ask identities.
- **Python + NumPy + SciPy:** used for exact enumeration, array-safe plotting inputs, and an independent `scipy.stats.binom` cross-check of every majority-error calculation. A NumPy fixed-width integer overflow found during this check was corrected by coercing combinatorial exponents to unbounded Python integers before the final figures were regenerated.
- **Matplotlib:** used to generate all three finite-model figures and visually inspected after regeneration.
- **Primary literature sources plus arXiv/Semantic Scholar/OpenAlex/Google Scholar discovery checks:** used for the bounded novelty audit; only primary papers support technical comparisons.
- **JAX/NumPyro, CVXPy, and NetworkX:** installed and smoke-tested in the project toolchain, but not used as research evidence here because the selected exact pilot has no differentiable training theorem, convex relaxation, causal graph, or graph-dynamics claim. They become relevant only to the amortized or expanded mechanism.
- **Markdown:** used as the reproducible report format. MLflow, Hydra, and DVC are installed but were not used to imply an experiment run: X64H has not been implemented. The implementation handoff requires configuration, run tracking, and artifact versioning before final hidden tests.
- **Coq/Isabelle and Mathematica/Maple:** not executed; no secondary-prover or proprietary-algebra result is claimed.

### Proof dependency graph

```text
Injective R
  └─ invFun left inverse
      ├─ pointwise exact recovery
      └─ perfect weighted accuracy
           └─ indicator ≤ 1 + nonnegative weights
                └─ authored-inverse dominance

Distinct signatures
  └─ each surface signature has a unique semantic preimage
      └─ convention equality

Duplicate signatures
  └─ transposition preserves every signature
      └─ nontrivial observational automorphism

Injective binary signature map
  └─ finite-cardinality injection
      └─ k ≤ |Bool^m| = 2^m
```

### Next Lean targets

1. finite stochastic-channel Bayes decoder optimality;
2. denotational quotient decoder optimality;
3. exact separating-family equivalence specialized to `Finset` contexts;
4. majority-decoding union bound, after selecting a Mathlib probability interface;
5. safe-commitment bound over a finite calibrated probability mass function.

The KL posterior-concentration proposition is deferred because formalizing its measure-theoretic assumptions would be disproportionate before the finite experiment establishes that the mechanism is worth extending.

## E. Mechanism/Architecture Instantiation

### [11] Computational graph

X64H adds a convention layer without replacing Sentinel's evidence-authoritative executor or behavioral version-space states.

```text
frozen semantic grammar ──> candidate z ──> trusted execution on D
          │                     │                    │
          │                     └──── K_B(b | z) ───┤
          │                                          │
p(phi | history) ──> FT-SPCFG p_phi(u | z) ─────────┤
                                                     v
                                       joint posterior p(z, phi, b)
                                          │       │        │
                         match/mismatch Bayes   OTHER   clarification MI
                                          │       │        │
                                          └── Bayes-risk action selector
                                                    │
                           execute / ask / abstain / expand
```

The persistent store contains posterior sufficient state for \(\phi\), the frozen model hash, and observed-history hashes. It must not contain the generator's convention identifier, latent target forms, or future answers.

### Exact finite algorithm

```text
INPUT:
  frozen Φ, semantic forms Z, grammar G, prior pη, program map KB
  trusted executor, conflict models L0/L1, OTHER models Lo
  prior convention log-weights log wφ from observed history H
  current instruction u and demonstrations D

1. For each φ in Φ and z in Z:
     language[φ,z]  ← InsideLikelihood(Gφ, z, u)
     behavior[z]    ← TrustedReplayLikelihood(z, D)
     joint[φ,z]     ← wφ · pG(z) · language[φ,z] · behavior[z]

2. Normalize joint only inside the IN, M=0 component.
   Independently compute L0, L1, L⊥U, L⊥Z, L⊥B and normalize their
   frozen top-level mixture. Never discard OTHER before normalization.

3. Push joint mass through KB to immediate behavioral classes.
   Retain exact z or continuation-equivalence class for persistent programs.

4. For each allowed question q:
     enumerate answer distribution from joint + OTHER states
     score(q) ← I((Z,Φ,O,M); Aq | u,D,H) - λ c(q)

5. Compute Bayes risks for execute(best b), ask(best q), abstain, expand.
   Apply the calibrated probability gates as necessary, not sufficient,
   conditions for execution. Choose the minimum-risk permitted action.

6. After and only after an answer or trusted outcome is observed:
     wφ ← wφ · Σz pG(z) pφ(u|z) p(D|z) p(Aq|q,z,φ)
     normalize, persist with model hash, and append an observation audit.

OUTPUT:
  action, behavioral posterior, convention-class posterior,
  conflict posterior, OTHER posterior, selected question and audit trace
```

Use log space and `logsumexp`. Cache language charts by `(model_hash, φ, z, u)` and trusted behavior signatures by `(executor_hash, z, D)`. Any approximate pruning must have a separate completeness flag; pruned exact inference is not the exact arm.

### [12] Complexity analysis

Let:

- \(F=|\Phi|\);
- \(Z=|\mathcal Z_d|\);
- \(n=|u|\);
- \(|G|\) be the fixed binarized grammar size;
- \(|N|\) be its chart nonterminals;
- \(m=|D|\);
- \(C_{\mathrm{exec}}\) be one trusted execution cost;
- \(Q\) be the number of candidate questions;
- \(A\) bound the answer alphabet.

For a fixed finite convention and semantic form, binarized inside inference is conservatively \(O(|G|n^3)\). Explicit exact inference is therefore

\[
O\!\left(FZ|G|n^3+ZmC_{\mathrm{exec}}+QFZA\right)
\tag{31}
\]

per task. Retaining every chart costs

\[
O(FZ|N|n^2)
\tag{32}
\]

memory; streaming \((\phi,z)\) pairs reduces chart memory to \(O(|N|n^2)\) plus \(O(FZ)\) posterior scores. Across \(T\) tasks, the unamortized bound is linear in \(T\), but caching repeated semantic subtrees and surface fragments may reduce constants.

The finite lexical-convention space can itself grow as \(k!\). This is not hidden by the polynomial chart bound. X64H must report \(F\), generated candidates, pruned candidates, peak memory, chart time, replay time, and query-scoring time. The amortized arm is allowed to reduce computation, but its accuracy must be compared against exact posterior marginals on instances where exact inference completes.

### [9] Adaptation regret

Let the oracle \(f_\phi\) know the realized convention and use the same semantic grammar and decision loss. Define

\[
\operatorname{Regret}(T)=
\sum_{t=1}^T \ell(\widehat z_t,z_t)
-\sum_{t=1}^T \ell(f_\phi(u_t,D_t),z_t).
\tag{33}
\]

The empirical target is decreasing per-task regret with interaction count. Under the finite-model posterior-concentration proposition, bounded loss, and convergence of the adaptive predictive distribution to the oracle predictive distribution, expected one-step excess risk tends to zero; by the Cesàro argument, \(\operatorname{Regret}(T)/T\to0\). No finite-time rate is claimed without a likelihood-separation and concentration analysis.

Report both semantic regret and decision regret. Abstention should not be scored as a semantic error and then ignored; use a frozen abstention cost in decision regret and separately report coverage and selective accuracy.

### [9] Two-part MDL and human specification cost

Use a frozen, computable codebook. For one shared adaptive model \(M\) and episodes \(e\),

\[
L_{\mathrm{adapt}}=
L(M)+\sum_e\left[
L(\widehat\phi_e\mid M)+L(H_e\mid\widehat\phi_e,M)
\right].
\tag{34}
\]

For independently authored parsers \(A_e\),

\[
L_{\mathrm{authored}}=
\sum_e\left[L(A_e)+L(H_e\mid A_e)\right].
\tag{35}
\]

Use `-log2` probability codes plus an explicit finite grammar/rule code; do not invoke uncomputable Kolmogorov complexity. Learning is more economical only if

\[
\Delta_{\mathrm{MDL}}=L_{\mathrm{authored}}-L_{\mathrm{adapt}}>0
\tag{36}
\]

under the same codebook, with a convention-level confidence interval. Also log human authoring minutes and rule counts, but do not merge those units into bits unless a conversion is preregistered.

## F. Empirical Falsification Plan

### [13] Minimal synthetic tasks

Run two layers before the full protocol.

**Layer 0 — finite bijection sanity test.** Use \(k\in\{4,8,16\}\) semantic atoms, hidden lexical bijections, subset contexts, and exact or bit-flipped behavioral signatures. This must reproduce the separating criterion, symmetry counterexample, lower bound, and query calculations. The included script already supplies a reference result for \(k=4\) and a noise calculation for \(k=8\).

**Layer 1 — typed compositional convention test.** Use the existing X64 semantic logical forms and trusted executor. Add only the FT-SPCFG realization layer. Each convention changes a crossed subset of lexical mapping, context-conditioned sense, child order, multi-token phrase, optional function word, argument omission, and attachment. Semantic trees and surface derivations remain auditable.

The generator must compute convention equivalence classes and reject structurally invalid meta-grammar draws using a criterion frozen before final sampling. It may not reject a final convention because an inference arm performs poorly.

### X64H freeze protocol

Before final test sampling, commit and hash:

- semantic task grammar and typed executor;
- convention meta-grammar and structural-validity filter;
- prior over conventions and top-level `OTHER` priors;
- exact and amortized inference algorithms;
- parser features and candidate limits;
- semantic and behavioral query pools and costs;
- commitment and abstention risks;
- match/mismatch likelihoods and conflict threshold;
- open-world base distributions and expansion rules;
- all H1–H10 thresholds;
- evaluator, confidence-interval method, and leakage audit;
- development and validation generation procedures.

Only after that commit may hidden convention parameters and final seeds be sampled. Hashes, generated manifests, and process-restart state must be retained. A final-test failure cannot trigger edits to the convention family, parser, thresholds, or evaluator; such an edit starts X64I with new untouched seeds.

### Convention sampling and independent splits

One convention persists for an episode of tasks. Variation is systematic and factorized in the convention tuple, not redrawn independently for each utterance.

Hold out independently:

1. semantic tree skeletons and production combinations;
2. entire convention tuples and tuple-factor combinations;
3. atom-to-phrase lexical mappings;
4. bounded phrase constructions;
5. exact surface strings.

Use a crossed design so late tasks under a convention contain new semantic compositions. Repeating a task skeleton is logged and excluded from the primary transfer endpoint.

### Scale and resources

Default preregistration target:

- 64 development conventions;
- 32 validation conventions;
- 48 hidden test conventions;
- 64 sequential tasks per convention;
- four untouched final seeds, each contributing 12 conventions.

This fits the requested ranges and uses convention, not task, as the primary independent unit. On the M5 Max with 128 GB unified memory, exact inference should be benchmarked during development. The final scale may be reduced only through a validation-time, preregistered bottleneck rule, for example if projected exact inference exceeds 30 minutes per convention or 100 GB peak memory. The rule and reduced scale must be frozen before test seeds are sampled.

### Required arms

1. demonstrations only;
2. static authored parser that knows only the convention family;
3. static parser authored for one default convention;
4. exact Bayesian convention inference;
5. learned/amortized convention inference;
6. joint task-and-convention posterior;
7. joint posterior with random disagreement queries;
8. joint posterior with information-gain queries;
9. no persistent convention memory;
10. shuffled convention history;
11. oracle hidden convention;
12. oracle task meaning;
13. no open-world component;
14. no confirmation.

The static family-aware parser receives the frozen meta-family but not sampled \(\phi\). Arm 11 receives \(\phi\) and defines the semantic-channel ceiling. Arm 12 isolates language/convention error from behavioral search error.

### Central measurements

- task accuracy and calibrated decision loss by within-convention interaction index;
- semantic and decision adaptation regret;
- posterior mass on the true convention equivalence class;
- posterior mass on true task meaning and behavioral class;
- number, type, and cost of clarification questions;
- conflict AUROC, AUPRC, Brier score, and calibration error;
- unknown word, construction, meaning, and program detection separately;
- false confident in-class answers and selective coverage/accuracy;
- transfer to unseen semantic compositions under the same convention;
- cross-convention interference and recovery;
- persistence after process restart;
- two-part MDL, authored rule count, and authoring time;
- exact/amortized inference runtime, peak memory, candidate count, and posterior divergence.

Use convention-level paired bootstrap confidence intervals, stratified by final seed. Task-level intervals alone are anticonservative because tasks share \(\phi\). Report every seed, not only pooled means.

### [14] Preregistered gates

The numerical thresholds below are proposals to calibrate on validation and then freeze. They are not measured X64H results.

| Gate | Pass criterion |
|---|---|
| **H1 Testbed validity** | Oracle-convention accuracy \(\ge0.98\); static-family upper 95% CI below oracle lower CI and point accuracy \(<0.95\). |
| **H2 Convention identifiability** | Mean increase in true-equivalence-class log posterior odds has lower 95% CI \(>0\); final median class mass \(\ge0.80\). |
| **H3 Within-convention transfer** | Late-quartile decision regret is at least 20% below early-quartile regret with paired lower CI \(>0\), on unseen semantic compositions. |
| **H4 Reset control** | No-memory and shuffled-history arms remove at least 80% of the measured H3 gain. |
| **H5 Active clarification** | Information-gain queries reduce decision regret or total query cost by at least 10% versus random disagreement, with paired lower CI \(>0\); decision regret is primary. |
| **H6 Semantic induction** | Joint inference beats the static family-aware parser by at least five accuracy points and the paired 95% lower bound is \(>0\). |
| **H7 Conflict** | AUROC \(\ge0.85\), AUPRC exceeds prevalence by \(\ge0.25\), and frozen-bin ECE \(\le0.05\). |
| **H8 Open world** | Aggregate unknown recall \(\ge0.80\), each subtype reported, and false confident in-class answers \(\le1\%\). |
| **H9 No leakage** | Every target/convention/future-answer field is absent from pre-observation memory; hash and taint tests have zero violations. |
| **H10 Replication** | H2, H3, H5, and H6 have the preregistered direction in all four seeds and pooled convention-level intervals exclude zero. |

H2 evaluates equivalence-class mass. If the generator's class is non-singleton, exact identifier recovery is neither required nor claimed.

### Falsifiers

The framework fails this cycle if:

- hidden conventions are not identifiable even modulo frozen symmetry;
- the static family-aware parser remains perfect;
- gains disappear after removing repeated task structures;
- information-gain queries do not beat random queries;
- persistent memory does not improve later same-convention tasks;
- gains vanish on new semantic compositions;
- conflict scores remain uncalibrated;
- open-world cases become confident in-class singletons;
- the oracle convention arm fails, indicating a grammar/evaluator problem;
- exact Bayes fails, indicating model or identifiability failure rather than amortization;
- final-test failures trigger edits to the frozen family or parser.

### Diagnostic matrix

| Failure | Likely cause |
|---|---|
| Oracle convention fails | semantic/surface grammar, candidate pool, or executor is inadequate |
| Exact Bayes succeeds; amortized fails | recognition/inference approximation problem |
| Neither exact nor amortized succeeds | non-identifiability, wrong likelihood, or missing support |
| Same-convention transfer absent | convention posterior is not retained, not used, or not informative |
| Random and information-gain tie | query pool or posterior partitions are uninformative |
| Conflict fails | mismatch likelihood or conflict prior is misspecified |
| Static parser is perfect | convention leak, invariant decoding shortcut, or trivial family |
| Open-world false confidence persists | base likelihood, prior, or confirmation rule is inadequate |
| Restart loses gains | persistence format or model-hash restoration is defective |

### Exact plots

Generate at minimum:

1. accuracy, decision regret, query count, and true-class posterior versus task index;
2. convention-class reliability plot and entropy trajectory;
3. matched/mismatch score densities plus ROC and precision-recall curves;
4. unknown subtype reliability and false-confidence boundary map;
5. no-memory/shuffle/joint/oracle learning curves;
6. query policy regret-versus-cost frontier;
7. held-out semantic-composition performance by convention factor;
8. runtime and peak memory versus \(F\), \(Z\), tree depth, and utterance length;
9. MDL crossover versus number of conventions;
10. cross-convention interference matrix before and after restart.

The included Matplotlib references are:

- `figures/identifiability-probability.png`;
- `figures/noisy-signature-recovery.png`;
- `figures/query-policy-comparison.png`.

They visualize a simplified finite model only.

## G. Comparison to Existing Methods

### [15] Closest primary prior art

The separate `prior-art-audit.md` records 27 primary sources and exact novelty deltas. The closest families are:

- grounded weakly supervised CCG and execution: [Artzi and Zettlemoyer (2013)](https://aclanthology.org/Q13-1005/);
- denotation-supervised executable parsing: [Berant et al. (2013)](https://aclanthology.org/D13-1160/);
- Bayesian tree transduction: [Jones, Johnson, and Goldwater (2012)](https://aclanthology.org/P12-1051/);
- synchronous grammar with logical forms: [Wong and Mooney (2007)](https://aclanthology.org/P07-1121/);
- Bayesian program learning: [Lake, Salakhutdinov, and Tenenbaum (2015)](https://www.science.org/doi/10.1126/science.aab3050);
- program-library learning and joint language-program models: [DreamCoder](https://doi.org/10.1145/3453483.3454080) and [LAPS](https://proceedings.mlr.press/v139/wong21a.html);
- reusable nonparametric phrase structure: [Adaptor Grammars](https://papers.nips.cc/paper_files/paper/2006/hash/62f91ce9b820a491ee78c108636db089-Abstract.html);
- uncertain lexica and pragmatic inference: [Bergen, Levy, and Goodman (2016)](https://semprag.org/index.php/sp/article/view/sp.9.20);
- persistent partner adaptation and hierarchical conventions: [Hawkins et al. (2020a)](https://aclanthology.org/2020.conll-1.33/) and [Hawkins et al. (2020b)](https://escholarship.org/uc/item/7849q1dm);
- semantic information-theoretic clarification: [Tellex et al. (2012)](https://www.roboticsproceedings.org/rss08/p52.html);
- behavioral information-gain program queries: [Huang et al. (2022)](https://arxiv.org/abs/2205.07857);
- hierarchical Bayesian meta-learning: [Grant et al. (2018)](https://openreview.net/pdf?id=BJ_UL-k0b).

Batch information gain is not generally submodular; [Krause and Guestrin (2005)](https://www.cs.cmu.edu/~guestrin/Publications/UAI2005/uai2005.pdf) give both a counterexample and conditional-independence conditions. Adaptive greedy guarantees require the separate adaptive-submodularity conditions of [Golovin and Krause (2011)](https://jair.org/index.php/jair/article/view/10642), which this report does not assume for entropy reduction.

### Formal comparison

| Method family | Meaning/program posterior | Persistent hidden convention | Verifier evidence | Semantic query | Behavioral query | Explicit open-world mass |
|---|---:|---:|---:|---:|---:|---:|
| Grounded/weak semantic parsing | yes | no | yes/partial | no | no | usually no |
| Bayesian program induction | yes | no | behavioral examples | no | sometimes separate | no |
| LAPS / joint language-program learning | yes | no | yes | no | no | no |
| RSA / uncertain lexicon | semantic/referential | sometimes partner model | no exact executor | no | no | no |
| Persistent convention learning | usually referential | yes | interaction reward | limited | no | no |
| Interactive semantic parsing | current parse | no | yes/partial | yes | no | usually no |
| Active program synthesis | program set | no | yes | no | yes | no |
| **X64H proposal** | joint \(z,b\) | explicit \(p(\phi\mid H)\) | trusted exact/noisy channel | yes | yes | \(\bot_U,\bot_Z,\bot_B\) |

### Expressivity, efficiency, and robustness deltas

- **Expressivity:** FT-SPCFG adds an episode-level structured convention variable to executable semantic parsing; it is less expressive than unrestricted CCG, tree transducers, or adaptor grammars by design.
- **Efficiency:** exact inference is available only because \(\Phi\), \(\mathcal Z_d\), and grammar topology are finite and frozen. The factorial convention space is the main scaling risk.
- **Robustness:** X64H's proposed delta is explicit conflict, `OTHER`, abstention, reset controls, and evidence-authoritative execution—not a guarantee against misspecification.

### [16] What is genuinely new, if anything

No individual ingredient is new. A bounded primary-source audit found no source directly combining all six of:

1. executable behavioral version spaces;
2. a persistent hidden communication convention;
3. trusted verifier feedback;
4. active semantic questions;
5. active behavioral questions;
6. explicit expandable out-of-model mass and safe commitment.

Therefore the only defensible novelty statement is:

> **HYPOTHESIS — search-bounded composition-level novelty:** X64H may contribute a unified formal state and leak-resistant evaluation protocol for jointly adapting executable task meaning and a persistent hidden realization convention while choosing between semantic and behavioral clarification and retaining explicit model-inadequacy mass.

This is not proof of global novelty. It requires a broader citation graph and, more importantly, a passing experiment. Claims that latent conventions, Bayesian semantic parsing, information-gain querying, executable weak supervision, persistent adaptation, or open-world grammar learning are themselves new are prohibited.

## H. Failure Modes & Boundary Conditions

### Strongest objections first

1. **Joint relabeling can make \((z,\phi)\) non-identifiable.** Rebuttal: score equivalence-class mass and require separating contexts. If the class remains behaviorally broad, abstain; do not relabel failure as success.
2. **The meta-grammar is still authored.** Rebuttal: the claim is adaptation within a frozen family, not natural-language acquisition. H8 and held-out factors test limited model inadequacy; broader language remains out of scope.
3. **A static parser may exploit an invariant shortcut.** Rebuttal: H1, post-freeze sampling, crossed lexical/phrase splits, and generator taint tests are mandatory.
4. **Task repetition can imitate convention learning.** Rebuttal: primary H3 uses unseen semantic compositions and reset/shuffle controls.
5. **Exact inference may be computationally meaningless at useful scale.** Rebuttal: report the factorial \(F\), exact runtime, and memory; compare amortized inference only where exact marginals are available.
6. **`OTHER` can be calibrated to the generator rather than reality.** Rebuttal: report every unknown subtype, reliability, and false-confidence rate. No claim extends beyond the frozen synthetic family.
7. **Information gain can ask cheap but decision-irrelevant questions.** Rebuttal: compare pure MI with expected decision-risk reduction and report the regret-cost frontier.
8. **Behaviorally equivalent forms may diverge under later composition.** Rebuttal: preserve continuation signatures and merge only at the immediate-output layer.
9. **A mismatch model can absorb ordinary ambiguity or model failure.** Rebuttal: maintain separate \(M\) and \(O\), compare their predictive likelihoods, and calibrate on distinct generated conditions.
10. **MDL can be manipulated through the codebook.** Rebuttal: freeze one two-part codebook and report raw rule counts and authoring time as separate measurements.

### Adversarial cases

- homonyms whose disambiguating parent contexts never occur;
- paired arguments always co-occurring, producing duplicate signatures;
- phrase templates that encode an entire task, bypassing composition;
- omission of the only behaviorally discriminative argument;
- conventions with the same surface distribution but different latent rule labels;
- poisoned early clarification answers that dominate a persistent posterior;
- abrupt convention switches inside an episode;
- adversarially long or cyclic epsilon derivations;
- unknown strings assigned high in-class probability by a broad lexical base;
- fast-executor discrepancies that eliminate the true candidate.

### Identifiability and optimization pathologies

- posterior multimodality across lexical permutations;
- label switching and grammar automorphisms;
- likelihood plateaus when behavioral evidence is weak;
- particle or beam collapse in the amortized arm;
- overconfident variational approximations;
- convention-memory interference across speakers;
- confirmation bias from questions chosen under a misspecified posterior;
- hidden caps on candidate generation manufacturing false concentration;
- different stopping rules between random and information-gain query arms.

Every arm must share candidate support, answer access, maximum query budget, and stopping policy except for the ablated component.

## I. Iteration Step

### Weakest assumption

The weakest necessary assumption is that the true convention lies in a finite, frozen meta-family whose `OTHER` base distributions are informative enough to detect absence. This is exactly what makes the first experiment decisive and exactly what limits its external validity.

### Next-generation variant

If X64H passes H1–H10, X64I should replace only one assumption at a time:

1. add a grammar-expansion proposal under a finite reversible-jump or adaptor/Pitman–Yor prior;
2. preserve an exact finite subproblem as an inference oracle;
3. test abrupt and gradual convention drift with a change-point posterior;
4. add real human-authored controlled paraphrases without exposing their convention labels;
5. formalize the finite stochastic-channel theorem and safe-commitment bound in Lean.

If X64H fails H2, do not add a larger neural model. First diagnose the automorphism group, separating contexts, and support. If exact Bayes succeeds but amortized inference fails, refine only the inference model. If oracle convention fails, repair the semantic/surface grammar before interpreting any adaptation result.

### [17] Claude Code handoff

The direct implementation contract is in `CLAUDE-CODE-IMPLEMENTATION-SPEC.md`. Its required order is:

1. implement immutable typed data models and frozen hashes;
2. implement the finite convention generator and equivalence-class audit;
3. implement exact likelihoods and compare them with brute-force microcases;
4. preserve current trusted behavioral replay as authoritative;
5. add conflict and `OTHER` mixture likelihoods;
6. add query scoring and Bayes-risk action choice;
7. add persistence with answer-leakage taint tests;
8. implement all 14 arms through one shared evaluator;
9. run Layer 0 and Layer 1 development tests;
10. freeze, commit, sample hidden conventions, and never tune on final results.

### Required-final-response map

| Requested item | Location in this report |
|---|---|
| 1. verdict | A — Direct verdict |
| 2. formal definitions | A and B |
| 3. known-realizer theorem | C — Theorem 1 and extensions |
| 4. selected model | B — FT-SPCFG |
| 5. identifiability | C — Theorem 2, corollaries, symmetries |
| 6. posterior and conflict | B — Equations (4)–(11) |
| 7. active query | B and C — Equations (12), (29), (30) |
| 8. safe commitment | B and C — Equations (14)–(16) |
| 9. open world | B — Equations (17)–(20) |
| 10. MDL and regret | E — Equations (33)–(36) |
| 11. algorithm | E — Exact finite algorithm |
| 12. complexity | E — Equations (31), (32) |
| 13. X64H protocol | F |
| 14. gates and falsifiers | F |
| 15. prior art | G and `prior-art-audit.md` |
| 16. novelty delta | G — search-bounded hypothesis |
| 17. implementation spec | I and separate handoff file |

### Final scientific status

- **FORMALLY PROVED / REPRODUCED:** finite deterministic authored-inverse dominance; separating-signature sufficiency and swap counterexample; binary signature cardinality bound.
- **NUMERICALLY OBSERVED / MEASURED:** SymPy identities; exact finite identifiability probabilities; constructive minimum contexts for \(2\le k\le16\); noisy-majority probabilities; \(k=4\) optimal/greedy/random query values.
- **MATHEMATICALLY DERIVED / INFERRED:** stochastic Bayes decoder, quotient decoder, noise bound, Fano query bound, safe commitment, finite-convention concentration under iid correct-model assumptions.
- **HYPOTHESIS:** FT-SPCFG will adapt under hidden conventions, information-gain questions will improve regret, `OTHER` will prevent false confidence, and the integrated composition will beat a static family-aware parser.
- **UNKNOWN:** full-grammar identifiability, practical exact-inference scale, calibration under misspecification, transfer to human language, and any broader AGI relevance.
