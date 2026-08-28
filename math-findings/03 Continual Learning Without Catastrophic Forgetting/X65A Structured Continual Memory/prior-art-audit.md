# X65A Structured Continual Memory — Bounded Primary-Source Prior-Art and Novelty Audit

**Audit date:** 2026-08-27  
**Status:** bounded literature audit, not an exhaustive novelty determination  
**Research target:** a continual-memory system integrating exact finite hierarchical Bayesian latent components; typed episodic, semantic, procedural, and negative/revision memory; provenance and dependency tracking; verifier-constrained two-part-MDL consolidation; budgeted retrieval; graph-local probabilistic revision; procedural macro compounding; durable restart persistence; and causal, temporally valid no-leakage evaluation.

## Direct verdict

No individual X65A ingredient is novel. Primary literature already covers every major component separately, and several systems cover substantial bundles:

- **PlugMem** is the closest contemporary systems-level collision for typed episodic, semantic, and procedural memory graphs, provenance to source episodes, consolidation, and budgeted multi-hop retrieval.
- **Rosenbloom's graphical cognitive architecture** and **Soar** are strong earlier collisions for integrating procedural, semantic, and episodic memory in one architecture; Rosenbloom additionally uses a common probabilistic graphical substrate.
- **DreamCoder** is the strongest collision for Bayesian/MDL-guided consolidation into reusable procedural abstractions that compound future search.
- **MACLA** directly combines hierarchical procedural memory, Bayesian reliability estimates, and contrastive use of success and failure traces.
- **AGM, TMS/ATMS, provenance semirings, and probabilistic graphical models** establish the main mathematical ingredients of dependency-aware revision and local inference.
- **TRUSTMEM** and **MemGuard** are direct recent threats to any claim that verifier-gated consolidation or persistent verifier metadata is new.
- **AGENTCL** and the controlled study by Hu, Long, and Wang substantially narrow novelty claims about evaluating continual agent memory, compositional reuse, retrieval pollution, and forward transfer.

Within this bounded audit, **no located primary source reports and measures the entire X65A package in one system**. The defensible conclusion is therefore only a **provisional systems-integration gap**: the exact conjunction appears absent from the sources inspected. This is not evidence that X65A is categorically novel, and it is not yet a measured contribution unless X65A is implemented, compared against the strongest component-matched baselines, ablated, and evaluated under a genuinely causal no-leakage protocol.

## 1. Audit target and comparison rule

For this audit, the proposed X65A mechanism is decomposed into nine claims:

1. **Finite hierarchical Bayes:** exact posterior updating over a finite family of reusable latent components, with stated priors, likelihoods, and normalization conditions.
2. **Typed memory:** explicit episodic, semantic, procedural, and negative/revision records rather than one undifferentiated vector store.
3. **Provenance/dependencies:** claims and procedures retain derivational links to observations, verifiers, parents, and dependents.
4. **Verifier-constrained MDL consolidation:** a candidate abstraction is retained only if executable verification succeeds and a computable two-part code-length objective improves.
5. **Budgeted retrieval:** retrieval chooses a useful subset subject to explicit cost or capacity constraints.
6. **Graph-local probabilistic revision:** new evidence updates the dependent region while preserving unaffected marginals only under explicit conditional-independence assumptions.
7. **Procedural macro compounding:** verified procedures become callable abstractions that reduce later description length or search.
8. **Restart persistence:** durable state survives process restart with semantic equivalence checks.
9. **Causal no-leakage evaluation:** every prediction is made before its target becomes available, future-derived artifacts are excluded, and memory interventions distinguish genuine retained transfer from repetition or leakage.

The novelty question is not whether these labels co-occur. It is whether a prior measured system implements the same operational conjunction with comparable guarantees and controls.

### Evidence and threat labels

- **Very high threat:** a source directly implements four or more central elements, or independently covers a proposed headline contribution.
- **High threat:** a source integrates two or three central elements or invalidates a broad novelty statement.
- **Moderate threat:** a source establishes one ingredient, theorem family, or necessary caveat.
- **Low threat:** conceptual background with limited direct mechanism overlap.

“Not found” below means only “not found in this bounded search,” never “does not exist.”

## 2. Search method and scope

The search prioritized original papers, official proceedings pages, publisher DOI records, author-hosted manuscripts, and author repositories. It covered targeted combinations of the terms *continual/lifelong learning*, *catastrophic forgetting*, *experience replay*, *complementary learning systems*, *episodic semantic procedural memory*, *Bayesian continual learning*, *program library learning*, *MDL abstraction*, *macro operators*, *options*, *belief revision*, *truth maintenance*, *provenance*, *submodular retrieval*, *knapsack*, *memory lower bounds*, *persistent agent memory*, *verifier memory*, *memory consolidation*, *compositional transfer*, *prequential evaluation*, and *target leakage*.

The audit deliberately includes foundational work and direct recent collisions through the audit date. It does not use surveys as evidence for mechanism or priority. Surveys and search-result summaries were not treated as primary support.

## 3. Continual learning, catastrophic forgetting, replay, and Bayesian updating

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| Michael McCloskey and Neal J. Cohen, **“Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem”** (1989). [DOI](https://doi.org/10.1016/S0079-7421(08)60536-8) | Demonstrates that sequential acquisition can overwrite previously learned mappings in connectionist networks and analyzes rehearsal/interleaving effects. | Establishes the failure X65A is intended to avoid and the importance of retained prior evidence. | It supplies neither structured persistent memory nor a consolidation/revision architecture. | **Moderate**: catastrophic forgetting itself and rehearsal as a response are old problems. |
| James Kirkpatrick et al., **“Overcoming Catastrophic Forgetting in Neural Networks”** (2017). [PNAS/DOI](https://doi.org/10.1073/pnas.1611835114) | Elastic Weight Consolidation uses a Fisher-information-weighted quadratic penalty to protect parameters estimated to matter for previous tasks. | Importance-weighted retention is an alternative to explicit structured memory. | EWC protects distributed parameters; it does not maintain typed records, provenance, executable abstractions, or graph-local belief revision. | **Moderate**: X65A must compare against parameter-regularization baselines before attributing retention to its memory design. |
| David Lopez-Paz and Marc'Aurelio Ranzato, **“Gradient Episodic Memory for Continual Learning”** (2017). [NeurIPS paper page](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) | Stores episodic examples and constrains new gradients so they do not increase loss on remembered tasks; also formalizes backward and forward transfer measures. | Explicit episodic retention, bounded memory, interference control, and continual-transfer metrics overlap directly. | GEM's memory is an exemplar buffer and its constraint acts in parameter space; it lacks semantic/procedural typing, dependency revision, MDL macros, and durable structured state. | **High** for any broad claim that explicit episodic memory plus constrained updating is new. |
| Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim, **“Continual Learning with Deep Generative Replay”** (2017). [NeurIPS paper page](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html) | Trains a generator to reproduce prior-task pseudo-examples and interleaves them with current data while updating a solver. | Consolidated/generative replacement for raw replay is a direct alternative to X65A's compressed retained structure. | Generated examples need not preserve explicit provenance or executable semantics, and the method does not certify abstractions with a verifier. | **Moderate–high**: consolidation must beat strong replay and generative-replay controls at equal memory cost. |
| Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, and Richard E. Turner, **“Variational Continual Learning”** (2018). [OpenReview](https://openreview.net/forum?id=BkQqq0gRb) | Uses the previous approximate posterior as the prior for the next task and performs online variational inference, optionally with coresets. | Sequential Bayesian belief updating and retained sufficient information overlap with X65A's Bayesian component posterior. | VCL is approximate posterior inference over model parameters, not exact finite inference over typed reusable components or a provenance graph. | **High** against any claim that sequential Bayesian continual updating is new. |
| Soochan Lee, Junsoo Ha, Dongsu Zhang, and Gunhee Kim, **“A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning”** (2020). [arXiv](https://arxiv.org/abs/2001.00689) | A Bayesian nonparametric mixture allocates data to experts and expands the set of components as task regimes change, without task labels. | Persistent latent-component assignment, hierarchical sharing, and expansion closely resemble a Bayesian component memory. | Inference is neural/approximate and the components are experts, not typed evidence, claims, revisions, or verifier-certified programs. | **High**: latent reusable Bayesian components and adaptive component growth are established. |
| Samuel Kessler et al., **“Hierarchical Indian Buffet Neural Networks for Bayesian Continual Learning”** (2021). [PMLR](https://proceedings.mlr.press/v161/kessler21a.html) | Places a hierarchical Bayesian nonparametric prior over shared latent neural structure, permitting selective feature reuse and resource allocation across tasks. | Hierarchical latent components and controlled sharing are close to the statistical layer of X65A. | It does not provide exact finite inference or X65A's typed, provenance-bearing memory and verifier loop. | **High** for the hierarchical-Bayesian-sharing ingredient. |
| Soochan Lee, Hyeonseong Jeon, Jaehyeon Son, and Gunhee Kim, **“Learning to Continually Learn with the Bayesian Principle”** (2024). [PMLR](https://proceedings.mlr.press/v235/lee24j.html) | Learns a representation and performs sequential statistical Bayesian updates whose ideal form matches batch updating under the paper's assumptions. | Directly overlaps the claim that Bayes-consistent state can accumulate evidence without parameter forgetting. | Its retained statistical models are not a typed cognitive memory, and its guarantees do not imply lossless retention under bounded arbitrary histories. | **High** against treating exact/sequential Bayes as a novel continual-learning principle. |

### Research conclusion from this cluster

Exact finite Bayes is mathematically simpler than most of these systems, not newer. Its legitimate value would be auditability: finite sums, explicit support, exact normalization, inspectable component identities, and reproducible posterior state. That is an engineering and verification advantage, not a novelty claim. Moreover, exactness holds only for the chosen finite hypothesis family. It does not protect against omitted components, likelihood misspecification, dependence errors, or unmodeled distribution shift.

## 4. Complementary systems and typed episodic, semantic, and procedural memory

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| James L. McClelland, Bruce L. McNaughton, and Randall C. O'Reilly, **“Why There Are Complementary Learning Systems in the Hippocampus and Neocortex”** (1995). [Author-hosted paper](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf), [DOI](https://doi.org/10.1037/0033-295X.102.3.419) | Proposes rapid, sparse hippocampal storage of episodes alongside slow, interleaved neocortical integration of structured knowledge. | Fast episodic capture followed by slower semantic consolidation is central to X65A. | CLS is a computational-neuroscience theory, not an executable typed graph with verifier-gated MDL or procedural macro induction. | **Very high** against any claim that dual-rate episodic-to-semantic consolidation is new. |
| Endel Tulving, **“Episodic and Semantic Memory”** (1972). [Bibliographic record for the original chapter](https://cir.nii.ac.jp/crid/1574231874408386176?lang=en) | Distinguishes personally situated episodes from decontextualized semantic knowledge. | Supplies two of X65A's explicit memory types. | It is a psychological taxonomy, not a machine update, retrieval, or revision algorithm. | **Moderate**: the type distinction is foundational prior art. |
| Neal J. Cohen and Larry R. Squire, **“Preserved Learning and Retention of Pattern-Analyzing Skill in Amnesia: Dissociation of Knowing How and Knowing That”** (1980). [Science/DOI](https://doi.org/10.1126/science.7414331) | Empirically dissociates skill learning (“knowing how”) from declarative knowledge (“knowing that”). | Grounds X65A's procedural/declarative separation. | It neither defines a software memory schema nor studies learned executable macro libraries. | **Moderate**: procedural memory as a distinct category is not new. |
| Paul S. Rosenbloom, **“Combining Procedural and Declarative Knowledge in a Graphical Architecture”** (2010). [USC author manuscript](https://ict.usc.edu/pubs/Combining%20Procedural%20and%20Declarative%20Knowledge%20in%20a%20Graphical%20Architecture.pdf) | Represents procedural rules, semantic memory, episodic memory, and constraints in a common factor-graph architecture, using local graphical computations to produce global behavior. | One architecture integrates the same three positive memory types and a probabilistic graphical substrate; this is a direct ancestor of typed graph-local memory. | The reported architecture did not provide X65A's exact finite component posterior, verifier-constrained MDL consolidation, explicit negative/revision records, or causal no-leakage experiment. | **Very high**: typed multi-memory integration on a probabilistic graph is established prior art. |
| John E. Laird and Shiwali Mohan, **“A Case Study of Knowledge Integration Across Multiple Memories in Soar”** (2013). [AAAI primary PDF](https://cdn.aaai.org/ocs/7606/7606-32587-1-PB.pdf) | Demonstrates interactive task learning that coordinates procedural, semantic, and episodic memory within Soar. | Integrated typed memory, skill learning, and cross-memory knowledge use closely overlap X65A's architectural ambition. | Soar's learning and memory semantics differ; it does not implement the proposed Bayes–MDL–verifier conjunction. | **Very high** against a claim that coordinating episodic, semantic, and procedural memory in one continual agent is new. |
| Charles Packer et al., **“MemGPT: Towards LLMs as Operating Systems”** (2023). [arXiv](https://arxiv.org/abs/2310.08560) | Introduces memory tiers and agent-managed movement between limited context and external persistent storage for long-running, multi-session interaction. | Budget pressure, retrieval, durable external memory, and cross-session continuation overlap X65A's persistence goals. | It does not impose X65A's typed evidence schema, exact Bayesian revision, verifier-certified consolidation, or formal no-leakage controls. | **High**: external restart-capable agent memory is not new. |
| Ke Yang et al., **“PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents”** (2026). [arXiv](https://arxiv.org/abs/2603.03296), [author repository](https://github.com/TIMAN-group/PlugMem) | Standardizes episodic trajectories; extracts semantic propositions/concepts and procedural intents/prescriptions; stores episodic, semantic, and procedural memory graphs; preserves links to source episodes; performs abstraction-aware multi-hop retrieval, re-ranking/pruning, updating, evolution, and compression. | This is the closest located collision for typed E/S/P graphs, provenance, consolidation, budgeted retrieval, and cross-session memory in one agent module. | The pipeline uses learned language-model extraction/evaluation rather than exact finite hierarchical Bayes; it does not establish AGM/ATMS-style negative revision, verifier-certified two-part MDL, a formal macro-search theorem, or a causal no-future-information protocol. | **Very high**. X65A must treat PlugMem as a principal baseline and cannot present its typed graph, provenance, or budgeted consolidation alone as novel. |

### Research conclusion from this cluster

The four-way X65A type system is best understood as an explicit engineering ontology assembled from established cognitive and agent-memory distinctions. “Negative/revision memory” may be a useful first-class implementation choice, but negative evidence, failure traces, retractions, justifications, and nogoods all have close antecedents below. The novelty burden therefore lies in exact operational semantics and measured interaction among the types, not in naming them.

## 5. MDL consolidation, program libraries, options, and procedural macro compounding

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| Jorma Rissanen, **“Modeling by Shortest Data Description”** (1978). [DOI](https://doi.org/10.1016/0005-1098(78)90005-5) | Selects models by total description length of the model and data encoded with it, establishing the MDL principle. | X65A's two-part consolidation score is a direct application of this principle. | MDL does not prescribe a verifier, memory ontology, or continual-agent policy. | **Very high** against any novelty claim for description-length consolidation itself. |
| Kevin Ellis et al., **“DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning”** (2021). [PLDI paper page](https://pldi21.sigplan.org/details/pldi-2021-papers/55/DreamCoder-Bootstrapping-Inductive-Program-Synthesis-with-Wake-Sleep-Library-Learnin), [DOI](https://doi.org/10.1145/3453483.3454080), [author PDF](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf) | Alternates program search, library refactoring, and learned recognition; uses Bayesian/MDL pressure to compress solved programs into reusable abstractions, including multilayer libraries that make later synthesis easier. | Bayesian program hypotheses, executable verification by task behavior, MDL consolidation, procedural abstractions, and compounding search reuse are all close to X65A. | DreamCoder is a program-induction system, not a unified episodic/semantic/revision memory with provenance-local belief retraction, restart evaluation, or budgeted retrieval over heterogeneous records. | **Very high**: the core “verified reusable macro selected by compression and then reused to reduce search” story is established. |
| Richard E. Korf, **“Macro-Operators: A Weak Method for Learning”** (1985). [DOI](https://doi.org/10.1016/0004-3702(85)90012-8) | Compiles sequences of primitive operators into macro-operators and analyzes when macros reduce search and when they fail due to domain structure. | Direct antecedent for procedural macro formation and compounding search reduction. | It lacks probabilistic memory and modern verifier/MDL machinery. Crucially, it warns that macros can enlarge branching or fail without favorable decomposability. | **High**: procedural compounding and its failure modes are old. |
| Richard S. Sutton, Doina Precup, and Satinder Singh, **“Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning”** (1999). [DOI](https://doi.org/10.1016/S0004-3702(99)00052-1) | Defines an option by an initiation set, internal policy, and termination condition and supplies SMDP planning/learning semantics for composing temporally extended actions. | Typed callable procedures with preconditions and postconditions are structurally close to X65A procedures/macros. | Options optimize expected return rather than MDL and do not supply typed declarative memory or provenance revision. | **High** against novelty claims for reusable, composable skill abstractions. |
| Jorge A. Mendez and Eric Eaton, **“Lifelong Learning of Compositional Structures”** (2021). [OpenReview](https://openreview.net/forum?id=ADWd4TJO13G), [author PDF](https://lifelongml.seas.upenn.edu/papers/Mendez2021Lifelong.pdf) | Learns a library of reusable components across tasks, first assimilating a new task by composing existing modules and then accommodating it by adapting or adding components. | Continual compositional transfer, selective component reuse, and controlled library expansion overlap the latent-component and macro layers of X65A. | Components are learned model modules rather than verifier-certified symbolic records with explicit episodic/provenance/revision semantics. | **High**: compositional lifelong transfer through a growing reusable library is established. |
| Saman Forouzandeh et al., **“Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement”** (2025 preprint). [arXiv](https://arxiv.org/abs/2512.18950) | Stores structured procedures with goals, preconditions, actions, and postconditions; uses Bayesian reliability estimates to select them; contrasts successes and failures; and merges/abstracts procedures into a hierarchy. | Hierarchical procedural memory, Bayesian selection, positive/negative evidence, and compounding abstraction form a direct partial bundle of X65A. | It does not integrate exact joint finite Bayes over all memory types, a general provenance/dependency revision graph, verifier-certified MDL, or causal restart/no-leakage tests. | **Very high**, especially for claims about Bayesian procedural reliability or learning from failure traces. |

### Macro-compounding caveat

A reduction from enumerating all primitive sequences of length \(L\) to all macro sequences of length \(m<L\) changes a simple full \(d\)-ary enumeration count from \(d^L\) to \(d^m\), a ratio \(d^{L-m}\). That identity does **not** prove an end-to-end exponential speedup for a realistic synthesizer. Library lookup, macro matching, type checking, verification, branching induced by extra macros, partial observability, and failed abstraction attempts can dominate. Korf's analysis and DreamCoder's measured search/library loop are the relevant prior standards. X65A should report actual search nodes, wall time, verifier calls, and memory overhead, not only the toy enumeration ratio.

## 6. Belief revision, truth maintenance, provenance, and local probabilistic updating

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| Carlos E. Alchourrón, Peter Gärdenfors, and David Makinson, **“On the Logic of Theory Change: Partial Meet Contraction and Revision Functions”** (1985). [DOI](https://doi.org/10.2307/2274239) | Gives rational postulates and partial-meet constructions for contraction and revision of belief sets. | Formal retraction and revision of stored semantic claims overlap X65A's negative/revision memory. | AGM is qualitative and deductively closed; it does not provide probabilities, provenance execution, or resource-bounded agent memory. | **High** against novelty claims for principled belief revision. |
| Johan de Kleer, **“An Assumption-Based TMS”** (1986). [DOI](https://doi.org/10.1016/0004-3702(86)90080-9), [author PDF](https://www.dekleer.org/Publications/An%20Assumption-Based%20TMS.pdf) | Tracks assumption sets supporting propositions, maintains multiple contexts, derives nogoods, and identifies inconsistent environments without globally discarding all dependent beliefs. | Dependency/provenance graph, negative evidence, localized invalidation, and support-aware revision are close architectural ancestors. | ATMS is symbolic rather than probabilistic and does not use MDL or a learned procedural library. | **Very high**: dependency-aware revision and explicit negative inconsistency records are established. |
| Todd J. Green, Grigoris Karvounarakis, and Val Tannen, **“Provenance Semirings”** (2007). [DOI](https://doi.org/10.1145/1265530.1265535), [author PDF](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf) | Annotates data with semiring expressions that record alternative and joint derivations and supports principled propagation of provenance through positive relational queries. | Formal derivation lineage is directly relevant to X65A's provenance graph and dependency-aware invalidation. | The framework is for query provenance and positive relational algebra, not uncertain agent belief revision or procedural verification. | **High** against claims that derivation-carrying facts or algebraic provenance propagation are new. |
| Steffen L. Lauritzen and David J. Spiegelhalter, **“Local Computations with Probabilities on Graphical Structures and Their Application to Expert Systems”** (1988). [DOI](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x), [author PDF](https://www.stats.ox.ac.uk/~steffen/papers/LS88.pdf) | Converts graphical models into junction-tree form and performs exact probabilistic inference through local message passing; computational cost is controlled by clique size/treewidth. | Supplies the mathematical foundation for exact local propagation and factorization-delimited revision. | “Local computation” does not mean only nearby beliefs change. Evidence can propagate throughout a connected graph, and exact inference may be exponential in treewidth. | **Very high**: graph-local exact Bayes is established, and its conditions materially restrict X65A wording. |
| Tianyu Yang et al., **“TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory”** (2026 preprint). [arXiv](https://arxiv.org/abs/2606.25161) | Learns write/revise/delete memory transitions; a Memory Transition Verifier evaluates coverage, preservation, and faithfulness; preference/RL training favors trustworthy consolidation. | Verifier-constrained consolidation, revision operations, and long-term memory directly overlap a central X65A headline. | It does not use X65A's typed provenance graph, exact finite Bayes, explicit two-part MDL gate, or macro-compounding theorem/evaluation. | **Very high**: verifier-gated consolidation is not independently novel. |
| Haoyu Wang et al., **“MemGuard: Persisting Verifier Signals for LLM-Agent Memory Governance”** (2026 preprint). [arXiv](https://arxiv.org/abs/2608.21867) | Persists verifier-derived reward, confidence, labels, and uncertainty through retrieval, conflict resolution, summarization, and archival so later memory operations retain trust metadata. | Persistent verifier evidence, conflict-aware update, provenance-like trust metadata, and governed consolidation overlap X65A. | It does not provide exact finite hierarchical Bayes, the same typed cognitive schema, an AGM/ATMS semantics, or a preregistered no-leakage proof. | **Very high**: verifier signals surviving consolidation and retrieval are directly covered. |

### The exact locality claim X65A may make

Suppose the joint prior factorizes as

\[
p(x,y)=p_X(x)p_Y(y)
\]

and new evidence has likelihood \(p(e\mid x,y)=p(e\mid x)\). If the evidence has positive marginal probability, then

\[
p(y\mid e)=p_Y(y).
\]

This is a valid and useful independence lemma. It does **not** justify the unrestricted statement that “revision remains graph-local.” In a coupled model, changing evidence for one node may alter every connected posterior marginal. Exact updates can require junction-tree propagation, and worst-case cost is exponential in treewidth. The research claim must therefore say: unaffected marginals are preserved only when the encoded conditional independences and evidence locality establish d-separation/factorization; otherwise propagation is allowed to be global.

## 7. Budgeted retrieval, submodular selection, and knapsack caveats

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, **“An Analysis of Approximations for Maximizing Submodular Set Functions—I”** (1978). [DOI](https://doi.org/10.1007/BF01588971) | Establishes the classical greedy approximation for normalized monotone submodular maximization under a cardinality constraint. | Provides a possible guarantee for selecting a bounded set of memories when utility has diminishing returns. | Arbitrary learned relevance, contradiction resolution, dependency closure, and macro synergy need not be monotone or submodular. | **High**: any greedy guarantee must be inherited from proved utility structure, not asserted from the use of a budget. |
| Samir Khuller, Anna Moss, and Joseph Naor, **“The Budgeted Maximum Coverage Problem”** (1999). [DOI](https://doi.org/10.1016/S0020-0190(99)00031-9) | Studies weighted maximum coverage under item costs and gives an approximation algorithm for this NP-hard budgeted problem. | Models memory retrieval when records have unequal costs and cover useful evidence/features. | Coverage is a special monotone-submodular objective; it does not model arbitrary posterior value or dependent record bundles. | **Moderate–high**: explicit-cost retrieval is a known combinatorial optimization problem. |
| Maxim Sviridenko, **“A Note on Maximizing a Submodular Set Function Subject to a Knapsack Constraint”** (2004). [DOI](https://doi.org/10.1016/S0167-6377(03)00062-2) | Gives a \(1-1/e\) approximation for nonnegative monotone submodular maximization under one knapsack constraint using partial enumeration plus greedy completion. | This is the closest formal basis for cost-aware budgeted retrieval with a diminishing-returns utility. | Plain relevance-to-cost greedy is not the theorem's algorithm, and multiple budgets, prerequisite closure, negative utilities, or non-submodular interactions invalidate the guarantee. | **High**: X65A must state and test the exact assumptions before citing a knapsack approximation. |

### Retrieval conclusion

“Budgeted retrieval” is not itself a new mechanism. A defensible X65A theorem must define a set utility \(F(S)\), show whether it is normalized, nonnegative, monotone, and submodular, specify the exact cost constraint, and use an algorithm with the corresponding guarantee. If verifier dependencies require retrieving parent chains, or if two memories are useful only jointly, the feasible family and utility may violate standard one-knapsack assumptions. In that case X65A should report a heuristic and empirical regret/utility comparison, not a classical approximation ratio.

## 8. Bounded-memory impossibility and capacity limits

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| Jeremias Knoblauch, Hisham Husain, and Tom Diethe, **“Optimal Continual Learning Has Perfect Memory and Is NP-Hard”** (2020). [PMLR](https://proceedings.mlr.press/v119/knoblauch20a.html) | Under the paper's general continual-learning formulation, establishes computational hardness and conditions under which optimal continual learning requires perfect memory. | Directly limits any claim that a bounded structured store can be universally lossless. | The result is tied to its formal notion of optimality and problem class; it does not imply every practical task needs full replay. | **Very high** as a boundary condition, not as a design collision. |
| Xi Chen, Christos H. Papadimitriou, and Binghui Peng, **“Memory Bounds for Continual Learning”** (2022). [arXiv](https://arxiv.org/abs/2204.10830), [FOCS DOI](https://doi.org/10.1109/FOCS54457.2022.00056) | Derives memory lower bounds in a PAC continual-learning setting, including linear dependence on the number of tasks for one-pass learners in the studied regime and benefits from additional passes/improper learning. | Gives formal lower-bound context for X65A's memory budget and consolidation tradeoffs. | It studies a particular theoretical learning model rather than X65A's program/evidence graph. | **High**: bounded memory cannot be advertised as universally retaining all task-relevant history. |

### Relation to the finite pigeonhole theorem

If the set of possible histories is larger than the set of memory states, no memory encoder can be injective. That finite pigeonhole fact is correct but weak: it says some histories collide, not that the collision changes any task-relevant prediction. A meaningful X65A lower bound should construct a task family in which collided histories require different future actions or posterior predictions. Knoblauch et al. and Chen et al. show what stronger problem-dependent statements look like. The appropriate X65A claim is therefore: bounded state cannot preserve arbitrary histories exactly; success depends on task-relevant sufficient structure, approximation tolerance, and the tested distribution.

## 9. Long-lived agents, continual memory evaluation, persistence, and leakage

| Primary source | Exact mechanism | Overlap with X65A | Key difference | Novelty threat |
|---|---|---|---|---|
| Qisheng Hu, Quanyu Long, and Wenya Wang, **“When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents”** (2026 preprint). [arXiv](https://arxiv.org/abs/2604.27003) | Uses controlled sequential agent environments to study external experience reuse; distinguishes raw traces from procedural abstractions and measures transfer, forgetting, retrieval pollution, context competition, and memory dilution. | Directly overlaps continual agent memory, procedural consolidation, compositional transfer, and the need to separate apparent forward transfer from retained competence. | It does not implement X65A's exact typed Bayesian provenance graph or verifier-MDL gate. | **Very high** for experimental framing and failure analysis. X65A should reproduce its strongest relevant controls. |
| Yiheng Shu et al., **“AGENTCL: Toward Rigorous Evaluation of Continual Learning in Language Agents”** (2026 preprint). [arXiv](https://arxiv.org/abs/2606.02461) | Constructs controlled compositional streams in which prior subsolutions, evidence, and workflows can be reused; evaluates transfer and memory methods, including filtering unreliable consolidation. | Closely overlaps compositional continual-agent evaluation, reusable skills/evidence, and reliability-aware memory updates. | It is an evaluation framework rather than the same exact memory formalism and does not by itself prove causal freedom from every leakage channel. | **Very high**: X65A cannot claim rigorous compositional continual-memory evaluation without direct comparison. |
| Shachar Kaufman, Saharon Rosset, and Claudia Perlich, **“Leakage in Data Mining: Formulation, Detection, and Avoidance”** (2011). [DOI](https://doi.org/10.1145/2020408.2020496), [institutional record](https://cris.tau.ac.il/en/publications/leakage-in-data-mining-formulation-detection-and-avoidance-2/) | Formalizes target leakage as access to information unavailable at legitimate prediction time and analyzes leakage detection and prevention. | Direct foundation for X65A's no-future-target constraint and pipeline audit. | It does not supply X65A's memory architecture; more importantly, excluding literal labels is only one leakage defense. | **High** against treating no-leakage methodology as novel. |
| A. P. Dawid, **“Statistical Theory: The Prequential Approach”** (1984). [DOI](https://doi.org/10.2307/2981683) | Evaluates a sequence of probabilistic forecasts using only information available before each outcome, then accumulates predictive evidence over time. | Provides the correct temporal ordering for continual-memory evaluation: predict/commit first, reveal outcome second, then update memory. | It is a statistical evaluation principle rather than a memory implementation. | **High** for the causal-temporal evaluation protocol. |
| C. Mohan, Don Haderle, Bruce Lindsay, Hamid Pirahesh, and Peter Schwarz, **“ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging”** (1992). [DOI](https://doi.org/10.1145/128765.128770), [IBM Research page](https://research.ibm.com/publications/aries-a-transaction-recovery-method-supporting-fine-granularity-locking-and-partial-rollbacks-using-write-ahead-logging) | Uses write-ahead logging, repeating history during redo, and logged compensation for recovery after failure. | Establishes mature mechanisms and validation concerns for durable state and restart recovery. | It concerns database transactions, not semantic equivalence of learned agent memory. | **Moderate**: persistence across restart is an engineering requirement, not an algorithmic novelty. |

### What a genuine no-leakage result requires

The finite predicate “every stored pair \((t,a)\) has \(t<t_{\text{prediction}}\)” can mechanically rule out storing the literal future pair. It cannot rule out:

- future labels encoded through aliases, hashes, embeddings, summaries, cached gradients, or derived macros;
- task-generator state, random seeds, task identifiers, file order, timestamps, or metadata correlated with the future answer;
- evaluator artifacts or hand-authored rules that implicitly reveal the held-out target;
- retrieval from a store populated by a later run;
- tuning on final streams, threshold leakage, or post-hoc changes after inspecting outcomes.

An evidence-honest X65A evaluation should therefore use an append-only event log and enforce this order for every item:

1. instantiate the current memory snapshot and record its hash;
2. present only the legal observation prefix;
3. retrieve, predict, and commit the answer or abstention;
4. record the prediction and retrieval trace;
5. only then reveal verifier feedback or the target;
6. update memory and record every new node's parent observations, timestamps, verifier version, and code hash.

The test generator, evaluator, update algorithm, memory schema, thresholds, and ablations should be frozen before untouched streams are sampled. At minimum, compare persistent memory with empty-memory, no-update, shuffled-history, wrong-agent-history, raw-replay, and process-restart arms. A process-restart check must deserialize a committed snapshot into a fresh process and verify both byte/schema integrity and equality of specified predictive/retrieval behavior on a frozen probe suite.

This makes the evaluation *causally informative*; it still cannot prove the absence of every covert channel. The claim should remain “passed the enumerated leakage audits and interventions,” not “leakage is impossible.”

## 10. Direct integration collision matrix

Legend: **●** directly present; **◐** partial or close analogue; **—** not located in the cited source. A mark describes the published mechanism, not merely terminology.

| Component | Rosenbloom / Soar | DreamCoder | PlugMem | MACLA | ATMS + graphical inference | TRUSTMEM / MemGuard | Hu / AGENTCL | Proposed X65A conjunction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Exact finite hierarchical Bayesian latent components | ◐ | ◐ | — | ◐ | ◐ | — | — | ● |
| Typed episodic + semantic + procedural memory | ● | — | ● | procedural only | — | ◐ | ◐ | ● |
| First-class negative/revision records | ◐ | — | ◐ | ● | ● | ● | ◐ | ● |
| Explicit provenance/dependency graph | ◐ | ◐ | ● | ◐ | ● | ◐ | ◐ | ● |
| Verifier-constrained two-part MDL consolidation | — | ◐ | ◐ | — | — | verifier, not MDL | — | ● |
| Budgeted retrieval | ◐ | search budget | ● | ● | — | ◐ | ● | ● |
| Factorization-delimited probabilistic revision | ●/◐ | — | — | Bayesian reliability only | ● | ◐ | — | ● |
| Reusable procedural macros that compound | ●/◐ | ● | ●/◐ | ● | — | — | ◐ | ● |
| Durable restart persistence test | — | ◐ | ●/◐ | ◐ | — | ◐ | ◐ | ● |
| Preregistered causal no-future-information evaluation | — | — | — | — | — | — | ◐ | ● |

No column other than the proposed X65A conjunction contains every mark. However, the table also shows why a broad novelty claim would be misleading: most rows have multiple strong antecedents, and PlugMem, DreamCoder, Rosenbloom/Soar, MACLA, and the verifier-memory papers collectively leave a narrow integration delta.

## 11. Strongest novelty threats and required comparisons

### 11.1 PlugMem is the nearest architecture-level baseline

If X65A is evaluated without PlugMem or a faithful component-matched reimplementation, any gain could be due merely to typed memory graphs, source links, abstraction-aware retrieval, or compression—all already bundled by PlugMem. The decisive X65A ablation is not “structured memory versus no memory”; it is PlugMem-like typed provenance memory versus the same system plus exact finite Bayes, explicit revision dependencies, verifier-certified MDL, and the prequential leakage controls.

### 11.2 DreamCoder captures the compression-to-library mechanism

DreamCoder substantially occupies the territory of finding executable programs, refactoring them into a compressed library, and using the library to improve future search. X65A's remaining difference is cross-type continual memory and revision, not the basic MDL macro loop. Compare compression achieved, downstream search reduction, negative transfer, verification cost, and whether learned abstractions survive contradictory later evidence.

### 11.3 Rosenbloom, Soar, and ATMS block “first integrated cognitive memory” claims

Typed procedural, semantic, and episodic stores have already been integrated in cognitive architectures; factor graphs and truth-maintenance dependencies have already supported local computation and support-aware revision. X65A can still contribute a cleaner finite probabilistic semantics and an auditable empirical protocol, but it cannot claim to originate multi-memory cognitive integration.

### 11.4 MACLA, TRUSTMEM, and MemGuard narrow the recent agent-memory delta

MACLA covers hierarchical procedural abstractions selected with Bayesian reliability and refined from success/failure contrast. TRUSTMEM covers verifier-evaluated consolidation transitions. MemGuard covers persistence of verifier confidence and uncertainty through memory operations. X65A must distinguish its exact mathematical conjunction and measured consequences rather than relabeling these mechanisms.

### 11.5 AgentCL and Hu et al. raise the evaluation bar

Positive transfer on later tasks is insufficient if task repetition, retrieval contamination, or easy stream structure explains it. The strongest evaluation should show:

- forward transfer on genuinely new compositions;
- retention after interfering tasks;
- improvement eliminated by memory shuffling or deletion;
- gains surviving process restart;
- raw replay, parameter-only continual learning, and current structured-memory baselines at matched cost;
- benefits attributable separately to Bayes, typing, provenance, MDL, verifier gating, revision locality, and macros;
- calibrated abstention or revision when old memories become false;
- full accounting of storage, retrieval, verification, and consolidation cost.

## 12. Evidence-honest novelty delta

### Clearly not novel

The following must not be claimed as new by themselves:

- continual or lifelong learning;
- catastrophic-forgetting mitigation through replay, regularization, or retained state;
- complementary fast episodic and slow semantic learning;
- distinctions among episodic, semantic, and procedural memory;
- Bayesian continual updating or reusable latent components;
- MDL/model-compression objectives;
- reusable program libraries, macro-operators, options, or compositional skills;
- provenance and dependency graphs;
- belief revision, truth maintenance, nogoods, or support-based retraction;
- exact local inference in suitably factorized graphical models;
- submodular or knapsack-constrained selection;
- persistent external memory and restart recovery;
- verifier-gated memory updates or persistent verifier metadata;
- prequential prediction-before-update evaluation and target-leakage controls.

### Provisional remaining integration delta

The narrow defensible statement is:

> In this bounded primary-source audit, no located work simultaneously implements and empirically evaluates exact finite hierarchical Bayesian latent-component updating; first-class episodic, semantic, procedural, and negative/revision records; explicit provenance/dependency tracking; verifier-gated two-part-MDL consolidation; cost-bounded retrieval; factorization-delimited probabilistic revision; verified procedural macro reuse; durable restart; and preregistered no-future-information interventions in one continual agent.

This statement reports a search result, not a proof of novelty. It should be prefixed by “in this bounded audit” wherever reused. It does not establish priority, patentability, non-obviousness, or scientific value. The integration becomes a research contribution only if its conjunction produces a measured advantage that cannot be reproduced by simpler subsets or strong prior systems.

### Conditions that would eliminate even the provisional delta

The integration claim should be rejected or sharply narrowed if any of the following occurs:

1. A primary source is found that operationalizes substantially the same conjunction.
2. The exact Bayesian layer does not outperform or calibrate better than a simpler count/reliability model.
3. Typed stores do not outperform an untyped provenance graph at equal budget.
4. Verifier-constrained MDL does not beat verification-only or MDL-only consolidation.
5. Graph-local revision either violates its independence assumptions or gives no efficiency/accuracy benefit.
6. Macro reuse reduces a toy enumeration count but not actual search time or verifier calls.
7. Gains vanish under new compositions, process restart, or memory-shuffle controls.
8. Future-derived information enters through any unlogged channel.
9. PlugMem-, DreamCoder-, MACLA-, replay-, and parameter-regularization baselines explain the measured effect.

## 13. Recommended claim language for the X65A research record

**Acceptable:**

> X65A investigates a provisionally underexplored integration of established continual-learning, cognitive-memory, Bayesian-inference, truth-maintenance, MDL program-library, verifier-governance, and budgeted-retrieval mechanisms. A bounded primary-source audit did not locate a measured system containing the full conjunction. Novelty, if any, is therefore conditional on the integrated mechanism and its ablation-supported empirical behavior, not on any constituent technique.

**Not supported:**

> X65A introduces structured continual memory, Bayesian lifelong learning, verifier-based consolidation, graph-local revision, procedural macro learning, or leakage-free continual evaluation.

Each phrase in the unsupported sentence is already occupied by prior art or is too broad to establish from the present evidence.

## 14. Search limits

This audit was bounded by time, accessible indexing, and the stated mechanism vocabulary. It inspected primary material available through publisher/DOI pages, official conference archives, PMLR, NeurIPS, CVF/OpenReview/AAAI where relevant, arXiv, institutional author pages, and author repositories through **2026-08-27**. It used targeted keyword and known-lineage searches rather than a complete forward/backward citation graph.

It did **not** exhaustively search:

- patents or freedom-to-operate databases;
- dissertations, workshop-only manuscripts, technical reports, or books without accessible primary records;
- all non-English literature;
- every cognitive architecture, database provenance system, process-mining system, or case-based reasoning system;
- every 2025–2026 agent-memory preprint, a rapidly changing literature;
- unpublished industrial systems or repositories without papers;
- alternative terminology that may describe the same conjunction without using “structured continual memory.”

Recent preprints were included because they are direct technical collisions, but their claims may change during review and should be rechecked before a public novelty statement. Absence from this audit is not evidence of absence. A publication-facing claim requires a refreshed systematic search, citation chaining from the strongest collisions, and expert review immediately before submission.

## Bottom line

**Evidence-honest conclusion:** all X65A ingredients have clear prior art, and several recent systems cover large subsets. The full measured conjunction was not located in this bounded audit, leaving a defensible but narrow and provisional systems-integration hypothesis. The research should proceed only as a falsifiable comparison of that conjunction against PlugMem-like typed provenance memory, DreamCoder-like MDL library learning, MACLA-like Bayesian procedural memory, verifier-governed memory, replay/regularization baselines, and rigorous continual-agent evaluation. No categorical novelty claim is warranted.
