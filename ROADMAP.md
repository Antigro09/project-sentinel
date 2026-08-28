# Roadmap: X63 - X73 plus the Phase-2 world-model track

**Status: this is the plan of record.** Adopted 2026-08-25, after X62.

**Phase-2 amendment adopted 2026-08-28.** The historical X63–X73 capability
labels remain in force. A parallel architecture track now introduces learned,
action-conditioned latent world models earlier, without treating model size as
a milestone and without replacing the exact Sentinel reference.

Canonical Phase-2 documents:

- `SENTINEL-HYBRID-WORLD-MODEL-STRATEGY.md`;
- `docs/phase-2-continuous-world-model/`;
- `math-findings/01 Long-Horizon Credit Assignment + Compositional World Modeling/SHWM Action-Conditioned Hybrid World Model/`.

Branch topology:

```text
phase-1-exact-reference          exact committed reference at 5205543
phase-1-verifier                ongoing exact X64/X65 evidence line
phase-2-continuous-world-model  learned latent architecture line
```

The Phase-2 branch started at the same committed reference. Unfinished files in
the active Phase-1 checkout were not absorbed into the freeze or Phase 2.

It supersedes the five-gap plan (`~/.claude/plans/alright-can-we-make-validated-toast.md`),
which is reconciled at the bottom rather than discarded -- three of its five
gaps are closed and the other two reappear here under new numbers.

## The thesis of this roadmap

After X62, the project stops expanding the toy interpreter one primitive at
a time and starts crossing the gaps that actually separate this system from
human-level intelligence.

The immediate next step is **not** POSIX, not a larger model, and not
unrestricted self-modification. It is to remove the strongest remaining
source of human supervision: **we currently define what the task means by
supplying a target program or target behaviour.** The README says this in
its own words -- "not label-free discovery" -- and X64 is the experiment
that fixes it.

One line:

> X62 memory audit -> X63 minimal memory hierarchy -> X64 task/goal
> induction -> X65 lifelong learning -> X66 real coding -> X67 cross-domain
> transfer -> X68 uncertain reasoning -> X69 grounded causal models ->
> X70 multimodality -> X71 long-horizon agency -> X72 controlled
> self-improvement -> X73 integrated AGI evaluation.

The architecture amendment adds a second dependency line:

> SHWM Scale 0 pipeline -> Scale 1 representation contest -> Scale 2 world
> model contribution -> Scale 3 verifier contribution -> Scale 4 causal audit
> -> Scale 5 environment transfer -> rejoin X65A/X65B at Scale 6 continual
> learning -> Scale 7 GUI/code/X66 -> Scale 8 parameter/data study.

This is deliberate parallelism. Scales 0–5 may progress while the exact memory
line is audited. Scale 6 cannot claim continual learning until X65A and the
minimal X65B scoped-transfer contract pass. Scale 7 cannot claim real
software-engineering transfer until both lines pass.

Two unsupported premises are explicitly rejected:

- there is no accepted 15B–70B theoretical lower bound for AGI in the bounded
  primary-source audit;
- there is no established requirement for a continuous 1,024-dimensional
  latent.

Parameters, data, horizon, and modality diversity remain independent variables.
Continuous, discrete, and hybrid latents are matched experimental arms.

The ordering is a **dependency** order, not a priority order. Each step
supplies something the next one consumes.

---

## X63 - Implement only the memory architecture X62 proves necessary

X62 was an audit, so its result had to produce a concrete decision. It did,
and the branch is already taken:

- streaming 2/2, register 2/2, stack 2/2 -- **expressible**
- set **0/2**, associative **0/1** -- **not expressible**

The pre-registered rule fired: repeated failure on associative lookup,
arbitrary-distance sets, and symbol tables means **add an exact sparse
key-value store**. Freeze the head + stack + registers design as-is; add
nothing else.

Explicitly ruled out:

- **No fully enumerated writable tape.** Priced at 7,338 GB per behaviour.
- **No mechanism added because it looks "more general."** The ablation
  standard holds: a new primitive earns its place by a measured win on
  held-out tasks (EQTOP failed 0/4 and was cut; SAME passed 3/6; the
  register passed 4/5).

The store represents only keys and values **actually touched during
execution**. Candidates are checked by concrete execution plus
counterexample-guided refinement -- not by tabulating every reachable
memory configuration.

**This is the load-bearing change, and the first thing to price.** Every
experiment since X47 leans on a behaviour *table*: a candidate is evaluated
by looking up a precomputed row, which is what makes the search cheap
(0.107 MB per behaviour, ~10^6 evaluations per task). A sparse store's
state space is defined by the keys *touched* rather than the keys
*possible*, so that table may not survive. Measure the cost of concrete
execution against the table before building anything on top of it. If
CEGIS-with-execution is more than ~50x slower per candidate, the search
budgets of X58-X62 are gone and the whole approach needs re-planning, not
patching.

**X63a is done, and it priced the wrong risk.** Execution is **35x
faster** than the step the frontier actually pays (6.3 us vs 218 us), and
the store costs nothing on top. The table was never a speed device -- it is
a *resolution* device, bought with memory. What dies is the **gradient**:
the full signature separates 211 behaviours where outputs separate 22, and
across four calibration metrics the best output correlation with table
agreement is r = 0.190 at 8 of 123 levels. Restricting to the 17% of
situations a run can actually reach leaves 129 classes, so that resolution
is real rather than agreement on dead states.

**Therefore X63b is counterexample-guided, not distance-ranked:** localise
the first divergence between output and evidence and repair there (X57's
mechanism), which needs no global gradient. Emission also becomes
byte-valued -- index emission would force the store to hold *positions*,
which grow with the tape, reintroducing exactly the unboundedness the store
was meant to avoid.

**X63b/c are done, against an externally specified twelve-clause gate.**
The mechanism passes all of it: set 2/2, associative 1/1, held-out split by
axis (longer / unseen symbols / unseen keys / unseen values) 5/5, ablation
kills exactly those three tasks and no others, 1.06x runtime across a
1,000x key universe, no capacity bound at 200 keys, `reverse` preserved as
a control. Two clauses caught real defects: the differential test **passed
while being vacuous** until it was given a calibration arm, and the
**search regresses** against X62 on two tasks.

**Gate.** Solves held-out tasks requiring registers, nesting,
arbitrary-distance lookup, and references, **and** correctly reports tasks
outside its memory class -- keeping X62's three-way split of
*not-expressible / expressible-but-not-found / found*.

**Status: the memory half passes, the identification half does not, and
that is the finding.** CEGIS finds 10 of 10 and only 3 generalise. Four
explanations, four experiments: thin evidence is **not monotone and nearly
flat** (52 -> 56 -> 56 of 100 while the evidence doubles twice, two tasks
climbing to 10 and two collapsing); no simplicity bias is **refuted**
(56 -> 55); a missing shape is **refuted** -- forcing the witness's exact
shape still gives 0/10 on the task it was built for, and lowers the total
to 45; a weak search is **refuted** (10/10 found).

What remains is that **fitting the evidence does not identify the
program**. A correct program exists in exactly the shape the search was
handed and the search returns a different one. That is X64's subject
reached from the other direction, and the same thing
`measure_identifiability.py` reports about `ordered_targets`.

Which sharpens X64's design: a version space over *interpretations* is not
an enrichment of this machine, it is the missing piece this machine's
failure names. The four states it must distinguish map directly onto what
X63b conflates -- "my implementation is wrong" is the only one it can
currently represent, and "the request is underspecified" is the one that
was true 7 times out of 10.

This completes the substrate foundation. On its own it does not move the
project substantially toward AGI, and should not be described as if it did.

## X64 - Remove the hand-authored target oracle

**The most important capability transition on this list.**

**X64A is done and passes all eight of its gates** (`x64a_identify.py`,
`x64a` section of the README). Nine of eleven tasks are UNDERSPECIFIED by
the demonstrations alone -- the state X63 could not represent and answered
from anyway. Disagreement-maximising clarification queries reach 9/11
answered and 90/110 held-out in 30 queries, against random queries' 7/11 and
79.6 +/- 6.8 over 24 seeds, and match an oracle allowed to know the answers
in advance. Four of the eight gates were structurally unfailable and are
calibrated against two known-bad arms (`reckless`, which is X63's behaviour,
and `paranoid`). The honest limit is recorded: with nothing seeded, one task
is identified confidently and wrongly, because a system cannot see the
outside of its own hypothesis pool.

**X64B-1 and X64B-2 are done and pass all their gates** (9/9 and 10/10;
`x64b1_openworld.py`, `x64b2_language.py`).

B-1 attacks X64A's own limit -- that a singleton version space implies
uniqueness inside the hypothesis class and nothing more. Confirmation on
challenge inputs, an explicit none-of-the-above, and an expansion ladder
whose recovering rung MEASURES what was missing. Headline: **richer pools
can create wrong singletons that poorer pools avoid by reporting
inconsistency** -- 0, 0, 1, 0, 2 across the rungs, not monotonic, with the
richest rung tested producing the most. Expansion buys recall and can cost
safety at the same time.

B-2 maps WORDS to behavioural predicates rather than phrases to task
labels; an instruction means the conjunction of its words' constraints.
Language and demonstrations each reach 10/11 alone and 7/11 together in
silence: what language buys is QUESTIONS -- 8 against 14, fewest of any
policy without future knowledge. Three lexical corrections were forced by
the gates, including one word that alone selected a single behaviour, which
is a task identity wearing a word's clothes.

**X64C froze the lexicon by hash and met unseen compositions: 10/12 gates,
and the two failures fire pre-registered falsifiers.** The query advantage
does not survive -- it reverses, with demonstrations alone answering 10/12
and language cutting that to 3 -- and the gains are confined to the
development families. The cause is measured: 22 of 40 holdout instruction
forms exclude their own target against 0 of 30 on the development set,
because `brackets` carries a sense authored for one task and wrong for
another. Fitting a lexicon to the evaluation suite is what X64B-2 did, and
that is what it cost.

What survives: the failure mode is SAFE (22 false exclusions, 0 confident
errors), conflict detection transfers to unseen pairs (10/12 flagged, 0
forced), confirmation still earns its place (0 confident errors with, 2
without), and unseen FORM is not the problem -- unseen COMPOSITION is.

**Standing claim: Sentinel represents ambiguity, asks
disagreement-maximising questions, detects conflict, abstains when nothing
adequate exists, and fails safely when its language is wrong. It has NOT
been shown to understand language that was not authored around its
evaluation set.** X64 is not closed as evidence of language understanding.

**X64D replaced the authored lexicon with induced senses: 9/10 gates.**
Senses are indexed by (word, role) and induced by clustering the predicate
signatures of development examples; language RANKS candidates and can never
eliminate one, so the target is retained 126/126 against 77/126 for X64C's
authored filter; and the system answers only when the evidence -- not the
language -- has resolved the set, which took confident errors from four to
zero.

D7 fails and the failure is the result: **what cannot eliminate cannot
contradict.** Twelve conflict statistics all sit at chance precision, because
induction by intersection keeps only what examples share and a generic
constraint is satisfied by the wrong task as readily as the right one.
X64B-2 detected conflict because its senses were authored and sharp; X64C
measured what sharp authored senses cost on unseen compositions. D5 and D7
are in tension and this architecture buys D5.

One arm tempers the headline: role-blind joint scores 125/126 in 263
queries against induced joint's 125/126 in 267. The syntactic role earns
nothing measurable here -- roles make polysemy REPRESENTABLE, keeping
alternatives is what makes the system WORK.

**X64E replaced predicate-set senses with a distribution over typed logical
forms: 12/12 gates.** Conflict is posterior mass outside the demonstrations'
consistent set, and it separates at AUROC 0.996 (95% CI 0.986-1.000) where
X64D's twelve set-based statistics all sat at chance -- a set has no mass, a
distribution does. On the 66 conditions every arm covers, all are 100%
correct and the induced parser spends 2 queries against demonstrations-only
150, X64D 148, uniform 34, authored structure 8, role-blind 6.

What learning bought is CALIBRATION, not accuracy: the authored parser beats
the induced one on exact-form (1.00 vs 0.84) and loses downstream (8 queries
vs 2, conflict AUROC 0.988 vs 0.996), because the induced weights were
fitted to a likelihood and calibration is what the commit threshold and the
conflict mass consume.

**The F-1 audit corrects two of those claims.** The commitment rule was
misdescribed: the reported arm also commits when the behaviour posterior
exceeds 0.99, so language can authorise an answer while rivals remain.
Ranking questions is worth 196->150; authorising commitment is worth
150->2. And a paired bootstrap resampled by task meaning shows the query
advantage over the strongest controls INCLUDES ZERO (+0.139, CI
0.000-0.326 vs authored; +0.119, CI -0.070-0.302 vs role-blind) while the
conflict-calibration advantage excludes it (+0.279, CI 0.181-0.384). So
learning buys calibration, which survives, and not query efficiency, which
does not. On the full population main answers 114/129 against
demonstrations-only's 120/129 -- a coverage-for-efficiency trade, not
dominance.

**X64F broke the one-to-one realiser and passes 12/12 over three
independently seeded frozen splits.** The same nouns serve as either the
filter or the scope delimiter, so `X before Y` and `Y before X` have
identical multisets and different meanings -- 50 such collisions over 46 of
230 forms. On those, the contextual parser scores 0.58 against
bag-of-words' 0.22, paired difference +0.359 with a 95% CI of
(+0.220,+0.500) that EXCLUDES zero; across all cases the difference is
-0.026 with a CI that includes it. Context buys exactly the construction it
should and nothing else. The authored word-to-slot control collapses from
1.00 on X64E's realiser to 0.02-0.18 here, which is the point.

Still weak: the operational gain over bag-of-words is negligible (721
queries against 727) and only the comparison against demonstrations-only
survives an interval; 167 of 180 unknown-word cases are answered correctly
because the EVIDENCE resolved them, not because the words were understood;
and with the target removed the system answers 6 of 1080 conditions.

**X64G, the closure replication on four never-used seeds, FAILS G2 and X64
is NOT closed.** A properly written authored contextual parser -- word
order, attachment, phrase patterns, several senses per noun, development
vocabulary only -- scores 1.00 on every seed. The induced parser loses to it
by -0.326 overall (CI -0.366,-0.287) and -0.452 on collisions (CI
-0.581,-0.323). X64F's claim that learning beat authoring came from a
control that counted the induced parser's own features and scored 0.02-0.18.

What DOES replicate: the targeted contextual advantage over bag-of-words,
+0.468 (CI +0.339,+0.597) on collisions, 4/4 fresh seeds -- stronger than
X64F's +0.359 -- and non-inferiority on the unstratified population.

**THE BOTTLENECK IS THE TESTBED.** The surface language is generated by a
template grammar, so an inverse can be written, and a perfect inverse cannot
be beaten. No self-generated controlled language can show induction beating
authoring. That needs variation no static rule set can invert: noise, open
vocabulary, inconsistent constructions, or real text.

X65A-L is complete at the frozen Phase-2 baseline commit `5205543`; X65A as a
whole is not closed. X65A-P and the mandatory X65B-core scoped-transfer gate
remain pending. None of those states may be inferred from uncommitted work in
another checkout.

Today the system searches for a program whose desired behaviour a human has
already specified. A human-level system must infer:

1. What is being asked.
2. What successful completion would look like.
3. Which evidence would disambiguate competing interpretations.
4. Which program or plan satisfies the inferred objective.

Give it: an instruction, a few possibly-ambiguous demonstrations, and an
interactive environment where it may ask questions or request examples.
Give it **no target program**.

The system must maintain a **version space over task interpretations**, not
only over solution programs, and must distinguish four failure modes that
it currently conflates into one:

- "My implementation is wrong."
- "My interpretation of the request is wrong."
- "The request is underspecified."
- "The requested outcome is impossible under the available tools."

Staging, honestly stated. The relayed plan says "start with controlled
language, then paraphrases, then genuinely ambiguous instructions," and that
is the right order -- but note what we do and do not have. There is no
language model in the synthesis loop; the core is 1.4M parameters over
grids and the synthesiser works on byte tapes. So:

- **Stage A (no NL at all):** demonstrations only. The version space over
  interpretations is fully testable here, and it is the actual mechanism.
  Build and gate this first.
- **Stage B:** a controlled instruction language over the existing task
  vocabulary.
- **Stage C:** real paraphrase and ambiguity, which needs the
  `gpt-oss:120b` teacher already plumbed in `bootstrap/teacher.py` --
  proposer only, never evaluator.

**Gate.** On held-out task families and unseen paraphrases: infers the
intended objective, asks useful clarification questions when the evidence
genuinely does not determine the answer, and synthesises a correct solution
**without receiving a hand-authored target program.**

Until this works, the system is a synthesiser operated by a human
experiment designer. That sentence should stay in the README until the gate
passes.

## X65 - Lifelong memory and continual learning

Operate across a sequence of tasks without being reset. Four kinds of
memory, kept distinct:

- **Episodic** -- what happened during a particular attempt.
- **Semantic** -- facts and relationships believed true.
- **Procedural** -- reusable programs, strategies, abstractions.
- **Working** -- the stack, registers, and active task state from X63.

The experiment is not "can it retrieve an old record." It is whether
experience changes future competence. Measure: forward transfer, backward
retention, revision of a stored false belief, cross-domain interference,
memory growth, provenance, and skill selection without branching over the
whole library.

LifelongAgentBench (arXiv 2505.11942) is built around sequential
experience, transfer, and retention rather than isolated static tasks --
useful as an external evaluation pattern at this stage.

**Gate.** Later unseen tasks become measurably easier because of reusable prior
experience; earlier capabilities stay intact; false stored beliefs can be
revised locally; context switches prevent stale reuse; active memory remains
bounded; and all effects survive a clean restart. Exact replay and exact old
answers do not count as transfer.

Prior art in this repo, and the trap: `memory/curve.py` already caught a
fake 13x speedup that cost accuracy 58% -> 29%. Any compounding claim is
run against that check.

### X65A — structured continual memory

X65A is the exact finite reference for episodic, semantic, procedural, and
negative/revision memory under provenance, byte/retrieval budgets, leakage
checks, and real restart. It is not complete because a summary exists or a
development gate is green. Any post-fork audit in the active Phase-1 checkout
must be committed, run against the full suite, and receive an explicit phase
verdict before it changes this roadmap.

### X65B-core — context-scoped transfer before X66

X65B-core is now a mandatory operational dependency. It need not begin with a
full sheaf or category-theoretic implementation, but it must establish:

- explicit validity context and dependency/provenance edges;
- reuse on structurally related but nonidentical tasks;
- no stored final target program or answer;
- context-switch detection;
- stale, poisoned, superficially similar, and shuffled-memory controls;
- revision locality;
- composition of prior verified components under a fixed resource budget;
- restart persistence.

The gate is scoped transfer without harmful reuse. A finite factor-graph or
exact Bayesian mechanism is preferred over heavier mathematics if it passes.

SHWM Scale 6 consumes the committed X65A/X65B contract. Embedding retrieval is
not a substitute for this gate.

## X66 - Real software engineering

The first human-relevant domain, chosen because it offers executable
actions, strong feedback, unit tests, version control, reversible changes,
long-horizon structure, and a natural bridge from program synthesis.

Input: an issue description and an unfamiliar repository. **Not** a target
patch. It must inspect the repo, infer the architecture, form competing bug
hypotheses, generate or improve tests, modify multiple files, run the
suite, diagnose failures, preserve unrelated behaviour, record reusable
knowledge about the repo, and roll back bad changes.

Order: generated small repositories, then unseen open-source repositories,
then a contamination-controlled subset of realistic benchmarks. SWE-bench
(arXiv 2310.06770) is 2,294 real GitHub issues across 12 Python projects --
an eventual external gate, never the development environment.

**Entry dependency.** X66 begins only after:

- X65B-core demonstrates context-scoped reusable knowledge;
- SHWM's learned model has earned its place in controlled environments;
- observable verification reduces model exploitation;
- intervention/mechanism-shift tests pass at the claimed scope;
- Scale 6 shows that continuous high-dimensional memory retains, revises, and
  restarts under budget.

Scale 7 is the Phase-2 implementation point for X66. The same core memory,
uncertainty, planner, verifier, and revision interfaces must be used. A
separately trained repository-specific cognitive core is a specialist, not a
cross-domain result.

**Gate.** Solves held-out repository tasks with no hand-authored target
programs and no per-repository code, and knowledge from one repository
improves work in another without harmful transfer.

## X67 - Cross-domain transfer

Freeze the core architecture. Introduce domains needing different surface
skills: abstract visual transformation, debugging, structured document
analysis, tabular reasoning, interactive planning, scientific hypothesis
testing, tool-mediated information gathering.

The result that matters is not solving each domain independently. It is
**positive transfer between them** -- a debugging strategy improving causal
diagnosis; nesting learned in text helping with visual containment;
experiment-selection learned in synthetic worlds improving test generation.

ARC-AGI-2 (arXiv 2505.11831) is built on novel few-shot abstraction tasks
that stay accessible to humans; ARC-AGI-3 extends toward interactive tasks
involving exploration, planning, memory, and goal acquisition.

**Contamination discipline, carried from the current README:** the public
ARC-AGI-3 games are saturated by frontier scaffolds -- treat them as a
regression check, never as progress. The official leaderboard, not a paper's
self-report, is the comparison.

**Gate.** Architecture frozen, no domain-specific heads, no manually added
primitives: learning in one domain improves sample efficiency or success in
unrelated held-out domains.

## X68 - Refutation under uncertain and imperfect evidence

Exact replay is strongest when the environment is deterministic, the
observation complete, the evaluator correct, and a candidate cleanly
consistent or inconsistent. Human reasoning rarely has those properties.

Two modes:

- **Exact** -- code, formal logic, deterministic mechanics, hard
  constraints. Unchanged.
- **Uncertain** -- noisy perception, incomplete evidence, conflicting
  reports, delayed outcomes, human feedback.

The uncertain layer keeps weighted causal hypotheses while preserving
explicit alternatives, tracking observation noise, source reliability,
missing variables, calibration, contradiction, and **whether a failure
challenges the solution, the interpretation, the observation, or the
model.**

Do not replace exact refutation with a neural score. Embed the exact
constraints inside the broader evidential system.

**Gate.** Under controlled corruption, hidden variables, partial
observability, and delayed feedback: maintains calibrated uncertainty,
seeks useful evidence, and does **not** trigger grammar growth merely
because a sensor or evaluator is wrong. (X42 already asks this question in
miniature -- novelty or corrupted evidence -- and is the precedent.)

## X69 - Grounded causal world models

The system is currently handed a designed observation and action interface.
It must instead discover objects, persistent identities, events,
affordances, causes, hidden state, which observations belong to the same
entity, and which actions change which variables.

The test is not prediction accuracy. It is whether the learned
representation supports counterfactual prediction, planning, transfer,
object persistence, novel intervention, and the distinction between
correlation and mechanism.

**Gate.** Induces reusable latent entities and causal rules from partial
observation, transfers them to changed environments, and identifies when
its **representation** -- not merely a parameter -- is inadequate.

Note the continuity: latent-state inference is the existing differentiator.
`charge_period` is a counter appearing in no frame, inferred at 0.634
against a 0.298 prior. Any change that costs `charge_period` accuracy is
reverted regardless of what else it buys.

**Phase-2 amendment.** SHWM Scales 0–5 start testing this architecture before
the historical X69 capability label is closed. The distinction is important:
building an action-conditioned predictor is an architecture experiment;
closing X69 requires reusable latent entities/mechanisms, intervention and
counterfactual transfer, representation-inadequacy detection, and matched
correlational controls.

## X70 - Multimodal grounding

Only after a shared world-model representation exists. Text, images, audio,
video, and action traces map into the **same** entities, events, and
beliefs -- not five models whose outputs are concatenated.

A real test forces information across modalities: learn an object's
function from video and answer a text question; receive spoken correction
and update a visual plan; read documentation, inspect a GUI, act, and
connect the result to the same stored concept; recognise that two
differently-presented observations are the same object.

OSWorld (arXiv 2404.07972) and OSWorld 2.0 provide real computer
environments for multimodal agents across arbitrary applications -- later
-stage evaluation, once there is perception and a tool interface worth
evaluating.

**Gate.** Knowledge acquired in one modality improves reasoning and action
in another, with persistent cross-modal identity tracking. Attaching a
vision model to a text planner does not pass.

## X71 - Long-horizon autonomous execution

Persistent goal management: interpret a broad objective, create subgoals,
track dependencies, decide what to investigate, execute, detect divergence,
recover after interruption or process restart, preserve completed work,
reprioritise on new evidence, stop when success is established, and **ask
for authority before consequential actions.**

Evaluate work measured in hours or days of simulated effort, not longer
token traces. SWE-EVO is built around repository evolution spanning many
files and coordinated modifications -- the sustained competence short issue
benchmarks miss.

**Gate.** Completes extended tasks across interruptions, avoids repeating
completed work, recovers from failure, and holds the original objective
without human micromanagement.

## X72 - Controlled self-improvement

Late, deliberately -- after the evaluator, the memory, and a broad task
suite are trustworthy.

**May improve:** its skill library, search ordering, test-generation
strategies, memory schemas, retrieval policies, task decomposition, tool
wrappers.

**May not initially modify:** the immutable evaluator, audit logging,
permission boundaries, rollback machinery, the benchmark holdout, or the
rule deciding whether a change is accepted.

Every proposed modification runs sandboxed with versioning, canary
evaluation, held-out tests, regression suites, resource accounting,
reward-tampering checks, automatic rollback, and human-readable provenance.

One definitional point worth keeping: "autonomous improvement" means no
human-operated retraining pipeline, **not** "no weight updates." Humans
learn through physical change; banning parameter adaptation outright would
be an arbitrary restriction. What matters is that the system directs and
validates its own learning safely.

**Gate.** Repeatedly produces modifications that improve held-out
cross-domain performance without degrading old capabilities, corrupting its
evaluator, or exploiting the acceptance metric.

## X73 - Integrated evaluation

No preceding ability counts in isolation. The final system integrates
language understanding, goal inference, abstract reasoning, causal model
formation, persistent memory, continual learning, coding and tool use,
multimodal grounding, cross-domain transfer, long-horizon planning, and
safe self-improvement.

Freeze the architecture before the final evaluation. Then: unseen tasks,
unseen domains, sequential learning, no manually added primitives, no
target programs, imperfect feedback, multimodal interaction, long horizons,
human baselines, strict contamination controls, and compute and interaction
budgets.

Passing ARC-like puzzles, coding tasks, or OS tasks **separately** is not
sufficient. The evidence must show the same accumulated system learning and
operating across all of them.

## Phase-2 SHWM scale track

This section is the operational architecture sequence. Detailed arms, metrics,
and stop conditions are in
`docs/phase-2-continuous-world-model/EXPERIMENT-GATES.md`.

### Scale 0 — premise and throughput

Add typed latent/belief/dynamics/event/uncertainty/verifier contracts, two
frozen encoder adapters, content-addressed cache, transition/branch schema, two
environment adapters, 50M/200M configurations, continuous/discrete/hybrid fake
arms, restart, and resource accounting. Execute the singular 48-workload
development contract in
`docs/phase-2-continuous-world-model/SCALE-0-RUN-MATRIX.md`. No Phase-2 final
seeds and no capability claim.

Gate: every frozen cell and seed completes locally under one evaluator, exact
tests remain intact, no leakage occurs, all matching tolerances hold, and the
preregistered memory/storage/time ceilings are respected.

### Scale 1 — representation contest

Compare continuous, discrete, and hybrid state at equal trainable parameters,
data, optimizer, interactions, planner calls, and cumulative compute. Measure
one-step and multi-step prediction, action-effect discrimination, calibration,
and control.

Falsifier: representation changes prediction metrics but not planning,
transfer, or calibration.

### Scale 2 — learned world model earns its place

Compare reactive frozen features, no-action sequence model, model-free control,
Dreamer-style baseline, SHWM, and oracle simulator under equal real
interactions.

Gate: higher task success at equal interactions or fewer interactions to
matched success, with a paired interval excluding zero. Prediction alone does
not pass.

### Scale 3 — exact verifier earns its place

Compare world-model only, symbolic-only, hybrid without verifier, hybrid with
observable verification, and verifier plus structured memory.

Gate: verification reduces real rollout failures or improves calibration
without erasing the planning gain. Universal abstention fails.

### Scale 4 — intervention and causal audit

Change correlations, mechanics, policies, irrelevant appearance, and forced
actions. Keep insufficient action coverage separate from model inadequacy.

Gate: better intervention outcomes than an equal-size passive/correlational
predictor. Do not call ordinary next-state prediction causal understanding.

### Scale 5 — held-out environment transfer

Freeze the shared belief/dynamics core; permit only thin observation/action
adapters and bounded adaptation.

Gate: fewer interactions or optimization steps in a held-out environment, and
resetting the shared core/memory removes the gain.

### Scale 6 — continuous-scale continual learning

Integrate only the committed X65A/X65B contract. Test recurring environments,
mechanic changes, distractor gaps, stale/false memory, byte/retrieval budgets,
and genuine process restart.

Gate: forward transfer, retention, local revision, negative-transfer
resistance, bounded active growth, and restart survive matched controls.

### Scale 7 — computer and code domains

Add sandboxed GUI/tool environments and then X66 repositories using the same
core interfaces. OSWorld and SWE-bench remain later untouched gates.

Gate: transferable improvement without a domain-specific cognitive core or
harmful repository reuse.

### Scale 8 — parameter/data/horizon study

Only after Scales 2–5 pass, run at least three model sizes and three data sizes
under fixed evaluation. Estimate capability as a function of parameters,
transitions, and horizon. Scale the bottleneck identified by behavior, not the
largest number that fits.

## Post-world-model integrated path

If the early SHWM scales pass, proceed in this dependency order:

1. multimodal time-indexed belief state with modality ablations;
2. causal/mechanism and counterfactual evaluation;
3. grounded abstraction and few-shot concept formation;
4. lifelong multimodal memory plus controlled slow adaptation;
5. metacognitive act/ask/test/replan/abstain control;
6. hierarchical long-horizon agency with authority and stopping;
7. physical, digital, code, and social cross-domain transfer;
8. sandboxed controlled self-improvement with frozen evaluation and rollback;
9. frozen integrated human-level evaluation.

Each item must earn its place by ablation. A successful world model does not
mean that only scaling remains.

---

## Reconciliation with the five-gap plan

The previous plan is not thrown away. Where each gap went:

| gap | fate |
|---|---|
| **1. Compositional search** (observational-equivalence pruning) | **Closed.** X6 measured the behavioural quotient; signatures ~ K^0.43, 10,000 programs -> 377 behaviours. Behavioural dedup is load-bearing in every experiment since X47. |
| **2. Learning that compounds** (MDL library induction) | **Becomes X65**, widened from a rule library to four memory kinds with transfer/retention/revision measured separately. |
| **3. Knowing what you don't know** (version space + information gain) | **Folds into X64.** The version space moves up a level -- over task *interpretations*, not only hypotheses -- which is the harder and more valuable version. `measure_identifiability.py` stays the instrument. |
| **4. Goal inference** | **Becomes X64**, and is promoted to the single most important transition on the list. |
| **5. Breadth** (LLM proposes, verifier disposes) | **Ran as X5.** Returns as a *tool* inside X64 stage C and X66, never as an evaluator. |

## Discipline that carries forward

Unchanged from the existing README, and it applies to every experiment
above:

- **Pre-register the decision before the number exists.** X62's rule fired
  as written; that is the standard.
- **Report expressibility three ways:** not-expressible / expressible-but
  -not-found / found. X62 found these come apart on more than half its
  suite -- 4 of 7 expressible tasks were never found.
- **Ablate every new primitive** on held-out tasks. Cut what does not earn
  its place.
- **Price the mechanism before building on it.** X48, X56, X60, X61, and
  X62 each refuted the premise of a proposal more cheaply than building on
  it would have cost.
- **Multi-seed only; keep calibration arms.** X1's random arm caught a
  broken estimator that had fitted an impossible exponent.
- **Revert anything that loses ground** -- eleven changes so far, including
  a factored search that cut cost 60x and destroyed solve rate.
- **Check the shape before blaming the search.** X58, X59, and X60 were
  each a missing *shape* wearing the costume of a search failure. Probing a
  shape costs 26-78 evaluations against a 400-state budget.
- **Never quietly cripple a primitive.** Six instances so far, most
  recently X62's bounded `PUSH` in the interpreter, which judged a correct
  witness wrong. Bound the search abstraction, never the machine.
