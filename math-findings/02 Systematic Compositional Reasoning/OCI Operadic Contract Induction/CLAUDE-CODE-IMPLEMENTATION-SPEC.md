# Claude Code implementation specification — OCI-1

## Objective

Implement one isolated experiment, provisionally `OCI-1`, that tests whether proof-scoped operadic contracts produce measurable transfer to structurally out-of-distribution typed terms after hidden surface symbols are actively identified.

Do not integrate OCI into Sentinel's production path during this experiment. Reuse existing exact execution and evidence-authoritative policies where convenient, but keep all new code under a dedicated experiment package.

Success is not “the interpreter executes a new tree.” A static typed interpreter already does that. The decisive question is whether the complete loop—persistent active symbol binding plus verifier-gated equation induction—beats matched baselines on later hidden-symbol, held-out-tree tasks.

## Non-goals

- Natural-language parsing.
- Neural-scale training.
- Claiming that operads, term rewriting, contracts, or active queries are novel.
- Treating random tests as universal equation proofs.
- Promoting a rewrite system to a canonical normalizer without termination and confluence evidence.
- Editing the frozen task family after final hidden seeds are sampled.

## Proposed location

```text
experiments/oci/
  README.md
  pyproject.toml
  configs/
    base.yaml
    smoke.yaml
    frozen.yaml
  oci/
    types.py
    terms.py
    algebra.py
    contracts.py
    surface_convention.py
    probes.py
    posterior.py
    equations.py
    rewrite.py
    egraph_adapter.py
    generator.py
    splits.py
    arms.py
    metrics.py
    runner.py
    freeze.py
  tests/
    test_typed_terms.py
    test_algebra_homomorphism.py
    test_signature_identifiability.py
    test_unknown_primitive.py
    test_equation_scope.py
    test_rewrite_context_closure.py
    test_nontermination_guard.py
    test_confluence_guard.py
    test_no_leakage.py
    test_reset_control.py
  reports/
```

Use existing project conventions if they conflict with this illustrative layout.

## Core data model

### Types and generators

```python
@dataclass(frozen=True)
class Profile:
    inputs: tuple[str, ...]
    output: str

@dataclass(frozen=True)
class GeneratorId:
    value: str

@dataclass(frozen=True)
class GeneratorSpec:
    id: GeneratorId
    profile: Profile
    finite_executor_id: str
```

The hidden surface name must not include the generator ID, index, arity beyond declared profile information, canonical ordering, or serializer metadata that reveals its target.

### Terms

```python
@dataclass(frozen=True)
class Atom:
    type_id: str
    value: object

@dataclass(frozen=True)
class Apply:
    generator: GeneratorId
    children: tuple["Term", ...]

Term = Atom | Apply
```

Constructors must reject wrong arity or color immediately. Generate terms from profiles rather than generating untyped trees and filtering afterward.

### Contract records

```python
class CertificateKind(Enum):
    LEAN = "lean"
    SMT = "smt"
    EXHAUSTIVE_FINITE = "exhaustive_finite"
    BOUNDED_TEST = "bounded_test"

@dataclass(frozen=True)
class ContractScope:
    domain_id: str
    verifier_version: str
    assumptions: tuple[str, ...]

@dataclass(frozen=True)
class GeneratorContract:
    generator: GeneratorId
    profile: Profile
    precondition_id: str
    postcondition_id: str
    sensitivities: tuple[float, ...] | None
    local_error: float | None
    certificate_kind: CertificateKind
    scope: ContractScope
    evidence_hash: str

@dataclass(frozen=True)
class EquationRecord:
    lhs: Term
    rhs: Term
    profile: Profile
    certificate_kind: CertificateKind
    scope: ContractScope
    evidence_hash: str
    authoritative: bool
```

`BOUNDED_TEST` records must always have `authoritative=False`. The authoritative equation set may contain only proof-kernel, accepted SMT, or exhaustive-finite certificates with an exact matching scope.

### Surface binding posterior

For each new surface symbol and profile, maintain posterior mass over:

```text
known generator IDs + OTHER
```

Do not collapse observation-equivalent generators. Store an explicit partition and report the true equivalence/orbit class. Commit only when all surviving candidates are behaviorally equivalent for the declared downstream observation boundary, or when one singleton remains and `OTHER` is below the frozen threshold.

## Frozen algebra family

Begin with finite exact domains so equation certificates can be exhaustive.

- 4 colors.
- 12 generators: four unary, five binary, two nullary, one ternary.
- Domain cardinalities between 5 and 17.
- At least three same-profile generator pairs.
- At least one pair indistinguishable under the development probe pool but separable by a held-out legal probe.
- At least one exact observational automorphism under the full allowed probe set; scoring must accept its orbit rather than demand an arbitrary label.
- At least four valid equations: identity, one associativity-like law, one commuting conversion, and one typed fusion law.
- At least four near-laws that agree on a strict development subset and fail elsewhere.
- At least one rule orientation that loops and one pair creating a nonjoinable critical pair, both as negative controls.

The experiment generator must output a machine-readable declaration of colors, profiles, truth-table executors, equations, certificate scopes, and automorphisms.

## Splits

Hold out independently:

1. Tree shapes.
2. Tree depths: development at depth at most 3; final test at depth 4–10.
3. Generator combinations.
4. Surface permutations.
5. Surface strings.
6. Equation application paths.
7. Probe contexts.
8. Whole algebra seeds.

No final hidden algebra, convention, or seed may appear in development or validation.

## Freeze protocol

Before final testing, freeze and hash:

- algebra-family generator;
- exact primitive inventory and profile generator;
- surface-name generator;
- probe pool and active-query policy;
- `OTHER` prior and commitment thresholds;
- equation proposer;
- certificate admission policy;
- rewrite orientation and confluence guards;
- all arms;
- metrics and confidence-interval code;
- development/validation procedures;
- final seed-sampling procedure.

Write a manifest containing file hashes, environment lock hash, Git commit, dirty-state report, and configuration hash. Sample final hidden seeds only after the manifest exists. Final failures do not authorize edits to the frozen family.

## Algorithms

### Structural evaluation

Implement one recursive evaluator over typed terms. Every experimental arm that claims exact symbolic execution must call this same trusted evaluator. Do not duplicate a faster evaluator without equivalence tests.

### Active binding

For candidate set $G_\tau\cup\{OTHER\}$ and query $q$, compute the exact predictive answer distribution under the current posterior. Select

\[
q^*=\arg\max_q I(G;Y_q\mid H)/c(q).
\]

Also implement random legal queries with the same budget and stopping rule. Log the complete candidate partition before and after every answer.

### Equation induction

The first slice may enumerate same-profile terms up to a frozen size bound and propose equalities whose exhaustive finite truth tables match. For every proposal:

1. Check exact profile equality.
2. Record proposal provenance.
3. Exhaustively evaluate the declared finite domain.
4. Store the complete truth-table digest.
5. Mark authoritative only if every assignment agrees.
6. Orient only if a frozen well-founded measure strictly decreases.
7. Check all critical pairs within the finite term/rule bound.
8. If termination or confluence is not established, retain equality in a congruence structure and disable canonical-normal-form claims.

### Counterexample update

On a mismatch, localize in this order:

1. trusted-evaluator regression;
2. surface binding;
3. unknown primitive / `OTHER`;
4. contract-domain violation;
5. equation certificate scope;
6. rewrite implementation;
7. parser/term generator.

Never silently mutate an authoritative equation. Create a new version, retract dependents explicitly, and retain the failed evidence.

## Required arms

1. Flat exact-term memorizer.
2. Static typed interpreter with oracle bindings.
3. Static typed interpreter with ordinary active binding and no equation learning.
4. Random-query binding.
5. Information-gain binding.
6. Ruler-like enumerative equality learner with matched verifier access.
7. Full OCI without persistence.
8. Full OCI with shuffled contract history.
9. Full OCI without equation induction.
10. Full OCI without the certificate gate.
11. Full OCI.
12. Oracle binding + oracle equations.

The static typed interpreter with ordinary active binding is the decisive baseline. If it matches full OCI, the operadic contract layer has not earned a distinct claim.

## Measurements

Report per task and aggregated by frozen seed:

- exact output accuracy;
- selective risk, commitment coverage, and false commitment rate;
- accuracy conditioned on correct binding;
- posterior mass on the true observational orbit;
- queries per committed symbol;
- active-query regret versus random;
- early-versus-late performance under one persistent surface convention;
- reset and shuffled-history deltas;
- valid-equation recall and false-equation admission rate;
- certificate type and scope for every admitted equation;
- rewrite termination, critical-pair, and normal-form agreement statistics;
- actual numerical error divided by the declared bound, if continuous operators are added;
- runtime and peak memory by tree size, rule count, and e-graph size;
- two-part description length of generators, bindings, equations, and observations.

Use task-level bootstrap intervals nested inside seed-level summaries. Do not treat correlated terms from one algebra as independent seeds.

## Gates

- **O1 — Structural OOD validity:** every final tree is absent from training support while its non-`OTHER` primitives are in scope.
- **O2 — No answer leakage:** hidden generator IDs and future answers are absent from serialized surface inputs and memory.
- **O3 — Identifiability honesty:** scoring targets the observational orbit; nonseparating cases end ambiguous or abstained.
- **O4 — Exact extension:** with oracle bindings, trusted evaluation is exact on all generated held-out trees.
- **O5 — Active value:** information-gain queries beat random under matched budgets.
- **O6 — Persistence:** later same-convention tasks improve; reset/shuffle removes the gain.
- **O7 — Certificate precision:** no bounded-test near-law enters the authoritative quotient.
- **O8 — Rewrite honesty:** canonical normalization is enabled only for proved/finitely established convergent systems.
- **O9 — Incremental value:** full OCI beats static interpreter + ordinary active binding on a preregistered central metric.
- **O10 — Replication:** O5, O6, O7, and O9 hold across untouched seeds with uncertainty intervals.

## Falsifiers

The mechanism fails its first-cycle claim if:

- the static interpreter + active learner matches full OCI;
- apparent transfer survives a history reset or shuffle;
- active and random probes tie;
- hidden symbols are identifiable from leakage rather than answers;
- observationally symmetric generators are scored as arbitrary-label failures;
- near-laws are admitted as universal equations;
- normalization loops or yields divergent irreducibles while still being called canonical;
- oracle-bound OOD execution fails;
- gains vanish after holding out tree shape independently of surface names;
- or final-test results trigger edits to the frozen generator.

## Verification commands required in the final implementation report

The implementation report must include exact commands and outputs for:

```text
unit and property tests
trusted evaluator equivalence tests
smoke experiment
freeze/hash generation
four or more untouched final seeds
git status --short
git rev-parse HEAD
```

Also rerun the mathematical package checks from the project root:

```bash
source math-findings/activate-math-research.sh
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_symbolic_checks.py"
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_falsification.py"
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_resource_checks.py"
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_property_checks.py"
cd .math-research-tools/lean/sentinel_math
lake env lean "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/formal/OCI.lean"
```

## Required final report from Claude Code

Return:

1. experiment name and Git commit;
2. frozen-manifest hash;
3. repository cleanliness and test count;
4. exact distributions and sample sizes;
5. all arm definitions and equal-access audit;
6. metrics with uncertainty;
7. gate-by-gate pass/fail table;
8. counterexamples and negative results;
9. bugs or representation errors discovered;
10. any correction to the mathematical assumptions;
11. the cheapest decisive follow-up;
12. no AGI, breakthrough, or novelty claim beyond the bounded evidence.
