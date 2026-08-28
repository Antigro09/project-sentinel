# SHWM Architecture and Interface Contract

Status: implementation contract for Scale 0; architecture hypothesis beyond
that point.

## 1. Design rule

The learned world model proposes predictions and plans. Observable execution,
tests, constraints, and explicit Sentinel state decide whether a claim is
accepted. Embeddings may retrieve or rank; they do not create truth.

## 2. Component boundaries

```text
EnvironmentAdapter
  -> FrozenEncoderAdapter
  -> LatentProjector
  -> BeliefUpdater
  -> DynamicsModel
  -> Planner
  -> DecisionController
  -> VerifierBridge
  -> EnvironmentAdapter.step

VerifierBridge.counterexample
  -> RevisionEvent
  -> StructuredMemory
  -> Retriever
  -> BeliefUpdater
```

Each arrow crosses a typed, serializable boundary. Scale 0 must make every
boundary runnable with deterministic fake components before a neural model is
introduced.

## 3. Proposed source layout

```text
src/sentinel/wm/
  latent_contract.py       # types and Protocols only
  representations.py      # continuous/discrete/hybrid tagged state
  belief.py                # recurrent stochastic belief interface
  dynamics.py              # prediction distribution interface
  events.py                # verifier-facing event schema
  uncertainty.py           # aleatoric/epistemic/inadequacy decomposition
  cache.py                 # content-addressed frozen-feature cache
  dataset.py               # transition/sequence records and split manifests
  planner_bridge.py        # planner-facing rollout API
  verifier_bridge.py       # observable probes and counterexamples
  versioning.py            # model/encoder/config/evaluator digests

src/sentinel/env/adapters/
  base.py                  # reset/step/probe/branch contract
  synthetic_control.py     # deterministic generated preflight domain
  procgen_adapter.py       # optional Stratum-A adapter after dependency check
  crafter_adapter.py       # optional Stratum-B adapter after Scale-0 preflight

experiments/shwm/
  scale0_preflight.py
  freeze_manifest.py
  resource_report.py
  configs/

tests/shwm/
  test_contracts.py
  test_cache.py
  test_dataset.py
  test_splits.py
  test_verifier_bridge.py
  test_no_leakage.py
  test_restart.py
  test_resource_accounting.py
```

Existing `src/sentinel/wm/contract.py` remains the exact executable-world-model
contract. The new latent contract is additive. It must not redefine exact
`WorldModel` behavior or weaken existing verifier tests.

## 4. Core records

The following are language-neutral schemas; the first implementation may use
frozen Python dataclasses with canonical JSON serialization.

### ObservationEnvelope

```text
episode_id: str
step: int
timestamp_ns: int
modalities: map[str, bytes-or-array-reference]
structured_observation: canonical map
available_action_digest: sha256
environment_version: sha256
taint: set[str]
```

### EncoderIdentity

```text
provider
model_name
revision
weight_digest
preprocessing_digest
precision
frozen: true
license_record
```

### LatentObservation

```text
episode_id
step
encoder_identity
projector_digest
representation_kind: continuous | discrete | hybrid
continuous_values: optional tensor reference
discrete_codes: optional integer tensor reference
mask
source_observation_digest
```

### BeliefState

```text
episode_id
step
deterministic_state
stochastic_state_distribution
retrieved_memory_ids
aleatoric_uncertainty
epistemic_uncertainty
model_inadequacy_score
model_version
```

### TransitionRecord

```text
episode_id
step
observation_digest_t
latent_digest_t
belief_digest_t
action
action_propensity
collector_policy_digest
reward_or_progress
termination
observation_digest_t1
latent_digest_t1
structured_events
branch_group_id: optional
taint
```

`branch_group_id` links transitions collected from the same restored initial
state with different actions. This is required for the intervention audit.

### TransitionPrediction

```text
next_latent_distribution
event_distribution
reward_distribution
termination_probability
aleatoric_uncertainty
epistemic_uncertainty
model_inadequacy_score
rollout_support_scope
```

### VerificationResult

```text
accepted_observables
rejected_observables
unprobed_observables
counterexamples
constraint_violations
coverage
verifier_version
```

Accuracy and coverage are separate. A prediction that abstains on every probe
does not receive a successful capability label.

## 5. Frozen encoder contract

```python
class FrozenEncoderAdapter(Protocol):
    @property
    def identity(self) -> EncoderIdentity: ...

    def encode(self, observation: ObservationEnvelope) -> EncodedObservation: ...

    def health_check(self) -> EncoderHealth: ...
```

Requirements:

- weights remain frozen in Scale 0–2;
- every cache key includes the complete encoder and preprocessing identity;
- changing precision invalidates the cache unless equivalence is proved;
- inherited benchmark capability is evaluated separately;
- unavailable modalities are represented by masks, not zero vectors that can
  be mistaken for observations;
- raw inputs remain available to the trusted verifier where permitted.

Candidate names are configuration values, not dependencies until local license,
conversion, precision, and throughput preflight passes.

## 6. Representation arms

All arms implement:

```python
class LatentRepresentation(Protocol):
    kind: RepresentationKind
    dimension_budget: int

    def project(self, encoded: EncodedObservation) -> LatentObservation: ...
    def serialize(self, latent: LatentObservation) -> bytes: ...
    def validate(self, latent: LatentObservation) -> None: ...
```

The continuous arm uses real-valued stochastic state; the discrete arm uses
categorical stochastic state; the hybrid arm combines continuous perception
with discrete event/mechanism variables. The trainable parameter budget must be
equalized after implementation, not estimated from nominal hidden width.

## 7. Belief and dynamics contracts

```python
class BeliefUpdater(Protocol):
    def initial(self, batch: int) -> BeliefState: ...

    def update(
        self,
        previous: BeliefState,
        latent: LatentObservation,
        previous_action: Action | None,
        previous_outcome: Outcome | None,
        retrieved_memory: tuple[MemoryEntry, ...],
    ) -> BeliefState: ...


class ActionConditionedDynamics(Protocol):
    def predict(
        self,
        belief: BeliefState,
        action: Action,
    ) -> TransitionPrediction: ...

    def observe(
        self,
        prediction: TransitionPrediction,
        actual: VerifiedTransition,
    ) -> ModelUpdateEvidence: ...
```

The action parameter is mandatory. A no-action arm exists only as a falsifying
control.

## 8. Structured event head

The initial event schema is intentionally observable:

```text
OBJECT_APPEARED
OBJECT_DISAPPEARED
INVENTORY_CHANGED
FOCUS_MOVED
FILE_STATE_CHANGED
UI_STATE_CHANGED
ACTION_SUCCEEDED
ACTION_FAILED
CONSTRAINT_VIOLATED
GOAL_PROGRESS_CHANGED
UNKNOWN_EVENT
MISSING_EVENT_REPRESENTATION
```

The two unknown states differ: `UNKNOWN_EVENT` means an in-schema event whose
value is unresolved; `MISSING_EVENT_REPRESENTATION` means the schema itself is
inadequate. Expansion is proposed, tested, versioned, and never auto-promoted
from a single embedding anomaly.

## 9. Uncertainty contract

```text
aleatoric: distributional outcome spread conditional on accepted model
epistemic: posterior/ensemble uncertainty within the model class
inadequacy: evidence that the class or representation cannot explain the event
```

The implementation must support calibration tables and distinct thresholds.
One scalar may be exposed for a specific decision, but the stored evidence must
retain the three components.

## 10. Planning contract

The planner consumes a pure rollout interface and cannot mutate model state:

```python
class LatentRollout(Protocol):
    def expand(self, belief: BeliefState, action: Action) -> RolloutNode: ...
    def score(self, node: RolloutNode, goal: Goal) -> UtilityDistribution: ...
```

Scale 0 provides deterministic beam/CEM/MCTS adapters with fake dynamics so
accounting can be tested. Scale 1 chooses a planner per action space before
results are seen. Every run logs proposed nodes, depth, model calls, wall time,
and uncertainty penalty.

## 11. Verifier bridge contract

The bridge translates predicted events into exact probes. It never asks the
latent model whether its own prediction was correct.

```python
class VerifierBridge(Protocol):
    def required_probes(self, context: VerificationContext) -> ProbeSet: ...
    def requested_probes(self, prediction: TransitionPrediction) -> ProbeSet: ...
    def verify(self, prediction: TransitionPrediction, actual: ActualStep) -> VerificationResult: ...
    def counterexamples(self, result: VerificationResult) -> tuple[Counterexample, ...]: ...
```

All externally consequential actions pass this bridge or an explicit authority
gate. `required_probes` is supplied by the evaluator/benchmark contract and is
frozen independently of model output; it is always executed and cannot be
removed, weakened, or replaced by `requested_probes`. Model-requested probes
may add diagnostic coverage only. A benchmark adapter may define additional
required probes, but must do so before Phase-2 final seeds are sampled.

## 12. Memory bridge

Retrieval is two-stage:

1. embedding or symbolic index proposes up to a preregistered candidate budget;
2. validity context, provenance, verification scope, stale/contradicted state,
   and dependency checks decide what is admissible.

The belief state stores retrieved IDs and model versions so replay is auditable.
A memory proposal cannot be used when its verification scope does not cover the
current context unless the controller explicitly treats it as provisional and
requests a test.

## 13. Canonical serialization and restart

Persisted state includes:

- encoder/projector/world-model identities;
- optimizer and random states when training continuation is allowed;
- belief and environment continuation state;
- cache manifest, not implicit process cache;
- structured memory and provenance;
- split/freeze manifest;
- query/action budgets and counters;
- pending counterexamples and provisional revisions.

At restart, a clean process loads only permitted persisted state. Tests plant a
forbidden global cache and require divergence detection.

## 14. Failure boundaries

- A correct verifier with insufficient probes can miss a latent error.
- Accurate one-step prediction can produce unstable long rollouts.
- Low ensemble disagreement can occur under shared misspecification.
- A powerful frozen backbone can solve the task and hide a useless world model.
- Branch data can leak the hidden dynamic if split after collection.
- Event heads can become target-label channels.
- Continuous and discrete arms can have unequal effective capacity despite
  nominal parameter matching.
- Planner compute can create an apparent model advantage.
- Cache conversion can change features without changing model names.
- Memory can reintroduce stale mechanics after a context switch.

Each is mapped to a control in `EXPERIMENT-GATES.md`.
