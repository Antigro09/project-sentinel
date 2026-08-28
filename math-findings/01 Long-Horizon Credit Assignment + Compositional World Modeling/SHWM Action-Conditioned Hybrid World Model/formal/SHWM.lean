/-
Copyright (c) 2026 Anthony Cavero. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Anthony Cavero, OpenAI Codex
-/
import Mathlib

/-!
# Finite formal core for the Sentinel-Hybrid World Model

This file checks only architecture-independent finite claims used by the SHWM
research cycle:

* a finite-precision latent has finitely many machine states;
* open-loop action sequences grow exponentially with horizon;
* passive data can leave an untried intervention non-identifiable;
* nonnegative weighted component losses and ensemble disagreement remain
  nonnegative;
* a Lipschitz-style rollout-error recurrence has the stated finite-sum form;
* an exact observable verifier rejects any observable mismatch; and
* a probe can fail to detect a latent mismatch when it is not injective.

These theorems do not establish that SHWM learns useful representations,
plans successfully, identifies causal mechanisms, transfers across domains,
or has any AGI capability. Those are empirical hypotheses.
-/

namespace SentinelMath.SHWM

open scoped BigOperators

section FiniteCapacity

/-- A machine latent with `dimension` coordinates and `bits` binary bits per
coordinate has exactly `2 ^ (bits * dimension)` distinct code states. -/
theorem finitePrecisionLatent_cardinality (dimension bits : ℕ) :
    Fintype.card (Fin dimension → Fin (2 ^ bits)) =
      2 ^ (bits * dimension) := by
  simp [pow_mul]

/-- There are `branching ^ horizon` open-loop action sequences when each of
`horizon` positions has `branching` choices. -/
theorem actionSequence_cardinality (branching horizon : ℕ) :
    Fintype.card (Fin horizon → Fin branching) = branching ^ horizon := by
  simp

end FiniteCapacity

section VerifierQuotient

/-- Observable trace produced by an exact transition and a verifier probe. -/
def probeTrace {State Action Observable : Type*}
    (transition : State → Action → State) (probe : State → Observable) :
    State → List Action → List Observable
  | state, [] => [probe state]
  | state, action :: suffix =>
      probe state :: probeTrace transition probe (transition state action) suffix

/-- Two states are verifier-equivalent through horizon `horizon` when every
admissible action list of at most that length produces the same probe trace. -/
def VerifierEquivalent {State Action Observable : Type*}
    (transition : State → Action → State) (probe : State → Observable)
    (horizon : ℕ) (left right : State) : Prop :=
  ∀ actions, actions.length ≤ horizon →
    probeTrace transition probe left actions =
      probeTrace transition probe right actions

/-- Any finite-horizon plan score that depends only on the verifier trace is
identical on verifier-equivalent states. The guarantee is conditional on the
probe set and horizon encoded by `VerifierEquivalent`. -/
theorem verifierQuotient_sufficient_for_traceScore
    {State Action Observable Score : Type*}
    (transition : State → Action → State) (probe : State → Observable)
    (horizon : ℕ) (left right : State)
    (hequivalent : VerifierEquivalent transition probe horizon left right)
    (actions : List Action) (hwithin : actions.length ≤ horizon)
    (score : List Observable → Score) :
    score (probeTrace transition probe left actions) =
      score (probeTrace transition probe right actions) := by
  rw [hequivalent actions hwithin]

end VerifierQuotient

section PassiveNonidentifiability

/-- A model that predicts `false` after both actions. -/
def passiveKernelA (_action : Bool) : Bool := false

/-- A model agreeing on the observed action `false`, but predicting a
different result after the unobserved intervention `true`. -/
def passiveKernelB (action : Bool) : Bool := action

/-- The two models are observationally indistinguishable under the policy
that only chooses `false`. -/
theorem passiveModels_agree_on_observed_action :
    passiveKernelA false = passiveKernelB false := by
  rfl

/-- The same models disagree on the action intervention that the passive
policy never executes. -/
theorem passiveModels_disagree_on_intervention :
    passiveKernelA true ≠ passiveKernelB true := by
  decide

/-- Any finite passive trace containing only `false` actions receives exactly
the same predicted outputs from the two models. -/
theorem passiveTrace_indistinguishable (n : ℕ) :
    List.replicate n (passiveKernelA false) =
      List.replicate n (passiveKernelB false) := by
  rw [passiveModels_agree_on_observed_action]

end PassiveNonidentifiability

section NonnegativeObjectives

/-- A finite sum of nonnegative weighted nonnegative component losses is
nonnegative. -/
theorem weightedLoss_nonnegative
    {Index : Type*} [Fintype Index]
    (weight loss : Index → ℝ)
    (hweight : ∀ index, 0 ≤ weight index)
    (hloss : ∀ index, 0 ≤ loss index) :
    0 ≤ ∑ index, weight index * loss index := by
  exact Finset.sum_nonneg fun index _ =>
    mul_nonneg (hweight index) (hloss index)

/-- Squared ensemble disagreement is nonnegative for every finite ensemble. -/
theorem squaredDisagreement_nonnegative
    {Member : Type*} [Fintype Member]
    (prediction : Member → ℝ) (center : ℝ) :
    0 ≤ ∑ member, (prediction member - center) ^ 2 := by
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

end NonnegativeObjectives

section RolloutError

/-- Worst-case rollout error when each one-step update has additive error
`epsilon` and transports prior error with sensitivity `lipschitz`. -/
def rolloutError (lipschitz epsilon : ℝ) : ℕ → ℝ
  | 0 => 0
  | horizon + 1 => epsilon + lipschitz * rolloutError lipschitz epsilon horizon

/-- The rollout-error recurrence is a finite geometric accumulation of local
one-step errors. -/
theorem rolloutError_eq_geometric_sum
    (lipschitz epsilon : ℝ) (horizon : ℕ) :
    rolloutError lipschitz epsilon horizon =
      epsilon * ∑ i ∈ Finset.range horizon, lipschitz ^ i := by
  induction horizon with
  | zero => simp [rolloutError]
  | succ horizon ih =>
      rw [rolloutError, ih, Finset.sum_range_succ']
      simp_rw [pow_succ]
      rw [← Finset.sum_mul]
      ring

end RolloutError

section ObservableVerification

/-- The exact observable verifier accepts exactly when prediction and
observation are equal. -/
def observableAccept {Observable : Type*}
    (predicted observed : Observable) : Prop := predicted = observed

/-- Every observable counterexample is rejected by the exact verifier. -/
theorem observableMismatch_rejected
    {Observable : Type*} (predicted observed : Observable)
    (hmismatch : predicted ≠ observed) :
    ¬ observableAccept predicted observed := by
  exact hmismatch

/-- A deliberately non-injective probe: two different latent states have the
same observable output. -/
def constantProbe (_latent : Bool) : Unit := ()

/-- Observable verification cannot in general detect latent distinctions that
the probe erases. This is a boundary condition, not a defect in equality. -/
theorem noninjectiveProbe_hides_latent_mismatch :
    false ≠ true ∧ constantProbe false = constantProbe true := by
  constructor
  · decide
  · rfl

end ObservableVerification

end SentinelMath.SHWM
