/-
Copyright (c) 2026 Anthony Cavero. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Anthony Cavero, OpenAI Codex
-/
import Mathlib

/-!
# Finite theorems for X65A structured continual memory

These theorems isolate the finite claims that can be checked without assuming
the empirical success of the proposed memory architecture.  In particular,
`noDirectFutureTarget` rules out literal target storage, not inference of a
future answer from reusable structure.
-/

namespace SentinelMath.X65A

open scoped BigOperators

section BoundedMemory

/-- If the memory state space is smaller than the history space, no encoder can
preserve every history injectively.  This is the finite pigeonhole core of the
bounded-memory no-free-lunch result. -/
theorem boundedMemory_not_injective
    {History Memory : Type*} [Fintype History] [Fintype Memory]
    (hcard : Fintype.card Memory < Fintype.card History)
    (encode : History → Memory) :
    ¬ Function.Injective encode := by
  intro hinjective
  have hle : Fintype.card History ≤ Fintype.card Memory :=
    Fintype.card_le_of_injective encode hinjective
  exact (not_le_of_gt hcard) hle

end BoundedMemory

section NoDirectLeakage

/-- Every directly stored answer was observed strictly before task `t`. -/
def NoFutureTargets {Answer : Type*}
    (memory : List (ℕ × Answer)) (t : ℕ) : Prop :=
  ∀ entry ∈ memory, entry.1 < t

/-- A memory satisfying `NoFutureTargets` cannot literally contain the target
answer at the current or any later task index. -/
theorem noDirectFutureTarget
    {Answer : Type*} (memory : List (ℕ × Answer)) (t t' : ℕ) (answer : Answer)
    (hclean : NoFutureTargets memory t) (hfuture : t ≤ t') :
    (t', answer) ∉ memory := by
  intro hmember
  have hpast : t' < t := hclean (t', answer) hmember
  exact (not_lt_of_ge hfuture) hpast

end NoDirectLeakage

section Bayes

variable {Latent : Type*} [Fintype Latent]

/-- Evidence normalizer for a finite latent-component model. -/
noncomputable def evidenceMass
    (prior likelihood : Latent → ℝ) : ℝ :=
  ∑ latent, prior latent * likelihood latent

/-- Exact finite Bayesian update. -/
noncomputable def posterior
    (prior likelihood : Latent → ℝ) (latent : Latent) : ℝ :=
  prior latent * likelihood latent / evidenceMass prior likelihood

/-- Exact Bayesian updating preserves normalization whenever the evidence has
nonzero marginal probability. -/
theorem posterior_sum_one
    (prior likelihood : Latent → ℝ)
    (hpositive : evidenceMass prior likelihood ≠ 0) :
    ∑ latent, posterior prior likelihood latent = 1 := by
  unfold posterior
  rw [← Finset.sum_div]
  change evidenceMass prior likelihood / evidenceMass prior likelihood = 1
  exact div_self hpositive

end Bayes

section RevisionLocality

variable {Left Right : Type*} [Fintype Left]

/-- A revision whose new evidence depends only on the left latent component. -/
noncomputable def factorizedRevision
    (leftPrior leftEvidence : Left → ℝ) (rightBelief : Right → ℝ)
    (state : Left × Right) : ℝ :=
  posterior leftPrior leftEvidence state.1 * rightBelief state.2

/-- Under factorization and left-only evidence, marginalizing the revised joint
belief over the left component leaves every right-component weight unchanged. -/
theorem independentRevisionLeavesRightMarginal
    (leftPrior leftEvidence : Left → ℝ) (rightBelief : Right → ℝ)
    (hpositive : evidenceMass leftPrior leftEvidence ≠ 0) (right : Right) :
    ∑ left, factorizedRevision leftPrior leftEvidence rightBelief (left, right) =
      rightBelief right := by
  unfold factorizedRevision
  change (∑ left, posterior leftPrior leftEvidence left * rightBelief right) =
    rightBelief right
  rw [← Finset.sum_mul]
  rw [posterior_sum_one leftPrior leftEvidence hpositive]
  simp

/-- The executable graph update corresponding to a local left-component
revision preserves the unrelated right component exactly. -/
def reviseLeft (update : Left → Left) (state : Left × Right) : Left × Right :=
  (update state.1, state.2)

omit [Fintype Left] in
@[simp] theorem reviseLeft_preserves_right
    (update : Left → Left) (left : Left) (right : Right) :
    (reviseLeft update (left, right)).2 = right := rfl

end RevisionLocality

section PosteriorSufficiency

variable {Latent Outcome : Type*} [Fintype Latent]

/-- Posterior predictive distribution induced by a finite latent-component
posterior and a task channel. -/
noncomputable def posteriorPredictive
    (belief : Latent → ℝ) (channel : Latent → Outcome → ℝ)
    (outcome : Outcome) : ℝ :=
  ∑ latent, belief latent * channel latent outcome

/-- Histories inducing the same finite posterior induce the same prediction for
every future outcome.  Thus the posterior is a predictive sufficient statistic
under the conditional-independence model encoded by `channel`. -/
theorem finitePosterior_isSufficient
    (first second : Latent → ℝ) (channel : Latent → Outcome → ℝ)
    (hsame : first = second) :
    posteriorPredictive first channel = posteriorPredictive second channel := by
  simp [hsame]

end PosteriorSufficiency

section GrowthAndConsolidation

/-- Storing every nonempty episode requires at least one storage unit per
episode, hence raw replay grows at least linearly. -/
theorem rawReplay_linearGrowth
    (taskCount : ℕ) (episodeSize : Fin taskCount → ℕ)
    (hnonempty : ∀ task, 1 ≤ episodeSize task) :
    taskCount ≤ ∑ task, episodeSize task := by
  have hsum : (∑ _task : Fin taskCount, 1) ≤ ∑ task, episodeSize task := by
    apply Finset.sum_le_sum
    intro task _hmember
    exact hnonempty task
  simpa using hsum

/-- A shared component reduces two-part description length exactly when its
one-time overhead is smaller than the repeated per-episode saving. -/
theorem reusableComponent_reducesDescriptionLength
    (repetitions rawCost residualCost componentCost : ℝ)
    (hsaving : componentCost < repetitions * (rawCost - residualCost)) :
    componentCost + repetitions * residualCost < repetitions * rawCost := by
  linarith

end GrowthAndConsolidation

end SentinelMath.X65A
