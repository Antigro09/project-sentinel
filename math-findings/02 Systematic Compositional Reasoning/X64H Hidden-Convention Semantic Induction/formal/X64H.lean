/-
Copyright (c) 2026 Anthony Cavero. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Anthony Cavero, OpenAI Codex
-/
import Mathlib

/-!
# Finite theorems for X64H latent-convention induction

This file mechanically verifies the deterministic authored-inverse result
and the finite separating-signature criterion used by the X64H theory note.
-/

namespace SentinelMath.X64H

open scoped BigOperators

section AuthoredInverse

variable {Z U : Type*}

/-- Weighted exact-recovery accuracy before normalization of the weights. -/
noncomputable def weightedAccuracy
    [Fintype Z] [DecidableEq Z]
    (weight : Z → ℝ) (realizer : Z → U) (parser : U → Z) : ℝ :=
  ∑ z, weight z * if parser (realizer z) = z then 1 else 0

/-- An injective realizer's authored inverse recovers every generated meaning. -/
theorem authoredInverse_correct
    [Nonempty Z] (realizer : Z → U)
    (hinjective : Function.Injective realizer) (z : Z) :
    Function.invFun realizer (realizer z) = z := by
  exact Function.leftInverse_invFun hinjective z

/-- The authored inverse receives all nonnegative semantic prior mass. -/
theorem authoredInverse_perfect
    [Fintype Z] [Nonempty Z] [DecidableEq Z]
    (weight : Z → ℝ) (realizer : Z → U)
    (hinjective : Function.Injective realizer) :
    weightedAccuracy weight realizer (Function.invFun realizer) = ∑ z, weight z := by
  classical
  unfold weightedAccuracy
  apply Finset.sum_congr rfl
  intro z _hz
  rw [Function.leftInverse_invFun hinjective z]
  simp

/-- No parser has greater expected exact-recovery accuracy than a known inverse. -/
theorem noParserStrictlyImprovesKnownInjectiveRealizer
    [Fintype Z] [Nonempty Z] [DecidableEq Z]
    (weight : Z → ℝ) (hweight : ∀ z, 0 ≤ weight z)
    (realizer : Z → U) (hinjective : Function.Injective realizer)
    (parser : U → Z) :
    weightedAccuracy weight realizer parser ≤
      weightedAccuracy weight realizer (Function.invFun realizer) := by
  classical
  rw [authoredInverse_perfect weight realizer hinjective]
  unfold weightedAccuracy
  apply Finset.sum_le_sum
  intro z _hz
  by_cases hcorrect : parser (realizer z) = z
  · simp [hcorrect]
  · simp [hcorrect, hweight z]

end AuthoredInverse

section SeparatingSignatures

variable {A W S : Type*}

/-- A surface symbol's observed incidence signature under a convention. -/
def surfaceSignature (convention : A ≃ W) (signature : A → S) (word : W) : S :=
  signature (convention.symm word)

/-- Distinct semantic incidence signatures uniquely identify the convention. -/
theorem separatingSignaturesIdentifyConvention
    (signature : A → S) (hseparating : Function.Injective signature)
    (first second : A ≃ W)
    (hobservation : ∀ word,
      surfaceSignature first signature word = surfaceSignature second signature word) :
    first = second := by
  apply Equiv.ext
  intro atom
  have hsig := hobservation (first atom)
  have hpreimage : first.symm (first atom) = second.symm (first atom) :=
    hseparating hsig
  have hatom : atom = second.symm (first atom) := by
    simpa using hpreimage
  have hmapped : second atom = first atom := by
    simpa using congrArg second hatom
  exact hmapped.symm

/-- Duplicate signatures admit a nontrivial swap that preserves every observation. -/
theorem duplicateSignaturesGiveAutomorphism
    (signature : A → S) (a b : A) (hne : a ≠ b)
    (hsame : signature a = signature b) :
    ∃ permutation : A ≃ A,
      permutation ≠ Equiv.refl A ∧ ∀ x, signature (permutation x) = signature x := by
  classical
  refine ⟨Equiv.swap a b, ?_, ?_⟩
  · intro heq
    have happly := DFunLike.congr_fun heq a
    have hba : b = a := by simpa [hne] using happly
    exact hne hba.symm
  · intro x
    by_cases hxa : x = a
    · subst x
      simpa [hne] using hsame.symm
    · by_cases hxb : x = b
      · subst x
        simpa [hne] using hsame
      · rw [Equiv.swap_apply_of_ne_of_ne hxa hxb]

/-- Injective signatures are exactly those with no nontrivial observational automorphism. -/
theorem signaturesSeparateIffNoNontrivialAutomorphism
    (signature : A → S) :
    Function.Injective signature ↔
      ∀ permutation : A ≃ A,
        (∀ x, signature (permutation x) = signature x) → permutation = Equiv.refl A := by
  classical
  constructor
  · intro hinjective permutation hinvariant
    apply Equiv.ext
    intro x
    exact hinjective (hinvariant x)
  · intro hunique a b hsame
    by_contra hne
    obtain ⟨permutation, hnontrivial, hinvariant⟩ :=
      duplicateSignaturesGiveAutomorphism signature a b hne hsame
    exact hnontrivial (hunique permutation hinvariant)

/-- With m binary contexts, at most 2^m atoms can have distinct signatures. -/
theorem binarySignatureCardinalityBound
    [Fintype A] (m : ℕ) (signature : A → (Fin m → Bool))
    (hseparating : Function.Injective signature) :
    Fintype.card A ≤ 2 ^ m := by
  calc
    Fintype.card A ≤ Fintype.card (Fin m → Bool) :=
      Fintype.card_le_of_injective signature hseparating
    _ = 2 ^ m := by simp

end SeparatingSignatures

end SentinelMath.X64H
