/-
Copyright (c) 2026 Anthony Cavero. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Anthony Cavero, OpenAI Codex
-/
import Mathlib

/-!
# Mechanically checked finite core for Operadic Contract Induction

The full OCI proposal uses a typed colored operad.  This file checks the
binary-tree subcase needed by the first falsification experiment:

* a compositional evaluator is the unique fold extending primitive meanings;
* the result applies to terms outside any chosen training set;
* verifier-certified local rewrites preserve meaning in every tree context;
* a finite sequence of certified rewrites preserves meaning; and
* the scalar contractive error recurrence has the claimed finite-sum form.

No theorem below asserts that primitive meanings can be recovered without
identifying evidence, or that an arbitrary rewrite system terminates or is
confluent.
-/

namespace SentinelMath.OCI

open scoped BigOperators

/-- Binary terms form the mechanically checked sublanguage of the free
typed-operadic term language used in the paper theory. -/
inductive BTerm (Generator Atom : Type*) where
  | atom : Atom → BTerm Generator Atom
  | node : Generator → BTerm Generator Atom → BTerm Generator Atom →
      BTerm Generator Atom
deriving Repr

namespace BTerm

variable {Generator Atom X : Type*}

/-- The structural evaluator induced by meanings of atoms and generators. -/
def eval (atomSem : Atom → X) (nodeSem : Generator → X → X → X) :
    BTerm Generator Atom → X
  | .atom leaf => atomSem leaf
  | .node generator left right =>
      nodeSem generator (eval atomSem nodeSem left) (eval atomSem nodeSem right)

/-- Any evaluator respecting the two constructors is the structural evaluator. -/
theorem fold_unique
    (atomSem : Atom → X) (nodeSem : Generator → X → X → X)
    (candidate : BTerm Generator Atom → X)
    (hatom : ∀ leaf, candidate (.atom leaf) = atomSem leaf)
    (hnode : ∀ generator left right,
      candidate (.node generator left right) =
        nodeSem generator (candidate left) (candidate right)) :
    ∀ term, candidate term = eval atomSem nodeSem term := by
  intro term
  induction term with
  | atom leaf => simpa [eval] using hatom leaf
  | node generator left right left_ih right_ih =>
      rw [hnode, left_ih, right_ih]
      simp [eval]

/-- Structural correctness is independent of whether the term occurred in a
designated training set.  The out-of-distribution premise is intentionally
unused: the guarantee comes from constructor laws, not interpolation. -/
theorem exact_outside_training
    (atomSem : Atom → X) (nodeSem : Generator → X → X → X)
    (candidate : BTerm Generator Atom → X)
    (hatom : ∀ leaf, candidate (.atom leaf) = atomSem leaf)
    (hnode : ∀ generator left right,
      candidate (.node generator left right) =
        nodeSem generator (candidate left) (candidate right))
    (training : Set (BTerm Generator Atom)) (term : BTerm Generator Atom)
    (_hood : term ∉ training) :
    candidate term = eval atomSem nodeSem term := by
  exact fold_unique atomSem nodeSem candidate hatom hnode term

/-- A directed equation between binary terms. -/
structure RewriteRule (Generator Atom : Type*) where
  lhs : BTerm Generator Atom
  rhs : BTerm Generator Atom

/-- Context closure of a set of root rewrite rules. -/
inductive Step (rules : Set (RewriteRule Generator Atom)) :
    BTerm Generator Atom → BTerm Generator Atom → Prop where
  | root (rule : RewriteRule Generator Atom) (hmem : rule ∈ rules) :
      Step rules rule.lhs rule.rhs
  | left (generator : Generator) (right : BTerm Generator Atom)
      {before after : BTerm Generator Atom} (hstep : Step rules before after) :
      Step rules (.node generator before right) (.node generator after right)
  | right (generator : Generator) (left : BTerm Generator Atom)
      {before after : BTerm Generator Atom} (hstep : Step rules before after) :
      Step rules (.node generator left before) (.node generator left after)

/-- One certified rewrite preserves denotation in every binary-tree context. -/
theorem step_preserves_eval
    (atomSem : Atom → X) (nodeSem : Generator → X → X → X)
    (rules : Set (RewriteRule Generator Atom))
    (hsound : ∀ rule ∈ rules,
      eval atomSem nodeSem rule.lhs = eval atomSem nodeSem rule.rhs) :
    ∀ {before after}, Step rules before after →
      eval atomSem nodeSem before = eval atomSem nodeSem after := by
  intro before after hstep
  induction hstep with
  | root rule hmem => exact hsound rule hmem
  | left generator right hstep ih =>
      simpa only [eval] using
        congrArg (fun value => nodeSem generator value (eval atomSem nodeSem right)) ih
  | right generator left hstep ih =>
      simpa only [eval] using
        congrArg (fun value => nodeSem generator (eval atomSem nodeSem left) value) ih

/-- Any finite chain of certified rewrites preserves denotation. -/
theorem steps_preserve_eval
    (atomSem : Atom → X) (nodeSem : Generator → X → X → X)
    (rules : Set (RewriteRule Generator Atom))
    (hsound : ∀ rule ∈ rules,
      eval atomSem nodeSem rule.lhs = eval atomSem nodeSem rule.rhs)
    {before after : BTerm Generator Atom}
    (hsteps : Relation.ReflTransGen (Step rules) before after) :
    eval atomSem nodeSem before = eval atomSem nodeSem after := by
  induction hsteps with
  | refl => rfl
  | tail hprefix hlast ih =>
      exact ih.trans (step_preserves_eval atomSem nodeSem rules hsound hlast)

/-- A normalizer is semantically sound whenever it returns a term reachable by
certified rewrites.  Termination and confluence are separate obligations. -/
theorem certified_normalizer_sound
    (atomSem : Atom → X) (nodeSem : Generator → X → X → X)
    (rules : Set (RewriteRule Generator Atom))
    (hsound : ∀ rule ∈ rules,
      eval atomSem nodeSem rule.lhs = eval atomSem nodeSem rule.rhs)
    (normalize : BTerm Generator Atom → BTerm Generator Atom)
    (hreduces : ∀ term,
      Relation.ReflTransGen (Step rules) term (normalize term))
    (term : BTerm Generator Atom) :
    eval atomSem nodeSem (normalize term) = eval atomSem nodeSem term := by
  exact (steps_preserve_eval atomSem nodeSem rules hsound (hreduces term)).symm

end BTerm

section ErrorRecurrence

/-- Worst-case depth recurrence under aggregate child sensitivity `rho`. -/
def depthError (rho epsilon initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => epsilon + rho * depthError rho epsilon initial n

/-- The recurrence unfolds into transported initial error plus accumulated
local errors. -/
theorem depthError_eq_sum (rho epsilon initial : ℝ) (n : ℕ) :
    depthError rho epsilon initial n =
      rho ^ n * initial + epsilon * ∑ i ∈ Finset.range n, rho ^ i := by
  induction n with
  | zero => simp [depthError]
  | succ n ih =>
      rw [depthError, ih, Finset.sum_range_succ']
      simp_rw [pow_succ]
      rw [← Finset.sum_mul]
      ring

end ErrorRecurrence

end SentinelMath.OCI
