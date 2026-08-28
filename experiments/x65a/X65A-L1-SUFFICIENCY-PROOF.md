# X65A-L1 grounded-pair sufficiency: finite-algebra proof

Status: mathematical proof for the controlled, finite, authored X64H/X65A
model. This document was not checked by Lean or another proof assistant. The
generated-stream differential audit is corroboration only; it is not the
proof. Nothing here establishes sufficiency for natural language, arbitrary
non-indicator evidence, a learned likelihood model, or hidden per-record
weights.

## 1. Finite model and assumptions

Fix one alphabet stratum. Let `Phi` be its finite convention family and `Z`
its 32 typed meanings. Let `U3(phi,z)` be the deterministic three-role
calibration utterance. A persistent grounded observation is exactly a pair
`(z,u)` with likelihood

    K_(z,u)(phi) = 1[U3(phi,z) = u].                         (1)

The theorem assumes:

1. `Phi` and `Z` are finite and the prior on `Phi` is uniform.
2. Every persistent factor is (1). Transfer observations are current evidence
   only; they are not compressed into this statistic.
3. The sketch retains every grounded `(z,u)` pair. It does not drop a hidden
   likelihood magnitude, reliability, selection propensity, or provenance
   weight.
4. For current evidence `e=(u,D,pool)`, the selection-aware channel
   `W_e(phi,z)` is a nonnegative function of the public family and current
   evidence. It has no dependency on full-record metadata omitted by the
   sketch.
5. A legal clarification `(zq,a)` has deterministic answer channel
   `a=U3(phi,zq)` for in-family conventions.
6. `NEW_IDENTITY` begins with the same finite family prior. Its support is
   intersected by clarification answers. `OUT_OF_FAMILY` uses the frozen
   current-utterance likelihood `1/A^2`; no unmodelled alien semantic-answer
   channel is invented.
7. Downstream predictions, thresholds, abstentions, and tie rules are
   deterministic functions of the posteriors derived below.

The executable `validate_sufficiency_domain` check enumerates every realizable
grounded pair and every convention/factor entry in both strata. It checks the
indicator premise, exact support reconstruction, uniform-prior normalization,
absence of hidden persistent weights, exact sketch fields, and independence of
the current channel from omitted record fields.

## 2. Statistic and stored posterior

For a grounded history `g=((z1,u1),...,(zt,ut))`, define

    S_g = {phi in Phi : U3(phi,zi)=ui for every i}.           (2)

By (1), the persistent likelihood product is

    product_i K_(zi,ui)(phi) = 1[phi in S_g].                (3)

With uniform prior `1/|Phi|`, Bayes normalization of (3), when `S_g` is
nonempty, gives

    q(phi | g) = 1[phi in S_g] / |S_g|.                      (4)

The grounded-pair sketch reconstructs (2) by equality tests against `U3`.
Therefore it reconstructs every weight in (4), not merely the support of an
otherwise weighted posterior. Full-record identity, provenance, status,
timestamps, and counters do not appear in (2)-(4).

## 3. Selection-aware current likelihood and task posterior

The generator chooses the task meaning uniformly from the legal demonstration
consistency set `D`. For current utterance evidence `e`, define

    W_e(phi,z) = p(u | phi,z,D,pool,accepted).

`W_e` can be nonuniform among conventions in `S_g`; the proof does not assume
otherwise. Marginalizing the joint over `phi` and `z` gives the identity-local
current-evidence likelihood

    L(e | g)
      = sum_(phi in S_g) sum_(z in D)
          q(phi|g) (1/|D|) W_e(phi,z)
      = [sum_(phi in S_g,z in D) W_e(phi,z)]
          / [|S_g| |D|].                                    (5)

Every quantity on the right of (5) is reconstructed from the sketch, public
family, and current task. Hence full record and sketch have identical identity
likelihoods. The task marginal is

    p(z | e,g)
      = sum_(phi in S_g) W_e(phi,z)
          / sum_(phi in S_g,z' in D) W_e(phi,z'),             (6)

so it is identical as well. Equations (5)-(6) explicitly retain nonuniform
selection-aware weights.

## 4. Clarification and query utility

A legal answer `a` to meaning query `zq` adds the indicator
`1[U3(phi,zq)=a]`. Thus

    S_(g+(zq,a))
      = S_g intersect {phi : U3(phi,zq)=a}.                  (7)

Induction on the number of clarification answers applies (4)-(6) after every
reachable query path.

For query selection, first condition each identity component on current
evidence. Its convention marginal is

    q(phi | g,e)
      proportional to 1[phi in S_g] sum_(z in D) W_e(phi,z). (8)

The exact answer distribution for candidate query `zq` is

    p(a | zq,e,g)
      = sum_phi q(phi | g,e) 1[U3(phi,zq)=a].                 (9)

Latent-identity inference mixes (9) with the exact identity posterior, then
renormalizes over components with a defined in-family answer channel.
Equations (8)-(9), not uniform counting inside `S_g`, determine query entropy.
Because the sketch reconstructs `S_g` and `W_e` is record-independent, it
reconstructs (8)-(9), every query utility, and the selected query.

## 5. NEW, OUT, and decision

`NEW_IDENTITY` is the same derivation with initial support `Phi`; (7) updates
that support after every answer. It uses no omitted old-record field.
`OUT_OF_FAMILY` contributes the frozen current likelihood `1/A^2`, also
independent of old-record fields. The theorem does not claim a semantic-answer
model for OUT.

The identity posterior, task posterior, NEW/OUT mass, and query distribution
are therefore equal between full record and sketch. Assumption 7 then implies
equal prediction, threshold decision, abstention, and tie behavior.

## 6. Why the indicator premise is necessary

The same domain validator receives a planted countermodel. It leaves the
grounded-pair support unchanged but changes one surviving persistent factor
from `1` to `1/2` and marks that magnitude as a hidden speaker-reliability
weight. The support-only sketch still assigns mass `1/|S_g|`, while the full
posterior assigns the planted convention mass

    (1/2) / (|S_g|-1+1/2) = 1/(2|S_g|-1),

which differs whenever `|S_g|>1`. The validator rejects this model on both
`all_persistent_factors_are_indicators` and
`no_hidden_persistent_weights`. This is a support-preserving counterexample:
support equality alone is not sufficient.

## 7. Claim boundary

Under assumptions 1-7, the grounded-pair sketch is an exact sufficient
statistic for the listed X65A-L1 inference operations in both finite authored
alphabet strata. The proof does not cover persistent transfer likelihoods,
non-indicator calibration noise, hidden record-specific weights, arbitrary
query channels, natural language, lifelong learning, or AGI capability.
