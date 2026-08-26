"""Layer 0: the finite bijection sanity test, reproduced from the theory
package's reference enumeration.

Nothing here depends on Sentinel. It exists to check that this
implementation reads the theory the same way the reference script does,
before any of the FT-SPCFG machinery is trusted. The reference uses numpy,
scipy and matplotlib; this reimplementation uses the standard library only,
so agreement is evidence about the mathematics rather than about a shared
dependency.

Reference values to reproduce, from the brief:
  separating probability, k=4, m=1..4   0, 0.09375, 0.41015625, 0.66650390625
  posterior entropy over 4! bijections  4.58496 bits
  largest answer alphabet               6
  Fano-style lower bound                1.77371 questions
  optimal expected questions            2.0
  greedy information gain               2.0
  random disagreement                   2.857142857...
"""

from __future__ import annotations

import functools
import itertools
import math

Perm = tuple[int, ...]


def signatures(k: int, contexts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Atom -> its membership pattern across the observed contexts."""
    return tuple(tuple((c >> a) & 1 for c in contexts) for a in range(k))


def is_separating(k: int, contexts: tuple[int, ...]) -> bool:
    return len(set(signatures(k, contexts))) == k


def exact_separating_probability(k: int, m: int) -> float:
    total = separating = 0
    for family in itertools.product(range(1 << k), repeat=m):
        total += 1
        separating += is_separating(k, family)
    return separating / total


def closed_form_separating_probability(k: int, m: int) -> float:
    """(2^m)_k / (2^m)^k -- the falling factorial over all labellings."""
    n = 1 << int(m)
    if n < k:
        return 0.0
    return math.prod(range(n - k + 1, n + 1)) / (n ** k)


def apply_context(perm: Perm, context: int) -> int:
    surface = 0
    for atom, word in enumerate(perm):
        if (context >> atom) & 1:
            surface |= 1 << word
    return surface


def partition(hyps: tuple[Perm, ...], query: int) -> tuple[tuple[Perm, ...], ...]:
    parts: dict[int, list[Perm]] = {}
    for h in hyps:
        parts.setdefault(apply_context(h, query), []).append(h)
    return tuple(tuple(p) for _a, p in sorted(parts.items()))


def entropy_of_query(hyps: tuple[Perm, ...], query: int) -> float:
    n = len(hyps)
    return -sum((len(p) / n) * math.log2(len(p) / n)
                for p in partition(hyps, query))


def query_statistics(k: int) -> dict[str, float | int]:
    """Optimal, greedy-information-gain and random-disagreement policies over
    the same question pool, the same stopping rule and the same answer
    access -- which is the comparison X64H's arms 7 and 8 have to make."""
    hyps = tuple(itertools.permutations(range(k)))
    queries = tuple(range(1, (1 << k) - 1))

    @functools.lru_cache(maxsize=None)
    def optimal(state: tuple[Perm, ...]) -> float:
        if len(state) <= 1:
            return 0.0
        best = math.inf
        for q in queries:
            parts = partition(state, q)
            if len(parts) <= 1:
                continue
            v = 1.0 + sum((len(p) / len(state)) * optimal(p) for p in parts)
            best = min(best, v)
        return best

    @functools.lru_cache(maxsize=None)
    def greedy(state: tuple[Perm, ...]) -> float:
        if len(state) <= 1:
            return 0.0
        cands = [(entropy_of_query(state, q), -q, q) for q in queries
                 if len(partition(state, q)) > 1]
        q = max(cands)[2]
        return 1.0 + sum((len(p) / len(state)) * greedy(p)
                         for p in partition(state, q))

    @functools.lru_cache(maxsize=None)
    def random_disagreement(state: tuple[Perm, ...],
                            available: tuple[int, ...]) -> float:
        if len(state) <= 1:
            return 0.0
        informative = tuple(q for q in available
                            if len(partition(state, q)) > 1)
        if not informative:
            return math.inf
        acc = []
        for q in informative:
            rest = tuple(x for x in available if x != q)
            acc.append(1.0 + sum((len(p) / len(state))
                                 * random_disagreement(p, rest)
                                 for p in partition(state, q)))
        return sum(acc) / len(acc)

    max_answers = max(len(partition(hyps, q)) for q in queries)
    entropy = math.log2(math.factorial(k))
    return {
        "k": k,
        "hypotheses": len(hyps),
        "posterior_entropy_bits": entropy,
        "largest_answer_alphabet": max_answers,
        "entropy_lower_bound_questions": entropy / math.log2(max_answers),
        "optimal_expected_questions": optimal(hyps),
        "greedy_information_gain_expected_questions": greedy(hyps),
        "random_disagreement_expected_questions":
            random_disagreement(hyps, queries),
    }


def majority_bit_error(reps: int, noise: float) -> float:
    thr = reps // 2 + 1
    return sum(math.comb(reps, e) * noise ** e * (1 - noise) ** (reps - e)
               for e in range(thr, reps + 1))


def noise_recovery(k: int, m: int, noise: float, reps: int) -> dict[str, float]:
    be = majority_bit_error(reps, noise)
    return {
        "repetitions": reps,
        "exact_all_signature_success": (1 - be) ** (k * m),
        "hoeffding_success_lower_bound":
            1 - min(1.0, k * m * math.exp(-2 * reps * (0.5 - noise) ** 2)),
    }


def indistinguishable_example() -> dict[str, object]:
    """A context family that is NOT separating leaves a genuine automorphism:
    two atoms with identical signatures may be swapped without changing any
    observation. The evaluator must score the equivalence CLASS."""
    k, contexts = 4, (0b0011,)
    sig = signatures(k, contexts)
    a, b = next((i, j) for i in range(k) for j in range(i + 1, k)
                if sig[i] == sig[j])
    identity = tuple(range(k))
    swapped = list(identity)
    swapped[a], swapped[b] = swapped[b], swapped[a]
    swapped = tuple(swapped)
    return {
        "contexts": list(contexts),
        "duplicate_atoms": [a, b],
        "identity": list(identity),
        "swapped": list(swapped),
        "observationally_identical": all(
            apply_context(identity, c) == apply_context(swapped, c)
            for c in contexts),
        "class_size_at_least": 2,
    }
