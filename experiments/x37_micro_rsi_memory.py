"""X37: the other two pillars at micro-scale -- self-extension, and memory.

X36 put Pillars 1 and 4 on a universal substrate: hypotheses are
S-expression programs rather than settings of a fixed simulator, and the
system writes its own adversarial tests (3/8 exact from one example ->
8/8 after self-testing, no extra data supplied).

This adds the remaining two.

PILLAR 3 -- non-gradient self-extension. Hand the system a task its
vocabulary cannot express. `DUP`, `REV` and `CAT` can only rearrange and
repeat what the input already contains; they cannot FILTER. So
"keep only the A's" has no program in the space, refutation returns EMPTY,
and that emptiness is the trigger. Candidate primitive families are then
proposed, each refuted, and the least expressive viable one adopted -- the
X34 rule, on a new substrate. No weights move; the change is a new entry in
a vocabulary, applied immediately.

PILLAR 2 -- memory as structure, not transcript. What is worth keeping from
a solved task is not the conversation but the CONSTRAINT: which primitive
the world demanded, and which program survived. A later session loads those
constraints and starts from a collapsed space instead of re-deriving them.
Measured as the cost to re-solve, cold versus warm.

THE ANCHOR THAT REMAINS, stated plainly because X34 and X35 hit the same
wall: the candidate families below are hand-written. The system chooses
among them by evidence, and it declines families it cannot execute, but it
does not invent `FILTER` from nothing. Growth discovers WHICH capability
reality demands; it does not conjure capability the machine lacks. Closing
that gap needs primitives synthesised from a generative meta-grammar, which
is the next experiment, not this one.

MEASURED: see main().
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x36_micro_vm import (
    Evidence,
    Program,
    VMError,
    enumerate_programs,
    refute,
    render,
    size,
)

# ------------------------------------------------- extended interpreter

ALPHABET = "AB"


def evaluate_ext(prog: Program, x: str, extra: dict, depth: int = 0) -> str:
    """X36's evaluator plus whatever primitives have been ADOPTED.

    `extra` maps a head symbol to a Python callable. An adopted primitive is
    a dictionary entry, not a retrained weight: the vocabulary grows at the
    cost of one insertion, and the very next refutation uses it.
    """
    if depth > 32:
        raise VMError("recursion")
    if prog == "x":
        return x
    head = prog[0]
    if head in extra:
        args = [evaluate_ext(p, x, extra, depth + 1) for p in prog[1:]]
        return extra[head](*args)
    if head == "DUP":
        a = evaluate_ext(prog[1], x, extra, depth + 1)
        if len(a) * 2 > 512:
            raise VMError("overflow")
        return a + a
    if head == "REV":
        return evaluate_ext(prog[1], x, extra, depth + 1)[::-1]
    if head == "CAT":
        a = evaluate_ext(prog[1], x, extra, depth + 1)
        b = evaluate_ext(prog[2], x, extra, depth + 1)
        if len(a) + len(b) > 512:
            raise VMError("overflow")
        return a + b
    raise VMError(f"unknown head {head!r}")


# ------------------------------------------------- candidate families


@dataclass(frozen=True, slots=True)
class PrimitiveFamily:
    """A proposed extension to the vocabulary."""

    name: str
    arity: int
    variants: tuple
    """Concrete instantiations, e.g. FILTER-A and FILTER-B."""
    make: object
    """variant -> callable implementing the primitive."""
    executable: bool = True
    """False when nothing in the machine can run it. Such a family is
    DECLINED rather than adopted, exactly as in X35: exactness has to be
    inherited from a working implementation, not asserted."""

    @property
    def cardinality(self) -> int:
        return len(self.variants)


FAMILIES = (
    PrimitiveFamily("FILTER", 1, tuple(ALPHABET),
                    lambda ch: (lambda a: "".join(c for c in a if c == ch))),
    PrimitiveFamily("DROP", 1, tuple(ALPHABET),
                    lambda ch: (lambda a: "".join(c for c in a if c != ch))),
    PrimitiveFamily("SORT", 1, ("asc", "desc"),
                    lambda d: (lambda a: "".join(sorted(a, reverse=(d == "desc"))))),
    PrimitiveFamily("HEAD", 1, ("1", "2"),
                    lambda n: (lambda a: a[: int(n)])),
    PrimitiveFamily("ORACLE", 1, ("magic",),
                    lambda _: None, executable=False),
)


def programs_with(head: str, base: list[Program], max_size: int) -> list[Program]:
    """Base programs, plus programs that apply the new head once or twice."""
    out = list(base)
    small = [p for p in base if size(p) <= max_size - 1]
    for p in small:
        out.append((head, p))
    smaller = [p for p in base if size(p) <= max_size - 2]
    for p in smaller:
        out.append((head, (head, p)))
        out.append(("CAT", (head, p), p))
        out.append(("CAT", p, (head, p)))
    return out


def refute_ext(programs: list[Program], evidence: Evidence, extra: dict) -> list[Program]:
    survivors = []
    for prog in programs:
        ok = True
        for x, want in evidence.pairs:
            try:
                if evaluate_ext(prog, x, extra) != want:
                    ok = False
                    break
            except (VMError, TypeError):
                ok = False
                break
        if ok:
            survivors.append(prog)
    return survivors


# ------------------------------------------------------ Pillar 2: memory

MEMORY_PATH = Path("corpus/micro_memory.json")


def load_memory() -> dict:
    """Constraints carried between sessions -- structure, not transcript."""
    if not MEMORY_PATH.exists():
        return {"primitives": {}, "solutions": {}}
    return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))


def save_memory(mem: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(mem, indent=2), encoding="utf-8")


def build_extra(mem: dict) -> dict:
    """Recompile remembered primitives into callables.

    A remembered fact is "the world demanded FILTER-A", not the text of how
    that was discovered. Loading it costs a dictionary insertion, which is
    the whole point of Pillar 2: memory is an executable constraint rather
    than a transcript to re-read.
    """
    extra = {}
    for head, variant in mem.get("primitives", {}).items():
        for fam in FAMILIES:
            if fam.name == head and fam.executable:
                extra[f"{head}-{variant}"] = fam.make(variant)
    return extra


# ------------------------------------------------------------- the task

def task_filter_a(s: str) -> str:
    """Keep only the A's. Unreachable with DUP/REV/CAT, which can rearrange
    and repeat but never discard."""
    return "".join(c for c in s if c == "A")


TRAIN = ("ABA", "BAAB", "AB")
HELD_OUT = ("A", "B", "BBAA", "ABAB", "BA", "AAB")


def solve(base_programs, evidence, extra, max_size, heads):
    """Refute across the base space plus every adopted head."""
    space = list(base_programs)
    for head in heads:
        space = programs_with(head, space, max_size)
    return refute_ext(space, evidence, extra), len(space)


def main() -> int:
    print("X37: self-extension and memory on the universal substrate\n")
    base = enumerate_programs(max_size=7)
    evidence = Evidence(tuple((x, task_filter_a(x)) for x in TRAIN))
    print(f"base vocabulary: DUP, REV, CAT   ({len(base):,} programs)")
    print(f"task: keep only the A's   evidence: "
          f"{', '.join(f'{x}->{y!r}' for x, y in evidence.pairs)}\n")

    # ---------------------------------------------------------- Pillar 3
    print("PILLAR 3 -- self-extension")
    t0 = time.perf_counter()
    survivors = refute(base, evidence)
    print(f"  base refutation: {len(survivors)} survivors "
          f"({time.perf_counter()-t0:.2f}s)")
    if survivors:
        print("  vocabulary still speaks this task; no growth needed")
        return 1
    print("  TRIGGER: refutation EMPTY -- the vocabulary cannot express this\n")

    contest = {}
    for fam in FAMILIES:
        if not fam.executable:
            print(f"  propose {fam.name:8} DECLINED: no implementation exists "
                  "(exactness must be inherited, not asserted)")
            continue
        best = None
        for variant in fam.variants:
            head = f"{fam.name}-{variant}"
            extra = {head: fam.make(variant)}
            surv, space_n = solve(base, evidence, extra, 7, [head])
            if surv:
                best = (variant, surv, space_n) if best is None or len(surv) < len(best[1]) else best
        if best is None:
            print(f"  propose {fam.name:8} ELIMINATED: no variant explains the evidence")
        else:
            variant, surv, space_n = best
            contest[fam.name] = (variant, surv, space_n)
            print(f"  propose {fam.name:8} VIABLE via {fam.name}-{variant}: "
                  f"{len(surv)} survivors of {space_n:,}")

    if not contest:
        print("\n  every candidate eliminated; giving up honestly")
        return 1

    # Occam over vocabularies: the least expressive family that survives.
    adopted = min(contest, key=lambda n: (
        next(f.cardinality for f in FAMILIES if f.name == n), len(contest[n][1])))
    variant, surv, _ = contest[adopted]
    head = f"{adopted}-{variant}"
    chosen = min(surv, key=lambda p: (size(p), render(p)))
    extra = {head: next(f.make(variant) for f in FAMILIES if f.name == adopted)}
    exact = all(
        (lambda: _safe(chosen, q, extra) == task_filter_a(q))() for q in HELD_OUT
    )
    print(f"\n  ADOPTED {head}; chosen program {render(chosen)}")
    print(f"  exact on held-out inputs: {'YES' if exact else 'NO'}  "
          f"({', '.join(HELD_OUT)})")
    print(f"  weights changed: none. vocabulary grew by one entry.\n")

    # ---------------------------------------------------------- Pillar 2
    print("PILLAR 2 -- memory as constraint, not transcript")
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink()

    t0 = time.perf_counter()
    cold_space = 0
    for fam in FAMILIES:
        if fam.executable:
            cold_space += sum(
                solve(base, evidence, {f"{fam.name}-{v}": fam.make(v)}, 7,
                      [f"{fam.name}-{v}"])[1] for v in fam.variants)
    cold_dt = time.perf_counter() - t0

    mem = load_memory()
    mem["primitives"][adopted] = variant
    mem["solutions"][ "filter-a"] = render(chosen)
    save_memory(mem)
    print(f"  saved: primitive {adopted}->{variant}, solution {render(chosen)}")

    warm_mem = load_memory()
    warm_extra = build_extra(warm_mem)
    t1 = time.perf_counter()
    warm_surv, warm_space = solve(base, evidence, warm_extra, 7, list(warm_extra))
    warm_dt = time.perf_counter() - t1
    print(f"  fresh session, memory loaded: search space {warm_space:,} "
          f"(cold contest searched {cold_space:,})")
    print(f"  re-solve: {len(warm_surv)} survivors in {warm_dt:.3f}s "
          f"vs {cold_dt:.3f}s cold  -> {cold_dt/max(warm_dt,1e-9):.1f}x cheaper")
    print(f"  collapse: {cold_space/max(warm_space,1):.1f}x smaller space")
    return 0


def _safe(prog, x, extra):
    try:
        return evaluate_ext(prog, x, extra)
    except (VMError, TypeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
