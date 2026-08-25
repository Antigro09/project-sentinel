"""X61: half of ordinary parsing needs a working set that grows.

X60 ended on a question and the plan for this step assumed an answer: that
almost all compiler, parsing and translation work runs in a bounded working
set, so registers are enough and the next move is a bigger substrate. That is
measurable rather than arguable.

The minimum working set of a string-to-string task is its Myhill-Nerode index:
how many prefix classes must be told apart, where two prefixes are equivalent
when the output produced AFTER them agrees for every suffix. A bounded index
means k registers suffice once |alphabet|^k reaches it. A growing one means no
register machine does the task at any k.

MEASURED ON THE DIAGONAL, and that qualifier is the methodological point.
Sweeping prefix length with suffixes fixed caps the index at what a short
suffix can distinguish; sweeping suffix length with prefixes fixed caps it at
what a short prefix can reach. Both produced tidy plateaus for nested
brackets -- 4 and then 5 -- and both were artefacts of the horizon. Growing
prefix and suffix together is the only reading with no ceiling built in:

    task                           n=1     n=2     n=3     n=4     n=5
    strip comment                    2       2       2       2       2   bounded
    capture quoted                   2       2       2       2       2   bounded
    dedupe adjacent                  6       6       6       6       6   bounded
    emit matching first              6       6       6       6       6   bounded
    capture brackets (nested)        2       3       4       5       6   GROWING
    only at depth 2                  1       3       4       5       6   GROWING
    balanced prefix                  3       4       5       6       7   GROWING
    reverse                          6      31     156     781   3,906  GROWING

FOUR OF EIGHT. Not "almost all". The premise this step was to be built on is
false on this sample, and the split is even rather than lopsided.

WHAT IT MEANS FOR THE ARCHITECTURE, which is better news than the headline.
Every growing task is one this machine already solves, and it solves them with
the STACK -- the unbounded memory it has had since X50 and which X60's
register work never replaced. Registers and stack are not competing designs
for the same problem; they cover different tasks. X60's registers handle
`emit matching first` and `dedupe adjacent`, which no stack discipline
reaches; the stack handles nesting and reversal, which no bounded register
reaches at any k. The machine needed both, and now there is a measurement
saying why rather than an intuition.

WHAT THIS DOES NOT SHOW. Eight tasks over a five-byte alphabet, hand-chosen to
span the cases. Which half is larger in real work depends on the task mix and
nothing here samples one. It also says nothing about whether a task with a
bounded index is EASY -- index is a memory requirement, not a search cost, and
X51 spent three experiments on a task whose index is trivial.
"""

from __future__ import annotations

import itertools
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

ALPHA = "ab()#"


# ---- the tasks, written as plain functions so the measurement is about the
# ---- TASK and not about any machine that might implement it.

def strip_comment(s):
    return s.split("#")[0]


def capture_quoted(s):
    out, inside = [], False
    for c in s:
        if c == "#":                      # '#' doubles as the quote here
            inside = not inside
        elif inside:
            out.append(c)
    return "".join(out)


def capture_brackets(s):
    out, depth = [], 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth > 0:
            out.append(c)
    return "".join(out)


def only_at_depth_two(s):
    out, depth = [], 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth == 2:
            out.append(c)
    return "".join(out)


def emit_matching_first(s):
    return "".join(c for c in s[1:] if s and c == s[0])


def dedupe_adjacent(s):
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def reverse(s):
    return s[::-1]


def balanced_prefix(s):
    """Emit only while the brackets seen so far are balanced-or-open."""
    out, depth = [], 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                break
        out.append(c)
    return "".join(out)


TASKS = {
    "strip comment": strip_comment,
    "capture quoted": capture_quoted,
    "dedupe adjacent": dedupe_adjacent,
    "emit matching first": emit_matching_first,
    "capture brackets (nested)": capture_brackets,
    "only at depth 2": only_at_depth_two,
    "balanced prefix": balanced_prefix,
    "reverse": reverse,
}


def residual(f, p, suffixes):
    """What the task still produces after prefix `p`, for each suffix.

    None marks a suffix where the prefix's own output is not a prefix of the
    whole output -- the task retracted something -- which is itself a
    distinguishing observation rather than a case to skip.
    """
    base = f(p)
    out = []
    for s in suffixes:
        whole = f(p + s)
        out.append(whole[len(base):] if whole.startswith(base) else None)
    return tuple(out)


def nerode_index(f, length, alpha, suffix_len=3):
    prefixes = ["".join(t) for n in range(length + 1)
                for t in itertools.product(alpha, repeat=n)]
    suffixes = ["".join(t) for n in range(suffix_len + 1)
                for t in itertools.product(alpha, repeat=n)]
    return len({residual(f, p, suffixes) for p in prefixes})


def main() -> int:
    t0 = time.perf_counter()
    print("X61: minimum working set of real parsing tasks\n")
    print(f"alphabet {ALPHA!r}. Index = distinguishable prefix classes.\n")
    print("MEASURED IN BOTH DIRECTIONS, because measuring one is misleading.")
    print("A suffix horizon CAPS the index: with suffixes of length 3 you can")
    print("only tell bracket depths 0,1,2,3+ apart, so a genuinely unbounded")
    print("task reports a tidy plateau of 4. Growth in EITHER direction means")
    print("the working set is unbounded.\n")

    # THE DIAGONAL. Sweeping prefixes with suffixes fixed caps the index at
    # what a short suffix can distinguish; sweeping suffixes with prefixes
    # fixed caps it at what a short prefix can reach. Both looked like tidy
    # plateaus for nested brackets, and both were artefacts. Growing the two
    # together is the only reading with no ceiling built into it.
    ns = (1, 2, 3, 4, 5)
    head = (f'{"task":26} ' + " ".join(f"{'n='+str(n):>7}" for n in ns)
            + f'  {"verdict":>10}')
    print(head + "\n" + "-" * len(head))

    verdicts = {}
    for name, f in TASKS.items():
        idx = [nerode_index(f, n, ALPHA, suffix_len=n) for n in ns]
        bounded = idx[-1] == idx[-2]
        verdicts[name] = (idx, bounded)
        print(f"{name:26} " + " ".join(f"{v:>7,}" for v in idx)
              + f"  {('bounded' if bounded else 'GROWING'):>10}")

    print("\nWHAT THIS COSTS THE MACHINE")
    print(f"  one register over this alphabet holds {len(ALPHA)+1} states, "
          f"two hold {(len(ALPHA)+1)**2}.")
    for name, (idx, bounded) in verdicts.items():
        if bounded:
            k = 1 if idx[-1] <= len(ALPHA) + 1 else 2
            print(f"  {name:26} index {idx[-1]:>5}  -> {k} register(s)")
        else:
            print(f"  {name:26} index {idx[-1]:>5}+ -> a STACK, not registers")

    growing = [n for n, (_, b) in verdicts.items() if not b]
    print("\nREADING")
    print(f"  bounded working set : {len(verdicts)-len(growing)}/{len(verdicts)}")
    print(f"  growing working set : {len(growing)}/{len(verdicts)}")
    b = len(verdicts) - len(growing)
    print(f"\n  The premise under test was that almost all parsing runs in a")
    print(f"  bounded working set, so registers are enough. {b} of "
          f"{len(verdicts)} do.")
    print("  Every task that does not is one this machine already solves, and")
    print("  it solves them with the STACK -- the unbounded memory it has had")
    print("  since X50. Registers and stack are not competing designs; they")
    print("  cover different tasks, and which half is larger depends entirely")
    print("  on the task mix, not on anything measured here.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
