"""X65A-0: four separate graphs, and the boundary between them.

The addendum's rule is the whole point of splitting them:

    Never infer probabilistic independence merely from missing provenance
    edges.

A provenance graph records where a claim came from. A factor graph records
which variables are conditionally dependent. They are different objects
about different things, and the absence of an edge in the first says
nothing about the second. `ProbabilisticFactorGraph` therefore refuses to
be built from a `ProvenanceGraph`, and `independent()` consults only
DECLARED factor edges -- an undeclared pair is UNKNOWN, not independent.

`EvaluatorDependencyDAG` is ORACLE_ONLY. The agent writer cannot serialize
it, and only the oracle and scoring interfaces may read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import DependencyEdge, EdgeKind, Taint, TaintError


class _Graph:
    kind = "abstract"
    allowed_edges: frozenset = frozenset()

    def __init__(self):
        self.nodes: set = set()
        self.edges: list = []

    def add_node(self, n) -> None:
        self.nodes.add(n)

    def add_edge(self, e: DependencyEdge, justification: str) -> None:
        if e.edge_kind not in self.allowed_edges:
            raise TaintError(f"{self.kind} does not carry "
                             f"{e.edge_kind.value} edges")
        if justification not in self.JUSTIFICATIONS:
            raise TaintError(
                f"an edge in {self.kind} may be created only from "
                f"{sorted(self.JUSTIFICATIONS)}, not {justification!r}")
        self.nodes.add(e.source_id)
        self.nodes.add(e.target_id)
        self.edges.append(e)

    def successors(self, n):
        return [e.target_id for e in self.edges if e.source_id == n]

    def closure(self, n, context: str | None = None) -> set:
        """Context-compatible descendant closure. Revision may touch this
        set and nothing else."""
        seen, stack = set(), [n]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            for e in self.edges:
                if e.source_id != v:
                    continue
                if context is not None and e.context_predicate not in (
                        "*", context):
                    continue
                stack.append(e.target_id)
        return seen

    def canon(self):
        return {"kind": self.kind, "nodes": sorted(self.nodes),
                "edges": [e.canon() for e in self.edges]}


class ProvenanceGraph(_Graph):
    """source observations -> derived claims/programs."""
    kind = "provenance"
    allowed_edges = frozenset({EdgeKind.DERIVES, EdgeKind.SUPPORTS,
                               EdgeKind.CONTRADICTS, EdgeKind.INVALIDATES})
    JUSTIFICATIONS = frozenset({"observed_evidence", "verified_derivation"})


class ProgramDependencyGraph(_Graph):
    """procedure composition and effect dependencies."""
    kind = "program_dependency"
    allowed_edges = frozenset({EdgeKind.COMPOSES, EdgeKind.INVALIDATES})
    JUSTIFICATIONS = frozenset({"verified_composition"})


class ProbabilisticFactorGraph(_Graph):
    """conditional-dependence structure used by Bayesian revision.

    Dependence must be DECLARED. Three states, not two: declared dependent,
    declared independent, and unknown. Only the second licenses a locality
    claim."""
    kind = "probabilistic_factor"
    allowed_edges = frozenset({EdgeKind.SUPPORTS, EdgeKind.CONTEXT_GATES,
                               EdgeKind.INVALIDATES})
    JUSTIFICATIONS = frozenset({"declared_model", "verified_conditional_change"})

    def __init__(self):
        super().__init__()
        self._declared_independent: set = set()

    def declare_independent(self, a, b, justification: str) -> None:
        if justification not in self.JUSTIFICATIONS:
            raise TaintError("independence must be declared by the model or "
                             "supported by a verified conditional change")
        self._declared_independent.add(frozenset({a, b}))
        self.nodes.update({a, b})

    def independent(self, a, b) -> str:
        if frozenset({a, b}) in self._declared_independent:
            return "DECLARED_INDEPENDENT"
        if b in self.closure(a) or a in self.closure(b):
            return "DEPENDENT"
        return "UNKNOWN"

    @classmethod
    def from_provenance(cls, _p: ProvenanceGraph):
        raise TaintError(
            "probabilistic independence may not be inferred from provenance "
            "structure: a missing provenance edge records that nothing was "
            "derived, not that two variables are conditionally independent")


@dataclass
class EvaluatorDependencyDAG:
    """Hidden future task relationships. ORACLE_ONLY: readable by the
    oracle and scoring interfaces, never by the agent, and never
    serializable by the agent writer."""
    taint: Taint = Taint.ORACLE_ONLY
    edges: list = field(default_factory=list)
    nodes: set = field(default_factory=set)

    def add(self, src, dst, kind: str) -> None:
        self.edges.append((src, dst, kind))
        self.nodes.update({src, dst})

    def canon(self):
        raise TaintError("the evaluator dependency DAG is ORACLE_ONLY and "
                         "may not be serialized by the agent writer")

    def oracle_view(self, token: str):
        if token != "ORACLE":
            raise TaintError("evaluator DAG accessed without the oracle "
                             "channel")
        return list(self.edges)


def graph_set():
    return {"provenance": ProvenanceGraph(),
            "program_dependency": ProgramDependencyGraph(),
            "probabilistic_factor": ProbabilisticFactorGraph(),
            "evaluator": EvaluatorDependencyDAG()}
