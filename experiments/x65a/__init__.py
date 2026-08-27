"""X65A: Verified Dependency-Factored Memory.

Staged in six internal phases. This package currently implements X65A-0
only: schemas, canonical serialization, taint, leakage checks, exact
posterior microcases, and genuine process restart. Nothing here measures
continual learning, and no stream seed exists.

The X64H prerequisite is checked fail-closed by `prereq.py` before any
X65A artifact is produced.
"""
