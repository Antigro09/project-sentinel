"""The freeze seal: hashes cover every component, mutation breaks it, and no
final seed can be produced before the manifest is committed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "experiments")

from x64h import freeze0c as FZ
from x64h.types import TaintError
from x64h_0c_audit import ARMS_0C, BASE


def test_every_named_component_is_hashed():
    """Section I names the artifacts that must be frozen. Each one must
    appear as its own digest, or the seal does not cover it."""
    need = {"alphabet_families", "calibration_transfer_schedule",
            "teacher_selection_likelihood", "exact_inference", "query_policy",
            "conflict_model", "open_world_other", "evaluator", "gates",
            "bootstrap", "persistence", "structural"}
    assert need <= set(FZ.component_digests(BASE, ARMS_0C))


def test_the_digest_does_not_depend_on_the_commit():
    """`protocol.freeze_digest` folds HEAD in, so committing the manifest
    invalidates it. This one must not."""
    a = FZ.freeze_digest(BASE, ARMS_0C)
    b = FZ.freeze_digest(BASE, ARMS_0C)
    assert a == b
    assert FZ._commit() not in json.dumps(
        FZ.component_digests(BASE, ARMS_0C))


def test_mutating_a_frozen_source_breaks_the_seal(tmp_path):
    src = FZ.ROOT / "x64h" / "family.py"
    before = src.read_bytes()
    try:
        src.write_bytes(before + b"\n# planted mutation\n")
        assert FZ.freeze_digest(BASE, ARMS_0C) != json.loads(
            FZ.MANIFEST.read_text())["digest"]
        ok, why = FZ.manifest_committed(BASE, ARMS_0C)
        assert not ok and "changed" in why
        with pytest.raises(TaintError):
            FZ.release_final_seeds(BASE, ARMS_0C)
    finally:
        src.write_bytes(before)
    assert FZ.manifest_committed(BASE, ARMS_0C)[0]


def test_released_seeds_are_a_function_of_the_digest():
    d = json.loads(FZ.SEEDS.read_text())
    digest = d["digest"]
    for fam, seeds in d["seeds"].items():
        for i, s in enumerate(seeds):
            assert s == int(FZ._sha(f"{digest}:{fam}:{i}".encode())[:8], 16)


def test_final_seeds_do_not_collide_with_any_development_split():
    d = json.loads(FZ.SEEDS.read_text())
    used = set(range(400, 412)) | set(range(500, 508)) | set(range(700, 712))
    for seeds in d["seeds"].values():
        assert not (set(seeds) & used)
    assert len(set(d["seeds"]["shared"])) == len(d["seeds"]["shared"])


def test_at_least_four_seeds_per_family():
    d = json.loads(FZ.SEEDS.read_text())
    assert set(d["seeds"]) == {"shared", "disjoint_op"}
    assert all(len(v) >= 4 for v in d["seeds"].values())
