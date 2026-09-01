"""The Scale-1A-1 screen matrix is a contract, so it is tested like one.

Each guard here has a calibration arm: a case the guard must reject. A guard
that has never been shown to fail on a known-bad input has not been shown to do
anything.
"""

from __future__ import annotations

import pytest

from sentinel.wm import matrix_scale1 as m
from sentinel.wm.latent_contract import ContractViolation


# ---- arithmetic -------------------------------------------------------------------------


def test_counts_match_the_specification() -> None:
    a = m.matrix_arithmetic()
    assert a["screen_workloads"] == 72
    assert a["reactive_control_workloads"] == 12
    assert a["pooling_ablation_workloads"] == 3
    assert a["total_workloads"] == 87


def test_counts_are_the_product_of_their_factors() -> None:
    a = m.matrix_arithmetic()
    assert a["screen_workloads"] == a["interfaces"] * a["arms"] * a["seeds"]


def test_workload_list_length_agrees_with_the_arithmetic() -> None:
    assert len(m.all_workloads()) == m.matrix_arithmetic()["total_workloads"]


def test_every_workload_id_is_unique() -> None:
    ids = [w.workload_id for w in m.all_workloads()]
    assert len(set(ids)) == len(ids)


# ---- reactive controls are not world models ---------------------------------------------


def test_reactive_controls_are_excluded_from_the_world_model_count() -> None:
    a = m.matrix_arithmetic()
    assert a["world_model_workloads"] == 75
    assert a["world_model_workloads"] + a["reactive_control_workloads"] == a["total_workloads"]


def test_reactive_controls_carry_no_parameter_target() -> None:
    for cell in m.reactive_control_cells():
        assert cell.target_parameters is None


def test_is_world_model_flag_partitions_the_workloads() -> None:
    workloads = m.all_workloads()
    models = [w for w in workloads if w.is_world_model]
    controls = [w for w in workloads if not w.is_world_model]
    assert len(models) == 75
    assert len(controls) == 12


# ---- the alias table --------------------------------------------------------------------


def test_every_alias_names_an_interface_that_builds() -> None:
    assert m.assert_aliases_resolve() == dict(m.INTERFACE_ALIASES)


def test_alias_table_covers_every_screen_interface() -> None:
    assert set(m.INTERFACE_ALIASES) == set(m.SCREEN_INTERFACES)


def test_a_broken_alias_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration: the alias check must fail when an alias names nothing."""
    broken = dict(m.INTERFACE_ALIASES) | {"qwen_spatial_slots": "no_such_interface"}
    monkeypatch.setattr(m, "INTERFACE_ALIASES", broken)
    with pytest.raises(ContractViolation, match="do not build"):
        m.assert_aliases_resolve()


def test_unknown_interface_name_is_refused() -> None:
    with pytest.raises(ContractViolation, match="not a screen interface"):
        m.resolve_interface("qwen3_vl_4b_spatial_slots")  # implementation name, not spec name


def test_cells_expose_both_names() -> None:
    cell = m.screen_cells()[0]
    assert cell.interface == "qwen_spatial_slots"
    assert cell.implementation == "qwen3_vl_4b_spatial_slots"


def test_pooling_ablation_resolves_to_the_mean_pool_interface() -> None:
    (cell,) = m.pooling_ablation_cells()
    assert cell.implementation == m.POOLING_ABLATION_IMPLEMENTATION
    assert cell.arm == "continuous_action_recurrent"


# ---- seeds ------------------------------------------------------------------------------


def test_screen_seeds_are_accepted() -> None:
    m.assert_no_final_seed(m.SCREEN_SEEDS)


def test_scale0_seeds_are_refused() -> None:
    """Calibration: reusing a Scale-0 seed would screen on already-seen environments."""
    with pytest.raises(ContractViolation, match="collide with Scale-0"):
        m.assert_no_final_seed(m.SCALE0_DEVELOPMENT_SEEDS)


def test_an_unlisted_seed_is_refused() -> None:
    """Calibration: a final seed must not be reachable by passing it in directly."""
    with pytest.raises(ContractViolation, match="not screen seeds"):
        m.assert_no_final_seed((9999,))


def test_screen_seeds_are_disjoint_from_scale0() -> None:
    assert not set(m.SCREEN_SEEDS) & set(m.SCALE0_DEVELOPMENT_SEEDS)


# ---- freezing ---------------------------------------------------------------------------


def test_digest_is_stable_across_calls() -> None:
    assert m.matrix_digest() == m.matrix_digest()


def test_digest_changes_when_the_matrix_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration: a digest that never moves cannot detect an amendment."""
    before = m.matrix_digest()
    monkeypatch.setattr(m, "SCREEN_SEEDS", (7700, 7701))
    assert m.matrix_digest() != before


def test_all_six_arms_appear_at_every_screen_interface() -> None:
    by_interface: dict[str, set[str]] = {}
    for cell in m.screen_cells():
        by_interface.setdefault(cell.interface, set()).add(cell.arm)
    expected = {arm.name for arm in m.ARMS}
    assert set(by_interface) == set(m.SCREEN_INTERFACES)
    for interface, arms in by_interface.items():
        assert arms == expected, interface
