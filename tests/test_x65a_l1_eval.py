import sys
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x65a import l1_eval as E
from x65a import l1_inference as I
from x65a import l_suite as LS
from x64h import episode as EP
from x64h import family as F
from x65a.semantic_mem import surviving_mask


def test_legacy_q246_is_reproduced_and_not_called_a_budget():
    got = E.legacy_query_accounting()
    assert got["published_curve_reproduced"]
    assert got["budgets"][1]["task_accuracy"].numerator == 20
    assert got["budgets"][1]["task_accuracy"].denominator == 21
    assert got["budgets"][3]["queries_actually_asked"] == 295
    assert got["budgets"][3]["mean_over_all_tasks"].numerator == 59
    assert got["budgets"][3]["mean_over_all_tasks"].denominator == 24
    assert got["metrics_internally_consistent"]
    assert got["calibration"]["fires"]


def _cases(seed=400):
    fam = F.Family(F.FamilySpec(overlap="disjoint_op"))
    beh = EP.behaviour_table(fam.forms)
    ids = LS.build_identities(fam, seed)
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap="disjoint_op"), ids, seed, n_per=1)
    masks = [surviving_mask(fam, i.grounded) for i in ids]
    cases = []
    for probe in E.distinct_returning_population(probes):
        initial = I.make_latent_state(fam, beh, probe.task, masks)
        cases.append(E.EvaluationCase(
            initial, probe, seed, probe.equivalence))
    return fam, beh, probes, tuple(cases)


def test_memoryless_population_is_not_the_first_equivalent_pair():
    _fam, _beh, probes, _cases0 = _cases()
    matched = E.memoryless_population(probes, "all_matched_scored")
    distinct = E.memoryless_population(
        probes, "all_distinct_returning_slots")
    assert matched == E.matched_scored_population(probes)
    assert len(distinct) > 2
    assert len({p.slot for p in distinct}) == len(distinct)
    assert {p.slot for p in distinct} != {0, 1}


def test_aggregate_curve_is_prefix_consistent_and_uses_frozen_target():
    _fam, _beh, _probes, cases = _cases()
    curve = E.aggregate_accuracy_curve(
        cases, I.TASK_INFORMATION_GAIN, range(8), range(8),
        budgets=(0, 1, 2, 3, 4), target=Fraction(19, 20))
    assert curve["prefix_consistent"]
    assert curve["frozen_target_accuracy"] == Fraction(19, 20)
    assert tuple(curve["budgets"]) == (0, 1, 2, 3, 4)
    assert curve["minimum_questions_to_frozen_target"] == \
        E.minimum_questions_to_target(curve["budgets"], Fraction(19, 20))
    assert all(isinstance(row["task_accuracy"], Fraction)
               for row in curve["budgets"].values())


def test_per_task_truth_aware_hitting_time_is_forbidden():
    with pytest.raises(RuntimeError, match="frozen-target"):
        E.questions_to_correct()
