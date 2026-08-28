import sys
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import provisional as P
from x65a import latent_id as LI
from x65a import l1_safety as S
from x65a import l_suite as LS


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_nonvacuous_grounded_ood_unknown_and_restricted_cases(overlap):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    got = S.stratum_constructibility(fam, beh)
    alien = got["out_of_family_convention"]
    assert alien["constructible"]
    assert alien["family_membership_count"] == 0
    assert alien["grounded_contradiction"]["zero_survivors"]
    assert got["out_of_family_grounded_event"]["zero_survivors"]
    assert got["UNKNOWN_MEANING"]["derived_live_count"] == 0
    assert got["MISSING_REPRESENTATION"]["outcome"] == P.MISSING
    assert got["MISSING_REPRESENTATION"]["cause_posterior"][P.MISSING] == 1
    assert got["restricted_query_indistinguishable"]["constructible"]
    if overlap == "shared":
        assert not got["out_of_family_transfer_utterance"]["constructible"]
    else:
        assert got["out_of_family_transfer_utterance"]["constructible"]


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_restricted_promotions_are_empirical_and_calibration_fires(overlap):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    got = S.scope_audit(fam)
    assert got["constructible"]
    assert got["record"]["scope"]["status"] == "empirical"
    assert got["validation"]["passed"]
    assert got["false_global_promotions"] == 0
    assert got["calibration_false_global_promotions"] == 1
    assert got["calibration"]["fires"]
    assert not got["calibration"]["same_validator"]["passed"]


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_new_identity_metrics_are_derived_from_new_and_every_returning_trial(
        overlap):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    ids = LS.build_identities(fam, 6400, n=7)
    beh = EP.behaviour_table(fam.forms)
    got = S.new_identity_audit(fam, ids, beh)
    assert got["constructible"]
    c = got["confusion"]
    assert c["new_trials"] == 1
    assert c["returning_trials"] == len(ids) == 7
    assert got["precision"] == Fraction(
        c["true_new_created"],
        c["true_new_created"] + c["returning_false_new"])
    assert got["recall"] == Fraction(
        c["true_new_created"],
        c["true_new_created"] + c["new_false_negative"])
    assert got["false_new_rate_returning"] == Fraction(
        c["returning_false_new"], c["returning_trials"])
    assert got["forced_assimilation_rate"] == Fraction(
        c["new_forced_assimilation"], c["new_trials"])
    assert got["unresolved_new_rate"] == Fraction(
        c["new_unresolved"], c["new_trials"])
    promotion = got["promotion"]
    assert promotion["arm"] == "main"
    assert promotion["provisional_branch_opened"]
    assert promotion["confirmed_unchanged_while_provisional"]
    assert promotion["outcome"] == P.PROMOTE
    assert promotion["queries_used"] > 0
    assert promotion["confirmed_before"]["grounded"] == []
    assert promotion["branch"]["provisional_grounded"]
    assert promotion["confirmed_after"]["grounded"]
    assert promotion["validation"]["passed"]
    assert got["successfully_promoted_new_records"] == 1
    assert got["store_records_before_promotion"] == 7
    assert got["store_records_after_creation"] == 8
    assert not got["later_reuse_used_stable_identity"]
    assert got["later_reuse_shortlist"] is not None
    assert got["later_reuse_identity_posterior"] is not None
    assert got["later_reuse_task_posterior"] is not None
    assert got["later_reuse_action"] is not None
    assert got["later_reuse_query_history"] is not None
    assert got["later_reuse_of_new_records"] == int(
        got["successfully_promoted_new_records"] == 1
        and got["later_reuse_identity_top"] == got["promoted_record_key"]
        and got["later_reuse_identity_decision"] == LI.ASSIGN_EXISTING
        and got["promoted_record_key"] in got["later_reuse_shortlist"]
        and got["later_reuse_task_accuracy"] == 1
        and got["later_reuse_unresolved_branches"] == 0)
    assert got["later_reuse_validation"]["passed"] == bool(
        got["later_reuse_of_new_records"])
    assert got["arms"]["no_new_unresolved"]["new_recall"] == 0
    assert got["arms"]["no_new_unresolved"]["new_false_negative"] == 1
    assert got["arms"]["always_reuse_nearest"]["forced_assimilation_rate"] == 1
    assert got["arms"]["no_new_forced"]["forced_assimilation_rate"] == 1
    assert got["arms"]["always_create_new"]["returning_false_new"] == len(ids)
    assert got["arms"]["always_create_new"]["precision"] == Fraction(1, 8)
    assert got["arms"]["oracle_new_returning_status"]["precision"] == 1
    assert got["arms"]["oracle_new_returning_status"]["recall"] == 1
    assert got["promotion_calibration"]["same_validator_rejections"]
    assert got["reuse_calibration"]["same_validator_rejections"]
    assert got["reuse_validator_control"]["validation"]["passed"]
    assert all(got["calibration_fired"].values())


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_strict_later_reuse_preserves_the_frozen_unresolved_failure(overlap):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    got = S.new_identity_audit(
        fam, LS.build_identities(fam, 6400, n=7), beh, 6400)
    assert got["successfully_promoted_new_records"] == 1
    assert got["promoted_record_key"] in got["later_reuse_shortlist"]
    assert got["later_reuse_identity_top"] == got["promoted_record_key"]
    assert got["later_reuse_task_accuracy"] == 1
    assert got["later_reuse_identity_decision"] == LI.UNRESOLVED_IDENTITY
    assert got["later_reuse_unresolved_branches"] == 1
    assert not got["later_reuse_validation"]["passed"]
    assert got["later_reuse_of_new_records"] == 0
