"""X64H-0 pins: the overlapping-codebook testbed.

H1 failed at 789161e because conventions used disjoint vocabularies. These
pin the properties that fix should have, and the ones it measurably has not
yet achieved."""

import math
import random
import sys

import pytest

sys.path.insert(0, "experiments")

from x64h import codebook as K
from x64h import semantic as S
from x64h import validity as V

FAM = K.full_family()
FS = K.forms()


def test_no_word_belongs_to_a_subset_of_conventions():
    """The X64H fix, stated as a property: every codeword is used by every
    convention, so the vocabulary of an utterance carries no information
    about which convention produced it."""
    la = K.leak_audit(FAM[:120], FS)
    assert la["words_not_used_by_every_convention"] == 0
    assert la["lengths_not_used_by_every_convention"] == 0
    assert la["leak_free"] is True


def test_a_planted_unique_token_leak_is_caught():
    planted = FAM[:40] + (K.Codebook(("o1", "o2"), ("z9", "c2", "c3", "c4"),
                                     K.W, 0),)
    assert K.leak_audit(planted, FS)["leak_free"] is False


def test_one_utterance_does_not_identify_the_convention():
    """The failure H1 named. A single utterance must leave many conventions
    live; at 789161e it left one."""
    phi, z = FAM[300], FS[5]
    u = K.realise(phi, z, ("O", "F"))
    live = sum(1 for p in FAM if any(K.consistent(p, zz, u) for zz in FS))
    assert live > len(FAM) // 4, f"only {live} of {len(FAM)} survive"


def test_the_convention_is_fixed_inside_an_episode():
    """V12. The convention must change only at the declared boundary."""
    utt, beh = V.precompute(FAM[:64], FS)
    cls = K.equivalence_classes(FAM[:64], FS)
    zs = random.Random(3).sample(list(FS), 12)
    seen = []

    class Probe(list):
        def __getitem__(self, i):
            seen.append(i)
            return list.__getitem__(self, i)

    r = V.run_episode(FAM[:64], FS, utt, beh, cls, 7, zs, "oracle",
                      n_demos=2, rng=random.Random(1))
    assert len(r.correct) == len(zs)
    ch = [random.Random(50 + t).randrange(64) for t in range(len(zs))]
    r2 = V.run_episode(FAM[:64], FS, utt, beh, cls, 7, zs, "persist",
                       n_demos=2, rng=random.Random(1), changing=ch)
    assert len(r2.correct) == len(zs)


def test_the_convention_posterior_concentrates_across_an_episode():
    """V4. If this stops holding the family is ambiguous but not learnable,
    which is the failure mode the sweep spent the most time on."""
    utt, beh = V.precompute(FAM, FS)
    cls = K.equivalence_classes(FAM, FS)
    zs = random.Random(11).sample(list(FS), 24)
    r = V.run_episode(FAM, FS, utt, beh, cls, 404, zs, "persist",
                      n_demos=2, rng=random.Random(5), alpha=1.0, schedule=3)
    assert r.entropy[0] > r.entropy[-1] + 3.0, r.entropy[:3] + r.entropy[-3:]
    assert r.true_class_mass[-1] > r.true_class_mass[0] + 0.5


def test_teaching_and_withholding_demonstrations_differ():
    """The mixed schedule is load-bearing: a uniform one gives either
    ambiguity or learnability, never both."""
    utt, _b = V.precompute(FAM, FS)
    phi_i, z_i = 300, 5
    u = K.realise(FAM[phi_i], FS[z_i], ("O", "F"))
    teach = V.teacher_demo(FS, utt, phi_i, z_i, u, 2, S.UNIVERSE[:24],
                           exposed=("O", "F"), alpha=0.0)
    hold = V.teacher_demo(FS, utt, phi_i, z_i, u, 2, S.UNIVERSE[:24],
                          exposed=("O", "F"), alpha=1.0)
    n_teach = len({S.denote(z) for z in FS
                   if all(S.execute(z)(x) == y for x, y in teach)})
    n_hold = len({S.denote(z) for z in FS
                  if all(S.execute(z)(x) == y for x, y in hold)})
    assert n_hold >= n_teach, (n_teach, n_hold)


def test_no_final_manifest_and_no_final_seed():
    """The standing constraint for the whole of X64H-0."""
    from x64h import protocol as PR
    from x64h.types import TaintError
    ok, _why = PR.manifest_committed()
    assert ok is False
    with pytest.raises(TaintError):
        PR.release_final_seeds()
