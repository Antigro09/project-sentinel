"""Pinned recurrence certificates, replayed against the live environment.

Gate K2 asserts that legally reachable states exist which share a complete
AgentVisiblePacket, differ in hidden phase, and reach a different public outcome
under the same action. A count in a report is not that assertion; these are, and
they fail if the environment, the renderer or the packet schema drifts.

The pinned routes below have different lengths on purpose. Under packet v1 the
step travelled in `timestamp_ns`, so pairs at different steps could not alias and
the certificate count was 9,581. Under v2 timing is the constant `delta_t` and
the same pairs alias, taking the count to 39,556. These cases are exactly the ones
v1 could not express, so they would silently stop being certificates if the step
ever returned to the packet.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS, ProceduralVisualV2Adapter
from sentinel.wm.authority import AuthorityGate
from sentinel.wm.versioning import digest_array

# layout, route_a, route_b, expected polarity a/b, expected successors a/b
CERTIFICATES = (
    (90_000, (2,), (2, 0, 2), 0, 1,
     (86.0, 99.0, 110.0, 97.0), (110.0, 97.0, 86.0, 99.0)),
    (90_000, (2,), (0, 2, 2, 0, 2), 0, 1,
     (86.0, 99.0, 110.0, 97.0), (110.0, 97.0, 86.0, 99.0)),
    (90_000, (2,), (0, 0, 2, 2, 0, 2), 0, 1,
     (86.0, 99.0, 110.0, 97.0), (110.0, 97.0, 86.0, 99.0)),
)


def _replay(adapter, gate, layout, route):
    adapter.reset(layout)
    for action in route:
        adapter.step(action, gate.authorize_evaluator(action, "cert"))
    truth = adapter.snapshot().reveal("evaluator")
    snapshot = adapter.snapshot()
    successors = []
    for candidate in ACTIONS:
        adapter.restore(snapshot)
        adapter.step(candidate, gate.authorize_evaluator(candidate, "cert-succ"))
        successors.append(float(adapter.probes().values["observable_signature"]))
    adapter.restore(snapshot)
    return {
        "frame": digest_array(adapter.frame()).digest,
        "goal": adapter.goal_text(),
        "polarity": int(truth["polarity"]),
        "position": tuple(int(v) for v in truth["position"]),
        "blocked": bool(truth["last_blocked"]),
        "step": int(truth["step"]),
        "successors": tuple(successors),
    }


@pytest.mark.parametrize(
    "layout,route_a,route_b,pol_a,pol_b,succ_a,succ_b", CERTIFICATES)
def test_pinned_recurrence_certificate(layout, route_a, route_b, pol_a, pol_b, succ_a, succ_b):
    gate = AuthorityGate(gate_id="cert-test")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    a = _replay(adapter, gate, layout, route_a)
    b = _replay(adapter, gate, layout, route_b)

    # Same everything an agent can see...
    assert a["frame"] == b["frame"], "frames must be identical"
    assert a["goal"] == b["goal"]
    assert a["position"] == b["position"]
    assert a["blocked"] == b["blocked"], "action_result must match"
    # ...different hidden phase...
    assert a["polarity"] == pol_a and b["polarity"] == pol_b
    assert a["polarity"] != b["polarity"]
    # ...and the same action goes somewhere else.
    assert a["successors"] == succ_a
    assert b["successors"] == succ_b
    assert all(x != y for x, y in zip(a["successors"], b["successors"]))


def test_certificates_span_different_steps() -> None:
    """These pairs exist only because v2 removed the step from the packet."""
    gate = AuthorityGate(gate_id="cert-step")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    for layout, route_a, route_b, *_ in CERTIFICATES:
        a = _replay(adapter, gate, layout, route_a)
        b = _replay(adapter, gate, layout, route_b)
        assert a["step"] != b["step"], (
            "a pinned certificate landed at equal steps; it no longer exercises the "
            "channel that packet v1 leaked")


def test_previous_action_matches_so_the_packet_truly_aliases() -> None:
    """`previous_action` is public, so a certificate needs it equal on both sides."""
    for layout, route_a, route_b, *_ in CERTIFICATES:
        assert route_a[-1] == route_b[-1], (
            "the two routes end on different actions, so the packets differ in a "
            "public field and this is not an alias")
