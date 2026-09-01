"""H. Freeze everything Stage 1A-1 will be measured against.

A freeze is only worth the things it can detect changing. So this hashes the
source of every component rather than a description of it: the environment, the
schema, the splits, the interfaces, the probe configuration, the controls, and
the gate thresholds. If any of those files moves, the manifest digest moves, and
a later result cannot silently claim to have been produced under this contract.

The certificates are recomputed at freeze time rather than copied, because a
certificate that was true of an earlier environment and is stale now is worse
than none.

    .venv-shwm/bin/python experiments/shwm/scale1_freeze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    GOAL_PHRASES,
    SWITCH_COUNT,
    build_hidden_state_certificate,
    build_language_certificate,
)
from sentinel.wm.interfaces import build_interfaces, interface_report  # noqa: E402
from sentinel.wm.packet import (  # noqa: E402
    MAX_GOAL_TOKENS,
    SLOT_COUNT,
    SLOT_WIDTH,
    build_vocabulary,
)
from sentinel.wm.provenance import environment_state, git_state  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED, Stratum  # noqa: E402
from sentinel.wm.versioning import digest_file, digest_of  # noqa: E402

FROZEN_SOURCES = (
    "src/sentinel/env/adapters/procedural_visual_v2.py",
    "src/sentinel/env/adapters/base.py",
    "src/sentinel/wm/packet.py",
    "src/sentinel/wm/interfaces.py",
    "src/sentinel/wm/splits_v2.py",
    "src/sentinel/wm/hidden_state_audit.py",
    "src/sentinel/wm/latent_contract.py",
    "src/sentinel/wm/versioning.py",
    "experiments/shwm/feature_qualification.py",
    "experiments/shwm/feature_sufficiency.py",
)


def probe_configuration() -> dict[str, Any]:
    import feature_qualification as qualification
    from feature_sufficiency import PENALTIES, RFF_BANDWIDTHS as SUFFICIENCY_BANDWIDTHS

    return {
        "family": "random-Fourier ridge, closed form",
        "rff_width": qualification.RFF_WIDTH,
        "rff_bandwidths": list(qualification.RFF_BANDWIDTHS),
        "sufficiency_bandwidths": list(SUFFICIENCY_BANDWIDTHS),
        "ridge_penalties": list(PENALTIES),
        "readout_width": qualification.READOUT_WIDTH,
        "readout": "one fixed random projection per interface, identical procedure",
        "history_steps": "full episode, so the window reaches the reset frame",
        "selection": "penalty and bandwidth chosen on a level-disjoint validation slice",
    }


def control_definitions() -> dict[str, Any]:
    return {
        "positive_control": {
            "name": "oracle_structured_state",
            "role": "evaluator-only upper bound; must win or the probe cannot read the variable",
            "admissible_as_model_input": False,
        },
        "negative_control": {
            "name": "shuffled labels",
            "role": "must sit at baseline; a positive margin means the protocol leaks",
        },
        "representation_floors": [
            "raw_lowres_spatial",
            "fixed_random_spatial_projection",
        ],
        "ablation": "mean pooling, which is the spatial interface with position deleted",
        "note": (
            "A fixed random projection is a frozen matrix drawn once and is not a learned "
            "representation. In the S1.2 investigation it matched or beat both 4B backbones "
            "on the pooled interface, which is why it is a first-class arm rather than a "
            "footnote."
        ),
    }


def gate_definitions() -> dict[str, Any]:
    return {
        "S1.2_v2": {
            "oracle_calibrated": "oracle mean intervention margin > 0.2 on dynamics_clean",
            "intervention": "a non-oracle interface with mean intervention margin > 0.05",
            "hidden_phase": "a non-oracle interface with mean hidden-phase margin > 0.05",
            "primary_stratum": Stratum.DYNAMICS_CLEAN.value,
            "appearance_shift_may_veto": False,
        },
        "construct_validity": {
            "hidden_state_exercised": "the hidden variable changes in at least 50% of episodes",
            "language_required": "some reachable state has the instruction change the best action",
            "reachability": "invariance is only argued over states a trajectory can produce",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/freeze-manifest.json")
    parser.add_argument(
        "--qualification",
        type=Path,
        default=REPO / "artifacts/shwm/scale1/feature-qualification.json",
    )
    arguments = parser.parse_args()

    sources = {path: digest_file(REPO / path) for path in FROZEN_SOURCES}
    hidden = build_hidden_state_certificate(9000, max_depth=8)
    language = build_language_certificate(9000, max_depth=6)
    interfaces = build_interfaces()

    environment = {
        "name": "procedural_visual_v2",
        "switch_count": SWITCH_COUNT,
        "hidden_variable": "polarity",
        "hidden_update_rule": "flips on entering a switch cell",
        "hidden_revealed": "reset frame only",
        "clock_exposed": False,
        "goal_selection": "language",
        "goal_phrases": dict(GOAL_PHRASES),
        "source_digest": sources["src/sentinel/env/adapters/procedural_visual_v2.py"],
    }
    schema = {
        "slot_count": SLOT_COUNT,
        "slot_width": SLOT_WIDTH,
        "max_goal_tokens": MAX_GOAL_TOKENS,
        "vocabulary": build_vocabulary(GOAL_PHRASES.values()),
        "modalities": ["image", "structured", "goal", "text", "audio(declared, absent)"],
        "source_digest": sources["src/sentinel/wm/packet.py"],
    }
    splits = {
        "strata": [s.value for s in Stratum],
        "primary": Stratum.DYNAMICS_CLEAN.value,
        "canonical_appearance_seed": CANONICAL_APPEARANCE_SEED,
        "lineage": "root, clone point and depth; descendants inherit the clone's lineage",
        "source_digest": sources["src/sentinel/wm/splits_v2.py"],
    }
    branches = {
        "design": "clone the exact state, execute every legal action, restore",
        "actions_per_state": 4,
        "targets": ["successor_0", "successor_1", "successor_2", "successor_3"],
        "hidden_state_use": "evaluator-only scoring; never a training feature",
    }

    qualification_digest = (
        digest_file(arguments.qualification) if arguments.qualification.exists() else None
    )

    manifest: dict[str, Any] = {
        "phase": "SHWM-SCALE-1A-0",
        "git": git_state(REPO),
        "environment_probe": environment_state(),
        "frozen_sources": sources,
        "v2_environment": environment,
        "multimodal_schema": schema,
        "splits": splits,
        "intervention_branches": branches,
        "interfaces": interface_report(interfaces),
        "token_resampling": {
            "qwen3_vl_4b": "64 visual tokens on an 8x8 grid, 2x2 mean-pooled to 4x4 slots",
            "gemma3_4b": "256 visual tokens on a 16x16 grid, 4x4 mean-pooled to 4x4 slots",
            "visual_span_selection": "the candidate image token id that actually repeats",
        },
        "probe_configuration": probe_configuration(),
        "controls": control_definitions(),
        "gates": gate_definitions(),
        "certificates": {
            "hidden_state": hidden.canonical_dict(),
            "language": language,
        },
        "qualification_artifact_digest": qualification_digest,
        "final_seed_file": None,
        "created_before_final_seed": True,
    }
    manifest["manifest_digest"] = digest_of(
        {k: v for k, v in manifest.items() if k not in ("git", "environment_probe")}
    )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    print(f"frozen sources        : {len(sources)}")
    for path, digest in sources.items():
        print(f"  {digest[7:19]}  {path}")
    print(f"hidden-state cert     : histories {list(hidden.history_a)} / {list(hidden.history_b)}, "
          f"polarity {hidden.polarity_a}/{hidden.polarity_b}")
    print(f"language cert         : best action {language['best_action_alpha']} vs "
          f"{language['best_action_beta']} with pixels fixed")
    print(f"interfaces            : {len(interfaces)} at "
          f"({SLOT_COUNT}, {SLOT_WIDTH}), "
          f"{manifest['interfaces']['total_trainable_adapter_parameters']:,} trainable adapter params")
    print(f"qualification artifact: {qualification_digest[:24] if qualification_digest else 'ABSENT'}")
    print(f"manifest digest       : {manifest['manifest_digest']}")
    print(f"written               : {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
