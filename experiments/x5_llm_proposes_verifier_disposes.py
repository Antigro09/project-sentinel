"""X5: does verification turn an unreliable proposer into a reliable system?

The pairing thesis. A pretrained model brings language, code and tool use;
this project brings a verifier that can prove a hypothesis wrong. Today's
LLM agents hallucinate how a tool behaves and have no way to check -- which
is exactly the hole the verifier fills.

The experiment: gpt-oss:120b proposes world models as programs, the
verifier scores them against real episodes, `first_divergence` localises
the error, and the model is asked to repair rather than resample.

Conditions:
  llm-only        first proposal, no verification        <- the baseline
  llm + verify    best of K proposals by fitness
  llm + repair    K proposals, each repaired R times using the divergence

The claim is that repair beats resampling at equal budget. If it does not,
verification is worth less than it looks and the pairing is weaker than
argued.

Staged: `bootstrap/teacher.py` already has the ollama plumbing and the
propose/verify/repair loop from Phase 2 -- reuse it rather than rewriting.
"""

raise SystemExit("staged: reuse bootstrap/teacher.py plumbing")
