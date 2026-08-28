# SHWM Action-Conditioned Hybrid World Model

## Direct verdict

The architecture direction is mathematically coherent as a falsifiable hybrid
research program. The finite results justify action interventions, bounded
latent-capacity language, horizon-aware rollout auditing, and separate verifier
coverage. They do not establish that a learned world model improves Sentinel.

## Surviving construct

**Verifier-Quotiented Interventional Belief Dynamics (VQ-IBD)** maintains a
posterior over latent state/dynamics while merging states only relative to an
explicit finite horizon and trusted verifier-probe traces. Action-conditioned
branch data exposes distinctions that passive observation cannot identify.

This is implemented architecturally as the **Sentinel-Hybrid World Model
(SHWM)**: frozen perceptual grounding, trainable stochastic dynamics, latent
planning, exact observable verification, and structured memory/revision.

## Evidence status

| Claim | Status |
|---|---|
| finite latent code cardinality | `REPRODUCED — mechanically checked in Lean` |
| \(B^H\) action-sequence cardinality | `REPRODUCED — mechanically checked in Lean` |
| finite-horizon probe quotient preserves probe-trace score | `REPRODUCED — mechanically checked in Lean` |
| passive-policy non-identifiability construction | `REPRODUCED — Lean plus exact enumeration` |
| weighted loss/disagreement nonnegativity | `REPRODUCED — mechanically checked in Lean` |
| rollout-error finite-sum recurrence | `REPRODUCED — Lean plus SymPy` |
| exact mismatch rejection plus missing-probe boundary | `REPRODUCED — mechanically checked in Lean` |
| action conditioning fits two-transition collision | `MEASURED` JAX toy only |
| recurrent history fits a separate observation-alias collision | `MEASURED` JAX toy only |
| frozen Scale-0 matrix arithmetic is internally consistent | `MEASURED` configuration diagnostic only |
| local resource arithmetic and toolchain | `MEASURED` diagnostics only |
| SHWM improves planning, transfer, or continual learning | `UNKNOWN` |
| VQ-IBD/SHWM integration is novel | `HYPOTHESIS`, bounded audit only |

## Artifacts

| Artifact | Purpose |
|---|---|
| [Research cycle](shwm-theory-cycle.md) | strict A–I theory report |
| [Prior-art audit](prior-art-audit.md) | bounded primary-source comparison and novelty collisions |
| [Implementation specification](CLAUDE-CODE-IMPLEMENTATION-SPEC.md) | direct coding contract |
| [Verification report](VERIFICATION.md) | executed commands, results, limitations, and failures |
| [Lean file](formal/SHWM.lean) | mechanically checked finite theorems |
| [SymPy checks](shwm_symbolic_checks.py) | recurrence, posterior, capacity, and resource arithmetic |
| [Exact checks](shwm_exact_checks.py) | passive/intervention enumeration and finite bounds |
| [Property checks](shwm_property_checks.py) | 1,200 derandomized helper cases |
| [Resource checks](shwm_resource_checks.py) | JAX/SciPy/CVXPy/NetworkX/NumPyro/Matplotlib/Hydra/MLflow diagnostics |
| [Hydra config](configs/shwm.yaml) | seed, arms, dimensions, sizes, and assumptions |
| [DVC pipeline](dvc.yaml) | reproducible artifact stages |
| `results/` | DVC-managed deterministic JSON summaries; local MLflow database excluded |
| `figures/` | DVC-managed Matplotlib plots |
| `artifact-checksums.sha256` | integrity log |

## Reproduce

Activate or directly use the dedicated Phase-1 math environment documented in
`../../TOOLCHAIN.md`, then from this directory run:

```text
python shwm_symbolic_checks.py
python shwm_exact_checks.py
python shwm_property_checks.py
python shwm_resource_checks.py
dvc repro
```

Compile Lean from the existing Mathlib workspace:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/elan/bin/lake env lean \
  "/absolute/path/to/formal/SHWM.lean"
```

The first real implementation task is Scale 0 only. No Phase-2/SHWM final seed
has been sampled and no scaled capability test has been run; inherited X64H
seed artifacts are not Phase-2 seeds.

No DVC remote is configured. The generated JSON/PNG outputs exist in this local
worktree and cache; a fresh checkout regenerates them with `dvc repro` from the
committed scripts, config, and lock file rather than receiving them as Git
blobs.
