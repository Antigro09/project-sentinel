## Run identity

| field                  | value                                                                   |
|------------------------|-------------------------------------------------------------------------|
| mode                   | matrix                                                                  |
| is_matrix_run          | True                                                                    |
| commit                 | c64cd651c76656881a02e990227b6ba1e88cef59                                |
| branch                 | phase-2-continuous-world-model                                          |
| tracked tree dirty     | True                                                                    |
| python                 | 3.12.14                                                                 |
| mlx                    | 0.32.2                                                                  |
| platform               | macOS-26.5.1-arm64-arm-64bit                                            |
| freeze manifest digest | sha256:c01f6f1c7d23787b3cff3fe217642fd8ce6a3349d5e9ac16fe428a7e7741df45 |

## Backbone preflight

| encoder     | repository                | verdict  | licence    | gated  | revision     | parameters    | blocked by |
|-------------|---------------------------|----------|------------|--------|--------------|---------------|------------|
| qwen3_vl_4b | Qwen/Qwen3-VL-4B-Instruct | runnable | apache-2.0 | False  | ebb281ec70b0 | 4,437,815,808 | -          |
| gemma3_4b   | google/gemma-3-4b-it      | runnable | gemma      | manual | 093f9f388b31 | 4,300,079,472 | -          |

`matrix_may_run` = **True**

> both frozen encoder families are runnable


## Dataset

| slot        | transitions | distinct obs | width | payload MB | index MB | cache hit | tx/s |
|-------------|-------------|--------------|-------|------------|----------|-----------|------|
| gemma3_4b   | 100,000     | 46,443       | 2560  | 243.7      | 26.8     | 1.000     | 2673 |
| qwen3_vl_4b | 100,000     | 46,443       | 2560  | 243.7      | 26.8     | 1.000     | 2729 |

Shared raw transition digest: `sha256:40366d749bc09b75b4215f9ab04edc151eef52c4c89d4b750a3f7b058e24a3a7`

Split manifest digest: `sha256:316b2f876210f334c08ab6d15f66896444f38c23cc5bce7d638dad6d4d5f0309`

Dataset build wall clock: 82.0 s (ceiling 28,800 s)


### Split audit, per family

| slot        | family            | transitions | branch groups | obs overlap | tuple overlap | disjointness |
|-------------|-------------------|-------------|---------------|-------------|---------------|--------------|
| gemma3_4b   | procedural_visual | 50,000      | 5,192         | 0.0000      | 0.0000        | asserted     |
| gemma3_4b   | synthetic_control | 50,000      | 4,545         | 0.9773      | 0.3634        | measured     |
| qwen3_vl_4b | procedural_visual | 50,000      | 5,192         | 0.0000      | 0.0000        | asserted     |
| qwen3_vl_4b | synthetic_control | 50,000      | 4,545         | 0.9773      | 0.3634        | measured     |

## Parameter accounting

| cell                             | target      | actual trainable | drift    | width | belief | core | frozen        |
|----------------------------------|-------------|------------------|----------|-------|--------|------|---------------|
| qwen3_vl_4b.continuous.50M.w512  | 50,000,000  | 50,003,698       | +0.0074% | 512   | 1124   | 4502 | 4,437,815,808 |
| qwen3_vl_4b.continuous.200M.w512 | 200,000,000 | 200,006,842      | +0.0034% | 512   | 2332   | 9330 | 4,437,815,808 |
| qwen3_vl_4b.discrete.50M.w512    | 50,000,000  | 50,001,712       | +0.0034% | 512   | 1118   | 4474 | 4,437,815,808 |
| qwen3_vl_4b.discrete.200M.w512   | 200,000,000 | 199,994,974      | -0.0025% | 512   | 2329   | 9316 | 4,437,815,808 |
| qwen3_vl_4b.hybrid.50M.w512      | 50,000,000  | 50,001,968       | +0.0039% | 512   | 1118   | 4474 | 4,437,815,808 |
| qwen3_vl_4b.hybrid.200M.w512     | 200,000,000 | 199,995,230      | -0.0024% | 512   | 2329   | 9316 | 4,437,815,808 |
| gemma3_4b.continuous.50M.w512    | 50,000,000  | 50,003,698       | +0.0074% | 512   | 1124   | 4502 | 4,300,079,472 |
| gemma3_4b.continuous.200M.w512   | 200,000,000 | 200,006,842      | +0.0034% | 512   | 2332   | 9330 | 4,300,079,472 |
| gemma3_4b.discrete.50M.w512      | 50,000,000  | 50,001,712       | +0.0034% | 512   | 1118   | 4474 | 4,300,079,472 |
| gemma3_4b.discrete.200M.w512     | 200,000,000 | 199,994,974      | -0.0025% | 512   | 2329   | 9316 | 4,300,079,472 |
| gemma3_4b.hybrid.50M.w512        | 50,000,000  | 50,001,968       | +0.0039% | 512   | 1118   | 4474 | 4,300,079,472 |
| gemma3_4b.hybrid.200M.w512       | 200,000,000 | 199,995,230      | -0.0024% | 512   | 2329   | 9316 | 4,300,079,472 |
| qwen3_vl_4b.hybrid.50M.w256      | 50,000,000  | 49,996,678       | -0.0066% | 256   | 1153   | 4616 | 4,437,815,808 |
| qwen3_vl_4b.hybrid.50M.w1024     | 50,000,000  | 50,000,414       | +0.0008% | 1024  | 1037   | 4155 | 4,437,815,808 |
| gemma3_4b.hybrid.50M.w256        | 50,000,000  | 49,996,678       | -0.0066% | 256   | 1153   | 4616 | 4,300,079,472 |
| gemma3_4b.hybrid.50M.w1024       | 50,000,000  | 50,000,414       | +0.0008% | 1024  | 1037   | 4155 | 4,300,079,472 |

## Throughput and memory, per cell

| cell                             | seeds | wall s | cold load s | device peak GiB | upd/s | positions/s | measured/estimated |
|----------------------------------|-------|--------|-------------|-----------------|-------|-------------|--------------------|
| qwen3_vl_4b.continuous.50M.w512  | 3     | 15.2   | 0.01        | 1.63            | 13.51 | 13,839      | 2.79               |
| qwen3_vl_4b.continuous.200M.w512 | 3     | 49.7   | 0.03        | 3.83            | 4.06  | 4,155       | 1.67               |
| qwen3_vl_4b.discrete.50M.w512    | 3     | 16.0   | 0.01        | 1.75            | 12.80 | 13,104      | 3.00               |
| qwen3_vl_4b.discrete.200M.w512   | 3     | 51.8   | 0.03        | 3.91            | 3.88  | 3,977       | 1.71               |
| qwen3_vl_4b.hybrid.50M.w512      | 3     | 15.9   | 0.01        | 1.66            | 12.83 | 13,134      | 2.84               |
| qwen3_vl_4b.hybrid.200M.w512     | 3     | 51.9   | 0.03        | 3.91            | 3.88  | 3,976       | 1.71               |
| gemma3_4b.continuous.50M.w512    | 3     | 16.0   | 0.01        | 1.63            | 12.78 | 13,085      | 2.79               |
| gemma3_4b.continuous.200M.w512   | 3     | 52.1   | 0.02        | 3.83            | 3.87  | 3,959       | 1.67               |
| gemma3_4b.discrete.50M.w512      | 3     | 16.3   | 0.01        | 1.75            | 12.51 | 12,809      | 3.00               |
| gemma3_4b.discrete.200M.w512     | 3     | 52.8   | 0.02        | 3.91            | 3.82  | 3,908       | 1.71               |
| gemma3_4b.hybrid.50M.w512        | 3     | 16.2   | 0.01        | 1.66            | 12.62 | 12,923      | 2.84               |
| gemma3_4b.hybrid.200M.w512       | 3     | 52.5   | 0.03        | 3.91            | 3.84  | 3,927       | 1.71               |
| qwen3_vl_4b.hybrid.50M.w256      | 3     | 16.3   | 0.01        | 1.64            | 12.53 | 12,826      | 2.81               |
| qwen3_vl_4b.hybrid.50M.w1024     | 3     | 15.5   | 0.01        | 1.63            | 13.21 | 13,530      | 2.80               |
| gemma3_4b.hybrid.50M.w256        | 3     | 16.2   | 0.01        | 1.64            | 12.60 | 12,904      | 2.81               |
| gemma3_4b.hybrid.50M.w1024       | 3     | 15.3   | 0.01        | 1.63            | 13.39 | 13,707      | 2.80               |

## Planner and verifier, per workload

| quantity                        | value   |
|---------------------------------|---------|
| planner invocations             | 300     |
| candidate sequences             | 19,200  |
| model calls                     | 338,800 |
| distinct plans                  | 298     |
| rollouts per second             | 959,524 |
| planner wall seconds            | 0.35    |
| verifier verifications          | 512     |
| planted mismatches              | 256     |
| detection rate                  | 1.000   |
| mean probe coverage             | 1.000   |
| authorised actions              | 512     |
| denied actions                  | 0       |
| online environment interactions | 0       |

Process resident high-water across the whole sequence: **13.01 GiB** of the 112 GiB per-process ceiling. That figure is cumulative by construction (`ru_maxrss` never falls), so it is the number the ceiling is checked against and *not* any single workload's cost; the device peak column above is the per-workload figure.


## Gate

Each clause the run matrix states, evaluated by name.

| clause                                  | result |
|-----------------------------------------|--------|
| all_48_workloads_complete               | PASS   |
| all_three_seeds_retained_for_every_cell | PASS   |
| both_frozen_encoders_runnable           | PASS   |
| hard_resource_ceilings_hold             | PASS   |
| is_a_matrix_run                         | PASS   |
| matching_rules_hold                     | PASS   |
| no_leakage_detected                     | PASS   |
| no_undeclared_process_state             | PASS   |
| restart_and_artifact_checks_pass        | PASS   |
| tracked_tree_clean_for_run_inputs       | PASS   |

Clauses the driver cannot settle, and how they are settled:

| clause                        | how                                                                                                                                                     |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| exact_full_suite_green        | not checked by this driver; run `uv run pytest -q` at the reported commit and record the result alongside this artefact                                 |
| no_phase_2_final_seed_sampled | enforced by sentinel.wm.provenance.FinalSeedGuard, which refuses to load a final seed without a committed post-freeze manifest; no such manifest exists |

| check                      | result                       |
|----------------------------|------------------------------|
| workloads completed        | 48/48                        |
| matching failures          | 0                            |
| resource envelope failures | 0                            |
| undeclared process state   | 0                            |
| failures                   | 0                            |
| total wall clock           | 25.9 min (ceiling 4,320 min) |
| artifact storage           | 0.97 GiB (ceiling 200 GiB)   |
| Scale-0 gate passed        | True                         |

> PASS requires mode=matrix with both frozen backbones runnable. A dry run measures the pipeline and can never pass the Scale-0 gate.


## Restart equivalence

| field                | value                                 |
|----------------------|---------------------------------------|
| workload             | qwen3_vl_4b.continuous.50M.w512.s6600 |
| checkpoint at update | 100                                   |
| weights match        | True                                  |
| loss history match   | True                                  |
