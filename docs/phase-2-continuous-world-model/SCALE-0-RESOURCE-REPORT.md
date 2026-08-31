## Run identity

| field                  | value                                                                   |
|------------------------|-------------------------------------------------------------------------|
| mode                   | dry_run                                                                 |
| is_matrix_run          | False                                                                   |
| commit                 | ae586d1e4f0bd7e51b75fef1eff3bc68e6e17513                                |
| branch                 | phase-2-continuous-world-model                                          |
| tracked tree dirty     | True                                                                    |
| python                 | 3.12.14                                                                 |
| mlx                    | 0.32.2                                                                  |
| platform               | macOS-26.5.1-arm64-arm-64bit                                            |
| freeze manifest digest | sha256:389cf4718ae42e8d0b816e6891b5be8af042d51208efa04a3e48f30645b37856 |

## Backbone preflight

| encoder     | repository                | verdict  | licence    | gated  | revision     | parameters    | blocked by                                                              |
|-------------|---------------------------|----------|------------|--------|--------------|---------------|-------------------------------------------------------------------------|
| qwen3_vl_4b | Qwen/Qwen3-VL-4B-Instruct | runnable | apache-2.0 | False  | ebb281ec70b0 | 4,437,815,808 | -                                                                       |
| gemma3_4b   | google/gemma-3-4b-it      | blocked  | gemma      | manual | 093f9f388b31 | 4,300,079,472 | gated_licence_requires_account_holder_acceptance, no_local_access_token |

`matrix_may_run` = **False**

> gemma3_4b (google/gemma-3-4b-it): blocked -- gated_licence_requires_account_holder_acceptance; no_local_access_token


## Dataset

| slot        | transitions | distinct obs | width | payload MB | index MB | cache hit | tx/s |
|-------------|-------------|--------------|-------|------------|----------|-----------|------|
| gemma3_4b   | 100,000     | 46,443       | 512   | 53.5       | 26.7     | 0.728     | 2085 |
| qwen3_vl_4b | 100,000     | 46,443       | 512   | 53.5       | 26.7     | 0.728     | 2091 |

Shared raw transition digest: `sha256:40366d749bc09b75b4215f9ab04edc151eef52c4c89d4b750a3f7b058e24a3a7`

Split manifest digest: `sha256:316b2f876210f334c08ab6d15f66896444f38c23cc5bce7d638dad6d4d5f0309`

Dataset build wall clock: 97.8 s (ceiling 28,800 s)


### Split audit, per family

| slot        | family            | transitions | branch groups | obs overlap | tuple overlap | disjointness |
|-------------|-------------------|-------------|---------------|-------------|---------------|--------------|
| gemma3_4b   | procedural_visual | 50,000      | 5,192         | 0.0000      | 0.0000        | asserted     |
| gemma3_4b   | synthetic_control | 50,000      | 4,545         | 0.9773      | 0.3634        | measured     |
| qwen3_vl_4b | procedural_visual | 50,000      | 5,192         | 0.0000      | 0.0000        | asserted     |
| qwen3_vl_4b | synthetic_control | 50,000      | 4,545         | 0.9773      | 0.3634        | measured     |

## Parameter accounting

| cell                             | target      | actual trainable | drift    | width | belief | core | frozen |
|----------------------------------|-------------|------------------|----------|-------|--------|------|--------|
| qwen3_vl_4b.continuous.50M.w512  | 50,000,000  | 49,998,794       | -0.0024% | 512   | 1137   | 4551 | 0      |
| qwen3_vl_4b.continuous.200M.w512 | 200,000,000 | 199,996,108      | -0.0019% | 512   | 2338   | 9356 | 0      |
| qwen3_vl_4b.discrete.50M.w512    | 50,000,000  | 50,000,440       | +0.0009% | 512   | 1130   | 4529 | 0      |
| qwen3_vl_4b.discrete.200M.w512   | 200,000,000 | 200,001,648      | +0.0008% | 512   | 2335   | 9343 | 0      |
| qwen3_vl_4b.hybrid.50M.w512      | 50,000,000  | 50,000,696       | +0.0014% | 512   | 1130   | 4529 | 0      |
| qwen3_vl_4b.hybrid.200M.w512     | 200,000,000 | 200,001,904      | +0.0010% | 512   | 2335   | 9343 | 0      |
| gemma3_4b.continuous.50M.w512    | 50,000,000  | 49,998,794       | -0.0024% | 512   | 1137   | 4551 | 0      |
| gemma3_4b.continuous.200M.w512   | 200,000,000 | 199,996,108      | -0.0019% | 512   | 2338   | 9356 | 0      |
| gemma3_4b.discrete.50M.w512      | 50,000,000  | 50,000,440       | +0.0009% | 512   | 1130   | 4529 | 0      |
| gemma3_4b.discrete.200M.w512     | 200,000,000 | 200,001,648      | +0.0008% | 512   | 2335   | 9343 | 0      |
| gemma3_4b.hybrid.50M.w512        | 50,000,000  | 50,000,696       | +0.0014% | 512   | 1130   | 4529 | 0      |
| gemma3_4b.hybrid.200M.w512       | 200,000,000 | 200,001,904      | +0.0010% | 512   | 2335   | 9343 | 0      |
| qwen3_vl_4b.hybrid.50M.w256      | 50,000,000  | 49,998,008       | -0.0040% | 256   | 1159   | 4643 | 0      |
| qwen3_vl_4b.hybrid.50M.w1024     | 50,000,000  | 50,003,302       | +0.0066% | 1024  | 1064   | 4258 | 0      |
| gemma3_4b.hybrid.50M.w256        | 50,000,000  | 49,998,008       | -0.0040% | 256   | 1159   | 4643 | 0      |
| gemma3_4b.hybrid.50M.w1024       | 50,000,000  | 50,003,302       | +0.0066% | 1024  | 1064   | 4258 | 0      |

## Throughput and memory, per cell

| cell                             | seeds | wall s | cold load s | device peak GiB | upd/s | positions/s | measured/estimated |
|----------------------------------|-------|--------|-------------|-----------------|-------|-------------|--------------------|
| qwen3_vl_4b.continuous.50M.w512  | 3     | 16.0   | 0.01        | 1.64            | 12.84 | 13,149      | 2.80               |
| qwen3_vl_4b.continuous.200M.w512 | 3     | 50.2   | 0.03        | 3.82            | 4.01  | 4,107       | 1.67               |
| qwen3_vl_4b.discrete.50M.w512    | 3     | 15.9   | 0.01        | 1.76            | 12.90 | 13,207      | 3.02               |
| qwen3_vl_4b.discrete.200M.w512   | 3     | 52.7   | 0.03        | 3.94            | 3.82  | 3,908       | 1.72               |
| qwen3_vl_4b.hybrid.50M.w512      | 3     | 15.9   | 0.01        | 1.64            | 12.83 | 13,133      | 2.81               |
| qwen3_vl_4b.hybrid.200M.w512     | 3     | 52.4   | 0.03        | 3.92            | 3.84  | 3,933       | 1.71               |
| gemma3_4b.continuous.50M.w512    | 3     | 16.1   | 0.01        | 1.64            | 12.72 | 13,030      | 2.80               |
| gemma3_4b.continuous.200M.w512   | 3     | 50.0   | 0.03        | 3.82            | 4.03  | 4,122       | 1.67               |
| gemma3_4b.discrete.50M.w512      | 3     | 15.9   | 0.01        | 1.76            | 12.88 | 13,192      | 3.02               |
| gemma3_4b.discrete.200M.w512     | 3     | 52.5   | 0.03        | 3.94            | 3.84  | 3,927       | 1.72               |
| gemma3_4b.hybrid.50M.w512        | 3     | 16.0   | 0.01        | 1.64            | 12.79 | 13,094      | 2.81               |
| gemma3_4b.hybrid.200M.w512       | 3     | 53.1   | 0.03        | 3.92            | 3.79  | 3,880       | 1.71               |
| qwen3_vl_4b.hybrid.50M.w256      | 3     | 16.4   | 0.01        | 1.64            | 12.47 | 12,766      | 2.81               |
| qwen3_vl_4b.hybrid.50M.w1024     | 3     | 15.3   | 0.01        | 1.63            | 13.37 | 13,690      | 2.79               |
| gemma3_4b.hybrid.50M.w256        | 3     | 16.4   | 0.01        | 1.64            | 12.45 | 12,748      | 2.81               |
| gemma3_4b.hybrid.50M.w1024       | 3     | 15.3   | 0.01        | 1.63            | 13.37 | 13,690      | 2.79               |

## Planner and verifier, per workload

| quantity                        | value   |
|---------------------------------|---------|
| planner invocations             | 300     |
| candidate sequences             | 19,200  |
| model calls                     | 338,800 |
| distinct plans                  | 298     |
| rollouts per second             | 967,958 |
| planner wall seconds            | 0.35    |
| verifier verifications          | 512     |
| planted mismatches              | 256     |
| detection rate                  | 1.000   |
| mean probe coverage             | 1.000   |
| authorised actions              | 512     |
| denied actions                  | 0       |
| online environment interactions | 0       |

Process resident high-water across the whole sequence: **12.15 GiB** of the 112 GiB per-process ceiling. That figure is cumulative by construction (`ru_maxrss` never falls), so it is the number the ceiling is checked against and *not* any single workload's cost; the device peak column above is the per-workload figure.


## Gate

| check                      | result                       |
|----------------------------|------------------------------|
| workloads completed        | 48/48                        |
| matching failures          | 0                            |
| resource envelope failures | 0                            |
| undeclared process state   | 0                            |
| failures                   | 0                            |
| total wall clock           | 26.2 min (ceiling 4,320 min) |
| artifact storage           | 0.62 GiB (ceiling 200 GiB)   |
| Scale-0 gate passed        | False                        |

> PASS requires mode=matrix with both frozen backbones runnable. A dry run measures the pipeline and can never pass the Scale-0 gate.


## Restart equivalence

| field                | value                                 |
|----------------------|---------------------------------------|
| workload             | qwen3_vl_4b.continuous.50M.w512.s6600 |
| checkpoint at update | 100                                   |
| weights match        | True                                  |
| loss history match   | True                                  |
