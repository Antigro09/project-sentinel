# SHWM Action-Conditioned Hybrid World Model — Bounded Prior-Art and Claim Audit

**Audit date:** 2026-08-28
**Status:** bounded and non-exhaustive primary-source audit
**Object audited:** the proposed Sentinel-Hybrid World Model (SHWM), especially its scale assumptions, latent-state assumptions, benchmark placement, causal language, and possible integration-level novelty
**Empirical status of SHWM:** **UNKNOWN**. No SHWM training run, benchmark result, ablation, or replication was inspected or executed for this audit.

## Evidence-label policy

- **MEASURED:** reported by an executed experiment in a cited primary source. Unless explicitly stated otherwise, these measurements are source-reported and were not independently reproduced here.
- **REPRODUCED:** independently repeated or pinned by tests. No external result in this audit receives this label.
- **INFERRED:** a reasoned conclusion from cited evidence.
- **HYPOTHESIS:** a falsifiable proposal not yet established by SHWM experiments.
- **SPECULATIVE:** a longer-range architectural possibility.
- **UNKNOWN:** evidence is insufficient.
- **RETRACTED:** contradicted or invalidated. No SHWM empirical result is retracted because none was supplied.

## Direct verdict

The high-level direction—learn action-conditioned latent dynamics while retaining exact symbolic checks, provenance, revision, and abstention—is mathematically coherent as a research program. It is not yet an empirical result, an AGI architecture theorem, or a categorical novelty claim.

The strongest defensible conclusion is narrow:

> **HYPOTHESIS / bounded novelty delta.** In this bounded audit, no located primary source reports one measured system combining (i) Sentinel-style exact executable verification with explicit provenance and dependency-aware revision, (ii) learned action-conditioned hybrid latent dynamics used for planning, and (iii) a preregistered, frozen causal ablation protocol that intervenes on action channels, latent components, verifier pathways, and retained memory. This is a provisional systems-integration gap, not proof of novelty. It becomes a contribution only if the integration produces a measured effect that survives the frozen ablations and held-out evaluations.

No individual ingredient is new: latent world models, action conditioning, frozen encoders, continuous or discrete latent variables, recurrent hybrid states, ensembles, latent-space planning, hierarchical planning, multimodal representation learning, audio-visual prediction, external memory, provenance records, belief revision, and machine-checkable verification all have substantial prior art.

## Strongest objections first

1. **No parameter-count necessity has been established.** The proposed 15B–70B range can be an engineering budget, but this audit found no accepted theorem making it a lower bound for AGI or for SHWM. “AGI” and success criteria would first need formal definitions, and a parameter count is architecture-, precision-, and task-dependent.
2. **No 1,024-dimensional continuous-latent necessity has been established.** The literature contains successful continuous, discrete, and hybrid representations. Dimension by itself is not an invariant measure of usable information without assumptions about precision, topology, noise, distortion, observability, and the task family.
3. **Predictive dynamics are not automatically causal dynamics.** Learning \(p(o_{t+1}\mid h_t,a_t)\) from logged trajectories does not in general identify \(p(o_{t+1}\mid h_t,\operatorname{do}(a_t))\). Hidden confounding, policy-induced correlations, insufficient state, and support gaps can preserve prediction accuracy while breaking interventions or counterfactuals.
4. **The closest world-model components are already strong collisions.** PlaNet, Dreamer/RSSM, MuZero, PETS, V-JEPA 2-AC, Director, and recent hierarchical latent planners already cover much of action-conditioned latent prediction, uncertainty, planning, and hierarchy.
5. **MLX support is not a feasibility proof.** Official MLX documentation establishes quantization and LoRA/QLoRA mechanisms, not that a particular Qwen3-VL or Gemma 3 multimodal stack fits, trains stably, preserves accuracy, or meets latency targets on the intended machine.
6. **OSWorld and SWE-bench are poor first development environments for the core dynamics claim.** They confound world modeling with GUI grounding, software tooling, language understanding, and evaluator integration. They are useful later external gates after a controlled causal-dynamics result exists.
7. **The proposed integration delta is unmeasured.** Without component ablations, the learned world model might add no value beyond Sentinel’s verifier/memory, or the verifier might merely reject errors without improving planning. Both possibilities must remain live falsifiers.

## Claim-by-claim audit

| # | Exact claim | Verdict | Evidence-honest status |
|---|---|---|---|
| 1 | No accepted theorem was found establishing a 15B–70B AGI parameter lower bound. | **Supported only as a bounded search result.** No such theorem was located; this does not prove that none exists. | **INFERRED** from a bounded audit; global absence remains **UNKNOWN**. |
| 2 | Chinchilla gives an empirical compute-optimal scaling relation, not an AGI threshold. | **Supported.** Hoffmann et al. fit and test an empirical loss/compute relation for autoregressive language models. | **MEASURED** by the paper; AGI extrapolation rejected as **unsupported**. |
| 3 | No source establishes that open-world systems require a continuous 1,024-dimensional latent. | **Supported only as a bounded search result.** Located work treats 1,024 as an architecture choice or experimental scale, not a necessity theorem. | **INFERRED** from a bounded audit; universal absence remains **UNKNOWN**. |
| 4 | Current official Qwen3-VL and Gemma 3 sizes should be verified. | **Verified as a dated snapshot.** The Qwen3-VL report lists dense 2B/4B/8B/32B and MoE 30B-A3B/235B-A22B variants; the Gemma 3 report covers 1B/4B/12B/27B and current official documentation also lists a later 270M core variant. | **MEASURED** from official reports/model documentation on 2026-08-28. |
| 5 | V-JEPA 2 and DreamerV3 must not be inflated into AGI or causality results. | **Supported.** Both demonstrate important but bounded prediction/control results; neither identifies general causal structure or establishes AGI. | Source results **MEASURED**; stronger claims **unsupported**. |
| 6 | Benchmarks must be assigned to development versus later gates. | **Supported as a research-design conclusion.** Procgen/Crafter isolate useful dynamics properties; OSWorld/SWE-bench add major unrelated confounds. | **INFERRED**. |
| 7 | Individual SHWM components are not new. | **Supported.** Direct prior-art collisions are documented below. | **INFERRED** from primary sources. |
| 8 | A narrow novelty delta may remain in the exact Sentinel + learned dynamics + frozen causal-ablation integration. | **Provisionally defensible, not established.** No directly matching system was located in this bounded audit. | **HYPOTHESIS**; categorical novelty is **UNKNOWN**. |
| 9 | Search date, attempts, URLs, and inaccessible sources must be recorded. | **Completed below.** | **MEASURED** audit metadata. |

## 1. Parameter scale: what Chinchilla does and does not say

### 1.1 Chinchilla is an empirical compute-allocation result

**Primary source:** Jordan Hoffmann et al., “Training Compute-Optimal Large Language Models” (2022), arXiv:2203.15556, <https://arxiv.org/abs/2203.15556>.

The paper studies transformer language-model loss under a fixed training-compute budget. It reports experiments over more than 400 models from 70M to over 16B parameters and 5B–500B training tokens, then trains the predicted compute-optimal 70B-parameter Chinchilla model using the same compute budget as the 280B-parameter Gopher model. Its central relation is empirical: under the studied regime, model size and training tokens should grow at approximately equal rates for compute-optimal training.

What it establishes:

- **MEASURED:** a fitted and experimentally checked compute-optimal scaling relationship for the studied autoregressive language-model family and data regime;
- **MEASURED:** a 70B model trained on substantially more data outperformed several larger undertrained models on the reported evaluations;
- **INFERRED:** parameter count alone is a poor proxy for the allocation of a fixed compute budget.

What it does not establish:

- no theorem that 70B parameters are necessary or sufficient for AGI;
- no lower bound for multimodal world modeling, causal reasoning, autonomous continual learning, or Sentinel’s acceptance criteria;
- no result that a 15B model is too small, or that a 70B model is enough, for SHWM;
- no local-hardware feasibility result.

The phrase “Chinchilla-optimal” therefore must mean an empirical compute/data allocation heuristic under stated assumptions, not an AGI threshold.

### 1.2 Why a bare AGI parameter lower bound is under-specified

A meaningful lower-bound theorem would have to fix at least:

\[
(\mathcal T,\;\mathcal A,\;\mathcal P,\;\epsilon,\;\delta,\;B,\;\Pi),
\]

where \(\mathcal T\) is a formal task family, \(\mathcal A\) the permitted architecture class, \(\mathcal P\) the data/environment distribution, \(\epsilon\) a performance tolerance, \(\delta\) a failure probability, \(B\) a compute or sample budget, and \(\Pi\) a numerical-precision model. “AGI” without such a target is not a theorem-ready property.

**Derived observation, unchecked:** parameter count is also not functionally canonical. One may add unused parameters or introduce function-preserving reparameterizations without changing behavior. This does not rule out lower bounds for a carefully constrained representation class, but it does rule out reading raw parameter count as an architecture-independent measure of intelligence.

**Bounded-search conclusion:** no accepted primary-source theorem establishing a 15B–70B AGI lower bound was located. The correct global status is **UNKNOWN**, not “proved nonexistent.”

## 2. Latent topology and the unsupported 1,024-dimensional requirement

### 2.1 Directly relevant representation families

| Family | Primary foundation | Mechanism | What it implies for SHWM |
|---|---|---|---|
| Continuous stochastic/deterministic | Danijar Hafner et al., “Learning Latent Dynamics for Planning from Pixels” (PlaNet, ICML 2019), <https://proceedings.mlr.press/v97/hafner19a.html> | Image encoder with deterministic and stochastic latent transition components; online latent-space planning. | Continuous latents are viable, but no fixed dimension is established as necessary. |
| Discrete | Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, “Neural Discrete Representation Learning” (VQ-VAE, NeurIPS 2017), <https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html> | Vector-quantized categorical codes with learned embeddings. | Discrete representations are a serious alternative, not a toy exception. |
| Discrete world model | Danijar Hafner et al., “Mastering Atari with Discrete World Models” (DreamerV2, 2020/ICLR 2021), <https://arxiv.org/abs/2010.02193> | Discrete stochastic world state; actor and critic learn from latent imagination. | Discrete latent control can perform strongly across many visual tasks. |
| Recurrent hybrid state | Danijar Hafner et al., “Mastering Diverse Control Tasks through World Models” (DreamerV3, Nature 2025), <https://doi.org/10.1038/s41586-025-08744-2> | RSSM combines continuous deterministic recurrent state \(h_t\) with discrete stochastic state \(z_t\), conditioned on actions. | A hybrid state is already established prior art; SHWM must specify a narrower contribution. |
| Joint-embedding prediction | Mido Assran et al., “V-JEPA 2” (2025), <https://arxiv.org/abs/2506.09985> | Predicts future embeddings; V-JEPA 2-AC adds action-conditioned latent planning after robot-video post-training. | Reconstruction-free latent prediction and action-conditioned planning are not new. |
| Identifiable JEPA under assumptions | David Klindt, Yann LeCun, Randall Balestriero, “When Does LeJEPA Learn a World Model?” (2026 preprint), <https://arxiv.org/abs/2605.26379> | Proves linear recovery under stationary additive-noise transitions and Gaussian regularization, with additional planning assumptions; experiments range from 2D to 1,024D. | The theorem concerns identifiability under restrictive assumptions. Testing 1,024D does not prove 1,024 dimensions are required. |

### 2.2 Why dimension alone cannot carry the requirement

Let \(s_t\) denote a task-relevant state and \(e(s_t)\in\mathbb R^d\) a learned encoding. A dimension lower bound is meaningful only after specifying, for example:

- finite precision or noise per coordinate;
- continuity/Lipschitz or robustness constraints on \(e\) and its decoder;
- a distortion criterion and acceptable planning regret;
- the intrinsic state family and observation process;
- whether history may be stored outside the latent;
- whether exact symbolic variables, episodic memory, or retrieved context count as part of the state.

Without such constraints, dimension and information capacity can be traded against coordinate precision and external state. At finite \(b\)-bit precision, a rough storage ceiling scales with \(db\); at ideal real precision, that interpretation fails. A “1,024 continuous dimensions” rule is therefore an engineering choice until attached to a formal rate-distortion, sufficient-statistic, observability, or control-regret result.

**Verdict:** this audit found no source establishing that open-world systems require a continuous \(1{,}024\)-dimensional latent. Continuous, discrete, and hybrid designs all remain live candidates. The audit does not select a winner. SHWM should preregister a matched-capacity comparison rather than encode the answer in advance.

## 3. Current official Qwen3-VL and Gemma 3 size ranges

These are dated model-catalog facts, not capability thresholds.

### 3.1 Qwen3-VL snapshot

**Official technical report:** Qwen Team, “Qwen3-VL Technical Report” (2025), arXiv:2511.21631, <https://arxiv.org/abs/2511.21631>.
**Official repository:** <https://github.com/QwenLM/Qwen3-VL>.
**Official Qwen collection:** <https://huggingface.co/collections/Qwen/qwen3-vl>.

The official named lineup located on 2026-08-28 includes:

- dense 2B, 4B, 8B, and 32B variants;
- MoE 30B-A3B and 235B-A22B variants;
- Instruct and Thinking editions, with multiple quantized distributions.

Representative official model cards:

- <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>
- <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct>
- <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct>
- <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct>

The report lists dense **2B, 4B, 8B, and 32B** models and MoE
**30B-A3B and 235B-A22B** models. The A3B/A22B suffixes are active-parameter
labels and must not be confused with total model size. Neither total nor active
parameters prove local memory use or throughput.

### 3.2 Gemma 3 snapshot

**Official model card:** <https://ai.google.dev/gemma/docs/core/model_card_3>
**Official current lineup:** <https://ai.google.dev/gemma/docs/get_started>
**Technical report:** Gemma Team, “Gemma 3 Technical Report” (2025), arXiv:2503.19786, <https://arxiv.org/abs/2503.19786>.

The current core Gemma 3 lineup is:

- 270M and 1B: text input to text output;
- 4B, 12B, and 27B: text/image input to text output.

The current core range is therefore **270M–27B**. The original 2025 technical report describes the then-released 1B–27B family; the official current model card now includes 270M. Gemma 3n E2B/E4B is a separate variant family and should not be silently merged into the core Gemma 3 parameter range.

### 3.3 MLX does not close the deployment question

**Official MLX quantization API:** <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html>
**Official MLX-LM repository:** <https://github.com/ml-explore/mlx-lm>
**Official LoRA/QLoRA guide:** <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>

The official API documents affine 2/3/4/5/6/8-bit group quantization and MXFP4, MXFP8, and NVFP4 modes. MLX-LM documents LoRA, DoRA, full fine-tuning, and QLoRA when the base model is quantized.

This supports the statement **“MLX provides relevant mechanisms.”** It does not support any of the following without a model-specific run:

- a particular Qwen3-VL or Gemma 3 multimodal checkpoint is fully supported by the current MLX path;
- weights, KV cache, optimizer state, vision tower, world-model modules, and activations fit simultaneously;
- quantization preserves the needed visual, planning, or causal-ablation behavior;
- LoRA training reaches the desired result;
- the intended latency or energy budget is met.

Those remain **UNKNOWN** until measured with exact checkpoint revisions, precisions, sequence/video lengths, batch sizes, and peak unified-memory telemetry.

## 4. What V-JEPA 2 and DreamerV3 actually demonstrate

### 4.1 V-JEPA 2

**Primary paper:** Mido Assran et al., “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning” (2025), arXiv:2506.09985, <https://arxiv.org/abs/2506.09985>.
**Official Meta research page:** <https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/>
**Official project page:** <https://ai.meta.com/research/vjepa/>
**Official repository:** <https://github.com/facebookresearch/vjepa2>

**MEASURED by the source:** V-JEPA 2 is pretrained action-free on more than one million hours of video/images and reports strong motion understanding and action-anticipation results. After alignment to an 8B language model, it reports video-QA results. V-JEPA 2-AC is post-trained as an action-conditioned latent world model with less than 62 hours of unlabeled DROID robot video and is deployed on Franka arms in two labs for image-goal reaching, grasping, and pick-and-place without data from those deployment environments or task-specific rewards.

Required qualification: “zero-shot” here means no data from the two deployment environments and no task-specific training/reward. It does not mean no robot data, no post-training, or arbitrary unseen embodiment/task generalization.

Not established:

- general causal-variable discovery or counterfactual identification;
- open-world continual adaptation with retained revision-safe memory;
- language, audio, tools, and long-horizon autonomous agency in one system;
- exact safety or semantic correctness of imagined trajectories;
- AGI.

### 4.2 DreamerV3 and RSSM

**Primary paper:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap, “Mastering Diverse Control Tasks through World Models” (Nature 2025), DOI:10.1038/s41586-025-08744-2, <https://www.nature.com/articles/s41586-025-08744-2>.
**Official author repository:** <https://github.com/danijar/dreamerv3>

DreamerV3 learns an RSSM, reconstructs observations, predicts rewards/continuations, and trains actor/critic networks on imagined latent trajectories. The RSSM itself is already a hybrid latent mechanism: a deterministic recurrent state and a discrete stochastic state are updated using actions.

**MEASURED by the source:** one fixed hyperparameter configuration is applied across more than 150 tasks in eight domains; model sizes from 12M to 400M are studied; the paper reports strong results across Atari, Procgen, control suites, DMLab, BSuite, and Minecraft. The Minecraft result uses the MineRL competition action space, including abstract crafting actions, and the source reports one GPU for roughly nine days per agent.

Important boundary: fixed hyperparameters across tasks demonstrate algorithmic robustness, not one continually learning universal model. The agents are trained per task/domain setting; the paper itself lists a single cross-domain world model as future work.

Not established:

- causal identification under hidden confounding or novel interventions;
- persistent cross-domain learning without catastrophic forgetting;
- exact symbolic verification or dependency-aware revision;
- multimodal text/audio grounding in the world model;
- AGI.

## 5. Strongest novelty collisions

| Component SHWM might otherwise overclaim | Closest primary source(s) | Collision and remaining distinction |
|---|---|---|
| Learned action-conditioned latent dynamics and MPC | PlaNet (Hafner et al., 2019), <https://proceedings.mlr.press/v97/hafner19a.html> | Direct collision on learning latent dynamics from pixels and online planning. Sentinel verification/provenance is not part of PlaNet. |
| Learned latent model plus tree search | Julian Schrittwieser et al., “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model” (MuZero, Nature 2020), <https://www.nature.com/articles/s41586-020-03051-4> | Direct collision on learning planning-relevant latent dynamics/reward/value/policy without reconstructing full observations. Not an exact semantic verifier or continual revision system. |
| Recurrent hybrid latent state and imagination | DreamerV3/RSSM, <https://doi.org/10.1038/s41586-025-08744-2> | Direct collision on hybrid deterministic/stochastic action-conditioned state and imagined rollouts. |
| Uncertainty ensembles for model-based control | Kurtland Chua et al., “Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models” (PETS, NeurIPS 2018), <https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html> | Direct collision on probabilistic ensembles, trajectory sampling, and model-predictive control. An ensemble is not itself a novelty delta. |
| Frozen pretrained perception | Suraj Nair et al., “R3M: A Universal Visual Representation for Robot Manipulation” (CoRL 2022/2023), <https://proceedings.mlr.press/v205/nair23a.html> | Direct collision on a frozen video-pretrained perception module for downstream policy learning. |
| Action-conditioned joint-embedding planning | V-JEPA 2-AC, <https://arxiv.org/abs/2506.09985> | Very close collision on large video pretraining followed by action-conditioned latent robot planning. |
| Hierarchical latent planning | Danijar Hafner et al., “Deep Hierarchical Planning from Pixels” (Director, NeurIPS 2022), <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html> | High-level latent goals plus low-level behavior already exist. |
| Multi-timescale visual latent MPC | Wancong Zhang et al., “Hierarchical Planning with Latent World Models” (2026 preprint), <https://arxiv.org/abs/2604.03208> | Very recent direct collision: shared-latent models at multiple temporal scales, latent subgoals, macro-actions, and hierarchical MPC. Preprint status limits evidential weight but increases novelty risk. |
| One multimodal multi-task policy | Scott Reed et al., “A Generalist Agent” (Gato, 2022), <https://arxiv.org/abs/2205.06175>; official page <https://deepmind.google/blog/a-generalist-agent/> | Same weights serialize observations/actions/text across many tasks and embodiments. Gato is a policy model, not a learned planning world model, but multimodal generalism is not new. |
| External/persistent agent memory | Joon Sung Park et al., “Generative Agents” (2023), <https://arxiv.org/abs/2304.03442>; Charles Packer et al., “MemGPT” (2023), <https://arxiv.org/abs/2310.08560>; Guanzhi Wang et al., “Voyager” (2023), <https://arxiv.org/abs/2305.16291> | Experience streams, reflection, tiered persistent memory, executable skill libraries, environment feedback, and self-verification all predate SHWM. Exact provenance-linked probabilistic revision is a narrower possible distinction. |
| Knowledge-centric agent memory graph | Ke Yang et al., “PlugMem” (2026 preprint), <https://arxiv.org/abs/2603.03296> | Recent collision on task-agnostic episodic-to-knowledge graph memory and retrieval. Preprint, but a serious novelty threat for any broad memory-graph claim. |
| Machine-checkable verification | George C. Necula, “Proof-Carrying Code” (POPL 1997), DOI:10.1145/263699.263712, <https://doi.org/10.1145/263699.263712> | Machine-checkable evidence for safe execution is longstanding. Sentinel’s contribution cannot be “verification exists”; it must be the measured coupling to uncertain latent planning. |
| Provenance and dependency revision | W3C PROV-O Recommendation (2013), <https://www.w3.org/TR/prov-o/>; Jon Doyle, “A Truth Maintenance System” (1979), DOI:10.1016/0004-3702(79)90008-0, <https://doi.org/10.1016/0004-3702(79)90008-0> | Provenance vocabularies, reasons/dependencies, contradiction-triggered revision, and dependency-directed backtracking are established. The possible delta is their exact operational integration with learned world-model uncertainty and causal ablations. |

## 6. Predictive versus causal world models

### 6.1 The mathematical mismatch

A predictive world model typically estimates

\[
P_\theta(o_{t+1},r_t\mid h_t,a_t),
\]

from trajectories generated by a behavior policy. A causal planning claim requires some version of

\[
P(o_{t+1},r_t\mid h_t,\operatorname{do}(a_t)).
\]

These coincide only under explicit assumptions such as adequate state/causal sufficiency, consistency of the action intervention, no unblocked hidden common causes of action and outcome after conditioning on state, positivity/support for the tested actions, and mechanism invariance across the evaluation shift. When actions are selected by a policy using unrecorded information, or when \(h_t\) aliases causally different states, observational transition fitting is not enough.

Primary foundations and warnings:

- Judea Pearl, “Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution” (2018), <https://arxiv.org/abs/1801.04016>, distinguishes statistical association from intervention/counterfactual reasoning.
- Bernhard Schölkopf et al., “Towards Causal Representation Learning” (2021), <https://arxiv.org/abs/2102.11107>, identifies recovery of high-level causal variables from low-level observations as an open problem tied to transfer and generalization.
- Pim de Haan, Dinesh Jayaraman, Sergey Levine, “Causal Confusion in Imitation Learning” (NeurIPS 2019), <https://proceedings.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html>, shows that predictive access to action effects/nuisance correlates can worsen deployed control and uses targeted interventions to disambiguate causes.
- Minne Li et al., “Causal World Models by Unsupervised Deconfounding of Physical Dynamics” (2020), <https://arxiv.org/abs/2012.14228>, explicitly targets hidden confounding in world models and counterfactual prediction.

**Audit conclusion:** action conditioning is necessary for controllable prediction but is not sufficient evidence of causal identification. SHWM should use “action-conditioned predictive model” until intervention tests justify stronger wording.

### 6.2 Minimum frozen causal-ablation requirements for the proposed delta

Before final evaluation, freeze datasets/generators, seeds, interventions, metrics, thresholds, and all arms. At minimum compare:

1. full SHWM;
2. no learned dynamics;
3. dynamics with actions masked or shuffled;
4. observationally matched but interventionally shifted trajectories;
5. single model versus ensemble at matched compute;
6. continuous versus discrete versus hybrid latent at matched storage/compute;
7. flat versus hierarchical planning at matched planning budget;
8. verifier enabled versus bypassed;
9. provenance/dependency revision enabled versus append-only memory;
10. valid history versus shuffled/reset history;
11. frozen encoder versus controlled fine-tuning;
12. oracle state/dynamics ceilings.

The decisive metric is not only one-step prediction. It should include multi-step intervention error, planning regret under policy shift, calibration of ensemble/open-world uncertainty, verifier catch rate and false-rejection rate, revision correctness after contradicted evidence, memory interference, and task success on held-out dynamics. If the full system does not beat the best simpler arm under equal compute, the integration claim is falsified.

## 7. Multimodal and audio-visual prior art

Multimodality is not a novelty delta.

- Hassan Akbari et al., “VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text” (NeurIPS 2021), <https://proceedings.neurips.cc/paper/2021/hash/cb3213ada48302953cb0f166464ab356-Abstract.html>, learns video/audio/text representations using multimodal contrastive objectives.
- Rohit Girdhar et al., “ImageBind: One Embedding Space to Bind Them All” (CVPR 2023), <https://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.html>, aligns image, text, audio, depth, thermal, and IMU representations through image-paired data.
- Dan Kondratyuk et al., “VideoPoet: A Large Language Model for Zero-Shot Video Generation” (2023), <https://arxiv.org/abs/2312.14125>, handles image/video/text/audio conditioning and multimodal generation.
- Jiahua Wang et al., “Audio-Visual World Models: Towards Multisensory Imagination in Sight and Sound” (2025 preprint), <https://arxiv.org/abs/2512.00883>, proposes synchronized binaural audio/visual/action/reward modeling and evaluates audio-visual navigation. Its priority claims are author claims from a recent preprint, not independently verified here, but it is a direct novelty collision for broad “first audio-visual world model” language.

These works do not establish that a common embedding is causal, sufficient for planning, or robust under intervention. SHWM may test whether exact provenance and verifier feedback improve such a model, but cannot claim that shared multimodal latent spaces or audio-visual prediction are new.

## 8. Benchmark audit and recommended role

| Benchmark | Primary source and official description | What it measures | Recommended SHWM role | Main limitation |
|---|---|---|---|---|
| Procgen | Karl Cobbe et al., “Leveraging Procedural Generation to Benchmark Reinforcement Learning” (2019), <https://arxiv.org/abs/1912.01588>; official repo <https://github.com/openai/procgen> | Sample efficiency and generalization across 16 procedurally generated games/level distributions. | **Early development and held-out-level test.** Useful for action-conditioned prediction, uncertainty, and OOD level generalization. | Mostly compact synthetic visual control; default tasks were designed to minimize memory demands. Not an open-world or causal proof. |
| Crafter | Danijar Hafner, “Benchmarking the Spectrum of Agent Capabilities” (2021), <https://arxiv.org/abs/2109.06780> | Visual open-world survival with semantically meaningful achievements, exploration, crafting, and longer dependencies. | **Development/stress environment after Procgen.** Good for planning depth, sparse achievement transfer, and memory ablations. | Finite authored mechanics and synthetic observations; still not broad causal or multimodal validation. |
| OSWorld | Tianbao Xie et al., “OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments” (2024), <https://arxiv.org/abs/2404.07972>; official project <https://osworld-v1.xlang.ai/>; official repo <https://github.com/xlang-ai/OSWorld> | Execution-based evaluation of hundreds of tasks across real desktop/web applications and operating systems. | **Later external integration gate.** Use only after GUI perception, tool execution, language, persistence, and safety interfaces are stable. | Heavily confounds world-model quality with GUI grounding, operational knowledge, tool reliability, and changing external software. |
| SWE-bench | Carlos E. Jimenez et al., “SWE-bench: Can Language Models Resolve Real-World GitHub Issues?” (ICLR 2024), <https://openreview.net/forum?id=VTF8yNQM66>; official project <https://www.swebench.com/original.html> | Patch generation for 2,294 GitHub issue instances from 12 Python repositories, judged with fail-to-pass and pass-to-pass tests; Verified is a 500-instance engineer-screened subset. | **Later external software-engineering gate**, not a core SHWM dynamics environment. | Primarily tests code understanding, retrieval, editing, tool use, and test-driven verification; weak isolation of learned sensorimotor latent dynamics. |

**INFERRED routing decision:** use controlled Procgen-style interventions first, then Crafter for longer dependencies. Keep OSWorld and SWE-bench untouched as later integration gates. Success on the first two would not imply success on the latter two; failure on OSWorld/SWE-bench would not diagnose a world-model defect without component controls.

## 9. Gato and the boundary between a generalist policy and a world model

Scott Reed et al., “A Generalist Agent” (2022), arXiv:2205.06175, <https://arxiv.org/abs/2205.06175>; official DeepMind page <https://deepmind.google/blog/a-generalist-agent/>.

Gato serializes text, images, observations, and actions into token sequences and uses the same transformer weights across many tasks and embodiments. **MEASURED by the source:** the same policy can perform language, image-captioning, Atari, simulated control, and real robot-arm behaviors, conditioned by context.

Gato is a strong collision for claims that one multimodal network or one token interface across tasks is novel. It is not, however, an action-conditioned latent simulator used for explicit look-ahead planning, nor does it provide causal identification, exact verification, provenance-linked revision, or continual autonomous learning. It is therefore an important baseline concept, not a direct implementation duplicate of the proposed SHWM integration.

## 10. Provisional novelty map

### Clearly not novel

- predictive latent world models;
- action-conditioned latent transitions;
- recurrent stochastic state-space models;
- continuous, discrete, or hybrid latents;
- reconstruction-free/joint-embedding prediction;
- frozen pretrained encoders;
- ensembles for epistemic uncertainty;
- model-predictive control, latent imagination, tree search, and hierarchical latent subgoals;
- multimodal shared embeddings and audio-visual prediction;
- persistent external memory and executable skill libraries;
- provenance graphs, reason/dependency tracking, and belief revision;
- machine-checkable verification or proof-carrying artifacts.

### Possible but unestablished delta

The exact composition below was not located as one measured system in the bounded source set:

\[
\begin{aligned}
&\text{learned action-conditioned hybrid dynamics} \\
&+\ \text{exact executable verifier} \\
&+\ \text{provenance/dependency-aware revision} \\
&+\ \text{persistent memory} \\
&+\ \text{frozen causal ablations}.
\end{aligned}
\]

Even this composition is only potentially publishable if it yields a nontrivial measured result, for example:

- verifier/provenance feedback reduces long-horizon planning regret beyond rejection-only baselines;
- graph-local revision corrects a learned dynamics error without erasing unrelated competence;
- persistent verified memory improves held-out same-mechanism tasks and survives restart;
- intervention generalization improves relative to equally sized predictive-only models;
- open-world uncertainty prevents confident in-class planning when dynamics are absent from the model family.

If the effect comes only from a larger encoder, more data, extra planning compute, or hand-authored evaluator access, the claimed integration delta is not isolated.

## 11. Search protocol, limits, and inaccessible sources

### Search date and source classes attempted

Search performed on **2026-08-28**. The audit used targeted title/abstract searches and citation chaining restricted to primary or official sources:

- arXiv paper records;
- Nature, PMLR, NeurIPS Proceedings, CVF Open Access, OpenReview, and ACM official proceedings pages;
- official Meta AI, Google DeepMind/Google AI, Qwen, Apple MLX, W3C, Procgen, OSWorld, and SWE-bench pages;
- official author or project repositories where relevant.

Targeted negative searches included combinations of:

- `AGI parameter lower bound theorem`;
- `artificial general intelligence minimum parameters 15B 70B`;
- `1024-dimensional latent open-world requirement`;
- `world model latent dimension lower bound`;
- action-conditioned latent planning, causal world models, hierarchical world models, audio-visual world models, and continual agent memory.

### Database/API attempts that did not yield usable coverage

The following direct endpoints were attempted but rejected by this session’s safe-URL layer, so they contribute **no** negative evidence:

- arXiv export API: <https://export.arxiv.org/api/query?search_query=all:%22AGI%22%20AND%20all:%22parameter%20lower%20bound%22&start=0&max_results=20>
- Semantic Scholar Graph API: <https://api.semanticscholar.org/graph/v1/paper/search?query=AGI%20parameter%20lower%20bound&limit=20&fields=title,year,url,externalIds>
- OpenAlex API: <https://api.openalex.org/works?search=AGI%20parameter%20lower%20bound&per-page=25>
- Google Scholar query: <https://scholar.google.com/scholar?q=%22AGI%22+%22parameter+lower+bound%22>

Additional access limits:

- the ACM page for Pearl’s related article at <https://dl.acm.org/doi/10.1145/3241036> returned HTTP 403 in-session; the author’s arXiv version was accessible;
- the IEEE landing page for DOI:10.1109/JPROC.2021.3058954 required JavaScript/bot verification; the authors’ arXiv version of “Towards Causal Representation Learning” was accessible;
- the Qwen3-VL technical report and official repository were located after the initial audit pass; the size discussion above was corrected before this package was finalized;
- very recent 2026 preprints may change status or content after this audit.

### Bounded-search limits

This was not a systematic review and does not establish global absence. It did not exhaust patents, dissertations, non-English work, every workshop, closed industrial systems, all citations of every source, or every paper released near the audit date. Search-index coverage and title/abstract terminology can miss semantically related work. “No directly matching source located” must therefore be read literally, not as “no such work exists.”

## Final evidence-honest conclusion

- **MEASURED from primary sources:** strong prior art exists for action-conditioned latent world models, discrete and hybrid RSSMs, ensemble uncertainty, latent and hierarchical planning, frozen video encoders, multimodal/audio-visual representations, external agent memory, executable verification, provenance, and revision.
- **INFERRED:** Chinchilla does not justify an AGI size threshold; neither the literature nor the audited model cards justify a mandatory 15B–70B model or continuous 1,024D latent.
- **INFERRED:** V-JEPA 2 and DreamerV3 are important bounded demonstrations, not evidence of AGI or general causal identification.
- **INFERRED:** Procgen and Crafter are defensible development environments; OSWorld and SWE-bench are later external integration gates.
- **HYPOTHESIS:** exact Sentinel verification/provenance/revision may add value when coupled to learned action-conditioned dynamics.
- **UNKNOWN:** whether SHWM is feasible on the target machine, whether it outperforms simpler baselines, whether its latent variables support interventions, and whether its proposed integration is novel beyond this bounded audit.

The research direction should proceed only with frozen claims and falsifiers. A positive result would be a measured integration effect under intervention and ablation—not the existence of a large multimodal model, a 1,024-dimensional latent, or a world-model module by itself.
