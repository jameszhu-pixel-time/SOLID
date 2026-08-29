# Method-to-code correspondence

## Scope

The implementation uses the `verl/` package namespace and the standard veRL
PPO trainer architecture. SOLID-specific logic is concentrated in the files
listed below; unchanged framework modules provide distributed training,
rollout, checkpoint, and model support.

## Paper algorithm

| Method component | Implementation | Contract |
| --- | --- | --- |
| Execute rollout programs and capture objective, status, and LP | `verl/trainer/ppo/ttrl_utils.py::get_solver_feedback` and `verl/or_utils/executor.py` | Returns one objective, execution status, and candidate LP per rollout. |
| Normalize objective direction and form tolerance clusters | `verl/trainer/ppo/ttrl_utils.py::solver_majority_vote` | Minimization uses `+z`, maximization uses `-z`; finite executable objectives within `1e-6` form complete-linkage clusters. |
| Select majority objective and pseudo-reference | `solver_majority_vote` and `select_reference_lp_from_feedback` | Largest cluster wins; dispersion and rollout order break ties. The usable LP nearest the cluster median is selected. |
| TTRL sequence reward | `apply_ttrl_gt`, `verl/or_utils/batch_score_gurobi.py` | Majority-cluster membership becomes the label-free answer reward used by GRPO. |
| Parse and canonicalize LP artifacts | `verl/or_utils/lp_step_error.py` | Produces anonymous variable, objective, row, and complete-LP signatures. |
| Map LP differences to response sections | `localize_solver_info` and `build_solver_info_weight_mask` | Variable differences map to Step 3, objective differences to Step 4, constraints to Step 5, and emitted/executed code differences to Step 9. |
| Exclude majority samples from teacher KL | `build_teacher_kl_sample_mask` | Majority-cluster responses receive GRPO only; non-majority responses remain eligible for localized teacher KL. |
| Build privileged teacher prompt | `verl/trainer/ppo/teacher_prompt_utils.py` and `RayPPOTrainer._maybe_build_self_distillation_batch` | Inserts the selected objective and LP only for no-gradient contextual re-scoring. |
| Compute contextual teacher log probabilities | `DataParallelPPOActor.compute_log_prob` | The teacher sees privileged context; the deployable student sees the original prompt. |
| Apply masked reverse KL | `core_algos.kl_penalty(..., "low_var_kl")`, `apply_stage_kl_random_flip`, and actor update code | Uses the sampled reverse-KL estimator `exp(delta) - delta - 1`, multiplied by sample and section masks. |
| Refresh the on-policy teacher | `DataParallelPPOActor._update_teacher` | With `teacher_update_rate=1.0`, the teacher becomes the current actor after each optimizer update. |

## TTRL baseline

The baseline and SOLID share program execution, majority voting, TTRL reward,
GRPO, rollout count, batch size, and optimizer settings. The launcher
`scripts/run_ttrl.sh` changes only:

- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.actor.bootstrap_kl=False`

This removes contextual teacher KL and its LP-structured mask while preserving
the outcome-only group-relative objective.

## LP localization behavior

The parser intentionally performs deterministic structural comparison, not
mathematical-equivalence proving:

- variable names, constraint labels, and record order are ignored;
- variable types, bounds, objective coefficients, matrix columns, row
  coefficients, senses, and right-hand sides are compared;
- fixed variables are folded into the objective constant;
- invalid or missing candidate LPs activate only the observable code section;
- a missing reference yields no teacher signal;
- nonlinear and advanced solver constructs are outside the supported grammar.

## Paper defaults

`scripts/run_solid.sh` encodes the matched-budget defaults:

- 24 rollouts per prompt;
- training batch size 32;
- learning rate `1e-6`;
- GRPO clipping inherited from the standard actor config (`0.2`);
- reverse-KL coefficient `1e-3`;
- rollout temperature `1.0`;
- prompt/response limits 4,096/32,768;
- four GPUs;
- majority objective voting and majority-LP selection;
- teacher refresh rate `1.0`.

Models, datasets, licenses, physical GPU identifiers, and output locations are
runtime inputs and are intentionally absent from the repository.
