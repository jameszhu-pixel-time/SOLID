#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to a local model or model identifier.}"
: "${TRAIN_FILES:?Set TRAIN_FILES as a Hydra list, for example [\"/path/train.parquet\"].}"
: "${VAL_FILES:?Set VAL_FILES as a Hydra list, for example [\"/path/val.parquet\"].}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TASK="${TASK:-.}"
export REWARD_MAX_WORKERS="${REWARD_MAX_WORKERS:-64}"
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
export WANDB_MODE=disabled

CONFIG_NAME="${CONFIG_NAME:-sdpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-solid}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${EXPERIMENT_NAME}}"
RAY_TMP_DIR="${RAY_TMP_DIR:-${RUN_ROOT}/ray/${EXPERIMENT_NAME}}"

SOLVER_NAME="${SOLVER_NAME:-gurobi}"
case "${SOLVER_NAME}" in
  gurobi)
    TEMPLATE_STYLE="${TEMPLATE_STYLE:-gurobi}"
    REWARD_FUNCTION="${REWARD_FUNCTION:-stage1_reward_gurobi}"
    ;;
  copt)
    TEMPLATE_STYLE="${TEMPLATE_STYLE:-copt}"
    REWARD_FUNCTION="${REWARD_FUNCTION:-compute_score_simple_copt}"
    ;;
  *)
    echo "SOLVER_NAME must be 'gurobi' or 'copt', got '${SOLVER_NAME}'." >&2
    exit 2
    ;;
esac

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ROLLOUT_N="${ROLLOUT_N:-24}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
SELF_DISTILL_MAX_REPROMPT_LEN="${SELF_DISTILL_MAX_REPROMPT_LEN:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-38912}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-1.0}"
SUCCESS_REWARD_THRESHOLD="${SUCCESS_REWARD_THRESHOLD:-2.0}"
SOLID_ENABLE="${SOLID_ENABLE:-true}"

ROLLOUT_REQUIRED_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TEACHER_REQUIRED_MODEL_LEN=$((SELF_DISTILL_MAX_REPROMPT_LEN + MAX_RESPONSE_LENGTH))
REQUIRED_MODEL_LEN="${ROLLOUT_REQUIRED_MODEL_LEN}"
if ((TEACHER_REQUIRED_MODEL_LEN > REQUIRED_MODEL_LEN)); then
  REQUIRED_MODEL_LEN="${TEACHER_REQUIRED_MODEL_LEN}"
fi
if ((MAX_MODEL_LEN < REQUIRED_MODEL_LEN)); then
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} is smaller than required length ${REQUIRED_MODEL_LEN}." >&2
  exit 2
fi

CUSTOM_REWARD_PATH="${PROJECT_ROOT}/verl/or_utils/batch_score_gurobi.py"
COMMON_ARGS=(
  "hydra.run.dir=${RUN_DIR}/hydra"
  "hydra.output_subdir=null"
  "max_model_len=${MAX_MODEL_LEN}"
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.chat_template_style=${TEMPLATE_STYLE}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=False"
  "algorithm.rollout_correction.rollout_is=token"
  "algorithm.rollout_correction.rollout_is_threshold=2.0"
  "ttrl.enable=True"
  "ttrl.voting_strategy=majority"
  "ttrl.solver_name=${SOLVER_NAME}"
  "ttrl.n_votes_per_prompt=${ROLLOUT_N}"
  "ttrl.n_samples_per_prompt=${ROLLOUT_N}"
  "ttrl.use_or_template=True"
  "ttrl.lp_selection_target=voted_answer"
  "trainer.apply_OR_distill=True"
  "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
  "trainer.logger=[\"console\"]"
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
  "trainer.rollout_data_dir=${RUN_DIR}/rollouts"
  "trainer.validation_data_dir=${RUN_DIR}/validation"
  "custom_reward_function.path=${CUSTOM_REWARD_PATH}"
  "custom_reward_function.name=${REWARD_FUNCTION}"
  "reward_model.reward_manager=batch"
  "+ray_kwargs.ray_init._temp_dir=${RAY_TMP_DIR}"
)

if [[ "${SOLID_ENABLE}" == "true" ]]; then
  METHOD_ARGS=(
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_ref_log_prob_source=teacher"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.self_distillation.max_reprompt_len=${SELF_DISTILL_MAX_REPROMPT_LEN}"
    "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE}"
    "actor_rollout_ref.actor.self_distillation.success_reward_threshold=${SUCCESS_REWARD_THRESHOLD}"
    "actor_rollout_ref.actor.self_distillation.use_or_template=True"
    "actor_rollout_ref.actor.self_distillation.or_template_style=${TEMPLATE_STYLE}"
    "actor_rollout_ref.actor.bootstrap_kl=True"
    "actor_rollout_ref.actor.bootstrap_kl_decay=solver_info"
    "actor_rollout_ref.actor.bootstrap_kl_style=${TEMPLATE_STYLE}"
  )
else
  METHOD_ARGS=(
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.bootstrap_kl=False"
  )
fi

mkdir -p "${RUN_DIR}" "${RAY_TMP_DIR}"
COMMAND=(
  bash "${PROJECT_ROOT}/training/launch.sh"
  "${EXPERIMENT_NAME}"
  "${CONFIG_NAME}"
  "${COMMON_ARGS[@]}"
  "${METHOD_ARGS[@]}"
  "$@"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
else
  "${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
fi

