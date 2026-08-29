# https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math_batch.py
"""
The Stage-1 reward function is based on three criteria: format correctness, execution success, and objective function verification.
"""

import re
import numpy as np
import requests
import json
from collections import Counter
from verl.or_utils.executor import PythonExecutor
from verl.or_utils.content_utils import extract_code_block, extract_code_block_step, extract_obj, extract_step_blocks, normalize_step_title
# from utils import load_jsonl
# url = "http://10.200.250.35:8000/execute"

def code_reward(code_excu_result):
    return code_excu_result=='Done'

def answer_reward(solver_result, ans, code_excu_result, cri = 1e-3):
    if isinstance(ans, str):
        try:
            ans = float(ans)
        except (ValueError, TypeError):
            ans = None
    if isinstance(solver_result, str):
        try:
            solver_result = float(solver_result)
        except (ValueError, TypeError):
            solver_result = None
    abs_err = np.abs(ans) if ans else 1
    if (ans is None and solver_result is None and code_excu_result=='Done'):
        abs_err = 0
    if ans and solver_result:
        abs_err = np.abs(ans - solver_result) / (np.abs(ans) + 1)
    if ans is None:
        ans = 1
    return abs_err <cri

# Code validity receives the largest format weight.
def format_reward(processed_str: str, order:bool=False) -> bool:
    minus_score = 0

    tags = {
        'plan_start':('<plan>', 1),
        'plan_end': ('</plan>', 1),
        'model_start': ('<model', 1),
        'model_end': ('</model>', 1),
        'python_start': ('<python>', 1),
        'python_end': ('</python>', 1)
    }

    position = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        position[tag_name] = pos = processed_str.find(tag_str)

        if count != expected_count:
            if "python" not in tag_name:
                minus_score += 1/8
            else:
                minus_score += 1/4
                
    # Verify tag order
    order_set = [
    position['plan_start'], position['plan_end'],
    position['model_start'], position['model_end'],
    position['python_start'], position['python_end']
]

    if order_set[1] > min(order_set[2:5]):
        minus_score += 1/3

    flag = 0
    for i in range(0, 6, 2):
        if order_set[i] > order_set[i + 1]:
            flag = 1
            break
    if flag == 1:
        minus_score += 1/3

    if order_set[4] <= max(order_set[:4]):
        minus_score += 1/3

    return 2 - minus_score

# by Batch   solution_str, (all rollout response lists) for gurobipy!!
def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
    order = False
    format_score = 0.5
    ans_score = 1.
    # sol_score = 2.
    code_score = 1.
    executor = PythonExecutor()
    response = executor.batch_apply([extract_code_block(solution_str, 'gurobi') for solution_str in solution_strs])
    
    obj_result =[response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]
    """
    # sol_result = [response[1][i] for i in range(len(solution_strs))]
    # if 'sol' in extra_infos[0]:
    #     sol = [sol_reward(extra_infos[i]['sol'], sol_result[i]) for i in range(len(ground_truths))]
    # else:
    #     sol = [0 for i in range(len(ground_truths))]
    """
    format_ = [format_reward(solution_strs[i], order) for i in range(len(solution_strs))]
    code_ = [code_reward(code_excu_result[i]) for i in range(len(code_excu_result))]
    ans = [answer_reward(obj_result[i], ground_truths[i], code_excu_result[i]) for i in range(len(ground_truths))]
    rewards = [ans[i] * ans_score + format_[i] * format_score + code_[i] * code_score for i in range(len(ans))]
    return rewards

#todo: provide a <step></step> version for STepORLM and coptpy code

def compute_score_copt(data_sources, solution_strs, ground_truths, extra_infos):
    """COPT/StepORLM version of compute_score.
    Uses extract_code_block_step for <step> format with ```python blocks.
    """
    from verl.or_utils.content_utils import extract_code_block_step, extract_step_blocks
    format_score = 0.
    ans_score = 1.
    code_score = 0.
    executor = PythonExecutor()
    response = executor.batch_apply(
        [extract_code_block_step(s, 'copt') for s in solution_strs])

    obj_result = [response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]

    def format_reward_step(processed_str: str) -> float:
        steps = extract_step_blocks(processed_str)
        minus_score = 0
        if len(steps) == 0:
            minus_score += 1.5
        elif len(steps) < 9:
            minus_score += (9 - len(steps)) * 0.1
        has_code = any('coptpy' in s or 'import' in s for s in steps) if steps else False
        if not has_code:
            minus_score += 0.5
        return max(0, 2 - minus_score)

    format_ = [format_reward_step(solution_strs[i]) for i in range(len(solution_strs))]
    code_ = [code_reward(code_excu_result[i]) for i in range(len(code_excu_result))]
    ans = [answer_reward(obj_result[i], ground_truths[i], code_excu_result[i]) for i in range(len(ground_truths))]
    rewards = [ans[i] * ans_score + format_[i] * format_score + code_[i] * code_score for i in range(len(ans))]
    return rewards


def compute_score_simple(data_sources, solution_strs, ground_truths, extra_infos):
    """Gurobi simplified reward: answer relative error only.
    Extracts code from the current 9-step format or legacy code blocks, executes,
    and compares obj with ground_truth. Reward: correct -> 1.0, wrong -> 0.0.
    """
    executor = PythonExecutor()
    response = executor.batch_apply(
        [extract_code_block_step(s, 'gurobi') for s in solution_strs])
    obj_result = [response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]
    rewards = []
    for i in range(len(solution_strs)):
        majority_score = _majority_membership_score(
            extra_infos[i] if i < len(extra_infos) else None,
            ground_truths[i],
            code_excu_result[i],
        )
        if majority_score is None:
            answer_score = _stage1_answer_reward(
                obj_result[i],
                ground_truths[i],
                code_excu_result[i],
            )
            pass_rate = _stage1_answer_pass_rate(
                obj_result[i],
                ground_truths[i],
            )
        else:
            answer_score, pass_rate = majority_score
        rewards.append(
            {
                "score": answer_score,
                "answer_score": answer_score,
                "pass_rate": pass_rate,
                "code_score": float(code_reward(code_excu_result[i])),
                "solver_obj": obj_result[i],
            }
        )
    return rewards


def compute_score_simple_copt(data_sources, solution_strs, ground_truths, extra_infos):
    """COPT/StepORLM simplified reward: answer relative error only.
    Extracts code from <step> ```python blocks, executes, compares obj with ground_truth.
    Reward: correct (within 1e-6 relative error) → 1.0, wrong → 0.0
    """
    from verl.or_utils.content_utils import extract_code_block_step
    executor = PythonExecutor()
    response = executor.batch_apply(
        [extract_code_block_step(s, 'copt') for s in solution_strs])
    obj_result = [response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]
    rewards = []
    for i in range(len(solution_strs)):
        majority_score = _majority_membership_score(
            extra_infos[i] if i < len(extra_infos) else None,
            ground_truths[i],
            code_excu_result[i],
        )
        if majority_score is None:
            answer_score = _stage1_answer_reward(
                obj_result[i],
                ground_truths[i],
                code_excu_result[i],
            )
            pass_rate = _stage1_answer_pass_rate(
                obj_result[i],
                ground_truths[i],
            )
        else:
            answer_score, pass_rate = majority_score
        rewards.append(
            {
                "score": answer_score,
                "answer_score": answer_score,
                "pass_rate": pass_rate,
                "code_score": float(code_reward(code_excu_result[i])),
                "solver_obj": obj_result[i],
            }
        )
    return rewards

STAGE1_GUROBI_STEPS = [
    "Problem Description",
    "Sets and Parameters",
    "Decision Variables",
    "Objective Function",
    "Constraints",
    "Mathematical Model",
    "Nonlinear Relationships",
    "Final Model",
    "Python Code Using gurobipy",
]


STAGE1_GUROBI_STEP_TITLES_NORM = tuple(normalize_step_title(step) for step in STAGE1_GUROBI_STEPS)


def _stage1_block_has_title_and_body(block: str, expected: str) -> bool:
    block_norm = normalize_step_title(block)
    expected_norm = normalize_step_title(expected)
    if expected_norm not in block_norm:
        return False

    # Repeated titles, numbered title lines, and Markdown-only decorations should
    # not count as body.
    for line in str(block or "").splitlines():
        line_norm = normalize_step_title(line)
        if not line_norm:
            continue
        line_norm = re.sub(r"^\d+\s+", "", line_norm).strip()
        if line_norm == expected_norm:
            continue
        if line_norm.startswith(expected_norm + " "):
            line_norm = line_norm[len(expected_norm):].strip()
        if line_norm and line_norm not in STAGE1_GUROBI_STEP_TITLES_NORM:
            return True
    return False


def _stage1_code_block_in_step(block: str) -> bool:
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", block or "", re.DOTALL | re.IGNORECASE)
    if not fence:
        return False
    code = fence.group(1).lower()
    return "gurobipy" in code or "model.optimize" in code


def _stage1_gurobi_format_details(solution_str: str) -> tuple[float, dict[str, float]]:
    """Score the required 9-step Gurobi response format.

    Each expected stage contributes 0.1. The expected stage title and content
    must appear in the corresponding <step> block so the reward favors the exact
    ordered flow from verl/or_utils/gurobi_prompt.py. The code stage must keep
    its fenced Python code inside the same <step>...</step> block.
    """
    blocks = extract_step_blocks(solution_str)
    details = {}
    total = 0.0

    for idx, expected in enumerate(STAGE1_GUROBI_STEPS):
        block = blocks[idx] if idx < len(blocks) else ""
        has_title_and_body = _stage1_block_has_title_and_body(block, expected)
        if idx == len(STAGE1_GUROBI_STEPS) - 1:
            ok = float(has_title_and_body and _stage1_code_block_in_step(block))
        else:
            ok = float(has_title_and_body)
        details[f"stage_{idx + 1}_format"] = ok
        total += 0.1 * ok

    details["stage_count"] = float(len(blocks))
    return total, details


def _stage1_answer_reward(solver_result, ground_truth, code_excu_result, cri: float = 1e-6) -> float:
    if code_excu_result != "Done":
        return 0.0
    if solver_result is None and ground_truth is None:
        return 1.0
    try:
        solver_value = float(solver_result)
        truth_value = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    rel_err = np.abs(solver_value - truth_value) / (np.abs(truth_value) + 1.0)
    return float(rel_err < cri)


def _stage1_answer_pass_rate(solver_result, ground_truth, cri: float = 1e-6) -> float:
    if solver_result is None or ground_truth is None:
        return 0.0
    try:
        solver_value = float(solver_result)
        truth_value = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    rel_err = np.abs(solver_value - truth_value) / (np.abs(truth_value) + 1.0)
    return float(rel_err < cri)


def _majority_membership_score(
    extra_info,
    ground_truth,
    code_exec_result,
) -> tuple[float, float] | None:
    """Return (answer reward, pass rate) when scoring a solver-vote target."""

    if not isinstance(extra_info, dict):
        return None
    cluster_member = extra_info.get("majority_cluster_member")
    majority_vote_target = extra_info.get("majority_vote_target")
    if (
        cluster_member is None
        or majority_vote_target is None
        or not _stage1_answer_pass_rate(majority_vote_target, ground_truth)
    ):
        return None
    return (
        float(bool(cluster_member) and code_reward(code_exec_result)),
        float(bool(cluster_member)),
    )


# stage1
def stage1_reward_gurobi(data_sources, solution_strs, ground_truths, extra_infos): #currently gurobipy version
    """
    SEOP: for stage 1 of self-enhanced onpolicy distillation. In this stage, model is trained to fit in the format
    of multilple stage in verl/or_utils/gurobi_prompt.py:
    reward is assigned to complete reward that:
    1. format score for every stage: 0.1 * 9 #nine stage inference
    2. correctly execute( code reward): 0.1
    3. answer score: 1.0
    addtional metrics to integrate: 1.stage scores.
    use grpo to train.
    """
    executor = PythonExecutor()
    code_blocks = [extract_code_block_step(solution_str, "gurobi") for solution_str in solution_strs]
    response = executor.batch_apply(code_blocks)

    obj_result = [response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]

    rewards = []
    for i, solution_str in enumerate(solution_strs):
        stage_score, stage_details = _stage1_gurobi_format_details(solution_str)
        code_score = 0.1 if code_reward(code_excu_result[i]) else 0.0
        majority_score = _majority_membership_score(
            extra_infos[i] if i < len(extra_infos) else None,
            ground_truths[i],
            code_excu_result[i],
        )
        if majority_score is None:
            answer_score = _stage1_answer_reward(
                obj_result[i],
                ground_truths[i],
                code_excu_result[i],
            )
            pass_rate = _stage1_answer_pass_rate(
                obj_result[i],
                ground_truths[i],
            )
        else:
            answer_score, pass_rate = majority_score
        total_score = stage_score + code_score + answer_score

        rewards.append(
            {
                "score": total_score,
                "stage_score": stage_score,
                "code_score": code_score,
                "answer_score": answer_score,
                "pass_rate": pass_rate,
                "solver_obj": obj_result[i],
                **stage_details,
            }
        )

    return rewards


def stage1_reward(data_sources, solution_strs, ground_truths, extra_infos):
    return stage1_reward_gurobi(data_sources, solution_strs, ground_truths, extra_infos)
