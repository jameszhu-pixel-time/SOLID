# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

# from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade
from verl.utils.reward_score.ttrl_math import simplify_expression_string
from verl.or_utils.executor import PythonExecutor
# from .utils import load_jsonl
def extract_extra_info(batch):
    """Returns flattened extra info in batch, grouped by key.
    Each per-prompt list is flattened so the result aligns with
    batch.repeat(n) which expands num_prompts -> num_prompts * n.

    Returns:
        tuple of (solved_objective_flat, solution_flat, code_exec_res_flat,
        candidate_lp_formulation_flat)
    """
    Required_keys = [
        "solved_objective",
        "solution",
        "code_exec_res",
        "candidate_lp_formulations",
    ]
    grouped = {k: [] for k in Required_keys}
    for data_items in batch:
        for key in Required_keys:
            val = data_items.non_tensor_batch["extra_info"].get(key, None)
            if isinstance(val, list):
                grouped[key].extend(val)
            else:
                grouped[key].append(val)
    return (grouped[k] for k in Required_keys)
                
    

def _safe_to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.lower() == "none":
            return None
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _is_zero_objective(x, tol=1e-12):
    numeric = _safe_to_float(x)
    return numeric is not None and abs(numeric) <= tol


def _answer_reward_like_compute_score(solver_result, ground_truth, code_exec_res="Done", cri=1e-6):
    pred = _safe_to_float(solver_result)
    gt = _safe_to_float(ground_truth)

    if code_exec_res != "Done":
        return False
    if pred is None or gt is None:
        return False

    rel_err = abs(pred - gt) / (abs(gt) + 1.0)
    return rel_err < cri


def _answer_only_correct(solver_result, ground_truth, cri=1e-6):
    pred = _safe_to_float(solver_result)
    gt = _safe_to_float(ground_truth)

    if pred is None or gt is None:
        return False

    rel_err = abs(pred - gt) / (abs(gt) + 1.0)
    return rel_err < cri


@dataclass(frozen=True)
class SolverMajorityVote:
    """Deterministic solver-objective vote and its selected LP reference."""

    voted_objective: str
    normalized_center: float | None
    member_indices: tuple[int, ...]
    reference_index: int | None
    ratio: float


def _solver_status_done(status: Any) -> bool:
    return isinstance(status, str) and status.strip().lower() == "done"


def _objective_direction_multiplier(lp_content: str | None) -> float | None:
    """Map minimization to +1 and maximization to -1 for a parseable LP."""

    if not lp_content:
        return None
    try:
        from verl.or_utils.lp_step_error import parse_lp_text

        model = parse_lp_text(lp_content)
    except (TypeError, ValueError, ArithmeticError, KeyError, IndexError):
        return None
    return 1.0 if model.objective_sense == "minimize" else -1.0


def solver_majority_vote(
    objectives: list[Any],
    exec_results: list[Any],
    candidate_lps: list[str | None],
    *,
    tolerance: float = 1e-6,
) -> SolverMajorityVote:
    """Cluster executable, direction-normalized objectives and select one LP.

    Objective values within ``tolerance`` are connected into vote clusters.
    Clusters are ranked by size, mean absolute deviation from their median,
    and earliest rollout index.  The reference is the usable LP in the winning
    cluster closest to its median, with rollout order as the final tie-break.
    """

    if not (
        len(objectives) == len(exec_results) == len(candidate_lps)
    ):
        raise ValueError(
            "objectives, execution results, and candidate LPs must have equal lengths"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    valid: list[tuple[int, float, float, float, str | None]] = []
    for index, (objective, status, candidate_lp) in enumerate(
        zip(objectives, exec_results, candidate_lps, strict=True)
    ):
        raw_value = _safe_to_float(objective)
        if (
            not _solver_status_done(status)
            or raw_value is None
            or not math.isfinite(raw_value)
        ):
            continue
        parsed_direction = _objective_direction_multiplier(candidate_lp)
        direction = parsed_direction if parsed_direction is not None else 1.0
        valid.append(
            (
                index,
                raw_value,
                direction * raw_value,
                direction,
                candidate_lp if parsed_direction is not None else None,
            )
        )

    if not valid:
        return SolverMajorityVote("None", None, (), None, 0.0)

    # Complete-linkage intervals prevent tolerance chaining: every pair of
    # objectives in one cluster differs by at most ``tolerance``.
    components: list[list[int]] = []
    sorted_positions = sorted(
        range(len(valid)),
        key=lambda position: (valid[position][2], valid[position][0]),
    )
    for position in sorted_positions:
        if (
            not components
            or valid[position][2] - valid[components[-1][0]][2] > tolerance
        ):
            components.append([position])
        else:
            components[-1].append(position)

    def component_summary(component: list[int]) -> tuple[float, float, int]:
        values = [valid[position][2] for position in component]
        center = float(np.median(values))
        dispersion = sum(abs(value - center) for value in values) / len(values)
        first_rollout = min(valid[position][0] for position in component)
        return center, dispersion, first_rollout

    winning_component = min(
        components,
        key=lambda component: (
            -len(component),
            component_summary(component)[1],
            component_summary(component)[2],
        ),
    )
    normalized_center, _, _ = component_summary(winning_component)
    member_indices = tuple(
        sorted(valid[position][0] for position in winning_component)
    )

    usable_reference_positions = [
        position
        for position in winning_component
        if valid[position][4]
    ]
    reference_index = None
    representative_position = min(
        winning_component,
        key=lambda position: valid[position][0],
    )
    voted_objective = str(
        normalized_center / valid[representative_position][3]
    )
    if usable_reference_positions:
        reference_position = min(
            usable_reference_positions,
            key=lambda position: (
                abs(valid[position][2] - normalized_center),
                valid[position][0],
            ),
        )
        reference_index = valid[reference_position][0]
        reference_direction = valid[reference_position][3]
        voted_objective = str(normalized_center / reference_direction)

    return SolverMajorityVote(
        voted_objective=voted_objective,
        normalized_center=normalized_center,
        member_indices=member_indices,
        reference_index=reference_index,
        ratio=len(winning_component) / len(valid),
    )


def build_teacher_kl_sample_mask(
    self_distillation_mask: "torch.Tensor",
    seq_scores: "torch.Tensor",
    extra_infos: list[dict[str, Any]],
    *,
    success_reward_threshold: float,
    require_majority_membership: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor", bool]:
    """Build the per-sample teacher-KL gate.

    Solver-vote membership is authoritative when every sample carries it:
    majority-cluster samples are excluded and non-majority samples remain
    eligible. Generic self-distillation callers without vote metadata retain
    the historical reward-threshold gate.
    """

    if self_distillation_mask.ndim != 1 or seq_scores.ndim != 1:
        raise ValueError(
            "self_distillation_mask and seq_scores must both be rank-1"
        )
    batch_size = self_distillation_mask.shape[0]
    if seq_scores.shape[0] != batch_size or len(extra_infos) != batch_size:
        raise ValueError(
            "mask, scores, and extra_infos must have the same batch size"
        )

    membership_available = batch_size > 0 and all(
        isinstance(item, dict)
        and item.get("majority_cluster_member") is not None
        for item in extra_infos
    )
    if require_majority_membership and not membership_available:
        raise ValueError(
            "solver-vote teacher gating requires majority membership for "
            "every sample"
        )
    if membership_available:
        excluded_mask = torch.tensor(
            [
                bool(item["majority_cluster_member"])
                for item in extra_infos
            ],
            dtype=torch.bool,
            device=self_distillation_mask.device,
        )
    else:
        excluded_mask = (
            seq_scores.to(self_distillation_mask.device)
            >= success_reward_threshold
        )

    teacher_kl_mask = (
        self_distillation_mask.float() * (~excluded_mask).float()
    )
    return teacher_kl_mask, excluded_mask, membership_available


def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt
    assert n_samples_per_prompt <= n_votes_per_prompt, "n_samples_per_prompt shoud be less than n_votes"
    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]

def select_top_k_per_prompt_result(batch, n_votes_per_prompt, n_samples_per_prompt):
    assert n_samples_per_prompt <= n_votes_per_prompt
    num_prompts = len(batch)

    for i in range(num_prompts):
        data_item = batch[i]

        obj_ls = data_item.non_tensor_batch["extra_info"]["solved_objective"]
        sol_ls = data_item.non_tensor_batch["extra_info"]["solution"]
        code_ls = data_item.non_tensor_batch["extra_info"]["code_exec_res"]

        assert len(obj_ls) == n_votes_per_prompt, f"prompt {i}: obj len={len(obj_ls)}"
        assert len(sol_ls) == n_votes_per_prompt, f"prompt {i}: sol len={len(sol_ls)}"
        assert len(code_ls) == n_votes_per_prompt, f"prompt {i}: code len={len(code_ls)}"

        data_item.non_tensor_batch["extra_info"]["solved_objective"] = obj_ls[:n_samples_per_prompt]
        data_item.non_tensor_batch["extra_info"]["solution"] = sol_ls[:n_samples_per_prompt]
        data_item.non_tensor_batch["extra_info"]["code_exec_res"] = code_ls[:n_samples_per_prompt]

    return batch
# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch

#warning always do this before apply original_gt
def apply_ttrl_gt(
    batch,
    gen_batch_output,
    n,
    tokenizer,
    voting_strategy="majority",
    solver_name="gurobi",
    extract_lp: bool = False,
):
    """
    Apply the voted ground truth to the batch.
    get the results and vote using the specified strategy.
    Args:
        voting_strategy: "majority" | "zero_filtered_majority" | "complexity_aligned" | "golden_answer"
    Warning: inplace modification.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    ##TODO check reward model with qids;
    # qids_to_resp = {}
    # for i in range(num_prompts):
    #     qid = data_item.non_tensor_batch["extra_info"]["qid"]
    #     data_item = gen_batch_output[i]
    #     response = data_item.batch["prompts"]
    #     prompt_ids = data_item.batch["prompts"]
    #     prompt_length = prompt_ids.shape[-1]
    #     response_ids = data_item.batch["responses"]
    #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    #     valid_response_ids = response_ids[:valid_response_length]
    #     response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    #     if qid not in list(qids_to_resp.keys()):
    #         qids_to_resp[qid]=[response_str]
    #     else:
    #         qids_to_resp[qid].appex wnd(response_str)
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j] #gen_batch is flat;
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)
    if len(model_outputs) > 0:
        print(f"[DEBUG] generation detokenized: {model_outputs[0]}")
    # if len(model_outputs) > 10:
    #     print(f"[DEBUG] generation detokenized: {model_outputs[10]}")
    # if len(model_outputs) > 100:
    #     print(f"[DEBUG] generation detokenized: {model_outputs[100]}")
    extract_mi = True
    batch_obj, batch_sol, batch_report, batch_model_infos, batch_lp_contents = get_solver_feedback(
        model_outputs,
        extract_model_info=extract_mi,
        solver_name=solver_name,
        extract_lp=extract_lp,
    )
    original_gt_list = [
        batch[i].non_tensor_batch["reward_model"]["ground_truth"]
        for i in range(num_prompts)
    ]
    solver_vote_outcomes: list[SolverMajorityVote] | None = None
    if voting_strategy == "golden_answer":
        majority_gt_list, majority_ratio_list = _batch_golden_answer_vote(
            batch_obj, batch_report, original_gt_list, n)
    elif voting_strategy in ("zero_filtered_majority", "complexity_aligned"):
        majority_gt_list, majority_ratio_list = _batch_self_enhanced_vote(
            batch_obj, batch_report, n, strategy=voting_strategy,
            model_infos=batch_model_infos if extract_mi else None)
    else:
        solver_vote_outcomes = []
        for prompt_index in range(num_prompts):
            start = prompt_index * n
            end = start + n
            solver_vote_outcomes.append(
                solver_majority_vote(
                    batch_obj[start:end],
                    batch_report[start:end],
                    batch_lp_contents[start:end],
                )
            )
        majority_gt_list = [
            outcome.voted_objective for outcome in solver_vote_outcomes
        ]
        majority_ratio_list = [
            outcome.ratio for outcome in solver_vote_outcomes
        ]
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):##broadcast
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        start = i * n
        end = start + n

        data_item.non_tensor_batch["extra_info"]["solved_objective"] = batch_obj[start:end]
        data_item.non_tensor_batch["extra_info"]["solution"] = batch_sol[start:end]
        data_item.non_tensor_batch["extra_info"]["code_exec_res"] = batch_report[start:end]
        data_item.non_tensor_batch["extra_info"]["model_infos"] = batch_model_infos[start:end]
        data_item.non_tensor_batch["extra_info"]["candidate_lp_formulations"] = batch_lp_contents[start:end]
        if solver_vote_outcomes is not None:
            outcome = solver_vote_outcomes[i]
            member_set = set(outcome.member_indices)
            data_item.non_tensor_batch["extra_info"]["majority_cluster_members"] = [
                rollout_index in member_set for rollout_index in range(n)
            ]
            data_item.non_tensor_batch["extra_info"]["majority_reference_index"] = (
                outcome.reference_index
            )
            data_item.non_tensor_batch["extra_info"]["majority_vote_center_normalized"] = (
                outcome.normalized_center
            )

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _batch_golden_answer_vote(
    model_outputs: List[str],
    code_exec_results: List[str],
    ground_truths: List[str],
    n: int,
) -> tuple[List[str], List[float]]:
    """Use the dataset golden answer as the vote target.

    The returned ratio is the share of executable numeric candidates whose
    objective matches the golden answer. LP extraction later uses this target to
    select a correct candidate's formulation.
    """
    assert len(model_outputs) % n == 0
    assert len(model_outputs) == len(code_exec_results)
    n_prompts = len(model_outputs) // n
    assert len(ground_truths) == n_prompts

    gt_list = []
    ratio_list = []
    for i in range(n_prompts):
        start, end = i * n, (i + 1) * n
        golden = ground_truths[i]
        golden_value = _safe_to_float(golden)
        if golden_value is None:
            gt_list.append("None")
            ratio_list.append(0.0)
            continue

        gt, ratio = _golden_answer_vote(model_outputs[start:end], code_exec_results[start:end], golden)
        gt_list.append(gt)
        ratio_list.append(ratio)

    return gt_list, ratio_list


def _golden_answer_vote(objectives: list, exec_results: list, golden_answer) -> tuple[str, float]:
    """Return the golden answer and the correct-candidate ratio."""
    assert len(objectives) == len(exec_results)
    if _safe_to_float(golden_answer) is None:
        return "None", 0.0

    valid_count = 0
    matched_count = 0
    for obj, status in zip(objectives, exec_results):
        if status != "Done" or _safe_to_float(obj) is None:
            continue
        valid_count += 1
        if _answer_only_correct(obj, golden_answer):
            matched_count += 1

    return str(golden_answer), matched_count / valid_count if valid_count > 0 else 0.0


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = model_outputs
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    model_answers = [answer for answer in model_answers if not _is_zero_objective(answer)]
    if len(model_answers) == 0:
        return "None", 0.0

    counter = Counter(model_answers)

    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_answers)
    return majority_answer, majority_ratio


def _batch_self_enhanced_vote(model_outputs: List[str], code_exec_results: List[str],
                              n: int, strategy: str = "zero_filtered_majority",
                              model_infos: Optional[List[dict]] = None) -> tuple[List[str], List[float]]:
    """Batch self-enhanced voting with configurable strategy.
    Args:
        strategy: "majority" | "zero_filtered_majority" | "complexity_aligned"
        model_infos: optional list of model_info dicts (required for complexity_aligned)
    """
    gt_list = []
    ratio_list = []
    assert len(model_outputs) % n == 0
    assert len(code_exec_results) == len(model_outputs)
    n_prompts = len(model_outputs) // n

    for i in range(n_prompts):
        start, end = i * n, (i + 1) * n
        prompt_outputs = model_outputs[start:end]
        prompt_exec = code_exec_results[start:end]

        if strategy == "zero_filtered_majority":
            gt, ratio = _zero_filtered_majority_vote(prompt_outputs, prompt_exec)
        elif strategy == "complexity_aligned":
            prompt_mi = model_infos[start:end] if model_infos else [{} for _ in range(n)]
            gt, ratio = _complexity_aligned_vote(prompt_outputs, prompt_exec, prompt_mi)
        else:
            gt, ratio = _majority_vote(prompt_outputs)

        gt_list.append(gt)
        ratio_list.append(ratio)

    return gt_list, ratio_list


# === Model Info Extraction ===
MODEL_INFO_MARKER = "__TTRL_MODEL_INFO__"
VARIABLE_PATTERN = re.compile(
    r"Var:\s*(?P<name>[^,]+),\s*Type:\s*(?P<type>[^,]+),\s*"
    r"LB:\s*(?P<lb>[^,]+),\s*UB:\s*(?P<ub>[^,]+),\s*Obj:\s*(?P<obj>[^,\s]+)"
)
CONSTRAINT_PATTERN = re.compile(
    r"Constr:\s*(?P<name>[^,]+),\s*Sense:\s*(?P<sense>[^,]+),\s*RHS:\s*(?P<rhs>[^,\s]+)"
)
MATRIX_ENTRY_PATTERN = re.compile(
    r"A\[\d+,\d+\]\s*=\s*(?P<weight>[-+0-9.eE]+)\s*"
    r"\(constr=(?P<constraint>[^,]+),\s*var=(?P<variable>[^)]+)\)"
)

COMPLEXITY_ALIGNED_BETA = 40.0
COMPLEXITY_ALIGNED_OVERRIDE_MARGIN = 1.0
COMPLEXITY_ALIGNED_CONSENSUS_MIN_COUNT = 1
COMPLEXITY_ALIGNED_ALIGNMENT_FLOOR = 1e-3
COMPLEXITY_ALIGNED_CHANNEL_WEIGHTS = (0.45, 0.35, 0.20)
ZERO_ABS_TOL = 1e-6
# PLACEHOLDER_VOTING_HELPERS

def _build_model_introspection_snippet(solver_name: str = "gurobi") -> str:
    """Generate Python code to append to scripts for model info extraction.
    Supports both gurobi and copt solvers."""
    if solver_name == "copt":
        return '''
try:
    import json as _json
    _model_var = None
    for _name in ['model', 'm', 'prob', 'mdl']:
        if _name in dir() or _name in globals():
            _candidate = globals().get(_name) or locals().get(_name)
            if hasattr(_candidate, 'getVars') and hasattr(_candidate, 'getConstrs'):
                _model_var = _candidate
                break
    if _model_var is not None:
        _vars = []
        for _v in _model_var.getVars():
            _vars.append(f"Var: {_v.name}, Type: {_v.type}, LB: {_v.lb}, UB: {_v.ub}, Obj: {_v.obj}")
        _constrs = []
        for _ci, _c in enumerate(_model_var.getConstrs()):
            _sense = getattr(_c, 'sense', '?')
            _rhs = getattr(_c, 'rhs', 0.0)
            _constrs.append(f"Constr: {_c.name}, Sense: {_sense}, RHS: {_rhs}")
        _entries = []
        for _ci, _c in enumerate(_model_var.getConstrs()):
            _row = _model_var.getRow(_c)
            for _i in range(_row.getSize()):
                _entries.append(f"A[{_ci},{_row.getVar(_i).getIdx()}] = {_row.getCoeff(_i)}  (constr={_c.name}, var={_row.getVar(_i).name})")
        _nv = len(_model_var.getVars())
        _nc = len(_model_var.getConstrs())
        _info = {"variables": _vars, "constraints": _constrs, "matrix_entries": _entries, "num_variables": _nv, "num_constraints": _nc}
        print("''' + MODEL_INFO_MARKER + '''" + _json.dumps(_info))
    else:
        print("''' + MODEL_INFO_MARKER + '''{}")
except Exception:
    print("''' + MODEL_INFO_MARKER + '''{}")
'''
    else:
        return '''
try:
    import json as _json
    _model_var = None
    for _name in ['model', 'm', 'prob', 'mdl']:
        if _name in dir() or _name in globals():
            _candidate = globals().get(_name) or locals().get(_name)
            if hasattr(_candidate, 'getVars') and hasattr(_candidate, 'getConstrs'):
                _model_var = _candidate
                break
    if _model_var is not None:
        _vars = []
        for _v in _model_var.getVars():
            _vars.append(f"Var: {_v.VarName}, Type: {_v.VType}, LB: {_v.LB}, UB: {_v.UB}, Obj: {_v.Obj}")
        _constrs = []
        for _c in _model_var.getConstrs():
            _constrs.append(f"Constr: {_c.ConstrName}, Sense: {_c.Sense}, RHS: {_c.RHS}")
        _entries = []
        for _c in _model_var.getConstrs():
            _row = _model_var.getRow(_c)
            for _i in range(_row.size()):
                _entries.append(f"A[{_c.index},{_row.getVar(_i).index}] = {_row.getCoeff(_i)}  (constr={_c.ConstrName}, var={_row.getVar(_i).VarName})")
        _info = {"variables": _vars, "constraints": _constrs, "matrix_entries": _entries, "num_variables": _model_var.NumVars, "num_constraints": _model_var.NumConstrs}
        print("''' + MODEL_INFO_MARKER + '''" + _json.dumps(_info))
    else:
        print("''' + MODEL_INFO_MARKER + '''{}")
except Exception:
    print("''' + MODEL_INFO_MARKER + '''{}")
'''


def _parse_model_info(output_str: str) -> dict:
    """Parse model info from executor output containing MODEL_INFO_MARKER."""
    if not output_str or MODEL_INFO_MARKER not in output_str:
        return {}
    try:
        marker_idx = output_str.rfind(MODEL_INFO_MARKER)
        json_str = output_str[marker_idx + len(MODEL_INFO_MARKER):]
        newline_idx = json_str.find("\n")
        if newline_idx >= 0:
            json_str = json_str[:newline_idx]
        return json.loads(json_str.strip())
    except (json.JSONDecodeError, ValueError):
        return {}


def _normalize_number(value) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(number):
        return 0.0
    return round(number, 6)


def _normalize_sense(value) -> str:
    text = str(value).strip()
    if text in {"<", "<="}:
        return "<="
    if text in {">", ">="}:
        return ">="
    if text in {"=", "=="}:
        return "=="
    return text


def _is_zero_val(value, tol=ZERO_ABS_TOL) -> bool:
    try:
        return abs(float(value)) <= tol
    except (TypeError, ValueError):
        return False


def _ca_parse_variables(model_info: dict) -> list:
    rows = []
    for raw in (model_info or {}).get("variables", []):
        match = VARIABLE_PATTERN.search(str(raw))
        if not match:
            continue
        row = match.groupdict()
        row["name"] = row["name"].strip()
        row["type"] = row["type"].strip()
        row["lb"] = _normalize_number(row["lb"])
        row["ub"] = _normalize_number(row["ub"])
        row["obj"] = _normalize_number(row["obj"])
        rows.append(row)
    return rows


def _ca_parse_constraints(model_info: dict) -> list:
    rows = []
    for raw in (model_info or {}).get("constraints", []):
        match = CONSTRAINT_PATTERN.search(str(raw))
        if not match:
            continue
        row = match.groupdict()
        row["name"] = row["name"].strip()
        row["sense"] = _normalize_sense(row["sense"])
        row["rhs"] = _normalize_number(row["rhs"])
        rows.append(row)
    return rows


def _ca_parse_matrix_entries(model_info: dict) -> list:
    rows = []
    for raw in (model_info or {}).get("matrix_entries", []):
        match = MATRIX_ENTRY_PATTERN.search(str(raw))
        if not match:
            continue
        row = match.groupdict()
        row["variable"] = row["variable"].strip()
        row["constraint"] = row["constraint"].strip()
        row["weight"] = _normalize_number(row["weight"])
        rows.append(row)
    return rows


def _ca_sample_weighted_complexity(model_info: dict) -> float:
    """Weighted complexity = num_variables + num_constraints - zero_rhs_eq_count."""
    if not model_info:
        return 0.0
    num_vars = float(model_info.get("num_variables", 0) or 0)
    num_constrs = float(model_info.get("num_constraints", 0) or 0)
    zero_eq_count = sum(
        1 for c in _ca_parse_constraints(model_info)
        if c["sense"] == "==" and _is_zero_val(c["rhs"])
    )
    return num_vars + num_constrs - zero_eq_count


def _ca_variable_node_signature(variable: dict) -> tuple:
    var_type = str(variable["type"])
    return ("var", variable["obj"], variable["lb"], variable["ub"],
            var_type in {"I", "B"}, var_type == "B")


def _ca_constraint_node_signature(constraint: dict) -> tuple:
    return ("constr", constraint["rhs"], constraint["sense"])


def _ca_graph_encoding(model_info: dict) -> dict:
    variables = _ca_parse_variables(model_info)
    constraints = _ca_parse_constraints(model_info)
    matrix_entries = _ca_parse_matrix_entries(model_info)
    variable_nodes = {v["name"]: _ca_variable_node_signature(v) for v in variables}
    constraint_nodes = {c["name"]: _ca_constraint_node_signature(c) for c in constraints}
    edge_sigs = [
        ("edge", constraint_nodes.get(e["constraint"]),
         variable_nodes.get(e["variable"]), e["weight"])
        for e in matrix_entries
        if e["constraint"] in constraint_nodes and e["variable"] in variable_nodes
    ]
    return {"variable_nodes": variable_nodes, "constraint_nodes": constraint_nodes,
            "edge_signatures": sorted(edge_sigs, key=repr)}


def _ca_graph_variable_signatures(model_info: dict) -> set:
    return set(_ca_graph_encoding(model_info)["variable_nodes"].values())


def _ca_graph_edge_signatures(model_info: dict) -> set:
    return set(_ca_graph_encoding(model_info)["edge_signatures"])


def _ca_graph_nonzero_rhs_signatures(model_info: dict) -> set:
    return {
        sig for sig in _ca_graph_encoding(model_info)["constraint_nodes"].values()
        if len(sig) >= 3 and not _is_zero_val(sig[1])
    }


def _ca_consensus_signatures(model_infos: list, extractor, min_count: int) -> set:
    counter: Counter = Counter()
    for mi in model_infos:
        if mi:
            counter.update(extractor(mi))
    return {sig for sig, count in counter.items() if count >= min_count}


def _ca_jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    if not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _ca_alignment_score(model_info: dict, consensus: dict,
                        weights: tuple = COMPLEXITY_ALIGNED_CHANNEL_WEIGHTS) -> float:
    var_score = _ca_jaccard(_ca_graph_variable_signatures(model_info), consensus["variables"])
    edge_score = _ca_jaccard(_ca_graph_edge_signatures(model_info), consensus["edges"])
    rhs_score = _ca_jaccard(_ca_graph_nonzero_rhs_signatures(model_info), consensus["nonzero_rhs"])
    w_var, w_edge, w_rhs = weights
    return w_var * var_score + w_edge * edge_score + w_rhs * rhs_score


def _ca_sample_weight(model_info: dict, consensus: dict, r_max: float,
                      beta: float, alignment_floor: float) -> float:
    r = _ca_sample_weighted_complexity(model_info)
    alignment = max(_ca_alignment_score(model_info, consensus), alignment_floor)
    return math.exp((r - r_max) / beta) * alignment


def _count_based_complexity_vote(
    grouped: dict, majority_key: float, beta: float, override_margin: float,
    alpha: float = 0.6, top_k: int = 5,
) -> tuple:
    """Hybrid complexity vote using count-based model_info (NumVars/NumConstrs).
    Combines majority vote score with top-k complexity vote score.
    alpha=0.6 balances majority (60%) with complexity signal (40%)."""
    all_candidates = []
    for answer, samples in grouped.items():
        for s in samples:
            mi = s["model_info"]
            nv = float(mi.get("num_variables") or mi.get("NumVars") or 0)
            nc = float(mi.get("num_constraints") or mi.get("NumConstrs") or 0)
            all_candidates.append((nv + nc, answer))

    if not all_candidates:
        total_valid = sum(len(s) for s in grouped.values())
        ratio = len(grouped[majority_key]) / total_valid if total_valid > 0 else 0.0
        return str(majority_key), ratio

    vote_counts = Counter(v for _, v in all_candidates)
    max_count = max(vote_counts.values())
    sorted_by_complexity = sorted(all_candidates, reverse=True)
    top = sorted_by_complexity[:top_k]
    top_votes = Counter(v for _, v in top)
    max_top = max(top_votes.values()) if top_votes else 1

    scores = {}
    for ans in set(vote_counts.keys()):
        maj = vote_counts[ans] / max_count
        comp = top_votes.get(ans, 0) / max_top if max_top > 0 else 0
        scores[ans] = alpha * maj + (1 - alpha) * comp

    selected = max(scores, key=scores.get)
    total_valid = sum(len(s) for s in grouped.values())
    ratio = len(grouped[selected]) / total_valid if total_valid > 0 else 0.0
    return str(selected), ratio


def _zero_filtered_majority_vote(objectives: list, exec_results: list) -> tuple:
    """Zero-filtered majority: filter zeros + infeasible, then majority vote.
    Replicates zero_filtered_majority_strategy from ttrl_analysis."""
    assert len(objectives) == len(exec_results)
    valid_answers = []
    for obj, status in zip(objectives, exec_results):
        if status != "Done":
            continue
        val = _safe_to_float(obj)
        if val is None:
            continue
        if abs(val) <= ZERO_ABS_TOL:
            continue
        valid_answers.append(round(val, 4))

    if not valid_answers:
        return "None", 0.0

    counter = Counter(valid_answers)
    best_answer, best_count = counter.most_common(1)[0]
    ratio = best_count / len(valid_answers)
    return str(best_answer), ratio


def _complexity_aligned_vote(
    objectives: list, exec_results: list, model_infos: list,
    beta: float = COMPLEXITY_ALIGNED_BETA,
    override_margin: float = COMPLEXITY_ALIGNED_OVERRIDE_MARGIN,
    consensus_min_count: int = COMPLEXITY_ALIGNED_CONSENSUS_MIN_COUNT,
    alignment_floor: float = COMPLEXITY_ALIGNED_ALIGNMENT_FLOOR,
) -> tuple:
    """Complexity-aligned voting: exponential weighting by complexity * alignment.
    Replicates complexity_aligned_strategy from ttrl_analysis.
    Falls back to zero_filtered_majority when model_infos unavailable."""
    assert len(objectives) == len(exec_results) == len(model_infos)

    grouped: dict = defaultdict(list)
    for i, (obj, status, mi) in enumerate(zip(objectives, exec_results, model_infos)):
        if status != "Done":
            continue
        val = _safe_to_float(obj)
        if val is None or abs(val) <= ZERO_ABS_TOL:
            continue
        key = round(val, 4)
        grouped[key].append({"idx": i, "model_info": mi or {}})

    if not grouped:
        return "None", 0.0

    majority_key = max(grouped.keys(), key=lambda k: (len(grouped[k]), -k))
    all_model_infos = [s["model_info"] for samples in grouped.values() for s in samples]
    has_structural_info = any(mi.get("num_variables") and mi.get("variables") for mi in all_model_infos)
    has_count_info = any(
        mi.get("num_variables") or mi.get("NumVars") for mi in all_model_infos
    )

    if not has_structural_info:
        if has_count_info:
            return _count_based_complexity_vote(
                grouped, majority_key, beta, override_margin)
        return _zero_filtered_majority_vote(objectives, exec_results)

    consensus = {
        "variables": _ca_consensus_signatures(
            all_model_infos, _ca_graph_variable_signatures, consensus_min_count),
        "edges": _ca_consensus_signatures(
            all_model_infos, _ca_graph_edge_signatures, consensus_min_count),
        "nonzero_rhs": _ca_consensus_signatures(
            all_model_infos, _ca_graph_nonzero_rhs_signatures, consensus_min_count),
    }
    r_max = max(_ca_sample_weighted_complexity(mi) for mi in all_model_infos)

    group_scores: dict = {}
    for answer, samples in grouped.items():
        total_weight = 0.0
        for s in samples:
            total_weight += _ca_sample_weight(
                s["model_info"], consensus, r_max, beta, alignment_floor)
        group_scores[answer] = total_weight
    # weigh aggregate
    top_answer = max(group_scores, key=lambda k: group_scores[k])
    top_score = group_scores[top_answer]
    majority_score = group_scores.get(majority_key, 0.0)

    if abs(top_answer - majority_key) < 1e-4:
        selected = majority_key
    elif majority_score > 0 and top_score / majority_score >= override_margin:
        selected = top_answer
    else:
        selected = majority_key

    total_valid = sum(len(s) for s in grouped.values())
    ratio = len(grouped[selected]) / total_valid if total_valid > 0 else 0.0
    return str(selected), ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []
    answer_correct = []
    answer_only_correct = []
    code_success = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(str(data_item.non_tensor_batch["reward_model"]["majority_gt"]))
        gt_label.append(str(data_item.non_tensor_batch["reward_model"]["original_gt"]))
        answer_correct.append(
            _answer_reward_like_compute_score(
                data_item.non_tensor_batch["extra_info"]["solved_objective"],
                data_item.non_tensor_batch["reward_model"]["original_gt"],
                data_item.non_tensor_batch["extra_info"]["code_exec_res"],
            )
        )
        answer_only_correct.append(
            _answer_only_correct(
                data_item.non_tensor_batch["extra_info"]["solved_objective"],
                data_item.non_tensor_batch["reward_model"]["original_gt"],
            )
        )
        code_success.append(data_item.non_tensor_batch["extra_info"]["code_exec_res"] == "Done")

    ttrl_metrics = _batch_compute_ttrl_metrics(
        majority_reward,
        gt_reward,
        majority_label,
        gt_label,
        answer_correct,
        answer_only_correct,
        code_success,
        n=n,
    )
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    answer_correct: List[bool],
    answer_only_correct: List[bool],
    code_success: List[bool],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(answer_correct) == len(majority_reward)
    assert len(answer_only_correct) == len(majority_reward)
    assert len(code_success) == len(majority_reward)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]
        prompt_answer_correct = answer_correct[i * n:(i + 1) * n]
        prompt_answer_only_correct = answer_only_correct[i * n:(i + 1) * n]
        prompt_code_success = code_success[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(
            prompt_majority_reward,
            prompt_gt_reward,
            prompt_majority_label,
            prompt_gt_label,
            prompt_answer_correct,
            prompt_answer_only_correct,
            prompt_code_success,
        )
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    answer_correct: List[bool],
    answer_only_correct: List[bool],
    code_success: List[bool],
    ):    
    assert len(majority_reward) == len(gt_reward)
    assert len(answer_correct) == len(majority_reward)
    assert len(answer_only_correct) == len(majority_reward)
    assert len(code_success) == len(majority_reward)

    hit_rate = 1.0 if _answer_reward_like_compute_score(majority_label, gt_label) else 0.0
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    sample_answer_accuracy = sum(answer_correct) / len(answer_correct)
    pass_rate = sum(answer_only_correct) / len(answer_only_correct)
    sample_code_pass_rate = sum(code_success) / len(code_success)
    answer_pass_k = 1.0 if any(answer_correct) else 0.0
    reward_pass_k = 1.0 if sum(gt_reward) >= 1 else 0.0
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        "sample_answer_accuracy": sample_answer_accuracy,
        "pass_rate": pass_rate,
        "sample_code_pass_rate": sample_code_pass_rate,
        f"pass@{len(majority_reward)}": answer_pass_k,
        f"reward_pass@{len(majority_reward)}": reward_pass_k,
        f"actual_pass@{len(answer_correct)}": answer_pass_k,
    }
    return ttrl_metric


def compute_multi_strategy_metrics(batch, n):
    """Compute label_accuracy and reward_accuracy for multiple voting strategies.
    Logs all strategies to wandb for real-time comparison.
    """
    assert len(batch) % n == 0
    num_prompts = len(batch) // n

    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    strategies = ["majority", "zero_filtered_majority", "complexity_aligned", "golden_answer"]
    per_strategy = {s: {"label_hits": [], "reward_hits": []} for s in strategies}

    for i in range(num_prompts):
        prompt_objectives = []
        prompt_exec_results = []
        prompt_model_infos = []
        prompt_majority_members = []
        stored_majority_gt = None
        original_gt = None

        for j in range(n):
            data_item = batch[idx[i * n + j]]
            extra_info = data_item.non_tensor_batch["extra_info"]
            reward_model_info = data_item.non_tensor_batch["reward_model"]
            prompt_objectives.append(extra_info["solved_objective"])
            prompt_exec_results.append(extra_info["code_exec_res"])
            prompt_majority_members.append(
                extra_info.get("majority_cluster_member")
            )
            mi = extra_info.get("model_infos", {})
            prompt_model_infos.append(mi if isinstance(mi, dict) else {})
            if stored_majority_gt is None:
                stored_majority_gt = reward_model_info.get("majority_gt")
            if original_gt is None:
                original_gt = str(reward_model_info["original_gt"])

        has_majority_membership = all(
            member is not None for member in prompt_majority_members
        )

        for strategy in strategies:
            if strategy == "majority":
                voted_gt = (
                    str(stored_majority_gt)
                    if stored_majority_gt is not None
                    else _majority_vote(prompt_objectives)[0]
                )
            elif strategy == "zero_filtered_majority":
                voted_gt, _ = _zero_filtered_majority_vote(prompt_objectives, prompt_exec_results)
            elif strategy == "complexity_aligned":
                has_mi = any(mi.get("num_variables") for mi in prompt_model_infos)
                if has_mi:
                    voted_gt, _ = _complexity_aligned_vote(
                        prompt_objectives, prompt_exec_results, prompt_model_infos)
                else:
                    voted_gt, _ = _zero_filtered_majority_vote(
                        prompt_objectives, prompt_exec_results)
            elif strategy == "golden_answer":
                voted_gt, _ = _golden_answer_vote(prompt_objectives, prompt_exec_results, original_gt)
            else:
                voted_gt, _ = _majority_vote(prompt_objectives)

            label_hit = 1.0 if _answer_reward_like_compute_score(voted_gt, original_gt) else 0.0
            per_strategy[strategy]["label_hits"].append(label_hit)

            reward_match = 0
            for j in range(n):
                if strategy == "majority" and has_majority_membership:
                    r_voted = float(
                        bool(prompt_majority_members[j])
                        and _solver_status_done(prompt_exec_results[j])
                    )
                else:
                    r_voted = 1.0 if _answer_reward_like_compute_score(
                        prompt_objectives[j],
                        voted_gt,
                        prompt_exec_results[j],
                    ) else 0.0
                r_original = 1.0 if _answer_reward_like_compute_score(
                    prompt_objectives[j], original_gt, prompt_exec_results[j]) else 0.0
                if r_voted == r_original:
                    reward_match += 1
            per_strategy[strategy]["reward_hits"].append(reward_match / n)

    metrics = {}
    for strategy in strategies:
        hits = per_strategy[strategy]
        if hits["label_hits"]:
            metrics[f"voting/{strategy}/label_accuracy"] = sum(hits["label_hits"]) / len(hits["label_hits"])
            metrics[f"voting/{strategy}/reward_accuracy"] = sum(hits["reward_hits"]) / len(hits["reward_hits"])
    return metrics


def get_solver_feedback(
    solution_strs: list[str],
    extract_model_info: bool = False,
    solver_name: str = "gurobi",
    extract_lp: bool = False,
):
    """
    given solution strs, get feedback and store in batches
    input: solution strs
    code execution has three results:
    obj_result: number
    solution: ...
    code_excu_result: exec code
    return: obj_result, solution, code_excu_result, model_infos, lp_contents
    """
    executor = PythonExecutor()
    introspection_snippet = _build_model_introspection_snippet(solver_name) if extract_model_info else ""

    def append_feedback_and_lp(code: str, lp_path: str) -> str:
        if not code:
            return code
        from verl.or_utils.content_utils import insert_lp_write

        code = insert_lp_write(code, lp_path)
        if extract_model_info:
            code = code + introspection_snippet
        return code

    from verl.or_utils.content_utils import extract_code_block_step
    extracted_codes = [
        extract_code_block_step(solution_str, solver_name)
        for solution_str in solution_strs
    ]
    if extract_lp:
        with tempfile.TemporaryDirectory(prefix="verl_solver_info_") as temporary_dir:
            lp_paths = [
                str(Path(temporary_dir) / f"candidate_{i:06d}.lp")
                for i in range(len(solution_strs))
            ]
            response = executor.batch_apply([
                append_feedback_and_lp(extracted_codes[i], lp_paths[i])
                for i in range(len(solution_strs))
            ])
            lp_contents = []
            for lp_path in lp_paths:
                try:
                    lp_contents.append(Path(lp_path).read_text(encoding="utf-8"))
                except (OSError, UnicodeError):
                    lp_contents.append("")
    else:
        response = executor.batch_apply([
            code + introspection_snippet if code and extract_model_info else code
            for code in extracted_codes
        ])
        lp_contents = ["" for _ in solution_strs]

    obj_result =[response[0][i] for i in range(len(solution_strs))]
    sol_result = [response[1][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]

    model_infos = []
    if extract_model_info:
        for i in range(len(solution_strs)):
            stdout_str = sol_result[i] if sol_result[i] else ""
            model_infos.append(_parse_model_info(stdout_str))
    else:
        model_infos = [{} for _ in range(len(solution_strs))]

    return obj_result, sol_result, code_excu_result, model_infos, lp_contents


def select_reference_lp_from_feedback(
    batch,
    n: int,
    selection_target: str = "voted_answer",
) -> list[str]:
    """Select the prompt reference LP from already-executed rollout feedback.

    selection_target:
        "voted_answer": select candidates matching reward_model.majority_gt.
        "golden_answer": select candidates matching reward_model.original_gt.

    Returns:
        One selected LP string per prompt, or an empty string if no usable
        matching rollout exists.
    """
    num_prompts = len(batch)
    reference_lp_contents = []

    for i in range(num_prompts):
        data_item = batch[i]
        objectives = data_item.non_tensor_batch["extra_info"]["solved_objective"]
        exec_results = data_item.non_tensor_batch["extra_info"]["code_exec_res"]
        candidate_lps = data_item.non_tensor_batch["extra_info"].get(
            "candidate_lp_formulations",
            [],
        )
        if not (
            len(objectives) == len(exec_results) == len(candidate_lps) == n
        ):
            raise ValueError(
                f"prompt {i} feedback lengths must all equal n={n}, got "
                f"objectives={len(objectives)}, exec_results={len(exec_results)}, "
                f"candidate_lps={len(candidate_lps)}"
            )
        reward_model_info = data_item.non_tensor_batch["reward_model"]
        target_answer = (
            reward_model_info.get("original_gt")
            if selection_target == "golden_answer"
            else reward_model_info.get("majority_gt")
        )

        winner_idx = None
        has_stored_majority_reference = False
        if selection_target == "voted_answer":
            extra_info = data_item.non_tensor_batch["extra_info"]
            has_stored_majority_reference = (
                "majority_reference_index" in extra_info
            )
            stored_winner = extra_info.get("majority_reference_index")
            if isinstance(stored_winner, (int, np.integer)):
                stored_winner = int(stored_winner)
                if (
                    0 <= stored_winner < n
                    and _solver_status_done(exec_results[stored_winner])
                    and candidate_lps[stored_winner]
                ):
                    winner_idx = stored_winner
            if has_stored_majority_reference and winner_idx is None:
                reference_lp_contents.append("")
                continue
        for j, (obj, status, candidate_lp) in enumerate(
            zip(objectives, exec_results, candidate_lps, strict=True)
        ):
            if winner_idx is not None:
                break
            if (
                _solver_status_done(status)
                and candidate_lp
                and _safe_to_float(obj) is not None
            ):
                pred = _safe_to_float(obj)
                gt = _safe_to_float(target_answer)
                if gt is not None and abs(pred - gt) / (abs(gt) + 1.0) < 1e-6:
                    winner_idx = j
                    break

        if winner_idx is None:
            reference_lp_contents.append("")
            continue

        reference_lp_contents.append(candidate_lps[winner_idx])

    return reference_lp_contents



# Tag patterns for stage boundary detection
_GUROBI_CLOSE_TAGS = ["</plan>", "</model>", "</python>"]
_STEP_OPEN_TAG = "<step>"
_STEP_CLOSE_TAG = "</step>"
_COPT_CLOSE_TAG = _STEP_CLOSE_TAG
_GUROBI_MAX_STAGES = 3
_COPT_MAX_STAGES = 9
_STEP_MAX_STAGES = 9
_STEP_TITLES = (
    "Problem Description",
    "Sets and Parameters",
    "Decision Variables",
    "Objective Function",
    "Constraints",
    "Mathematical Model",
    "Nonlinear Relationships",
    "Final Model",
    "Python Code",
)


def _find_token_subsequence(token_ids: list[int], pattern_ids: list[int]) -> list[int]:
    """Find all start positions where pattern_ids appears in token_ids."""
    positions = []
    plen = len(pattern_ids)
    if plen == 0:
        return positions
    for i in range(len(token_ids) - plen + 1):
        if token_ids[i:i + plen] == pattern_ids:
            positions.append(i + plen - 1)
    return positions


def _close_tag_token_patterns(tokenizer, tag: str) -> list[list[int]]:
    """Tokenize a close tag in common contexts.

    BPE tokenizers may merge the final ">" with a following newline or merge
    a leading space with "</". Matching only tokenizer.encode(tag) misses those
    valid close tags.
    """
    variants = [
        tag,
        f"{tag}\n",
        f"{tag}\n\n",
        f"\n{tag}",
        f"\n{tag}\n",
        f" {tag}",
        f" {tag}\n",
    ]
    patterns = []
    seen = set()
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        key = tuple(ids)
        if ids and key not in seen:
            patterns.append(ids)
            seen.add(key)
    return patterns


def _find_close_tag_positions(token_ids: list[int], patterns: list[list[int]]) -> list[int]:
    positions = []
    for pattern in patterns:
        positions.extend(_find_token_subsequence(token_ids, pattern))
    return sorted(set(positions))


def _find_tag_start_positions(token_ids: list[int], patterns: list[list[int]]) -> list[int]:
    """Find tag starts and merge overlapping context-tokenization variants."""

    spans = []
    for pattern in patterns:
        for end in _find_token_subsequence(token_ids, pattern):
            spans.append((end - len(pattern) + 1, end))
    if not spans:
        return []

    spans.sort()
    starts = []
    group_starts = [spans[0][0]]
    group_end = spans[0][1]
    for start, end in spans[1:]:
        if start <= group_end:
            group_starts.append(start)
            group_end = max(group_end, end)
            continue
        starts.append(max(group_starts))
        group_starts = [start]
        group_end = end
    starts.append(max(group_starts))
    return starts


def _markdown_heading_token_patterns(tokenizer, title: str) -> list[list[int]]:
    """Tokenize a StepORLM Markdown heading in common BPE contexts.

    The deployed StepORLM/COPT prompt asks for ``<step>`` tags, but the model
    reliably emits the nine canonical ``##`` headings while sometimes
    producing missing or duplicated tags.  Heading prefixes therefore provide
    the stable boundary signal.  Prefix matching intentionally accepts
    ``## Python Code Using `coptpy``` and optional trailing colons.
    """

    headings = [f"## {title}", f"### {title}", f"## ### {title}"]
    variants = []
    for heading in headings:
        variants.extend([
            heading,
            f"{heading}:",
            f"{heading}\n",
            f"{heading}:\n",
            f"\n{heading}",
            f"\n{heading}:",
            f"\n{heading}\n",
            f"\n{heading}:\n",
        ])
    patterns = []
    seen = set()
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        key = tuple(ids)
        if ids and key not in seen:
            patterns.append(ids)
            seen.add(key)
    return patterns


def _find_ordered_markdown_heading_starts(
    token_ids: list[int],
    patterns_by_title: list[list[list[int]]],
) -> list[int]:
    """Return one increasing token start for every canonical heading."""

    ordered_starts = []
    previous = -1
    for patterns in patterns_by_title:
        starts = _find_tag_start_positions(token_ids, patterns)
        start = next((position for position in starts if position > previous), None)
        if start is None:
            return []
        ordered_starts.append(start)
        previous = start
    return ordered_starts


def _has_ordered_markdown_step_headings(text: str) -> bool:
    """Check the canonical nine headings while allowing later extra headings."""

    headings = re.findall(r"(?m)^\s*#{2,6}\s+(.+?)\s*$", text)
    normalized = [
        re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip()
        for heading in headings
    ]
    cursor = 0
    for title in _STEP_TITLES:
        expected = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        matched = False
        while cursor < len(normalized):
            candidate = normalized[cursor]
            cursor += 1
            if candidate == expected or (
                expected == "python code" and candidate.startswith("python code ")
            ):
                matched = True
                break
        if not matched:
            return False
    return True


def _set_step_boundaries_from_open_tags(
    boundaries: "torch.Tensor",
    batch_index: int,
    open_positions: list[int],
    response_length: int,
) -> None:
    """Partition stages at the next ``<step>`` tag, not at title close tags.

    This supports both ``<step>Title\nbody</step>`` and the common StepORLM
    form ``<step>Title</step>\nbody``. Padding after the final stage is masked
    later by the response mask.
    """

    stage_count = min(len(open_positions), boundaries.shape[1])
    for stage_index in range(stage_count):
        if stage_index + 1 < len(open_positions):
            boundaries[batch_index, stage_index] = open_positions[stage_index + 1] - 1
        else:
            boundaries[batch_index, stage_index] = response_length - 1


def compute_stage_boundaries(
    response_ids: "torch.Tensor",
    tokenizer,
    style: str = "gurobi",
) -> "torch.Tensor":
    """Find end-of-stage token positions in response_ids.

    Args:
        response_ids: (batch_size, response_length) token IDs
        tokenizer: tokenizer with encode method
        style: "gurobi" or "copt". New gurobi and copt prompts use
            9-step StepORLM-style tags; legacy gurobi uses 3 stages.

    Returns:
        (batch_size, max_stages) tensor. Values are token positions
        marking the end of each stage. -1 = no stage at that index.
    """
    batch_size, resp_len = response_ids.shape
    step_open_patterns = _close_tag_token_patterns(tokenizer, _STEP_OPEN_TAG)
    markdown_patterns = [
        _markdown_heading_token_patterns(tokenizer, title)
        for title in _STEP_TITLES
    ]
    if style == "copt":
        close_tags = [_COPT_CLOSE_TAG] * _COPT_MAX_STAGES
        max_stages = _COPT_MAX_STAGES
    else:
        step_patterns = _close_tag_token_patterns(tokenizer, _STEP_CLOSE_TAG)
        legacy_tag_patterns = [
            _close_tag_token_patterns(tokenizer, tag)
            for tag in _GUROBI_CLOSE_TAGS
        ]
        boundaries = torch.full(
            (batch_size, _STEP_MAX_STAGES), -1, dtype=torch.long
        )

        any_step_tags = False
        for b in range(batch_size):
            row = response_ids[b].tolist()
            heading_positions = _find_ordered_markdown_heading_starts(
                row,
                markdown_patterns,
            )
            open_positions = _find_tag_start_positions(row, step_open_patterns)
            if heading_positions:
                any_step_tags = True
                _set_step_boundaries_from_open_tags(
                    boundaries,
                    b,
                    heading_positions,
                    resp_len,
                )
            elif open_positions:
                any_step_tags = True
                _set_step_boundaries_from_open_tags(
                    boundaries,
                    b,
                    open_positions,
                    resp_len,
                )
            else:
                step_positions = _find_close_tag_positions(row, step_patterns)
                if step_positions:
                    any_step_tags = True
                    for s, pos in enumerate(step_positions[:_STEP_MAX_STAGES]):
                        boundaries[b, s] = pos
                else:
                    for s, patterns in enumerate(legacy_tag_patterns):
                        legacy_positions = _find_close_tag_positions(row, patterns)
                        if legacy_positions:
                            boundaries[b, s] = legacy_positions[0]

        if any_step_tags:
            return boundaries

        close_tags = _GUROBI_CLOSE_TAGS
        tag_token_patterns = legacy_tag_patterns
        max_stages = _GUROBI_MAX_STAGES

    # Tokenize close tags (without special tokens)
    if style == "copt":
        tag_token_patterns = []
        for tag in close_tags:
            tag_token_patterns.append(_close_tag_token_patterns(tokenizer, tag))

    boundaries = torch.full(
        (batch_size, max_stages), -1, dtype=torch.long
    )

    for b in range(batch_size):
        row = response_ids[b].tolist()
        if style == "copt":
            heading_positions = _find_ordered_markdown_heading_starts(
                row,
                markdown_patterns,
            )
            open_positions = _find_tag_start_positions(row, step_open_patterns)
            if heading_positions:
                _set_step_boundaries_from_open_tags(
                    boundaries,
                    b,
                    heading_positions,
                    resp_len,
                )
            elif open_positions:
                _set_step_boundaries_from_open_tags(
                    boundaries,
                    b,
                    open_positions,
                    resp_len,
                )
            else:
                positions = _find_close_tag_positions(row, tag_token_patterns[0])
                for s, pos in enumerate(positions[:max_stages]):
                    boundaries[b, s] = pos
        else:
            for s, patterns in enumerate(tag_token_patterns):
                positions = _find_close_tag_positions(row, patterns)
                if positions:
                    boundaries[b, s] = positions[0]

    return boundaries


def build_stage_weight_mask(
    stage_boundaries: "torch.Tensor",
    response_length: int,
    decay: str = "linear",
    gamma: float = 2.0,
    random_mask_prob: float = 0.5,
    generator: "torch.Generator | None" = None,
) -> "torch.Tensor":
    """Build per-token KL weight mask based on stage boundaries.

    Args:
        stage_boundaries: (batch_size, max_stages) end-of-stage positions (-1 = absent)
        response_length: length of response dimension
        decay: "linear" | "exponential" | "uniform" | "random_flipped"
            | "random_stage"
            - linear: weight = (stage_idx + 1) / num_stages (increasing)
            - exponential: weight = gamma^(stage_idx / (num_stages - 1)) (exponentially increasing)
            - uniform: weight = 1.0 (identical for every stage)
            - random_flipped: neutral here; dynamic flipping uses build_stage_id_mask
            - random_stage: independently set each detected stage to weight 0
              with probability ``random_mask_prob``; otherwise weight 1
        gamma: base for exponential decay (default 2.0)
        random_mask_prob: probability of masking a detected stage in
            ``random_stage`` mode
        generator: optional CPU torch generator for reproducible random masks

    Returns:
        (batch_size, response_length) float tensor of per-token KL weights.
        Samples with no detected stages get uniform weight 1.0.
    """
    batch_size, max_stages = stage_boundaries.shape
    mask = torch.ones(batch_size, response_length, dtype=torch.float32)
    if decay == "random_stage" and not 0.0 <= random_mask_prob <= 1.0:
        raise ValueError(
            "random_mask_prob must be in [0, 1], "
            f"got {random_mask_prob}"
        )

    for b in range(batch_size):
        bounds = stage_boundaries[b]
        valid = bounds[bounds >= 0]
        num_stages = len(valid)
        if num_stages == 0:
            continue

        if decay == "uniform":
            weights = [1.0] * num_stages
        elif decay == "exponential":
            weights = [gamma ** (i / (num_stages - 1)) if num_stages > 1 else 1.0 for i in range(num_stages)]
        elif decay == "random_flipped":
            # Dynamic random flipping is computed after per-token KL is available.
            # This static path keeps the weight neutral if called directly.
            weights = [1.0] * num_stages
        elif decay == "random_stage":
            weights = (
                torch.rand(num_stages, generator=generator)
                .ge(random_mask_prob)
                .to(torch.float32)
                .tolist()
            )
        elif decay == "problem_info":
            weights = [1.0] * num_stages
            weights[0] = 0 ##problem discription no changes:
        else:  # linear
            weights = [(i + 1) / num_stages for i in range(num_stages)]
        # print(f"[DEBUG] Decay mode activates:{decay}")
        prev_end = 0
        for s in range(num_stages):
            end = int(valid[s].item()) + 1
            end = min(end, response_length)
            mask[b, prev_end:end] = weights[s]
            prev_end = end
        if prev_end < response_length:
            mask[b, prev_end:] = weights[-1]

    return mask


def compute_random_stage_mask_metrics(
    stage_boundaries: "torch.Tensor",
    kl_stage_weights: "torch.Tensor",
    response_mask: "torch.Tensor",
) -> dict[str, float]:
    """Report aggregate and per-stage activation for ``random_stage`` masks."""

    expected_shape = tuple(response_mask.shape)
    if tuple(kl_stage_weights.shape) != expected_shape:
        raise ValueError(
            "KL stage weights and response mask must have identical shapes, "
            f"got weights={tuple(kl_stage_weights.shape)}, mask={expected_shape}"
        )
    if stage_boundaries.shape[0] != response_mask.shape[0]:
        raise ValueError(
            "stage boundaries and response mask must have the same batch size, "
            f"got boundaries={stage_boundaries.shape[0]}, "
            f"mask={response_mask.shape[0]}"
        )

    valid_response_mask = response_mask.bool()
    valid_token_count = max(float(valid_response_mask.sum().item()), 1.0)
    enabled_token_count = float(
        ((kl_stage_weights > 0) & valid_response_mask).sum().item()
    )
    metrics = {
        "bootstrap_kl/random_stage_enabled_token_fraction": (
            enabled_token_count / valid_token_count
        ),
        "bootstrap_kl/random_stage_masked_token_fraction": (
            1.0 - enabled_token_count / valid_token_count
        ),
    }

    response_length = response_mask.shape[1]
    for stage_index in range(stage_boundaries.shape[1]):
        present_count = 0
        masked_count = 0
        for batch_index in range(stage_boundaries.shape[0]):
            end_boundary = int(
                stage_boundaries[batch_index, stage_index].item()
            )
            if end_boundary < 0:
                continue
            start = (
                0
                if stage_index == 0
                else int(
                    stage_boundaries[batch_index, stage_index - 1].item()
                )
                + 1
            )
            start = min(max(start, 0), response_length)
            end = min(max(end_boundary + 1, start), response_length)
            stage_valid_mask = valid_response_mask[batch_index, start:end]
            if not stage_valid_mask.any():
                continue
            present_count += 1
            stage_weights = kl_stage_weights[batch_index, start:end][
                stage_valid_mask
            ]
            if bool((stage_weights == 0).all().item()):
                masked_count += 1

        step = stage_index + 1
        metrics[f"bootstrap_kl/random_stage_step_{step}_present_samples"] = (
            float(present_count)
        )
        metrics[f"bootstrap_kl/random_stage_step_{step}_masked_fraction"] = (
            masked_count / present_count if present_count else 0.0
        )

    return metrics


def build_solver_info_weight_mask(
    stage_boundaries: "torch.Tensor",
    response_length: int,
    candidate_lp_contents: list[str | None],
    reference_lp_contents: list[str | None],
    response_texts: list[str] | None = None,
    code_exec_results: list[str | None] | None = None,
) -> tuple["torch.Tensor", list[Any]]:
    """Build independent Step 3/4/5/9 masks from solver information.

    Anonymous variable bounds/coefficient-matrix differences enable Step 3,
    objective differences enable Step 4, normalized constraint-row differences
    enable Step 5, and complete-LP or execution failures enable Step 9.
    Missing/invalid candidate artifacts therefore activate only Step 9.
    Samples lacking a usable reference or valid stage boundaries receive no
    solver-info signal. Exact masking requires all nine ordered boundaries.
    """
    from verl.or_utils.lp_step_error import SolverInfoLocalization, localize_solver_info

    batch_size = stage_boundaries.shape[0]
    if len(candidate_lp_contents) != batch_size or len(reference_lp_contents) != batch_size:
        raise ValueError(
            "candidate/reference LP counts must match stage-boundary batch size, "
            f"got {len(candidate_lp_contents)=}, {len(reference_lp_contents)=}, {batch_size=}"
        )
    if response_texts is not None and len(response_texts) != batch_size:
        raise ValueError(
            "response text count must match stage-boundary batch size, "
            f"got {len(response_texts)=}, {batch_size=}"
        )
    if code_exec_results is not None and len(code_exec_results) != batch_size:
        raise ValueError(
            "code execution result count must match stage-boundary batch size, "
            f"got {len(code_exec_results)=}, {batch_size=}"
        )

    def has_canonical_step_sequence(batch_index: int) -> bool:
        if response_texts is None:
            return True
        response_text = response_texts[batch_index]
        if _has_ordered_markdown_step_headings(response_text):
            return True
        from verl.or_utils.content_utils import extract_step_blocks, normalize_step_title

        blocks = extract_step_blocks(response_text)
        if len(blocks) != len(_STEP_TITLES):
            return False
        for expected, block in zip(_STEP_TITLES, blocks, strict=True):
            expected_normalized = normalize_step_title(expected)
            block_normalized = normalize_step_title(block)
            if expected_normalized not in block_normalized:
                return False
        return True

    mask = torch.zeros(
        (batch_size, response_length),
        dtype=torch.float32,
    )
    results = []
    reference_cache = {}
    localization_started = time.perf_counter()
    slowest_seconds = 0.0
    slowest_index = -1
    for batch_index, (candidate_lp, reference_lp) in enumerate(
        zip(candidate_lp_contents, reference_lp_contents, strict=True)
    ):
        sample_started = time.perf_counter()
        result = localize_solver_info(
            candidate_lp,
            reference_lp,
            reference_cache=reference_cache,
        )
        sample_seconds = time.perf_counter() - sample_started
        if sample_seconds > slowest_seconds:
            slowest_seconds = sample_seconds
            slowest_index = batch_index
        if sample_seconds >= 1.0:
            print(
                "[solver_info] slow localization "
                f"index={batch_index} seconds={sample_seconds:.3f} "
                f"candidate_chars={len(candidate_lp or '')} "
                f"reference_chars={len(reference_lp or '')} "
                f"status={result.status} fallback_reason={result.fallback_reason}",
                flush=True,
            )
        bounds = stage_boundaries[batch_index]
        valid = bounds[bounds >= 0]
        complete_boundaries = (
            valid.numel() == _STEP_MAX_STAGES
            and bool(torch.all(valid[1:] > valid[:-1]).item())
            and has_canonical_step_sequence(batch_index)
        )
        if result.status == "localized" and not complete_boundaries:
            result = SolverInfoLocalization(
                status="fallback",
                reported_step=None,
                difference_count=result.difference_count,
                category_counts=result.category_counts,
                fallback_reason="invalid_stage_boundaries",
            )

        if result.status == "localized":
            error_steps = set(result.error_steps)
            if code_exec_results is not None:
                code_status = code_exec_results[batch_index]
                if not isinstance(code_status, str) or code_status.strip().lower() != "done":
                    error_steps.add(9)

            invalid_steps = error_steps.difference({3, 4, 5, 9})
            if invalid_steps:
                result = SolverInfoLocalization(
                    status="fallback",
                    reported_step=None,
                    difference_count=result.difference_count,
                    category_counts=result.category_counts,
                    fallback_reason="invalid_reported_step",
                )
            else:
                ordered_error_steps = tuple(sorted(error_steps))
                result = SolverInfoLocalization(
                    status=result.status,
                    reported_step=result.reported_step,
                    difference_count=result.difference_count,
                    category_counts=result.category_counts,
                    fallback_reason=result.fallback_reason,
                    error_steps=ordered_error_steps,
                )
                for step in ordered_error_steps:
                    start = 0 if step == 1 else int(valid[step - 2].item()) + 1
                    end = int(valid[step - 1].item()) + 1
                    start = min(max(start, 0), response_length)
                    end = min(max(end, start), response_length)
                    mask[batch_index, start:end].fill_(1.0)
        results.append(result)

    localization_seconds = time.perf_counter() - localization_started
    print(
        "[solver_info] localization summary "
        f"samples={batch_size} seconds={localization_seconds:.3f} "
        f"reference_cache_entries={len(reference_cache)} "
        f"slowest_index={slowest_index} slowest_seconds={slowest_seconds:.3f}",
        flush=True,
    )

    return mask, results


def compute_solver_info_mask_metrics(
    stage_boundaries: "torch.Tensor",
    solver_info_results: list[Any],
    kl_stage_weights: "torch.Tensor",
    response_mask: "torch.Tensor",
) -> dict[str, float]:
    """Report solver-info mask rates by source and by enabled token count.

    ``mask_*_fraction`` and ``mask_*_sample_fraction`` are sample-level
    activation rates. ``mask_*_token_fraction`` uses all valid response tokens
    in the batch as its denominator. Fallback samples carry no KL signal.
    """

    batch_size = len(solver_info_results)
    expected_shape = tuple(response_mask.shape)
    if stage_boundaries.shape[0] != batch_size:
        raise ValueError(
            "solver-info result count must match stage-boundary batch size, "
            f"got {batch_size=}, stage_batch_size={stage_boundaries.shape[0]}"
        )
    if tuple(kl_stage_weights.shape) != expected_shape:
        raise ValueError(
            "KL stage weights and response mask must have identical shapes, "
            f"got weights={tuple(kl_stage_weights.shape)}, mask={expected_shape}"
        )
    if response_mask.shape[0] != batch_size:
        raise ValueError(
            "solver-info result count must match response-mask batch size, "
            f"got {batch_size=}, mask_batch_size={response_mask.shape[0]}"
        )

    sample_denominator = max(batch_size, 1)
    token_denominator = max(float(response_mask.sum().item()), 1.0)
    metrics: dict[str, float] = {}

    for step in (3, 4, 5, 9):
        sample_indices = [
            index
            for index, result in enumerate(solver_info_results)
            if result.status == "localized" and step in result.error_steps
        ]
        enabled_tokens = 0.0
        for index in sample_indices:
            bounds = stage_boundaries[index]
            start = 0 if step == 1 else int(bounds[step - 2].item()) + 1
            end = int(bounds[step - 1].item()) + 1
            start = min(max(start, 0), response_mask.shape[1])
            end = min(max(end, start), response_mask.shape[1])
            enabled_tokens += float(
                (
                    (kl_stage_weights[index, start:end] > 0)
                    * response_mask[index, start:end].bool()
                ).sum().item()
            )

        sample_fraction = len(sample_indices) / sample_denominator
        metrics[f"solver_info/mask_step_{step}_fraction"] = sample_fraction
        metrics[f"solver_info/mask_step_{step}_sample_fraction"] = sample_fraction
        metrics[f"solver_info/mask_step_{step}_token_fraction"] = (
            enabled_tokens / token_denominator
        )

    fallback_indices = [
        index
        for index, result in enumerate(solver_info_results)
        if result.status == "fallback"
    ]
    fallback_enabled_tokens = sum(
        float(
            (
                (kl_stage_weights[index] > 0)
                * response_mask[index].bool()
            ).sum().item()
        )
        for index in fallback_indices
    )
    fallback_fraction = len(fallback_indices) / sample_denominator
    metrics["solver_info/fallback_no_signal_fraction"] = fallback_fraction
    metrics["solver_info/fallback_no_signal_sample_fraction"] = fallback_fraction
    metrics["solver_info/fallback_enabled_token_fraction"] = (
        fallback_enabled_tokens / token_denominator
    )
    return metrics


def build_stage_id_mask(
    stage_boundaries: "torch.Tensor",
    response_length: int,
) -> "torch.Tensor":
    """Build per-token stage ids from stage boundaries.

    Returns:
        (batch_size, response_length) long tensor. Stage ids are 0-based.
        Samples with no detected stages get -1, so dynamic stage logic can
        leave them unmodified.
    """
    batch_size, _ = stage_boundaries.shape
    stage_ids = torch.full((batch_size, response_length), -1, dtype=torch.long)

    for b in range(batch_size):
        bounds = stage_boundaries[b]
        valid = bounds[bounds >= 0]
        num_stages = len(valid)
        if num_stages == 0:
            continue

        prev_end = 0
        for s in range(num_stages):
            end = int(valid[s].item()) + 1
            end = min(end, response_length)
            stage_ids[b, prev_end:end] = s
            prev_end = end
        if prev_end < response_length:
            stage_ids[b, prev_end:] = num_stages - 1

    return stage_ids
