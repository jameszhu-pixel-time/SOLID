from typing import Any
from importlib import util
from pathlib import Path


def unwrap_or_question_text(prompt_text: Any) -> Any:
    """Extract the original OR question from the rollout prompt text."""
    if not isinstance(prompt_text, str):
        return prompt_text
    text = prompt_text.strip()
    copt_prefix = (
        "Below is an optimization modeling question. Build a mathematical model and corresponding "
        "python code using coptpy that appropriately addresses the question:"
    )
    copt_backtick_prefix = (
        "Below is an optimization modeling question. Build a mathematical model and corresponding "
        "python code using `coptpy` that appropriately addresses the question:"
    )
    gurobi_prefix = (
        "Below is an optimization modeling question. Build a mathematical model and corresponding "
        "python code using gurobipy that appropriately addresses the question:"
    )
    legacy_gurobi_prefix = "Solve the following mathmetical modeling problem"
    for prefix in (copt_prefix, copt_backtick_prefix, gurobi_prefix, legacy_gurobi_prefix):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    copt_requirements_marker = "* Make sure to import necessary packages"
    if copt_requirements_marker in text:
        text = text.split(copt_requirements_marker, 1)[0].strip()
    for suffix in ("Think step by step.", "think step by step."):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def build_or_teacher_message(
    prompt_text: Any,
    lp_content: str | None,
    majority_gt: Any,
    or_style: str = "gurobi",
    fallback_messages: list[dict] | None = None,
) -> list[dict]:
    """Build the solver-specific reference teacher message used by SDPO."""
    if not lp_content:
        return fallback_messages or [{"role": "user", "content": prompt_text}]

    question_text = unwrap_or_question_text(prompt_text)
    if or_style == "copt":
        prompt_path = Path(__file__).resolve().parents[2] / "or_utils" / "copt_prompt.py"
        module_name = "_or_distill_copt_prompt"
        template_name = "copt_prompt_method2_lp"
    else:
        prompt_path = Path(__file__).resolve().parents[2] / "or_utils" / "gurobi_prompt.py"
        module_name = "_or_distill_gurobi_prompt"
        template_name = "gurobi_prompt_method2_noref"

    spec = util.spec_from_file_location(module_name, prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompt template from {prompt_path}")
    prompt_module = util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    prompt_template = getattr(prompt_module, template_name)
    system_tmpl = prompt_template["system"]
    user_tmpl = prompt_template["user"]

    if or_style == "copt":
        reprompt_text = user_tmpl.format(
            question=question_text,
            reference_obj=str(majority_gt or ""),
            reference_lp=lp_content,
        )
    else:
        reprompt_text = user_tmpl.format(
            question=question_text,
            reference_answer=lp_content,
        )
    return [
        {"role": "system", "content": system_tmpl.strip()},
        {"role": "user", "content": reprompt_text.strip()},
    ]
