import re
import subprocess
import textwrap
# ---------------------------------------
# Code-block extraction helpers
# ---------------------------------------

def strip_markdown_fence(code: str) -> str:
    """Remove a wrapping Markdown code fence from an extracted code string."""
    if code is None:
        return None

    cleaned = str(code).strip()
    fence = re.fullmatch(
        r'```[ \t]*(?:python|py)?[^\n]*\n?(.*?)\n?```[ \t]*',
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if fence:
        return fence.group(1).strip()
    return cleaned


def insert_print(code: str, solver_name: str) -> str:
    code = strip_markdown_fence(code)
    if code is None:
        return None
    # Detect the model variable dynamically.
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)  # Existing indentation
        model_name = model_match.group(2)  # Model variable name
        optimize_call = model_match.group(3)  # Solver invocation
        # Select the invocation and status API for the requested solver.
        if solver_name == "gurobi":
            pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
            status_check = (
                f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        elif solver_name == "copt":
            pattern = r'^(\s*)(' + model_name + r'\.solve\(\))'
            status_check = (
                f"{indent}if {model_name}.status == COPT.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        # Preserve the original indentation during replacement.
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
    return code

def insert_lp_generation(code: str, output_name: str, solver_name: str = "gurobi") -> str:
    code = strip_markdown_fence(code)
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    try:
        code = str(code)
    except:
        return None
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)
        model_name = model_match.group(2)
        optimize_call = model_match.group(3)
        pattern = r'^(\s*)(' + model_name + r'\.' + optimize_call + r'\(\))'
        if solver_name == "copt":
            optimal_const = "COPT.OPTIMAL"
        else:
            optimal_const = "GRB.OPTIMAL"
        status_check = (
            f"{indent}{model_name}.write('{output_name}')\n"
            f"{indent}if {model_name}.status == {optimal_const}:\n"
            f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
            f"{indent}else:\n"
            f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
        )
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
    return code


def insert_lp_write(code: str, output_name: str) -> str:
    """Write an LP beside the existing solve call without changing stdout.

    The solver-feedback path already injects objective/solution printing before
    this helper runs. Keeping LP export write-only lets one execution produce
    both the normal feedback and the LP used by solver-info localization.
    """
    code = strip_markdown_fence(code)
    if code is None:
        return None
    model_match = re.search(r"^(\s*)(\w+)\.(optimize|solve)\(\)", str(code), re.M)
    if model_match is None:
        return code
    indent, model_name, solve_method = model_match.groups()
    pattern = rf"^(\s*)({re.escape(model_name)}\.{solve_method}\(\))"
    lp_write = (
        f"{indent}try:\n"
        f"{indent}    {model_name}.write({output_name!r})\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass"
    )
    # Reuse the same output path after every solve call for this model. If a
    # rollout resolves after modifying the model, the last write is the final
    # formulation instead of a stale first snapshot.
    return re.sub(pattern, rf"\1\2\n{lp_write}", str(code), flags=re.M)

def extract_code_block(llm_output: str,solver_name) -> str:
    """
    Extract code between a fenced ``python`` block using DOTALL matching.
    Return an empty value when no block is found.
    """
    if llm_output is None:
        return None

    pattern = r'<python>(.*?)</python>'
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        code = strip_markdown_fence(match.group(1))
        if '```' in code:  # Handle a nested code fence.
            code = _extract_python_fence(code) or strip_markdown_fence(code)
        code = insert_print(code, solver_name)
        return code
    # The response may omit the explicit Python tag.
    code = _extract_python_fence(llm_output)
    if code:
        code = insert_print(code, solver_name)
        return code
    code = _extract_raw_solver_code(llm_output, solver_name)
    if code:
        return insert_print(code, solver_name)
    return None

def extract_block(llm_output,part_name):
    # Extract the requested structured section.
    pattern = rf'<{part_name}>(.*?)</{part_name}>'
    block = None
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        block = match.group(1).strip()
    return block

def extract_obj(str_log):
    """Extract objective value from log string.
    Supports multiple output formats:
    - 'Just print the best obj: VALUE'
    - 'obj: VALUE' / 'obj:VALUE'
    - 'Optimal objective: VALUE'
    """
    if not str_log:
        return None
    patterns = [
        r'Just print the best obj:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)',
        r'(?:^|\n)\s*obj:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)',
        r'Optimal objective[:\s]+(-?\d+\.?\d*(?:e[+-]?\d+)?)',
    ]
    for pat in patterns:
        match = re.search(pat, str_log, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    return None

def extract_sol(str_log):
    """Extract objective value from log string"""
    if 'Just print the best sol:' in str_log:
        sol_match = re.search(r'Just print the best sol:\s*\[([-\d.,\s]*)\]', str_log)
        best_sol = [float(x) for x in sol_match.group(1).split(',') if x.strip()] if sol_match else None
        if best_sol:
            best_sol.sort()
            return best_sol
        else:
            print(str_log)
            return [None]
    return [None]

def extract_integer_binary(str_log):
    """Extract objective value from log string"""
    return 'Integer Variables Exists' in str_log or 'Binary Variables Exists' in str_log
import re

def enforce_integer_variables(code):
    """
    Insert ``vtype=GRB.INTEGER`` into Gurobi ``addVar``/``addVars``
    calls while allowing arbitrary existing arguments.
    """
    # Capture the assignment, arbitrary arguments, and closing parenthesis.
    pattern = r'(\w+\s*=\s*\w+\.addVar[s]?)\(([\s\S]*?)(\)\n)'
    
    def replacer(match):
        var_assignment = match.group(1)  # For example, "x = m.addVar".
        params = match.group(2).rstrip()  # Existing arguments.
        closing = match.group(3)  # Closing parenthesis and newline.
        
        # Leave calls with an explicit variable type unchanged.
        if re.search(r'\bvtype\s*=', params):
            return match.group(0)
        
        # Add the type argument.
        if params:
            # Add a comma after the existing final argument when needed.
            if not params.endswith(','):
                params += ','
            new_params = f"{params} vtype=GRB.INTEGER"
        else:
            # A call without arguments can receive the type directly.
            new_params = "vtype=GRB.INTEGER"
        
        # Preserve the original closing parenthesis and newline.
        return f"{var_assignment}({new_params}{closing}"
    
    # Apply the transformation to every matching call.
    return re.sub(pattern, replacer, code, flags=re.MULTILINE)

def change_variable_types(str_log):
    # Toggle an explicitly declared INTEGER or CONTINUOUS type.
    if "Vtype" in str_log or "vtype" in str_log:
        if 'INTEGER' in str_log:
            return str_log.replace('INTEGER', 'CONTINUOUS')
        elif 'CONTINUOUS' in str_log:
            return str_log.replace('CONTINUOUS', 'INTEGER')
    # An omitted type is continuous by default; make it integer.
    else:
        return enforce_integer_variables(str_log)


STEP_ORLM_STEPS = [
    "Problem Description", "Sets and Parameters", "Decision Variables",
    "Objective Function", "Constraints", "Mathematical Model",
    "Nonlinear Relationships", "Final Model", "Python Code Using coptpy"
]

OR_STEP_TITLES = [
    "Problem Description",
    "Sets and Parameters",
    "Decision Variables",
    "Objective Function",
    "Constraints",
    "Mathematical Model",
    "Nonlinear Relationships",
    "Final Model",
]

OR_CODE_STEP_TITLES = {
    "copt": "Python Code Using coptpy",
    "gurobi": "Python Code Using gurobipy",
}


def normalize_step_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def extract_step_blocks(llm_output: str) -> list:
    """Extract content from <step>...</step> tags for StepORLM format.
    StepORLM expects exactly 9 steps. Falls back to STEP_N: pattern.
    Returns a list of step content strings.
    """
    pattern = r'<\s*step\s*>(.*?)<\s*/\s*step\s*>'
    matches = re.findall(pattern, llm_output or "", re.DOTALL | re.IGNORECASE)
    if matches:
        return [m.strip() for m in matches]
    fallback_pattern = r'STEP_\d+:\s*(.*?)(?=STEP_\d+:|$)'
    matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
    return [m.strip() for m in matches] if matches else []


def extract_step_block_by_title(llm_output: str, title: str) -> str:
    title_norm = normalize_step_title(title)
    for block in extract_step_blocks(llm_output):
        if title_norm in normalize_step_title(block):
            return block
    return None


def _extract_python_fence(text: str) -> str:
    match = re.search(r'```(?:python|py)\s*(.*?)```', text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return strip_markdown_fence(match.group(1))

    match = re.search(r'```\s*(.*?)```', text or "", re.DOTALL)
    if match:
        code = strip_markdown_fence(match.group(1))
        if _looks_like_python_code(code):
            return code
    return None


def _looks_like_python_code(text: str) -> bool:
    lowered = (text or "").lower()
    return "import " in lowered or "def " in lowered or "gurobi" in lowered or "copt" in lowered


def _looks_like_solver_code(text: str, solver_name: str) -> bool:
    lowered = (text or "").lower()
    solver_markers = {
        "copt": ("coptpy", "copt.", ".solve("),
        "gurobi": ("gurobipy", "grb.", ".optimize("),
    }
    return "import" in lowered or any(marker in lowered for marker in solver_markers.get(solver_name, ()))


def _extract_raw_solver_code(text: str, solver_name: str) -> str:
    if not text:
        return None

    solver_starts = {
        "gurobi": [r"import\s+gurobipy\b", r"from\s+gurobipy\s+import\b"],
        "copt": [r"import\s+coptpy\b", r"from\s+coptpy\s+import\b"],
    }
    for pattern in solver_starts.get(solver_name, []):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            code = text[match.start():].strip()
            fence_pos = code.find("```")
            if fence_pos >= 0:
                code = code[:fence_pos].strip()
            return code
    return None


def extract_code_block_from_steps(llm_output: str, solver_name: str = "copt") -> str:
    """Extract solver code using the established OR response priority.

    First use the original extraction path (<python>, global fenced code, raw
    solver imports). If that fails, fall back to the 9-step titled code block
    and finally the last <step> block.
    """
    code = extract_code_block(llm_output, solver_name)
    if code:
        return code

    steps = extract_step_blocks(llm_output)
    if steps:
        code_step = extract_step_block_by_title(llm_output, OR_CODE_STEP_TITLES.get(solver_name, "")) or steps[-1]
        code = _extract_python_fence(code_step)
        if code:
            return insert_print(code, solver_name)
        if _looks_like_solver_code(code_step, solver_name):
            return insert_print(code_step, solver_name)
    return None


def extract_code_block_copt(llm_output: str) -> str:
    """Extract code block for COPT solver from StepORLM output."""
    return extract_code_block_from_steps(llm_output, "copt")


def extract_code_block_step(llm_output: str, solver_name: str = "copt") -> str:
    """Extract code from the 9-step OR format with legacy fallback."""
    return extract_code_block_from_steps(llm_output, solver_name)


def get_step_orlm_system_prompt() -> str:
    """Return the StepORLM system prompt for COPT-based OR problem solving.
    The format matches the COPT inference prompt used by this package.
    """
    return (
        "You are a helpful assistant with expertise in mathematical modeling, "
        "Python code and the COPT solver. When the User provides an optimization "
        "question, you will analyze it, build a detailed mathematical model, "
        "and provide the COPT code to solve it.\n\n"
        "Your response should follow these steps:\n"
        "1. <step>Problem Description</step>\n"
        "2. <step>Sets and Parameters</step>\n"
        "3. <step>Decision Variables</step>\n"
        "4. <step>Objective Function</step>\n"
        "5. <step>Constraints</step>\n"
        "6. <step>Mathematical Model</step>\n"
        "7. <step>Nonlinear Relationships</step>\n"
        "8. <step>Final Model</step>\n"
        "9. <step>Python Code Using coptpy</step>\n\n"
        "The output must be in Markdown format, with each section enclosed "
        "within <step>...</step>.\n"
        "In the <step>Python Code Using coptpy</step> section, put the "
        "complete code in a ```python fenced code block.\n\n"
        "**COPT Code Requirements:**\n"
        '* Make sure to import necessary packages, such as "import coptpy as cp" '
        'and "from coptpy import COPT".\n'
        '* When you create a model, make sure to use "env = cp.Envr()" '
        'and "model = env.createModel(name)".\n'
        '* When you add a variable, use "vtype=COPT.CONTINUOUS", '
        '"vtype=COPT.INTEGER", or "vtype=COPT.BINARY".\n'
        "* Do not name variables and constraints.\n"
        '* Use "model.addConstr()" or "model.addConstrs()" to add constraints.\n'
        '* If you want to set "lb" or "ub" as infinity, please use '
        '"lb=COPT.INFINITY" or "ub=COPT.INFINITY".\n'
        '* When you set objective, you should use the "model.setObjective" '
        'method and use "COPT.MINIMIZE" or "COPT.MAXIMIZE".\n'
        '* Do not use "model.optimize()".\n'
        '* Make sure to use "model.solve()" to solve the question.\n'
        "* The code output statement is:\n"
        "if model.status == COPT.OPTIMAL:\n"
        "    solution = {var.getName(): var.X for var in model.getVars()}\n"
        '    print("Just print the best obj:", model.ObjVal)\n'
        "else:\n"
        '    print("No Solution")\n'
    )
