"""Simple LP fingerprint comparison for solver-informed KL masking.

The input is assumed to be regular linear LP text emitted by Gurobi/COPT.  A
variable name is used only while parsing one LP; names and row labels never
participate in candidate/reference matching.  Each variable is represented by
its bounds, normalized objective coefficient, and its coefficient in every
constraint (zero when it does not participate).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
# Gurobi emits both ``x_1`` and user-provided ``x[1]`` names, while COPT
# commonly emits ``x(1)``/``flow(1,2)``.  Index suffixes remain part of one LP
# variable token; their contents are deliberately not interpreted.
# Do not use a recursive-looking ``(?:plain+|nested)*`` expression here.  A
# generated, malformed variable name with an unmatched delimiter can make the
# stdlib regex engine explore exponentially many partitions.  LP variable
# suffixes contain no whitespace or operators, so consuming their contents as
# one flat token is sufficient and also supports COPT names such as
# ``y((0,_1),(2,_1))``.
NAME_SUFFIX_CONTENT = r"[^\s+*<>=:]*"
NAME = (
    r"[A-Za-z_][A-Za-z0-9_.$]*"
    rf"(?:(?:\({NAME_SUFFIX_CONTENT}\))|(?:\[{NAME_SUFFIX_CONTENT}\]))*"
)
TERM_RE = re.compile(rf"(?P<sign>[+-]?)\s*(?:(?P<coef>{NUMBER})\s+)?(?P<name>{NAME})")
RELATION_RE = re.compile(r"(<=|>=|=)")

# Solver-info is an auxiliary KL signal.  An adversarial/generated LP must not
# be allowed to hold the whole training driver indefinitely.  The current
# fingerprint is dense in variables x constraints, so both the source size and
# the number of dense coefficient cells need explicit budgets.  Inputs over a
# budget are treated as an observable Step-9 artifact failure.
MAX_LP_CHARS = 4 * 1024 * 1024
MAX_LP_LINES = 100_000
MAX_DENSE_COEFFICIENT_CELLS = 2_000_000

SECTION_ALIASES = {
    "minimize": "objective_min",
    "minimum": "objective_min",
    "min": "objective_min",
    "maximize": "objective_max",
    "maximum": "objective_max",
    "max": "objective_max",
    "subject to": "constraints",
    "such that": "constraints",
    "s.t.": "constraints",
    "st": "constraints",
    "bounds": "bounds",
    "binary": "binaries",
    "binaries": "binaries",
    "bin": "binaries",
    "general": "generals",
    "generals": "generals",
    "gen": "generals",
    "integer": "generals",
    "integers": "generals",
    "end": "end",
}
UNSUPPORTED_SECTIONS = {
    "semi-continuous",
    "semicontinuous",
    "semi-integer",
    "semiinteger",
    "sos",
    "general constraints",
    "indicator constraints",
    "lazy constraints",
    "user cuts",
}

CATEGORY_STEP = {
    "variable": 3,
    "bound": 3,
    "objective": 4,
    "constraint": 5,
    "nonlinear": 7,
    "code": 9,
}


class LPParseError(ValueError):
    """Raised when LP text cannot be represented without dropping facts."""


class LPComplexityError(LPParseError):
    """Raised when an LP exceeds the bounded solver-info comparison budget."""


@dataclass
class Variable:
    name: str
    vartype: str = "continuous"
    lb: float = 0.0
    ub: float = math.inf
    objective: float = 0.0


@dataclass
class Constraint:
    name: str
    terms: dict[str, float]
    sense: str
    rhs: float


@dataclass
class LPModel:
    objective_sense: str
    objective_constant: float
    variables: dict[str, Variable]
    constraints: list[Constraint]
    folded_fixed_variables: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverInfoLocalization:
    """Result consumed by the stage-mask builder."""

    status: str
    reported_step: int | None
    difference_count: int
    category_counts: dict[str, int]
    fallback_reason: str | None = None
    error_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class VariableFeatures:
    """Extracted features for one variable; ``name`` is display-only."""

    name: str
    variable_type: str
    bounds: tuple[float, float]
    objective_coefficient: float
    constraint_coefficients: tuple[float, ...]

    @property
    def parameter_signature(self) -> tuple[Any, ...]:
        """Step-3 column signature: bounds plus the LP coefficient matrix."""

        return (
            self.variable_type,
            self.bounds,
            self.constraint_coefficients,
        )

    @property
    def anonymous_signature(self) -> tuple[Any, ...]:
        """Name-free signature used by the simple comparison."""

        return (
            self.variable_type,
            self.bounds,
            self.objective_coefficient,
            self.constraint_coefficients,
        )


@dataclass(frozen=True)
class LPFeatures:
    """Name-free comparison inputs plus display metadata for reports."""

    objective_sense: str
    objective_constant: float
    variables: tuple[VariableFeatures, ...]
    constraint_names: tuple[str, ...]
    constraint_coefficients: tuple[tuple[float, ...], ...]
    constraint_senses: tuple[str, ...]
    constraint_rhs: tuple[float, ...]


@dataclass(frozen=True)
class _SparseVariableFingerprint:
    """Comparison-only bounded dense column fingerprint."""

    variable_type: str
    bounds: tuple[float, float]
    objective_coefficient: float
    constraint_coefficients: tuple[float, ...]

    @property
    def parameter_signature(self) -> tuple[Any, ...]:
        return (
            self.variable_type,
            self.bounds,
            self.constraint_coefficients,
        )

    @property
    def anonymous_signature(self) -> tuple[Any, ...]:
        return (
            self.variable_type,
            self.bounds,
            self.objective_coefficient,
            self.constraint_coefficients,
        )


@dataclass(frozen=True)
class _SparseLPFingerprint:
    """Bounded name-free, record-order-independent training representation."""

    objective_sense: str
    objective_constant: float
    variables: tuple[_SparseVariableFingerprint, ...]
    constraint_coefficients: tuple[tuple[float, ...], ...]
    constraint_senses: tuple[str, ...]
    constraint_rhs: tuple[float, ...]


def _clean_number(raw: str) -> float:
    lowered = raw.strip().lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    return float(raw)


def _parse_expression(
    raw: str,
    *,
    preserve_zero_terms: bool = False,
) -> tuple[dict[str, float], float]:
    text = raw.strip()
    if not text:
        return {}, 0.0

    terms: dict[str, float] = {}
    spans: list[tuple[int, int]] = []
    for match in TERM_RE.finditer(text):
        sign = -1.0 if match.group("sign") == "-" else 1.0
        coefficient = float(match.group("coef")) if match.group("coef") else 1.0
        name = match.group("name")
        terms[name] = terms.get(name, 0.0) + sign * coefficient
        spans.append(match.span())

    remainder_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        remainder_parts.append(text[cursor:start])
        cursor = end
    remainder_parts.append(text[cursor:])
    remainder = re.sub(r"\s+", "", " ".join(remainder_parts).strip())

    constant = 0.0
    if remainder and remainder not in {"+", "-"}:
        numeric_parts = re.findall(rf"[+-]?{NUMBER}", remainder)
        residue = remainder
        for part in numeric_parts:
            constant += float(part)
            residue = residue.replace(part, "", 1)
        if residue not in {"", "+", "-"}:
            raise LPParseError(f"cannot parse expression residue {residue!r}")

    if preserve_zero_terms:
        return terms, constant
    return {name: value for name, value in terms.items() if abs(value) > 1e-12}, constant


def _parse_bound(raw: str, bounds: dict[str, tuple[float | None, float | None]]) -> None:
    line = raw.strip()
    if not line:
        return

    free_match = re.fullmatch(rf"({NAME})\s+free", line, re.IGNORECASE)
    if free_match:
        bounds[free_match.group(1)] = (-math.inf, math.inf)
        return

    ranged = re.fullmatch(
        rf"([+-]?{NUMBER}|[+-]?(?:inf|infinity))\s*<=\s*({NAME})\s*<=\s*"
        rf"([+-]?{NUMBER}|[+-]?(?:inf|infinity))",
        line,
        re.IGNORECASE,
    )
    if ranged:
        bounds[ranged.group(2)] = (_clean_number(ranged.group(1)), _clean_number(ranged.group(3)))
        return

    variable_first = re.fullmatch(
        rf"({NAME})\s*(<=|>=|=)\s*([+-]?{NUMBER}|[+-]?(?:inf|infinity))",
        line,
        re.IGNORECASE,
    )
    if variable_first:
        name, sense, raw_value = variable_first.groups()
        value = _clean_number(raw_value)
        old_lb, old_ub = bounds.get(name, (None, None))
        if sense == "<=":
            bounds[name] = (old_lb, value)
        elif sense == ">=":
            bounds[name] = (value, old_ub)
        else:
            bounds[name] = (value, value)
        return

    lower_first = re.fullmatch(
        rf"([+-]?{NUMBER}|[+-]?(?:inf|infinity))\s*<=\s*({NAME})",
        line,
        re.IGNORECASE,
    )
    if lower_first:
        value, name = lower_first.groups()
        _, old_ub = bounds.get(name, (None, None))
        bounds[name] = (_clean_number(value), old_ub)
        return

    raise LPParseError(f"unsupported bound: {raw!r}")


def _fold_fixed_variables(model: LPModel) -> LPModel:
    fixed = {
        name: var.lb
        for name, var in model.variables.items()
        if math.isfinite(var.lb) and math.isfinite(var.ub) and abs(var.lb - var.ub) <= 1e-12
    }
    if not fixed:
        return model

    objective_constant = model.objective_constant
    for name, value in fixed.items():
        objective_constant += model.variables[name].objective * value

    constraints: list[Constraint] = []
    for row in model.constraints:
        rhs = row.rhs
        terms = dict(row.terms)
        for name, value in fixed.items():
            rhs -= terms.pop(name, 0.0) * value
        constraints.append(Constraint(name=row.name, terms=terms, sense=row.sense, rhs=rhs))

    return LPModel(
        objective_sense=model.objective_sense,
        objective_constant=objective_constant,
        variables={name: var for name, var in model.variables.items() if name not in fixed},
        constraints=constraints,
        folded_fixed_variables=fixed,
    )


def parse_lp_text(text: str, *, fold_fixed_variables: bool = True) -> LPModel:
    """Parse the linear subset of solver-exported LP text."""

    if not isinstance(text, str) or not text.strip():
        raise LPParseError("LP text is empty")
    if len(text) > MAX_LP_CHARS:
        raise LPComplexityError(
            f"LP text has {len(text)} characters, limit is {MAX_LP_CHARS}"
        )
    line_count = text.count("\n") + 1
    if line_count > MAX_LP_LINES:
        raise LPComplexityError(
            f"LP text has {line_count} lines, limit is {MAX_LP_LINES}"
        )

    section: str | None = None
    objective_sense: str | None = None
    objective_lines: list[str] = []
    constraint_lines: list[str] = []
    bound_lines: list[str] = []
    binary_names: list[str] = []
    general_names: list[str] = []

    for original in text.splitlines():
        stripped = original.strip()
        if not stripped or stripped.startswith("\\"):
            continue

        lowered = re.sub(r"\s+", " ", stripped.lower())
        if lowered in UNSUPPORTED_SECTIONS:
            raise LPParseError(f"unsupported LP section {stripped!r}")
        new_section = SECTION_ALIASES.get(lowered)
        if new_section:
            section = new_section
            if section == "objective_min":
                objective_sense = "minimize"
            elif section == "objective_max":
                objective_sense = "maximize"
            elif section == "end":
                break
            continue

        if section in {"objective_min", "objective_max"}:
            objective_lines.append(stripped)
        elif section == "constraints":
            # COPT can emit a fixed objective-constant variable immediately
            # after the last row without a ``Bounds`` header.
            copt_implicit_bound = re.fullmatch(
                rf"{NAME}\s*(?:<=|>=|=)\s*"
                rf"(?:[+-]?{NUMBER}|[+-]?(?:inf|infinity))",
                stripped,
                re.IGNORECASE,
            )
            if ":" not in stripped and copt_implicit_bound:
                bound_lines.append(stripped)
            else:
                constraint_lines.append(stripped)
        elif section == "bounds":
            bound_lines.append(stripped)
        elif section == "binaries":
            binary_names.extend(stripped.split())
        elif section == "generals":
            general_names.extend(stripped.split())
        else:
            raise LPParseError(f"content outside a supported section: {original!r}")

    if objective_sense is None:
        raise LPParseError("LP file has no Minimize/Maximize section")

    objective_raw = " ".join(objective_lines)
    if ":" in objective_raw:
        objective_raw = objective_raw.split(":", 1)[1].strip()
    objective_terms, objective_constant = _parse_expression(
        objective_raw,
        preserve_zero_terms=True,
    )

    assembled_rows: list[str] = []
    current: list[str] = []
    for line in constraint_lines:
        if ":" in line:
            if current:
                assembled_rows.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
        else:
            raise LPParseError(f"constraint continuation has no named row: {line!r}")
    if current:
        assembled_rows.append(" ".join(current))

    constraints: list[Constraint] = []
    for ordinal, raw in enumerate(assembled_rows):
        name, expression = raw.split(":", 1)
        relation = RELATION_RE.search(expression)
        if not relation:
            raise LPParseError(f"constraint has no relation: {raw!r}")
        lhs = expression[: relation.start()]
        rhs_raw = expression[relation.end() :].strip()
        if not re.fullmatch(rf"[+-]?{NUMBER}", rhs_raw):
            raise LPParseError(f"non-numeric RHS in {raw!r}")
        terms, lhs_constant = _parse_expression(lhs)
        constraints.append(
            Constraint(
                name=name.strip() or f"row_{ordinal}",
                terms=terms,
                sense=relation.group(1),
                rhs=float(rhs_raw) - lhs_constant,
            )
        )

    bounds: dict[str, tuple[float | None, float | None]] = {}
    for line in bound_lines:
        _parse_bound(line, bounds)

    variable_order: list[str] = []
    seen: set[str] = set()

    def remember(names: Iterable[str]) -> None:
        for name in names:
            if name not in seen:
                seen.add(name)
                variable_order.append(name)

    remember(objective_terms)
    for row in constraints:
        remember(row.terms)
    remember(bounds)
    remember(binary_names)
    remember(general_names)

    binary_set = set(binary_names)
    general_set = set(general_names)
    variables: dict[str, Variable] = {}
    for name in variable_order:
        lb, ub = bounds.get(name, (None, None))
        vartype = "binary" if name in binary_set else "integer" if name in general_set else "continuous"
        variables[name] = Variable(
            name=name,
            vartype=vartype,
            lb=0.0 if lb is None else lb,
            ub=(1.0 if vartype == "binary" else math.inf) if ub is None else ub,
            objective=objective_terms.get(name, 0.0),
        )

    model = LPModel(
        objective_sense=objective_sense,
        objective_constant=objective_constant,
        variables=variables,
        constraints=constraints,
    )
    return _fold_fixed_variables(model) if fold_fixed_variables else model


def extract_lp_features(model: LPModel) -> LPFeatures:
    """Extract deterministic features independent of names and record order."""

    dense_cells = len(model.variables) * len(model.constraints)
    if dense_cells > MAX_DENSE_COEFFICIENT_CELLS:
        raise LPComplexityError(
            f"dense report extraction requires {dense_cells} coefficient cells, "
            f"limit is {MAX_DENSE_COEFFICIENT_CELLS}"
        )

    variable_names = tuple(model.variables)
    variables = tuple(
        VariableFeatures(
            name=name,
            variable_type=variable.vartype,
            bounds=(variable.lb, variable.ub),
            objective_coefficient=variable.objective,
            constraint_coefficients=tuple(
                sorted(row.terms.get(name, 0.0) for row in model.constraints)
            ),
        )
        for name, variable in model.variables.items()
    )
    constraint_rows = tuple(
        tuple(
            sorted(row.terms.get(name, 0.0) for name in variable_names)
        )
        for row in model.constraints
    )
    return LPFeatures(
        objective_sense=model.objective_sense,
        objective_constant=model.objective_constant,
        variables=variables,
        constraint_names=tuple(row.name for row in model.constraints),
        constraint_coefficients=constraint_rows,
        constraint_senses=tuple(row.sense for row in model.constraints),
        constraint_rhs=tuple(row.rhs for row in model.constraints),
    )


def _extract_sparse_lp_fingerprint(model: LPModel) -> _SparseLPFingerprint:
    """Build a bounded, name-free and record-order-independent fingerprint."""

    dense_cells = len(model.variables) * len(model.constraints)
    if dense_cells > MAX_DENSE_COEFFICIENT_CELLS:
        raise LPComplexityError(
            f"canonical comparison requires {dense_cells} coefficient cells, "
            f"limit is {MAX_DENSE_COEFFICIENT_CELLS}"
        )
    variable_names = tuple(model.variables)
    variables = tuple(
        _SparseVariableFingerprint(
            variable_type=variable.vartype,
            bounds=(variable.lb, variable.ub),
            objective_coefficient=variable.objective,
            constraint_coefficients=tuple(
                sorted(row.terms.get(name, 0.0) for row in model.constraints)
            ),
        )
        for name, variable in model.variables.items()
    )
    constraint_rows = tuple(
        tuple(
            sorted(row.terms.get(name, 0.0) for name in variable_names)
        )
        for row in model.constraints
    )
    return _SparseLPFingerprint(
        objective_sense=model.objective_sense,
        objective_constant=model.objective_constant,
        variables=variables,
        constraint_coefficients=constraint_rows,
        constraint_senses=tuple(row.sense for row in model.constraints),
        constraint_rhs=tuple(row.rhs for row in model.constraints),
    )


def _feature_checks(
    candidate: LPFeatures | _SparseLPFingerprint,
    reference: LPFeatures | _SparseLPFingerprint,
) -> dict[str, bool]:
    """Compare exact multisets; variable and row names are deliberately absent."""

    candidate_rows = Counter(
        zip(
            candidate.constraint_coefficients,
            candidate.constraint_senses,
            candidate.constraint_rhs,
            strict=True,
        )
    )
    reference_rows = Counter(
        zip(
            reference.constraint_coefficients,
            reference.constraint_senses,
            reference.constraint_rhs,
            strict=True,
        )
    )
    checks = {
        "variable_count": len(candidate.variables) == len(reference.variables),
        "variable_type_multiset": Counter(
            variable.variable_type for variable in candidate.variables
        )
        == Counter(variable.variable_type for variable in reference.variables),
        "bounds_multiset": Counter(variable.bounds for variable in candidate.variables)
        == Counter(variable.bounds for variable in reference.variables),
        "constraint_coefficient_vector_multiset": Counter(
            variable.constraint_coefficients for variable in candidate.variables
        )
        == Counter(variable.constraint_coefficients for variable in reference.variables),
        "variable_parameter_signature_multiset": Counter(
            variable.parameter_signature for variable in candidate.variables
        )
        == Counter(variable.parameter_signature for variable in reference.variables),
        "objective_coefficient_multiset": Counter(
            variable.objective_coefficient for variable in candidate.variables
        )
        == Counter(variable.objective_coefficient for variable in reference.variables),
        "objective_sense": candidate.objective_sense == reference.objective_sense,
        "objective_constant": candidate.objective_constant == reference.objective_constant,
        "full_variable_signature_multiset": Counter(
            variable.anonymous_signature for variable in candidate.variables
        )
        == Counter(variable.anonymous_signature for variable in reference.variables),
        "constraint_row_multiset": candidate_rows == reference_rows,
    }
    checks["complete_normalized_lp"] = all(checks.values())
    return checks


def _features_as_dict(features: LPFeatures) -> dict[str, Any]:
    """Return a JSON-friendly representation used by reports and diagnostics."""

    return {
        "objective_sense": features.objective_sense,
        "objective_constant": features.objective_constant,
        "variables": [
            {
                "name": variable.name,
                "variable_type": variable.variable_type,
                "bounds": variable.bounds,
                "objective_coefficient": variable.objective_coefficient,
                "constraint_coefficients": variable.constraint_coefficients,
                "anonymous_signature": variable.anonymous_signature,
            }
            for variable in features.variables
        ],
        "constraints": [
            {
                "index": index,
                "name": name,
                "coefficients": coefficients,
                "sense": sense,
                "rhs": rhs,
            }
            for index, (name, coefficients, sense, rhs) in enumerate(
                zip(
                    features.constraint_names,
                    features.constraint_coefficients,
                    features.constraint_senses,
                    features.constraint_rhs,
                    strict=True,
                ),
                start=1,
            )
        ],
    }


def compare_lp_models(
    candidate: LPModel,
    reference: LPModel,
    *,
    include_feature_details: bool = True,
    _reference_fingerprint: _SparseLPFingerprint | None = None,
) -> dict[str, Any]:
    """Compare simple LP fingerprints and map failed checks to Steps 3/4/5.

    Training passes ``include_feature_details=False`` so comparison remains
    sparse.  Reports retain the previous dense, human-readable extraction.
    """

    candidate_features = _extract_sparse_lp_fingerprint(candidate)
    reference_features = (
        _reference_fingerprint
        if _reference_fingerprint is not None
        else _extract_sparse_lp_fingerprint(reference)
    )
    checks = _feature_checks(candidate_features, reference_features)
    differences: list[dict[str, Any]] = []

    def add(category: str, kind: str) -> None:
        differences.append(
            {
                "id": f"D{len(differences) + 1:03d}",
                "category": category,
                "kind": kind,
                "origin_step": CATEGORY_STEP[category],
            }
        )

    if not checks["variable_parameter_signature_multiset"]:
        add("variable", "variable_parameter_signature_multiset")
    if (
        not checks["objective_sense"]
        or not checks["objective_coefficient_multiset"]
        or not checks["objective_constant"]
    ):
        add("objective", "objective_fingerprint")
    elif (
        checks["variable_parameter_signature_multiset"]
        and not checks["full_variable_signature_multiset"]
    ):
        # The objective values match as a multiset, but they are attached to
        # different anonymous Step-3 parameter columns.
        add("objective", "full_variable_signature_multiset")
    if not checks["constraint_row_multiset"]:
        add("constraint", "constraint_row_multiset")
    if not checks["complete_normalized_lp"]:
        add("code", "complete_normalized_lp")

    category_counts = {
        category: sum(diff["category"] == category for diff in differences)
        for category in CATEGORY_STEP
    }
    candidate_details = (
        _features_as_dict(extract_lp_features(candidate))
        if include_feature_details
        else None
    )
    reference_details = (
        _features_as_dict(extract_lp_features(reference))
        if include_feature_details
        else None
    )
    return {
        "candidate": candidate_details,
        "reference": reference_details,
        "checks": checks,
        "differences": differences,
        "summary": {
            "difference_count": len(differences),
            "category_counts": category_counts,
            "earliest_lp_origin": min(
                (diff["origin_step"] for diff in differences),
                default=None,
            ),
        },
    }


def _format_number(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _format_vector(values: Iterable[float]) -> str:
    return "[" + ", ".join(_format_number(value) for value in values) + "]"


def _render_extraction_markdown(title: str, features: LPFeatures, heading_level: int) -> list[str]:
    heading = "#" * heading_level
    lines = [
        f"{heading} {title}",
        "",
        f"- Original objective sense: `{features.objective_sense}`",
        f"- Objective constant: `{_format_number(features.objective_constant)}`",
        f"- Decision variables: `{len(features.variables)}`",
        f"- Constraints: `{len(features.constraint_rhs)}`",
        "",
        "Variable names and record order are displayed but not compared.",
        "",
        "| Display name | Type | Bounds | Objective coefficient | Sorted constraint-column coefficients |",
        "|---|---:|---:|---:|---|",
    ]
    for variable in features.variables:
        bounds = _format_vector(variable.bounds)
        objective = _format_number(variable.objective_coefficient)
        coefficients = _format_vector(variable.constraint_coefficients)
        lines.append(
            f"| `{variable.name}` | `{variable.variable_type}` | `{bounds}` | "
            f"`{objective}` | `{coefficients}` |"
        )

    lines.extend(
        [
            "",
            "| Constraint | Display label | Sorted row coefficients | Sense | RHS |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for index, (name, coefficients, sense, rhs) in enumerate(
        zip(
            features.constraint_names,
            features.constraint_coefficients,
            features.constraint_senses,
            features.constraint_rhs,
            strict=True,
        ),
        start=1,
    ):
        lines.append(
            f"| C{index} | `{name}` | `{_format_vector(coefficients)}` | "
            f"`{sense}` | `{_format_number(rhs)}` |"
        )
    return lines


def render_lp_comparison_markdown(
    candidate_lp: str,
    reference_lp: str,
    *,
    title: str = "LP extraction and comparison report",
    heading_level: int = 1,
) -> str:
    """Render extracted features and exact comparison checks as Markdown."""

    candidate_model = parse_lp_text(candidate_lp)
    reference_model = parse_lp_text(reference_lp)
    candidate_features = extract_lp_features(candidate_model)
    reference_features = extract_lp_features(reference_model)
    comparison = compare_lp_models(candidate_model, reference_model)
    checks = comparison["checks"]
    heading = "#" * heading_level
    child_level = heading_level + 1
    lines = [
        f"{heading} {title}",
        "",
        (
            "Comparison rule: variable names, constraint labels, variable order, "
            "and constraint record order are ignored; missing coefficients are zero. "
            "Step 3 compares variable types, bounds, and anonymous parameter columns. "
            "Step 4 compares objective sense, coefficients, column attachment, and "
            "the objective constant. Step 5 compares anonymous row coefficients, "
            "senses, and RHS values. Step 9 compares the complete normalized LP. "
            "Fixed variables are folded into the objective constant first."
        ),
        "",
    ]
    lines.extend(_render_extraction_markdown("Candidate extraction", candidate_features, child_level))
    lines.append("")
    lines.extend(_render_extraction_markdown("Reference extraction", reference_features, child_level))
    lines.extend(
        [
            "",
            f"{'#' * child_level} Comparison",
            "",
            "| Step | Check | Equal |",
            "|---:|---|---:|",
            f"| Step 3 | Variable count | {'yes' if checks['variable_count'] else 'no'} |",
            f"| Step 3 | Variable-type multiset | {'yes' if checks['variable_type_multiset'] else 'no'} |",
            f"| Step 3 | Bounds multiset | {'yes' if checks['bounds_multiset'] else 'no'} |",
            "| Step 3 | Anonymous constraint-column multiset | "
            f"{'yes' if checks['constraint_coefficient_vector_multiset'] else 'no'} |",
            "| Step 3 | Type + bounds + parameter-column signature multiset | "
            f"{'yes' if checks['variable_parameter_signature_multiset'] else 'no'} |",
            f"| Step 4 | Objective sense | {'yes' if checks['objective_sense'] else 'no'} |",
            f"| Step 4 | Objective-coefficient multiset | {'yes' if checks['objective_coefficient_multiset'] else 'no'} |",
            f"| Step 4 | Objective constant | {'yes' if checks['objective_constant'] else 'no'} |",
            "| Step 4 | Complete anonymous variable-signature multiset | "
            f"{'yes' if checks['full_variable_signature_multiset'] else 'no'} |",
            "| Step 5 | `(anonymous row, sense, RHS)` multiset | "
            f"{'yes' if checks['constraint_row_multiset'] else 'no'} |",
            "| Step 9 | Complete normalized LP | "
            f"{'yes' if checks['complete_normalized_lp'] else 'no'} |",
            "",
        ]
    )
    error_steps = tuple(
        step
        for step, categories in (
            (3, ("variable", "bound")),
            (4, ("objective",)),
            (5, ("constraint",)),
            (9, ("code",)),
        )
        if any(
            comparison["summary"]["category_counts"].get(category, 0) > 0
            for category in categories
        )
    )
    lines.append(
        f"Final LP error steps: `{error_steps}`. Step 9 may also be activated "
        "by the upstream code-execution status."
    )
    return "\n".join(lines).rstrip() + "\n"


def localize_solver_info(
    candidate_lp: str | None,
    reference_lp: str | None,
    *,
    reference_cache: dict[str, tuple[LPModel, _SparseLPFingerprint]] | None = None,
) -> SolverInfoLocalization:
    """Locate independently observable Step 3/4/5/9 LP differences.

    An equivalent LP is a successful localization with no error steps. A
    missing or invalid candidate is observable as a Step-9 code artifact
    failure. Missing or invalid references still require a no-signal fallback.
    """

    empty_counts = {category: 0 for category in CATEGORY_STEP}
    code_only_counts = dict(empty_counts)
    code_only_counts["code"] = 1
    if not reference_lp:
        return SolverInfoLocalization("fallback", None, 0, empty_counts, "missing_reference_lp")

    try:
        reference_model: LPModel
        reference_fingerprint: _SparseLPFingerprint | None = None
        cached_reference = (
            reference_cache.get(reference_lp)
            if reference_cache is not None
            else None
        )
        if cached_reference is None:
            reference_model = parse_lp_text(reference_lp)
            if reference_cache is not None:
                reference_fingerprint = _extract_sparse_lp_fingerprint(reference_model)
                reference_cache[reference_lp] = (reference_model, reference_fingerprint)
        else:
            reference_model, reference_fingerprint = cached_reference
    except LPComplexityError:
        return SolverInfoLocalization("fallback", None, 0, empty_counts, "complexity_limit")
    except (LPParseError, TypeError, ValueError, ArithmeticError, KeyError, IndexError):
        return SolverInfoLocalization("fallback", None, 0, empty_counts, "parse_error")

    if not candidate_lp:
        return SolverInfoLocalization(
            "localized",
            9,
            1,
            code_only_counts,
            "missing_candidate_lp",
            error_steps=(9,),
        )
    try:
        candidate_model = parse_lp_text(candidate_lp)
        _extract_sparse_lp_fingerprint(candidate_model)
    except (LPComplexityError, LPParseError, TypeError, ValueError, ArithmeticError, KeyError, IndexError) as error:
        fallback_reason = (
            "complexity_limit" if isinstance(error, LPComplexityError) else "parse_error"
        )
        return SolverInfoLocalization(
            "localized",
            9,
            1,
            code_only_counts,
            fallback_reason,
            error_steps=(9,),
        )

    try:
        comparison = compare_lp_models(
            candidate_model,
            reference_model,
            include_feature_details=False,
            _reference_fingerprint=reference_fingerprint,
        )
    except LPComplexityError:
        return SolverInfoLocalization("fallback", None, 0, empty_counts, "complexity_limit")
    except (LPParseError, TypeError, ValueError, ArithmeticError, KeyError, IndexError):
        return SolverInfoLocalization("fallback", None, 0, empty_counts, "parse_error")

    difference_count = comparison["summary"]["difference_count"]
    category_counts = comparison["summary"]["category_counts"]
    reported_step = comparison["summary"]["earliest_lp_origin"]
    error_steps = tuple(
        step
        for step, categories in (
            (3, ("variable", "bound")),
            (4, ("objective",)),
            (5, ("constraint",)),
            (9, ("code",)),
        )
        if any(category_counts.get(category, 0) > 0 for category in categories)
    )
    if reported_step is None:
        return SolverInfoLocalization(
            "localized",
            None,
            difference_count,
            category_counts,
            error_steps=error_steps,
        )
    return SolverInfoLocalization(
        "localized",
        reported_step,
        difference_count,
        category_counts,
        error_steps=error_steps,
    )


def _main() -> None:
    """Generate one Markdown report containing one or more LP pairs."""

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("TITLE", "CANDIDATE_LP", "REFERENCE_LP"),
        help="comparison title followed by candidate and reference LP paths",
    )
    parser.add_argument("--output", type=Path, help="write Markdown to this path")
    args = parser.parse_args()

    sections = ["# LP structural fingerprint report", ""]
    for title, candidate_path, reference_path in args.pair:
        candidate_lp = Path(candidate_path).read_text(encoding="utf-8")
        reference_lp = Path(reference_path).read_text(encoding="utf-8")
        sections.append(
            render_lp_comparison_markdown(
                candidate_lp,
                reference_lp,
                title=title,
                heading_level=2,
            ).rstrip()
        )
        sections.append("")
    report = "\n".join(sections).rstrip() + "\n"

    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    _main()
