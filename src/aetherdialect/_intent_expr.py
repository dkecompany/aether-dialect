"""Expression parsing, intent JSON parsing, and structural parameter handling."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

import jsonschema
import sqlglot
from sqlglot import exp

from ._config import (
    CASE_RESULT_BARE_LABEL_RE,
    CASE_RESULT_REGISTRY_TOKEN_RE,
    CTE_DEFAULT_AGGS,
    CTE_FULL_AGGS,
    CTE_HAVING_COMPARE_OPS,
    CTE_NUMERIC_FILTER_OPS,
    CTE_OUTPUT_ALIAS_RE,
    DATE_INTERVAL_EXPR_SUBSTRINGS,
    DATE_UNIT_ALIAS_TO_CANONICAL,
    DATE_UNIT_KEYWORDS,
    IN_OPS,
    IN_STRING_SEPARATORS,
    INTEGER_SCALARS,
    INTENT_SCHEMA,
    NUMERIC_RESULT_AGGS,
    PLANNER_PROSE_FIELDS,
    NUMERIC_RESULT_SCALARS,
    REGISTRY_TOKEN_PATTERN,
    SCALAR_FUNC_DEFAULTS,
    SCALAR_FUNCTIONS_LEADING_ARG,
    is_structural_param_key,
)
from ._contracts_base import (
    CteOutputColumnMeta,
    LogicalIntent,
    SchemaGraph,
    VirtualColumnSpec,
    VirtualTableSpec,
)
from ._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ExprValue,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    WindowRegistryStep,
    WindowSpec,
    expr_registry_ref,
    register_parse_expr_string,
)
from ._core_utils import debug, normalize_op, safe_json_loads, stable_json


def _is_date_or_interval_expr(s: str) -> bool:
    """
    Return True when *s* looks like a date/interval literal (keep as ``right_expr``).

    Args:

        s: Candidate right-hand string.

    Returns:

        True if substring markers match ``DATE_INTERVAL_EXPR_SUBSTRINGS``.
    """
    if not s or not isinstance(s, str):
        return False
    lower = s.strip().lower()
    return any(p in lower for p in DATE_INTERVAL_EXPR_SUBSTRINGS)


def strip_leading_distinct_from_column_ref(ref: str) -> str:
    """
    Remove a leading SQL DISTINCT prefix from a column reference token.

    Args:

        ref: Raw reference substring, possibly ``DISTINCT tbl.col``.

    Returns:

        Reference without a leading ``DISTINCT`` keyword.
    """
    s = ref.strip()
    upper = s.upper()
    if upper.startswith("DISTINCT "):
        return s[9:].strip()
    return s


def expr_canonical_key(expr: NormalizedExpr) -> str:
    """
    Return a stable string key for the same logical aggregation or scalar expression.

    Used for HAVING dedup and diagnostics where ``primary_column`` is absent on aggregate LHS.
    """

    return stable_json(expr.to_dict())


def extract_columns_from_expr(expr: NormalizedExpr) -> list[str]:
    """
    Collect unique column refs from every nested term in *expr*.

    Walks the structural NormalizedExpr tree (including nested multiply/divide children, CAST inner expressions, etc.) and yields the leaf ``column_ref`` values in first-seen order. For ``raw_sql`` escape-hatch leaves, attempts a secondary sqlglot parse to recover any embedded column references.
    """
    cols: list[str] = []
    seen: set[str] = set()

    def _add(ref: str) -> None:
        if ref and ref not in seen:
            seen.add(ref)
            cols.append(ref)

    def _walk(node: NormalizedExpr) -> None:
        if node.string_literal:
            return
        if node.column_ref:
            _add(node.column_ref)
            return
        if node.raw_sql:
            try:
                to_parse = node.raw_sql.strip()
                if to_parse.upper().startswith("DISTINCT "):
                    to_parse = to_parse[9:].strip()
                tree = sqlglot.parse_one(to_parse, dialect=None)
                if tree is not None:
                    for col in tree.find_all(exp.Column):
                        tbl = col.table
                        name = col.name
                        ref = f"{tbl}.{name}" if tbl else name
                        _add(ref)
            except Exception:
                pass
        for arg in node.scalar_func_args:
            if isinstance(arg, NormalizedExpr):
                _walk(arg)
        for arg in node.inner_scalar_func_args:
            if isinstance(arg, NormalizedExpr):
                _walk(arg)
        for group in node.add_groups + node.sub_groups:
            for child in group.multiply + group.divide:
                _walk(child)
            for arg in group.scalar_func_args:
                if isinstance(arg, NormalizedExpr):
                    _walk(arg)
            for arg in group.inner_scalar_func_args:
                if isinstance(arg, NormalizedExpr):
                    _walk(arg)

    _walk(expr)
    return cols


def _remap_qualified_column_refs_in_raw_sql(fragment: str, replacer: Callable[[str], str]) -> str:
    """
    Apply *replacer* to qualified ``table.column`` tokens inside a raw SQL fragment.

    Used when :func:`replace_refs_in_expr` encounters a ``raw_sql`` leaf so CTE rewrites still reach escape-hatch SQL text.
    """
    if not (fragment and fragment.strip()):
        return fragment
    try:
        to_parse = fragment.strip()
        if to_parse.upper().startswith("DISTINCT "):
            to_parse = to_parse[9:].strip()
        tree = sqlglot.parse_one(to_parse, dialect=None)
    except Exception:
        return fragment
    changed = False
    for col in list(tree.find_all(exp.Column)):
        tbl = col.table
        name = col.name
        if not tbl or not name:
            continue
        ref = f"{tbl}.{name}"
        new_ref = replacer(ref)
        if new_ref == ref or "." not in new_ref:
            continue
        _nt, nn = new_ref.rsplit(".", 1)
        changed = True
        col.replace(exp.column(nn, table=_nt))
    if not changed:
        return fragment
    return tree.sql(dialect=None)


def replace_refs_in_expr(expr: NormalizedExpr, replacer: Callable[[str], str]) -> NormalizedExpr:
    """Map every leaf ``column_ref`` in *expr* through *replacer* and return a fresh tree copy."""

    def _walk(node: NormalizedExpr) -> NormalizedExpr:
        if node.string_literal:
            return node
        if node.column_ref:
            new_ref = replacer(node.column_ref)
            return replace(node, column_ref=new_ref)
        new_add: list[MulGroup] = []
        for g in node.add_groups:
            new_add.append(
                MulGroup(
                    coefficient=g.coefficient,
                    multiply=[_walk(m) for m in g.multiply],
                    divide=[_walk(d) for d in g.divide],
                    agg_func=g.agg_func,
                    scalar_func=g.scalar_func,
                    inner_scalar_func=g.inner_scalar_func,
                    scalar_func_args=list(g.scalar_func_args),
                    inner_scalar_func_args=list(g.inner_scalar_func_args),
                    coeff_param_key=g.coeff_param_key,
                    sarg_param_keys=list(g.sarg_param_keys),
                    isarg_param_keys=list(g.isarg_param_keys),
                    distinct=g.distinct,
                )
            )
        new_sub: list[MulGroup] = []
        for g in node.sub_groups:
            new_sub.append(
                MulGroup(
                    coefficient=g.coefficient,
                    multiply=[_walk(m) for m in g.multiply],
                    divide=[_walk(d) for d in g.divide],
                    agg_func=g.agg_func,
                    scalar_func=g.scalar_func,
                    inner_scalar_func=g.inner_scalar_func,
                    scalar_func_args=list(g.scalar_func_args),
                    inner_scalar_func_args=list(g.inner_scalar_func_args),
                    coeff_param_key=g.coeff_param_key,
                    sarg_param_keys=list(g.sarg_param_keys),
                    isarg_param_keys=list(g.isarg_param_keys),
                    distinct=g.distinct,
                )
            )
        out = replace(node, add_groups=new_add, sub_groups=new_sub)
        if out.raw_sql:
            new_raw = _remap_qualified_column_refs_in_raw_sql(out.raw_sql, replacer)
            if new_raw != out.raw_sql:
                out = replace(out, raw_sql=new_raw)
        return out

    return _walk(expr)


_AST_AGG_NODE_TO_NAME: dict[type[exp.Expression], str] = {
    exp.Sum: "sum",
    exp.Count: "count",
    exp.Avg: "avg",
    exp.Min: "min",
    exp.Max: "max",
}


def _ast_unwrap_paren(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _ast_literal_value(node: exp.Expression) -> int | float | str | None:
    """Coerce a leaf literal/identifier node to its scalar value, or None if not literal-shaped."""
    node = _ast_unwrap_paren(node)
    if isinstance(node, exp.Literal):
        if node.is_int:
            try:
                return int(node.this)
            except (TypeError, ValueError):
                return str(node.this)
        if node.is_number:
            try:
                v = float(node.this)
                return int(v) if v == int(v) else v
            except (TypeError, ValueError):
                return str(node.this)
        return str(node.this)
    if isinstance(node, exp.Var):
        return str(node.name)
    if isinstance(node, exp.Column):
        if not node.table:
            return str(node.name)
    return None


def _ast_render_token(node: exp.Expression) -> str:
    """Render a column/star/identifier node into the legacy ``table.column`` token form."""
    node = _ast_unwrap_paren(node)
    if isinstance(node, exp.Column):
        table = node.table
        col = node.name
        return f"{table}.{col}" if table else col
    if isinstance(node, exp.Star):
        return "*"
    raise ValueError(f"unsupported leaf node for column rendering: {type(node).__name__}")


def _ast_node_to_normalized(node: exp.Expression) -> NormalizedExpr:
    """Convert a sqlglot expression AST into a structural :class:`NormalizedExpr` tree."""
    node = _ast_unwrap_paren(node)

    if isinstance(node, exp.Case):
        return NormalizedExpr(raw_sql=node.sql(dialect=None))

    if isinstance(node, exp.Cast):
        inner = _ast_node_to_normalized(node.this)
        type_name = node.args.get("to").sql(dialect=None).upper() if node.args.get("to") else ""
        wrapper = NormalizedExpr(
            cast_type=type_name,
            add_groups=[MulGroup(multiply=[inner])],
        )
        return wrapper

    if isinstance(node, exp.Interval):
        val_node = node.this
        unit_node = node.args.get("unit")
        try:
            val = float(getattr(val_node, "this", val_node))
        except (TypeError, ValueError):
            val = 0.0
        unit_str = ""
        if unit_node is not None:
            unit_str = getattr(unit_node, "name", None) or unit_node.sql(dialect=None)
        return NormalizedExpr(interval=(val, str(unit_str).lower()))

    if isinstance(node, exp.CurrentDate):
        return NormalizedExpr(keyword="current_date")
    if isinstance(node, exp.CurrentTimestamp):
        return NormalizedExpr(keyword="current_timestamp")

    if isinstance(node, exp.Distinct):
        return NormalizedExpr(raw_sql=node.sql(dialect=None))

    agg_name = _AST_AGG_NODE_TO_NAME.get(type(node))
    if agg_name is not None:
        inner = node.this
        distinct_flag = False
        if isinstance(inner, exp.Distinct):
            distinct_flag = True
            exprs = inner.expressions or [inner.this]
            inner = exprs[0] if exprs else inner.this
        if inner is None:
            child = NormalizedExpr(star=True)
        else:
            child = _ast_node_to_normalized(inner)
        return NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=[child],
                    agg_func=agg_name,
                    distinct=distinct_flag,
                )
            ]
        )

    if isinstance(node, exp.DPipe):
        parts: list[exp.Expression] = []

        def _collect_dpipe(n: exp.Expression) -> None:
            n = _ast_unwrap_paren(n)
            if isinstance(n, exp.DPipe):
                _collect_dpipe(n.this)
                _collect_dpipe(n.expression)
            else:
                parts.append(n)

        _collect_dpipe(node)
        children = [_ast_node_to_normalized(p) for p in parts]
        return NormalizedExpr(
            add_groups=[MulGroup(multiply=children, scalar_func="concat", scalar_func_args=[])],
        )

    if isinstance(node, exp.Anonymous) or (
        isinstance(node, exp.Func) and not isinstance(node, (exp.Mul, exp.Add, exp.Sub, exp.Div, exp.Paren, exp.Neg))
    ):
        if isinstance(node, exp.Anonymous):
            func_name = (node.this or "").lower()
            args = list(node.expressions or [])
        else:
            func_name = (node.key or "").lower()
            collected: list[exp.Expression] = []
            seen_ids: set[int] = set()
            for v in node.args.values():
                if isinstance(v, exp.Expression) and id(v) not in seen_ids:
                    collected.append(v)
                    seen_ids.add(id(v))
                elif isinstance(v, list):
                    for e in v:
                        if isinstance(e, exp.Expression) and id(e) not in seen_ids:
                            collected.append(e)
                            seen_ids.add(id(e))
            args = collected

        if func_name == "concat":
            children = [_ast_node_to_normalized(a) for a in args]
            return NormalizedExpr(
                add_groups=[MulGroup(multiply=children, scalar_func="concat", scalar_func_args=[])],
            )

        if isinstance(node, exp.Extract):
            unit_node = node.this
            source_node = node.expression
            unit = getattr(unit_node, "name", None) or _ast_literal_value(unit_node) or ""
            source = _ast_node_to_normalized(source_node) if source_node is not None else NormalizedExpr(star=True)
            return NormalizedExpr(
                add_groups=[
                    MulGroup(
                        multiply=[source],
                        scalar_func="extract",
                        scalar_func_args=[str(unit).lower()],
                    )
                ]
            )

        if not args:
            return NormalizedExpr(add_groups=[MulGroup(multiply=[NormalizedExpr(star=True)], scalar_func=func_name)])

        expr_args: list[exp.Expression] = []
        trailing_scalars: list[int | float | str] = []
        for a in args:
            lit = _ast_literal_value(a)
            if lit is not None and not expr_args and len(args) > 1:
                trailing_scalars.append(lit)
            elif lit is not None and expr_args:
                trailing_scalars.append(lit)
            else:
                expr_args.append(a)

        if not expr_args:
            return NormalizedExpr(raw_sql=node.sql(dialect=None))

        children = [_ast_node_to_normalized(a) for a in expr_args]
        return NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=children,
                    scalar_func=func_name,
                    scalar_func_args=list(trailing_scalars),
                )
            ]
        )

    if isinstance(node, exp.Column):
        return NormalizedExpr.from_column(_ast_render_token(node))
    if isinstance(node, exp.Star):
        return NormalizedExpr(star=True)
    if isinstance(node, exp.Literal):
        try:
            v = float(node.this)
            return NormalizedExpr(add_values=[ExprValue(value=v)])
        except (TypeError, ValueError):
            return NormalizedExpr(raw_sql=node.sql(dialect=None))

    if isinstance(node, (exp.Add, exp.Sub, exp.Neg)):
        additive = _ast_collect_additive(node)
        return _additive_to_normalized(additive)

    if isinstance(node, (exp.Mul, exp.Div)):
        group = _ast_mul_chain_to_group(node)
        return NormalizedExpr(add_groups=[group])

    return NormalizedExpr(raw_sql=node.sql(dialect=None))


def _ast_collect_additive(node: exp.Expression) -> list[tuple[str, exp.Expression]]:
    node = _ast_unwrap_paren(node)
    if isinstance(node, exp.Add):
        return _ast_collect_additive(node.this) + _ast_collect_additive(node.expression)
    if isinstance(node, exp.Sub):
        left = _ast_collect_additive(node.this)
        right = _ast_collect_additive(node.expression)
        flipped = [("-" if s == "+" else "+", t) for s, t in right]
        return left + flipped
    if isinstance(node, exp.Neg):
        inner = _ast_collect_additive(node.this)
        return [("-" if s == "+" else "+", t) for s, t in inner]
    return [("+", node)]


def _ast_mul_chain_to_group(node: exp.Expression) -> MulGroup:
    """Walk a Mul/Div chain into a single MulGroup with nested NormalizedExpr children."""
    multiply: list[NormalizedExpr] = []
    divide: list[NormalizedExpr] = []
    coefficient = 1.0

    def _consume(n: exp.Expression, dividing: bool) -> None:
        nonlocal coefficient
        n = _ast_unwrap_paren(n)
        if isinstance(n, exp.Mul):
            _consume(n.this, dividing)
            _consume(n.expression, dividing)
            return
        if isinstance(n, exp.Div):
            _consume(n.this, dividing)
            _consume(n.expression, not dividing)
            return
        if isinstance(n, exp.Literal) and n.is_number:
            try:
                v = float(n.this)
            except (TypeError, ValueError):
                v = 1.0
            if dividing:
                if v == 0.0:
                    return
                coefficient /= v
            else:
                coefficient *= v
            return
        child = _ast_node_to_normalized(n)
        (divide if dividing else multiply).append(child)

    _consume(node, False)
    if not multiply:
        multiply = [NormalizedExpr(star=True)]
    return MulGroup(multiply=multiply, divide=divide, coefficient=coefficient)


def _additive_to_normalized(
    additive: list[tuple[str, exp.Expression]],
) -> NormalizedExpr:
    """Build a NormalizedExpr from a signed list of additive terms."""
    additive = [
        (s, t)
        for s, t in additive
        if not (
            isinstance(_ast_unwrap_paren(t), exp.Literal)
            and getattr(_ast_unwrap_paren(t), "is_number", False)
            and float(_ast_unwrap_paren(t).this) == 0.0
        )
    ]
    if not additive:
        return NormalizedExpr(add_values=[ExprValue(value=0.0)])
    add_groups: list[MulGroup] = []
    sub_groups: list[MulGroup] = []
    add_values: list[ExprValue] = []
    sub_values: list[ExprValue] = []
    for sign, raw in additive:
        n = _ast_unwrap_paren(raw)
        if isinstance(n, exp.Literal) and n.is_number:
            try:
                v = float(n.this)
            except (TypeError, ValueError):
                v = 0.0
            (add_values if sign == "+" else sub_values).append(ExprValue(value=abs(v)))
            continue
        sub = _ast_node_to_normalized(n)
        if (
            not sub.column_ref
            and not sub.star
            and not sub.keyword
            and not sub.cast_type
            and not sub.interval
            and not sub.raw_sql
            and len(sub.add_groups) == 1
            and not sub.sub_groups
            and not sub.add_values
            and not sub.sub_values
        ):
            grp = sub.add_groups[0]
            (add_groups if sign == "+" else sub_groups).append(grp)
        else:
            (add_groups if sign == "+" else sub_groups).append(MulGroup(multiply=[sub]))
    return NormalizedExpr(
        add_groups=add_groups,
        sub_groups=sub_groups,
        add_values=add_values,
        sub_values=sub_values,
    )


def _coerce_case_result_string_literal(parsed: NormalizedExpr, raw: str) -> NormalizedExpr:
    """
    Treat a bare unqualified identifier in a CASE result as a string literal.

    The LLM frequently writes CASE branch results like ``"active"`` without quoting, which sqlglot parses as a column reference. When the parsed expression is just a bare ``column_ref`` token (no ``.``, no functions, no registry id pattern) and the raw text matches a simple identifier shape, we emit a quoted string literal via ``raw_sql`` so SQL generation produces the user-intended literal.
    """

    if not raw or "." in raw:
        return parsed
    text = raw.strip()
    if not text or "'" in text or '"' in text:
        return parsed
    if not CASE_RESULT_BARE_LABEL_RE.match(text):
        return parsed
    if CASE_RESULT_REGISTRY_TOKEN_RE.match(text):
        return parsed
    if not parsed.column_ref:
        return parsed
    if (
        parsed.add_groups
        or parsed.sub_groups
        or parsed.add_values
        or parsed.sub_values
        or parsed.agg_func
        or parsed.scalar_func
        or parsed.inner_scalar_func
        or parsed.cast_type
        or parsed.interval is not None
        or parsed.keyword
        or parsed.star
    ):
        return parsed
    escaped = text.replace("'", "''")
    return NormalizedExpr(raw_sql=f"'{escaped}'")


def parse_expr_string(expr_str: str | dict) -> NormalizedExpr:
    """
    Parse LLM SQL text (or ``{"expr": ...}``) into a structural :class:`NormalizedExpr` tree.

    Uses sqlglot as the sole parser – no string-based fallback.
    """
    if isinstance(expr_str, dict):
        raw_str = expr_str.get("expr", "")
    elif isinstance(expr_str, str):
        raw_str = expr_str
    else:
        raw_str = str(expr_str) if expr_str else ""
    raw_str = (raw_str or "").strip()
    if not raw_str:
        return NormalizedExpr()
    try:
        tree = sqlglot.parse_one(raw_str, dialect=None)
    except Exception:
        return NormalizedExpr(raw_sql=raw_str)
    if tree is None:
        return NormalizedExpr(raw_sql=raw_str)
    if isinstance(tree, (exp.Command, exp.Select)):
        return NormalizedExpr(raw_sql=raw_str)
    additive = _ast_collect_additive(tree)
    if len(additive) == 1 and additive[0][0] == "+":
        return _ast_node_to_normalized(additive[0][1])
    return _additive_to_normalized(additive)


def _is_expr_numeric(
    expr: NormalizedExpr,
    schema: SchemaGraph,
    cte_steps: Sequence[RuntimeCteStep] | None = None,
) -> bool:
    """
    Heuristic: agg/scalar/column typing implies a numeric result.

    Args:

        expr: Candidate expression.

        schema: Column ``value_type`` lookup.

        cte_steps: Optional CTE steps consulted when ``expr.primary_column`` references
        a CTE-qualified column whose physical schema entry does not exist.

    Returns:

        True when treated as numeric for tagging.
    """
    if expr.agg_func and expr.agg_func in NUMERIC_RESULT_AGGS:
        return True
    if expr.scalar_func and expr.scalar_func in NUMERIC_RESULT_SCALARS:
        return True
    if expr.inner_scalar_func and expr.inner_scalar_func in NUMERIC_RESULT_SCALARS:
        return True
    for g in expr.add_groups + expr.sub_groups:
        if g.agg_func and g.agg_func in NUMERIC_RESULT_AGGS:
            return True
        if g.scalar_func and g.scalar_func in NUMERIC_RESULT_SCALARS:
            return True
        if g.inner_scalar_func and g.inner_scalar_func in NUMERIC_RESULT_SCALARS:
            return True
    col = expr.primary_column
    if col and "." in col:
        table, col_name = col.rsplit(".", 1)
        if table in schema.tables:
            meta = schema.tables[table].columns.get(col_name) or schema.tables[table].columns.get(col_name.lower())
            if meta and meta.value_type:
                return meta.value_type in ("integer", "number")
        cte_vt = _cte_output_value_type_for_numeric(table, col_name, cte_steps)
        if cte_vt is not None:
            return cte_vt in ("integer", "number")
    return len(expr.add_groups) + len(expr.sub_groups) > 1


def _cte_output_value_type_for_numeric(
    qualifier: str,
    output_alias: str,
    cte_steps: Sequence[RuntimeCteStep] | None,
) -> str | None:
    """Return the value_type for ``cte_name.output_alias`` from CTE output metadata, or None."""
    if not cte_steps or not qualifier:
        return None
    qualifier_lc = qualifier.lower()
    alias_lc = output_alias.lower()
    for cte in cte_steps:
        if (cte.cte_name or "").lower() != qualifier_lc:
            continue
        ocm = cte.output_column_metadata or {}
        meta = ocm.get(output_alias) or ocm.get(alias_lc)
        if meta and (meta.value_type or "").strip():
            return meta.value_type.lower()
        for sc in cte.select_cols or []:
            sc_alias = (sc.output_alias or "").lower()
            if sc_alias and sc_alias != alias_lc:
                continue
            agg = (sc.expr.agg_func or "").lower()
            scalar = (sc.expr.scalar_func or "").lower()
            inner = (sc.expr.inner_scalar_func or "").lower()
            if not agg and sc.expr.add_groups:
                g0 = sc.expr.add_groups[0]
                agg = (g0.agg_func or "").lower()
                if not scalar:
                    scalar = (g0.scalar_func or "").lower()
                if not inner:
                    inner = (g0.inner_scalar_func or "").lower()
                if not agg:
                    for g in sc.expr.add_groups:
                        for term in g.multiply + g.divide:
                            if term.agg_func:
                                agg = (term.agg_func or "").lower()
                                break
                            ts = (term.scalar_func or "").lower()
                            ti = (term.inner_scalar_func or "").lower()
                            if ts or ti:
                                scalar, inner = ts or scalar, ti or inner
                        if agg:
                            break
            if agg in NUMERIC_RESULT_AGGS:
                return "number"
            if scalar in NUMERIC_RESULT_SCALARS or inner in NUMERIC_RESULT_SCALARS:
                return "number"
            if getattr(sc.expr, "is_numeric", False):
                return "number"
        return None
    return None


def _tag_single_expr(
    expr: NormalizedExpr,
    schema: SchemaGraph,
    skip_value_injection: bool = False,
    cte_steps: Sequence[RuntimeCteStep] | None = None,
) -> NormalizedExpr:
    """
    Set ``is_numeric``; inject ``0.0`` offset for numeric groups unless skipped; strip non-numeric junk.

    Args:

        expr: Expression to tag.

        schema: Column metadata for ``_is_expr_numeric``.

        skip_value_injection: If True, no ``ExprValue(0.0)`` (filters/having LHS).

    Returns:

        Updated expression (copy via ``replace``).
    """
    numeric = _is_expr_numeric(expr, schema, cte_steps)
    bare_leaf = not expr.add_groups and not expr.sub_groups and bool(expr.column_ref or expr.star or expr.keyword)
    if numeric:
        if skip_value_injection:
            return replace(expr, is_numeric=True)
        need_offset = bool(expr.add_groups) and not expr.add_values and not bare_leaf
        return replace(
            expr,
            is_numeric=True,
            add_values=[ExprValue(value=0.0)] if need_offset else expr.add_values,
        )
    sanitized_groups = [
        (replace(g, coefficient=1.0, coeff_param_key="") if g.coefficient != 1.0 or g.coeff_param_key else g)
        for g in expr.add_groups
    ]
    sanitized_sub_groups = [
        (replace(g, coefficient=1.0, coeff_param_key="") if g.coefficient != 1.0 or g.coeff_param_key else g)
        for g in expr.sub_groups
    ]
    return replace(
        expr,
        is_numeric=False,
        add_groups=sanitized_groups,
        sub_groups=sanitized_sub_groups,
        add_values=[],
        sub_values=[],
    )


def tag_expr_numeric(intent: RuntimeIntent, schema: SchemaGraph) -> RuntimeIntent:
    """
    Run ``_tag_single_expr`` on every expression slot in *intent* (main + CTEs).

    Args:

        intent: Full intent.

        schema: Schema graph.

    Returns:

        Replaced intent with tagged expressions.
    """
    cte_seq = intent.cte_steps or []
    select_cols = [
        replace(sc, expr=_tag_single_expr(sc.expr, schema, cte_steps=cte_seq)) for sc in (intent.select_cols or [])
    ]
    order_by_cols = [
        replace(obc, expr=_tag_single_expr(obc.expr, schema, cte_steps=cte_seq)) for obc in (intent.order_by_cols or [])
    ]
    group_by_cols = [_tag_single_expr(g, schema, cte_steps=cte_seq) for g in (intent.group_by_cols or [])]
    filters_param = []
    for fp in intent.filters_param or []:
        left = _tag_single_expr(fp.left_expr, schema, skip_value_injection=True, cte_steps=cte_seq)
        right = _tag_single_expr(fp.right_expr, schema, cte_steps=cte_seq) if fp.right_expr else None
        filters_param.append(replace(fp, left_expr=left, right_expr=right))
    having_param = []
    for hp in intent.having_param or []:
        left = _tag_single_expr(hp.left_expr, schema, skip_value_injection=True, cte_steps=cte_seq)
        right = _tag_single_expr(hp.right_expr, schema, cte_steps=cte_seq) if hp.right_expr else None
        having_param.append(replace(hp, left_expr=left, right_expr=right))
    cte_steps = []
    for cte in cte_seq:
        cte_sc = [
            replace(sc, expr=_tag_single_expr(sc.expr, schema, cte_steps=cte_seq)) for sc in (cte.select_cols or [])
        ]
        cte_obc = [
            replace(obc, expr=_tag_single_expr(obc.expr, schema, cte_steps=cte_seq))
            for obc in (cte.order_by_cols or [])
        ]
        cte_gb = [_tag_single_expr(g, schema, cte_steps=cte_seq) for g in (cte.group_by_cols or [])]
        cte_fp = []
        for fp in cte.filters_param or []:
            left = _tag_single_expr(fp.left_expr, schema, skip_value_injection=True, cte_steps=cte_seq)
            right = _tag_single_expr(fp.right_expr, schema, cte_steps=cte_seq) if fp.right_expr else None
            cte_fp.append(replace(fp, left_expr=left, right_expr=right))
        cte_hp = []
        for hp in cte.having_param or []:
            left = _tag_single_expr(hp.left_expr, schema, skip_value_injection=True, cte_steps=cte_seq)
            right = _tag_single_expr(hp.right_expr, schema, cte_steps=cte_seq) if hp.right_expr else None
            cte_hp.append(replace(hp, left_expr=left, right_expr=right))
        cte_steps.append(
            replace(
                cte,
                select_cols=cte_sc,
                order_by_cols=cte_obc,
                group_by_cols=cte_gb,
                filters_param=cte_fp,
                having_param=cte_hp,
            )
        )
    return replace(
        intent,
        select_cols=select_cols,
        order_by_cols=order_by_cols,
        group_by_cols=group_by_cols,
        filters_param=filters_param,
        having_param=having_param,
        cte_steps=cte_steps,
    )


def classify_cte_expr(expr: NormalizedExpr) -> str:
    """
    Classify a CTE select expression by its structural type.

    Args:

        expr: NormalizedExpr from a CTE SelectCol.

    Returns:

        One of 'passthrough', 'aggregation', 'scalar', or 'computed'.
    """
    agg = expr.agg_func or (expr.add_groups[0].agg_func if expr.add_groups else "")
    has_arithmetic = (
        len(expr.add_groups) + len(expr.sub_groups) > 1
        or expr.add_values
        or expr.sub_values
        or any(g.divide or len(g.multiply) > 1 or g.coefficient != 1.0 for g in expr.add_groups)
    )
    if agg:
        return "aggregation"
    if has_arithmetic:
        return "computed"
    scalar = expr.scalar_func or (expr.add_groups[0].scalar_func if expr.add_groups else "")
    if scalar:
        return "scalar"
    return "passthrough"


def derive_cte_output_columns(select_cols: list[SelectCol], *, cte_ordinal: int = 1) -> list[str]:
    """
    Derive CTE output aliases with dedup suffixes; passthrough preserves base name case.

    Args:

        select_cols:
            CTE ``SelectCol`` list.

        cte_ordinal:
            1-based index of this CTE in ``intent.cte_steps`` for stable ``cteN_wMM`` / ``cteN_cMM`` naming.

    Returns:

        One alias string per select column, same order.
    """
    derived: list[str] = []
    seen: dict[str, int] = {}
    expr_counter = 0
    for sc in select_cols:
        expr = sc.expr
        kind = classify_cte_expr(expr)
        agg = expr.agg_func or (expr.add_groups[0].agg_func if expr.add_groups else "")
        scalar = expr.scalar_func or (expr.add_groups[0].scalar_func if expr.add_groups else "")
        base = expr.primary_column or ""
        if kind == "aggregation":
            if base == "*":
                name = "row_count" if agg == "count" else f"{agg}_star"
            elif "." in base:
                bare = base.split(".", 1)[1]
                name = f"{agg}_{bare}"
            else:
                name = f"{agg}_{base}"
        elif kind == "scalar":
            if "." in base:
                bare = base.split(".", 1)[1]
                name = f"{scalar}_{bare}"
            else:
                name = f"{scalar}_{base}"
        elif kind == "computed":
            expr_counter += 1
            name = f"expr{expr_counter}"
        else:
            if not base.strip():
                expr_counter += 1
                name = f"expr{expr_counter}"
            elif re.fullmatch(REGISTRY_TOKEN_PATTERN, base.strip().lower()):
                name = f"cte{cte_ordinal}_{base.lower()}"
            elif "." in base:
                name = base.split(".", 1)[1]
            else:
                name = base
        if kind != "passthrough":
            name = name.lower()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        derived.append(name)
    for n in derived:
        assert n.strip()
        assert not re.fullmatch(REGISTRY_TOKEN_PATTERN, n.strip().lower())
    return derived


def _data_and_value_type_for_window_spec(ws: WindowSpec, schema: SchemaGraph) -> tuple[str, str]:
    """Infer ``(data_type, value_type)`` for a window function projection."""

    fn = (ws.function or "").strip().lower()
    if fn in {"row_number", "rank", "dense_rank", "ntile"}:
        return "integer", "integer"
    if fn in {"sum", "avg"}:
        return "numeric", "number"
    if fn in {"lag", "lead", "first_value", "last_value"}:
        arg = ws.argument
        if arg is None:
            return "unknown", "string"
        bc = arg.primary_column
        if "." in bc:
            tbl, col = bc.split(".", 1)
            tm = schema.tables.get(tbl)
            cm = tm.columns.get(col) if tm else None
            if cm is None and tm:
                cm = tm.columns.get(col.lower())
            if cm and cm.data_type:
                dt = cm.data_type.lower().split("(")[0].strip()
                if dt in ("date", "timestamp", "timestamptz", "datetime"):
                    return dt, "date"
                if dt in (
                    "int",
                    "integer",
                    "bigint",
                    "smallint",
                    "float",
                    "double",
                    "numeric",
                    "decimal",
                    "real",
                ):
                    return "numeric", "number"
        return "numeric", "number"
    return "numeric", "number"


def build_cte_output_metadata(
    select_cols: list[SelectCol],
    output_columns: list[str],
    schema: SchemaGraph,
    window_registry: list[WindowRegistryStep] | None = None,
) -> dict[str, CteOutputColumnMeta]:
    """
    Produce ``CteOutputColumnMeta`` per output alias from expr kind + schema.

    Args:

        select_cols: CTE selects.

        output_columns: Aliases parallel to *select_cols*.

        schema: Base table column metadata.

    Returns:

        Map ``alias -> meta``.
    """
    result: dict[str, CteOutputColumnMeta] = {}
    for i, sc in enumerate(select_cols):
        if i >= len(output_columns):
            break
        out_col = output_columns[i]
        expr = sc.expr
        wr_by = {s.registry_id: s for s in (window_registry or [])}
        rid = expr_registry_ref(expr) or ""
        if rid.startswith("w"):
            step = wr_by.get(rid)
            if step is not None:
                data_type, value_type = _data_and_value_type_for_window_spec(step.window_spec, schema)
                result[out_col] = CteOutputColumnMeta(
                    source="window_registry",
                    agg_func=(step.window_spec.function or "").lower(),
                    role="numeric_measure",
                    filterable=True,
                    aggregatable=True,
                    data_type=data_type,
                    value_type=value_type,
                    groupable=False,
                    valid_filter_ops=list(CTE_NUMERIC_FILTER_OPS),
                    valid_aggregations=list(CTE_FULL_AGGS),
                    valid_having_ops=list(CTE_HAVING_COMPARE_OPS),
                    sensitivity=None,
                )
                continue
        kind = classify_cte_expr(expr)
        agg = expr.agg_func or (expr.add_groups[0].agg_func if expr.add_groups else "")
        scalar = expr.scalar_func or (expr.add_groups[0].scalar_func if expr.add_groups else "")
        base_col = expr.primary_column
        src_meta = None
        base_type = "unknown"
        if "." in base_col and base_col != "*":
            tbl, col = base_col.split(".", 1)
            if tbl in schema.tables:
                src_meta = schema.tables[tbl].columns.get(col) or schema.tables[tbl].columns.get(col.lower())
                if src_meta and src_meta.data_type:
                    base_type = src_meta.data_type.lower().split("(")[0].strip()
        if kind == "passthrough":
            role = src_meta.role if src_meta else None
            data_type = base_type
            filterable = src_meta.is_filterable if src_meta else True
            aggregatable = src_meta.is_aggregatable if src_meta else False
            groupable = src_meta.is_groupable if src_meta else True
            vf_ops = list(src_meta.get_valid_filter_ops()) if src_meta else list(CTE_NUMERIC_FILTER_OPS)
            v_aggs = sorted(src_meta.get_valid_aggregations()) if src_meta else list(CTE_DEFAULT_AGGS)
            vh_ops = list(src_meta.get_valid_having_ops()) if src_meta else list(CTE_HAVING_COMPARE_OPS)
            out_sensitivity = src_meta.sensitivity.value if src_meta else None
        elif kind == "aggregation":
            role = "numeric_measure"
            if agg == "count":
                data_type = "integer"
            elif agg == "avg":
                data_type = "numeric"
            elif agg in ("sum", "min", "max") and base_type != "unknown":
                data_type = base_type
            else:
                data_type = "integer" if agg == "count" else "numeric"
            filterable = True
            aggregatable = True
            groupable = False
            vf_ops = list(CTE_NUMERIC_FILTER_OPS)
            v_aggs = list(CTE_FULL_AGGS)
            vh_ops = list(CTE_HAVING_COMPARE_OPS)
            out_sensitivity = None
        elif kind == "scalar":
            if scalar in NUMERIC_RESULT_SCALARS:
                role = "numeric_measure"
                data_type = "integer" if scalar in INTEGER_SCALARS else "numeric"
                aggregatable = True
                groupable = False
                vf_ops = list(CTE_NUMERIC_FILTER_OPS)
                v_aggs = list(CTE_FULL_AGGS)
                vh_ops = list(CTE_HAVING_COMPARE_OPS)
                out_sensitivity = None
            else:
                role = src_meta.role if src_meta else None
                data_type = base_type
                aggregatable = src_meta.is_aggregatable if src_meta else False
                groupable = src_meta.is_groupable if src_meta else True
                vf_ops = list(src_meta.get_valid_filter_ops()) if src_meta else list(CTE_NUMERIC_FILTER_OPS)
                v_aggs = sorted(src_meta.get_valid_aggregations()) if src_meta else list(CTE_DEFAULT_AGGS)
                vh_ops = list(src_meta.get_valid_having_ops()) if src_meta else list(CTE_HAVING_COMPARE_OPS)
                out_sensitivity = src_meta.sensitivity.value if src_meta else None
            filterable = True
        else:
            role = "numeric_measure"
            data_type = "numeric"
            filterable = True
            aggregatable = True
            groupable = False
            vf_ops = list(CTE_NUMERIC_FILTER_OPS)
            v_aggs = list(CTE_FULL_AGGS)
            vh_ops = list(CTE_HAVING_COMPARE_OPS)
            out_sensitivity = None
        lineage_phys_table: str | None = None
        lineage_phys_column: str | None = None
        lineage_inherits_pk = False
        lineage_fk_to_table: str | None = None
        lineage_fk_to_column: str | None = None
        semantic_distinct_values: list[str] = []
        semantic_join_neighbors: list[tuple[str, str]] = []
        lift_lineage = kind == "passthrough" or (kind == "scalar" and scalar not in NUMERIC_RESULT_SCALARS)
        if lift_lineage and "." in base_col and base_col != "*" and src_meta is not None:
            ltbl, lcol = base_col.split(".", 1)
            if ltbl in schema.tables:
                lineage_phys_table = ltbl
                lineage_phys_column = lcol
                lineage_inherits_pk = bool(src_meta.is_primary_key)
                if src_meta.fk_target:
                    lineage_fk_to_table = src_meta.fk_target[0]
                    lineage_fk_to_column = src_meta.fk_target[1]
                semantic_distinct_values = list(src_meta.semantic_distinct_values)
                semantic_join_neighbors = list(src_meta.semantic_join_neighbors)
        result[out_col] = CteOutputColumnMeta(
            source=kind,
            agg_func=agg,
            role=role,
            filterable=filterable,
            aggregatable=aggregatable,
            data_type=data_type,
            groupable=groupable,
            valid_filter_ops=vf_ops,
            valid_aggregations=v_aggs,
            valid_having_ops=vh_ops,
            sensitivity=out_sensitivity,
            lineage_phys_table=lineage_phys_table,
            lineage_phys_column=lineage_phys_column,
            lineage_inherits_pk=lineage_inherits_pk,
            lineage_fk_to_table=lineage_fk_to_table,
            lineage_fk_to_column=lineage_fk_to_column,
            semantic_distinct_values=semantic_distinct_values,
            semantic_join_neighbors=semantic_join_neighbors,
        )
    return result


def build_virtual_table_specs(
    intent: RuntimeIntent,
    schema: SchemaGraph | None,
) -> dict[str, VirtualTableSpec]:
    """
    Build join-graph virtual nodes for each CTE that exposes output column metadata.

    Args:

        intent: Runtime intent including ``cte_steps`` with ``output_column_metadata``.

        schema: Physical schema (unused today; reserved for cross-checks).

    Returns:

        Map ``cte_name -> VirtualTableSpec`` for CTEs with at least one output column.
    """
    if schema is None:
        return {}
    out: dict[str, VirtualTableSpec] = {}
    for step in intent.cte_steps:
        cols: dict[str, VirtualColumnSpec] = {}
        meta_map = step.output_column_metadata or {}
        for alias in step.output_columns or []:
            meta = meta_map.get(alias)
            if not meta:
                cols[alias] = VirtualColumnSpec(None, None, False, None, [], [])
                continue
            fk_to: tuple[str, str] | None = None
            if meta.lineage_fk_to_table and meta.lineage_fk_to_column:
                fk_to = (meta.lineage_fk_to_table, meta.lineage_fk_to_column)
            cols[alias] = VirtualColumnSpec(
                meta.lineage_phys_table,
                meta.lineage_phys_column,
                meta.lineage_inherits_pk,
                fk_to,
                list(meta.semantic_distinct_values),
                list(meta.semantic_join_neighbors),
            )
        if cols:
            out[step.cte_name] = VirtualTableSpec(
                cte_name=step.cte_name,
                columns=cols,
                emission=getattr(step, "emission", "join_table") or "join_table",
            )
    return out


def _strip_angle_brackets(obj: Any) -> Any:
    """
    Recursively strip angle-bracket placeholders from all string values in a parsed LLM output.

    Args:

        obj: Parsed JSON value such as a dict, list, string, or other type.

    Returns:

        The same structure with <word> placeholders expanded into literal words.
    """
    if isinstance(obj, str):
        return re.sub(r"<(\w+)>", r"\1", obj)
    if isinstance(obj, list):
        return [_strip_angle_brackets(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _strip_angle_brackets(v) for k, v in obj.items()}
    return obj


def _normalize_order_direction(direction: str) -> str:
    """
    Normalize sort direction to ``asc`` or ``desc``.

    Args:

        direction: Raw direction token.

    Returns:

        ``desc`` if string contains ``desc``, else ``asc``.
    """
    if not isinstance(direction, str):
        return "asc"
    d = direction.strip().lower()
    return "desc" if "desc" in d else "asc"


def _strip_order_direction(expr_raw: object) -> tuple[str, str]:
    """
    Split ``expr ASC|DESC`` into expression and normalized direction.

    Args:

        expr_raw: Order-by expression string.

    Returns:

        ``(expr_without_suffix, 'asc'|'desc')``; default ``asc`` for non-strings.
    """
    if not isinstance(expr_raw, str):
        return ("", "asc")
    trimmed = expr_raw.lstrip()
    if not trimmed:
        return ("", "asc")
    upper = trimmed.upper()
    if upper.endswith(" DESC"):
        return (trimmed[:-5].rstrip(), "desc")
    if upper.endswith(" ASC"):
        return (trimmed[:-4].rstrip(), "asc")
    return (trimmed.rstrip(), "asc")


def _order_by_col_from_obc(obc: dict[str, Any]) -> OrderByCol:
    """Parse one LLM ``order_by_cols`` object into ``OrderByCol``."""

    expr_raw = obc.get("expr")
    if isinstance(expr_raw, str):
        expr_clean, dir_from_expr = _strip_order_direction(expr_raw)
        direction_src = obc.get("direction")
        if direction_src is None:
            merged = dir_from_expr
        elif isinstance(direction_src, str):
            merged = direction_src
        else:
            merged = dir_from_expr or "asc"
        return OrderByCol(
            expr=parse_expr_string(expr_clean),
            direction=_normalize_order_direction(merged),
        )
    payload = dict(obc)
    payload["direction"] = _normalize_order_direction(obc.get("direction", "asc"))
    col = OrderByCol.from_dict(payload)
    return OrderByCol(expr=col.expr, direction=_normalize_order_direction(col.direction))


def _coerce_filter_group_token(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip():
        t = x.strip()
        if t.lstrip("-").isdigit():
            return int(t)
    return None


def _filter_group_for_param(raw: Any) -> int | None:
    return _coerce_filter_group_token(raw)


def _parse_filter_param_from_llm(fp: dict[str, Any]) -> list[FilterParam]:
    """
    Parse one LLM filter object into at most one ``FilterParam`` row.

    Args:

        fp: Raw filter dict.

    Returns:

        Zero or one param; empty when ``left_expr`` is missing.
    """
    left_str = fp.get("left_expr") or fp.get("left_col") or ""
    if not left_str:
        return []
    right_str = fp.get("right_expr") or fp.get("right_col") or ""
    if right_str and "." not in right_str and not _is_date_or_interval_expr(right_str):
        right_str = ""
    fg = _filter_group_for_param(fp.get("filter_group"))
    left_ex = parse_expr_string(left_str)
    op = normalize_op(fp.get("op", "="))
    right_ex = parse_expr_string(right_str) if right_str else None
    vt = fp.get("value_type", "string")
    raw_v = fp.get("value")
    bop = fp.get("bool_op", "AND")
    return [
        FilterParam(
            left_expr=left_ex,
            op=op,
            right_expr=right_ex,
            value_type=vt,
            param_key="",
            raw_value=raw_v,
            bool_op=bop,
            filter_group=fg,
        )
    ]


def _parse_having_param_from_llm(hp: dict[str, Any]) -> list[HavingParam]:
    """
    Parse one LLM having object into at most one ``HavingParam`` row.

    Args:

        hp: Raw having dict.

    Returns:

        Zero or one param; empty when ``left_expr`` is missing.
    """
    left_str = hp.get("left_expr") or hp.get("left_agg") or ""
    if not left_str:
        return []
    right_str = hp.get("right_expr") or hp.get("right_agg") or ""
    if right_str and "." not in right_str and not _is_date_or_interval_expr(right_str):
        right_str = ""
    fg = _filter_group_for_param(hp.get("filter_group"))
    left_ex = parse_expr_string(left_str)
    op = normalize_op(hp.get("op", ">"))
    right_ex = parse_expr_string(right_str) if right_str else None
    vt = hp.get("value_type", "integer")
    raw_v = hp.get("value")
    bop = hp.get("bool_op", "AND")
    return [
        HavingParam(
            left_expr=left_ex,
            op=op,
            right_expr=right_ex,
            value_type=vt,
            param_key="",
            raw_value=raw_v,
            bool_op=bop,
            filter_group=fg,
        )
    ]


def _parse_select_col_from_llm(sc: dict[str, Any]) -> SelectCol:
    """
    Parse select-col dict with ``expr`` string or object.

    Args:

        sc: LLM select column object.

    Returns:

        ``SelectCol`` instance.
    """
    expr_str = sc.get("expr", "") or ""
    expr = parse_expr_string(expr_str) if (isinstance(expr_str, str) and expr_str.strip()) else NormalizedExpr()
    return SelectCol(expr=expr)


def _sanitize_cte_alias_token(s: str) -> str:
    """
    Lowercase and coerce *s* toward a snake_case identifier fragment.

    Args:

        s: Raw token from an LLM ``output_columns`` cell.

    Returns:

        Sanitized string (may still violate the strict alias regex).
    """

    t = s.lower()
    t = t.replace(".", "_")
    t = re.sub(r"[^a-z0-9_]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if t and t[0].isdigit():
        t = "_" + t
    return t


def _canonicalise_one_output_column_item(
    raw: str,
    select_cols: list[SelectCol],
    derived: list[str],
    index: int,
) -> str:
    """
    Map one raw ``output_columns`` string to a snake_case alias matching ``INTENT_SCHEMA``.

    Args:

        raw: One cell from the LLM.

        select_cols: Parsed CTE ``select_cols`` (same order as output columns).

        derived: Result of :func:`derive_cte_output_columns` when *select_cols* is non-empty.

        index: Position of this cell.

    Returns:

        Canonical alias string.
    """

    s = raw.strip()
    if re.search(r"\s+AS\s+", s, flags=re.I):
        s = re.split(r"\s+AS\s+", s, flags=re.I)[-1].strip()
    if "(" in s or ")" in s:
        if index < len(derived):
            return derived[index]
        return f"out_{index}"
    sanitized = _sanitize_cte_alias_token(s)
    if sanitized and CTE_OUTPUT_ALIAS_RE.match(sanitized):
        return sanitized
    if index < len(derived):
        return derived[index]
    return f"out_{index}"


def _canonicalise_cte_output_columns(intent_dict: dict[str, Any]) -> None:
    """
    Rewrite ``cte_steps[*].output_columns`` in place so each entry matches ``^[a-z_][a-z0-9_]*$``.

    Args:

        intent_dict: Root intent object from ``safe_json_loads`` before ``INTENT_SCHEMA`` validation.
    """

    ctes = intent_dict.get("cte_steps")
    if not isinstance(ctes, list):
        return
    for cte_ordinal, cte in enumerate(ctes, start=1):
        if not isinstance(cte, dict):
            continue
        select_raw = cte.get("select_cols", [])
        if not isinstance(select_raw, list):
            select_raw = []
        cte_select_cols: list[SelectCol] = []
        for sc in select_raw:
            if isinstance(sc, str):
                cte_select_cols.append(SelectCol(expr=parse_expr_string(sc)))
            elif isinstance(sc, dict):
                cte_select_cols.append(_parse_select_col_from_llm(sc))
        oc_raw = cte.get("output_columns")
        if oc_raw is None:
            continue
        if isinstance(oc_raw, str):
            oc_raw = [oc_raw]
        if not isinstance(oc_raw, list):
            continue
        derived = derive_cte_output_columns(cte_select_cols, cte_ordinal=cte_ordinal) if cte_select_cols else []
        new_oc = [
            _canonicalise_one_output_column_item(str(raw), cte_select_cols, derived, i) for i, raw in enumerate(oc_raw)
        ]
        cte["output_columns"] = new_oc


def _intent_schema_validation_error(parsed: dict[str, Any]) -> str | None:
    """
    Return a jsonschema validation message when *parsed* fails ``INTENT_SCHEMA``, else ``None``.
    """

    try:
        jsonschema.validate(instance=parsed, schema=INTENT_SCHEMA)
    except jsonschema.ValidationError as exc:
        ptr = "/" + "/".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "/"
        return f"Field {ptr} failed INTENT_SCHEMA: {exc.message}"
    return None


def _validate_intent_schema(parsed: dict[str, Any]) -> bool:
    """
    Validate parsed intent dict against INTENT_SCHEMA.

    Args:

        parsed: Parsed JSON dict from an LLM response.

    Returns:

        True if the structure is valid for the schema and False otherwise.
    """

    return _intent_schema_validation_error(parsed) is None


def _flatten_dpipe_chain_operands(node: exp.Expression) -> list[exp.Expression]:
    """Flatten a left-associative ``DPipe`` tree into ordered operand expressions."""

    if isinstance(node, exp.DPipe):
        return _flatten_dpipe_chain_operands(node.this) + [node.expression]
    return [node]


def _split_root_concat_or_dpipe_sql(expr: str) -> list[str] | None:
    """
    When *expr* is a top-level ``CONCAT`` or ``||`` chain, return one SQL string per operand.

    Args:

        expr: Raw expression text from the model.

    Returns:

        Operand SQL fragments, or ``None`` when no split applies or parsing fails.
    """

    t = expr.strip()
    if not t:
        return None
    try:
        tree = sqlglot.parse_one(t, dialect=None)
    except Exception:
        return None
    if isinstance(tree, exp.Concat):
        return [x.sql() for x in tree.expressions]
    if isinstance(tree, exp.DPipe):
        return [x.sql() for x in _flatten_dpipe_chain_operands(tree)]
    return None


def _expand_select_cols_raw_entries(select_cols_raw: list[Any]) -> list[Any]:
    """
    Expand top-level ``CONCAT`` / ``||`` dict or string entries into separate raw select entries.

    Args:

        select_cols_raw: Raw ``select_cols`` array from parsed intent JSON.

    Returns:

        Expanded raw entries suitable for :func:`_parse_select_col_from_llm` / :func:`parse_expr_string`.
    """

    out: list[Any] = []
    for entry in select_cols_raw:
        if isinstance(entry, dict) and isinstance(entry.get("expr"), str):
            parts = _split_root_concat_or_dpipe_sql(entry["expr"])
            if parts and len(parts) > 1:
                base_alias = str(entry.get("alias") or "").strip()
                for i, p in enumerate(parts):
                    row = dict(entry)
                    row["expr"] = p
                    if base_alias:
                        row["alias"] = f"{base_alias}__p{i}"
                    elif "alias" in row:
                        del row["alias"]
                    out.append(row)
                continue
        if isinstance(entry, str):
            parts = _split_root_concat_or_dpipe_sql(entry)
            if parts and len(parts) > 1:
                out.extend(parts)
                continue
        out.append(entry)
    return out


def parse_intent_response(
    raw: str,
    question: str,
    *,
    parse_detail_out: list[str] | None = None,
) -> RuntimeIntent | None:
    """
    Parse LLM JSON through ``INTENT_SCHEMA`` into ``RuntimeIntent``.

    Args:

        raw: Model output string.

        question: Fallback for ``natural_language``.

        parse_detail_out: When provided, append one human-readable failure line before returning ``None``.

    Returns:

        Intent, or ``None`` on parse/validation failure.
    """
    parsed = safe_json_loads(raw)
    if not parsed or not isinstance(parsed, dict):
        if parse_detail_out is not None:
            parse_detail_out.append("Intent response is not a JSON object.")
        return None
    parsed = _strip_angle_brackets(parsed)
    parsed.pop("intent_status", None)
    _canonicalise_cte_output_columns(parsed)
    schema_err = _intent_schema_validation_error(parsed)
    if schema_err:
        debug("[intent_expr.parse_intent_response] schema validation failed")
        if parse_detail_out is not None:
            parse_detail_out.append(f"Intent JSON failed INTENT_SCHEMA validation: {schema_err}")
        return None

    tables = parsed.get("tables", [])
    if isinstance(tables, str):
        tables = [tables]

    select_cols_raw_in = parsed.get("select_cols", [])
    if not isinstance(select_cols_raw_in, list):
        select_cols_raw_in = [select_cols_raw_in] if select_cols_raw_in else []
    select_cols_raw = _expand_select_cols_raw_entries(select_cols_raw_in)
    select_cols = []
    for sc in select_cols_raw:
        if isinstance(sc, str):
            select_cols.append(SelectCol(expr=parse_expr_string(sc)))
        elif isinstance(sc, dict):
            select_cols.append(_parse_select_col_from_llm(sc))

    group_by_cols_raw = parsed.get("group_by_cols", [])
    if isinstance(group_by_cols_raw, str):
        group_by_cols_raw = [group_by_cols_raw]
    group_by_cols = [parse_expr_string(g) for g in group_by_cols_raw]

    order_by_cols_raw = parsed.get("order_by_cols", [])
    order_by_cols = []
    for obc in order_by_cols_raw:
        if isinstance(obc, str):
            expr_clean, direction = _strip_order_direction(obc)
            order_by_cols.append(OrderByCol(expr=parse_expr_string(expr_clean), direction=direction))
        elif isinstance(obc, dict):
            order_by_cols.append(_order_by_col_from_obc(obc))

    filters_param_raw = parsed.get("filters_param", [])
    filters_param = []
    for fp in filters_param_raw:
        if isinstance(fp, dict):
            filters_param.extend(_parse_filter_param_from_llm(fp))

    having_param_raw = parsed.get("having_param", [])
    having_param = []
    for hp in having_param_raw:
        if isinstance(hp, dict):
            having_param.extend(_parse_having_param_from_llm(hp))

    cte_steps_raw = parsed.get("cte_steps", [])
    cte_steps = []
    for cte in cte_steps_raw:
        if isinstance(cte, dict):
            cte_sc_in = cte.get("select_cols", [])
            if not isinstance(cte_sc_in, list):
                cte_sc_in = [cte_sc_in] if cte_sc_in else []
            cte_select_cols = []
            for sc in _expand_select_cols_raw_entries(cte_sc_in):
                if isinstance(sc, str):
                    cte_select_cols.append(SelectCol(expr=parse_expr_string(sc)))
                elif isinstance(sc, dict):
                    cte_select_cols.append(_parse_select_col_from_llm(sc))

            cte_order_by = []
            for obc in cte.get("order_by_cols", []):
                if isinstance(obc, str):
                    expr_clean, direction = _strip_order_direction(obc)
                    cte_order_by.append(OrderByCol(expr=parse_expr_string(expr_clean), direction=direction))
                elif isinstance(obc, dict):
                    cte_order_by.append(_order_by_col_from_obc(obc))

            cte_filters = []
            for fp in cte.get("filters_param", []):
                if isinstance(fp, dict):
                    cte_filters.extend(_parse_filter_param_from_llm(fp))

            cte_having = []
            for hp in cte.get("having_param", []):
                if isinstance(hp, dict):
                    cte_having.extend(_parse_having_param_from_llm(hp))

            cte_output_columns_raw = cte.get("output_columns", [])
            if isinstance(cte_output_columns_raw, str):
                cte_output_columns_raw = [cte_output_columns_raw]
            cte_label = str(cte.get("cte_name") or "").strip() or "<unnamed>"
            if len(cte_output_columns_raw) != len(cte_select_cols):
                if parse_detail_out is not None:
                    parse_detail_out.append(
                        f"CTE {cte_label!r}: output_columns length {len(cte_output_columns_raw)} "
                        f"must equal select_cols length {len(cte_select_cols)}."
                    )
                return None
            cte_output_columns: list[str] = []
            for raw_name in cte_output_columns_raw:
                name = str(raw_name).strip()
                if not name or not CTE_OUTPUT_ALIAS_RE.match(name):
                    if parse_detail_out is not None:
                        parse_detail_out.append(
                            f"CTE {cte_label!r}: each output_columns entry must match "
                            f"^[a-z_][a-z0-9_]*$; got {raw_name!r}."
                        )
                    return None
                cte_output_columns.append(name)

            cte_group_by_raw = cte.get("group_by_cols", [])
            cte_group_by = [parse_expr_string(g) for g in cte_group_by_raw]

            cte_window_registry: list[WindowRegistryStep] = []
            for wr in cte.get("window_registry") or []:
                if isinstance(wr, dict):
                    cte_window_registry.append(WindowRegistryStep.from_dict(wr))
            cte_case_registry: list[CaseRegistryStep] = []
            for cr in cte.get("case_registry") or []:
                if isinstance(cr, dict):
                    cte_case_registry.append(CaseRegistryStep.from_dict(cr))

            cte_steps.append(
                RuntimeCteStep(
                    cte_name=cte.get("cte_name", ""),
                    description=cte.get("description"),
                    tables=cte.get("tables", []),
                    select_cols=cte_select_cols,
                    group_by_cols=cte_group_by,
                    order_by_cols=cte_order_by,
                    filters_param=cte_filters,
                    having_param=cte_having,
                    param_values={},
                    output_columns=cte_output_columns,
                    grain=cte.get("grain") or "row_level",
                    limit=cte.get("limit"),
                    window_registry=cte_window_registry,
                    case_registry=cte_case_registry,
                )
            )

    limit = parsed.get("limit")
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = None

    natural_language = parsed.get("natural_language", "").strip() or question
    debug(f"[intent_parse.full_intent_parse] extracted natural_language='{natural_language}'")

    has_agg = any(sc.is_aggregated for sc in select_cols)
    if group_by_cols:
        grain = "grouped"
    elif has_agg:
        grain = "scalar"
    else:
        grain = "row_level"

    main_window_registry: list[WindowRegistryStep] = []
    for wr in parsed.get("window_registry") or []:
        if isinstance(wr, dict):
            main_window_registry.append(WindowRegistryStep.from_dict(wr))
    main_case_registry: list[CaseRegistryStep] = []
    for cr in parsed.get("case_registry") or []:
        if isinstance(cr, dict):
            main_case_registry.append(CaseRegistryStep.from_dict(cr))
    return RuntimeIntent(
        tables=tables,
        grain=grain,
        select_cols=select_cols,
        group_by_cols=group_by_cols,
        order_by_cols=order_by_cols,
        filters_param=filters_param,
        having_param=having_param,
        param_values={},
        cte_steps=cte_steps,
        natural_language=natural_language,
        limit=limit,
        schema_invalid=False,
        window_registry=main_window_registry,
        case_registry=main_case_registry,
    )


def _is_nontrivial_group(g: MulGroup) -> bool:
    """
    Return True if a MulGroup warrants coefficient parameterization.

    Args:

        g: MulGroup to test.

    Returns:

        True when the group has an aggregation, a scalar function, division operands, or a non-unit coefficient.
    """
    return bool(g.agg_func or g.scalar_func or g.inner_scalar_func or g.divide or g.coefficient != 1.0)


def _assign_structural_group(g: MulGroup, idx: int, pv: dict[str, Any], is_numeric: bool = True) -> int:
    """
    Assign structural param keys to a single MulGroup and collect values.

    Args:

        g: The MulGroup to assign param keys to.

        idx: The current structural param index counter.

        pv: The param_values dict to populate with key and value pairs.

        is_numeric: If False, skip coefficient parameterization entirely.

    Returns:

        The updated index after all assignments.
    """
    if is_numeric and _is_nontrivial_group(g) and g.coefficient not in (1, 1.0):
        key = f"s{idx}"
        g.coeff_param_key = key
        pv[key] = g.coefficient
        idx += 1
    if not is_numeric:
        return idx
    for i, v in enumerate(g.scalar_func_args or []):
        key = f"s{idx}"
        if len(g.sarg_param_keys) <= i:
            g.sarg_param_keys.append(key)
        else:
            g.sarg_param_keys[i] = key
        pv[key] = v
        idx += 1
    for i, v in enumerate(g.inner_scalar_func_args or []):
        key = f"s{idx}"
        if len(g.isarg_param_keys) <= i:
            g.isarg_param_keys.append(key)
        else:
            g.isarg_param_keys[i] = key
        pv[key] = v
        idx += 1
    return idx


def _assign_structural_filter_param(fp: FilterParam, idx: int, pv: dict[str, Any]) -> int:
    """Assign structural keys to both sides of a filter predicate."""

    idx = _assign_structural_date_predicate_payload(fp, idx, pv)
    idx = _assign_structural_expr(fp.left_expr, idx, pv)
    if fp.right_expr is not None:
        idx = _assign_structural_expr(fp.right_expr, idx, pv)
    return idx


def _assign_structural_having_param(hp: HavingParam, idx: int, pv: dict[str, Any]) -> int:
    """Assign structural keys to a HAVING predicate."""

    idx = _assign_structural_date_predicate_payload(hp, idx, pv)
    idx = _assign_structural_expr(hp.left_expr, idx, pv)
    if hp.right_expr is not None:
        idx = _assign_structural_expr(hp.right_expr, idx, pv)
    return idx


def _assign_structural_case_when_expr(cw: CaseWhenExpr, idx: int, pv: dict[str, Any]) -> int:
    """Assign structural keys across CASE branches and else clause."""

    for br in cw.branches or []:
        idx = _assign_structural_filter_param(br.condition, idx, pv)
        idx = _assign_structural_expr(br.result, idx, pv)
    if cw.else_result is not None:
        idx = _assign_structural_expr(cw.else_result, idx, pv)
    return idx


def _assign_structural_select_col(sc: SelectCol, idx: int, pv: dict[str, Any]) -> int:
    """Assign structural keys for one SELECT column expression."""

    return _assign_structural_expr(sc.expr, idx, pv)


def _assign_structural_case_registry(
    registry: list[CaseRegistryStep] | None,
    idx: int,
    pv: dict[str, Any],
) -> int:
    """Assign structural keys across ``case_registry[*].case_when`` bodies."""

    for step in registry or []:
        idx = _assign_structural_case_when_expr(step.case_when, idx, pv)
    return idx


def _assign_structural_expr(expr: NormalizedExpr, idx: int, pv: dict[str, Any]) -> int:
    """
    Assign structural param keys to a single NormalizedExpr including ExprValue offsets and collect values.

    Args:

        expr: The NormalizedExpr to assign param keys to.

        idx: The current structural param index counter.

        pv: The param_values dict to populate with key and value pairs.

    Returns:

        The updated index after all assignments.
    """
    for g in expr.add_groups:
        idx = _assign_structural_group(g, idx, pv, is_numeric=expr.is_numeric)
    for g in expr.sub_groups:
        idx = _assign_structural_group(g, idx, pv, is_numeric=expr.is_numeric)
    if expr.is_numeric:
        for i, v in enumerate(expr.scalar_func_args or []):
            key = f"s{idx}"
            if len(expr.sarg_param_keys) <= i:
                expr.sarg_param_keys.append(key)
            else:
                expr.sarg_param_keys[i] = key
            pv[key] = v
            idx += 1
        for i, v in enumerate(expr.inner_scalar_func_args or []):
            key = f"s{idx}"
            if len(expr.isarg_param_keys) <= i:
                expr.isarg_param_keys.append(key)
            else:
                expr.isarg_param_keys[i] = key
            pv[key] = v
            idx += 1
    if expr.is_numeric:
        for ev in expr.add_values:
            if ev.value in (0, 0.0):
                continue
            key = f"s{idx}"
            ev.param_key = key
            pv[key] = ev.value
            idx += 1
        for ev in expr.sub_values:
            if ev.value in (0, 0.0):
                continue
            key = f"s{idx}"
            ev.param_key = key
            pv[key] = ev.value
            idx += 1
    return idx


def _infer_date_unit(column: str) -> str:
    """
    Infer a temporal unit keyword from a column name for date function default arguments.

    Args:

        column: Fully qualified or bare column name string.

    Returns:

        One of 'month', 'day', 'week', 'quarter', or 'year', defaulting to 'month' when no date keyword is found in the column name.
    """
    col_lower = column.lower()
    for keyword, unit in DATE_UNIT_KEYWORDS:
        if keyword in col_lower:
            return unit
    return "month"


def _fill_group_defaults(g: MulGroup) -> None:
    """
    Mutate *g*: fill missing scalar arg lists from defaults or date- unit inference.

    Args:

        g: Mul group to update in place.

    Returns:

        None.
    """
    if g.scalar_func and not g.scalar_func_args:
        func = g.scalar_func.lower()
        if func in SCALAR_FUNCTIONS_LEADING_ARG:
            leaf = g.multiply[0] if g.multiply else None
            col = leaf.column_ref if leaf is not None and leaf.column_ref and not leaf.star else ""
            g.scalar_func_args = [_infer_date_unit(col)]
        else:
            defaults = SCALAR_FUNC_DEFAULTS.get(func)
            if defaults is not None:
                g.scalar_func_args = list(defaults)
    if g.inner_scalar_func and not g.inner_scalar_func_args:
        func = g.inner_scalar_func.lower()
        if func in SCALAR_FUNCTIONS_LEADING_ARG:
            leaf = g.multiply[0] if g.multiply else None
            col = leaf.column_ref if leaf is not None and leaf.column_ref and not leaf.star else ""
            g.inner_scalar_func_args = [_infer_date_unit(col)]
        else:
            defaults = SCALAR_FUNC_DEFAULTS.get(func)
            if defaults is not None:
                g.inner_scalar_func_args = list(defaults)


def _fill_expr_defaults(expr: NormalizedExpr) -> None:
    """
    Mutate *expr* and nested groups: fill missing scalar arg lists.

    Args:

        expr: Expression to update in place.

    Returns:

        None.
    """
    if expr.scalar_func and not expr.scalar_func_args:
        func = expr.scalar_func.lower()
        if func in SCALAR_FUNCTIONS_LEADING_ARG:
            col = expr.primary_column
            expr.scalar_func_args = [_infer_date_unit(col)]
        else:
            defaults = SCALAR_FUNC_DEFAULTS.get(func)
            if defaults is not None:
                expr.scalar_func_args = list(defaults)
    if expr.inner_scalar_func and not expr.inner_scalar_func_args:
        func = expr.inner_scalar_func.lower()
        if func in SCALAR_FUNCTIONS_LEADING_ARG:
            col = expr.primary_column
            expr.inner_scalar_func_args = [_infer_date_unit(col)]
        else:
            defaults = SCALAR_FUNC_DEFAULTS.get(func)
            if defaults is not None:
                expr.inner_scalar_func_args = list(defaults)
    for g in expr.add_groups + expr.sub_groups:
        _fill_group_defaults(g)


def ensure_scalar_func_defaults(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Call ``_fill_expr_defaults`` on every expr in *intent* (main + CTEs).

    Args:

        intent: Intent to mutate in place.

    Returns:

        Same *intent* instance.
    """
    for cte in intent.cte_steps or []:
        for sc in cte.select_cols or []:
            _fill_expr_defaults(sc.expr)
        for g in cte.group_by_cols or []:
            _fill_expr_defaults(g)
        for obc in cte.order_by_cols or []:
            _fill_expr_defaults(obc.expr)
        for fp in cte.filters_param or []:
            _fill_expr_defaults(fp.left_expr)
            if fp.right_expr:
                _fill_expr_defaults(fp.right_expr)
        for hp in cte.having_param or []:
            _fill_expr_defaults(hp.left_expr)
            if hp.right_expr:
                _fill_expr_defaults(hp.right_expr)
    for sc in intent.select_cols or []:
        _fill_expr_defaults(sc.expr)
    for g in intent.group_by_cols or []:
        _fill_expr_defaults(g)
    for obc in intent.order_by_cols or []:
        _fill_expr_defaults(obc.expr)
    for fp in intent.filters_param or []:
        _fill_expr_defaults(fp.left_expr)
        if fp.right_expr:
            _fill_expr_defaults(fp.right_expr)
    for hp in intent.having_param or []:
        _fill_expr_defaults(hp.left_expr)
        if hp.right_expr:
            _fill_expr_defaults(hp.right_expr)
    return intent


def _assign_structural_window_registry(
    registry: list[WindowRegistryStep] | None,
    idx: int,
    pv: dict[str, Any],
) -> int:
    """Assign structural keys for window registry expressions nested in ``window_spec``."""

    for step in registry or []:
        ws = step.window_spec
        for part in ws.partition_by or []:
            idx = _assign_structural_expr(part, idx, pv)
        for ob in ws.order_by or []:
            idx = _assign_structural_expr(ob.expr, idx, pv)
        if ws.argument is not None:
            idx = _assign_structural_expr(ws.argument, idx, pv)
    return idx


def _assign_structural_date_predicate_payload(pred: FilterParam | HavingParam, idx: int, pv: dict[str, Any]) -> int:
    """Bind ``s*`` for calendar unit in ``date_diff`` / ``date_window`` dict payloads; clear ``raw_value``."""

    vt = (pred.value_type or "").lower()
    if vt not in ("date_diff", "date_window"):
        return idx
    rv = pred.raw_value
    if not isinstance(rv, dict) or "unit" not in rv:
        return idx
    key = f"s{idx}"
    ur = rv.get("unit", "day")
    if isinstance(ur, str):
        canon = _canonicalize_temporal_unit(ur)
        stored = canon if isinstance(canon, str) and canon else ur
    else:
        stored = str(ur) if ur is not None else "day"
    pv[key] = stored
    pred.param_key_unit = key
    pred.raw_value = None
    return idx + 1


def extract_structural_params(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Assign ``s*`` keys to limits, coeffs, and func args; fill ``param_values``.

    Args:

        intent: Tagged intent (``is_numeric``, etc.).

    Returns:

        ``replace`` copy with ``param_values`` and ``limit_param_key``.
    """
    pv: dict[str, Any] = dict(intent.param_values or {})
    idx = 1
    for cte in intent.cte_steps or []:
        if cte.limit is not None:
            key = f"s{idx}"
            pv[key] = cte.limit
            cte.limit_param_key = key
            idx += 1
        for sc in cte.select_cols or []:
            idx = _assign_structural_select_col(sc, idx, pv)
        idx = _assign_structural_window_registry(cte.window_registry, idx, pv)
        idx = _assign_structural_case_registry(cte.case_registry, idx, pv)
        for g in cte.group_by_cols or []:
            idx = _assign_structural_expr(g, idx, pv)
        for obc in cte.order_by_cols or []:
            idx = _assign_structural_expr(obc.expr, idx, pv)
        for fp in cte.filters_param or []:
            idx = _assign_structural_filter_param(fp, idx, pv)
        for hp in cte.having_param or []:
            idx = _assign_structural_having_param(hp, idx, pv)
    limit_param_key = ""
    if intent.limit is not None:
        key = f"s{idx}"
        pv[key] = intent.limit
        limit_param_key = key
        idx += 1
    for sc in intent.select_cols or []:
        idx = _assign_structural_select_col(sc, idx, pv)
    idx = _assign_structural_window_registry(intent.window_registry, idx, pv)
    idx = _assign_structural_case_registry(intent.case_registry, idx, pv)
    for g in intent.group_by_cols or []:
        idx = _assign_structural_expr(g, idx, pv)
    for obc in intent.order_by_cols or []:
        idx = _assign_structural_expr(obc.expr, idx, pv)
    for fp in intent.filters_param or []:
        idx = _assign_structural_filter_param(fp, idx, pv)
    for hp in intent.having_param or []:
        idx = _assign_structural_having_param(hp, idx, pv)
    debug(f"[intent_process.extract_structural_params] assigned {idx - 1} structural params")
    return replace(intent, param_values=pv, limit_param_key=limit_param_key)


def apply_default_structural_values(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Ensure structural ``s*`` slots in ``param_values`` are never unset before substitution.

    Args:

        intent: Runtime intent after deterministic repairs.

    Returns:

        Intent copy with missing structural keys coerced to numeric zero.
    """

    pv = dict(intent.param_values or {})
    changed = False
    for k in list(pv.keys()):
        if not is_structural_param_key(k):
            continue
        if pv[k] is None:
            pv[k] = 0
            changed = True
    return replace(intent, param_values=pv) if changed else intent


def cleared_param_runtime_intent(intent: RuntimeIntent) -> RuntimeIntent:
    """Return a deep copy of *intent* with all ``param_values`` cleared (main and CTEs)."""

    out = deepcopy(intent)
    empty_ctes = [replace(cte, param_values={}) for cte in (out.cte_steps or [])]
    return replace(out, param_values={}, limit_param_key="", cte_steps=empty_ctes)


def structural_s_key_assignment_order(intent: RuntimeIntent) -> list[str]:
    """Return structural ``s*`` keys in the same assignment order as ``extract_structural_params``."""

    blank = cleared_param_runtime_intent(intent)
    tagged = extract_structural_params(blank)
    return [k for k in tagged.param_values if is_structural_param_key(k)]


def collect_raw_param_values(intent: RuntimeIntent) -> dict[str, Any]:
    """
    Harvest ``raw_value`` into a dict and clear ``raw_value`` on each param (CTEs first).

    Walks ``filters_param``, ``having_param``, and ``case_registry[*].case_when`` branch conditions
    in both the main intent and every CTE so CASE literals are bound alongside WHERE / HAVING parameters.

    Args:

        intent: Intent to scan and mutate.

    Returns:

        ``param_key -> raw_value`` map.
    """
    pv: dict[str, Any] = {}

    def _harvest_filter(fp: FilterParam) -> None:
        vt = (fp.value_type or "").lower()
        if vt in ("date_window", "date_diff") and isinstance(fp.raw_value, dict):
            rv = dict(fp.raw_value)
            if fp.param_key:
                num = rv.get("amount")
                if num is not None and not isinstance(num, dict):
                    pv[fp.param_key] = int(num) if not isinstance(num, bool) else 0
            unit_val = rv.get("unit", "day")
            fp.raw_value = {"unit": unit_val}
            return
        if vt in ("date_window", "date_diff"):
            return
        if fp.op == "between" and fp.param_key and fp.param_key_hi and fp.raw_value is not None:
            bounds = _parse_between_raw_value(fp.raw_value)
            if bounds is not None:
                pv[fp.param_key] = bounds[0]
                pv[fp.param_key_hi] = bounds[1]
                fp.raw_value = None
            return
        if fp.param_key and fp.raw_value is not None:
            pv[fp.param_key] = fp.raw_value
            fp.raw_value = None

    def _harvest_having(hp: HavingParam) -> None:
        vt = (hp.value_type or "").lower()
        if vt in ("date_window", "date_diff") and isinstance(hp.raw_value, dict):
            rv = dict(hp.raw_value)
            if hp.param_key:
                num = rv.get("amount")
                if num is not None and not isinstance(num, dict):
                    pv[hp.param_key] = int(num) if not isinstance(num, bool) else 0
            hp.raw_value = {"unit": rv.get("unit", "day")}
            return
        if vt in ("date_window", "date_diff"):
            return
        if hp.param_key and hp.raw_value is not None:
            pv[hp.param_key] = hp.raw_value
            hp.raw_value = None

    def _harvest_case_registry(regs: list[CaseRegistryStep] | None) -> None:
        for step in regs or []:
            cw = step.case_when
            if cw is None:
                continue
            for branch in cw.branches or []:
                if branch.condition is not None:
                    _harvest_filter(branch.condition)

    for cte in intent.cte_steps or []:
        for fp in cte.filters_param or []:
            _harvest_filter(fp)
        for hp in cte.having_param or []:
            _harvest_having(hp)
        _harvest_case_registry(cte.case_registry)
    for fp in intent.filters_param or []:
        _harvest_filter(fp)
    for hp in intent.having_param or []:
        _harvest_having(hp)
    _harvest_case_registry(intent.case_registry)
    return pv


def _assign_case_registry_param_keys(
    case_registry: list[CaseRegistryStep] | None,
    start_idx: int,
) -> tuple[list[CaseRegistryStep], int]:
    """Allocate ``p*`` keys to CASE branch conditions inside ``case_registry`` entries."""

    if not case_registry:
        return list(case_registry or []), start_idx
    idx = start_idx
    out: list[CaseRegistryStep] = []
    for step in case_registry:
        cw = step.case_when
        if cw is None or not cw.branches:
            out.append(step)
            continue
        new_branches: list[CaseWhenBranch] = []
        for branch in cw.branches:
            cond = branch.condition
            if cond is None:
                new_branches.append(branch)
                continue
            if cond.op in ("is null", "is not null") or cond.right_expr is not None or cond.param_key:
                new_branches.append(branch)
                continue
            cvt = (cond.value_type or "").lower()
            if cvt in ("date_window", "date_diff") and isinstance(cond.raw_value, dict):
                new_cond = replace(cond, param_key=f"p{idx}")
                idx += 1
                new_branches.append(replace(branch, condition=new_cond))
                continue
            if cvt in ("date_window", "date_diff"):
                new_branches.append(branch)
                continue
            if cond.op == "between":
                lo_key = f"p{idx}"
                hi_key = f"p{idx + 1}"
                idx += 2
                new_cond = replace(cond, param_key=lo_key, param_key_hi=hi_key)
                new_branches.append(replace(branch, condition=new_cond))
                continue
            new_cond = replace(cond, param_key=f"p{idx}")
            idx += 1
            new_branches.append(replace(branch, condition=new_cond))
        new_cw = replace(cw, branches=new_branches)
        out.append(replace(step, case_when=new_cw))
    return out, idx


def assign_param_keys(
    filters_param: list[FilterParam],
    having_param: list[HavingParam],
    cte_steps: list[RuntimeCteStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
) -> tuple[
    list[FilterParam],
    list[HavingParam],
    list[RuntimeCteStep],
    list[CaseRegistryStep],
    int,
]:
    """
    Assign ``p*`` keys to filter/having value params and CASE-branch conditions (CTEs first).

    Skips null operators, date_window/date_diff value types, and predicates whose RHS is a
    column expression. Each CTE ``case_registry`` is keyed before main filters/having; main
    ``case_registry`` follows.

    Args:

        filters_param: Main filters.

        having_param: Main having list.

        cte_steps: Optional CTE chain to key before main.

        case_registry: Main-query CASE registry rows (``c01``, …).

    Returns:

        ``(filters, having, cte_steps, case_registry, next_index)``.
    """
    idx = 1
    updated_cte_steps: list[RuntimeCteStep] = []
    for cte in cte_steps or []:
        cte_fp = []
        for fp in cte.filters_param or []:
            vt = (fp.value_type or "").lower()
            if fp.op in ("is null", "is not null") or fp.right_expr is not None:
                cte_fp.append(fp)
            elif vt in ("date_window", "date_diff") and isinstance(fp.raw_value, dict):
                cte_fp.append(replace(fp, param_key=f"p{idx}"))
                idx += 1
            elif vt in ("date_window", "date_diff"):
                cte_fp.append(fp)
            else:
                cte_fp.append(replace(fp, param_key=f"p{idx}"))
                idx += 1
        cte_hp = []
        for hp in cte.having_param or []:
            if hp.right_expr is not None:
                cte_hp.append(hp)
            else:
                cte_hp.append(replace(hp, param_key=f"p{idx}"))
                idx += 1
        new_cte_case_registry, idx = _assign_case_registry_param_keys(cte.case_registry, idx)
        updated_cte_steps.append(
            replace(
                cte,
                filters_param=cte_fp,
                having_param=cte_hp,
                case_registry=new_cte_case_registry,
            )
        )
    new_filters_param = []
    for fp in filters_param:
        vt = (fp.value_type or "").lower()
        if fp.op in ("is null", "is not null") or fp.right_expr is not None:
            new_filters_param.append(fp)
        elif vt in ("date_window", "date_diff") and isinstance(fp.raw_value, dict):
            new_filters_param.append(replace(fp, param_key=f"p{idx}"))
            idx += 1
        elif vt in ("date_window", "date_diff"):
            new_filters_param.append(fp)
        else:
            new_filters_param.append(replace(fp, param_key=f"p{idx}"))
            idx += 1
    new_having_param = []
    for hp in having_param:
        if hp.right_expr is not None:
            new_having_param.append(hp)
        else:
            new_having_param.append(replace(hp, param_key=f"p{idx}"))
            idx += 1
    new_case_registry, idx = _assign_case_registry_param_keys(case_registry, idx)
    return (
        new_filters_param,
        new_having_param,
        updated_cte_steps,
        new_case_registry,
        idx,
    )


def _parse_between_raw_value(raw_value: Any) -> tuple[Any, Any] | None:
    """
    Parse BETWEEN bounds from list or delimited string.

    Args:

        raw_value: ``between`` operand.

    Returns:

        ``(low, high)`` or ``None``.
    """
    if isinstance(raw_value, list) and len(raw_value) == 2:
        return raw_value[0], raw_value[1]
    if isinstance(raw_value, str):
        for sep in (" AND ", " and ", ",", " - "):
            parts = raw_value.split(sep, 1)
            if len(parts) == 2:
                lo = parts[0].strip()
                hi = parts[1].strip()
                if lo and hi:
                    return lo, hi
    return None


def _decompose_between_param_list(
    params: list[FilterParam] | list[HavingParam],
) -> list[FilterParam] | list[HavingParam]:
    """
    Replace each ``between`` with ``>=`` and ``<=`` pair.

    Args:

        params: Filter or having list.

    Returns:

        Expanded list (same element types).
    """
    result: list = []
    for p in params:
        if p.op.lower() != "between":
            result.append(p)
            continue
        bounds = _parse_between_raw_value(p.raw_value)
        if bounds is not None:
            result.append(replace(p, op=">=", raw_value=bounds[0], bool_op="AND"))
            result.append(replace(p, op="<=", raw_value=bounds[1]))
        else:
            result.append(replace(p, op=">=", bool_op="AND"))
            result.append(replace(p, op="<="))
    return result


def _normalize_case_registry_between(
    case_registry: list[CaseRegistryStep] | None,
) -> list[CaseRegistryStep] | None:
    """
    Canonicalise BETWEEN raw_values inside CASE branches stored in the case_registry.

    Branch predicates live under ``case_registry[*].case_when.branches[*].condition``.

    Args:

        case_registry: Registry list to scan; ``None`` is returned unchanged.

    Returns:

        New registry list with BETWEEN bounds expanded to ``[lo, hi]`` tuples, or the
        original value when no rewrites apply.
    """
    if not case_registry:
        return case_registry
    out: list[CaseRegistryStep] = []
    changed = False
    for step in case_registry:
        cw = step.case_when
        if cw is None or not cw.branches:
            out.append(step)
            continue
        new_branches: list[CaseWhenBranch] = []
        step_changed = False
        for branch in cw.branches:
            cond = branch.condition
            if cond is None or cond.op != "between":
                new_branches.append(branch)
                continue
            bounds = _parse_between_raw_value(cond.raw_value)
            if bounds is None:
                new_branches.append(branch)
                continue
            new_cond = replace(cond, raw_value=[bounds[0], bounds[1]])
            new_branches.append(replace(branch, condition=new_cond))
            step_changed = True
        if step_changed:
            changed = True
            out.append(replace(step, case_when=replace(cw, branches=new_branches)))
        else:
            out.append(step)
    return out if changed else case_registry


def decompose_between_params(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Decompose BETWEEN filter and having operators into paired >= and <= conditions.

    Applies to filters_param, having_param, and their counterparts in CTE steps. CASE branch
    conditions under ``case_registry[*].case_when`` cannot be split into two flat predicates;
    instead their raw_value is normalised to a ``(lo, hi)`` tuple and rendered via the BETWEEN
    arm of ``_render_case_branch_sql`` after two param keys are allocated.

    Args:

        intent: RuntimeIntent containing filters_param, having_param, and cte_steps to process.

    Returns:

        New RuntimeIntent with all BETWEEN operators split into >= and <= pairs in flat lists
        and canonicalised raw_values inside CASE branches.
    """
    new_fp = _decompose_between_param_list(intent.filters_param or [])
    new_hp = _decompose_between_param_list(intent.having_param or [])
    new_case_registry = _normalize_case_registry_between(intent.case_registry)
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_fp = _decompose_between_param_list(cte.filters_param or [])
        cte_hp = _decompose_between_param_list(cte.having_param or [])
        cte_cr = _normalize_case_registry_between(cte.case_registry)
        new_cte_steps.append(
            replace(
                cte,
                filters_param=cte_fp,
                having_param=cte_hp,
                case_registry=cte_cr,
            )
        )
    return replace(
        intent,
        filters_param=new_fp,
        having_param=new_hp,
        case_registry=new_case_registry,
        cte_steps=new_cte_steps,
    )


def _parse_in_string_to_list(raw_value: str) -> list[str]:
    """
    Parse a string-encoded IN-list into a list of stripped string elements.

    Handles formats such as "R, PG-13", "'R','PG-13'", and "1, 2, 3" and strips leading and trailing quotes on each element.

    Args:

        raw_value: String representation of an IN-list value.

    Returns:

        List of individual value strings with surrounding quotes removed.
    """
    parts = IN_STRING_SEPARATORS.split(raw_value)
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def _normalize_in_param_list(
    params: list[FilterParam] | list[HavingParam],
) -> list[FilterParam] | list[HavingParam]:
    """
    Convert string raw_values to lists for IN / NOT IN operators.

    Args:

        params: Filter or having params to normalise.

    Returns:

        New list with string IN-values parsed into Python lists.
    """
    result: list = []
    for p in params:
        if p.op.lower() not in IN_OPS or not isinstance(p.raw_value, str):
            result.append(p)
            continue
        parsed = _parse_in_string_to_list(p.raw_value)
        if len(parsed) > 1:
            result.append(replace(p, raw_value=parsed))
        else:
            result.append(p)
    return result


def normalize_in_raw_values(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Parse string IN-list ``raw_value`` into Python lists (main + CTEs).

    Args:

        intent: Intent to update.

    Returns:

        Intent with list ``raw_value`` where applicable.
    """
    new_fp = _normalize_in_param_list(intent.filters_param or [])
    new_hp = _normalize_in_param_list(intent.having_param or [])
    new_cr = _normalize_in_case_registry_raw_values(intent.case_registry)
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_fp = _normalize_in_param_list(cte.filters_param or [])
        cte_hp = _normalize_in_param_list(cte.having_param or [])
        cte_cr = _normalize_in_case_registry_raw_values(cte.case_registry)
        new_cte_steps.append(
            replace(cte, filters_param=cte_fp, having_param=cte_hp, case_registry=cte_cr),
        )
    return replace(
        intent,
        filters_param=new_fp,
        having_param=new_hp,
        case_registry=new_cr,
        cte_steps=new_cte_steps,
    )


def _normalize_in_case_registry_raw_values(
    case_registry: list[CaseRegistryStep] | None,
) -> list[CaseRegistryStep] | None:
    """Canonicalise IN/NOT IN ``raw_value`` from string to list inside registry CASE branch conditions."""

    if not case_registry:
        return case_registry
    out: list[CaseRegistryStep] = []
    changed_any = False
    for step in case_registry:
        cw = step.case_when
        if cw is None or not cw.branches:
            out.append(step)
            continue
        new_branches: list[CaseWhenBranch] = []
        step_changed = False
        for branch in cw.branches:
            cond = branch.condition
            if cond is None or cond.op.lower() not in IN_OPS or not isinstance(cond.raw_value, str):
                new_branches.append(branch)
                continue
            parsed = _parse_in_string_to_list(cond.raw_value)
            if len(parsed) <= 1:
                new_branches.append(branch)
                continue
            new_cond = replace(cond, raw_value=parsed)
            new_branches.append(replace(branch, condition=new_cond))
            step_changed = True
        if step_changed:
            changed_any = True
            out.append(replace(step, case_when=replace(cw, branches=new_branches)))
        else:
            out.append(step)
    return out if changed_any else case_registry


def _tag_case_registry_condition_scope(case_registry: list[Any] | None) -> list[Any]:
    """Set ``condition_scope`` on every CASE registry step based on aggregate use."""

    if not case_registry:
        return list(case_registry or [])
    out: list[Any] = []
    for step in case_registry:
        cw = getattr(step, "case_when", None)
        if cw is None or not cw.branches:
            out.append(step)
            continue
        new_scope = "having" if cw.has_aggregated_condition else "filter"
        if cw.condition_scope == new_scope:
            out.append(step)
            continue
        out.append(replace(step, case_when=replace(cw, condition_scope=new_scope)))
    return out


def tag_case_when_condition_scope(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Tag every ``CaseWhenExpr`` with ``condition_scope`` (``"filter"`` vs ``"having"``).

    A branch condition that references SQL aggregates (e.g. ``SUM(amount) > 100``) implies the parent scope must use GROUP BY semantics — surfaced as ``"having"`` so downstream validators can gate aggregation consistency.
    """

    new_registry = _tag_case_registry_condition_scope(intent.case_registry)
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_cr = _tag_case_registry_condition_scope(getattr(cte, "case_registry", None))
        new_cte_steps.append(replace(cte, case_registry=cte_cr))
    return replace(
        intent,
        case_registry=new_registry,
        cte_steps=new_cte_steps,
    )


def _canonicalize_temporal_unit(unit: Any) -> Any:
    """
    Map a colloquial temporal-grain token to its canonical form.

    Args:

        unit: Candidate unit value (string or other).

    Returns:

        Canonical singular unit when *unit* is a recognised alias; otherwise the value unchanged.
    """
    if not isinstance(unit, str):
        return unit
    canonical = DATE_UNIT_ALIAS_TO_CANONICAL.get(unit.lower().strip())
    return canonical if canonical is not None else unit


def _normalize_date_unit_in_raw_value(raw_value: Any) -> Any:
    """
    Canonicalize ``unit`` in a date filter dict via ``DATE_UNIT_ALIAS_TO_CANONICAL``.

    Args:

        raw_value: Filter ``raw_value`` (expect dict with ``unit``).

    Returns:

        Updated dict or unchanged *raw_value*.
    """
    if not isinstance(raw_value, dict):
        return raw_value
    unit = raw_value.get("unit")
    if isinstance(unit, str):
        canonical = _canonicalize_temporal_unit(unit)
        if isinstance(canonical, str) and canonical != unit:
            return {**raw_value, "unit": canonical}
    return raw_value


def normalize_date_diff_raw_values(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Canonicalize ``unit`` in date_window and date_diff filters via ``DATE_UNIT_ALIAS_TO_CANONICAL``.
    Coerce legacy numeric scalars to structured ``{unit, amount}`` payloads for both value types.

    Args:

        intent: RuntimeIntent whose date filters may use colloquial units.

    Returns:

        Updated intent with canonical units, or unchanged when nothing matches.
    """

    def _process(params: list) -> list:
        out = []
        for p in params or []:
            if p.value_type in ("date_window", "date_diff") and p.raw_value is not None:
                if isinstance(p.raw_value, (int, float)) and not isinstance(p.raw_value, bool):
                    if p.value_type == "date_diff":
                        out.append(replace(p, raw_value={"unit": "day", "amount": int(p.raw_value)}))
                    else:
                        out.append(replace(p, raw_value={"unit": "day", "amount": int(p.raw_value)}))
                elif isinstance(p.raw_value, dict):
                    out.append(replace(p, raw_value=_normalize_date_unit_in_raw_value(p.raw_value)))
                else:
                    out.append(p)
            else:
                out.append(p)
        return out

    new_fp = _process(intent.filters_param or [])
    new_hp = _process(intent.having_param or [])
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_fp = _process(cte.filters_param or [])
        cte_hp = _process(cte.having_param or [])
        new_cte_steps.append(replace(cte, filters_param=cte_fp, having_param=cte_hp))
    return replace(intent, filters_param=new_fp, having_param=new_hp, cte_steps=new_cte_steps)


def _canonicalize_group_temporal_args(g: MulGroup) -> None:
    """
    Mutate *g*: canonicalize the leading temporal-unit arg for date_trunc/date_part/extract.

    Args:

        g: Mul group to update in place.

    Returns:

        None.
    """
    if g.scalar_func and g.scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and g.scalar_func_args:
        g.scalar_func_args[0] = _canonicalize_temporal_unit(g.scalar_func_args[0])
    if g.inner_scalar_func and g.inner_scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and g.inner_scalar_func_args:
        g.inner_scalar_func_args[0] = _canonicalize_temporal_unit(g.inner_scalar_func_args[0])


def _canonicalize_expr_temporal_args(expr: NormalizedExpr) -> None:
    """
    Mutate *expr* and its nested groups: canonicalize temporal-unit leading args.

    Args:

        expr: Expression to update in place.

    Returns:

        None.
    """
    if expr.scalar_func and expr.scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG and expr.scalar_func_args:
        expr.scalar_func_args[0] = _canonicalize_temporal_unit(expr.scalar_func_args[0])
    if (
        expr.inner_scalar_func
        and expr.inner_scalar_func.lower() in SCALAR_FUNCTIONS_LEADING_ARG
        and expr.inner_scalar_func_args
    ):
        expr.inner_scalar_func_args[0] = _canonicalize_temporal_unit(expr.inner_scalar_func_args[0])
    for g in expr.add_groups + expr.sub_groups:
        _canonicalize_group_temporal_args(g)


def canonicalize_temporal_unit_args(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Canonicalize temporal-unit leading args of ``scalar_func_args`` across every expr in *intent*.

    Args:

        intent: Intent to mutate in place.

    Returns:

        Same *intent* instance.
    """
    for cte in intent.cte_steps or []:
        for sc in cte.select_cols or []:
            _canonicalize_expr_temporal_args(sc.expr)
        for g in cte.group_by_cols or []:
            _canonicalize_expr_temporal_args(g)
        for obc in cte.order_by_cols or []:
            _canonicalize_expr_temporal_args(obc.expr)
        for fp in cte.filters_param or []:
            _canonicalize_expr_temporal_args(fp.left_expr)
            if fp.right_expr:
                _canonicalize_expr_temporal_args(fp.right_expr)
        for hp in cte.having_param or []:
            _canonicalize_expr_temporal_args(hp.left_expr)
            if hp.right_expr:
                _canonicalize_expr_temporal_args(hp.right_expr)
    for sc in intent.select_cols or []:
        _canonicalize_expr_temporal_args(sc.expr)
    for g in intent.group_by_cols or []:
        _canonicalize_expr_temporal_args(g)
    for obc in intent.order_by_cols or []:
        _canonicalize_expr_temporal_args(obc.expr)
    for fp in intent.filters_param or []:
        _canonicalize_expr_temporal_args(fp.left_expr)
        if fp.right_expr:
            _canonicalize_expr_temporal_args(fp.right_expr)
    for hp in intent.having_param or []:
        _canonicalize_expr_temporal_args(hp.left_expr)
        if hp.right_expr:
            _canonicalize_expr_temporal_args(hp.right_expr)
    return intent


def _is_plain_column_expr(expr: NormalizedExpr) -> bool:
    """
    True for a single ``table.col`` term with no arithmetic.

    Args:

        expr: Expression to test.

    Returns:

        True if exactly one bare column MulGroup or a flat column-ref leaf.
    """
    if expr.sub_groups:
        return False
    if expr.add_values:
        return False
    if expr.column_ref and not expr.add_groups:
        return True
    if len(expr.add_groups) != 1:
        return False
    group = expr.add_groups[0]
    if group.agg_func or group.scalar_func or group.inner_scalar_func:
        return False
    if group.divide:
        return False
    if len(group.multiply) != 1:
        return False
    leaf = group.multiply[0]
    return bool(leaf.column_ref) and not leaf.add_groups and not leaf.sub_groups


def _reclassify_date_diff_param(
    fp: FilterParam,
) -> FilterParam:
    """
    If ``date_diff`` targets a plain column, retype to ``date_window`` preserving ``amount``.

    Args:

        fp: Candidate filter.

    Returns:

        Rewritten or original *fp*.
    """
    if fp.value_type != "date_diff":
        return fp
    if not isinstance(fp.raw_value, dict):
        return fp
    if not _is_plain_column_expr(fp.left_expr):
        return fp
    rv = fp.raw_value
    amount = rv.get("amount")
    if amount is None:
        return fp
    new_rv = {"unit": rv.get("unit", "day"), "amount": int(amount)}
    return replace(fp, value_type="date_window", raw_value=new_rv)


def repair_misclassified_date_diff(
    intent: RuntimeIntent,
) -> RuntimeIntent:
    """
    Apply ``_reclassify_date_diff_param`` to main and CTE filters.

    Args:

        intent: Intent to fix.

    Returns:

        Updated intent.
    """

    def _process(params: list[FilterParam]) -> list[FilterParam]:
        return [_reclassify_date_diff_param(fp) for fp in params]

    new_fp = _process(intent.filters_param or [])
    new_cte_steps = []
    for cte in intent.cte_steps or []:
        cte_fp = _process(cte.filters_param or [])
        new_cte_steps.append(replace(cte, filters_param=cte_fp))
    return replace(intent, filters_param=new_fp, cte_steps=new_cte_steps)


def concat_logical_intent_prose(logical: LogicalIntent) -> str:
    """
    Concatenate every planner prose field from the top intent and each CTE step.

    Args:

        logical: Parsed planner intent whose string fields should be flattened.

    Returns:

        Lowercase-friendly haystack text; empty fields are skipped so spacing stays stable.
    """

    chunks: list[str] = []
    for field in PLANNER_PROSE_FIELDS:
        chunks.append(str(getattr(logical, field, "") or ""))
    for step in logical.cte_steps:
        for field in PLANNER_PROSE_FIELDS:
            chunks.append(str(getattr(step, field, "") or ""))
    return " ".join(c for c in chunks if c.strip())


register_parse_expr_string(parse_expr_string)
