"""Execute and validate SQL, score confidence, validate CTE chains, and classify user rejections."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ._config import PolicyConfig
from ._constants import (
    DIAG_TO_FAILURE_CATEGORY,
    PERMISSION_DENIED_USER_MESSAGE,
    SCALAR_FUNCTIONS_NUMERIC,
    SCALAR_FUNCTIONS_STRING,
    SCALAR_FUNCTIONS_TEMPORAL,
    UNBOUND_PYFORMAT_PLACEHOLDER_RE,
    VALID_AGGREGATION_FUNCTIONS,
)
from ._contracts_base import (
    AccessError,
    ColumnRole,
    EngineContext,
    FailureCategory,
    HavingParam,
    LogicalIntent,
    OrderByCol,
    SqlDiagnostic,
    SqlDiagnosticCode,
)
from ._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from ._contracts_schema import (
    CaseRegistryStep,
    CteOutputColumnMeta,
    IntentIssue,
    IntentValidationResult,
    SchemaGraph,
    WindowRegistryStep,
    make_intent_issue,
)
from ._core_utils import (
    debug,
    pipeline_trace,
    reconcile_execute_bind_params,
    stable_json,
)
from ._dialect import Dialect
from ._intent_resolve import (
    check_qualified_refs_exist,
    collect_column_refs_for_cte_step,
    collect_column_refs_for_post_processing,
    resolve_column_map,
)
from ._schema_graph import assert_consumer_sql_in_scope
from ._validation_schema import (
    extract_agg_col,
    extract_col_from_scalar_wrapper,
    extract_functions_from_term,
    filter_param_to_having_param,
    get_col_type,
    iterate_case_branch_conditions,
    validate_case_when_schema,
    validate_contains_array_filters,
    validate_date_diff_left_expr_is_subtraction,
    validate_date_diff_units,
    validate_date_window_units,
    validate_filter_ops_per_column,
    validate_filter_value_type_alignment,
    validate_filters_schema,
    validate_group_by_cols_schema,
    validate_having_ops_per_column,
    validate_having_schema,
    validate_join_path_reachability,
    validate_join_path_reachability_for_tables,
    validate_no_between_ops,
    validate_null_filters,
    validate_order_by_cols_schema,
    validate_redundant_extract_year_column_literals,
    validate_scope_registries,
    validate_select_cols_schema,
    validate_selectability,
    validate_window_partition_group_by_alignment,
    validate_window_spec_schema,
)
from ._validation_semantic import (
    validate_agg_vs_agg_having,
    validate_arith_expression_semantics,
    validate_case_branch_aggregation_consistency,
    validate_concat_mulgroups_in_runtime,
    validate_count_threshold_missing_having,
    validate_cte_dependency_grains,
    validate_cte_grain_consistency,
    validate_cte_join_key_exposure,
    validate_denied_references,
    validate_deny_bare_select,
    validate_empty_window,
    validate_expr_vs_expr_filters,
    validate_filter_expr_types,
    validate_filter_no_aggregation,
    validate_for_each_grouping,
    validate_grain_consistency,
    validate_grouped_requires_aggregation,
    validate_having_expr_types,
    validate_having_operator_is_numeric,
    validate_having_requires_aggregation,
    validate_logical_intent_numeric_coverage,
    validate_mixed_aggregation_in_mulgroup,
    validate_no_nested_aggregation,
    validate_non_selectable_predicates,
    validate_order_by_aggregation_context,
    validate_order_by_expr_types,
    validate_predicate_bool_op_filter_group_hints,
    validate_predicate_sidedness,
    validate_question_agg_keyword_coverage,
    validate_question_distinct_hint,
    validate_select_expr_types,
    validate_select_group_by_membership,
    validate_semantic_contradictions,
    validate_sensitivity_group_by,
    validate_sensitivity_order_by,
    validate_threshold_missing_having,
)


def _scalar_subquery_cte_names(intent: RuntimeIntent | None) -> frozenset[str]:
    """Lowercased CTE names whose pipeline emission is ``scalar_subquery``."""
    if intent is None:
        return frozenset()
    names: list[str] = []
    for cte in intent.cte_steps or []:
        if getattr(cte, "emission", "") != "scalar_subquery":
            continue
        cn = (cte.cte_name or "").strip().lower()
        if cn:
            names.append(cn)
    return frozenset(names)


def _classify_explain_sql_failure(message: str) -> str:
    """Bucket database ``EXPLAIN`` failures for seed-warmup telemetry and policy. Returns one of ``explain_transient``, ``explain_schema``, ``explain_semantic``, or ``explain_failed``."""
    low = message.lower()
    transient_markers = (
        "timeout",
        "timed out",
        "statement_timeout",
        "query cancelled",
        "query canceled",
        "canceled",
        "cancelled",
        "connection reset",
        "could not connect",
        "connection refused",
        "temporarily unavailable",
        "server closed the connection",
        "broken pipe",
        "deadlock detected",
    )
    if any(x in low for x in transient_markers):
        return "explain_transient"
    schema_markers = (
        "does not exist",
        "undefined_column",
        "undefined column",
        "invalid catalog name",
        "unknown column",
        "column not found",
        "no such table",
        "relation ",
        "not found in catalog",
    )
    if any(x in low for x in schema_markers):
        return "explain_schema"
    semantic_markers = (
        "cannot cast",
        "invalid input syntax",
        "type mismatch",
        "operator does not exist",
        "division by zero",
    )
    if any(x in low for x in semantic_markers):
        return "explain_semantic"
    return "explain_failed"


def _classify_explain_error(dialect_name: str, exc_text: str) -> FailureCategory:
    """Map an ``EXPLAIN`` exception string to ``FailureCategory`` for. telemetry and outcomes."""
    _ = dialect_name
    code = _classify_explain_sql_failure(exc_text or "")
    if code == "explain_transient":
        return FailureCategory.EXECUTION_TIMEOUT
    if code == "explain_schema":
        return FailureCategory.EXECUTION_SCHEMA_ERROR
    if code == "explain_semantic":
        return FailureCategory.EXECUTION_SEMANTIC_ERROR
    return FailureCategory.EXECUTION_EXPLAIN_FAILED


def _format_explain_validation_error(code: str, raw: str) -> str:
    return f"[{code}] {raw}"


def _column_resolution_issues(
    columns: list[str],
    tables: list[str],
    schema: SchemaGraph,
    context: str,
) -> list[IntentIssue]:
    """Run ``resolve_column_map`` and prefix messages with *context* for non-main scopes."""
    _, raw = resolve_column_map(columns, schema, tables)
    if context == "main query":
        return raw
    out: list[IntentIssue] = []
    slug = re.sub(r"[^\w]+", "_", context).strip("_")
    for iss in raw:
        out.append(
            make_intent_issue(
                issue_id=f"{iss.issue_id}_{slug}",
                category=iss.category,
                severity=iss.severity,
                message=f"{context}: {iss.message}",
                context={**iss.context, "location": context},
            ),
        )
    return out


def _enforce_select_only(sql: str, dialect: Dialect) -> tuple[bool, str]:
    """Check whether SQL is a safe SELECT-only statement."""
    s = sql.lower().strip()
    for p in PolicyConfig.FORBIDDEN_SQL:
        if re.search(p, s, re.IGNORECASE):
            return False, "forbidden_sql"
    if dialect.parse_select(sql) is None:
        return False, "not_select"
    return True, "ok"


def validate_sql(
    dialect: Dialect,
    sql: str,
    params: dict[str, Any] | None = None,
    *,
    schema: SchemaGraph | None = None,
    intent: RuntimeIntent | None = None,
) -> tuple[bool, str | None, FailureCategory | None, list[SqlDiagnostic]]:
    """Validate SQL as a safe ``SELECT`` and syntactically valid via. the. dialect AST. ``EXPLAIN`` is always attempted via ``dialect.explain_sql`` when the dialect still has a usable backend; it is skipped only after the dialect has self-disabled EXPLAIN in response to a permission-denied response from the database (see :meth:`Dialect._disable_explain_on_permission_denied`)."""
    debug(f"[validation_execute.validate_sql] checking SQL length={len(sql)}")
    pipeline_trace(
        "validation_execute.validate_sql.input",
        lambda: stable_json({"sql": sql, "params": params or {}}),
    )
    if UNBOUND_PYFORMAT_PLACEHOLDER_RE.search(sql):
        debug("[validation_execute.validate_sql] unbound pyformat placeholder detected")
        pipeline_trace(
            "validation_execute.validate_sql.FAILED_unbound_placeholder",
            lambda: sql,
        )
        return (
            False,
            "unbound_placeholder",
            FailureCategory.UNBOUND_PLACEHOLDER,
            [
                SqlDiagnostic(
                    code=SqlDiagnosticCode.PARAM_UNBOUND,
                    message="unbound pyformat placeholder",
                )
            ],
        )
    ok, reason = _enforce_select_only(sql, dialect)
    if not ok:
        debug(f"[validation_execute.validate_sql] enforce_select_only FAILED: {reason}")
        pipeline_trace(
            "validation_execute.validate_sql.FAILED_enforce_select_only",
            lambda: f"reason={reason}\n{sql}",
        )
        cat = FailureCategory.SCHEMA if reason == "not_select" else FailureCategory.OTHER
        diag_code = SqlDiagnosticCode.NOT_SELECT if reason == "not_select" else SqlDiagnosticCode.MULTIPLE_STATEMENTS
        return False, reason, cat, [SqlDiagnostic(code=diag_code, message=reason or "")]
    declared_params = set(params.keys()) if params else set()
    scalar_cte_names = _scalar_subquery_cte_names(intent)
    ast_diags = dialect.ast_validate_full(
        sql,
        schema=schema,
        declared_params=declared_params,
        scalar_cte_names=scalar_cte_names,
    )
    if ast_diags:
        first = ast_diags[0]
        ast_err = first.message if first.message else first.code.value
        debug(f"[validation_execute.validate_sql] AST validation failed: {ast_err}")
        pipeline_trace(
            "validation_execute.validate_sql.FAILED_ast",
            lambda: f"ast_err={ast_err}\n{sql}",
        )
        return (
            False,
            f"SQL structure error: {ast_err}",
            FailureCategory(DIAG_TO_FAILURE_CATEGORY.get(first.code.value, FailureCategory.SCHEMA_VALIDATION.value)),
            list(ast_diags),
        )
    debug("[validation_execute.validate_sql] structural validation succeeded")
    explain_diags: list[SqlDiagnostic] = []
    if dialect.can_explain():
        ok, explain_diags, explain_err = dialect.explain_diagnose(sql, params, schema=schema, intent=intent)
        if not ok:
            raw = (explain_err or "").strip()
            if explain_diags and explain_diags[0].code == SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED:
                msg = explain_diags[0].message or raw
                return (
                    False,
                    msg,
                    FailureCategory.EXECUTION_COST_EXCEEDED,
                    list(explain_diags),
                )
            code = _classify_explain_sql_failure(raw)
            formatted = _format_explain_validation_error(code, raw)
            ecat = _classify_explain_error(getattr(dialect, "name", ""), raw)
            debug(f"[validation_execute.validate_sql] explain_sql failed ({code}): {explain_err}")
            pipeline_trace(
                "validation_execute.validate_sql.FAILED_explain",
                lambda: f"explain_err={explain_err}\n{sql}\nparams={stable_json(params or {})}",
            )
            hard_diags = (
                list(explain_diags)
                if explain_diags
                else [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=raw)]
            )
            return False, formatted, ecat, hard_diags
        debug("[validation_execute.validate_sql] explain_sql passed")
    pipeline_trace(
        "validation_execute.validate_sql.PASSED",
        lambda: sql,
    )
    return True, None, None, list(explain_diags)


def execute_guarded_sql(
    dialect: Dialect,
    sql: str,
    params: dict[str, Any] | None = None,
    *,
    schema: SchemaGraph | None = None,
    intent: RuntimeIntent | None = None,
    schema_role: str = "owner",
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Validate *sql* then execute via *dialect*, enforcing consumer scope when applicable."""
    if schema is not None:
        runtime_cfg_ctx = schema_context
        ctx = runtime_cfg_ctx if runtime_cfg_ctx is not None else EngineContext()
        gate_active = (
            schema_role == "consumer"
            or getattr(ctx, "name", "master") != "master"
            or bool(ctx.allow_objects or ctx.deny_objects or ctx.deny_columns or ctx.allow_columns)
            or visible_objects is not None
        )
        if gate_active and not assert_consumer_sql_in_scope(sql, dialect, ctx, schema, visible_objects):
            raise AccessError("execute", PERMISSION_DENIED_USER_MESSAGE)
    ok, err, _cat, _diags = validate_sql(
        dialect,
        sql,
        params,
        schema=schema,
        intent=intent,
    )
    if not ok:
        raise ValueError(err or "sql validation failed")
    exec_params = reconcile_execute_bind_params(sql, params)
    return dialect.execute(sql, exec_params)


def compute_confidence(
    best_score: float,
    score_gap: float,
    used_new_tables: bool,
    shape_penalty: float,
    negative_pen: float,
    colmap_pen: float,
    num_cte_pen: float = 0.0,
    explain_pen: float = 0.0,
) -> float:
    """Compute overall confidence score for a query result. Combines. template similarity score, gap to next-best, new-table penalty, shape distance penalty, negative memory penalty, column-map penalty, CTE count penalty, and EXPLAIN soft-diagnostic penalty into a single float."""
    debug("[validation_execute.compute_confidence] inputs:")
    debug(
        f"[validation_execute.compute_confidence] used_new_tables={used_new_tables}, shape_penalty={shape_penalty:.3f}, num_cte_pen={num_cte_pen:.3f}, explain_pen={explain_pen:.3f}"
    )
    c = 0.0
    c += 0.62 * best_score
    debug(f"[validation_execute.compute_confidence] +{0.62 * best_score:.3f} from best_score")
    gap_contrib = 0.18 * max(0.0, min(1.0, score_gap * 2.0))
    c += gap_contrib
    debug(f"[validation_execute.compute_confidence] +{gap_contrib:.3f} from score_gap")
    if used_new_tables:
        c -= 0.12
        debug("[validation_execute.compute_confidence] -0.120 from used_new_tables")
    shape_deduct = 0.18 * shape_penalty
    c -= shape_deduct
    debug(f"[validation_execute.compute_confidence] -{shape_deduct:.3f} from shape_penalty")
    neg_deduct = 0.35 * min(1.0, max(0.0, negative_pen))
    c -= neg_deduct
    debug(f"[validation_execute.compute_confidence] -{neg_deduct:.3f} from negative_pen")
    colmap_deduct = 0.20 * min(1.0, max(0.0, colmap_pen))
    c -= colmap_deduct
    debug(f"[validation_execute.compute_confidence] -{colmap_deduct:.3f} from colmap_pen")
    cte_deduct = 0.10 * min(1.0, max(0.0, num_cte_pen))
    c -= cte_deduct
    debug(f"[validation_execute.compute_confidence] -{cte_deduct:.3f} from num_cte_pen")
    explain_deduct = 0.08 * min(1.0, max(0.0, explain_pen))
    c -= explain_deduct
    debug(f"[validation_execute.compute_confidence] -{explain_deduct:.3f} from explain_pen")
    result = max(0.0, min(1.0, c))
    if best_score == 0.0:
        cold_start_floor = max(0.0, 0.50 - neg_deduct - colmap_deduct - cte_deduct - explain_deduct)
        if result < cold_start_floor:
            debug(
                f"[validation_execute.compute_confidence] cold-start floor applied: {result:.3f} → {cold_start_floor:.3f}"
            )
            result = cold_start_floor
    debug(f"[validation_execute.compute_confidence] FINAL confidence={result:.3f}")
    return result


def canonicalize_rejection_reason(text: str) -> str:
    """Normalise a rejection summary to a single line, max 160. characters, no trailing sentence punctuation."""
    s = " ".join((text or "").split())
    if not s:
        return ""
    s = s.splitlines()[0].strip()
    s = re.sub(r"[\s.?!,;:]+$", "", s)
    if len(s) > 160:
        s = s[:160].rstrip()
    return s


def _merge_cte_projection_columns_into_outputs(
    meta_by_col: dict[str, CteOutputColumnMeta],
    output_columns: list[str],
) -> None:
    """Ensure *meta_by_col* contains a permissive entry for every bare name in *output_columns*. LLM payloads sometimes omit ``output_column_metadata`` keys even when ``output_columns`` is authoritative; schema validators key off this map."""
    for oc in output_columns:
        bare = oc.split(".")[-1].strip()
        if not bare:
            continue
        if any(k.lower() == bare.lower() for k in meta_by_col):
            continue
        meta_by_col[bare] = CteOutputColumnMeta(source="output_column_projection")


def _cte_names_from_column_refs(refs: list[str], cte_names_lower: set[str]) -> set[str]:
    """Return lowercase CTE names that appear as qualified table prefixes in *refs*."""
    found: set[str] = set()
    for ref in refs:
        if "." not in ref:
            continue
        tbl = ref.split(".", 1)[0].strip().lower()
        if tbl in cte_names_lower:
            found.add(tbl)
    return found


def _validate_main_query_cte_usage(
    intent: RuntimeIntent,
    cte_outputs: dict[str, list[str]],
    cte_steps: list[RuntimeCteStep] | None = None,
) -> list[IntentIssue]:
    """Validate that the main query references CTE outputs correctly. Checks for unreferenced CTEs (computed via transitive closure through ``cte_steps[*].tables`` so a CTE consumed by another used CTE is itself considered used), main-query select columns not present in their referenced CTE outputs, ``column_map`` references missing from CTE outputs, and filter column references missing from CTE outputs."""
    issues: list[IntentIssue] = []
    debug(
        f"[validation_execute.validate_main_query_cte_usage] checking main query uses CTEs: {list(cte_outputs.keys())}"
    )
    if not cte_outputs:
        return issues
    intent_tables = set(t.lower() for t in (intent.tables or []))
    cte_names_lower = {c.lower() for c in cte_outputs.keys()}
    used: set[str] = intent_tables & cte_names_lower
    used |= _cte_names_from_column_refs(collect_column_refs_for_post_processing(intent), cte_names_lower)

    def _canonical_cte_key(nm: str) -> str | None:
        return next((x for x in cte_outputs if x.lower() == nm.lower()), None)

    def _cte_col_list(nm: str) -> list[str]:
        ck = _canonical_cte_key(nm)
        return list(cte_outputs.get(ck or "", []) or [])

    if cte_steps:
        steps_by_name = {c.cte_name.lower(): c for c in cte_steps}
        frontier = set(used)
        while frontier:
            nxt: set[str] = set()
            for name in frontier:
                step = steps_by_name.get(name)
                if step is None:
                    continue
                for t in step.tables or []:
                    tl = t.lower()
                    if tl in cte_names_lower and tl not in used:
                        nxt.add(tl)
                for tl in _cte_names_from_column_refs(collect_column_refs_for_cte_step(step), cte_names_lower):
                    if tl not in used:
                        nxt.add(tl)
            used |= nxt
            frontier = nxt
    unreferenced_ctes = cte_names_lower - used
    if unreferenced_ctes and cte_steps:
        steps_by_name = {c.cte_name.lower(): c for c in cte_steps if c.cte_name}
        for u in sorted(unreferenced_ctes):
            st = steps_by_name.get(u)
            em = getattr(st, "emission", "join_table") if st is not None else "join_table"
            sev = "error" if em == "scalar_subquery" else "warning"
            issues.append(
                make_intent_issue(
                    issue_id=f"cte_unreferenced_{u}",
                    category=FailureCategory.CTE_USAGE,
                    severity=sev,
                    message=f"CTE {u!r} is defined but not used in the main query or referenced CTE chain",
                    context={"unreferenced": [u], "emission": em},
                )
            )
        debug(f"[validation_execute.validate_main_query_cte_usage] unreferenced CTEs: {unreferenced_ctes}")
    elif unreferenced_ctes:
        issues.append(
            make_intent_issue(
                issue_id=f"cte_unreferenced_{','.join(sorted(unreferenced_ctes))}",
                category=FailureCategory.CTE_USAGE,
                severity="warning",
                message=f"CTEs defined but not used in main query: {sorted(unreferenced_ctes)}",
                context={"unreferenced": list(unreferenced_ctes)},
            )
        )
        debug(f"[validation_execute.validate_main_query_cte_usage] unreferenced CTEs: {unreferenced_ctes}")
    for sc in intent.select_cols or []:
        col_expr = sc.expr.primary_column
        if not col_expr or "." not in col_expr:
            continue
        table_ref, col_name = col_expr.rsplit(".", 1)
        if table_ref.lower() in cte_names_lower:
            cte_cols = _cte_col_list(table_ref)
            if col_name.lower() not in {c.lower() for c in cte_cols}:
                issues.append(
                    make_intent_issue(
                        issue_id=f"main_col_not_in_cte_{table_ref}_{col_name}",
                        category=FailureCategory.CTE_COLUMN_REFERENCE,
                        severity="error",
                        message=f"Main query references column '{col_name}' not in CTE '{table_ref}'",
                        context={
                            "cte": table_ref,
                            "column": col_name,
                            "available": cte_cols,
                        },
                    )
                )
                debug(f"[validation_execute.validate_main_query_cte_usage] column {col_name} not in CTE {table_ref}")
    column_map = intent.column_map or {}
    for col_name, source in column_map.items():
        if source.lower() in cte_names_lower:
            cte_cols = _cte_col_list(source)
            if col_name.lower() not in {c.lower() for c in cte_cols}:
                issues.append(
                    make_intent_issue(
                        issue_id=f"main_colmap_not_in_cte_{source}_{col_name}",
                        category=FailureCategory.CTE_COLUMN_REFERENCE,
                        severity="error",
                        message=f"column_map references '{col_name}' from CTE '{source}' but column not in CTE outputs",
                        context={
                            "cte": source,
                            "column": col_name,
                            "available": cte_cols,
                        },
                    )
                )
                debug(f"[validation_execute.validate_main_query_cte_usage] column_map {col_name} not in CTE {source}")
    for fp in intent.filters_param or []:
        col_expr = fp.left_expr.primary_column
        if not col_expr or "." not in col_expr:
            continue
        table_ref, col_name = col_expr.rsplit(".", 1)
        if table_ref.lower() in cte_names_lower:
            cte_cols = _cte_col_list(table_ref)
            if col_name.lower() not in {c.lower() for c in cte_cols}:
                issues.append(
                    make_intent_issue(
                        issue_id=f"main_filter_not_in_cte_{table_ref}_{col_name}",
                        category=FailureCategory.CTE_COLUMN_REFERENCE,
                        severity="error",
                        message=f"Main query filter references column '{col_name}' not in CTE '{table_ref}'",
                        context={
                            "cte": table_ref,
                            "column": col_name,
                            "available": cte_cols,
                        },
                    )
                )
                debug(f"[validation_execute.validate_main_query_cte_usage] filter {col_name} not in CTE {table_ref}")
    if issues:
        debug(f"[validation_execute.validate_main_query_cte_usage] found {len(issues)} issues")
    else:
        debug("[validation_execute.validate_main_query_cte_usage] all CTE references valid")
    return issues


def _validate_cte_output_types(cte_steps: list[RuntimeCteStep], schema: SchemaGraph) -> list[IntentIssue]:
    """Validate that CTE output column types are consistent with. aggregation usage. Warns when ``SUM`` or ``AVG`` is applied to a column whose inferred type is not numeric."""
    issues: list[IntentIssue] = []
    cte_output_types: dict[str, dict[str, str]] = {}
    debug(f"[validation_execute.validate_cte_output_types] validating {len(cte_steps)} CTE output types")
    for cte in cte_steps:
        cte_name = cte.cte_name
        output_cols = cte.output_columns or []
        select_cols = cte.select_cols or []
        col_types: dict[str, str] = {}
        for col in output_cols:
            table_name: str | None
            if "." in col:
                table_name, col_name = col.rsplit(".", 1)
            else:
                col_name = col
                table_name = cte.tables[0] if cte.tables else None
            for sc in select_cols:
                if sc.is_aggregated:
                    term = sc.expr.primary_term
                    agg_name = term.split("(")[0].lower() if "(" in term else ""
                    alias = f"{agg_name}_{sc.expr.primary_column.replace('.', '_')}"
                    if col_name.lower() == alias.lower() or col_name.lower().startswith(f"{agg_name}_"):
                        col_types[col_name] = "numeric"
                        break
            else:
                if table_name and table_name in schema.tables:
                    schema_col = schema.tables[table_name].columns.get(col_name)
                    if schema_col and schema_col.value_type:
                        col_types[col_name] = schema_col.value_type
                elif table_name and table_name in cte_output_types:
                    dep_types = cte_output_types[table_name]
                    if col_name in dep_types:
                        col_types[col_name] = dep_types[col_name]
        cte_output_types[cte_name] = col_types
    for cte in cte_steps:
        cte_name = cte.cte_name
        select_cols = cte.select_cols or []
        numeric_aggs = {"sum", "avg"}
        for sc in select_cols:
            if not sc.is_aggregated:
                continue
            term = sc.expr.primary_term
            agg_func = term.split("(")[0].lower() if "(" in term else ""
            if agg_func not in numeric_aggs:
                continue
            col_expr = sc.expr.primary_column
            if not col_expr:
                continue
            agg_table: str | None
            if "." in col_expr:
                agg_table, col_name = col_expr.rsplit(".", 1)
            else:
                col_name = col_expr
                agg_table = cte.tables[0] if cte.tables else None
            if agg_table in cte_output_types:
                col_type = cte_output_types[agg_table].get(col_name, "")
                if col_type and col_type not in ("integer", "number"):
                    issues.append(
                        make_intent_issue(
                            issue_id=f"cte_agg_type_mismatch_{cte_name}_{agg_func}_{col_name}",
                            category=FailureCategory.CTE_TYPE_CONSISTENCY,
                            severity="warning",
                            message=f"CTE '{cte_name}' applies {agg_func.upper()} to non-numeric column '{col_name}' (type: {col_type})",
                            context={
                                "cte_name": cte_name,
                                "agg": agg_func,
                                "column": col_name,
                                "type": col_type,
                            },
                        )
                    )
                    debug(
                        f"[validation_execute.validate_cte_output_types] type mismatch: {agg_func} on {col_name}({col_type})"
                    )
    if issues:
        debug(f"[validation_execute.validate_cte_output_types] found {len(issues)} type issues")
    else:
        debug("[validation_execute.validate_cte_output_types] all CTE output types valid")
    return issues


def _validate_cte_cardinality(cte_steps: list[RuntimeCteStep]) -> list[IntentIssue]:
    """Validate CTE cardinality expectations are internally consistent. Warns when scalar-grain CTEs have ``expected_rows != 'one'``, ``LIMIT 1`` CTEs have ``expected_rows != 'one'``, and notes when a many-row CTE depends on a single-row CTE."""
    issues: list[IntentIssue] = []
    cte_expected_rows: dict[str, str] = {}
    debug(f"[validation_execute.validate_cte_cardinality] validating {len(cte_steps)} CTE cardinalities")
    for cte in cte_steps:
        cte_expected_rows[cte.cte_name] = cte.expected_rows or "many"
    for cte in cte_steps:
        cte_name = cte.cte_name
        expected = cte.expected_rows or "many"
        grain = cte.grain
        if grain == "scalar" and expected != "one":
            issues.append(
                make_intent_issue(
                    issue_id=f"cte_scalar_cardinality_{cte_name}",
                    category=FailureCategory.CTE_CARDINALITY,
                    severity="warning",
                    message=f"CTE '{cte_name}' has scalar grain but expected_rows='{expected}'",
                    context={
                        "cte_name": cte_name,
                        "grain": grain,
                        "expected_rows": expected,
                    },
                )
            )
            debug(f"[validation_execute.validate_cte_cardinality] scalar CTE with expected_rows={expected}")
        if cte.limit == 1 and expected != "one":
            issues.append(
                make_intent_issue(
                    issue_id=f"cte_limit1_cardinality_{cte_name}",
                    category=FailureCategory.CTE_CARDINALITY,
                    severity="warning",
                    message=f"CTE '{cte_name}' has LIMIT 1 but expected_rows='{expected}'",
                    context={
                        "cte_name": cte_name,
                        "limit": 1,
                        "expected_rows": expected,
                    },
                )
            )
            debug(f"[validation_execute.validate_cte_cardinality] LIMIT 1 CTE with expected_rows={expected}")
        for table in cte.tables or []:
            if table in cte_expected_rows:
                dep_expected = cte_expected_rows[table]
                if expected in {"few", "many"} and dep_expected == "one":
                    debug(
                        f"[validation_execute.validate_cte_cardinality] cardinality note dropped: {cte_name}({expected}) <- {table}({dep_expected})"
                    )
    if issues:
        debug(f"[validation_execute.validate_cte_cardinality] found {len(issues)} cardinality issues")
    else:
        debug("[validation_execute.validate_cte_cardinality] all cardinalities valid")
    return issues


def _validate_case_branches_for_scope(
    *,
    select_cols: list[SelectCol],
    case_registry: list[CaseRegistryStep] | None,
    window_registry: list[WindowRegistryStep] | None,
    schema: SchemaGraph,
    allowed_tables: set[str],
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]],
    cte_steps: list[RuntimeCteStep],
    location_prefix: str,
    param_values: Mapping[str, Any] | None,
) -> list[IntentIssue]:
    """Apply the filter/HAVING-shaped validators against every CASE. branch condition. Walks ``case_registry[*].case_when`` branch conditions via :func:`iterate_case_branch_conditions`. For each branch the helper routes the condition through the same validators that the main and CTE bodies use for ``filters_param`` (when ``condition_scope`` is ``"filter"``) or ``having_param`` (when ``condition_scope`` is ``"having"``). Each branch is treated as a one-element list so that operator, type-alignment, NULL, array, and date-unit rules surface inside CASE branches with the same diagnostics as flat WHERE/HAVING predicates. Flat ``filters_param`` / ``having_param`` lists still run ``validate_no_between_ops`` at body scope so decomposable ``between`` rows are caught there; CASE branch conditions skip that check because each branch holds a single ``FilterParam`` and shares the CASE renderer path for ``between``, ``in``, and ``not in`` (:func:`aetherdialect._sql_gen._render_case_branch_sql`)."""
    issues: list[IntentIssue] = []
    for cond, branch_scope, location in iterate_case_branch_conditions(
        select_cols, case_registry, window_registry, location_prefix
    ):
        f_list = [cond]
        if branch_scope == "having":
            h_list = [filter_param_to_having_param(cond)]
            issues.extend(
                validate_having_schema(
                    h_list,
                    schema,
                    allowed_tables,
                    cte_outputs,
                    location,
                    param_values=param_values,
                )
            )
            issues.extend(validate_having_ops_per_column(h_list, schema, cte_outputs, location))
            issues.extend(validate_having_agg_per_role(h_list, schema, cte_outputs, location))
            issues.extend(validate_having_expr_types(h_list, schema, cte_outputs, location))
            issues.extend(validate_having_operator_is_numeric(h_list, location))
            issues.extend(validate_having_requires_aggregation(h_list, location, group_by_cols=[]))
            issues.extend(validate_agg_vs_agg_having(h_list, schema, cte_outputs, location))
            continue
        issues.extend(
            validate_filters_schema(
                f_list,
                schema,
                allowed_tables,
                cte_outputs,
                location,
                param_values=param_values,
            )
        )
        issues.extend(validate_filter_ops_per_column(f_list, schema, cte_outputs, location))
        issues.extend(validate_filter_no_aggregation(f_list, location))
        issues.extend(validate_filter_expr_types(f_list, schema, cte_outputs, location))
        issues.extend(validate_null_filters(f_list, cte_steps, location))
        issues.extend(
            validate_filter_value_type_alignment(f_list, schema, location, cte_outputs, param_values=param_values)
        )
        issues.extend(validate_contains_array_filters(f_list, schema, cte_outputs, location))
        issues.extend(validate_date_window_units(f_list, cte_steps, location, scope_param_values=param_values))
        issues.extend(validate_date_diff_units(f_list, cte_steps, location, scope_param_values=param_values))
        issues.extend(validate_date_diff_left_expr_is_subtraction(f_list, cte_steps, location))
        issues.extend(validate_expr_vs_expr_filters(f_list, schema, cte_outputs, location))
        issues.extend(validate_predicate_sidedness(f_list, [], location))
    return issues


def validate_semantics(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    *,
    post_binding: bool = False,
    numeric_coverage_logical: LogicalIntent | None = None,
) -> IntentValidationResult:
    """Run the full semantic validation suite against a. ``RuntimeIntent``. Validators are grouped by tier. Tier 1 structural checks run on the main query body and again on each CTE body with that scope's ``tables`` and ``param_values``. Tier 2 NL-dependent checks use ``intent.natural_language`` for the main query and ``cte.description`` for CTE scopes. Tier 3 cross-scope checks run once on the full intent (CTE naming, dependency grains, and related rules). Applies every schema-level and semantic-level validation function to the main query and each CTE step, then aggregates CTE grain compatibility checks across the full CTE chain."""
    all_issues: list[IntentIssue] = []
    debug(f"[validation_execute.validate_semantics] post_binding={post_binding}")
    debug("[validation_execute.validate_semantics] running semantic validation suite")
    pipeline_trace(
        "validation_execute.validate_semantics.input_intent",
        lambda: stable_json(intent.to_dict()),
    )
    if not intent.tables:
        all_issues.append(
            make_intent_issue(
                issue_id="tables_empty",
                category=FailureCategory.STRUCTURAL,
                severity="error",
                message="Intent has no tables specified",
                context={},
            )
        )
        debug("[validation_execute.validate_semantics] tables empty")
    cte_steps = intent.cte_steps or []
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] = {}
    for cte in cte_steps:
        if not cte.cte_name:
            all_issues.append(
                make_intent_issue(
                    issue_id="cte_name_empty",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message="CTE step has no name specified",
                    context={},
                )
            )
        if not cte.output_columns:
            all_issues.append(
                make_intent_issue(
                    issue_id=f"cte_output_columns_empty_{cte.cte_name}",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message=f"CTE '{cte.cte_name}' has no output columns specified",
                    context={"cte_name": cte.cte_name},
                )
            )
        else:
            cte_outputs[cte.cte_name] = dict(cte.output_column_metadata or {})
            _merge_cte_projection_columns_into_outputs(cte_outputs[cte.cte_name], list(cte.output_columns or []))
    if cte_steps:
        cte_names = [c.cte_name for c in cte_steps]
        if len(cte_names) != len(set(cte_names)):
            duplicates = [n for n in cte_names if cte_names.count(n) > 1]
            all_issues.append(
                make_intent_issue(
                    issue_id=f"cte_duplicate_names_{','.join(sorted(set(duplicates)))}",
                    category=FailureCategory.CTE_STRUCTURE,
                    severity="error",
                    message=f"Duplicate CTE names: {sorted(set(duplicates))}",
                    context={"duplicates": list(set(duplicates))},
                )
            )
        for i, cte in enumerate(cte_steps):
            for table in cte.tables or []:
                if table.lower() in {n.lower() for n in cte_names[i + 1 :]}:
                    all_issues.append(
                        make_intent_issue(
                            issue_id=f"cte_forward_ref_{cte.cte_name}_{table}",
                            category=FailureCategory.CTE_STRUCTURE,
                            severity="error",
                            message=f"CTE '{cte.cte_name}' forward-references CTE '{table}' defined later",
                            context={"cte_name": cte.cte_name, "forward_ref": table},
                        )
                    )
        known_tables = set(schema.tables.keys())
        for i, cte in enumerate(cte_steps):
            available = known_tables | {c.cte_name for c in cte_steps[:i]}
            for table in cte.tables or []:
                if table.lower() not in {t.lower() for t in available}:
                    all_issues.append(
                        make_intent_issue(
                            issue_id=f"cte_unknown_table_{cte.cte_name}_{table}",
                            category=FailureCategory.CTE_TABLE_REFERENCE,
                            severity="error",
                            message=f"CTE '{cte.cte_name}' references unknown table '{table}'",
                            context={"cte_name": cte.cte_name, "table": table},
                        )
                    )
        cte_outputs_list = {c.cte_name: (c.output_columns or []) for c in cte_steps}
        all_issues.extend(_validate_main_query_cte_usage(intent, cte_outputs_list, cte_steps))
        all_issues.extend(_validate_cte_output_types(cte_steps, schema))
        all_issues.extend(_validate_cte_cardinality(cte_steps))
    allowed_tables = set(intent.tables or [])
    all_issues.extend(
        validate_scope_registries(
            context="main query",
            window_registry=list(intent.window_registry or []),
            case_registry=list(intent.case_registry or []),
            select_cols=intent.select_cols or [],
            group_by_cols=intent.group_by_cols or [],
            order_by_cols=intent.order_by_cols or [],
            filters_param=intent.filters_param or [],
            having_param=intent.having_param or [],
        )
    )
    all_issues.extend(
        validate_select_cols_schema(
            intent.select_cols or [],
            schema,
            allowed_tables,
            cte_outputs,
            "main query",
            window_registry=list(intent.window_registry or []),
            case_registry=list(intent.case_registry or []),
        )
    )
    all_issues.extend(validate_deny_bare_select(intent, schema))
    all_issues.extend(validate_denied_references(intent, schema))
    all_issues.extend(validate_sensitivity_group_by(intent, schema))
    all_issues.extend(validate_sensitivity_order_by(intent, schema))
    all_issues.extend(validate_non_selectable_predicates(intent, schema))
    all_issues.extend(validate_empty_window(intent, schema))
    all_issues.extend(
        validate_window_spec_schema(
            intent.select_cols or [],
            schema,
            cte_outputs,
            "main query",
            window_registry=list(intent.window_registry or []),
            case_registry=list(intent.case_registry or []),
        )
    )
    all_issues.extend(
        validate_window_partition_group_by_alignment(
            grain=intent.grain or "row_level",
            group_by_cols=intent.group_by_cols or [],
            window_registry=list(intent.window_registry or []),
            context="main query",
        )
    )
    all_issues.extend(
        validate_case_when_schema(
            intent.select_cols or [],
            schema,
            allowed_tables,
            cte_outputs,
            "main query",
            window_registry=list(intent.window_registry or []),
            case_registry=list(intent.case_registry or []),
            param_values=intent.param_values,
        )
    )
    all_issues.extend(validate_contains_array_filters(intent.filters_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(
        validate_selectability(
            intent.select_cols or [],
            schema,
            cte_outputs,
            "main query",
            window_registry=list(intent.window_registry or []),
            case_registry=list(intent.case_registry or []),
        )
    )
    all_issues.extend(validate_join_path_reachability(intent, schema, "main query"))
    all_issues.extend(
        validate_order_by_cols_schema(
            intent.order_by_cols or [],
            schema,
            allowed_tables,
            cte_outputs,
            "main query",
        )
    )
    all_issues.extend(
        validate_group_by_cols_schema(
            intent.group_by_cols or [],
            schema,
            allowed_tables,
            cte_outputs,
            "main query",
        )
    )
    all_issues.extend(
        validate_filters_schema(
            intent.filters_param or [],
            schema,
            allowed_tables,
            cte_outputs,
            "main query",
            param_values=intent.param_values,
        )
    )
    all_issues.extend(
        _column_resolution_issues(
            collect_column_refs_for_post_processing(intent),
            intent.tables or [],
            schema,
            "main query",
        )
    )
    all_issues.extend(
        validate_having_schema(
            intent.having_param or [],
            schema,
            allowed_tables,
            cte_outputs,
            "main query",
            param_values=intent.param_values,
        )
    )
    all_issues.extend(validate_filter_ops_per_column(intent.filters_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_having_agg_per_role(intent.having_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_having_ops_per_column(intent.having_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_select_agg_per_role(intent.select_cols or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_select_agg_semantics(intent.select_cols or [], schema, "main query"))
    all_issues.extend(validate_order_by_agg_per_role(intent.order_by_cols or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_order_by_agg_semantics(intent.order_by_cols or [], schema, "main query"))
    all_issues.extend(
        validate_scalar_func_type_semantics(intent.select_cols or [], intent.order_by_cols or [], schema, "main query")
    )
    all_issues.extend(validate_null_filters(intent.filters_param or [], cte_steps, "main query"))
    all_issues.extend(
        validate_date_window_units(
            intent.filters_param or [],
            cte_steps,
            "main query",
            scope_param_values=intent.param_values,
        )
    )
    all_issues.extend(
        validate_date_diff_units(
            intent.filters_param or [],
            cte_steps,
            "main query",
            scope_param_values=intent.param_values,
        )
    )
    all_issues.extend(
        validate_redundant_extract_year_column_literals(
            intent.filters_param or [],
            cte_steps,
            "main query",
        )
    )
    all_issues.extend(validate_column_types(intent.select_cols or [], schema, "main query"))
    all_issues.extend(
        validate_filter_value_type_alignment(
            intent.filters_param or [],
            schema,
            "main query",
            cte_outputs,
            param_values=intent.param_values,
        )
    )
    all_issues.extend(validate_no_between_ops(intent.filters_param or [], intent.having_param or [], "main query"))
    all_issues.extend(
        validate_grain_consistency(
            intent.grain,
            intent.select_cols or [],
            intent.group_by_cols or [],
            intent.having_param or [],
            "main query",
        )
    )
    all_issues.extend(
        validate_grouped_requires_aggregation(
            intent.grain,
            intent.select_cols or [],
            intent.group_by_cols or [],
            "main query",
            having_param=intent.having_param or [],
        )
    )
    all_issues.extend(
        validate_case_branch_aggregation_consistency(
            intent.case_registry,
            intent.group_by_cols or [],
            "main query",
        )
    )
    all_issues.extend(
        validate_semantic_contradictions(
            intent.select_cols or [],
            intent.natural_language,
            intent.grain,
            intent.expected_rows,
            "main query",
        )
    )
    all_issues.extend(
        validate_predicate_bool_op_filter_group_hints(
            intent.natural_language,
            intent.filters_param or [],
            intent.having_param or [],
            "main query",
        )
    )
    all_issues.extend(
        validate_threshold_missing_having(
            intent.natural_language,
            intent.select_cols or [],
            intent.having_param or [],
            intent.grain,
            "main query",
        )
    )
    all_issues.extend(
        validate_count_threshold_missing_having(
            intent.natural_language,
            intent.tables or [],
            intent.having_param or [],
            schema,
            "main query",
        )
    )
    all_issues.extend(
        validate_logical_intent_numeric_coverage(
            numeric_coverage_logical,
            intent.filters_param or [],
            intent.having_param or [],
            intent.limit,
            "main query",
            param_values=intent.param_values,
            case_registry=intent.case_registry,
        )
    )
    all_issues.extend(
        validate_question_distinct_hint(
            intent.natural_language,
            intent.select_cols or [],
            "main query",
            distinct_select_index=intent.distinct_select_index,
        )
    )
    all_issues.extend(
        validate_question_agg_keyword_coverage(
            intent.natural_language,
            intent.select_cols or [],
            intent.having_param or [],
            "main query",
            intent.cte_steps or [],
        )
    )
    _has_agg = any(sc.is_aggregated for sc in (intent.select_cols or [])) or bool(intent.having_param)
    all_issues.extend(
        validate_for_each_grouping(
            intent.natural_language,
            intent.group_by_cols or [],
            schema,
            _has_agg,
            "main query",
        )
    )
    all_issues.extend(validate_expr_vs_expr_filters(intent.filters_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_agg_vs_agg_having(intent.having_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(
        _validate_case_branches_for_scope(
            select_cols=intent.select_cols or [],
            case_registry=list(intent.case_registry or []),
            window_registry=list(intent.window_registry or []),
            schema=schema,
            allowed_tables=allowed_tables,
            cte_outputs=cte_outputs,
            cte_steps=cte_steps,
            location_prefix="main query",
            param_values=intent.param_values,
        )
    )
    all_issues.extend(validate_scalar_expression_semantics(intent.select_cols or [], schema, "main query"))
    all_issues.extend(
        validate_arith_expression_semantics(
            intent.filters_param or [],
            intent.having_param or [],
            schema,
            cte_outputs,
            "main query",
        )
    )
    all_issues.extend(validate_concat_mulgroups_in_runtime(intent, "main query"))
    all_issues.extend(validate_temporal_columns(intent.select_cols or [], schema, "main query"))
    all_issues.extend(validate_pk_fk_aggregation(intent.select_cols or [], schema, "main query"))
    all_issues.extend(validate_select_expr_types(intent.select_cols or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_order_by_expr_types(intent.order_by_cols or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_filter_expr_types(intent.filters_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_having_expr_types(intent.having_param or [], schema, cte_outputs, "main query"))
    all_issues.extend(validate_filter_no_aggregation(intent.filters_param or [], "main query"))
    all_issues.extend(
        validate_having_requires_aggregation(
            intent.having_param or [],
            "main query",
            group_by_cols=intent.group_by_cols or [],
        )
    )
    all_issues.extend(validate_having_operator_is_numeric(intent.having_param or [], "main query"))
    all_issues.extend(validate_predicate_sidedness(intent.filters_param or [], intent.having_param or [], "main query"))
    all_issues.extend(
        validate_no_nested_aggregation(
            intent.select_cols or [],
            intent.order_by_cols or [],
            intent.filters_param or [],
            intent.having_param or [],
            "main query",
        )
    )
    all_issues.extend(
        validate_mixed_aggregation_in_mulgroup(
            intent.select_cols or [],
            intent.order_by_cols or [],
            intent.filters_param or [],
            intent.having_param or [],
            "main query",
        )
    )
    all_issues.extend(validate_order_by_aggregation_context(intent.order_by_cols or [], intent.grain, "main query"))
    all_issues.extend(
        validate_select_group_by_membership(
            intent.select_cols or [],
            intent.group_by_cols or [],
            intent.grain,
            "main query",
        )
    )
    for cte in cte_steps:
        cte_context = f"CTE '{cte.cte_name}'"
        cte_allowed = set(cte.tables or [])
        all_issues.extend(
            validate_scope_registries(
                context=cte_context,
                window_registry=list(cte.window_registry or []),
                case_registry=list(cte.case_registry or []),
                select_cols=cte.select_cols or [],
                group_by_cols=cte.group_by_cols or [],
                order_by_cols=cte.order_by_cols or [],
                filters_param=cte.filters_param or [],
                having_param=cte.having_param or [],
            )
        )
        all_issues.extend(
            validate_select_cols_schema(
                cte.select_cols or [],
                schema,
                cte_allowed,
                cte_outputs,
                cte_context,
                window_registry=list(cte.window_registry or []),
                case_registry=list(cte.case_registry or []),
            )
        )
        all_issues.extend(
            validate_window_spec_schema(
                cte.select_cols or [],
                schema,
                cte_outputs,
                cte_context,
                window_registry=list(cte.window_registry or []),
                case_registry=list(cte.case_registry or []),
            )
        )
        all_issues.extend(
            validate_window_partition_group_by_alignment(
                grain=cte.grain or "row_level",
                group_by_cols=cte.group_by_cols or [],
                window_registry=list(cte.window_registry or []),
                context=cte_context,
            )
        )
        all_issues.extend(
            validate_case_when_schema(
                cte.select_cols or [],
                schema,
                cte_allowed,
                cte_outputs,
                cte_context,
                window_registry=list(cte.window_registry or []),
                case_registry=list(cte.case_registry or []),
                param_values=cte.param_values,
            )
        )
        all_issues.extend(validate_contains_array_filters(cte.filters_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(
            validate_selectability(
                cte.select_cols or [],
                schema,
                cte_outputs,
                cte_context,
                window_registry=list(cte.window_registry or []),
                case_registry=list(cte.case_registry or []),
            )
        )
        all_issues.extend(validate_join_path_reachability_for_tables(cte.tables or [], schema, cte_context))
        all_issues.extend(
            validate_order_by_cols_schema(cte.order_by_cols or [], schema, cte_allowed, cte_outputs, cte_context)
        )
        all_issues.extend(
            validate_group_by_cols_schema(cte.group_by_cols or [], schema, cte_allowed, cte_outputs, cte_context)
        )
        all_issues.extend(
            validate_filters_schema(
                cte.filters_param or [],
                schema,
                cte_allowed,
                cte_outputs,
                cte_context,
                param_values=cte.param_values,
            )
        )
        all_issues.extend(
            _column_resolution_issues(
                collect_column_refs_for_cte_step(cte),
                cte.tables or [],
                schema,
                cte_context,
            )
        )
        all_issues.extend(
            validate_having_schema(
                cte.having_param or [],
                schema,
                cte_allowed,
                cte_outputs,
                cte_context,
                param_values=cte.param_values,
            )
        )
        all_issues.extend(validate_filter_ops_per_column(cte.filters_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(
            validate_date_window_units(
                cte.filters_param or [],
                [],
                cte_context,
                scope_param_values=cte.param_values,
            )
        )
        all_issues.extend(
            validate_date_diff_units(
                cte.filters_param or [],
                [],
                cte_context,
                scope_param_values=cte.param_values,
            )
        )
        all_issues.extend(validate_having_agg_per_role(cte.having_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_having_ops_per_column(cte.having_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_select_agg_per_role(cte.select_cols or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_select_agg_semantics(cte.select_cols or [], schema, cte_context))
        all_issues.extend(validate_order_by_agg_per_role(cte.order_by_cols or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_order_by_agg_semantics(cte.order_by_cols or [], schema, cte_context))
        all_issues.extend(
            validate_scalar_func_type_semantics(cte.select_cols or [], cte.order_by_cols or [], schema, cte_context)
        )
        all_issues.extend(validate_column_types(cte.select_cols or [], schema, cte_context))
        all_issues.extend(
            validate_filter_value_type_alignment(
                cte.filters_param or [],
                schema,
                cte_context,
                cte_outputs,
                param_values=cte.param_values,
            )
        )
        all_issues.extend(validate_no_between_ops(cte.filters_param or [], cte.having_param or [], cte_context))
        all_issues.extend(validate_cte_grain_consistency(cte, cte_context))
        all_issues.extend(
            validate_grouped_requires_aggregation(
                cte.grain,
                cte.select_cols or [],
                cte.group_by_cols or [],
                cte_context,
                having_param=cte.having_param or [],
            )
        )
        all_issues.extend(
            validate_case_branch_aggregation_consistency(
                getattr(cte, "case_registry", None),
                cte.group_by_cols or [],
                cte_context,
            )
        )
        all_issues.extend(
            validate_semantic_contradictions(
                cte.select_cols or [],
                cte.description or "",
                cte.grain,
                cte.expected_rows,
                cte_context,
            )
        )
        all_issues.extend(
            validate_predicate_bool_op_filter_group_hints(
                cte.description or "",
                cte.filters_param or [],
                cte.having_param or [],
                cte_context,
            )
        )
        all_issues.extend(validate_expr_vs_expr_filters(cte.filters_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_agg_vs_agg_having(cte.having_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(
            _validate_case_branches_for_scope(
                select_cols=cte.select_cols or [],
                case_registry=list(cte.case_registry or []),
                window_registry=list(cte.window_registry or []),
                schema=schema,
                allowed_tables=cte_allowed,
                cte_outputs=cte_outputs,
                cte_steps=cte_steps,
                location_prefix=cte_context,
                param_values=cte.param_values,
            )
        )
        all_issues.extend(validate_scalar_expression_semantics(cte.select_cols or [], schema, cte_context))
        all_issues.extend(
            validate_arith_expression_semantics(
                cte.filters_param or [],
                cte.having_param or [],
                schema,
                cte_outputs,
                cte_context,
            )
        )
        all_issues.extend(validate_temporal_columns(cte.select_cols or [], schema, cte_context))
        all_issues.extend(validate_pk_fk_aggregation(cte.select_cols or [], schema, cte_context))
        all_issues.extend(validate_select_expr_types(cte.select_cols or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_order_by_expr_types(cte.order_by_cols or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_filter_expr_types(cte.filters_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_having_expr_types(cte.having_param or [], schema, cte_outputs, cte_context))
        all_issues.extend(validate_filter_no_aggregation(cte.filters_param or [], cte_context))
        all_issues.extend(
            validate_having_requires_aggregation(
                cte.having_param or [],
                cte_context,
                group_by_cols=cte.group_by_cols or [],
            )
        )
        all_issues.extend(validate_having_operator_is_numeric(cte.having_param or [], cte_context))
        all_issues.extend(validate_predicate_sidedness(cte.filters_param or [], cte.having_param or [], cte_context))
        all_issues.extend(
            validate_no_nested_aggregation(
                cte.select_cols or [],
                cte.order_by_cols or [],
                cte.filters_param or [],
                cte.having_param or [],
                cte_context,
            )
        )
        all_issues.extend(
            validate_mixed_aggregation_in_mulgroup(
                cte.select_cols or [],
                cte.order_by_cols or [],
                cte.filters_param or [],
                cte.having_param or [],
                cte_context,
            )
        )
        all_issues.extend(validate_order_by_aggregation_context(cte.order_by_cols or [], cte.grain, cte_context))
        all_issues.extend(
            validate_select_group_by_membership(cte.select_cols or [], cte.group_by_cols or [], cte.grain, cte_context)
        )
        if cte.output_columns:
            cte_outputs[cte.cte_name] = dict(cte.output_column_metadata or {})
            _merge_cte_projection_columns_into_outputs(cte_outputs[cte.cte_name], list(cte.output_columns))
    if cte_steps:
        all_issues.extend(
            validate_cte_dependency_grains(
                cte_steps,
                intent.grain,
                main_tables=intent.tables or [],
                select_cols=intent.select_cols or [],
            )
        )
        all_issues.extend(validate_cte_join_key_exposure(intent))
    if all_issues:
        debug(f"[validation_execute.validate_semantics] found {len(all_issues)} total issues")
    else:
        debug("[validation_execute.validate_semantics] all validations passed")
    for idx, iss in enumerate(all_issues):
        debug(
            f"[validation_execute.validate_semantics] issue[{idx}]: "
            f"{iss.issue_id} | {iss.category} | {iss.severity} | {iss.message} | context={iss.context}"
        )

        def _issue_trace_body(issue: IntentIssue = iss) -> str:
            return stable_json(issue.to_dict())

        pipeline_trace(
            f"validation_execute.validate_semantics.issue[{idx}]",
            _issue_trace_body,
        )
    result = IntentValidationResult(issues=all_issues)
    pipeline_trace(
        "validation_execute.validate_semantics.result",
        lambda: stable_json(result.to_dict()),
    )
    return result


def curated_warmup_semantic_issues(intent: RuntimeIntent, schema: SchemaGraph) -> list[str]:
    """Semantic checks aligned with live pipeline validation after deterministic repairs and lite post-processing. Delegates to :func:`validate_semantics` with ``post_binding=True`` so NL-conditioned rules stay disabled while the full schema-tier suite runs (including scope-registries via that suite)."""
    vr = validate_semantics(intent, schema, post_binding=True)
    return list(
        dict.fromkeys(i.message for i in vr.issues if (i.severity or "").lower() == "error"),
    )


def curated_warmup_post_binding_issues(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    final_sql: str,
) -> list[str]:
    """Post-substitution parity checks mirroring :func:`_post_processing_revalidation_passes` without LLM use."""
    _ = final_sql
    msgs: list[str] = []
    _, qerr = check_qualified_refs_exist(intent, schema)
    msgs.extend(qerr)
    vr = validate_semantics(intent, schema, post_binding=True)
    msgs.extend(i.message for i in vr.issues if (i.severity or "").lower() == "error")
    return list(dict.fromkeys(msgs))


def validate_having_agg_per_role(
    having_param: list[HavingParam],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that HAVING aggregation functions are valid for each. column's role."""
    issues = []
    if not having_param:
        return []
    cte_outputs = cte_outputs or {}
    for hp in having_param:
        agg_expr = hp.left_expr.primary_term
        if not agg_expr:
            continue
        func, actual_target, _ = extract_agg_col(agg_expr)
        if not func or not actual_target or actual_target == "*":
            continue
        if "." not in actual_target:
            continue
        table_name, col_name = actual_target.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                if cte_meta.valid_aggregations and func not in cte_meta.valid_aggregations:
                    issues.append(
                        make_intent_issue(
                            issue_id=f"having_agg_invalid_for_cte_{context}_{actual_target}_{func}",
                            category=FailureCategory.HAVING_VALIDITY,
                            severity="error",
                            message=f"Aggregation '{func.upper()}' not valid for CTE column '{actual_target}' (role={cte_meta.role}) in HAVING for {context}. Valid: {sorted(cte_meta.valid_aggregations)}",
                            context={
                                "column": actual_target,
                                "function": func,
                                "role": cte_meta.role,
                                "valid_aggs": sorted(cte_meta.valid_aggregations),
                                "location": context,
                            },
                        )
                    )
            continue
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        valid_aggs = col_meta.get_valid_aggregations()
        if func not in valid_aggs:
            issues.append(
                make_intent_issue(
                    issue_id=f"having_agg_invalid_for_role_{context}_{actual_target}_{func}",
                    category=FailureCategory.HAVING_VALIDITY,
                    severity="error",
                    message=f"Aggregation '{func.upper()}' not valid for column '{actual_target}' (role={col_meta.role}) in HAVING for {context}. Valid: {sorted(valid_aggs)}",
                    context={
                        "column": actual_target,
                        "function": func,
                        "role": col_meta.role,
                        "valid_aggs": sorted(valid_aggs),
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_having_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_select_agg_per_role(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that SELECT aggregation functions are valid for each. column's role."""
    issues = []
    if not select_cols:
        return []
    cte_outputs = cte_outputs or {}
    for sc in select_cols:
        _, agg_func = extract_functions_from_term(sc.expr.primary_term)
        if not agg_func:
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                if cte_meta.valid_aggregations:
                    func_lower = agg_func.lower()
                    if func_lower not in cte_meta.valid_aggregations:
                        issues.append(
                            make_intent_issue(
                                issue_id=f"select_agg_invalid_for_cte_{context}_{actual_col}_{agg_func}",
                                category=FailureCategory.AGGREGATION_VALIDITY,
                                severity="error",
                                message=f"Aggregation '{agg_func.upper()}' not valid for CTE column '{actual_col}' (role={cte_meta.role}) in {context}. Valid: {sorted(cte_meta.valid_aggregations)}",
                                context={
                                    "column": actual_col,
                                    "function": agg_func,
                                    "role": cte_meta.role,
                                    "valid_aggs": sorted(cte_meta.valid_aggregations),
                                    "location": context,
                                },
                            )
                        )
            continue
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        valid_aggs = col_meta.get_valid_aggregations()
        func_lower = agg_func.lower()
        if func_lower not in valid_aggs:
            issues.append(
                make_intent_issue(
                    issue_id=f"select_agg_invalid_for_role_{context}_{actual_col}_{agg_func}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"Aggregation '{agg_func.upper()}' not valid for column '{actual_col}' (role={col_meta.role}) in {context}. Valid: {sorted(valid_aggs)}",
                    context={
                        "column": actual_col,
                        "function": agg_func,
                        "role": col_meta.role,
                        "valid_aggs": sorted(valid_aggs),
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_select_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_select_agg_semantics(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that SELECT aggregation functions are semantically. appropriate for column types. Errors for SUM/AVG on non-numeric columns; warns for MIN/MAX on FREE_TEXT columns."""
    issues = []
    if not select_cols:
        return []
    numeric_aggs = {"sum", "avg"}
    for sc in select_cols:
        _, agg_func = extract_functions_from_term(sc.expr.primary_term)
        if not agg_func:
            continue
        func_lower = agg_func
        if func_lower not in numeric_aggs and func_lower not in {"min", "max"}:
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        col_meta = schema.tables[table_name].columns.get(col_name)
        if not col_meta:
            continue
        vt = col_meta.value_type
        numeric = vt in ("integer", "number")
        temporal = vt == "date"
        if func_lower in numeric_aggs and not numeric:
            issues.append(
                make_intent_issue(
                    issue_id=f"invalid_agg_semantics_{func_lower}_{table_name}_{col_name}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="error",
                    message=f"Cannot {func_lower.upper()} on {actual_col} (type={col_meta.data_type}): {func_lower.upper()} requires numeric column",
                    context={
                        "aggregation": func_lower,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_schema.validate_select_agg_semantics] invalid {func_lower.upper()} on {actual_col}")
        elif func_lower in {"min", "max"} and not numeric and not temporal:
            col_role = col_meta.role if col_meta.role else None
            if col_role == ColumnRole.FREE_TEXT.value:
                issues.append(
                    make_intent_issue(
                        issue_id=f"questionable_agg_{func_lower}_{table_name}_{col_name}",
                        category=FailureCategory.AGGREGATION_SEMANTICS,
                        severity="warning",
                        message=f"Questionable {func_lower.upper()} on {actual_col} (type={col_meta.data_type}): {func_lower.upper()} on free text is semantically meaningless",
                        context={
                            "aggregation": func_lower,
                            "column": actual_col,
                            "data_type": col_meta.data_type,
                            "location": context,
                        },
                    )
                )
                debug(
                    f"[validation_schema.validate_select_agg_semantics] questionable {func_lower.upper()} on {actual_col}"
                )
    if issues:
        debug(f"[validation_schema.validate_select_agg_semantics] found {len(issues)} semantic issues")
    else:
        debug("[validation_schema.validate_select_agg_semantics] no semantic issues")
    return issues


def validate_order_by_agg_per_role(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    cte_outputs: dict[str, dict[str, CteOutputColumnMeta]] | None = None,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that ORDER BY aggregation functions are valid for each. column's role."""
    issues = []
    if not order_by_cols:
        return []
    cte_outputs = cte_outputs or {}
    for obc in order_by_cols:
        _, agg_func = extract_functions_from_term(obc.expr.primary_term)
        if not agg_func:
            continue
        col_expr = obc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in cte_outputs:
            cte_cols = cte_outputs[table_name]
            matched_key = next((c for c in cte_cols if c.lower() == col_name.lower()), None)
            if matched_key:
                cte_meta = cte_cols[matched_key]
                if cte_meta.valid_aggregations:
                    func_lower = agg_func.lower()
                    if func_lower not in cte_meta.valid_aggregations:
                        issues.append(
                            make_intent_issue(
                                issue_id=f"order_by_agg_invalid_for_cte_{context}_{actual_col}_{agg_func}",
                                category=FailureCategory.AGGREGATION_VALIDITY,
                                severity="error",
                                message=f"Aggregation '{agg_func.upper()}' not valid for CTE column '{actual_col}' (role={cte_meta.role}) in order_by for {context}. Valid: {sorted(cte_meta.valid_aggregations)}",
                                context={
                                    "column": actual_col,
                                    "function": agg_func,
                                    "role": cte_meta.role,
                                    "valid_aggs": sorted(cte_meta.valid_aggregations),
                                    "location": context,
                                },
                            )
                        )
            continue
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        valid_aggs = col_meta.get_valid_aggregations()
        func_lower = agg_func.lower()
        if func_lower not in valid_aggs:
            issues.append(
                make_intent_issue(
                    issue_id=f"order_by_agg_invalid_for_role_{context}_{actual_col}_{agg_func}",
                    category=FailureCategory.AGGREGATION_VALIDITY,
                    severity="error",
                    message=f"Aggregation '{agg_func.upper()}' not valid for column '{actual_col}' (role={col_meta.role}) in order_by for {context}. Valid: {sorted(valid_aggs)}",
                    context={
                        "column": actual_col,
                        "function": agg_func,
                        "role": col_meta.role,
                        "valid_aggs": sorted(valid_aggs),
                        "location": context,
                    },
                )
            )
    debug(f"[validation_schema.validate_order_by_agg_per_role] {len(issues)} issues in {context}")
    return issues


def validate_order_by_agg_semantics(
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that ORDER BY aggregation functions are semantically. appropriate for column types. Errors for SUM/AVG on non-numeric columns; warns for MIN/MAX on FREE_TEXT columns."""
    issues = []
    if not order_by_cols:
        return []
    numeric_aggs = {"sum", "avg"}
    for obc in order_by_cols:
        _, agg_func = extract_functions_from_term(obc.expr.primary_term)
        if not agg_func:
            continue
        func_lower = agg_func
        if func_lower not in numeric_aggs and func_lower not in {"min", "max"}:
            continue
        col_expr = obc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if actual_col == "*":
            continue
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        col_meta = schema.tables[table_name].columns.get(col_name)
        if not col_meta:
            continue
        vt = col_meta.value_type
        numeric = vt in ("integer", "number")
        temporal = vt == "date"
        if func_lower in numeric_aggs and not numeric:
            issues.append(
                make_intent_issue(
                    issue_id=f"invalid_order_by_agg_semantics_{func_lower}_{table_name}_{col_name}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="error",
                    message=f"Cannot {func_lower.upper()} on {actual_col} (type={col_meta.data_type}) in ORDER BY: {func_lower.upper()} requires numeric column",
                    context={
                        "aggregation": func_lower,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_schema.validate_order_by_agg_semantics] invalid {func_lower.upper()} on {actual_col}")
        elif func_lower in {"min", "max"} and not numeric and not temporal:
            col_role = col_meta.role if col_meta.role else None
            if col_role == ColumnRole.FREE_TEXT.value:
                issues.append(
                    make_intent_issue(
                        issue_id=f"questionable_order_by_agg_{func_lower}_{table_name}_{col_name}",
                        category=FailureCategory.AGGREGATION_SEMANTICS,
                        severity="warning",
                        message=f"Questionable {func_lower.upper()} on {actual_col} (type={col_meta.data_type}) in ORDER BY: {func_lower.upper()} on free text is semantically meaningless",
                        context={
                            "aggregation": func_lower,
                            "column": actual_col,
                            "data_type": col_meta.data_type,
                            "location": context,
                        },
                    )
                )
                debug(
                    f"[validation_schema.validate_order_by_agg_semantics] questionable {func_lower.upper()} on {actual_col}"
                )
    if issues:
        debug(f"[validation_schema.validate_order_by_agg_semantics] found {len(issues)} semantic issues")
    else:
        debug("[validation_schema.validate_order_by_agg_semantics] no semantic issues")
    return issues


def validate_scalar_func_type_semantics(
    select_cols: list[SelectCol],
    order_by_cols: list[OrderByCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that scalar functions are appropriate for column types. and aggregation context. Errors when a non-aggregate-compatible scalar wraps an aggregation, or when a type-specific scalar (string, numeric, temporal) is applied to the wrong column type."""
    issues = []

    def check_scalar_semantics(
        scalar_func: str, col_expr: str, agg_func: str | None, location: str
    ) -> list[IntentIssue]:
        """
        Check scalar function semantics for one term.

        Args:

            scalar_func: Outer scalar function name.
            col_expr: Column expression string.
            agg_func: Inner aggregation function if present.
            location: Location label for issues.

        Returns:

            List of `IntentIssue` objects for violations.
        """
        inner_issues = []
        func_lower = scalar_func.lower()
        if agg_func and func_lower not in SCALAR_FUNCTIONS_NUMERIC:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_on_agg_invalid_{location}_{func_lower}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' cannot wrap aggregation '{agg_func.upper()}' in {location}. Only {sorted(SCALAR_FUNCTIONS_NUMERIC)} allowed on aggregates",
                    context={
                        "scalar": scalar_func,
                        "aggregation": agg_func,
                        "location": location,
                        "allowed": sorted(SCALAR_FUNCTIONS_NUMERIC),
                    },
                )
            )
            return inner_issues
        if agg_func:
            return inner_issues
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if not actual_col or "." not in actual_col or actual_col == "*":
            return inner_issues
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            return inner_issues
        col_meta = schema.tables[table_name].columns.get(col_name) or schema.tables[table_name].columns.get(
            col_name.lower()
        )
        if not col_meta:
            return inner_issues
        vt = col_meta.value_type
        string = vt == "string"
        numeric = vt in ("integer", "number")
        temporal = vt == "date"
        if func_lower in SCALAR_FUNCTIONS_STRING and not string:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_type_mismatch_{location}_{func_lower}_{actual_col}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' requires string column, got '{actual_col}' (type={col_meta.data_type}) in {location}",
                    context={
                        "scalar": scalar_func,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "expected_type": "string",
                        "location": location,
                    },
                )
            )
        elif func_lower in SCALAR_FUNCTIONS_NUMERIC and not numeric:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_type_mismatch_{location}_{func_lower}_{actual_col}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' requires numeric column, got '{actual_col}' (type={col_meta.data_type}) in {location}",
                    context={
                        "scalar": scalar_func,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "expected_type": "numeric",
                        "location": location,
                    },
                )
            )
        elif func_lower in SCALAR_FUNCTIONS_TEMPORAL and not temporal:
            inner_issues.append(
                make_intent_issue(
                    issue_id=f"scalar_type_mismatch_{location}_{func_lower}_{actual_col}",
                    category=FailureCategory.SCALAR_SEMANTICS,
                    severity="error",
                    message=f"Scalar '{scalar_func}' requires temporal column, got '{actual_col}' (type={col_meta.data_type}) in {location}",
                    context={
                        "scalar": scalar_func,
                        "column": actual_col,
                        "data_type": col_meta.data_type,
                        "expected_type": "date/timestamp",
                        "location": location,
                    },
                )
            )
        return inner_issues

    for idx, sc in enumerate(select_cols or []):
        sc_scalar, sc_agg = extract_functions_from_term(sc.expr.primary_term)
        if sc_scalar:
            issues.extend(check_scalar_semantics(sc_scalar, sc.expr.primary_column, sc_agg, f"select_cols[{idx}]"))
    for idx, obc in enumerate(order_by_cols or []):
        obc_scalar, obc_agg = extract_functions_from_term(obc.expr.primary_term)
        if obc_scalar:
            issues.extend(
                check_scalar_semantics(
                    obc_scalar,
                    obc.expr.primary_column,
                    obc_agg,
                    f"order_by_cols[{idx}]",
                )
            )
    if issues:
        debug(
            f"[validation_schema.validate_scalar_func_type_semantics] found {len(issues)} semantic issues in {context}"
        )
    else:
        debug(f"[validation_schema.validate_scalar_func_type_semantics] no semantic issues in {context}")
    return issues


def validate_column_types(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that operations match their column types (heuristic. checks). Warns for numeric aggregations on text columns, date operations on non-date columns, and string operations on numeric columns."""
    issues = []
    debug("[validation_schema.validate_column_types] checking type consistency")
    numeric_aggs = {"sum", "avg", "average", "total", "mean"}
    date_ops = {"latest", "earliest", "recent", "oldest", "newest", "before", "after"}
    string_ops = {"contains", "starts", "ends", "like", "match"}
    for sc in select_cols:
        _, agg_func = extract_functions_from_term(sc.expr.primary_term)
        if not agg_func:
            continue
        func_lower = agg_func
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        table_meta = schema.tables[table_name]
        col_meta = table_meta.columns.get(col_name) or table_meta.columns.get(col_name.lower())
        if not col_meta:
            continue
        vt = col_meta.value_type
        if vt:
            numeric = vt in ("integer", "number")
            date = vt == "date"
            text = vt == "string"
        else:
            numeric = any(
                hint in col_name.lower()
                for hint in [
                    "amount",
                    "price",
                    "total",
                    "count",
                    "qty",
                    "quantity",
                    "rate",
                    "cost",
                    "num",
                ]
            )
            date = any(
                hint in col_name.lower()
                for hint in [
                    "date",
                    "time",
                    "created",
                    "updated",
                    "at",
                    "day",
                    "year",
                    "month",
                ]
            )
            text = any(
                hint in col_name.lower() for hint in ["name", "title", "description", "email", "address", "text"]
            )
        if func_lower in numeric_aggs and text and not numeric:
            issues.append(
                make_intent_issue(
                    issue_id=f"numeric_on_text_{table_name}_{col_name}",
                    category=FailureCategory.TYPE_MISMATCH,
                    severity="warning",
                    message=f"Attempting numeric aggregation ({func_lower}) on text column '{col_name}' (type: {col_meta.data_type})",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "type": col_meta.data_type,
                        "agg": func_lower,
                        "location": context,
                    },
                )
            )
            debug("[validation_schema.validate_column_types] type_mismatch: numeric_on_text")
        if func_lower in date_ops and not date:
            issues.append(
                make_intent_issue(
                    issue_id=f"date_on_non_date_{table_name}_{col_name}",
                    category=FailureCategory.TYPE_MISMATCH,
                    severity="warning",
                    message=f"Attempting date operation ({func_lower}) on non-date column '{col_name}' (type: {col_meta.data_type})",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "type": col_meta.data_type,
                        "op": func_lower,
                        "location": context,
                    },
                )
            )
            debug("[validation_schema.validate_column_types] type_mismatch: date_on_non_date")
        if func_lower in string_ops and numeric and "_id" not in col_name.lower():
            issues.append(
                make_intent_issue(
                    issue_id=f"string_on_numeric_{table_name}_{col_name}",
                    category=FailureCategory.TYPE_MISMATCH,
                    severity="warning",
                    message=f"Attempting string operation ({func_lower}) on numeric column '{col_name}' (type: {col_meta.data_type})",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "type": col_meta.data_type,
                        "op": func_lower,
                        "location": context,
                    },
                )
            )
            debug("[validation_schema.validate_column_types] TYPE MISMATCH: string op on numeric column")
    if issues:
        debug(f"[validation_schema.validate_column_types] FAILED with {len(issues)} issues")
    else:
        debug("[validation_schema.validate_column_types] PASSED")
    return issues


def validate_scalar_expression_semantics(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that scalar functions are applied to semantically. appropriate column types."""
    issues = []
    debug("[validation_semantic.validate_scalar_expression_semantics] checking scalar semantics")
    numeric_scalars = {"abs", "round", "ceil", "floor", "sqrt"}
    string_scalars = {"upper", "lower", "trim", "ltrim", "rtrim", "length"}
    for sc in select_cols:
        outer_func, _, _ = extract_agg_col(sc.expr.primary_term)
        if not outer_func or outer_func in VALID_AGGREGATION_FUNCTIONS:
            continue
        func_lower = outer_func
        col_type = get_col_type(sc.expr.primary_column, schema, {})
        if col_type:
            numeric = col_type in ("integer", "number")
            text = col_type == "string"
            if func_lower in numeric_scalars and not numeric and not sc.is_aggregated:
                issues.append(
                    make_intent_issue(
                        issue_id=f"numeric_scalar_on_non_numeric_{sc.expr.primary_column}_{func_lower}",
                        category=FailureCategory.SCALAR_SEMANTIC,
                        severity="warning",
                        message=f"Numeric scalar '{func_lower}' on non-numeric column '{sc.expr.primary_column}' (type: {col_type})",
                        context={
                            "column": sc.expr.primary_column,
                            "scalar": func_lower,
                            "type": col_type,
                            "location": context,
                        },
                    )
                )
            if func_lower in string_scalars and not text:
                issues.append(
                    make_intent_issue(
                        issue_id=f"string_scalar_on_non_string_{sc.expr.primary_column}_{func_lower}",
                        category=FailureCategory.SCALAR_SEMANTIC,
                        severity="warning",
                        message=f"String scalar '{func_lower}' on non-string column '{sc.expr.primary_column}' (type: {col_type})",
                        context={
                            "column": sc.expr.primary_column,
                            "scalar": func_lower,
                            "type": col_type,
                            "location": context,
                        },
                    )
                )
    debug(f"[validation_semantic.validate_scalar_expression_semantics] {len(issues)} issues in {context}")
    return issues


def validate_temporal_columns(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate temporal aggregates against date-type columns in the. intent."""
    issues = []
    temporal_ops = {"latest", "recent", "last", "first", "earliest", "oldest", "newest"}
    agg_funcs: set[str] = set()
    for sc in select_cols:
        if not sc.is_aggregated:
            continue
        fn, _, _ = extract_agg_col(sc.expr.primary_term)
        if fn:
            agg_funcs.add(fn)
    if not (agg_funcs & temporal_ops):
        return []
    debug("[validation_semantic.validate_temporal_columns] checking temporal column presence")
    has_date_column = False
    for sc in select_cols:
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name in schema.tables:
            col_meta = schema.tables[table_name].columns.get(col_name)
            if col_meta:
                if col_meta.value_type == "date":
                    has_date_column = True
                    break
        if any(hint in col_name.lower() for hint in ["date", "time", "created", "updated", "at"]):
            has_date_column = True
            break
    if not has_date_column:
        issues.append(
            make_intent_issue(
                issue_id=f"temporal_no_date_col_{','.join(sorted(agg_funcs & temporal_ops))}",
                category=FailureCategory.MISSING_TEMPORAL_COLUMN,
                severity="warning",
                message=f"Intent uses temporal operation ({agg_funcs & temporal_ops}) but no date/time column identified",
                context={
                    "temporal_ops": list(agg_funcs & temporal_ops),
                    "location": context,
                },
            )
        )
        debug("[validation_semantic.validate_temporal_columns] AMBIGUITY: temporal ops but no date column")
    return issues


def validate_pk_fk_aggregation(
    select_cols: list[SelectCol],
    schema: SchemaGraph,
    context: str = "main",
) -> list[IntentIssue]:
    """Validate that primary-key and foreign-key columns are not. aggregated with SUM or AVG."""
    issues = []
    suspicious_aggs = {"sum", "avg"}
    debug("[validation_semantic.validate_pk_fk_aggregation] checking PK/FK aggregation")
    for sc in select_cols:
        if not sc.is_aggregated:
            continue
        func_lower, _, _ = extract_agg_col(sc.expr.primary_term)
        if not func_lower or func_lower not in suspicious_aggs:
            continue
        col_expr = sc.expr.primary_column
        if not col_expr:
            continue
        actual_col = extract_col_from_scalar_wrapper(col_expr)
        if "." not in actual_col:
            continue
        table_name, col_name = actual_col.rsplit(".", 1)
        if table_name not in schema.tables:
            continue
        col_meta = schema.tables[table_name].columns.get(col_name)
        if col_meta and (col_meta.is_primary_key or col_meta.is_foreign_key):
            issues.append(
                make_intent_issue(
                    issue_id=f"agg_on_pk_fk_{table_name}_{col_name}_{func_lower}",
                    category=FailureCategory.AGGREGATION_SEMANTICS,
                    severity="warning",
                    message=f"{func_lower.upper()} on PK/FK column {actual_col} is suspicious",
                    context={
                        "table": table_name,
                        "column": col_name,
                        "agg": func_lower,
                        "location": context,
                    },
                )
            )
            debug(f"[validation_semantic.validate_pk_fk_aggregation] {func_lower.upper()} on PK/FK: {actual_col}")
    return issues
