"""LLM-backed QSim: fill skeletons, parse intents, generate questions, and adaptive skeleton selection."""

from __future__ import annotations

import math
import random
from typing import Any

from ._config import (
    AGG_PATTERN,
    TABLE_COL_PATTERN,
    VALID_FILTER_VALUE_TYPES,
    VALID_HAVING_OPS,
    VALID_HAVING_VALUE_TYPES,
    PolicyConfig,
    QSimConfig,
)
from ._contracts_base import (
    QSIM_COMPLEXITY_TIER_SPECS,
    QSIM_SUPPORTED_ADVANCED_FEATURES,
    DatabaseFeatureCapability,
    LlmJsonExhausted,
    QSimSkeleton,
    RetryFailureContext,
    SchemaGraph,
    SkeletonPool,
    rebalance_complexity_target_proportions,
)
from ._contracts_core import (
    ComplexityTier,
    QSimFilter,
    QSimHaving,
    QSimIntent,
    classify_qsim_intent_complexity,
    classify_qsim_skeleton_complexity,
    qsim_intent_matches_target_tier,
)
from ._core_utils import debug, llm_json
from ._qsim import (
    build_fk_adjacency,
    build_schema_context,
    compute_intent_id,
    decompose_between_filter,
    enumerate_table_sets,
    generate_all_skeletons,
    get_aggregatable_columns,
    get_comparable_column_pairs,
    get_filterable_columns,
    get_groupable_columns,
    is_connected,
    load_or_create_skeletons,
    validate_column_exists,
)
from ._utils import generate_question

_QSIM_FILL_SYSTEM = (
    "You are a SQL intent generator."
    " Given structural skeleton constraints, generate a valid query intent filling in specific columns and filters from the available schema."
    " Rules:"
    " 1. STRICTLY follow skeleton constraints for tables, aggregation presence, filter count, groupby count, having, orderby."
    " 2. Use ONLY columns from the provided schema in the specified tables."
    " 3. For filters, choose columns with meaningful filter potential (status, category, date columns)."
    " 4. For aggregation, use COUNT/SUM/AVG/MIN/MAX wrapping table.column in select_cols; aggregate numeric columns only (not IDs or foreign keys); COUNT may use any column or *."
    " 5. For groupby, choose categorical or temporal columns that make semantic sense."
    " 6. Ensure all column references use table.column format from the specified tables."
    " 7. Return ONLY valid JSON matching the specified format."
    " 8. For expr_comparison (filter expr-vs-expr), both expressions must be from different tables with compatible types."
    " 9. DISTINCT is only valid for non-aggregated queries."
    " 10. For orderby, include ASC or DESC direction suffix."
    " 11. DO NOT return columns or tables not in the provided schema."
    " 12. For having, expression must be an aggregation matching a select_cols aggregation."
)


def _allocate_tier_int_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    """
    Convert fractional tier weights into nonnegative integers summing to *total*.

    Args:

        weights: Normalized tier weights keyed by string tier names.

        total: Total intents budget.

    Returns:

        Integer quota per tier key.
    """

    keys = [
        ComplexityTier.SIMPLE.value,
        ComplexityTier.MODERATE.value,
        ComplexityTier.COMPLEX.value,
        ComplexityTier.HIGHLY_COMPLEX.value,
    ]
    raw = [max(0.0, float(weights.get(k, 0.0))) * float(total) for k in keys]
    floors = [int(math.floor(r + 1e-12)) for r in raw]
    rem = int(total) - sum(floors)
    order = sorted(range(len(keys)), key=lambda i: raw[i] - floors[i], reverse=True)
    for j in range(max(0, rem)):
        floors[order[j % len(order)]] += 1
    return dict(zip(keys, floors, strict=False))


def _skeleton_bucket_lookup_key(target: ComplexityTier) -> str:
    """
    Map a sampled target tier to skeleton buckets derived from :func:`classify_qsim_skeleton_complexity`.

    Args:

        target: Tier quota chosen for this draw.

    Returns:

        Skeleton bucket key string (highly complex draws reuse complex skeleton shapes).
    """

    if target == ComplexityTier.HIGHLY_COMPLEX:
        return ComplexityTier.COMPLEX.value
    return target.value


def _tier_spec_lines(target: ComplexityTier) -> tuple[str, str]:
    """
    Resolve summary and example sketch strings for a tier target.

    Args:

        target: Sampled complexity tier.

    Returns:

        ``(summary, example_sketch)`` tuple for prompt insertion.
    """

    for spec in QSIM_COMPLEXITY_TIER_SPECS:
        if spec.tier == target:
            return spec.summary, spec.example_sketch
    return "", ""


def _advanced_feature_allowed(feature_id: str, cap: DatabaseFeatureCapability) -> bool:
    """
    Return whether an advanced feature remains plausible on this schema snapshot.

    Args:

        feature_id: Stable feature identifier from :data:`QSIM_SUPPORTED_ADVANCED_FEATURES`.

        cap: Database capability snapshot.

    Returns:

        False when prerequisite columns or join depth are absent.
    """

    if feature_id in ("date_window_filter", "date_diff_shapes"):
        return cap.has_date_columns
    if feature_id == "unnest_array_column":
        return cap.has_array_columns
    if feature_id in ("multi_cte_chain", "scalar_cte_bridge", "self_join_via_cte"):
        return cap.fk_edge_count >= 1 and cap.max_fk_chain_depth >= 1
    if feature_id == "window_partition_order":
        return cap.has_window_capable_table_sets
    if feature_id == "case_when_select":
        return cap.has_categorical_columns
    if feature_id in ("having_aggregate_compare",):
        return cap.has_numeric_measures
    return True


def _advanced_feature_prompt_block(cap: DatabaseFeatureCapability) -> str:
    """
    Format capability-filtered advanced feature bullets for the skeleton-fill prompt.

    Args:

        cap: Schema-derived feasibility snapshot.

    Returns:

        Bullet text listing allowed advanced shapes only.
    """

    lines: list[str] = []
    for spec in QSIM_SUPPORTED_ADVANCED_FEATURES:
        if not _advanced_feature_allowed(spec.feature_id, cap):
            continue
        lines.append(f"- {spec.feature_id}: {spec.summary} Example: {spec.example_fragment}")
    return "\n".join(lines) if lines else "(none feasible on this schema)"


def _sample_select_col_count_geometric(p: float, rng: random.Random, cap_cols: int = 12) -> int:
    """
    Sample a SELECT-column count biased toward small integers.

    Args:

        p: Success probability per Bernoulli trial tail.

        rng: Deterministic RNG for reproducibility.

        cap_cols: Upper bound on returned count.

    Returns:

        Positive integer column target.
    """

    n = 1
    while n < cap_cols and rng.random() > p:
        n += 1
    return n


def _build_merged_tier_buckets(
    schema: SchemaGraph,
    column_roles: dict[str, str],
) -> dict[str, list[tuple[QSimSkeleton, list[str]]]]:
    """
    Flatten A/B/C skeleton tiers into complexity buckets for weighted sampling.

    Args:

        schema: Schema graph.

        column_roles: Column role hints from profiling.

    Returns:

        Mapping tier value -> skeleton plus concrete table-set list.
    """

    merged: dict[str, list[tuple[QSimSkeleton, list[str]]]] = {
        ComplexityTier.SIMPLE.value: [],
        ComplexityTier.MODERATE.value: [],
        ComplexityTier.COMPLEX.value: [],
        ComplexityTier.HIGHLY_COMPLEX.value: [],
    }
    for nt in (1, 2, 3):
        pool = _build_skeleton_pool(schema, column_roles, num_tables=nt)
        for tk in pool.table_set_keys:
            ts = tk.split("|")
            for tier_dict in (
                pool.tier_a_by_table_set,
                pool.tier_b_by_table_set,
                pool.tier_c_by_table_set,
            ):
                for skel in tier_dict[tk]:
                    ct = classify_qsim_skeleton_complexity(skel)
                    merged[ct.value].append((skel, ts))
    for k in merged:
        random.shuffle(merged[k])
    return merged


def _pop_matching_skeleton(
    bucket: list[tuple[QSimSkeleton, list[str]]],
    need_filters: bool,
    need_having: bool,
) -> tuple[QSimSkeleton, list[str]] | None:
    """
    Pop the next skeleton from *bucket* honoring filter and HAVING coverage needs.

    Args:

        bucket: Mutable list of skeleton candidates.

        need_filters: When True, require ``num_filters > 0``.

        need_having: When True, require ``num_having > 0``.

    Returns:

        Removed skeleton entry, or None when no candidate matches.
    """

    for i, (sk, ts) in enumerate(bucket):
        if need_filters and sk.num_filters == 0:
            continue
        if need_having and sk.num_having == 0:
            continue
        bucket.pop(i)
        return sk, ts
    return None


def _pick_weighted_tier(tier_remaining: dict[str, int], rng: random.Random) -> str | None:
    """
    Sample the next tier to fill using remaining quota counts as weights.

    Args:

        tier_remaining: Mutable tier quotas.

        rng: Deterministic RNG.

    Returns:

        Tier key with positive remainder, or None when exhausted.
    """

    active = [(k, v) for k, v in tier_remaining.items() if v > 0]
    if not active:
        return None
    keys = [a[0] for a in active]
    weights = [a[1] for a in active]
    return str(rng.choices(keys, weights=weights, k=1)[0])


def _has_aggregation(select_cols: list[str]) -> bool:
    """
    Return True if any select column string matches an aggregation pattern.

    Args:

        select_cols: SQL select-column strings to inspect.

    Returns:

        True if at least one string matches `AGG(...)`, else False.
    """
    return any(AGG_PATTERN.match(sc) for sc in select_cols)


def _extract_agg_info(expr: str) -> tuple[str, str] | None:
    """
    Extract aggregation function and inner column from a SQL aggregation expression.

    Args:

        expr: Expression such as `COUNT(table.col)` or `SUM(table.amount)`.

    Returns:

        contents, or None if not an aggregation.
    """
    m = AGG_PATTERN.match(expr.strip())
    if m:
        return (m.group(1).lower(), m.group(2).strip())
    return None


def _extract_tables_from_expr(expr: str) -> set[str]:
    """
    Extract table names from a SQL expression containing `table.column` references.

    Args:

        expr: SQL expression that may contain `table.column` tokens.

    Returns:

        Set of table names found in `expr`.
    """
    return {m.group(1) for m in TABLE_COL_PATTERN.finditer(expr)}


def _validate_skeleton_constraints(response: dict[str, Any], skeleton: QSimSkeleton) -> tuple[bool, list[str]]:
    """
    Validate an LLM response dict against structural skeleton constraints.

    Args:

        response: Description.

        skeleton: `QSimSkeleton` whose constraints the response must satisfy.

    Returns:

        constraint failures.
    """
    violations = []

    select_cols_raw = response.get("select_cols", [])
    has_agg = any(AGG_PATTERN.match(sc) for sc in select_cols_raw if isinstance(sc, str))

    if skeleton.has_aggregation and not has_agg:
        violations.append("skeleton requires aggregation but no aggregated select_cols found")
    if not skeleton.has_aggregation and has_agg:
        violations.append("skeleton forbids aggregation but aggregated select_cols found")

    filters = response.get("filters", [])
    if skeleton.num_filters > 0 and len(filters) == 0 and not skeleton.has_expr_comparison:
        violations.append(f"skeleton requires {skeleton.num_filters} filters but got 0")

    groupby = response.get("groupby_cols", [])
    if skeleton.num_groupby > 0 and len(groupby) == 0:
        violations.append(f"skeleton requires {skeleton.num_groupby} groupby but got 0")
    if skeleton.num_groupby == 0 and len(groupby) > 0:
        violations.append(f"skeleton forbids groupby but got {len(groupby)}")

    having = response.get("having", [])
    if len(having) != skeleton.num_having:
        violations.append(f"skeleton requires {skeleton.num_having} having clause(s) but got {len(having)}")

    has_distinct = response.get("distinct", False)
    if skeleton.has_distinct and not has_distinct:
        violations.append(f"skeleton requires distinct but got distinct={has_distinct}")
    if not skeleton.has_distinct and has_distinct:
        violations.append(f"skeleton forbids distinct but got distinct={has_distinct}")

    expr_comparison = response.get("expr_comparison") or response.get("column_comparison")
    if skeleton.has_expr_comparison and not expr_comparison:
        violations.append("skeleton requires expr_comparison but got none")

    orderby_cols = response.get("orderby_cols", [])
    if skeleton.has_orderby and len(orderby_cols) == 0:
        violations.append("skeleton requires orderby but got none")
    if not skeleton.has_orderby and len(orderby_cols) > 0:
        violations.append("skeleton forbids orderby but got orderby_cols")

    return (len(violations) == 0, violations)


def _build_retry_guidance(failure_ctx: RetryFailureContext, schema: SchemaGraph, column_roles: dict[str, str]) -> str:
    """
    Build retry guidance text for the LLM from a previous failure context.

    Args:

        failure_ctx: Description.

        schema: Schema graph for suggesting columns on missing tables.

        column_roles: Map of column key to role (reserved for extensions).

    Returns:

        Multi-line string to append to the fill prompt.
    """
    guidance_parts = []

    guidance_parts.append(f"\n\n    RETRY GUIDANCE (Attempt {failure_ctx.attempt_number + 2}):")
    guidance_parts.append(f"    Previous attempt failed: {failure_ctx.failure_type}")
    guidance_parts.append(f"    Required tables: {failure_ctx.required_tables}")
    guidance_parts.append(f"    Tables you used: {list(failure_ctx.used_tables)}")
    guidance_parts.append(f"    Tables you MUST include: {list(failure_ctx.missing_tables)}")

    for missing_table in failure_ctx.missing_tables:
        table_ir = schema.tables.get(missing_table)
        if table_ir:
            cols = list(table_ir.columns.keys())[:5]
            guidance_parts.append(f"    Available columns in {missing_table}: {cols}")

    guidance_parts.append(
        f"    FIX: Add filters, select_cols, groupby_cols, or aggregation from {list(failure_ctx.missing_tables)}"
    )

    return "\n".join(guidance_parts)


def _llm_fill_intent(
    skeleton: QSimSkeleton,
    schema: SchemaGraph,
    column_roles: dict[str, str],
    *,
    target_tier: ComplexityTier | None = None,
    select_col_target: int | None = None,
    advanced_feature_lines: str | None = None,
) -> QSimIntent | None:
    """
    Fill a structural skeleton via the LLM; validate and retry with guidance on failure.

    Args:

        skeleton: Description.

        schema: Schema graph for columns and FK relationships.

        column_roles: Map of `table.column` to role string.

        target_tier: Optional sampled complexity band for conditioning and conformance checking.

        select_col_target: Preferred SELECT-list cardinality from geometric sampling.

        advanced_feature_lines: Capability-filtered advanced feature bullets.

    Returns:

        `QSimIntent` on success, or None after retries are exhausted.
    """
    context = build_schema_context(skeleton.tables, schema)

    all_filterable = []
    all_groupable = []
    all_aggregatable = []
    for table in skeleton.tables:
        all_filterable.extend(get_filterable_columns(table, schema, column_roles))
        all_groupable.extend(get_groupable_columns(table, schema, column_roles))
        all_aggregatable.extend(get_aggregatable_columns(table, schema, column_roles))

    filterable_list = [col_key for col_key, _ in all_filterable]

    effective_filters = skeleton.num_filters
    if skeleton.has_expr_comparison:
        effective_filters = max(0, skeleton.num_filters - 1)

    if skeleton.has_aggregation:
        agg_instruction = (
            "MUST include at least one aggregated select column (COUNT/SUM/AVG/MIN/MAX wrapping table.column)"
        )
    else:
        agg_instruction = "NO aggregation - all select_cols must be plain table.column references"

    filter_instruction = (
        f"MUST include {skeleton.num_filters} filter conditions"
        if skeleton.num_filters > 0
        else "DO NOT include filters"
    )
    groupby_instruction = (
        f"MUST include {skeleton.num_groupby} GROUP BY columns"
        if skeleton.num_groupby > 0
        else "DO NOT include GROUP BY"
    )
    orderby_instruction = (
        "MUST include ORDER BY clause (non-empty orderby_cols)" if skeleton.has_orderby else "DO NOT include ORDER BY"
    )

    if skeleton.num_having > 0:
        having_instruction = (
            f"MUST include exactly {skeleton.num_having} HAVING condition(s) with aggregation. "
            "The having array length must match."
        )
    else:
        having_instruction = "DO NOT include HAVING (having must be an empty array)"

    distinct_instruction = "Use SELECT DISTINCT (no aggregations)" if skeleton.has_distinct else ""

    expr_comparison_instruction = ""
    comparable_pairs = []
    if skeleton.has_expr_comparison:
        comparable_pairs = get_comparable_column_pairs(skeleton.tables, schema, column_roles)
        if comparable_pairs:
            pairs_str = ", ".join([f"{t1}.{c1} vs {t2}.{c2}" for t1, c1, t2, c2, _ in comparable_pairs[:5]])
            expr_comparison_instruction = (
                f"MUST include an expr-vs-expr comparison (e.g., {pairs_str}). "
                "Choose columns and operator that make logical sense. "
                "DO NOT set expr_comparison to null."
            )

    filterable_constraint = (
        f"\n        FILTERABLE COLUMNS (MUST use ONLY these for filters): {filterable_list}"
        if effective_filters > 0
        else ""
    )
    aggregatable_constraint = (
        f"\n        AGGREGATABLE COLUMNS (use for SUM/AVG/MIN/MAX): {all_aggregatable}"
        if skeleton.has_aggregation and all_aggregatable
        else ""
    )
    groupable_constraint = (
        f"\n        GROUPABLE COLUMNS (MUST use for GROUP BY): {all_groupable}" if skeleton.num_groupby > 0 else ""
    )

    optional_instructions = []
    if distinct_instruction:
        optional_instructions.append(distinct_instruction)
    if expr_comparison_instruction:
        optional_instructions.append(expr_comparison_instruction)
    optional_str = "\n        - ".join([""] + optional_instructions) if optional_instructions else ""

    tier_extra = ""
    if target_tier is not None:
        tsumm, tex = _tier_spec_lines(target_tier)
        sel_hint = int(select_col_target) if select_col_target is not None else 1
        feat_blk = advanced_feature_lines or ""
        tier_extra = f"""
        TARGET COMPLEXITY BAND: {target_tier.value}
        BAND GUIDANCE: {tsumm}
        EXAMPLE SKETCH: {tex}
        AIM FOR APPROXIMATELY {sel_hint} DISTINCT SELECT LIST ENTRIES (respect aggregation rules above).
        DATABASE-SUPPORTED ADVANCED SHAPES (only where compatible with this skeleton):
        {feat_blk}
        """

    user_prompt = f"""
        Schema:
        {context}
        {filterable_constraint}{aggregatable_constraint}{groupable_constraint}

        CRITICAL REQUIREMENTS (MUST follow exactly):
        - Tables: {skeleton.tables}
        - {agg_instruction}
        - {filter_instruction}
        - {groupby_instruction}
        - {orderby_instruction}
        - {having_instruction}{optional_str}
        {tier_extra}

        Return JSON:
        {{
        "select_cols": ["table.column" | "COUNT(table.column)" | "SUM(table.column)" | "AVG(table.column)" | "MIN(table.column)" | "MAX(table.column)", ...],
        "filters": [{{"column": "table.column", "op": "=" | ">" | "<" | ">=" | "<=" | "!=" | "like" | "between" | "in" | "not in" | "is null" | "is not null", "value_type": "categorical" | "numeric_categorical" | "numeric" | "temporal" | "boolean" | "null"}}],
        "groupby_cols": ["table.column", ...],
        "orderby_cols": ["table.column ASC" | "table.column DESC" | "COUNT(table.column) DESC", ...],
        "having": [{{"expression": "COUNT(table.column)" | "SUM(table.column)" | "AVG(table.column)" | "MIN(table.column)" | "MAX(table.column)", "op": "=" | "!=" | ">" | "<" | ">=" | "<=" | "in" | "not in" | "between", "value_type": "number" | "integer"}}],
        "expr_comparison": {{"left_column": "table.column", "op": "=" | ">" | "<" | ">=" | "<=" | "!=", "right_column": "table.column"}} | null,
        "distinct": true | false
        }}
    """

    last_failure_reason = None
    failure_context = None
    for attempt in range(PolicyConfig.MAX_STAGE_B_REPAIRS + 1):
        prompt_with_context = user_prompt
        if failure_context and attempt > 0:
            retry_guidance = _build_retry_guidance(failure_context, schema, column_roles)
            prompt_with_context = f"{user_prompt}{retry_guidance}"
        elif last_failure_reason and attempt > 0:
            prompt_with_context = f"{user_prompt}\n\n    PREVIOUS ATTEMPT FAILED: {last_failure_reason}\n    Please fix this issue in your response."

        try:
            result = llm_json(_QSIM_FILL_SYSTEM, prompt_with_context, task="default")
        except LlmJsonExhausted as exc:
            last_failure_reason = f"LLM returned no parseable JSON ({exc})"
            failure_context = None
            debug(
                f"[qsim_ops.llm_fill_intent] attempt {attempt + 1}/{(PolicyConfig.MAX_STAGE_B_REPAIRS + 1)} exhausted: {last_failure_reason} for skeleton tables={skeleton.tables}"
            )
            continue

        debug(
            f"[qsim_ops.llm_fill_intent] attempt {attempt + 1} LLM returned: select_cols={len(result.get('select_cols', []))}, filters_count={len(result.get('filters', []))}, groupby_count={len(result.get('groupby_cols', []))}, having_count={len(result.get('having', []))}, expr_comparison={result.get('expr_comparison') or result.get('column_comparison')}, distinct={result.get('distinct')}"
        )

        is_valid, violations = _validate_skeleton_constraints(result, skeleton)
        if not is_valid:
            last_failure_reason = "; ".join(violations)
            failure_context = None
            debug(
                f"[qsim_ops.llm_fill_intent] attempt {attempt + 1}/{(PolicyConfig.MAX_STAGE_B_REPAIRS + 1)} SKELETON_CONSTRAINT_VIOLATION: {violations}"
            )
            continue

        parse_result = _parse_llm_response(result, skeleton, schema, column_roles)

        if isinstance(parse_result, tuple) and len(parse_result) == 3:
            failure_type, used_tables, missing_tables = parse_result
            failure_context = RetryFailureContext(
                failure_type=failure_type,
                required_tables=skeleton.tables,
                used_tables=used_tables,
                missing_tables=missing_tables,
                attempt_number=attempt,
            )
            last_failure_reason = None
            debug(
                f"[qsim_ops.llm_fill_intent] attempt {attempt + 1}/{(PolicyConfig.MAX_STAGE_B_REPAIRS + 1)} failed: {failure_type} for skeleton tables={skeleton.tables}, missing={missing_tables}"
            )
            continue

        if parse_result:
            if target_tier is not None:
                classified = classify_qsim_intent_complexity(parse_result)
                if not qsim_intent_matches_target_tier(classified, target_tier):
                    last_failure_reason = f"tier_conformance: classified={classified.value} target={target_tier.value}"
                    failure_context = None
                    debug(
                        f"[qsim_ops.llm_fill_intent] attempt {attempt + 1}/{(PolicyConfig.MAX_STAGE_B_REPAIRS + 1)} "
                        f"TIER_MISMATCH classified={classified.value} target={target_tier.value}"
                    )
                    continue
            debug(
                f"[qsim_ops.llm_fill_intent] SUCCESS: intent_id={parse_result.intent_id}, grain={parse_result.grain}, filters={len(parse_result.filters_param)}, groupby={len(parse_result.group_by_cols)}, distinct={parse_result.distinct}"
            )
            return parse_result

        last_failure_reason = "Response validation failed (filters/columns rejected)"
        failure_context = None
        debug(
            f"[qsim_ops.llm_fill_intent] attempt {attempt + 1}/{(PolicyConfig.MAX_STAGE_B_REPAIRS + 1)} failed: parse_llm_response returned None for skeleton tables={skeleton.tables}, LLM response keys={list(result.keys())}"
        )

    debug(
        f"[qsim_ops.llm_fill_intent] FINAL_FAILURE: exhausted {(PolicyConfig.MAX_STAGE_B_REPAIRS + 1)} attempts for skeleton tables={skeleton.tables}, has_agg={skeleton.has_aggregation}, num_filters={skeleton.num_filters}"
    )
    return None


def _parse_llm_response(
    response: dict[str, Any],
    skeleton: QSimSkeleton,
    schema: SchemaGraph,
    column_roles: dict[str, str],
) -> Any | None:
    """
    Parse and validate LLM JSON into a `QSimIntent` or retry context tuple.

    Args:

        response: Parsed LLM JSON with intent fields.

        skeleton: Skeleton constraints the response must satisfy.

        schema: Schema graph for existence and metadata checks.

        column_roles: Map of `table.column` to role string.

    Returns:

        coverage retry.
    """
    select_cols_raw = response.get("select_cols", [])
    filter_dicts = response.get("filters", [])
    groupby_cols = response.get("groupby_cols", [])
    orderby_cols_raw = response.get("orderby_cols", [])
    having_dicts = response.get("having", [])
    expr_comparison_dict = response.get("expr_comparison") or response.get("column_comparison")
    has_distinct = response.get("distinct", False)

    has_agg = _has_aggregation(select_cols_raw)

    if skeleton.has_aggregation and not has_agg:
        debug("[qsim_ops.parse_llm_response] REJECTED: skeleton requires aggregation but none in select_cols")
        return None
    if skeleton.num_filters > 0 and len(filter_dicts) == 0 and not skeleton.has_expr_comparison:
        debug(
            f"[qsim_ops.parse_llm_response] REJECTED: skeleton requires {skeleton.num_filters} filters but none provided"
        )
        return None
    if skeleton.num_groupby > 0 and len(groupby_cols) == 0:
        debug(
            f"[qsim_ops.parse_llm_response] REJECTED: skeleton requires {skeleton.num_groupby} groupby cols but none provided"
        )
        return None

    if skeleton.has_orderby and len(orderby_cols_raw) == 0:
        debug("[qsim_ops.parse_llm_response] REJECTED: skeleton requires orderby but none provided")
        return None
    if not skeleton.has_orderby and len(orderby_cols_raw) > 0:
        debug("[qsim_ops.parse_llm_response] REJECTED: skeleton forbids orderby but orderby_cols provided")
        return None

    if skeleton.has_distinct and skeleton.has_aggregation:
        debug("[qsim_ops.parse_llm_response] REJECTED: DISTINCT not allowed with aggregation")
        return None

    select_cols: list[str] = []
    for sc in select_cols_raw:
        if not isinstance(sc, str) or not sc.strip():
            continue
        sc = sc.strip()
        agg_info = _extract_agg_info(sc)
        if agg_info:
            agg_func, agg_inner = agg_info
            if agg_inner != "*":
                if not validate_column_exists(agg_inner, skeleton.tables, schema):
                    debug(f"[qsim_ops.parse_llm_response] REJECTED_SELECT: {sc}, reason=agg_column_not_found")
                    continue
            select_cols.append(f"{agg_func.upper()}({agg_inner})")
        else:
            if not validate_column_exists(sc, skeleton.tables, schema):
                debug(f"[qsim_ops.parse_llm_response] REJECTED_SELECT: {sc}, reason=column_not_found")
                continue
            select_cols.append(sc)

    if not select_cols:
        debug("[qsim_ops.parse_llm_response] REJECTED: no valid select_cols remaining")
        return None

    aggregated_tables: set[str] = set()
    if has_agg and groupby_cols:
        for gcol in groupby_cols:
            if "." in gcol:
                aggregated_tables.add(gcol.split(".")[0])

    filters: list[QSimFilter] = []
    filter_columns_used: set[str] = set()
    for _filter_idx, fd in enumerate(filter_dicts):
        col = fd.get("column", "")
        if not validate_column_exists(col, skeleton.tables, schema):
            debug(f"[qsim_ops.parse_llm_response] REJECTED_FILTER: col={col}, reason=column_not_found")
            continue

        table, col_name = col.split(".", 1)
        col_meta = schema.tables[table].columns.get(col_name)
        if not col_meta or not col_meta.is_filterable or not col_meta.is_visible:
            debug(f"[qsim_ops.parse_llm_response] REJECTED_FILTER: col={col}, reason=not_filterable")
            continue

        if col not in filter_columns_used and len(filter_columns_used) >= QSimConfig.MAX_FILTER_COLUMNS + 1:
            debug(
                f"[qsim_ops.parse_llm_response] REJECTED_FILTER: col={col}, reason=max_filter_columns_exceeded (>{QSimConfig.MAX_FILTER_COLUMNS + 1})"
            )
            continue

        if col not in filter_columns_used and len(filter_columns_used) >= QSimConfig.MAX_FILTER_COLUMNS:
            debug(
                f"[qsim_ops.parse_llm_response] WARNING_FILTER: col={col}, using {len(filter_columns_used) + 1} distinct columns (preferred max={QSimConfig.MAX_FILTER_COLUMNS})"
            )

        op = fd.get("op", "=")
        valid_ops = col_meta.get_valid_filter_ops()

        if op not in valid_ops:
            debug(f"[qsim_ops.parse_llm_response] REJECTED_FILTER: col={col}, reason=invalid_operator_{op}_for_type")
            continue

        if has_agg and col_meta.is_foreign_key and op == "=":
            fk_target_table = col_meta.fk_target[0] if col_meta.fk_target else None

            if fk_target_table and fk_target_table in aggregated_tables:
                debug(
                    f"[qsim_ops.parse_llm_response] REJECTED_FILTER: col={col}, reason=circular_fk_to_aggregated_table"
                )
                continue

            if table in aggregated_tables:
                debug(
                    f"[qsim_ops.parse_llm_response] REJECTED_FILTER: col={col}, reason=fk_filter_on_aggregated_source_table"
                )
                continue

        value_type = fd.get("value_type", "categorical")
        if op in ("is null", "is not null"):
            value_type = "null"
        elif value_type not in VALID_FILTER_VALUE_TYPES and value_type != "null":
            value_type = "categorical"

        filter_columns_used.add(col)

        qf = QSimFilter(column=col, op=op, value_type=value_type)
        if op == "between":
            decomposed = decompose_between_filter(qf)
            filters.extend(decomposed)
            debug(f"[qsim_ops.parse_llm_response] DECOMPOSED_BETWEEN: col={col} into >= and <=")
        else:
            filters.append(qf)
            debug(f"[qsim_ops.parse_llm_response] ACCEPTED_FILTER: col={col}, op={op}, value_type={value_type}")

    if skeleton.has_expr_comparison and expr_comparison_dict:
        left_col_full = expr_comparison_dict.get("left_column", "")
        right_col_full = expr_comparison_dict.get("right_column", "")
        cmp_op = expr_comparison_dict.get("op", "=")

        if left_col_full and right_col_full and "." in left_col_full and "." in right_col_full:
            left_table, left_col_name = left_col_full.split(".", 1)
            right_table, right_col_name = right_col_full.split(".", 1)

            left_valid = validate_column_exists(left_col_full, skeleton.tables, schema)
            right_valid = validate_column_exists(right_col_full, skeleton.tables, schema)

            if left_valid and right_valid:
                left_meta = schema.tables[left_table].columns.get(left_col_name)
                right_meta = schema.tables[right_table].columns.get(right_col_name)

                if left_meta and right_meta:
                    left_is_numeric = left_meta.value_type in ("integer", "number")
                    right_is_numeric = right_meta.value_type in ("integer", "number")
                    left_is_temporal = left_meta.value_type == "date"
                    right_is_temporal = right_meta.value_type == "date"

                    left_role = column_roles.get(f"{left_table}.{left_col_name}", left_meta.role or "unknown")
                    right_role = column_roles.get(f"{right_table}.{right_col_name}", right_meta.role or "unknown")

                    semantic_compatible = False
                    rejection_reason = None

                    if left_role == right_role and left_role != "unknown":
                        semantic_compatible = True
                    elif left_is_temporal and right_is_temporal:
                        semantic_compatible = True
                    elif left_is_numeric and right_is_numeric and left_role == right_role:
                        semantic_compatible = True
                    else:
                        rejection_reason = f"role_mismatch: left={left_col_full}(role={left_role}) vs right={right_col_full}(role={right_role})"

                    if semantic_compatible:
                        value_type = "temporal" if left_is_temporal else "numeric"
                        filters.append(
                            QSimFilter(
                                column=left_col_full,
                                op=cmp_op,
                                value_type=value_type,
                                right_column=right_col_full,
                            )
                        )
                        debug(
                            f"[qsim_ops.parse_llm_response] ACCEPTED_COLUMN_COMPARISON: {left_col_full} {cmp_op} {right_col_full}, roles={left_role}={right_role}"
                        )
                    else:
                        debug(
                            f"[qsim_ops.parse_llm_response] DISCARDED_EXPR_COMPARISON: {left_col_full} {cmp_op} {right_col_full}, reason={rejection_reason}"
                        )
                else:
                    debug("[qsim_ops.parse_llm_response] DISCARDED_EXPR_COMPARISON: column metadata not found")
            else:
                debug(
                    f"[qsim_ops.parse_llm_response] DISCARDED_EXPR_COMPARISON: column validation failed left={left_valid} right={right_valid}"
                )
        else:
            debug("[qsim_ops.parse_llm_response] DISCARDED_EXPR_COMPARISON: invalid column format")

    total_filter_elements = len(filters)
    if skeleton.num_filters > 0 and total_filter_elements == 0:
        debug(
            f"[qsim_ops.parse_llm_response] INSUFFICIENT_FILTERS: requested={skeleton.num_filters}, validated_filters={len(filters)}, rejecting_intent"
        )
        return None

    if has_agg and groupby_cols:
        for sc in select_cols:
            agg_info = _extract_agg_info(sc)
            if agg_info:
                _, agg_inner = agg_info
                agg_inner_base = agg_inner.split(".")[-1] if "." in agg_inner else agg_inner
                for gcol in groupby_cols:
                    gcol_base = gcol.split(".")[-1] if "." in gcol else gcol
                    if agg_inner == gcol:
                        debug(
                            f"[qsim_ops.parse_llm_response] REJECTED: agg_inner={agg_inner} matches groupby_col={gcol}, reason=exact_self_grouping"
                        )
                        return None
                    if agg_inner_base == gcol_base:
                        debug(
                            f"[qsim_ops.parse_llm_response] REJECTED: agg_inner={agg_inner} matches groupby_col={gcol}, reason=base_name_self_grouping"
                        )
                        return None

    having: list[QSimHaving] = []
    for hd in having_dicts:
        h_expression = hd.get("expression", "")
        h_op = hd.get("op", ">")
        if h_op not in VALID_HAVING_OPS:
            h_op = ">"
        h_value_type = hd.get("value_type", "number")
        if h_value_type not in VALID_HAVING_VALUE_TYPES:
            h_value_type = "number"
        right_expr = hd.get("right_expression", "")

        h_agg_info = _extract_agg_info(h_expression)
        if not h_agg_info:
            debug(
                f"[qsim_ops.parse_llm_response] REJECTED_HAVING: expression={h_expression}, reason=no_aggregation_pattern"
            )
            continue

        h_agg_func, h_agg_inner = h_agg_info
        if h_agg_inner != "*" and not validate_column_exists(h_agg_inner, skeleton.tables, schema):
            debug(f"[qsim_ops.parse_llm_response] REJECTED_HAVING: expression={h_expression}, reason=column_not_found")
            continue

        if right_expr:
            right_agg_info = _extract_agg_info(right_expr)
            if not right_agg_info:
                debug(
                    f"[qsim_ops.parse_llm_response] REJECTED_HAVING: right_expression={right_expr}, reason=no_aggregation_pattern"
                )
                continue
            right_agg_func, right_agg_inner = right_agg_info
            if right_agg_inner != "*" and not validate_column_exists(right_agg_inner, skeleton.tables, schema):
                debug(
                    f"[qsim_ops.parse_llm_response] REJECTED_HAVING: right_expression={right_expr}, reason=column_not_found"
                )
                continue
            having.append(
                QSimHaving(
                    expression=f"{h_agg_func.upper()}({h_agg_inner})",
                    op=h_op,
                    value_type="expression",
                    right_expression=f"{right_agg_func.upper()}({right_agg_inner})",
                )
            )
        else:
            having.append(
                QSimHaving(
                    expression=f"{h_agg_func.upper()}({h_agg_inner})",
                    op=h_op,
                    value_type=h_value_type,
                )
            )

    validated_groupby: list[str] = []
    for gcol in groupby_cols:
        if validate_column_exists(gcol, skeleton.tables, schema):
            validated_groupby.append(gcol)
        else:
            debug(f"[qsim_ops.parse_llm_response] REJECTED_GROUPBY: col={gcol}, reason=column_not_found")

    order_by_cols: list[str] = []
    for ob in orderby_cols_raw:
        ob_clean = ob.strip()
        direction = "ASC"
        if ob_clean.upper().endswith(" DESC"):
            direction = "DESC"
            ob_clean = ob_clean[:-5].strip()
        elif ob_clean.upper().endswith(" ASC"):
            ob_clean = ob_clean[:-4].strip()

        agg_info = _extract_agg_info(ob_clean)
        if agg_info:
            agg_func, agg_inner = agg_info
            if agg_inner != "*" and not validate_column_exists(agg_inner, skeleton.tables, schema):
                debug(f"[qsim_ops.parse_llm_response] REJECTED_ORDERBY: {ob}, reason=column_not_found")
                continue
            order_by_cols.append(f"{agg_func.upper()}({agg_inner}) {direction}")
        else:
            if not validate_column_exists(ob_clean, skeleton.tables, schema):
                debug(f"[qsim_ops.parse_llm_response] REJECTED_ORDERBY: {ob}, reason=column_not_found")
                continue
            order_by_cols.append(f"{ob_clean} {direction}")

    grain = "row_level"
    if has_agg:
        grain = "grouped" if validated_groupby else "scalar"

    use_distinct = skeleton.has_distinct and has_distinct and grain == "row_level"

    if skeleton.has_distinct and not use_distinct:
        if not has_distinct:
            debug(
                "[qsim_ops.parse_llm_response] DISTINCT_REJECTED: LLM returned distinct=false despite skeleton.has_distinct=True"
            )
        elif grain != "row_level":
            debug(
                f"[qsim_ops.parse_llm_response] DISTINCT_REJECTED: grain={grain} incompatible with DISTINCT (requires row_level)"
            )

    if len(skeleton.tables) >= 3:
        tables_used: set[str] = set()
        for sc in select_cols:
            tables_used.update(_extract_tables_from_expr(sc))
        for col in validated_groupby:
            tables_used.update(_extract_tables_from_expr(col))
        for f in filters:
            tables_used.update(_extract_tables_from_expr(f.column))
            if f.right_column:
                tables_used.update(_extract_tables_from_expr(f.right_column))
        for ob in order_by_cols:
            tables_used.update(_extract_tables_from_expr(ob))

        missing_tables = set(skeleton.tables) - tables_used
        if missing_tables:
            debug(
                f"[qsim_ops.parse_llm_response] REJECTED_THREE_TABLE: tables={skeleton.tables}, used={tables_used}, missing={missing_tables}"
            )
            return ("three_table_violation", tables_used, missing_tables)

    intent_id_val = compute_intent_id(
        {
            "tables": skeleton.tables,
            "grain": grain,
            "select_cols": select_cols,
            "group_by_cols": validated_groupby,
            "filters_param": [f.to_dict() for f in filters],
            "having_param": [h.to_dict() for h in having],
            "distinct": use_distinct,
        }
    )

    return QSimIntent(
        intent_id=intent_id_val,
        tables=skeleton.tables,
        grain=grain,
        select_cols=select_cols,
        group_by_cols=validated_groupby,
        order_by_cols=order_by_cols,
        filters_param=filters,
        having_param=having,
        param_values={},
        distinct=use_distinct,
    )


def _generate_question_from_intent(intent: QSimIntent, schema: SchemaGraph) -> str | None:
    """
    Produce natural language for an intent using `generate_question`.

    Args:

        intent: Intent with filters, groupings, and param placeholders.

        schema: Schema graph for the generator.

    Returns:

        Question string, or None if generation fails.
    """
    filter_descriptions = []
    for idx, f in enumerate(intent.filters_param):
        if f.is_expr_comparison:
            cond = f"{f.op} {f.right_column}"
        else:
            cond = f"{f.op} {intent.param_values.get(f'f{idx}', '?')}"
        filter_descriptions.append({"column": f.column, "condition": cond})

    having_descriptions = []
    for hidx, h in enumerate(intent.having_param):
        if h.is_expression_comparison:
            cond = f"{h.op} {h.right_expression}"
        else:
            cond = f"{h.op} {intent.param_values.get(f'h{hidx}', '?')}"
        having_descriptions.append({"expression": h.expression, "condition": cond})

    return generate_question(
        intent.tables,
        intent.select_cols,
        filter_descriptions,
        intent.group_by_cols,
        having_descriptions,
        schema,
    )


def generate_all_questions(intents: list[QSimIntent], schema: SchemaGraph) -> list[QSimIntent]:
    """
    Return intents with generated NL `question` fields where generation succeeds.

    Args:

        intents: Intents to verbalize.

        schema: Schema graph for question generation.

    Returns:

        are omitted.
    """
    debug(f"[qsim_ops.generate_all_questions] generating: {len(intents)} questions")

    results: list[QSimIntent] = []

    for i, intent in enumerate(intents):
        if i > 0 and i % 10 == 0:
            debug(f"[qsim_ops.generate_all_questions] progress: {i}/{len(intents)}")

        question = _generate_question_from_intent(intent, schema)
        if question:
            intent_with_question = QSimIntent(
                intent_id=intent.intent_id,
                tables=intent.tables,
                grain=intent.grain,
                select_cols=intent.select_cols,
                group_by_cols=intent.group_by_cols,
                order_by_cols=intent.order_by_cols,
                filters_param=intent.filters_param,
                having_param=intent.having_param,
                param_values=intent.param_values,
                question=question,
                variant_idx=intent.variant_idx,
                limit=intent.limit,
                distinct=intent.distinct,
            )
            results.append(intent_with_question)
        else:
            debug(f"[qsim_ops.generate_all_questions] failed: {intent.intent_id}")

    debug(f"[qsim_ops.generate_all_questions] complete: {len(results)} questions")
    return results


def _is_no_variance_skeleton(skeleton: QSimSkeleton) -> bool:
    """
    Return True if the skeleton has no filters and no HAVING (no value-sampling variance).

    Args:

        skeleton: Skeleton to inspect.

    Returns:

        True when ``num_filters == 0`` and ``num_having == 0``.
    """
    return skeleton.num_filters == 0 and skeleton.num_having == 0


def _compute_skeleton_complexity_tier(skeleton: QSimSkeleton) -> str:
    """
    Assign complexity tier A, B, or C from skeleton features for stratified pools.

    Args:

        skeleton: Skeleton to score.

    Returns:

        `"A"` (high), `"B"` (medium), or `"C"` (low) by composite score.
    """
    score = 0
    score += skeleton.num_filters * 2
    score += 3 if skeleton.has_aggregation else 0
    score += skeleton.num_groupby * 2
    score += 3 if skeleton.num_having > 0 else 0
    score += 1 if skeleton.has_orderby else 0
    score += 2 if skeleton.has_distinct else 0
    score += 3 if skeleton.has_expr_comparison else 0

    if score >= 8:
        return "A"
    elif score >= 4:
        return "B"
    else:
        return "C"


def _compute_table_set_richness(tables: list[str], schema: SchemaGraph, column_roles: dict[str, str]) -> int:
    """
    Score a table set from filterable, aggregatable, groupable, and comparable-column counts.

    Args:

        tables: Table names in the candidate set.

        schema: Schema graph for column capabilities.

        column_roles: Map of column key to role string.

    Returns:

        Integer score (weighted counts across tables and comparable pairs).
    """
    filterable_count = 0
    aggregatable_count = 0
    groupable_count = 0

    for table in tables:
        filterable_count += len(get_filterable_columns(table, schema, column_roles))
        aggregatable_count += len(get_aggregatable_columns(table, schema, column_roles))
        groupable_count += len(get_groupable_columns(table, schema, column_roles))

    comparable_pairs = len(get_comparable_column_pairs(tables, schema, column_roles))

    return filterable_count * 2 + aggregatable_count * 3 + groupable_count * 2 + comparable_pairs * 2


def _build_skeleton_pool(
    schema: SchemaGraph, column_roles: dict[str, str], num_tables: int | None = None
) -> SkeletonPool:
    """
    Build a tiered `SkeletonPool` from enumerated table sets and generated skeletons.

    Args:

        schema: Schema graph for skeleton generation and richness scoring.

        column_roles: Map of column key to role string.

        num_tables: If set, keep only table sets of this size.

    Returns:

        `SkeletonPool` with A/B/C tiers and shuffle state for selection.
    """
    table_sets = enumerate_table_sets(schema)

    if num_tables is not None:
        table_sets = [ts for ts in table_sets if len(ts) == num_tables]

    scored_sets = [(ts, _compute_table_set_richness(ts, schema, column_roles)) for ts in table_sets]
    scored_sets.sort(key=lambda x: x[1], reverse=True)

    tier_a_by_table_set: dict[str, list[QSimSkeleton]] = {}
    tier_b_by_table_set: dict[str, list[QSimSkeleton]] = {}
    tier_c_by_table_set: dict[str, list[QSimSkeleton]] = {}

    for table_set, _ in scored_sets:
        table_key = "|".join(sorted(table_set))
        tier_a_by_table_set[table_key] = []
        tier_b_by_table_set[table_key] = []
        tier_c_by_table_set[table_key] = []

        skeletons = generate_all_skeletons(table_set, schema, column_roles)
        for skel in skeletons:
            tier = _compute_skeleton_complexity_tier(skel)
            if tier == "A":
                tier_a_by_table_set[table_key].append(skel)
            elif tier == "B":
                tier_b_by_table_set[table_key].append(skel)
            else:
                tier_c_by_table_set[table_key].append(skel)

    for table_key in tier_a_by_table_set:
        random.shuffle(tier_a_by_table_set[table_key])
        random.shuffle(tier_b_by_table_set[table_key])
        random.shuffle(tier_c_by_table_set[table_key])

    table_set_keys = list(tier_a_by_table_set.keys())
    tier_a_indices = {k: 0 for k in table_set_keys}
    tier_b_indices = {k: 0 for k in table_set_keys}
    tier_c_indices = {k: 0 for k in table_set_keys}

    total_a = sum(len(v) for v in tier_a_by_table_set.values())
    total_b = sum(len(v) for v in tier_b_by_table_set.values())
    total_c = sum(len(v) for v in tier_c_by_table_set.values())

    debug(f"[qsim_ops.build_skeleton_pool] built pool: tier_a={total_a}, tier_b={total_b}, tier_c={total_c}")
    return SkeletonPool(
        tier_a_by_table_set=tier_a_by_table_set,
        tier_b_by_table_set=tier_b_by_table_set,
        tier_c_by_table_set=tier_c_by_table_set,
        table_set_keys=table_set_keys,
        tier_a_indices=tier_a_indices,
        tier_b_indices=tier_b_indices,
        tier_c_indices=tier_c_indices,
    )


def _select_next_skeleton(
    pool: SkeletonPool, need_filters: bool, need_having: bool
) -> tuple[QSimSkeleton, list[str]] | None:
    """
    Select the next skeleton by round-robin table sets; prefer tier A, then B, then C.

    Args:

        pool: Pool with tier lists and iteration indices.

        need_filters: If True, require `num_filters > 0`.

        need_having: If True, require ``num_having > 0``.

    Returns:

        Tuple `(skeleton, table_set)`, or None if no skeleton matches.
    """

    def matches_needs(skel: QSimSkeleton) -> bool:
        if need_filters and skel.num_filters == 0:
            return False
        if need_having and skel.num_having == 0:
            return False
        return True

    start_idx = pool.current_table_idx
    attempts = 0
    max_attempts = len(pool.table_set_keys)

    tiers = [
        ("a", pool.tier_a_by_table_set, pool.tier_a_indices),
        ("b", pool.tier_b_by_table_set, pool.tier_b_indices),
        ("c", pool.tier_c_by_table_set, pool.tier_c_indices),
    ]

    while attempts < max_attempts:
        table_idx = (start_idx + attempts) % len(pool.table_set_keys)
        table_key = pool.table_set_keys[table_idx]
        table_set = table_key.split("|")

        for _tier_name, tier_dict, indices_dict in tiers:
            skeletons = tier_dict[table_key]
            current_idx = indices_dict[table_key]

            for i in range(current_idx, len(skeletons)):
                skel = skeletons[i]
                if matches_needs(skel):
                    indices_dict[table_key] = i + 1
                    pool.current_table_idx = (table_idx + 1) % len(pool.table_set_keys)
                    return skel, table_set

        attempts += 1

    return None


def _normalize_qsim_intent(intent: QSimIntent, schema: SchemaGraph) -> QSimIntent:
    """
    Return a canonical `QSimIntent`: grain, deduped columns, pruned tables, new `intent_id`.

    Args:

        intent: Intent to normalize.

        schema: Schema graph for FK connectivity.

    Returns:

        New normalized `QSimIntent`.
    """
    grain = intent.grain
    has_agg = _has_aggregation(intent.select_cols)

    if grain == "grouped":
        if not intent.group_by_cols:
            grain = "row_level"
    else:
        if has_agg:
            grain = "grouped" if intent.group_by_cols else "scalar"

    normalized_select = sorted(set(intent.select_cols))
    normalized_orderby = sorted(intent.order_by_cols)

    tables_used: set[str] = set()
    for sc in normalized_select:
        tables_used.update(_extract_tables_from_expr(sc))
    for col in intent.group_by_cols:
        tables_used.update(_extract_tables_from_expr(col))
    for ob in normalized_orderby:
        tables_used.update(_extract_tables_from_expr(ob))
    for f in intent.filters_param:
        tables_used.update(_extract_tables_from_expr(f.column))
        if f.right_column:
            tables_used.update(_extract_tables_from_expr(f.right_column))
    for h in intent.having_param:
        tables_used.update(_extract_tables_from_expr(h.expression))

    tables_used.discard("")

    normalized_tables = intent.tables
    if tables_used and len(tables_used) < len(intent.tables):
        adj = build_fk_adjacency(schema)
        if is_connected(list(tables_used), adj):
            normalized_tables = sorted(tables_used)
            debug(f"[qsim_ops.normalize_qsim_intent] removed unnecessary tables: {set(intent.tables) - tables_used}")

    table_prefixed_group_by = []
    for col in intent.group_by_cols:
        if "." not in col:
            if normalized_tables:
                col = f"{normalized_tables[0]}.{col}"
        table_prefixed_group_by.append(col)

    intent_id_val = compute_intent_id(
        {
            "tables": normalized_tables,
            "grain": grain,
            "select_cols": normalized_select,
            "group_by_cols": table_prefixed_group_by,
            "filters_param": [f.to_dict() for f in intent.filters_param],
            "having_param": [h.to_dict() for h in intent.having_param],
            "distinct": intent.distinct,
        }
    )

    return QSimIntent(
        intent_id=intent_id_val,
        tables=normalized_tables,
        grain=grain,
        select_cols=normalized_select,
        group_by_cols=table_prefixed_group_by,
        order_by_cols=normalized_orderby,
        filters_param=intent.filters_param,
        having_param=intent.having_param,
        param_values=intent.param_values,
        question=intent.question,
        variant_idx=intent.variant_idx,
        limit=intent.limit,
        distinct=intent.distinct,
    )


def generate_all_intents(
    schema: SchemaGraph,
    column_roles: dict[str, str],
    num_intents: int | None = None,
    *,
    rng_seed: int | None = None,
) -> list[QSimIntent]:
    """
    Generate diverse ``QSimIntent`` rows using tier-balanced skeleton sampling and coverage targets.

    Args:

        schema: Schema graph for pools and LLM filling.

        column_roles: Map of column key to role string.

        num_intents: Target count; defaults to ``QSimConfig.INTENT_TYPES``.

        rng_seed: RNG seed for reproducible skeleton and geometric draws.

    Returns:

        Normalized, deduplicated intents up to ``num_intents``.
    """

    seed_val = rng_seed if rng_seed is not None else QSimConfig.RANDOM_SEED
    random.seed(seed_val)
    rng = random.Random(seed_val)
    load_or_create_skeletons(schema, column_roles)

    if num_intents is None:
        num_intents = QSimConfig.INTENT_TYPES

    cap = schema.database_feature_capability
    weights = rebalance_complexity_target_proportions(QSimConfig.COMPLEXITY_TARGET_PROPORTIONS, cap)
    quotas = _allocate_tier_int_quotas(weights, num_intents)
    tier_remaining: dict[str, int] = dict(quotas)
    adv_txt = _advanced_feature_prompt_block(cap)

    min_with_filters = int(num_intents * QSimConfig.MIN_FILTER_RATIO)
    min_with_having = int(num_intents * QSimConfig.MIN_HAVING_RATIO)
    min_three_table = int(num_intents * QSimConfig.MIN_THREE_TABLE_RATIO)
    max_no_variance = int(num_intents * QSimConfig.MAX_NO_VARIANCE_RATIO)

    debug(
        f"[qsim_ops.generate_all_intents] targeting {num_intents} intents tier_quotas={tier_remaining} "
        f"rebalanced_weights={weights} schema_tables={cap.table_count}",
    )

    merged_buckets = _build_merged_tier_buckets(schema, column_roles)

    intents: list[QSimIntent] = []
    seen_ids: set[str] = set()
    table_set_usage: dict[str, int] = {}
    no_variance_count = 0

    consecutive_duplicates = 0
    consecutive_failures = 0

    while len(intents) < num_intents:
        if consecutive_duplicates >= QSimConfig.MAX_CONSECUTIVE_DUPLICATES:
            debug("[qsim_ops.generate_all_intents] EARLY_EXIT: consecutive duplicate cap")
            break
        if consecutive_failures >= QSimConfig.MAX_CONSECUTIVE_FAILURES:
            debug("[qsim_ops.generate_all_intents] EARLY_EXIT: consecutive failure cap")
            break
        if sum(tier_remaining.values()) <= 0:
            debug("[qsim_ops.generate_all_intents] STOP: tier quotas exhausted")
            break

        tier_key = _pick_weighted_tier(tier_remaining, rng)
        if tier_key is None:
            break

        target_enum = ComplexityTier(str(tier_key))
        bucket_key = _skeleton_bucket_lookup_key(target_enum)
        bucket = merged_buckets.setdefault(bucket_key, [])

        current_with_filters = len([i for i in intents if i.filters_param])
        current_with_having = len([i for i in intents if i.having_param])
        need_filters = current_with_filters < min_with_filters
        need_having = current_with_having < min_with_having

        selection = _pop_matching_skeleton(bucket, need_filters, need_having)
        if selection is None:
            tier_remaining[str(tier_key)] = 0
            debug(f"[qsim_ops.generate_all_intents] exhausted skeleton bucket for tier={tier_key}")
            continue

        skeleton, table_set = selection

        if _is_no_variance_skeleton(skeleton) and no_variance_count >= max_no_variance:
            bucket.append((skeleton, table_set))
            debug(
                f"[qsim_ops.generate_all_intents] SKIPPING: no-variance budget exceeded ({no_variance_count}/{max_no_variance})",
            )
            continue

        sel_target = _sample_select_col_count_geometric(QSimConfig.SELECT_COL_GEOMETRIC_P, rng)

        intent = _llm_fill_intent(
            skeleton,
            schema,
            column_roles,
            target_tier=target_enum,
            select_col_target=sel_target,
            advanced_feature_lines=adv_txt,
        )

        if not intent:
            consecutive_failures += 1
            bucket.append((skeleton, table_set))
            debug(f"[qsim_ops.generate_all_intents] LLM failed, consecutive_failures={consecutive_failures}")
            continue

        consecutive_failures = 0

        normalized = _normalize_qsim_intent(intent, schema)

        if normalized.intent_id in seen_ids:
            consecutive_duplicates += 1
            bucket.append((skeleton, table_set))
            debug(
                f"[qsim_ops.generate_all_intents] DUPLICATE intent_id={normalized.intent_id}, "
                f"consecutive_duplicates={consecutive_duplicates}",
            )
            continue

        consecutive_duplicates = 0

        if _is_no_variance_skeleton(skeleton):
            no_variance_count += 1

        table_set_key = "|".join(sorted(table_set))
        table_set_usage[table_set_key] = table_set_usage.get(table_set_key, 0) + 1

        intents.append(normalized)
        seen_ids.add(normalized.intent_id)
        tier_remaining[str(tier_key)] -= 1
        debug(
            f"[qsim_ops.generate_all_intents] ADDED intent_id={normalized.intent_id}, tier={tier_key}, "
            f"tables={normalized.tables}, filters={len(normalized.filters_param)}, "
            f"having={len(normalized.having_param)}, total={len(intents)}/{num_intents}",
        )

    final_with_filters = len([i for i in intents if i.filters_param])
    final_with_having = len([i for i in intents if i.having_param])
    single_count = len([i for i in intents if len(i.tables) == 1])
    two_count = len([i for i in intents if len(i.tables) == 2])
    three_count = len([i for i in intents if len(i.tables) >= 3])

    debug(
        f"[qsim_ops.generate_all_intents] generated {len(intents)} intents: single={single_count}, two={two_count}, three={three_count}",
    )
    debug(
        f"[qsim_ops.generate_all_intents] coverage: with_filters={final_with_filters}/{min_with_filters}, "
        f"with_having={final_with_having}/{min_with_having}, three_table={three_count}/{min_three_table}, "
        f"no_variance={no_variance_count}/{max_no_variance}",
    )
    debug(
        f"[qsim_ops.generate_all_intents] table_set_usage: "
        f"{dict(sorted(table_set_usage.items(), key=lambda x: x[1], reverse=True)[:10])}",
    )

    return intents


def greedy_cover_indices_by_atoms(
    atoms_per_row: list[frozenset[str]],
    universe: frozenset[str],
) -> list[int]:
    """
    Greedy set-cover ordering of row indices over a discrete atom universe.

    Args:

        atoms_per_row: Coverage atom sets aligned to candidate row indices.

        universe: Full atom set to cover (typically the union over feasible rows).

    Returns:

        Indices chosen in greedy marginal-gain order until exhaustion or stagnation.
    """

    uncovered = set(universe)
    picked: list[int] = []
    available = list(range(len(atoms_per_row)))
    while uncovered and available:
        best_i = -1
        best_gain = -1
        for idx in available:
            gain = len(atoms_per_row[idx] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = idx
        if best_i < 0 or best_gain <= 0:
            break
        picked.append(best_i)
        uncovered -= atoms_per_row[best_i]
        available.remove(best_i)
    return picked
