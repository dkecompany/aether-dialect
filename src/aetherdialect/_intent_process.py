"""LLM intent parsing, repair loops, template matching, and union helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, NamedTuple
from types import MappingProxyType

import jsonschema

from ._config import (
    DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART,
    DIAGNOSTIC_CODE_STAGE_A_RETRY,
    DIAGNOSTIC_CODE_STAGE_B_REPAIR,
    LOGICAL_INTENT_SCHEMA,
    MAX_NON_AGG_COL_DIFF,
    MAX_STAGE_A_RETRIES,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    VALID_HAVING_OPS,
    GenerationPath,
    PolicyConfig,
    diagnostic_debug_enabled,
    diagnostic_pipeline_trace_full_enabled,
)
from ._contracts_base import (
    ConfigError,
    CteIntent,
    FailureCategory,
    IntentIssue,
    LogicalIntent,
    SchemaGraph,
    make_intent_issue,
)
from ._contracts_core import (
    CaseRegistryStep,
    ConcreteIntent,
    FeedbackKind,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RejectionBucket,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    StructuralCompareResult,
    Template,
    WindowRegistryStep,
    concrete_intent_to_runtime_skeleton,
    expr_registry_ref,
    intent_prompt_structural_index,
    runtime_intent_to_concrete,
)
from ._core_utils import (
    debug,
    llm_chat,
    notify,
    pipeline_trace_lazy,
    safe_json_loads,
    stable_json,
)
from ._dialect import extra_filter_ops_for_engine
from ._intent_expr import (
    assign_param_keys,
    build_cte_output_metadata,
    canonicalize_temporal_unit_args,
    collect_raw_param_values,
    decompose_between_params,
    derive_cte_output_columns,
    ensure_scalar_func_defaults,
    extract_structural_params,
    normalize_date_diff_raw_values,
    normalize_in_raw_values,
    parse_intent_response,
    repair_misclassified_date_diff,
    replace_refs_in_expr,
    tag_case_when_condition_scope,
    tag_expr_numeric,
)
from ._intent_repair import (
    align_filter_value_type_to_exprs,
    auto_repair_filter_having,
    dedup_contradictory_filters,
    dedup_value_vs_right_expr,
    drop_invalid_case_registry_entries,
    enforce_sensitivity_policy_intent,
    expand_fk_select_to_descriptive,
    lift_distinct_modifier_in_multiply,
    normalize_boolean_filter_values,
    normalize_in_filter_types,
    normalize_null_filter_values,
    normalize_pk_distinct,
    qualify_cte_output_columns,
    reconcile_tables,
    repair_array_filters_intent,
    repair_case_when_intent,
    repair_cumulative_phrasing_window_intent,
    repair_fk_filter_type_mismatch,
    repair_intent_placeholder_tokens,
    repair_null_equality_filters,
    replace_unknown_scalar_funcs,
    resolve_filter_value_case,
    runtime_intent_has_instructional_placeholders,
    strip_impossible_having,
    strip_join_conditions,
    strip_spurious_group_by,
)
from ._intent_resolve import (
    _coerce_filter_group_list,
    _coerce_having_group_list,
    _normalized_expr_is_absent,
    apply_aggregatability_gate,
    attribute_post_stage_b_issue,
    canonicalize_registry_ids,
    check_qualified_refs_exist,
    coerce_filter_group_mode,
    collect_column_refs_for_post_processing,
    enforce_case_branch_param_keys,
    enforce_cte_grain_consistency,
    enforce_grain_consistency,
    lift_distinct_select_from_raw_sql,
    normalize_count_star,
    normalize_cte_names,
    normalize_filters_havings,
    qualify_cte_count_star_mulgroups,
    prune_unused_cte_output_columns,
    prune_unused_cte_steps,
    reorder_cte_steps_by_dag,
    resolve_column_map,
    resolve_cte_column_maps,
    rewrite_cte_output_refs_to_aliases,
    rewrite_main_query_refs_to_final_cte_columns,
    simplify_exprs,
    sort_select_cols,
    validate_cte_dependencies,
    validate_tables_exist,
)
from ._sql_gen import classify_cte_emission
from ._utils import (
    QuestionReuseMatch,
    body_similarity_key,
    body_similarity_key_for_concrete,
    intent_key,
    match_question_against_template_history,
    template_instance_key_from_parts,
)
from ._validation_execute import validate_semantics

_TEMPLATES_MODULE: Any = None

_INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES = (
    "Semantically-required join intermediates. The join enumerator only considers the SHORTEST FK paths "
    "connecting the tables in `tables`, and any table listed in `tables` whose columns are not referenced "
    "elsewhere in the intent (in `select_cols`, filters, `group_by_cols`, `having_param`, `order_by_cols`, "
    "or registry expressions) is pruned before enumeration. Therefore, listing an extra table in `tables` "
    "alone is never enough to force the join through it. "
    "When the descriptions on the endpoint tables (or on tables related to them) indicate that two "
    "semantically distinct FK paths connect the same pair of tables, and one path is strictly LONGER than "
    "the other, the shorter path will always be chosen unless you force the longer one explicitly. "
    "To force the longer path, reference at least one column from each required intermediate table inside "
    "the intent. Adding such a column to `select_cols` is preferred — typically the intermediate table's "
    "primary key or its most descriptive column — because it both keeps the table from being pruned and "
    "surfaces context to the user that this specific semantic was intended. "
    "Concretely, suppose the intent connects `tbl_a` to `tbl_b` and the descriptions indicate two semantic "
    "paths: `tbl_a -> tbl_x -> tbl_b` for one semantic and `tbl_a -> tbl_y -> tbl_z -> tbl_b` for another. "
    "If the question requires the second semantic, add at least one column from `tbl_y` or one column from "
    "`tbl_z` (preferably to `select_cols`, ideally a primary key or the most descriptive column from each). "
    "If the question requires the first semantic, no extra columns are needed because the shorter path is "
    "already what the resolver will pick. When descriptions are silent about distinct paths, or when only "
    "one FK path exists between the endpoints, no extra columns are needed."
)

_INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE = (
    "`natural_language` field shape. The `natural_language` field must read like a question a non-technical "
    "user would ask, not a description of how the SQL is computed. Even when the intent uses CTEs, derived "
    "steps, or window functions, compress the overall information need into a single conversational sentence. "
    "Never enumerate steps, never reference CTE names, never mention SQL operators. The reader of this field "
    'is the end user being asked "I understood: ; is this what you wanted?"'
)


def _dedupe_prior_question_feedback_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return *rows* de-duplicated by ``(intent_structural_hash, summary prefix)``, preserving order."""

    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ihash = str(row.get("intent_structural_hash", "") or "")
        summary = str(row.get("summary", "") or "")
        key = (ihash, summary[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _in_turn_row_from_semantic_errors(
    errors: list[IntentIssue],
    schema_hash: str,
    intent: RuntimeIntent,
) -> dict[str, str]:
    """Build one ``to_prompt_row``-shaped dict from semantic validation errors."""

    max_b = PolicyConfig.MAX_SUMMARY_BULLETS
    lines = [f"[{e.category.value}] {e.message}" for e in errors[:max_b]]
    summary = "\n".join(lines)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    tplm = _TEMPLATES_MODULE
    if tplm is None:
        ish, ipay = "", ""
    else:
        ish, ipay = tplm.compute_intent_structural_signature(intent)
    return {
        "kind": FeedbackKind.VALIDATION_FAILURE.value,
        "summary": summary,
        "buckets": RejectionBucket.OTHER.value,
        "effective_structural_hash": schema_hash,
        "intent_structural_hash": ish,
        "intent_payload": ipay,
        "created_at": ts,
        "updated_at": ts,
        "is_post_restart": "False",
    }


def register_templates_module(module: Any) -> None:
    global _TEMPLATES_MODULE
    _TEMPLATES_MODULE = module


INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR = (
    "Instructional placeholder tokens appear in expression strings. Replace "
    "each with exact table.column names from schema_info. Do not leave "
    "angle-bracket markup, table_N or column_N instructional tokens, or "
    "synthetic shape tokens from the prompt (tbl_a, tbl_b, col_a, date_a, date_b)."
)

REPAIR_INSTRUCTIONS: dict[str, str] = {
    "extract_epoch": (
        "Do not use EXTRACT(EPOCH FROM ...). Use date column subtraction "
        "(e.g. tbl_a.date_a - tbl_b.date_b) or other supported "
        "date functions for time differences."
    ),
    "grain_validity": "Use one of the allowed grain values: 'scalar', 'grouped', or 'row_level'.",
    "grain_consistency": "Ensure grain matches the query structure. 'grouped' requires group_by_cols and aggregation in select_cols. 'row_level' means no GROUP BY and no aggregation. 'scalar' means a single aggregated value with no GROUP BY.",
    "schema_validation": "One or more tables or columns do not exist in the schema. Check allowed_tables and use only exact column names from each table.",
    "unknown_table": "The table does not exist in the schema. Remove it from tables and rewrite any references to use only tables that appear in allowed_tables.",
    "unknown_column": "The column does not exist in its table. Check the schema for available columns with a similar name or meaning and replace the reference.",
    "semantic_contradiction": "The intent contains contradictory operations. Keep only the aggregation or pattern that matches the question.",
    "expression_type": "Arithmetic expressions require numeric columns. Ensure all operands in arithmetic and all comparison sides have compatible types.",
    "filter_aggregation": "Conditions with aggregation functions (COUNT, SUM, AVG, MIN, MAX) belong in having_param, not filters_param. Move the condition.",
    "having_aggregation": "HAVING conditions must have an aggregation function in left_expr. Conditions without aggregation belong in filters_param.",
    "filter_semantic": "Fix the filter comparison: remove self-comparisons and ensure type compatibility between left and right expressions.",
    "having_semantic": "Fix the HAVING comparison: remove self-comparisons and ensure type compatibility between aggregated expressions.",
    "nested_aggregation": "Nested aggregation (e.g. SUM(COUNT(...))) is not allowed. Use a CTE: compute the inner aggregation in a CTE step, then aggregate the CTE output in the main query.",
    "mixed_aggregation": "An expression cannot mix aggregated and bare column terms. Either wrap all terms in an aggregation function or add bare columns to group_by_cols.",
    "group_by_membership": "Every non-aggregated column in select_cols must appear in group_by_cols when grain is 'grouped'. Add the missing column to group_by_cols or wrap it in an aggregation.",
    "order_by_aggregation": "ORDER BY cannot contain aggregation when grain is 'row_level'. Change grain to 'grouped' or remove the aggregation from order_by_cols.",
    "aggregation_hint": "The question contains a quantity-comparison phrase that typically requires COUNT or SUM aggregation with GROUP BY and HAVING. Add aggregation in select_cols, group_by_cols on the entity, having_param with the threshold, and set grain to 'grouped'.",
    "column_schema": "A referenced column does not exist in its table. Check the schema for the correct column name.",
    "table_schema": "A referenced table does not exist in the schema. Use only tables from allowed_tables.",
    "filter_type_ops": "The filter operator is not compatible with the column data type. Use an appropriate operator for the column type.",
    "null_filter": "NULL checks must use 'is null' or 'is not null' operator with no value or value_type field.",
    "between_filter": (
        "Use op 'between' with value as a two-element array [lower, upper]. "
        "The pipeline decomposes it into >= and <= automatically."
    ),
    "date_window": (
        "For relative time filters (e.g. 'last 90 days', 'past 6 months'), "
        "use value_type 'date_window' with the date column in left_expr, "
        'op \'>=\' and value as {"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. The amount is interpreted as an ISO half-open window: "
        "the start is N units before today, the end is exclusive. Prefer unit 'day' whenever "
        "the question phrases the window in days, weeks, or fortnights (convert weeks to days as N*7) "
        "so the start/end boundary matches the schema's daily granularity."
    ),
    "date_diff": (
        "For date-difference filters comparing two date columns "
        "(e.g. tbl_a.date_b - tbl_a.date_a compared to N days), use value_type 'date_diff' with "
        "left_expr as the date subtraction expression (later minus earlier per the question), "
        "op as the comparison operator, and value as "
        '{"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. Prefer unit 'day' whenever the question phrases the duration in "
        "days or weeks (convert weeks to days as N*7). Do NOT use date_diff for relative date-window "
        "filters; use date_window instead."
    ),
    "agg_role": "SUM and AVG should only be applied to numeric measure columns. Use COUNT for non-measure columns, or select a numeric column.",
    "agg_type": "SUM and AVG require a numeric column. The referenced column is not numeric. Use COUNT instead or choose a numeric column.",
    "for_each_grouping": "The question contains a 'for each', 'per', or 'by' phrase implying a GROUP BY on the referenced entity. Add the entity's identifying column to group_by_cols, include it as a non-aggregated entry in select_cols, and set grain to 'grouped'.",
    "scalar_func_type": "The scalar function is applied to an incompatible column type. Ensure the column type matches what the function expects (e.g. YEAR needs a date column, UPPER needs a text column).",
    "threshold_missing_having": "The question contains a threshold phrase (e.g. 'more than', 'at least') and the intent already has aggregation, but no HAVING condition is defined. Add a HAVING clause that compares the aggregated column to the numeric threshold in the question.",
    "cte_structure": "CTE steps require a cte_name string, an output_columns list of alias strings, and valid tables.",
    "cte_grain_consistency": "CTE grain must match its structure: same rules as the main query regarding grain, group_by_cols, and aggregation.",
    "cte_table_reference": "A CTE references an unknown table. A CTE can only reference schema tables or CTEs defined earlier in the same WITH list.",
    "cte_grain_compatibility": "A row_level query or CTE depends on an aggregated CTE. Ensure the grain is compatible with upstream CTE grains.",
    "cte_aggregation": "A CTE has HAVING conditions but no aggregation in its select_cols. Add aggregation or remove the HAVING.",
    "missing_scoping_table": "The question explicitly mentions a schema table that is missing from the intent. Add the table to 'tables', join it via its foreign-key relationship, and include relevant columns in select_cols or filters as appropriate.",
    "agg_keyword_missing": "The question asks for an aggregation (total, count, average, sum, etc.) but the intent has no aggregated column and no HAVING condition. Add the appropriate aggregation function to select_cols, include all tables needed to compute the aggregated value, and set grain to 'grouped' with the correct group_by_cols.",
}

if PolicyConfig.MAX_STAGE_B_REPAIRS < 1:
    raise ConfigError("PolicyConfig.MAX_STAGE_B_REPAIRS must be >= 1")
if PolicyConfig.MAX_FRESH_RESTARTS < 0:
    raise ConfigError("PolicyConfig.MAX_FRESH_RESTARTS must be >= 0")


@dataclass
class RestartBudget:
    """Mutable counter bounding the number of fresh full-parse restarts per top-level invocation."""

    fresh_restarts_left: int

    @classmethod
    def default(cls) -> RestartBudget:
        """Construct a budget initialised from :data:`PolicyConfig.MAX_FRESH_RESTARTS`."""
        return cls(fresh_restarts_left=PolicyConfig.MAX_FRESH_RESTARTS)


_PLANNER_NL_CONVENTIONS_BODY: dict[str, Any] = {
    "mandatory": [
        "Copy every literal that constrains rows or ordering into the matching prose field; the encoder never sees the original question.",
        "List only semantic base tables in tables; omit junction tables.",
        "Reference window and case output names from select when you define them in window or case using as <registry_name>.",
        "Never emit SQL set operators, EXISTS, NOT EXISTS, LATERAL, param_key, raw_value, wNN, cNN, filter_group, or IR vocabulary.",
        "Never use as <name> inside select, filter, having, group_by, order_by, or limit; select output aliases are assigned later by the pipeline.",
        "Leave group_by, order_by, and limit empty unless the question explicitly asks for grouping, ordering, or a row cap; do not invent presentation-layer sorting or grouping for context.",
        "Project select prose to primary keys, primary human-readable label columns such as names or titles, and every column referenced by stated filters, ordering, grouping, having, or limits; never enumerate every physical column unless the question explicitly asks for all columns, every column, or complete row dumps.",
        "Set schema_invalid to true in the planner JSON only when the question cannot be represented with the listed semantic base tables and prose fields; otherwise omit schema_invalid or set it to false. The structural formatter never authors this flag.",
    ],
    "recommended": [
        "Qualify columns as tbl_a.col_x when multiple listed tables share a column name.",
        "Describe aggregates with phrases such as sum of tbl_b.col_y.",
        "Describe anti-existence with plain language or tbl_b.pk is null style wording.",
    ],
}

PLANNER_NL_CONVENTIONS: Mapping[str, Any] = MappingProxyType(_PLANNER_NL_CONVENTIONS_BODY)

PLANNER_NL_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "tables": ["tbl_a", "tbl_b"],
        "select": "tbl_a.col_x and the sum of tbl_b.col_y per tbl_a.col_x",
        "filter": "tbl_a rows whose tbl_a.col_z equals 'L1'",
        "group_by": "tbl_a.col_x",
        "having": "the sum of tbl_b.col_y is greater than 100",
        "order_by": "the sum of tbl_b.col_y descending",
        "limit": "10",
        "window": "",
        "case": "",
        "cte_steps": [],
    },
    {
        "tables": ["tbl_a", "tbl_b"],
        "select": "tbl_a.col_x",
        "filter": "tbl_a rows where tbl_b.pk is null",
        "group_by": "",
        "having": "",
        "order_by": "",
        "limit": None,
        "window": "",
        "case": "",
        "cte_steps": [],
    },
    {
        "tables": ["tbl_a"],
        "select": "tbl_a.col_x, tbl_a.col_y, and metric_x from the step_a step",
        "filter": "rows where metric_x is at most 3",
        "group_by": "",
        "having": "",
        "order_by": "tbl_a.col_y ascending, metric_x ascending",
        "limit": None,
        "window": "",
        "case": "",
        "cte_steps": [
            {
                "name": "step_a",
                "depends_on": ["tbl_a"],
                "tables": ["tbl_a"],
                "select": "tbl_a.pk and metric_x from the per-group ranking",
                "filter": "",
                "group_by": "",
                "having": "",
                "order_by": "",
                "limit": None,
                "window": "rank tbl_a.col_z descending partitioned by tbl_a.col_y as metric_x",
                "case": "",
            }
        ],
    },
    {
        "tables": ["customer"],
        "select": "customer.customer_id and customer.first_name",
        "filter": "customer rows whose customer.active equals true",
        "group_by": "",
        "having": "",
        "order_by": "",
        "limit": None,
        "window": "",
        "case": "",
        "cte_steps": [],
    },
)

ENCODER_NL_PHRASE_MAPPINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "=": ("equals", "equal to", "is", "are", "="),
        "!=": ("not equal", "not equals", "different from", "!="),
        ">": ("greater than", "more than", "above", "over", ">"),
        ">=": ("at least", "greater than or equal to", ">="),
        "<": ("less than", "below", "under", "<"),
        "<=": ("at most", "less than or equal to", "<="),
        "like": ("like", "matches", "contains pattern"),
        "not like": ("not like", "does not match"),
        "in": ("in", "one of", "any of", "among"),
        "not in": ("not in", "none of", "not among"),
        "between": ("between", "from", "to"),
        "is null": ("is null", "is absent", "is missing", "has no", "without"),
        "is not null": ("is not null", "has", "with", "present"),
        "sum": ("sum of", "total of", "sum", "total"),
        "avg": ("average of", "avg of", "mean of", "average"),
        "min": ("minimum of", "min of", "smallest", "lowest"),
        "max": ("maximum of", "max of", "largest", "highest"),
        "count": ("count of", "number of", "how many"),
        "asc": ("ascending", "asc", "lowest first"),
        "desc": ("descending", "desc", "highest first"),
    }
)

ENCODER_NL_TO_IR_GUIDANCE: tuple[str, ...] = (
    "Match planner prose phrases against nl_phrase_mappings case-insensitively to pick IR operators and aggregates.",
    "Infer value_type from literal form: single-quoted text is string; bare digits are integer or number; ISO dates are date; true or false are boolean; is null operator uses value_type null without a value.",
    "Emit AggregateCol with alias empty string; never invent select-column aliases.",
    "WindowRegistryStep.name and CaseRegistryStep.name come from as <registry_name> inside planner window and case prose; CTE names come from cte_steps[].name.",
)

ENCODER_IR_ASSEMBLY_RULES: tuple[str, ...] = (
    "filter_group is OR-of-AND groups parsed from planner prose.",
    "Emit raw_value for every Filter and Having literal; never emit param_key or param_values.",
    "Every column_ref is table.column using only structural_schema_for_chosen_tables identifiers.",
    "Runtime tables lists are overwritten to match the planner immediately after Stage B parse; emit tables consistent with planner tables at the main query and each CTE with the same cte_name.",
    "For CONCAT expressions, place every concat argument into the same MulGroup.multiply list under scalar_func='concat'. Do not introduce divisors or coefficients for CONCAT groups. Use only COUNT as the outer aggregation wrapper for a CONCAT MulGroup; SUM, AVG, MIN, and MAX are not valid as that outer aggregation.",
)


INTENT_CRITICAL_RULES: tuple[str, ...] = (
    "Every JSON object uses only keys listed in structural_json_keys for its structural type "
    "(the root object follows RuntimeIntent; nested rows follow their named type; SQL expressions "
    "are always a single string field such as expr or left_expr per structural_json_keys.sql_expression). "
    "Do not emit extra sibling keys at any level.",
    "Qualify every column reference as table.column using names from the schema text and allowed_tables; "
    "never emit bare column names except the bare wNN and cNN registry tokens on select_cols that point at window_registry and case_registry entries. "
    "Qualify columns inside every window_registry.window_spec.partition_by, order_by, and argument, and inside every case_registry case_when branch (condition sides, result, else_result).",
    "Use only tables and columns from the provided schema text and allowed_tables; do not invent identifiers.",
    "Join path discovery, foreign-key traversal, and bridge or junction tables are handled only by the downstream engine after this JSON is parsed; never refuse or shrink the intent because tables look disconnected in the structural payload.",
    "Do not judge whether the question is answerable from schema connectivity; translate the planner prose into IR using only the listed planner tables and their columns.",
    "Grain must match structure: grouped requires group_by_cols and aggregation in select_cols; "
    "row_level means no GROUP BY and no aggregation in select_cols; "
    "scalar is a single aggregated result with no GROUP BY.",
    "Row-level predicates belong in filters_param; predicates on aggregates belong in having_param. "
    "Never put join predicates in filters_param or having_param.",
    "SUM and AVG apply only to numeric measure columns; use COUNT for non-measure columns.",
    "Nested aggregation (e.g. SUM(COUNT(...))) is forbidden; compute inner aggregates in a CTE step, then aggregate in the main query.",
    "Do not use EXTRACT(EPOCH FROM ...) for time differences; subtract date columns directly or use supported date functions.",
    "CTE output_columns are snake_case alias tokens matching ^[a-z_][a-z0-9_]*$; never qualified table.column, never function call text, never AS clauses; align positionally with select_cols.",
    "Relative date-window filters use value_type date_window with value {\"unit\", \"amount\"}; "
    "column-to-column date spans use value_type date_diff with value {\"unit\", \"amount\"}. Use singular unit names.",
    "BETWEEN uses op between with value [lower, upper]. NULL checks use op is null or is not null without a value field.",
    "filter_group (integer) labels OR-of-AND blocks: predicates sharing a filter_group are joined by AND; "
    "distinct filter_group values are joined by OR. Use bool_op only when every row has filter_group unset "
    "(a single AND chain or a flat AND/OR chain). Do not put bool_op on rows that carry filter_group.",
    "window_registry defines registry_id and window_spec; select_cols reference entries with bare wNN tokens. "
    "Never put a window_spec key on a select_cols entry.",
    "case_registry defines registry_id and case_when with non-empty branches; select_cols reference entries with bare cNN tokens. "
    "When the question asks for conditional labels or buckets over columns, populate case_registry rather than dropping the derived column.",
    "SELECT DISTINCT prefixes the column expr with the bare token DISTINCT and a space "
    "(e.g. 'DISTINCT tbl_a.col_a'). Use COUNT(DISTINCT tbl_a.col_a) for distinct counts; "
    "do not wrap DISTINCT around arbitrary expressions except as COUNT(DISTINCT ...). "
    "Do not embed COUNT(*) inside arithmetic subexpressions—use COUNT(*) only as a top-level aggregate where appropriate.",
    "Arithmetic combines expressions with +, -, *, /; aggregations may wrap arithmetic (e.g. SUM(tbl_a.col_a * tbl_a.col_b)). "
    "Subtract date columns directly (tbl_a.date_a - tbl_a.date_b) for day differences.",
    "String concatenation uses CONCAT(expr1, ' ', expr2, ...) in expr strings; do not use the SQL || operator (pipe-pipe).",
    "Apply scalar functions such as ROUND after aggregates when needed (e.g. ROUND(SUM(tbl_a.col_a), 2)).",
    "Use exact identifiers from the provided schema text; never leave synthetic shape tokens from this prompt "
    "(tbl_a, tbl_b, col_a, date_a, date_b), generic instructional tokens (table_N, column_N), or angle-bracket markup in expressions.",
)

INTENT_PARSE_RULES_APPEND: tuple[str, ...] = (
    "output_format lists every required top-level key; use [] for empty arrays and null for unused scalars.",
    "natural_language is required: a short, direct description of what the query returns using real entity names from the question and schema.",
    (
        "Constructs the intermediate representation cannot emit directly "
        "(set difference, EXISTS / NOT EXISTS, anti-joins, correlated subqueries, lateral joins) "
        "should be reformulated using available primitives such as IS NULL / IS NOT NULL filters on "
        "left-joined sources or conditional aggregations inside GROUP BY."
    ),
    (
        "For (predicate_A OR predicate_B) emit two filters_param objects with different filter_group integers "
        "and no bool_op fields. "
        "For (predicate_A AND predicate_B) OR (predicate_C AND predicate_D) emit four objects with "
        "filter_group values 1, 1, 2, 2 and no bool_op. "
        "Each disjunct gets its own integer filter_group; predicates inside the same disjunct share the same "
        "filter_group. Do not emit bool_op on rows that have a filter_group. "
        "Use qualified column references from the schema for left_expr; bind literals via value placeholders as elsewhere in these rules. "
        "Do not nest filter_group as an array; each filters_param element must include op."
    ),
)

PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE: str = (
    "When the question can be answered with a single SELECT whose only added structure is a window function "
    "over the primary table, do not introduce a CTE. Emit the window function directly in the main SELECT. "
    "CTEs are appropriate when reused across multiple SELECT bodies or when the question explicitly asks for "
    "staged computation."
)

INTENT_FORMAT_REPAIR_JSON_RULES: tuple[str, ...] = (
    "Return JSON only with no prose, markdown fences, or trailing commas.",
    "Preserve intent content while correcting syntax; ensure all required fields are present.",
    "Use [] for empty array fields and null for absent optional scalars.",
)


_INTENT_STAGE_A_SYSTEM = (
    "You are a logical intent planner for text-to-SQL. Output ONLY valid JSON matching the logical_intent_json_schema "
    "embedded in the user payload. Follow nl_conventions and nl_examples. Never emit SQL, IR operators, EXISTS, "
    "NOT EXISTS, UNION, INTERSECT, or EXCEPT. The downstream structural formatter does not see the original question; "
    "your JSON is the sole source of truth for literals and table choice."
)

_INTENT_STAGE_B_SYSTEM = (
    "You are a structural intent formatter for text-to-SQL. Output ONLY valid JSON matching output_format. "
    "Translate logical_intent natural language into the runtime IR using nl_phrase_mappings for phrasing hints. "
    "logical_intent is the sole source of truth. Use only identifiers from structural_schema_for_chosen_tables. "
    "NEVER emit param_key, param_values, or any harvested-literal mapping; emit raw_value for each Filter and "
    "Having literal and leave SelectCol.alias and AggregateCol.alias empty strings."
)

_LOGICAL_DECOMPOSITION_GUIDANCE: tuple[str, ...] = (
    "Describe each prose field thoroughly enough that a structural converter can build the IR without re-reading the question.",
    "Only tables may name real schema tables; every other field is natural language.",
    "Use cte_steps when the question needs a reusable intermediate; each step lists name, depends_on, tables, and the same prose fields as the top level.",
    "Put window definitions in the window prose field and case definitions in the case prose field; use as <registry_name> only inside those two fields.",
    "Never describe joins as explicit paths; the engine discovers FK paths.",
    PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE,
)

_FORMAT_STRUCTURAL_GUIDANCE: tuple[str, ...] = (
    "Window semantics belong in window_registry only; map logical_intent.window and each cte_steps[].window prose into WindowRegistryStep rows with wNN ids.",
    "Case semantics belong in case_registry only; map logical_intent.case and each cte_steps[].case prose into CaseRegistryStep rows with cNN ids.",
    "Reference registry outputs from select_cols using window_ref or case_ref tokens alongside other projected columns.",
    "Do not encode row filters as CASE branches; use filters_param for row membership.",
    "Never put AVG, SUM, COUNT, MIN, MAX calls or OVER (...) frames inside filters_param.right_expr as raw_sql. "
    "Compare to aggregates using an extra cte_steps row, a scalar subquery shape allowed by the IR, or window_registry references.",
)

_INTENT_ANSWER_STYLE_GUIDANCE: tuple[str, ...] = _LOGICAL_DECOMPOSITION_GUIDANCE + _FORMAT_STRUCTURAL_GUIDANCE


def _logical_coerce_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return []


def _logical_coerce_optional_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _logical_coerce_limit(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    s = str(v).strip()
    return s or None


def _logical_coerce_bool(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _cte_intent_from_obj(obj: dict[str, Any]) -> CteIntent:
    """Materialise one :class:`CteIntent` from a planner JSON object."""

    if not isinstance(obj, dict):
        return CteIntent(name="", tables=(), select="")
    return CteIntent(
        name=str(obj.get("name", "") or "").strip(),
        depends_on=tuple(_logical_coerce_str_list(obj.get("depends_on"))),
        tables=tuple(_logical_coerce_str_list(obj.get("tables"))),
        select=str(obj.get("select", "") or "").strip(),
        filter=_logical_coerce_optional_str(obj.get("filter")),
        group_by=_logical_coerce_optional_str(obj.get("group_by")),
        having=_logical_coerce_optional_str(obj.get("having")),
        order_by=_logical_coerce_optional_str(obj.get("order_by")),
        limit=_logical_coerce_limit(obj.get("limit")),
        window=_logical_coerce_optional_str(obj.get("window")),
        case=_logical_coerce_optional_str(obj.get("case")),
    )


def logical_intent_from_parsed(d: dict[str, Any]) -> LogicalIntent:
    """Materialise a :class:`LogicalIntent` from validated planner JSON."""

    ctes: list[CteIntent] = []
    for c in d.get("cte_steps") or []:
        if isinstance(c, dict):
            ctes.append(_cte_intent_from_obj(c))
    return LogicalIntent(
        tables=tuple(_logical_coerce_str_list(d.get("tables"))),
        select=str(d.get("select", "") or "").strip(),
        filter=_logical_coerce_optional_str(d.get("filter")),
        group_by=_logical_coerce_optional_str(d.get("group_by")),
        having=_logical_coerce_optional_str(d.get("having")),
        order_by=_logical_coerce_optional_str(d.get("order_by")),
        limit=_logical_coerce_limit(d.get("limit")),
        window=_logical_coerce_optional_str(d.get("window")),
        case=_logical_coerce_optional_str(d.get("case")),
        cte_steps=tuple(ctes),
        schema_invalid=_logical_coerce_bool(d.get("schema_invalid")),
    )


def _logical_intent_schema_issues(parsed: dict[str, Any] | None) -> list[IntentIssue]:
    """Return Stage A JSON-schema violations as logical-stage issues."""

    if parsed is None or not isinstance(parsed, dict):
        return [
            make_intent_issue(
                issue_id="json_schema_violation_logical_root",
                category=FailureCategory.INTENT_SCHEMA_INVALID_ABORT,
                severity="error",
                message="Stage A logical intent JSON must be a single object.",
                responsible_stage="logical",
            )
        ]
    try:
        jsonschema.validate(instance=parsed, schema=LOGICAL_INTENT_SCHEMA)
    except jsonschema.ValidationError as exc:
        return [
            make_intent_issue(
                issue_id="json_schema_violation_logical_detail",
                category=FailureCategory.INTENT_SCHEMA_INVALID_ABORT,
                severity="error",
                message=str(exc.message),
                context={"path": [str(p) for p in exc.path]},
                responsible_stage="logical",
            )
        ]
    return []


def _parse_logical_intent_response(
    raw: str,
    schema_graph: SchemaGraph,
) -> tuple[LogicalIntent | None, list[IntentIssue]]:
    """Parse and validate a planner model payload against schema and graph rules."""

    obj = safe_json_loads(raw.strip())
    issues = _logical_intent_schema_issues(obj if isinstance(obj, dict) else None)
    if issues:
        return None, issues
    assert isinstance(obj, dict)
    li = logical_intent_from_parsed(obj)
    if not li.tables or not li.select.strip():
        return None, [
            make_intent_issue(
                issue_id="logical_intent_empty_core",
                category=FailureCategory.INTENT_SCHEMA_INVALID_ABORT,
                severity="error",
                message="Planner must populate tables and a non-empty select field.",
                responsible_stage="logical",
            )
        ]
    issues = list(validate_tables_exist(li.tables, schema_graph))
    issues.extend(validate_cte_dependencies(li))
    for step in li.cte_steps:
        issues.extend(validate_tables_exist(step.tables, schema_graph))
    if issues:
        return None, issues
    return li, []


def _serialized_prior_feedback_rows(rows: list[dict[str, str]] | None) -> str:
    """Serialise merged feedback rows for Stage A as compact JSON text."""

    if not rows:
        return ""
    return stable_json({"items": rows})


def _build_intent_logical_prompt(
    question: str,
    schema_literal_json: str,
    prior_question_feedback: str,
    logical_decomposition_guidance: str,
    prior_user_corrections: tuple[str, ...],
    prior_grounding_failures: tuple[str, ...],
) -> str:
    """Build the planner user JSON instructing the model to emit LOGICAL_INTENT_SCHEMA JSON."""

    body: dict[str, Any] = {
        "task": "Decompose the question into planner JSON matching logical_intent_json_schema.",
        "question": question,
        "schema_literal_json": schema_literal_json,
        "logical_intent_json_schema": LOGICAL_INTENT_SCHEMA,
        "nl_conventions": json.loads(stable_json(dict(PLANNER_NL_CONVENTIONS))),
        "nl_examples": [json.loads(stable_json(x)) for x in PLANNER_NL_EXAMPLES],
        "logical_schema_rules": [
            "Do not plan UNION, INTERSECT, or EXCEPT; describe set-like needs with plain language or CTE steps.",
            "Use the filter prose field for row predicates; never name EXISTS or NOT EXISTS.",
            "Use cte_steps for self-comparisons and per-entity top-N style questions without prescribing IR shapes.",
        ],
    }
    if prior_question_feedback.strip():
        body["prior_question_feedback"] = prior_question_feedback
    if logical_decomposition_guidance.strip():
        body["logical_decomposition_guidance"] = logical_decomposition_guidance
    if prior_user_corrections:
        body["prior_user_corrections"] = list(prior_user_corrections)
    if prior_grounding_failures:
        body["prior_grounding_failures"] = list(prior_grounding_failures)
    return stable_json(body)


def _logical_intent_to_serialisable(logical: LogicalIntent) -> dict[str, Any]:
    """Convert logical intent into JSON-serialisable dict for the encoder."""

    return {
        "tables": list(logical.tables),
        "select": logical.select,
        "filter": logical.filter,
        "group_by": logical.group_by,
        "having": logical.having,
        "order_by": logical.order_by,
        "limit": logical.limit,
        "window": logical.window,
        "case": logical.case,
        "schema_invalid": logical.schema_invalid,
        "cte_steps": [
            {
                "name": c.name,
                "depends_on": list(c.depends_on),
                "tables": list(c.tables),
                "select": c.select,
                "filter": c.filter,
                "group_by": c.group_by,
                "having": c.having,
                "order_by": c.order_by,
                "limit": c.limit,
                "window": c.window,
                "case": c.case,
            }
            for c in logical.cte_steps
        ],
    }


def _propagate_planner_schema_invalid_flag(intent: RuntimeIntent, logical: LogicalIntent) -> RuntimeIntent:
    """
    Overwrite runtime schema_invalid with the planner-only flag after structural parsing.

    Args:

        intent: Runtime intent produced by the structural formatter and post-alignment steps.

        logical: Stage A logical intent whose schema_invalid field is the sole source of truth for this signal.

    Returns:

        RuntimeIntent whose schema_invalid matches logical.schema_invalid regardless of formatter output.
    """

    return replace(intent, schema_invalid=logical.schema_invalid)


def _build_intent_format_prompt(logical: LogicalIntent, structural_schema_json: str) -> str:
    """Build the encoder user JSON that maps logical intent into runtime IR JSON."""

    parse_filter_ops = [
        "=",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "like",
        "not like",
    ]
    parse_filter_ops.extend(extra_filter_ops_for_engine())
    parse_filter_ops.extend(
        [
            "in",
            "not in",
            "is null",
            "is not null",
            "between",
            "contains",
        ]
    )
    structural_obj = json.loads(structural_schema_json)
    body: dict[str, Any] = {
        "task": (
            "Translate logical_intent into runtime intent JSON that conforms to field_specifications and output_format. "
            "Use only identifiers present in structural_schema_for_chosen_tables. Output ONLY valid JSON."
        ),
        "logical_intent": _logical_intent_to_serialisable(logical),
        "structural_schema_for_chosen_tables": structural_obj,
        "structural_json_keys": intent_prompt_structural_index(),
        "critical_rules": list(INTENT_CRITICAL_RULES),
        "parse_rules_append": list(INTENT_PARSE_RULES_APPEND),
        "field_specifications": dict(RuntimeIntent.PROMPT_FIELD_SPEC.items()),
        "output_format": RuntimeIntent.prompt_example_dict(),
        "nl_phrase_mappings": {k: list(v) for k, v in ENCODER_NL_PHRASE_MAPPINGS.items()},
        "nl_to_ir_guidance": list(ENCODER_NL_TO_IR_GUIDANCE),
        "ir_assembly_rules": list(ENCODER_IR_ASSEMBLY_RULES),
        "operator_reference": {
            "filter_ops": parse_filter_ops,
            "having_ops": sorted(VALID_HAVING_OPS),
        },
        "value_type_reference": {
            "filter": [
                "string",
                "integer",
                "number",
                "date",
                "boolean",
                "null",
                "date_window",
                "date_diff",
            ],
            "having": ["integer", "number"],
        },
        "structural_pattern_rules": [
            "Existence filters: INNER join plus DISTINCT on the X grain; anti-join uses LEFT join plus IS NULL on Y keys.",
            "Self-references: separate CTEs per view of the same base table, then join on the entity key—no inline self-join.",
            "Correlated lookups: prefer ROW_NUMBER partitioned per entity with outer filter row_number = 1—no LATERAL or correlated scalar subqueries.",
            "Mixed aggregates with row detail: compute aggregates in a grouped CTE and join back for row-level columns.",
            "Planner having prose maps to HAVING on the aggregated source, not outer WHERE.",
        ],
        "format_structural_guidance": list(_FORMAT_STRUCTURAL_GUIDANCE),
    }
    return stable_json(body)


def _resolve_repair_instruction(issue: IntentIssue) -> str:
    """Return a targeted fix instruction for a semantic issue."""
    return REPAIR_INSTRUCTIONS.get(issue.category.value, issue.message)


def _classify_schema_error(error_message: str) -> FailureCategory:
    """
    Classify a schema enforcement error into a specific category.

    Args:

        error_message: Error string produced by check_qualified_refs_exist.

    Returns:

        ``UNKNOWN_TABLE``, ``UNKNOWN_COLUMN``, or ``SCHEMA_VALIDATION`` as a fallback.
    """
    lower = error_message.lower()
    if "unknown table" in lower:
        return FailureCategory.UNKNOWN_TABLE
    if "unknown" in lower and "column" in lower:
        return FailureCategory.UNKNOWN_COLUMN
    return FailureCategory.SCHEMA_VALIDATION


def _build_intent_semantic_repair_prompt(
    question: str,
    current_intent_json: str,
    errors: list[IntentIssue],
    warnings: list[IntentIssue],
    schema_literal_json: str,
    *,
    prior_question_feedback: list[dict[str, str]] | None = None,
) -> str:
    """
    Build a user-prompt for the LLM to repair semantic issues in a parsed intent.

    Errors are presented as errors_to_fix with targeted fix instructions sourced from REPAIR_INSTRUCTIONS and warnings are presented as non- binding suggestions.

    Args:

        question: Original natural language question.

        current_intent_json: JSON string of the current flawed parsed intent.

        errors: IntentIssue objects with severity equal to "error".

    warnings: IntentIssue objects with severity equal to "warning".

        schema_literal_json: Compact JSON schema literal for the LLM context.

        prior_question_feedback: Optional rows scoped to this question and schema hash.

    Returns:

        JSON-formatted prompt string ready to send as the user message.
    """
    errors_to_fix: list[dict[str, str]] = []
    for err in errors:
        errors_to_fix.append(
            {
                "category": err.category,
                "issue": err.message,
                "fix": _resolve_repair_instruction(err),
            }
        )

    suggestions: list[str] = [w.message for w in warnings]

    body: dict[str, Any] = {
        "task": (
            "Fix every error listed in errors_to_fix. Follow each "
            "fix instruction exactly. Suggestions are optional "
            "improvements. Output corrected intent JSON only."
        ),
        "critical_rules": list(INTENT_CRITICAL_RULES),
        "errors_to_fix": errors_to_fix,
        "suggestions": suggestions,
        "field_specifications": dict(RuntimeIntent.PROMPT_FIELD_SPEC.items()),
        "structural_json_keys": intent_prompt_structural_index(),
        "current_intent": current_intent_json,
        "question": question,
        "schema_info": schema_literal_json,
        "output_format": RuntimeIntent.prompt_example_dict(),
    }
    if prior_question_feedback:
        body["prior_question_feedback"] = {
            "instruction": (
                "Each entry below summarizes a known-bad shape for this question; do not repeat the listed mistakes."
            ),
            "items": list(prior_question_feedback),
        }
    return stable_json(body)


def _build_intent_format_repair_prompt(question: str, raw_response: str, parse_error: str) -> str:
    """
    Build the JSON format-repair user message for a bad LLM response.

    Args:

        question: User question.

        raw_response: Malformed model output.

        parse_error: Parse failure reason.

    Returns:

        JSON user payload string for ``llm_chat``.
    """
    prompt = stable_json(
        {
            "task": "The previous response was not valid JSON. Fix the formatting errors and return ONLY valid JSON.",
            "question": question,
            "invalid_response": raw_response,
            "parse_error": parse_error,
            "field_specifications": dict(RuntimeIntent.PROMPT_FIELD_SPEC.items()),
            "structural_json_keys": intent_prompt_structural_index(),
            "instructions": list(INTENT_FORMAT_REPAIR_JSON_RULES) + list(INTENT_CRITICAL_RULES),
            "output_format": RuntimeIntent.prompt_example_dict(),
        }
    )
    return prompt


def _build_intent_parse_prompt(
    question: str,
    schema_literal_json: str,
    table_list: list[str],
    prior_question_feedback: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """
    Build ``(system, user_json)`` strings for the initial intent LLM call.

    Args:

        question: User question.

        schema_literal_json: Compact JSON schema literal for prompts.

        table_list: Allowed table names.

        prior_question_feedback: Optional summarized failures for this question from persisted memory and the current attempt.

    Returns:

        System prompt and stable-JSON user payload.
    """
    system = (
        "You are a deterministic intent parser for text-to-SQL. "
        "Output ONLY valid JSON that matches the required format. "
        "Identical inputs must produce identical outputs. The "
        "natural_language field is a single short sentence in plain English that describes the structured intent you just produced — what the query computes. Read your own SELECT expressions, FROM tables, filters, grouping, and ordering, then describe that result. "
        "Aggregation words like \"count\", \"sum\", \"average\", \"minimum\", \"maximum\", \"total\" are encouraged whenever the intent uses them. Use the same domain nouns as the schema (table and column names rendered in plain English). Do not mention SQL syntax tokens (JOIN, GROUP BY, WHERE, ORDER BY, LIMIT, etc.). Reuse the user's domain words when they correctly name the intent's tables/columns/aggregations; only avoid copying the question verbatim when the question contains filler words, polite phrasing, or wording the structured intent does not actually reflect. "
        + _INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES
        + " "
        + _INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE
    )

    parse_filter_ops = [
        "=",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "like",
        "not like",
    ]

    parse_filter_ops.extend(extra_filter_ops_for_engine())
    parse_filter_ops.extend(
        [
            "in",
            "not in",
            "is null",
            "is not null",
            "between",
            "contains",
        ]
    )

    user_payload: dict[str, object] = {
        "task": (
            "Parse the question into a schema-aware intent JSON. "
            "Do not write SQL. Extract all literals needed for "
            "filters, HAVING, limits, and parameters."
        ),
        "question": question,
        "schema_summary": schema_literal_json,
        "allowed_tables": table_list,
        "naming_conventions": {
            "shape_token_table_a": "tbl_a",
            "shape_token_table_b": "tbl_b",
            "shape_token_column_a": "col_a",
            "shape_token_date_a": "date_a",
            "shape_token_date_b": "date_b",
            "note": (
                "tbl_a, tbl_b, col_a, date_a, date_b appear only to illustrate JSON shape. "
                "Replace every expression with real identifiers from schema_summary or allowed_tables."
            ),
        },
        "field_specifications": dict(RuntimeIntent.PROMPT_FIELD_SPEC.items()),
        "structural_json_keys": intent_prompt_structural_index(),
        "expression_format": {
            "description": (
                "expr strings mirror SQL using qualified columns, arithmetic, aggregation, and scalar functions. "
                "The ``rules`` array begins with the same canonical strings as ``critical_rules`` in repair prompts, "
                "then appends parse-only lines (output_format keys and natural_language)."
            ),
        },
        "rules": list(INTENT_CRITICAL_RULES) + list(INTENT_PARSE_RULES_APPEND),
        "logical_decomposition_guidance": list(_LOGICAL_DECOMPOSITION_GUIDANCE),
        "format_structural_guidance": list(_FORMAT_STRUCTURAL_GUIDANCE),
        "output_format": RuntimeIntent.prompt_example_dict(),
        "operator_reference": {
            "filter_ops": parse_filter_ops,
            "having_ops": sorted(VALID_HAVING_OPS),
        },
        "value_type_reference": {
            "filter": [
                "string",
                "integer",
                "number",
                "date",
                "boolean",
                "null",
                "date_window",
                "date_diff",
            ],
            "having": ["integer", "number"],
        },
    }

    if prior_question_feedback:
        user_payload["prior_question_feedback"] = {
            "instruction": (
                "Each entry below summarizes a known-bad shape for this question; do not repeat the listed mistakes."
            ),
            "items": list(prior_question_feedback),
        }

    user = stable_json(user_payload)
    return system, user


def _format_repair_loop(
    system: str,
    raw: str,
    question: str,
    max_retries: int = PolicyConfig.MAX_STAGE_B_REPAIRS,
) -> tuple[RuntimeIntent | None, int]:
    """
    Parse *raw* to ``RuntimeIntent``, optionally calling format- repair LLM rounds.

    Args:

        system: System prompt.

        raw: Model output string.

        question: User question (for parse context).

        max_retries: Max format-repair attempts after parse failure or placeholder leakage in parsed expressions.

    Returns:

        ``(intent_or_none, extra_llm_calls)``.
    """

    def _acceptable(candidate: RuntimeIntent | None) -> bool:
        return candidate is not None and not runtime_intent_has_instructional_placeholders(candidate)

    llm_calls = 0
    parse_detail: list[str] = []
    intent = parse_intent_response(raw, question, parse_detail_out=parse_detail)
    if _acceptable(intent):
        if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled() and intent is not None:
            pipeline_trace_lazy(
                "intent_after_parse_intent_response.initial",
                lambda: stable_json(intent.to_dict()),
            )
        return intent, llm_calls
    for _ in range(max_retries):
        if intent is None:
            parse_error = parse_detail[-1] if parse_detail else "JSON parse failed"
        else:
            parse_error = INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR
        parse_detail.clear()
        repair_prompt = _build_intent_format_repair_prompt(question, raw, parse_error)
        raw = llm_chat(system, repair_prompt, task="intent")
        llm_calls += 1
        intent = parse_intent_response(raw, question, parse_detail_out=parse_detail)
        if _acceptable(intent):
            if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled() and intent is not None:
                pipeline_trace_lazy(
                    "intent_after_parse_intent_response.format_repair",
                    lambda it=intent: stable_json(it.to_dict()),
                )
            return intent, llm_calls
    return intent, llm_calls


def _compute_error_signature_issues(
    issues: list[IntentIssue],
) -> frozenset[tuple[FailureCategory, str]]:
    """
    Frozenset of ``(category, message)`` for *issues*.

    Args:

        issues: ``IntentIssue`` list (usually errors).

    Returns:

        Hashable signature for oscillation checks.
    """
    return frozenset((iss.category, iss.message) for iss in issues)


def _compute_error_signature_strings(errors: list[str]) -> frozenset[str]:
    """
    Frozenset copy of *errors* for history comparison.

    Args:

        errors: Raw error strings.

    Returns:

        Hashable signature.
    """
    return frozenset(errors)


def _detect_oscillation(history: list[frozenset[str]]) -> bool:
    """
    Return True on repeated identical signatures or A-B-A-B alternation.

    Args:

        history: Recent signature frozensets, oldest first.

    Returns:

        True when repair should stop for oscillation.
    """
    if len(history) >= 2 and history[-1] == history[-2]:
        return True
    if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
        return True
    return False


def _normalize_cte_output_aliases(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """
    Align CTE ``output_columns`` with ``derive_cte_output_columns`` and remap refs.

    Args:

        intent: Intent whose CTE outputs should be canonical.

        schema_graph: Metadata for ``build_cte_output_metadata``.

    Returns:

        Intent with remapped ``cte.*`` references across main and CTEs.
    """
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return intent

    alias_map: dict[str, str] = {}
    refreshed_cte_steps = []
    for idx, cte in enumerate(cte_steps):
        old_oc = list(cte.output_columns or [])
        new_oc = derive_cte_output_columns(cte.select_cols or [], cte_ordinal=idx + 1)
        ocm = build_cte_output_metadata(
            cte.select_cols or [],
            new_oc,
            schema_graph,
            window_registry=cte.window_registry,
        )
        for old_col, new_col in zip(old_oc, new_oc, strict=False):
            if old_col != new_col:
                alias_map[f"{cte.cte_name}.{old_col}"] = f"{cte.cte_name}.{new_col}"
        refreshed_cte_steps.append(
            replace(
                cte,
                output_columns=new_oc,
                output_column_metadata=ocm,
            )
        )
    intent = replace(intent, cte_steps=refreshed_cte_steps)

    if not alias_map:
        return intent

    debug(f"[_normalize_cte_output_aliases] remap: {alias_map}")

    def _remap_alias(s: str) -> str:
        return alias_map.get(s, s)

    def _remap_expr(expr: NormalizedExpr) -> NormalizedExpr:
        return replace_refs_in_expr(expr, _remap_alias)

    def _remap_cte_step(cte: RuntimeCteStep) -> RuntimeCteStep:
        return replace(
            cte,
            select_cols=[replace(sc, expr=_remap_expr(sc.expr)) for sc in (cte.select_cols or [])],
            order_by_cols=[replace(obc, expr=_remap_expr(obc.expr)) for obc in (cte.order_by_cols or [])],
            group_by_cols=[_remap_expr(g) for g in (cte.group_by_cols or [])],
            filters_param=[
                replace(
                    fp,
                    left_expr=_remap_expr(fp.left_expr),
                    right_expr=(_remap_expr(fp.right_expr) if fp.right_expr else None),
                )
                for fp in (cte.filters_param or [])
            ],
            having_param=[
                replace(
                    hp,
                    left_expr=_remap_expr(hp.left_expr),
                    right_expr=(_remap_expr(hp.right_expr) if hp.right_expr else None),
                )
                for hp in (cte.having_param or [])
            ],
        )

    refreshed_cte_steps = [_remap_cte_step(cte) for cte in intent.cte_steps]

    intent = replace(
        intent,
        cte_steps=refreshed_cte_steps,
        select_cols=[replace(sc, expr=_remap_expr(sc.expr)) for sc in (intent.select_cols or [])],
        order_by_cols=[replace(obc, expr=_remap_expr(obc.expr)) for obc in (intent.order_by_cols or [])],
        group_by_cols=[_remap_expr(g) for g in (intent.group_by_cols or [])],
        filters_param=[
            replace(
                fp,
                left_expr=_remap_expr(fp.left_expr),
                right_expr=(_remap_expr(fp.right_expr) if fp.right_expr else None),
            )
            for fp in (intent.filters_param or [])
        ],
        having_param=[
            replace(
                hp,
                left_expr=_remap_expr(hp.left_expr),
                right_expr=(_remap_expr(hp.right_expr) if hp.right_expr else None),
            )
            for hp in (intent.having_param or [])
        ],
    )

    return intent


def _trace_intent_after_deterministic_step(
    step_name: str,
    before: RuntimeIntent,
    after: RuntimeIntent,
    changed_fields: list[str],
) -> None:
    """Emit a per-step JSON diff trace, suppressed entirely when nothing changed."""
    if not changed_fields:
        return
    after_dict = after.to_dict()
    before_dict = before.to_dict()
    diff_payload = {
        field_name: {
            "before": before_dict.get(field_name),
            "after": after_dict.get(field_name),
        }
        for field_name in changed_fields
    }
    if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled():
        pipeline_trace_lazy(
            f"intent_after_deterministic_repair.{step_name}",
            lambda: stable_json(diff_payload),
        )


def _intent_changed_fields(before: RuntimeIntent, after: RuntimeIntent) -> list[str]:
    """Return sorted top-level field names that differ between two intents."""
    before_dict = before.to_dict()
    after_dict = after.to_dict()
    return [
        field_name for field_name in sorted(before_dict.keys()) if before_dict[field_name] != after_dict[field_name]
    ]


def _summarize_intent_changes(
    before: RuntimeIntent,
    after: RuntimeIntent,
) -> str:
    """Return a concise top-level field diff summary between two intents."""
    changed_fields = _intent_changed_fields(before, after)
    if not changed_fields:
        return "no_changes"
    return ", ".join(changed_fields)


def _deterministic_repair_step(
    step_name: str,
    intent: RuntimeIntent,
    transform: Callable[[RuntimeIntent], RuntimeIntent],
) -> RuntimeIntent:
    """Apply *transform* to *intent* and optionally record a trace snapshot."""
    out = transform(intent)
    changed_fields = _intent_changed_fields(intent, out)
    summary = ", ".join(changed_fields) if changed_fields else "no_changes"
    debug(f"[intent_process._deterministic_repair_step] {step_name}: {summary}")
    _trace_intent_after_deterministic_step(step_name, intent, out, changed_fields)
    return out


def apply_deterministic_repairs(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
    natural_language: str = "",
) -> RuntimeIntent:
    """
    Run the ordered deterministic repair chain (grain, filters, CTEs, policies, ...).

    Args:

        intent: Intent after LLM parse.

        schema_graph: Schema graph.

        natural_language: Question text (e.g. filter casing); may be empty.

    Returns:

        Fully normalized intent.
    """

    def process(
        fp: list[FilterParam],
        hp: list[HavingParam],
        gbc: list[Any],
    ) -> tuple[list[FilterParam], list[HavingParam]]:
        return auto_repair_filter_having(fp, hp, group_by_cols=gbc)

    def cte_grain_consistency_only(i: RuntimeIntent) -> RuntimeIntent:
        new_cte_steps = [enforce_cte_grain_consistency(cte) for cte in (i.cte_steps or [])]
        return replace(i, cte_steps=new_cte_steps)

    def _remap_distinct_select_index(
        old_cols: list[Any],
        new_cols: list[Any],
        old_index: int,
    ) -> int:
        """
        Translate ``distinct_select_index`` after select-col reordering.

        The index points to the column that originally carried ``DISTINCT``; after sorting we locate the same identity in the new list and return its new position. Returns ``-1`` when no DISTINCT was set or the original column cannot be located.
        """
        if old_index < 0 or old_index >= len(old_cols):
            return old_index if old_index < 0 else -1
        target = old_cols[old_index]
        for k, sc in enumerate(new_cols):
            if sc is target:
                return k
        return -1

    def auto_repair_filter_having_all_scopes(i: RuntimeIntent) -> RuntimeIntent:
        repaired_fp, repaired_hp = process(
            i.filters_param or [],
            i.having_param or [],
            i.group_by_cols or [],
        )
        out = replace(i, filters_param=repaired_fp, having_param=repaired_hp)
        if not out.cte_steps:
            return out
        new_cte_steps = []
        for cte in out.cte_steps:
            fp, hp = process(
                cte.filters_param or [],
                cte.having_param or [],
                cte.group_by_cols or [],
            )
            new_cte_steps.append(replace(cte, filters_param=fp, having_param=hp))
        return replace(out, cte_steps=new_cte_steps)

    def sort_select_and_order_by_all_scopes(i: RuntimeIntent) -> RuntimeIntent:
        sorted_main = sort_select_cols(i.select_cols or [])
        main_distinct = _remap_distinct_select_index(i.select_cols or [], sorted_main, i.distinct_select_index)
        out = replace(
            i,
            select_cols=sorted_main,
            order_by_cols=list(i.order_by_cols or []),
            distinct_select_index=main_distinct,
        )
        if not out.cte_steps:
            return out
        new_cte_steps = []
        for cte in out.cte_steps:
            sorted_cte = sort_select_cols(cte.select_cols or [])
            cte_distinct = _remap_distinct_select_index(cte.select_cols or [], sorted_cte, cte.distinct_select_index)
            new_cte_steps.append(
                replace(
                    cte,
                    select_cols=sorted_cte,
                    order_by_cols=list(cte.order_by_cols or []),
                    distinct_select_index=cte_distinct,
                )
            )
        return replace(out, cte_steps=new_cte_steps)

    intent = _deterministic_repair_step(
        "repair_intent_placeholder_tokens",
        intent,
        lambda x: repair_intent_placeholder_tokens(x, schema_graph),
    )
    intent = _deterministic_repair_step("normalize_count_star", intent, normalize_count_star)
    intent = _deterministic_repair_step("dedup_value_vs_right_expr", intent, dedup_value_vs_right_expr)
    intent = _deterministic_repair_step(
        "qualify_cte_count_star_mulgroups",
        intent,
        lambda x: qualify_cte_count_star_mulgroups(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "lift_distinct_select_from_raw_sql",
        intent,
        lambda x: lift_distinct_select_from_raw_sql(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "normalize_cte_output_aliases",
        intent,
        lambda x: _normalize_cte_output_aliases(x, schema_graph),
    )
    intent = _deterministic_repair_step("canonicalize_registry_ids", intent, canonicalize_registry_ids)
    intent = _deterministic_repair_step("reorder_cte_steps_by_dag", intent, reorder_cte_steps_by_dag)
    intent = _deterministic_repair_step("normalize_cte_names", intent, normalize_cte_names)
    intent = _deterministic_repair_step(
        "rewrite_main_query_refs_to_final_cte_columns",
        intent,
        rewrite_main_query_refs_to_final_cte_columns,
    )
    intent = _deterministic_repair_step("qualify_cte_output_columns", intent, qualify_cte_output_columns)
    intent = _deterministic_repair_step(
        "derive_tables_from_intent",
        intent,
        lambda x: reconcile_tables(x),
    )
    intent = _deterministic_repair_step("replace_unknown_scalar_funcs", intent, replace_unknown_scalar_funcs)
    intent = _deterministic_repair_step(
        "enforce_grain_consistency",
        intent,
        lambda x: enforce_grain_consistency(x, schema_graph),
    )
    intent = _deterministic_repair_step("strip_spurious_group_by", intent, strip_spurious_group_by)
    intent = _deterministic_repair_step("decompose_between_params", intent, decompose_between_params)
    intent = _deterministic_repair_step("auto_repair_filter_having", intent, auto_repair_filter_having_all_scopes)
    intent = _deterministic_repair_step("coerce_filter_group_mode", intent, coerce_filter_group_mode)
    intent = _deterministic_repair_step("normalize_filters_havings", intent, normalize_filters_havings)
    intent = _deterministic_repair_step("repair_null_equality_filters", intent, repair_null_equality_filters)
    intent = _deterministic_repair_step(
        "strip_join_conditions",
        intent,
        lambda x: strip_join_conditions(x, schema_graph),
    )
    intent = _deterministic_repair_step("cte_grain_consistency", intent, cte_grain_consistency_only)
    intent = _deterministic_repair_step("sort_select_and_order_by", intent, sort_select_and_order_by_all_scopes)
    intent = _deterministic_repair_step(
        "lift_distinct_modifier_in_multiply", intent, lift_distinct_modifier_in_multiply
    )
    intent = _deterministic_repair_step("simplify_exprs", intent, simplify_exprs)
    intent = _deterministic_repair_step("normalize_in_raw_values", intent, normalize_in_raw_values)
    intent = _deterministic_repair_step("repair_misclassified_date_diff", intent, repair_misclassified_date_diff)
    intent = _deterministic_repair_step("normalize_date_diff_raw_values", intent, normalize_date_diff_raw_values)
    intent = _deterministic_repair_step("canonicalize_temporal_unit_args", intent, canonicalize_temporal_unit_args)
    intent = _deterministic_repair_step("strip_impossible_having", intent, strip_impossible_having)
    intent = _deterministic_repair_step(
        "repair_fk_filter_type_mismatch",
        intent,
        lambda x: repair_fk_filter_type_mismatch(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "resolve_filter_value_case",
        intent,
        lambda x: resolve_filter_value_case(x, schema_graph, natural_language),
    )
    intent = _deterministic_repair_step(
        "normalize_in_filter_types",
        intent,
        lambda x: normalize_in_filter_types(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "normalize_boolean_filter_values",
        intent,
        lambda x: normalize_boolean_filter_values(x, schema_graph),
    )
    intent = _deterministic_repair_step("normalize_null_filter_values", intent, normalize_null_filter_values)
    intent = _deterministic_repair_step(
        "expand_fk_select_to_descriptive",
        intent,
        lambda x: expand_fk_select_to_descriptive(x, schema_graph),
    )
    intent = _deterministic_repair_step("dedup_contradictory_filters", intent, dedup_contradictory_filters)
    intent = _deterministic_repair_step(
        "repair_cumulative_phrasing_window_intent",
        intent,
        lambda x: repair_cumulative_phrasing_window_intent(x, natural_language),
    )
    intent = _deterministic_repair_step(
        "repair_case_when_intent",
        intent,
        lambda x: repair_case_when_intent(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "drop_invalid_case_registry_entries",
        intent,
        lambda x: drop_invalid_case_registry_entries(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "repair_array_filters_intent",
        intent,
        lambda x: repair_array_filters_intent(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "enforce_sensitivity_policy_intent",
        intent,
        lambda x: enforce_sensitivity_policy_intent(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "tail_enforce_grain_consistency",
        intent,
        lambda x: enforce_grain_consistency(x, schema_graph),
    )
    intent = _deterministic_repair_step("coerce_filter_group_mode", intent, coerce_filter_group_mode)
    intent = _deterministic_repair_step("tail_normalize_filters_havings", intent, normalize_filters_havings)
    return intent


def _apply_post_processing(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
    question: str,
) -> tuple[RuntimeIntent | None, list[IntentIssue]]:
    """
    Resolve columns, wire CTEs, assign params, prune tables; fail if params missing.

    Args:

        intent: Intent that passed semantic validation.

        schema_graph: Schema graph.

        question: User question.

    Returns:

        ``(ready_intent_or_none, resolution_issues)``; issues include ``column_ambiguous`` errors when
        bare names span multiple tables. ``None`` intent when required param values are missing.
    """
    all_cols = collect_column_refs_for_post_processing(intent)
    column_map, col_issues = resolve_column_map(all_cols, schema_graph, intent.tables or [])
    intent = replace(intent, column_map=column_map)

    if intent.cte_steps:
        intent = replace(intent, cte_steps=resolve_cte_column_maps(intent.cte_steps))

    if intent.cte_steps:
        debug(f"[_apply_post_processing] CTE chain: {[c.cte_name for c in intent.cte_steps]}")
        alias_map: dict[str, str] = {}
        refreshed_cte_steps = []
        for idx, cte in enumerate(intent.cte_steps):
            old_oc = list(cte.output_columns or [])
            new_oc = derive_cte_output_columns(cte.select_cols or [], cte_ordinal=idx + 1)
            ocm = build_cte_output_metadata(
                cte.select_cols or [],
                new_oc,
                schema_graph,
                window_registry=cte.window_registry,
            )
            debug(
                f"[_apply_post_processing] CTE '{cte.cte_name}' "
                f"tables={cte.tables} grain={cte.grain} "
                f"old_oc={old_oc} new_oc={new_oc}"
            )
            for old_col, new_col in zip(old_oc, new_oc, strict=False):
                if old_col != new_col:
                    alias_map[f"{cte.cte_name}.{old_col}"] = f"{cte.cte_name}.{new_col}"
            refreshed_cte_steps.append(replace(cte, output_columns=new_oc, output_column_metadata=ocm))
        intent = replace(intent, cte_steps=refreshed_cte_steps)
        if alias_map:
            debug(f"[_apply_post_processing] CTE alias remap: {alias_map}")

            def _remap_alias(s: str) -> str:
                return alias_map.get(s, s)

            def _remap_expr(expr: NormalizedExpr) -> NormalizedExpr:
                return replace_refs_in_expr(expr, _remap_alias)

            intent = replace(
                intent,
                select_cols=[replace(sc, expr=_remap_expr(sc.expr)) for sc in (intent.select_cols or [])],
                order_by_cols=[replace(obc, expr=_remap_expr(obc.expr)) for obc in (intent.order_by_cols or [])],
                group_by_cols=[_remap_expr(g) for g in (intent.group_by_cols or [])],
                filters_param=[
                    replace(
                        fp,
                        left_expr=_remap_expr(fp.left_expr),
                        right_expr=(_remap_expr(fp.right_expr) if fp.right_expr else None),
                    )
                    for fp in (intent.filters_param or [])
                ],
                having_param=[
                    replace(
                        hp,
                        left_expr=_remap_expr(hp.left_expr),
                        right_expr=(_remap_expr(hp.right_expr) if hp.right_expr else None),
                    )
                    for hp in (intent.having_param or [])
                ],
            )

    if intent.cte_steps:
        intent = qualify_cte_output_columns(intent)

    intent = rewrite_cte_output_refs_to_aliases(intent)

    if intent.cte_steps:
        intent = prune_unused_cte_steps(intent)
        intent = prune_unused_cte_output_columns(intent, schema_graph)
        intent = prune_unused_cte_steps(intent)

    if intent.cte_steps:
        emission_ctes: list[RuntimeCteStep] = []
        for cte in intent.cte_steps:
            em = classify_cte_emission(cte, intent, schema_graph)
            emission_ctes.append(replace(cte, emission=em))
        intent = replace(intent, cte_steps=emission_ctes)

    filters_param, having_param, cte_steps, case_registry, _ = assign_param_keys(
        intent.filters_param or [],
        intent.having_param or [],
        intent.cte_steps,
        intent.case_registry or [],
    )
    intent = replace(
        intent,
        filters_param=filters_param,
        having_param=having_param,
        cte_steps=cte_steps,
        case_registry=case_registry,
    )
    intent = enforce_case_branch_param_keys(intent)

    intent = reconcile_tables(intent)
    intent = normalize_pk_distinct(intent, schema_graph)

    intent = tag_expr_numeric(intent, schema_graph)
    intent = align_filter_value_type_to_exprs(intent, schema_graph)
    intent = tag_case_when_condition_scope(intent)

    all_pv = collect_raw_param_values(intent)
    if intent.param_values:
        all_pv = {**dict(intent.param_values), **all_pv}

    def _expected_keys_from_case_registry(regs: list[Any] | None) -> list[str]:
        out: list[str] = []
        for step in regs or []:
            cw = getattr(step, "case_when", None)
            if cw is None:
                continue
            for branch in cw.branches or []:
                cond = branch.condition
                if cond is None:
                    continue
                if cond.op == "between" and cond.param_key and cond.param_key_hi:
                    out.extend([cond.param_key, cond.param_key_hi])
                elif cond.param_key and cond.op not in ("is null", "is not null") and not cond.right_expr:
                    out.append(cond.param_key)
        return out

    expected_keys: list[str] = []
    for cte in intent.cte_steps or []:
        for fp in cte.filters_param or []:
            if fp.param_key and fp.op not in ("is null", "is not null") and not fp.right_expr:
                expected_keys.append(fp.param_key)
        for hp in cte.having_param or []:
            if hp.param_key and not hp.right_expr:
                expected_keys.append(hp.param_key)
        expected_keys.extend(_expected_keys_from_case_registry(cte.case_registry))
    for fp in intent.filters_param or []:
        if fp.param_key and fp.op not in ("is null", "is not null") and not fp.right_expr:
            expected_keys.append(fp.param_key)
    for hp in intent.having_param or []:
        if hp.param_key and not hp.right_expr:
            expected_keys.append(hp.param_key)
    expected_keys.extend(_expected_keys_from_case_registry(intent.case_registry))
    missing_keys = [k for k in expected_keys if k not in all_pv]
    if missing_keys:
        debug(f"[intent_process.apply_post_processing] missing_param_values — auto-terminating: {missing_keys}")
        return None, col_issues

    intent = replace(intent, param_values=all_pv)

    if intent.cte_steps:
        new_cte_steps = []
        for cte in intent.cte_steps:
            cte_pks: set[str] = set()
            for fp in cte.filters_param or []:
                if fp.param_key:
                    cte_pks.add(fp.param_key)
            for hp in cte.having_param or []:
                if hp.param_key:
                    cte_pks.add(hp.param_key)
            for step in cte.case_registry or []:
                cw = getattr(step, "case_when", None)
                if cw is None:
                    continue
                for branch in cw.branches or []:
                    cond = branch.condition
                    if cond is not None and cond.param_key:
                        cte_pks.add(cond.param_key)
                    if cond is not None and getattr(cond, "param_key_hi", ""):
                        cte_pks.add(cond.param_key_hi)
            cte_pv = {k: v for k, v in all_pv.items() if k in cte_pks}
            new_cte_steps.append(replace(cte, param_values=cte_pv))
        intent = replace(intent, cte_steps=new_cte_steps)

    intent = ensure_scalar_func_defaults(intent)
    intent = apply_aggregatability_gate(intent, schema_graph)
    intent = extract_structural_params(intent)
    intent = simplify_exprs(intent)

    if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled():
        pipeline_trace_lazy("intent_after_post_processing", lambda: stable_json(intent.to_dict()))

    return intent, col_issues


def _align_runtime_tables_to_planner(
    runtime: RuntimeIntent,
    logical: LogicalIntent,
) -> RuntimeIntent:
    """
    Overwrite runtime main and matching-CTE tables with the planner's authoritative lists.

    The planner is the source of truth for which tables the query needs, including join
    bridges. Encoder drift on the tables field is silently corrected here so the downstream
    deterministic pipeline (reconcile_tables, JOIN engine) sees consistent state. Encoder
    column hallucinations remain caught by check_qualified_refs_exist.

    Args:

        runtime: Encoder output runtime intent before deterministic repair.

        logical: Planner output logical intent.

    Returns:

        Runtime intent whose main tables list matches the planner's tables list. For every
        runtime cte_steps entry whose cte_name matches a planner CteIntent.name, that CTE's
        tables list is aligned with the planner's CTE tables list. Unmatched CTE steps are
        returned unchanged.
    """

    aligned_main = list(logical.tables)
    planner_cte_by_name = {s.name: s for s in (logical.cte_steps or ()) if s.name}
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in runtime.cte_steps or []:
        cname = (cte.cte_name or "").strip()
        if cname and cname in planner_cte_by_name:
            planned = planner_cte_by_name[cname]
            new_cte_steps.append(replace(cte, tables=list(planned.tables)))
        else:
            new_cte_steps.append(cte)
    return replace(runtime, tables=aligned_main, cte_steps=new_cte_steps)


def _structural_tables_for_logical(logical: LogicalIntent) -> tuple[str, ...]:
    """Return sorted union of planner base tables and every CTE step table list."""

    names: set[str] = set(logical.tables)
    for step in logical.cte_steps:
        names.update(step.tables)
    return tuple(sorted(names))


def _issue_to_planner_hint(issue: IntentIssue) -> str:
    """Return a compact NL hint appended to planner restarts."""

    cat = issue.category
    ctx = issue.context or {}
    if cat == FailureCategory.UNKNOWN_TABLE:
        name = str(ctx.get("table", ctx.get("name", ""))).strip()
        return f"Table {name or 'requested'} is not in the schema; pick only real schema table names."
    if cat == FailureCategory.WRONG_TABLES:
        return f"Your table set does not satisfy the question: {issue.message}"
    if cat == FailureCategory.WRONG_COLUMN_SELECTION:
        return f"Your select prose does not match the question: {issue.message}"
    if cat == FailureCategory.WRONG_FILTER_LOGIC:
        return "Your filter prose does not match the question predicate logic."
    if cat == FailureCategory.MISSING_NUMERIC_FILTER:
        val = str(ctx.get("value", "")).strip()
        return (
            f"Your prose did not mention value {val}; include literal values verbatim."
            if val
            else "Include every literal value from the question verbatim in the appropriate prose field."
        )
    if cat == FailureCategory.CTE_TABLE_REFERENCE:
        return f"CTE dependency issue: {issue.message}"
    if cat == FailureCategory.MISSING_TEMPORAL_COLUMN:
        return f"Temporal coverage issue: {issue.message}"
    return (issue.message or "").strip()


def _collect_post_stage_b_validation_issues(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
    post_resolution_issues: list[IntentIssue],
    logical: LogicalIntent | None = None,
) -> list[IntentIssue]:
    """
    Aggregate schema-ref strings, semantic validation, and post-resolution issues after Stage B.

    Used by :func:`full_intent_parse` to route logical versus format retries before the schema and semantic repair loop.

    Args:

        intent: Post-processed runtime intent candidate.

        schema_graph: Active schema graph.

        post_resolution_issues: Column resolution or binding issues from post-processing.

        logical: Planner intent when available for table fidelity, numeric coverage source, and literal attribution.

    Returns:

        Combined issues including inferred ``responsible_stage`` where applicable.
    """

    issues: list[IntentIssue] = list(post_resolution_issues)
    _, schema_errors = check_qualified_refs_exist(intent, schema_graph)
    for idx, err in enumerate(schema_errors):
        issues.append(
            make_intent_issue(
                issue_id=f"schema_ref_post_stage_b_{idx}",
                category=_classify_schema_error(err),
                severity="error",
                message=err,
                context={},
                responsible_stage="format",
            )
        )
    vr = validate_semantics(
        intent,
        schema_graph,
        post_binding=True,
        numeric_coverage_logical=logical,
    )
    if logical is not None:
        issues.extend(attribute_post_stage_b_issue(iss, logical) for iss in vr.issues)
    else:
        issues.extend(vr.issues)
    return issues


def apply_runtime_post_processing_lite(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
    *,
    question_fallback: str = "",
) -> tuple[RuntimeIntent | None, list[IntentIssue]]:
    """
    Apply deterministic column resolution and normalization from :func:`_apply_post_processing`.

    Args:

        intent: Runtime intent after deterministic structural repairs.

        schema_graph: Active schema graph.

        question_fallback: Text substituted when ``intent.natural_language`` is empty.

    Returns:

        Same contract as :func:`_apply_post_processing`: ``(intent_or_none, column_issues)``.
    """

    q = (intent.natural_language or "").strip() or question_fallback
    return _apply_post_processing(intent, schema_graph, q)


def _attempt_fresh_restart(
    question: str,
    schema_graph: SchemaGraph,
    max_retries: int,
    llm_calls: int,
    store: dict[str, Any] | None,
    in_turn_summaries: list[dict[str, str]],
    budget: RestartBudget,
    reason: str,
    last_intent: RuntimeIntent | None,
    *,
    prior_user_corrections: tuple[str, ...] = (),
    persist_template_learning: bool = True,
) -> tuple[RuntimeIntent | None, list[str], int]:
    """
    Re-run a full intent parse once after repair exhaustion, bounded by *budget*.

    When *reason* is in :data:`PolicyConfig.SEMANTIC_RESTART_REASONS` and in-memory semantic error rows exist, one ``VALIDATION_FAILURE`` summary is persisted before attempting the restart.

    Decrements :attr:`RestartBudget.fresh_restarts_left` and recurses into
    :func:`full_intent_parse` with the same shared budget so a nested restart cannot occur.

    Args:

        question: Normalised user question text.

        schema_graph: Schema graph for validation and table listing.

        max_retries: JSON format-repair attempts for the new parse.

        llm_calls: LLM calls already spent before this restart.

        store: Template store; optional validation summaries are persisted at the restart boundary.

        in_turn_summaries: Current-turn feedback rows not yet written to the store.

        budget: Shared :class:`RestartBudget` controlling how many restarts remain.

        reason: Short label describing which exit branch triggered the restart attempt.

        last_intent: Most recent intent from the repair loop, if any.

    Returns:

        ``(intent_or_none, warnings, new_llm_call_total)``.
    """
    tpl = _TEMPLATES_MODULE
    should_persist = (
        store is not None
        and last_intent is not None
        and reason in PolicyConfig.SEMANTIC_RESTART_REASONS
        and bool(in_turn_summaries)
    )
    if should_persist and persist_template_learning:
        flat_errs: list[str] = []
        for row in in_turn_summaries:
            if not isinstance(row, dict):
                continue
            for line in str(row.get("summary", "")).split("\n"):
                t = line.strip()
                if t:
                    flat_errs.append(t)
        is_post_restart = budget.fresh_restarts_left < PolicyConfig.MAX_FRESH_RESTARTS
        ent = tpl.summarize_failure_for_memory(
            question=question,
            intent=last_intent,
            kind=FeedbackKind.VALIDATION_FAILURE,
            schema_hash=schema_graph.effective_structural_hash,
            validator_errors=flat_errs or None,
            is_post_restart=is_post_restart,
        )
        tpl.record_question_feedback(store, question, ent)
        tpl.save_template_store(store)
    if budget.fresh_restarts_left <= 0:
        debug(
            f"[intent_process._attempt_fresh_restart] fresh restart denied (reason={reason}, "
            f"fresh_restarts_left=0); terminating."
        )
        return None, [], llm_calls
    budget.fresh_restarts_left -= 1
    debug(
        f"[intent_process._attempt_fresh_restart] fresh restart triggered (reason={reason}, "
        f"fresh_restarts_left={budget.fresh_restarts_left}, in_turn={len(in_turn_summaries)}, "
        f"store={'yes' if store is not None else 'no'})"
    )
    notify(
        f"Fresh intent parse restart (reason={reason}).",
        stage="intent",
        code=DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART,
        details=(("fresh_restarts_left", str(budget.fresh_restarts_left)),),
    )
    intent, warns, inner = full_intent_parse(
        question,
        schema_graph,
        max_retries=max_retries,
        store=store,
        in_turn_seed=[],
        budget=budget,
        prior_user_corrections=prior_user_corrections,
        persist_template_learning=persist_template_learning,
    )
    return intent, warns, llm_calls + inner


def _invoke_intent_parse_with_hints(
    question: str,
    schema_graph: SchemaGraph,
    *,
    max_retries: int = PolicyConfig.MAX_STAGE_B_REPAIRS,
    store: dict[str, Any] | None = None,
    in_turn_seed: list[dict[str, str]] | None = None,
    extra_user_feedback: list[str] | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    budget: RestartBudget | None = None,
    persist_template_learning: bool = True,
) -> tuple[RuntimeIntent | None, list[str], int]:
    """Run :func:`full_intent_parse` with persisted question feedback merged from *store*."""

    return full_intent_parse(
        question,
        schema_graph,
        max_retries=max_retries,
        store=store,
        in_turn_seed=in_turn_seed,
        extra_user_feedback=extra_user_feedback,
        prior_user_corrections=prior_user_corrections,
        budget=budget,
        persist_template_learning=persist_template_learning,
    )


def full_intent_parse(
    question: str,
    schema_graph: SchemaGraph,
    max_retries: int = PolicyConfig.MAX_STAGE_B_REPAIRS,
    *,
    store: dict[str, Any] | None = None,
    in_turn_seed: list[dict[str, str]] | None = None,
    extra_user_feedback: list[str] | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    budget: RestartBudget | None = None,
    persist_template_learning: bool = True,
) -> tuple[RuntimeIntent | None, list[str], int]:
    """
    End-to-end parse: Stage A logical JSON, Stage B IR JSON, post-bind validation loop, then schema and semantic repair.

    After Stage B, the engine applies deterministic repairs and post-processing, then aggregates schema-ref,
    semantic, and resolution issues. Any issue attributed to the logical stage restarts Stage A with accumulated
    grounding failures (bounded by :data:`MAX_STAGE_A_RETRIES`). Format-only issues trigger the same repair prompt
    shape as semantic repair (bounded by *max_retries*). On success the pipeline continues into
    :func:`_run_schema_semantic_repair_loop` for schema oscillation handling, semantic repair, and phase-G checks.

    Args:

        question: Natural language question.

        schema_graph: Schema and literal text provider.

        max_retries: JSON format-repair attempts per parse.

        store: Optional template store for ``question_feedback`` reads.

        in_turn_seed: Current-turn prompt rows merged ahead of persisted feedback.

        extra_user_feedback: Optional user-supplied rejection reasons merged into prior feedback rows.

        prior_user_corrections: Short operator or conversation hints appended to Stage A as ``prior_user_corrections``.

        budget: Shared :class:`RestartBudget` controlling fresh-restart count; constructed from
        :data:`PolicyConfig.MAX_FRESH_RESTARTS` when ``None``.

    Returns:

        ``(intent, warning_messages, llm_call_count)``; intent is ``None`` on failure.
    """
    tpl = _TEMPLATES_MODULE

    if budget is None:
        budget = RestartBudget.default()
    table_list = sorted(schema_graph.tables.keys())
    schema_literal_json = schema_graph.schema_literal_json
    llm_calls = 0
    seed_rows = [dict(r) for r in (in_turn_seed or []) if isinstance(r, dict)]
    if extra_user_feedback:
        for line in extra_user_feedback:
            t = (line or "").strip()
            if t:
                seed_rows.append({"summary": t, "source": "user_refinement"})
    persisted: list[dict[str, str]] = []
    if store is not None:
        persisted = tpl.collect_question_feedback_for_prompt(store, question, schema_graph.effective_structural_hash)
    merged_feedback = _dedupe_prior_question_feedback_rows(seed_rows + persisted)
    prior_fb_text = _serialized_prior_feedback_rows(merged_feedback)
    answer_style_text = stable_json(list(_LOGICAL_DECOMPOSITION_GUIDANCE))
    system_b = _INTENT_STAGE_B_SYSTEM
    prior_grounding_failures: tuple[str, ...] = ()
    max_a_attempts = MAX_STAGE_A_RETRIES + 1
    attempt_a = 0
    while attempt_a < max_a_attempts:
        user_a = _build_intent_logical_prompt(
            question,
            schema_literal_json,
            prior_fb_text,
            answer_style_text,
            prior_user_corrections,
            prior_grounding_failures,
        )
        raw_a = llm_chat(_INTENT_STAGE_A_SYSTEM, user_a, task="intent")
        llm_calls += 1
        debug(f"[intent_process.full_intent_parse] Stage A raw_llm_response (attempt {attempt_a + 1}): {raw_a}")
        logical_candidate, logical_issues = _parse_logical_intent_response(raw_a, schema_graph)
        if logical_candidate is None:
            if attempt_a >= max_a_attempts - 1:
                debug("[intent_process.full_intent_parse] Stage A exhausted after logical validation failures")
                return _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "stage_a_exhausted",
                    None,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                )
            notify(
                "Stage A logical intent retry after schema or core-field validation failures.",
                stage="intent",
                code=DIAGNOSTIC_CODE_STAGE_A_RETRY,
                details=(("attempt", str(attempt_a + 1)),),
            )
            prior_grounding_failures = prior_grounding_failures + tuple(_issue_to_planner_hint(iss) for iss in logical_issues)
            attempt_a += 1
            continue

        logical = logical_candidate
        structural_json = schema_graph.structural_schema_literal_json(_structural_tables_for_logical(logical))
        user_b = _build_intent_format_prompt(logical, structural_json)
        raw_b = llm_chat(system_b, user_b, task="intent")
        llm_calls += 1
        debug(f"[intent_process.full_intent_parse] Stage B raw_llm_response: {raw_b}")
        intent, fmt_calls = _format_repair_loop(system_b, raw_b, question, max_retries)
        llm_calls += fmt_calls

        if not intent:
            debug("[intent_process.full_intent_parse] Stage B format repair exhausted")
            return _attempt_fresh_restart(
                question,
                schema_graph,
                max_retries,
                llm_calls,
                store,
                seed_rows,
                budget,
                "stage_b_format_exhausted",
                None,
                prior_user_corrections=prior_user_corrections,
                persist_template_learning=persist_template_learning,
            )

        debug(f"[intent_process.full_intent_parse] normalized intent after Stage B:\n{stable_json(intent.to_dict())}")

        intent = _align_runtime_tables_to_planner(intent, logical)
        intent = _propagate_planner_schema_invalid_flag(intent, logical)

        avoid_rows = _dedupe_prior_question_feedback_rows(list(seed_rows) + persisted)
        logical_restart = False
        b_repairs_used = 0
        while True:
            intent = apply_deterministic_repairs(intent, schema_graph, question)
            result, post_issues = _apply_post_processing(intent, schema_graph, question)
            if result is None:
                debug("[intent_process.full_intent_parse] post-processing missing params — terminating")
                return None, [], llm_calls
            merged_issues = _collect_post_stage_b_validation_issues(result, schema_graph, post_issues, logical)
            errors = [iss for iss in merged_issues if iss.severity == "error"]
            if not errors:
                intent = result
                break
            if any(iss.responsible_stage == "logical" for iss in errors):
                if attempt_a >= max_a_attempts - 1:
                    return _attempt_fresh_restart(
                        question,
                        schema_graph,
                        max_retries,
                        llm_calls,
                        store,
                        seed_rows,
                        budget,
                        "stage_a_orchestrator_exhausted",
                        None,
                        prior_user_corrections=prior_user_corrections,
                        persist_template_learning=persist_template_learning,
                    )
                notify(
                    "Stage A retry after post-bind validation flagged logical grounding issues.",
                    stage="intent",
                    code=DIAGNOSTIC_CODE_STAGE_A_RETRY,
                    details=(("attempt", str(attempt_a + 1)), ("phase", "post_stage_b")),
                )
                prior_grounding_failures = prior_grounding_failures + tuple(
                    _issue_to_planner_hint(iss) for iss in errors if iss.responsible_stage == "logical"
                )
                logical_restart = True
                break
            if b_repairs_used >= max_retries:
                debug("[intent_process.full_intent_parse] post-Stage B format repair budget exhausted")
                return _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "post_stage_b_format_exhausted",
                    intent,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                )
            warnings = [iss for iss in merged_issues if iss.severity == "warning"]
            intent_json = stable_json(result.to_prompt_dict())
            debug(
                f"[intent_process.full_intent_parse] post-Stage B repair errors: "
                f"{[(e.category, e.message) for e in errors]}"
            )
            repair_prompt = _build_intent_semantic_repair_prompt(
                question,
                intent_json,
                errors,
                warnings,
                schema_literal_json,
                prior_question_feedback=avoid_rows or None,
            )
            notify(
                "Stage B post-bind format repair LLM invocation.",
                stage="intent",
                code=DIAGNOSTIC_CODE_STAGE_B_REPAIR,
                details=(("phase", "post_stage_b"), ("repair_round", str(b_repairs_used + 1))),
            )
            rollback_intent = result
            repaired_raw = llm_chat(system_b, repair_prompt, task="intent")
            llm_calls += 1
            repaired, fmt_rep_calls = _format_repair_loop(system_b, repaired_raw, question, max_retries)
            llm_calls += fmt_rep_calls
            if not repaired:
                debug("[intent_process.full_intent_parse] format repair exhausted after post-Stage B repair")
                return _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "post_stage_b_format_repair_exhausted",
                    intent,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                )
            intent = repaired
            intent = _align_runtime_tables_to_planner(intent, logical)
            intent = _propagate_planner_schema_invalid_flag(intent, logical)
            if not _runtime_intent_select_cols_have_substance(intent) or _runtime_intent_case_registry_has_empty_branches(
                intent
            ):
                debug("[intent_process.full_intent_parse] post-Stage B repair_reverted_empty_select")
                intent = rollback_intent
            b_repairs_used += 1

        if logical_restart:
            attempt_a += 1
            continue

        in_turn_live: list[dict[str, str]] = list(seed_rows)

        repaired_intent, sem_warns, llm_calls, planner_hints = _run_schema_semantic_repair_loop(
            intent=intent,
            question=question,
            system=system_b,
            schema_graph=schema_graph,
            schema_literal_json=schema_literal_json,
            table_list=table_list,
            max_retries=max_retries,
            llm_calls=llm_calls,
            store=store,
            in_turn_summaries=in_turn_live,
            budget=budget,
            logical=logical,
            prior_user_corrections=prior_user_corrections,
            persist_template_learning=persist_template_learning,
        )
        if planner_hints is not None:
            if attempt_a >= max_a_attempts - 1:
                return _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "stage_a_after_schema_semantic_exhausted",
                    None,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                )
            notify(
                "Stage A retry after schema/semantic validation surfaced logical-stage issues.",
                stage="intent",
                code=DIAGNOSTIC_CODE_STAGE_A_RETRY,
                details=(("attempt", str(attempt_a + 1)), ("phase", "schema_semantic_loop")),
            )
            prior_grounding_failures = prior_grounding_failures + planner_hints
            attempt_a += 1
            continue

        return repaired_intent, sem_warns, llm_calls

    debug("[intent_process.full_intent_parse] Stage A outer budget exhausted (unexpected)")
    return _attempt_fresh_restart(
        question,
        schema_graph,
        max_retries,
        llm_calls,
        store,
        seed_rows,
        budget,
        "stage_a_outer_exhausted",
        None,
        prior_user_corrections=prior_user_corrections,
        persist_template_learning=persist_template_learning,
    )


def _runtime_intent_select_cols_have_substance(intent: RuntimeIntent) -> bool:
    """
    Return False when any select column expression is structurally empty in the main query or a CTE.
    """

    def scope_ok(cols: list[SelectCol]) -> bool:
        return all(not _normalized_expr_is_absent(sc.expr) for sc in cols or [])

    if not scope_ok(intent.select_cols):
        return False
    return all(scope_ok(step.select_cols) for step in intent.cte_steps or [])


def _runtime_intent_case_registry_has_empty_branches(intent: RuntimeIntent) -> bool:
    """
    Return True when any case-registry step has no branches (main query or a CTE).
    """

    def scope_bad(steps: list[CaseRegistryStep]) -> bool:
        return any(not step.case_when.branches for step in steps or [])

    if scope_bad(intent.case_registry):
        return True
    return any(scope_bad(step.case_registry) for step in intent.cte_steps or [])


def _run_schema_semantic_repair_loop(
    intent: RuntimeIntent,
    question: str,
    system: str,
    schema_graph: SchemaGraph,
    schema_literal_json: str,
    table_list: list[str],
    max_retries: int,
    llm_calls: int,
    store: dict[str, Any] | None = None,
    in_turn_summaries: list[dict[str, str]] | None = None,
    budget: RestartBudget | None = None,
    *,
    logical: LogicalIntent | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    persist_template_learning: bool = True,
) -> tuple[RuntimeIntent | None, list[str], int, tuple[str, ...] | None]:
    """
    Schema + semantic repair loops, post-processing, and phase-G revalidation.

    Each level performs one initial validation followed by up to :data:`PolicyConfig.MAX_STAGE_B_REPAIRS` repair attempts (so total = original + ``MAX_STAGE_B_REPAIRS`` validations). Exhaustion or oscillation funnels into :func:`_attempt_fresh_restart`, which is bounded by the shared *budget*.

    Returns:

        ``(intent, warnings, llm_calls, planner_restart_hints)``. When *planner_restart_hints* is not ``None``, the caller should retry the planner (Stage A) with those hints instead of treating the tuple as success.
    """
    semantic_warnings: list[str] = []
    seen_warning_ids: set[str] = set()
    semantic_error_history: list[frozenset[str]] = []
    accumulated_failure_hints: list[str] = []
    in_turn: list[dict[str, str]] = list(in_turn_summaries) if in_turn_summaries is not None else []
    if budget is None:
        budget = RestartBudget.default()
    max_repair = PolicyConfig.MAX_STAGE_B_REPAIRS
    sem_iterations = max_repair + 1
    schema_iterations = max_repair + 1

    for sem_round in range(sem_iterations):
        debug(f"[intent_process._run_schema_semantic_repair_loop] semantic round {sem_round + 1}/{sem_iterations}")
        tpl = _TEMPLATES_MODULE

        persisted_rows: list[dict[str, str]] = []
        if store is not None:
            persisted_rows = tpl.collect_question_feedback_for_prompt(
                store, question, schema_graph.effective_structural_hash
            )
        avoid_rows = _dedupe_prior_question_feedback_rows(list(in_turn) + persisted_rows)
        intent = apply_deterministic_repairs(intent, schema_graph, question)
        debug(
            f"[intent_process._run_schema_semantic_repair_loop] full intent after deterministic repairs:\n"
            f"{stable_json(intent.to_dict())}"
        )

        schema_error_history: list[frozenset[str]] = []
        schema_resolved = False
        for schema_sub in range(schema_iterations):
            intent, schema_errors = check_qualified_refs_exist(intent, schema_graph)
            if not schema_errors:
                debug(
                    f"[intent_process._run_schema_semantic_repair_loop] schema validation passed on sub-round {schema_sub + 1}/{schema_iterations}"
                )
                schema_resolved = True
                break
            debug(
                f"[intent_process._run_schema_semantic_repair_loop] schema sub-round {schema_sub + 1}/{schema_iterations}: "
                f"{len(schema_errors)} errors"
            )
            sig = _compute_error_signature_strings(schema_errors)
            schema_error_history.append(sig)
            if _detect_oscillation(schema_error_history):
                debug(
                    "[intent_process._run_schema_semantic_repair_loop] schema oscillation detected — breaking sub-loop"
                )
                accumulated_failure_hints.extend(schema_errors)
                break
            if schema_sub >= schema_iterations - 1:
                accumulated_failure_hints.extend(schema_errors)
                break
            schema_issues = [
                make_intent_issue(
                    issue_id=f"schema_error_{idx}",
                    category=_classify_schema_error(err),
                    severity="error",
                    message=err,
                    responsible_stage="format",
                )
                for idx, err in enumerate(schema_errors)
            ]
            for iss in schema_issues:
                debug(
                    f"[intent_process._run_schema_semantic_repair_loop]   issue_id={iss.issue_id} message={iss.message}"
                )
            intent_before_schema_llm = intent
            intent_json = stable_json(intent.to_prompt_dict())
            debug(
                f"[intent_process._run_schema_semantic_repair_loop] intent being sent to schema repair LLM:\n{intent_json}"
            )
            debug(
                f"[intent_process._run_schema_semantic_repair_loop] schema errors_to_fix: "
                f"{[(iss.category, iss.message) for iss in schema_issues]}"
            )
            repair_prompt = _build_intent_semantic_repair_prompt(
                question,
                intent_json,
                schema_issues,
                [],
                schema_literal_json,
                prior_question_feedback=avoid_rows or None,
            )
            notify(
                "Stage B schema repair LLM invocation.",
                stage="intent",
                code=DIAGNOSTIC_CODE_STAGE_B_REPAIR,
                details=(
                    ("phase", "schema"),
                    ("semantic_round", str(sem_round + 1)),
                    ("schema_sub_round", str(schema_sub + 1)),
                ),
            )
            repaired_raw = llm_chat(system, repair_prompt, task="intent")
            llm_calls += 1
            repaired, fmt_calls = _format_repair_loop(system, repaired_raw, question, max_retries)
            llm_calls += fmt_calls
            if not repaired:
                debug(
                    "[intent_process._run_schema_semantic_repair_loop] format repair exhausted after schema repair — terminating"
                )
                fi, fw, fc = _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    in_turn,
                    budget,
                    "schema_format_repair_exhausted",
                    intent,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                )
                return fi, fw, fc, None
            intent = repaired
            if logical is not None:
                intent = _align_runtime_tables_to_planner(intent, logical)
                intent = _propagate_planner_schema_invalid_flag(intent, logical)
            if not _runtime_intent_select_cols_have_substance(intent) or _runtime_intent_case_registry_has_empty_branches(
                intent
            ):
                debug("[intent_process._run_schema_semantic_repair_loop] repair_reverted_empty_select")
                intent = intent_before_schema_llm
            debug(
                f"[intent_process._run_schema_semantic_repair_loop] normalized intent after schema repair:\n"
                f"{stable_json(intent.to_dict())}"
            )
            intent = apply_deterministic_repairs(intent, schema_graph, question)

        if not schema_resolved:
            debug(
                "[intent_process._run_schema_semantic_repair_loop] schema errors persist after sub-loop — terminating"
            )
            fi, fw, fc = _attempt_fresh_restart(
                question,
                schema_graph,
                max_retries,
                llm_calls,
                store,
                in_turn,
                budget,
                "schema_unresolved",
                intent,
                prior_user_corrections=prior_user_corrections,
                persist_template_learning=persist_template_learning,
            )
            return fi, fw, fc, None

        validation_result = validate_semantics(
            intent,
            schema_graph,
            numeric_coverage_logical=logical,
        )
        if logical is not None:
            validation_result.issues = [attribute_post_stage_b_issue(i, logical) for i in validation_result.issues]
        debug(
            f"[intent_process._run_schema_semantic_repair_loop] semantic validation completed: issues={len(validation_result.issues)}"
        )

        errors = [iss for iss in validation_result.issues if iss.severity == "error"]
        warnings = [iss for iss in validation_result.issues if iss.severity == "warning"]

        if logical is not None:
            logical_issues = [iss for iss in validation_result.issues if iss.responsible_stage == "logical"]
            if logical_issues:
                hints = tuple(dict.fromkeys(_issue_to_planner_hint(iss) for iss in logical_issues))
                return None, [], llm_calls, hints

        for iss in validation_result.issues:
            debug(f"[intent_process._run_schema_semantic_repair_loop]   issue_id={iss.issue_id} message={iss.message}")
        for w in warnings:
            if w.issue_id not in seen_warning_ids:
                seen_warning_ids.add(w.issue_id)
                semantic_warnings.append(w.message)

        if not errors:
            debug(f"[intent_process._run_schema_semantic_repair_loop] no semantic errors in round {sem_round + 1}")
            break

        debug(
            f"[intent_process._run_schema_semantic_repair_loop] {len(errors)} errors, {len(warnings)} warnings in round {sem_round + 1}"
        )
        in_turn.append(_in_turn_row_from_semantic_errors(errors, schema_graph.effective_structural_hash, intent))
        accumulated_failure_hints.extend(iss.message for iss in errors)

        sig = _compute_error_signature_issues(errors)
        semantic_error_history.append(sig)
        if _detect_oscillation(semantic_error_history):
            debug(
                "[intent_process._run_schema_semantic_repair_loop] semantic oscillation detected — trying fresh restart"
            )
            fi, fw, fc = _attempt_fresh_restart(
                question,
                schema_graph,
                max_retries,
                llm_calls,
                store,
                in_turn,
                budget,
                "semantic_oscillation",
                intent,
                prior_user_corrections=prior_user_corrections,
                persist_template_learning=persist_template_learning,
            )
            return fi, fw, fc, None

        if sem_round >= sem_iterations - 1:
            debug(
                "[intent_process._run_schema_semantic_repair_loop] semantic errors persist after max rounds — trying fresh restart"
            )
            fi, fw, fc = _attempt_fresh_restart(
                question,
                schema_graph,
                max_retries,
                llm_calls,
                store,
                in_turn,
                budget,
                "semantic_max_rounds",
                intent,
                prior_user_corrections=prior_user_corrections,
                persist_template_learning=persist_template_learning,
            )
            return fi, fw, fc, None

        intent_before_semantic_llm = intent
        intent_json = stable_json(intent.to_prompt_dict())
        debug(
            f"[intent_process._run_schema_semantic_repair_loop] intent being sent to semantic repair LLM:\n{intent_json}"
        )
        debug(
            f"[intent_process._run_schema_semantic_repair_loop] errors_to_fix: {[(e.category, e.message) for e in errors]}"
        )
        repair_prompt = _build_intent_semantic_repair_prompt(
            question,
            intent_json,
            errors,
            warnings,
            schema_literal_json,
            prior_question_feedback=avoid_rows or None,
        )
        notify(
            "Stage B semantic repair LLM invocation.",
            stage="intent",
            code=DIAGNOSTIC_CODE_STAGE_B_REPAIR,
            details=(
                ("phase", "semantic"),
                ("semantic_round", str(sem_round + 1)),
            ),
        )
        repaired_raw = llm_chat(system, repair_prompt, task="intent")
        llm_calls += 1
        repaired, fmt_calls = _format_repair_loop(system, repaired_raw, question, max_retries)
        llm_calls += fmt_calls

        if not repaired:
            debug(
                "[intent_process._run_schema_semantic_repair_loop] format repair exhausted after semantic repair — trying fresh restart"
            )
            fi, fw, fc = _attempt_fresh_restart(
                question,
                schema_graph,
                max_retries,
                llm_calls,
                store,
                in_turn,
                budget,
                "semantic_format_repair_exhausted",
                intent,
                prior_user_corrections=prior_user_corrections,
                persist_template_learning=persist_template_learning,
            )
            return fi, fw, fc, None

        intent = repaired
        if logical is not None:
            intent = _align_runtime_tables_to_planner(intent, logical)
            intent = _propagate_planner_schema_invalid_flag(intent, logical)
        if not _runtime_intent_select_cols_have_substance(intent) or _runtime_intent_case_registry_has_empty_branches(
            intent
        ):
            debug("[intent_process._run_schema_semantic_repair_loop] repair_reverted_empty_select")
            intent = intent_before_semantic_llm
        debug(
            f"[intent_process._run_schema_semantic_repair_loop] normalized intent after semantic repair:\n"
            f"{stable_json(intent.to_dict())}"
        )
        pipeline_trace_lazy(
            "intent_process._run_schema_semantic_repair_loop.after_llm_semantic_repair",
            lambda it=intent: stable_json(it.to_dict()),
        )

    result, post_issues = _apply_post_processing(intent, schema_graph, question)
    if result is None:
        return None, [], llm_calls, None
    if any(i.severity == "error" for i in post_issues):
        debug(
            "[intent_process._run_schema_semantic_repair_loop] post-processing column resolution errors — terminating"
        )
        return None, [], llm_calls, None

    if not _phase_g_post_validation_passes(result, schema_graph):
        debug("[intent_process._run_schema_semantic_repair_loop] phase-G soft recovery: attempted")
        recovered = apply_deterministic_repairs(result, schema_graph, question)
        if _phase_g_post_validation_passes(recovered, schema_graph):
            debug("[intent_process._run_schema_semantic_repair_loop] phase-G soft recovery: succeeded")
            result = recovered
        else:
            debug("[intent_process._run_schema_semantic_repair_loop] phase-G soft recovery: failed")
            debug(
                "[intent_process._run_schema_semantic_repair_loop] phase-G post-processing revalidation failed — terminating"
            )
            return None, [], llm_calls, None

    debug(
        f"[intent_process._run_schema_semantic_repair_loop] parsed intent with {len(result.tables or [])} tables, "
        f"{len(result.filters_param or [])} filters, {llm_calls} LLM calls"
    )
    pipeline_trace_lazy(
        "intent_process._run_schema_semantic_repair_loop.final_intent",
        lambda: stable_json(result.to_dict()),
    )

    if logical is not None:
        result = _propagate_planner_schema_invalid_flag(result, logical)

    return result, semantic_warnings, llm_calls, None


def _phase_g_post_validation_passes(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> bool:
    """
    Lightweight revalidation after post-processing.

    Runs ``check_qualified_refs_exist`` and ``validate_semantics`` once with no repair loop or LLM call. A post-processing step (param resolution, CTE alias rewrite, table pruning) should never invalidate what earlier phases proved, but any surfaced error is treated as terminal because there is no safe way to recover at this point without re-entering the full repair pipeline.
    """
    _, schema_errors = check_qualified_refs_exist(intent, schema_graph)
    if schema_errors:
        debug(f"[intent_process._phase_g_post_validation_passes] schema errors: {len(schema_errors)}")
        for err in schema_errors:
            debug(f"[intent_process._phase_g_post_validation_passes]   {err}")
        return False
    validation_result = validate_semantics(intent, schema_graph, post_binding=True)
    final_errors = [iss for iss in validation_result.issues if iss.severity == "error"]
    if final_errors:
        debug(f"[intent_process._phase_g_post_validation_passes] semantic errors: {len(final_errors)}")
        for iss in final_errors:
            debug(f"[intent_process._phase_g_post_validation_passes]   {iss.category}: {iss.message}")
        return False
    return True


def _compute_filters_similarity(filters1: list[FilterParam], filters2: list[FilterParam]) -> float:
    """
    Jaccard similarity on ``signature_key``, with a small penalty when coerced
    ``(signature_key, filter_group)`` wiring differs.

    Coercion matches the main pipeline so flat ``bool_op=OR`` rows align with
    ``filter_group`` disjuncts for template reuse.

    Args:

        filters1: First filter list.

        filters2: Second filter list.

    Returns:

        Score in ``[0, 1]``; ``1.0`` when both empty.
    """
    if not filters1 and not filters2:
        return 1.0
    if not filters1 or not filters2:
        return 0.0
    c1 = _coerce_filter_group_list(list(filters1))
    c2 = _coerce_filter_group_list(list(filters2))
    keys1 = {fp.signature_key for fp in c1}
    keys2 = {fp.signature_key for fp in c2}
    score = _jaccard(keys1, keys2)
    if score > 0:
        g1 = any(fp.filter_group is not None for fp in c1)
        g2 = any(fp.filter_group is not None for fp in c2)
        if len(c1) >= 2 and len(c1) == len(c2) and sorted(fp.signature_key for fp in c1) == sorted(
            fp.signature_key for fp in c2
        ):
            if g1 != g2:
                return score
        sig_fg1 = sorted((fp.signature_key, fp.filter_group) for fp in c1)
        sig_fg2 = sorted((fp.signature_key, fp.filter_group) for fp in c2)
        if sig_fg1 != sig_fg2:
            score *= 0.9
    return score


def _compute_having_similarity(having1: list[HavingParam], having2: list[HavingParam]) -> float:
    """
    Same as ``_compute_filters_similarity`` for having clauses.

    Args:

        having1: First having list.

        having2: Second having list.

    Returns:

        Score in ``[0, 1]``.
    """
    if not having1 and not having2:
        return 1.0
    if not having1 or not having2:
        return 0.0
    c1 = _coerce_having_group_list(list(having1))
    c2 = _coerce_having_group_list(list(having2))
    keys1 = {hp.signature_key for hp in c1}
    keys2 = {hp.signature_key for hp in c2}
    score = _jaccard(keys1, keys2)
    if score > 0:
        g1 = any(hp.filter_group is not None for hp in c1)
        g2 = any(hp.filter_group is not None for hp in c2)
        if len(c1) >= 2 and len(c1) == len(c2) and sorted(hp.signature_key for hp in c1) == sorted(
            hp.signature_key for hp in c2
        ):
            if g1 != g2:
                return score
        sig_fg1 = sorted((hp.signature_key, hp.filter_group) for hp in c1)
        sig_fg2 = sorted((hp.signature_key, hp.filter_group) for hp in c2)
        if sig_fg1 != sig_fg2:
            score *= 0.9
    return score


def _compute_select_cols_similarity(cols1: list[SelectCol], cols2: list[SelectCol]) -> float:
    """
    Jaccard similarity on select ``signature_key``.

    Args:

        cols1: First select list.

        cols2: Second select list.

    Returns:

        Score in ``[0, 1]``.
    """
    if not cols1 and not cols2:
        return 1.0
    if not cols1 or not cols2:
        return 0.0
    keys1 = {sc.signature_key for sc in cols1}
    keys2 = {sc.signature_key for sc in cols2}
    return _jaccard(keys1, keys2)


def _compute_order_by_cols_similarity(cols1: list[OrderByCol], cols2: list[OrderByCol]) -> float:
    """
    Jaccard similarity on order-by ``signature_key``.

    Args:

        cols1: First order-by list.

        cols2: Second order-by list.

    Returns:

        Score in ``[0, 1]``.
    """
    if not cols1 and not cols2:
        return 1.0
    if not cols1 or not cols2:
        return 0.0
    keys1 = {obc.signature_key for obc in cols1}
    keys2 = {obc.signature_key for obc in cols2}
    return _jaccard(keys1, keys2)


def _base_similarity(
    tables1: list[str],
    tables2: list[str],
    select1: list[SelectCol],
    select2: list[SelectCol],
    group1: list[NormalizedExpr],
    group2: list[NormalizedExpr],
    order1: list[OrderByCol],
    order2: list[OrderByCol],
    filters1: list[FilterParam],
    filters2: list[FilterParam],
    having1: list[HavingParam],
    having2: list[HavingParam],
) -> float:
    """
    Weighted blend of Jaccard-like scores (tables, filters, select, group, order, having).

    Args:

        filters1, filters2, having1, having2: Parallel clause lists from two intents.

    Returns:

        Score in ``[0, 1]``.
    """
    tables_sim = _jaccard(set(tables1), set(tables2))
    select_sim = _compute_select_cols_similarity(select1, select2)
    group_sim = _jaccard({g.signature_key for g in group1}, {g.signature_key for g in group2})
    order_sim = _compute_order_by_cols_similarity(order1, order2)
    filter_sim = _compute_filters_similarity(filters1, filters2)
    having_sim = _compute_having_similarity(having1, having2)
    return (
        0.30 * tables_sim
        + 0.25 * filter_sim
        + 0.15 * select_sim
        + 0.15 * group_sim
        + 0.08 * order_sim
        + 0.07 * having_sim
    )


def _cte_step_similarity(cte1: RuntimeCteStep, cte2: RuntimeCteStep) -> float:
    """
    ``_base_similarity`` applied to two CTE bodies.

    Args:

        cte1: First step.

        cte2: Second step.

    Returns:

        Score in ``[0, 1]``.
    """
    return _base_similarity(
        cte1.tables or [],
        cte2.tables or [],
        cte1.select_cols or [],
        cte2.select_cols or [],
        cte1.group_by_cols or [],
        cte2.group_by_cols or [],
        cte1.order_by_cols or [],
        cte2.order_by_cols or [],
        cte1.filters_param or [],
        cte2.filters_param or [],
        cte1.having_param or [],
        cte2.having_param or [],
    )


def intent_similarity(intent1: RuntimeIntent | ConcreteIntent, intent2: RuntimeIntent | ConcreteIntent) -> float:
    """
    Blend weighted main-body clause similarity with per-step CTE similarity (position-aligned).

    Main body uses :func:`_base_similarity` (tables, filters, select, group, order, having). It does not use :func:`aetherdialect._utils.intent_key`; keys hash serialised clauses while similarity uses clause-wise scores, so equal keys imply high similarity but the converse is not guaranteed.

    Args:

        intent1: First intent.

        intent2: Second intent.

    Returns:

        Score in ``[0, 1]``.
    """
    base_sim = _base_similarity(
        intent1.tables or [],
        intent2.tables or [],
        intent1.select_cols or [],
        intent2.select_cols or [],
        intent1.group_by_cols or [],
        intent2.group_by_cols or [],
        intent1.order_by_cols or [],
        intent2.order_by_cols or [],
        intent1.filters_param or [],
        intent2.filters_param or [],
        intent1.having_param or [],
        intent2.having_param or [],
    )
    ctes1 = intent1.cte_steps or []
    ctes2 = intent2.cte_steps or []
    n_cte = max(len(ctes1), len(ctes2))
    if n_cte == 0:
        return base_sim
    intent_weight = {1: 0.7, 2: 0.6}.get(n_cte, 0.4)
    cte_total_weight = 1.0 - intent_weight
    cte_per_weight = cte_total_weight / n_cte
    cte_score = 0.0
    for i in range(n_cte):
        if i < len(ctes1) and i < len(ctes2):
            cte_score += cte_per_weight * _cte_step_similarity(ctes1[i], ctes2[i])
    return intent_weight * base_sim + cte_score


def _jaccard(set1: set[str], set2: set[str]) -> float:
    """
    |intersection| / |union| for string sets; ``1.0`` when both empty.

    Args:

        set1: First set.

        set2: Second set.

    Returns:

        Jaccard coefficient in ``[0, 1]``.
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0


@dataclass(frozen=True, slots=True)
class TrustedTemplateHit:
    """Trusted template row selected by fuzzy question match against ``value_history`` (paths 2.x)."""

    template: Template
    reuse_hit: QuestionReuseMatch


def find_trusted_template_match(
    question: str,
    templates: list[Template],
    *,
    shape_question_index: dict[str, list[str]] | None = None,
    question_token_index: dict[str, list[Any]] | None = None,
    intent_key_index: dict[str, list[str]] | None = None,
    union_family_index: dict[str, list[str]] | None = None,
    candidate_intent: RuntimeIntent | None = None,
) -> TrustedTemplateHit | None:
    """
    If ``trust_level>=1`` and a history question fuzzy-matches, return the template and match details.

    Args:

        question: Raw or normalized user question; normalized inside ``match_question_against_template_history``.

        templates: Candidate templates.

        shape_question_index: Optional coarse index keyed by ``compute_shape_form``.

        question_token_index: Optional inverted index for question-token fingerprints.

        intent_key_index: Optional inverted index from structural intent key to template ids.

        union_family_index: Optional inverted index (body key and ``body|join`` composite) to template ids.

        candidate_intent: When provided with both indexes, templates are narrowed to
            ``(union_family[body|join] ∪ union_family[body]) ∩ intent_key_index[intent_key(runtime)]``
            before fuzzy history matching; when that intersection is empty, narrowing is skipped
            (shape + question-token indexes still apply).

    Returns:

        :class:`TrustedTemplateHit` on hit, else ``None``.
    """
    if not templates:
        return None

    scan_templates: list[Template] = list(templates)
    if (
        candidate_intent is not None
        and union_family_index is not None
        and intent_key_index is not None
    ):
        bk = body_similarity_key(candidate_intent)
        jk = join_path_key_runtime(candidate_intent)
        uf_key = f"{bk}|{jk}"
        union_ids: set[str] = set()
        for key in (uf_key, bk):
            raw_ids = union_family_index.get(key)
            if isinstance(raw_ids, list):
                union_ids.update(str(x) for x in raw_ids)
        raw_ik = intent_key_index.get(intent_key(candidate_intent))
        intent_ids: set[str] = set()
        if isinstance(raw_ik, list):
            intent_ids.update(str(x) for x in raw_ik)
        narrowed_ids = union_ids & intent_ids
        if narrowed_ids:
            narrowed = [t for t in templates if t.id in narrowed_ids]
            if narrowed:
                scan_templates = narrowed

    hit = match_question_against_template_history(
        question,
        scan_templates,
        shape_question_index=shape_question_index,
        question_token_index=question_token_index,
    )
    if hit is None:
        return None
    for tpl in templates:
        if tpl.id == hit.template_id:
            debug(f"[intent_process.find_trusted_template_match] fuzzy match with template {tpl.id}")
            return TrustedTemplateHit(template=tpl, reuse_hit=hit)
    return None


def cte_structural_signature(steps: list) -> list[tuple[str, str]]:
    """
    Sorted ``(cte_name, body_sig)`` tuples excluding select columns (union logic).

    Args:

        steps: CTE steps.

    Returns:

        Sorted signature list.
    """
    sigs: list[tuple[str, str]] = []
    for cte in steps:
        parts: list[str] = [
            cte.grain or "row_level",
            ",".join(sorted(cte.tables or [])),
            ",".join(sorted(f.signature_key for f in (cte.filters_param or []))),
            ",".join(sorted(g.signature_key for g in (cte.group_by_cols or []))),
            ",".join(sorted(o.signature_key for o in (cte.order_by_cols or []))),
            ",".join(sorted(h.signature_key for h in (cte.having_param or []))),
            ",".join(sorted(s.signature_key for s in (cte.window_registry or []))),
            ",".join(sorted(s.signature_key for s in (cte.case_registry or []))),
        ]
        sigs.append((cte.cte_name, "|".join(parts)))
    return sorted(sigs, key=lambda t: t[0])


def _structural_body_matches(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
) -> bool:
    """
    True when tables, grain, limit, filters, group/order/having, and CTE skeletons match.

    Args:

        intent: New runtime intent.

        concrete: Template's concrete intent.

    Returns:

        Whether non-select structure is identical.
    """
    if (intent.grain or "row_level") != (concrete.grain or "row_level"):
        return False
    if sorted(intent.tables or []) != sorted(concrete.tables or []):
        return False
    if intent.limit != concrete.limit:
        return False

    i_filters = sorted(f.signature_key for f in (intent.filters_param or []))
    c_filters = sorted(f.signature_key for f in (concrete.filters_param or []))
    if i_filters != c_filters:
        return False

    i_gb = sorted(g.signature_key for g in (intent.group_by_cols or []))
    c_gb = sorted(g.signature_key for g in (concrete.group_by_cols or []))
    if i_gb != c_gb:
        return False

    i_ob = sorted(o.signature_key for o in (intent.order_by_cols or []))
    c_ob = sorted(o.signature_key for o in (concrete.order_by_cols or []))
    if i_ob != c_ob:
        return False

    i_hav = sorted(h.signature_key for h in (intent.having_param or []))
    c_hav = sorted(h.signature_key for h in (concrete.having_param or []))
    if i_hav != c_hav:
        return False

    if cte_structural_signature(intent.cte_steps or []) != cte_structural_signature(concrete.cte_steps or []):
        return False

    def _reg_sig(
        wr: list[WindowRegistryStep] | None,
        cr: list[CaseRegistryStep] | None,
    ) -> str:
        w = ",".join(sorted(s.signature_key for s in (wr or [])))
        c = ",".join(sorted(s.signature_key for s in (cr or [])))
        return f"{w}|{c}"

    if _reg_sig(intent.window_registry, intent.case_registry) != _reg_sig(
        concrete.window_registry, concrete.case_registry
    ):
        return False

    return True


def select_col_diff(
    intent_cols: list[SelectCol],
    concrete_cols: list[SelectCol],
) -> tuple[bool, int]:
    """
    Compare agg select keys for equality; count symmetric diff of non-agg keys.

    Args:

        intent_cols: Runtime select list.

        concrete_cols: Template select list.

    Returns:

        ``(aggregates_match, non_agg_symmetric_diff_count)``.
    """
    i_agg = sorted(s.signature_key for s in intent_cols if s.is_aggregated)
    c_agg = sorted(s.signature_key for s in concrete_cols if s.is_aggregated)
    agg_match = i_agg == c_agg

    i_non = set(s.signature_key for s in intent_cols if not s.is_aggregated)
    c_non = set(s.signature_key for s in concrete_cols if not s.is_aggregated)
    non_agg_diff = len(i_non.symmetric_difference(c_non))

    return agg_match, non_agg_diff


def _diff_cols_span_disjoint_tables(
    intent_cols: list[SelectCol],
    concrete_cols: list[SelectCol],
    intent_tables: list[str],
    concrete_tables: list[str],
) -> bool:
    """
    True if any differing non-agg column sits on a table outside both table sets' intersection.

    Args:

        intent_cols: Runtime selects.

        concrete_cols: Template selects.

        intent_tables: Runtime ``tables``.

        concrete_tables: Template ``tables``.

    Returns:

        Whether the diff spans disjoint table namespaces.
    """
    shared_tables = set(intent_tables or []) & set(concrete_tables or [])
    i_non = {s.signature_key for s in intent_cols if not s.is_aggregated}
    c_non = {s.signature_key for s in concrete_cols if not s.is_aggregated}
    diff_keys = i_non.symmetric_difference(c_non)
    if not diff_keys:
        return False
    all_cols = list(intent_cols) + list(concrete_cols)
    for sc in all_cols:
        if sc.is_aggregated or sc.signature_key not in diff_keys:
            continue
        term = sc.expr.primary_term
        tbl = term.split(".")[0] if "." in term else ""
        if tbl and tbl not in shared_tables:
            return True
    return False


def _non_agg_select_signature_keys(select_cols: list[SelectCol] | None) -> set[str]:
    """Return ``signature_key`` values for non-aggregated select columns."""

    return {sc.signature_key for sc in select_cols or [] if not sc.is_aggregated}


def select_col_is_plain_column(sc: SelectCol) -> bool:
    """
    Return True when *sc* is a bare ``table.column`` reference with no transforms.

    Path 4 widening only inlines select columns that need no expression rebuild — neither aggregates, scalar/inner-scalar functions, coefficients, expression composition, registry window/case references, nor CASE expressions are tolerated.
    """

    rid = expr_registry_ref(sc.expr) or ""
    if rid.startswith("w") or rid.startswith("c"):
        return False
    if sc.is_aggregated:
        return False
    expr = sc.expr
    if expr.scalar_func or expr.inner_scalar_func or expr.agg_func:
        return False
    if expr.cast_type or expr.interval is not None or expr.raw_sql or expr.keyword:
        return False
    if expr.sub_groups or expr.add_values or expr.sub_values:
        return False
    if expr.column_ref and not expr.add_groups:
        return True
    if len(expr.add_groups) != 1:
        return False
    g = expr.add_groups[0]
    if g.divide or g.agg_func or g.scalar_func or g.inner_scalar_func:
        return False
    if g.coefficient != 1.0 or g.coeff_param_key or g.distinct:
        return False
    if len(g.multiply) != 1:
        return False
    leaf = g.multiply[0]
    if not leaf.column_ref:
        return False
    if leaf.add_groups or leaf.sub_groups or leaf.scalar_func or leaf.inner_scalar_func:
        return False
    return True


def _diff_select_cols_are_plain_columns(
    intent_cols: list[SelectCol],
    concrete_cols: list[SelectCol],
) -> bool:
    """Return True when every non-aggregated symmetric-diff select column is a plain column ref."""

    i_map = {sc.signature_key: sc for sc in intent_cols if not sc.is_aggregated}
    c_map = {sc.signature_key: sc for sc in concrete_cols if not sc.is_aggregated}
    diff_keys = (set(i_map) | set(c_map)) - (set(i_map) & set(c_map))
    for k in diff_keys:
        sc = i_map.get(k) or c_map.get(k)
        if sc is None:
            continue
        if not select_col_is_plain_column(sc):
            return False
    return True


class UnionSelectColumnDelta(str, Enum):
    """Select-list delta between runtime intent and template concrete intent (non-aggregated keys)."""

    EQUAL = "equal"
    TEMPLATE_ONLY_EXTRA = "template_only_extra"
    INTENT_ONLY_EXTRA = "intent_only_extra"
    BOTH_EXTRA = "both_extra"


def resolve_sql_path(
    *,
    matched_template: Template | None,
    cols_changed: bool,
    union_sql_path: GenerationPath | None,
) -> GenerationPath:
    """
    Resolve the persisted :class:`GenerationPath` for template-scoped SQL generation.

    When a template row is in play but no precomputed union path was supplied, infer
    ``INTENT_DIRECT_MATCH`` versus union widen codes from column-change flags only.

    Args:

        matched_template: Accepted template chosen for reuse, if any.

        cols_changed: Whether merged union columns differ from the template concrete list.

        union_sql_path: Path from structural union analysis, when already computed.

    Returns:

        Canonical path ``1``–``5`` family member; :attr:`GenerationPath.FRESH` when no template.
    """

    if matched_template is None:
        return GenerationPath.FRESH
    if union_sql_path is not None:
        return union_sql_path
    return GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN if cols_changed else GenerationPath.INTENT_DIRECT_MATCH


def generation_path_for_eligible_union(
    *,
    cols_changed: bool,
    select_equal: bool,
    delta: UnionSelectColumnDelta,
) -> GenerationPath:
    """
    Map structural union facts to the canonical :class:`GenerationPath` for this match.

    Args:

        cols_changed: Whether merged union column keys differ from the template concrete list.

        select_equal: Whether aggregate rules hold with zero symmetric non-agg diff.

        delta: Non-aggregated key-set relationship between runtime and concrete selects.

    Returns:

        ``INTENT_DIRECT_MATCH``, ``RUNTIME_SUBSET_TEMPLATE_WIDE``, or union widen codes ``4.1`` / ``4.2``.
    """

    if not cols_changed and delta is UnionSelectColumnDelta.TEMPLATE_ONLY_EXTRA:
        return GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE
    if not cols_changed and select_equal and delta is UnionSelectColumnDelta.EQUAL:
        return GenerationPath.INTENT_DIRECT_MATCH
    if cols_changed and delta is UnionSelectColumnDelta.INTENT_ONLY_EXTRA:
        return GenerationPath.UNION_TEMPLATE_WIDEN
    if cols_changed and delta is UnionSelectColumnDelta.BOTH_EXTRA:
        return GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN
    if cols_changed and delta is UnionSelectColumnDelta.TEMPLATE_ONLY_EXTRA:
        return GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN
    if not cols_changed:
        return GenerationPath.INTENT_DIRECT_MATCH
    return GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN


def reconcile_template_store_until_stable(
    templates: dict[str, Template],
    *,
    max_iterations: int = 16,
    template_store_view: Any | None = None,
) -> int:
    """
    Run paired union reconcilers until a full pass removes no template ids.

    Args:

        templates: Accepted template id map (mutated in place).

        max_iterations: Safety cap to avoid infinite loops on pathological stores.

        template_store_view: When a :class:`~aetherdialect._templates.TemplateStoreView` is supplied,
            reconcilers read ``union_family_index`` from its header indexes; after each pass the
            view's matcher indexes are refreshed from the live template bodies.

    Returns:

        Count of template ids removed across all iterations.
    """

    from ._templates import TemplateStoreView, refresh_template_store_indexes

    total = 0
    for _ in range(max_iterations):
        r1 = reconcile_union_family_after_mutation(
            templates,
            union_family_index=None,
            template_store_view=template_store_view,
        )
        r2 = reconcile_union_family_body_join_after_mutation(
            templates,
            union_family_index=None,
            template_store_view=template_store_view,
        )
        n = len(r1) + len(r2)
        total += n
        if isinstance(template_store_view, TemplateStoreView):
            refresh_template_store_indexes(
                template_store_view,
                template_objs=list(templates.values()),
            )
        if n == 0:
            break
    return total


def _mulgroup_strict_union_safe(mg: MulGroup) -> bool:
    """Return False when a multiply group carries scalar transforms or non-trivial coefficients."""

    if mg.coefficient != 1.0 or (mg.coeff_param_key or ""):
        return False
    if mg.scalar_func or mg.inner_scalar_func:
        return False
    if mg.sarg_param_keys or mg.isarg_param_keys:
        return False
    return True


def _expr_strict_union_safe(expr: NormalizedExpr) -> bool:
    """Return False when outer expression uses scalar transforms reserved for fresh-SQL paths."""

    if expr.scalar_func or expr.inner_scalar_func:
        return False
    if expr.sarg_param_keys or expr.isarg_param_keys:
        return False
    for g in expr.add_groups + expr.sub_groups:
        if not _mulgroup_strict_union_safe(g):
            return False
    return True


def _select_col_strict_union_safe(sc: SelectCol) -> bool:
    """Return False when window, CASE, or expression tree blocks union merge."""

    rid = expr_registry_ref(sc.expr) or ""
    if rid.startswith("w") or rid.startswith("c"):
        return False
    return _expr_strict_union_safe(sc.expr)


def _concrete_step_select_cols_safe(select_cols: list[SelectCol] | None) -> bool:
    """Return True when every non-aggregated concrete select passes the strict union gate."""

    for sc in select_cols or []:
        if sc.is_aggregated:
            continue
        if not _select_col_strict_union_safe(sc):
            return False
    return True


def _runtime_intent_union_select_structures_safe(intent: RuntimeIntent) -> bool:
    """Return True when main and CTE selects satisfy strict union merge rules."""

    if not _concrete_step_select_cols_safe(intent.select_cols):
        return False
    for step in intent.cte_steps or []:
        if not _concrete_step_select_cols_safe(step.select_cols):
            return False
    return True


def _union_sql_eligibility_strict_shape(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
) -> bool:
    """Return True when both sides pass scalar- and coefficient-gates for union comparison."""

    if not _runtime_intent_union_select_structures_safe(intent):
        return False
    rt = concrete_intent_to_runtime_skeleton(concrete)
    return _runtime_intent_union_select_structures_safe(rt)


def classify_union_merge_case(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
) -> UnionSelectColumnDelta:
    """Classify how runtime select keys differ from template concrete selects (non-aggregated only)."""

    i_keys = _non_agg_select_signature_keys(intent.select_cols)
    c_keys = _non_agg_select_signature_keys(concrete.select_cols)
    i_only = i_keys - c_keys
    c_only = c_keys - i_keys
    if not i_only and not c_only:
        return UnionSelectColumnDelta.EQUAL
    if i_only and c_only:
        return UnionSelectColumnDelta.BOTH_EXTRA
    if i_only:
        return UnionSelectColumnDelta.INTENT_ONLY_EXTRA
    return UnionSelectColumnDelta.TEMPLATE_ONLY_EXTRA


def _join_path_signature_hash(layers: list[list[str]]) -> str:
    """SHA-256 hex of layered join path signatures (main then CTEs)."""

    return hashlib.sha256(stable_json(layers).encode("utf-8")).hexdigest()


def join_path_key_runtime(intent: RuntimeIntent) -> str:
    """Stable join fingerprint for a runtime intent."""

    layers: list[list[str]] = [list(intent.chosen_join_path_signature or [])]
    for step in intent.cte_steps or []:
        layers.append(list(step.chosen_join_path_signature or []))
    return _join_path_signature_hash(layers)


def join_path_key_concrete(concrete: ConcreteIntent) -> str:
    """Stable join fingerprint for a concrete intent signature."""

    layers: list[list[str]] = [list(concrete.chosen_join_path_signature or [])]
    for step in concrete.cte_steps or []:
        layers.append(list(step.chosen_join_path_signature or []))
    return _join_path_signature_hash(layers)


def join_runtime_intent_has_join_fingerprint(intent: RuntimeIntent) -> bool:
    """Return True when the runtime intent carries at least one non-empty join path signature layer."""

    if intent.chosen_join_path_signature:
        return True
    return any(bool(s.chosen_join_path_signature) for s in (intent.cte_steps or []))


def join_runtime_matches_template_concrete(intent: RuntimeIntent, tmpl: Template) -> bool:
    """
    Return True when runtime join layers match the template concrete fingerprint.

    When the runtime intent has no join fingerprint yet, returns True so callers can defer gating.
    """

    if not join_runtime_intent_has_join_fingerprint(intent):
        return True
    return join_path_key_runtime(intent) == join_path_key_concrete(tmpl.intent_signature)


def union_runtime_concrete_compatibility(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
) -> tuple[list[SelectCol], bool, UnionSelectColumnDelta, int] | None:
    """
    Return union columns, column-change flag, merge case, and non-agg diff when structural union rules pass.

    Does not evaluate template trust; callers apply trust when comparing accepted templates.
    """

    if not _structural_body_matches(intent, concrete):
        return None
    if not _union_sql_eligibility_strict_shape(intent, concrete):
        return None
    agg_match, non_agg_diff = select_col_diff(
        intent.select_cols or [],
        concrete.select_cols or [],
    )
    if not agg_match:
        return None
    if non_agg_diff > MAX_NON_AGG_COL_DIFF:
        return None
    if not _diff_select_cols_are_plain_columns(
        intent.select_cols or [],
        concrete.select_cols or [],
    ):
        return None
    if intent.distinct_select_index != concrete.distinct_select_index:
        return None
    if _diff_cols_span_disjoint_tables(
        intent.select_cols or [],
        concrete.select_cols or [],
        intent.tables or [],
        concrete.tables or [],
    ):
        return None
    union_cols, cols_changed, merge_case = compute_intent_union(intent, concrete)
    return union_cols, cols_changed, merge_case, non_agg_diff


def _structural_compare_from_union_row(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
    sql_fp_val: str,
    row: tuple[list[SelectCol], bool, UnionSelectColumnDelta, int] | None,
    *,
    mode: Literal["full", "warmup_gold_store_check"],
) -> StructuralCompareResult:
    """Build a ``StructuralCompareResult`` from a precomputed optional union-compatibility row."""

    agg_match, non_agg_diff = select_col_diff(
        intent.select_cols or [],
        concrete.select_cols or [],
    )
    select_equal = bool(agg_match) and non_agg_diff == 0
    score: float | None = None
    if mode == "full":
        score = float(intent_similarity(intent, concrete))
    if row is None:
        return StructuralCompareResult(
            non_agg_symmetric_diff=non_agg_diff,
            union_eligible=False,
            similarity_score=score,
        )
    union_cols, cols_changed, merge_case, nad = row
    union_path = generation_path_for_eligible_union(
        cols_changed=cols_changed,
        select_equal=select_equal,
        delta=merge_case,
    )
    return StructuralCompareResult(
        non_agg_symmetric_diff=nad,
        union_eligible=True,
        union_cols=list(union_cols),
        cols_changed=cols_changed,
        union_sql_path=union_path,
        similarity_score=score,
    )


def structural_compare_runtime(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
    sql_fp_val: str,
    *,
    mode: Literal["full", "warmup_gold_store_check"] = "full",
) -> StructuralCompareResult:
    """
    Compare *intent* to a stored concrete signature and SQL fingerprint without template trust gating.

    Union eligibility matches accepted-template rules except ``trust_level`` is not consulted.
    """

    row = union_runtime_concrete_compatibility(intent, concrete)
    return _structural_compare_from_union_row(intent, concrete, sql_fp_val, row, mode=mode)


def structural_compare(
    intent: RuntimeIntent,
    tmpl: Template,
    *,
    mode: Literal["full", "warmup_gold_store_check"] = "full",
) -> StructuralCompareResult:
    """
    Compare *intent* to *tmpl* for body key, join path, union eligibility, and ``union_sql_path``.

    *mode* ``full`` attaches ``similarity_score`` via :func:`intent_similarity`; ``warmup_gold_store_check`` omits that score.
    """

    concrete = tmpl.intent_signature
    row = union_template_compatibility(intent, tmpl)
    return _structural_compare_from_union_row(intent, concrete, tmpl.sql_fp, row, mode=mode)


def union_template_compatibility(
    intent: RuntimeIntent,
    tmpl: Template,
) -> tuple[list[SelectCol], bool, UnionSelectColumnDelta, int] | None:
    """Return union columns, column-change flag, merge case, and non-agg diff when *tmpl* is eligible."""

    if tmpl.trust_level < 1:
        return None
    return union_runtime_concrete_compatibility(intent, tmpl.intent_signature)


def collect_structural_match_templates(
    intent: RuntimeIntent,
    templates: dict[str, Template],
) -> list[Template]:
    """Trusted templates with same body similarity key as *intent* and union-compatible without column change."""

    ibk = body_similarity_key(intent)
    out: list[Template] = []
    for tmpl in templates.values():
        cr = structural_compare(intent, tmpl, mode="warmup_gold_store_check")
        if not cr.union_eligible or cr.cols_changed:
            continue
        if body_similarity_key_for_concrete(tmpl.intent_signature) != ibk:
            continue
        out.append(tmpl)
    out.sort(key=lambda t: t.id)
    return out


def _union_family_index_from_templates(templates: dict[str, Template]) -> dict[str, list[str]]:
    """Inverted union-family buckets (body key and ``body|join`` composite) used by reconcilers."""

    idx: dict[str, set[str]] = defaultdict(set)
    for tid, tmpl in templates.items():
        bk = body_similarity_key_for_concrete(tmpl.intent_signature)
        jk = join_path_key_concrete(tmpl.intent_signature)
        idx[bk].add(tid)
        idx[f"{bk}|{jk}"].add(tid)
    return {k: sorted(v) for k, v in idx.items()}


def reconcile_union_family_after_mutation(
    templates: dict[str, Template],
    *,
    union_family_index: dict[str, list[str]] | None = None,
    template_store_view: Any | None = None,
) -> list[str]:
    """
    Merge accepted templates that share the same warmup ``template_instance_key`` into one row.

    Args:

        templates: Live accepted-template map (mutated in place).

        union_family_index: Optional inverted union-family index (body keys and ``body|join`` keys).

        template_store_view: When *union_family_index* is omitted, read persisted
            ``union_family_index`` from this view's header indexes.

    Returns:

        Template ids removed after merging value history into the lexicographically smallest keeper id.
    """

    removed: list[str] = []

    def _merge_instance_group(tids_sorted: list[str]) -> None:
        nonlocal removed
        if len(tids_sorted) <= 1:
            return
        keeper_id = tids_sorted[0]
        keeper = templates[keeper_id]
        for dup_id in tids_sorted[1:]:
            if dup_id not in templates:
                continue
            dup = templates[dup_id]
            for i in range(len(dup.value_history.questions)):
                q = dup.value_history.questions[i]
                pv = dup.value_history.param_values[i] if i < len(dup.value_history.param_values) else {}
                nl = dup.value_history.natural_language[i] if i < len(dup.value_history.natural_language) else ""
                keeper.value_history.add(pv, q, nl)
            del templates[dup_id]
            removed.append(dup_id)

    ufi = union_family_index
    if not ufi and template_store_view is not None:
        from ._templates import TemplateStoreView

        if isinstance(template_store_view, TemplateStoreView):
            raw = template_store_view._indexes.get(TEMPLATE_UNION_FAMILY_INDEX_KEY)
            if isinstance(raw, dict) and any(isinstance(v, list) and v for v in raw.values()):
                ufi = {str(k): [str(x) for x in v] for k, v in raw.items() if isinstance(v, list)}
    if not ufi:
        ufi = _union_family_index_from_templates(templates)

    if ufi:
        for bk, cand_tids in ufi.items():
            if "|" in bk:
                continue
            uniq_tids = sorted({t for t in cand_tids if t in templates})
            if len(uniq_tids) <= 1:
                continue
            by_inst: dict[str, list[str]] = defaultdict(list)
            for tid in uniq_tids:
                tmpl = templates[tid]
                inst = template_instance_key_from_parts(
                    bk,
                    join_path_key_concrete(tmpl.intent_signature),
                    tmpl.sql_fp,
                )
                by_inst[inst].append(tid)
            for _inst, tids in by_inst.items():
                _merge_instance_group(sorted(tids))
        return removed

    by_inst: dict[str, list[str]] = defaultdict(list)
    for tid, tmpl in list(templates.items()):
        inst = template_instance_key_from_parts(
            body_similarity_key_for_concrete(tmpl.intent_signature),
            join_path_key_concrete(tmpl.intent_signature),
            tmpl.sql_fp,
        )
        by_inst[inst].append(tid)
    for _inst, tids in by_inst.items():
        _merge_instance_group(sorted(tids))
    return removed


class UnionMatchCandidate(NamedTuple):
    """One accepted template row eligible for union-style reuse with resolved column metadata."""

    template: Template
    union_cols: list[SelectCol]
    cols_changed: bool
    union_sql_path: GenerationPath
    non_agg_symmetric_diff: int


def list_union_match_candidates(
    intent: RuntimeIntent,
    templates: dict[str, Template],
) -> list[UnionMatchCandidate]:
    """
    List every trusted template that passes structural union gates for *intent*.

    Used for paths ``3`` and ``4.x`` so the join phase can pick among templates that differ only in stored join fingerprints.
    """

    def _collect() -> list[UnionMatchCandidate]:
        rows: list[UnionMatchCandidate] = []
        for tmpl in templates.values():
            cr = structural_compare(intent, tmpl, mode="full")
            if not cr.union_eligible or cr.union_sql_path is None:
                continue
            rows.append(
                UnionMatchCandidate(
                    tmpl,
                    cr.union_cols,
                    cr.cols_changed,
                    cr.union_sql_path,
                    cr.non_agg_symmetric_diff,
                )
            )
        rows.sort(key=lambda r: (r.non_agg_symmetric_diff, len(r.union_cols), r.template.id))
        return rows

    rows = _collect()
    if len(rows) >= 2:
        head = (rows[0].non_agg_symmetric_diff, len(rows[0].union_cols))
        tie_n = sum(1 for r in rows if (r.non_agg_symmetric_diff, len(r.union_cols)) == head)
        if tie_n > 1:
            reconcile_template_store_until_stable(templates)
            rows = _collect()
    return rows


def pick_union_match_for_runtime_join(
    intent: RuntimeIntent,
    candidates: Sequence[UnionMatchCandidate],
) -> UnionMatchCandidate | None:
    """
    Choose the union candidate whose stored join fingerprint matches the runtime intent.

    When the runtime intent has no join fingerprint yet and every candidate shares one join key, returns the lexicographically smallest stable pick. When candidates disagree on join keys and the runtime intent is not pinned yet, returns ``None`` so the join LLM can run first.
    """

    if not candidates:
        return None
    jkeys = {join_path_key_concrete(c.template.intent_signature) for c in candidates}
    if len(jkeys) > 1 and not join_runtime_intent_has_join_fingerprint(intent):
        return None
    if len(jkeys) == 1 and not join_runtime_intent_has_join_fingerprint(intent):
        return min(
            candidates,
            key=lambda c: (c.non_agg_symmetric_diff, len(c.union_cols), c.template.id),
        )
    filtered = [c for c in candidates if join_runtime_matches_template_concrete(intent, c.template)]
    if not filtered:
        return None
    return min(
        filtered,
        key=lambda c: (c.non_agg_symmetric_diff, len(c.union_cols), c.template.id),
    )


def reconcile_union_family_body_join_after_mutation(
    templates: dict[str, Template],
    *,
    union_family_index: dict[str, list[str]] | None = None,
    template_store_view: Any | None = None,
) -> list[str]:
    """
    Merge templates that share the same warmup ``(body_key, join_path_key)`` when union rules allow.

    Args:

        templates: Live accepted-template map (mutated in place).

        union_family_index: Optional inverted union-family index (``body|join`` composite keys).

        template_store_view: When *union_family_index* is omitted, read persisted
            ``union_family_index`` from this view's header indexes.

    Returns:

        Template ids removed after value-history merge into the lexicographically smallest keeper id.
    """

    removed: list[str] = []

    def _merge_family_group(tids_sorted: list[str]) -> None:
        nonlocal removed
        if len(tids_sorted) <= 1:
            return
        keeper_id = tids_sorted[0]
        keeper = templates[keeper_id]
        keeper_rt = concrete_intent_to_runtime_skeleton(keeper.intent_signature)
        for dup_id in tids_sorted[1:]:
            if dup_id not in templates:
                continue
            dup = templates[dup_id]
            row = union_template_compatibility(keeper_rt, dup)
            if row is None:
                continue
            for i in range(len(dup.value_history.questions)):
                q = dup.value_history.questions[i]
                pv = dup.value_history.param_values[i] if i < len(dup.value_history.param_values) else {}
                nl = dup.value_history.natural_language[i] if i < len(dup.value_history.natural_language) else ""
                keeper.value_history.add(pv, q, nl)
            del templates[dup_id]
            removed.append(dup_id)

    ufi = union_family_index
    if not ufi and template_store_view is not None:
        from ._templates import TemplateStoreView

        if isinstance(template_store_view, TemplateStoreView):
            raw = template_store_view._indexes.get(TEMPLATE_UNION_FAMILY_INDEX_KEY)
            if isinstance(raw, dict) and any(isinstance(v, list) and v for v in raw.values()):
                ufi = {str(k): [str(x) for x in v] for k, v in raw.items() if isinstance(v, list)}
    if not ufi:
        ufi = _union_family_index_from_templates(templates)

    if ufi:
        for key, cand_tids in ufi.items():
            if "|" not in key:
                continue
            bk, jk = key.split("|", 1)
            uniq_tids = sorted({t for t in cand_tids if t in templates})
            if len(uniq_tids) <= 1:
                continue
            _merge_family_group(uniq_tids)
        return removed

    by_fam: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tid, tmpl in list(templates.items()):
        c = tmpl.intent_signature
        bk = body_similarity_key_for_concrete(c)
        jk = join_path_key_concrete(c)
        by_fam[(bk, jk)].append(tid)
    for _fam, tids in by_fam.items():
        _merge_family_group(sorted(tids))
    return removed


def match_template_for_union(
    intent: RuntimeIntent,
    templates: dict[str, Template],
) -> tuple[Template, list[SelectCol], bool, GenerationPath] | None:
    """
    Pick a ``trust_level>=1`` template with matching body and small select diff.

    Args:

        intent: Validated runtime intent.

        templates: Id -> template map.

    Returns:

        ``(template, union_select_cols, cols_changed, union_sql_path)`` or ``None``.
    """

    rows = list_union_match_candidates(intent, templates)
    if not rows:
        return None
    best = rows[0]
    debug(
        f"[intent_process.match_template_for_union] matched template={best.template.id} "
        f"non_agg_diff={best.non_agg_symmetric_diff} union_cols={len(best.union_cols)} sql_path={best.union_sql_path.code}"
    )
    return best.template, best.union_cols, best.cols_changed, best.union_sql_path


def compute_intent_union(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
) -> tuple[list[SelectCol], bool, UnionSelectColumnDelta]:
    """
    Merge selects by ``signature_key``: concrete order first, then new from *intent*.

    Args:

        intent: Runtime intent.

        concrete: Matched template intent.

    Returns:

        ``(union_cols, changed_vs_concrete, merge_case)``.
    """
    seen_keys: set[str] = set()
    union_cols: list[SelectCol] = []
    for sc in concrete.select_cols or []:
        key = sc.signature_key
        if key not in seen_keys:
            seen_keys.add(key)
            union_cols.append(sc)
    for sc in intent.select_cols or []:
        key = sc.signature_key
        if key not in seen_keys:
            seen_keys.add(key)
            union_cols.append(sc)

    cols_changed = sorted(seen_keys) != sorted(s.signature_key for s in (concrete.select_cols or []))

    sorted_union = sort_select_cols(union_cols)
    merge_case = classify_union_merge_case(intent, concrete)
    debug(
        f"[intent_process.compute_intent_union] union_cols={len(sorted_union)} "
        f"cols_changed={cols_changed} merge_case={merge_case.value}"
    )
    return sorted_union, cols_changed, merge_case
