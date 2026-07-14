"""LLM intent parsing, repair loops, template matching, and union helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple

from ._config import (
    PolicyConfig,
)
from ._constants import (
    ASK_PHASE_B,
    ASK_PHASE_C,
    ASK_PHASE_D,
    ASK_PHASE_E,
    ASK_PHASE_F,
    ASK_PHASE_G,
    ASK_PHASE_H,
    COMPOSE_SUPPORTED_CAPABILITIES,
    DIAGNOSTIC_CODE_COMPOSE_REPAIR,
    DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART,
    DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY,
    ENCODER_IR_ASSEMBLY_RULES,
    ENCODER_NL_PHRASE_MAPPINGS,
    ENCODER_NL_TO_IR_GUIDANCE,
    EXPR_TABLE_COLUMN_REF_RE,
    FORMAT_STRUCTURAL_GUIDANCE,
    GROUND_SUPPORTED_CAPABILITIES,
    IDENTIFIER_RE,
    INTENT_COMPOSE_SYSTEM,
    INTENT_CRITICAL_RULES,
    INTENT_FORMAT_REPAIR_JSON_RULES,
    INTENT_GROUND_SYSTEM,
    INTENT_INTERPRET_SYSTEM,
    INTENT_PARSE_RULES_APPEND,
    INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR,
    INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE,
    INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES,
    INTERPRET_PLAN_SCHEMA,
    INTERPRET_SUPPORTED_CAPABILITIES,
    LOGICAL_DECOMPOSITION_GUIDANCE,
    LOGICAL_INTENT_SCHEMA,
    MAX_NON_AGG_COL_DIFF,
    NATURAL_LANGUAGE_REFUSAL_PARSE_ERROR,
    PLANNER_NL_CONVENTIONS,
    PLANNER_PROSE_FIELDS,
    REGISTRY_TOKEN_PATTERN,
    REPAIR_INSTRUCTIONS,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    VALID_HAVING_OPS,
    GenerationPath,
)
from ._contracts_base import (
    FailureCategory,
    FilterParam,
    HavingParam,
    LogicalIntent,
    MulGroup,
    NormalizedExpr,
    expr_registry_ref,
)
from ._contracts_core import (
    ConcreteCteStep,
    ConcreteIntent,
    FeedbackKind,
    InterpretPlan,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    StructuralCompareResult,
    Template,
    concrete_intent_to_runtime_skeleton,
    intent_prompt_structural_index,
)
from ._contracts_schema import (
    CaseRegistryStep,
    IntentIssue,
    SchemaGraph,
    WindowRegistryStep,
    make_intent_issue,
)
from ._core_utils import (
    debug,
    diagnostic_debug_enabled,
    diagnostic_pipeline_trace_full_enabled,
    notify,
    pipeline_trace,
    stable_json,
)
from ._dialect import extra_filter_ops_for_engine
from ._intent_expr import (
    RestartBudget,
    assign_param_keys,
    build_cte_output_metadata,
    canonicalize_temporal_unit_args,
    classify_cte_expr,
    collect_raw_param_values,
    decompose_between_params,
    dedupe_prior_question_feedback_rows,
    derive_cte_output_columns,
    ensure_scalar_func_defaults,
    extract_structural_params,
    get_templates_module,
    in_turn_row_from_semantic_errors,
    intent_similarity,
    interpret_plan_for_ground,
    is_template_store_view,
    normalize_date_diff_raw_values,
    normalize_in_raw_values,
    parse_intent_response,
    parse_interpret_plan_response,
    parse_logical_intent_response,
    promote_date_subtraction_to_date_diff,
    refresh_template_store_indexes_for_view,
    repair_misclassified_date_diff,
    replace_refs_in_expr,
    runtime_intent_has_refusal_natural_language,
    serialized_prior_feedback_rows,
    tag_case_when_condition_scope,
    tag_expr_numeric,
)
from ._intent_repair import (
    align_filter_value_type_to_exprs,
    apply_filters_to_main_and_ctes,
    auto_repair_filter_having,
    dedup_contradictory_filters,
    dedup_extract_year_vs_column_literal,
    dedup_value_vs_right_expr,
    drop_invalid_case_registry_entries,
    enforce_sensitivity_policy_intent,
    expand_fk_select_to_descriptive,
    expand_shared_pk_tables_for_refs,
    lift_distinct_modifier_in_multiply,
    normalize_boolean_filter_values,
    normalize_in_filter_types,
    normalize_null_filter_values,
    normalize_pk_distinct,
    promote_temporal_keyword_rhs,
    reconcile_tables,
    repair_array_filters_intent,
    repair_case_when_intent,
    repair_cumulative_phrasing_window_intent,
    repair_fk_filter_type_mismatch,
    repair_intent_placeholder_tokens,
    repair_null_equality_filters,
    replace_unknown_scalar_funcs,
    runtime_intent_has_instructional_placeholders,
    sanitize_table_names,
    strip_impossible_having,
    strip_join_conditions,
    strip_spurious_group_by,
)
from ._intent_resolve import (
    UnionSelectColumnDelta,
    apply_aggregatability_gate,
    attribute_post_compose_issue,
    canonicalize_registry_ids,
    check_qualified_refs_exist,
    coerce_filter_group_mode,
    collect_column_refs_for_post_processing,
    compute_intent_union,
    enforce_case_branch_param_keys,
    enforce_cte_grain_consistency,
    enforce_grain_consistency,
    ensure_cte_output_columns_exposure,
    join_path_key_concrete,
    join_path_key_runtime,
    lift_distinct_select_from_raw_sql,
    normalize_count_star,
    normalize_cte_names,
    normalize_filters_havings,
    normalized_expr_is_absent,
    prune_unused_cte_output_columns,
    prune_unused_cte_steps,
    qualify_count_star_mulgroups,
    qualify_cte_output_columns,
    rename_window_registry_steps,
    reorder_cte_steps_by_dag,
    repair_window_partition_group_by_alignment,
    resolve_column_map,
    resolve_cte_column_maps,
    resolve_filter_value_case,
    resolve_window_registry_filter_rhs,
    rewrite_cte_output_refs_to_aliases,
    rewrite_main_query_refs_to_final_cte_columns,
    simplify_exprs,
    sort_select_cols,
    strip_redundant_identifier_group_by,
)
from ._llm_provider import llm_chat
from ._sql_gen import classify_cte_emission, render_feedback_sql
from ._utils import (
    QuestionReuseMatch,
    body_similarity_key,
    body_similarity_key_for_concrete,
    intent_key,
    match_question_against_template_history,
    template_instance_key_from_parts,
)
from ._validation_execute import validate_semantics


def build_intent_interpret_prompt(
    question: str,
    domain_payload: str,
    prior_question_feedback: str,
    prior_user_corrections: tuple[str, ...],
    prior_attempt_failures: tuple[str, ...] = (),
) -> str:
    """Build the Interpret user JSON instructing the model to emit INTERPRET_PLAN_SCHEMA JSON. ``prior_attempt_failures`` carries hints from earlier interpret/ground attempts in the same ask. It is empty on the first attempt (so that prompt is unchanged) and non-empty on retries, which both informs the replan and gives each retry a distinct, deterministically replayable cache key - without it, a re-interpreted retry would reuse the first attempt's key while the live model resampled a different plan, making the recorded chain impossible to replay."""
    body: dict[str, Any] = {
        "task": "Read the question against the domain schema and produce a thinking pathway (interpret_plan).",
        "question": question,
        "schema_domain": json.loads(domain_payload),
        "supported_capabilities": list(INTERPRET_SUPPORTED_CAPABILITIES),
        "interpret_plan_schema": INTERPRET_PLAN_SCHEMA,
    }
    if prior_question_feedback.strip():
        body["prior_question_feedback"] = prior_question_feedback
    if prior_user_corrections:
        body["prior_user_corrections"] = list(prior_user_corrections)
    if prior_attempt_failures:
        body["prior_attempt_failures"] = list(prior_attempt_failures)
    return stable_json(body)


def build_intent_ground_prompt(
    question: str,
    interpret_plan: InterpretPlan,
    ground_payload: str,
    prior_question_feedback: str,
    logical_decomposition_guidance: str,
    prior_user_corrections: tuple[str, ...],
    prior_grounding_failures: tuple[str, ...],
) -> str:
    """Build the Ground user JSON instructing the model to emit LOGICAL_INTENT_SCHEMA JSON."""
    body: dict[str, Any] = {
        "task": "Convert interpret_plan into logical intent JSON with natural-language clause fields bound to schema identifiers.",
        "question": question,
        "interpret_plan": interpret_plan_for_ground(interpret_plan),
        "schema_literal_json": json.loads(ground_payload),
        "logical_intent_json_schema": LOGICAL_INTENT_SCHEMA,
        "nl_conventions": json.loads(stable_json(dict(PLANNER_NL_CONVENTIONS))),
        "logical_schema_rules": [
            "Do not plan UNION, INTERSECT, or EXCEPT; describe set-like needs with plain language or CTE steps.",
            "Use the filter prose field for row predicates; never name EXISTS or NOT EXISTS.",
            "Use cte_steps for self-comparisons and per-entity top-N style questions without prescribing IR shapes.",
            "After structural encoding, only tables with qualified column references in that scope remain in join scope; "
            "name every required table with explicit table.column in the appropriate prose field.",
            "Omitting a junction_table from tables is correct when its columns appear in prose; the tables list alone never keeps a table in join scope.",
            "When membership or existence requires link_table or junction_table, name junction_table.column or bridge_table.column in select prose "
            "(not only join-equality narration in filter prose, which downstream steps strip).",
            "Per-entity breakdown phrasing maps to group_by, not row-level DISTINCT deduplication.",
            "Existence or membership conditions belong in filter prose with binding columns named.",
            "Time-window and duration comparisons must name bound date column(s) in filter prose and preserve unit and amount in that prose.",
            "For cte_steps, tables lists base schema tables and prior cte_steps names; there is no depends_on field.",
            "When schema shows role temporal with type integer on a column, treat it as a day-count duration for elapsed-time comparisons, not a calendar date.",
        ],
        "supported_capabilities": list(GROUND_SUPPORTED_CAPABILITIES),
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


def logical_intent_to_serialisable(logical: LogicalIntent) -> dict[str, Any]:
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
        "cte_steps": [
            {
                "name": c.name,
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


def _propagate_interpret_schema_invalid_flag(intent: RuntimeIntent, plan: InterpretPlan) -> RuntimeIntent:
    """Overwrite runtime schema_invalid with the Interpret-only flag after structural parsing."""
    return replace(intent, schema_invalid=plan.schema_invalid)


def finalize_planner_schema_invalid_flag(
    intent: RuntimeIntent,
    plan: InterpretPlan,
    schema_graph: SchemaGraph,
) -> RuntimeIntent:
    """Copy Interpret ``schema_invalid`` onto the runtime intent after structural parsing."""
    del schema_graph
    return replace(intent, schema_invalid=plan.schema_invalid)


def _build_intent_compose_prompt(logical: LogicalIntent, structural_payload: str) -> str:
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
    parse_filter_ops.extend(sorted(extra_filter_ops_for_engine()))
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
    structural_obj = json.loads(structural_payload)
    body: dict[str, Any] = {
        "task": (
            "Encode logical_intent into runtime intent JSON conforming to field_specifications and output_format. "
            "You are a structural encoder only; do not re-plan. Use only identifiers in "
            "structural_schema_for_chosen_tables. Output ONLY valid JSON."
        ),
        "logical_intent": logical_intent_to_serialisable(logical),
        "logical_to_ir_field_map": {
            "select": "select_cols",
            "filter": "filters_param",
            "group_by": "group_by_cols",
            "having": "having_param",
            "order_by": "order_by_cols",
            "limit": "limit",
            "window": "window_registry",
            "case": "case_registry",
            "cte_steps": "cte_steps",
        },
        "cte_tables_encoding": (
            "Each runtime cte_steps row uses cte_name from logical name and tables from logical tables. "
            "Logical tables may list base schema tables and prior logical cte_steps names."
        ),
        "structural_schema_for_chosen_tables": structural_obj,
        "structural_json_keys": intent_prompt_structural_index(),
        "critical_rules": list(INTENT_CRITICAL_RULES),
        "parse_rules_append": list(INTENT_PARSE_RULES_APPEND),
        "field_specifications": dict(RuntimeIntent.PROMPT_FIELD_SPEC.items()),
        "output_format": RuntimeIntent.prompt_example_dict(),
        "supported_capabilities": list(COMPOSE_SUPPORTED_CAPABILITIES),
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
            "Translate each populated logical_intent prose field into its mapped IR slot only.",
            "Planner having prose maps to having_param on the aggregated source, not outer filters_param.",
            "Do not invent filters, grouping, tables, or columns absent from logical_intent prose fields.",
            "Join path discovery is downstream; do not author join predicates in filters_param or having_param.",
        ],
        "format_structural_guidance": list(FORMAT_STRUCTURAL_GUIDANCE),
        "compose_priority_rules": [
            "Translate each populated clause prose field mechanically into its IR slot.",
            "Use nl_phrase_mappings and operator_reference only to choose operators and aggregates.",
            "Do not re-interpret semantics; logical_intent is the full contract because the question is not in this payload.",
        ],
    }
    return stable_json(body)


def resolve_repair_instruction(issue: IntentIssue) -> str:
    """Return a targeted fix instruction for a semantic issue."""
    return REPAIR_INSTRUCTIONS.get(issue.category.value, issue.message)


def classify_schema_error(error_message: str) -> FailureCategory:
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


def build_intent_semantic_repair_prompt(
    question: str,
    current_intent_json: str,
    errors: list[IntentIssue],
    warnings: list[IntentIssue],
    schema_literal_json: str,
    *,
    prior_question_feedback: list[dict[str, str]] | None = None,
) -> str:
    """Build a user-prompt for the LLM to repair semantic issues in a. parsed intent. Errors are presented as errors_to_fix with targeted fix instructions sourced from REPAIR_INSTRUCTIONS and warnings are presented as non- binding suggestions. Args: question: Original natural language question. current_intent_json: JSON string of the current flawed parsed intent. errors: IntentIssue objects with severity equal to "error". warnings: IntentIssue objects with severity equal to "warning". schema_literal_json: Compact JSON schema literal for the LLM context. prior_question_feedback: Optional rows scoped to this question and schema hash. Returns: JSON-formatted prompt string ready to send as the user message."""
    errors_to_fix: list[dict[str, str]] = []
    for err in errors:
        errors_to_fix.append(
            {
                "category": err.category,
                "issue": err.message,
                "fix": resolve_repair_instruction(err),
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


def build_intent_format_repair_prompt(question: str, raw_response: str, parse_error: str) -> str:
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


def build_intent_parse_prompt(
    question: str,
    schema_literal_json: str,
    table_list: list[str],
    prior_question_feedback: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """Build ``(system, user_json)`` strings for the initial intent LLM. call. Args: question: User question. schema_literal_json: Compact JSON schema literal for prompts. table_list: Allowed table names. prior_question_feedback: Optional summarized failures for this question from persisted memory and the current attempt. Returns: System prompt and stable-JSON user payload."""
    system = (
        "You are a deterministic intent parser for text-to-SQL. "
        "Output ONLY valid JSON that matches the required format. "
        "Identical inputs must produce identical outputs. The "
        "natural_language field is a single short sentence in plain English that describes the structured intent you just produced — what the query computes. Read your own SELECT expressions, FROM tables, filters, grouping, and ordering, then describe that result. "
        'Aggregation words like "count", "sum", "average", "minimum", "maximum", "total" are encouraged whenever the intent uses them. Use the same domain nouns as the schema (table and column names rendered in plain English). Do not mention SQL syntax tokens (JOIN, GROUP BY, WHERE, ORDER BY, LIMIT, etc.). Reuse the user\'s domain words when they correctly name the intent\'s tables/columns/aggregations; only avoid copying the question verbatim when the question contains filler words, polite phrasing, or wording the structured intent does not actually reflect. '
        + INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES
        + " "
        + INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE
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

    parse_filter_ops.extend(sorted(extra_filter_ops_for_engine()))
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
            "qualified_column_reference": "table.column",
            "note": (
                "table.column, other_table.other_column, and bare table names such as table or other_table "
                "appear only to illustrate JSON shape. Replace every expression with real identifiers "
                "from schema_summary or allowed_tables."
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
        "logical_decomposition_guidance": list(LOGICAL_DECOMPOSITION_GUIDANCE),
        "format_structural_guidance": list(FORMAT_STRUCTURAL_GUIDANCE),
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
    max_retries: int = PolicyConfig.MAX_ASK_COMPOSE_REPAIRS,
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
        if candidate is None or runtime_intent_has_instructional_placeholders(candidate):
            return False
        return not runtime_intent_has_refusal_natural_language(candidate)

    llm_calls = 0
    parse_detail: list[str] = []
    intent = parse_intent_response(raw, question, parse_detail_out=parse_detail)
    if _acceptable(intent):
        assert intent is not None
        accepted_intent = intent
        if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled():
            pipeline_trace(
                "intent_after_parse_intent_response.initial",
                lambda: stable_json(accepted_intent.to_dict()),
            )
        return intent, llm_calls
    for _ in range(max_retries):
        if intent is None:
            parse_error = parse_detail[-1] if parse_detail else "JSON parse failed"
        elif runtime_intent_has_refusal_natural_language(intent):
            parse_error = NATURAL_LANGUAGE_REFUSAL_PARSE_ERROR
        else:
            parse_error = INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR
        parse_detail.clear()
        repair_prompt = build_intent_format_repair_prompt(question, raw, parse_error)
        raw = llm_chat(system, repair_prompt, task="intent")
        llm_calls += 1
        intent = parse_intent_response(raw, question, parse_detail_out=parse_detail)
        if _acceptable(intent):
            assert intent is not None
            accepted_intent = intent
            if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled():
                pipeline_trace(
                    "intent_after_parse_intent_response.format_repair",
                    stable_json(accepted_intent.to_dict()),
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


def _detect_oscillation_strings(history: list[frozenset[str]]) -> bool:
    """Return True on repeated identical string-error signatures or A-B- A-B alternation."""
    if len(history) >= 2 and history[-1] == history[-2]:
        return True
    if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
        return True
    return False


def _detect_oscillation(history: list[frozenset[tuple[FailureCategory, str]]]) -> bool:
    """Return True on repeated identical signatures or A-B-A-B. alternation. Args: history: Recent signature frozensets, oldest first. Returns: True when repair should stop for oscillation."""
    if len(history) >= 2 and history[-1] == history[-2]:
        return True
    if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
        return True
    return False


def _derive_window_registry_output_alias(
    step: WindowRegistryStep,
    *,
    explicit_alias: str | None,
    cte_ordinal: int,
    registry_id: str,
) -> str:
    """Derive a CTE output alias for a ``window_registry`` select token."""
    if explicit_alias:
        return explicit_alias
    ws = step.window_spec
    fn = (ws.function or "").strip().lower()
    if fn in {"row_number", "rank", "dense_rank", "ntile"}:
        return f"cte{cte_ordinal}_{registry_id.lower()}"
    arg = ws.argument
    base = ""
    if arg is not None:
        col = arg.primary_column or ""
        base = "*" if col == "*" else col.split(".")[-1]
    if fn == "count":
        if base == "*":
            return "row_count"
        if base:
            return f"count_{base}"
    elif fn and base and base != "*":
        return f"{fn}_{base}"
    return f"cte{cte_ordinal}_{registry_id.lower()}"


def _derive_cte_output_columns_resolved(
    cte: RuntimeCteStep,
    *,
    cte_ordinal: int,
) -> list[str]:
    """Derive CTE ``output_columns`` including window-registry select tokens."""
    wr_by = {s.registry_id: s for s in (cte.window_registry or [])}
    old_oc = list(cte.output_columns or [])
    derived: list[str] = []
    seen: dict[str, int] = {}
    expr_counter = 0
    for i, sc in enumerate(cte.select_cols or []):
        explicit_raw = old_oc[i].strip() if i < len(old_oc) and old_oc[i] else ""
        explicit = explicit_raw.lower() if explicit_raw else None
        if explicit and re.fullmatch(REGISTRY_TOKEN_PATTERN, explicit):
            explicit = None
        rid = expr_registry_ref(sc.expr) or ""
        if rid.startswith("w"):
            step = wr_by.get(rid)
            if step is not None:
                name = _derive_window_registry_output_alias(
                    step,
                    explicit_alias=explicit,
                    cte_ordinal=cte_ordinal,
                    registry_id=rid,
                )
            else:
                name = explicit or f"cte{cte_ordinal}_{rid.lower()}"
        elif rid.startswith("c"):
            name = explicit or f"cte{cte_ordinal}_{rid.lower()}"
        else:
            single = derive_cte_output_columns([sc], cte_ordinal=cte_ordinal)
            if single:
                name = single[0]
            else:
                expr_counter += 1
                name = f"expr{expr_counter}"
        kind = classify_cte_expr(sc.expr)
        if kind != "passthrough" and not rid.startswith(("w", "c")):
            name = name.lower()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        derived.append(name)
    return derived


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
        new_oc = _derive_cte_output_columns_resolved(cte, cte_ordinal=idx + 1)
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
            window_registry=rename_window_registry_steps(cte.window_registry, alias_map),
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
        window_registry=rename_window_registry_steps(intent.window_registry, alias_map),
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
        pipeline_trace(
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
        """Translate ``distinct_select_index`` after select-col reordering. The index points to the column that originally carried ``DISTINCT``; after sorting we locate the same identity in the new list and return its new position. Returns ``-1`` when no DISTINCT was set or the original column cannot be located."""
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
        "dedup_extract_year_vs_column_literal",
        intent,
        lambda x: apply_filters_to_main_and_ctes(x, dedup_extract_year_vs_column_literal),
    )
    intent = _deterministic_repair_step(
        "repair_intent_placeholder_tokens",
        intent,
        lambda x: repair_intent_placeholder_tokens(x, schema_graph),
    )
    intent = _deterministic_repair_step("normalize_count_star", intent, normalize_count_star)
    intent = _deterministic_repair_step(
        "promote_temporal_keyword_rhs",
        intent,
        promote_temporal_keyword_rhs,
    )
    intent = _deterministic_repair_step("dedup_value_vs_right_expr", intent, dedup_value_vs_right_expr)
    intent = _deterministic_repair_step(
        "qualify_count_star_mulgroups",
        intent,
        lambda x: qualify_count_star_mulgroups(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "lift_distinct_select_from_raw_sql",
        intent,
        lambda x: lift_distinct_select_from_raw_sql(x, schema_graph),
    )
    intent = _deterministic_repair_step("canonicalize_registry_ids", intent, canonicalize_registry_ids)
    intent = _deterministic_repair_step("reorder_cte_steps_by_dag", intent, reorder_cte_steps_by_dag)
    intent = _deterministic_repair_step("normalize_cte_names", intent, normalize_cte_names)
    intent = _deterministic_repair_step(
        "normalize_cte_output_aliases",
        intent,
        lambda x: _normalize_cte_output_aliases(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "rewrite_main_query_refs_to_final_cte_columns",
        intent,
        rewrite_main_query_refs_to_final_cte_columns,
    )
    intent = _deterministic_repair_step(
        "ensure_cte_output_columns_exposure",
        intent,
        ensure_cte_output_columns_exposure,
    )
    intent = _deterministic_repair_step("qualify_cte_output_columns", intent, qualify_cte_output_columns)
    intent = _deterministic_repair_step(
        "derive_tables_from_intent",
        intent,
        lambda x: reconcile_tables(x),
    )
    intent = _deterministic_repair_step(
        "expand_shared_pk_tables_for_refs",
        intent,
        lambda x: expand_shared_pk_tables_for_refs(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "sanitize_table_names",
        intent,
        lambda x: sanitize_table_names(x, schema_graph),
    )
    intent = _deterministic_repair_step("replace_unknown_scalar_funcs", intent, replace_unknown_scalar_funcs)
    intent = _deterministic_repair_step(
        "enforce_grain_consistency",
        intent,
        lambda x: enforce_grain_consistency(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "repair_window_partition_group_by_alignment",
        intent,
        lambda x: repair_window_partition_group_by_alignment(x, schema_graph),
    )
    intent = _deterministic_repair_step(
        "strip_redundant_identifier_group_by",
        intent,
        lambda x: strip_redundant_identifier_group_by(x, schema_graph),
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
    intent = _deterministic_repair_step(
        "promote_date_subtraction_to_date_diff",
        intent,
        promote_date_subtraction_to_date_diff,
    )
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
        lambda x: repair_array_filters_intent(x, schema_graph, natural_language),
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
    Resolve columns, wire CTEs, assign params, prune tables; fail if.

    params missing. Args: intent: Intent that passed semantic

    validation. schema_graph: Schema graph. question: User question.

    Returns:

        ``(ready_intent_or_none, resolution_issues)``; issues include
        ``column_ambiguous`` errors when bare names span multiple tables.
        ``None`` intent when required param values are missing.
    """
    all_cols = collect_column_refs_for_post_processing(intent)
    column_map, col_issues = resolve_column_map(all_cols, schema_graph, intent.tables or [])
    intent = replace(intent, column_map=column_map)

    if intent.cte_steps:
        intent = replace(intent, cte_steps=resolve_cte_column_maps(intent.cte_steps))

    if intent.cte_steps:
        intent = qualify_cte_output_columns(intent)

    intent = rewrite_cte_output_refs_to_aliases(intent)
    intent = resolve_window_registry_filter_rhs(intent)

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
                    if cond.param_key:
                        cte_pks.add(cond.param_key)
                    if cond.param_key_hi:
                        cte_pks.add(cond.param_key_hi)
            cte_pv = {k: v for k, v in all_pv.items() if k in cte_pks}
            new_cte_steps.append(replace(cte, param_values=cte_pv))
        intent = replace(intent, cte_steps=new_cte_steps)

    intent = ensure_scalar_func_defaults(intent)
    intent = apply_aggregatability_gate(intent, schema_graph)
    intent = extract_structural_params(intent)
    intent = simplify_exprs(intent)

    if diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled():
        pipeline_trace("intent_after_post_processing", lambda: stable_json(intent.to_dict()))

    return intent, col_issues


def _align_runtime_tables_to_planner(
    runtime: RuntimeIntent,
    logical: LogicalIntent,
) -> RuntimeIntent:
    """Overwrite runtime main and matching-CTE tables with the. planner's. authoritative lists. The planner is the source of truth for which tables the query needs, including join bridges. Encoder drift on the tables field is silently corrected here so the downstream deterministic pipeline (reconcile_tables, JOIN engine) sees consistent state. Encoder column hallucinations remain caught by check_qualified_refs_exist. Args: runtime: Encoder output runtime intent before deterministic repair. logical: Planner output logical intent. Returns: Runtime intent whose main tables list matches the planner's tables list. For every runtime cte_steps entry whose cte_name matches a planner CteIntent.name, that CTE's tables list is aligned with the planner's CTE tables list. Unmatched CTE steps are returned unchanged."""
    aligned_main = list(logical.tables)
    planner_cte_by_name = {s.name: s for s in (logical.cte_steps or ()) if s.name}
    planner_cte_names = [s.name for s in (logical.cte_steps or ()) if s.name]
    new_cte_steps: list[RuntimeCteStep] = []
    for cte in runtime.cte_steps or []:
        cname = (cte.cte_name or "").strip()
        if cname and cname in planner_cte_by_name:
            planned = planner_cte_by_name[cname]
            new_cte_steps.append(replace(cte, tables=list(planned.tables)))
        else:
            new_cte_steps.append(cte)
    return replace(
        runtime,
        tables=aligned_main,
        cte_steps=new_cte_steps,
        planner_cte_names=planner_cte_names,
    )


def _base_tables_from_prose_text(text: str) -> set[str]:
    """Return base table names from qualified ``table.column`` tokens in planner prose."""
    if not text:
        return set()
    tables: set[str] = set()
    for match in EXPR_TABLE_COLUMN_REF_RE.finditer(text):
        token = match.group(0)
        parts = token.split(".", 1)
        if len(parts) != 2:
            continue
        tbl, col = parts
        if IDENTIFIER_RE.match(tbl) and IDENTIFIER_RE.match(col):
            tables.add(tbl)
    return tables


def _base_tables_from_logical_prose(logical: LogicalIntent) -> set[str]:
    """Collect base table names referenced as qualified column tokens across logical prose fields."""
    names: set[str] = set()
    for field in PLANNER_PROSE_FIELDS:
        text = getattr(logical, field, "") or ""
        if isinstance(text, str):
            names.update(_base_tables_from_prose_text(text))
    for step in logical.cte_steps:
        for field in PLANNER_PROSE_FIELDS:
            text = getattr(step, field, "") or ""
            if isinstance(text, str):
                names.update(_base_tables_from_prose_text(text))
    return names


def _structural_tables_for_logical(logical: LogicalIntent) -> tuple[str, ...]:
    """Return sorted union of planner table lists and base tables named as qualified column tokens in prose."""
    names: set[str] = set(logical.tables)
    for step in logical.cte_steps:
        names.update(step.tables)
    names.update(_base_tables_from_logical_prose(logical))
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


def _collect_post_compose_validation_issues(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
    post_resolution_issues: list[IntentIssue],
    logical: LogicalIntent | None = None,
) -> list[IntentIssue]:
    """
    Aggregate schema-ref strings, semantic validation, and post-

    resolution issues after the Compose phase.

    Used by :func:`full_intent_parse` to route Ground versus Compose

    retries before the schema and semantic repair loop. Args: intent:

    Post-processed runtime intent candidate. schema_graph: Active schema

    graph. post_resolution_issues: Column resolution or binding issues

    from post-processing. logical: Planner intent when available for

    table fidelity, numeric coverage source, and literal attribution.

    Returns: Combined issues including inferred ``responsible_stage``
        where applicable.
    """
    issues: list[IntentIssue] = list(post_resolution_issues)
    _, schema_errors = check_qualified_refs_exist(intent, schema_graph)
    for idx, err in enumerate(schema_errors):
        issues.append(
            make_intent_issue(
                issue_id=f"schema_ref_post_compose_{idx}",
                category=classify_schema_error(err),
                severity="error",
                message=err,
                context={},
                responsible_stage="compose",
            )
        )
    vr = validate_semantics(
        intent,
        schema_graph,
        post_binding=True,
        numeric_coverage_logical=logical,
    )
    if logical is not None:
        issues.extend(attribute_post_compose_issue(iss, logical) for iss in vr.issues)
    else:
        issues.extend(vr.issues)
    return issues


def apply_runtime_post_processing_lite(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
    *,
    question_fallback: str = "",
) -> tuple[RuntimeIntent | None, list[IntentIssue]]:
    """Apply deterministic column resolution and normalization from. :func:`_apply_post_processing`. Args: intent: Runtime intent after deterministic structural repairs. schema_graph: Active schema graph. question_fallback: Text substituted when ``intent.natural_language`` is empty. Returns: Same contract as :func:`_apply_post_processing`: ``(intent_or_none, column_issues)``."""
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
    visible_objects: frozenset[str] | None = None,
    allowed_columns: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    deny_columns: frozenset[str] | None = None,
    description_overlay: dict[str, Any] | None = None,
) -> tuple[RuntimeIntent | None, list[str], int, InterpretPlan | None]:
    """Re-run a full intent parse once after repair exhaustion, bounded. by *budget*. When *reason* is in :data:`PolicyConfig.SEMANTIC_RESTART_REASONS` and in-memory semantic error rows exist, one ``VALIDATION_FAILURE`` summary is persisted before attempting the restart. Decrements:attr:`RestartBudget.fresh_restarts_left` and recurses into :func:`full_intent_parse` with the same shared budget so a nested restart cannot occur. Args: question: Normalised user question text. schema_graph: Schema graph for validation and table listing. max_retries: JSON format-repair attempts for the new parse. llm_calls: LLM calls already spent before this restart. store: Template store; optional validation summaries are persisted at the restart boundary. in_turn_summaries: Current-turn feedback rows not yet written to the store. budget: Shared :class:`RestartBudget` controlling how many restarts remain. reason: Short label describing which exit branch triggered the restart attempt. last_intent: Most recent intent from the repair loop, if any. Returns: ``(intent_or_none, warnings, new_llm_call_total)``."""
    tpl = get_templates_module()
    should_persist = (
        store is not None
        and last_intent is not None
        and reason in PolicyConfig.SEMANTIC_RESTART_REASONS
        and bool(in_turn_summaries)
    )
    if should_persist and persist_template_learning and tpl is not None:
        flat_errs: list[str] = []
        for row in in_turn_summaries:
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
            sql=render_feedback_sql(last_intent, schema_graph) if last_intent is not None else None,
        )
        tpl.record_question_feedback(store, question, ent)
        tpl.save_template_store(store)
    if budget.fresh_restarts_left <= 0:
        debug(
            f"[intent_process._attempt_fresh_restart] fresh restart denied (reason={reason}, "
            f"fresh_restarts_left=0); terminating."
        )
        return None, [], llm_calls, None
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
    intent, warns, inner, plan = full_intent_parse(
        question,
        schema_graph,
        max_retries=max_retries,
        store=store,
        in_turn_seed=[],
        budget=budget,
        prior_user_corrections=prior_user_corrections,
        persist_template_learning=persist_template_learning,
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
        deny_objects=deny_objects,
        deny_columns=deny_columns,
        description_overlay=description_overlay,
    )
    return intent, warns, llm_calls + inner, plan


def invoke_intent_parse_with_hints(
    question: str,
    schema_graph: SchemaGraph,
    *,
    max_retries: int = PolicyConfig.MAX_ASK_COMPOSE_REPAIRS,
    store: dict[str, Any] | None = None,
    in_turn_seed: list[dict[str, str]] | None = None,
    extra_user_feedback: list[str] | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    budget: RestartBudget | None = None,
    persist_template_learning: bool = True,
    visible_objects: frozenset[str] | None = None,
    allowed_columns: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    deny_columns: frozenset[str] | None = None,
    description_overlay: dict[str, Any] | None = None,
) -> tuple[RuntimeIntent | None, list[str], int, InterpretPlan | None]:
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
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
        deny_objects=deny_objects,
        deny_columns=deny_columns,
        description_overlay=description_overlay,
    )


def parse_intent_for_question(
    question: str,
    schema_graph: SchemaGraph,
    *,
    max_retries: int = PolicyConfig.MAX_ASK_COMPOSE_REPAIRS,
    store: dict[str, Any] | None = None,
    in_turn_seed: list[dict[str, str]] | None = None,
    extra_user_feedback: list[str] | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    budget: RestartBudget | None = None,
    persist_template_learning: bool = True,
    visible_objects: frozenset[str] | None = None,
    allowed_columns: frozenset[str] | None = None,
) -> tuple[RuntimeIntent | None, list[str], int, InterpretPlan | None]:
    """Parse one question into a :class:`RuntimeIntent` (alias of :func:`invoke_intent_parse_with_hints`)."""
    return invoke_intent_parse_with_hints(
        question,
        schema_graph,
        max_retries=max_retries,
        store=store,
        in_turn_seed=in_turn_seed,
        extra_user_feedback=extra_user_feedback,
        prior_user_corrections=prior_user_corrections,
        budget=budget,
        persist_template_learning=persist_template_learning,
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
    )


def full_intent_parse(
    question: str,
    schema_graph: SchemaGraph,
    *,
    max_retries: int = PolicyConfig.MAX_ASK_COMPOSE_REPAIRS,
    store: dict[str, Any] | None = None,
    in_turn_seed: list[dict[str, str]] | None = None,
    extra_user_feedback: list[str] | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    budget: RestartBudget | None = None,
    persist_template_learning: bool = True,
    visible_objects: frozenset[str] | None = None,
    allowed_columns: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    deny_columns: frozenset[str] | None = None,
    description_overlay: dict[str, Any] | None = None,
) -> tuple[RuntimeIntent | None, list[str], int, InterpretPlan | None]:
    """End-to-end parse: Interpret, Ground, Compose, then post-compose validation and repair."""
    tpl = get_templates_module()
    if budget is None:
        budget = RestartBudget.default()
    table_filter, column_filter = schema_graph._resolve_payload_filters(
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
        deny_objects=deny_objects,
        deny_columns=deny_columns,
    )
    if column_filter:
        table_list = sorted(table_filter if table_filter else schema_graph.tables.keys())
    elif table_filter:
        table_list = sorted(table_filter)
    else:
        table_list = sorted(schema_graph.tables.keys())
    interpret_payload = schema_graph.schema_payload_interpret(
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
        deny_objects=deny_objects,
        deny_columns=deny_columns,
        description_overlay=description_overlay,
    )
    ground_payload = schema_graph.schema_payload_ground(
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
        deny_objects=deny_objects,
        deny_columns=deny_columns,
        description_overlay=description_overlay,
    )
    llm_calls = 0
    seed_rows = [dict(r) for r in (in_turn_seed or []) if isinstance(r, dict)]
    if extra_user_feedback:
        for line in extra_user_feedback:
            t = (line or "").strip()
            if t:
                seed_rows.append({"summary": t, "source": "user_refinement"})
    persisted: list[dict[str, str]] = []
    if store is not None and tpl is not None:
        persisted = tpl.collect_question_feedback_for_prompt(store, question, schema_graph.schema_graph_id)
    merged_feedback = dedupe_prior_question_feedback_rows(seed_rows + persisted)
    prior_fb_text = serialized_prior_feedback_rows(merged_feedback)
    answer_style_text = stable_json(list(LOGICAL_DECOMPOSITION_GUIDANCE))
    system_compose = INTENT_COMPOSE_SYSTEM
    prior_grounding_failures: tuple[str, ...] = ()
    prior_attempt_failures: tuple[str, ...] = ()
    max_a_attempts = PolicyConfig.MAX_ASK_INTERPRET_GROUND_RETRIES + 1
    attempt_a = 0
    interpret_plan: InterpretPlan | None = None
    while attempt_a < max_a_attempts:
        user_interpret = build_intent_interpret_prompt(
            question,
            interpret_payload,
            prior_fb_text,
            prior_user_corrections,
            prior_attempt_failures,
        )
        raw_interpret = llm_chat(INTENT_INTERPRET_SYSTEM, user_interpret, task="intent")
        llm_calls += 1
        debug(f"[{ASK_PHASE_B}] raw_llm_response (attempt {attempt_a + 1}): {raw_interpret}")
        interpret_candidate, interpret_issues = parse_interpret_plan_response(raw_interpret)
        if interpret_candidate is None:
            if attempt_a >= max_a_attempts - 1:
                debug(f"[{ASK_PHASE_B}] exhausted after interpret validation failures")
                intent, warns, calls, plan = _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "interpret_exhausted",
                    None,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                    visible_objects=visible_objects,
                    allowed_columns=allowed_columns,
                    deny_objects=deny_objects,
                    deny_columns=deny_columns,
                    description_overlay=description_overlay,
                )
                return intent, warns, calls, plan
            notify(
                "Interpret retry after plan validation failures.",
                stage="intent",
                code=DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY,
                details=(("attempt", str(attempt_a + 1)),),
            )
            prior_attempt_failures = prior_attempt_failures + tuple(
                _issue_to_planner_hint(iss) for iss in interpret_issues
            )
            attempt_a += 1
            continue

        interpret_plan = interpret_candidate
        pipeline_trace(
            ASK_PHASE_B,
            stable_json(
                {
                    "approach": interpret_plan.approach,
                    "grounding": [{"ref": r, "used_for": u} for r, u in interpret_plan.grounding],
                }
            ),
        )
        user_ground = build_intent_ground_prompt(
            question,
            interpret_plan,
            ground_payload,
            prior_fb_text,
            answer_style_text,
            prior_user_corrections,
            prior_grounding_failures,
        )
        raw_ground = llm_chat(INTENT_GROUND_SYSTEM, user_ground, task="intent")
        llm_calls += 1
        debug(f"[{ASK_PHASE_C}] raw_llm_response (attempt {attempt_a + 1}): {raw_ground}")
        logical_candidate, logical_issues = parse_logical_intent_response(raw_ground, schema_graph)
        if logical_candidate is None:
            if attempt_a >= max_a_attempts - 1:
                debug(f"[{ASK_PHASE_C}] exhausted after ground validation failures")
                intent, warns, calls, plan = _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "ground_exhausted",
                    None,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                    visible_objects=visible_objects,
                    allowed_columns=allowed_columns,
                    deny_objects=deny_objects,
                    deny_columns=deny_columns,
                    description_overlay=description_overlay,
                )
                return intent, warns, calls, plan
            notify(
                "Ground retry after schema or core-field validation failures.",
                stage="intent",
                code=DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY,
                details=(("attempt", str(attempt_a + 1)),),
            )
            new_hints = tuple(_issue_to_planner_hint(iss) for iss in logical_issues)
            prior_grounding_failures = prior_grounding_failures + new_hints
            prior_attempt_failures = prior_attempt_failures + new_hints
            attempt_a += 1
            continue

        logical = logical_candidate
        structural_json = schema_graph.schema_payload_compose(_structural_tables_for_logical(logical))
        user_compose = _build_intent_compose_prompt(logical, structural_json)
        raw_compose = llm_chat(system_compose, user_compose, task="intent")
        llm_calls += 1
        debug(f"[{ASK_PHASE_D}] raw_llm_response: {raw_compose}")
        intent, fmt_calls = _format_repair_loop(system_compose, raw_compose, question, max_retries)
        llm_calls += fmt_calls

        if not intent:
            debug(f"[{ASK_PHASE_D}] format repair exhausted")
            intent, warns, calls, plan = _attempt_fresh_restart(
                question,
                schema_graph,
                max_retries,
                llm_calls,
                store,
                seed_rows,
                budget,
                "compose_format_exhausted",
                None,
                prior_user_corrections=prior_user_corrections,
                persist_template_learning=persist_template_learning,
                visible_objects=visible_objects,
                allowed_columns=allowed_columns,
                deny_objects=deny_objects,
                deny_columns=deny_columns,
            )
            return intent, warns, calls, plan

        debug(f"[{ASK_PHASE_D}] normalized intent:\n{stable_json(intent.to_dict())}")

        intent = _align_runtime_tables_to_planner(intent, logical)
        intent = _propagate_interpret_schema_invalid_flag(intent, interpret_plan)

        avoid_rows = dedupe_prior_question_feedback_rows(list(seed_rows) + persisted)
        logical_restart = False
        b_repairs_used = 0
        compose_repair_json = schema_graph.schema_payload_compose(intent.tables or table_list)
        while True:
            intent = apply_deterministic_repairs(intent, schema_graph, question)
            result, post_issues = _apply_post_processing(intent, schema_graph, question)
            if result is None:
                debug(f"[{ASK_PHASE_H}] post-processing missing params — terminating")
                return None, [], llm_calls, interpret_plan
            merged_issues = _collect_post_compose_validation_issues(result, schema_graph, post_issues, logical)
            errors = [iss for iss in merged_issues if iss.severity == "error"]
            if not errors:
                intent = result
                break
            if (
                errors
                and not any(iss.responsible_stage == "ground" for iss in errors)
                and all(
                    (iss.issue_id or "").startswith(("non_selectable_filter", "non_selectable_having"))
                    or iss.category == FailureCategory.ACCESS_POLICY
                    for iss in errors
                )
            ):
                debug(f"[{ASK_PHASE_H}] post-compose unfixable policy errors — terminating")
                return None, [], llm_calls, interpret_plan
            if any(iss.responsible_stage == "ground" for iss in errors):
                if attempt_a >= max_a_attempts - 1:
                    intent, warns, calls, plan = _attempt_fresh_restart(
                        question,
                        schema_graph,
                        max_retries,
                        llm_calls,
                        store,
                        seed_rows,
                        budget,
                        "ground_orchestrator_exhausted",
                        None,
                        prior_user_corrections=prior_user_corrections,
                        persist_template_learning=persist_template_learning,
                        visible_objects=visible_objects,
                        allowed_columns=allowed_columns,
                        deny_objects=deny_objects,
                        deny_columns=deny_columns,
                    )
                    return intent, warns, calls, plan
                notify(
                    "Ground retry after phase validation flagged Ground-stage issues.",
                    stage="intent",
                    code=DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY,
                    details=(
                        ("attempt", str(attempt_a + 1)),
                        ("phase", ASK_PHASE_H),
                    ),
                )
                prior_grounding_failures = prior_grounding_failures + tuple(
                    _issue_to_planner_hint(iss) for iss in errors if iss.responsible_stage == "ground"
                )
                logical_restart = True
                break
            if b_repairs_used >= max_retries:
                debug(f"[{ASK_PHASE_E}] post-compose format repair budget exhausted")
                intent, warns, calls, plan = _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "post_compose_format_exhausted",
                    intent,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                    visible_objects=visible_objects,
                    allowed_columns=allowed_columns,
                    deny_objects=deny_objects,
                    deny_columns=deny_columns,
                    description_overlay=description_overlay,
                )
                return intent, warns, calls, plan
            warnings = [iss for iss in merged_issues if iss.severity == "warning"]
            intent_json = stable_json(result.to_prompt_dict())
            debug(f"[{ASK_PHASE_E}] post-compose repair errors: {[(e.category, e.message) for e in errors]}")
            compose_repair_json = schema_graph.schema_payload_compose(result.tables or table_list)
            repair_prompt = build_intent_semantic_repair_prompt(
                question,
                intent_json,
                errors,
                warnings,
                compose_repair_json,
                prior_question_feedback=avoid_rows or None,
            )
            notify(
                "Compose repair LLM invocation.",
                stage="intent",
                code=DIAGNOSTIC_CODE_COMPOSE_REPAIR,
                details=(
                    ("phase", ASK_PHASE_E),
                    ("repair_round", str(b_repairs_used + 1)),
                ),
            )
            rollback_intent = result
            repaired_raw = llm_chat(system_compose, repair_prompt, task="intent")
            llm_calls += 1
            repaired, fmt_rep_calls = _format_repair_loop(system_compose, repaired_raw, question, max_retries)
            llm_calls += fmt_rep_calls
            if not repaired:
                debug(f"[{ASK_PHASE_E}] format repair exhausted after post-compose repair")
                intent, warns, calls, plan = _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "post_compose_format_repair_exhausted",
                    intent,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                    visible_objects=visible_objects,
                    allowed_columns=allowed_columns,
                    deny_objects=deny_objects,
                    deny_columns=deny_columns,
                    description_overlay=description_overlay,
                )
                return intent, warns, calls, plan
            intent = repaired
            intent = _align_runtime_tables_to_planner(intent, logical)
            intent = _propagate_interpret_schema_invalid_flag(intent, interpret_plan)
            if not _runtime_intent_select_cols_have_substance(
                intent
            ) or _runtime_intent_case_registry_has_empty_branches(intent):
                debug(f"[{ASK_PHASE_E}] repair_reverted_empty_select")
                intent = rollback_intent
            b_repairs_used += 1

        if logical_restart:
            attempt_a += 1
            continue

        in_turn_live: list[dict[str, str]] = list(seed_rows)

        repaired_intent, sem_warns, llm_calls, planner_hints = _run_schema_semantic_repair_loop(
            intent=intent,
            question=question,
            system=system_compose,
            schema_graph=schema_graph,
            schema_literal_json=compose_repair_json,
            table_list=table_list,
            max_retries=max_retries,
            llm_calls=llm_calls,
            store=store,
            in_turn_summaries=in_turn_live,
            budget=budget,
            logical=logical,
            interpret_plan=interpret_plan,
            prior_user_corrections=prior_user_corrections,
            persist_template_learning=persist_template_learning,
            visible_objects=visible_objects,
            allowed_columns=allowed_columns,
            deny_objects=deny_objects,
            deny_columns=deny_columns,
        )
        if planner_hints is not None:
            if attempt_a >= max_a_attempts - 1:
                intent, warns, calls, plan = _attempt_fresh_restart(
                    question,
                    schema_graph,
                    max_retries,
                    llm_calls,
                    store,
                    seed_rows,
                    budget,
                    "ground_after_schema_semantic_exhausted",
                    None,
                    prior_user_corrections=prior_user_corrections,
                    persist_template_learning=persist_template_learning,
                    visible_objects=visible_objects,
                    allowed_columns=allowed_columns,
                    deny_objects=deny_objects,
                    deny_columns=deny_columns,
                    description_overlay=description_overlay,
                )
                return intent, warns, calls, plan
            notify(
                "Ground retry after schema/semantic validation surfaced logical-stage issues.",
                stage="intent",
                code=DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY,
                details=(
                    ("attempt", str(attempt_a + 1)),
                    ("phase", ASK_PHASE_G),
                ),
            )
            prior_grounding_failures = prior_grounding_failures + planner_hints
            attempt_a += 1
            continue

        return repaired_intent, sem_warns, llm_calls, interpret_plan

    debug(f"[{ASK_PHASE_C}] outer budget exhausted (unexpected)")
    intent, warns, calls, plan = _attempt_fresh_restart(
        question,
        schema_graph,
        max_retries,
        llm_calls,
        store,
        seed_rows,
        budget,
        "ground_outer_exhausted",
        None,
        prior_user_corrections=prior_user_corrections,
        persist_template_learning=persist_template_learning,
        visible_objects=visible_objects,
        allowed_columns=allowed_columns,
        deny_objects=deny_objects,
        deny_columns=deny_columns,
    )
    return intent, warns, calls, plan


def _runtime_intent_select_cols_have_substance(intent: RuntimeIntent) -> bool:
    """Return False when any select column expression is structurally empty in the main query or a CTE."""

    def scope_ok(cols: list[SelectCol]) -> bool:
        return all(not normalized_expr_is_absent(sc.expr) for sc in cols or [])

    if not scope_ok(intent.select_cols):
        return False
    return all(scope_ok(step.select_cols) for step in intent.cte_steps or [])


def _runtime_intent_case_registry_has_empty_branches(intent: RuntimeIntent) -> bool:
    """Return True when any case-registry step has no branches (main query or a CTE)."""

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
    interpret_plan: InterpretPlan | None = None,
    prior_user_corrections: tuple[str, ...] = (),
    persist_template_learning: bool = True,
    visible_objects: frozenset[str] | None = None,
    allowed_columns: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    deny_columns: frozenset[str] | None = None,
) -> tuple[RuntimeIntent | None, list[str], int, tuple[str, ...] | None]:
    """
    Schema + semantic repair loops, post-processing, and post-processing revalidation. Each level performs one initial validation followed by up to :data:`PolicyConfig.MAX_ASK_COMPOSE_REPAIRS` repair attempts (so total = original + ``MAX_ASK_COMPOSE_REPAIRS`` validations). Exhaustion or oscillation funnels into :func:`_attempt_fresh_restart`, which is bounded by the shared *budget*.

    Returns:

        ``(intent, warnings, llm_calls, planner_restart_hints)``. When *planner_restart_hints* is not ``None``, the caller should retry the planner (Interpret/Ground phase) with those hints instead of treating the tuple as success.
    """
    semantic_warnings: list[str] = []
    seen_warning_ids: set[str] = set()
    semantic_error_history: list[frozenset[tuple[FailureCategory, str]]] = []
    in_turn: list[dict[str, str]] = list(in_turn_summaries) if in_turn_summaries is not None else []
    if budget is None:
        budget = RestartBudget.default()
    max_repair = PolicyConfig.MAX_ASK_COMPOSE_REPAIRS
    sem_iterations = max_repair + 1
    schema_iterations = max_repair + 1

    for sem_round in range(sem_iterations):
        debug(f"[{ASK_PHASE_G}] semantic round {sem_round + 1}/{sem_iterations}")
        tpl = get_templates_module()

        persisted_rows: list[dict[str, str]] = []
        if store is not None and tpl is not None:
            persisted_rows = tpl.collect_question_feedback_for_prompt(store, question, schema_graph.schema_graph_id)
        avoid_rows = dedupe_prior_question_feedback_rows(list(in_turn) + persisted_rows)
        intent = apply_deterministic_repairs(intent, schema_graph, question)
        debug(f"[{ASK_PHASE_E}] repairs:\n{stable_json(intent.to_dict())}")

        schema_error_history: list[frozenset[str]] = []
        schema_resolved = False
        for schema_sub in range(schema_iterations):
            intent, schema_errors = check_qualified_refs_exist(intent, schema_graph)
            if not schema_errors:
                debug(f"[{ASK_PHASE_F}] validation passed on sub-round {schema_sub + 1}/{schema_iterations}")
                schema_resolved = True
                break
            debug(f"[{ASK_PHASE_F}] sub-round {schema_sub + 1}/{schema_iterations}: {len(schema_errors)} errors")
            schema_sig = _compute_error_signature_strings(schema_errors)
            schema_error_history.append(schema_sig)
            if _detect_oscillation_strings(schema_error_history):
                debug(f"[{ASK_PHASE_F}] schema oscillation detected — breaking sub-loop")
                break
            if schema_sub >= schema_iterations - 1:
                break
            schema_issues = [
                make_intent_issue(
                    issue_id=f"schema_error_{idx}",
                    category=classify_schema_error(err),
                    severity="error",
                    message=err,
                    responsible_stage="compose",
                )
                for idx, err in enumerate(schema_errors)
            ]
            for iss in schema_issues:
                debug(f"[{ASK_PHASE_F}]   issue_id={iss.issue_id} message={iss.message}")
            intent_before_schema_llm = intent
            intent_json = stable_json(intent.to_prompt_dict())
            debug(f"[{ASK_PHASE_F}] intent being sent to schema repair LLM:\n{intent_json}")
            debug(f"[{ASK_PHASE_F}] errors_to_fix: {[(iss.category, iss.message) for iss in schema_issues]}")
            repair_prompt = build_intent_semantic_repair_prompt(
                question,
                intent_json,
                schema_issues,
                [],
                schema_literal_json,
                prior_question_feedback=avoid_rows or None,
            )
            notify(
                "Compose schema repair LLM invocation.",
                stage="intent",
                code=DIAGNOSTIC_CODE_COMPOSE_REPAIR,
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
                debug(f"[{ASK_PHASE_F}] format repair exhausted after schema repair — terminating")
                fi, fw, fc, _ = _attempt_fresh_restart(
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
            if interpret_plan is not None:
                intent = _propagate_interpret_schema_invalid_flag(intent, interpret_plan)
            if not _runtime_intent_select_cols_have_substance(
                intent
            ) or _runtime_intent_case_registry_has_empty_branches(intent):
                debug(f"[{ASK_PHASE_F}] repair_reverted_empty_select")
                intent = intent_before_schema_llm
            debug(f"[{ASK_PHASE_F}] normalized intent after schema repair:\n{stable_json(intent.to_dict())}")
            intent = apply_deterministic_repairs(intent, schema_graph, question)

        if not schema_resolved:
            debug(f"[{ASK_PHASE_F}] schema errors persist after sub-loop — terminating")
            fi, fw, fc, _ = _attempt_fresh_restart(
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
            validation_result.issues = [attribute_post_compose_issue(i, logical) for i in validation_result.issues]
        debug(f"[{ASK_PHASE_G}] validation completed: issues={len(validation_result.issues)}")

        errors = [iss for iss in validation_result.issues if iss.severity == "error"]
        warnings = [iss for iss in validation_result.issues if iss.severity == "warning"]

        if logical is not None:
            logical_issues = [iss for iss in validation_result.issues if iss.responsible_stage == "ground"]
            if logical_issues:
                hints = tuple(dict.fromkeys(_issue_to_planner_hint(iss) for iss in logical_issues))
                return None, [], llm_calls, hints

        for iss in validation_result.issues:
            debug(f"[{ASK_PHASE_G}]   issue_id={iss.issue_id} message={iss.message}")
        for w in warnings:
            if w.issue_id not in seen_warning_ids:
                seen_warning_ids.add(w.issue_id)
                semantic_warnings.append(w.message)

        if not errors:
            debug(f"[{ASK_PHASE_G}] no semantic errors in round {sem_round + 1}")
            break

        debug(f"[{ASK_PHASE_G}] {len(errors)} errors, {len(warnings)} warnings in round {sem_round + 1}")
        in_turn.append(in_turn_row_from_semantic_errors(errors, schema_graph.effective_structural_hash, intent))

        semantic_sig = _compute_error_signature_issues(errors)
        semantic_error_history.append(semantic_sig)
        if _detect_oscillation(semantic_error_history):
            debug(f"[{ASK_PHASE_G}] semantic oscillation detected — trying fresh restart")
            fi, fw, fc, _ = _attempt_fresh_restart(
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
            debug(f"[{ASK_PHASE_G}] semantic errors persist after max rounds — trying fresh restart")
            fi, fw, fc, _ = _attempt_fresh_restart(
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
        debug(f"[{ASK_PHASE_G}] intent being sent to semantic repair LLM:\n{intent_json}")
        debug(f"[{ASK_PHASE_G}] errors_to_fix: {[(e.category, e.message) for e in errors]}")
        repair_prompt = build_intent_semantic_repair_prompt(
            question,
            intent_json,
            errors,
            warnings,
            schema_literal_json,
            prior_question_feedback=avoid_rows or None,
        )
        notify(
            "Compose semantic repair LLM invocation.",
            stage="intent",
            code=DIAGNOSTIC_CODE_COMPOSE_REPAIR,
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
            debug(f"[{ASK_PHASE_G}] format repair exhausted after semantic repair — trying fresh restart")
            fi, fw, fc, _ = _attempt_fresh_restart(
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
            if interpret_plan is not None:
                intent = _propagate_interpret_schema_invalid_flag(intent, interpret_plan)
        if not _runtime_intent_select_cols_have_substance(intent) or _runtime_intent_case_registry_has_empty_branches(
            intent
        ):
            debug(f"[{ASK_PHASE_G}] repair_reverted_empty_select")
            intent = intent_before_semantic_llm
        debug(f"[{ASK_PHASE_G}] normalized intent after semantic repair:\n{stable_json(intent.to_dict())}")
        pipeline_trace(
            ASK_PHASE_G,
            stable_json(intent.to_dict()),
        )

    result, post_issues = _apply_post_processing(intent, schema_graph, question)
    if result is None:
        return None, [], llm_calls, None
    if any(i.severity == "error" for i in post_issues):
        debug(f"[{ASK_PHASE_H}] post-processing column resolution errors — terminating")
        return None, [], llm_calls, None

    if not _post_processing_revalidation_passes(result, schema_graph):
        debug(f"[{ASK_PHASE_H}] post-processing revalidation soft recovery: attempted")
        recovered = apply_deterministic_repairs(result, schema_graph, question)
        if _post_processing_revalidation_passes(recovered, schema_graph):
            debug(f"[{ASK_PHASE_H}] post-processing revalidation soft recovery: succeeded")
            result = recovered
        else:
            debug(f"[{ASK_PHASE_H}] post-processing revalidation soft recovery: failed")
            debug(f"[{ASK_PHASE_H}] post-processing revalidation failed — terminating")
            return None, [], llm_calls, None

    debug(
        f"[{ASK_PHASE_H}] parsed intent with {len(result.tables or [])} tables, "
        f"{len(result.filters_param or [])} filters, {llm_calls} LLM calls"
    )
    assert result is not None
    final_result = result
    pipeline_trace(
        ASK_PHASE_H,
        lambda: stable_json(final_result.to_dict()),
    )

    if interpret_plan is not None:
        result = finalize_planner_schema_invalid_flag(result, interpret_plan, schema_graph)

    return result, semantic_warnings, llm_calls, None


def _post_processing_revalidation_passes(
    intent: RuntimeIntent,
    schema_graph: SchemaGraph,
) -> bool:
    """Lightweight revalidation after post-processing. Runs ``check_qualified_refs_exist`` and ``validate_semantics`` once with no repair loop or LLM call. A post-processing step (param resolution, CTE alias rewrite, table pruning) should never invalidate what earlier phases proved, but any surfaced error is treated as terminal because there is no safe way to recover at this point without re-entering the full repair pipeline."""
    _, schema_errors = check_qualified_refs_exist(intent, schema_graph)
    if schema_errors:
        debug(f"[intent_process._post_processing_revalidation_passes] schema errors: {len(schema_errors)}")
        for err in schema_errors:
            debug(f"[intent_process._post_processing_revalidation_passes]   {err}")
        return False
    validation_result = validate_semantics(intent, schema_graph, post_binding=True)
    final_errors = [iss for iss in validation_result.issues if iss.severity == "error"]
    if final_errors:
        debug(f"[intent_process._post_processing_revalidation_passes] semantic errors: {len(final_errors)}")
        for iss in final_errors:
            debug(f"[intent_process._post_processing_revalidation_passes]   {iss.category}: {iss.message}")
        return False
    return True


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
    if candidate_intent is not None and union_family_index is not None and intent_key_index is not None:
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


def cte_structural_signature(steps: Sequence[RuntimeCteStep | ConcreteCteStep]) -> list[tuple[str, str]]:
    """Sorted ``(cte_name, body_sig)`` tuples excluding select columns. (union logic). Args: steps: CTE steps. Returns: Sorted signature list."""
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
    """True when tables, grain, limit, filters, group/order/having, and. CTE skeletons match. Args: intent: New runtime intent. concrete: Template's concrete intent. Returns: Whether non-select structure is identical."""
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
    """Compare agg select keys for equality; count symmetric diff of. non-agg keys. Args: intent_cols: Runtime select list. concrete_cols: Template select list. Returns: ``(aggregates_match, non_agg_symmetric_diff_count)``."""
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
    """True if any differing non-agg column sits on a table outside. both. table sets' intersection. Args: intent_cols: Runtime selects. concrete_cols: Template selects. intent_tables: Runtime ``tables``. concrete_tables: Template ``tables``. Returns: Whether the diff spans disjoint table namespaces."""
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


def _select_col_is_plain_column(sc: SelectCol) -> bool:
    """Return True when *sc* is a bare ``table.column`` reference with no transforms. Path 4 widening only inlines select columns that need no expression rebuild — neither aggregates, scalar/inner-scalar functions, coefficients, expression composition, registry window/case references, nor CASE expressions are tolerated."""
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
        if not _select_col_is_plain_column(sc):
            return False
    return True


def resolve_sql_path(
    *,
    matched_template: Template | None,
    cols_changed: bool,
    union_sql_path: GenerationPath | None,
) -> GenerationPath:
    """Resolve the persisted :class:`GenerationPath` for template- scoped. SQL generation. When a template row is in play but no precomputed union path was supplied, infer ``INTENT_DIRECT_MATCH`` versus union widen codes from column-change flags only. Args: matched_template: Accepted template chosen for reuse, if any. cols_changed: Whether merged union columns differ from the template concrete list. union_sql_path: Path from structural union analysis, when already computed. Returns: Canonical path ``1``–``5`` family member; :attr:`GenerationPath.FRESH` when no template."""
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
    distinct_select_index: int = -1,
) -> GenerationPath:
    """Map structural union facts to the canonical. :class:`GenerationPath` for this match. Args: cols_changed: Whether merged union column keys differ from the template concrete list. select_equal: Whether aggregate rules hold with zero symmetric non- agg diff. delta: Non-aggregated key-set relationship between runtime and concrete selects. Returns: ``INTENT_DIRECT_MATCH``, ``RUNTIME_SUBSET_TEMPLATE_WIDE``, or union widen codes ``4.1`` / ``4.2``."""
    if distinct_select_index >= 0 and delta is UnionSelectColumnDelta.TEMPLATE_ONLY_EXTRA:
        return GenerationPath.INTENT_DIRECT_MATCH
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
    """Run paired union reconcilers until a full pass removes no. template ids. Args: templates: Accepted template id map (mutated in place). max_iterations: Safety cap to avoid infinite loops on pathological stores. template_store_view: When a :class:`~aetherdialect._templates.TemplateStoreView` is supplied, reconcilers read ``union_family_index`` from its header indexes; after each pass the view's matcher indexes are refreshed from the live template bodies. Returns: Count of template ids removed across all iterations."""
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
        if is_template_store_view(template_store_view):
            refresh_template_store_indexes_for_view(
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


def _join_runtime_intent_has_join_fingerprint(intent: RuntimeIntent) -> bool:
    """Return True when the runtime intent carries at least one non- empty join path signature layer."""
    if intent.chosen_join_path_signature:
        return True
    return any(bool(s.chosen_join_path_signature) for s in (intent.cte_steps or []))


def _join_runtime_matches_template_concrete(intent: RuntimeIntent, tmpl: Template) -> bool:
    """Return True when runtime join layers match the template concrete fingerprint. When the runtime intent has no join fingerprint yet, returns True so callers can defer gating."""
    if not _join_runtime_intent_has_join_fingerprint(intent):
        return True
    return join_path_key_runtime(intent) == join_path_key_concrete(tmpl.intent_signature)


def union_runtime_concrete_compatibility(
    intent: RuntimeIntent,
    concrete: ConcreteIntent,
) -> tuple[list[SelectCol], bool, UnionSelectColumnDelta, int] | None:
    """Return union columns, column-change flag, merge case, and non-agg diff when structural union rules pass. Does not evaluate template trust; callers apply trust when comparing accepted templates."""
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
        distinct_select_index=intent.distinct_select_index,
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
    """Compare *intent* to a stored concrete signature and SQL fingerprint without template trust gating. Union eligibility matches accepted-template rules except ``trust_level`` is not consulted."""
    row = union_runtime_concrete_compatibility(intent, concrete)
    return _structural_compare_from_union_row(intent, concrete, sql_fp_val, row, mode=mode)


def structural_compare(
    intent: RuntimeIntent,
    tmpl: Template,
    *,
    mode: Literal["full", "warmup_gold_store_check"] = "full",
) -> StructuralCompareResult:
    """Compare *intent* to *tmpl* for body key, join path, union eligibility, and ``union_sql_path``. *mode* ``full`` attaches ``similarity_score`` via :func:`intent_similarity`; ``warmup_gold_store_check`` omits that score."""
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
    """Merge accepted templates that share the same warmup. ``template_instance_key`` into one row. Args: templates: Live accepted-template map (mutated in place). union_family_index: Optional inverted union-family index (body keys and ``body|join`` keys). template_store_view: When *union_family_index* is omitted, read persisted ``union_family_index`` from this view's header indexes. Returns: Template ids removed after merging value history into the lexicographically smallest keeper id."""
    removed: list[str] = []

    def _merge_instance_group(tids_sorted: list[str]) -> None:
        nonlocal removed
        if len(tids_sorted) <= 1:
            return
        keeper_id = tids_sorted[0]
        keeper = templates.get(keeper_id)
        if keeper is None:
            return
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
    if not ufi and is_template_store_view(template_store_view):
        indexes = getattr(template_store_view, "_indexes", None)
        if isinstance(indexes, dict):
            raw = indexes.get(TEMPLATE_UNION_FAMILY_INDEX_KEY)
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
                tmpl = templates.get(tid)
                if tmpl is None:
                    continue
                inst = template_instance_key_from_parts(
                    bk,
                    join_path_key_concrete(tmpl.intent_signature),
                    tmpl.sql_fp,
                )
                by_inst[inst].append(tid)
            for _inst, tids in by_inst.items():
                _merge_instance_group(sorted(tids))
        return removed

    by_inst_body: dict[str, list[str]] = defaultdict(list)
    for tid, tmpl in list(templates.items()):
        inst = template_instance_key_from_parts(
            body_similarity_key_for_concrete(tmpl.intent_signature),
            join_path_key_concrete(tmpl.intent_signature),
            tmpl.sql_fp,
        )
        by_inst_body[inst].append(tid)
    for _inst, tids in by_inst_body.items():
        _merge_instance_group(sorted(tids))
    return removed


class _UnionMatchCandidate(NamedTuple):
    """One accepted template row eligible for union-style reuse with resolved column metadata."""

    template: Template
    union_cols: list[SelectCol]
    cols_changed: bool
    union_sql_path: GenerationPath
    non_agg_symmetric_diff: int


def list_union_match_candidates(
    intent: RuntimeIntent,
    templates: dict[str, Template],
) -> list[_UnionMatchCandidate]:
    """List every trusted template that passes structural union gates for *intent*. Used for paths ``3`` and ``4.x`` so the join phase can pick among templates that differ only in stored join fingerprints."""

    def _collect() -> list[_UnionMatchCandidate]:
        rows: list[_UnionMatchCandidate] = []
        for tmpl in templates.values():
            cr = structural_compare(intent, tmpl, mode="full")
            if not cr.union_eligible or cr.union_sql_path is None:
                continue
            rows.append(
                _UnionMatchCandidate(
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
    candidates: Sequence[_UnionMatchCandidate],
) -> _UnionMatchCandidate | None:
    """Choose the union candidate whose stored join fingerprint matches the runtime intent. When the runtime intent has no join fingerprint yet and every candidate shares one join key, returns the lexicographically smallest stable pick. When candidates disagree on join keys and the runtime intent is not pinned yet, returns ``None`` so the join LLM can run first."""
    if not candidates:
        return None
    jkeys = {join_path_key_concrete(c.template.intent_signature) for c in candidates}
    if len(jkeys) > 1 and not _join_runtime_intent_has_join_fingerprint(intent):
        return None
    if len(jkeys) == 1 and not _join_runtime_intent_has_join_fingerprint(intent):
        return min(
            candidates,
            key=lambda c: (c.non_agg_symmetric_diff, len(c.union_cols), c.template.id),
        )
    filtered = [c for c in candidates if _join_runtime_matches_template_concrete(intent, c.template)]
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
        keeper = templates.get(keeper_id)
        if keeper is None:
            return
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
    if not ufi and is_template_store_view(template_store_view):
        indexes = getattr(template_store_view, "_indexes", None)
        if isinstance(indexes, dict):
            raw = indexes.get(TEMPLATE_UNION_FAMILY_INDEX_KEY)
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
    """Pick a ``trust_level>=1`` template with matching body and small. select diff. Args: intent: Validated runtime intent. templates: Id -> template map. Returns: ``(template, union_select_cols, cols_changed, union_sql_path)`` or ``None``."""
    rows = list_union_match_candidates(intent, templates)
    if not rows:
        return None
    best = rows[0]
    debug(
        f"[intent_process.match_template_for_union] matched template={best.template.id} "
        f"non_agg_diff={best.non_agg_symmetric_diff} union_cols={len(best.union_cols)} sql_path={best.union_sql_path.code}"
    )
    return best.template, best.union_cols, best.cols_changed, best.union_sql_path
