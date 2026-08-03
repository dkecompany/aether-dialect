"""Interactive text-to-SQL orchestration: intent parsing, joins, SQL generation, validation, and template/negative memory."""

from __future__ import annotations

import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import Context, copy_context
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar, cast

import pandas

from ._config import EngineConfig, PolicyConfig, llm_credentials_configured
from ._refusal_diagnostics import (
    emit_session_refusal_diagnostic,
    refusal_diagnostic_code_for_exception,
    refusal_message_for_exception,
)
from ._constants import (
    ASK_PHASE_A,
    ASK_PHASE_B,
    ASK_PHASE_H,
    ASK_PHASE_J,
    ASK_PHASE_K,
    ASK_PHASE_L,
    ASK_PHASE_M,
    ASK_PHASE_N,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_EXECUTED,
    DIAGNOSTIC_CODE_FEDERATION_JOIN_CANDIDATE_CAP,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_GENERATED,
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
    DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED,
    DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
    AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER,
    DIAGNOSTIC_CODE_REUSE_HIT,
    DIAGNOSTIC_CODE_REUSE_MISS,
    DISPLAY_ALIAS_PROMPT_KEY_ORDER,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_MAPPINGS_VERSION,
    FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER,
    INTERACTIVE_STAGE_DIRECT_REUSE,
    INTERACTIVE_STAGE_INTENT_CONFIRM,
    JOIN_CHOICE_SCOPE_MAIN,
    PERMISSION_DENIED_USER_MESSAGE,
    PIPELINE_BUG_SQL_VALIDATION,
    PLAN_PREVIEW_INTENT_PARSE_FAILED,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    SHAPE_QUESTION_INDEX_KEY,
    SOFT_DIAGNOSTIC_CODES,
    SQL_BIND_TOKEN_RE,
    TEMPLATE_INTENT_KEY_INDEX_KEY,
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    MASTER_AETHERSPACE_NAME,
)
from ._contracts_base import (
    AccessError,
    AggregateJoinFanOutError,
    ClauseWidenedRowsetError,
    ComparisonJoinScopeExceededError,
    ConfigError,
    EngineContext,
    EngineIdentity,
    FailureCategory,
    FederationManifest,
    FederationMappings,
    FederationCoordinatorConfig,
    FederationMemberExecutionError,
    FederationPartialFailureError,
    FederationPlanTemplate,
    FederationTurnCancelledError,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    NoJoinPathError,
    ProbeCtePlacementError,
    PipelineSuspended,
    RetryableError,
    SpaceContext,
    SqlDiagnostic,
    StatementTimeoutError,
    TemplateExecutionResult,
    PlanPreviewResult,
    WriteQueueEvent,
    expr_registry_ref,
    having_leaves,
    where_leaves,
)
from ._contracts_core import (
    ConcreteIntent,
    DirectReuseSuspendContext,
    FederatedExecutionOutcome,
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    FederatedSqlBundle,
    FederatedSqlOutcome,
    FederatedStatementRecord,
    FederationExecutionContext,
    FeedbackKind,
    GenerationPath,
    InteractiveTailSnapshot,
    InterpretPlan,
    QuestionFormStorage,
    RefinementContext,
    RefinementRetry,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    SqlGenerationOutcome,
    Template,
    TemplateMatch,
    UserFeedbackRejectSuspendContext,
    ValueHistory,
    concrete_cte_to_runtime,
    concrete_intent_to_runtime_skeleton,
    runtime_intent_to_concrete,
)
from ._contracts_schema import SchemaGraph, resolve_intent_visible_objects
from ._core_utils import (
    InteractiveChoicePort,
    RephraseHint,
    active_federation_execution_context,
    bind_params_for_sql,
    business_knowledge_scope,
    debug,
    emit_ask_phase,
    emit_write_queue_event,
    federation_turn_cancelled,
    interactive_yes_no,
    invalid_input,
    is_structural_param_key,
    join_resolved_scope_tables,
    llm_usage_attribution,
    normalize_question,
    note_interactive_turn,
    notify,
    phase_timer,
    pipeline_trace,
    pop_engine_identity,
    pop_federation_execution_context,
    print_info,
    print_query_result,
    print_rephrase_hint,
    progress,
    prompt,
    prompt_cache_schema_scope,
    prompt_json,
    push_engine_identity,
    push_federation_execution_context,
    reconcile_execute_bind_params,
    reduce_structural_sql_placeholders,
    safe_json_loads,
    schema_prompt_cache_id,
    stable_json,
    terminated,
)
from ._dialect import (
    Dialect,
    active_sqlglot_dialect,
    compute_sql_fp,
    extra_where_ops_for_engine,
    finalize_executable_sql,
    get_dialect,
    get_runtime_config_class,
    list_engines,
    sql_outer_select_aliases,
)
from ._federation import (
    CoordinatorMemberFrame,
    FederationCapExceededError,
    FederationConfigError,
    FederationRuntimeError,
    apply_projected_keys_to_intent,
    column_where_value_type,
    coordinator_member_row_count,
    coordinator_residual_bind_map,
    credit_federation_plan_accept,
    declared_table_for_source_column,
    delete_unaccepted_federation_plan_template,
    dialect_streams_arrow_to_coordinator,
    distinct_semijoin_keys,
    effective_union_specs,
    execute_federation_coordinator,
    enforce_federation_plan_timeout,
    federation_coordinator_spill_dir,
    federation_member_execution_batches,
    federation_member_parallelism_cap,
    federation_member_resolved_limits,
    federation_member_schema_graph_ids,
    federation_member_timeout_error,
    federation_plan_combine_hash,
    federation_plan_combine_kind,
    federation_plan_timeout_deadline,
    federation_plan_is_degenerate,
    federation_plan_matches_template,
    federation_plan_residual_hash,
    federation_plan_sql_shape,
    federation_plan_step_fingerprints,
    federation_plan_topology_identity,
    federation_residual_column_headers,
    federation_scaled_join_candidate_cap,
    federation_scaled_join_path_tie_cap,
    federation_stage_execution_waves,
    inject_filter_keys_where,
    inject_semijoin_where,
    intersect_member_where_ops,
    load_federation_plan_templates,
    lookup_federation_plan_template_for_question,
    member_feedback_q_norm,
    member_frame_column_names,
    member_schema_slice,
    member_stage_for_source,
    order_federation_execution_steps,
    plan_federated_intent,
    record_federation_join_feedback,
    render_federation_glue,
    resolve_federated_member_schema,
    resolve_source_column_table,
    revalidate_prepared_federation_plan,
    save_federation_plan_template,
    schema_spans_multiple_sources,
    semijoin_key_columns,
    semijoin_key_is_allowed,
    semijoin_key_passes_distinct_floor,
    source_by_table_from_schema,
    source_row_cap_for_source,
    reducing_edge_allowed_for_target,
    split_qualified_column,
    source_semijoin_enabled,
    source_timeout_for_source,
    member_guard_limit_kwargs,
    stamp_federation_member_template,
    template_is_federation_plan_fragment,
    validate_federated_sub_intent,
    validate_member_frame_projection,
)
from ._intent_expr import (
    build_virtual_table_specs,
    cleared_param_runtime_intent,
    extract_structural_params,
    narrow_bind_map_for_sub_intent,
    structural_s_key_assignment_order,
)
from ._intent_process import (
    apply_runtime_post_processing,
    find_trusted_template_match,
    invoke_intent_parse_with_hints,
    list_union_match_candidates,
    pick_union_match_for_runtime_join,
    reconcile_template_store_until_stable,
    resolve_sql_path,
    structural_compare,
)
from ._intent_repair import (
    append_table_scope_repairs,
    apply_diagnostic_repairs,
    drop_redundant_resolved_join_where_predicates,
    expand_shared_pk_tables_for_refs,
)
from ._intent_resolve import join_path_key_concrete, prune_unused_cte_steps
from ._llm_provider import llm_chat
from ._schema_graph import assert_consumer_intent_in_scope, assert_consumer_sql_in_scope, assert_intent_in_scope
from ._sql_gen import (
    ScopeClass,
    build_deterministic_sql,
    build_display_sql,
    canonicalize_stored_join_path_signature,
    cte_emission_map,
    cte_to_intent_for_ranking,
    emit_join_orphan_rate_diagnostics,
    first_base_non_j00_candidate_id,
    generate_col_alias,
    get_join_choice_from_llm,
    inject_join_into_deterministic_sql,
    join_candidate_map,
    join_candidate_spans_tables,
    join_choice_scope_key_cte,
    join_hints_multi,
    join_scope_pass1_plan,
    join_scope_pass2_llm_scopes,
    merge_join_hints_for_na_scopes,
    probe_cte_names,
    render_select_col_sql,
    select_col_prefers_llm_display_alias,
    tables_in_join_scope,
)
from ._templates import (
    TemplateStoreView,
    artifacts_dir_for_template_store,
    delete_rejected_templates_matching_question,
    handles_referenced_in_sql_param,
    has_any_rejection_history_for_question,
    insert_template,
    join_fingerprint_from_concrete_intent,
    join_fingerprint_from_runtime_intent,
    lookup_join_feedback_for_question,
    param_keys_from_intent_signature,
    param_slot_prompt_payload,
    path_bucket,
    primary_template_q_norm,
    promote_rejected_to_template,
    promote_trust,
    record_deterministic_join_failure_feedback,
    record_per_question_feedback,
    record_question_feedback,
    record_template_feedback,
    record_value_history_on_accept,
    reject_out_per_question,
    resolve_template_for_question,
    resolve_template_ref,
    save_template_store,
    should_auto_accept_for_question,
    sqlglot_dialect_for_template_fingerprint,
    summarize_failure_for_memory,
    template_is_live,
    template_schema_refs,
    template_visible_to_callers,
    templates_to_store,
)
from ._utils import exact_question_match, flatten_param_values, intent_key, sql_shape
from ._validation_execute import (
    canonicalize_rejection_reason,
    enforce_probe_cte_anchor_placement_post_resolution,
    execute_guarded_arrow_table,
    execute_guarded_sql,
    validate_aggregate_join_fan_out,
    validate_sql,
    temporary_dialect_member_limits,
)
from ._validation_schema import (
    validate_clause_widened_rowset,
    validate_comparison_join_scope_or_raise,
    validate_intent_join_reachability,
    validate_join_path_reachability_for_tables,
)


def _execution_scope_gate_active(
    schema_context: EngineContext | None,
    execution_visible_objects: frozenset[str] | None,
    schema_role: str,
    *,
    context_name: str = MASTER_AETHERSPACE_NAME,
) -> bool:
    """Return True when the execution-time context/RBAC gate should run."""
    if schema_role == "consumer":
        return True
    ctx = schema_context if schema_context is not None else EngineContext()
    norm_name = str(context_name or MASTER_AETHERSPACE_NAME).strip().lower() or MASTER_AETHERSPACE_NAME
    if norm_name != MASTER_AETHERSPACE_NAME:
        return True
    if ctx.allow_objects or ctx.deny_objects or ctx.deny_columns or ctx.allow_columns:
        return True
    return execution_visible_objects is not None


def _row_structural_values_match_defaults(
    row_params: dict[str, Any], structural_defaults: dict[str, Any] | None, sql_param: str
) -> bool:
    """Return True when every ``:s`` referenced in SQL matches structural defaults in *row_params*."""
    sd = structural_defaults or {}
    s_keys = set(re.findall(r":(s\d+)", sql_param))
    for sk in s_keys:
        if sk not in row_params:
            continue
        if sk not in sd:
            return False
        if row_params[sk] != sd[sk]:
            return False
    return True


def extract_fuzzy_reuse_params(
    q_norm: str,
    template: Template,
    *,
    history_index: int,
    literal_structural_only: bool,
    schema: SchemaGraph | None = None,
) -> dict[str, Any]:
    """Extract p- and s-parameter values from a question for fuzzy template reuse via one LLM call."""
    p_key_names, s_key_names = param_keys_from_intent_signature(
        template.intent_signature, literal_structural_only=literal_structural_only
    )
    all_keys = p_key_names + ([] if literal_structural_only else s_key_names)
    vh = template.value_history
    idx = max(0, min(history_index, len(vh.questions) - 1)) if vh.questions else 0
    prev_pv = vh.param_values[idx] if vh.param_values else {}
    if not all_keys:
        if literal_structural_only:
            return dict(prev_pv)
        return {}
    example_pv = {}
    for k in all_keys:
        example_pv[k] = prev_pv.get(k, "example")
    matched_question = vh.questions[idx] if vh.questions else ""
    matched_values = prev_pv
    system = (
        "You are a deterministic parameter value extractor for text-to-SQL."
        " Output ONLY valid JSON that matches the requested format."
    )
    user = prompt_json(
        {
            "task": "The parameter slots below describe literal bind handles from the stored intent. Extract the correct value from the question for every param_key listed.",
            "param_slots": param_slot_prompt_payload(template.intent_signature, all_keys),
            "matched_question": matched_question,
            "matched_values": matched_values,
            "param_keys": all_keys,
            "question": q_norm,
            "extraction_rules": [
                "p-params (p1, p2, ...) are filter/having values: strings, numbers, dates, booleans, or arrays.",
                "s-params (s1, s2, ...) are structural numeric values: LIMIT, coefficients, function arguments.",
                "Quoted text or named entities → string/enum values. Preserve original case.",
                "Digits in the question → integer or number values. Use numeric type, not string.",
                "'in'/'not in' operators → value is an array of extracted items.",
                "Date expressions → convert to YYYY-MM-DD format.",
                "true/false/yes/no keywords → boolean values.",
                "s1, s2, ... correspond to LIMIT values, coefficients, or function arguments in expressions.",
            ],
            "output_format": {
                "param_values": example_pv,
            },
        },
        FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER,
    )

    def _reuse_llm_chat() -> str:
        with prompt_cache_schema_scope(schema_prompt_cache_id(schema)):
            return llm_chat(system, user, task="default")

    raw = _reuse_llm_chat()
    parsed = safe_json_loads(raw)
    if not parsed or not isinstance(parsed, dict):
        raw2 = _reuse_llm_chat()
        parsed = safe_json_loads(raw2)
    if not parsed or not isinstance(parsed, dict):
        debug(f"[{ASK_PHASE_A}] JSON parse failed")
        return {}
    pv_raw = parsed.get("param_values", {})
    result: dict[str, Any] = {}
    for k in all_keys:
        if k in pv_raw:
            val = pv_raw[k]
            if k.startswith("s") and isinstance(val, str):
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            result[k] = val
    if literal_structural_only:
        for sk in s_key_names:
            if sk in prev_pv:
                result[sk] = prev_pv[sk]
    debug(f"[{ASK_PHASE_A}] extracted {len(result)}/{len(all_keys)} params")
    return result


def _extract_reuse_params_literal_only(
    q_norm: str, template: Template, *, history_index: int, schema: SchemaGraph | None = None
) -> dict[str, Any]:
    """Paths ``2.1``: LLM fills ``p*`` only; structural ``s*`` come from the exemplar row and defaults."""
    return extract_fuzzy_reuse_params(
        q_norm, template, history_index=history_index, literal_structural_only=True, schema=schema
    )


def _extract_reuse_params_full(
    q_norm: str, template: Template, *, history_index: int, schema: SchemaGraph | None = None
) -> dict[str, Any]:
    """Paths ``2.2``: LLM fills both ``p*`` and ``s*`` keys present in the intent signature."""
    return extract_fuzzy_reuse_params(
        q_norm, template, history_index=history_index, literal_structural_only=False, schema=schema
    )


def load_pipeline_resources(
    schema: SchemaGraph | None = None,
    store: Any = None,
    templates: dict[str, Any] | None = None,
    rejected: dict[str, Any] | None = None,
    schema_terms: set[str] | None = None,
    dialect: Dialect | None = None,
) -> tuple[Dialect, SchemaGraph, Any, dict[str, Any], dict[str, Any], set[str]]:
    """Validate inputs, build dialect, and return pipeline resource. bundle."""
    if not llm_credentials_configured():
        raise RuntimeError("No OpenAI/Azure OpenAI API key configured")

    debug("loading schema")
    if dialect is None:
        dialect = get_dialect(EngineConfig.TYPE, EngineConfig.RUNTIME)
    if EngineConfig.TYPE not in list_engines():
        raise ValueError(f"Unsupported engine type: {EngineConfig.TYPE}")

    if schema is None:
        raise RuntimeError("Schema must be provided to load_pipeline_resources")
    if store is None:
        raise RuntimeError("Store must be provided to load_pipeline_resources")
    if templates is None:
        raise RuntimeError("Templates must be provided to load_pipeline_resources")
    if rejected is None:
        raise RuntimeError("Rejected must be provided to load_pipeline_resources")
    if schema_terms is None:
        raise RuntimeError("Schema terms must be provided to load_pipeline_resources")

    debug(f"templates_loaded={len(templates)}")
    debug(f"[{ASK_PHASE_A}] templates: {len(templates)} approved, {len(rejected)} rejected")
    debug(f"[{ASK_PHASE_A}] schema_terms: {len(schema_terms)} terms")

    return dialect, schema, store, templates, rejected, schema_terms


def match_question_level_template_reuse(
    candidate_question: str,
    templates: dict[str, Any],
    *,
    template_store: dict[str, Any] | TemplateStoreView | None = None,
    candidate_intent: RuntimeIntent | None = None,
    schema: SchemaGraph | None = None,
) -> TemplateMatch:
    """Detect fuzzy question match against trusted template. ``value_history`` for direct SQL reuse."""
    debug(f"[{ASK_PHASE_A}] checking exact fuzzy match")
    templates_list = list(templates.values()) if isinstance(templates, dict) else templates
    idx: dict[str, list[str]] | None = None
    qtok_idx: dict[str, list[Any]] | None = None
    intent_idx: dict[str, list[str]] | None = None
    uf_idx: dict[str, list[str]] | None = None
    if template_store is not None:
        raw_idx = template_store.get(SHAPE_QUESTION_INDEX_KEY)
        if isinstance(raw_idx, dict):
            idx = {str(k): [str(x) for x in v] for k, v in raw_idx.items() if isinstance(v, list)}
        raw_qtok = template_store.get(TEMPLATE_QUESTION_TOKEN_INDEX_KEY)
        if isinstance(raw_qtok, dict):
            qtok_idx = raw_qtok
        raw_ik = template_store.get(TEMPLATE_INTENT_KEY_INDEX_KEY)
        if isinstance(raw_ik, dict):
            intent_idx = {str(k): v for k, v in raw_ik.items() if isinstance(v, list)}
        raw_uf = template_store.get(TEMPLATE_UNION_FAMILY_INDEX_KEY)
        if isinstance(raw_uf, dict):
            uf_idx = {str(k): [str(x) for x in v] for k, v in raw_uf.items() if isinstance(v, list)}
    hit = find_trusted_template_match(
        candidate_question,
        templates_list,
        shape_question_index=idx,
        question_token_index=qtok_idx,
        intent_key_index=intent_idx,
        union_family_index=uf_idx,
        candidate_intent=candidate_intent,
    )

    if hit is not None:
        ref_tmpl = hit.template
        if schema is not None:
            live_ok, stale_reasons = template_is_live(template_schema_refs(ref_tmpl), schema)
            if not live_ok:
                debug(f"[{ASK_PHASE_A}] question_reuse_template_not_live: {','.join(stale_reasons)}")
                hit = None
    if hit is not None:
        ref_tmpl = hit.template
        reuse_hit = hit.reuse_hit
        debug("exact question match found (trust>=1, score=1.0)")
        debug(f"[{ASK_PHASE_A}] AUTO-DECISION: fuzzy question match for template '{ref_tmpl.id}'")
        return TemplateMatch(
            intent=None,
            best_template=ref_tmpl,
            similarity_score=1.0,
            reuse_type="direct_reuse",
            reuse_candidate_normalized=reuse_hit.candidate_normalized,
            reuse_history_index=reuse_hit.history_index,
        )

    if templates_list:
        notify("No trusted template matched for direct SQL reuse.", stage="pipeline", code=DIAGNOSTIC_CODE_REUSE_MISS)
    return TemplateMatch(
        intent=None, best_template=None, similarity_score=0.0, reuse_type="none", reuse_candidate_normalized=None
    )


def build_interactive_tail_snapshot(
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema_terms: set[str],
    dialect: Any,
    semantic_warnings: list[Any],
    has_union_match: bool,
    cols_changed: bool,
    matched_template: Template | None,
    union_select_cols: list[SelectCol] | None,
    structural_match_templates: list[Template],
    ikey: str,
    intent_sim: float,
    union_sql_path: GenerationPath | None = None,
    union_candidate_template_ids: Sequence[str] | None = None,
    form_storage: QuestionFormStorage | None = None,
    interpretation: InterpretPlan | None = None,
) -> InteractiveTailSnapshot:
    """Freeze the interactive bundle needed to resume after intent or. hard-block prompts."""
    uc = tuple(union_select_cols) if union_select_cols is not None else None
    st = tuple(structural_match_templates)
    sw = tuple(semantic_warnings) if semantic_warnings else ()
    ucid = tuple(union_candidate_template_ids) if union_candidate_template_ids else ()
    return InteractiveTailSnapshot(
        q_norm=q_norm,
        intent=intent,
        schema=schema,
        store=cast(dict[str, Any], store),
        templates=templates,
        rejected=rejected,
        schema_terms=set(schema_terms),
        dialect=dialect,
        semantic_warnings=sw,
        has_union_match=has_union_match,
        cols_changed=cols_changed,
        matched_template=matched_template,
        union_select_cols=uc,
        structural_match_templates=st,
        ikey=ikey,
        intent_sim=intent_sim,
        union_sql_path=union_sql_path,
        union_candidate_template_ids=ucid,
        form_storage=form_storage,
        interpretation=interpretation,
    )


def _refinement_ctx_for_feedback(
    choice_port: InteractiveChoicePort | None, refinement_ctx: RefinementContext | None
) -> RefinementContext | None:
    """Resolve optional refinement state from an explicit argument or an attached interactive session."""
    if refinement_ctx is not None:
        return refinement_ctx
    return getattr(choice_port, "_refinement_ctx", None)


def refinement_retry_available(ctx: RefinementContext | None) -> bool:
    """Return True when another silent refinement parse may run after recording rejection."""
    if ctx is None:
        return False
    if ctx.block_further_refinement:
        return False
    if ctx.pending_retry:
        return True
    return ctx.refinement_rounds_executed < PolicyConfig.MAX_USER_REFINEMENTS


def _artifact_dir_for_template_store(store: Any) -> str:
    if isinstance(store, TemplateStoreView):
        return artifacts_dir_for_template_store(store._store_dir)
    return artifacts_dir_for_template_store(EngineConfig.TEMPLATE_STORE_DIR)


def _emit_reader_write_queue_event(store: Any, event: WriteQueueEvent) -> None:
    emit_write_queue_event(_artifact_dir_for_template_store(store), event)


def parse_intent_via_llm(
    q_norm: str,
    schema: SchemaGraph,
    templates: dict[str, Any],
    store: dict[str, Any] | TemplateStoreView,
    choice_port: InteractiveChoicePort | None = None,
    *,
    extra_user_feedback: list[str] | None = None,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
) -> tuple[RuntimeIntent | None, list[str], int, InterpretPlan | None]:
    """Parse intent with. :func:`aetherdialect._intent_process._invoke_in. tent_parse_with_hints` and optional user confirmation on schema- invalid paths."""
    emit_ask_phase(ASK_PHASE_B)
    debug("intent via LLM")
    debug(f"[{ASK_PHASE_B}] calling invoke_intent_parse_with_hints")
    resolved_ctx = _refinement_ctx_for_feedback(choice_port, refinement_ctx)
    conv_corr: tuple[str, ...] = ()
    if resolved_ctx is not None:
        raw_hints = getattr(resolved_ctx, "conversation_rejection_hints", None)
        if isinstance(raw_hints, tuple):
            conv_corr = raw_hints
    if resolved_ctx is not None and resolved_ctx.accumulated_reasons:
        seed_lines = list(resolved_ctx.accumulated_reasons)
    else:
        seed_lines = list(extra_user_feedback or [])
    intent_visible_objects = (
        resolve_intent_visible_objects(
            visible_objects=getattr(choice_port, "visible_objects", None),
            execution_visible_objects=getattr(choice_port, "execution_visible_objects", None),
        )
        if choice_port is not None
        else None
    )
    intent, semantic_warnings, llm_calls, interpret_plan = invoke_intent_parse_with_hints(
        q_norm,
        schema,
        store=cast(dict[str, Any] | None, store),
        extra_user_feedback=seed_lines if seed_lines else None,
        prior_user_corrections=conv_corr,
        persist_template_learning=persist_template_learning,
        visible_objects=intent_visible_objects,
        allowed_columns=getattr(choice_port, "space_columns", None) if choice_port is not None else None,
        deny_objects=getattr(choice_port, "space_deny_objects", None) if choice_port is not None else None,
        deny_columns=getattr(choice_port, "space_deny_columns", None) if choice_port is not None else None,
        description_overlay=getattr(choice_port, "space_description_overlay", None)
        if choice_port is not None
        else None,
    )
    if intent is not None:
        pipeline_trace("pipeline.parse_intent_via_llm.intent_complete", lambda: stable_json(intent.to_dict()))

    if intent is None:
        print_rephrase_hint(RephraseHint.INTENT_PARSE_FAILED)
        if persist_template_learning and isinstance(store, TemplateStoreView):
            save_template_store(store)
        return None, semantic_warnings, llm_calls, interpret_plan

    if semantic_warnings:
        debug(f"Semantic warnings after repair: {len(semantic_warnings)} remaining")

    debug(f"[{ASK_PHASE_B}] tables={intent.tables}, grain={intent.grain}")
    return intent, semantic_warnings, llm_calls, interpret_plan


def generate_join_candidates(
    intent: RuntimeIntent, schema: SchemaGraph
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Build join hint payloads for the main query and each multi-table. CTE."""
    debug(f"[{ASK_PHASE_J}] generating join candidates")
    virtual_specs = build_virtual_table_specs(intent, schema)
    cross_product_cap: int | None = None
    tie_cap: int | None = None
    if schema_spans_multiple_sources(schema):
        member_count = len({meta.source_id for meta in schema.tables.values() if meta.source_id})
        cross_product_cap = federation_scaled_join_candidate_cap(max(1, member_count))
        tie_cap = federation_scaled_join_path_tie_cap(max(1, member_count))
        pipeline_trace(
            DIAGNOSTIC_CODE_FEDERATION_JOIN_CANDIDATE_CAP, lambda: f"cap={cross_product_cap} members={member_count}"
        )

    def _scope_only_j00(hint_dict: dict[str, Any]) -> bool:
        cands = hint_dict.get("candidates") or []
        return (not cands) or (len(cands) == 1 and cands[0].get("candidate_id") == "J00")

    scope_main = tables_in_join_scope(intent.tables, schema, virtual_specs)
    join_candidates = join_hints_multi(
        schema,
        scope_main,
        intent,
        virtual_specs=virtual_specs,
        include_semantic=False,
        cross_product_cap=cross_product_cap,
        tie_cap=tie_cap,
    )
    if len(scope_main) >= 2 and _scope_only_j00(join_candidates):
        join_candidates = join_hints_multi(
            schema,
            scope_main,
            intent,
            virtual_specs=virtual_specs,
            include_semantic=True,
            cross_product_cap=cross_product_cap,
            tie_cap=tie_cap,
        )
    cmap = join_candidate_map(join_candidates)
    debug(f"[{ASK_PHASE_J}] {len(join_candidates.get('candidates', []))} join candidates")
    for c in join_candidates.get("candidates", []):
        debug(f"[{ASK_PHASE_J}] {c.get('candidate_id')}: {c.get('join_path_signature', [])}")

    cte_join_hints: dict[str, dict[str, Any]] = {}
    cte_steps = intent.cte_steps or []
    for cte in cte_steps:
        cte_name = cte.cte_name
        cte_tables = list(cte.tables or [])
        if len(cte_tables) >= 2:
            cte_intent = cte_to_intent_for_ranking(cte)
            scope_cte = tables_in_join_scope(cte_tables, schema, virtual_specs)
            cte_hints = join_hints_multi(
                schema,
                scope_cte,
                cte_intent,
                virtual_specs=virtual_specs,
                include_semantic=False,
                cross_product_cap=cross_product_cap,
                tie_cap=tie_cap,
            )
            if len(scope_cte) >= 2 and _scope_only_j00(cte_hints):
                cte_hints = join_hints_multi(
                    schema,
                    scope_cte,
                    cte_intent,
                    virtual_specs=virtual_specs,
                    include_semantic=True,
                    cross_product_cap=cross_product_cap,
                    tie_cap=tie_cap,
                )
            cte_join_hints[cte_name] = cte_hints
            debug(
                f"[{ASK_PHASE_J}] CTE '{cte_name}': {len(cte_hints.get('candidates', []))} candidates (CTE-specific ranking)"
            )
        else:
            cte_join_hints[cte_name] = {
                "candidates": [
                    {
                        "candidate_id": "J00",
                        "predicates": [],
                        "join_path_signature": [],
                        "edge_kinds": [],
                        "candidate_tier": "base",
                        "edge_count": 0,
                        "description": "No join needed (single table CTE)",
                    }
                ]
            }
            debug(f"[{ASK_PHASE_J}] CTE '{cte_name}': single table, J00 assigned")

    pipeline_trace(
        "pipeline.generate_join_candidates.full",
        lambda: stable_json(
            {
                "scope_main_tables": scope_main,
                "join_candidates": join_candidates,
                "cmap": {k: list(v) for k, v in cmap.items()},
                "cte_join_hints": cte_join_hints,
            }
        ),
    )
    return join_candidates, cmap, cte_join_hints


def _join_preset_scope_from_concrete(conc: ConcreteIntent) -> dict[str, str]:
    """Build join-choice preset scope from stored template join candidate ids."""
    preset: dict[str, str] = {}
    main_cid = str(conc.chosen_join_candidate_id or "").strip()
    if main_cid and main_cid != "J00":
        preset[JOIN_CHOICE_SCOPE_MAIN] = main_cid
    for cte in conc.cte_steps or []:
        ccid = str(cte.chosen_join_candidate_id or "").strip()
        if ccid and ccid != "J00":
            preset[join_choice_scope_key_cte(cte.cte_name)] = ccid
    return preset


def _sql_phase_join_resources(
    intent: RuntimeIntent, schema: SchemaGraph, matched_template: Template | None, union_sql_path: GenerationPath | None
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Produce join candidate payloads for the SQL phase (always fully enumerated)."""
    jc, cmap, hints = generate_join_candidates(intent, schema)
    if matched_template is not None and union_sql_path == GenerationPath.INTENT_DIRECT_MATCH:
        conc = matched_template.intent_signature
        cid = str(getattr(conc, "chosen_join_candidate_id", "") or "").strip()
        if cid and cid != "J00":
            intent.chosen_join_candidate_id = cid
            intent.chosen_join_path_signature = canonicalize_stored_join_path_signature(
                list(conc.chosen_join_path_signature or []),
                from_anchor=intent.tables[0] if intent.tables else None,
            )
            conc_ctes = conc.cte_steps or []
            for idx, step in enumerate(intent.cte_steps or []):
                if idx >= len(conc_ctes):
                    break
                cs = conc_ctes[idx]
                ccid = str(getattr(cs, "chosen_join_candidate_id", "") or "").strip()
                if ccid and ccid != "J00":
                    step.chosen_join_candidate_id = ccid
                    step.chosen_join_path_signature = canonicalize_stored_join_path_signature(
                        list(getattr(cs, "chosen_join_path_signature", []) or []),
                        from_anchor=(step.tables[0] if step.tables else None),
                    )
    return jc, cmap, hints


def _join_matches_template_intent(matched: Template | None, intent: RuntimeIntent) -> bool | None:
    """Compare stored template join fingerprint to the runtime intent join fingerprint."""
    if matched is None:
        return None
    return join_fingerprint_from_concrete_intent(matched.intent_signature) == join_fingerprint_from_runtime_intent(
        intent
    )


def _resolve_joins_for_intent_placeholder(
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    join_candidates: dict[str, Any],
    cmap: dict[str, list[str]],
    cte_join_hints: dict[str, dict[str, Any]] | None,
    structural_defaults_src: dict[str, Any] | None = None,
    store: dict[str, Any] | TemplateStoreView | None = None,
) -> None:
    """Populate runtime join fields from deterministic SQL before pinning a union template. Paths touched: ``3``, ``4.x`` when multiple stored join fingerprints require an LLM join choice."""
    intent = prune_unused_cte_steps(intent)
    join_candidates, cmap, cte_join_hints = generate_join_candidates(intent, schema)

    prior_fb: list[str] | None = None
    if store is not None:
        prior_fb = lookup_join_feedback_for_question(store, q_norm)

    anchor_main, anchor_cte = _join_signatures_for_deterministic_from_anchor(cmap, cte_join_hints, intent)
    deterministic_sql = build_deterministic_sql(
        intent,
        cte_join_hints,
        schema,
        dialect,
        join_signature_for_from_anchor=anchor_main if anchor_main else None,
        cte_join_signatures_for_from_anchor=anchor_cte if anchor_cte else None,
    )
    _resolve_joins_fresh(
        deterministic_sql,
        intent,
        cmap,
        cte_join_hints,
        q_norm,
        join_candidates,
        schema=schema,
        structural_defaults=structural_defaults_src,
        dialect=dialect,
        prior_join_feedback=prior_fb,
    )


def prepare_union_match_join_phase(
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    templates: dict[str, Any],
    store: dict[str, Any] | TemplateStoreView | None = None,
) -> tuple[
    Template | None,
    list[SelectCol] | None,
    bool,
    GenerationPath | None,
    bool,
    dict[str, Any],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
]:
    """Resolve union-template reuse versus join LLM when body matches span multiple join fingerprints. Paths touched: ``3``, ``4.1``, ``4.2``, ``4.3`` for interactive and live runners."""
    candidates = list_union_match_candidates(intent, templates, schema=schema)
    if not candidates:
        jc, cmap, hints = generate_join_candidates(intent, schema)
        return None, None, False, None, False, jc, cmap, hints

    candidate_join_keys = {join_path_key_concrete(c.template.intent_signature) for c in candidates}
    if len(candidate_join_keys) == 1 and len(candidates) == 1:
        chosen = candidates[0]
        jc, cmap, hints = _sql_phase_join_resources(intent, schema, chosen.template, chosen.union_sql_path)
        return (chosen.template, chosen.union_cols, chosen.cols_changed, chosen.union_sql_path, True, jc, cmap, hints)
    if len(candidate_join_keys) == 1:
        chosen = min(candidates, key=lambda c: (c.non_agg_symmetric_diff, len(c.union_cols), c.template.id))
        jc, cmap, hints = _sql_phase_join_resources(intent, schema, chosen.template, chosen.union_sql_path)
        return (chosen.template, chosen.union_cols, chosen.cols_changed, chosen.union_sql_path, True, jc, cmap, hints)
    jc, cmap, hints = generate_join_candidates(intent, schema)
    _resolve_joins_for_intent_placeholder(
        q_norm, intent, schema, dialect, jc, cmap, hints, structural_defaults_src=None, store=store
    )
    union_chosen = pick_union_match_for_runtime_join(intent, candidates)
    if union_chosen is None:
        return None, None, False, None, False, jc, cmap, hints
    jc2, cmap2, hints2 = _sql_phase_join_resources(intent, schema, union_chosen.template, union_chosen.union_sql_path)
    return (
        union_chosen.template,
        union_chosen.union_cols,
        union_chosen.cols_changed,
        union_chosen.union_sql_path,
        True,
        jc2,
        cmap2,
        hints2,
    )


def _template_effective_sql_display_param(tmpl: Template, dialect: Dialect) -> str:
    """Return user-facing display SQL for a template row, recomputing when storage omits it. Paths touched: ``1``, ``2.1``, ``2.2`` direct reuse and any reader after storage trim."""
    rt = concrete_intent_to_runtime_skeleton(tmpl.intent_signature)
    return build_display_sql(tmpl.sql_param, rt, tmpl.display_alias_map or None, dialect=dialect)


def enriched_display_alias_map(
    q_norm: str, sql_param: str, disp: RuntimeIntent, base: dict[str, str] | None
) -> dict[str, str]:
    """Merge persisted ``display_alias_map`` with LLM-suggested headers for complex select columns. Simple columns use deterministic aliases only (no LLM). Paths touched: ``3``, ``4.1``, ``4.2``, ``4.3``, ``5``."""
    out = dict(base or ())
    cols = list(disp.select_cols or [])
    if not cols:
        return out
    targets: list[tuple[str, str]] = []
    for sc in cols:
        expr_str = render_select_col_sql(sc)
        sk = sc.signature_key
        if select_col_prefers_llm_display_alias(sc) and sk not in out:
            targets.append((sk, expr_str.strip()[:500]))
    if not targets:
        return out
    system = (
        'You output ONLY JSON: {"aliases": {<signature_key>: <short_snake_case_label>}}. '
        "Every signature_key listed in the user payload must appear exactly once. "
        "Values: unique ascii lower_snake_case, max 48 chars, no spaces."
    )
    user = prompt_json(
        {
            "task": "Short result-grid column headers for SQL SELECT expressions.",
            "question": q_norm,
            "columns": [{"signature_key": k, "sql_expr": e} for k, e in targets],
        },
        DISPLAY_ALIAS_PROMPT_KEY_ORDER,
    )
    raw = llm_chat(system, user, task="default")
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, dict):
        return out
    block = parsed.get("aliases", parsed)
    if not isinstance(block, dict):
        return out
    allowed = {k for k, _ in targets}
    seen_vals: set[str] = {str(v).strip().lower() for v in out.values() if isinstance(v, str) and str(v).strip()}
    for sk, alias in block.items():
        if sk not in allowed or not isinstance(alias, str):
            continue
        piece = alias.strip().lower().replace(" ", "_")[:48]
        if not piece:
            continue
        base_a = piece
        n = 2
        while piece in seen_vals:
            piece = f"{base_a}_{n}"
            n += 1
        seen_vals.add(piece)
        out[str(sk)] = piece
    return out


def _validation_sql_for_explain(sql: str, intent: RuntimeIntent, dialect: Any) -> str:
    """Choose canonical parameterized SQL for validation before execution rewrites."""
    param_sql = (intent.sql_param or "").strip()
    if param_sql and SQL_BIND_TOKEN_RE.search(param_sql):
        base = param_sql
    else:
        base = sql
    if dialect is None:
        return base
    explain_fn = getattr(dialect, "explain_validation_sql", None)
    if not callable(explain_fn):
        return base
    return cast(str, explain_fn(base, dict(intent.param_values or {})))


def _run_sql_validation_cascade(
    sql: str,
    intent: RuntimeIntent,
    dialect: Any,
    schema: SchemaGraph | None = None,
    *,
    max_query_cost_rows: float | None = None,
    max_query_cost_bytes: float | None = None,
    profile_timeout_ms: int | None = None,
) -> tuple[bool, str, FailureCategory | None, list[SqlDiagnostic]]:
    """Run join reachability checks, then `validate_sql` (AST plus optional EXPLAIN)."""
    if schema is not None:
        reach_issues = validate_intent_join_reachability(intent, schema)
        reach_errors = [i for i in reach_issues if i.severity == "error"]
        if reach_errors:
            first = reach_errors[0]
            out_err = first.message
            debug(f"[{ASK_PHASE_K}] join reachability failed before validate_sql: {out_err}")
            pipeline_trace(
                "pipeline._run_sql_validation_cascade.reachability_failed",
                lambda: stable_json({"err": out_err, "issue_id": first.issue_id}),
            )
            return False, out_err, first.category, []
    validation_sql = _validation_sql_for_explain(sql, intent, dialect)
    with temporary_dialect_member_limits(
        dialect,
        max_query_cost_rows=max_query_cost_rows,
        max_query_cost_bytes=max_query_cost_bytes,
        profile_timeout_ms=profile_timeout_ms,
    ):
        ok, err, cat, diags = validate_sql(
            dialect,
            validation_sql,
            bind_params_for_sql(validation_sql, intent.param_values),
            schema=schema,
            intent=intent,
        )
    out_err = "" if ok and err is None else (err or "")
    debug(f"[{ASK_PHASE_K}] validate_sql ok={ok}, err={out_err}, diagnostics={len(diags)}")
    pipeline_trace(
        "pipeline._run_sql_validation_cascade.result",
        lambda: stable_json(
            {
                "ok": ok,
                "err": out_err,
                "failure_category": getattr(cat, "value", cat),
                "sql": sql,
                "param_values": dict(intent.param_values or {}),
            }
        ),
    )
    return ok, out_err, cat, diags


def other_template_owns_question_string(templates: dict[str, Any], exclude_id: str, q_norm: str) -> bool:
    """Return True when another accepted template already lists *q_norm* in value history."""
    for tid, tmpl in templates.items():
        if tid == exclude_id:
            continue
        for hq in tmpl.value_history.questions or []:
            if hq and exact_question_match(q_norm, hq, label=f"dedup_{tid}"):
                return True
    return False


def _maybe_record_value_history_accept(
    templates: dict[str, Any],
    tmpl: Template,
    intent: RuntimeIntent,
    q_norm: str,
    form_storage: QuestionFormStorage | None,
    schema: SchemaGraph,
) -> None:
    """Record accepted question form(s) on *tmpl* when no other. template. already claims the keys."""
    all_pv = flatten_param_values(intent)
    nl = intent.natural_language or ""
    pq = form_storage.corrected if form_storage is not None else q_norm
    if other_template_owns_question_string(templates, tmpl.id, pq):
        return
    nopt = form_storage.normalized_optional if form_storage is not None else None
    if nopt and nopt != pq and other_template_owns_question_string(templates, tmpl.id, nopt):
        return
    record_value_history_on_accept(
        tmpl.value_history,
        param_values=all_pv,
        natural_language=nl,
        form_storage=form_storage,
        q_norm_fallback=q_norm,
        schema=schema,
        tables_hint=sorted(intent.tables or []),
    )


def best_accepted_template_similarity(intent: RuntimeIntent, templates: dict[str, Any]) -> float:
    """Return the highest structural similarity between *intent* and. any. accepted template signature."""
    if not templates:
        return 0.0
    scores: list[float] = []
    for t in templates.values():
        cr = structural_compare(intent, t, mode="full")
        s = cr.similarity_score
        scores.append(float(s) if s is not None else 0.0)
    return max(scores, default=0.0)


def clear_planner_schema_invalid_after_user_accept(intent: RuntimeIntent) -> None:
    """Drop the ephemeral planner schema-invalid hint after the user accepts intent confirmation."""
    if intent.schema_invalid:
        intent.schema_invalid = False
        debug("[pipeline] cleared planner schema_invalid after user accepted intent confirmation")


def should_skip_intent_confirmation(
    intent: RuntimeIntent, store: dict[str, Any] | None, q_norm: str | None, semantic_warnings: list[Any] | None
) -> bool:
    """Return True when intent confirmation may be skipped. Returns. False when the parsed intent is schema-invalid, when there are semantic warnings, or when the same canonicalised question has any prior rejection recorded in ``question_feedback``."""
    if getattr(intent, "schema_invalid", False):
        return False
    if semantic_warnings:
        return False
    if store is not None and q_norm and has_any_rejection_history_for_question(store, q_norm):
        return False
    return True


def _should_prompt_direct_reuse_user(
    ref_tmpl: Template, _rejected: dict[str, Any], intent: RuntimeIntent, q_norm: str, *, reuse_history_index: int
) -> bool:
    """Return True when direct SQL reuse must ask the user instead of auto-accepting."""
    return not should_auto_accept_for_question(ref_tmpl, q_norm, reuse_history_index=reuse_history_index)


@dataclass(frozen=True)
class PathSelectionState:
    """Structured inputs for :func:`_choose_generation_path` after union SQL path resolution."""

    has_matched_template: bool
    resolved_union_path: GenerationPath | None
    matched_template_id: str
    structural_matches: int
    cols_changed: bool
    retry_depth: int


def _choose_generation_path(state: PathSelectionState) -> GenerationPath:
    """Return the SQL generation branch: template-driven paths ``3``/``4`` when a template is in play, else ``5`` (fresh)."""
    if not state.has_matched_template:
        return GenerationPath.FRESH
    return state.resolved_union_path or GenerationPath.FRESH


def align_template_to_widened_intent(template: Template, intent: RuntimeIntent, dialect: Any) -> None:
    """Copy widened SQL artifacts and identity fields from *intent* onto. *template*. Used after union paths ``4.1`` and ``4.2`` so execution SQL caches match widened projections."""
    all_pv = dict(flatten_param_values(intent))
    template.sql_param = intent.sql_param or template.sql_param
    sig_id = (
        template.intent_signature.intent_id
        if template.intent_signature and template.intent_signature.intent_id
        else "union"
    )
    template.intent_signature = runtime_intent_to_concrete(intent, sig_id)
    template.intent_key = intent_key(intent)
    member_source_id = str(getattr(template, "member_source_id", "") or "") or None
    sg_dialect = sqlglot_dialect_for_template_fingerprint(dialect, member_source_id)
    template.sql_fp = compute_sql_fp(template.sql_param or "", sqlglot_dialect=sg_dialect)
    template.structural_defaults = {k: v for k, v in all_pv.items() if is_structural_param_key(k)}
    sig_aliases: dict[str, str] = {}
    for sc in intent.select_cols or []:
        alias = generate_col_alias(sc)
        if alias:
            sig_aliases[sc.signature_key] = alias
    template.display_alias_map = {**template.display_alias_map, **sig_aliases}


def _sql_validation_refusal_outcome(
    exc: Exception,
    *,
    generation_path: GenerationPath,
    matched_template: Template | None,
    structural_match_templates: tuple[Template, ...] | list[Template] | None,
) -> SqlGenerationOutcome:
    """Return a failed SQL-generation outcome with a stable refusal diagnostic when applicable."""
    refusal_code = refusal_diagnostic_code_for_exception(exc)
    message = refusal_message_for_exception(exc)
    if refusal_code:
        emit_session_refusal_diagnostic(refusal_code, message)
    structural_tpl = tuple(structural_match_templates or ())
    return SqlGenerationOutcome(
        "",
        False,
        generation_path,
        matched_template,
        structural_tpl,
        sql_validation_error=message,
        join_matches_template=None,
        error_kind=None,
        refusal_diagnostic_code=refusal_code,
    )


def _join_path_failure_outcome(
    exc: NoJoinPathError,
    *,
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    store: dict[str, Any] | TemplateStoreView,
    generation_path: GenerationPath,
    matched_template: Template | None,
    structural_match_templates: tuple[Template, ...] | list[Template] | None,
    persist_template_learning: bool,
) -> SqlGenerationOutcome:
    """Record join feedback, print the join-path hint, and return a failed outcome."""
    debug(f"[{ASK_PHASE_K}] {exc}")
    record_deterministic_join_failure_feedback(store, q_norm, exc, intent=intent, schema=schema)
    print_rephrase_hint(RephraseHint.JOIN_PATH_UNAVAILABLE)
    if persist_template_learning:
        save_template_store(store)
    refusal_code = refusal_diagnostic_code_for_exception(exc)
    if refusal_code:
        emit_session_refusal_diagnostic(refusal_code, exc.user_message)
    structural_tpl = tuple(structural_match_templates or ())
    return SqlGenerationOutcome(
        "",
        False,
        generation_path,
        matched_template,
        structural_tpl,
        sql_validation_error=exc.user_message,
        join_matches_template=None,
        error_kind=None,
        refusal_diagnostic_code=refusal_code,
    )


def generate_and_validate_sql(
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    join_candidates: dict[str, Any],
    cmap: dict[str, list[str]],
    dialect: Any,
    store: dict[str, Any] | TemplateStoreView,
    cte_join_hints: dict[str, dict[str, Any]] | None = None,
    matched_template: Template | None = None,
    union_select_cols: list[SelectCol] | None = None,
    cols_changed: bool = False,
    structural_match_templates: list[Template] | None = None,
    union_sql_path: GenerationPath | None = None,
    persist_template_learning: bool = True,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
    schema_role: str = "owner",
    context_name: str = MASTER_AETHERSPACE_NAME,
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    member_source_id: str | None = None,
    allowed_where_ops: frozenset[str] | None = None,
    join_preset_scope: dict[str, str] | None = None,
    max_query_cost_rows: float | None = None,
    max_query_cost_bytes: float | None = None,
    profile_timeout_ms: int | None = None,
) -> SqlGenerationOutcome:
    """Generate SQL from template reuse or deterministic build, then. validate once."""
    emit_ask_phase(ASK_PHASE_J, source=member_source_id)
    intent = prune_unused_cte_steps(intent)
    structural_tpl = tuple(structural_match_templates or ())
    if getattr(intent, "schema_invalid", False):
        print_rephrase_hint(RephraseHint.SCHEMA_INVALID_DECLINED)
        if persist_template_learning:
            save_template_store(store)
        return SqlGenerationOutcome(
            "",
            False,
            GenerationPath.INTENT_DIRECT_MATCH,
            None,
            structural_tpl,
            sql_validation_error="intent schema_invalid",
            error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
        )
    space_tables = frozenset(space_allowed_tables or ())
    space_columns = frozenset(space_allowed_columns or ())
    if space_tables or space_columns:
        if not assert_intent_in_scope(intent, space_tables, space_columns, schema):
            return SqlGenerationOutcome(
                "",
                False,
                GenerationPath.INTENT_DIRECT_MATCH,
                None,
                structural_tpl,
                sql_validation_error="intent out of aetherspace scope",
                error_kind=FailureCategory.DENIED_REFERENCE.value,
            )
    scope_ctx = schema_context if schema_context is not None else EngineContext()
    if _execution_scope_gate_active(scope_ctx, visible_objects, schema_role, context_name=context_name):
        if not assert_consumer_intent_in_scope(intent, scope_ctx, schema, visible_objects):
            return SqlGenerationOutcome(
                "",
                False,
                GenerationPath.INTENT_DIRECT_MATCH,
                None,
                structural_tpl,
                sql_validation_error="intent out of execution scope",
                error_kind=FailureCategory.ACCESS_POLICY.value,
            )
    if allowed_where_ops:
        for fp in where_leaves(intent.where) or []:
            if fp.op not in allowed_where_ops:
                return SqlGenerationOutcome(
                    "",
                    False,
                    GenerationPath.INTENT_DIRECT_MATCH,
                    None,
                    structural_tpl,
                    sql_validation_error=(f"filter operator {fp.op!r} is not supported by federation members"),
                    error_kind=FailureCategory.DENIED_REFERENCE.value,
                )
        for hp in having_leaves(intent.having) or []:
            if hp.op not in allowed_where_ops:
                return SqlGenerationOutcome(
                    "",
                    False,
                    GenerationPath.INTENT_DIRECT_MATCH,
                    None,
                    structural_tpl,
                    sql_validation_error=(f"having operator {hp.op!r} is not supported by federation members"),
                    error_kind=FailureCategory.DENIED_REFERENCE.value,
                )
    join_candidates, cmap, cte_join_hints = generate_join_candidates(intent, schema)
    debug("sql generation")
    debug(f"[{ASK_PHASE_K}] tables={intent.tables or []}")
    debug(f"[{ASK_PHASE_K}] grain={intent.grain or 'unknown'}")
    debug(f"[{ASK_PHASE_K}] select_cols={[s.expr.primary_term for s in (intent.select_cols or [])]}")
    debug(f"[{ASK_PHASE_K}] where={len(where_leaves(intent.where) or [])}")
    debug(f"[{ASK_PHASE_K}] having={len(having_leaves(intent.having) or [])}")
    debug(f"[{ASK_PHASE_K}] cte_join_hints={list(cte_join_hints.keys()) if cte_join_hints else None}")
    resolved_union_path = resolve_sql_path(
        matched_template=matched_template, cols_changed=cols_changed, union_sql_path=union_sql_path
    )
    routing = _choose_generation_path(
        PathSelectionState(
            has_matched_template=matched_template is not None,
            resolved_union_path=resolved_union_path,
            matched_template_id=(matched_template.id if matched_template else ""),
            structural_matches=len(structural_tpl),
            cols_changed=cols_changed,
            retry_depth=0,
        )
    )
    active_path = routing
    debug(f"[{ASK_PHASE_K}] generation path={active_path}")

    structural_defaults_src: dict[str, Any] | None = None
    if matched_template:
        tmpl_sd = getattr(matched_template, "structural_defaults", None)
        structural_defaults_src = tmpl_sd if tmpl_sd else None

    params = dict(flatten_param_values(intent))
    debug(f"[{ASK_PHASE_K}] params={params}")

    prior_join_fb = lookup_join_feedback_for_question(
        cast(dict[str, Any], store), q_norm, member_source_id=member_source_id
    )

    generation_path_label = active_path
    matched_for_outcome: Template | None = None
    fall_through_to_fresh = False

    if matched_template and routing in (
        GenerationPath.INTENT_DIRECT_MATCH,
        GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE,
        GenerationPath.UNION_TEMPLATE_WIDEN,
        GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
    ):
        live_ok, stale_reasons = template_is_live(template_schema_refs(matched_template), schema)
        if not live_ok:
            debug(f"[{ASK_PHASE_K}] template_not_live: {','.join(stale_reasons)}")
            matched_template = None
            routing = GenerationPath.FRESH
            active_path = GenerationPath.FRESH
            fall_through_to_fresh = True

    if (
        matched_template
        and routing
        in (
            GenerationPath.INTENT_DIRECT_MATCH,
            GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE,
        )
        and not fall_through_to_fresh
    ):
        generation_path_label = resolved_union_path
        matched_for_outcome = matched_template
        tpl_sql_param = matched_template.sql_param
        merge_structural_defaults_for_reuse(
            tpl_sql_param, params, getattr(matched_template, "structural_defaults", None)
        )
        intent.param_values = dict(intent.param_values or {})
        for k, v in params.items():
            intent.param_values.setdefault(k, v)
        if resolved_union_path == GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE and union_select_cols:
            intent.select_cols = list(union_select_cols)
        anchor_main, anchor_cte = _join_signatures_for_deterministic_from_anchor(cmap, cte_join_hints, intent)
        deterministic_sql = build_deterministic_sql(
            intent,
            cte_join_hints,
            schema,
            dialect,
            join_signature_for_from_anchor=anchor_main if anchor_main else None,
            cte_join_signatures_for_from_anchor=anchor_cte if anchor_cte else None,
        )
        pipeline_trace("pipeline.generate_and_validate_sql.deterministic_sql.path_3", lambda: deterministic_sql)
        effective_join_preset = dict(join_preset_scope or {})
        effective_join_preset.update(_join_preset_scope_from_concrete(matched_template.intent_signature))
        try:
            sql_param, _cte_join_ids = _resolve_joins_fresh(
                deterministic_sql,
                intent,
                cmap,
                cte_join_hints,
                q_norm,
                join_candidates,
                schema=schema,
                structural_defaults=structural_defaults_src,
                dialect=dialect,
                prior_join_feedback=prior_join_fb,
                join_preset_scope=effective_join_preset or None,
            )
        except NoJoinPathError as exc:
            return _join_path_failure_outcome(
                exc,
                q_norm=q_norm,
                intent=intent,
                schema=schema,
                store=store,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
                persist_template_learning=persist_template_learning,
            )
        except (
            AggregateJoinFanOutError,
            ClauseWidenedRowsetError,
            ComparisonJoinScopeExceededError,
            JoinInjectionAlignmentError,
            JoinInjectionFailedError,
            ProbeCtePlacementError,
        ) as exc:
            debug(f"[{ASK_PHASE_K}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                save_template_store(store)
            return _sql_validation_refusal_outcome(
                exc,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
            )
        intent.sql_param = sql_param
        subs_params = dict(flatten_param_values(intent))
        sql = finalize_substitute_sql(intent, structural_defaults_src=structural_defaults_src, params=subs_params)
        jm3 = _join_matches_template_intent(matched_template, intent)
        debug(f"[{ASK_PHASE_K}] path 3: deterministic sql with fresh joins")
        path_3_payload = {
            "chosen_join_candidate_id": intent.chosen_join_candidate_id,
            "chosen_join_path_signature": intent.chosen_join_path_signature,
            "deterministic_sql": deterministic_sql,
            "sql_param": sql_param,
            "sql_substituted": sql,
            "join_matches_template": jm3,
        }
        pipeline_trace("pipeline.generate_and_validate_sql.path_3", lambda: stable_json(path_3_payload))
        ok_c, err_c, cat_c, diags_c = _run_sql_validation_cascade(
            sql,
            intent,
            dialect,
            schema=schema,
            max_query_cost_rows=max_query_cost_rows,
            max_query_cost_bytes=max_query_cost_bytes,
            profile_timeout_ms=profile_timeout_ms,
        )
        if ok_c:
            return SqlGenerationOutcome(
                sql,
                True,
                resolved_union_path,
                matched_template,
                structural_tpl,
                None,
                jm3,
                None,
                sum(1 for d in diags_c if d.code.value in SOFT_DIAGNOSTIC_CODES),
            )
        debug(
            f"[{ASK_PHASE_K}] template path {resolved_union_path.code} "
            f"SQL validation failed: {err_c}; falling through to fresh"
        )
        fall_through_to_fresh = True

    if (
        matched_template
        and routing
        in (
            GenerationPath.UNION_TEMPLATE_WIDEN,
            GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
        )
        and not fall_through_to_fresh
    ):
        generation_path_label = resolved_union_path
        matched_for_outcome = matched_template
        gen_intent = replace(intent, select_cols=list(union_select_cols or []), param_values=dict(intent.param_values))
        gen_intent = extract_structural_params(gen_intent)
        params = flatten_param_values(gen_intent)
        intent.param_values = gen_intent.param_values
        anchor_main, anchor_cte = _join_signatures_for_deterministic_from_anchor(cmap, cte_join_hints, gen_intent)
        deterministic_sql = build_deterministic_sql(
            gen_intent,
            cte_join_hints,
            schema,
            dialect,
            join_signature_for_from_anchor=anchor_main if anchor_main else None,
            cte_join_signatures_for_from_anchor=anchor_cte if anchor_cte else None,
        )
        pipeline_trace("pipeline.generate_and_validate_sql.deterministic_sql.path_4", lambda: deterministic_sql)
        try:
            sql_param, _cte_join_ids = _resolve_joins_fresh(
                deterministic_sql,
                gen_intent,
                cmap,
                cte_join_hints,
                q_norm,
                join_candidates,
                schema=schema,
                structural_defaults=structural_defaults_src,
                dialect=dialect,
                prior_join_feedback=prior_join_fb,
                join_preset_scope=join_preset_scope,
            )
        except NoJoinPathError as exc:
            return _join_path_failure_outcome(
                exc,
                q_norm=q_norm,
                intent=gen_intent,
                schema=schema,
                store=store,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
                persist_template_learning=persist_template_learning,
            )
        except (
            AggregateJoinFanOutError,
            ClauseWidenedRowsetError,
            ComparisonJoinScopeExceededError,
            JoinInjectionAlignmentError,
            JoinInjectionFailedError,
            ProbeCtePlacementError,
        ) as exc:
            debug(f"[{ASK_PHASE_K}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                save_template_store(store)
            return _sql_validation_refusal_outcome(
                exc,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
            )
        intent.chosen_join_candidate_id = gen_intent.chosen_join_candidate_id
        intent.chosen_join_path_signature = list(gen_intent.chosen_join_path_signature)
        intent.sql_param = sql_param
        sql = finalize_substitute_sql(intent, structural_defaults_src=structural_defaults_src, params=dict(params))
        debug(f"[{ASK_PHASE_K}] path 4: rebuilt deterministic SQL with union cols")
        path_4_final = {
            "chosen_join_candidate_id": intent.chosen_join_candidate_id,
            "chosen_join_path_signature": intent.chosen_join_path_signature,
            "sql_param": sql_param,
            "sql_substituted": sql,
        }
        pipeline_trace("pipeline.generate_and_validate_sql.path_4.final", lambda: stable_json(path_4_final))
        if resolved_union_path == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN and union_select_cols:
            intent.select_cols = list(union_select_cols)

    elif fall_through_to_fresh or not (
        matched_template
        and routing
        in (
            GenerationPath.INTENT_DIRECT_MATCH,
            GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE,
            GenerationPath.UNION_TEMPLATE_WIDEN,
            GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
        )
    ):
        generation_path_label = GenerationPath.FRESH
        matched_for_outcome = None
        anchor_main, anchor_cte = _join_signatures_for_deterministic_from_anchor(cmap, cte_join_hints, intent)
        deterministic_sql = build_deterministic_sql(
            intent,
            cte_join_hints,
            schema,
            dialect,
            join_signature_for_from_anchor=anchor_main if anchor_main else None,
            cte_join_signatures_for_from_anchor=anchor_cte if anchor_cte else None,
        )
        pipeline_trace("pipeline.generate_and_validate_sql.deterministic_sql.path_5", lambda: deterministic_sql)
        try:
            sql_param, cte_join_ids = _resolve_joins_fresh(
                deterministic_sql,
                intent,
                cmap,
                cte_join_hints,
                q_norm,
                join_candidates,
                schema=schema,
                structural_defaults=structural_defaults_src,
                dialect=dialect,
                prior_join_feedback=prior_join_fb,
                join_preset_scope=join_preset_scope,
            )
        except NoJoinPathError as exc:
            return _join_path_failure_outcome(
                exc,
                q_norm=q_norm,
                intent=intent,
                schema=schema,
                store=store,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
                persist_template_learning=persist_template_learning,
            )
        except (
            AggregateJoinFanOutError,
            ClauseWidenedRowsetError,
            ComparisonJoinScopeExceededError,
            JoinInjectionAlignmentError,
            JoinInjectionFailedError,
            ProbeCtePlacementError,
        ) as exc:
            debug(f"[{ASK_PHASE_K}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                save_template_store(store)
            return _sql_validation_refusal_outcome(
                exc,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
            )
        pipeline_trace(
            "pipeline.generate_and_validate_sql.after_resolve_joins_fresh",
            lambda: stable_json(
                {
                    "chosen_join_candidate_id": intent.chosen_join_candidate_id,
                    "chosen_join_path_signature": intent.chosen_join_path_signature,
                    "cte_join_ids": cte_join_ids,
                    "sql_param_before_normalize": sql_param,
                }
            ),
        )
        intent.sql_param = sql_param
        sql = finalize_substitute_sql(intent, structural_defaults_src=structural_defaults_src, params=params)
        debug(f"[{ASK_PHASE_K}] path 5: fresh deterministic SQL")
        path_5_final = {
            "chosen_join_candidate_id": intent.chosen_join_candidate_id,
            "chosen_join_path_signature": intent.chosen_join_path_signature,
            "sql_param": sql_param,
            "sql_substituted": sql,
        }
        pipeline_trace("pipeline.generate_and_validate_sql.path_5.final", lambda: stable_json(path_5_final))

    ok, err, vcat, vdiags = _run_sql_validation_cascade(
        sql,
        intent,
        dialect,
        schema=schema,
        max_query_cost_rows=max_query_cost_rows,
        max_query_cost_bytes=max_query_cost_bytes,
        profile_timeout_ms=profile_timeout_ms,
    )

    if not ok:
        repaired_intent, did_repair = apply_diagnostic_repairs(intent, schema, vdiags)
        if did_repair:
            try:
                anchor_main_r, anchor_cte_r = _join_signatures_for_deterministic_from_anchor(
                    cmap, cte_join_hints, repaired_intent
                )
                deterministic_sql_r = build_deterministic_sql(
                    repaired_intent,
                    cte_join_hints,
                    schema,
                    dialect,
                    join_signature_for_from_anchor=(anchor_main_r if anchor_main_r else None),
                    cte_join_signatures_for_from_anchor=(anchor_cte_r if anchor_cte_r else None),
                )
                sql_param_r, _cte_join_ids_r = _resolve_joins_fresh(
                    deterministic_sql_r,
                    repaired_intent,
                    cmap,
                    cte_join_hints,
                    q_norm,
                    join_candidates,
                    schema=schema,
                    structural_defaults=structural_defaults_src,
                    dialect=dialect,
                    prior_join_feedback=prior_join_fb,
                    join_preset_scope=join_preset_scope,
                )
                repaired_intent.sql_param = sql_param_r
                sql_r = finalize_substitute_sql(
                    repaired_intent, structural_defaults_src=structural_defaults_src, params=dict(params)
                )
                ok_r, err_r, vcat_r, vdiags_r = _run_sql_validation_cascade(
                    sql_r,
                    repaired_intent,
                    dialect,
                    schema=schema,
                    max_query_cost_rows=max_query_cost_rows,
                    max_query_cost_bytes=max_query_cost_bytes,
                    profile_timeout_ms=profile_timeout_ms,
                )
                if ok_r:
                    debug(f"[{ASK_PHASE_K}] B.3 diagnostic repair succeeded on retry")
                    intent = repaired_intent
                    sql = sql_r
                    vdiags = vdiags_r
                    ok = True
                    err = err_r
                    vcat = vcat_r
                else:
                    debug(f"[{ASK_PHASE_K}] B.3 retry still failed: {err_r}")
            except NoJoinPathError as exc:
                return _join_path_failure_outcome(
                    exc,
                    q_norm=q_norm,
                    intent=repaired_intent,
                    schema=schema,
                    store=store,
                    generation_path=generation_path_label,
                    matched_template=matched_for_outcome,
                    structural_match_templates=structural_tpl,
                    persist_template_learning=persist_template_learning,
                )
            except (
                AggregateJoinFanOutError,
                ClauseWidenedRowsetError,
                ComparisonJoinScopeExceededError,
                JoinInjectionAlignmentError,
                JoinInjectionFailedError,
                ProbeCtePlacementError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                debug(f"[{ASK_PHASE_K}] B.3 retry rebuild raised: {exc}")

    if not ok:
        print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
        if persist_template_learning:
            save_template_store(store)
        ek = vcat.value if vcat is not None else PIPELINE_BUG_SQL_VALIDATION
        return SqlGenerationOutcome(
            sql,
            False,
            generation_path_label,
            matched_for_outcome,
            structural_tpl,
            sql_validation_error=err or None,
            join_matches_template=_join_matches_template_intent(matched_for_outcome, intent),
            error_kind=ek,
        )

    join_matches_for_outcome: bool | None = None
    if matched_for_outcome is not None and generation_path_label in (
        GenerationPath.UNION_TEMPLATE_WIDEN,
        GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
    ):
        join_matches_for_outcome = _join_matches_template_intent(matched_for_outcome, intent)
        if join_matches_for_outcome and persist_template_learning:
            align_template_to_widened_intent(matched_for_outcome, intent, dialect)
    elif matched_for_outcome is not None:
        join_matches_for_outcome = _join_matches_template_intent(matched_for_outcome, intent)

    return SqlGenerationOutcome(
        sql,
        True,
        generation_path_label,
        matched_for_outcome,
        structural_tpl,
        None,
        join_matches_for_outcome,
        None,
        sum(1 for d in vdiags if d.code.value in SOFT_DIAGNOSTIC_CODES),
    )


def _join_signatures_for_deterministic_from_anchor(
    cmap: dict[str, list[str]], cte_join_hints: dict[str, dict[str, Any]] | None, intent: RuntimeIntent
) -> tuple[list[str], dict[str, list[str]]]:
    """Return join-path signatures when exactly one non-``J00`` candidate exists per scope."""
    main_sig: list[str] = []
    cte_sigs: dict[str, list[str]] = {}
    tbls = intent.tables or []
    if len(tbls) >= 2:
        multi = {k: v for k, v in cmap.items() if k != "J00"}
        if len(multi) == 1:
            main_sig = canonicalize_stored_join_path_signature(
                list(next(iter(multi.values()))),
                from_anchor=tbls[0] if tbls else None,
            )
    if cte_join_hints and intent.cte_steps:
        for step in intent.cte_steps:
            step_tables = step.tables or []
            if len(step_tables) < 2:
                continue
            hint = cte_join_hints.get(step.cte_name)
            if not hint:
                continue
            candidates = hint.get("candidates", [])
            non_j00 = [c for c in candidates if c.get("candidate_id") != "J00"]
            if len(non_j00) == 1:
                sig = non_j00[0].get("join_path_signature", [])
                if sig:
                    cte_sigs[step.cte_name] = canonicalize_stored_join_path_signature(
                        list(sig),
                        from_anchor=step_tables[0] if step_tables else None,
                    )
    return main_sig, cte_sigs


def _build_per_carrier_join_payloads(
    intent: RuntimeIntent,
    cte_join_hints: dict[str, dict[str, Any]] | None,
    cte_join_ids: dict[str, str],
    candidate_id: str,
    cmap: dict[str, list[str]],
    main_candidates: list[dict[str, Any]],
    multi_table: bool,
    multi_table_candidates: dict[str, list[str]],
) -> tuple[list[list[str]], list[list[str]], str]:
    """Build ``join_sigs_ordered`` and ``edge_kinds_ordered`` aligned with ``intent.cte_steps`` plus one main slot. Invariant: ``len(join_sigs_ordered) == len(intent.cte_steps or []) + 1``. Scalar-subquery CTEs contribute empty signatures; multi-table CTEs use ``cte_join_hints``. May update *cte_join_ids* and *candidate_id* when resolving fallbacks."""
    join_sigs_ordered: list[list[str]] = []
    edge_kinds_ordered: list[list[str]] = []
    for cte in intent.cte_steps or []:
        if getattr(cte, "emission", "join_table") == "scalar_subquery":
            join_sigs_ordered.append([])
            edge_kinds_ordered.append([])
            continue
        if (cte.tables or []) and len(cte.tables) >= 2:
            cte_sig: list[str] = []
            cte_kinds: list[str] = []
            if cte_join_hints and cte.cte_name in cte_join_hints:
                cte_cid = cte_join_ids.get(cte.cte_name)
                if cte_cid is None:
                    cte_cid = first_base_non_j00_candidate_id(cte_join_hints[cte.cte_name]) or "J00"
                    cte_join_ids[cte.cte_name] = cte_cid
                else:
                    cte_join_ids[cte.cte_name] = cte_cid
                for cand in cte_join_hints[cte.cte_name].get("candidates", []):
                    if cand.get("candidate_id") == cte_cid:
                        cte_sig = cand.get("join_path_signature", [])
                        cte_kinds = list(cand.get("edge_kinds", []) or [])
                        break
            join_sigs_ordered.append(cte_sig)
            edge_kinds_ordered.append(cte_kinds)
        else:
            join_sigs_ordered.append([])
            edge_kinds_ordered.append([])
    main_sig: list[str] = []
    main_kinds: list[str] = []
    out_candidate_id = candidate_id
    if multi_table:
        main_sig = list(cmap.get(out_candidate_id, []))
        for cand in main_candidates:
            if cand.get("candidate_id") == out_candidate_id:
                main_kinds = list(cand.get("edge_kinds", []) or [])
                break
        if not main_sig and multi_table_candidates:
            for fid in sorted(multi_table_candidates.keys()):
                cand_sig = cmap.get(fid, [])
                if cand_sig:
                    out_candidate_id = fid
                    main_sig = list(cand_sig)
                    main_kinds = []
                    for cand in main_candidates:
                        if cand.get("candidate_id") == out_candidate_id:
                            main_kinds = list(cand.get("edge_kinds", []) or [])
                            break
                    debug(
                        f"[{ASK_PHASE_J}] empty join signature for "
                        f"prior choice — using fallback candidate_id={out_candidate_id}"
                    )

                    def _main_join_fallback_trace(cid: str = out_candidate_id, sig: list[Any] = main_sig) -> str:
                        return stable_json(
                            {
                                "candidate_id": cid,
                                "join_path_signature": sig,
                            }
                        )

                    pipeline_trace(
                        "pipeline._build_per_carrier_join_payloads.main_join_fallback", _main_join_fallback_trace
                    )
                    break
    join_sigs_ordered.append(main_sig)
    edge_kinds_ordered.append(main_kinds)
    return join_sigs_ordered, edge_kinds_ordered, out_candidate_id


def _resolve_joins_fresh(
    deterministic_sql: str,
    intent: RuntimeIntent,
    cmap: dict[str, list[str]],
    cte_join_hints: dict[str, dict[str, Any]] | None,
    q_norm: str,
    join_candidates: dict[str, Any],
    schema: SchemaGraph | None = None,
    structural_defaults: dict[str, Any] | None = None,
    dialect: Any | None = None,
    prior_join_feedback: list[str] | None = None,
    join_preset_scope: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Choose join candidate(s) and inject join SQL into deterministic. SQL."""
    cte_join_ids: dict[str, str] = {}

    main_needs_join = bool(intent.tables) and len(intent.tables) > 1
    cte_needs_join = any(
        getattr(cte, "emission", "join_table") != "scalar_subquery" and cte.tables and len(cte.tables) > 1
        for cte in (intent.cte_steps or [])
    )
    if not main_needs_join and not cte_needs_join:
        sql_param = deterministic_sql
        intent.chosen_join_candidate_id = "J00"
        intent.chosen_join_path_signature = []
        intent.resolved_join_tables = list(intent.tables or [])
        for cte_step in intent.cte_steps or []:
            cte_step.resolved_join_tables = list(cte_step.tables or [])
        pipeline_trace(
            "pipeline._resolve_joins_fresh.no_join_required",
            lambda: stable_json(
                {
                    "sql_param": sql_param,
                    "chosen_join_candidate_id": intent.chosen_join_candidate_id,
                    "chosen_join_path_signature": intent.chosen_join_path_signature,
                    "cte_join_ids": cte_join_ids,
                }
            ),
        )
        return sql_param, cte_join_ids

    if schema is not None:
        virtual_specs = build_virtual_table_specs(intent, schema)
        scope_main = tables_in_join_scope(intent.tables, schema, virtual_specs)
        main_tables_list = list(scope_main)
    else:
        virtual_specs = {}
        main_tables_list = list(intent.tables or [])
    scalar_cte_names_in_intent: set[str] = {
        cte.cte_name
        for cte in (intent.cte_steps or [])
        if (cte.grain or "row_level") == "scalar"
        and getattr(cte, "emission", "join_table") == "join_table"
        and cte.cte_name
    }
    if scalar_cte_names_in_intent:
        main_tables_list = [t for t in main_tables_list if t not in scalar_cte_names_in_intent]
    multi_table = len(main_tables_list) >= 2

    for cte in intent.cte_steps or []:
        if getattr(cte, "emission", "join_table") == "scalar_subquery":
            continue
        if not cte.tables or len(cte.tables) < 2:
            continue
        cte_hints = (cte_join_hints or {}).get(cte.cte_name) or {}
        cte_cids = {
            c.get("candidate_id") for c in cte_hints.get("candidates", []) if isinstance(c.get("candidate_id"), str)
        }
        if not (cte_cids - {"J00"}):
            debug(f"[{ASK_PHASE_J}] CTE '{cte.cte_name}' has no FK or semantic join path → NoJoinPathError")
            raise NoJoinPathError(f"CTE '{cte.cte_name}'", list(cte.tables))

    main_candidates = list(join_candidates.get("candidates") or [])
    cte_scopes: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    for cte in intent.cte_steps or []:
        if getattr(cte, "emission", "join_table") == "scalar_subquery":
            continue
        if not cte.tables or len(cte.tables) < 2:
            continue
        hints_entry = (cte_join_hints or {}).get(cte.cte_name) or {}
        cands = list(hints_entry.get("candidates") or [])
        if schema is not None:
            tbls = list(tables_in_join_scope(cte.tables, schema, virtual_specs))
        else:
            tbls = list(cte.tables or [])
        cte_scopes.append((cte.cte_name, tbls, cands))

    preset, pass1_llm, accept_na_map, scope_class = join_scope_pass1_plan(
        main_multi_table=multi_table,
        main_tables=main_tables_list,
        main_candidates=main_candidates,
        cte_scopes=cte_scopes,
        forbid_na=schema is None,
    )
    if join_preset_scope:
        preset.update(join_preset_scope)
        resolved = frozenset(join_preset_scope)
        pass1_llm = [scope for scope in pass1_llm if str(scope.get("scope")) not in resolved]

    params_full = dict(flatten_param_values(intent))
    det_for_llm, _ = reduce_structural_sql_placeholders(deterministic_sql, params_full, structural_defaults)

    merged_scope: dict[str, str] = dict(preset)
    if pass1_llm:
        accept_slice = {str(s["scope"]): accept_na_map[str(s["scope"])] for s in pass1_llm if s.get("scope")}
        merged_scope = get_join_choice_from_llm(
            q_norm,
            det_for_llm,
            llm_scopes=pass1_llm,
            preset_choices=preset,
            accept_na_by_scope=accept_slice,
            require_final=False,
            schema=schema,
            prior_join_feedback=prior_join_feedback,
        )

    pass1_keys = frozenset(str(s["scope"]) for s in pass1_llm if s.get("scope"))
    na_keys = frozenset(sk for sk in pass1_keys if merged_scope.get(sk) == "NA" and accept_na_map.get(sk, False))
    if na_keys and schema is not None:
        join_candidates = {
            **join_candidates,
            "candidates": list(join_candidates.get("candidates") or []),
        }
        hints_in = dict(cte_join_hints or {})
        gen_main, gen_cte = merge_join_hints_for_na_scopes(
            join_candidates, hints_in, intent, schema, virtual_specs, na_keys
        )
        join_candidates["candidates"] = gen_main["candidates"]
        hints_for_pass2 = dict(hints_in)
        for k, v in gen_cte.items():
            hints_for_pass2[k] = v
        cmap = join_candidate_map(join_candidates)
        pass2_llm = join_scope_pass2_llm_scopes(
            na_keys, join_candidates, hints_for_pass2, intent, schema, virtual_specs
        )
        preset2 = {k: v for k, v in merged_scope.items() if v != "NA"}
        accept2 = {str(s["scope"]): False for s in pass2_llm if s.get("scope")}
        merged_scope = get_join_choice_from_llm(
            q_norm,
            det_for_llm,
            llm_scopes=pass2_llm,
            preset_choices=preset2,
            accept_na_by_scope=accept2,
            require_final=True,
            schema=schema,
            prior_join_feedback=prior_join_feedback,
        )

    cmap = join_candidate_map(join_candidates)
    multi_table_candidates = {k: v for k, v in cmap.items() if k != "J00"}

    candidate_id = merged_scope.get(JOIN_CHOICE_SCOPE_MAIN, "J00")
    cte_join_ids = {}
    for cte in intent.cte_steps or []:
        if getattr(cte, "emission", "join_table") == "scalar_subquery":
            continue
        if not cte.tables or len(cte.tables) < 2:
            continue
        sk_cte = join_choice_scope_key_cte(cte.cte_name)
        if sk_cte in merged_scope:
            cte_join_ids[cte.cte_name] = merged_scope[sk_cte]

    def _fallback_main_join_id() -> str:
        fb = first_base_non_j00_candidate_id(join_candidates)
        return fb if fb else "J00"

    if multi_table and (candidate_id in (None, "NA") or candidate_id not in cmap):
        candidate_id = _fallback_main_join_id()
    if cte_join_hints:
        for cname in cte_join_hints:
            hints_c = cte_join_hints[cname]
            valid_ids = {
                c.get("candidate_id") for c in hints_c.get("candidates", []) if isinstance(c.get("candidate_id"), str)
            }
            vid = cte_join_ids.get(cname)
            if vid is None or vid == "NA" or vid not in valid_ids:
                cte_join_ids[cname] = first_base_non_j00_candidate_id(hints_c) or "J00"

    for sk, sc in scope_class.items():
        if sk == JOIN_CHOICE_SCOPE_MAIN:
            err_label = "main query"
            err_tables = list(main_tables_list)
            comp_only = list(intent.comparison_only_tables or [])
            vid_final: str | None = candidate_id
        elif sk.startswith("cte:"):
            cte_nm = sk.split(":", 1)[1]
            err_label = f"CTE '{cte_nm}'"
            err_tables = list(next((t for n, t, _ in cte_scopes if n == cte_nm), []))
            matched_cte = next((c for c in intent.cte_steps or [] if c.cte_name == cte_nm), None)
            comp_only = list(matched_cte.comparison_only_tables or []) if matched_cte else []
            vid_final = cte_join_ids.get(cte_nm)
        else:
            continue
        if comp_only and set(comp_only) & set(err_tables) and sc == ScopeClass.semantic_only:
            overlap = sorted(set(comp_only) & set(err_tables))
            raise ComparisonJoinScopeExceededError(
                err_label,
                (
                    f"Cross-table comparison tables {', '.join(overlap)} can only be joined through "
                    "profile-inferred relationships. The comparison does not imply a relationship. "
                    "Declare foreign_keys_add or a semantic override when the relationship is real."
                ),
            )
        if sc != ScopeClass.semantic_only:
            continue
        if vid_final in (None, "", "NA"):
            raise NoJoinPathError(err_label, err_tables)

    if multi_table and len(main_tables_list) >= 2:
        chosen_cand = next((c for c in main_candidates if c.get("candidate_id") == candidate_id), None)
        if chosen_cand is not None and not join_candidate_spans_tables(chosen_cand, main_tables_list):
            for fid in sorted(multi_table_candidates.keys()):
                alt_cand = next((c for c in main_candidates if c.get("candidate_id") == fid), None)
                if alt_cand is not None and join_candidate_spans_tables(alt_cand, main_tables_list):
                    candidate_id = fid
                    break

    def _validate_scope_span(scope_key: str, chosen_id: str, scope_tables: list[str], hints: dict[str, Any]) -> None:
        if not scope_tables or len(scope_tables) < 2:
            return
        if chosen_id in (None, "", "NA", "J00"):
            return
        cand = next((c for c in hints.get("candidates", []) if c.get("candidate_id") == chosen_id), None)
        if cand is None:
            return
        if not join_candidate_spans_tables(cand, scope_tables):
            raise NoJoinPathError(
                scope_key if scope_key != JOIN_CHOICE_SCOPE_MAIN else "main query", list(scope_tables)
            )

    _validate_scope_span(JOIN_CHOICE_SCOPE_MAIN, candidate_id, main_tables_list, join_candidates)
    if cte_join_hints:
        for cname, hints_c in cte_join_hints.items():
            sk = join_choice_scope_key_cte(cname)
            if sk not in merged_scope:
                continue
            tbls = list(next((t for n, t, _ in cte_scopes if n == cname), []))
            _validate_scope_span(sk, merged_scope.get(sk, "J00"), tbls, hints_c)

    def _signature_for_candidate(hints: dict[str, Any], cid: str) -> list[str]:
        cand = next((c for c in hints.get("candidates", []) if c.get("candidate_id") == cid), None)
        return list(cand.get("join_path_signature", []) or []) if cand else []

    def _edge_kinds_for_candidate(hints: dict[str, Any], cid: str) -> list[str]:
        cand = next((c for c in hints.get("candidates", []) if c.get("candidate_id") == cid), None)
        return list(cand.get("edge_kinds", []) or []) if cand else []

    if multi_table:
        intent.resolved_join_tables = join_resolved_scope_tables(
            _signature_for_candidate(join_candidates, candidate_id),
            main_tables_list,
        )
        bridge_tables = sorted(set(intent.resolved_join_tables or []) - set(intent.tables or []))
        if bridge_tables:
            updated = append_table_scope_repairs(
                intent,
                scope_label="main query",
                added=bridge_tables,
                add_reason="join_bridge",
            )
            intent.table_scope_repairs = updated.table_scope_repairs
    else:
        intent.resolved_join_tables = list(intent.tables or [])
    for cte_step in intent.cte_steps or []:
        cte_tbls = list(cte_step.tables or [])
        if len(cte_tbls) >= 2 and cte_step.cte_name in cte_join_ids:
            if schema is not None:
                scope_tbls = list(tables_in_join_scope(cte_tbls, schema, virtual_specs))
            else:
                scope_tbls = cte_tbls
            hints_c = (cte_join_hints or {}).get(cte_step.cte_name) or {}
            cte_step.resolved_join_tables = join_resolved_scope_tables(
                _signature_for_candidate(hints_c, cte_join_ids[cte_step.cte_name]),
                scope_tbls,
            )
            bridge_tables = sorted(set(cte_step.resolved_join_tables or []) - set(cte_tbls))
            if bridge_tables:
                updated = append_table_scope_repairs(
                    intent,
                    scope_label=f"CTE '{cte_step.cte_name}'",
                    added=bridge_tables,
                    add_reason="join_bridge",
                )
                intent.table_scope_repairs = updated.table_scope_repairs
        else:
            cte_step.resolved_join_tables = cte_tbls

    if schema is not None:

        def _raise_if_join_scope_unreachable(tables: list[str], context: str) -> None:
            if len(tables) < 2:
                return
            issues = validate_join_path_reachability_for_tables(tables, schema, context)
            if any(i.severity == "error" for i in issues):
                raise NoJoinPathError(context, list(tables))

        _raise_if_join_scope_unreachable(list(intent.resolved_join_tables or []), "main query")
        for cte_step in intent.cte_steps or []:
            rt = list(cte_step.resolved_join_tables or [])
            if len(rt) >= 2:
                _raise_if_join_scope_unreachable(rt, f"CTE '{cte_step.cte_name}'")

        def _raise_if_aggregate_join_fan_out(
            scope_intent: RuntimeIntent,
            signature: list[str],
            context: str,
            anchor: str | None,
        ) -> None:
            issues = validate_aggregate_join_fan_out(
                scope_intent,
                schema,
                context,
                join_signature=signature,
                from_anchor=anchor,
            )
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                raise AggregateJoinFanOutError(context, errors[0].message)

        def _raise_if_clause_widened_rowset(
            scope_intent: RuntimeIntent,
            signature: list[str],
            context: str,
            anchor: str | None,
        ) -> None:
            issues = validate_clause_widened_rowset(
                scope_intent,
                schema,
                context,
                join_signature=signature,
                from_anchor=anchor,
            )
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                raise ClauseWidenedRowsetError(context, errors[0].message)

        if multi_table:
            main_sig = _signature_for_candidate(join_candidates, candidate_id)
            main_anchor = intent.tables[0] if intent.tables else None
            _raise_if_aggregate_join_fan_out(intent, main_sig, "main query", main_anchor)
            _raise_if_clause_widened_rowset(intent, main_sig, "main query", main_anchor)
        for cte_step in intent.cte_steps or []:
            cte_tbls = list(cte_step.tables or [])
            if len(cte_tbls) < 2 or cte_step.cte_name not in cte_join_ids:
                continue
            hints_c = (cte_join_hints or {}).get(cte_step.cte_name) or {}
            cte_sig = _signature_for_candidate(hints_c, cte_join_ids[cte_step.cte_name])
            cte_anchor = cte_tbls[0] if cte_tbls else None
            cte_scope_intent = RuntimeIntent(
                tables=cte_tbls,
                grain=cte_step.grain or "row_level",
                select_cols=list(cte_step.select_cols or []),
                group_by_cols=list(cte_step.group_by_cols or []),
                order_by_cols=list(cte_step.order_by_cols or []),
                where=cte_step.where,
                having=cte_step.having,
                limit=cte_step.limit,
                limit_param_key=cte_step.limit_param_key or "",
                distinct_select_index=cte_step.distinct_select_index,
                distinct_on=list(cte_step.distinct_on or []),
                chosen_join_path_signature=cte_sig,
            )
            _raise_if_aggregate_join_fan_out(
                cte_scope_intent,
                cte_sig,
                f"CTE '{cte_step.cte_name}'",
                cte_anchor,
            )
            _raise_if_clause_widened_rowset(
                cte_scope_intent,
                cte_sig,
                f"CTE '{cte_step.cte_name}'",
                cte_anchor,
            )

        def _raise_if_comparison_join_scope(
            *,
            scope_label: str,
            scope_tables: list[str],
            comparison_only: list[str],
            signature: list[str],
            edge_kinds: list[str],
            from_anchor: str | None,
            where_params: list[Any] | None = None,
            having_params: list[Any] | None = None,
        ) -> None:
            validate_comparison_join_scope_or_raise(
                scope_label=scope_label,
                scope_tables=scope_tables,
                comparison_only=comparison_only,
                signature=signature,
                edge_kinds=edge_kinds,
                from_anchor=from_anchor,
                where_params=where_params,
                having_params=having_params,
            )

        if multi_table:
            _raise_if_comparison_join_scope(
                scope_label="main query",
                scope_tables=list(main_tables_list),
                comparison_only=list(intent.comparison_only_tables or []),
                signature=_signature_for_candidate(join_candidates, candidate_id),
                edge_kinds=_edge_kinds_for_candidate(join_candidates, candidate_id),
                from_anchor=intent.tables[0] if intent.tables else None,
                where_params=where_leaves(intent.where),
                having_params=having_leaves(intent.having),
            )
        for cte_step in intent.cte_steps or []:
            cte_tbls = list(cte_step.tables or [])
            if len(cte_tbls) < 2 or cte_step.cte_name not in cte_join_ids:
                continue
            hints_c = (cte_join_hints or {}).get(cte_step.cte_name) or {}
            cid = cte_join_ids[cte_step.cte_name]
            _raise_if_comparison_join_scope(
                scope_label=f"CTE '{cte_step.cte_name}'",
                scope_tables=cte_tbls,
                comparison_only=list(cte_step.comparison_only_tables or []),
                signature=_signature_for_candidate(hints_c, cid),
                edge_kinds=_edge_kinds_for_candidate(hints_c, cid),
                from_anchor=cte_tbls[0] if cte_tbls else None,
                where_params=where_leaves(cte_step.where),
                having_params=having_leaves(cte_step.having),
            )

    if not multi_table:
        debug(f"[{ASK_PHASE_J}] single-table intent → J00")
    elif len(multi_table_candidates) == 1 and candidate_id != "J00":
        debug(f"[{ASK_PHASE_J}] resolved candidate_id={candidate_id}")
    elif pass1_llm:
        debug(f"[{ASK_PHASE_J}] LLM chose candidate_id={candidate_id}")

    join_sigs_ordered, edge_kinds_ordered, candidate_id = _build_per_carrier_join_payloads(
        intent, cte_join_hints, cte_join_ids, candidate_id, cmap, main_candidates, multi_table, multi_table_candidates
    )

    intent, join_where_dropped = drop_redundant_resolved_join_where_predicates(
        intent,
        schema,
        join_sigs_ordered=join_sigs_ordered,
        edge_kinds_ordered=edge_kinds_ordered,
    )
    if join_where_dropped:
        anchor_main, anchor_cte = _join_signatures_for_deterministic_from_anchor(cmap, cte_join_hints, intent)
        deterministic_sql = build_deterministic_sql(
            intent,
            cte_join_hints,
            schema,
            dialect,
            join_signature_for_from_anchor=anchor_main if anchor_main else None,
            cte_join_signatures_for_from_anchor=anchor_cte if anchor_cte else None,
        )

    main_sig = join_sigs_ordered[-1] if join_sigs_ordered else []
    sql_param = inject_join_into_deterministic_sql(
        deterministic_sql,
        join_sigs_ordered,
        edge_kinds_ordered=edge_kinds_ordered,
        schema=schema,
        dialect=dialect,
        cte_emissions=cte_emission_map(intent.cte_steps),
        preserve_tables=list(intent.preserve_tables or []),
        probe_cte_names=probe_cte_names(intent.cte_steps),
    )
    emit_join_orphan_rate_diagnostics(
        intent,
        schema,
        join_signature=canonicalize_stored_join_path_signature(
            list(main_sig if multi_table else cmap.get(candidate_id, []) or []),
            from_anchor=intent.tables[0] if intent.tables else None,
        ),
        edge_kinds=edge_kinds_ordered[-1] if edge_kinds_ordered else None,
        from_anchor=intent.tables[0] if intent.tables else None,
        preserve_tables=list(intent.preserve_tables or []),
    )
    intent.chosen_join_candidate_id = candidate_id
    raw_main_sig = main_sig if multi_table else cmap.get(candidate_id, [])
    intent.chosen_join_path_signature = canonicalize_stored_join_path_signature(
        list(raw_main_sig or []),
        from_anchor=intent.tables[0] if intent.tables else None,
    )
    if cte_join_ids and intent.cte_steps:
        for cte_step in intent.cte_steps:
            if cte_step.cte_name in cte_join_ids:
                cte_step.chosen_join_candidate_id = cte_join_ids[cte_step.cte_name]
                if cte_join_hints and cte_step.cte_name in cte_join_hints:
                    for cand in cte_join_hints[cte_step.cte_name].get("candidates", []):
                        if cand.get("candidate_id") == cte_step.chosen_join_candidate_id:
                            cte_step.chosen_join_path_signature = canonicalize_stored_join_path_signature(
                                list(cand.get("join_path_signature", []) or []),
                                from_anchor=(cte_step.tables[0] if cte_step.tables else None),
                            )
                            break

    enforce_probe_cte_anchor_placement_post_resolution(intent)

    pipeline_trace(
        "pipeline._resolve_joins_fresh.resolved",
        lambda: stable_json(
            {
                "candidate_id": intent.chosen_join_candidate_id,
                "chosen_join_path_signature": list(intent.chosen_join_path_signature or []),
                "resolved_join_tables": list(intent.resolved_join_tables or []),
                "join_sigs_ordered": join_sigs_ordered,
                "cte_join_ids": cte_join_ids,
                "sql_param": sql_param,
                "deterministic_sql": deterministic_sql,
            }
        ),
    )
    return sql_param, cte_join_ids


def finalize_substitute_sql(
    intent: RuntimeIntent, *, structural_defaults_src: dict[str, Any] | None, params: dict[str, Any]
) -> str:
    """Substitute bound parameters into ``intent.sql_param`` and return the executable SQL. ``intent.sql_param`` is already canonical because the compositional SQL builder emits column-left predicates from the intent layer; no post-SQL normalization is applied. Ensures ``intent.sql_param`` is non-empty when present."""
    sql_param = intent.sql_param or ""
    intent.sql_param = sql_param
    return finalize_executable_sql(sql_param, params, structural_defaults_src, sqlglot_dialect=active_sqlglot_dialect())


def stamp_sql_shape(
    sql: str,
    intent: RuntimeIntent,
    *,
    generation_path: GenerationPath | None = None,
    federated_plan: FederatedPlan | None = None,
) -> None:
    """Stamp ``intent.sql_shape`` from rendered SQL or a federated plan."""
    if generation_path is GenerationPath.FEDERATION_PLAN and federated_plan is not None:
        actual_shape = federation_plan_sql_shape(federated_plan)
    else:
        actual_shape = sql_shape(sql, intent, sqlglot_dialect=active_sqlglot_dialect())
    intent.sql_shape = actual_shape
    debug(f"sql_shape={actual_shape}")


def emit_explain_soft_diagnostics(count: int) -> None:
    """Surface EXPLAIN soft-diagnostic findings as structured diagnostics instead of a confidence penalty."""
    n = max(0, int(count))
    if n <= 0:
        return
    notify(
        f"EXPLAIN reported {n} soft diagnostic finding(s).",
        stage="validation",
        code=next(iter(SOFT_DIAGNOSTIC_CODES), DIAGNOSTIC_CODE_ENGINE_INFO),
        details=(("explain_soft_diagnostics", str(n)),),
    )


def credit_federation_accept(
    *,
    q_norm: str,
    federation_dir: str,
    plan_id: str,
    steps: Sequence[FederatedPreparedStep],
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]],
    schemas_by_source: Mapping[str, SchemaGraph] | None = None,
    form_storage: QuestionFormStorage | None = None,
    pending_plan_template: FederationPlanTemplate | None = None,
) -> None:
    """Credit accept feedback to each member template and the federation plan record."""
    path = GenerationPath.FEDERATION_PLAN
    for step in steps:
        tmpl = step.matched_template
        if tmpl is None:
            continue
        member_store = stores_by_source.get(step.source_id)
        if member_store is None:
            raise FederationConfigError(
                f"federation member store missing for source_id {step.source_id!r}; "
                "each member must have its own artifact tree"
            )
        member_templates: dict[str, Any] = {}
        if isinstance(member_store, TemplateStoreView):
            member_templates = cast(dict[str, Any], member_store["templates"])
        else:
            member_templates = cast(dict[str, Any], member_store.get("templates", {}))
        record_template_feedback(tmpl, accept=True)
        member_q = member_feedback_q_norm(step.source_id, q_norm)
        record_per_question_feedback(tmpl, member_q, accept=True, path=path_bucket(path))
        promote_trust(tmpl, member_q)
        stamp_federation_member_template(tmpl, plan_id=plan_id, source_id=step.source_id)
        member_schema = (schemas_by_source or {}).get(step.source_id)
        if member_schema is not None and not other_template_owns_question_string(member_templates, tmpl.id, member_q):
            _maybe_record_value_history_accept(
                member_templates, tmpl, step.sub_intent, member_q, form_storage, member_schema
            )
        save_template_store(member_store)
    if federation_dir and plan_id:
        member_ids = tuple(
            (step.source_id, str(step.matched_template.id))
            for step in steps
            if step.matched_template is not None and str(getattr(step.matched_template, "id", "") or "").strip()
        )
        credit_federation_plan_accept(
            federation_dir,
            plan_id,
            q_norm,
            member_template_ids=member_ids or None,
            pending_plan_template=pending_plan_template,
        )


def complete_user_feedback_reject(
    ctx: UserFeedbackRejectSuspendContext,
    *,
    needs_reason: bool,
    reject_reason: str,
    choice_port: InteractiveChoicePort | None = None,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
    federation_dir: str | None = None,
    federation_plan_id: str | None = None,
    cross_source_join_feedback: bool = False,
) -> dict[str, str] | None:
    """Persist user SQL rejection feedback into templates, rejected store, and negative memory."""
    intent = ctx.intent
    sql = ctx.sql
    schema = ctx.schema
    store: dict[str, Any] | TemplateStoreView = ctx.store
    templates = ctx.templates
    q_norm = ctx.q_norm
    matched_template = ctx.matched_template

    join_feedback_recorded = False

    if choice_port is not None:
        choice_port._pending_federation_plan_template = None

    feedback_template = matched_template

    if needs_reason:
        norm_reason = canonicalize_rejection_reason(reject_reason or "")
    else:
        norm_reason = "user_rejected"
        reject_reason = ""

    last_bucket: str | None = None

    if persist_template_learning:
        if feedback_template is not None:
            record_template_feedback(feedback_template, accept=False)
            try:
                resolved_path_for_reject = GenerationPath.parse(ctx.generation_path)
            except (KeyError, ValueError):
                resolved_path_for_reject = None
            path_bucket_value = path_bucket(resolved_path_for_reject)
            record_per_question_feedback(feedback_template, q_norm, accept=False, path=path_bucket_value)
            _, template_deleted = reject_out_per_question(templates, feedback_template, q_norm)
            entry_fb = summarize_failure_for_memory(
                question=q_norm,
                intent=intent,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash=schema.effective_structural_hash,
                user_reason=norm_reason,
            )
            if federation_plan_id and federation_dir:
                record_federation_join_feedback(str(federation_dir), str(federation_plan_id), entry_fb.summary)
                join_feedback_recorded = True
            elif (
                cross_source_join_feedback
                and federation_dir
                and federation_plan_id
                and RejectionBucket.WRONG_TABLES_OR_JOINS in entry_fb.buckets
            ):
                record_federation_join_feedback(str(federation_dir), str(federation_plan_id), entry_fb.summary)
                join_feedback_recorded = True
            else:
                record_question_feedback(store, q_norm, entry_fb)
            last_bucket = entry_fb.buckets[0].value if entry_fb.buckets else RejectionBucket.OTHER.value
            if template_deleted:
                templates_to_store(store, templates)

        else:
            entry_fb = summarize_failure_for_memory(
                question=q_norm,
                intent=intent,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash=schema.effective_structural_hash,
                user_reason=norm_reason,
            )
            if federation_plan_id and federation_dir:
                record_federation_join_feedback(str(federation_dir), str(federation_plan_id), entry_fb.summary)
                join_feedback_recorded = True
            elif (
                cross_source_join_feedback
                and federation_dir
                and federation_plan_id
                and RejectionBucket.WRONG_TABLES_OR_JOINS in entry_fb.buckets
            ):
                record_federation_join_feedback(str(federation_dir), str(federation_plan_id), entry_fb.summary)
                join_feedback_recorded = True
            else:
                record_question_feedback(store, q_norm, entry_fb)
            last_bucket = entry_fb.buckets[0].value if entry_fb.buckets else RejectionBucket.OTHER.value

        store = templates_to_store(store, templates)
        save_template_store(store)
    else:
        gpath = ctx.generation_path
        gcode = gpath.code if isinstance(gpath, GenerationPath) else str(gpath)
        mid = ctx.matched_template.id if ctx.matched_template is not None else ""
        mrej_id = getattr(ctx.matched_rejected_template, "id", "") or ""
        ctx_doc = {
            "intent": intent.to_dict(),
            "sql": sql,
            "q_norm": q_norm,
            "generation_path": gcode,
            "matched_template_id": mid,
            "matched_rejected_template_id": str(mrej_id),
            "needs_reason": bool(needs_reason),
            "reject_reason": reject_reason if needs_reason else "",
        }
        ev = WriteQueueEvent(
            kind="template_reject",
            schema_graph_id=str(schema.schema_graph_id or ""),
            schema_hash=str(schema.effective_structural_hash or ""),
            produced_at=datetime.now(timezone.utc).isoformat(),
            payload=(("ctx_json", stable_json(ctx_doc)),),
        )
        _emit_reader_write_queue_event(store, ev)
        last_bucket = RejectionBucket.OTHER.value
    ctx_ref = _refinement_ctx_for_feedback(choice_port, refinement_ctx)
    reason_line = ((reject_reason or "").strip() if needs_reason else "") or norm_reason
    if ctx_ref is not None and refinement_retry_available(ctx_ref):
        ctx_ref.accumulated_reasons.append(reason_line)
        ctx_ref.pending_retry = True
        raise RefinementRetry
    if choice_port is None:
        notify(
            "\nFeedback recorded — your note will guide the next attempt.",
            stage="pipeline",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )
        print_rephrase_hint(RephraseHint.USER_REJECTED_RESULT, rejection_bucket=last_bucket)
    fn_note = getattr(choice_port, "note_turn_outcome", None)
    if callable(fn_note):
        bk = str(last_bucket or RejectionBucket.OTHER.value).strip().upper()
        fn_note(outcome="user_declined", rejection_bucket=bk)
    if federation_plan_id and federation_dir and not join_feedback_recorded:
        delete_unaccepted_federation_plan_template(str(federation_dir), str(federation_plan_id))
    return {
        "category": str(last_bucket or RejectionBucket.OTHER.value),
        "normalized_reason": norm_reason,
        "reject_reason": reject_reason if needs_reason else "",
    }


def handle_user_feedback(
    choice: str,
    intent: RuntimeIntent,
    sql: str,
    schema: SchemaGraph,
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Any],
    rejected: dict[str, Any],
    q_norm: str,
    generation_path: GenerationPath | str,
    matched_template: Template | None,
    matched_rejected_template: Any | None,
    dialect: Any | None = None,
    structural_match_templates: tuple[Template, ...] | list[Template] | None = None,
    choice_port: InteractiveChoicePort | None = None,
    join_matches_template: bool | None = None,
    form_storage: QuestionFormStorage | None = None,
    persist_template_learning: bool = True,
    federated_steps: Sequence[FederatedPreparedStep] | None = None,
    federation_dir: str | None = None,
    federation_plan_id: str | None = None,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None = None,
    schemas_by_source: Mapping[str, SchemaGraph] | None = None,
    member_source_id: str | None = None,
    federated_plan: FederatedPlan | None = None,
    pending_plan_template: FederationPlanTemplate | None = None,
) -> dict[str, str] | None:
    """Persist accept/reject feedback into templates, rejected store, and negative memory. Accept and reject paths use *matched_template* as the sole accepted- template target when applicable; there is no ``intent_key`` re- resolution of templates."""
    if choice not in ("y", "n"):
        invalid_input("Invalid choice — please answer y or n.")
        if persist_template_learning:
            save_template_store(store)
        return None

    record_q = member_feedback_q_norm(member_source_id, q_norm) if member_source_id else q_norm

    resolved_path = GenerationPath.parse(generation_path)
    if resolved_path is GenerationPath.FEDERATION_PLAN and federated_plan is not None:
        stamp_sql_shape(sql, intent, generation_path=resolved_path, federated_plan=federated_plan)
    else:
        intent.sql_shape = sql_shape(sql, intent, sqlglot_dialect=active_sqlglot_dialect())
    feedback_template = matched_template
    if (
        resolved_path
        in (
            GenerationPath.EXACT_QUESTION_REUSE,
            GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL,
            GenerationPath.FUZZY_REUSE_FULL_PARAMS,
            GenerationPath.INTENT_DIRECT_MATCH,
            GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE,
            GenerationPath.UNION_TEMPLATE_WIDEN,
            GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
        )
        and matched_template is None
    ):
        raise RuntimeError(f"Missing matched_template for generation path {resolved_path.code} ({resolved_path.label})")

    if choice == "y":
        promoted = False
        if persist_template_learning:
            delete_rejected_templates_matching_question(cast(dict[str, Any], store), record_q)

            if resolved_path == GenerationPath.FEDERATION_PLAN:
                fed_steps = federated_steps or ()
                credit_federation_accept(
                    q_norm=q_norm,
                    federation_dir=str(federation_dir or ""),
                    plan_id=str(federation_plan_id or ""),
                    steps=fed_steps,
                    stores_by_source=stores_by_source or {},
                    schemas_by_source=schemas_by_source,
                    form_storage=form_storage,
                    pending_plan_template=pending_plan_template,
                )
                if choice_port is not None:
                    choice_port._pending_federation_plan_template = None
                promoted = True

            if matched_rejected_template is not None:
                new_tmpl = promote_rejected_to_template(
                    cast(dict[str, Any], store),
                    templates,
                    record_q,
                    intent,
                    sql,
                    schema.schema_graph_id,
                    effective_structural_hash=schema.effective_structural_hash,
                    form_storage=form_storage,
                )
                debug(f"promoted prior-negative-memory path to template {new_tmpl.id}")
                promoted = True

            if not promoted and resolved_path == GenerationPath.FRESH:
                debug(f"[{ASK_PHASE_N}] insert_template path 5")
                sm_list = list(structural_match_templates or [])
                insert_template(
                    store,
                    templates,
                    schema,
                    q_norm,
                    intent,
                    sql,
                    dialect=dialect,
                    structural_match_templates=sm_list,
                    form_storage=form_storage,
                    record_accept=True,
                    member_source_id=member_source_id,
                )
            elif not promoted and matched_template is not None and resolved_path == GenerationPath.EXACT_QUESTION_REUSE:
                tmpl = matched_template
                record_template_feedback(tmpl, accept=True)
                record_per_question_feedback(tmpl, record_q, accept=True, path=path_bucket(resolved_path))
                promote_trust(tmpl, record_q)
                if not other_template_owns_question_string(templates, tmpl.id, record_q):
                    _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif (
                not promoted
                and matched_template is not None
                and resolved_path
                in (GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL, GenerationPath.FUZZY_REUSE_FULL_PARAMS)
            ):
                tmpl = matched_template
                record_template_feedback(tmpl, accept=True)
                record_per_question_feedback(tmpl, record_q, accept=True, path=path_bucket(resolved_path))
                promote_trust(tmpl, record_q)
                if not other_template_owns_question_string(templates, tmpl.id, record_q):
                    _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif not promoted and matched_template is not None and resolved_path == GenerationPath.INTENT_DIRECT_MATCH:
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    insert_template(
                        store,
                        templates,
                        schema,
                        q_norm,
                        intent,
                        sql,
                        dialect=dialect,
                        structural_match_templates=sm_list,
                        form_storage=form_storage,
                        record_accept=True,
                        member_source_id=member_source_id,
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, record_q, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif (
                not promoted
                and matched_template is not None
                and resolved_path == GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE
            ):
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    insert_template(
                        store,
                        templates,
                        schema,
                        q_norm,
                        intent,
                        sql,
                        dialect=dialect,
                        structural_match_templates=sm_list,
                        form_storage=form_storage,
                        record_accept=True,
                        member_source_id=member_source_id,
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, record_q, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif not promoted and matched_template is not None and resolved_path == GenerationPath.UNION_TEMPLATE_WIDEN:
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    insert_template(
                        store,
                        templates,
                        schema,
                        q_norm,
                        intent,
                        sql,
                        dialect=dialect,
                        structural_match_templates=sm_list,
                        form_storage=form_storage,
                        record_accept=True,
                        member_source_id=member_source_id,
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, record_q, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
                    old_skeleton = concrete_intent_to_runtime_skeleton(tmpl.intent_signature)
                    new_skeleton = cleared_param_runtime_intent(intent)
                    key_remap = _structural_key_remap_from_assignment_order(old_skeleton, new_skeleton)
                    _remap_value_history_structural_keys(tmpl.value_history, key_remap)
                    align_template_to_widened_intent(tmpl, intent, dialect)
                    reconcile_template_store_until_stable(
                        templates, template_store_view=(store if isinstance(store, TemplateStoreView) else None)
                    )
            elif (
                not promoted
                and matched_template is not None
                and resolved_path == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN
            ):
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    insert_template(
                        store,
                        templates,
                        schema,
                        q_norm,
                        intent,
                        sql,
                        dialect=dialect,
                        structural_match_templates=sm_list,
                        form_storage=form_storage,
                        record_accept=True,
                        member_source_id=member_source_id,
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, record_q, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
                    old_skeleton = concrete_intent_to_runtime_skeleton(tmpl.intent_signature)
                    new_skeleton = cleared_param_runtime_intent(intent)
                    key_remap = _structural_key_remap_from_assignment_order(old_skeleton, new_skeleton)
                    _remap_value_history_structural_keys(tmpl.value_history, key_remap)
                    align_template_to_widened_intent(tmpl, intent, dialect)
                    reconcile_template_store_until_stable(
                        templates, template_store_view=(store if isinstance(store, TemplateStoreView) else None)
                    )

            store = templates_to_store(store, templates)
            save_template_store(store)
        else:
            replay = {
                "generation_path": resolved_path.code,
                "q_norm": q_norm,
                "sql": sql,
                "intent": intent.to_dict(),
                "matched_template_id": (matched_template.id if matched_template is not None else ""),
                "matched_rejected_id": getattr(matched_rejected_template, "id", "") or "",
                "join_matches": join_matches_template,
                "structural_ids": ",".join(t.id for t in (structural_match_templates or []) if getattr(t, "id", None)),
                "form_storage": (asdict(form_storage) if form_storage is not None else None),
                "promoted_from_rejected": matched_rejected_template is not None,
            }
            ev = WriteQueueEvent(
                kind="template_accept",
                schema_graph_id=str(schema.schema_graph_id or ""),
                schema_hash=str(schema.effective_structural_hash or ""),
                produced_at=datetime.now(timezone.utc).isoformat(),
                payload=(("replay_json", stable_json(replay)),),
            )
            _emit_reader_write_queue_event(store, ev)

        return None

    else:
        needs_reason = feedback_template is None
        ctx_rej = UserFeedbackRejectSuspendContext(
            intent=intent,
            sql=sql,
            schema=schema,
            store=cast(dict[str, Any], store),
            templates=templates,
            rejected=rejected,
            q_norm=record_q,
            generation_path=generation_path,
            matched_template=matched_template,
            matched_rejected_template=matched_rejected_template,
            dialect=dialect,
            structural_match_templates=structural_match_templates,
        )
        if needs_reason:
            if choice_port is not None and not choice_port.has_pending_choice():
                raise PipelineSuspended(PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT, "What was wrong?", ctx_rej)
            try:
                print_info(
                    "What was wrong?",
                    footer=(
                        "Tip: a single sentence is enough — for example 'wrong table', "
                        "'missing date filter', or 'should aggregate by month'."
                    ),
                )
                if choice_port is not None:
                    cons = getattr(choice_port, "_consume_next_queued_choice", None)
                    if not callable(cons):
                        raise TypeError("choice port must implement _consume_next_queued_choice for free-text feedback")
                    reject_reason = (cons() or "").strip()
                else:
                    reject_reason = prompt("").strip()
            except (EOFError, KeyboardInterrupt):
                terminated()
                if persist_template_learning:
                    save_template_store(store)
                return None
            if not reject_reason:
                invalid_input()
                if persist_template_learning:
                    save_template_store(store)
                return None
            return complete_user_feedback_reject(
                ctx_rej,
                needs_reason=True,
                reject_reason=reject_reason,
                choice_port=choice_port,
                persist_template_learning=persist_template_learning,
                federation_dir=federation_dir,
                federation_plan_id=federation_plan_id,
                cross_source_join_feedback=resolved_path == GenerationPath.FEDERATION_PLAN,
            )
        return complete_user_feedback_reject(
            ctx_rej,
            needs_reason=False,
            reject_reason="",
            choice_port=choice_port,
            persist_template_learning=persist_template_learning,
            federation_dir=federation_dir,
            federation_plan_id=federation_plan_id,
            cross_source_join_feedback=resolved_path == GenerationPath.FEDERATION_PLAN,
        )


def _most_frequent_natural_language(vh: ValueHistory) -> str:
    """Get the most frequently occurring natural_language from a. ValueHistory."""
    if not vh.natural_language:
        return ""
    non_empty = [nl for nl in vh.natural_language if nl]
    if not non_empty:
        return ""
    counts = Counter(non_empty)
    return counts.most_common(1)[0][0]


def extract_column_headers(sql: str) -> list[str]:
    """Parse the outermost ``SELECT`` projection list and return. display. column names / aliases via AST."""
    return sql_outer_select_aliases(sql, sqlglot_dialect=active_sqlglot_dialect())


def _federated_result_column_headers(
    *,
    row_width: int,
    column_names: Sequence[str] | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
    federated_plan: FederatedPlan | None = None,
) -> tuple[str, ...] | None:
    """Resolve federated display column names from structured plan or bundle metadata."""
    if column_names and len(column_names) == row_width:
        return tuple(str(c) for c in column_names)
    if federated_bundle is not None and federated_bundle.column_names:
        bundle_hdrs = tuple(str(c) for c in federated_bundle.column_names)
        if len(bundle_hdrs) == row_width:
            return bundle_hdrs
    if federated_plan is not None:
        residual_hdr = federation_residual_column_headers(federated_plan)
        if residual_hdr and len(residual_hdr) == row_width:
            return residual_hdr
    return None


def intent_result_column_headers(
    intent: RuntimeIntent,
    *,
    row_width: int | None = None,
    template_display_alias_map: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Derive display column names from intent projection without parsing SQL."""
    if not intent.select_cols:
        return ()
    headers: list[str] = []
    display_aliases = dict(template_display_alias_map or {})
    column_map = dict(intent.column_map or {})
    for sc in intent.select_cols:
        name = ""
        col_ref = (sc.expr.column_ref or sc.expr.primary_column or sc.expr.primary_term or "").strip()
        registry_id = (expr_registry_ref(sc.expr) or "").strip()
        if registry_id and registry_id in display_aliases:
            name = display_aliases[registry_id].strip()
        if not name and (sc.output_alias or "").strip():
            name = (sc.output_alias or "").strip()
        if not name and col_ref and col_ref in column_map:
            name = str(column_map[col_ref]).strip()
        if not name:
            name = generate_col_alias(sc).strip()
        if not name and col_ref:
            name = col_ref.rsplit(".", 1)[-1]
        headers.append(name or f"c{len(headers)}")
    if row_width is not None and len(headers) != row_width:
        return ()
    return tuple(headers)


def result_columns_for_session(
    sql: str | None,
    rows: list[tuple[Any, ...]] | None,
    *,
    intent: RuntimeIntent | None = None,
    template_display_alias_map: Mapping[str, str] | None = None,
    generation_path: GenerationPath | None = None,
    federated_plan: FederatedPlan | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
    column_names: Sequence[str] | None = None,
) -> tuple[str, ...] | None:
    """Derive display column names for programmatic ``SessionStep`` consumers."""
    if not rows:
        return None
    n = len(rows[0])
    if column_names and len(column_names) == n:
        return tuple(column_names)
    federated_turn = (
        generation_path is GenerationPath.FEDERATION_PLAN or federated_plan is not None or federated_bundle is not None
    )
    if federated_turn:
        fed_hdrs = _federated_result_column_headers(
            row_width=n,
            column_names=column_names,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
        )
        if fed_hdrs is not None:
            return fed_hdrs
        if intent is not None:
            intent_hdrs = intent_result_column_headers(
                intent,
                row_width=n,
                template_display_alias_map=template_display_alias_map,
            )
            if intent_hdrs:
                return intent_hdrs
        return tuple(f"c{i}" for i in range(n))
    if intent is not None:
        intent_hdrs = intent_result_column_headers(
            intent,
            row_width=n,
            template_display_alias_map=template_display_alias_map,
        )
        if intent_hdrs:
            return intent_hdrs
    hdrs = tuple(extract_column_headers(sql or ""))
    if hdrs and len(hdrs) == n:
        return tuple(hdrs)
    return tuple(f"c{i}" for i in range(n))


def _structural_key_remap_from_assignment_order(old_intent: RuntimeIntent, new_intent: RuntimeIntent) -> dict[str, str]:
    """Map old structural ``s*`` keys to new keys when assignment sequences align in length."""
    old_seq = structural_s_key_assignment_order(old_intent)
    new_seq = structural_s_key_assignment_order(new_intent)
    if len(old_seq) != len(new_seq):
        return {}
    return dict(zip(old_seq, new_seq, strict=True))


def _remap_value_history_structural_keys(history: ValueHistory, key_remap: dict[str, str]) -> None:
    """Rewrite structural keys in *history* rows using *key_remap* (dict-only, deterministic)."""
    if not key_remap:
        return
    for i, row in enumerate(history.param_values):
        updated: dict[str, Any] = {}
        for k, v in row.items():
            nk = key_remap.get(k, k)
            updated[nk] = v
        history.param_values[i] = updated


def merge_structural_defaults_for_reuse(
    sql_param: str, new_params: dict[str, Any], structural_defaults: dict[str, Any] | None
) -> int:
    """Fill missing structural keys referenced as ``:sN`` in param SQL. from template defaults. Used by direct SQL reuse and by path 3 (``INTENT_DIRECT_MATCH``)."""
    sd = structural_defaults or {}
    s_keys = set(re.findall(r":(s\d+)", sql_param))
    added = 0
    for sk in s_keys:
        if sk not in new_params and sk in sd:
            new_params[sk] = sd[sk]
            added += 1
    return added


def _validate_replay_aggregate_join_fan_out(
    intent: RuntimeIntent,
    schema: SchemaGraph,
) -> AggregateJoinFanOutError | None:
    """Refuse replay when stored join paths would fan out aggregates in the reconstructed intent."""
    tables = list(intent.tables or [])
    if len(tables) >= 2:
        sig = list(intent.chosen_join_path_signature or [])
        if sig:
            issues = validate_aggregate_join_fan_out(
                intent,
                schema,
                "main query",
                join_signature=sig,
                from_anchor=tables[0],
            )
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                return AggregateJoinFanOutError("main query", errors[0].message)
    for cte in intent.cte_steps or []:
        cte_tbls = list(cte.tables or [])
        if len(cte_tbls) < 2:
            continue
        sig = list(cte.chosen_join_path_signature or [])
        if not sig:
            continue
        cte_scope = RuntimeIntent(
            tables=cte_tbls,
            grain=cte.grain or "row_level",
            select_cols=list(cte.select_cols or []),
            group_by_cols=list(cte.group_by_cols or []),
            order_by_cols=list(cte.order_by_cols or []),
            where=cte.where,
            having=cte.having,
            limit=cte.limit,
            limit_param_key=cte.limit_param_key or "",
            distinct_select_index=cte.distinct_select_index,
            distinct_on=list(cte.distinct_on or []),
            chosen_join_path_signature=sig,
        )
        context = f"CTE '{cte.cte_name}'"
        issues = validate_aggregate_join_fan_out(
            cte_scope,
            schema,
            context,
            join_signature=sig,
            from_anchor=cte_tbls[0],
        )
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            return AggregateJoinFanOutError(context, errors[0].message)
    return None


def _federation_contract_kwargs_for_reuse(
    reuse_path: GenerationPath, federated_plan: FederatedPlan | None
) -> dict[str, Any]:
    """Column contract kwargs for direct reuse when replaying a federated plan."""
    if reuse_path is not GenerationPath.FEDERATION_PLAN and federated_plan is None:
        return {}
    kwargs: dict[str, Any] = {}
    if reuse_path is GenerationPath.FEDERATION_PLAN:
        kwargs["generation_path"] = reuse_path
    if federated_plan is not None:
        kwargs["federated_plan"] = federated_plan
        residual = federation_residual_column_headers(federated_plan)
        if residual:
            kwargs["column_names"] = residual
    return kwargs


def complete_direct_sql_reuse_user_choice(
    ctx: DirectReuseSuspendContext,
    choice: str | None,
    *,
    choice_port: InteractiveChoicePort | None = None,
    persist_template_learning: bool = True,
) -> SqlGenerationOutcome:
    """Apply the user's confirmation after a deferred direct-reuse. prompt."""
    ref_tmpl = ctx.ref_tmpl
    q_norm = ctx.q_norm
    store = ctx.store
    templates = ctx.templates
    rejected = ctx.rejected
    intent = ctx.intent
    sql = ctx.sql
    rows = list(ctx.rows)
    sd_reuse = ctx.sd_reuse
    reuse_path = ctx.reuse_path

    normalised = "y" if choice == "y" else "n"
    if normalised == "y":
        debug(f"[{ASK_PHASE_A}] user_accepted_reuse")
        fed_contract = _federation_contract_kwargs_for_reuse(reuse_path, None)
        if intent.grain != "scalar":
            dfw = build_result_dataframe(
                rows,
                intent,
                sql,
                structural_defaults=sd_reuse,
                q_norm=q_norm,
                template_display_alias_map=getattr(ref_tmpl, "display_alias_map", None),
                column_names=ctx.headers,
                **fed_contract,
            )
            if dfw is not None:
                save_result_csv(dfw, output_path=results_csv_output_path(store))
        row_tuples = [tuple(r) for r in rows]
        cols = ctx.headers if ctx.headers else result_columns_for_session(sql, row_tuples, **fed_contract)
        note_interactive_turn(choice_port, outcome="success", sql=sql, rows=row_tuples, columns=cols, intent=intent)
    else:
        debug(f"[{ASK_PHASE_A}] user_rejected_reuse")
    handle_user_feedback(
        normalised,
        intent,
        sql,
        ctx.schema,
        store,
        templates,
        rejected,
        q_norm,
        reuse_path,
        ref_tmpl,
        matched_rejected_template=None,
        dialect=ctx.dialect,
        structural_match_templates=None,
        choice_port=choice_port,
        join_matches_template=True,
        form_storage=ctx.form_storage,
        persist_template_learning=persist_template_learning,
    )
    return SqlGenerationOutcome(ctx.sql, True, reuse_path, ref_tmpl, (), None, True, None)


def execute_reuse_with_params(
    q_norm: str,
    ref_tmpl: Template,
    new_params: dict[str, Any],
    dialect: Any,
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema: SchemaGraph,
    *,
    existing_nl: str | None = None,
    choice_port: InteractiveChoicePort | None = None,
    reuse_row_idx: int = 0,
    reuse_path: GenerationPath,
    matched_idx: int = -1,
    literal_structural_only: bool = False,
    form_storage: QuestionFormStorage | None = None,
    persist_template_learning: bool = True,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
    schema_role: str = "owner",
    context_name: str = MASTER_AETHERSPACE_NAME,
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    prompt: bool = True,
    record_question: str | None = None,
    on_param_incomplete: Literal["return_none", "raise"] = "return_none",
    federated_plan: FederatedPlan | None = None,
) -> SqlGenerationOutcome | None:
    """
    Bind *new_params* to *ref_tmpl*, validate, execute, and record feedback.

    Args:

        q_norm: Normalized question text for scope and display.
        ref_tmpl: Trusted template whose SQL is reused.
        new_params: Complete bind map after overlay and structural defaults.
        dialect: Active dialect runner.
        store: Template store backing persistence.
        templates: In-memory template map.
        rejected: Rejected-template memory map.
        schema: Schema graph for validation.
        existing_nl: Optional natural-language label for the intent skeleton.
        choice_port: Optional programmatic session receiving turn outcomes.
        reuse_row_idx: History row used for trust and auto-accept heuristics.
        reuse_path: Generation path recorded on acceptance.
        matched_idx: Exact history index when wording matched a stored row.
        literal_structural_only: Whether structural params stayed at defaults.
        form_storage: Optional question-form metadata for history recording.
        persist_template_learning: Whether writer mode may persist learning.
        schema_context: Engine execution context for consumer scope gates.
        visible_objects: Consumer-visible object set.
        schema_role: ``owner`` or ``consumer``.
        space_allowed_tables: AetherSpace table allow-list.
        space_allowed_columns: AetherSpace column allow-list.
        prompt: When False, skip the direct-reuse confirmation prompt.
        record_question: When set, normalized text stored in value history.
        on_param_incomplete: ``return_none`` or ``raise`` when bind map is incomplete.

    Returns:

        :class:`SqlGenerationOutcome` on success, ``None`` when reuse aborts quietly.

    Raises:

        ConfigError: When *on_param_incomplete* is ``raise`` and bind slots are missing.
    """
    n_bf = merge_structural_defaults_for_reuse(
        ref_tmpl.sql_param, new_params, getattr(ref_tmpl, "structural_defaults", None)
    )
    if n_bf:
        debug(f"[{ASK_PHASE_A}] backfilled_structural_from_sql_param: {n_bf} keys")

    p_count = len(re.findall(r":p\d+", ref_tmpl.sql_param or ""))
    s_count = len(re.findall(r":s\d+", ref_tmpl.sql_param or ""))
    expected_param_count = p_count + s_count

    if len(new_params) < expected_param_count:
        msg = f"parameter bind map incomplete: {len(new_params)}/{expected_param_count}"
        debug(f"[{ASK_PHASE_A}] param_extraction_incomplete: {msg}")
        if on_param_incomplete == "raise":
            raise ConfigError(msg)
        return None

    live_ok, stale_reasons = template_is_live(template_schema_refs(ref_tmpl), schema)
    if not live_ok:
        debug(f"[{ASK_PHASE_A}] template_not_live: {','.join(stale_reasons)}")
        if on_param_incomplete == "raise":
            raise ConfigError("stored template no longer matches current schema join paths")
        return None

    sd_reuse = getattr(ref_tmpl, "structural_defaults", None)
    reuse_nl = existing_nl if existing_nl else _most_frequent_natural_language(ref_tmpl.value_history)
    concrete_cte_steps = ref_tmpl.intent_signature.cte_steps or []
    runtime_cte_steps = [concrete_cte_to_runtime(c) for c in concrete_cte_steps]

    for cte in runtime_cte_steps:
        keys: set[str] = set()
        for fp in where_leaves(cte.where) or []:
            if fp.param_key:
                keys.add(fp.param_key)
        for hp in having_leaves(cte.having) or []:
            if hp.param_key:
                keys.add(hp.param_key)
        cte.param_values = {k: v for k, v in new_params.items() if k in keys}

    intent = RuntimeIntent(
        tables=ref_tmpl.intent_signature.tables or [],
        grain=ref_tmpl.intent_signature.grain or "row_level",
        select_cols=ref_tmpl.intent_signature.select_cols or [],
        group_by_cols=ref_tmpl.intent_signature.group_by_cols or [],
        order_by_cols=ref_tmpl.intent_signature.order_by_cols or [],
        where=ref_tmpl.intent_signature.where,
        having=ref_tmpl.intent_signature.having,
        param_values=new_params,
        cte_steps=runtime_cte_steps,
        column_map=ref_tmpl.intent_signature.column_map or {},
        natural_language=reuse_nl,
        chosen_join_candidate_id=ref_tmpl.chosen_join_candidate_id,
        chosen_join_path_signature=ref_tmpl.chosen_join_path_signature or [],
    )

    fan_out_err = _validate_replay_aggregate_join_fan_out(intent, schema)
    if fan_out_err is not None:
        debug(f"[{ASK_PHASE_A}] aggregate_join_fan_out: {fan_out_err.message_for_caller}")
        if on_param_incomplete == "raise":
            raise fan_out_err
        return None

    intent.sql_param = ref_tmpl.sql_param or ""
    exec_sql = dialect.finalize_render(
        ref_tmpl.sql_param or "",
        new_params,
        schema=schema,
        intent=intent,
        execution_sql_override=None,
        structural_defaults=sd_reuse,
    )
    debug(f"[{ASK_PHASE_A}] params_substituted: {new_params}")
    debug(f"[{ASK_PHASE_A}] final_sql: {exec_sql}")

    space_tables = frozenset(space_allowed_tables or ())
    space_columns = frozenset(space_allowed_columns or ())
    if space_tables or space_columns:
        if not assert_intent_in_scope(intent, space_tables, space_columns, schema):
            debug(f"[{ASK_PHASE_A}] intent out of aetherspace scope")
            return SqlGenerationOutcome(
                "",
                False,
                GenerationPath.INTENT_DIRECT_MATCH,
                ref_tmpl,
                (),
                sql_validation_error="intent out of aetherspace scope",
                error_kind=FailureCategory.DENIED_REFERENCE.value,
            )
    scope_ctx = schema_context if schema_context is not None else EngineContext()
    if _execution_scope_gate_active(scope_ctx, visible_objects, schema_role, context_name=context_name):
        if not assert_consumer_intent_in_scope(intent, scope_ctx, schema, visible_objects):
            debug(f"[{ASK_PHASE_A}] intent out of execution scope")
            return SqlGenerationOutcome(
                "",
                False,
                GenerationPath.INTENT_DIRECT_MATCH,
                ref_tmpl,
                (),
                sql_validation_error="intent out of execution scope",
                error_kind=FailureCategory.ACCESS_POLICY.value,
            )
    ok, err, _vcat, _vdiags = _run_sql_validation_cascade(exec_sql, intent, dialect, schema=schema)
    if not ok:
        debug(f"[{ASK_PHASE_A}] validation_failed: {err}")
        if on_param_incomplete == "raise":
            raise ConfigError(str(err or "SQL validation failed"))
        return None

    notify("Direct SQL reuse: validated template parameters and SQL.", stage="pipeline", code=DIAGNOSTIC_CODE_REUSE_HIT)

    try:
        progress("Executing SQL...")
        rows = dialect.execute(exec_sql, reconcile_execute_bind_params(exec_sql, new_params))
    except AccessError:
        debug(f"[{ASK_PHASE_A}] execute permission denied — continuing to intent parse")
        if on_param_incomplete == "raise":
            raise ConfigError("execute permission denied") from None
        return None

    display_base = _template_effective_sql_display_param(ref_tmpl, dialect=dialect)
    display_sql = (
        finalize_executable_sql(
            display_base, new_params, sd_reuse, sqlglot_dialect=dialect.sqlglot_dialect, for_display=True
        )
        if display_base and new_params
        else (display_base or exec_sql)
    )

    record_q = record_question if record_question is not None else q_norm
    normalised_choice = "y"
    fed_contract = _federation_contract_kwargs_for_reuse(reuse_path, federated_plan)
    row_tuples_preview = [tuple(r) for r in rows]
    resolved_headers = result_columns_for_session(
        display_sql,
        row_tuples_preview,
        generation_path=fed_contract.get("generation_path"),
        federated_plan=federated_plan,
        column_names=fed_contract.get("column_names"),
    )
    display_headers = list(resolved_headers) if resolved_headers else None
    if prompt and _should_prompt_direct_reuse_user(
        ref_tmpl, rejected, intent, q_norm, reuse_history_index=reuse_row_idx
    ):
        if choice_port is None:
            print_query_result(rows, display_sql, headers=display_headers)
        ctx = DirectReuseSuspendContext(
            q_norm=q_norm,
            ref_tmpl=ref_tmpl,
            dialect=dialect,
            store=cast(dict[str, Any], store),
            templates=templates,
            rejected=rejected,
            schema=schema,
            intent=intent,
            sql=exec_sql,
            rows=tuple(tuple(r) for r in rows),
            display_sql=display_sql,
            headers=tuple(display_headers) if display_headers else None,
            is_exact=matched_idx >= 0,
            reuse_path=reuse_path,
            sd_reuse=sd_reuse,
            form_storage=form_storage,
        )
        if choice_port is not None and not choice_port.has_pending_choice():
            raise PipelineSuspended(PIPELINE_SUSPEND_ID_DIRECT_REUSE, "Is this correct?", ctx)
        choice = interactive_yes_no(
            INTERACTIVE_STAGE_DIRECT_REUSE, "Is this correct?", ["y", "n"], choice_port=choice_port
        )
        normalised_choice = "y" if choice == "y" else "n"
        if normalised_choice == "y":
            debug(f"[{ASK_PHASE_A}] user_accepted_reuse")
        else:
            debug(f"[{ASK_PHASE_A}] user_rejected_reuse")
    else:
        if choice_port is None:
            print_query_result(rows, display_sql, headers=display_headers)
        debug(f"[{ASK_PHASE_A}] auto_accepted")

    if normalised_choice == "y" and intent.grain != "scalar":
        dfw = build_result_dataframe(
            rows,
            intent,
            exec_sql,
            structural_defaults=sd_reuse,
            q_norm=q_norm,
            template_display_alias_map=getattr(ref_tmpl, "display_alias_map", None),
            **fed_contract,
        )
        if dfw is not None:
            save_result_csv(dfw, output_path=results_csv_output_path(store))

    handle_user_feedback(
        normalised_choice,
        intent,
        exec_sql,
        schema,
        store,
        templates,
        rejected,
        record_q,
        reuse_path,
        ref_tmpl,
        matched_rejected_template=None,
        dialect=dialect,
        structural_match_templates=None,
        choice_port=choice_port,
        join_matches_template=True,
        form_storage=form_storage,
        persist_template_learning=persist_template_learning,
    )
    if normalised_choice == "y":
        row_tuples = [tuple(r) for r in rows]
        cols = result_columns_for_session(
            exec_sql,
            row_tuples,
            generation_path=fed_contract.get("generation_path"),
            federated_plan=federated_plan,
            column_names=fed_contract.get("column_names"),
        )
        note_interactive_turn(
            choice_port,
            outcome="success",
            sql=exec_sql,
            rows=row_tuples,
            columns=cols,
            intent=intent,
            matched_template=ref_tmpl,
            template_history_index=reuse_row_idx,
        )
    return SqlGenerationOutcome(exec_sql, True, reuse_path, ref_tmpl, (), None, True, None)


def _reuse_runtime_intent_from_template(
    ref_tmpl: Template, new_params: dict[str, Any], *, existing_nl: str | None = None
) -> RuntimeIntent:
    """Build a runtime intent skeleton for question-level reuse with fresh bind values."""
    reuse_nl = existing_nl if existing_nl else _most_frequent_natural_language(ref_tmpl.value_history)
    concrete_cte_steps = ref_tmpl.intent_signature.cte_steps or []
    runtime_cte_steps = [concrete_cte_to_runtime(c) for c in concrete_cte_steps]
    for cte in runtime_cte_steps:
        keys: set[str] = set()
        for fp in where_leaves(cte.where) or []:
            if fp.param_key:
                keys.add(fp.param_key)
        for hp in having_leaves(cte.having) or []:
            if hp.param_key:
                keys.add(hp.param_key)
        cte.param_values = {k: v for k, v in new_params.items() if k in keys}
    intent = RuntimeIntent(
        tables=ref_tmpl.intent_signature.tables or [],
        grain=ref_tmpl.intent_signature.grain or "row_level",
        select_cols=ref_tmpl.intent_signature.select_cols or [],
        group_by_cols=ref_tmpl.intent_signature.group_by_cols or [],
        order_by_cols=ref_tmpl.intent_signature.order_by_cols or [],
        where=ref_tmpl.intent_signature.where,
        having=ref_tmpl.intent_signature.having,
        param_values=new_params,
        cte_steps=runtime_cte_steps,
        column_map=ref_tmpl.intent_signature.column_map or {},
        natural_language=reuse_nl,
        chosen_join_candidate_id=ref_tmpl.chosen_join_candidate_id,
        chosen_join_path_signature=ref_tmpl.chosen_join_path_signature or [],
    )
    intent.sql_param = ref_tmpl.sql_param or ""
    return intent


def _member_template_for_plan_template(
    cached: FederationPlanTemplate, stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]]
) -> Template | None:
    """Load the first member template referenced by a federation plan template."""
    for source_id, tmpl_id in cached.member_template_ids:
        member_store = stores_by_source.get(source_id)
        if member_store is None:
            continue
        if isinstance(member_store, TemplateStoreView):
            templates = cast(dict[str, Any], member_store["templates"])
        else:
            templates = cast(dict[str, Any], member_store.get("templates", {}))
        tmpl = templates.get(tmpl_id)
        if tmpl is not None:
            return cast(Template, tmpl)
    return None


def _space_context_for_plan_reuse(
    *,
    choice_port: InteractiveChoicePort | None = None,
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> SpaceContext | None:
    """Build federation space scope for plan replay from reuse gate kwargs."""
    space_tables = frozenset(space_allowed_tables or ())
    space_columns = frozenset(space_allowed_columns or ())
    space_deny_objects: frozenset[str] = frozenset()
    space_deny_columns: frozenset[str] = frozenset()
    if gate_kwargs_by_source:
        sample = next(iter(gate_kwargs_by_source.values()), {})
        if not space_tables:
            space_tables = frozenset(sample.get("space_allowed_tables") or ())
        if not space_columns:
            space_columns = frozenset(sample.get("space_allowed_columns") or ())
    if choice_port is not None:
        if not space_tables:
            space_tables = frozenset(getattr(choice_port, "space_tables", None) or ())
        if not space_columns:
            space_columns = frozenset(getattr(choice_port, "space_columns", None) or ())
        space_deny_objects = frozenset(getattr(choice_port, "space_deny_objects", None) or ())
        space_deny_columns = frozenset(getattr(choice_port, "space_deny_columns", None) or ())
    if not space_tables and not space_columns and not space_deny_objects and not space_deny_columns:
        return None
    return SpaceContext(
        tables=space_tables,
        columns=space_columns,
        deny_objects=space_deny_objects,
        deny_columns=space_deny_columns,
    )


def _exact_reuse_param_row(q_norm: str, ref_tmpl: Template) -> dict[str, Any]:
    """Return stored bind values for an exact question match, else the first history row."""
    vh = ref_tmpl.value_history
    for i, hist_q in enumerate(vh.questions):
        if hist_q and q_norm == hist_q:
            if i < len(vh.param_values):
                return dict(vh.param_values[i])
            return {}
    if vh.param_values:
        return dict(vh.param_values[0])
    return {}


def try_federation_plan_intake_reuse(
    q_norm: str,
    composite_schema: SchemaGraph,
    dialect: Any,
    *,
    federation_dir: str | None = None,
    federation_manifest: FederationManifest | None = None,
    federation_mappings: FederationMappings | None = None,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> SqlGenerationOutcome | None:
    """Replay a stored federation plan from question intake before interpret."""
    if not federation_dir or federation_manifest is None or not stores_by_source:
        return None
    cached = lookup_federation_plan_template_for_question(federation_dir, q_norm)
    if cached is None:
        return None
    ref_tmpl = _member_template_for_plan_template(cached, stores_by_source)
    if ref_tmpl is None:
        return None
    new_params = _exact_reuse_param_row(q_norm, ref_tmpl)
    return _try_federation_plan_question_reuse(
        q_norm,
        ref_tmpl,
        new_params,
        composite_schema,
        dialect,
        federation_dir=federation_dir,
        federation_manifest=federation_manifest,
        federation_mappings=federation_mappings,
        stores_by_source=stores_by_source,
        dialects_by_source=dialects_by_source,
        source_runtimes=source_runtimes,
        member_graphs=member_graphs,
        gate_kwargs_by_source=gate_kwargs_by_source,
        existing_nl=q_norm,
        space=_space_context_for_plan_reuse(gate_kwargs_by_source=gate_kwargs_by_source),
    )


def _resolve_federation_plan_template_for_reuse(
    federation_dir: str | None, q_norm: str, ref_tmpl: Template
) -> FederationPlanTemplate | None:
    if not federation_dir:
        return None
    cached = lookup_federation_plan_template_for_question(federation_dir, q_norm)
    if cached is not None:
        return cached
    if template_is_federation_plan_fragment(ref_tmpl):
        plan_id = str(getattr(ref_tmpl, "federation_plan_id", "") or "")
        if plan_id:
            return load_federation_plan_templates(federation_dir).get(plan_id)
    return None


def _try_federation_plan_question_reuse(
    q_norm: str,
    ref_tmpl: Template,
    new_params: dict[str, Any],
    composite_schema: SchemaGraph,
    dialect: Any,
    *,
    federation_dir: str | None = None,
    federation_manifest: FederationManifest | None = None,
    federation_mappings: FederationMappings | None = None,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    existing_nl: str | None = None,
    on_param_incomplete: Literal["return_none", "raise"] = "return_none",
    space: SpaceContext | None = None,
) -> SqlGenerationOutcome | None:
    """Replay a stored federation plan instead of rendering composite display SQL."""
    if not federation_dir or federation_manifest is None or not stores_by_source:
        return None
    cached = _resolve_federation_plan_template_for_reuse(federation_dir, q_norm, ref_tmpl)
    if cached is None or not cached.member_template_ids:
        return None
    intent = _reuse_runtime_intent_from_template(ref_tmpl, new_params, existing_nl=existing_nl)
    plan = plan_federated_intent(
        intent,
        composite_schema,
        federation_manifest,
        federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION),
        member_graphs=member_graphs,
        space=space,
    )
    if plan.ineligible_reason:
        if on_param_incomplete == "raise":
            raise ConfigError(plan.ineligible_reason)
        return None
    step_fps = federation_plan_step_fingerprints(
        plan,
        intent_key_fn=intent_key,
        manifest=federation_manifest,
        member_graphs=member_graphs,
    )
    manifest_hash_value, member_tuple_hash_value = ("", "")
    if isinstance(member_graphs, dict) and member_graphs and isinstance(federation_manifest, FederationManifest):
        manifest_hash_value, member_tuple_hash_value = federation_plan_topology_identity(
            member_graphs, federation_manifest
        )
    if not federation_plan_matches_template(
        plan,
        cached,
        step_fingerprints=step_fps,
        manifest_hash_value=manifest_hash_value,
        member_tuple_hash_value=member_tuple_hash_value,
    ):
        if on_param_incomplete == "raise":
            raise ConfigError("federation plan no longer matches stored plan template")
        return None
    try:
        fed_prep = replay_federated_prepare_from_plan_template(
            plan,
            cached,
            composite_schema,
            stores_by_source=stores_by_source,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            default_dialect=dialect,
            manifest=federation_manifest,
            member_graphs=member_graphs,
            gate_kwargs_by_source=gate_kwargs_by_source,
            q_norm=q_norm,
        )
    except FederationConfigError:
        if on_param_incomplete == "raise":
            raise ConfigError("federation plan replay failed") from None
        return None
    if not fed_prep.success:
        if on_param_incomplete == "raise":
            raise ConfigError(fed_prep.sql_validation_error or "federation plan replay failed")
        return None
    try:
        execute_federated_prepare(
            fed_prep,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            manifest=federation_manifest,
            q_norm=q_norm,
            federation_dir=federation_dir,
            member_graphs=member_graphs,
            gate_kwargs_by_source=gate_kwargs_by_source,
        )
    except (FederationConfigError, FederationRuntimeError) as exc:
        if on_param_incomplete == "raise":
            raise ConfigError(str(exc)) from exc
        return None
    notify(
        "Federation plan replay succeeded for question-level reuse.",
        stage="pipeline",
        code=DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
        source_id="composite",
        details=(("phase", "prepare"), ("plan_id", cached.plan_id)),
    )
    return SqlGenerationOutcome(
        sql=fed_prep.display_sql,
        success=True,
        generation_path=GenerationPath.FEDERATION_PLAN,
        matched_template=None,
        federated_steps=tuple(fed_prep.steps),
        federation_plan_id=cached.plan_id,
        federation_dir=federation_dir,
    )


def force_reuse_saved_question(
    question_old: str,
    question_new: str,
    new_values: dict[str, Any],
    dialect: Any,
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema: SchemaGraph,
    *,
    choice_port: InteractiveChoicePort | None = None,
    persist_template_learning: bool = True,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
    schema_role: str = "owner",
    context_name: str = MASTER_AETHERSPACE_NAME,
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    federation_dir: str | None = None,
    federation_manifest: FederationManifest | None = None,
    federation_mappings: FederationMappings | None = None,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> SqlGenerationOutcome:
    """
    Re-execute a stored template with caller-supplied bind values and a new question row.

    Args:

        question_old: Prior question text that identifies the stored template.
        question_new: New natural-language question recorded in value history.
        new_values: Changed bind values keyed by template handles (``p1``, ``s1``, …).
        dialect: Active dialect runner.
        store: Template store backing persistence.
        templates: In-memory template map.
        rejected: Rejected-template memory map.
        schema: Schema graph for validation.
        choice_port: Optional programmatic session receiving turn outcomes.
        persist_template_learning: Whether writer mode may persist learning.
        schema_context: Engine execution context for consumer scope gates.
        visible_objects: Consumer-visible object set.
        schema_role: ``owner`` or ``consumer``.
        space_allowed_tables: AetherSpace table allow-list.
        space_allowed_columns: AetherSpace column allow-list.

    Returns:

        :class:`SqlGenerationOutcome` for the executed query.

    Raises:

        ConfigError: When no template matches *question_old* or bind values are invalid.
    """
    resolved = resolve_template_for_question(question_old, templates, template_store=store)
    if resolved is None:
        raise ConfigError(f"No stored template matches question {question_old!r}")
    ref_tmpl, hist_idx = resolved
    q_new_norm = normalize_question(question_new)
    expected_handles = set(handles_referenced_in_sql_param(ref_tmpl.sql_param or ""))
    for key in new_values:
        if key not in expected_handles:
            raise ConfigError(f"Unknown parameter handle {key!r} for template {ref_tmpl.id}")
    base_row = (
        dict(ref_tmpl.value_history.param_values[hist_idx])
        if ref_tmpl.value_history.param_values and hist_idx < len(ref_tmpl.value_history.param_values)
        else {}
    )
    merged = dict(base_row)
    merged.update(new_values)
    fed_outcome = _try_federation_plan_question_reuse(
        q_new_norm,
        ref_tmpl,
        merged,
        schema,
        dialect,
        federation_dir=federation_dir,
        federation_manifest=federation_manifest,
        federation_mappings=federation_mappings,
        stores_by_source=stores_by_source,
        dialects_by_source=dialects_by_source,
        source_runtimes=source_runtimes,
        member_graphs=member_graphs,
        gate_kwargs_by_source=gate_kwargs_by_source,
        on_param_incomplete="raise",
        space=_space_context_for_plan_reuse(
            choice_port=choice_port,
            space_allowed_tables=space_allowed_tables,
            space_allowed_columns=space_allowed_columns,
            gate_kwargs_by_source=gate_kwargs_by_source,
        ),
    )
    if fed_outcome is not None:
        return fed_outcome
    outcome = execute_reuse_with_params(
        q_new_norm,
        ref_tmpl,
        merged,
        dialect,
        store,
        templates,
        rejected,
        schema,
        choice_port=choice_port,
        reuse_row_idx=hist_idx,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        matched_idx=-1,
        literal_structural_only=False,
        form_storage=QuestionFormStorage(corrected=question_new.strip()),
        persist_template_learning=persist_template_learning,
        schema_context=schema_context,
        visible_objects=visible_objects,
        schema_role=schema_role,
        space_allowed_tables=space_allowed_tables,
        space_allowed_columns=space_allowed_columns,
        prompt=False,
        record_question=q_new_norm,
        on_param_incomplete="raise",
    )
    if outcome is None or not outcome.success:
        raise ConfigError("Forced template reuse failed validation or execution")
    return outcome


def execute_stored_template_by_ref(
    template_ref: str,
    params: dict[str, Any],
    *,
    question: str | None,
    dialect: Any,
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema: SchemaGraph,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
    schema_role: str = "owner",
    persist_template_learning: bool = False,
) -> TemplateExecutionResult:
    """
    Execute one stored template identified by id or ``sql_fp`` with caller bind values.

    Raises:

        ConfigError: When the ref is unknown, bind values are invalid, or execution fails.
    """
    tmpl = resolve_template_ref(template_ref, templates)
    if tmpl is None or not template_visible_to_callers(tmpl):
        raise ConfigError(f"unknown template ref {template_ref!r}")
    expected_handles = set(handles_referenced_in_sql_param(tmpl.sql_param or ""))
    for key in params:
        if key not in expected_handles:
            raise ConfigError(f"Unknown parameter handle {key!r} for template {tmpl.id}")
    vh = tmpl.value_history
    hist_idx = 0
    if vh.questions:
        primary_q = primary_template_q_norm(tmpl)
        hist_idx = vh.questions.index(primary_q) if primary_q in vh.questions else 0
    base_row = dict(vh.param_values[hist_idx]) if vh.param_values and hist_idx < len(vh.param_values) else {}
    merged = dict(base_row)
    merged.update(params)
    q_norm = normalize_question(question) if question else primary_template_q_norm(tmpl)
    if not q_norm:
        raise ConfigError("template has no stored question row; pass question=")
    outcome = execute_reuse_with_params(
        q_norm,
        tmpl,
        merged,
        dialect,
        store,
        templates,
        rejected,
        schema,
        reuse_row_idx=hist_idx,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        matched_idx=hist_idx if q_norm in vh.questions else -1,
        persist_template_learning=persist_template_learning,
        schema_context=schema_context,
        visible_objects=visible_objects,
        schema_role=schema_role,
        prompt=False,
        record_question=q_norm,
        on_param_incomplete="raise",
    )
    if outcome is None or not outcome.success:
        raise ConfigError("template execution failed validation or execution")
    sd_reuse = getattr(tmpl, "structural_defaults", None)
    exec_bind = reconcile_execute_bind_params(outcome.sql, merged)
    rows = dialect.execute(outcome.sql, exec_bind)
    display_base = _template_effective_sql_display_param(tmpl, dialect=dialect)
    display_sql = (
        finalize_executable_sql(display_base, merged, sd_reuse, sqlglot_dialect=dialect.sqlglot_dialect)
        if display_base and merged
        else (display_base or outcome.sql)
    )
    row_tuples = tuple(tuple(r) for r in rows)
    cols = result_columns_for_session(display_sql, list(row_tuples)) or ()
    return TemplateExecutionResult(
        rows=row_tuples,
        sql=outcome.sql,
        display_sql=display_sql,
        columns=tuple(cols),
    )


def handle_direct_sql_reuse(
    q_norm: str,
    ref_tmpl: Template,
    dialect: Any,
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema: SchemaGraph,
    existing_nl: str | None = None,
    choice_port: InteractiveChoicePort | None = None,
    reuse_history_index: int | None = None,
    form_storage: QuestionFormStorage | None = None,
    persist_template_learning: bool = True,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
    schema_role: str = "owner",
    context_name: str = MASTER_AETHERSPACE_NAME,
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    federation_dir: str | None = None,
    federation_manifest: FederationManifest | None = None,
    federation_mappings: FederationMappings | None = None,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> SqlGenerationOutcome | None:
    """Reuse a template’s SQL: extract params, validate, execute, and. record feedback."""
    emit_ask_phase(ASK_PHASE_A)
    vh = ref_tmpl.value_history
    matched_idx = -1
    for i, hist_q in enumerate(vh.questions):
        if hist_q and q_norm == hist_q:
            matched_idx = i
            break

    literal_structural_only = False
    reuse_row_idx = 0
    if matched_idx >= 0:
        new_params = dict(vh.param_values[matched_idx])
        debug(f"[{ASK_PHASE_A}] zero_distance_match: index={matched_idx}")
        reuse_row_idx = matched_idx
    else:
        hi = reuse_history_index
        if hi is None:
            if vh.questions:
                freq_q = Counter(vh.questions).most_common(1)[0][0]
                hi = vh.questions.index(freq_q)
            else:
                hi = 0
        reuse_row_idx = hi
        prev_row = vh.param_values[hi] if hi < len(vh.param_values) else {}
        literal_structural_only = _row_structural_values_match_defaults(
            prev_row, getattr(ref_tmpl, "structural_defaults", None), ref_tmpl.sql_param or ""
        )
        new_params = (
            _extract_reuse_params_literal_only(q_norm, ref_tmpl, history_index=hi, schema=schema)
            if literal_structural_only
            else _extract_reuse_params_full(q_norm, ref_tmpl, history_index=hi, schema=schema)
        )
        debug(f"[{ASK_PHASE_A}] llm_reuse_extraction: {len(new_params)} params")

    reuse_path = (
        GenerationPath.EXACT_QUESTION_REUSE
        if matched_idx >= 0
        else (
            GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL
            if literal_structural_only
            else GenerationPath.FUZZY_REUSE_FULL_PARAMS
        )
    )
    fed_outcome = _try_federation_plan_question_reuse(
        q_norm,
        ref_tmpl,
        new_params,
        schema,
        dialect,
        federation_dir=federation_dir,
        federation_manifest=federation_manifest,
        federation_mappings=federation_mappings,
        stores_by_source=stores_by_source,
        dialects_by_source=dialects_by_source,
        source_runtimes=source_runtimes,
        member_graphs=member_graphs,
        gate_kwargs_by_source=gate_kwargs_by_source,
        existing_nl=existing_nl,
        space=_space_context_for_plan_reuse(
            choice_port=choice_port,
            space_allowed_tables=space_allowed_tables,
            space_allowed_columns=space_allowed_columns,
            gate_kwargs_by_source=gate_kwargs_by_source,
        ),
    )
    if fed_outcome is not None:
        return fed_outcome
    if federation_manifest is not None:
        debug(f"[{ASK_PHASE_A}] federation_active: direct SQL reuse blocked")
        return None
    return execute_reuse_with_params(
        q_norm,
        ref_tmpl,
        new_params,
        dialect,
        store,
        templates,
        rejected,
        schema,
        existing_nl=existing_nl,
        choice_port=choice_port,
        reuse_row_idx=reuse_row_idx,
        reuse_path=reuse_path,
        matched_idx=matched_idx,
        literal_structural_only=literal_structural_only,
        form_storage=form_storage,
        persist_template_learning=persist_template_learning,
        schema_context=schema_context,
        visible_objects=visible_objects,
        schema_role=schema_role,
        space_allowed_tables=space_allowed_tables,
        space_allowed_columns=space_allowed_columns,
        prompt=True,
        record_question=None,
        on_param_incomplete="return_none",
    )


def _intent_decline_feedback_bucket(
    intent: RuntimeIntent,
    store: dict[str, Any] | TemplateStoreView,
    q_norm: str,
    schema: SchemaGraph | None,
    choice_port: InteractiveChoicePort | None,
    suspend_tail: InteractiveTailSnapshot | None,
    default_user_reason: str,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
) -> str | None:
    """Collect optional decline text, persist intent rejection feedback, and return the bucket label."""
    if not q_norm or schema is None:
        return None
    if getattr(intent, "schema_invalid", False):
        return None
    feedback_body = (
        "What should change about this interpretation?\n"
        "Tip: a single sentence is enough — for example 'wrong table', "
        "'missing date filter', or 'should aggregate by month'."
    )
    if choice_port is not None and suspend_tail is not None:
        raise PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_FEEDBACK, feedback_body, suspend_tail)
    feedback = ""
    try:
        print_info(feedback_body)
        if choice_port is not None and choice_port.has_pending_choice():
            cons = getattr(choice_port, "_consume_next_queued_choice", None)
            if not callable(cons):
                raise TypeError("choice port must implement _consume_next_queued_choice for intent feedback")
            feedback = (cons() or "").strip()
        else:
            feedback = prompt("").strip()
    except (EOFError, KeyboardInterrupt):
        terminated()
    entry = summarize_failure_for_memory(
        question=q_norm,
        intent=intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=schema.effective_structural_hash,
        user_reason=feedback or default_user_reason,
    )
    if persist_template_learning:
        record_question_feedback(store, q_norm, entry)
        save_template_store(store)
    else:
        ev = WriteQueueEvent(
            kind="feedback_record",
            schema_graph_id=str(schema.schema_graph_id or ""),
            schema_hash=str(schema.effective_structural_hash or ""),
            produced_at=datetime.now(timezone.utc).isoformat(),
            payload=(("q_norm", q_norm), ("entry_json", stable_json(entry.to_dict()))),
        )
        _emit_reader_write_queue_event(store, ev)
    ctx_ref = _refinement_ctx_for_feedback(choice_port, refinement_ctx)
    reason_line = (feedback or "").strip() or default_user_reason
    if ctx_ref is not None and refinement_retry_available(ctx_ref):
        ctx_ref.accumulated_reasons.append(reason_line)
        ctx_ref.pending_retry = True
        raise RefinementRetry
    return entry.buckets[0].value if entry.buckets else RejectionBucket.OTHER.value


def compose_intent_confirm_session_message(
    intent: RuntimeIntent, semantic_warnings: list[Any] | None
) -> tuple[str, tuple[str, ...]]:
    """Build the multi-line intent-confirmation body and parallel structured warning strings."""
    parts: list[str] = []
    warn_out: list[str] = []
    if getattr(intent, "schema_invalid", False):
        parts.append("This question may not match the current database schema.")
    if semantic_warnings:
        parts.append("Semantic warnings:")
        for w in semantic_warnings:
            if isinstance(w, str):
                parts.append(f"  - {w}")
                warn_out.append(w.strip())
            elif isinstance(w, dict):
                msg = str(w.get("message", w))
                parts.append(f"  - {msg}")
                warn_out.append(msg.strip())
            else:
                parts.append(f"  - {w!s}")
                warn_out.append(str(w).strip())
    nl = intent.natural_language or f"Query {', '.join(intent.tables or [])} for data"
    parts.append(f"I understood: {nl}")
    return "\n".join(parts), tuple(warn_out)


def confirm_intent_with_user(
    intent: RuntimeIntent,
    store: dict[str, Any] | TemplateStoreView,
    semantic_warnings: list[Any] | None = None,
    similarity_score: float = 0.0,
    has_union_match: bool = False,
    cols_changed: bool = False,
    rejected: dict[str, Any] | None = None,
    q_norm: str | None = None,
    schema: SchemaGraph | None = None,
    choice_port: InteractiveChoicePort | None = None,
    suspend_tail: InteractiveTailSnapshot | None = None,
    intent_already_confirmed: bool = False,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
    force_intent_confirm: bool = False,
) -> bool:
    """Prompt for intent confirmation or auto-proceed from similarity. and warnings."""
    if intent_already_confirmed:
        return True
    if not force_intent_confirm and should_skip_intent_confirmation(
        intent, cast(dict[str, Any] | None, store), q_norm or "", semantic_warnings
    ):
        debug(
            f"[{ASK_PHASE_H}] auto_proceed: similarity={similarity_score:.3f} "
            f"has_union={has_union_match} cols_changed={cols_changed}"
        )
        return True
    body_lines, _ = compose_intent_confirm_session_message(intent, semantic_warnings)
    if choice_port is None or choice_port.has_pending_choice() or suspend_tail is None:
        print_info(body_lines)
    if choice_port is not None and not choice_port.has_pending_choice() and suspend_tail is not None:
        raise PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "Is this correct?", suspend_tail)
    intent_choice = interactive_yes_no(
        INTERACTIVE_STAGE_INTENT_CONFIRM, "Is this correct?", ["y", "n"], choice_port=choice_port
    )
    if intent_choice is None or intent_choice != "y":
        debug(f"[{ASK_PHASE_H}] user_rejected_intent")
        if getattr(intent, "schema_invalid", False):
            if persist_template_learning:
                save_template_store(store)
            print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=None)
            return False
        rejection_bucket = _intent_decline_feedback_bucket(
            intent,
            store,
            q_norm or "",
            schema,
            choice_port,
            suspend_tail,
            "User declined intent confirmation",
            refinement_ctx=refinement_ctx,
            persist_template_learning=persist_template_learning,
        )
        if persist_template_learning:
            save_template_store(store)
        print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=rejection_bucket)
        return False
    debug(f"[{ASK_PHASE_H}] user_confirmed_intent")
    clear_planner_schema_invalid_after_user_accept(intent)
    return True


def _final_display_sql_for_results(
    intent: RuntimeIntent,
    sql: str,
    structural_defaults: dict[str, Any] | None,
    *,
    q_norm: str = "",
    template_display_alias_map: dict[str, str] | None = None,
) -> str:
    """Resolve executable display SQL for printing or CSV export."""
    if intent.expected_rows == "one":
        display_param = intent.sql_param or ""
    elif not (intent.select_cols or []):
        display_param = intent.sql_param or ""
    else:
        try:
            dialect = get_dialect(EngineConfig.TYPE, EngineConfig.RUNTIME)
        except ValueError:
            dialect = None
        if dialect is None:
            display_param = intent.sql_param or ""
        else:
            d_aliases = enriched_display_alias_map(q_norm, intent.sql_param or "", intent, template_display_alias_map)
            display_param = build_display_sql(intent.sql_param or "", intent, d_aliases, dialect=dialect)
    if display_param and intent.param_values:
        return finalize_executable_sql(
            display_param,
            intent.param_values,
            structural_defaults,
            sqlglot_dialect=active_sqlglot_dialect(),
            for_display=True,
        )
    return display_param or sql


def build_result_dataframe(
    rows: list[tuple[Any, ...]],
    intent: RuntimeIntent,
    sql: str,
    structural_defaults: dict[str, Any] | None = None,
    *,
    q_norm: str = "",
    template_display_alias_map: dict[str, str] | None = None,
    column_names: Sequence[str] | None = None,
    generation_path: GenerationPath | None = None,
    federated_plan: FederatedPlan | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
) -> pandas.DataFrame | None:
    """Build a row-level ``DataFrame`` for programmatic session steps, or ``None`` for scalar grain."""
    fed_hdr: tuple[str, ...] | None = None
    if rows:
        fed_hdr = _federated_result_column_headers(
            row_width=len(rows[0]),
            column_names=column_names,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
        )
    if intent.grain == "scalar":
        return None
    if column_names is not None and rows and len(column_names) == len(rows[0]):
        return pandas.DataFrame(rows, columns=list(column_names))
    if fed_hdr is not None:
        return pandas.DataFrame(rows, columns=list(fed_hdr))
    if generation_path is GenerationPath.FEDERATION_PLAN or federated_plan is not None or federated_bundle is not None:
        if rows:
            intent_hdrs = intent_result_column_headers(
                intent,
                row_width=len(rows[0]),
                template_display_alias_map=template_display_alias_map,
            )
            if intent_hdrs:
                return pandas.DataFrame(rows, columns=list(intent_hdrs))
            return pandas.DataFrame(rows, columns=[f"c{i}" for i in range(len(rows[0]))])
        return pandas.DataFrame(rows)
    intent_hdrs = intent_result_column_headers(
        intent,
        row_width=len(rows[0]) if rows else None,
        template_display_alias_map=template_display_alias_map,
    )
    if intent_hdrs:
        return pandas.DataFrame(rows, columns=list(intent_hdrs))
    display_sql = _final_display_sql_for_results(
        intent, sql, structural_defaults, q_norm=q_norm, template_display_alias_map=template_display_alias_map
    )
    hdr = extract_column_headers(display_sql)
    if hdr:
        return pandas.DataFrame(rows, columns=hdr)
    return pandas.DataFrame(rows)


def display_final_results_to_stdout(
    q_norm: str,
    intent: RuntimeIntent,
    sql: str,
    rows: list[tuple[Any, ...]],
    structural_defaults: dict[str, Any] | None = None,
    *,
    template_display_alias_map: dict[str, str] | None = None,
    generation_path: GenerationPath | None = None,
    federated_plan: FederatedPlan | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
    column_names: Sequence[str] | None = None,
) -> None:
    """Print result rows to stdout using the resolved display SQL."""
    display_sql = _final_display_sql_for_results(
        intent, sql, structural_defaults, q_norm=q_norm, template_display_alias_map=template_display_alias_map
    )
    hdr: list[str] | None = None
    federated_turn = (
        generation_path is GenerationPath.FEDERATION_PLAN
        or federated_plan is not None
        or federated_bundle is not None
        or column_names is not None
    )
    if column_names and rows and len(column_names) == len(rows[0]):
        hdr = list(column_names)
    elif federated_turn and rows:
        fed_hdr = _federated_result_column_headers(
            row_width=len(rows[0]),
            column_names=column_names,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
        )
        if fed_hdr is not None:
            hdr = list(fed_hdr)
        else:
            intent_hdrs = intent_result_column_headers(
                intent,
                row_width=len(rows[0]),
                template_display_alias_map=template_display_alias_map,
            )
            hdr = list(intent_hdrs) if intent_hdrs else [f"c{i}" for i in range(len(rows[0]))]
    elif not federated_turn:
        intent_hdrs = intent_result_column_headers(
            intent,
            row_width=len(rows[0]),
            template_display_alias_map=template_display_alias_map,
        )
        if intent_hdrs:
            hdr = list(intent_hdrs)
        else:
            hdr = extract_column_headers(display_sql)
    print_query_result(rows, display_sql, headers=hdr)


def results_csv_output_path(
    store: dict[str, Any] | TemplateStoreView | None = None,
    *,
    artifacts_dir: str | None = None,
    csv_dir: str | None = None,
) -> str:
    """Resolve the destination path for ``results.csv`` from explicit dirs or a template store."""
    if csv_dir:
        return os.path.join(csv_dir, "results.csv")
    if artifacts_dir:
        return os.path.join(artifacts_dir, "results.csv")
    if store is not None:
        return os.path.join(_artifact_dir_for_template_store(store), "results.csv")
    return os.path.join(os.getcwd(), "results.csv")


def save_result_csv(
    df: pandas.DataFrame,
    *,
    output_path: str | os.PathLike[str] | None = None,
) -> None:
    """Write *df* to ``results.csv`` at *output_path* or the process working directory."""
    dest = os.fspath(output_path) if output_path is not None else os.path.join(os.getcwd(), "results.csv")
    df.to_csv(dest, index=False)
    debug(f"results saved to {dest}")


def _execute_intent_sql_rows(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    structural_defaults: dict[str, Any] | None,
    *,
    timeout_ms: int | None = None,
    sql_override: str | None = None,
    bind_map: Mapping[str, Any] | None = None,
    return_column_names: bool = False,
    gate_kwargs: Mapping[str, Any] | None = None,
) -> list[tuple[Any, ...]] | tuple[list[tuple[Any, ...]], tuple[str, ...] | None]:
    """Finalize and execute one intent's SQL, returning row tuples."""
    exec_params = dict(bind_map) if bind_map is not None else dict(flatten_param_values(intent))
    base_sql = sql_override if sql_override is not None else (intent.sql_param or "")
    exec_sql = dialect.finalize_render(
        base_sql,
        dict(flatten_param_values(intent)),
        schema=schema,
        intent=intent,
        execution_sql_override=sql_override,
        structural_defaults=structural_defaults,
    )
    exec_bind = reconcile_execute_bind_params(exec_sql, exec_params)
    if gate_kwargs is not None:
        schema_role = str(gate_kwargs.get("schema_role", "owner") or "owner")
        schema_context = gate_kwargs.get("schema_context")
        visible_objects = gate_kwargs.get("visible_objects")
        context_name = str(gate_kwargs.get("context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
        ctx = schema_context if schema_context is not None else EngineContext()
        gate_active = _execution_scope_gate_active(ctx, visible_objects, schema_role, context_name=context_name)
        if gate_active and not assert_consumer_sql_in_scope(exec_sql, dialect, ctx, schema, visible_objects):
            raise AccessError("execute", PERMISSION_DENIED_USER_MESSAGE)
    backend = getattr(dialect, "result_backend", None)
    if return_column_names and backend is not None:
        fetch_with_cols = getattr(backend, "fetch_rows_with_columns", None)
        if callable(fetch_with_cols):
            if timeout_ms is not None:
                rows, cols = fetch_with_cols(exec_sql, exec_bind, timeout_ms=int(timeout_ms))
            else:
                rows, cols = fetch_with_cols(exec_sql, exec_bind)
            return list(rows), tuple(str(c) for c in cols) if cols else None
    if timeout_ms is not None and backend is not None:
        rows = list(backend.fetch_rows(exec_sql, exec_bind, timeout_ms=int(timeout_ms)))
    else:
        rows = list(dialect.execute(exec_sql, exec_bind))
    if return_column_names:
        return rows, None
    return rows


def _format_federated_sql_display(per_source_sql: Sequence[tuple[str, str]], glue_sql: str) -> str:
    """Join per-source SQL and coordinator glue for session display."""
    parts: list[str] = []
    for idx, (_source_id, sql) in enumerate(per_source_sql, start=1):
        parts.append(f"-- statement_{idx}\n{sql}")
    if glue_sql.strip():
        parts.append(f"-- coordinator\n{glue_sql}")
    return "\n\n".join(parts)


def replay_federated_prepare_from_plan_template(
    plan: FederatedPlan,
    cached_template: FederationPlanTemplate,
    composite_schema: SchemaGraph,
    *,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]],
    dialects_by_source: Mapping[str, Any] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    default_dialect: Any,
    manifest: FederationManifest | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    q_norm: str = "",
) -> FederatedPrepareOutcome:
    """Rebuild a federated prepare outcome from cached member templates without LLM generation."""
    if not cached_template.member_template_ids:
        raise FederationConfigError("federation plan template has no member template ids")
    member_id_map = dict(cached_template.member_template_ids)
    prepared: list[FederatedPreparedStep] = []
    per_source: list[tuple[str, str]] = []
    gate_map = dict(gate_kwargs_by_source or {})
    degenerate = federation_plan_is_degenerate(plan)
    for step in plan.steps:
        tmpl_id = member_id_map.get(step.source_id)
        if not tmpl_id:
            raise FederationConfigError(
                f"federation plan template missing member template id for source {step.source_id!r}"
            )
        member_store = stores_by_source.get(step.source_id)
        if member_store is None:
            raise FederationConfigError(f"federation member store missing for source {step.source_id!r}")
        if isinstance(member_store, TemplateStoreView):
            templates = cast(dict[str, Any], member_store["templates"])
        else:
            templates = cast(dict[str, Any], member_store.get("templates", {}))
        tmpl = templates.get(tmpl_id)
        if tmpl is None:
            raise FederationConfigError(f"federation member template {tmpl_id!r} missing for source {step.source_id!r}")
        source_dialect, sub_schema = _federated_step_sql_context(
            step,
            composite_schema,
            dialect=default_dialect,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            manifest=manifest,
            member_graphs=member_graphs,
        )
        live_ok, _stale_reasons = template_is_live(template_schema_refs(tmpl), sub_schema)
        if not live_ok:
            raise FederationConfigError(
                f"federation member template {tmpl_id!r} is stale for source {step.source_id!r}"
            )
        sub_intent = (
            step.sub_intent if degenerate else apply_projected_keys_to_intent(step.sub_intent, step.projected_keys)
        )
        replay_sig = list(
            tmpl.chosen_join_path_signature or getattr(tmpl.intent_signature, "chosen_join_path_signature", None) or []
        )
        if replay_sig:
            sub_intent = RuntimeIntent.from_dict(sub_intent.to_dict())
            sub_intent.chosen_join_path_signature = replay_sig
        if not degenerate:
            sub_intent = expand_shared_pk_tables_for_refs(sub_intent, sub_schema)
            processed, post_issues = apply_runtime_post_processing(sub_intent, sub_schema, question_fallback=q_norm)
            if processed is None:
                return FederatedPrepareOutcome(
                    success=False,
                    plan=plan,
                    display_sql="",
                    sql_validation_error="federated member post-processing incomplete",
                    error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
                    source_id=step.source_id,
                    phase="prepare",
                )
            blocking = [issue for issue in post_issues if getattr(issue, "severity", "") == "error"]
            if blocking:
                return FederatedPrepareOutcome(
                    success=False,
                    plan=plan,
                    display_sql="",
                    sql_validation_error=str(getattr(blocking[0], "message", blocking[0])),
                    error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
                    source_id=step.source_id,
                    phase="prepare",
                )
            sub_intent = processed
            slice_error = validate_federated_sub_intent(sub_intent, sub_schema)
            if slice_error:
                return FederatedPrepareOutcome(
                    success=False,
                    plan=plan,
                    display_sql="",
                    sql_validation_error=slice_error,
                    error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
                    source_id=step.source_id,
                    phase="prepare",
                )
        replay_intent = RuntimeIntent.from_dict(sub_intent.to_dict())
        fan_out_err = _validate_replay_aggregate_join_fan_out(replay_intent, sub_schema)
        if fan_out_err is not None:
            raise AggregateJoinFanOutError(fan_out_err.scope_label, fan_out_err.message_for_caller)
        step_gates = dict(gate_map.get(step.source_id, {}))
        if manifest is not None and "allowed_where_ops" not in step_gates:
            engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
            allowed_where_ops = intersect_member_where_ops(dialects_by_source, engine_types_by_source=engine_types)
            binding = next((item for item in manifest.sources if item.source_id == step.source_id), None)
            if binding is not None:
                member_ops = extra_where_ops_for_engine(binding.engine)
                step_gates["allowed_where_ops"] = allowed_where_ops & (member_ops | set(FEDERATION_BASE_WHERE_OPS))
        runtime = dict(source_runtimes or {}).get(step.source_id)
        identity_token = None
        if runtime is not None:
            identity_token = push_engine_identity(_engine_identity_for_source_runtime(runtime))
        gen_out = None
        guard_limits = member_guard_limit_kwargs(manifest, step.source_id)
        try:
            gen_out = generate_and_validate_sql(
                q_norm,
                sub_intent,
                sub_schema,
                {},
                {},
                source_dialect,
                member_store,
                matched_template=tmpl,
                persist_template_learning=False,
                member_source_id=None if degenerate else step.source_id,
                **step_gates,
                **guard_limits,
            )
        finally:
            if identity_token is not None:
                pop_engine_identity(identity_token)
        if gen_out is None or not gen_out.success:
            return FederatedPrepareOutcome(
                success=False,
                plan=plan,
                display_sql="",
                sql_validation_error=(
                    gen_out.sql_validation_error if gen_out is not None else "federated replay validation failed"
                ),
                error_kind=gen_out.error_kind if gen_out is not None else None,
                source_id=step.source_id,
                phase="prepare",
            )
        tmpl_sd = (
            getattr(gen_out.matched_template, "structural_defaults", None)
            if gen_out.matched_template is not None
            else getattr(tmpl, "structural_defaults", None)
        )
        isolated_intent = RuntimeIntent.from_dict(sub_intent.to_dict())
        prepared.append(
            FederatedPreparedStep(
                source_id=step.source_id,
                sub_intent=isolated_intent,
                sql=gen_out.sql,
                structural_defaults=tmpl_sd,
                matched_template=gen_out.matched_template or tmpl,
            )
        )
        per_source.append((step.source_id, gen_out.sql))
    glue = render_federation_glue(
        plan, {sid: f"src_{sid}" for sid, _ in per_source}, schema=composite_schema, manifest=manifest
    )
    display_sql = _format_federated_sql_display(per_source, glue)
    return FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql=display_sql,
        steps=tuple(prepared),
        per_source_sql=tuple(per_source),
        glue_sql=glue,
        composite_schema_graph_id=str(composite_schema.schema_graph_id or ""),
        combine_hash=federation_plan_combine_hash(plan),
        step_fingerprints=federation_plan_step_fingerprints(
            plan, intent_key_fn=intent_key, manifest=manifest, member_graphs=member_graphs
        ),
        member_schema_graph_ids=federation_member_schema_graph_ids(plan, member_graphs),
        member_resolved_limits=federation_member_resolved_limits(plan, manifest) if manifest is not None else (),
    )


def _engine_identity_for_source_runtime(runtime: Any) -> EngineIdentity:
    """Bind federation SQL generation to the per-source runtime config (incl. DuckDB SCHEMA)."""
    engine_type = str(getattr(runtime, "engine", "") or "")
    runtime_config = getattr(getattr(runtime, "dialect", None), "config", None)
    if runtime_config is None:
        runtime_config = get_runtime_config_class(engine_type)
    return EngineIdentity(engine_type=engine_type, runtime_config=runtime_config)


def _federated_step_sql_context(
    step: SourceStep,
    composite_schema: SchemaGraph,
    *,
    dialect: Any,
    dialects_by_source: Mapping[str, Any] | None,
    source_runtimes: Mapping[str, Any] | None,
    manifest: FederationManifest | None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
) -> tuple[Any, SchemaGraph]:
    """Resolve per-source dialect and member schema slice for one federated step."""
    runtime = dict(source_runtimes or {}).get(step.source_id)
    dialect_map = dict(dialects_by_source or {})
    if runtime is not None and getattr(runtime, "dialect", None) is not None:
        source_dialect = runtime.dialect
    else:
        source_dialect = dialect_map.get(step.source_id, dialect)
    sub_schema = resolve_federated_member_schema(
        step.source_id, composite_schema, manifest=manifest, member_graphs=member_graphs
    )
    return source_dialect, sub_schema


def _federation_batch_join_scope_key(scope_counter: list[int]) -> str:
    scope_key = f"jc{scope_counter[0]}"
    scope_counter[0] += 1
    return scope_key


def _federation_batch_member_join_presets(
    q_norm: str,
    plan: FederatedPlan,
    composite_schema: SchemaGraph,
    *,
    dialect: Any,
    dialects_by_source: Mapping[str, Any] | None,
    manifest: FederationManifest | None,
    member_graphs: Mapping[str, SchemaGraph] | None,
    source_runtimes: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Resolve join choices for all federation members in one LLM round- trip."""
    all_llm_scopes: list[dict[str, Any]] = []
    preset: dict[str, str] = {}
    accept_na: dict[str, bool] = {}
    scope_to_member: dict[str, tuple[str, str]] = {}
    scope_counter = [0]

    for step in plan.steps:
        sub_intent = apply_projected_keys_to_intent(step.sub_intent, step.projected_keys)
        _source_dialect, sub_schema = _federated_step_sql_context(
            step,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            manifest=manifest,
            member_graphs=member_graphs,
        )
        step_join_candidates, _step_cmap, step_cte_hints = generate_join_candidates(sub_intent, sub_schema)
        virtual_specs = build_virtual_table_specs(sub_intent, sub_schema)
        main_tables_list = list(tables_in_join_scope(sub_intent.tables, sub_schema, virtual_specs))
        main_candidates = list(step_join_candidates.get("candidates") or [])
        cte_scopes: list[tuple[str, list[str], list[dict[str, Any]]]] = []
        for cte in sub_intent.cte_steps or []:
            if getattr(cte, "emission", "join_table") == "scalar_subquery":
                continue
            if not cte.tables or len(cte.tables) < 2:
                continue
            hints_entry = (step_cte_hints or {}).get(cte.cte_name) or {}
            cands = list(hints_entry.get("candidates") or [])
            tbls = list(tables_in_join_scope(cte.tables, sub_schema, virtual_specs))
            cte_scopes.append((cte.cte_name, tbls, cands))
        step_preset, pass1_llm, accept_na_map, _scope_class = join_scope_pass1_plan(
            main_multi_table=len(main_tables_list) >= 2,
            main_tables=main_tables_list,
            main_candidates=main_candidates,
            cte_scopes=cte_scopes,
            forbid_na=False,
        )
        step_scope_ids: dict[str, str] = {}

        def _prefix_step_scope(local: str) -> str:
            if local not in step_scope_ids:
                step_scope_ids[local] = _federation_batch_join_scope_key(scope_counter)
                scope_to_member[step_scope_ids[local]] = (step.source_id, local)
            return step_scope_ids[local]

        for scope_key, choice in step_preset.items():
            preset[_prefix_step_scope(scope_key)] = choice
        for scope_key, allow in accept_na_map.items():
            accept_na[_prefix_step_scope(scope_key)] = allow
        for scope in pass1_llm:
            local = str(scope["scope"])
            prefixed = _prefix_step_scope(local)
            all_llm_scopes.append({**scope, "scope": prefixed})

    if not all_llm_scopes:
        return {}

    if federation_turn_cancelled():
        _raise_federation_turn_cancelled(source_id="composite", phase="prepare")

    with llm_usage_attribution(phase="join_choice"):
        merged = get_join_choice_from_llm(
            q_norm,
            "SELECT 1",
            llm_scopes=all_llm_scopes,
            preset_choices=preset,
            accept_na_by_scope=accept_na,
            require_final=False,
            schema=composite_schema,
        )
    by_source: dict[str, dict[str, str]] = {}
    for scoped_key, choice in merged.items():
        mapped = scope_to_member.get(scoped_key)
        if mapped is None:
            continue
        source_id, local_key = mapped
        by_source.setdefault(source_id, {})[local_key] = choice
    return by_source


def persist_federated_member_stores(
    plan: FederatedPlan,
    *,
    store: dict[str, Any] | TemplateStoreView,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None,
) -> None:
    """Persist member template stores after a full federated prepare succeeds."""
    store_map = dict(stores_by_source or {})
    seen: set[int] = set()
    for step in plan.steps:
        if store_map and step.source_id not in store_map:
            raise FederationConfigError(f"federation member store missing for source_id {step.source_id!r}")
        step_store = store_map[step.source_id] if store_map else store
        store_key = id(step_store)
        if store_key in seen:
            continue
        seen.add(store_key)
        if isinstance(step_store, TemplateStoreView):
            save_template_store(step_store)


def prepare_federated_sql_plan(
    q_norm: str,
    plan: FederatedPlan,
    composite_schema: SchemaGraph,
    *,
    dialect: Any,
    dialects_by_source: Mapping[str, Any] | None,
    join_candidates: dict[str, Any],
    cmap: dict[str, list[str]],
    store: dict[str, Any] | TemplateStoreView,
    cte_join_hints: dict[str, dict[str, Any]] | None = None,
    persist_template_learning: bool = True,
    stores_by_source: Mapping[str, TemplateStoreView] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    manifest: FederationManifest | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    **gate_kwargs: Any,
) -> FederatedPrepareOutcome:
    """Generate per-source SQL for a federated plan without executing it."""
    emit_ask_phase(ASK_PHASE_M)
    debug(f"[{ASK_PHASE_M}] combine steps={len(plan.steps)} residual={plan.residual is not None}")
    if plan.ineligible_reason:
        return FederatedPrepareOutcome(
            success=False,
            plan=plan,
            display_sql="",
            sql_validation_error=plan.ineligible_reason,
            error_kind=FailureCategory.DENIED_REFERENCE.value,
            phase="prepare",
        )
    if not plan.steps:
        return FederatedPrepareOutcome(
            success=False, plan=plan, display_sql="", sql_validation_error="empty federated plan", phase="prepare"
        )
    degenerate = federation_plan_is_degenerate(plan)
    if federation_turn_cancelled():
        _raise_federation_turn_cancelled(source_id="composite", phase="prepare")
    join_presets_by_source = (
        _federation_batch_member_join_presets(
            q_norm,
            plan,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            manifest=manifest,
            member_graphs=member_graphs,
            source_runtimes=source_runtimes,
        )
        if not degenerate and len(plan.steps) > 1
        else {}
    )
    per_source: list[tuple[str, str]] = []
    prepared: list[FederatedPreparedStep] = []
    store_map = dict(stores_by_source or {})
    gate_map = dict(gate_kwargs_by_source or {})
    runtime_map = dict(source_runtimes or {})
    for step in plan.steps:
        if federation_turn_cancelled():
            _raise_federation_turn_cancelled(source_id=step.source_id, phase="prepare")
        source_dialect, sub_schema = _federated_step_sql_context(
            step,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            manifest=manifest,
            member_graphs=member_graphs,
        )
        if store_map and step.source_id not in store_map:
            raise FederationConfigError(f"federation member store missing for source_id {step.source_id!r}")
        step_store = store_map[step.source_id] if store_map else store
        step_gates = dict(gate_map.get(step.source_id, gate_kwargs))
        if manifest is not None and "allowed_where_ops" not in step_gates:
            engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
            allowed_where_ops = intersect_member_where_ops(dialects_by_source, engine_types_by_source=engine_types)
            binding = next((item for item in manifest.sources if item.source_id == step.source_id), None)
            if binding is not None:
                member_ops = extra_where_ops_for_engine(binding.engine)
                step_gates["allowed_where_ops"] = allowed_where_ops & (member_ops | set(FEDERATION_BASE_WHERE_OPS))
        runtime = runtime_map.get(step.source_id)
        sub_intent = (
            step.sub_intent if degenerate else apply_projected_keys_to_intent(step.sub_intent, step.projected_keys)
        )
        if not degenerate:
            sub_intent = expand_shared_pk_tables_for_refs(sub_intent, sub_schema)
            processed, post_issues = apply_runtime_post_processing(sub_intent, sub_schema, question_fallback=q_norm)
            if processed is None:
                return FederatedPrepareOutcome(
                    success=False,
                    plan=plan,
                    display_sql="",
                    sql_validation_error="federated member post-processing incomplete",
                    error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
                    source_id=step.source_id,
                    phase="prepare",
                )
            blocking = [issue for issue in post_issues if getattr(issue, "severity", "") == "error"]
            if blocking:
                return FederatedPrepareOutcome(
                    success=False,
                    plan=plan,
                    display_sql="",
                    sql_validation_error=str(getattr(blocking[0], "message", blocking[0])),
                    error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
                    source_id=step.source_id,
                    phase="prepare",
                )
            sub_intent = processed
            slice_error = validate_federated_sub_intent(sub_intent, sub_schema)
            if slice_error:
                return FederatedPrepareOutcome(
                    success=False,
                    plan=plan,
                    display_sql="",
                    sql_validation_error=slice_error,
                    error_kind=FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value,
                    source_id=step.source_id,
                    phase="prepare",
                )
        identity_token = None
        if runtime is not None:
            identity = _engine_identity_for_source_runtime(runtime)
            identity_token = push_engine_identity(identity)
        gen_out = None
        try:
            for attempt in range(2):
                with phase_timer(
                    "federation_generation",
                    source_id=step.source_id,
                    code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_GENERATED,
                    phase="prepare",
                ):
                    with llm_usage_attribution(phase="generation", source_id=step.source_id):
                        gen_out = generate_and_validate_sql(
                            q_norm,
                            sub_intent,
                            sub_schema,
                            join_candidates,
                            cmap,
                            source_dialect,
                            step_store,
                            cte_join_hints=cte_join_hints,
                            persist_template_learning=False,
                            member_source_id=None if degenerate else step.source_id,
                            join_preset_scope=join_presets_by_source.get(step.source_id),
                            **step_gates,
                        )
                if gen_out.success or attempt == 1:
                    break
        finally:
            if identity_token is not None:
                pop_engine_identity(identity_token)
        if gen_out is None or not gen_out.success:
            notify(
                "Federation member SQL generation failed during prepare.",
                stage="generation",
                code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
                level="error",
                source_id=step.source_id,
                details=(
                    ("source_id", step.source_id),
                    ("phase", "prepare"),
                    ("error_kind", str(gen_out.error_kind or "") if gen_out is not None else ""),
                ),
            )
            return FederatedPrepareOutcome(
                success=False,
                plan=plan,
                display_sql="",
                sql_validation_error=(
                    gen_out.sql_validation_error if gen_out is not None else "federated generation failed"
                ),
                error_kind=gen_out.error_kind if gen_out is not None else None,
                source_id=step.source_id,
                phase="prepare",
            )
        tmpl_sd = (
            getattr(gen_out.matched_template, "structural_defaults", None)
            if gen_out.matched_template is not None
            else None
        )
        per_source.append((step.source_id, gen_out.sql))
        isolated_intent = RuntimeIntent.from_dict(sub_intent.to_dict())
        prepared.append(
            FederatedPreparedStep(
                source_id=step.source_id,
                sub_intent=isolated_intent,
                sql=gen_out.sql,
                structural_defaults=tmpl_sd,
                matched_template=gen_out.matched_template,
            )
        )
    if degenerate:
        display_sql = per_source[0][1] if per_source else ""
        glue = ""
    else:
        glue = render_federation_glue(
            plan, {sid: f"src_{sid}" for sid, _ in per_source}, schema=composite_schema, manifest=manifest
        )
        display_sql = _format_federated_sql_display(per_source, glue)
    return FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql=display_sql,
        steps=tuple(prepared),
        per_source_sql=tuple(per_source),
        glue_sql=glue,
        composite_schema_graph_id=str(composite_schema.schema_graph_id or ""),
        combine_hash=federation_plan_combine_hash(plan),
        step_fingerprints=federation_plan_step_fingerprints(
            plan, intent_key_fn=intent_key, manifest=manifest, member_graphs=member_graphs
        ),
        member_schema_graph_ids=federation_member_schema_graph_ids(plan, member_graphs),
        member_resolved_limits=federation_member_resolved_limits(plan, manifest) if manifest is not None else (),
    )


def _member_statement_record(
    *,
    source_id: str,
    statement: str,
    row_count: int,
    runtime_map: Mapping[str, Any],
    manifest: FederationManifest | None,
    duration_ms: int | None = None,
    combine_kind: str = "",
) -> FederatedStatementRecord:
    runtime = runtime_map.get(source_id)
    engine_name = str(getattr(runtime, "engine", "") or "unknown")
    return FederatedStatementRecord(
        source_id=source_id,
        engine=engine_name,
        statement=statement,
        row_count=row_count,
        read_instant=datetime.now(timezone.utc).isoformat(),
        row_cap=source_row_cap_for_source(manifest, source_id) if manifest is not None else None,
        timeout_ms=source_timeout_for_source(manifest, source_id) if manifest is not None else None,
        duration_ms=duration_ms,
        phase="member",
        combine_kind=combine_kind,
    )


def _execute_degenerate_federation_plan(
    prepared: FederatedPrepareOutcome,
    composite_schema: SchemaGraph,
    *,
    dialect: Any,
    dialects_by_source: Mapping[str, Any] | None,
    source_runtimes: Mapping[str, Any] | None,
    manifest: FederationManifest | None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    gate_kwargs: Mapping[str, Any] | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> FederatedExecutionOutcome:
    """Execute a one-member federated plan directly on the member engine (no coordinator)."""
    step = prepared.plan.steps[0]
    prep_step = next((item for item in prepared.steps if item.source_id == step.source_id), None)
    if prep_step is None:
        raise FederationRuntimeError("degenerate federation plan missing prepared step")
    source_dialect, sub_schema = _federated_step_sql_context(
        step,
        composite_schema,
        dialect=dialect,
        dialects_by_source=dialects_by_source,
        source_runtimes=source_runtimes,
        manifest=manifest,
        member_graphs=member_graphs,
    )
    gate_map = dict(gate_kwargs_by_source or {})
    step_gates = dict(gate_map.get(step.source_id, gate_kwargs or {}))
    runtime_map = dict(source_runtimes or {})
    member_runtime = runtime_map.get(step.source_id)
    identity_token = None
    if member_runtime is not None:
        identity_token = push_engine_identity(_engine_identity_for_source_runtime(member_runtime))
    t0 = time.perf_counter()
    try:
        exec_params = dict(flatten_param_values(prep_step.sub_intent))
        exec_sql = source_dialect.finalize_render(
            prep_step.sql,
            exec_params,
            schema=sub_schema,
            intent=prep_step.sub_intent,
            execution_sql_override=prep_step.sql,
            structural_defaults=prep_step.structural_defaults,
        )
        exec_bind = reconcile_execute_bind_params(exec_sql, exec_params)
        guard_limits = member_guard_limit_kwargs(manifest, step.source_id)
        try:
            rows = execute_guarded_sql(
                source_dialect,
                exec_sql,
                exec_bind,
                schema=sub_schema,
                intent=prep_step.sub_intent,
                schema_role=str(step_gates.get("schema_role", "owner") or "owner"),
                schema_context=step_gates.get("schema_context"),
                visible_objects=step_gates.get("visible_objects"),
                context_name=str(step_gates.get("context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME),
                **guard_limits,
            )
        except StatementTimeoutError as exc:
            raise federation_member_timeout_error(step.source_id, exc) from exc
    finally:
        if identity_token is not None:
            pop_engine_identity(identity_token)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    row_values = rows
    runtime_map = dict(source_runtimes or {})
    statement = _member_statement_record(
        source_id=step.source_id,
        statement=prep_step.sql,
        row_count=len(row_values),
        runtime_map=runtime_map,
        manifest=manifest,
        duration_ms=duration_ms,
    )
    notify(
        f"federation member {step.source_id!r} returned {len(row_values)} rows",
        stage="execution",
        code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED,
        source_id=step.source_id,
        duration_ms=duration_ms,
        details=(("phase", "member"), ("row_count", str(len(row_values)))),
    )
    bundle = FederatedSqlBundle(
        statements=(statement,),
        display_sql=prepared.display_sql,
        column_names=(),
        read_window=((step.source_id, statement.read_instant),),
    )
    return FederatedExecutionOutcome(rows=tuple(tuple(row) for row in row_values), bundle=bundle)


def execute_federated_prepare(
    prepared: FederatedPrepareOutcome,
    composite_schema: SchemaGraph,
    *,
    dialect: Any,
    dialects_by_source: Mapping[str, Any] | None,
    source_runtimes: Mapping[str, Any] | None = None,
    coordinator_row_cap: int | None = None,
    manifest: FederationManifest | None = None,
    q_norm: str = "",
    join_candidates: dict[str, Any] | None = None,
    cmap: dict[str, list[str]] | None = None,
    store: dict[str, Any] | TemplateStoreView | None = None,
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    federation_dir: str | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    turn_session: Any | None = None,
    **gate_kwargs: Any,
) -> FederatedExecutionOutcome:
    """Execute a prepared federated plan and return coordinator rows plus a statement bundle."""
    emit_ask_phase(ASK_PHASE_L)
    debug(f"[{ASK_PHASE_L}] execute steps={len(prepared.steps)}")
    try:
        revalidate_prepared_federation_plan(prepared, composite_schema, manifest=manifest, member_graphs=member_graphs)
    except FederationConfigError:
        raise
    gate_map = dict(gate_kwargs_by_source or {})
    if federation_plan_is_degenerate(prepared.plan):
        return _execute_degenerate_federation_plan(
            prepared,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            manifest=manifest,
            member_graphs=member_graphs,
            gate_kwargs=gate_kwargs,
            gate_kwargs_by_source=gate_map,
        )
    statements: list[FederatedStatementRecord] = []
    dialect_map = dict(dialects_by_source or {})
    runtime_map = dict(source_runtimes or {})
    prepared_by_source = {step.source_id: step for step in prepared.steps}
    execution_steps = (
        order_federation_execution_steps(prepared.plan, schema=composite_schema, manifest=manifest)
        if prepared.plan.steps
        else prepared.plan.steps
    )
    stage_waves = federation_stage_execution_waves(prepared.plan, execution_steps, schema=composite_schema)
    semijoin_cap = manifest.coordinator.semijoin_key_cap if manifest is not None else 50_000
    parent_params: dict[str, Any] = {}
    if prepared.steps:
        parent_params = dict(flatten_param_values(prepared.steps[0].sub_intent))
    coordinator_params = coordinator_residual_bind_map(prepared.plan, parent_params)
    executed_sql_by_source: dict[str, str] = {}
    fed_ctx = FederationExecutionContext(
        plan_id=stable_json(
            federation_plan_step_fingerprints(
                prepared.plan, intent_key_fn=intent_key, manifest=manifest, member_graphs=member_graphs
            )
        ),
        audit_emit=(
            getattr(getattr(turn_session, "_owner", None), "_audit_emit", None)
            if turn_session is not None
            and callable(getattr(getattr(turn_session, "_owner", None), "_audit_emit", None))
            else None
        ),
    )
    plan_timeout_ms = (
        manifest.coordinator.plan_timeout_ms if manifest is not None else FederationCoordinatorConfig().plan_timeout_ms
    )
    fed_ctx.plan_started_monotonic = time.perf_counter()
    fed_ctx.plan_deadline_monotonic = federation_plan_timeout_deadline(
        plan_timeout_ms,
        started_at=fed_ctx.plan_started_monotonic,
    )
    if turn_session is not None:
        turn_session._active_federation_execution_context = fed_ctx
    fed_token = push_federation_execution_context(fed_ctx)
    frames: dict[str, pandas.DataFrame | CoordinatorMemberFrame] = {}
    executed: dict[str, pandas.DataFrame | CoordinatorMemberFrame] = {}
    combine_kind = federation_plan_combine_kind(prepared.plan)
    try:
        if effective_union_specs(prepared.plan) and len(execution_steps) > 1:
            for wave in stage_waves:
                if wave.stage.kind != "member":
                    emit_ask_phase(ASK_PHASE_L, stage=wave.stage)
                    continue
                member_wave = wave.member_steps
                if not member_wave:
                    continue
                for step in member_wave:
                    emit_ask_phase(ASK_PHASE_L, source=step.source_id, stage=wave.stage)
                try:
                    _enforce_active_federation_plan_timeout()
                except FederationCapExceededError as exc:
                    _raise_partial_member_failure(exc, source_id=member_wave[0].source_id, phase="member", succeeded=())
                wave_frames = _execute_federation_steps_parallel(
                    member_wave,
                    prepared_by_source=prepared_by_source,
                    composite_schema=composite_schema,
                    dialect_map=dialect_map,
                    dialect=dialect,
                    manifest=manifest,
                    q_norm=q_norm,
                    join_candidates=join_candidates,
                    cmap=cmap,
                    store=store,
                    gate_kwargs=gate_kwargs,
                    gate_kwargs_by_source=gate_map,
                    source_runtimes=source_runtimes,
                    executed_sql_by_source=executed_sql_by_source,
                    plan=prepared.plan,
                    semijoin_cap=semijoin_cap,
                    executed_shared=executed,
                )
                frames.update(wave_frames)
                wave_frames.clear()
            for step in execution_steps:
                frame = frames.get(step.source_id)
                prep_step = prepared_by_source.get(step.source_id)
                if prep_step is None:
                    raise FederationRuntimeError(f"federation source {step.source_id!r} has no prepared step")
                statements.append(
                    _member_statement_record(
                        source_id=step.source_id,
                        statement=executed_sql_by_source.get(step.source_id, prep_step.sql),
                        row_count=coordinator_member_row_count(frame),
                        runtime_map=runtime_map,
                        manifest=manifest,
                    )
                )
        else:
            succeeded: list[tuple[str, int, str]] = []
            for wave in stage_waves:
                if wave.stage.kind != "member":
                    emit_ask_phase(ASK_PHASE_L, stage=wave.stage)
                    continue
                member_wave = wave.member_steps
                if not member_wave:
                    continue
                for step in member_wave:
                    emit_ask_phase(ASK_PHASE_L, source=step.source_id, stage=wave.stage)
                try:
                    _enforce_active_federation_plan_timeout()
                except FederationCapExceededError as exc:
                    _raise_partial_member_failure(
                        exc, source_id=member_wave[0].source_id, phase="member", succeeded=succeeded
                    )
                if len(member_wave) > 1:
                    wave_frames = _execute_federation_steps_parallel(
                        member_wave,
                        prepared_by_source=prepared_by_source,
                        composite_schema=composite_schema,
                        dialect_map=dialect_map,
                        dialect=dialect,
                        manifest=manifest,
                        q_norm=q_norm,
                        join_candidates=join_candidates,
                        cmap=cmap,
                        store=store,
                        gate_kwargs=gate_kwargs,
                        gate_kwargs_by_source=gate_map,
                        source_runtimes=source_runtimes,
                        executed_sql_by_source=executed_sql_by_source,
                        plan=prepared.plan,
                        semijoin_cap=semijoin_cap,
                        executed_shared=executed,
                    )
                    for step in member_wave:
                        frame = wave_frames.get(step.source_id)
                        prep_step = prepared_by_source.get(step.source_id)
                        if prep_step is None:
                            continue
                        row_count = coordinator_member_row_count(frame)
                        statements.append(
                            _member_statement_record(
                                source_id=step.source_id,
                                statement=executed_sql_by_source.get(step.source_id, prep_step.sql),
                                row_count=row_count,
                                runtime_map=runtime_map,
                                manifest=manifest,
                            )
                        )
                        succeeded.append((step.source_id, row_count, datetime.now(timezone.utc).isoformat()))
                        if frame is not None:
                            frames[step.source_id] = frame
                    wave_frames.clear()
                    continue
                for step in member_wave:
                    if federation_turn_cancelled():
                        member_dialect = dialect_map.get(step.source_id, dialect)
                        _raise_federation_turn_cancelled(
                            source_id=step.source_id,
                            phase="member",
                            succeeded=succeeded,
                            dialect=member_dialect,
                        )
                    t0 = time.perf_counter()
                    try:
                        result_frame = _execute_federation_source_step_with_cancel(
                            step,
                            member_dialect=dialect_map.get(step.source_id, dialect),
                            succeeded=succeeded,
                            prepared_by_source=prepared_by_source,
                            composite_schema=composite_schema,
                            dialect_map=dialect_map,
                            dialect=dialect,
                            manifest=manifest,
                            executed=executed,
                            plan=prepared.plan,
                            semijoin_cap=semijoin_cap,
                            q_norm=q_norm,
                            join_candidates=join_candidates,
                            cmap=cmap,
                            store=store,
                            gate_kwargs=gate_map.get(step.source_id, gate_kwargs),
                            source_runtimes=source_runtimes,
                            executed_sql_by_source=executed_sql_by_source,
                        )
                    except Exception as exc:
                        _raise_partial_member_failure(
                            exc, source_id=step.source_id, phase="member", succeeded=succeeded
                        )
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    prep_step = prepared_by_source.get(step.source_id)
                    if prep_step is not None:
                        read_instant = datetime.now(timezone.utc).isoformat()
                        row_count = coordinator_member_row_count(result_frame)
                        statements.append(
                            _member_statement_record(
                                source_id=step.source_id,
                                statement=executed_sql_by_source.get(step.source_id, prep_step.sql),
                                row_count=row_count,
                                runtime_map=runtime_map,
                                manifest=manifest,
                                duration_ms=duration_ms,
                            )
                        )
                        succeeded.append((step.source_id, row_count, read_instant))
                        notify(
                            f"federation member {step.source_id!r} returned {row_count} rows",
                            stage="execution",
                            code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED,
                            source_id=step.source_id,
                            duration_ms=duration_ms,
                            details=(("phase", "member"), ("row_count", str(row_count))),
                        )
                    if result_frame is not None:
                        executed[step.source_id] = result_frame
                        frames[step.source_id] = result_frame
        if federation_turn_cancelled():
            _raise_federation_turn_cancelled(
                source_id="coordinator",
                phase="coordinator",
                succeeded=tuple(
                    (record.source_id, record.row_count, record.read_instant)
                    for record in statements
                    if record.phase == "member"
                ),
            )
        try:
            _enforce_active_federation_plan_timeout()
        except FederationCapExceededError as exc:
            _raise_partial_member_failure(
                exc,
                source_id="coordinator",
                phase="coordinator",
                succeeded=tuple(
                    (record.source_id, record.row_count, record.read_instant)
                    for record in statements
                    if record.phase == "member"
                ),
            )
        glue_sql = prepared.glue_sql or render_federation_glue(
            prepared.plan,
            {sid: f"src_{sid}" for sid in frames},
            schema=composite_schema,
            manifest=manifest,
            param_values=coordinator_params,
        )
        spill_dir = federation_coordinator_spill_dir(federation_dir)
        executed.clear()
        try:
            result_df = execute_federation_coordinator(
                frames,
                prepared.plan,
                row_cap=coordinator_row_cap,
                spill_row_threshold=(manifest.coordinator.spill_row_threshold if manifest is not None else None),
                spill_dir=spill_dir,
                schema=composite_schema,
                param_values=coordinator_params,
                total_input_byte_cap=(manifest.coordinator.total_input_byte_cap if manifest is not None else None),
                semijoin_key_cap=(manifest.coordinator.semijoin_key_cap if manifest is not None else None),
                coordinator_timeout_ms=(manifest.coordinator.coordinator_timeout_ms if manifest is not None else None),
            )
        except Exception as exc:
            if isinstance(exc, FederationConfigError):
                raise
            _raise_partial_member_failure(
                exc,
                source_id="coordinator",
                phase="coordinator",
                succeeded=tuple(
                    (record.source_id, record.row_count, record.read_instant)
                    for record in statements
                    if record.phase == "member"
                ),
            )
        executed.clear()
        frames.clear()
        coord_engine = "duckdb"
        notify(
            f"federation coordinator returned {len(result_df)} rows",
            stage="execution",
            code=DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_EXECUTED,
            source_id="coordinator",
            details=(("phase", "coordinator"), ("row_count", str(len(result_df)))),
        )
        statements.append(
            FederatedStatementRecord(
                source_id="coordinator",
                engine=coord_engine,
                statement=glue_sql,
                row_count=len(result_df),
                read_instant=datetime.now(timezone.utc).isoformat(),
                row_cap=coordinator_row_cap,
                phase="coordinator",
                combine_kind=combine_kind,
            )
        )
        column_names = federation_residual_column_headers(prepared.plan)
        if not column_names and not result_df.empty:
            column_names = tuple(str(c) for c in result_df.columns)
        rows: tuple[tuple[Any, ...], ...] = ()
        if not result_df.empty:
            rows = tuple(tuple(row) for row in result_df.itertuples(index=False, name=None))
        bundle = FederatedSqlBundle(
            statements=tuple(statements),
            display_sql=prepared.display_sql,
            column_names=column_names,
            read_window=tuple(
                (record.source_id, record.read_instant)
                for record in statements
                if record.phase == "member" and record.read_instant
            ),
        )
        return FederatedExecutionOutcome(rows=rows, bundle=bundle)
    finally:
        pop_federation_execution_context(fed_token)
        if turn_session is not None and getattr(turn_session, "_active_federation_execution_context", None) is fed_ctx:
            turn_session._active_federation_execution_context = None


def _as_member_execution_error(
    exc: BaseException, source_id: str, *, phase: str = "member"
) -> FederationMemberExecutionError:
    if isinstance(exc, FederationMemberExecutionError):
        return exc
    return FederationMemberExecutionError(str(exc), source_id=source_id, phase=phase)


def _emit_federation_semijoin_key_transfer_audit(
    *,
    source_member: str,
    target_member: str,
    column: str,
    key_count: int,
) -> None:
    """Record a semijoin key transfer in the active federation audit sink when configured."""
    fed_ctx = active_federation_execution_context()
    audit_emit = getattr(fed_ctx, "audit_emit", None) if fed_ctx is not None else None
    if not callable(audit_emit):
        return
    audit_emit(
        AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER,
        details=(
            ("source_member", str(source_member)),
            ("target_member", str(target_member)),
            ("column", str(column)),
            ("key_count", str(key_count)),
        ),
    )


def _member_execution_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RetryableError):
        return True
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, RetryableError):
        return True
    return False


def _federation_execution_failure_phase(raw: str) -> Literal["prepare", "member", "coordinator"]:
    if raw == "prepare":
        return "prepare"
    if raw == "coordinator":
        return "coordinator"
    return "member"


def _federated_prepare_outcome_for_execution_failure(
    prepared: FederatedPrepareOutcome,
    exc: BaseException,
) -> FederatedPrepareOutcome:
    """Stamp *prepared* with structured failure fields for execution- time errors."""
    if isinstance(exc, FederationPartialFailureError):
        return replace(
            prepared,
            success=False,
            source_id=exc.source_id,
            phase=_federation_execution_failure_phase(exc.phase),
            sql_validation_error=str(exc),
            error_kind="federation_execution_failed",
        )
    source_id = str(getattr(exc, "source_id", "") or "")
    phase = str(getattr(exc, "phase", "") or "")
    if isinstance(exc, FederationConfigError):
        return replace(
            prepared,
            success=False,
            source_id=source_id,
            phase=_federation_execution_failure_phase(phase or "prepare"),
            sql_validation_error=str(exc),
            error_kind="federation_execution_failed",
        )
    return replace(
        prepared,
        success=False,
        source_id=source_id,
        phase=_federation_execution_failure_phase(phase),
        sql_validation_error=str(exc),
        error_kind="federation_execution_failed",
    )


def _enforce_active_federation_plan_timeout() -> None:
    fed_ctx = active_federation_execution_context()
    if fed_ctx is None:
        return
    started_at = fed_ctx.plan_started_monotonic
    if started_at is None:
        started_at = time.perf_counter()
    enforce_federation_plan_timeout(fed_ctx.plan_deadline_monotonic, started_at=started_at)


def _raise_federation_turn_cancelled(
    *,
    source_id: str,
    phase: str,
    succeeded: Sequence[tuple[str, int, str]] = (),
    dialect_map: Mapping[str, Any] | None = None,
    batch: Sequence[SourceStep] | None = None,
    dialect: Any | None = None,
) -> None:
    """Cancel in-flight member statements and raise a structured cancellation outcome."""
    if dialect_map is not None and batch is not None:
        for step in batch:
            member_dialect = dialect_map.get(step.source_id, dialect)
            if member_dialect is not None:
                member_dialect.cancel_statement()
    elif dialect is not None:
        dialect.cancel_statement()
    notify(
        "Federated turn cancelled.",
        stage="execution",
        code=DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
        level="error",
        source_id=source_id,
        details=(
            ("source_id", source_id),
            ("phase", phase),
            ("succeeded", ",".join(item[0] for item in succeeded)),
        ),
    )
    raise FederationTurnCancelledError(
        "federated turn cancelled",
        source_id=source_id,
        phase=phase,
        succeeded=tuple(succeeded),
    )


def _execute_federation_source_step_with_cancel(
    step: SourceStep,
    *,
    member_dialect: Any,
    succeeded: Sequence[tuple[str, int, str]],
    **kwargs: Any,
) -> pandas.DataFrame | CoordinatorMemberFrame | None:
    """Run one member step in a worker thread so cancellation can interrupt it."""
    result_box: list[pandas.DataFrame | CoordinatorMemberFrame | None] = []
    error_box: list[BaseException] = []
    done = threading.Event()

    def _run() -> None:
        try:
            result_box.append(_execute_federation_source_step(step, **kwargs))
        except BaseException as exc:
            error_box.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    while not done.wait(timeout=0.02):
        if federation_turn_cancelled():
            member_dialect.cancel_statement()
            _raise_federation_turn_cancelled(
                source_id=step.source_id,
                phase="member",
                succeeded=succeeded,
                dialect=member_dialect,
            )
    worker.join(timeout=30.0)
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def _raise_partial_member_failure(
    exc: BaseException, *, source_id: str, phase: str, succeeded: Sequence[tuple[str, int, str]]
) -> None:
    if isinstance(exc, FederationTurnCancelledError):
        raise exc
    fed_ctx = active_federation_execution_context()
    if fed_ctx is not None:
        fed_ctx.cancel()
    member_exc = _as_member_execution_error(exc, source_id, phase=phase)
    notify(
        "Federation member execution failed.",
        stage="execution",
        code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
        level="error",
        source_id=member_exc.source_id,
        details=(("source_id", member_exc.source_id), ("phase", member_exc.phase), ("message", str(member_exc))),
    )
    raise FederationPartialFailureError(
        str(member_exc),
        source_id=member_exc.source_id,
        phase=member_exc.phase,
        succeeded=tuple(succeeded),
        retryable=_member_execution_retryable(exc),
    ) from member_exc


def _execute_federation_source_step(
    step: SourceStep,
    *,
    prepared_by_source: Mapping[str, FederatedPreparedStep],
    composite_schema: SchemaGraph,
    dialect_map: Mapping[str, Any],
    dialect: Any,
    manifest: FederationManifest | None,
    executed: Mapping[str, pandas.DataFrame | CoordinatorMemberFrame],
    plan: FederatedPlan,
    semijoin_cap: int,
    q_norm: str,
    join_candidates: dict[str, Any] | None,
    cmap: dict[str, list[str]] | None,
    store: dict[str, Any] | TemplateStoreView | None,
    gate_kwargs: Mapping[str, Any],
    source_runtimes: Mapping[str, Any] | None = None,
    executed_sql_by_source: dict[str, str] | None = None,
) -> pandas.DataFrame | CoordinatorMemberFrame | None:
    prep_step = prepared_by_source.get(step.source_id)
    if prep_step is None:
        raise FederationRuntimeError(f"federation source {step.source_id!r} has no prepared step")
    try:
        return _execute_federation_source_step_body(
            step,
            prep_step=prep_step,
            prepared_by_source=prepared_by_source,
            composite_schema=composite_schema,
            dialect_map=dialect_map,
            dialect=dialect,
            manifest=manifest,
            executed=executed,
            plan=plan,
            semijoin_cap=semijoin_cap,
            q_norm=q_norm,
            join_candidates=join_candidates,
            cmap=cmap,
            store=store,
            gate_kwargs=gate_kwargs,
            source_runtimes=source_runtimes,
            executed_sql_by_source=executed_sql_by_source,
        )
    except FederationMemberExecutionError:
        raise
    except Exception as exc:
        raise _as_member_execution_error(exc, step.source_id) from exc


def _execute_federation_source_step_body(
    step: SourceStep,
    *,
    prep_step: FederatedPreparedStep,
    prepared_by_source: Mapping[str, FederatedPreparedStep],
    composite_schema: SchemaGraph,
    dialect_map: Mapping[str, Any],
    dialect: Any,
    manifest: FederationManifest | None,
    executed: Mapping[str, pandas.DataFrame | CoordinatorMemberFrame],
    plan: FederatedPlan,
    semijoin_cap: int,
    q_norm: str,
    join_candidates: dict[str, Any] | None,
    cmap: dict[str, list[str]] | None,
    store: dict[str, Any] | TemplateStoreView | None,
    gate_kwargs: Mapping[str, Any],
    source_runtimes: Mapping[str, Any] | None = None,
    executed_sql_by_source: dict[str, str] | None = None,
) -> pandas.DataFrame | CoordinatorMemberFrame | None:
    runtime_map = dict(source_runtimes or {})
    member_runtime = runtime_map.get(step.source_id)
    identity_token = None
    if member_runtime is not None:
        identity_token = push_engine_identity(_engine_identity_for_source_runtime(member_runtime))
    try:
        return _execute_federation_source_step_body_impl(
            step,
            prep_step=prep_step,
            prepared_by_source=prepared_by_source,
            composite_schema=composite_schema,
            dialect_map=dialect_map,
            dialect=dialect,
            manifest=manifest,
            executed=executed,
            plan=plan,
            semijoin_cap=semijoin_cap,
            q_norm=q_norm,
            join_candidates=join_candidates,
            cmap=cmap,
            store=store,
            gate_kwargs=gate_kwargs,
            source_runtimes=source_runtimes,
            executed_sql_by_source=executed_sql_by_source,
        )
    finally:
        if identity_token is not None:
            pop_engine_identity(identity_token)


def _execute_federation_source_step_body_impl(
    step: SourceStep,
    *,
    prep_step: FederatedPreparedStep,
    prepared_by_source: Mapping[str, FederatedPreparedStep],
    composite_schema: SchemaGraph,
    dialect_map: Mapping[str, Any],
    dialect: Any,
    manifest: FederationManifest | None,
    executed: Mapping[str, pandas.DataFrame | CoordinatorMemberFrame],
    plan: FederatedPlan,
    semijoin_cap: int,
    q_norm: str,
    join_candidates: dict[str, Any] | None,
    cmap: dict[str, list[str]] | None,
    store: dict[str, Any] | TemplateStoreView | None,
    gate_kwargs: Mapping[str, Any],
    source_runtimes: Mapping[str, Any] | None = None,
    executed_sql_by_source: dict[str, str] | None = None,
) -> pandas.DataFrame | CoordinatorMemberFrame | None:
    execution_intent = prep_step.sub_intent
    sql_override: str | None = None
    source_dialect, sub_schema = _federated_step_sql_context(
        step,
        composite_schema,
        dialect=dialect,
        dialects_by_source=dialect_map,
        source_runtimes=source_runtimes,
        manifest=manifest,
    )
    if manifest is not None and executed and not effective_union_specs(plan):
        source_by_table = source_by_table_from_schema(composite_schema)
        semijoin_distinct_floor = int(manifest.coordinator.semijoin_key_distinct_floor)
        member_stage = member_stage_for_source(plan, step.source_id)
        reducing_edges = member_stage.reducing_edges if member_stage is not None else ()
        if reducing_edges:
            for edge in reducing_edges:
                if not source_semijoin_enabled(manifest, step.source_id):
                    notify(
                        f"semijoin reduction skipped for {step.source_id!r}",
                        stage="federation",
                        code=DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED,
                        source_id=step.source_id,
                        details=(
                            ("phase", "prepare"),
                            ("reason", "semijoin_disabled"),
                            ("driving_source_id", edge.driving_source_id),
                            ("target_source_id", edge.target_source_id),
                        ),
                    )
                    continue
                driving_frame = executed.get(edge.driving_source_id)
                if driving_frame is None:
                    continue
                driving_table = resolve_source_column_table(
                    composite_schema,
                    edge.driving_source_id,
                    edge.driving_key,
                    manifest=manifest,
                    source_by_table=source_by_table,
                    declared_table=declared_table_for_source_column(
                        plan,
                        edge.driving_source_id,
                        edge.driving_key,
                        manifest=manifest,
                        schema=composite_schema,
                        source_by_table=source_by_table,
                    ),
                )
                target_table = resolve_source_column_table(
                    composite_schema,
                    edge.target_source_id,
                    edge.target_key,
                    manifest=manifest,
                    source_by_table=source_by_table,
                    declared_table=declared_table_for_source_column(
                        plan,
                        edge.target_source_id,
                        edge.target_key,
                        manifest=manifest,
                        schema=composite_schema,
                        source_by_table=source_by_table,
                    ),
                )
                if driving_table is None or target_table is None:
                    raise FederationRuntimeError(
                        f"cannot resolve reducing-edge tables for {edge.driving_source_id!r} -> {edge.target_source_id!r}"
                    )
                if not semijoin_key_is_allowed(composite_schema, driving_table, edge.driving_key):
                    raise FederationRuntimeError(
                        f"reducing driving key {driving_table}.{edge.driving_key!r} is not allowed"
                    )
                if not semijoin_key_is_allowed(composite_schema, target_table, edge.target_key):
                    raise FederationRuntimeError(
                        f"reducing target key {target_table}.{edge.target_key!r} is not allowed"
                    )
                if not semijoin_key_passes_distinct_floor(
                    composite_schema, driving_table, edge.driving_key, floor=semijoin_distinct_floor
                ):
                    notify(
                        f"semijoin reduction skipped for {step.source_id!r}",
                        stage="federation",
                        code=DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED,
                        source_id=step.source_id,
                        details=(
                            ("phase", "prepare"),
                            ("reason", "low_cardinality"),
                            ("driving_source_id", edge.driving_source_id),
                            ("target_source_id", edge.target_source_id),
                        ),
                    )
                    continue
                keys = distinct_semijoin_keys(driving_frame, edge.driving_key, cap=semijoin_cap)
                if keys is None:
                    raise FederationCapExceededError(
                        f"federation semijoin key cap exceeded for member {step.source_id!r}: "
                        f"distinct keys on {edge.driving_key!r} exceed cap {semijoin_cap}",
                        limit_key="semijoin_key_cap",
                        source_id=step.source_id,
                    )
                _emit_federation_semijoin_key_transfer_audit(
                    source_member=edge.driving_source_id,
                    target_member=edge.target_source_id,
                    column=edge.driving_key,
                    key_count=len(keys),
                )
                value_type = column_where_value_type(composite_schema, target_table, edge.target_key)
                if edge.edge_kind == "filter_keys":
                    execution_intent = inject_filter_keys_where(
                        execution_intent, edge.target_key, keys, value_type=value_type
                    )
                else:
                    execution_intent = inject_semijoin_where(
                        execution_intent, edge.target_key, keys, value_type=value_type
                    )
        elif source_semijoin_enabled(manifest, step.source_id):
            for driving_id, driving_frame in executed.items():
                pair = semijoin_key_columns(plan, driving_id, step.source_id)
                if pair is None:
                    continue
                driving_key, target_key = pair
                join_allowed = False
                for join in manifest.cross_source_joins:
                    left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
                    right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
                    left_src = manifest.table_namespace.get(left_tbl, "")
                    right_src = manifest.table_namespace.get(right_tbl, "")
                    if left_src != driving_id and right_src != driving_id:
                        continue
                    if left_src != step.source_id and right_src != step.source_id:
                        continue
                    if left_src == right_src:
                        continue
                    key_match = (
                        left_src == driving_id
                        and left_col == driving_key
                        and right_src == step.source_id
                        and right_col == target_key
                    ) or (
                        right_src == driving_id
                        and right_col == driving_key
                        and left_src == step.source_id
                        and left_col == target_key
                    )
                    if not key_match:
                        continue
                    join_allowed = reducing_edge_allowed_for_target(
                        step.source_id, join, manifest, schema=composite_schema
                    )
                    break
                if not join_allowed:
                    continue
                driving_table = resolve_source_column_table(
                    composite_schema,
                    driving_id,
                    driving_key,
                    manifest=manifest,
                    source_by_table=source_by_table,
                    declared_table=declared_table_for_source_column(
                        plan,
                        driving_id,
                        driving_key,
                        manifest=manifest,
                        schema=composite_schema,
                        source_by_table=source_by_table,
                    ),
                )
                target_table = resolve_source_column_table(
                    composite_schema,
                    step.source_id,
                    target_key,
                    manifest=manifest,
                    source_by_table=source_by_table,
                    declared_table=declared_table_for_source_column(
                        plan,
                        step.source_id,
                        target_key,
                        manifest=manifest,
                        schema=composite_schema,
                        source_by_table=source_by_table,
                    ),
                )
                if driving_table is None or target_table is None:
                    raise FederationRuntimeError(
                        f"cannot resolve semijoin tables for {driving_id!r} -> {step.source_id!r}"
                    )
                if not semijoin_key_is_allowed(composite_schema, driving_table, driving_key):
                    raise FederationRuntimeError(f"semijoin driving key {driving_table}.{driving_key!r} is not allowed")
                if not semijoin_key_is_allowed(composite_schema, target_table, target_key):
                    raise FederationRuntimeError(f"semijoin target key {target_table}.{target_key!r} is not allowed")
                if not semijoin_key_passes_distinct_floor(
                    composite_schema, driving_table, driving_key, floor=semijoin_distinct_floor
                ):
                    notify(
                        f"semijoin reduction skipped for {step.source_id!r}",
                        stage="federation",
                        code=DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED,
                        source_id=step.source_id,
                        details=(
                            ("phase", "prepare"),
                            ("reason", "low_cardinality"),
                            ("driving_source_id", driving_id),
                            ("target_source_id", step.source_id),
                        ),
                    )
                    continue
                keys = distinct_semijoin_keys(driving_frame, driving_key, cap=semijoin_cap)
                if keys is None:
                    raise FederationCapExceededError(
                        f"federation semijoin key cap exceeded for member {step.source_id!r}: "
                        f"distinct keys on {driving_key!r} exceed cap {semijoin_cap}",
                        limit_key="semijoin_key_cap",
                        source_id=step.source_id,
                    )
                _emit_federation_semijoin_key_transfer_audit(
                    source_member=driving_id,
                    target_member=step.source_id,
                    column=driving_key,
                    key_count=len(keys),
                )
                value_type = column_where_value_type(composite_schema, target_table, target_key)
                execution_intent = inject_semijoin_where(execution_intent, target_key, keys, value_type=value_type)
        if execution_intent is not prep_step.sub_intent:
            slice_error = validate_federated_sub_intent(execution_intent, sub_schema)
            if slice_error:
                raise FederationRuntimeError(
                    f"reduction-reduced sub-intent invalid for {step.source_id!r}: {slice_error}"
                )
    if execution_intent is not prep_step.sub_intent:
        if join_candidates is None or cmap is None or store is None:
            raise FederationRuntimeError("semijoin reduction requires SQL regeneration context")
        regen = generate_and_validate_sql(
            q_norm,
            execution_intent,
            sub_schema,
            join_candidates,
            cmap,
            source_dialect,
            store,
            persist_template_learning=False,
            **gate_kwargs,
        )
        if not regen.success:
            raise FederationRuntimeError(regen.sql_validation_error or "federated semijoin regen failed")
        sql_override = regen.sql
    parent_params = dict(flatten_param_values(prep_step.sub_intent))
    narrowed_bind = narrow_bind_map_for_sub_intent(execution_intent, parent_params)
    space_tables = frozenset(gate_kwargs.get("space_allowed_tables") or ())
    space_columns = frozenset(gate_kwargs.get("space_allowed_columns") or ())
    if space_tables or space_columns:
        if not assert_intent_in_scope(execution_intent, space_tables, space_columns, sub_schema):
            raise FederationRuntimeError(
                f"member {step.source_id!r} statement is outside the session aetherspace scope"
            )
    schema_context = gate_kwargs.get("schema_context")
    visible_objects = gate_kwargs.get("visible_objects")
    schema_role = str(gate_kwargs.get("schema_role", "owner") or "owner")
    context_name = str(gate_kwargs.get("context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
    if schema_context is not None and _execution_scope_gate_active(
        schema_context, visible_objects, schema_role, context_name=context_name
    ):
        if not assert_consumer_intent_in_scope(execution_intent, schema_context, sub_schema, visible_objects):
            raise FederationRuntimeError(f"member {step.source_id!r} statement is outside the execution scope")
    executed_sql = sql_override or prep_step.sql
    exec_params = dict(narrowed_bind) if narrowed_bind is not None else dict(flatten_param_values(execution_intent))
    exec_sql = source_dialect.finalize_render(
        executed_sql,
        dict(flatten_param_values(execution_intent)),
        schema=sub_schema,
        intent=execution_intent,
        execution_sql_override=sql_override,
        structural_defaults=prep_step.structural_defaults,
    )
    exec_bind = reconcile_execute_bind_params(exec_sql, exec_params)
    guard_limits = member_guard_limit_kwargs(manifest, step.source_id)
    column_names = member_frame_column_names(step)
    if execution_intent.grain == "scalar":
        result_frame: pandas.DataFrame | CoordinatorMemberFrame | None = None
    elif dialect_streams_arrow_to_coordinator(source_dialect):
        try:
            arrow_table = execute_guarded_arrow_table(
                source_dialect,
                exec_sql,
                exec_bind,
                schema=sub_schema,
                intent=execution_intent,
                schema_role=schema_role,
                schema_context=schema_context,
                visible_objects=visible_objects,
                context_name=context_name,
                **guard_limits,
            )
        except StatementTimeoutError as exc:
            raise federation_member_timeout_error(step.source_id, exc) from exc
        result_frame = CoordinatorMemberFrame(kind="arrow", table=arrow_table, column_names=column_names)
    else:
        try:
            step_rows = execute_guarded_sql(
                source_dialect,
                exec_sql,
                exec_bind,
                schema=sub_schema,
                intent=execution_intent,
                schema_role=schema_role,
                schema_context=schema_context,
                visible_objects=visible_objects,
                context_name=context_name,
                **guard_limits,
            )
        except StatementTimeoutError as exc:
            raise federation_member_timeout_error(step.source_id, exc) from exc
        if step_rows:
            result_frame = pandas.DataFrame(
                step_rows, columns=column_names or [f"c{i}" for i in range(len(step_rows[0]))]
            )
        else:
            result_frame = pandas.DataFrame(columns=list(column_names) if column_names else None)
    executed_sql = sql_override or prep_step.sql
    if executed_sql_by_source is not None:
        executed_sql_by_source[step.source_id] = executed_sql
    if manifest is not None and result_frame is not None:
        row_cap = source_row_cap_for_source(manifest, step.source_id)
        row_count = coordinator_member_row_count(result_frame)
        if row_count > row_cap:
            raise FederationCapExceededError(
                f"federation row cap exceeded for source {step.source_id!r}: {row_count} rows > cap {row_cap}",
                limit_key="row_cap",
                source_id=step.source_id,
            )
    validate_member_frame_projection(step, result_frame)
    return result_frame


_T = TypeVar("_T")


def _run_in_captured_context(
    ctx: Context,
    fn: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    return ctx.run(fn, *args, **kwargs)


def _execute_federation_steps_parallel(
    execution_steps: Sequence[SourceStep],
    *,
    prepared_by_source: Mapping[str, FederatedPreparedStep],
    composite_schema: SchemaGraph,
    dialect_map: Mapping[str, Any],
    dialect: Any,
    manifest: FederationManifest | None,
    q_norm: str,
    join_candidates: dict[str, Any] | None,
    cmap: dict[str, list[str]] | None,
    store: dict[str, Any] | TemplateStoreView | None,
    gate_kwargs: Mapping[str, Any],
    gate_kwargs_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    executed_sql_by_source: dict[str, str] | None = None,
    plan: FederatedPlan | None = None,
    semijoin_cap: int = 0,
    executed_shared: dict[str, pandas.DataFrame | CoordinatorMemberFrame] | None = None,
) -> dict[str, pandas.DataFrame | CoordinatorMemberFrame]:
    frames: dict[str, pandas.DataFrame | CoordinatorMemberFrame] = {}
    succeeded: list[tuple[str, int, str]] = []
    gate_map = dict(gate_kwargs_by_source or {})
    shared_executed = executed_shared if executed_shared is not None else {}
    active_plan = plan if plan is not None else FederatedPlan(steps=())
    batches = federation_member_execution_batches(execution_steps, manifest, plan=active_plan)
    for batch in batches:
        if federation_turn_cancelled():
            _raise_federation_turn_cancelled(
                source_id=batch[0].source_id if batch else "composite",
                phase="member",
                succeeded=succeeded,
            )
        try:
            _enforce_active_federation_plan_timeout()
        except FederationCapExceededError as exc:
            _raise_partial_member_failure(exc, source_id=batch[0].source_id, phase="member", succeeded=succeeded)
        max_workers = federation_member_parallelism_cap(manifest, len(batch))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures: dict[Any, str] = {}
            for step in batch:
                captured_ctx = copy_context()
                future = pool.submit(
                    _run_in_captured_context,
                    captured_ctx,
                    _execute_federation_source_step,
                    step,
                    prepared_by_source=prepared_by_source,
                    composite_schema=composite_schema,
                    dialect_map=dialect_map,
                    dialect=dialect,
                    manifest=manifest,
                    executed=dict(shared_executed),
                    plan=active_plan,
                    semijoin_cap=semijoin_cap,
                    q_norm=q_norm,
                    join_candidates=join_candidates,
                    cmap=cmap,
                    store=store,
                    gate_kwargs=gate_map.get(step.source_id, gate_kwargs),
                    source_runtimes=source_runtimes,
                    executed_sql_by_source=executed_sql_by_source,
                )
                futures[future] = step.source_id
            pending = set(futures.keys())
            for future in as_completed(futures):
                if federation_turn_cancelled():
                    for other in pending:
                        if not other.done():
                            other.cancel()
                    _raise_federation_turn_cancelled(
                        source_id=futures[future],
                        phase="member",
                        succeeded=succeeded,
                        dialect_map=dialect_map,
                        batch=batch,
                        dialect=dialect,
                    )
                source_id = futures[future]
                pending.discard(future)
                try:
                    frame = future.result()
                except Exception as exc:
                    for other in pending:
                        if not other.done():
                            other.cancel()
                    _raise_partial_member_failure(exc, source_id=source_id, phase="member", succeeded=succeeded)
                if frame is not None:
                    frames[source_id] = frame
                    shared_executed[source_id] = frame
                    succeeded.append(
                        (source_id, coordinator_member_row_count(frame), datetime.now(timezone.utc).isoformat())
                    )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return frames


def persist_federated_warmup_learning(
    q_norm: str,
    parent_intent: RuntimeIntent,
    prepared: FederatedPrepareOutcome,
    composite_schema: SchemaGraph,
    *,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]],
    dialects_by_source: Mapping[str, Any] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    federation_dir: str | None = None,
    federation_manifest: FederationManifest | None = None,
    form_storage: QuestionFormStorage | None = None,
    question_phrases: Sequence[str] | None = None,
) -> list[Any]:
    """Write per-member templates and a composite plan record after a successful federated warmup turn."""
    if not prepared.success or not prepared.steps:
        return []
    store_map = dict(stores_by_source)
    dialect_map = dict(dialects_by_source or {})
    graphs = dict(member_graphs or {})
    for step in prepared.steps:
        if step.source_id not in store_map:
            raise FederationConfigError(f"federation member store missing for source_id {step.source_id!r}")
    plan_id = intent_key(parent_intent)
    member_template_ids: list[tuple[str, str]] = []
    created: list[Any] = []
    phrases = [p for p in (question_phrases or ()) if p and str(p).strip()]
    for step in prepared.steps:
        member_store = store_map[step.source_id]
        if isinstance(member_store, TemplateStoreView):
            member_templates = cast(dict[str, Any], member_store["templates"])
        else:
            member_templates = cast(dict[str, Any], member_store.setdefault("templates", {}))
        member_graph = graphs.get(step.source_id)
        sub_schema = member_schema_slice(
            composite_schema, step.source_id, manifest=federation_manifest, member_graph=member_graph
        )
        source_dialect = dialect_map.get(step.source_id)
        member_q = member_feedback_q_norm(step.source_id, q_norm)
        member_form = form_storage
        if member_form is None and phrases:
            first = phrases[0].strip()
            member_form = QuestionFormStorage(
                corrected=first,
                normalized_optional=normalize_question(first) if normalize_question(first) != first else None,
            )
        tmpl = insert_template(
            member_store,
            member_templates,
            sub_schema,
            member_q,
            step.sub_intent,
            step.sql,
            dialect=source_dialect,
            form_storage=member_form,
            member_source_id=step.source_id,
            federation_plan_id=plan_id,
            federation_plan_only=True,
            record_accept=True,
        )
        stamp_federation_member_template(tmpl, plan_id=plan_id, source_id=step.source_id)
        save_template_store(member_store)
        member_template_ids.append((step.source_id, str(tmpl.id)))
        created.append(tmpl)
    if federation_dir:
        manifest_hash_value = ""
        member_tuple_hash_value = ""
        if federation_manifest is not None and graphs:
            manifest_hash_value, member_tuple_hash_value = federation_plan_topology_identity(
                graphs, federation_manifest
            )
        save_federation_plan_template(
            federation_dir,
            FederationPlanTemplate(
                plan_id=plan_id,
                composite_schema_graph_id=str(
                    prepared.composite_schema_graph_id or composite_schema.schema_graph_id or ""
                ),
                intent_key=plan_id,
                step_fingerprints=prepared.step_fingerprints,
                combine_hash=prepared.combine_hash or federation_plan_combine_hash(prepared.plan),
                question=q_norm,
                accepted_questions=(q_norm,),
                member_template_ids=tuple(member_template_ids),
                residual_hash=federation_plan_residual_hash(prepared.plan),
                manifest_hash=manifest_hash_value,
                member_tuple_hash=member_tuple_hash_value,
            ),
        )
    return created


def execute_federated_warmup_intent(
    q_norm: str,
    intent: RuntimeIntent,
    composite_schema: SchemaGraph,
    dialect: Any,
    *,
    federation_manifest: FederationManifest,
    federation_mappings: FederationMappings | None = None,
    stores_by_source: Mapping[str, TemplateStoreView | dict[str, Any]] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    federation_dir: str | None = None,
    persist_template_learning: bool = True,
) -> tuple[bool, str | None, list[Any] | None, str, FederatedPrepareOutcome | None]:
    """Plan, prepare, and execute a warmup intent through the federated coordinator. Member generation during prepare never persists learning. When ``persist_template_learning`` is True the caller must invoke :func:`persist_federated_warmup_learning` only after the full warmup turn succeeds; failed and partially-failed turns must not call it."""
    if not stores_by_source:
        return False, "federation warmup requires per-member template stores", None, "", None
    plan = plan_federated_intent(
        intent,
        composite_schema,
        federation_manifest,
        federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION),
        member_graphs=member_graphs,
    )
    if plan.ineligible_reason:
        return False, plan.ineligible_reason, None, "", None
    join_candidates, cmap, cte_join_hints = generate_join_candidates(intent, composite_schema)
    try:
        outcome = execute_federated_sql_plan(
            q_norm,
            plan,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            join_candidates=join_candidates,
            cmap=cmap,
            store={},
            cte_join_hints=cte_join_hints,
            persist_template_learning=False,
            source_runtimes=source_runtimes,
            manifest=federation_manifest,
            member_graphs=member_graphs,
            stores_by_source=cast(Mapping[str, TemplateStoreView], stores_by_source),
            federation_dir=federation_dir,
        )
    except (FederationConfigError, FederationRuntimeError, FederationPartialFailureError) as exc:
        return False, str(exc), None, "", None
    if not outcome.success:
        return (False, outcome.sql_validation_error or "federated warmup prepare failed", None, outcome.sql or "", None)
    rows = [tuple(row) for row in outcome.rows]
    learning_payload = outcome.prepared if persist_template_learning else None
    return True, None, rows, outcome.sql, learning_payload


def execute_federated_sql_plan(
    q_norm: str,
    plan: FederatedPlan,
    composite_schema: SchemaGraph,
    *,
    dialect: Any,
    dialects_by_source: Mapping[str, Any] | None,
    join_candidates: dict[str, Any],
    cmap: dict[str, list[str]],
    store: dict[str, Any] | TemplateStoreView,
    cte_join_hints: dict[str, dict[str, Any]] | None = None,
    coordinator_row_cap: int | None = None,
    manifest: FederationManifest | None = None,
    source_runtimes: Mapping[str, Any] | None = None,
    persist_template_learning: bool = True,
    **gate_kwargs: Any,
) -> FederatedSqlOutcome:
    """Generate and execute per-source SQL, then combine frames in the coordinator."""
    prepared = prepare_federated_sql_plan(
        q_norm,
        plan,
        composite_schema,
        dialect=dialect,
        dialects_by_source=dialects_by_source,
        join_candidates=join_candidates,
        cmap=cmap,
        store=store,
        cte_join_hints=cte_join_hints,
        persist_template_learning=persist_template_learning,
        source_runtimes=source_runtimes,
        manifest=manifest,
        **gate_kwargs,
    )
    if not prepared.success:
        return FederatedSqlOutcome(
            success=False,
            sql="",
            rows=(),
            prepared=prepared,
            sql_validation_error=prepared.sql_validation_error,
            error_kind=prepared.error_kind,
        )
    try:
        exec_outcome = execute_federated_prepare(
            prepared,
            composite_schema,
            dialect=dialect,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            coordinator_row_cap=coordinator_row_cap,
            manifest=manifest,
            q_norm=q_norm,
            join_candidates=join_candidates,
            cmap=cmap,
            store=store,
            **gate_kwargs,
        )
    except (FederationConfigError, FederationRuntimeError, FederationPartialFailureError) as exc:
        failed_prepared = _federated_prepare_outcome_for_execution_failure(prepared, exc)
        return FederatedSqlOutcome(
            success=False,
            sql=prepared.display_sql,
            rows=(),
            prepared=failed_prepared,
            sql_validation_error=failed_prepared.sql_validation_error,
            error_kind=failed_prepared.error_kind,
        )
    store_map = dict(gate_kwargs.get("stores_by_source") or {})
    if persist_template_learning and store_map:
        persist_federated_member_stores(prepared.plan, store=store, stores_by_source=store_map)
    return FederatedSqlOutcome(
        success=True,
        sql=prepared.display_sql,
        rows=exec_outcome.rows,
        per_source_sql=prepared.per_source_sql,
        bundle=exec_outcome.bundle,
        prepared=prepared,
    )


def _join_path_from_intent(intent: RuntimeIntent) -> tuple[str, ...]:
    signature = tuple(str(item) for item in (intent.chosen_join_path_signature or []) if str(item).strip())
    if signature:
        return signature
    for step in intent.cte_steps or ():
        cte_sig = tuple(str(item) for item in (step.chosen_join_path_signature or []) if str(item).strip())
        if cte_sig:
            return cte_sig
    return ()


def build_plan_preview_from_intent(
    question: str,
    intent: RuntimeIntent,
    *,
    federated_plan: Any | None = None,
) -> PlanPreviewResult:
    """Project one parsed intent into a caller-visible turn plan summary."""
    tables = tuple(sorted(str(table) for table in (intent.tables or ()) if str(table).strip()))
    join_path = _join_path_from_intent(intent)
    if federated_plan is None:
        return PlanPreviewResult(
            question=question,
            tables=tables,
            join_path=join_path,
        )
    ineligible = str(getattr(federated_plan, "ineligible_reason", None) or "") or None
    member_ids = tuple(
        sorted(
            {
                str(step.source_id)
                for step in (getattr(federated_plan, "steps", ()) or ())
                if str(step.source_id).strip()
            }
        )
    )
    combine = getattr(federated_plan, "combine", None)
    union_specs = getattr(federated_plan, "union_specs", ()) or ()
    federates = bool(ineligible is None and (len(member_ids) > 1 or combine or union_specs))
    return PlanPreviewResult(
        question=question,
        tables=tables,
        join_path=join_path,
        member_source_ids=member_ids,
        federates=federates,
        ineligible_reason=ineligible,
    )


@contextmanager
def _owner_business_knowledge_scope_for_preview(owner: Any):
    holder = getattr(owner, "_business_knowledge", None)
    if holder is None:
        yield
        return
    with business_knowledge_scope(**holder.scope_kwargs()):
        yield


def _parse_intent_for_preview(
    owner: Any,
    question: str,
    schema_graph: SchemaGraph,
    *,
    visible_objects: frozenset[str] | None,
    execution_visible_objects: frozenset[str] | None,
    space_columns: frozenset[str] | None,
    space_deny_objects: frozenset[str] | None,
    space_deny_columns: frozenset[str] | None,
    space_description_overlay: dict[str, Any] | None,
    store: dict[str, Any] | None,
) -> RuntimeIntent | None:
    q_norm = normalize_question(question)
    intent_visible = resolve_intent_visible_objects(
        visible_objects=visible_objects,
        execution_visible_objects=execution_visible_objects,
    )
    with _owner_business_knowledge_scope_for_preview(owner):
        intent, _warns, _calls, _plan = invoke_intent_parse_with_hints(
            q_norm,
            schema_graph,
            store=store,
            persist_template_learning=False,
            visible_objects=intent_visible,
            allowed_columns=space_columns,
            deny_objects=space_deny_objects,
            deny_columns=space_deny_columns,
            description_overlay=space_description_overlay,
        )
    return intent


def preview_plan_on_engine(
    engine: Any,
    question: str,
    *,
    visible_objects: frozenset[str] | None = None,
    execution_visible_objects: frozenset[str] | None = None,
    space_columns: frozenset[str] | None = None,
    space_deny_objects: frozenset[str] | None = None,
    space_deny_columns: frozenset[str] | None = None,
    space_description_overlay: dict[str, Any] | None = None,
) -> PlanPreviewResult:
    """Plan one engine turn without generating or executing SQL."""
    q_norm = normalize_question(question)
    intent = _parse_intent_for_preview(
        engine,
        question,
        engine._schema_graph,
        visible_objects=visible_objects,
        execution_visible_objects=execution_visible_objects,
        space_columns=space_columns,
        space_deny_objects=space_deny_objects,
        space_deny_columns=space_deny_columns,
        space_description_overlay=space_description_overlay,
        store=getattr(engine, "_store", None),
    )
    if intent is None:
        return PlanPreviewResult(
            question=q_norm,
            tables=(),
            join_path=(),
            ineligible_reason=PLAN_PREVIEW_INTENT_PARSE_FAILED,
        )
    return build_plan_preview_from_intent(q_norm, intent)


def preview_plan_on_federation(
    federation: Any,
    question: str,
    *,
    space: SpaceContext | None = None,
    visible_objects: frozenset[str] | None = None,
    execution_visible_objects: frozenset[str] | None = None,
    space_columns: frozenset[str] | None = None,
    space_deny_objects: frozenset[str] | None = None,
    space_deny_columns: frozenset[str] | None = None,
    space_description_overlay: dict[str, Any] | None = None,
) -> PlanPreviewResult:
    """Plan one federation turn without generating or executing SQL."""
    q_norm = normalize_question(question)
    manifest = getattr(federation, "_federation_manifest", None)
    if manifest is None:
        return PlanPreviewResult(
            question=q_norm,
            tables=(),
            join_path=(),
            ineligible_reason="federation manifest not loaded",
        )
    intent = _parse_intent_for_preview(
        federation,
        question,
        federation._schema_graph,
        visible_objects=visible_objects,
        execution_visible_objects=execution_visible_objects,
        space_columns=space_columns,
        space_deny_objects=space_deny_objects,
        space_deny_columns=space_deny_columns,
        space_description_overlay=space_description_overlay,
        store=getattr(federation, "_store", None),
    )
    if intent is None:
        return PlanPreviewResult(
            question=q_norm,
            tables=(),
            join_path=(),
            ineligible_reason=PLAN_PREVIEW_INTENT_PARSE_FAILED,
        )
    mappings = getattr(federation, "_federation_mappings", None)
    member_graphs = getattr(federation, "_federation_member_graphs", None) or {}
    federated_plan = plan_federated_intent(
        intent,
        federation._schema_graph,
        manifest,
        mappings,
        space=space,
        member_graphs=member_graphs,
    )
    return build_plan_preview_from_intent(q_norm, intent, federated_plan=federated_plan)
