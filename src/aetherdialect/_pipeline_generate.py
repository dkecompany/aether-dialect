"""Pipeline generate path: scope gates, intent parse, joins, validate, reuse choice."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, cast

from ._config import EngineConfig, PolicyConfig
from ._constants import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_FEDERATION_JOIN_CANDIDATE_CAP,
    DIAGNOSTIC_CODE_REUSE_MISS,
    INTERACTIVE_STAGE_INTENT_CONFIRM,
    JOIN_CHOICE_SCOPE_MAIN,
    MASTER_AETHERSPACE_NAME,
    PIPELINE_BUG_SQL_VALIDATION,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    SHAPE_QUESTION_INDEX_KEY,
    SOFT_DIAGNOSTIC_CODES,
    SQL_BIND_TOKEN_RE,
    TEMPLATE_INTENT_KEY_INDEX_KEY,
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    VALUE_TYPE_NORMALIZATION,
)
from ._constants_runtime import (
    ASK_PHASE_A,
    ASK_PHASE_B,
    ASK_PHASE_H,
    ASK_PHASE_J,
    ASK_PHASE_K,
    ASK_PHASE_N,
    DISPLAY_ALIAS_PROMPT_KEY_ORDER,
    FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER,
)
from ._contracts_base import (
    EngineContext,
    FailureCategory,
    FederationConfigError,
    FederationContext,
    PredicateGroup,
    SqlDiagnostic,
)
from ._contracts_core import (
    AggregateJoinFanOutError,
    ClauseWidenedRowsetError,
    ComparisonJoinScopeExceededError,
    ConcreteIntent,
    FederatedPlan,
    FederatedPreparedStep,
    FeedbackKind,
    GenerationPath,
    InteractiveTailSnapshot,
    InterpretPlan,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    JoinPathKeyTypeError,
    NoJoinPathError,
    PipelineSessionMarker,
    PipelineSuspended,
    ProbeCtePlacementError,
    QuestionFormStorage,
    RefinementContext,
    RefinementRetry,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
    SqlGenerationOutcome,
    Template,
    TemplateMatch,
    UserFeedbackRejectSuspendContext,
    ValueHistory,
    WriteQueueEvent,
)
from ._contracts_schema import (
    FederationPlanTemplate,
    SchemaGraph,
)
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._federation_execute import (
    credit_federation_plan_accept,
    delete_unaccepted_federation_plan_template,
    federation_user_facing_ineligible_message,
    mirror_federation_plan_join_feedback,
    record_federation_join_feedback,
)
from ._federation_manifest import (
    federation_plan_sql_shape,
    federation_residual_column_headers,
    federation_scaled_join_candidate_cap,
    federation_scaled_join_path_tie_cap,
    member_feedback_q_norm,
    schema_spans_multiple_sources,
    stamp_federation_member_template,
)
from ._intent_bind import join_path_key_concrete, prune_unused_cte_steps
from ._intent_expr import (
    build_virtual_table_specs,
    cleared_param_runtime_intent,
    extract_structural_params,
    join_resolved_scope_tables,
    structural_s_key_assignment_order,
)
from ._intent_loop import (
    find_trusted_template_match,
    invoke_intent_parse_with_hints,
    list_union_match_candidates,
    pick_union_match_for_runtime_join,
    reconcile_template_store_until_stable,
    resolve_sql_path,
    structural_compare,
)
from ._intent_normalize import (
    append_table_scope_repairs,
    apply_diagnostic_repairs,
    drop_redundant_resolved_join_where_predicates,
)
from ._knowledge_join import merge_preserve_tables_with_notes_defaults
from ._llm_provider import LLMProvider
from ._schema_graph import (
    assert_consumer_intent_in_scope,
    assert_intent_in_scope,
    effective_execution_visible_tables,
)
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
from ._templates import TemplateRefs, TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    InteractiveChoicePort,
    RephraseHint,
    bind_params_for_sql,
    bound_engine_runtime_config,
    debug,
    emit_ask_phase,
    emit_session_refusal_diagnostic,
    interactive_yes_no,
    invalid_input,
    is_structural_param_key,
    notify,
    pipeline_trace,
    print_info,
    print_rephrase_hint,
    prompt,
    prompt_cache_schema_scope,
    prompt_json,
    reduce_structural_sql_placeholders,
    refusal_diagnostic_code_for_exception,
    refusal_diagnostic_code_for_federation_reason,
    refusal_message_for_exception,
    safe_json_loads,
    schema_prompt_cache_id,
    stable_json,
    terminated,
)
from ._utils_artifacts import emit_write_queue_event
from ._utils_intent import (
    exact_question_match,
    flatten_param_values,
    intent_key,
    resolve_caller_visible_tables,
    sql_shape,
)
from ._validation_rules import (
    validate_aggregate_join_fan_out,
    validate_clause_widened_rowset,
    validate_join_path_key_types,
)
from ._validation_shape import (
    validate_comparison_join_scope_or_raise,
    validate_intent_join_reachability,
    validate_join_path_reachability_for_tables,
)
from ._validation_sql import (
    canonicalize_rejection_reason,
    enforce_probe_cte_anchor_placement_post_resolution,
    temporary_dialect_member_limits,
    validate_sql,
)


def execution_scope_gate_active(
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


def space_scope_gate_active(
    space_tables: frozenset[str],
    space_columns: frozenset[str],
    space_deny_tables: frozenset[str] | None = None,
    space_deny_columns: frozenset[str] | None = None,
) -> bool:
    """Return True when an aetherspace allow/deny gate should run."""
    return bool(space_tables or space_columns or space_deny_tables or space_deny_columns)


def _post_resolution_scope_outcome(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    *,
    space_tables: frozenset[str],
    space_columns: frozenset[str],
    space_deny_tables: frozenset[str] | None = None,
    space_deny_columns: frozenset[str] | None = None,
    schema_context: EngineContext | None,
    visible_objects: frozenset[str] | None,
    schema_role: str,
    context_name: str,
    generation_path: GenerationPath,
    matched_template: Template | None,
    structural_tpl: tuple[Template, ...],
) -> SqlGenerationOutcome | None:
    """Re-run aetherspace and consumer scope gates after bridge tables are resolved."""
    deny_tables = frozenset(space_deny_tables or ())
    deny_columns = frozenset(space_deny_columns or ())
    if space_scope_gate_active(space_tables, space_columns, deny_tables, deny_columns):
        if not assert_intent_in_scope(
            intent, space_tables, space_columns, schema, deny_tables=deny_tables, deny_columns=deny_columns
        ):
            return SqlGenerationOutcome(
                "",
                False,
                generation_path,
                matched_template,
                structural_tpl,
                sql_validation_error="intent out of aetherspace scope",
                error_kind=FailureCategory.DENIED_REFERENCE.value,
            )
    scope_ctx = schema_context if schema_context is not None else EngineContext()
    if execution_scope_gate_active(scope_ctx, visible_objects, schema_role, context_name=context_name):
        if not assert_consumer_intent_in_scope(intent, scope_ctx, schema, visible_objects):
            return SqlGenerationOutcome(
                "",
                False,
                generation_path,
                matched_template,
                structural_tpl,
                sql_validation_error="intent out of execution scope",
                error_kind=FailureCategory.ACCESS_POLICY.value,
            )
    return None


def _raise_if_join_path_key_types(
    signature: list[str],
    schema: SchemaGraph,
    context: str,
) -> None:
    issues = validate_join_path_key_types(signature, schema, context)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise JoinPathKeyTypeError(context, errors[0].message)


def row_structural_values_match_defaults(
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


def _python_bind_kind(value: Any) -> str:
    """Classify a bound parameter value for fuzzy-reuse schema checks."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    return "unknown"


def _bind_kinds_compatible(exemplar_kind: str, candidate_kind: str) -> bool:
    if exemplar_kind == candidate_kind:
        return True
    if exemplar_kind in ("integer", "number") and candidate_kind in ("integer", "number"):
        return True
    return False


def _value_matches_slot_domain(value: Any, value_type: str) -> bool:
    vt = VALUE_TYPE_NORMALIZATION.get(
        (value_type or "string").strip().lower(), (value_type or "string").strip().lower()
    )
    if vt in ("date_window", "date_diff", "unknown", "null"):
        return True
    kind = _python_bind_kind(value)
    if vt == "boolean":
        return kind == "boolean"
    if vt == "integer":
        return kind == "integer"
    if vt == "number":
        return kind in ("integer", "number")
    if vt in ("string", "date", "binary"):
        return kind == "string"
    return True


def reuse_params_match_value_schema(
    params: Mapping[str, Any],
    schema_row: Mapping[str, Any],
    intent_sig: ConcreteIntent,
) -> bool:
    """Return True when extracted bind values match exemplar ``param_values`` types and slot domains."""
    if not schema_row:
        return True
    slot_meta = TemplateOps.collect_param_slot_meta(intent_sig)
    for key, exemplar in schema_row.items():
        if key not in params:
            continue
        candidate = params[key]
        ex_kind = _python_bind_kind(exemplar)
        cand_kind = _python_bind_kind(candidate)
        if not _bind_kinds_compatible(ex_kind, cand_kind):
            return False
        if ex_kind == "array" and cand_kind == "array":
            ex_items = list(exemplar) if isinstance(exemplar, (list, tuple)) else []
            if ex_items:
                item_kind = _python_bind_kind(ex_items[0])
                cand_items = list(candidate) if isinstance(candidate, (list, tuple)) else []
                if not all(_bind_kinds_compatible(item_kind, _python_bind_kind(item)) for item in cand_items):
                    return False
        meta = slot_meta.get(key)
        if meta is not None and not _value_matches_slot_domain(candidate, meta.value_type):
            return False
    return True


def extract_fuzzy_reuse_params(
    q_norm: str,
    template: Template,
    *,
    history_index: int,
    literal_structural_only: bool,
    schema: SchemaGraph | None = None,
    schema_context: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Extract p- and s-parameter values from a question for fuzzy template reuse via one LLM call."""
    p_key_names, s_key_names = TemplateOps.param_keys_from_intent_signature(
        template.intent_signature, literal_structural_only=literal_structural_only
    )
    all_keys = p_key_names + ([] if literal_structural_only else s_key_names)
    vh = template.value_history
    idx = max(0, min(history_index, len(vh.questions) - 1)) if vh.questions else 0
    prev_pv = vh.param_values[idx] if vh.param_values else {}
    if schema is not None and prev_pv:
        prev_pv = TemplateOps.redact_param_values_for_caller(
            template,
            prev_pv,
            schema=schema,
            schema_context=schema_context,
            visible_objects=visible_objects,
        )
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
            "param_slots": TemplateOps.param_slot_prompt_payload(template.intent_signature, all_keys),
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
            return LLMProvider.chat(system, user, task="default")

    raw = _reuse_llm_chat()
    parsed = safe_json_loads(raw)
    if not parsed or not isinstance(parsed, dict):
        raw2 = _reuse_llm_chat()
        parsed = safe_json_loads(raw2)
    if not isinstance(parsed, dict):
        raise ValueError(f"[{ASK_PHASE_A}] fuzzy reuse LLM JSON is not an object after retries")
    pv_raw = parsed.get("param_values")
    if not isinstance(pv_raw, dict):
        raise ValueError(
            f"[{ASK_PHASE_A}] fuzzy reuse LLM JSON missing dict 'param_values'; got {type(pv_raw).__name__}"
        )
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


def extract_reuse_params_literal_only(
    q_norm: str,
    template: Template,
    *,
    history_index: int,
    schema: SchemaGraph | None = None,
    schema_context: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> dict[str, Any]:
    """LLM fills ``p*`` bind handles only; structural ``s*`` values come from the exemplar row and defaults."""
    return extract_fuzzy_reuse_params(
        q_norm,
        template,
        history_index=history_index,
        literal_structural_only=True,
        schema=schema,
        schema_context=schema_context,
        visible_objects=visible_objects,
    )


def extract_reuse_params_full(
    q_norm: str,
    template: Template,
    *,
    history_index: int,
    schema: SchemaGraph | None = None,
    schema_context: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> dict[str, Any]:
    """LLM fills both ``p*`` and ``s*`` keys present in the intent signature."""
    return extract_fuzzy_reuse_params(
        q_norm,
        template,
        history_index=history_index,
        literal_structural_only=False,
        schema=schema,
        schema_context=schema_context,
        visible_objects=visible_objects,
    )


def load_pipeline_resources(
    schema: SchemaGraph | None = None,
    store: Any = None,
    templates: dict[str, Any] | None = None,
    rejected: dict[str, Any] | None = None,
    schema_terms: set[str] | None = None,
    dialect: Dialect | None = None,
) -> tuple[Dialect, SchemaGraph, Any, dict[str, Any], dict[str, Any], set[str]]:
    """Validate inputs, build dialect, and return pipeline resource bundle."""
    if not EngineConfig.llm_credentials_configured():
        raise RuntimeError("No OpenAI/Azure OpenAI API key configured")

    debug("loading schema")
    if dialect is None:
        runtime_cfg = bound_engine_runtime_config()
        dialect = DialectRegistry.get_dialect(EngineConfig.TYPE, runtime_cfg)
    if EngineConfig.TYPE not in DialectRegistry.list_engines():
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
    visible_tables: frozenset[str] | None = None,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> TemplateMatch:
    """Detect fuzzy question match against trusted template. ``value_history`` for direct SQL reuse."""
    debug(f"[{ASK_PHASE_A}] checking exact fuzzy match")
    caller_tables = resolve_caller_visible_tables(
        visible_tables=visible_tables,
        schema=schema,
        schema_context=schema_context,
        visible_objects=visible_objects,
    )
    templates_list = TemplateOps.caller_scoped_templates(templates, visible_tables=caller_tables)
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
        visible_tables=caller_tables,
    )

    if hit is not None:
        ref_tmpl = hit.template
        if schema is not None:
            live_ok, stale_reasons = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(ref_tmpl), schema)
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
    """Freeze the interactive bundle needed to resume after intent or hard-block prompts."""
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


def refinement_ctx_for_feedback(
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


def artifact_dir_for_template_store(store: Any) -> str:
    if isinstance(store, TemplateStoreView):
        return TemplateOps.artifacts_dir_for_template_store(store._store_dir)
    if isinstance(store, dict):
        store_dir = store.get("_store_dir") or store.get("store_dir")
        if store_dir:
            return TemplateOps.artifacts_dir_for_template_store(str(store_dir))
        if store.get("templates") or store.get("question_feedback"):
            return TemplateOps.artifacts_dir_for_template_store(EngineConfig.TEMPLATE_STORE_DIR)
    raise ValueError("template store requires _store_dir or a TemplateStoreView for artifact path resolution")


def emit_reader_write_queue_event(store: Any, event: WriteQueueEvent) -> None:
    """Reader sessions do not enqueue durable write-queue events; learning stays session-local."""
    return


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
    """Parse intent with :func:`aetherdialect._intent_loop._invoke_inten t_parse_with_hints` and optional user confirmation on schema-invalid paths. LLM schema payloads use space scope only. Credential RBAC is enforced at SQL generation/execution via ``execution_visible_objects``, not in prompts."""
    emit_ask_phase(ASK_PHASE_B)
    debug("intent via LLM")
    debug(f"[{ASK_PHASE_B}] calling invoke_intent_parse_with_hints")
    resolved_ctx = refinement_ctx_for_feedback(choice_port, refinement_ctx)
    conv_corr: tuple[str, ...] = ()
    if resolved_ctx is not None:
        raw_hints = getattr(resolved_ctx, "conversation_rejection_hints", None)
        if isinstance(raw_hints, tuple):
            conv_corr = raw_hints
    if resolved_ctx is not None and resolved_ctx.accumulated_reasons:
        seed_lines = list(resolved_ctx.accumulated_reasons)
    else:
        seed_lines = list(extra_user_feedback or [])
    intent_visible_objects = getattr(choice_port, "visible_objects", None) if choice_port is not None else None
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
            TemplateOps.save_template_store(store)
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
    return TemplateRefs.join_fingerprint_from_concrete_intent(
        matched.intent_signature
    ) == TemplateRefs.join_fingerprint_from_runtime_intent(intent)


def _resolve_joins_before_union_pin(
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    join_candidates: dict[str, Any],
    cmap: dict[str, list[str]],
    cte_join_hints: dict[str, dict[str, Any]] | None,
    structural_defaults_src: dict[str, Any] | None = None,
    store: dict[str, Any] | TemplateStoreView | None = None,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> None:
    """Populate runtime join fields from deterministic SQL before pinning a union template, including when multiple stored join fingerprints require an LLM join choice."""
    intent = prune_unused_cte_steps(intent)
    join_candidates, cmap, cte_join_hints = generate_join_candidates(intent, schema)

    prior_fb: list[str] | None = None
    avoid_join_ids: frozenset[str] = frozenset()
    if store is not None:
        visible_tables = effective_execution_visible_tables(schema, schema_context, visible_objects)
        prior_fb = TemplateOps.lookup_join_feedback_for_question(
            store,
            q_norm,
            schema_graph_id=schema.schema_graph_id,
            visible_tables=visible_tables,
            schema=schema,
        )
        avoid_join_ids = TemplateOps.lookup_join_avoid_candidate_ids_for_question(
            store,
            q_norm,
            schema_graph_id=schema.schema_graph_id,
            visible_tables=visible_tables,
            schema=schema,
        )

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
        avoided_candidate_ids=avoid_join_ids or None,
    )


def prepare_union_match_join_phase(
    q_norm: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    dialect: Any,
    templates: dict[str, Any],
    store: dict[str, Any] | TemplateStoreView | None = None,
    *,
    visible_tables: frozenset[str] | None = None,
    schema_context: EngineContext | None = None,
    visible_objects: frozenset[str] | None = None,
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
    """Resolve union-template reuse versus join LLM when body matches span multiple join fingerprints for interactive and live runners."""
    caller_tables = resolve_caller_visible_tables(
        visible_tables=visible_tables,
        schema=schema,
        schema_context=schema_context,
        visible_objects=visible_objects,
    )
    candidates = list_union_match_candidates(intent, templates, schema=schema, visible_tables=caller_tables)
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
    _resolve_joins_before_union_pin(
        q_norm,
        intent,
        schema,
        dialect,
        jc,
        cmap,
        hints,
        structural_defaults_src=None,
        store=store,
        schema_context=schema_context,
        visible_objects=visible_objects,
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


def template_effective_sql_display_param(tmpl: Template, dialect: Dialect) -> str:
    """Return user-facing display SQL for a template row, recomputing when storage omits it. Used by direct reuse and any reader after storage trim."""
    rt = tmpl.intent_signature.to_runtime_skeleton()
    return build_display_sql(tmpl.sql_param, rt, tmpl.display_alias_map or None, dialect=dialect)


def enriched_display_alias_map(
    q_norm: str, sql_param: str, disp: RuntimeIntent, base: dict[str, str] | None
) -> dict[str, str]:
    """Merge persisted ``display_alias_map`` with LLM-suggested headers for complex select columns. Simple columns use deterministic aliases only (no LLM)."""
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
    raw = LLMProvider.chat(system, user, task="default")
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"enriched_display_alias_map: LLM JSON is not an object; got {type(parsed).__name__}")
    if "aliases" not in parsed:
        raise ValueError("enriched_display_alias_map: LLM JSON missing 'aliases' key")
    block = parsed["aliases"]
    if not isinstance(block, dict):
        raise ValueError(f"enriched_display_alias_map: 'aliases' must be a dict; got {type(block).__name__}")
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


def run_sql_validation_cascade(
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
            emit_ask_phase(ASK_PHASE_K)
            pipeline_trace(
                "pipeline.run_sql_validation_cascade.reachability_failed",
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
    emit_ask_phase(ASK_PHASE_K)
    pipeline_trace(
        "pipeline.run_sql_validation_cascade.result",
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
    """Record accepted question form(s) on *tmpl* when no other template already claims the keys."""
    all_pv = flatten_param_values(intent)
    nl = intent.natural_language or ""
    pq = form_storage.corrected if form_storage is not None else q_norm
    if other_template_owns_question_string(templates, tmpl.id, pq):
        return
    nopt = form_storage.normalized_optional if form_storage is not None else None
    if nopt and nopt != pq and other_template_owns_question_string(templates, tmpl.id, nopt):
        return
    TemplateOps.record_value_history_on_accept(
        tmpl.value_history,
        param_values=all_pv,
        natural_language=nl,
        form_storage=form_storage,
        q_norm_fallback=q_norm,
        schema=schema,
        tables_hint=sorted(intent.tables or []),
    )


def best_accepted_template_similarity(
    intent: RuntimeIntent,
    templates: dict[str, Any],
    *,
    visible_tables: frozenset[str] | None = None,
) -> float:
    """Return the highest structural similarity between *intent* and any accepted template signature."""
    if not templates:
        return 0.0
    scores: list[float] = []
    for t in templates.values():
        if not TemplateOps.template_enumerable_by_caller(t, visible_tables=visible_tables):
            continue
        cr = structural_compare(intent, t, mode="full")
        s = cr.similarity_score
        scores.append(float(s) if s is not None else 0.0)
    return max(scores, default=0.0)


def clear_interpret_schema_invalid_after_user_accept(intent: RuntimeIntent) -> None:
    """Drop the ephemeral interpret schema-invalid hint after the user accepts intent confirmation."""
    if intent.schema_invalid:
        intent.schema_invalid = False
        debug("[pipeline] cleared interpret schema_invalid after user accepted intent confirmation")


def should_skip_intent_confirmation(
    intent: RuntimeIntent,
    store: dict[str, Any] | None,
    q_norm: str | None,
    semantic_warnings: list[Any] | None,
    *,
    schema_graph_id: str | None = None,
) -> bool:
    """Return True when intent confirmation may be skipped. Returns. False when the parsed intent is schema-invalid, when there are semantic warnings, or when the same canonicalised question has any prior rejection recorded in ``question_feedback``."""
    if getattr(intent, "schema_invalid", False):
        return False
    if semantic_warnings:
        return False
    if (
        store is not None
        and q_norm
        and TemplateOps.has_any_rejection_history_for_question(store, q_norm, schema_graph_id)
    ):
        return False
    return True


def should_prompt_direct_reuse_user(
    ref_tmpl: Template, _rejected: dict[str, Any], intent: RuntimeIntent, q_norm: str, *, reuse_history_index: int
) -> bool:
    """Return True when direct SQL reuse must ask the user instead of auto-accepting."""
    return not TemplateOps.should_auto_accept_for_question(ref_tmpl, q_norm, reuse_history_index=reuse_history_index)


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
        "Tip: a single sentence is enough — for example 'wrong table', 'missing date filter', or 'should aggregate by month'."
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
    entry = TemplateOps.summarize_failure_for_memory(
        question=q_norm,
        intent=intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=schema.effective_structural_hash,
        user_reason=feedback or default_user_reason,
    )
    if persist_template_learning:
        TemplateOps.record_question_feedback(store, q_norm, entry)
        TemplateOps.save_template_store(store)
    else:
        ev = WriteQueueEvent(
            kind="feedback_record",
            schema_graph_id=str(schema.schema_graph_id or ""),
            schema_hash=str(schema.effective_structural_hash or ""),
            produced_at=datetime.now(UTC).isoformat(),
            payload=(("q_norm", q_norm), ("entry_json", stable_json(entry.to_dict()))),
        )
        emit_reader_write_queue_event(store, ev)
    ctx_ref = refinement_ctx_for_feedback(choice_port, refinement_ctx)
    reason_line = (feedback or "").strip() or default_user_reason
    if (
        ctx_ref is not None
        and refinement_retry_available(ctx_ref)
        and not (choice_port is not None and isinstance(choice_port, PipelineSessionMarker))
    ):
        ctx_ref.accumulated_reasons.append(reason_line)
        ctx_ref.pending_retry = True
        raise RefinementRetry
    return entry.buckets[0].value if entry.buckets else RejectionBucket.OTHER.value


def compose_intent_confirm_session_message(
    intent: RuntimeIntent,
    semantic_warnings: list[Any] | None,
    *,
    approach: str | None = None,
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
    nl = (approach or "").strip() or (intent.natural_language or "").strip()
    if not nl:
        nl = f"Query {', '.join(intent.tables or [])} for data"
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
    """Prompt for intent confirmation or auto-proceed from similarity and warnings."""
    if intent_already_confirmed:
        return True
    if not force_intent_confirm and should_skip_intent_confirmation(
        intent,
        cast(dict[str, Any] | None, store),
        q_norm or "",
        semantic_warnings,
        schema_graph_id=(schema.schema_graph_id if schema is not None else None),
    ):
        debug(
            f"[{ASK_PHASE_H}] auto_proceed: similarity={similarity_score:.3f} "
            f"has_union={has_union_match} cols_changed={cols_changed}"
        )
        return True
    body_lines, _ = compose_intent_confirm_session_message(
        intent,
        semantic_warnings,
        approach=getattr(getattr(suspend_tail, "interpretation", None), "approach", None),
    )
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
                TemplateOps.save_template_store(store)
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
            TemplateOps.save_template_store(store)
        print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=rejection_bucket)
        return False
    debug(f"[{ASK_PHASE_H}] user_confirmed_intent")
    clear_interpret_schema_invalid_after_user_accept(intent)
    return True


def _choose_generation_path(
    *,
    has_matched_template: bool,
    resolved_union_path: GenerationPath | None,
    matched_template_id: str = "",
    structural_matches: int = 0,
    cols_changed: bool = False,
    retry_depth: int = 0,
) -> GenerationPath:
    """Return the SQL generation branch: a template-driven ``GenerationPath`` when a template is in play, else ``FRESH``."""
    del matched_template_id, structural_matches, cols_changed, retry_depth
    if not has_matched_template:
        return GenerationPath.FRESH
    return resolved_union_path or GenerationPath.FRESH


def align_template_to_widened_intent(template: Template, intent: RuntimeIntent, dialect: Any) -> None:
    """Copy widened SQL artifacts and identity fields from *intent* onto *template*. Used after ``UNION_TEMPLATE_WIDEN`` and ``UNION_TEMPLATE_AND_RUNTIME_WIDEN`` so execution SQL caches match widened projections."""
    all_pv = dict(flatten_param_values(intent))
    template.sql_param = intent.sql_param or template.sql_param
    sig_id = (
        template.intent_signature.intent_id
        if template.intent_signature and template.intent_signature.intent_id
        else "union"
    )
    template.intent_signature = intent.to_concrete(sig_id)
    template.intent_key = intent_key(intent)
    member_source_id = str(getattr(template, "member_source_id", "") or "") or None
    sg_dialect = TemplateStoreView.sqlglot_dialect_for_template_fingerprint(dialect, member_source_id)
    template.sql_fp = Dialect.compute_sql_fp(template.sql_param or "", sqlglot_dialect=sg_dialect)
    template.structural_defaults = {k: v for k, v in all_pv.items() if is_structural_param_key(k)}
    sig_aliases: dict[str, str] = {}
    for sc in intent.select_cols or []:
        alias = generate_col_alias(sc)
        if alias:
            sig_aliases[sc.signature_key] = alias
    template.display_alias_map = {**template.display_alias_map, **sig_aliases}


def sql_validation_refusal_outcome(
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


def federation_ineligible_refusal_outcome(
    reason: str,
    *,
    generation_path: GenerationPath,
    matched_template: Template | None,
    structural_match_templates: tuple[Template, ...] | list[Template] | None = (),
) -> SqlGenerationOutcome:
    """Return a failed reuse outcome for a federation ineligible reason."""
    refusal_code = refusal_diagnostic_code_for_federation_reason(reason)
    message = federation_user_facing_ineligible_message(reason)
    if refusal_code:
        emit_session_refusal_diagnostic(
            refusal_code,
            message,
            stage="validation",
            source_id="composite",
            details=(("phase", "prepare"), ("reason", str(reason or ""))),
        )
    structural_tpl = tuple(structural_match_templates or ())
    return SqlGenerationOutcome(
        "",
        False,
        generation_path,
        matched_template,
        structural_tpl,
        sql_validation_error=message,
        join_matches_template=None,
        error_kind=FailureCategory.DENIED_REFERENCE.value,
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
    TemplateOps.record_deterministic_join_failure_feedback(store, q_norm, exc, intent=intent, schema=schema)
    print_rephrase_hint(RephraseHint.JOIN_PATH_UNAVAILABLE)
    if persist_template_learning:
        TemplateOps.save_template_store(store)
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
    space_deny_tables: frozenset[str] | None = None,
    space_deny_columns: frozenset[str] | None = None,
    member_source_id: str | None = None,
    allowed_where_ops: frozenset[str] | None = None,
    join_preset_scope: dict[str, str] | None = None,
    max_query_cost_rows: float | None = None,
    max_query_cost_bytes: float | None = None,
    profile_timeout_ms: int | None = None,
) -> SqlGenerationOutcome:
    """Generate SQL from template reuse or deterministic build, then validate once."""
    emit_ask_phase(ASK_PHASE_J, source=member_source_id)
    intent = prune_unused_cte_steps(intent)
    structural_tpl = tuple(structural_match_templates or ())
    if getattr(intent, "schema_invalid", False):
        print_rephrase_hint(RephraseHint.SCHEMA_INVALID_DECLINED)
        if persist_template_learning:
            TemplateOps.save_template_store(store)
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
    deny_tables = frozenset(space_deny_tables or ())
    deny_columns = frozenset(space_deny_columns or ())
    if space_scope_gate_active(space_tables, space_columns, deny_tables, deny_columns):
        if not assert_intent_in_scope(
            intent, space_tables, space_columns, schema, deny_tables=deny_tables, deny_columns=deny_columns
        ):
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
    if execution_scope_gate_active(scope_ctx, visible_objects, schema_role, context_name=context_name):
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
        for fp in PredicateGroup.where_leaves(intent.where) or []:
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
        for hp in PredicateGroup.having_leaves(intent.having) or []:
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
    emit_ask_phase(ASK_PHASE_K, source=member_source_id)
    debug(f"[{ASK_PHASE_K}] tables={intent.tables or []}")
    debug(f"[{ASK_PHASE_K}] grain={intent.grain or 'unknown'}")
    debug(f"[{ASK_PHASE_K}] select_cols={[s.expr.primary_term for s in (intent.select_cols or [])]}")
    debug(f"[{ASK_PHASE_K}] where={len(PredicateGroup.where_leaves(intent.where) or [])}")
    debug(f"[{ASK_PHASE_K}] having={len(PredicateGroup.having_leaves(intent.having) or [])}")
    debug(f"[{ASK_PHASE_K}] cte_join_hints={list(cte_join_hints.keys()) if cte_join_hints else None}")
    resolved_union_path = resolve_sql_path(
        matched_template=matched_template, cols_changed=cols_changed, union_sql_path=union_sql_path
    )
    routing = _choose_generation_path(
        has_matched_template=matched_template is not None,
        resolved_union_path=resolved_union_path,
        matched_template_id=(matched_template.id if matched_template else ""),
        structural_matches=len(structural_tpl),
        cols_changed=cols_changed,
        retry_depth=0,
    )
    active_path = routing
    debug(f"[{ASK_PHASE_K}] generation path={active_path}")

    structural_defaults_src: dict[str, Any] | None = None
    if matched_template:
        tmpl_sd = getattr(matched_template, "structural_defaults", None)
        structural_defaults_src = tmpl_sd if tmpl_sd else None

    params = dict(flatten_param_values(intent))
    debug(f"[{ASK_PHASE_K}] params={params}")

    prior_join_fb = TemplateOps.lookup_join_feedback_for_question(
        cast(dict[str, Any], store),
        q_norm,
        schema_graph_id=schema.schema_graph_id,
        member_source_id=member_source_id,
        visible_tables=effective_execution_visible_tables(schema, schema_context, visible_objects),
        schema=schema,
    )
    avoid_join_ids = TemplateOps.lookup_join_avoid_candidate_ids_for_question(
        cast(dict[str, Any], store),
        q_norm,
        schema_graph_id=schema.schema_graph_id,
        member_source_id=member_source_id,
        visible_tables=effective_execution_visible_tables(schema, schema_context, visible_objects),
        schema=schema,
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
        live_ok, stale_reasons = TemplateRefs.template_is_live(
            TemplateRefs.template_schema_refs(matched_template), schema
        )
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
                avoided_candidate_ids=avoid_join_ids or None,
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
            JoinPathKeyTypeError,
            ProbeCtePlacementError,
        ) as exc:
            debug(f"[{ASK_PHASE_K}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                TemplateOps.save_template_store(store)
            return sql_validation_refusal_outcome(
                exc,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
            )
        scope_outcome = _post_resolution_scope_outcome(
            intent,
            schema,
            space_tables=space_tables,
            space_columns=space_columns,
            schema_context=schema_context,
            visible_objects=visible_objects,
            schema_role=schema_role,
            context_name=context_name,
            generation_path=generation_path_label,
            matched_template=matched_for_outcome,
            structural_tpl=structural_tpl,
        )
        if scope_outcome is not None:
            return scope_outcome
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
        ok_c, err_c, cat_c, diags_c = run_sql_validation_cascade(
            sql,
            intent,
            dialect,
            schema=schema,
            max_query_cost_rows=max_query_cost_rows,
            max_query_cost_bytes=max_query_cost_bytes,
            profile_timeout_ms=profile_timeout_ms,
        )
        if ok_c:
            soft_findings = tuple(d for d in diags_c if d.code.value in SOFT_DIAGNOSTIC_CODES)
            return SqlGenerationOutcome(
                sql,
                True,
                resolved_union_path,
                matched_template,
                structural_tpl,
                None,
                jm3,
                None,
                None,
                explain_soft_diagnostics=len(soft_findings),
                explain_soft_findings=soft_findings,
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
                avoided_candidate_ids=avoid_join_ids or None,
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
            JoinPathKeyTypeError,
            ProbeCtePlacementError,
        ) as exc:
            debug(f"[{ASK_PHASE_K}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                TemplateOps.save_template_store(store)
            return sql_validation_refusal_outcome(
                exc,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
            )
        scope_outcome = _post_resolution_scope_outcome(
            gen_intent,
            schema,
            space_tables=space_tables,
            space_columns=space_columns,
            schema_context=schema_context,
            visible_objects=visible_objects,
            schema_role=schema_role,
            context_name=context_name,
            generation_path=generation_path_label,
            matched_template=matched_for_outcome,
            structural_tpl=structural_tpl,
        )
        if scope_outcome is not None:
            return scope_outcome
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
                avoided_candidate_ids=avoid_join_ids or None,
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
            JoinPathKeyTypeError,
            ProbeCtePlacementError,
        ) as exc:
            debug(f"[{ASK_PHASE_K}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                TemplateOps.save_template_store(store)
            return sql_validation_refusal_outcome(
                exc,
                generation_path=generation_path_label,
                matched_template=matched_for_outcome,
                structural_match_templates=structural_tpl,
            )
        scope_outcome = _post_resolution_scope_outcome(
            intent,
            schema,
            space_tables=space_tables,
            space_columns=space_columns,
            schema_context=schema_context,
            visible_objects=visible_objects,
            schema_role=schema_role,
            context_name=context_name,
            generation_path=generation_path_label,
            matched_template=matched_for_outcome,
            structural_tpl=structural_tpl,
        )
        if scope_outcome is not None:
            return scope_outcome
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

    ok, err, vcat, vdiags = run_sql_validation_cascade(
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
                    avoided_candidate_ids=avoid_join_ids or None,
                    join_preset_scope=join_preset_scope,
                )
                repaired_intent.sql_param = sql_param_r
                sql_r = finalize_substitute_sql(
                    repaired_intent, structural_defaults_src=structural_defaults_src, params=dict(params)
                )
                ok_r, err_r, vcat_r, vdiags_r = run_sql_validation_cascade(
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
            TemplateOps.save_template_store(store)
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

    soft_findings = tuple(d for d in vdiags if d.code.value in SOFT_DIAGNOSTIC_CODES)
    return SqlGenerationOutcome(
        sql,
        True,
        generation_path_label,
        matched_for_outcome,
        structural_tpl,
        None,
        join_matches_for_outcome,
        None,
        None,
        explain_soft_diagnostics=len(soft_findings),
        explain_soft_findings=soft_findings,
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
    avoided_candidate_ids: frozenset[str] | None = None,
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
            avoided_candidate_ids=avoided_candidate_ids,
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
            avoided_candidate_ids=avoided_candidate_ids,
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
            _raise_if_join_path_key_types(main_sig, schema, "main query")
            _raise_if_aggregate_join_fan_out(intent, main_sig, "main query", main_anchor)
            _raise_if_clause_widened_rowset(intent, main_sig, "main query", main_anchor)
        for cte_step in intent.cte_steps or []:
            cte_tbls = list(cte_step.tables or [])
            if len(cte_tbls) < 2 or cte_step.cte_name not in cte_join_ids:
                continue
            hints_c = (cte_join_hints or {}).get(cte_step.cte_name) or {}
            cte_sig = _signature_for_candidate(hints_c, cte_join_ids[cte_step.cte_name])
            cte_anchor = cte_tbls[0] if cte_tbls else None
            _raise_if_join_path_key_types(cte_sig, schema, f"CTE '{cte_step.cte_name}'")
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
                where_params=PredicateGroup.where_leaves(intent.where),
                having_params=PredicateGroup.having_leaves(intent.having),
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
                where_params=PredicateGroup.where_leaves(cte_step.where),
                having_params=PredicateGroup.having_leaves(cte_step.having),
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

    main_preserve_tables = merge_preserve_tables_with_notes_defaults(
        intent.preserve_tables,
        schema,
        query_tables=intent.tables or [],
    )
    main_sig = join_sigs_ordered[-1] if join_sigs_ordered else []
    sql_param = inject_join_into_deterministic_sql(
        deterministic_sql,
        join_sigs_ordered,
        edge_kinds_ordered=edge_kinds_ordered,
        schema=schema,
        dialect=dialect,
        cte_emissions=cte_emission_map(intent.cte_steps),
        preserve_tables=main_preserve_tables,
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
        preserve_tables=main_preserve_tables,
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
    return Dialect.finalize_executable_sql(
        sql_param, params, structural_defaults_src, sqlglot_dialect=Dialect.active_sqlglot_dialect()
    )


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
        actual_shape = sql_shape(sql, intent, sqlglot_dialect=Dialect.active_sqlglot_dialect())
    intent.sql_shape = actual_shape
    debug(f"sql_shape={actual_shape}")


def emit_explain_soft_diagnostics(findings: Sequence[Any]) -> None:
    """Surface EXPLAIN soft-diagnostic findings as structured diagnostics instead of a confidence penalty."""
    if not findings:
        return
    for finding in findings:
        raw_code = finding.code if isinstance(finding.code, str) else finding.code.value
        if raw_code not in SOFT_DIAGNOSTIC_CODES:
            continue
        message = getattr(finding, "message", None) or f"EXPLAIN soft diagnostic: {raw_code}"
        notify(
            message,
            stage="validation",
            code=raw_code,
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
        TemplateOps.record_template_feedback(tmpl, accept=True)
        member_q = member_feedback_q_norm(step.source_id, q_norm)
        TemplateOps.record_per_question_feedback(tmpl, member_q, accept=True, path=TemplateOps.path_bucket(path))
        TemplateOps.promote_trust(tmpl, member_q)
        stamp_federation_member_template(tmpl, plan_id=plan_id, source_id=step.source_id)
        member_schema = (schemas_by_source or {}).get(step.source_id)
        if member_schema is not None and not other_template_owns_question_string(member_templates, tmpl.id, member_q):
            _maybe_record_value_history_accept(
                member_templates, tmpl, step.sub_intent, member_q, form_storage, member_schema
            )
        TemplateOps.save_template_store(member_store)
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


def _persist_federation_join_feedback(
    federation_dir: str,
    federation_plan_id: str,
    summary: str,
    *,
    q_norm: str,
) -> None:
    record_federation_join_feedback(federation_dir, federation_plan_id, summary, q_norm=q_norm)
    mirror_federation_plan_join_feedback(federation_dir, federation_plan_id, summary)


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
        TemplateOps.delete_pending_templates_for_question(store, templates, q_norm)
        if feedback_template is not None:
            TemplateOps.record_template_feedback(feedback_template, accept=False)
            try:
                resolved_path_for_reject = GenerationPath.parse(ctx.generation_path)
            except (KeyError, ValueError):
                resolved_path_for_reject = None
            path_bucket_value = TemplateOps.path_bucket(resolved_path_for_reject)
            TemplateOps.record_per_question_feedback(feedback_template, q_norm, accept=False, path=path_bucket_value)
            _, template_deleted = TemplateOps.reject_out_per_question(templates, feedback_template, q_norm)
            entry_fb = TemplateOps.summarize_failure_for_memory(
                question=q_norm,
                intent=intent,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash=schema.effective_structural_hash,
                user_reason=norm_reason,
            )
            if federation_plan_id and federation_dir:
                _persist_federation_join_feedback(
                    str(federation_dir), str(federation_plan_id), entry_fb.summary, q_norm=q_norm
                )
                join_feedback_recorded = True
            elif (
                cross_source_join_feedback
                and federation_dir
                and federation_plan_id
                and RejectionBucket.WRONG_TABLES_OR_JOINS in entry_fb.buckets
            ):
                _persist_federation_join_feedback(
                    str(federation_dir), str(federation_plan_id), entry_fb.summary, q_norm=q_norm
                )
                join_feedback_recorded = True
            else:
                TemplateOps.record_question_feedback(store, q_norm, entry_fb)
            last_bucket = entry_fb.buckets[0].value if entry_fb.buckets else RejectionBucket.OTHER.value
            if template_deleted:
                TemplateOps.templates_to_store(store, templates)

        else:
            entry_fb = TemplateOps.summarize_failure_for_memory(
                question=q_norm,
                intent=intent,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash=schema.effective_structural_hash,
                user_reason=norm_reason,
            )
            if federation_plan_id and federation_dir:
                _persist_federation_join_feedback(
                    str(federation_dir), str(federation_plan_id), entry_fb.summary, q_norm=q_norm
                )
                join_feedback_recorded = True
            elif (
                cross_source_join_feedback
                and federation_dir
                and federation_plan_id
                and RejectionBucket.WRONG_TABLES_OR_JOINS in entry_fb.buckets
            ):
                _persist_federation_join_feedback(
                    str(federation_dir), str(federation_plan_id), entry_fb.summary, q_norm=q_norm
                )
                join_feedback_recorded = True
            else:
                TemplateOps.record_question_feedback(store, q_norm, entry_fb)
            last_bucket = entry_fb.buckets[0].value if entry_fb.buckets else RejectionBucket.OTHER.value

        store = TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)
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
            produced_at=datetime.now(UTC).isoformat(),
            payload=(("ctx_json", stable_json(ctx_doc)),),
        )
        try:
            emit_reader_write_queue_event(store, ev)
        except ValueError:
            if federation_dir:
                event_space = MASTER_AETHERSPACE_NAME
                if choice_port is not None:
                    sn = getattr(choice_port, "space_name", None)
                    if sn is not None and str(sn).strip():
                        event_space = str(sn).strip().lower()
                emit_write_queue_event(federation_dir, ev, space_name=event_space)
            else:
                raise
        last_bucket = RejectionBucket.OTHER.value
    ctx_ref = refinement_ctx_for_feedback(choice_port, refinement_ctx)
    reason_line = ((reject_reason or "").strip() if needs_reason else "") or norm_reason
    if (
        ctx_ref is not None
        and refinement_retry_available(ctx_ref)
        and not (choice_port is not None and isinstance(choice_port, PipelineSessionMarker))
    ):
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
            TemplateOps.save_template_store(store)
        return None

    emit_ask_phase(ASK_PHASE_N, source=member_source_id)

    record_q = member_feedback_q_norm(member_source_id, q_norm) if member_source_id else q_norm

    resolved_path = GenerationPath.parse(generation_path)
    if resolved_path is GenerationPath.FEDERATION_PLAN and federated_plan is not None:
        stamp_sql_shape(sql, intent, generation_path=resolved_path, federated_plan=federated_plan)
    else:
        intent.sql_shape = sql_shape(sql, intent, sqlglot_dialect=Dialect.active_sqlglot_dialect())
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
            TemplateOps.delete_rejected_templates_matching_question(cast(dict[str, Any], store), record_q)

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
                composite = matched_template
                if composite is None or not TemplateOps.template_is_pending(composite):
                    composite = TemplateOps.find_pending_template_for_question(templates, record_q)
                if composite is not None and TemplateOps.template_is_pending(composite):
                    TemplateOps.approve_pending_template(
                        store,
                        templates,
                        composite,
                        intent=intent,
                        q_norm=record_q,
                        form_storage=form_storage,
                        schema=schema,
                    )
                promoted = True

            if matched_rejected_template is not None:
                new_tmpl = TemplateOps.promote_rejected_to_template(
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
                pending = TemplateOps.find_pending_template_for_question(templates, record_q)
                if pending is not None:
                    TemplateOps.approve_pending_template(
                        store,
                        templates,
                        pending,
                        intent=intent,
                        q_norm=record_q,
                        form_storage=form_storage,
                        schema=schema,
                    )
                else:
                    sm_list = list(structural_match_templates or [])
                    TemplateOps.insert_template(
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
                if TemplateOps.template_is_pending(tmpl):
                    TemplateOps.approve_pending_template(
                        store,
                        templates,
                        tmpl,
                        intent=intent,
                        q_norm=record_q,
                        form_storage=form_storage,
                        schema=schema,
                    )
                else:
                    TemplateOps.record_template_feedback(tmpl, accept=True)
                    TemplateOps.record_per_question_feedback(
                        tmpl, record_q, accept=True, path=TemplateOps.path_bucket(resolved_path)
                    )
                    TemplateOps.promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif (
                not promoted
                and matched_template is not None
                and resolved_path
                in (GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL, GenerationPath.FUZZY_REUSE_FULL_PARAMS)
            ):
                tmpl = matched_template
                TemplateOps.record_template_feedback(tmpl, accept=True)
                TemplateOps.record_per_question_feedback(
                    tmpl, record_q, accept=True, path=TemplateOps.path_bucket(resolved_path)
                )
                TemplateOps.promote_trust(tmpl, record_q)
                if not other_template_owns_question_string(templates, tmpl.id, record_q):
                    _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif not promoted and matched_template is not None and resolved_path == GenerationPath.INTENT_DIRECT_MATCH:
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    TemplateOps.insert_template(
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
                    TemplateOps.record_template_feedback(tmpl, accept=True)
                    TemplateOps.record_per_question_feedback(
                        tmpl, record_q, accept=True, path=TemplateOps.path_bucket(resolved_path)
                    )
                    TemplateOps.promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif (
                not promoted
                and matched_template is not None
                and resolved_path == GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE
            ):
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    TemplateOps.insert_template(
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
                    TemplateOps.record_template_feedback(tmpl, accept=True)
                    TemplateOps.record_per_question_feedback(
                        tmpl, record_q, accept=True, path=TemplateOps.path_bucket(resolved_path)
                    )
                    TemplateOps.promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
            elif not promoted and matched_template is not None and resolved_path == GenerationPath.UNION_TEMPLATE_WIDEN:
                if join_matches_template is False:
                    sm_list = list(structural_match_templates or [])
                    TemplateOps.insert_template(
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
                    TemplateOps.record_template_feedback(tmpl, accept=True)
                    TemplateOps.record_per_question_feedback(
                        tmpl, record_q, accept=True, path=TemplateOps.path_bucket(resolved_path)
                    )
                    TemplateOps.promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
                    old_skeleton = tmpl.intent_signature.to_runtime_skeleton()
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
                    TemplateOps.insert_template(
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
                    TemplateOps.record_template_feedback(tmpl, accept=True)
                    TemplateOps.record_per_question_feedback(
                        tmpl, record_q, accept=True, path=TemplateOps.path_bucket(resolved_path)
                    )
                    TemplateOps.promote_trust(tmpl, record_q)
                    if not other_template_owns_question_string(templates, tmpl.id, record_q):
                        _maybe_record_value_history_accept(templates, tmpl, intent, record_q, form_storage, schema)
                    old_skeleton = tmpl.intent_signature.to_runtime_skeleton()
                    new_skeleton = cleared_param_runtime_intent(intent)
                    key_remap = _structural_key_remap_from_assignment_order(old_skeleton, new_skeleton)
                    _remap_value_history_structural_keys(tmpl.value_history, key_remap)
                    align_template_to_widened_intent(tmpl, intent, dialect)
                    reconcile_template_store_until_stable(
                        templates, template_store_view=(store if isinstance(store, TemplateStoreView) else None)
                    )

            store = TemplateOps.templates_to_store(store, templates)
            TemplateOps.save_template_store(store)
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
                produced_at=datetime.now(UTC).isoformat(),
                payload=(("replay_json", stable_json(replay)),),
            )
            emit_reader_write_queue_event(store, ev)

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
                        "Tip: a single sentence is enough — for example 'wrong table', 'missing date filter', or 'should aggregate by month'."
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
                    TemplateOps.save_template_store(store)
                return None
            if not reject_reason:
                invalid_input()
                if persist_template_learning:
                    TemplateOps.save_template_store(store)
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


def most_frequent_natural_language(vh: ValueHistory) -> str:
    """Get the most frequently occurring natural_language from a. ValueHistory."""
    if not vh.natural_language:
        return ""
    non_empty = [nl for nl in vh.natural_language if nl]
    if not non_empty:
        return ""
    counts = Counter(non_empty)
    return counts.most_common(1)[0][0]


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
    """Fill missing structural keys referenced as ``:sN`` in param SQL from template defaults. Used by direct SQL reuse and ``INTENT_DIRECT_MATCH``."""
    sd = structural_defaults or {}
    s_keys = set(re.findall(r":(s\d+)", sql_param))
    added = 0
    for sk in s_keys:
        if sk not in new_params and sk in sd:
            new_params[sk] = sd[sk]
            added += 1
    return added


def federation_contract_kwargs_for_reuse(
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
