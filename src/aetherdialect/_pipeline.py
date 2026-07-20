"""Interactive text-to-SQL orchestration: intent parsing, joins, SQL generation, validation, and template/negative memory."""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, cast

import pandas

from ._config import (
    ConfigError,
    EngineConfig,
    PolicyConfig,
    llm_credentials_configured,
)
from ._constants import (
    ASK_PHASE_A,
    ASK_PHASE_B,
    ASK_PHASE_H,
    ASK_PHASE_I,
    ASK_PHASE_J,
    ASK_PHASE_L,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_REUSE_HIT,
    DIAGNOSTIC_CODE_REUSE_MISS,
    INTERACTIVE_STAGE_DIRECT_REUSE,
    INTERACTIVE_STAGE_INTENT_CONFIRM,
    JOIN_CHOICE_SCOPE_MAIN,
    PIPELINE_BUG_SQL_VALIDATION,
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
    GenerationPath,
)
from ._contracts_base import (
    AccessError,
    EngineContext,
    FailureCategory,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    NoJoinPathError,
    PipelineSuspended,
    SqlDiagnostic,
    WriteQueueEvent,
)
from ._contracts_core import (
    DirectReuseSuspendContext,
    FeedbackKind,
    InteractiveTailSnapshot,
    InterpretPlan,
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
    concrete_cte_to_runtime,
    concrete_intent_to_runtime_skeleton,
    runtime_intent_to_concrete,
)
from ._contracts_schema import (
    SchemaGraph,
    SQLShape,
)
from ._core_utils import (
    InteractiveChoicePort,
    RephraseHint,
    bind_params_for_sql,
    debug,
    emit_write_queue_event,
    interactive_yes_no,
    invalid_input,
    is_structural_param_key,
    normalize_question,
    note_interactive_turn,
    notify,
    pipeline_trace,
    print_info,
    print_query_result,
    print_rephrase_hint,
    progress,
    prompt,
    reconcile_execute_bind_params,
    reduce_structural_sql_placeholders,
    safe_json_loads,
    stable_json,
    terminated,
)
from ._dialect import (
    Dialect,
    active_sqlglot_dialect,
    compute_sql_fp,
    finalize_executable_sql,
    get_dialect,
    list_engines,
    sql_outer_select_aliases,
)
from ._intent_expr import (
    build_virtual_table_specs,
    cleared_param_runtime_intent,
    extract_structural_params,
    structural_s_key_assignment_order,
)
from ._intent_process import (
    find_trusted_template_match,
    invoke_intent_parse_with_hints,
    list_union_match_candidates,
    pick_union_match_for_runtime_join,
    reconcile_template_store_until_stable,
    resolve_sql_path,
    structural_compare,
)
from ._intent_repair import apply_diagnostic_repairs, collect_referenced_tables, expand_shared_pk_tables_for_refs
from ._intent_resolve import join_path_key_concrete, prune_unused_cte_steps
from ._llm_provider import llm_chat
from ._schema_graph import assert_consumer_intent_in_scope, assert_intent_in_scope
from ._sql_gen import (
    ScopeClass,
    build_deterministic_sql,
    build_display_sql,
    cte_to_intent_for_ranking,
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
    render_feedback_sql,
    render_select_col_sql,
    select_col_prefers_llm_display_alias,
    tables_in_join_scope,
)
from ._templates import (
    TemplateStoreView,
    artifacts_dir_for_template_store,
    compute_question_feedback_penalty,
    delete_rejected_templates_matching_question,
    handles_referenced_in_sql_param,
    has_any_rejection_history_for_question,
    insert_template,
    join_fingerprint_from_concrete_intent,
    join_fingerprint_from_runtime_intent,
    lookup_join_feedback_for_question,
    path_bucket,
    promote_rejected_to_template,
    promote_trust,
    record_per_question_feedback,
    record_question_feedback,
    record_template_feedback,
    record_value_history_on_accept,
    reject_out_per_question,
    resolve_template_for_question,
    save_template_store,
    should_auto_accept_for_question,
    summarize_failure_for_memory,
    templates_to_store,
)
from ._utils import (
    exact_question_match,
    extract_tables_from_sql,
    flatten_param_values,
    intent_key,
    sql_shape,
)
from ._validation_execute import (
    canonicalize_rejection_reason,
    compute_confidence,
    validate_sql,
)


def _execution_scope_gate_active(
    schema_context: EngineContext | None,
    execution_visible_objects: frozenset[str] | None,
    schema_role: str,
) -> bool:
    """Return True when the execution-time context/RBAC gate should run."""
    if schema_role == "consumer":
        return True
    ctx = schema_context if schema_context is not None else EngineContext()
    if getattr(ctx, "name", "master") != "master":
        return True
    if ctx.allow_objects or ctx.deny_objects or ctx.deny_columns or ctx.allow_columns:
        return True
    return execution_visible_objects is not None


def _row_structural_values_match_defaults(
    row_params: dict[str, Any],
    structural_defaults: dict[str, Any] | None,
    sql_param: str,
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
) -> dict[str, Any]:
    """Extract p- and s-parameter values from a question for fuzzy. template reuse via one LLM call."""
    sql_param = template.sql_param or ""
    p_keys = sorted(set(re.findall(r":p\d+", sql_param)))
    s_keys = sorted(set(re.findall(r":s\d+", sql_param)))
    p_key_names = [k.lstrip(":") for k in p_keys]
    s_key_names = [k.lstrip(":") for k in s_keys]
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
    user = stable_json(
        {
            "task": "The parameterized SQL below uses placeholders for literal values. Extract the correct value from the question for every param_key listed.",
            "parameterized_sql": sql_param,
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
        }
    )
    raw = llm_chat(system, user, task="default")
    parsed = safe_json_loads(raw)
    if not parsed or not isinstance(parsed, dict):
        raw2 = llm_chat(system, user, task="default")
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
    q_norm: str,
    template: Template,
    *,
    history_index: int,
) -> dict[str, Any]:
    """Paths ``2.1``: LLM fills ``p*`` only; structural ``s*`` come from the exemplar row and defaults."""
    return extract_fuzzy_reuse_params(q_norm, template, history_index=history_index, literal_structural_only=True)


def _extract_reuse_params_full(
    q_norm: str,
    template: Template,
    *,
    history_index: int,
) -> dict[str, Any]:
    """Paths ``2.2``: LLM fills both ``p*`` and ``s*`` keys present in template SQL."""
    return extract_fuzzy_reuse_params(q_norm, template, history_index=history_index, literal_structural_only=False)


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
        notify(
            "No trusted template matched for direct SQL reuse.",
            stage="pipeline",
            code=DIAGNOSTIC_CODE_REUSE_MISS,
        )
    return TemplateMatch(
        intent=None,
        best_template=None,
        similarity_score=0.0,
        reuse_type="none",
        reuse_candidate_normalized=None,
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
    choice_port: InteractiveChoicePort | None,
    refinement_ctx: RefinementContext | None,
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
    intent, semantic_warnings, llm_calls, interpret_plan = invoke_intent_parse_with_hints(
        q_norm,
        schema,
        store=cast(dict[str, Any] | None, store),
        extra_user_feedback=seed_lines if seed_lines else None,
        prior_user_corrections=conv_corr,
        persist_template_learning=persist_template_learning,
        visible_objects=getattr(choice_port, "visible_objects", None) if choice_port is not None else None,
        allowed_columns=getattr(choice_port, "space_columns", None) if choice_port is not None else None,
        deny_objects=getattr(choice_port, "space_deny_objects", None) if choice_port is not None else None,
        deny_columns=getattr(choice_port, "space_deny_columns", None) if choice_port is not None else None,
        description_overlay=getattr(choice_port, "space_description_overlay", None)
        if choice_port is not None
        else None,
    )
    if intent is not None:
        pipeline_trace(
            "pipeline.parse_intent_via_llm.intent_complete",
            lambda: stable_json(intent.to_dict()),
        )

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
    debug(f"[{ASK_PHASE_I}] generating join candidates")
    virtual_specs = build_virtual_table_specs(intent, schema)

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
    )
    if len(scope_main) >= 2 and _scope_only_j00(join_candidates):
        join_candidates = join_hints_multi(
            schema,
            scope_main,
            intent,
            virtual_specs=virtual_specs,
            include_semantic=True,
        )
    cmap = join_candidate_map(join_candidates)
    debug(f"[{ASK_PHASE_I}] {len(join_candidates.get('candidates', []))} join candidates")
    for c in join_candidates.get("candidates", []):
        debug(f"[{ASK_PHASE_I}] {c.get('candidate_id')}: {c.get('join_path_signature', [])}")

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
            )
            if len(scope_cte) >= 2 and _scope_only_j00(cte_hints):
                cte_hints = join_hints_multi(
                    schema,
                    scope_cte,
                    cte_intent,
                    virtual_specs=virtual_specs,
                    include_semantic=True,
                )
            cte_join_hints[cte_name] = cte_hints
            debug(
                f"[{ASK_PHASE_I}] CTE '{cte_name}': {len(cte_hints.get('candidates', []))} candidates (CTE-specific ranking)"
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
            debug(f"[{ASK_PHASE_I}] CTE '{cte_name}': single table, J00 assigned")

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


def _sql_phase_join_resources(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    matched_template: Template | None,
    union_sql_path: GenerationPath | None,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Produce join candidate payloads for the SQL phase (always fully enumerated)."""
    jc, cmap, hints = generate_join_candidates(intent, schema)
    if matched_template is not None and union_sql_path == GenerationPath.INTENT_DIRECT_MATCH:
        conc = matched_template.intent_signature
        cid = str(getattr(conc, "chosen_join_candidate_id", "") or "").strip()
        if cid and cid != "J00":
            intent.chosen_join_candidate_id = cid
            intent.chosen_join_path_signature = list(conc.chosen_join_path_signature or [])
            conc_ctes = conc.cte_steps or []
            for idx, step in enumerate(intent.cte_steps or []):
                if idx >= len(conc_ctes):
                    break
                cs = conc_ctes[idx]
                ccid = str(getattr(cs, "chosen_join_candidate_id", "") or "").strip()
                if ccid and ccid != "J00":
                    step.chosen_join_candidate_id = ccid
                    step.chosen_join_path_signature = list(getattr(cs, "chosen_join_path_signature", []) or [])
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
    candidates = list_union_match_candidates(intent, templates)
    if not candidates:
        jc, cmap, hints = generate_join_candidates(intent, schema)
        return None, None, False, None, False, jc, cmap, hints

    candidate_join_keys = {join_path_key_concrete(c.template.intent_signature) for c in candidates}
    if len(candidate_join_keys) == 1 and len(candidates) == 1:
        chosen = candidates[0]
        jc, cmap, hints = _sql_phase_join_resources(intent, schema, chosen.template, chosen.union_sql_path)
        return (
            chosen.template,
            chosen.union_cols,
            chosen.cols_changed,
            chosen.union_sql_path,
            True,
            jc,
            cmap,
            hints,
        )
    if len(candidate_join_keys) == 1:
        chosen = min(
            candidates,
            key=lambda c: (c.non_agg_symmetric_diff, len(c.union_cols), c.template.id),
        )
        jc, cmap, hints = _sql_phase_join_resources(intent, schema, chosen.template, chosen.union_sql_path)
        return (
            chosen.template,
            chosen.union_cols,
            chosen.cols_changed,
            chosen.union_sql_path,
            True,
            jc,
            cmap,
            hints,
        )
    jc, cmap, hints = generate_join_candidates(intent, schema)
    _resolve_joins_for_intent_placeholder(
        q_norm,
        intent,
        schema,
        dialect,
        jc,
        cmap,
        hints,
        structural_defaults_src=None,
        store=store,
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
    return build_display_sql(
        tmpl.sql_param,
        rt,
        tmpl.display_alias_map or None,
        dialect=dialect,
    )


def enriched_display_alias_map(
    q_norm: str,
    sql_param: str,
    disp: RuntimeIntent,
    base: dict[str, str] | None,
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
    user = stable_json(
        {
            "task": "Short result-grid column headers for SQL SELECT expressions.",
            "question": q_norm,
            "columns": [{"signature_key": k, "sql_expr": e} for k, e in targets],
        }
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


def _validation_sql_for_explain(
    sql: str,
    intent: RuntimeIntent,
    dialect: Any,
) -> str:
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
) -> tuple[bool, str, FailureCategory | None, list[SqlDiagnostic]]:
    """Run `validate_sql` (AST plus optional EXPLAIN)."""
    validation_sql = _validation_sql_for_explain(sql, intent, dialect)
    ok, err, cat, diags = validate_sql(
        dialect,
        validation_sql,
        bind_params_for_sql(validation_sql, intent.param_values),
        schema=schema,
        intent=intent,
    )
    out_err = "" if ok and err is None else (err or "")
    debug(f"[{ASK_PHASE_J}] validate_sql ok={ok}, err={out_err}, diagnostics={len(diags)}")
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


def other_template_owns_question_string(
    templates: dict[str, Any],
    exclude_id: str,
    q_norm: str,
) -> bool:
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
    intent: RuntimeIntent,
    store: dict[str, Any] | None,
    q_norm: str | None,
    semantic_warnings: list[Any] | None,
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
    ref_tmpl: Template,
    _rejected: dict[str, Any],
    intent: RuntimeIntent,
    q_norm: str,
    *,
    reuse_history_index: int,
) -> bool:
    """Return True when direct SQL reuse must ask the user instead of auto-accepting."""
    return not should_auto_accept_for_question(
        ref_tmpl,
        q_norm,
        reuse_history_index=reuse_history_index,
    )


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


def align_template_to_widened_intent(
    template: Template,
    intent: RuntimeIntent,
    dialect: Any,
) -> None:
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
    template.sql_fp = compute_sql_fp(template.sql_param or "", sqlglot_dialect=active_sqlglot_dialect())
    template.structural_defaults = {k: v for k, v in all_pv.items() if is_structural_param_key(k)}
    sig_aliases: dict[str, str] = {}
    for sc in intent.select_cols or []:
        alias = generate_col_alias(sc)
        if alias:
            sig_aliases[sc.signature_key] = alias
    template.display_alias_map = {**template.display_alias_map, **sig_aliases}


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
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
) -> SqlGenerationOutcome:
    """Generate SQL from template reuse or deterministic build, then. validate once."""
    intent = prune_unused_cte_steps(intent)
    intent = expand_shared_pk_tables_for_refs(intent, schema)
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
    if _execution_scope_gate_active(scope_ctx, visible_objects, schema_role):
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
    join_candidates, cmap, cte_join_hints = generate_join_candidates(intent, schema)
    debug("sql generation")
    debug(f"[{ASK_PHASE_J}] tables={intent.tables or []}")
    debug(f"[{ASK_PHASE_J}] grain={intent.grain or 'unknown'}")
    debug(f"[{ASK_PHASE_J}] select_cols={[s.expr.primary_term for s in (intent.select_cols or [])]}")
    debug(f"[{ASK_PHASE_J}] filters_param={len(intent.filters_param or [])}")
    debug(f"[{ASK_PHASE_J}] having_param={len(intent.having_param or [])}")
    debug(f"[{ASK_PHASE_J}] cte_join_hints={list(cte_join_hints.keys()) if cte_join_hints else None}")
    resolved_union_path = resolve_sql_path(
        matched_template=matched_template,
        cols_changed=cols_changed,
        union_sql_path=union_sql_path,
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
    debug(f"[{ASK_PHASE_J}] generation path={active_path}")

    structural_defaults_src: dict[str, Any] | None = None
    if matched_template:
        tmpl_sd = getattr(matched_template, "structural_defaults", None)
        structural_defaults_src = tmpl_sd if tmpl_sd else None

    params = dict(flatten_param_values(intent))
    debug(f"[{ASK_PHASE_J}] params={params}")

    prior_join_fb = lookup_join_feedback_for_question(cast(dict[str, Any], store), q_norm)

    generation_path_label = active_path
    matched_for_outcome: Template | None = None

    if matched_template and routing in (
        GenerationPath.INTENT_DIRECT_MATCH,
        GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE,
    ):
        generation_path_label = resolved_union_path
        matched_for_outcome = matched_template
        tpl_sql_param = matched_template.sql_param
        merge_structural_defaults_for_reuse(
            tpl_sql_param,
            params,
            getattr(matched_template, "structural_defaults", None),
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
        pipeline_trace(
            "pipeline.generate_and_validate_sql.deterministic_sql.path_3",
            lambda: deterministic_sql,
        )
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
            )
        except (
            NoJoinPathError,
            JoinInjectionAlignmentError,
            JoinInjectionFailedError,
        ) as exc:
            debug(f"[{ASK_PHASE_J}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                save_template_store(store)
            return SqlGenerationOutcome(
                "",
                False,
                generation_path_label,
                matched_for_outcome,
                structural_tpl,
                sql_validation_error=str(exc),
                join_matches_template=None,
                error_kind=None,
            )
        intent.sql_param = sql_param
        subs_params = dict(flatten_param_values(intent))
        sql = finalize_substitute_sql(
            intent,
            structural_defaults_src=structural_defaults_src,
            params=subs_params,
        )
        jm3 = _join_matches_template_intent(matched_template, intent)
        debug(f"[{ASK_PHASE_J}] path 3: deterministic sql with fresh joins")
        path_3_payload = {
            "chosen_join_candidate_id": intent.chosen_join_candidate_id,
            "chosen_join_path_signature": intent.chosen_join_path_signature,
            "deterministic_sql": deterministic_sql,
            "sql_param": sql_param,
            "sql_substituted": sql,
            "join_matches_template": jm3,
        }
        pipeline_trace(
            "pipeline.generate_and_validate_sql.path_3",
            lambda: stable_json(path_3_payload),
        )
        ok_c, err_c, cat_c, diags_c = _run_sql_validation_cascade(sql, intent, dialect, schema=schema)
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
            f"[{ASK_PHASE_J}] template path {resolved_union_path.code} "
            f"SQL validation failed: {err_c}; terminating without fresh fallback"
        )
        if persist_template_learning:
            save_template_store(store)
        ek = cat_c.value if cat_c is not None else PIPELINE_BUG_SQL_VALIDATION
        return SqlGenerationOutcome(
            sql,
            False,
            resolved_union_path,
            matched_template,
            structural_tpl,
            sql_validation_error=err_c or None,
            join_matches_template=jm3,
            error_kind=ek,
        )

    elif matched_template and routing in (
        GenerationPath.UNION_TEMPLATE_WIDEN,
        GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
    ):
        generation_path_label = resolved_union_path
        matched_for_outcome = matched_template
        gen_intent = replace(
            intent,
            select_cols=list(union_select_cols or []),
            param_values=dict(intent.param_values),
        )
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
        pipeline_trace(
            "pipeline.generate_and_validate_sql.deterministic_sql.path_4",
            lambda: deterministic_sql,
        )
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
            )
        except (
            NoJoinPathError,
            JoinInjectionAlignmentError,
            JoinInjectionFailedError,
        ) as exc:
            debug(f"[{ASK_PHASE_J}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                save_template_store(store)
            return SqlGenerationOutcome(
                "",
                False,
                generation_path_label,
                matched_for_outcome,
                structural_tpl,
                sql_validation_error=str(exc),
                join_matches_template=None,
                error_kind=None,
            )
        intent.chosen_join_candidate_id = gen_intent.chosen_join_candidate_id
        intent.chosen_join_path_signature = list(gen_intent.chosen_join_path_signature)
        intent.sql_param = sql_param
        sql = finalize_substitute_sql(
            intent,
            structural_defaults_src=structural_defaults_src,
            params=dict(params),
        )
        debug(f"[{ASK_PHASE_J}] path 4: rebuilt deterministic SQL with union cols")
        path_4_final = {
            "chosen_join_candidate_id": intent.chosen_join_candidate_id,
            "chosen_join_path_signature": intent.chosen_join_path_signature,
            "sql_param": sql_param,
            "sql_substituted": sql,
        }
        pipeline_trace(
            "pipeline.generate_and_validate_sql.path_4.final",
            lambda: stable_json(path_4_final),
        )
        if resolved_union_path == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN and union_select_cols:
            intent.select_cols = list(union_select_cols)

    else:
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
        pipeline_trace(
            "pipeline.generate_and_validate_sql.deterministic_sql.path_5",
            lambda: deterministic_sql,
        )
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
            )
        except (
            NoJoinPathError,
            JoinInjectionAlignmentError,
            JoinInjectionFailedError,
        ) as exc:
            debug(f"[{ASK_PHASE_J}] {exc}")
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
            if persist_template_learning:
                save_template_store(store)
            return SqlGenerationOutcome(
                "",
                False,
                generation_path_label,
                matched_for_outcome,
                structural_tpl,
                sql_validation_error=str(exc),
                join_matches_template=None,
                error_kind=None,
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
        sql = finalize_substitute_sql(
            intent,
            structural_defaults_src=structural_defaults_src,
            params=params,
        )
        debug(f"[{ASK_PHASE_J}] path 5: fresh deterministic SQL")
        path_5_final = {
            "chosen_join_candidate_id": intent.chosen_join_candidate_id,
            "chosen_join_path_signature": intent.chosen_join_path_signature,
            "sql_param": sql_param,
            "sql_substituted": sql,
        }
        pipeline_trace(
            "pipeline.generate_and_validate_sql.path_5.final",
            lambda: stable_json(path_5_final),
        )

    ok, err, vcat, vdiags = _run_sql_validation_cascade(sql, intent, dialect, schema=schema)

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
                )
                repaired_intent.sql_param = sql_param_r
                sql_r = finalize_substitute_sql(
                    repaired_intent,
                    structural_defaults_src=structural_defaults_src,
                    params=dict(params),
                )
                ok_r, err_r, vcat_r, vdiags_r = _run_sql_validation_cascade(
                    sql_r, repaired_intent, dialect, schema=schema
                )
                if ok_r:
                    debug(f"[{ASK_PHASE_J}] B.3 diagnostic repair succeeded on retry")
                    intent = repaired_intent
                    sql = sql_r
                    vdiags = vdiags_r
                    ok = True
                    err = err_r
                    vcat = vcat_r
                else:
                    debug(f"[{ASK_PHASE_J}] B.3 retry still failed: {err_r}")
            except (
                NoJoinPathError,
                JoinInjectionAlignmentError,
                JoinInjectionFailedError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                debug(f"[{ASK_PHASE_J}] B.3 retry rebuild raised: {exc}")

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
    cmap: dict[str, list[str]],
    cte_join_hints: dict[str, dict[str, Any]] | None,
    intent: RuntimeIntent,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return join-path signatures when exactly one non-``J00`` candidate exists per scope."""
    main_sig: list[str] = []
    cte_sigs: dict[str, list[str]] = {}
    tbls = intent.tables or []
    if len(tbls) >= 2:
        multi = {k: v for k, v in cmap.items() if k != "J00"}
        if len(multi) == 1:
            main_sig = list(next(iter(multi.values())))
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
                    cte_sigs[step.cte_name] = list(sig)
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
                        f"[{ASK_PHASE_I}] empty join signature for "
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
                        "pipeline._build_per_carrier_join_payloads.main_join_fallback",
                        _main_join_fallback_trace,
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
            debug(
                f"[{ASK_PHASE_I}] CTE '{cte.cte_name}' has no FK or semantic join path → NoJoinPathError",
            )
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

    params_full = dict(flatten_param_values(intent))
    det_for_llm, _ = reduce_structural_sql_placeholders(
        deterministic_sql,
        params_full,
        structural_defaults,
    )

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
        hints_in = dict(cte_join_hints or {})
        gen_main, gen_cte = merge_join_hints_for_na_scopes(
            join_candidates,
            hints_in,
            intent,
            schema,
            virtual_specs,
            na_keys,
        )
        join_candidates["candidates"] = gen_main["candidates"]
        hints_for_pass2 = dict(hints_in)
        for k, v in gen_cte.items():
            hints_for_pass2[k] = v
            if cte_join_hints is not None:
                cte_join_hints[k] = v
        cmap = join_candidate_map(join_candidates)
        pass2_llm = join_scope_pass2_llm_scopes(
            na_keys,
            join_candidates,
            hints_for_pass2,
            intent,
            schema,
            virtual_specs,
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
        if sc != ScopeClass.semantic_only:
            continue
        if sk == JOIN_CHOICE_SCOPE_MAIN:
            vid_final: str | None = candidate_id
            err_label = "main query"
            err_tables = list(main_tables_list)
        elif sk.startswith("cte:"):
            cte_nm = sk.split(":", 1)[1]
            vid_final = cte_join_ids.get(cte_nm)
            err_label = f"CTE '{cte_nm}'"
            err_tables = list(next((t for n, t, _ in cte_scopes if n == cte_nm), []))
        else:
            continue
        if vid_final in (None, "", "NA"):
            raise NoJoinPathError(err_label, err_tables)

    if multi_table and len(main_tables_list) >= 2:
        chosen_cand = next(
            (c for c in main_candidates if c.get("candidate_id") == candidate_id),
            None,
        )
        if chosen_cand is not None and not join_candidate_spans_tables(chosen_cand, main_tables_list):
            for fid in sorted(multi_table_candidates.keys()):
                alt_cand = next(
                    (c for c in main_candidates if c.get("candidate_id") == fid),
                    None,
                )
                if alt_cand is not None and join_candidate_spans_tables(alt_cand, main_tables_list):
                    candidate_id = fid
                    break

    def _validate_scope_span(scope_key: str, chosen_id: str, scope_tables: list[str], hints: dict[str, Any]) -> None:
        if not scope_tables or len(scope_tables) < 2:
            return
        if chosen_id in (None, "", "NA", "J00"):
            return
        cand = next(
            (c for c in hints.get("candidates", []) if c.get("candidate_id") == chosen_id),
            None,
        )
        if cand is None:
            return
        if not join_candidate_spans_tables(cand, scope_tables):
            raise NoJoinPathError(
                scope_key if scope_key != JOIN_CHOICE_SCOPE_MAIN else "main query",
                list(scope_tables),
            )

    _validate_scope_span(JOIN_CHOICE_SCOPE_MAIN, candidate_id, main_tables_list, join_candidates)
    if cte_join_hints:
        for cname, hints_c in cte_join_hints.items():
            sk = join_choice_scope_key_cte(cname)
            if sk not in merged_scope:
                continue
            tbls = list(next((t for n, t, _ in cte_scopes if n == cname), []))
            _validate_scope_span(sk, merged_scope.get(sk, "J00"), tbls, hints_c)

    if not multi_table:
        debug(f"[{ASK_PHASE_I}] single-table intent → J00")
    elif len(multi_table_candidates) == 1 and candidate_id != "J00":
        debug(f"[{ASK_PHASE_I}] resolved candidate_id={candidate_id}")
    elif pass1_llm:
        debug(f"[{ASK_PHASE_I}] LLM chose candidate_id={candidate_id}")

    join_sigs_ordered, edge_kinds_ordered, candidate_id = _build_per_carrier_join_payloads(
        intent,
        cte_join_hints,
        cte_join_ids,
        candidate_id,
        cmap,
        main_candidates,
        multi_table,
        multi_table_candidates,
    )

    sql_param = inject_join_into_deterministic_sql(
        deterministic_sql,
        join_sigs_ordered,
        edge_kinds_ordered=edge_kinds_ordered,
        schema=schema,
        dialect=dialect,
    )
    intent.chosen_join_candidate_id = candidate_id
    main_sig = join_sigs_ordered[-1] if join_sigs_ordered else []
    intent.chosen_join_path_signature = main_sig if multi_table else cmap.get(candidate_id, [])
    if cte_join_ids and intent.cte_steps:
        for cte_step in intent.cte_steps:
            if cte_step.cte_name in cte_join_ids:
                cte_step.chosen_join_candidate_id = cte_join_ids[cte_step.cte_name]
                if cte_join_hints and cte_step.cte_name in cte_join_hints:
                    for cand in cte_join_hints[cte_step.cte_name].get("candidates", []):
                        if cand.get("candidate_id") == cte_step.chosen_join_candidate_id:
                            cte_step.chosen_join_path_signature = cand.get(
                                "join_path_signature",
                                [],
                            )
                            break

    pipeline_trace(
        "pipeline._resolve_joins_fresh.resolved",
        lambda: stable_json(
            {
                "candidate_id": intent.chosen_join_candidate_id,
                "chosen_join_path_signature": list(intent.chosen_join_path_signature or []),
                "join_sigs_ordered": join_sigs_ordered,
                "cte_join_ids": cte_join_ids,
                "sql_param": sql_param,
                "deterministic_sql": deterministic_sql,
            }
        ),
    )
    return sql_param, cte_join_ids


def finalize_substitute_sql(
    intent: RuntimeIntent,
    *,
    structural_defaults_src: dict[str, Any] | None,
    params: dict[str, Any],
) -> str:
    """Substitute bound parameters into ``intent.sql_param`` and return the executable SQL. ``intent.sql_param`` is already canonical because the compositional SQL builder emits column-left predicates from the intent layer; no post-SQL normalization is applied. Ensures ``intent.sql_param`` is non-empty when present."""
    sql_param = intent.sql_param or ""
    intent.sql_param = sql_param
    return finalize_executable_sql(
        sql_param,
        params,
        structural_defaults_src,
        sqlglot_dialect=active_sqlglot_dialect(),
    )


def compute_final_metrics(
    sql: str,
    intent: RuntimeIntent,
    schema: SchemaGraph,
    templates: dict[str, Any],
    join_candidates: dict[str, Any],
    store: dict[str, Any] | TemplateStoreView,
    q_norm: str = "",
    explain_soft_diagnostics: int = 0,
) -> float:
    """Compute final confidence from similarity, shape drift, negative. memory, and EXPLAIN soft diagnostics."""
    known_tables = sorted(schema.tables.keys())
    sql_tables = extract_tables_from_sql(sql, known_tables, sqlglot_dialect=active_sqlglot_dialect())
    ref_tables = collect_referenced_tables(
        intent.select_cols,
        intent.order_by_cols,
        intent.group_by_cols,
        intent.filters_param,
        intent.having_param,
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    expected_tables = set(intent.tables or []) | ref_tables
    used_new_tables = any(t not in expected_tables for t in sql_tables)

    scored: list[tuple[Any, float]] = []
    for t in templates.values():
        cr = structural_compare(intent, t, mode="full")
        raw = cr.similarity_score
        scored.append((t, float(raw) if raw is not None else 0.0))
    scored.sort(key=lambda x: (-x[1], x[0].id))
    best_score = scored[0][1] if scored else 0.0
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    gap = best_score - second_score

    predicted_num_cte = len(intent.cte_steps or [])
    has_agg = any(s.is_aggregated for s in (intent.select_cols or []))
    predicted_shape = SQLShape(
        num_joins=max(0, len(intent.tables or []) - 1),
        has_group_by=(intent.grain == "grouped"),
        has_agg=has_agg,
        num_cte=predicted_num_cte,
    )
    actual_shape = sql_shape(sql, intent, sqlglot_dialect=active_sqlglot_dialect())
    shape_pen = _shape_distance(predicted_shape, actual_shape)

    num_cte_pen = float(abs(predicted_num_cte - actual_shape.num_cte))

    neg_pen = compute_question_feedback_penalty(cast(dict[str, Any], store), q_norm, schema.schema_graph_id)

    conf = compute_confidence(
        best_score,
        gap,
        used_new_tables,
        shape_pen,
        neg_pen,
        0.0,
        num_cte_pen,
        min(1.0, 0.34 * max(0, int(explain_soft_diagnostics))),
    )

    intent.sql_shape = actual_shape

    debug(f"confidence={round(conf, 3)}")
    debug(f"sql_tables={sql_tables}")
    debug(f"shape={actual_shape}")

    return conf


def complete_user_feedback_reject(
    ctx: UserFeedbackRejectSuspendContext,
    *,
    needs_reason: bool,
    reject_reason: str,
    choice_port: InteractiveChoicePort | None = None,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
) -> dict[str, str] | None:
    """Persist user SQL rejection feedback into templates, rejected store, and negative memory."""
    intent = ctx.intent
    sql = ctx.sql
    schema = ctx.schema
    store: dict[str, Any] | TemplateStoreView = ctx.store
    templates = ctx.templates
    q_norm = ctx.q_norm
    matched_template = ctx.matched_template

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
            record_per_question_feedback(
                feedback_template,
                q_norm,
                accept=False,
                path=path_bucket_value,
            )
            _, template_deleted = reject_out_per_question(templates, feedback_template, q_norm)
            entry_fb = summarize_failure_for_memory(
                question=q_norm,
                intent=intent,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash=schema.effective_structural_hash,
                user_reason=norm_reason,
                sql=sql,
            )
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
                sql=sql,
            )
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
        print_rephrase_hint(
            RephraseHint.USER_REJECTED_RESULT,
            rejection_bucket=last_bucket,
        )
    fn_note = getattr(choice_port, "note_turn_outcome", None)
    if callable(fn_note):
        bk = str(last_bucket or RejectionBucket.OTHER.value).strip().upper()
        fn_note(outcome="user_declined", rejection_bucket=bk)
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
) -> dict[str, str] | None:
    """Persist accept/reject feedback into templates, rejected store, and negative memory. Accept and reject paths use *matched_template* as the sole accepted- template target when applicable; there is no ``intent_key`` re- resolution of templates."""
    if choice not in ("y", "n"):
        invalid_input("Invalid choice — please answer y or n.")
        if persist_template_learning:
            save_template_store(store)
        return None

    intent.sql_shape = sql_shape(sql, intent, sqlglot_dialect=active_sqlglot_dialect())
    resolved_path = GenerationPath.parse(generation_path)
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
            delete_rejected_templates_matching_question(cast(dict[str, Any], store), q_norm)

            if matched_rejected_template is not None:
                new_tmpl = promote_rejected_to_template(
                    cast(dict[str, Any], store),
                    templates,
                    q_norm,
                    intent,
                    sql,
                    schema.schema_graph_id,
                    effective_structural_hash=schema.effective_structural_hash,
                    form_storage=form_storage,
                )
                debug(f"promoted prior-negative-memory path to template {new_tmpl.id}")
                promoted = True

            if not promoted and resolved_path == GenerationPath.FRESH:
                debug(f"[{ASK_PHASE_L}] insert_template path 5")
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
                )
            elif not promoted and matched_template is not None and resolved_path == GenerationPath.EXACT_QUESTION_REUSE:
                tmpl = matched_template
                record_template_feedback(tmpl, accept=True)
                record_per_question_feedback(tmpl, q_norm, accept=True, path=path_bucket(resolved_path))
                promote_trust(tmpl, q_norm)
                if not other_template_owns_question_string(templates, tmpl.id, q_norm):
                    _maybe_record_value_history_accept(
                        templates,
                        tmpl,
                        intent,
                        q_norm,
                        form_storage,
                        schema,
                    )
            elif (
                not promoted
                and matched_template is not None
                and resolved_path
                in (
                    GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL,
                    GenerationPath.FUZZY_REUSE_FULL_PARAMS,
                )
            ):
                tmpl = matched_template
                record_template_feedback(tmpl, accept=True)
                record_per_question_feedback(tmpl, q_norm, accept=True, path=path_bucket(resolved_path))
                promote_trust(tmpl, q_norm)
                if not other_template_owns_question_string(templates, tmpl.id, q_norm):
                    _maybe_record_value_history_accept(
                        templates,
                        tmpl,
                        intent,
                        q_norm,
                        form_storage,
                        schema,
                    )
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
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, q_norm, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, q_norm)
                    if not other_template_owns_question_string(templates, tmpl.id, q_norm):
                        _maybe_record_value_history_accept(
                            templates,
                            tmpl,
                            intent,
                            q_norm,
                            form_storage,
                            schema,
                        )
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
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, q_norm, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, q_norm)
                    if not other_template_owns_question_string(templates, tmpl.id, q_norm):
                        _maybe_record_value_history_accept(
                            templates,
                            tmpl,
                            intent,
                            q_norm,
                            form_storage,
                            schema,
                        )
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
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, q_norm, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, q_norm)
                    if not other_template_owns_question_string(templates, tmpl.id, q_norm):
                        _maybe_record_value_history_accept(
                            templates,
                            tmpl,
                            intent,
                            q_norm,
                            form_storage,
                            schema,
                        )
                    old_skeleton = concrete_intent_to_runtime_skeleton(tmpl.intent_signature)
                    new_skeleton = cleared_param_runtime_intent(intent)
                    key_remap = _structural_key_remap_from_assignment_order(old_skeleton, new_skeleton)
                    _remap_value_history_structural_keys(tmpl.value_history, key_remap)
                    align_template_to_widened_intent(tmpl, intent, dialect)
                    reconcile_template_store_until_stable(
                        templates,
                        template_store_view=(store if isinstance(store, TemplateStoreView) else None),
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
                    )
                else:
                    tmpl = matched_template
                    record_template_feedback(tmpl, accept=True)
                    record_per_question_feedback(tmpl, q_norm, accept=True, path=path_bucket(resolved_path))
                    promote_trust(tmpl, q_norm)
                    if not other_template_owns_question_string(templates, tmpl.id, q_norm):
                        _maybe_record_value_history_accept(
                            templates,
                            tmpl,
                            intent,
                            q_norm,
                            form_storage,
                            schema,
                        )
                    old_skeleton = concrete_intent_to_runtime_skeleton(tmpl.intent_signature)
                    new_skeleton = cleared_param_runtime_intent(intent)
                    key_remap = _structural_key_remap_from_assignment_order(old_skeleton, new_skeleton)
                    _remap_value_history_structural_keys(tmpl.value_history, key_remap)
                    align_template_to_widened_intent(tmpl, intent, dialect)
                    reconcile_template_store_until_stable(
                        templates,
                        template_store_view=(store if isinstance(store, TemplateStoreView) else None),
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
            q_norm=q_norm,
            generation_path=generation_path,
            matched_template=matched_template,
            matched_rejected_template=matched_rejected_template,
            dialect=dialect,
            structural_match_templates=structural_match_templates,
        )
        if needs_reason:
            if choice_port is not None and not choice_port.has_pending_choice():
                raise PipelineSuspended(
                    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
                    "What was wrong?",
                    ctx_rej,
                )
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
            )
        return complete_user_feedback_reject(
            ctx_rej,
            needs_reason=False,
            reject_reason="",
            choice_port=choice_port,
            persist_template_learning=persist_template_learning,
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


def result_columns_for_session(
    sql: str | None,
    rows: list[tuple[Any, ...]] | None,
) -> tuple[str, ...] | None:
    """Derive display column names for programmatic ``SessionStep`` consumers."""
    if not rows:
        return None
    n = len(rows[0])
    hdrs = extract_column_headers(sql or "")
    if hdrs and len(hdrs) == n:
        return tuple(hdrs)
    return tuple(f"c{i}" for i in range(n))


def _structural_key_remap_from_assignment_order(
    old_intent: RuntimeIntent,
    new_intent: RuntimeIntent,
) -> dict[str, str]:
    """Map old structural ``s*`` keys to new keys when assignment sequences align in length."""
    old_seq = structural_s_key_assignment_order(old_intent)
    new_seq = structural_s_key_assignment_order(new_intent)
    if len(old_seq) != len(new_seq):
        return {}
    return dict(zip(old_seq, new_seq, strict=True))


def _remap_value_history_structural_keys(
    history: ValueHistory,
    key_remap: dict[str, str],
) -> None:
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
    sql_param: str,
    new_params: dict[str, Any],
    structural_defaults: dict[str, Any] | None,
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
        if intent.grain != "scalar":
            dfw = build_result_dataframe(
                rows,
                intent,
                sql,
                structural_defaults=sd_reuse,
                q_norm=q_norm,
                template_display_alias_map=getattr(ref_tmpl, "display_alias_map", None),
            )
            if dfw is not None:
                save_result_csv(dfw)
        row_tuples = [tuple(r) for r in rows]
        cols = result_columns_for_session(sql, row_tuples)
        note_interactive_turn(
            choice_port,
            outcome="success",
            sql=sql,
            rows=row_tuples,
            columns=cols,
            intent=intent,
        )
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
    return SqlGenerationOutcome(
        ctx.sql,
        True,
        reuse_path,
        ref_tmpl,
        (),
        None,
        True,
        None,
    )


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
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    prompt: bool = True,
    record_question: str | None = None,
    on_param_incomplete: Literal["return_none", "raise"] = "return_none",
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
        ref_tmpl.sql_param,
        new_params,
        getattr(ref_tmpl, "structural_defaults", None),
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

    sd_reuse = getattr(ref_tmpl, "structural_defaults", None)
    reuse_nl = existing_nl if existing_nl else _most_frequent_natural_language(ref_tmpl.value_history)
    concrete_cte_steps = ref_tmpl.intent_signature.cte_steps or []
    runtime_cte_steps = [concrete_cte_to_runtime(c) for c in concrete_cte_steps]

    for cte in runtime_cte_steps:
        keys: set[str] = set()
        for fp in cte.filters_param or []:
            if fp.param_key:
                keys.add(fp.param_key)
        for hp in cte.having_param or []:
            if hp.param_key:
                keys.add(hp.param_key)
        cte.param_values = {k: v for k, v in new_params.items() if k in keys}

    intent = RuntimeIntent(
        tables=ref_tmpl.intent_signature.tables or [],
        grain=ref_tmpl.intent_signature.grain or "row_level",
        select_cols=ref_tmpl.intent_signature.select_cols or [],
        group_by_cols=ref_tmpl.intent_signature.group_by_cols or [],
        order_by_cols=ref_tmpl.intent_signature.order_by_cols or [],
        filters_param=ref_tmpl.intent_signature.filters_param or [],
        having_param=ref_tmpl.intent_signature.having_param or [],
        param_values=new_params,
        cte_steps=runtime_cte_steps,
        column_map=ref_tmpl.intent_signature.column_map or {},
        natural_language=reuse_nl,
        chosen_join_candidate_id=ref_tmpl.chosen_join_candidate_id,
        chosen_join_path_signature=ref_tmpl.chosen_join_path_signature or [],
    )

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
    if _execution_scope_gate_active(scope_ctx, visible_objects, schema_role):
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

    notify(
        "Direct SQL reuse: validated template parameters and SQL.",
        stage="pipeline",
        code=DIAGNOSTIC_CODE_REUSE_HIT,
    )

    try:
        progress("Executing SQL...")
        rows = dialect.execute(
            exec_sql,
            reconcile_execute_bind_params(exec_sql, new_params),
        )
    except AccessError:
        debug(f"[{ASK_PHASE_A}] execute permission denied — continuing to intent parse")
        if on_param_incomplete == "raise":
            raise ConfigError("execute permission denied") from None
        return None

    display_base = _template_effective_sql_display_param(ref_tmpl, dialect=dialect)
    display_sql = (
        finalize_executable_sql(
            display_base,
            new_params,
            sd_reuse,
            sqlglot_dialect=dialect.sqlglot_dialect,
        )
        if display_base and new_params
        else (display_base or exec_sql)
    )

    record_q = record_question if record_question is not None else q_norm
    normalised_choice = "y"
    if prompt and _should_prompt_direct_reuse_user(
        ref_tmpl,
        rejected,
        intent,
        q_norm,
        reuse_history_index=reuse_row_idx,
    ):
        hdr = extract_column_headers(display_sql)
        if choice_port is None:
            print_query_result(rows, display_sql, headers=hdr)
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
            headers=tuple(hdr) if hdr else None,
            is_exact=matched_idx >= 0,
            reuse_path=reuse_path,
            sd_reuse=sd_reuse,
            form_storage=form_storage,
        )
        if choice_port is not None and not choice_port.has_pending_choice():
            raise PipelineSuspended(
                PIPELINE_SUSPEND_ID_DIRECT_REUSE,
                "Is this correct?",
                ctx,
            )
        choice = interactive_yes_no(
            INTERACTIVE_STAGE_DIRECT_REUSE,
            "Is this correct?",
            ["y", "n"],
            choice_port=choice_port,
        )
        normalised_choice = "y" if choice == "y" else "n"
        if normalised_choice == "y":
            debug(f"[{ASK_PHASE_A}] user_accepted_reuse")
        else:
            debug(f"[{ASK_PHASE_A}] user_rejected_reuse")
    else:
        if choice_port is None:
            print_query_result(rows, display_sql, headers=extract_column_headers(display_sql))
        debug(f"[{ASK_PHASE_A}] auto_accepted")

    if normalised_choice == "y" and intent.grain != "scalar":
        dfw = build_result_dataframe(
            rows,
            intent,
            exec_sql,
            structural_defaults=sd_reuse,
            q_norm=q_norm,
            template_display_alias_map=getattr(ref_tmpl, "display_alias_map", None),
        )
        if dfw is not None:
            save_result_csv(dfw)

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
        cols = result_columns_for_session(exec_sql, row_tuples)
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
    return SqlGenerationOutcome(
        exec_sql,
        True,
        reuse_path,
        ref_tmpl,
        (),
        None,
        True,
        None,
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
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
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
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
) -> SqlGenerationOutcome | None:
    """Reuse a template’s SQL: extract params, validate, execute, and. record feedback."""
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
            prev_row,
            getattr(ref_tmpl, "structural_defaults", None),
            ref_tmpl.sql_param or "",
        )
        new_params = (
            _extract_reuse_params_literal_only(q_norm, ref_tmpl, history_index=hi)
            if literal_structural_only
            else _extract_reuse_params_full(q_norm, ref_tmpl, history_index=hi)
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
        raise PipelineSuspended(
            PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
            feedback_body,
            suspend_tail,
        )
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
        sql=render_feedback_sql(intent, schema),
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
            payload=(
                ("q_norm", q_norm),
                ("entry_json", stable_json(entry.to_dict())),
            ),
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
        raise PipelineSuspended(
            PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
            "Is this correct?",
            suspend_tail,
        )
    intent_choice = interactive_yes_no(
        INTERACTIVE_STAGE_INTENT_CONFIRM,
        "Is this correct?",
        ["y", "n"],
        choice_port=choice_port,
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
            d_aliases = enriched_display_alias_map(
                q_norm,
                intent.sql_param or "",
                intent,
                template_display_alias_map,
            )
            display_param = build_display_sql(
                intent.sql_param or "",
                intent,
                d_aliases,
                dialect=dialect,
            )
    if display_param and intent.param_values:
        return finalize_executable_sql(
            display_param,
            intent.param_values,
            structural_defaults,
            sqlglot_dialect=active_sqlglot_dialect(),
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
) -> pandas.DataFrame | None:
    """Build a row-level ``DataFrame`` for programmatic session steps, or ``None`` for scalar grain."""
    if intent.grain == "scalar":
        return None
    display_sql = _final_display_sql_for_results(
        intent,
        sql,
        structural_defaults,
        q_norm=q_norm,
        template_display_alias_map=template_display_alias_map,
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
) -> None:
    """Print result rows to stdout using the resolved display SQL."""
    display_sql = _final_display_sql_for_results(
        intent,
        sql,
        structural_defaults,
        q_norm=q_norm,
        template_display_alias_map=template_display_alias_map,
    )
    print_query_result(rows, display_sql, headers=extract_column_headers(display_sql))


def save_result_csv(df: pandas.DataFrame) -> None:
    """Write *df* to ``results.csv`` in the process working directory."""
    output_path = os.path.join(os.getcwd(), "results.csv")
    df.to_csv(output_path, index=False)
    debug(f"results saved to {output_path}")


def _shape_distance(a: SQLShape, b: SQLShape) -> float:
    """Compute distance between two SQL shapes."""
    d = 0.0
    d += (1.0 / 3.0) * min(1.0, abs(a.num_joins - b.num_joins) / 4.0)
    d += (1.0 / 3.0) * (0.0 if a.has_group_by == b.has_group_by else 1.0)
    d += (1.0 / 3.0) * (0.0 if a.has_agg == b.has_agg else 1.0)
    return d
