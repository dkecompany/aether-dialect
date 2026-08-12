"""Pipeline execute path: param reuse, federation prepare/execute, results/CSV."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import Context, copy_context
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast

import pandas
import sqlglot

from ._config import EngineConfig, EngineLimits
from ._constants import (
    AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER,
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_EXECUTED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_GENERATED,
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
    DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED,
    DIAGNOSTIC_CODE_FEDERATION_TIME_ANCHOR,
    DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
    DIAGNOSTIC_CODE_REUSE_HIT,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_MAPPINGS_VERSION,
    INTERACTIVE_STAGE_DIRECT_REUSE,
    MASTER_AETHERSPACE_NAME,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
)
from ._constants_runtime import (
    ASK_PHASE_A,
    ASK_PHASE_L,
    ASK_PHASE_M,
    PERMISSION_DENIED_USER_MESSAGE,
)
from ._contracts_base import (
    ConfigError,
    EngineContext,
    EngineIdentity,
    FailureCategory,
    FederationCapExceededError,
    FederationConfigError,
    FederationContext,
    FederationMemberExecutionError,
    FederationPartialFailureError,
    FederationRuntimeError,
    FederationTurnCancelledError,
    PredicateGroup,
    ResultCapExceededError,
    RetryableError,
    SpaceContext,
    StatementTimeoutError,
)
from ._contracts_core import (
    AccessError,
    CoordinatorMemberFrame,
    DirectReuseSuspendContext,
    FederatedExecutionOutcome,
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    FederatedSqlBundle,
    FederatedSqlOutcome,
    FederatedStatementRecord,
    FederationExecutionContext,
    GenerationPath,
    PipelineSuspended,
    QuestionFormStorage,
    RuntimeIntent,
    SourceStep,
    SqlGenerationOutcome,
    Template,
    TemplateExecutionResult,
)
from ._contracts_schema import (
    FederationCoordinatorConfig,
    FederationManifest,
    FederationMappings,
    FederationPlanTemplate,
    SchemaGraph,
)
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._dialect_sqlglot_helper import ResultBackend
from ._federation_compose import member_schema_slice, split_qualified_column
from ._federation_execute import (
    column_where_value_type,
    coordinator_member_row_count,
    dialect_streams_arrow_to_coordinator,
    distinct_semijoin_keys,
    enforce_federation_plan_timeout,
    execute_federation_coordinator,
    federation_coordinator_spill_dir,
    federation_member_execution_batches,
    federation_member_parallelism_cap,
    federation_member_resolved_limits,
    federation_member_schema_graph_ids,
    federation_member_timeout_error,
    federation_plan_combine_hash,
    federation_plan_combine_kind,
    federation_plan_matches_template,
    federation_plan_residual_hash,
    federation_plan_step_fingerprints,
    federation_plan_timeout_deadline,
    federation_plan_topology_identity,
    federation_stage_execution_waves,
    federation_user_facing_ineligible_message,
    inject_filter_keys_where,
    inject_semijoin_where,
    load_federation_plan_templates,
    lookup_federation_plan_template_for_question,
    member_frame_column_names,
    member_guard_limit_kwargs,
    order_federation_execution_steps,
    revalidate_prepared_federation_plan,
    save_federation_plan_template,
    semijoin_key_columns,
    source_row_cap_for_source,
    validate_federated_sub_intent,
    validate_member_frame_projection,
)
from ._federation_manifest import (
    emit_federation_member_timezone_mismatch_diagnostics,
    federation_residual_column_headers,
    intersect_member_where_ops,
    member_feedback_q_norm,
    resolve_anchored_temporal_bind,
    stamp_federation_member_template,
    template_is_federation_plan_fragment,
)
from ._federation_plan import (
    apply_projected_keys_to_intent,
    coordinator_residual_bind_map,
    declared_table_for_source_column,
    effective_union_specs,
    federation_plan_is_degenerate,
    member_stage_for_source,
    plan_federated_intent,
    reducing_edge_allowed_for_target,
    render_federation_glue,
    resolve_federated_member_schema,
    resolve_source_column_table,
    semijoin_key_is_allowed,
    semijoin_key_passes_distinct_floor,
    source_by_table_from_schema,
    source_semijoin_enabled,
    source_timeout_for_source,
)
from ._intent_expr import (
    build_virtual_table_specs,
    narrow_bind_map_for_sub_intent,
)
from ._intent_loop import apply_runtime_post_processing
from ._intent_normalize import (
    expand_shared_pk_tables_for_refs,
)
from ._pipeline_generate import (
    artifact_dir_for_template_store,
    enriched_display_alias_map,
    execution_scope_gate_active,
    extract_reuse_params_full,
    extract_reuse_params_literal_only,
    federation_contract_kwargs_for_reuse,
    federation_ineligible_refusal_outcome,
    generate_and_validate_sql,
    generate_join_candidates,
    handle_user_feedback,
    merge_structural_defaults_for_reuse,
    most_frequent_natural_language,
    reuse_params_match_value_schema,
    row_structural_values_match_defaults,
    run_sql_validation_cascade,
    should_prompt_direct_reuse_user,
    space_scope_gate_active,
    sql_validation_refusal_outcome,
    template_effective_sql_display_param,
)
from ._schema_graph import assert_consumer_intent_in_scope, assert_consumer_sql_in_scope, assert_intent_in_scope
from ._sql_gen import (
    build_display_sql,
    generate_col_alias,
    get_join_choice_from_llm,
    join_scope_pass1_plan,
    tables_in_join_scope,
)
from ._templates import TemplateRefs, TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    InteractiveChoicePort,
    active_engine_limits,
    active_engine_runtime_config,
    active_federation_execution_context,
    debug,
    emit_ask_phase,
    federation_turn_cancelled,
    interactive_yes_no,
    llm_usage_attribution,
    normalize_question,
    note_interactive_turn,
    notify,
    phase_timer,
    pop_engine_identity,
    pop_engine_limits,
    pop_federation_execution_context,
    print_query_result,
    progress,
    push_engine_identity,
    push_engine_limits,
    push_federation_execution_context,
    reconcile_execute_bind_params,
    stable_json,
)
from ._utils_intent import flatten_param_values, intent_key
from ._validation_sql import (
    assert_execution_parameters_validated,
    execute_guarded_arrow_table,
    execute_guarded_sql,
    validate_execute_join_semantics,
    validate_replay_join_semantics,
    validate_sql,
)

_T = TypeVar("_T")


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
    schema_context: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
    schema_role: str = "owner",
    context_name: str = MASTER_AETHERSPACE_NAME,
    space_allowed_tables: frozenset[str] | None = None,
    space_allowed_columns: frozenset[str] | None = None,
    space_deny_tables: frozenset[str] | None = None,
    space_deny_columns: frozenset[str] | None = None,
    prompt: bool = True,
    record_question: str | None = None,
    on_param_incomplete: Literal["return_none", "raise"] = "return_none",
    federated_plan: FederatedPlan | None = None,
    validation_only: bool = False,
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

    schema_row = (
        dict(ref_tmpl.value_history.param_values[reuse_row_idx])
        if ref_tmpl.value_history.param_values and reuse_row_idx < len(ref_tmpl.value_history.param_values)
        else {}
    )
    if schema_row and not reuse_params_match_value_schema(new_params, schema_row, ref_tmpl.intent_signature):
        msg = "parameter bind map has incompatible types for template reuse"
        debug(f"[{ASK_PHASE_A}] param_extraction_type_mismatch: {msg}")
        if on_param_incomplete == "raise":
            raise ConfigError(msg)
        return None

    live_ok, stale_reasons = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(ref_tmpl), schema)
    if not live_ok:
        debug(f"[{ASK_PHASE_A}] template_not_live: {','.join(stale_reasons)}")
        if on_param_incomplete == "raise":
            raise ConfigError("stored template no longer matches current schema join paths")
        return None

    sd_reuse = getattr(ref_tmpl, "structural_defaults", None)
    reuse_nl = existing_nl if existing_nl else most_frequent_natural_language(ref_tmpl.value_history)
    concrete_cte_steps = ref_tmpl.intent_signature.cte_steps or []
    runtime_cte_steps = [c.to_runtime() for c in concrete_cte_steps]

    for cte in runtime_cte_steps:
        keys: set[str] = set()
        for fp in PredicateGroup.where_leaves(cte.where) or []:
            if fp.param_key:
                keys.add(fp.param_key)
        for hp in PredicateGroup.having_leaves(cte.having) or []:
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
        limit=ref_tmpl.intent_signature.limit,
        limit_param_key=ref_tmpl.intent_signature.limit_param_key or "",
        distinct_select_index=ref_tmpl.intent_signature.distinct_select_index,
        distinct_on=list(ref_tmpl.intent_signature.distinct_on or []),
    )

    replay_err = validate_replay_join_semantics(intent, schema)
    if replay_err is not None:
        debug(f"[{ASK_PHASE_A}] replay join semantics: {replay_err.message_for_caller}")
        if on_param_incomplete == "raise":
            raise replay_err
        return sql_validation_refusal_outcome(
            replay_err,
            generation_path=reuse_path,
            matched_template=ref_tmpl,
            structural_match_templates=(),
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
    deny_tables = frozenset(space_deny_tables or ())
    deny_columns = frozenset(space_deny_columns or ())
    if space_scope_gate_active(space_tables, space_columns, deny_tables, deny_columns):
        if not assert_intent_in_scope(
            intent, space_tables, space_columns, schema, deny_tables=deny_tables, deny_columns=deny_columns
        ):
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
    try:
        assert_execution_parameters_validated(intent, schema)
    except ConfigError as exc:
        debug(f"[{ASK_PHASE_A}] parameter_validation_failed: {exc}")
        if on_param_incomplete == "raise":
            raise
        return None
    ok, err, _vcat, _vdiags = run_sql_validation_cascade(exec_sql, intent, dialect, schema=schema)
    if not ok:
        debug(f"[{ASK_PHASE_A}] validation_failed: {err}")
        if on_param_incomplete == "raise":
            raise ConfigError(str(err or "SQL validation failed"))
        return None

    if validation_only:
        return SqlGenerationOutcome(exec_sql, True, reuse_path, ref_tmpl, (), None, True, None)

    notify("Direct SQL reuse: validated template parameters and SQL.", stage="pipeline", code=DIAGNOSTIC_CODE_REUSE_HIT)

    try:
        progress("Executing SQL...")
        rows = execute_guarded_sql(
            dialect,
            exec_sql,
            new_params,
            schema=schema,
            intent=intent,
            schema_role=schema_role,
            schema_context=scope_ctx,
            visible_objects=visible_objects,
            context_name=context_name,
        )
    except AccessError:
        debug(f"[{ASK_PHASE_A}] execute permission denied — continuing to intent parse")
        if on_param_incomplete == "raise":
            raise ConfigError("execute permission denied") from None
        return None

    display_base = template_effective_sql_display_param(ref_tmpl, dialect=dialect)
    display_sql = (
        Dialect.finalize_executable_sql(
            display_base, new_params, sd_reuse, sqlglot_dialect=dialect.sqlglot_dialect, for_display=True
        )
        if display_base and new_params
        else (display_base or exec_sql)
    )

    record_q = record_question if record_question is not None else q_norm
    normalised_choice = "y"
    fed_contract = federation_contract_kwargs_for_reuse(reuse_path, federated_plan)
    row_tuples_preview = [tuple(r) for r in rows]
    resolved_headers = result_columns_for_session(
        display_sql,
        row_tuples_preview,
        generation_path=fed_contract.get("generation_path"),
        federated_plan=federated_plan,
        column_names=fed_contract.get("column_names"),
    )
    display_headers = list(resolved_headers) if resolved_headers else None
    if prompt and should_prompt_direct_reuse_user(
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
            save_result_csv_for_store(dfw, store)

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
    reuse_nl = existing_nl if existing_nl else most_frequent_natural_language(ref_tmpl.value_history)
    concrete_cte_steps = ref_tmpl.intent_signature.cte_steps or []
    runtime_cte_steps = [c.to_runtime() for c in concrete_cte_steps]
    for cte in runtime_cte_steps:
        keys: set[str] = set()
        for fp in PredicateGroup.where_leaves(cte.where) or []:
            if fp.param_key:
                keys.add(fp.param_key)
        for hp in PredicateGroup.having_leaves(cte.having) or []:
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


def member_template_for_plan_template(
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


def _exact_reuse_param_row(
    q_norm: str,
    ref_tmpl: Template,
    *,
    schema: SchemaGraph | None = None,
    schema_context: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Return stored bind values for an exact question match, else the first history row."""
    vh = ref_tmpl.value_history
    for i, hist_q in enumerate(vh.questions):
        if hist_q and q_norm == hist_q:
            if i < len(vh.param_values):
                row = dict(vh.param_values[i])
                break
            row = {}
            break
    else:
        row = dict(vh.param_values[0]) if vh.param_values else {}
    if schema is not None and row:
        row = TemplateOps.redact_param_values_for_caller(
            ref_tmpl,
            row,
            schema=schema,
            schema_context=schema_context,
            visible_objects=visible_objects,
        )
    return row


def try_federation_plan_inplace_reuse(
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
    ref_tmpl = member_template_for_plan_template(cached, stores_by_source)
    if ref_tmpl is None:
        return None
    new_params = _exact_reuse_param_row(
        q_norm,
        ref_tmpl,
        schema=composite_schema,
    )
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
        return federation_ineligible_refusal_outcome(
            plan.ineligible_reason,
            generation_path=GenerationPath.FEDERATION_PLAN,
            matched_template=ref_tmpl,
        )
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
    schema_context: EngineContext | FederationContext | None = None,
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
    resolved = TemplateOps.resolve_template_for_question(question_old, templates, template_store=store)
    if resolved is None:
        raise ConfigError(f"No stored template matches question {question_old!r}")
    ref_tmpl, hist_idx = resolved
    q_new_norm = normalize_question(question_new)
    expected_handles = set(TemplateOps.handles_referenced_in_sql_param(ref_tmpl.sql_param or ""))
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
    schema_context: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
    visible_tables: frozenset[str] | None = None,
    schema_role: str = "owner",
    persist_template_learning: bool = False,
) -> TemplateExecutionResult:
    """
    Execute one stored template identified by id or ``sql_fp`` with caller bind values.

    Raises:

        ConfigError: When the ref is unknown, bind values are invalid, or execution fails.
    """
    tmpl = TemplateOps.resolve_template_ref(template_ref, templates)
    if tmpl is None or not TemplateOps.template_enumerable_by_caller(tmpl, visible_tables=visible_tables):
        raise ConfigError(f"unknown template ref {template_ref!r}")
    approval = getattr(tmpl, "approval_state", None)
    if approval is not None and str(getattr(approval, "value", approval)).lower() == "pending":
        raise ConfigError(f"template {tmpl.id!r} is pending approval")
    expected_handles = {
        h for h in TemplateOps.handles_referenced_in_sql_param(tmpl.sql_param or "") if re.fullmatch(r"p\d+", h)
    }
    for key in params:
        if key not in expected_handles:
            raise ConfigError(f"Unknown parameter handle {key!r} for template {tmpl.id}")
    vh = tmpl.value_history
    hist_idx = 0
    if vh.questions:
        primary_q = TemplateOps.primary_template_q_norm(tmpl)
        hist_idx = vh.questions.index(primary_q) if primary_q in vh.questions else 0
    base_row = dict(vh.param_values[hist_idx]) if vh.param_values and hist_idx < len(vh.param_values) else {}
    merged = dict(base_row)
    merged.update(params)
    q_norm = normalize_question(question) if question else TemplateOps.primary_template_q_norm(tmpl)
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
        validation_only=True,
    )
    if outcome is None or not outcome.success:
        raise ConfigError("template execution failed validation or execution")
    reuse_intent = _reuse_runtime_intent_from_template(tmpl, merged)
    exec_bind = reconcile_execute_bind_params(outcome.sql, merged)
    rows = execute_guarded_sql(
        dialect,
        outcome.sql,
        exec_bind,
        schema=schema,
        intent=reuse_intent,
        schema_role=schema_role,
        schema_context=schema_context,
        visible_objects=visible_objects,
    )
    sd_reuse = getattr(tmpl, "structural_defaults", None)
    display_base = template_effective_sql_display_param(tmpl, dialect=dialect)
    display_sql = (
        Dialect.finalize_executable_sql(
            display_base, merged, sd_reuse, sqlglot_dialect=dialect.sqlglot_dialect, for_display=True
        )
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
    schema_context: EngineContext | FederationContext | None = None,
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
    """Reuse a template’s SQL: extract params, validate, execute, and record feedback."""
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
        if schema is not None:
            new_params = TemplateOps.redact_param_values_for_caller(
                ref_tmpl,
                new_params,
                schema=schema,
                schema_context=schema_context,
                visible_objects=visible_objects,
            )
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
        literal_structural_only = row_structural_values_match_defaults(
            prev_row, getattr(ref_tmpl, "structural_defaults", None), ref_tmpl.sql_param or ""
        )
        new_params = (
            extract_reuse_params_literal_only(
                q_norm,
                ref_tmpl,
                history_index=hi,
                schema=schema,
                schema_context=schema_context,
                visible_objects=visible_objects,
            )
            if literal_structural_only
            else extract_reuse_params_full(
                q_norm,
                ref_tmpl,
                history_index=hi,
                schema=schema,
                schema_context=schema_context,
                visible_objects=visible_objects,
            )
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


def extract_column_headers(sql: str) -> list[str]:
    """Parse the outermost ``SELECT`` projection list and return display column names / aliases via AST."""
    return Dialect.sql_outer_select_aliases(sql, sqlglot_dialect=Dialect.active_sqlglot_dialect())


def federated_result_column_headers(
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
        registry_id = (sc.expr.registry_ref() or "").strip()
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
        fed_hdrs = federated_result_column_headers(
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
    display_sql = final_display_sql_for_results(
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
        fed_hdr = federated_result_column_headers(
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
    elif not federated_turn and rows:
        intent_hdrs = intent_result_column_headers(
            intent,
            row_width=len(rows[0]),
            template_display_alias_map=template_display_alias_map,
        )
        if intent_hdrs:
            hdr = list(intent_hdrs)
        else:
            hdr = extract_column_headers(display_sql)
    elif not rows:
        intent_hdrs = intent_result_column_headers(
            intent,
            template_display_alias_map=template_display_alias_map,
        )
        if intent_hdrs:
            hdr = list(intent_hdrs)
        elif column_names:
            hdr = list(column_names)
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
        return os.path.join(artifact_dir_for_template_store(store), "results.csv")
    raise ValueError(
        "results.csv path requires artifacts_dir, csv_dir, or a template store with a persisted artifacts root"
    )


def save_result_csv(
    df: pandas.DataFrame,
    *,
    output_path: str | os.PathLike[str] | None = None,
) -> None:
    """Write *df* to ``results.csv`` at *output_path*."""
    if output_path is None:
        raise ValueError(
            "save_result_csv requires an explicit output_path or artifacts_dir via results_csv_output_path"
        )
    dest = os.fspath(output_path)
    df.to_csv(dest, index=False, lineterminator="\n")
    debug(f"results saved to {dest}")


def _estimate_result_row_bytes(row: tuple[Any, ...]) -> int:
    """Estimate the in-memory byte size of one fetched result row."""
    total = 0
    for val in row:
        if isinstance(val, (bytes, bytearray, memoryview)):
            total += len(val)
        elif isinstance(val, str):
            total += len(val.encode("utf-8", errors="replace"))
        elif val is not None:
            total += sys.getsizeof(val)
    return total


def _push_result_row_limit_sql(sql: str, intent: RuntimeIntent, engine: str, max_rows: int) -> str:
    """Push ``LIMIT max_rows + 1`` when the intent does not already carry a tighter limit."""
    intent_limit = intent.limit
    if intent_limit is not None and int(intent_limit) <= int(max_rows):
        return sql
    dialect_name = DialectRegistry.sqlglot_dialect_for_engine(engine) or engine or "duckdb"
    try:
        parsed = sqlglot.parse_one(sql, read=dialect_name)
        if isinstance(parsed, sqlglot.exp.Select):
            limited = parsed.limit(int(max_rows) + 1, copy=True)
            return limited.sql(dialect=dialect_name)
    except (sqlglot.errors.ParseError, AttributeError, TypeError, ValueError):
        pass
    stripped = sql.rstrip().rstrip(";")
    return f"{stripped} LIMIT {int(max_rows) + 1}"


def _fetch_capped_result_rows(
    backend: ResultBackend,
    sql: str,
    params: dict[str, Any] | None,
    *,
    timeout_ms: int | None = None,
) -> list[tuple[Any, ...]]:
    """Fetch result rows with streaming row and byte caps from active engine limits."""
    limits = active_engine_limits()
    batch_rows = int(limits.result_fetch_batch_rows or 10_000)
    max_rows = limits.max_result_rows
    max_bytes = limits.max_result_bytes
    rows: list[tuple[Any, ...]] = []
    total_bytes = 0
    for batch in backend.fetch_rows_batched(
        sql,
        params,
        batch_rows=batch_rows,
        max_rows=max_rows,
        max_bytes=max_bytes,
        timeout_ms=timeout_ms,
    ):
        for row in batch:
            next_rows = len(rows) + 1
            next_bytes = total_bytes + _estimate_result_row_bytes(row)
            if max_rows is not None and next_rows > max_rows:
                raise ResultCapExceededError(
                    f"result row cap exceeded: {next_rows} rows > cap {max_rows}",
                    limit_key="row_cap",
                )
            if max_bytes is not None and next_bytes > max_bytes:
                raise ResultCapExceededError(
                    f"result byte cap exceeded: {next_bytes} bytes > cap {max_bytes}",
                    limit_key="byte_cap",
                )
            rows.append(row)
            total_bytes = next_bytes
    return rows


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
    gk = dict(gate_kwargs or {})
    schema_role = str(gk.get("schema_role", "owner") or "owner")
    runtime_schema_context = cast(EngineContext | FederationContext | None, gk.get("schema_context"))
    visible_objects = cast(frozenset[str] | None, gk.get("visible_objects"))
    context_name = str(gk.get("context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
    backend = getattr(dialect, "result_backend", None)
    if return_column_names and backend is not None:
        fetch_with_cols = getattr(backend, "fetch_rows_with_columns", None)
        if callable(fetch_with_cols):
            exec_bind = reconcile_execute_bind_params(exec_sql, exec_params)
            if schema is not None:
                ctx: EngineContext | FederationContext = (
                    runtime_schema_context if runtime_schema_context is not None else EngineContext()
                )
                gate_active = execution_scope_gate_active(
                    cast(EngineContext | None, runtime_schema_context),
                    visible_objects,
                    schema_role,
                    context_name=context_name,
                )
                if gate_active and not assert_consumer_sql_in_scope(exec_sql, dialect, ctx, schema, visible_objects):
                    raise AccessError("execute", PERMISSION_DENIED_USER_MESSAGE, reason="scope")
            if schema is not None and intent is not None:
                validate_execute_join_semantics(intent, schema)
                assert_execution_parameters_validated(intent, schema)
            ok, err, _cat, _diags = validate_sql(dialect, exec_sql, exec_bind, schema=schema, intent=intent)
            if not ok:
                raise ValueError(err or "sql validation failed")
            if timeout_ms is not None:
                rows, cols = fetch_with_cols(exec_sql, exec_bind, timeout_ms=int(timeout_ms))
            else:
                rows, cols = fetch_with_cols(exec_sql, exec_bind)
            return list(rows), tuple(str(c) for c in cols) if cols else None
    rows = list(
        execute_guarded_sql(
            dialect,
            exec_sql,
            exec_params,
            schema=schema,
            intent=intent,
            timeout_ms=timeout_ms,
            schema_role=schema_role,
            schema_context=runtime_schema_context,
            visible_objects=visible_objects,
            context_name=context_name,
        )
    )
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
        live_ok, _stale_reasons = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(tmpl), sub_schema)
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
        replay_err = validate_replay_join_semantics(replay_intent, sub_schema)
        if replay_err is not None:
            raise replay_err
        step_gates = dict(gate_map.get(step.source_id, {}))
        if manifest is not None and "allowed_where_ops" not in step_gates:
            engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
            allowed_where_ops = intersect_member_where_ops(dialects_by_source, engine_types_by_source=engine_types)
            binding = next((item for item in manifest.sources if item.source_id == step.source_id), None)
            if binding is not None:
                member_ops = DialectRegistry.extra_where_ops_for_engine(binding.engine)
                step_gates["allowed_where_ops"] = allowed_where_ops & (member_ops | set(FEDERATION_BASE_WHERE_OPS))
        runtime = dict(source_runtimes or {}).get(step.source_id)
        identity_token = None
        limits_token = None
        if runtime is not None:
            identity_token = push_engine_identity(_engine_identity_for_source_runtime(runtime))
            limits_token = push_engine_limits(_engine_limits_for_source_runtime(runtime))
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
                **cast(Any, step_gates),
                **cast(Any, guard_limits),
            )
        finally:
            if limits_token is not None:
                pop_engine_limits(limits_token)
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
    if runtime_config is None or isinstance(runtime_config, type):
        raise RuntimeError(
            f"federation source runtime for {engine_type!r} has no per-engine runtime configuration instance"
        )
    return EngineIdentity(engine_type=engine_type, runtime_config=runtime_config)


def _engine_limits_for_source_runtime(runtime: Any) -> EngineLimits:
    limits = getattr(runtime, "limits", None)
    return limits if isinstance(limits, EngineLimits) else EngineLimits()


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
        step_source_id = step.source_id

        def _prefix_step_scope(
            local: str,
            *,
            _ids: dict[str, str] = step_scope_ids,
            _source_id: str = step_source_id,
        ) -> str:
            if local not in _ids:
                _ids[local] = _federation_batch_join_scope_key(scope_counter)
                scope_to_member[_ids[local]] = (_source_id, local)
            return _ids[local]

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
            TemplateOps.save_template_store(step_store)


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
        ineligible = str(plan.ineligible_reason)
        return FederatedPrepareOutcome(
            success=False,
            plan=plan,
            display_sql="",
            sql_validation_error=federation_user_facing_ineligible_message(ineligible),
            error_kind=FailureCategory.DENIED_REFERENCE.value,
            phase="prepare",
        )
    if not plan.steps:
        return FederatedPrepareOutcome(
            success=False, plan=plan, display_sql="", sql_validation_error="empty federated plan", phase="prepare"
        )
    temporal_bind = resolve_anchored_temporal_bind(plan.steps[0].sub_intent)
    fed_ctx = active_federation_execution_context()
    fed_token = None
    if fed_ctx is None:
        fed_ctx = FederationExecutionContext(
            plan_id=stable_json(
                federation_plan_step_fingerprints(
                    plan,
                    intent_key_fn=intent_key,
                    manifest=manifest,
                    member_graphs=member_graphs,
                    temporal_bind=temporal_bind,
                )
            ),
            temporal_bind=temporal_bind,
        )
        fed_token = push_federation_execution_context(fed_ctx)
    elif temporal_bind is not None and fed_ctx.temporal_bind is None:
        fed_ctx.temporal_bind = temporal_bind
    if temporal_bind is not None:
        notify(
            "Federated turn bound relative temporal predicates to a single anchor.",
            stage="federation",
            code=DIAGNOSTIC_CODE_FEDERATION_TIME_ANCHOR,
            source_id="composite",
            details=(("anchor_iso", temporal_bind.anchor_iso), ("phase", "prepare")),
        )
    emit_federation_member_timezone_mismatch_diagnostics(manifest, plan, schema=composite_schema)
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
                member_ops = DialectRegistry.extra_where_ops_for_engine(binding.engine)
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
        limits_token = None
        if runtime is not None:
            identity = _engine_identity_for_source_runtime(runtime)
            identity_token = push_engine_identity(identity)
            limits_token = push_engine_limits(_engine_limits_for_source_runtime(runtime))
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
            if limits_token is not None:
                pop_engine_limits(limits_token)
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
            if fed_token is not None:
                pop_federation_execution_context(fed_token)
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
    try:
        if degenerate:
            display_sql = per_source[0][1] if per_source else ""
            glue = ""
        else:
            glue = render_federation_glue(
                plan, {sid: f"src_{sid}" for sid, _ in per_source}, schema=composite_schema, manifest=manifest
            )
            display_sql = _format_federated_sql_display(per_source, glue)
        outcome = FederatedPrepareOutcome(
            success=True,
            plan=plan,
            display_sql=display_sql,
            steps=tuple(prepared),
            per_source_sql=tuple(per_source),
            glue_sql=glue,
            composite_schema_graph_id=str(composite_schema.schema_graph_id or ""),
            combine_hash=federation_plan_combine_hash(plan),
            step_fingerprints=federation_plan_step_fingerprints(
                plan,
                intent_key_fn=intent_key,
                manifest=manifest,
                member_graphs=member_graphs,
                temporal_bind=temporal_bind,
            ),
            member_schema_graph_ids=federation_member_schema_graph_ids(plan, member_graphs),
            member_resolved_limits=federation_member_resolved_limits(plan, manifest) if manifest is not None else (),
        )
    finally:
        if fed_token is not None:
            pop_federation_execution_context(fed_token)
    return outcome


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
        read_instant=datetime.now(UTC).isoformat(),
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
    limits_token = None
    if member_runtime is not None:
        identity_token = push_engine_identity(_engine_identity_for_source_runtime(member_runtime))
        limits_token = push_engine_limits(_engine_limits_for_source_runtime(member_runtime))
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
                **cast(Any, guard_limits),
            )
        except StatementTimeoutError as exc:
            raise federation_member_timeout_error(step.source_id, exc) from exc
    finally:
        if limits_token is not None:
            pop_engine_limits(limits_token)
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
        turn_session.active_federation_execution_context = fed_ctx
    fed_token = push_federation_execution_context(fed_ctx)
    frames: dict[str, pandas.DataFrame | CoordinatorMemberFrame] = {}
    executed: dict[str, pandas.DataFrame | CoordinatorMemberFrame] = {}
    combine_kind = federation_plan_combine_kind(prepared.plan)
    try:
        if effective_union_specs(prepared.plan) and len(execution_steps) > 1:
            for wave in stage_waves:
                if wave.stage.kind != "member":
                    emit_ask_phase(ASK_PHASE_L, stage=cast(Any, wave.stage))
                    continue
                member_wave = wave.member_steps
                if not member_wave:
                    continue
                for step in member_wave:
                    emit_ask_phase(ASK_PHASE_L, source=step.source_id, stage=cast(Any, wave.stage))
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
                    emit_ask_phase(ASK_PHASE_L, stage=cast(Any, wave.stage))
                    continue
                member_wave = wave.member_steps
                if not member_wave:
                    continue
                for step in member_wave:
                    emit_ask_phase(ASK_PHASE_L, source=step.source_id, stage=cast(Any, wave.stage))
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
                        succeeded.append((step.source_id, row_count, datetime.now(UTC).isoformat()))
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
                        read_instant = datetime.now(UTC).isoformat()
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
                read_instant=datetime.now(UTC).isoformat(),
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
        if turn_session is not None and getattr(turn_session, "active_federation_execution_context", None) is fed_ctx:
            turn_session.active_federation_execution_context = None


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
    if isinstance(exc, FederationCapExceededError) and exc.limit_key == "timeout_ms":
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
                Dialect.cancel_in_flight_statement(member_dialect)
    elif dialect is not None:
        Dialect.cancel_in_flight_statement(dialect)
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
            Dialect.cancel_in_flight_statement(member_dialect)
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
    limits_token = None
    if member_runtime is not None:
        identity_token = push_engine_identity(_engine_identity_for_source_runtime(member_runtime))
        limits_token = push_engine_limits(_engine_limits_for_source_runtime(member_runtime))
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
        if limits_token is not None:
            pop_engine_limits(limits_token)
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
    deny_tables = frozenset(gate_kwargs.get("space_deny_tables") or ())
    deny_columns = frozenset(gate_kwargs.get("space_deny_columns") or ())
    if space_scope_gate_active(space_tables, space_columns, deny_tables, deny_columns):
        if not assert_intent_in_scope(
            execution_intent,
            space_tables,
            space_columns,
            sub_schema,
            deny_tables=deny_tables,
            deny_columns=deny_columns,
        ):
            raise FederationRuntimeError(
                f"member {step.source_id!r} statement is outside the session aetherspace scope"
            )
    schema_context = gate_kwargs.get("schema_context")
    visible_objects = gate_kwargs.get("visible_objects")
    schema_role = str(gate_kwargs.get("schema_role", "owner") or "owner")
    context_name = str(gate_kwargs.get("context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
    if schema_context is not None and execution_scope_gate_active(
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
                **cast(Any, guard_limits),
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
                **cast(Any, guard_limits),
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
                    succeeded.append((source_id, coordinator_member_row_count(frame), datetime.now(UTC).isoformat()))
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
        tmpl = TemplateOps.insert_template(
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
        TemplateOps.save_template_store(member_store)
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


def complete_direct_sql_reuse_user_choice(
    ctx: DirectReuseSuspendContext,
    choice: str | None,
    *,
    choice_port: InteractiveChoicePort | None = None,
    persist_template_learning: bool = True,
) -> SqlGenerationOutcome:
    """Apply the user's confirmation after a deferred direct-reuse prompt."""
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
        fed_contract = federation_contract_kwargs_for_reuse(reuse_path, None)
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
                save_result_csv_for_store(dfw, store)
        row_tuples = [tuple(r) for r in rows]
        cols = ctx.headers if ctx.headers else result_columns_for_session(sql, row_tuples, **fed_contract)
        note_interactive_turn(
            choice_port,
            outcome="success",
            sql=sql,
            rows=row_tuples,
            columns=cols,
            intent=intent,
            matched_template=ref_tmpl,
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
    return SqlGenerationOutcome(ctx.sql, True, reuse_path, ref_tmpl, (), None, True, None)


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
        fed_hdr = federated_result_column_headers(
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
    display_sql = final_display_sql_for_results(
        intent, sql, structural_defaults, q_norm=q_norm, template_display_alias_map=template_display_alias_map
    )
    hdr = extract_column_headers(display_sql)
    if hdr:
        return pandas.DataFrame(rows, columns=hdr)
    return pandas.DataFrame(rows)


def save_result_csv_for_store(
    df: pandas.DataFrame,
    store: dict[str, Any] | TemplateStoreView,
    *,
    artifacts_dir: str | None = None,
    csv_dir: str | None = None,
) -> None:
    """Write ``results.csv`` when an explicit destination can be resolved from *store*."""
    try:
        output_path = results_csv_output_path(
            store,
            artifacts_dir=artifacts_dir,
            csv_dir=csv_dir,
        )
    except ValueError:
        debug("save_result_csv skipped: no explicit artifacts destination")
        return
    save_result_csv(df, output_path=output_path)


def final_display_sql_for_results(
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
            runtime_cfg = active_engine_runtime_config()
            dialect = DialectRegistry.get_dialect(EngineConfig.TYPE, runtime_cfg)
        except ValueError:
            dialect = None
        if dialect is None:
            display_param = intent.sql_param or ""
        else:
            d_aliases = enriched_display_alias_map(q_norm, intent.sql_param or "", intent, template_display_alias_map)
            display_param = build_display_sql(intent.sql_param or "", intent, d_aliases, dialect=dialect)
    if display_param and intent.param_values:
        return Dialect.finalize_executable_sql(
            display_param,
            intent.param_values,
            structural_defaults,
            sqlglot_dialect=Dialect.active_sqlglot_dialect(),
            for_display=True,
        )
    return display_param or sql
