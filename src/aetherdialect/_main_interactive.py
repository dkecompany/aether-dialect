"""Interactive intent-to-SQL phase, suspend dispatch, and seed/qsim runners."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

from ._config import (
    EngineConfig,
    EngineRuntimeConfig,
    PolicyConfig,
    QSimConfig,
    SeedWarmupConfig,
)
from ._constants import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED,
    DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE,
    DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
    DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
    DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
    DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED,
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION,
    INTERACTIVE_STAGE_SQL_FEEDBACK,
    JSON_COMPACT_SEPARATORS,
    MASTER_AETHERSPACE_NAME,
    NORMALIZED_SEEDS_TXT,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_EXECUTE,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_SQL,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    QSIM_QUESTIONS_PATTERN,
    SECRET_ENV_KEYS,
    SEED_NORMALIZATION_JSON,
    SESSION_KIND_ERROR,
    TRUST_AUTO_ACCEPT_THRESHOLD,
)
from ._constants_runtime import (
    MIGRATION_HEADER_BY_TIER,
    PERMISSION_DENIED_USER_MESSAGE,
    PERMISSION_DRIFT_CONTACT_ADMIN_MESSAGE,
    WARMUP_PHASE_B,
    WARMUP_PHASE_C,
    WARMUP_PHASE_D,
    WARMUP_PHASE_E,
    WARMUP_PHASE_F,
    WARMUP_PHASE_G,
    WARMUP_PHASE_I,
    WARMUP_PHASE_K,
)
from ._contracts_base import (
    Diagnostic,
    EngineContext,
    FailureCategory,
    FederationCapExceededError,
    FederationConfigError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationPartialFailureError,
    FederationTurnCancelledError,
    MigrationReport,
    PredicateGroup,
    RetryableError,
    SchemaRole,
    SpaceContext,
)
from ._contracts_core import (
    AccessError,
    DirectReuseSuspendContext,
    FederatedPlan,
    FederatedPrepareOutcome,
    FederatedSqlBundle,
    FeedbackKind,
    GenerationPath,
    IntentInterpretation,
    IntentSummary,
    InteractiveChoicePort,
    InteractiveTailSnapshot,
    InterpretPlan,
    LLMConfig,
    PipelineSessionMarker,
    PipelineSuspended,
    QuestionFormStorage,
    RefinementContext,
    RefinementRetry,
    RephraseHint,
    RuntimeConfig,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    SessionOutcome,
    SessionStep,
    SqlExecuteSuspendContext,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
    TurnPolicySnapshot,
    UserFeedbackRejectSuspendContext,
)
from ._contracts_schema import (
    FederationManifest,
    FederationMappings,
    FederationPlanTemplate,
    QSimSummary,
    SchemaGraph,
    SeedWarmupSummary,
)
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._expansion_ops import (
    expand_gold_intents,
)
from ._federation_compose import (
    assert_federation_sql_history_warmup_allowed,
    assert_query_log_warmup_allowed,
)
from ._federation_execute import (
    check_federation_member_drift_at_turn_start,
    clear_federated_turn_state,
    federation_ineligible_answerable_hint,
    federation_plan_combine_hash,
    federation_plan_matches_template,
    federation_plan_residual_hash,
    federation_plan_step_fingerprints,
    federation_plan_topology_identity,
    federation_user_facing_error_message,
    federation_user_facing_ineligible_message,
    lookup_federation_plan_template,
)
from ._federation_manifest import (
    owner_is_aether_federation,
    resolve_anchored_temporal_bind,
)
from ._federation_plan import (
    plan_federated_intent,
    qsim_intent_eligible_on_federation,
    resolve_federated_combine,
)
from ._intent_expr import interpret_plan_is_unanswerable
from ._intent_loop import (
    collect_structural_match_templates,
    list_union_match_candidates,
    match_template_for_union,
    reconcile_template_store_until_stable,
    structural_compare,
)
from ._main_spaces import MainSpaceOps
from ._pipeline_execute import (
    build_result_dataframe,
    complete_direct_sql_reuse_user_choice,
    display_final_results_to_stdout,
    execute_federated_prepare,
    persist_federated_member_stores,
    prepare_federated_sql_plan,
    replay_federated_prepare_from_plan_template,
    result_columns_for_session,
    save_result_csv_for_store,
)
from ._pipeline_generate import (
    best_accepted_template_similarity,
    build_interactive_tail_snapshot,
    clear_interpret_schema_invalid_after_user_accept,
    complete_user_feedback_reject,
    confirm_intent_with_user,
    emit_explain_soft_diagnostics,
    execution_scope_gate_active,
    generate_and_validate_sql,
    handle_user_feedback,
    parse_intent_via_llm,
    prepare_union_match_join_phase,
    refinement_retry_available,
    stamp_sql_shape,
)
from ._qsim import (
    generate_all_intents,
    generate_all_questions,
    instantiate_all,
)
from ._schema_graph import (
    assert_consumer_intent_in_scope,
    compute_schema_limits,
    effective_execution_visible_tables,
)
from ._schema_reflect import (
    emit_materialized_view_answer_diagnostics,
)
from ._seed_warmup import (
    JoinCacheEntry,
    JoinCacheKey,
    SeedWarmupCacheSession,
)
from ._sql_to_intent import (
    compute_sql_history_content_hash,
    convert_sql_to_intent,
    dedup_runtime_intents,
    fetch_query_log,
    load_sql_history_statements,
    seed_warmup_intent_from_runtime_intent,
)
from ._templates import TemplateRefs, TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    active_engine_runtime_config,
    debug,
    diagnostic_debug_enabled,
    drain_llm_usage_records,
    emit_llm_usage_summary_diagnostics,
    emit_session_refusal_diagnostic,
    failure_kind_is_permission_denied,
    interactive_yes_no,
    note_interactive_turn,
    notify,
    permission_denied_detail_logging_enabled,
    print_rephrase_hint,
    progress,
    refusal_diagnostic_code_for_federation_reason,
    refusal_user_text_for_code,
)
from ._utils_intent import (
    body_similarity_key,
    enumerate_zero_row_equality_where,
    flatten_param_values,
    intent_key,
    patch_where_literal_on_intent,
    zero_row_where_remediation_candidates,
    zero_row_where_suggestions,
)
from ._validation_sql import execute_guarded_sql


class MainInteractiveOps:
    """Interactive intent-to-SQL phase, suspend dispatch, and seed/qsim runners."""

    @staticmethod
    def owner_from_choice_port(choice_port: InteractiveChoicePort | None) -> Any | None:
        if choice_port is None:
            return None
        return getattr(choice_port, "_owner", None)

    @staticmethod
    def note_access_error_turn(choice_port: InteractiveChoicePort | None, exc: AccessError) -> None:
        """Map scope vs warehouse ``AccessError`` to the correct interactive outcome."""
        if permission_denied_detail_logging_enabled():
            debug(f"[main_execution] access error detail: {exc!r} reason={getattr(exc, 'reason', None)!r}")
        reason = getattr(exc, "reason", "warehouse")
        owner = MainInteractiveOps.owner_from_choice_port(choice_port)
        schema_role = str(getattr(owner, "_schema_role", "owner") or "owner")
        if reason == "scope" or schema_role == "consumer":
            note_interactive_turn(choice_port, outcome="permission_denied", error=None, sql=None, intent=None)
            return
        note_interactive_turn(
            choice_port,
            outcome="validation_failed",
            error=str(exc),
            sql=None,
            intent=None,
        )

    @staticmethod
    def failure_category_for_terminal_step(step: SessionStep) -> str | None:
        """Map a terminal error :class:`SessionStep` to a coarse failure category string."""
        if step.kind != SESSION_KIND_ERROR:
            return None
        err = step.error
        if err is None:
            return None
        if err.limit_key:
            if err.limit_key == "timeout_ms":
                return FailureCategory.EXECUTION_TIMEOUT.value
            return FailureCategory.EXECUTION_OTHER_ERROR.value
        if err.phase or err.source_id:
            return FailureCategory.EXECUTION_OTHER_ERROR.value
        for d in step.diagnostics:
            code_u = (d.code or "").upper()
            if code_u in {"EXPLAIN_COST_EXCEEDED", "EXPLAIN_COST"} or "explain_cost_exceeded" in (d.code or "").lower():
                return FailureCategory.EXECUTION_COST_EXCEEDED.value
        if err.code == SessionOutcome.COST_EXCEEDED:
            return FailureCategory.EXECUTION_COST_EXCEEDED.value
        if err.code == SessionOutcome.EXECUTION_TIMEOUT:
            return FailureCategory.EXECUTION_TIMEOUT.value
        if err.code in {SessionOutcome.FORBIDDEN, SessionOutcome.UNANSWERABLE}:
            return FailureCategory.PERMISSION_ERROR.value
        if err.code == SessionOutcome.PARSE_FAILED:
            return FailureCategory.INTENT_ERROR.value
        blob = " ".join([*(x.message for x in step.diagnostics)]).lower()
        if ("cost" in blob or "explain_cost" in blob) and ("exceed" in blob or "cap" in blob):
            return FailureCategory.EXECUTION_COST_EXCEEDED.value
        if "timeout" in blob or "statement_timeout" in blob:
            return FailureCategory.EXECUTION_TIMEOUT.value
        permission_markers = (
            "permission denied",
            "access_policy",
            "denied_reference",
            "contact your administrator",
            "out of execution scope",
            "insufficient privilege",
            "insufficient privileges",
        )
        if any(marker in blob for marker in permission_markers):
            return FailureCategory.PERMISSION_ERROR.value
        intent_markers = (
            "intent_parse_failed",
            "intent_schema_invalid",
            "intent schema_invalid",
            "schema_invalid",
            "could not compose intent",
            "intent_error",
        )
        if any(marker in blob for marker in intent_markers):
            return FailureCategory.INTENT_ERROR.value
        transport_auth_markers = (
            "password authentication",
            "authentication failed",
            "invalid credentials",
            "login failed",
            "unauthorized",
            "access token",
            "invalid token",
            "connection refused",
            "connection reset",
            "could not connect",
            "server closed the connection",
            "name or service not known",
            "network unreachable",
            "broken pipe",
        )
        if any(marker in blob for marker in transport_auth_markers):
            return FailureCategory.TRANSPORT_AUTH.value
        return FailureCategory.EXECUTION_OTHER_ERROR.value

    @staticmethod
    def federation_error_step_fields(exc: BaseException) -> dict[str, Any]:
        """Extract structured federation attribution fields from a runtime error."""
        if isinstance(exc, FederationPartialFailureError):
            return {
                "federation_source_id": exc.source_id or None,
                "federation_phase": exc.phase or None,
                "federation_succeeded": tuple(exc.succeeded),
            }
        if isinstance(exc, FederationTurnCancelledError):
            return {
                "federation_source_id": exc.source_id or None,
                "federation_phase": exc.phase or None,
                "federation_succeeded": tuple(exc.succeeded),
            }
        if isinstance(exc, FederationMemberExecutionError):
            return {
                "federation_source_id": exc.source_id or None,
                "federation_phase": exc.phase or None,
                "federation_succeeded": (),
            }
        if isinstance(exc, FederationCapExceededError):
            return {
                "federation_source_id": exc.source_id or None,
                "federation_phase": "member" if exc.source_id else "coordinator",
                "federation_limit_key": exc.limit_key or None,
                "federation_succeeded": (),
            }
        if isinstance(exc, FederationMemberProbeError):
            return {
                "federation_source_id": exc.source_id or None,
                "federation_phase": "prepare",
                "federation_succeeded": (),
            }
        rejection_bucket = getattr(exc, "rejection_bucket", None)
        if rejection_bucket:
            return {
                "federation_source_id": getattr(exc, "source_id", None) or None,
                "federation_phase": getattr(exc, "phase", None) or None,
                "rejection_bucket": str(rejection_bucket),
                "federation_succeeded": (),
            }
        return {}

    @staticmethod
    def federation_error_diagnostics(exc: BaseException) -> tuple[Diagnostic, ...]:
        """Build turn diagnostics for a structured federation terminal error."""
        fields = MainInteractiveOps.federation_error_step_fields(exc)
        if not fields:
            return ()
        source_id = str(fields.get("federation_source_id") or "") or "composite"
        phase = str(fields.get("federation_phase") or "execution")
        user_message = federation_user_facing_error_message(exc)
        details: list[tuple[str, str]] = [
            ("message", user_message),
            ("source_id", source_id),
            ("phase", phase),
        ]
        if fields.get("federation_limit_key"):
            details.append(("limit_key", str(fields["federation_limit_key"])))
        succeeded = fields.get("federation_succeeded") or ()
        if succeeded:
            details.append(("succeeded", ",".join(item[0] for item in succeeded)))
        detail_tuple = tuple(details)
        if isinstance(exc, FederationCapExceededError):
            return (
                Diagnostic(
                    stage="execution",
                    level="error",
                    code=DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED,
                    message=user_message,
                    details=detail_tuple,
                    source_id=source_id,
                    phase=phase,
                ),
            )
        if isinstance(exc, FederationMalformedMemberAnswerError):
            return (
                Diagnostic(
                    stage="execution",
                    level="error",
                    code=DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
                    message=user_message,
                    details=detail_tuple,
                    source_id=source_id,
                    phase=phase,
                ),
            )
        if isinstance(exc, FederationJoinFanOutError):
            return (
                Diagnostic(
                    stage="execution",
                    level="error",
                    code=DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
                    message=user_message,
                    details=detail_tuple,
                    source_id=source_id,
                    phase=phase,
                ),
            )
        if isinstance(exc, FederationMemberExecutionError):
            return (
                Diagnostic(
                    stage="execution",
                    level="error",
                    code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
                    message=user_message,
                    details=detail_tuple,
                    source_id=source_id,
                    phase=phase,
                ),
            )
        if isinstance(exc, FederationTurnCancelledError):
            return (
                Diagnostic(
                    stage="execution",
                    level="error",
                    code=DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
                    message=user_message,
                    details=detail_tuple,
                    source_id=source_id,
                    phase=phase,
                ),
            )
        if isinstance(exc, FederationMemberProbeError):
            return (
                Diagnostic(
                    stage="execution",
                    level="error",
                    code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED,
                    message=user_message,
                    details=detail_tuple,
                    source_id=source_id,
                    phase=phase,
                ),
            )
        return (
            Diagnostic(
                stage="execution",
                level="error",
                code=DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
                message=user_message,
                details=detail_tuple,
                source_id=source_id,
                phase=phase,
            ),
        )

    @staticmethod
    def interactive_attach_refinement_ctx(
        choice_port: InteractiveChoicePort | None, refinement_ctx: RefinementContext
    ) -> None:
        """Bind turn-local refinement state to an interactive session when supported."""
        if choice_port is None:
            return
        attach = getattr(choice_port, "_attach_refinement_ctx", None)
        if callable(attach):
            attach(refinement_ctx)

    @staticmethod
    def persist_template_learning_for_pipeline_session(port: Any | None) -> bool:
        """Return whether template-store and question-feedback mutations may be written for this choice-port session."""
        if port is None:
            return True
        return getattr(port, "_session_mode", "writer") == "writer"

    @staticmethod
    def interactive_run_intent_pass(
        *,
        corrected_text: str,
        q_norm: str,
        dialect: Any,
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Any],
        rejected: dict[str, Any],
        schema_terms: set[str],
        choice_port: InteractiveChoicePort | None,
        form_storage: QuestionFormStorage | None,
        refinement_ctx: RefinementContext,
        persist_template_learning: bool,
    ) -> bool:
        """Parse intent once and continue through confirmation and SQL feedback."""
        MainSpaceOps.raise_if_session_turn_cancelled()
        if refinement_ctx.pending_retry:
            refinement_ctx.pending_retry = False
            if refinement_ctx.skip_refinement_increment_once:
                refinement_ctx.skip_refinement_increment_once = False
            else:
                refinement_ctx.refinement_rounds_executed += 1
        msg = "Refining intent..." if refinement_ctx.accumulated_reasons else "Processing intent..."
        progress(msg)
        parsed_intent, semantic_warnings, _, interpret_plan = parse_intent_via_llm(
            corrected_text,
            schema,
            templates,
            store,
            choice_port=choice_port,
            refinement_ctx=refinement_ctx,
            persist_template_learning=persist_template_learning,
        )
        if parsed_intent is None:
            if interpret_plan is not None and interpret_plan_is_unanswerable(interpret_plan):
                code = DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
                note_interactive_turn(
                    choice_port,
                    outcome="not_available_in_context",
                    error=refusal_user_text_for_code(code),
                    refusal_diagnostic_code=code,
                )
            else:
                note_interactive_turn(choice_port, outcome="parse_failed", error="Intent parse failed.")
            return False
        ik = intent_key(parsed_intent)
        if refinement_ctx.refinement_rounds_executed > 0 and ik == refinement_ctx.last_intent_key:
            refinement_ctx.block_further_refinement = True
        refinement_ctx.last_intent_key = ik
        MainInteractiveOps._run_interactive_post_intent_parse(
            q_norm,
            parsed_intent,
            semantic_warnings,
            dialect,
            schema,
            store,
            templates,
            rejected,
            schema_terms,
            choice_port,
            form_storage=form_storage,
            refinement_ctx=refinement_ctx,
            persist_template_learning=persist_template_learning,
            interpret_plan=interpret_plan,
        )
        return True

    @staticmethod
    def intent_interpretation_from_plan(plan: InterpretPlan | None) -> IntentInterpretation | None:
        """Project an :class:`InterpretPlan` into session-step traceability."""
        if plan is None:
            return None
        return IntentInterpretation(approach=plan.approach, grounding=plan.grounding)

    @staticmethod
    def build_intent_summary(intent: RuntimeIntent) -> IntentSummary:
        """Project a :class:`RuntimeIntent` into a compact :class:`IntentSummary` for session steps."""
        sel = tuple(sc.expr.signature_key for sc in intent.select_cols or [])
        flt = tuple(fp.signature_key for fp in PredicateGroup.where_leaves(intent.where) or [])
        gb = tuple(e.signature_key for e in intent.group_by_cols or [])
        ob = tuple(f"{oc.expr.signature_key} {oc.direction}" for oc in intent.order_by_cols or [])
        return IntentSummary(
            tables=tuple(intent.tables or ()),
            select_cols=sel,
            filters=flt,
            group_by=gb,
            order_by=ob,
            limit=intent.limit,
            natural_language=(intent.natural_language or "").strip(),
        )

    @staticmethod
    def _gold_intent_store_union_widen_blocks_warmup(si: SeedWarmupIntent, templates: dict[str, Template]) -> bool:
        """Return True when a gold row matches the store only via disallowed ``UNION_TEMPLATE_WIDEN`` / ``UNION_TEMPLATE_AND_RUNTIME_WIDEN`` paths."""
        if (si.source or "gold") != "gold":
            return False
        rt = si.to_runtime_intent()
        for tmpl in templates.values():
            if tmpl.trust_level < 1:
                continue
            cr = structural_compare(rt, tmpl, mode="warmup_gold_store_check")
            if cr.union_sql_path in (
                GenerationPath.UNION_TEMPLATE_WIDEN,
                GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
            ):
                return True
        return False

    @staticmethod
    def _get_next_qsim_version(artifacts_dir: str) -> int:
        """Return the next monotonic QSim version for an artifacts directory."""
        pattern = os.path.join(artifacts_dir, "qsim_questions_v*.txt")
        existing = glob.glob(pattern)
        versions: list[int] = []
        for fpath in existing:
            base = os.path.basename(fpath)
            if not base.startswith("qsim_questions_v") or not base.endswith(".txt"):
                continue
            core = base[len("qsim_questions_v") : -len(".txt")]
            try:
                versions.append(int(core))
            except ValueError:
                continue
        summary_path = os.path.join(artifacts_dir, "qsim_summary.json")
        qsim_dir = os.path.join(artifacts_dir, "qsim")
        index_path = os.path.join(qsim_dir, "index.jsonl")
        if os.path.isfile(index_path):
            try:
                with open(index_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if isinstance(row, dict) and row.get("version") is not None:
                            try:
                                versions.append(int(row["version"]))
                            except (TypeError, ValueError):
                                continue
            except (json.JSONDecodeError, OSError):
                pass
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summaries = json.load(f)
                if isinstance(summaries, list):
                    for row in summaries:
                        if isinstance(row, dict) and row.get("version") is not None:
                            try:
                                versions.append(int(row["version"]))
                            except (TypeError, ValueError):
                                continue
            except (json.JSONDecodeError, OSError):
                pass
        return max(versions) + 1 if versions else 1

    @staticmethod
    def format_qsim_summary_line(s: QSimSummary) -> str:
        """Single-line human summary for one QSim run."""
        return f"  v{s.version}: intents={s.num_intents}  questions={s.num_questions}  seed={s.seed}"

    @staticmethod
    def format_seed_warmup_summary(s: SeedWarmupSummary) -> str:
        """Multi-line human summary for one seed warmup report."""
        pct = (100.0 * s.success / s.total) if s.total else 0.0
        return "\n".join(
            (
                f"Seed warmup summary (version {s.version}):",
                f"  Gold intents:          {s.gold_intents_total} total",
                f"  Synthetic unique:      {s.unique_prompts}",
                f"  Attempted:             {s.total}",
                f"  Succeeded:             {s.success}  ({pct:.1f}%)",
                f"  Failed:                {s.failed}",
                f"  Templates inserted:    {s.templates_added}",
            )
        )

    @staticmethod
    def print_migration_applied(report: MigrationReport, sink: Callable[[str], None]) -> None:
        """Emit a per-tier user-friendly migration summary."""
        tier_key = report.tier.value
        header = MIGRATION_HEADER_BY_TIER.get(tier_key)
        if header is None:
            return
        sink(header)
        if report.renamed_tables:
            sink(f"  Renamed {len(report.renamed_tables)} table(s).")
        if report.renamed_columns:
            sink(f"  Renamed {len(report.renamed_columns)} column(s).")
        if report.dropped_tables:
            sink(f"  Removed {len(report.dropped_tables)} dropped table(s) from learning.")
        if report.added_tables:
            sink(f"  Added {len(report.added_tables)} table(s) to the schema.")
        if report.added_columns:
            sink(f"  Added {len(report.added_columns)} column(s) to the schema.")
        if report.value_type_changed_columns:
            sink(f"  Re-checked {len(report.value_type_changed_columns)} column(s) whose type changed.")
        if report.refreshed_descriptions:
            sink(f"  Refreshed {len(report.refreshed_descriptions)} description(s).")
        if report.remapped_templates:
            sink(f"  Updated {report.remapped_templates} learned template(s) to new names.")
        if report.destroyed_templates:
            sink(f"  Removed {report.destroyed_templates} learned template(s) that no longer fit.")
        if report.surgically_invalidated:
            sink(f"  Marked {report.surgically_invalidated} learned template(s) as stale pending re-check.")

    @staticmethod
    def _redact_display_value(field_label: str, raw: object) -> str:
        """Return ``***`` for known secret env keys or secret-like field names."""
        label = field_label.strip()
        upper = label.upper()
        if upper in SECRET_ENV_KEYS:
            return "***"
        low = label.lower()
        for token in ("password", "token", "api_key", "secret"):
            if token in low:
                return "***"
        if raw is None:
            return ""
        return str(raw)

    @staticmethod
    def describe_runtime_config(
        runtime: RuntimeConfig,
        llm: LLMConfig,
        *,
        schema_role: SchemaRole = SchemaRole.OWNER,
    ) -> str:
        """Build a redacted multi-line snapshot of engine, schema scope, DB, and LLM settings."""
        lines: list[str] = []
        lines.append(f"Engine:          {runtime.engine}")
        lines.append(f"Artifacts dir:   {os.path.abspath(runtime.artifacts_dir)}")
        ctx = runtime.engine_context
        deny_cols = sorted(ctx.deny_columns)
        deny_objs = sorted(ctx.deny_objects)
        if schema_role == SchemaRole.CONSUMER:
            lines.append(f"Schema context:  deny_columns={len(deny_cols)} deny_objects={len(deny_objs)}")
        else:
            lines.append(f"Schema context:  deny_columns={deny_cols!r} deny_objects={deny_objs!r}")
        runtime_cls = cast(type[EngineRuntimeConfig], DialectRegistry.get_runtime_config_class(runtime.engine))
        try:
            runtime_cfg = active_engine_runtime_config()
        except RuntimeError:
            runtime_cfg = runtime_cls()
        fields = runtime_cfg.connection_slug_fields()
        redacted = runtime_cls.redacted_fields()
        lines.append(f"{runtime.engine}:")
        for key, value in fields.items():
            display = MainInteractiveOps._redact_display_value(key, value) if key in redacted else value
            lines.append(f"  {key}: {display}")
        lines.append("LLM:")
        lines.append(f"  provider:   {llm.provider}")
        if llm.provider == "azure":
            base = EngineConfig.azure_base_url() or ""
            lines.append(f"  base_url:   {base}")
            lines.append(
                f"  api_key:    {MainInteractiveOps._redact_display_value('api_key', EngineConfig.AZURE_API_TOKEN)}"
            )
        else:
            lines.append(f"  base_url:   {EngineConfig.OPENAI_BASE_URL or ''}")
            lines.append(f"  api_key:    {MainInteractiveOps._redact_display_value('api_key', EngineConfig.API_TOKEN)}")
        return "\n".join(lines)

    @staticmethod
    def qsim_run_once(
        num_intents: int | None = None,
        num_questions: int | None = None,
        seed: int | None = None,
        artifacts_dir: str | None = None,
        schema: SchemaGraph | None = None,
        *,
        federation_manifest: FederationManifest | None = None,
        federation_mappings: FederationMappings | None = None,
    ) -> None:
        """Run full QSim (intents, values, NL questions) and write versioned question text plus summary."""
        if num_intents is None:
            num_intents = QSimConfig.INTENT_TYPES
        if num_questions is None:
            num_questions = QSimConfig.QUESTIONS_COUNT
        if seed is None:
            seed = QSimConfig.RANDOM_SEED

        random.seed(seed)

        debug(f"Starting question simulation: {num_intents} intent types, {num_questions} questions, seed={seed}")

        if schema is None:
            raise RuntimeError("Schema must be provided to qsim_run_once")

        has_profiled_columns = any(
            col.role is not None for table in schema.tables.values() for col in table.columns.values()
        )
        if not has_profiled_columns:
            raise RuntimeError(
                "Schema profiling failed - no column roles found. Check database connection and column data."
            )

        total_cols = sum(len(t.columns) for t in schema.tables.values())
        debug(f"  Loaded {len(schema.tables)} tables, {total_cols} columns with metadata")

        column_roles: dict[str, str] = {}
        for table_name, table_meta in schema.tables.items():
            for col_name, col_meta in table_meta.columns.items():
                if col_meta.role:
                    column_roles[f"{table_name}.{col_name}"] = col_meta.role

        base_dir = artifacts_dir or "."
        os.makedirs(base_dir, exist_ok=True)
        version = MainInteractiveOps._get_next_qsim_version(base_dir)
        qsim_trace_path = os.path.join(base_dir, f"qsim_trace_v{version}.json")
        qsim_trace_rows_path = os.path.join(base_dir, f"qsim_trace_rows_v{version}.jsonl")
        intent_trace_rows: list[dict[str, Any]] = []
        instantiation_trace_rows: list[dict[str, Any]] = []
        question_trace_rows: list[dict[str, Any]] = []
        intent_trace_summary: dict[str, Any] = {}
        instantiation_trace_summary: dict[str, Any] = {}
        question_trace_summary: dict[str, Any] = {}

        debug("Generating QSimIntent structures...")
        intents = generate_all_intents(
            schema,
            column_roles,
            num_intents,
            rng_seed=seed,
            trace_rows=intent_trace_rows,
            trace_summary=intent_trace_summary,
        )
        if federation_manifest is not None:
            intents = [
                intent
                for intent in intents
                if qsim_intent_eligible_on_federation(
                    intent.tables or [], schema, federation_manifest, federation_mappings
                )
            ]
        debug(f"  Generated {len(intents)} QSimIntent structures")

        debug("Instantiating QSimIntents with values...")
        instantiated = instantiate_all(
            intents,
            schema,
            num_questions,
            rng_seed=seed,
            trace_rows=instantiation_trace_rows,
            trace_summary=instantiation_trace_summary,
        )
        debug(f"  Created {len(instantiated)} QSimIntent variants with values")

        debug("Generating NL questions via LLM...")
        results = generate_all_questions(
            instantiated, schema, trace_rows=question_trace_rows, trace_summary=question_trace_summary
        )
        debug(f"  Generated {len(results)} QSimIntents with questions")

        parent_ids = [
            (intent.intent_id.rsplit("_v", 1)[0] if "_v" in intent.intent_id else intent.intent_id)
            for intent in results
        ]
        intent_counts = Counter(parent_ids)
        debug(f"  Questions per intent type: {dict(intent_counts)}")

        qname = QSIM_QUESTIONS_PATTERN.format(version=version)
        qsim_questions_path = os.path.join(base_dir, qname)
        qsim_dir = os.path.join(base_dir, "qsim")
        os.makedirs(qsim_dir, exist_ok=True)
        run_id = str(version)
        qsim_summary_path = os.path.join(qsim_dir, f"summary_{run_id}.json")
        qsim_index_path = os.path.join(qsim_dir, "index.jsonl")

        debug(f"Saving QSim questions to {qsim_questions_path}...")
        with open(qsim_questions_path, "w", encoding="utf-8") as f:
            for i, qintent in enumerate(results, 1):
                f.write(f"{i}. {qintent.question}\n")

        summary_entry = QSimSummary(version=version, num_intents=len(intents), num_questions=len(results), seed=seed)
        MainSpaceOps.write_json_atomic(qsim_summary_path, summary_entry.to_dict())
        index_row = {
            "run_id": run_id,
            "version": version,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(qsim_index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n")

        table_arity_histogram = dict(Counter(len(intent.tables) for intent in results))
        where_histogram = dict(Counter(len(intent.where) for intent in results))
        having_histogram = dict(Counter(len(intent.having_param) for intent in results))
        grain_histogram = dict(Counter(str(intent.grain or "") for intent in results))
        qsim_trace_payload = {
            "version": version,
            "seed": seed,
            "requested_num_intents": num_intents,
            "requested_num_questions": num_questions,
            "generated_num_intents": len(intents),
            "generated_num_variants": len(instantiated),
            "generated_num_questions": len(results),
            "question_artifact": os.path.basename(qsim_questions_path),
            "summary_artifact": os.path.relpath(qsim_summary_path, base_dir).replace(os.sep, "/"),
            "stages": {
                "intent_generation": intent_trace_summary,
                "instantiation": instantiation_trace_summary,
                "question_generation": question_trace_summary,
            },
            "coverage": {
                "table_arity_histogram": table_arity_histogram,
                "where_histogram": where_histogram,
                "having_histogram": having_histogram,
                "grain_histogram": grain_histogram,
            },
        }
        MainSpaceOps.write_json_atomic(qsim_trace_path, qsim_trace_payload)
        MainSpaceOps.write_jsonl_atomic(
            qsim_trace_rows_path, intent_trace_rows + instantiation_trace_rows + question_trace_rows
        )

        debug(f"Question simulation complete: {len(results)} questions saved")
        notify(f"QSim version: {version}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

        if results and diagnostic_debug_enabled():
            debug("[main_execution.qsim_run_once] samples:")
            for i, item in enumerate(results[:5]):
                debug(f"[main_execution.qsim_run_once]   {i + 1}. {item.question}")

        notify(
            MainInteractiveOps.format_qsim_summary_line(summary_entry),
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        return None

    @staticmethod
    def _load_questions_from_qsim_txt(path: str) -> list[str]:
        """Load numbered natural-language questions from a QSim ``.txt`` artifact."""
        questions: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dot = line.find(". ")
                if dot >= 0 and line[:dot].isdigit():
                    questions.append(line[dot + 2 :].strip())
                else:
                    questions.append(line)
        return questions

    @staticmethod
    def _get_questions_only(questions: list[str], *, output_path: str) -> None:
        """Print and save a numbered list of natural-language questions."""
        for i, q in enumerate(questions, 1):
            notify(f"{i}. {q}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

        with open(output_path, "w", encoding="utf-8") as f:
            for i, q in enumerate(questions, 1):
                f.write(f"{i}. {q}\n")

    @staticmethod
    def print_questions_bundle(version: int, artifacts_dir: str) -> None:
        """Load QSim questions for a version, print them, and mirror lines to ``qsim_v{version}_questions.txt`` in the working directory."""
        path = MainSpaceOps.resolve_qsim_path(version, artifacts_dir)
        questions = MainInteractiveOps._load_questions_from_qsim_txt(path)
        ver = int(version)
        out_path = os.path.join(artifacts_dir, f"qsim_v{ver}_questions.txt")
        MainInteractiveOps._get_questions_only(questions, output_path=out_path)

    @staticmethod
    def seed_warmup_run_once(
        schema: SchemaGraph,
        dialect: Any,
        seed_filepath: str,
        output_dir: str,
        store: dict[str, Any] | TemplateStoreView | None = None,
        templates: dict[str, Template] | None = None,
        interactive_gold: bool = True,
        seed: int | None = None,
        *,
        abort_on_gold_failure: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
        federation_manifest: Any | None = None,
        federation_mappings: Any | None = None,
        stores_by_source: dict[str, Any] | None = None,
        dialects_by_source: Mapping[str, Any] | None = None,
        source_runtimes: Mapping[str, Any] | None = None,
        member_graphs: Mapping[str, SchemaGraph] | None = None,
        federation_dir: str | None = None,
    ) -> None:
        """Execute the seed warmup pipeline: gold build, expansion, execute, stratified sampling, NL LLM, and template writes."""
        if seed is None:
            seed = SeedWarmupConfig.RANDOM_SEED
        random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)
        version = SeedWarmupCacheSession.get_next_seed_warmup_version(output_dir)
        report_name = SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version)
        report_filepath = os.path.join(output_dir, report_name)

        debug(f"Starting seed warmup run version {version}")
        notify(f"Seed warmup run version: {version}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

        schema_stats = schema.ensure_schema_stats()
        limits = compute_schema_limits(schema_stats)
        debug(
            f"Computed SchemaLimits: max_where_predicates={limits.max_where_predicates}, "
            f"max_groupby={limits.max_groupby}, "
            f"max_tables={limits.max_tables}"
        )

        debug(f"[{WARMUP_PHASE_B}] Gold build: seed normalization and gold intent generation")
        gold_intents_raw, gold_funnel, failure_trace_body, seed_norm_bundle = (
            SeedWarmupCacheSession.run_gold_intent_generation(
                schema, seed_filepath, interactive=interactive_gold, seed_warmup_version=version
            )
        )
        gold_warmup_intents = [SeedWarmupIntent.from_dict(d) if isinstance(d, dict) else d for d in gold_intents_raw]
        for row in gold_warmup_intents:
            row.source = "gold"
        debug(f"Gold intents: {len(gold_warmup_intents)}")
        seed_questions_loaded = int(gold_funnel.get("seed_questions_loaded", 0))
        gold_failed_count = int(gold_funnel.get("gold_failed", 0))
        gold_user_rejected_count = int(gold_funnel.get("gold_user_rejected", 0))
        notify(
            f"{WARMUP_PHASE_B} complete: seed normalization and gold intent generation "
            f"(seed_questions={seed_questions_loaded}, "
            f"gold_intents={len(gold_warmup_intents)}, "
            f"parse_failed={gold_failed_count}, "
            f"user_rejected={gold_user_rejected_count}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        if abort_on_gold_failure and (
            gold_failed_count > 0 or len(gold_warmup_intents) < seed_questions_loaded or gold_user_rejected_count > 0
        ):
            raise SystemExit(
                "Seed warmup aborted after ingest: "
                f"gold_intents={len(gold_warmup_intents)}, "
                f"seed_questions={seed_questions_loaded}, "
                f"parse_failed={gold_failed_count}, "
                f"user_rejected={gold_user_rejected_count}"
            )

        debug(f"[{WARMUP_PHASE_C}] Expansion: deterministic multi-depth expand_gold_intents")
        expanded_only = expand_gold_intents(gold_warmup_intents, schema, limits, pool_key=output_dir)
        full_pool: list[SeedWarmupIntent] = list(gold_warmup_intents) + expanded_only
        pool_body_tier: set[tuple[str, str]] = set()
        deduped_pool: list[SeedWarmupIntent] = []
        for pool_intent in full_pool:
            bk = body_similarity_key(pool_intent.to_runtime_intent())
            tier = pool_intent.complexity_tier().value
            key = (bk, tier)
            if key in pool_body_tier:
                continue
            pool_body_tier.add(key)
            deduped_pool.append(pool_intent)

        debug(f"[{WARMUP_PHASE_D}] Pool union and body dedupe (body_key,tier): {len(deduped_pool)} unique rows")

        tmpl_map: dict[str, Template] = templates if templates is not None else {}
        blocked_gold_rows = [
            row
            for row in deduped_pool
            if (row.source or "gold") == "gold"
            and MainInteractiveOps._gold_intent_store_union_widen_blocks_warmup(row, tmpl_map)
        ]
        gold_warmup_blocked_union_template_widen = len(blocked_gold_rows)
        warmup_queue = [
            row
            for row in deduped_pool
            if not MainInteractiveOps._gold_intent_store_union_widen_blocks_warmup(row, tmpl_map)
        ]
        debug(
            f"[{WARMUP_PHASE_D}] Gold vs store classification: gold_warmup_blocked_union_template_widen={gold_warmup_blocked_union_template_widen}; "
            f"queue {len(warmup_queue)} (expanded children keep distinct (body_key,tier))"
        )
        debug(f"[{WARMUP_PHASE_G}] Synthetic rows filtered by template_instance_key / ledger inside execute loop")

        notify(
            "Expansion and deduplication complete: pool size vs store classification "
            f"(expanded_synthetics={len(expanded_only)}, "
            f"unique_body_tier_rows={len(deduped_pool)}, "
            f"blocked_union_template_widen={gold_warmup_blocked_union_template_widen}, "
            f"queued_for_warmup={len(warmup_queue)}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        debug("Join resolution (cached per table-set)")
        join_cache: dict[JoinCacheKey, JoinCacheEntry] = {}
        for gold in gold_warmup_intents:
            SeedWarmupCacheSession.resolve_joins_for_table_set(
                gold.tables or [], schema, gold.intent_id or "gold", join_cache
            )
        debug(f"Join cache seeded with {len(join_cache)} table-set entries")
        debug(f"[{WARMUP_PHASE_E}] Join cache seeded from gold table sets (reuse across pool)")
        notify(
            f"{WARMUP_PHASE_E} complete: join cache seeded (table_sets={len(join_cache)}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        with open(seed_filepath, "rb") as _seed_f:
            seed_content_sha256 = hashlib.sha256(_seed_f.read()).hexdigest()
        warmup_cache_session = SeedWarmupCacheSession.open_seed_warmup_cache_session(
            output_dir, schema, seed_content_sha256
        )
        debug(f"[{WARMUP_PHASE_F}] Seed warmup cache manifest aligned to schema_hash and seed_content_hash")

        debug(
            f"[{WARMUP_PHASE_G}] Execute and validate; stratified sampling after successes; "
            f"[{WARMUP_PHASE_I}] question LLM, realism, templates only on full run"
        )
        next_id = int(store.get("next_id", 1)) if store else 1
        join_intent_index = {row.intent_id: row for row in deduped_pool if getattr(row, "intent_id", None)}
        store_keys = SeedWarmupCacheSession.accepted_template_instance_keys(tmpl_map)
        results, new_templates, updated_next_id, warmup_funnel = SeedWarmupCacheSession.run_seed_warmup_execution(
            warmup_queue,
            schema,
            dialect,
            next_id,
            join_cache=join_cache,
            join_resolver_intent_index=join_intent_index,
            store_instance_keys=store_keys,
            accepted_templates=tmpl_map,
            warmup_cache=warmup_cache_session,
            warmup_report_version=version,
            warmup_lattice_root=output_dir,
            max_kept_intents=max_kept_intents,
            federation_manifest=federation_manifest,
            federation_mappings=federation_mappings,
            stores_by_source=stores_by_source,
            dialects_by_source=dialects_by_source,
            source_runtimes=source_runtimes,
            member_graphs=member_graphs,
            federation_dir=federation_dir,
        )
        SeedWarmupCacheSession.save_seed_warmup_cache_zip(
            output_dir,
            warmup_cache_session.manifest,
            warmup_cache_session.work_units,
            gold_intent_dicts=[g.to_dict() for g in gold_warmup_intents],
        )
        for blocked in blocked_gold_rows:
            blocked_result = SeedWarmupResult(blocked.to_runtime_intent(), "")
            blocked_result.failure_code = "blocked_by_store_union_template_widen"
            blocked_result.failure_stage = "gold_store_classification"
            blocked_result.drop_reason_category = "gold_store_classification"
            blocked_result.error = (
                "Gold intent skipped: existing store template covers it only via disallowed "
                "UNION_TEMPLATE_WIDEN / UNION_TEMPLATE_AND_RUNTIME_WIDEN paths."
            )
            results.append(blocked_result)
        debug(f"Seed warmup execution results: {len(results)} rows, templates: {len(new_templates)}")
        exec_validation_drop = int(warmup_funnel.get("validation_drop", 0))
        exec_realism_drop = int(warmup_funnel.get("realism_drop", 0))
        exec_question_gen_failed = int(warmup_funnel.get("question_generation_failed", 0))
        exec_early_failed = int(warmup_funnel.get("early_pipeline_failed", 0))
        exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
        exec_total = len(results)
        exec_success = sum(1 for r in results if r.success)
        notify(
            f"{WARMUP_PHASE_G} complete: per-intent SQL build, validation, execution, realism gate "
            f"(processed={exec_total}, "
            f"success={exec_success}, "
            f"validation_drop={exec_validation_drop}, "
            f"realism_drop={exec_realism_drop}, "
            f"question_gen_failed={exec_question_gen_failed}, "
            f"early_pipeline_failed={exec_early_failed}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        if federation_manifest is None and store is not None and templates is not None:
            store_any: Any = store
            template_store_view = store_any if isinstance(store_any, TemplateStoreView) else None
            writable_store: dict[str, Any] = store if isinstance(store, dict) else {}
            for tmpl in new_templates:
                TemplateRefs.merge_seed_warmup_templates_into_store(templates, [tmpl])
            template_store_view = store if isinstance(store, TemplateStoreView) else None
            writable_store = store if isinstance(store, dict) else {}
            reconcile_template_store_until_stable(templates, template_store_view=template_store_view)
            writable_store["next_id"] = updated_next_id
            saved_store = TemplateOps.templates_to_store(writable_store, templates)
            TemplateOps.save_template_store(saved_store)

        templates_added = (
            len(new_templates)
            if (federation_manifest is not None or (store is not None and templates is not None))
            else 0
        )

        exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
        run_mode = "full"
        registry_snapshot = {
            "run_mode": run_mode,
            "schema_hash": schema.effective_structural_hash,
            "seed_content_hash": seed_content_sha256,
            "policy_version": warmup_cache_session.manifest.get("policy_version"),
            "code_version": warmup_cache_session.manifest.get("code_version"),
            "template_store_size_at_start": len(tmpl_map),
            "template_store_next_id_at_start": next_id,
            "template_store_next_id_at_end": updated_next_id,
            "work_units_total": len(warmup_cache_session.work_units),
            "work_units_touched_this_run": len(warmup_cache_session.touched_work_unit_ids),
        }
        SeedWarmupCacheSession.save_seed_warmup_report(
            results,
            report_filepath,
            funnel={
                "seed_warmup_version": version,
                "registry_snapshot": registry_snapshot,
                **gold_funnel,
                "synthetic_unique_body_keys": len(deduped_pool),
                "synthetic_runnable_count": len(warmup_queue),
                **SeedWarmupCacheSession.warmup_pool_operator_feature_stats(warmup_queue),
                "gold_prompts_count": seed_questions_loaded,
                "templates_added": templates_added,
                "execute_ok_count": exec_ok_ct,
                **warmup_funnel,
                "gold_warmup_blocked_union_template_widen": gold_warmup_blocked_union_template_widen,
            },
        )

        norm_json: str | None = None
        norm_txt: str | None = None
        if seed_norm_bundle is not None:
            norm_json, norm_txt = seed_norm_bundle
        bundle_path = os.path.join(output_dir, SeedWarmupConfig.SEED_WARMUP_BUNDLE_PATTERN.format(version=version))
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if norm_json is not None:
                zf.writestr(SEED_NORMALIZATION_JSON, norm_json)
            if norm_txt is not None:
                zf.writestr(NORMALIZED_SEEDS_TXT, norm_txt)
            if failure_trace_body:
                zf.writestr("gold_intent_failures_trace.txt", failure_trace_body)

        notify(
            f"{WARMUP_PHASE_K} complete: seed warmup report and bundle zip"
            + (", template store updated" if store is not None and templates is not None else "")
            + f" (templates_added={templates_added}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        success_rate = round(success / total, 3) if total > 0 else 0.0

        debug(f"SEED WARMUP COMPLETE: {len(new_templates)} synthetic templates created")
        summary = SeedWarmupSummary(
            version=version,
            total=total,
            success=success,
            failed=failed,
            success_rate=success_rate,
            seed_questions_loaded=seed_questions_loaded,
            gold_intents_total=int(gold_funnel.get("gold_intents_total", len(gold_warmup_intents))),
            unique_prompts=len(warmup_queue),
            gold_new=int(gold_funnel.get("gold_new", 0)),
            gold_skipped=int(gold_funnel.get("gold_skipped", 0)),
            gold_failed=int(gold_funnel.get("gold_failed", 0)),
            gold_user_rejected=int(gold_funnel.get("gold_user_rejected", 0)),
            deduped_prompts_count=len(deduped_pool),
            gold_prompts_count=seed_questions_loaded,
            templates_added=templates_added,
            validation_drop=int(warmup_funnel.get("validation_drop", 0)),
            realism_drop=int(warmup_funnel.get("realism_drop", 0)),
            question_generation_failed=int(warmup_funnel.get("question_generation_failed", 0)),
            early_pipeline_failed=int(warmup_funnel.get("early_pipeline_failed", 0)),
        )
        notify(
            MainInteractiveOps.format_seed_warmup_summary(summary),
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        return None

    @staticmethod
    def _raw_db_connection_for_query_log(dialect: Any) -> Any:
        """Best-effort DBAPI handle for query-log probes."""
        eng = getattr(dialect, "engine", None)
        if eng is not None:
            try:
                return eng.raw_connection()
            except (OSError, AttributeError, TypeError, RuntimeError):
                pass
        return getattr(dialect, "connection", None)

    @staticmethod
    def _dialect_name_for_query_log(dialect: Any) -> str:
        """Return the dialect label used for query-log dispatch."""
        label = getattr(dialect, "dialect_label", None)
        if isinstance(label, str) and label.strip():
            return label.strip().lower()
        return str(getattr(dialect, "name", "postgresql")).strip().lower()

    @staticmethod
    def _run_seed_warmup_sql_history_pipeline(
        *,
        schema: SchemaGraph,
        dialect: Any,
        output_dir: str,
        store: dict[str, Any] | None,
        templates: dict[str, Template] | None,
        sql_texts: list[str],
        sql_history_content_hash: str,
        seed: int | None,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> None:
        """Execute seed warmup over converted SQL-history intents sharing cache keyed by *sql_history_content_hash*."""
        if seed is None:
            seed = SeedWarmupConfig.RANDOM_SEED
        random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)
        version = SeedWarmupCacheSession.get_next_seed_warmup_version(output_dir)
        report_name = SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version)
        report_filepath = os.path.join(output_dir, report_name)

        debug(f"Starting SQL-history seed warmup run version {version}")
        notify(
            f"SQL-history seed warmup run version: {version}",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        schema_stats = schema.ensure_schema_stats()
        limits = compute_schema_limits(schema_stats)
        debug(
            f"Computed SchemaLimits: max_where_predicates={limits.max_where_predicates}, "
            f"max_groupby={limits.max_groupby}, "
            f"max_tables={limits.max_tables}"
        )

        converted_pairs: list[tuple[str, Any]] = []
        fail_by_hash: dict[str, str] = {}
        for sql_line in sql_texts:
            cr = convert_sql_to_intent(sql_line, schema, dialect, verify_via_execute=True)
            if cr.intent is None:
                fc = cr.failure_code or "unknown"
                fail_by_hash[cr.sql_hash] = fc
                continue
            converted_pairs.append((sql_line, cr.intent))

        runtimes = [rt for _, rt in converted_pairs]
        deduped_rt = dedup_runtime_intents(runtimes)
        warmup_queue: list[SeedWarmupIntent] = []
        for idx, rt in enumerate(deduped_rt):
            bk = body_similarity_key(rt)
            warmup_queue.append(
                seed_warmup_intent_from_runtime_intent(rt, intent_id=f"sqlhist_{idx}_{bk[:24]}", seed_index=idx)
            )

        seed_questions_loaded = len(sql_texts)
        gold_warmup_intents = list(warmup_queue)
        deduped_pool = list(warmup_queue)
        gold_funnel = {
            "seed_questions_loaded": seed_questions_loaded,
            "gold_failed": len(fail_by_hash),
            "gold_new": 0,
            "gold_skipped": 0,
            "gold_user_rejected": 0,
            "gold_intents_total": len(warmup_queue),
        }
        gold_failed_count = int(gold_funnel.get("gold_failed", 0))
        notify(
            f"{WARMUP_PHASE_B} complete: SQL history conversion "
            f"(lines={seed_questions_loaded}, "
            f"converted_intents={len(warmup_queue)}, "
            f"conversion_failed={gold_failed_count}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        tmpl_map: dict[str, Template] = templates if templates is not None else {}
        gold_warmup_blocked_union_template_widen = 0
        if expand:
            debug(f"[{WARMUP_PHASE_C}] Expansion: deterministic multi-depth expand_gold_intents")
            expanded_only = expand_gold_intents(gold_warmup_intents, schema, limits, pool_key=output_dir)
            full_pool: list[SeedWarmupIntent] = list(gold_warmup_intents) + expanded_only
            pool_body_tier: set[tuple[str, str]] = set()
            deduped_pool = []
            for pool_intent in full_pool:
                bk = body_similarity_key(pool_intent.to_runtime_intent())
                tier = pool_intent.complexity_tier().value
                key = (bk, tier)
                if key in pool_body_tier:
                    continue
                pool_body_tier.add(key)
                deduped_pool.append(pool_intent)
            debug(f"[{WARMUP_PHASE_D}] Pool union and body dedupe (body_key,tier): {len(deduped_pool)} unique rows")
            blocked_gold_rows = [
                row
                for row in deduped_pool
                if (row.source or "gold") == "gold"
                and MainInteractiveOps._gold_intent_store_union_widen_blocks_warmup(row, tmpl_map)
            ]
            gold_warmup_blocked_union_template_widen = len(blocked_gold_rows)
            warmup_queue = [
                row
                for row in deduped_pool
                if not MainInteractiveOps._gold_intent_store_union_widen_blocks_warmup(row, tmpl_map)
            ]
            debug(
                f"[{WARMUP_PHASE_D}] Gold vs store classification: gold_warmup_blocked_union_template_widen={gold_warmup_blocked_union_template_widen}; "
                f"queue {len(warmup_queue)} (expanded children keep distinct (body_key,tier))"
            )
            notify(
                "Expansion and deduplication complete: pool size vs store classification "
                f"(expanded_synthetics={len(expanded_only)}, "
                f"unique_body_tier_rows={len(deduped_pool)}, "
                f"blocked_union_template_widen={gold_warmup_blocked_union_template_widen}, "
                f"queued_for_warmup={len(warmup_queue)}).",
                stage="cli",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                level="info",
            )

        join_cache: dict[JoinCacheKey, JoinCacheEntry] = {}
        join_seed_rows = gold_warmup_intents if expand else warmup_queue
        for row in join_seed_rows:
            SeedWarmupCacheSession.resolve_joins_for_table_set(
                row.tables or [], schema, row.intent_id or "sqlhist", join_cache
            )
        debug(f"Join cache seeded with {len(join_cache)} table-set entries")

        warmup_cache_session = SeedWarmupCacheSession.open_seed_warmup_cache_session(
            output_dir, schema, sql_history_content_sha256=sql_history_content_hash
        )
        debug(f"[{WARMUP_PHASE_F}] Seed warmup cache manifest aligned to schema_hash and sql_history_content_hash")

        next_id = int(store.get("next_id", 1)) if store else 1
        join_intent_index = {row.intent_id: row for row in deduped_pool if getattr(row, "intent_id", None)}
        store_keys = SeedWarmupCacheSession.accepted_template_instance_keys(tmpl_map)
        results, new_templates, updated_next_id, warmup_funnel = SeedWarmupCacheSession.run_seed_warmup_execution(
            warmup_queue,
            schema,
            dialect,
            next_id,
            join_cache=join_cache,
            join_resolver_intent_index=join_intent_index,
            store_instance_keys=store_keys,
            accepted_templates=tmpl_map,
            warmup_cache=warmup_cache_session,
            warmup_report_version=version,
            warmup_lattice_root=output_dir,
            max_kept_intents=max_kept_intents,
        )
        SeedWarmupCacheSession.save_seed_warmup_cache_zip(
            output_dir,
            warmup_cache_session.manifest,
            warmup_cache_session.work_units,
            gold_intent_dicts=[g.to_dict() for g in gold_warmup_intents],
        )

        debug(f"SQL-history seed warmup execution results: {len(results)} rows, templates: {len(new_templates)}")
        exec_validation_drop = int(warmup_funnel.get("validation_drop", 0))
        exec_realism_drop = int(warmup_funnel.get("realism_drop", 0))
        exec_question_gen_failed = int(warmup_funnel.get("question_generation_failed", 0))
        exec_early_failed = int(warmup_funnel.get("early_pipeline_failed", 0))
        exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
        exec_total = len(results)
        exec_success = sum(1 for r in results if r.success)
        notify(
            f"{WARMUP_PHASE_G} complete: per-intent SQL build, validation, execution "
            f"(processed={exec_total}, "
            f"success={exec_success}, "
            f"validation_drop={exec_validation_drop}, "
            f"realism_drop={exec_realism_drop}, "
            f"question_gen_failed={exec_question_gen_failed}, "
            f"early_pipeline_failed={exec_early_failed}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        if store is not None and templates is not None:
            store_any: Any = store
            template_store_view = store_any if isinstance(store_any, TemplateStoreView) else None
            writable_store: dict[str, Any] = store if isinstance(store, dict) else {}
            for tmpl in new_templates:
                TemplateRefs.merge_seed_warmup_templates_into_store(templates, [tmpl])
            reconcile_template_store_until_stable(templates, template_store_view=template_store_view)
            writable_store["next_id"] = updated_next_id
            saved_store = TemplateOps.templates_to_store(writable_store, templates)
            TemplateOps.save_template_store(saved_store)

        templates_added = len(new_templates) if store is not None and templates is not None else 0

        exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
        run_mode = "full"
        registry_snapshot = {
            "run_mode": run_mode,
            "schema_hash": schema.effective_structural_hash,
            "sql_history_content_hash": sql_history_content_hash,
            "seed_content_hash": warmup_cache_session.manifest.get("seed_content_hash"),
            "policy_version": warmup_cache_session.manifest.get("policy_version"),
            "code_version": warmup_cache_session.manifest.get("code_version"),
            "template_store_size_at_start": len(tmpl_map),
            "template_store_next_id_at_start": next_id,
            "template_store_next_id_at_end": updated_next_id,
            "work_units_total": len(warmup_cache_session.work_units),
            "work_units_touched_this_run": len(warmup_cache_session.touched_work_unit_ids),
        }
        SeedWarmupCacheSession.save_seed_warmup_report(
            results,
            report_filepath,
            funnel={
                "seed_warmup_version": version,
                "registry_snapshot": registry_snapshot,
                **gold_funnel,
                "sql_history_conversion_failures": len(fail_by_hash),
                "synthetic_unique_body_keys": len(deduped_pool),
                "synthetic_runnable_count": len(warmup_queue),
                **SeedWarmupCacheSession.warmup_pool_operator_feature_stats(warmup_queue),
                "gold_prompts_count": seed_questions_loaded,
                "templates_added": templates_added,
                "execute_ok_count": exec_ok_ct,
                **warmup_funnel,
                "gold_warmup_blocked_union_template_widen": gold_warmup_blocked_union_template_widen,
            },
        )

        notify(
            f"{WARMUP_PHASE_K} complete: seed warmup report"
            + (", template store updated" if store is not None and templates is not None else "")
            + f" (templates_added={templates_added}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        success_rate = round(success / total, 3) if total > 0 else 0.0

        debug(f"SQL-HISTORY SEED WARMUP COMPLETE: {len(new_templates)} synthetic templates created")
        summary = SeedWarmupSummary(
            version=version,
            total=total,
            success=success,
            failed=failed,
            success_rate=success_rate,
            seed_questions_loaded=seed_questions_loaded,
            gold_intents_total=int(gold_funnel.get("gold_intents_total", len(gold_warmup_intents))),
            unique_prompts=len(warmup_queue),
            gold_new=int(gold_funnel.get("gold_new", 0)),
            gold_skipped=int(gold_funnel.get("gold_skipped", 0)),
            gold_failed=int(gold_funnel.get("gold_failed", 0)),
            gold_user_rejected=int(gold_funnel.get("gold_user_rejected", 0)),
            deduped_prompts_count=len(deduped_pool),
            gold_prompts_count=seed_questions_loaded,
            templates_added=templates_added,
            validation_drop=int(warmup_funnel.get("validation_drop", 0)),
            realism_drop=int(warmup_funnel.get("realism_drop", 0)),
            question_generation_failed=int(warmup_funnel.get("question_generation_failed", 0)),
            early_pipeline_failed=int(warmup_funnel.get("early_pipeline_failed", 0)),
        )
        notify(
            MainInteractiveOps.format_seed_warmup_summary(summary),
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

    @staticmethod
    def run_seed_warmup_from_history_execution(
        self_engine: Any,
        sql_history_filepath: str,
        *,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
        seed: int | None = None,
    ) -> None:
        """Drive :meth:`SeedWarmupCacheSession.run_seed_warmup_execution` from a newline-oriented SQL history file."""
        assert_federation_sql_history_warmup_allowed(self_engine)
        schema = self_engine._schema_graph
        dialect = self_engine._dialect
        output_dir = str(self_engine._artifacts_dir)
        store = self_engine._store
        templates = self_engine._templates
        statements = load_sql_history_statements(sql_history_filepath)
        content_hash = compute_sql_history_content_hash(statements)
        MainInteractiveOps._run_seed_warmup_sql_history_pipeline(
            schema=schema,
            dialect=dialect,
            output_dir=output_dir,
            store=store,
            templates=templates,
            sql_texts=statements,
            sql_history_content_hash=content_hash,
            seed=seed,
            expand=expand,
            max_kept_intents=max_kept_intents,
        )

    @staticmethod
    def run_seed_warmup_from_query_log_execution(
        self_engine: Any,
        *,
        lookback_days: int = 730,
        max_queries: int = 5000,
        min_runs: int = 1,
        user_filter: str | None = None,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
        seed: int | None = None,
    ) -> None:
        """Drive :meth:`SeedWarmupCacheSession.run_seed_warmup_execution` from the engine query log."""
        assert_query_log_warmup_allowed(self_engine)
        schema = self_engine._schema_graph
        dialect = self_engine._dialect
        output_dir = str(self_engine._artifacts_dir)
        store = self_engine._store
        templates = self_engine._templates
        conn = MainInteractiveOps._raw_db_connection_for_query_log(dialect)
        dialect_name = MainInteractiveOps._dialect_name_for_query_log(dialect)
        try:
            sql_texts = fetch_query_log(
                dialect_name,
                conn,
                lookback_days=lookback_days,
                max_queries=max_queries,
                min_runs=min_runs,
                user_filter=user_filter,
            )
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                close()
        content_hash = compute_sql_history_content_hash(sql_texts)
        MainInteractiveOps._run_seed_warmup_sql_history_pipeline(
            schema=schema,
            dialect=dialect,
            output_dir=output_dir,
            store=store,
            templates=templates,
            sql_texts=sql_texts,
            sql_history_content_hash=content_hash,
            seed=seed,
            expand=expand,
            max_kept_intents=max_kept_intents,
        )

    @staticmethod
    def _suspend_preview_rows(rows: Sequence[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
        return tuple(tuple(r) for r in rows[:10])

    @staticmethod
    def _freeze_sql_parameters(intent: Any) -> tuple[tuple[str, Any], ...]:
        return tuple((str(k), v) for k, v in flatten_param_values(intent).items())

    @staticmethod
    def _suspend_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _sql_feedback_suspend_context(
        snap_post: InteractiveTailSnapshot,
        sql: str,
        rows: list[tuple[Any, ...]],
        tmpl_sd: dict[str, Any] | None,
        gen_out: SqlGenerationOutcome,
        matched_rejected_template: Any,
        force_feedback: bool,
        execution_intent: Any,
        *,
        federated_prepare: FederatedPrepareOutcome | None = None,
        federated_bundle: FederatedSqlBundle | None = None,
    ) -> SqlFeedbackSuspendContext:
        """Build a frozen payload for deferred SQL accept/reject."""
        return SqlFeedbackSuspendContext(
            tail=snap_post,
            execution_intent=execution_intent,
            sql=sql,
            preview_rows=MainInteractiveOps._suspend_preview_rows(rows),
            sql_parameters=MainInteractiveOps._freeze_sql_parameters(execution_intent),
            suspended_at=MainInteractiveOps._suspend_now(),
            tmpl_sd=tmpl_sd,
            gen_out=gen_out,
            matched_rejected_template=matched_rejected_template,
            force_feedback=force_feedback,
            federated_prepare=federated_prepare,
            federated_bundle=federated_bundle,
        )

    @staticmethod
    def _federation_execute_confirm_prompt(
        gen_out: SqlGenerationOutcome, fed_prep: FederatedPrepareOutcome | None, manifest: FederationManifest | None
    ) -> str:
        """Return execute-gate wording that states how many member databases a federated plan spans."""
        if gen_out.generation_path is GenerationPath.FEDERATION_PLAN and fed_prep is not None and manifest is not None:
            source_ids = {step.source_id for step in (fed_prep.plan.steps or ())}
            count = len(source_ids)
            if count > 1:
                return f"Execute federated plan across {count} database(s)?"
            if count == 1:
                return "Execute federated plan on one member database?"
        return "Execute this SQL?"

    @staticmethod
    def snapshot_turn_policy() -> TurnPolicySnapshot:
        """Freeze per-turn policy knobs at suspend for federation and execute resume."""
        return TurnPolicySnapshot(
            max_compose_repairs=PolicyConfig.MAX_ASK_COMPOSE_REPAIRS,
            max_interpret_ground_retries=PolicyConfig.MAX_ASK_INTERPRET_GROUND_RETRIES,
            trust_auto_accept_threshold=TRUST_AUTO_ACCEPT_THRESHOLD,
        )

    @staticmethod
    def _sql_execute_suspend_context(
        snap_post: InteractiveTailSnapshot,
        sql: str,
        tmpl_sd: dict[str, Any] | None,
        gen_out: SqlGenerationOutcome,
        matched_rejected_template: Any,
        force_feedback: bool,
        execution_intent: Any,
        rows: tuple[tuple[Any, ...], ...] = (),
        *,
        federated_prepare: FederatedPrepareOutcome | None = None,
        federation_plan_id: str = "",
        federation_exec_context: Mapping[str, Any] | None = None,
        turn_policy: TurnPolicySnapshot | None = None,
    ) -> SqlExecuteSuspendContext:
        """Build a frozen payload for the separated execute step."""
        exec_ctx_pairs: tuple[tuple[str, Any], ...] = ()
        if isinstance(federation_exec_context, Mapping):
            exec_ctx_pairs = tuple((str(k), v) for k, v in federation_exec_context.items())
        return SqlExecuteSuspendContext(
            tail=snap_post,
            execution_intent=execution_intent,
            sql=sql,
            gen_out=gen_out,
            matched_rejected_template=matched_rejected_template,
            force_feedback=force_feedback,
            tmpl_sd=tmpl_sd,
            preview_rows=MainInteractiveOps._suspend_preview_rows(rows),
            sql_parameters=MainInteractiveOps._freeze_sql_parameters(execution_intent),
            suspended_at=MainInteractiveOps._suspend_now(),
            federated_prepare=federated_prepare,
            federation_plan_id=str(federation_plan_id or gen_out.federation_plan_id or ""),
            federation_exec_context=exec_ctx_pairs,
            turn_policy=turn_policy if turn_policy is not None else MainInteractiveOps.snapshot_turn_policy(),
        )

    @staticmethod
    def _federation_exec_context_from_pairs(
        pairs: tuple[tuple[str, Any], ...] | Sequence[tuple[str, Any]] | None,
    ) -> dict[str, Any]:
        if not pairs:
            return {}
        return {str(k): v for k, v in pairs}

    @staticmethod
    def _handle_federation_turn_cancelled_interactive(
        choice_port: InteractiveChoicePort | None,
        owner: Any | None,
        exc: FederationTurnCancelledError,
    ) -> None:
        """Record a structured federation cancellation terminal outcome."""
        notify(
            "Federated turn cancelled during execution.",
            stage="execution",
            code=DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
            level="error",
            source_id=exc.source_id,
            details=(
                ("source_id", exc.source_id),
                ("phase", exc.phase),
                ("succeeded", ",".join(s[0] for s in exc.succeeded)),
                ("message", str(exc)),
            ),
        )
        clear_federated_turn_state(choice_port)
        print_rephrase_hint(RephraseHint.FEDERATION_TURN_CANCELLED)
        note_interactive_turn(
            choice_port,
            outcome="federation_turn_cancelled",
            error=None,
            federation_source_id=exc.source_id,
            federation_phase=exc.phase,
            federation_succeeded=exc.succeeded,
            failure_kind=FailureCategory.FEDERATION_TURN_CANCELLED.value,
        )

    @staticmethod
    def _handle_federation_partial_failure_interactive(
        choice_port: InteractiveChoicePort | None,
        owner: Any | None,
        exc: FederationPartialFailureError,
    ) -> None:
        """Record a structured federation partial-failure terminal outcome without persisting turn artifacts."""
        notify(
            "Federation partial failure during execution.",
            stage="execution",
            code=DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
            level="error",
            source_id=exc.source_id,
            details=(
                ("source_id", exc.source_id),
                ("phase", exc.phase),
                ("succeeded", ",".join(s[0] for s in exc.succeeded)),
                ("message", str(exc)),
            ),
        )
        clear_federated_turn_state(choice_port)
        print_rephrase_hint(RephraseHint.FEDERATION_PARTIAL_FAILURE)
        note_interactive_turn(
            choice_port,
            outcome="federation_partial_failure",
            error=None,
            federation_source_id=exc.source_id,
            federation_phase=exc.phase,
            federation_succeeded=exc.succeeded,
            failure_kind=FailureCategory.EXECUTION_OTHER_ERROR.value,
            retryable=isinstance(exc, RetryableError),
        )

    @staticmethod
    def _verify_federation_execute_resume(ctx: SqlExecuteSuspendContext) -> None:
        """Ensure the federated plan approved at suspend matches the resume payload."""
        expected = str(ctx.federation_plan_id or ctx.gen_out.federation_plan_id or "")
        actual = str(ctx.gen_out.federation_plan_id or "")
        if expected and actual and expected != actual:
            raise FederationInvariantError(
                f"federation plan id mismatch on execute resume: expected {expected!r}, got {actual!r}"
            )

    @staticmethod
    def _federation_failure_attribution(fed_prep: FederatedPrepareOutcome | None) -> dict[str, Any]:
        """Extract federation source/phase/failure_kind for session terminal outcomes."""
        if fed_prep is None:
            return {
                "federation_source_id": None,
                "federation_phase": None,
                "failure_kind": None,
            }
        return {
            "federation_source_id": str(fed_prep.source_id or "") or None,
            "federation_phase": str(fed_prep.phase) or None,
            "failure_kind": str(fed_prep.error_kind or "") or None,
        }

    @staticmethod
    def session_step_federation_fields_from_snap(snap: Mapping[str, Any], raw_outcome: str) -> dict[str, Any]:
        """Copy federation attribution from a stored turn outcome onto SessionStep fields."""
        del raw_outcome
        return {}

    @staticmethod
    def _run_sql_execution_for_gen_out(
        *,
        intent: Any,
        exec_schema: SchemaGraph,
        exec_dialect: Any,
        tmpl_sd: dict[str, Any] | None,
        gen_out: SqlGenerationOutcome,
        owner: Any | None,
        choice_port: InteractiveChoicePort | None,
        q_norm: str = "",
        join_candidates: dict[str, Any] | None = None,
        cmap: dict[str, list[str]] | None = None,
        store: dict[str, Any] | TemplateStoreView | None = None,
        federated_prepare: FederatedPrepareOutcome | None = None,
        federation_exec_context: Mapping[str, Any] | None = None,
    ) -> tuple[list[tuple[Any, ...]], FederatedSqlBundle | None]:
        """Execute SQL for the current generation outcome, including federated coordinator paths."""
        fed_prep = federated_prepare
        federated_bundle: FederatedSqlBundle | None = None
        if (
            getattr(gen_out, "generation_path", None) is GenerationPath.FEDERATION_PLAN
            and isinstance(fed_prep, FederatedPrepareOutcome)
            and fed_prep.success
        ):
            fed_manifest = (
                getattr(owner, "_federation_manifest", None)
                if owner is not None and owner_is_aether_federation(owner)
                else None
            )
            if not isinstance(fed_manifest, FederationManifest):
                fed_manifest = None
            row_cap = fed_manifest.coordinator.row_cap if fed_manifest is not None else None
            gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
            exec_ctx = dict(federation_exec_context or {})
            progress("Executing federated SQL...")
            turn_session = choice_port if isinstance(choice_port, PipelineSessionMarker) else None
            exec_outcome = execute_federated_prepare(
                fed_prep,
                exec_schema,
                dialect=exec_dialect,
                dialects_by_source=getattr(owner, "_federation_dialects", None) if owner is not None else None,
                source_runtimes=getattr(owner, "_federation_source_runtimes", None) if owner is not None else None,
                coordinator_row_cap=row_cap,
                manifest=fed_manifest,
                q_norm=str(exec_ctx.get("q_norm") or q_norm),
                join_candidates=exec_ctx.get("join_candidates", join_candidates),
                cmap=exec_ctx.get("cmap", cmap),
                store=store,
                gate_kwargs_by_source=(
                    MainSpaceOps.federation_gate_kwargs_by_source(
                        owner, choice_port, fed_manifest, getattr(owner, "_federation_dialects", None)
                    )
                    if owner is not None and fed_manifest is not None
                    else None
                ),
                federation_dir=getattr(owner, "_federation_storage_dir", None) if owner is not None else None,
                member_graphs=(
                    getattr(owner, "_federation_member_graphs", None)
                    if owner is not None and isinstance(getattr(owner, "_federation_member_graphs", None), dict)
                    else None
                ),
                turn_session=turn_session,
                **gate_kwargs,
            )
            if (
                owner is not None
                and MainInteractiveOps.persist_template_learning_for_pipeline_session(choice_port)
                and isinstance(fed_manifest, FederationManifest)
            ):
                member_graphs = getattr(owner, "_federation_member_graphs", None)
                if isinstance(member_graphs, dict) and member_graphs:
                    member_stores = MainSpaceOps.federation_stores_by_source(
                        owner,
                        member_graphs,
                        space_name=MainSpaceOps.session_space_name_for_federation(owner, choice_port),
                    )
                    if member_stores:
                        persist_federated_member_stores(
                            fed_prep.plan,
                            store=store or {},
                            stores_by_source=member_stores,
                        )
            federated_bundle = exec_outcome.bundle
            return list(exec_outcome.rows), federated_bundle
        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        return MainInteractiveOps._run_pipeline_sql_rows(
            intent=intent,
            schema=exec_schema,
            dialect=exec_dialect,
            tmpl_sd=tmpl_sd,
            gate_kwargs=gate_kwargs,
        ), None

    @staticmethod
    def _run_pipeline_sql_rows(
        *,
        intent: Any,
        schema: SchemaGraph,
        dialect: Any,
        tmpl_sd: dict[str, Any] | None,
        gate_kwargs: Mapping[str, Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        """Finalize and execute pipeline SQL, returning row tuples."""
        exec_params = dict(flatten_param_values(intent))
        exec_sql = dialect.finalize_render(
            intent.sql_param or "",
            exec_params,
            schema=schema,
            intent=intent,
            execution_sql_override=None,
            structural_defaults=tmpl_sd,
        )
        progress("Executing SQL...")
        gk = dict(gate_kwargs or {})
        return list(
            execute_guarded_sql(
                dialect,
                exec_sql,
                exec_params,
                schema=schema,
                intent=intent,
                schema_role=str(gk.get("schema_role", "owner") or "owner"),
                schema_context=gk.get("schema_context"),
                visible_objects=gk.get("visible_objects"),
                context_name=str(gk.get("context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME),
            )
        )

    @staticmethod
    def try_zero_row_where_remediation(
        intent: Any,
        schema: SchemaGraph,
        dialect: Any,
        tmpl_sd: dict[str, Any] | None,
        *,
        gate_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, list[tuple[Any, ...]] | None]:
        """Attempt filter literal auto-fix after a zero-row execute using cached distinct values."""
        if not PolicyConfig.ZERO_ROW_WHERE_AUTO_FIX_ENABLED:
            return intent, None
        for where_param, literal, column, cached in enumerate_zero_row_equality_where(intent, schema):
            for candidate in zero_row_where_remediation_candidates(literal, cached):
                trial_intent = patch_where_literal_on_intent(intent, where_param, candidate)
                try:
                    trial_rows = MainInteractiveOps._run_pipeline_sql_rows(
                        intent=trial_intent,
                        schema=schema,
                        dialect=dialect,
                        tmpl_sd=tmpl_sd,
                        gate_kwargs=gate_kwargs,
                    )
                except AccessError:
                    continue
                if len(trial_rows) > 0:
                    notify(
                        f"Adjusted filter {column} from {literal!r} to {candidate!r}.",
                        stage="execution",
                        code=DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED,
                    )
                    return trial_intent, trial_rows
        return intent, None

    @staticmethod
    def _offer_sql_feedback_after_execute(
        *,
        q_norm: str,
        intent: Any,
        sql: str,
        rows: list[tuple[Any, ...]],
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Any],
        rejected: dict[str, Any],
        dialect: Any,
        choice_port: InteractiveChoicePort | None,
        snap_post: InteractiveTailSnapshot,
        tmpl_sd: dict[str, Any] | None,
        gen_out: SqlGenerationOutcome,
        matched_rejected_template: Any,
        force_feedback: bool,
        persist_template_learning: bool = True,
        owner: Any | None = None,
        federated_prepare: FederatedPrepareOutcome | None = None,
        federated_bundle: FederatedSqlBundle | None = None,
    ) -> None:
        """Collect SQL feedback after execution (shared by inline and deferred execute paths)."""
        emit_materialized_view_answer_diagnostics(intent, schema)
        if len(rows) > int(PolicyConfig.RESULT_ROW_COUNT_SOFT_WARNING):
            notify(
                f"Query result row count {len(rows)} exceeds the soft warning threshold.",
                stage="execution",
                code=DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
            )

        if len(rows) == 0:
            for suggestion in zero_row_where_suggestions(intent, schema):
                notify(suggestion, stage="execution", code=DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION)

        need_sql_feedback_prompt = force_feedback or TemplateOps.should_prompt_sql_feedback(
            store, q_norm, gen_out.matched_template
        )
        is_session = choice_port is not None and isinstance(choice_port, PipelineSessionMarker)
        if not is_session:
            display_final_results_to_stdout(
                q_norm,
                intent,
                sql,
                rows,
                structural_defaults=tmpl_sd,
                template_display_alias_map=(
                    getattr(gen_out.matched_template, "display_alias_map", None) if gen_out.matched_template else None
                ),
                **MainSpaceOps.federation_result_contract_kwargs(
                    gen_out, federated_prepare=federated_prepare, federated_bundle=federated_bundle
                ),
            )

        sql_prompt = "Is this correct?"
        if need_sql_feedback_prompt and persist_template_learning and gen_out.matched_template is None:
            pending = TemplateOps.upsert_pending_template(
                store,
                templates,
                schema,
                q_norm,
                intent,
                sql,
                dialect=dialect,
            )
            gen_out = replace(gen_out, matched_template=pending)
            MainSpaceOps.persist_template_store(
                MainInteractiveOps.owner_from_choice_port(choice_port) or owner,
                store,
            )
        if need_sql_feedback_prompt:
            if choice_port is not None and not choice_port.has_pending_choice():
                raise PipelineSuspended(
                    PIPELINE_SUSPEND_ID_SQL,
                    sql_prompt,
                    MainInteractiveOps._sql_feedback_suspend_context(
                        snap_post,
                        sql,
                        rows,
                        tmpl_sd,
                        gen_out,
                        matched_rejected_template,
                        force_feedback,
                        intent,
                        federated_prepare=federated_prepare,
                        federated_bundle=federated_bundle,
                    ),
                )
            choice = interactive_yes_no(INTERACTIVE_STAGE_SQL_FEEDBACK, sql_prompt, ["y", "n"], choice_port=choice_port)
        else:
            debug("[AUTO-ACCEPT] trust ceiling with prior accept on this question")
            choice = "y"
        if choice is None:
            note_interactive_turn(choice_port, outcome="user_declined", error="User cancelled SQL feedback.")
            if persist_template_learning:
                MainSpaceOps.persist_template_store(MainInteractiveOps.owner_from_choice_port(choice_port), store)
            return None

        if choice == "y" and intent.grain != "scalar":
            df_full = build_result_dataframe(
                rows,
                intent,
                sql,
                structural_defaults=tmpl_sd,
                q_norm=q_norm,
                template_display_alias_map=(
                    getattr(gen_out.matched_template, "display_alias_map", None) if gen_out.matched_template else None
                ),
                **MainSpaceOps.federation_result_contract_kwargs(
                    gen_out, federated_prepare=federated_prepare, federated_bundle=federated_bundle
                ),
            )
            if df_full is not None:
                art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
                artifacts_dir = str(art) if art is not None else None
                save_result_csv_for_store(df_full, store, artifacts_dir=artifacts_dir)

        feedback_result = handle_user_feedback(
            choice,
            intent,
            sql,
            schema,
            store,
            templates,
            rejected,
            q_norm,
            gen_out.generation_path,
            gen_out.matched_template,
            matched_rejected_template,
            dialect=dialect,
            structural_match_templates=gen_out.structural_match_templates,
            choice_port=choice_port,
            join_matches_template=gen_out.join_matches_template,
            form_storage=snap_post.form_storage,
            persist_template_learning=persist_template_learning,
            **MainSpaceOps.federation_feedback_kwargs(
                owner, gen_out, choice_port=choice_port, federated_prepare=federated_prepare
            ),
        )
        emit_llm_usage_summary_diagnostics(drain_llm_usage_records())
        row_tuples = [tuple(r) for r in rows]
        cols = result_columns_for_session(
            sql,
            row_tuples,
            intent=intent,
            **MainSpaceOps.federation_result_contract_kwargs(
                gen_out, federated_prepare=federated_prepare, federated_bundle=federated_bundle
            ),
        )
        if choice == "n":
            rb: str | None = None
            if isinstance(feedback_result, dict):
                rb = str(feedback_result.get("category") or "").strip().upper() or None
            note_interactive_turn(
                choice_port,
                outcome="intent_rejected",
                sql=MainSpaceOps.resolved_session_step_sql(
                    sql,
                    gen_out=gen_out,
                    federated_bundle=federated_bundle,
                    federated_plan=federated_prepare.plan if federated_prepare is not None else None,
                    generation_path=gen_out.generation_path,
                ),
                rows=row_tuples,
                columns=cols,
                intent=intent,
                rejection_bucket=rb,
                federated_bundle=federated_bundle,
                federated_plan=federated_prepare.plan if federated_prepare is not None else None,
                generation_path=gen_out.generation_path,
            )
        else:
            note_interactive_turn(
                choice_port,
                outcome="success",
                sql=MainSpaceOps.resolved_session_step_sql(
                    sql,
                    gen_out=gen_out,
                    federated_bundle=federated_bundle,
                    federated_plan=federated_prepare.plan if federated_prepare is not None else None,
                    generation_path=gen_out.generation_path,
                ),
                rows=row_tuples,
                columns=cols,
                intent=intent,
                matched_template=gen_out.matched_template,
                federated_bundle=federated_bundle,
                federated_plan=federated_prepare.plan if federated_prepare is not None else None,
                generation_path=gen_out.generation_path,
            )

    @staticmethod
    def _run_interactive_join_through_feedback(
        q_norm: str,
        intent: Any,
        semantic_warnings: list[str],
        dialect: Any,
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Any],
        rejected: dict[str, Any],
        schema_terms: set[str],
        choice_port: InteractiveChoicePort | None,
        has_union_match: bool,
        cols_changed: bool,
        matched_template: Any,
        union_select_cols: Any,
        structural_match_templates: Any,
        ikey: str,
        intent_sim: float,
        union_sql_path: GenerationPath | None = None,
        form_storage: QuestionFormStorage | None = None,
        persist_template_learning: bool = True,
    ) -> None:
        """Run join resolution, generation, execution, and feedback after intent is confirmed."""
        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        caller_visible_tables = effective_execution_visible_tables(
            schema,
            gate_kwargs.get("schema_context"),
            gate_kwargs.get("visible_objects"),
        )
        (
            matched_template,
            union_select_cols,
            cols_changed,
            union_sql_path,
            has_union_match,
            join_candidates,
            cmap,
            cte_join_hints,
        ) = prepare_union_match_join_phase(
            q_norm,
            intent,
            schema,
            dialect,
            templates,
            store=store,
            visible_tables=caller_visible_tables,
            schema_context=gate_kwargs.get("schema_context"),
            visible_objects=gate_kwargs.get("visible_objects"),
        )

        union_cand_ids = [
            c.template.id for c in list_union_match_candidates(intent, templates, visible_tables=caller_visible_tables)
        ]
        snap_post = build_interactive_tail_snapshot(
            q_norm,
            intent,
            schema,
            store,
            templates,
            rejected,
            schema_terms,
            dialect,
            semantic_warnings,
            has_union_match,
            cols_changed,
            matched_template,
            union_select_cols,
            structural_match_templates,
            ikey,
            intent_sim,
            union_sql_path=union_sql_path,
            union_candidate_template_ids=union_cand_ids,
            form_storage=form_storage,
        )
        matched_rejected_template = None

        MainInteractiveOps._run_sql_phase_after_intent_confirm(
            q_norm=q_norm,
            intent=intent,
            schema=schema,
            store=store,
            templates=templates,
            rejected=rejected,
            dialect=dialect,
            choice_port=choice_port,
            snap_post=snap_post,
            join_candidates=join_candidates,
            cmap=cmap,
            cte_join_hints=cte_join_hints,
            matched_template=matched_template,
            union_select_cols=union_select_cols,
            cols_changed=cols_changed,
            structural_match_templates=structural_match_templates,
            union_sql_path=union_sql_path,
            matched_rejected_template=matched_rejected_template,
            persist_template_learning=persist_template_learning,
        )

    @staticmethod
    def _federation_space_for_choice_port(choice_port: InteractiveChoicePort | None) -> SpaceContext | None:
        if choice_port is None:
            return None
        space_tables = getattr(choice_port, "space_tables", None)
        space_columns = getattr(choice_port, "space_columns", None)
        space_deny_objects = getattr(choice_port, "space_deny_objects", None)
        space_deny_columns = getattr(choice_port, "space_deny_columns", None)
        if not space_tables and not space_columns and not space_deny_objects and not space_deny_columns:
            return None
        return SpaceContext(
            tables=frozenset(space_tables or ()),
            columns=frozenset(space_columns or ()),
            deny_objects=frozenset(space_deny_objects or ()),
            deny_columns=frozenset(space_deny_columns or ()),
        )

    @staticmethod
    def _handle_federation_ineligible_plan(
        plan: FederatedPlan,
        *,
        choice_port: InteractiveChoicePort | None,
        store: dict[str, Any] | TemplateStoreView,
        owner: Any | None,
        persist_template_learning: bool,
    ) -> None:
        ineligible_reason = str(plan.ineligible_reason or "")
        user_message = federation_user_facing_ineligible_message(ineligible_reason)
        refusal_code = refusal_diagnostic_code_for_federation_reason(ineligible_reason)
        details = (("phase", "prepare"), ("reason", ineligible_reason))
        if refusal_code:
            emit_session_refusal_diagnostic(
                refusal_code,
                user_message,
                stage="validation",
                source_id="composite",
                details=details,
            )
        notify(
            user_message,
            stage="validation",
            code=DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE,
            source_id="composite",
            phase="prepare",
            details=details,
        )
        answerable = federation_ineligible_answerable_hint(plan.ineligible_reason)
        if answerable:
            notify(answerable, stage="rephrase_hint", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
        print_rephrase_hint(RephraseHint.FEDERATION_INELIGIBLE)
        note_interactive_turn(
            choice_port,
            outcome="validation_failed",
            error=user_message,
            failure_kind=FailureCategory.DENIED_REFERENCE.value,
            refusal_diagnostic_code=refusal_code,
        )
        clear_federated_turn_state(choice_port)

    @staticmethod
    def _check_federation_eligibility_before_confirm(
        intent: Any,
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        choice_port: InteractiveChoicePort | None,
        *,
        persist_template_learning: bool = True,
    ) -> bool:
        """Return False when a federated plan is ineligible and the turn should stop."""
        owner = MainInteractiveOps.owner_from_choice_port(choice_port)
        fed_manifest = (
            getattr(owner, "_federation_manifest", None)
            if owner is not None and owner_is_aether_federation(owner)
            else None
        )
        if not isinstance(fed_manifest, FederationManifest):
            return True
        fed_mappings = (
            getattr(owner, "_federation_mappings", None)
            if owner is not None and owner_is_aether_federation(owner)
            else None
        )
        if not isinstance(fed_mappings, FederationMappings):
            fed_mappings = None
        plan = plan_federated_intent(
            intent,
            schema,
            fed_manifest,
            fed_mappings,
            space=MainInteractiveOps._federation_space_for_choice_port(choice_port),
            member_graphs=(
                getattr(owner, "_federation_member_graphs", None)
                if owner is not None and isinstance(getattr(owner, "_federation_member_graphs", None), dict)
                else None
            ),
        )
        if not plan.ineligible_reason:
            return True
        MainInteractiveOps._handle_federation_ineligible_plan(
            plan, choice_port=choice_port, store=store, owner=owner, persist_template_learning=persist_template_learning
        )
        return False

    @staticmethod
    def _check_consumer_rbac_before_confirm(
        intent: Any,
        schema: SchemaGraph,
        choice_port: InteractiveChoicePort | None,
    ) -> bool:
        """Return False when declared intent tables/cols fall outside security scope (pre-confirm)."""
        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        schema_role = str(gate_kwargs.get("schema_role") or "owner")
        visible_objects = gate_kwargs.get("visible_objects")
        schema_context = gate_kwargs.get("schema_context")
        context_name = str(gate_kwargs.get("context_name") or "master")
        if not execution_scope_gate_active(schema_context, visible_objects, schema_role, context_name=context_name):
            return True
        scope_ctx = schema_context if schema_context is not None else EngineContext()
        if assert_consumer_intent_in_scope(
            intent,
            scope_ctx,
            schema,
            visible_objects,
            declared_tables_only=True,
        ):
            return True
        notify(
            PERMISSION_DENIED_USER_MESSAGE,
            stage="validation",
            code="PERMISSION_DENIED",
            details=(("reason", "scope"), ("phase", "pre_confirm")),
        )
        note_interactive_turn(
            choice_port,
            outcome="permission_denied",
            error=None,
            failure_kind=FailureCategory.ACCESS_POLICY.value,
        )
        return False

    @staticmethod
    def _run_sql_phase_after_intent_confirm(
        *,
        q_norm: str,
        intent: Any,
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Any],
        rejected: dict[str, Any],
        dialect: Any,
        choice_port: InteractiveChoicePort | None,
        snap_post: InteractiveTailSnapshot,
        join_candidates: dict[str, Any],
        cmap: dict[str, list[str]],
        cte_join_hints: dict[str, dict[str, Any]],
        matched_template: Any,
        union_select_cols: Any,
        cols_changed: bool,
        structural_match_templates: Any,
        union_sql_path: GenerationPath | None,
        matched_rejected_template: Any,
        persist_template_learning: bool = True,
    ) -> None:
        """Generate, execute and collect feedback after intent confirmation."""
        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        owner = MainInteractiveOps.owner_from_choice_port(choice_port)
        fed_manifest = (
            getattr(owner, "_federation_manifest", None)
            if owner is not None and owner_is_aether_federation(owner)
            else None
        )
        if not isinstance(fed_manifest, FederationManifest):
            fed_manifest = None
        fed_mappings = (
            getattr(owner, "_federation_mappings", None)
            if owner is not None and owner_is_aether_federation(owner)
            else None
        )
        if not isinstance(fed_mappings, FederationMappings):
            fed_mappings = None
        gen_out: SqlGenerationOutcome | None = None
        fed_prep_outcome: FederatedPrepareOutcome | None = None
        federation_exec_ctx: dict[str, Any] | None = None
        if fed_manifest is not None:
            if owner is not None:
                check_federation_member_drift_at_turn_start(owner, manifest=fed_manifest)
            fed_space = MainInteractiveOps._federation_space_for_choice_port(choice_port)
            plan = plan_federated_intent(
                intent,
                schema,
                fed_manifest,
                fed_mappings,
                space=fed_space,
                member_graphs=(
                    getattr(owner, "_federation_member_graphs", None)
                    if owner is not None and isinstance(getattr(owner, "_federation_member_graphs", None), dict)
                    else None
                ),
            )
            if plan.ineligible_reason:
                MainInteractiveOps._handle_federation_ineligible_plan(
                    plan,
                    choice_port=choice_port,
                    store=store,
                    owner=owner,
                    persist_template_learning=persist_template_learning,
                )
                return None
            if plan.steps:
                temporal_bind = resolve_anchored_temporal_bind(intent)
                plan = resolve_federated_combine(q_norm, plan, fed_manifest, schema, temporal_bind=temporal_bind)
                fed_dir = getattr(owner, "_federation_storage_dir", None) if owner is not None else None
                composite_id = str(schema.schema_graph_id or "")
                ikey = intent_key(intent)
                member_graphs = getattr(owner, "_federation_member_graphs", None) if owner is not None else None
                manifest_hash_value, member_tuple_hash_value = (
                    federation_plan_topology_identity(member_graphs, fed_manifest)
                    if isinstance(member_graphs, dict) and member_graphs
                    else ("", "")
                )
                cached_plan_template = (
                    lookup_federation_plan_template(
                        fed_dir,
                        composite_id,
                        ikey,
                        manifest_hash_value=manifest_hash_value,
                        member_tuple_hash_value=member_tuple_hash_value,
                    )
                    if fed_dir
                    else None
                )
                step_fps = federation_plan_step_fingerprints(
                    plan,
                    intent_key_fn=intent_key,
                    manifest=fed_manifest,
                    member_graphs=member_graphs if isinstance(member_graphs, dict) else None,
                    temporal_bind=temporal_bind,
                )
                plan_cache_hit = cached_plan_template is not None and federation_plan_matches_template(
                    plan,
                    cached_plan_template,
                    step_fingerprints=step_fps,
                    manifest_hash_value=manifest_hash_value,
                    member_tuple_hash_value=member_tuple_hash_value,
                )
                member_stores = (
                    MainSpaceOps.federation_stores_by_source(
                        owner,
                        member_graphs or {},
                        space_name=MainSpaceOps.session_space_name_for_federation(owner, choice_port),
                    )
                    if owner is not None and member_graphs
                    else None
                )
                fed_prep = None
                if plan_cache_hit and cached_plan_template is not None and member_stores:
                    try:
                        fed_prep = replay_federated_prepare_from_plan_template(
                            plan,
                            cached_plan_template,
                            schema,
                            stores_by_source=member_stores,
                            dialects_by_source=getattr(owner, "_federation_dialects", None),
                            source_runtimes=getattr(owner, "_federation_source_runtimes", None),
                            default_dialect=dialect,
                            manifest=fed_manifest,
                            member_graphs=member_graphs if isinstance(member_graphs, dict) else None,
                            q_norm=q_norm,
                        )
                    except FederationConfigError:
                        fed_prep = None
                    if fed_prep is not None:
                        notify(
                            "Replaying cached federation plan from member templates.",
                            stage="generation",
                            code=DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
                            source_id="composite",
                            details=(("phase", "prepare"), ("plan_id", ikey)),
                        )
                if fed_prep is None:
                    fed_prep = prepare_federated_sql_plan(
                        q_norm,
                        plan,
                        schema,
                        dialect=dialect,
                        dialects_by_source=getattr(owner, "_federation_dialects", None),
                        join_candidates=join_candidates,
                        cmap=cmap,
                        store=store,
                        cte_join_hints=cte_join_hints,
                        persist_template_learning=persist_template_learning,
                        stores_by_source=member_stores,
                        gate_kwargs_by_source=(
                            MainSpaceOps.federation_gate_kwargs_by_source(
                                owner, choice_port, fed_manifest, getattr(owner, "_federation_dialects", None)
                            )
                            if owner is not None
                            else None
                        ),
                        source_runtimes=getattr(owner, "_federation_source_runtimes", None),
                        manifest=fed_manifest,
                        member_graphs=member_graphs if isinstance(member_graphs, dict) else None,
                        **gate_kwargs,
                    )
                if not fed_prep.success:
                    fed_prep_outcome = fed_prep
                    clear_federated_turn_state(choice_port)
                    gen_out = SqlGenerationOutcome(
                        "",
                        False,
                        GenerationPath.INTENT_DIRECT_MATCH,
                        None,
                        sql_validation_error=fed_prep.sql_validation_error,
                        error_kind=fed_prep.error_kind,
                    )
                else:
                    federation_exec_ctx = {
                        "join_candidates": join_candidates,
                        "cmap": cmap,
                        "q_norm": q_norm,
                        "temporal_bind": temporal_bind.anchor_iso if temporal_bind else None,
                    }
                    fed_prep_outcome = fed_prep
                    if choice_port is not None and fed_dir and not plan_cache_hit:
                        member_template_ids = tuple(
                            (step.source_id, str(step.matched_template.id))
                            for step in fed_prep.steps
                            if step.matched_template is not None
                            and str(getattr(step.matched_template, "id", "") or "").strip()
                        )
                        choice_port._pending_federation_plan_template = FederationPlanTemplate(
                            plan_id=ikey,
                            composite_schema_graph_id=composite_id,
                            intent_key=ikey,
                            step_fingerprints=step_fps,
                            combine_hash=federation_plan_combine_hash(plan),
                            question=q_norm,
                            member_template_ids=member_template_ids,
                            residual_hash=federation_plan_residual_hash(plan),
                            manifest_hash=manifest_hash_value,
                            member_tuple_hash=member_tuple_hash_value,
                        )
                    gen_out = SqlGenerationOutcome(
                        fed_prep.display_sql,
                        True,
                        GenerationPath.FEDERATION_PLAN,
                        None,
                        federated_steps=fed_prep.steps,
                        federation_plan_id=ikey,
                        federation_dir=fed_dir or "",
                    )
        elif owner is not None:
            clear_federated_turn_state(choice_port)
        if gen_out is None:
            gen_out = generate_and_validate_sql(
                q_norm,
                intent,
                schema,
                join_candidates,
                cmap,
                dialect,
                store,
                cte_join_hints=cte_join_hints,
                matched_template=matched_template,
                union_select_cols=union_select_cols,
                cols_changed=cols_changed,
                structural_match_templates=structural_match_templates,
                union_sql_path=union_sql_path,
                persist_template_learning=persist_template_learning,
                **gate_kwargs,
            )
        if not gen_out.success:
            err_text = str(gen_out.sql_validation_error or "")
            ek = str(gen_out.error_kind or "")
            schema_role = str(gate_kwargs.get("schema_role") or "owner")
            fed_attr = MainInteractiveOps._federation_failure_attribution(fed_prep_outcome)
            if ek == FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value:
                if schema_role == "consumer":
                    note_interactive_turn(
                        choice_port,
                        outcome="schema_invalid_declined",
                        error=gen_out.sql_validation_error,
                        sql=None,
                        intent=None,
                        **fed_attr,
                    )
                else:
                    print_rephrase_hint(RephraseHint.SCHEMA_INVALID_DECLINED)
                    note_interactive_turn(
                        choice_port, outcome="schema_invalid_declined", error=gen_out.sql_validation_error, **fed_attr
                    )
                if persist_template_learning:
                    MainSpaceOps.persist_template_store(MainInteractiveOps.owner_from_choice_port(choice_port), store)
                clear_federated_turn_state(choice_port)
                return None
            perm_denied = ek == "explain_permission_denied" or Dialect.is_permission_denied_error(err_text)
            scope_denied = failure_kind_is_permission_denied(ek, err_text)
            if scope_denied or (schema_role == "consumer" and perm_denied):
                drift_message = None
                if schema_role == "consumer" and perm_denied and not scope_denied:
                    drift_message = PERMISSION_DRIFT_CONTACT_ADMIN_MESSAGE
                note_interactive_turn(
                    choice_port,
                    outcome="permission_denied",
                    error=drift_message,
                    sql=None,
                    intent=None,
                    **fed_attr,
                )
            else:
                note_interactive_turn(
                    choice_port,
                    outcome="validation_failed",
                    error=gen_out.sql_validation_error,
                    refusal_diagnostic_code=getattr(gen_out, "refusal_diagnostic_code", None),
                    **fed_attr,
                )
            if persist_template_learning:
                MainSpaceOps.persist_template_store(MainInteractiveOps.owner_from_choice_port(choice_port), store)
            clear_federated_turn_state(choice_port)
            return None

        sql = gen_out.sql
        tmpl_sd = getattr(gen_out.matched_template, "structural_defaults", None) if gen_out.matched_template else None
        stamp_sql_shape(
            sql,
            intent,
            generation_path=getattr(gen_out, "generation_path", None),
            federated_plan=fed_prep_outcome.plan if fed_prep_outcome is not None else None,
        )
        emit_explain_soft_diagnostics(getattr(gen_out, "explain_soft_findings", ()))
        force_feedback = TemplateOps.should_prompt_sql_feedback(store, snap_post.q_norm, gen_out.matched_template)
        exec_dialect = dialect
        exec_schema = schema
        snap_for_exec = snap_post
        is_session = choice_port is not None and isinstance(choice_port, PipelineSessionMarker)
        if is_session and choice_port is not None and not choice_port.has_pending_choice():
            raise PipelineSuspended(
                PIPELINE_SUSPEND_ID_EXECUTE,
                MainInteractiveOps._federation_execute_confirm_prompt(gen_out, fed_prep_outcome, fed_manifest),
                MainInteractiveOps._sql_execute_suspend_context(
                    snap_for_exec,
                    sql,
                    tmpl_sd,
                    gen_out,
                    matched_rejected_template,
                    force_feedback,
                    intent,
                    federated_prepare=fed_prep_outcome,
                    federation_plan_id=str(gen_out.federation_plan_id or ""),
                    federation_exec_context=federation_exec_ctx,
                ),
            )
        debug("executing SQL")
        federated_bundle: FederatedSqlBundle | None = None
        try:
            rows, federated_bundle = MainInteractiveOps._run_sql_execution_for_gen_out(
                intent=intent,
                exec_schema=exec_schema,
                exec_dialect=exec_dialect,
                tmpl_sd=tmpl_sd,
                gen_out=gen_out,
                owner=owner,
                choice_port=choice_port,
                q_norm=q_norm,
                join_candidates=join_candidates,
                cmap=cmap,
                store=store,
                federated_prepare=fed_prep_outcome,
                federation_exec_context=federation_exec_ctx,
            )
        except FederationPartialFailureError as exc:
            MainInteractiveOps._handle_federation_partial_failure_interactive(choice_port, owner, exc)
            return None
        except FederationTurnCancelledError as exc:
            MainInteractiveOps._handle_federation_turn_cancelled_interactive(choice_port, owner, exc)
            return None
        except AccessError as exc:
            MainInteractiveOps.note_access_error_turn(choice_port, exc)
            clear_federated_turn_state(choice_port)
            return None
        if len(rows) == 0:
            gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
            fixed_intent, fixed_rows = MainInteractiveOps.try_zero_row_where_remediation(
                intent, exec_schema, exec_dialect, tmpl_sd, gate_kwargs=gate_kwargs
            )
            if fixed_rows is not None:
                intent = fixed_intent
                rows = fixed_rows
        MainInteractiveOps._offer_sql_feedback_after_execute(
            q_norm=q_norm,
            intent=intent,
            sql=sql,
            rows=rows,
            schema=schema,
            store=store,
            templates=templates,
            rejected=rejected,
            dialect=dialect,
            choice_port=choice_port,
            snap_post=snap_post,
            tmpl_sd=tmpl_sd,
            gen_out=gen_out,
            matched_rejected_template=matched_rejected_template,
            force_feedback=force_feedback,
            persist_template_learning=persist_template_learning,
            owner=owner,
            federated_prepare=fed_prep_outcome,
            federated_bundle=federated_bundle,
        )

    @staticmethod
    def _run_interactive_after_parsed_intent(
        q_norm: str,
        intent: Any,
        semantic_warnings: list[str],
        dialect: Any,
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Any],
        rejected: dict[str, Any],
        schema_terms: set[str],
        choice_port: InteractiveChoicePort | None,
        intent_already_confirmed: bool = False,
        form_storage: QuestionFormStorage | None = None,
        refinement_ctx: RefinementContext | None = None,
        persist_template_learning: bool = True,
        interpret_plan: InterpretPlan | None = None,
    ) -> None:
        """Match union templates, confirm intent with the user, then continue through SQL feedback."""
        ikey = intent_key(intent)
        debug(f"[main_execution._run_interactive_after_parsed_intent] intent_key: {ikey[:32]}")

        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        caller_visible_tables = effective_execution_visible_tables(
            schema,
            gate_kwargs.get("schema_context"),
            gate_kwargs.get("visible_objects"),
        )

        union_result = match_template_for_union(intent, templates, visible_tables=caller_visible_tables)
        structural_match_templates = collect_structural_match_templates(
            intent, templates, visible_tables=caller_visible_tables
        )
        matched_template = None
        union_select_cols = None
        cols_changed = False
        union_sql_path: GenerationPath | None = None
        has_union_match = union_result is not None
        if union_result is not None:
            matched_template, union_select_cols, cols_changed, union_sql_path = union_result

        intent_sim = best_accepted_template_similarity(intent, templates, visible_tables=caller_visible_tables)
        union_cand_ids = [
            c.template.id for c in list_union_match_candidates(intent, templates, visible_tables=caller_visible_tables)
        ]
        snap_pre = build_interactive_tail_snapshot(
            q_norm,
            intent,
            schema,
            store,
            templates,
            rejected,
            schema_terms,
            dialect,
            semantic_warnings,
            has_union_match,
            cols_changed,
            matched_template,
            union_select_cols,
            structural_match_templates,
            ikey,
            intent_sim,
            union_sql_path=union_sql_path,
            union_candidate_template_ids=union_cand_ids,
            form_storage=form_storage,
            interpretation=interpret_plan,
        )
        if not MainInteractiveOps._check_federation_eligibility_before_confirm(
            intent, schema, store, choice_port, persist_template_learning=persist_template_learning
        ):
            return None
        if not MainInteractiveOps._check_consumer_rbac_before_confirm(intent, schema, choice_port):
            return None
        if not confirm_intent_with_user(
            intent,
            store,
            semantic_warnings,
            similarity_score=intent_sim,
            has_union_match=has_union_match,
            cols_changed=cols_changed,
            rejected=rejected,
            q_norm=q_norm,
            schema=schema,
            choice_port=choice_port,
            suspend_tail=snap_pre,
            intent_already_confirmed=intent_already_confirmed,
            refinement_ctx=refinement_ctx,
            persist_template_learning=persist_template_learning,
        ):
            note_interactive_turn(choice_port, outcome="user_declined", error="User declined intent confirmation.")
            return None

        MainInteractiveOps._run_interactive_join_through_feedback(
            q_norm,
            intent,
            semantic_warnings,
            dialect,
            schema,
            store,
            templates,
            rejected,
            schema_terms,
            choice_port,
            has_union_match,
            cols_changed,
            matched_template,
            union_select_cols,
            structural_match_templates,
            ikey,
            intent_sim,
            union_sql_path,
            form_storage=form_storage,
            persist_template_learning=persist_template_learning,
        )

    @staticmethod
    def _run_interactive_after_parsed_intent_from_tail(
        tail: InteractiveTailSnapshot,
        choice_port: InteractiveChoicePort | None,
        refinement_ctx: RefinementContext | None = None,
        persist_template_learning: bool = True,
    ) -> None:
        """Resume after a deferred intent confirmation using a frozen tail snapshot."""
        MainInteractiveOps._run_interactive_after_parsed_intent(
            tail.q_norm,
            tail.intent,
            cast(list[str], list(tail.semantic_warnings)),
            tail.dialect,
            tail.schema,
            tail.store,
            tail.templates,
            tail.rejected,
            tail.schema_terms,
            choice_port,
            intent_already_confirmed=True,
            form_storage=tail.form_storage,
            refinement_ctx=refinement_ctx,
            persist_template_learning=persist_template_learning,
        )

    @staticmethod
    def _run_interactive_post_intent_parse(
        q_norm: str,
        intent: Any,
        semantic_warnings: list[str],
        dialect: Any,
        schema: SchemaGraph,
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Any],
        rejected: dict[str, Any],
        schema_terms: set[str],
        choice_port: InteractiveChoicePort | None,
        form_storage: QuestionFormStorage | None = None,
        refinement_ctx: RefinementContext | None = None,
        persist_template_learning: bool = True,
        interpret_plan: InterpretPlan | None = None,
    ) -> None:
        """Continue the interactive pipeline after a parsed intent (joins through feedback)."""
        MainInteractiveOps._run_interactive_after_parsed_intent(
            q_norm,
            intent,
            semantic_warnings,
            dialect,
            schema,
            store,
            templates,
            rejected,
            schema_terms,
            choice_port,
            form_storage=form_storage,
            refinement_ctx=refinement_ctx,
            persist_template_learning=persist_template_learning,
            interpret_plan=interpret_plan,
        )

    @staticmethod
    def _complete_interactive_execute(
        ctx: SqlExecuteSuspendContext, choice: str | None, *, choice_port: InteractiveChoicePort | None = None
    ) -> None:
        """Run deferred execution after the separated execute step, then continue to SQL feedback."""
        tail = ctx.tail
        persist_tl = MainInteractiveOps.persist_template_learning_for_pipeline_session(choice_port)
        if choice is None or choice != "y":
            note_interactive_turn(choice_port, outcome="user_declined", error="User declined SQL execution.")
            if persist_tl:
                MainSpaceOps.persist_template_store(MainInteractiveOps.owner_from_choice_port(choice_port), tail.store)
            clear_federated_turn_state(choice_port)
            return None
        execution_intent = ctx.execution_intent
        owner = MainInteractiveOps.owner_from_choice_port(choice_port)
        MainInteractiveOps._verify_federation_execute_resume(ctx)
        fed_prep = ctx.federated_prepare
        exec_ctx = MainInteractiveOps._federation_exec_context_from_pairs(ctx.federation_exec_context)
        federated_bundle: FederatedSqlBundle | None = None
        if fed_prep is not None:
            try:
                rows, federated_bundle = MainInteractiveOps._run_sql_execution_for_gen_out(
                    intent=execution_intent,
                    exec_schema=tail.schema,
                    exec_dialect=tail.dialect,
                    tmpl_sd=ctx.tmpl_sd,
                    gen_out=ctx.gen_out,
                    owner=owner,
                    choice_port=choice_port,
                    q_norm=str(exec_ctx.get("q_norm") or tail.q_norm),
                    join_candidates=exec_ctx.get("join_candidates"),
                    cmap=exec_ctx.get("cmap"),
                    store=tail.store,
                    federated_prepare=fed_prep,
                    federation_exec_context=exec_ctx,
                )
            except FederationPartialFailureError as exc:
                MainInteractiveOps._handle_federation_partial_failure_interactive(choice_port, owner, exc)
                return None
            except FederationTurnCancelledError as exc:
                MainInteractiveOps._handle_federation_turn_cancelled_interactive(choice_port, owner, exc)
                return None
            except AccessError as exc:
                MainInteractiveOps.note_access_error_turn(choice_port, exc)
                clear_federated_turn_state(choice_port)
                return None
        else:
            if ctx.gen_out.generation_path is GenerationPath.FEDERATION_PLAN:
                note_interactive_turn(
                    choice_port,
                    outcome="error",
                    error="Federated prepare outcome missing; cannot execute federation plan.",
                )
                clear_federated_turn_state(choice_port)
                return None
            exec_dialect = tail.dialect
            exec_schema = tail.schema
            fed_manifest = (
                getattr(owner, "_federation_manifest", None)
                if owner is not None and owner_is_aether_federation(owner)
                else None
            )
            fed_mappings = (
                getattr(owner, "_federation_mappings", None)
                if owner is not None and owner_is_aether_federation(owner)
                else None
            )
            if not isinstance(fed_mappings, FederationMappings):
                fed_mappings = None
            if isinstance(fed_manifest, FederationManifest) and owner is not None:
                single_source = MainSpaceOps.federation_single_source_sql_context(
                    owner, execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
                )
                if single_source is not None:
                    exec_dialect, exec_schema = single_source
            try:
                gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
                rows = MainInteractiveOps._run_pipeline_sql_rows(
                    intent=execution_intent,
                    schema=exec_schema,
                    dialect=exec_dialect,
                    tmpl_sd=ctx.tmpl_sd,
                    gate_kwargs=gate_kwargs,
                )
            except AccessError as exc:
                MainInteractiveOps.note_access_error_turn(choice_port, exc)
                return None
        if len(rows) == 0:
            exec_dialect = tail.dialect
            exec_schema = tail.schema
            fed_manifest = (
                getattr(owner, "_federation_manifest", None)
                if owner is not None and owner_is_aether_federation(owner)
                else None
            )
            fed_mappings = (
                getattr(owner, "_federation_mappings", None)
                if owner is not None and owner_is_aether_federation(owner)
                else None
            )
            if not isinstance(fed_mappings, FederationMappings):
                fed_mappings = None
            if isinstance(fed_manifest, FederationManifest) and owner is not None:
                single_source = MainSpaceOps.federation_single_source_sql_context(
                    owner, execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
                )
                if single_source is not None:
                    exec_dialect, exec_schema = single_source
            gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
            fixed_intent, fixed_rows = MainInteractiveOps.try_zero_row_where_remediation(
                execution_intent, exec_schema, exec_dialect, ctx.tmpl_sd, gate_kwargs=gate_kwargs
            )
            if fixed_rows is not None:
                execution_intent = fixed_intent
                rows = fixed_rows
        stamp_sql_shape(
            ctx.sql,
            execution_intent,
            generation_path=ctx.gen_out.generation_path,
            federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
        )
        emit_explain_soft_diagnostics(getattr(ctx.gen_out, "explain_soft_findings", ()))
        owner = MainInteractiveOps.owner_from_choice_port(choice_port)
        MainInteractiveOps._offer_sql_feedback_after_execute(
            q_norm=tail.q_norm,
            intent=execution_intent,
            sql=ctx.sql,
            rows=rows,
            schema=tail.schema,
            store=tail.store,
            templates=tail.templates,
            rejected=tail.rejected,
            dialect=tail.dialect,
            choice_port=choice_port,
            snap_post=tail,
            tmpl_sd=ctx.tmpl_sd,
            gen_out=ctx.gen_out,
            matched_rejected_template=ctx.matched_rejected_template,
            force_feedback=ctx.force_feedback,
            persist_template_learning=persist_tl,
            owner=owner,
            federated_prepare=ctx.federated_prepare,
            federated_bundle=federated_bundle,
        )

    @staticmethod
    def _reexecute_suspend_sql_rows(
        ctx: SqlExecuteSuspendContext | SqlFeedbackSuspendContext,
        *,
        choice_port: InteractiveChoicePort | None = None,
    ) -> tuple[list[tuple[Any, ...]], FederatedSqlBundle | None]:
        """Re-run validated SQL after a deferred suspend instead of replaying preview rows."""
        tail = ctx.tail
        owner = MainInteractiveOps.owner_from_choice_port(choice_port)
        fed_prep = ctx.federated_prepare
        federated_bundle: FederatedSqlBundle | None = (
            ctx.federated_bundle if isinstance(ctx, SqlFeedbackSuspendContext) else None
        )
        exec_ctx: dict[str, Any] = {}
        if isinstance(ctx, SqlExecuteSuspendContext):
            exec_ctx = MainInteractiveOps._federation_exec_context_from_pairs(ctx.federation_exec_context)
        if fed_prep is not None:
            rows, federated_bundle = MainInteractiveOps._run_sql_execution_for_gen_out(
                intent=ctx.execution_intent,
                exec_schema=tail.schema,
                exec_dialect=tail.dialect,
                tmpl_sd=ctx.tmpl_sd,
                gen_out=ctx.gen_out,
                owner=owner,
                choice_port=choice_port,
                q_norm=str(exec_ctx.get("q_norm") or tail.q_norm),
                join_candidates=exec_ctx.get("join_candidates"),
                cmap=exec_ctx.get("cmap"),
                store=tail.store,
                federated_prepare=fed_prep,
                federation_exec_context=exec_ctx or None,
            )
            return [tuple(r) for r in rows], federated_bundle
        exec_dialect = tail.dialect
        exec_schema = tail.schema
        fed_manifest = (
            getattr(owner, "_federation_manifest", None)
            if owner is not None and owner_is_aether_federation(owner)
            else None
        )
        fed_mappings = (
            getattr(owner, "_federation_mappings", None)
            if owner is not None and owner_is_aether_federation(owner)
            else None
        )
        if not isinstance(fed_mappings, FederationMappings):
            fed_mappings = None
        if isinstance(fed_manifest, FederationManifest) and owner is not None:
            single_source = MainSpaceOps.federation_single_source_sql_context(
                owner, ctx.execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
            )
            if single_source is not None:
                exec_dialect, exec_schema = single_source
        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        rows = MainInteractiveOps._run_pipeline_sql_rows(
            intent=ctx.execution_intent,
            schema=exec_schema,
            dialect=exec_dialect,
            tmpl_sd=ctx.tmpl_sd,
            gate_kwargs=gate_kwargs,
        )
        return rows, federated_bundle

    @staticmethod
    def _complete_interactive_sql_feedback(
        ctx: SqlFeedbackSuspendContext, choice: str | None, *, choice_port: InteractiveChoicePort | None = None
    ) -> None:
        """Apply accept or reject after a deferred final-SQL prompt."""
        tail = ctx.tail
        intent = ctx.execution_intent
        sql = ctx.sql
        tmpl_sd = ctx.tmpl_sd
        federated_bundle = ctx.federated_bundle
        rows: list[tuple[Any, ...]] = []
        persist_tl = MainInteractiveOps.persist_template_learning_for_pipeline_session(choice_port)
        if choice is None:
            if persist_tl:
                MainSpaceOps.persist_template_store(MainInteractiveOps.owner_from_choice_port(choice_port), tail.store)
            return None
        try:
            rows, federated_bundle = MainInteractiveOps._reexecute_suspend_sql_rows(ctx, choice_port=choice_port)
        except AccessError as exc:
            MainInteractiveOps.note_access_error_turn(choice_port, exc)
            return None
        if choice == "y":
            if intent.grain != "scalar":
                df_full = build_result_dataframe(
                    rows,
                    intent,
                    sql,
                    structural_defaults=tmpl_sd,
                    q_norm=tail.q_norm,
                    template_display_alias_map=(
                        getattr(ctx.gen_out.matched_template, "display_alias_map", None)
                        if ctx.gen_out.matched_template
                        else None
                    ),
                    **MainSpaceOps.federation_result_contract_kwargs(
                        ctx.gen_out, federated_prepare=ctx.federated_prepare, federated_bundle=federated_bundle
                    ),
                )
                if df_full is not None:
                    owner = MainInteractiveOps.owner_from_choice_port(choice_port)
                    art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
                    artifacts_dir = str(art) if art is not None else None
                    save_result_csv_for_store(df_full, tail.store, artifacts_dir=artifacts_dir)
        feedback_result = handle_user_feedback(
            choice,
            intent,
            sql,
            tail.schema,
            tail.store,
            tail.templates,
            tail.rejected,
            tail.q_norm,
            ctx.gen_out.generation_path,
            ctx.gen_out.matched_template,
            ctx.matched_rejected_template,
            dialect=tail.dialect,
            structural_match_templates=ctx.gen_out.structural_match_templates,
            choice_port=choice_port,
            join_matches_template=ctx.gen_out.join_matches_template,
            form_storage=tail.form_storage,
            persist_template_learning=persist_tl,
            **MainSpaceOps.federation_feedback_kwargs(
                MainInteractiveOps.owner_from_choice_port(choice_port),
                ctx.gen_out,
                choice_port=choice_port,
                federated_prepare=ctx.federated_prepare,
            ),
        )
        emit_llm_usage_summary_diagnostics(drain_llm_usage_records())
        row_tuples = [tuple(r) for r in rows]
        cols = result_columns_for_session(
            sql,
            row_tuples,
            intent=intent,
            **MainSpaceOps.federation_result_contract_kwargs(
                ctx.gen_out, federated_prepare=ctx.federated_prepare, federated_bundle=federated_bundle
            ),
        )
        if choice == "n":
            rb: str | None = None
            if isinstance(feedback_result, dict):
                rb = str(feedback_result.get("category") or "").strip().upper() or None
            note_interactive_turn(
                choice_port,
                outcome="intent_rejected",
                sql=MainSpaceOps.resolved_session_step_sql(
                    sql,
                    gen_out=ctx.gen_out,
                    federated_bundle=federated_bundle,
                    federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
                    generation_path=ctx.gen_out.generation_path,
                ),
                rows=row_tuples,
                columns=cols,
                intent=intent,
                rejection_bucket=rb,
                federated_bundle=federated_bundle,
                federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
                generation_path=ctx.gen_out.generation_path,
            )
        else:
            note_interactive_turn(
                choice_port,
                outcome="success",
                sql=MainSpaceOps.resolved_session_step_sql(
                    sql,
                    gen_out=ctx.gen_out,
                    federated_bundle=federated_bundle,
                    federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
                    generation_path=ctx.gen_out.generation_path,
                ),
                rows=row_tuples,
                columns=cols,
                intent=intent,
                matched_template=ctx.gen_out.matched_template,
                federated_bundle=federated_bundle,
                federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
                generation_path=ctx.gen_out.generation_path,
            )

    @staticmethod
    def _complete_intent_rejection_feedback(
        tail: InteractiveTailSnapshot, feedback: str | None, choice_port: InteractiveChoicePort | None
    ) -> None:
        """Persist free-text feedback after the user declines an intent."""
        body = (feedback or "").strip() or "user_declined_intent"
        entry = TemplateOps.summarize_failure_for_memory(
            question=tail.q_norm,
            intent=tail.intent,
            kind=FeedbackKind.INTENT_REJECTED,
            schema_hash=tail.schema.effective_structural_hash,
            user_reason=body,
        )
        persist_tl = MainInteractiveOps.persist_template_learning_for_pipeline_session(choice_port)
        if persist_tl:
            TemplateOps.record_question_feedback(tail.store, tail.q_norm, entry)
            MainSpaceOps.persist_template_store(MainInteractiveOps.owner_from_choice_port(choice_port), tail.store)
        rb = entry.buckets[0].value if entry.buckets else None
        ctx_ref = getattr(choice_port, "_refinement_ctx", None)
        reason_line = body
        if (
            ctx_ref is not None
            and refinement_retry_available(ctx_ref)
            and not (choice_port is not None and isinstance(choice_port, PipelineSessionMarker))
        ):
            ctx_ref.accumulated_reasons.append(reason_line)
            ctx_ref.pending_retry = True
            raise RefinementRetry
        print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=rb)
        note_interactive_turn(choice_port, outcome="user_declined", error="User declined intent confirmation.")

    @staticmethod
    def dispatch_pipeline_resume(session: Any, suspended: PipelineSuspended) -> None:
        """Drive the next pipeline segment after the caller enqueued a programmatic choice."""
        sid = suspended.state_id
        payload = suspended.payload
        persist_tl = MainInteractiveOps.persist_template_learning_for_pipeline_session(session)
        if sid == PIPELINE_SUSPEND_ID_DIRECT_REUSE:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, DirectReuseSuspendContext):
                raise TypeError("direct reuse resume expects DirectReuseSuspendContext")
            complete_direct_sql_reuse_user_choice(
                payload, ch, choice_port=session, persist_template_learning=persist_tl
            )
            return
        if sid == PIPELINE_SUSPEND_ID_INTENT_CONFIRM:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, InteractiveTailSnapshot):
                raise TypeError("intent resume expects InteractiveTailSnapshot")
            if ch is None:
                raise PipelineSuspended("empty_choice_queue", "interactive choice queue is empty", None)
            if ch != "y":
                if getattr(payload.intent, "schema_invalid", False):
                    print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=None)
                note_interactive_turn(session, outcome="user_declined", error="User declined intent confirmation.")
                if persist_tl:
                    MainSpaceOps.persist_template_store(
                        MainInteractiveOps.owner_from_choice_port(session), payload.store
                    )
                return
            clear_interpret_schema_invalid_after_user_accept(payload.intent)
            MainInteractiveOps._run_interactive_after_parsed_intent_from_tail(
                payload,
                session,
                refinement_ctx=getattr(session, "_refinement_ctx", None),
                persist_template_learning=persist_tl,
            )
            return
        if sid == PIPELINE_SUSPEND_ID_EXECUTE:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, SqlExecuteSuspendContext):
                raise TypeError("execute resume expects SqlExecuteSuspendContext")
            MainInteractiveOps._complete_interactive_execute(payload, ch, choice_port=session)
            return
        if sid == PIPELINE_SUSPEND_ID_SQL:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, SqlFeedbackSuspendContext):
                raise TypeError("SQL feedback resume expects SqlFeedbackSuspendContext")
            MainInteractiveOps._complete_interactive_sql_feedback(payload, ch, choice_port=session)
            return
        if sid == PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, UserFeedbackRejectSuspendContext):
                raise TypeError("user feedback reject resume expects UserFeedbackRejectSuspendContext")
            complete_user_feedback_reject(
                payload,
                needs_reason=True,
                reject_reason=ch or "",
                choice_port=session,
                refinement_ctx=getattr(session, "_refinement_ctx", None),
                persist_template_learning=persist_tl,
            )
            return
        if sid == PIPELINE_SUSPEND_ID_INTENT_FEEDBACK:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, InteractiveTailSnapshot):
                raise TypeError("intent feedback resume expects InteractiveTailSnapshot")
            MainInteractiveOps._complete_intent_rejection_feedback(payload, ch, session)
            return
        raise RuntimeError(f"unknown pipeline suspend id: {sid!r}")
