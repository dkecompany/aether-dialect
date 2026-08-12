"""Suspend/session serde helpers and PipelineSession."""

from __future__ import annotations

import base64
import contextlib
import os
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pandas

from ._config import (
    EngineConfig,
)
from ._constants import (
    AUDIT_EVENT_ASK_CANCELLED,
    AUDIT_EVENT_ASK_SUSPEND,
    DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
    DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED,
    DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
    MASTER_AETHERSPACE_NAME,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_EXECUTE,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_SQL,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    SESSION_KIND_ERROR,
    SESSION_KIND_IDLE,
    SESSION_KIND_META,
    SESSION_KIND_RESULT,
    SESSION_PERSISTENCE_FORMAT_VERSION,
    SUSPEND_ID_TO_SESSION_KIND,
    SUSPEND_STATE_FORMAT_VERSION,
)
from ._constants_runtime import (
    SESSION_PROMPT_REASON,
    SESSION_PROMPT_YESNO,
)
from ._contracts_base import (
    ConfigError,
    Diagnostic,
    DiagnosticSeverity,
    DomainKnowledgeEntry,
    DomainKnowledgeHolder,
    DomainKnowledgeState,
    EngineIdentity,
    FederationCapExceededError,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationPartialFailureError,
    ParamValue,
    SchemaRole,
    SessionActiveError,
    SessionTurnCancelledError,
    SuspendedSessionExpiredError,
)
from ._contracts_core import (
    DirectReuseSuspendContext,
    FederatedPlan,
    FederationExecutionContext,
    GenerationPath,
    IntentSummary,
    InteractiveChoicePort,
    InteractiveTailSnapshot,
    InterpretPlan,
    ParameterBinding,
    PipelineSessionMarker,
    PipelineSuspended,
    QuestionFormStorage,
    RefinementContext,
    RuntimeIntent,
    SessionError,
    SessionOutcome,
    SessionStep,
    SqlExecuteSuspendContext,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
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
    federation_user_facing_error_message,
    probe_federation_member_liveness,
)
from ._federation_manifest import (
    owner_is_aether_federation,
)
from ._llm_provider import (
    SandboxRuntimeState,
)
from ._main_init import MainInitOps
from ._main_interactive import MainInteractiveOps
from ._main_spaces import MainSpaceOps
from ._pipeline_execute import (
    build_result_dataframe,
    intent_result_column_headers,
    result_columns_for_session,
)
from ._pipeline_generate import compose_intent_confirm_session_message
from ._templates import TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    active_turn_id,
    apply_refusal_timing_floor,
    debug,
    details_with_turn_id,
    diagnostic_segment,
    domain_knowledge_scope,
    drain_diagnostic_collector,
    format_versions_match,
    llm_call_audit_details,
    llm_execution_scope,
    llm_turn_audit_details,
    llm_turn_cost_diagnostic,
    llm_usage_session_scope,
    mint_turn_id,
    normalize_question,
    owner_limits_scope,
    pop_ask_phase_callback,
    pop_audit_emit,
    pop_engine_identity,
    pop_session_turn_cancel,
    pop_turn_id,
    pop_turn_timing,
    prompt_cache_schema_scope,
    push_ask_phase_callback,
    push_audit_emit,
    push_engine_identity,
    push_session_turn_cancel,
    push_turn_id,
    push_turn_timing,
    reset_diagnostic_collector,
    reset_turn_llm_scope,
    sanitize_audit_details_for_egress,
    sanitize_federation_diagnostics_for_egress,
    sanitize_session_step_for_egress,
    schema_prompt_cache_id,
    session_error_from_terminal_message,
    session_error_from_turn_snap,
    set_diagnostic_collector,
    set_turn_llm_scope,
    snapshot_llm_usage_records,
    summarize_llm_turn_usage,
    take_and_clear_orphan_diagnostics,
    turn_elapsed_ms,
)


class MainSessionSerdeOps:
    """Suspend/session serde helpers and PipelineSession."""

    @staticmethod
    def _check_suspend_state_format_version(payload: dict[str, Any]) -> None:
        stored = payload.get("format_version")
        if not format_versions_match(stored, SUSPEND_STATE_FORMAT_VERSION):
            raise ConfigError(
                f"suspend state payload has format_version {stored!r}; "
                f"this build expects {SUSPEND_STATE_FORMAT_VERSION}."
            )

    @staticmethod
    def _check_session_persistence_format_version(payload: dict[str, Any]) -> None:
        stored = payload.get("format_version")
        if not format_versions_match(stored, SESSION_PERSISTENCE_FORMAT_VERSION):
            raise ConfigError(
                f"session persistence payload has format_version {stored!r}; "
                f"this build expects {SESSION_PERSISTENCE_FORMAT_VERSION}."
            )

    @staticmethod
    def _serialize_diagnostic(diag: Diagnostic) -> dict[str, Any]:
        out: dict[str, Any] = {
            "stage": diag.stage,
            "level": diag.level.value if isinstance(diag.level, DiagnosticSeverity) else diag.level,
            "code": diag.code,
            "message": diag.message,
            "details": [list(pair) for pair in diag.details],
            "phase": diag.phase,
            "remediation": diag.remediation,
            "subject": diag.subject,
            "count": diag.count,
        }
        if diag.duration_ms is not None:
            out["duration_ms"] = diag.duration_ms
        if diag.source_id is not None:
            out["source_id"] = diag.source_id
        return out

    @staticmethod
    def _deserialize_diagnostic(raw: dict[str, Any]) -> Diagnostic:
        details_raw = raw.get("details") or []
        details = tuple(tuple(pair) for pair in details_raw)
        duration_ms = raw.get("duration_ms")
        if duration_ms is not None:
            duration_ms = int(duration_ms)
        source_id = raw.get("source_id")
        if source_id is not None:
            source_id = str(source_id)
        if "phase" in raw:
            phase = None if raw["phase"] is None else str(raw["phase"])
        else:
            phase = str(raw["stage"])
        remediation = raw.get("remediation")
        if remediation is not None:
            remediation = str(remediation)
        subject = raw.get("subject")
        if subject is not None:
            subject = str(subject)
        return Diagnostic(
            stage=str(raw["stage"]),
            level=str(raw["level"]),
            code=str(raw["code"]),
            message=str(raw["message"]),
            details=details,
            duration_ms=duration_ms,
            source_id=source_id,
            phase=phase,
            remediation=remediation,
            subject=subject,
            count=int(raw.get("count", 1)),
        )

    @staticmethod
    def _serialize_intent_summary(summary: IntentSummary) -> dict[str, Any]:
        return {
            "tables": list(summary.tables),
            "select_cols": list(summary.select_cols),
            "filters": list(summary.filters),
            "group_by": list(summary.group_by),
            "order_by": list(summary.order_by),
            "limit": summary.limit,
            "natural_language": summary.natural_language,
        }

    @staticmethod
    def _deserialize_intent_summary(raw: dict[str, Any]) -> IntentSummary:
        limit = raw.get("limit")
        if limit is not None:
            limit = int(limit)
        return IntentSummary(
            tables=tuple(raw.get("tables") or ()),
            select_cols=tuple(raw.get("select_cols") or ()),
            filters=tuple(raw.get("filters") or ()),
            group_by=tuple(raw.get("group_by") or ()),
            order_by=tuple(raw.get("order_by") or ()),
            limit=limit,
            natural_language=str(raw.get("natural_language") or ""),
        )

    @staticmethod
    def _deserialize_param_value(raw: Any) -> ParamValue:
        if isinstance(raw, list):
            return [item for item in raw]
        if isinstance(raw, (str, int, float, bool)):
            return raw
        raise ValueError(f"unsupported parameter value type: {type(raw)!r}")

    @staticmethod
    def _serialize_parameter_binding(binding: ParameterBinding) -> dict[str, Any]:
        return {
            "handle": binding.handle,
            "current_value": binding.current_value,
            "display_name": binding.display_name,
            "column_expr": binding.column_expr,
        }

    @staticmethod
    def _serialize_session_error(error: SessionError) -> dict[str, Any]:
        return {
            "code": error.code.value,
            "detail_code": error.detail_code,
            "source_id": error.source_id,
            "phase": error.phase,
            "limit_key": error.limit_key,
        }

    @staticmethod
    def _deserialize_session_error(raw: dict[str, Any]) -> SessionError:
        code_raw = str(raw["code"])
        return SessionError(
            code=SessionOutcome(code_raw),
            detail_code=raw.get("detail_code"),
            source_id=raw.get("source_id"),
            phase=raw.get("phase"),
            limit_key=raw.get("limit_key"),
        )

    @staticmethod
    def _deserialize_parameter_binding(raw: dict[str, Any]) -> ParameterBinding:
        current_raw = raw.get("current_value")
        current_value = None if current_raw is None else MainSessionSerdeOps._deserialize_param_value(current_raw)
        return ParameterBinding(
            handle=str(raw["handle"]),
            current_value=current_value,
            display_name=str(raw.get("display_name") or ""),
            column_expr=str(raw.get("column_expr") or ""),
        )

    @staticmethod
    def _json_encode_session_cell(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        return value

    @staticmethod
    def _serialize_dataframe(df: pandas.DataFrame) -> dict[str, Any]:
        records = [
            {key: MainSessionSerdeOps._json_encode_session_cell(val) for key, val in row.items()}
            for row in df.to_dict(orient="records")
        ]
        return {
            "columns": list(df.columns),
            "records": records,
        }

    @staticmethod
    def _deserialize_dataframe(raw: dict[str, Any]) -> pandas.DataFrame:
        columns = list(raw.get("columns") or [])
        records = raw.get("records") or []
        if not columns:
            return pandas.DataFrame(records)
        return pandas.DataFrame.from_records(records, columns=columns)

    @staticmethod
    def serialize_session_step(step: SessionStep) -> dict[str, Any]:
        """Return a JSON-serialisable dict for *step*."""
        payload: dict[str, Any] = {
            "format_version": SESSION_PERSISTENCE_FORMAT_VERSION,
            "done": step.done,
            "prompt": step.prompt,
            "kind": step.kind,
            "sql": step.sql,
            "answer": step.answer,
            "reply_shape": step.reply_shape,
            "semantic_warnings": list(step.semantic_warnings),
            "diagnostics": [MainSessionSerdeOps._serialize_diagnostic(d) for d in step.diagnostics],
            "parameters": [MainSessionSerdeOps._serialize_parameter_binding(p) for p in step.parameters],
            "data_truncated": step.data_truncated,
            "template_id": step.template_id,
            "turn_id": step.turn_id,
            "elapsed_ms": step.elapsed_ms,
        }
        if step.error is not None:
            payload["error"] = MainSessionSerdeOps._serialize_session_error(step.error)
        if step.data is not None:
            payload["data"] = MainSessionSerdeOps._serialize_dataframe(step.data)
        if step.intent_summary is not None:
            payload["intent_summary"] = MainSessionSerdeOps._serialize_intent_summary(step.intent_summary)
        if step.llm_usage is not None:
            payload["llm_usage"] = asdict(step.llm_usage)
        return payload

    @staticmethod
    def deserialize_session_step(payload: dict[str, Any]) -> SessionStep:
        """Rebuild a :class:`SessionStep` from *payload*, refusing on version mismatch."""
        MainSessionSerdeOps._check_session_persistence_format_version(payload)
        data_out: pandas.DataFrame | None = None
        data_raw = payload.get("data")
        if data_raw is not None:
            data_out = MainSessionSerdeOps._deserialize_dataframe(data_raw)
        intent_summary_raw = payload.get("intent_summary")
        intent_summary = (
            MainSessionSerdeOps._deserialize_intent_summary(intent_summary_raw)
            if intent_summary_raw is not None
            else None
        )
        diagnostics_raw = payload.get("diagnostics") or []
        diagnostics = tuple(MainSessionSerdeOps._deserialize_diagnostic(d) for d in diagnostics_raw)
        parameters_raw = payload.get("parameters") or []
        parameters = tuple(MainSessionSerdeOps._deserialize_parameter_binding(p) for p in parameters_raw)
        reply_shape = payload.get("reply_shape")
        if reply_shape is not None and reply_shape not in ("yes_no", "free_text"):
            raise ValueError(f"invalid reply_shape: {reply_shape!r}")
        sql_raw = payload.get("sql")
        if sql_raw is not None and not isinstance(sql_raw, (str, dict)):
            raise ValueError(f"invalid sql payload type: {type(sql_raw)!r}")
        error_raw = payload.get("error")
        error_out: SessionError | None = None
        if isinstance(error_raw, dict):
            error_out = MainSessionSerdeOps._deserialize_session_error(error_raw)
        return SessionStep(
            done=bool(payload["done"]),
            prompt=payload.get("prompt"),
            kind=str(payload["kind"]),
            sql=sql_raw,
            data=data_out,
            answer=payload.get("answer"),
            intent_summary=intent_summary,
            diagnostics=diagnostics,
            error=error_out,
            reply_shape=reply_shape,
            semantic_warnings=tuple(payload.get("semantic_warnings") or ()),
            parameters=parameters,
            data_truncated=bool(payload.get("data_truncated", False)),
            template_id=str(payload["template_id"]) if payload.get("template_id") is not None else None,
            turn_id=str(payload["turn_id"]) if payload.get("turn_id") is not None else None,
            elapsed_ms=int(payload["elapsed_ms"]) if payload.get("elapsed_ms") is not None else None,
        )

    @staticmethod
    def _owner_learning_refs(owner: Any | None) -> dict[str, Any]:
        if owner is None:
            return {"schema": None, "store": {}, "templates": {}, "rejected": {}, "dialect": None}
        return {
            "schema": getattr(owner, "_schema_graph", None),
            "store": getattr(owner, "_store", {}) or {},
            "templates": getattr(owner, "_templates", {}) or {},
            "rejected": getattr(owner, "_rejected", {}) or {},
            "dialect": getattr(owner, "_dialect", None),
        }

    @staticmethod
    def _serialize_question_form_storage(form_storage: QuestionFormStorage | None) -> dict[str, Any] | None:
        if form_storage is None:
            return None
        return {
            "corrected": form_storage.corrected,
            "normalized_optional": form_storage.normalized_optional,
            "normalized_negative_memory_dropped": form_storage.normalized_negative_memory_dropped,
            "accept_via_normalized_lookup_only": form_storage.accept_via_normalized_lookup_only,
        }

    @staticmethod
    def _deserialize_question_form_storage(raw: dict[str, Any] | None) -> QuestionFormStorage | None:
        if not isinstance(raw, dict):
            return None
        return QuestionFormStorage(
            corrected=str(raw.get("corrected") or ""),
            normalized_optional=raw.get("normalized_optional"),
            normalized_negative_memory_dropped=bool(raw.get("normalized_negative_memory_dropped", False)),
            accept_via_normalized_lookup_only=bool(raw.get("accept_via_normalized_lookup_only", False)),
        )

    @staticmethod
    def _serialize_interpret_plan(plan: InterpretPlan | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "approach": plan.approach,
            "tables": list(plan.tables),
            "grounding": [list(pair) for pair in plan.grounding],
            "schema_invalid": plan.schema_invalid,
            "missing": plan.missing,
        }

    @staticmethod
    def _deserialize_interpret_plan(raw: dict[str, Any] | None) -> InterpretPlan | None:
        if not isinstance(raw, dict):
            return None
        grounding_raw = raw.get("grounding") or []
        return InterpretPlan(
            approach=str(raw.get("approach") or ""),
            tables=tuple(raw.get("tables") or ()),
            grounding=tuple(tuple(pair) for pair in grounding_raw),
            schema_invalid=bool(raw.get("schema_invalid", False)),
            missing=str(raw.get("missing") or ""),
        )

    @staticmethod
    def _serialize_template_ref(template: Template | None) -> dict[str, Any] | None:
        if template is None:
            return None
        return template.to_dict()

    @staticmethod
    def _deserialize_template_ref(raw: dict[str, Any] | None) -> Template | None:
        if not isinstance(raw, dict):
            return None
        return Template.from_dict(raw)

    @staticmethod
    def _serialize_sql_generation_outcome(gen_out: SqlGenerationOutcome) -> dict[str, Any]:
        union_path = (
            gen_out.generation_path.value
            if isinstance(gen_out.generation_path, GenerationPath)
            else gen_out.generation_path
        )
        return {
            "sql": gen_out.sql,
            "success": gen_out.success,
            "generation_path": union_path,
            "matched_template": MainSessionSerdeOps._serialize_template_ref(gen_out.matched_template),
            "join_matches_template": gen_out.join_matches_template,
            "error_kind": gen_out.error_kind,
            "refusal_diagnostic_code": gen_out.refusal_diagnostic_code,
            "federation_plan_id": gen_out.federation_plan_id,
        }

    @staticmethod
    def _deserialize_sql_generation_outcome(raw: dict[str, Any]) -> SqlGenerationOutcome:
        matched = MainSessionSerdeOps._deserialize_template_ref(raw.get("matched_template"))
        generation_path = GenerationPath(str(raw.get("generation_path") or GenerationPath.INTENT_DIRECT_MATCH.value))
        return SqlGenerationOutcome(
            sql=str(raw.get("sql") or ""),
            success=bool(raw.get("success", False)),
            generation_path=generation_path,
            matched_template=matched,
            join_matches_template=raw.get("join_matches_template"),
            error_kind=raw.get("error_kind"),
            refusal_diagnostic_code=raw.get("refusal_diagnostic_code"),
            federation_plan_id=str(raw.get("federation_plan_id") or ""),
        )

    @staticmethod
    def _serialize_interactive_tail_snapshot(tail: InteractiveTailSnapshot) -> dict[str, Any]:
        union_sql_path = tail.union_sql_path.value if tail.union_sql_path is not None else None
        return {
            "q_norm": tail.q_norm,
            "intent": tail.intent.to_dict(),
            "schema_terms": sorted(tail.schema_terms),
            "semantic_warnings": list(tail.semantic_warnings),
            "has_union_match": tail.has_union_match,
            "cols_changed": tail.cols_changed,
            "matched_template": MainSessionSerdeOps._serialize_template_ref(tail.matched_template),
            "union_select_cols": list(tail.union_select_cols) if tail.union_select_cols is not None else None,
            "structural_match_templates": [
                MainSessionSerdeOps._serialize_template_ref(tmpl)
                for tmpl in tail.structural_match_templates
                if tmpl is not None
            ],
            "ikey": tail.ikey,
            "intent_sim": tail.intent_sim,
            "union_sql_path": union_sql_path,
            "union_candidate_template_ids": list(tail.union_candidate_template_ids),
            "form_storage": MainSessionSerdeOps._serialize_question_form_storage(tail.form_storage),
            "interpretation": MainSessionSerdeOps._serialize_interpret_plan(tail.interpretation),
        }

    @staticmethod
    def _deserialize_interactive_tail_snapshot(raw: dict[str, Any], *, owner: Any | None) -> InteractiveTailSnapshot:
        refs = MainSessionSerdeOps._owner_learning_refs(owner)
        intent = RuntimeIntent.from_dict(raw.get("intent") or {})
        matched_template = MainSessionSerdeOps._deserialize_template_ref(raw.get("matched_template"))
        structural_raw = raw.get("structural_match_templates") or []
        structural_match_templates = tuple(
            tmpl
            for tmpl in (MainSessionSerdeOps._deserialize_template_ref(item) for item in structural_raw)
            if tmpl is not None
        )
        union_path_raw = raw.get("union_sql_path")
        union_sql_path = GenerationPath(str(union_path_raw)) if union_path_raw else None
        union_select_raw = raw.get("union_select_cols")
        union_select_cols = tuple(union_select_raw) if isinstance(union_select_raw, list) else None
        return InteractiveTailSnapshot(
            q_norm=str(raw.get("q_norm") or ""),
            intent=intent,
            schema=refs["schema"],
            store=refs["store"],
            templates=refs["templates"],
            rejected=refs["rejected"],
            schema_terms=set(raw.get("schema_terms") or ()),
            dialect=refs["dialect"],
            semantic_warnings=tuple(raw.get("semantic_warnings") or ()),
            has_union_match=bool(raw.get("has_union_match", False)),
            cols_changed=bool(raw.get("cols_changed", False)),
            matched_template=matched_template,
            union_select_cols=union_select_cols,
            structural_match_templates=structural_match_templates,
            ikey=str(raw.get("ikey") or ""),
            intent_sim=float(raw.get("intent_sim") or 0.0),
            union_sql_path=union_sql_path,
            union_candidate_template_ids=tuple(raw.get("union_candidate_template_ids") or ()),
            form_storage=MainSessionSerdeOps._deserialize_question_form_storage(raw.get("form_storage")),
            interpretation=MainSessionSerdeOps._deserialize_interpret_plan(raw.get("interpretation")),
        )

    @staticmethod
    def _serialize_pipeline_suspend_payload(state_id: str, payload: Any | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        if state_id == PIPELINE_SUSPEND_ID_SQL and isinstance(payload, SqlFeedbackSuspendContext):
            return {
                "type": "sql_feedback",
                "sql": payload.sql,
                "preview_rows": [list(row) for row in payload.preview_rows],
                "sql_parameters": [list(pair) for pair in payload.sql_parameters],
                "suspended_at": payload.suspended_at.isoformat() if payload.suspended_at is not None else None,
                "tmpl_sd": payload.tmpl_sd,
                "force_feedback": payload.force_feedback,
                "execution_intent": payload.execution_intent.to_dict(),
                "tail": MainSessionSerdeOps._serialize_interactive_tail_snapshot(payload.tail),
                "gen_out": MainSessionSerdeOps._serialize_sql_generation_outcome(payload.gen_out),
            }
        if state_id == PIPELINE_SUSPEND_ID_DIRECT_REUSE and isinstance(payload, DirectReuseSuspendContext):
            reuse_path = (
                payload.reuse_path.value if isinstance(payload.reuse_path, GenerationPath) else payload.reuse_path
            )
            return {
                "type": "direct_reuse",
                "q_norm": payload.q_norm,
                "sql": payload.sql,
                "display_sql": payload.display_sql,
                "rows": [list(row) for row in payload.rows],
                "headers": list(payload.headers) if payload.headers is not None else None,
                "is_exact": payload.is_exact,
                "reuse_path": reuse_path,
                "sd_reuse": payload.sd_reuse,
                "intent": payload.intent.to_dict(),
                "ref_tmpl": MainSessionSerdeOps._serialize_template_ref(payload.ref_tmpl),
                "form_storage": MainSessionSerdeOps._serialize_question_form_storage(payload.form_storage),
            }
        if state_id == PIPELINE_SUSPEND_ID_EXECUTE and isinstance(payload, SqlExecuteSuspendContext):
            return {
                "type": "sql_execute",
                "sql": payload.sql,
                "preview_rows": [list(row) for row in payload.preview_rows],
                "sql_parameters": [list(pair) for pair in payload.sql_parameters],
                "suspended_at": payload.suspended_at.isoformat() if payload.suspended_at is not None else None,
                "tmpl_sd": payload.tmpl_sd,
                "force_feedback": payload.force_feedback,
                "execution_intent": payload.execution_intent.to_dict(),
                "tail": MainSessionSerdeOps._serialize_interactive_tail_snapshot(payload.tail),
                "gen_out": MainSessionSerdeOps._serialize_sql_generation_outcome(payload.gen_out),
                "federation_plan_id": payload.federation_plan_id,
            }
        if state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM and isinstance(payload, InteractiveTailSnapshot):
            return {
                "type": "intent_confirm",
                "tail": MainSessionSerdeOps._serialize_interactive_tail_snapshot(payload),
            }
        raise ConfigError(f"unsupported suspended payload for state_id {state_id!r}")

    @staticmethod
    def _deserialize_pipeline_suspend_payload(
        state_id: str, raw: dict[str, Any] | None, *, owner: Any | None
    ) -> Any | None:
        if not isinstance(raw, dict):
            return None
        payload_type = str(raw.get("type") or "")
        if payload_type == "sql_feedback":
            suspended_at = raw.get("suspended_at")
            suspended_dt = datetime.fromisoformat(suspended_at) if isinstance(suspended_at, str) else None
            tail = MainSessionSerdeOps._deserialize_interactive_tail_snapshot(raw.get("tail") or {}, owner=owner)
            execution_intent = RuntimeIntent.from_dict(raw.get("execution_intent") or tail.intent.to_dict())
            return SqlFeedbackSuspendContext(
                tail=tail,
                execution_intent=execution_intent,
                sql=str(raw.get("sql") or ""),
                preview_rows=tuple(tuple(row) for row in (raw.get("preview_rows") or [])),
                sql_parameters=tuple((str(k), v) for k, v in (raw.get("sql_parameters") or [])),
                suspended_at=suspended_dt,
                tmpl_sd=raw.get("tmpl_sd"),
                gen_out=MainSessionSerdeOps._deserialize_sql_generation_outcome(raw.get("gen_out") or {}),
                matched_rejected_template=None,
                force_feedback=bool(raw.get("force_feedback", False)),
            )
        if payload_type == "direct_reuse":
            refs = MainSessionSerdeOps._owner_learning_refs(owner)
            ref_tmpl = MainSessionSerdeOps._deserialize_template_ref(raw.get("ref_tmpl"))
            if ref_tmpl is None:
                raise ConfigError("direct reuse suspend payload is missing ref_tmpl")
            return DirectReuseSuspendContext(
                q_norm=str(raw.get("q_norm") or ""),
                ref_tmpl=ref_tmpl,
                dialect=refs["dialect"],
                store=refs["store"],
                templates=refs["templates"],
                rejected=refs["rejected"],
                schema=refs["schema"],
                intent=RuntimeIntent.from_dict(raw.get("intent") or {}),
                sql=str(raw.get("sql") or ""),
                rows=tuple(tuple(row) for row in (raw.get("rows") or [])),
                display_sql=str(raw.get("display_sql") or ""),
                headers=tuple(raw.get("headers") or ()) if raw.get("headers") is not None else None,
                is_exact=bool(raw.get("is_exact", False)),
                reuse_path=GenerationPath(str(raw.get("reuse_path") or GenerationPath.EXACT_QUESTION_REUSE.value)),
                sd_reuse=raw.get("sd_reuse"),
                form_storage=MainSessionSerdeOps._deserialize_question_form_storage(raw.get("form_storage")),
            )
        if payload_type == "sql_execute":
            suspended_at = raw.get("suspended_at")
            suspended_dt = datetime.fromisoformat(suspended_at) if isinstance(suspended_at, str) else None
            tail = MainSessionSerdeOps._deserialize_interactive_tail_snapshot(raw.get("tail") or {}, owner=owner)
            execution_intent = RuntimeIntent.from_dict(raw.get("execution_intent") or tail.intent.to_dict())
            return SqlExecuteSuspendContext(
                tail=tail,
                execution_intent=execution_intent,
                sql=str(raw.get("sql") or ""),
                gen_out=MainSessionSerdeOps._deserialize_sql_generation_outcome(raw.get("gen_out") or {}),
                matched_rejected_template=None,
                force_feedback=bool(raw.get("force_feedback", False)),
                tmpl_sd=raw.get("tmpl_sd"),
                preview_rows=tuple(tuple(row) for row in (raw.get("preview_rows") or [])),
                sql_parameters=tuple((str(k), v) for k, v in (raw.get("sql_parameters") or [])),
                suspended_at=suspended_dt,
                federation_plan_id=str(raw.get("federation_plan_id") or ""),
            )
        if payload_type == "intent_confirm":
            return MainSessionSerdeOps._deserialize_interactive_tail_snapshot(raw.get("tail") or {}, owner=owner)
        if payload_type == "empty":
            return None
        raise ConfigError(f"unsupported suspended payload type {payload_type!r} for state_id {state_id!r}")

    @staticmethod
    def serialize_suspended_state(
        state_id: str,
        message: str,
        choice_queue: list[tuple[str, str]],
        turn_question: str | None,
        *,
        resume_choice_stage_id: str | None = None,
        suspend_payload: Any | None = None,
        mode: str | None = None,
        space_name: str | None = None,
        data_row_cap: int | None = None,
        policy_ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Capture suspended-session fields for later restoration, including full resume payload."""
        serialized_payload: dict[str, Any]
        suspended_at: datetime
        if suspend_payload is None:
            serialized_payload = {"type": "empty", "state_id": state_id}
            suspended_at = datetime.now(UTC)
        else:
            suspended_at_raw = getattr(suspend_payload, "suspended_at", None)
            if not isinstance(suspended_at_raw, datetime):
                suspended_at = datetime.now(UTC)
            elif suspended_at_raw.tzinfo is None:
                suspended_at = suspended_at_raw.replace(tzinfo=UTC)
            else:
                suspended_at = suspended_at_raw
            serialized_payload = (
                MainSessionSerdeOps._serialize_pipeline_suspend_payload(state_id, suspend_payload) or {}
            )
        payload: dict[str, Any] = {
            "format_version": SUSPEND_STATE_FORMAT_VERSION,
            "state_id": state_id,
            "message": message,
            "choice_queue": [list(pair) for pair in choice_queue],
            "turn_question": turn_question,
            "suspended_at": suspended_at.isoformat(),
            "payload": serialized_payload,
            "policy_ttl_seconds": int(policy_ttl_seconds) if policy_ttl_seconds is not None else None,
        }
        if resume_choice_stage_id is not None:
            payload["resume_choice_stage_id"] = resume_choice_stage_id
        if mode is not None:
            payload["mode"] = mode
        if space_name is not None:
            payload["space_name"] = space_name
        if data_row_cap is not None:
            payload["data_row_cap"] = int(data_row_cap)
        return payload

    @staticmethod
    def deserialize_suspended_state(payload: dict[str, Any], *, owner: Any | None = None) -> dict[str, Any]:
        """Rebuild suspended-session fields from *payload*, refusing on version mismatch or hollow payload."""
        MainSessionSerdeOps._check_suspend_state_format_version(payload)
        choice_queue_raw = payload.get("choice_queue") or []
        choice_queue = [tuple(pair) for pair in choice_queue_raw]
        turn_question = payload.get("turn_question")
        if turn_question is not None:
            turn_question = str(turn_question)
        resume_choice_stage_id = payload.get("resume_choice_stage_id")
        if resume_choice_stage_id is not None:
            resume_choice_stage_id = str(resume_choice_stage_id)
        mode = payload.get("mode")
        if mode is not None:
            mode = str(mode)
        space_name = payload.get("space_name")
        if space_name is not None:
            space_name = str(space_name)
        data_row_cap = payload.get("data_row_cap")
        if data_row_cap is not None:
            data_row_cap = int(data_row_cap)
        suspend_payload_raw = payload.get("payload")
        if suspend_payload_raw is None:
            suspend_payload_raw = payload.get("suspend_payload")
        if not isinstance(suspend_payload_raw, dict):
            raise ConfigError("suspend state payload is missing or hollow; cannot restore session")
        if str(suspend_payload_raw.get("type") or "") == "empty":
            suspend_payload = None
        else:
            suspend_payload = MainSessionSerdeOps._deserialize_pipeline_suspend_payload(
                str(payload["state_id"]),
                suspend_payload_raw,
                owner=owner,
            )
        suspended_at_raw = payload.get("suspended_at")
        suspended_at: datetime | None = None
        if isinstance(suspended_at_raw, str) and suspended_at_raw.strip():
            try:
                suspended_at = datetime.fromisoformat(suspended_at_raw)
            except ValueError as exc:
                raise ConfigError(f"invalid suspended_at {suspended_at_raw!r}") from exc
            if suspended_at.tzinfo is None:
                suspended_at = suspended_at.replace(tzinfo=UTC)
        if suspended_at is not None and hasattr(suspend_payload, "suspended_at"):
            payload_obj: Any = suspend_payload
            try:
                object.__setattr__(payload_obj, "suspended_at", suspended_at)
            except Exception:
                payload_obj.suspended_at = suspended_at
        policy_ttl = payload.get("policy_ttl_seconds")
        if policy_ttl is not None:
            policy_ttl = int(policy_ttl)
        result: dict[str, Any] = {
            "state_id": str(payload["state_id"]),
            "message": str(payload.get("message") or ""),
            "choice_queue": choice_queue,
            "turn_question": turn_question,
            "resume_choice_stage_id": resume_choice_stage_id,
            "suspend_payload": suspend_payload,
            "suspended_at": suspended_at,
            "policy_ttl_seconds": policy_ttl,
        }
        if mode is not None:
            result["mode"] = mode
        if space_name is not None:
            result["space_name"] = space_name
        if data_row_cap is not None:
            result["data_row_cap"] = data_row_cap
        return result

    @staticmethod
    @contextlib.contextmanager
    def _owner_domain_knowledge_scope(owner: Any) -> Any:
        """Bind the owner's stored domain knowledge for nested pipeline work."""
        holder = getattr(owner, "_domain_knowledge", None)
        if not isinstance(holder, DomainKnowledgeHolder):
            yield
            return
        with domain_knowledge_scope(**holder.scope_kwargs()):
            yield

    @staticmethod
    @contextlib.contextmanager
    def _session_domain_knowledge_scope(session: Any) -> Any:
        """Bind caller-scoped domain knowledge for ask-time parsing."""
        owner = getattr(session, "_owner", None)
        holder = getattr(owner, "_domain_knowledge", None) if owner is not None else None
        engine_entries: tuple[DomainKnowledgeEntry, ...] = ()
        if isinstance(holder, DomainKnowledgeHolder):
            engine_entries = holder.entries()
        schema = getattr(owner, "_schema_graph", None) if owner is not None else None
        if owner is None or not isinstance(schema, SchemaGraph):
            if isinstance(holder, DomainKnowledgeHolder):
                with domain_knowledge_scope(**holder.scope_kwargs()):
                    yield
            else:
                yield
            return
        scope_ctx, visible, space_tables, space_snapshot = MainInitOps.meta_visibility_knobs(
            owner, schema, None, pipeline_session=session
        )
        scoped = MainSpaceOps.derive_caller_scoped_domain_knowledge(
            engine_entries=engine_entries,
            schema=schema,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_snapshot=space_snapshot,
            space_tables=space_tables,
        )
        digest = DomainKnowledgeState.digest_for(scoped)
        with domain_knowledge_scope(entries=scoped, digest=digest):
            yield


class PipelineSession(PipelineSessionMarker, InteractiveChoicePort):
    """Programmatic driver for one interactive turn at a time via ask and step. When used as the interactive choice port, the internal pipeline calls :meth:`has_pending_choice` and :meth:`take_yes_no`. :meth:`note_turn_outcome` records the latest turn for :meth:`step` consumers. Builtin ``dir`` on this class lists only ask, ask_until_done, awaiting_prompt, reset, and step. Writer turns take the owner's ``_pipeline_writer_lock`` only around artifact mutations and write-queue drains, releasing it across model calls and database execution; reader turns never take that lock. Only one turn may be in flight per session instance at a time."""

    __slots__ = (
        "_owner",
        "_choice_queue",
        "_suspended",
        "_resume_choice_stage_id",
        "_last_turn_outcome",
        "_session_busy",
        "_session_busy_lock",
        "_refinement_ctx",
        "_session_mode",
        "_visible_objects",
        "_execution_visible_objects",
        "_space_name",
        "_space_tables",
        "_space_columns",
        "_space_deny_objects",
        "_space_deny_columns",
        "_space_description_overlay",
        "_turn_question",
        "_pending_conversation_rejection_hints",
        "_turn_llm_usage_start",
        "_turn_llm_scope_tok",
        "_turn_accumulated_diagnostics",
        "_turn_cancel_event",
        "_data_row_cap",
        "active_federation_execution_context",
        "_pending_federation_plan_template",
        "_pending_terminal_step",
    )

    def __init__(
        self,
        owner: Any,
        *,
        mode: Literal["reader", "writer"] = "writer",
        visible_objects: frozenset[str] | None = None,
        execution_visible_objects: frozenset[str] | None = None,
        space_name: str = "master",
        space_tables: frozenset[str] | None = None,
        space_columns: frozenset[str] | None = None,
        space_deny_objects: frozenset[str] | None = None,
        space_deny_columns: frozenset[str] | None = None,
        space_description_overlay: dict[str, Any] | None = None,
        data_row_cap: int | None = None,
    ) -> None:
        if mode not in ("reader", "writer"):
            raise ValueError("mode must be 'reader' or 'writer'")
        self._owner = owner
        self._session_mode = mode
        self._visible_objects = visible_objects
        self._execution_visible_objects = execution_visible_objects
        self._space_name = space_name
        self._space_tables = space_tables if space_tables else frozenset()
        self._space_columns = space_columns if space_columns else frozenset()
        self._space_deny_objects = space_deny_objects if space_deny_objects else frozenset()
        self._space_deny_columns = space_deny_columns if space_deny_columns else frozenset()
        self._space_description_overlay = space_description_overlay
        self._choice_queue: deque[tuple[str, str]] = deque()
        self._suspended: PipelineSuspended | None = None
        self._restored_suspended_at: datetime | None = None
        self._restored_policy_ttl_seconds: int | None = None
        self._resume_choice_stage_id: str | None = None
        self._last_turn_outcome: dict[str, Any] | None = None
        self._session_busy = False
        self._session_busy_lock = threading.Lock()
        self._refinement_ctx: RefinementContext | None = None
        self._turn_question: str | None = None
        self._pending_conversation_rejection_hints: tuple[str, ...] = ()
        self._turn_llm_usage_start = 0
        self._turn_llm_scope_tok: Any = None
        self._turn_llm_usage_summary: Any = None
        self._turn_accumulated_diagnostics: list[Diagnostic] = []
        self._turn_cancel_event = threading.Event()
        self._data_row_cap = int(data_row_cap) if data_row_cap is not None and int(data_row_cap) > 0 else None
        self.active_federation_execution_context: FederationExecutionContext | None = None
        self._pending_federation_plan_template: FederationPlanTemplate | None = None
        self._pending_terminal_step: SessionStep | None = None

    def _session_schema_role(self) -> SchemaRole:
        role = getattr(self._owner, "_schema_role", SchemaRole.OWNER)
        if role == SchemaRole.CONSUMER or role == "consumer":
            return SchemaRole.CONSUMER
        return SchemaRole.OWNER

    def _egress_session_step(self, step: SessionStep) -> SessionStep:
        if self._session_schema_role() != SchemaRole.CONSUMER:
            return step
        return sanitize_session_step_for_egress(step)

    def _acquire_session_turn(self) -> None:
        """Mark this session as busy under the session lock."""
        with self._session_busy_lock:
            if self._session_busy:
                raise SessionActiveError("Cannot start a new question while a turn is in progress.")
            self._session_busy = True

    def _release_session_turn(self) -> None:
        """Clear the busy flag under the session lock."""
        with self._session_busy_lock:
            self._session_busy = False

    @property
    def visible_objects(self) -> frozenset[str] | None:
        """When set, names tables included in the intent-stage effective context payload (aetherspace only)."""
        return self._visible_objects

    @property
    def execution_visible_objects(self) -> frozenset[str] | None:
        """When set, names tables granted to a consumer at execution time."""
        return self._execution_visible_objects

    @property
    def space_name(self) -> str:
        """Active aetherspace name for this session."""
        return self._space_name

    @property
    def space_tables(self) -> frozenset[str]:
        """Allowed tables for the active aetherspace (empty means unrestricted)."""
        return self._space_tables

    @property
    def space_columns(self) -> frozenset[str]:
        """Allowed qualified columns for the active aetherspace (empty means unrestricted)."""
        return self._space_columns

    @property
    def space_deny_objects(self) -> frozenset[str]:
        """Denied tables/views for the active aetherspace payload and execution gates."""
        return self._space_deny_objects

    @property
    def space_deny_columns(self) -> frozenset[str]:
        """Denied qualified columns for the active aetherspace payload and execution gates."""
        return self._space_deny_columns

    @property
    def space_description_overlay(self) -> dict[str, Any] | None:
        """Per-space table/column description overlay from the aetherspace snapshot."""
        return self._space_description_overlay

    def _apply_data_row_cap(self, data_out: pandas.DataFrame | None) -> tuple[pandas.DataFrame | None, bool]:
        """Trim tabular step data to the session row cap when configured."""
        if data_out is None or self._data_row_cap is None:
            return data_out, False
        if len(data_out) <= self._data_row_cap:
            return data_out, False
        return data_out.head(self._data_row_cap), True

    def export_serialized_state(self) -> dict[str, Any]:
        """Return a versioned serialisable snapshot of suspend state for this session."""
        if self._suspended is None:
            raise ConfigError("no suspended session state to export")
        limits = getattr(self._owner, "limits", None)
        ttl = getattr(limits, "suspended_session_ttl_seconds", None) if limits is not None else None
        return MainSessionSerdeOps.serialize_suspended_state(
            self._suspended.state_id,
            self._suspended.message_for_caller,
            list(self._choice_queue),
            self._turn_question,
            resume_choice_stage_id=self._resume_choice_stage_id,
            suspend_payload=self._suspended.payload,
            mode=self._session_mode,
            space_name=self._space_name,
            data_row_cap=self._data_row_cap,
            policy_ttl_seconds=ttl,
        )

    @classmethod
    def restore_serialized_state(cls, owner: Any, payload: dict[str, Any]) -> PipelineSession:
        """Rebuild a session from :meth:`export_serialized_state` output."""
        fields = MainSessionSerdeOps.deserialize_suspended_state(payload, owner=owner)
        mode_raw = fields.get("mode")
        if mode_raw not in ("reader", "writer"):
            raise ConfigError("restored session payload missing mode")
        mode = cast(Literal["reader", "writer"], mode_raw)
        space_name = fields.get("space_name") or str(
            getattr(owner, "_active_space_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME
        )
        session_factory = getattr(owner, "session", None)
        if not callable(session_factory):
            raise ConfigError("restore_serialized_state requires an engine with session()")
        sess = cast(
            PipelineSession,
            session_factory(
                mode=mode,
                space=str(space_name),
                data_row_cap=fields.get("data_row_cap"),
            ),
        )
        sess._suspended = PipelineSuspended(
            fields["state_id"],
            fields["message"],
            fields.get("suspend_payload"),
        )
        sess._choice_queue = deque(fields["choice_queue"])
        sess._turn_question = fields["turn_question"]
        sess._resume_choice_stage_id = fields["resume_choice_stage_id"]
        sess._restored_suspended_at = fields.get("suspended_at")
        sess._restored_policy_ttl_seconds = fields.get("policy_ttl_seconds")
        sess._session_busy = True
        return sess

    def _audit_ask_emit(
        self, event_type: str, *, question: str | None = None, details: tuple[tuple[str, str], ...] = ()
    ) -> None:
        fn = getattr(self._owner, "_audit_emit", None)
        if not callable(fn):
            return
        if self._session_schema_role() == SchemaRole.CONSUMER:
            details = sanitize_audit_details_for_egress(details)
        owner_schema = getattr(self._owner, "_schema_graph", None)
        schema_hash_val: str | None = None
        if owner_schema is not None:
            schema_hash_val = getattr(owner_schema, "effective_structural_hash", None)
        fn(
            event_type,
            question=question,
            schema_hash=schema_hash_val,
            details=details_with_turn_id(details),
            turn_id=active_turn_id(),
        )

    def _turn_llm_usage_records(self) -> tuple[Any, ...]:
        records = snapshot_llm_usage_records()
        if self._turn_llm_usage_start >= len(records):
            return ()
        return records[self._turn_llm_usage_start :]

    def _emit_turn_llm_usage(
        self, *, question: str | None, diagnostics: tuple[Diagnostic, ...] = ()
    ) -> tuple[Diagnostic, ...]:
        """Emit per-turn LLM audit events and advance the turn cursor without draining usage. Live and sandbox invoice flush snapshot the session accumulator after each ask. Draining here would wipe intent and default usage so invoices keep only schema noise."""
        records = self._turn_llm_usage_records()
        if not records:
            self._turn_llm_usage_summary = None
            return diagnostics
        provider_raw = str(getattr(getattr(self._owner, "_llm_config", None), "provider", "openai"))
        provider: Literal["openai", "azure", "sandbox"]
        normalized = EngineConfig.normalize_llm_provider(provider_raw)
        if normalized in ("openai", "azure", "sandbox"):
            provider = cast(Literal["openai", "azure", "sandbox"], normalized)
        else:
            provider = "openai"
        self._turn_llm_usage_summary = summarize_llm_turn_usage(records, provider=provider)
        for record in records:
            self._audit_ask_emit("llm_call", question=question, details=llm_call_audit_details(record))
        self._audit_ask_emit("llm_turn", question=question, details=llm_turn_audit_details(records, provider=provider))
        cost_diag = llm_turn_cost_diagnostic(records, provider=provider)
        out = diagnostics
        if cost_diag is not None:
            out = diagnostics + (cost_diag,)
        self._turn_llm_usage_start = len(snapshot_llm_usage_records())
        return out

    def _extend_turn_accumulated_diagnostics(self, merged: tuple[Diagnostic, ...]) -> None:
        """Append *merged* to the active turn accumulator, deduping by (code, phase, subject)."""
        index: dict[tuple[str, Any, Any], int] = {}
        for i, existing in enumerate(self._turn_accumulated_diagnostics):
            key = (
                existing.code,
                getattr(existing, "phase", None),
                getattr(existing, "subject", None),
            )
            index[key] = i
        for d in merged:
            key = (d.code, getattr(d, "phase", None), getattr(d, "subject", None))
            if key in index:
                i = index[key]
                retained = self._turn_accumulated_diagnostics[i]
                new_count = getattr(retained, "count", 1) + getattr(d, "count", 1)
                self._turn_accumulated_diagnostics[i] = replace(retained, count=new_count)
                continue
            index[key] = len(self._turn_accumulated_diagnostics)
            self._turn_accumulated_diagnostics.append(d)

    def _terminal_turn_diagnostics(self, turn_diagnostics: tuple[Diagnostic, ...] = ()) -> tuple[Diagnostic, ...]:
        """Diagnostics for a terminal :class:`SessionStep` (full turn accumulation plus *turn_diagnostics*)."""
        accumulated = getattr(self, "_turn_accumulated_diagnostics", ())
        return tuple(accumulated) + turn_diagnostics

    def _mk_step(self, *, diagnostics: tuple[Diagnostic, ...] = (), **kw: Any) -> SessionStep:
        """Build a :class:`SessionStep`, attaching drained pipeline diagnostics."""
        drained = drain_diagnostic_collector()
        merged = diagnostics + drained
        self._extend_turn_accumulated_diagnostics(merged)
        merged_kw = dict(kw)
        merged_kw["diagnostics"] = merged
        if merged_kw.get("done") and "llm_usage" not in merged_kw:
            merged_kw["llm_usage"] = getattr(self, "_turn_llm_usage_summary", None)
        if merged_kw.get("done"):
            if "turn_id" not in merged_kw:
                merged_kw["turn_id"] = active_turn_id()
            if "elapsed_ms" not in merged_kw:
                merged_kw["elapsed_ms"] = turn_elapsed_ms()
        return SessionStep(**merged_kw)

    def _attach_refinement_ctx(self, ctx: RefinementContext | None) -> None:
        """Bind :class:`RefinementContext` for silent in-turn retries after user rejection."""
        self._refinement_ctx = ctx

    def _resources(
        self,
    ) -> tuple[SchemaGraph, dict[str, Any] | TemplateStoreView, dict[str, Any], dict[str, Any], set[str]]:
        """Return the schema graph and template backing structures from the owning facade."""
        owner = self._owner
        space_name = TemplateOps.validate_space_name(self._space_name)
        cached_store = MainSpaceOps.owner_template_store_for_space(owner, space_name)
        store: dict[str, Any] | TemplateStoreView
        if isinstance(cached_store, TemplateStoreView):
            store = cached_store
        elif isinstance(cached_store, dict) and cached_store:
            store = cached_store
        else:
            graph_id = str(getattr(owner._schema_graph, "schema_graph_id", "") or "")
            raw_ad = getattr(owner, "_artifacts_dir", None)
            if isinstance(raw_ad, (str, Path)):
                store = TemplateOps.load_template_store(
                    graph_id, owner._schema_graph, space_name=space_name, artifacts_dir=str(raw_ad)
                )
            else:
                store = TemplateOps.load_template_store(graph_id, owner._schema_graph, space_name=space_name)
            MainSpaceOps.sync_owner_template_cache(owner, store, space_name=space_name)
        templates_by_space = getattr(owner, "_templates_by_space", None)
        if isinstance(templates_by_space, dict) and space_name in templates_by_space:
            templates = templates_by_space[space_name]
        else:
            templates = TemplateOps.store_to_templates(store)
            if isinstance(templates_by_space, dict):
                templates_by_space[space_name] = templates
        return (owner._schema_graph, store, templates, owner._rejected, owner._schema_terms)

    def __dir__(self) -> list[str]:
        """Return names intended for interactive discovery."""
        return sorted(("accept_until_done", "ask", "ask_until_done", "awaiting_prompt", "reset", "step"))

    def __enter__(self) -> PipelineSession:
        """Return *self* for ``with`` blocks."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        """Cancel any in-flight work and reset partial turn state when leaving a ``with`` block."""
        self.cancel()
        self.reset()
        return False

    def _reset_after_turn(self) -> None:
        """Clear partial turn state after a completed or abandoned interactive pass."""
        if self._turn_llm_scope_tok is not None:
            reset_turn_llm_scope(self._turn_llm_scope_tok)
            self._turn_llm_scope_tok = None
        self._turn_llm_usage_start = 0
        self._turn_llm_usage_summary = None
        self._turn_accumulated_diagnostics = []
        self._choice_queue.clear()
        self._suspended = None
        self._resume_choice_stage_id = None
        self._last_turn_outcome = None
        self._refinement_ctx = None
        self._turn_question = None
        self._pending_conversation_rejection_hints = ()
        self._turn_cancel_event.clear()
        self.active_federation_execution_context = None
        self._pending_federation_plan_template = None
        self._pending_terminal_step = None

    def reset(self) -> None:
        """Clear suspend state, queued programmatic answers, and partial turn state."""
        self._reset_after_turn()
        self._release_session_turn()

    def cancel(self) -> bool:
        """Cancel the in-flight turn owned by this session when one is in progress. Safe to call from another thread. Cancellation is cooperative: federation workers observe it between member stages or batches; non-federated work observes it at pipeline checkpoints."""
        if self._suspended is not None:
            self._turn_cancel_event.set()
            ctx = self.active_federation_execution_context
            if ctx is not None:
                ctx.cancel()
            self._pending_terminal_step = self._terminal_cancelled_step("Turn cancelled.")
            return True
        cancelled = False
        with self._session_busy_lock:
            busy = self._session_busy
        if busy:
            self._turn_cancel_event.set()
            dialect = getattr(self._owner, "_dialect", None)
            if dialect is not None:
                Dialect.cancel_in_flight_statement(dialect)
            cancelled = True
        ctx = self.active_federation_execution_context
        if ctx is not None:
            ctx.cancel()
            cancelled = True
        return cancelled

    def note_turn_outcome(
        self,
        *,
        outcome: str,
        error: str | None = None,
        sql: str | dict[str, str] | None = None,
        rows: list[tuple[Any, ...]] | None = None,
        columns: tuple[str, ...] | None = None,
        rejection_bucket: str | None = None,
        intent: RuntimeIntent | None = None,
        matched_template: Template | None = None,
        template_history_index: int | None = None,
        federated_bundle: Any | None = None,
        federation_source_id: str | None = None,
        federation_phase: str | None = None,
        federation_succeeded: tuple[tuple[str, int, str], ...] | None = None,
        failure_kind: str | None = None,
        federated_plan: FederatedPlan | None = None,
        generation_path: GenerationPath | None = None,
        retryable: bool | None = None,
        refusal_diagnostic_code: str | None = None,
    ) -> None:
        """Store the latest interactive turn outcome for :meth:`step` consumers."""
        self._last_turn_outcome = {
            "outcome": outcome,
            "error": error,
            "sql": sql,
            "rows": rows,
            "columns": list(columns) if columns is not None else None,
            "rejection_bucket": rejection_bucket,
            "intent": intent,
            "matched_template": matched_template,
            "template_history_index": template_history_index,
            "federated_bundle": federated_bundle,
            "federated_plan": federated_plan,
            "generation_path": generation_path,
            "federation_source_id": federation_source_id,
            "federation_phase": federation_phase,
            "federation_succeeded": tuple(federation_succeeded or ()),
            "failure_kind": failure_kind,
            "retryable": retryable,
            "refusal_diagnostic_code": refusal_diagnostic_code,
        }

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""
        return len(self._choice_queue) > 0

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Pop and normalise the next queued token against *options*."""
        if not self._choice_queue:
            raise PipelineSuspended("empty_choice_queue", "interactive choice queue is empty", None)
        sid_token, raw = self._choice_queue.popleft()
        expected = self._resume_choice_stage_id
        if expected is not None and sid_token != expected:
            raise PipelineSuspended(
                "choice_queue_mismatch", f"queued answer targeted {sid_token!r} but expected {expected!r}", None
            )
        return MainInitOps.normalise_yes_no(raw, options)

    def _consume_next_queued_choice(self) -> str | None:
        """Remove one raw queued token for resume paths that do not use :meth:`take_yes_no`."""
        if not self._choice_queue:
            return None
        sid_token, raw = self._choice_queue.popleft()
        expected = self._resume_choice_stage_id
        if expected is not None and sid_token != expected:
            raise PipelineSuspended(
                "choice_queue_mismatch", f"queued answer targeted {sid_token!r} but expected {expected!r}", None
            )
        return raw

    def awaiting_prompt(self) -> bool:
        """Return True when the session is waiting on :meth:`step` input."""
        return self._suspended is not None

    def _owner_engine_identity(self) -> EngineIdentity:
        """Resolve the owning engine identity for session-scoped diagnostic routing."""
        identity = getattr(self._owner, "_engine_identity", None)
        if isinstance(identity, EngineIdentity):
            return identity
        dialect_obj = getattr(self._owner, "_dialect", None)
        if isinstance(dialect_obj, str):
            engine_type = dialect_obj.strip().lower()
            runtime_cfg = getattr(self._owner, "_runtime_config", None)
            if runtime_cfg is None or isinstance(runtime_cfg, type):
                runtime_cfg = DialectRegistry.get_runtime_config_class(engine_type)()
            return EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
        if dialect_obj is not None:
            runtime_cfg = getattr(dialect_obj, "config", None)
            if runtime_cfg is None or isinstance(runtime_cfg, type):
                raise RuntimeError("engine session requires a per-engine runtime configuration instance")
            engine_type = str(getattr(dialect_obj, "name", getattr(self._owner, "dialect", "")))
            return EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
        runtime_cfg = getattr(self._owner, "_runtime_config", None)
        if runtime_cfg is None or isinstance(runtime_cfg, type):
            raise RuntimeError("engine session requires a per-engine runtime configuration instance")
        return EngineIdentity(engine_type=str(self._owner.dialect), runtime_config=runtime_cfg)

    def ask(self, question: str) -> SessionStep:
        """Start a new NL turn and return the first :class:`SessionStep` (prompt, result, or error)."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        if getattr(self._owner, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        if not question.strip():
            self._audit_ask_emit(
                "ask_blocked",
                question=question,
                details=(("reason", "empty_question"),),
            )
            st = self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error=SessionError(
                    code=SessionOutcome.NOT_A_QUESTION,
                    detail_code=DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
                ),
            )
            return st
        with self._session_busy_lock:
            if self._session_busy:
                self._audit_ask_emit(
                    "ask_blocked",
                    question=question,
                    details=(("reason", "session_active"),),
                )
                raise SessionActiveError("Cannot start a new question while a turn is in progress.")
            self._session_busy = True
        buf = diagnostic_segment()
        for _orph in take_and_clear_orphan_diagnostics(self._owner_engine_identity()):
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        self._pending_conversation_rejection_hints = ()
        try:
            return self._egress_session_step(self._drive_question_turn(question))
        finally:
            reset_diagnostic_collector(tok)

    def ask_until_done(self, question: str, *, on_confirm: Literal["y", "n"] = "y") -> SessionStep:
        """Run ``ask`` then auto-answer yes or no suspends with *on_confirm* until the turn ends."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        MainInitOps.validate_yes_no_reply_token(on_confirm, param="on_confirm")
        step = self.ask(question)
        while not step.done:
            if step.reply_shape != "yes_no":
                raise SessionActiveError(f"free-text suspend at kind={step.kind}; ask_until_done cannot answer")
            step = self.step(on_confirm)
        return self._egress_session_step(step)

    def accept_until_done(
        self, question: str, *, on_yes_no: Literal["y", "n"] = "y", on_free_text: str = "looks good"
    ) -> SessionStep:
        """Auto-answer yes-or-no and free-text suspends until the turn ends. Intended for sandbox tours and quick demos where every prompt can be confirmed automatically."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        MainInitOps.validate_yes_no_reply_token(on_yes_no, param="on_yes_no")
        step = self.ask(question)
        while not step.done:
            if step.reply_shape == "yes_no":
                step = self.step(on_yes_no)
            elif step.reply_shape == "free_text":
                step = self.step(on_free_text)
            else:
                break
        return self._egress_session_step(step)

    def _enforce_suspended_session_ttl(self) -> None:
        """Raise and reset when a deferred turn exceeds the configured suspension TTL."""
        suspended = self._suspended
        if suspended is None:
            return
        limits = getattr(self._owner, "limits", None)
        ttl = self._restored_policy_ttl_seconds
        if ttl is None:
            ttl = getattr(limits, "suspended_session_ttl_seconds", None) if limits is not None else None
        if ttl is None:
            return
        payload = suspended.payload
        suspended_at = self._restored_suspended_at
        if suspended_at is None:
            suspended_at = getattr(payload, "suspended_at", None)
        if suspended_at is None:
            return
        if suspended_at.tzinfo is None:
            suspended_at = suspended_at.replace(tzinfo=UTC)
        age_seconds = (datetime.now(UTC) - suspended_at).total_seconds()
        if age_seconds <= float(ttl):
            return
        self.reset()
        raise SuspendedSessionExpiredError(
            f"suspended session expired after {int(ttl)} seconds; call ask() to start a new turn"
        )

    def step(self, response: str | None = None) -> SessionStep:
        """Supply the next user answer for a suspended prompt."""
        if getattr(self._owner, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        self._enforce_suspended_session_ttl()
        buf = diagnostic_segment()
        for _orph in take_and_clear_orphan_diagnostics(self._owner_engine_identity()):
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        try:
            pending = self._pending_terminal_step
            if pending is not None:
                self._pending_terminal_step = None
                return self._egress_session_step(pending)
            if self._suspended is not None:
                return self._egress_session_step(self._step_pipeline_suspend(response or ""))
            if not self._session_busy:
                return self._egress_session_step(
                    self._mk_step(
                        done=True,
                        prompt=None,
                        kind=SESSION_KIND_IDLE,
                        error=SessionError(code=SessionOutcome.INTERNAL_ERROR),
                    )
                )
            return self._egress_session_step(
                self._mk_step(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_ERROR,
                    error=SessionError(code=SessionOutcome.INTERNAL_ERROR),
                )
            )
        finally:
            reset_diagnostic_collector(tok)

    def _step_pipeline_suspend(self, raw: str) -> SessionStep:
        """Resume the pipeline after :exc:`PipelineSuspended` using *raw* as the next answer."""
        if self._suspended is not None and self._suspended.state_id == PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT:
            text = (raw or "").strip()
            if not text:
                return self._mk_step(
                    done=False,
                    prompt=SESSION_PROMPT_REASON,
                    kind=SUSPEND_ID_TO_SESSION_KIND.get(PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT, SESSION_KIND_ERROR),
                    reply_shape="free_text",
                )
            self._choice_queue.append((PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT, text))
            return self._resume_from_suspend()
        if self._suspended is not None and self._suspended.state_id == PIPELINE_SUSPEND_ID_INTENT_FEEDBACK:
            text = (raw or "").strip()
            if not text:
                return self._mk_step(
                    done=False,
                    prompt=SESSION_PROMPT_REASON,
                    kind=SUSPEND_ID_TO_SESSION_KIND.get(PIPELINE_SUSPEND_ID_INTENT_FEEDBACK, SESSION_KIND_ERROR),
                    reply_shape="free_text",
                )
            self._choice_queue.append((PIPELINE_SUSPEND_ID_INTENT_FEEDBACK, text))
            return self._resume_from_suspend()
        suspended = self._suspended
        if suspended is None:
            return self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error=SessionError(code=SessionOutcome.INTERNAL_ERROR),
            )
        normalised = MainInitOps.normalise_yes_no(raw, ["y", "n"])
        if normalised is None:
            kind = SUSPEND_ID_TO_SESSION_KIND.get(suspended.state_id, SESSION_KIND_ERROR)
            return self._mk_step(
                done=False,
                prompt=SESSION_PROMPT_YESNO,
                kind=kind,
                reply_shape="yes_no",
            )
        sid = suspended.state_id
        self._choice_queue.append((sid, normalised))
        return self._resume_from_suspend()

    def _suspend_to_step(self, ex: PipelineSuspended) -> SessionStep:
        """Build a :class:`SessionStep` describing a deferred pipeline prompt."""
        kind = SUSPEND_ID_TO_SESSION_KIND.get(ex.state_id, SESSION_KIND_ERROR)
        payload = ex.payload
        sql_out: str | dict[str, str] | None = None
        data_out: pandas.DataFrame | None = None
        prompt_out = SESSION_PROMPT_YESNO
        isum: IntentSummary | None = None
        reply_shape: Literal["yes_no", "free_text"] | None = "yes_no"
        sem_w: tuple[str, ...] = ()
        parameters: tuple[ParameterBinding, ...] = ()
        template_id_out: str | None = None
        matched_for_params: Template | None = None
        intent_for_params: RuntimeIntent | None = None

        if ex.state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM and isinstance(payload, InteractiveTailSnapshot):
            _, sem_w = compose_intent_confirm_session_message(
                payload.intent,
                list(payload.semantic_warnings),
                approach=getattr(payload.interpretation, "approach", None) if payload.interpretation else None,
            )
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainInteractiveOps.build_intent_summary(payload.intent)
            reply_shape = "yes_no"
        elif ex.state_id == PIPELINE_SUSPEND_ID_EXECUTE and isinstance(payload, SqlExecuteSuspendContext):
            ctx_exec = payload
            sql_out = MainSpaceOps.resolved_session_step_sql(
                ctx_exec.sql,
                gen_out=ctx_exec.gen_out,
                federated_bundle=getattr(ctx_exec, "federated_bundle", None),
                federated_plan=(ctx_exec.federated_prepare.plan if ctx_exec.federated_prepare is not None else None),
                generation_path=ctx_exec.gen_out.generation_path,
            )
            if sql_out is None:
                sql_out = ctx_exec.sql
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainInteractiveOps.build_intent_summary(ctx_exec.execution_intent)
            reply_shape = "yes_no"
            sem_w = ()
            matched_for_params = ctx_exec.gen_out.matched_template
            intent_for_params = ctx_exec.execution_intent
        elif ex.state_id == PIPELINE_SUSPEND_ID_SQL and isinstance(payload, SqlFeedbackSuspendContext):
            ctxp = payload
            sql_out = MainSpaceOps.resolved_session_step_sql(
                ctxp.sql,
                gen_out=ctxp.gen_out,
                federated_bundle=ctxp.federated_bundle,
                federated_plan=ctxp.federated_prepare.plan if ctxp.federated_prepare is not None else None,
                generation_path=ctxp.gen_out.generation_path,
            )
            if sql_out is None:
                sql_out = ctxp.sql
            preview = list(ctxp.preview_rows)
            full_df = build_result_dataframe(
                preview,
                ctxp.execution_intent,
                ctxp.sql if isinstance(ctxp.sql, str) else "",
                structural_defaults=ctxp.tmpl_sd,
                q_norm=ctxp.tail.q_norm,
                template_display_alias_map=(
                    getattr(ctxp.gen_out.matched_template, "display_alias_map", None)
                    if ctxp.gen_out.matched_template
                    else None
                ),
                **MainSpaceOps.federation_result_contract_kwargs(
                    ctxp.gen_out, federated_prepare=ctxp.federated_prepare, federated_bundle=ctxp.federated_bundle
                ),
            )
            if full_df is not None:
                data_out = full_df.head(10)
            elif preview:
                alias_map = (
                    getattr(ctxp.gen_out.matched_template, "display_alias_map", None)
                    if ctxp.gen_out.matched_template
                    else None
                )
                hdrs = intent_result_column_headers(
                    ctxp.execution_intent,
                    row_width=len(preview[0]),
                    template_display_alias_map=alias_map,
                )
                if not hdrs:
                    maybe_hdrs = result_columns_for_session(
                        ctxp.sql if isinstance(ctxp.sql, str) else "",
                        preview,
                        intent=ctxp.execution_intent,
                        **MainSpaceOps.federation_result_contract_kwargs(
                            ctxp.gen_out,
                            federated_prepare=ctxp.federated_prepare,
                            federated_bundle=ctxp.federated_bundle,
                        ),
                    )
                    hdrs = maybe_hdrs or ()
                if hdrs and len(hdrs) == len(preview[0]):
                    data_out = pandas.DataFrame([list(r) for r in preview[:10]], columns=list(hdrs))
                else:
                    data_out = pandas.DataFrame([list(r) for r in preview[:10]])
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainInteractiveOps.build_intent_summary(ctxp.execution_intent)
            reply_shape = "yes_no"
            sem_w = ()
            matched_for_params = ctxp.gen_out.matched_template or ctxp.tail.matched_template
            intent_for_params = ctxp.execution_intent
        elif ex.state_id == PIPELINE_SUSPEND_ID_DIRECT_REUSE and isinstance(payload, DirectReuseSuspendContext):
            ctx = payload
            tmpl_sql = getattr(ctx.ref_tmpl, "sql_param", None)
            sql_out = (
                tmpl_sql if isinstance(tmpl_sql, str) and tmpl_sql.strip() else (ctx.display_sql or ctx.sql or None)
            )
            rows_list = list(ctx.rows)
            hdr = list(ctx.headers) if ctx.headers else None
            if rows_list:
                if hdr and len(hdr) == len(rows_list[0]):
                    data_out = pandas.DataFrame([list(r) for r in rows_list], columns=hdr).head(10)
                else:
                    data_out = pandas.DataFrame([list(r) for r in rows_list]).head(10)
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainInteractiveOps.build_intent_summary(ctx.intent)
            reply_shape = "yes_no"
            sem_w = ()
            matched_for_params = ctx.ref_tmpl if isinstance(ctx.ref_tmpl, Template) else None
            intent_for_params = ctx.intent
        elif ex.state_id == PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT:
            prompt_out = SESSION_PROMPT_REASON
            reply_shape = "free_text"
            sem_w = ()
        elif ex.state_id == PIPELINE_SUSPEND_ID_INTENT_FEEDBACK:
            prompt_out = SESSION_PROMPT_REASON
            reply_shape = "free_text"
            sem_w = ()
        else:
            prompt_out = SESSION_PROMPT_YESNO
            reply_shape = "yes_no"
            sem_w = ()

        if sql_out is not None:
            parameters = self._parameters_for_sql_bearing_step(
                sql=sql_out,
                matched_template=matched_for_params,
                intent=intent_for_params,
                question_nl=self._turn_question or "",
                persist_display_names=False,
            )
            templates_for_id: dict[str, Any] | None = None
            store_for_id: dict[str, Any] | TemplateStoreView | None = None
            q_for_id = ""
            if isinstance(payload, (SqlExecuteSuspendContext, SqlFeedbackSuspendContext)):
                templates_for_id = payload.tail.templates
                store_for_id = payload.tail.store
                q_for_id = payload.tail.q_norm
            elif isinstance(payload, DirectReuseSuspendContext):
                templates_for_id = payload.templates
                store_for_id = payload.store
                q_for_id = payload.q_norm
            template_id_out = self._resolve_session_step_template_id(
                matched_for_params,
                templates=templates_for_id,
                store=store_for_id,
                q_norm=q_for_id or self._turn_question or "",
            )

        self._audit_ask_emit(
            AUDIT_EVENT_ASK_SUSPEND,
            question=self._turn_question,
            details=(("state_id", ex.state_id), ("kind", kind)),
        )
        return self._mk_step(
            done=False,
            prompt=prompt_out,
            kind=kind,
            sql=sql_out,
            data=data_out,
            intent_summary=isum,
            reply_shape=reply_shape,
            semantic_warnings=sem_w,
            parameters=parameters,
            template_id=template_id_out,
        )

    def _finalize_pending_meta_step(self, pending: SessionStep) -> SessionStep:
        """Attach turn diagnostics and release the session after a metadata answer."""
        qtxt = self._turn_question or ""
        turn_diagnostics = self._emit_turn_llm_usage(question=qtxt, diagnostics=tuple(pending.diagnostics or ()))
        kind = str(pending.kind or SESSION_KIND_META)
        self._audit_ask_emit(
            "ask_done",
            question=qtxt,
            details=(("outcome", "meta"), ("kind", kind)),
        )
        step = self._mk_step(
            done=True,
            prompt=None,
            kind=kind,
            sql=None,
            data=None,
            answer=pending.answer,
            error=pending.error,
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
            reply_shape=None,
            intent_summary=None,
            semantic_warnings=(),
            parameters=(),
        )
        self._reset_after_turn()
        self._release_session_turn()
        return step

    def _completed_step(self) -> SessionStep:
        """Build a terminal :class:`SessionStep` after a full successful pipeline pass."""
        snap = self._last_turn_outcome or {}
        qtxt = self._turn_question or ""
        raw_outcome = str(snap.get("outcome") or "success")
        rows_raw = snap.get("rows")
        rows_tuple: tuple[tuple[Any, ...], ...] | None = None
        if isinstance(rows_raw, list):
            rows_tuple = tuple(tuple(r) for r in rows_raw)
        sql_val = snap.get("sql")
        sql_out: str | dict[str, str] | None
        if isinstance(sql_val, dict):
            sql_out = {str(k): str(v) for k, v in sql_val.items()}
        elif sql_val is not None:
            sql_out = str(sql_val)
        else:
            sql_out = None
        federated_bundle = snap.get("federated_bundle")
        generation_path = snap.get("generation_path")
        sql_out = MainSpaceOps.resolved_session_step_sql(
            sql_out,
            generation_path=generation_path if isinstance(generation_path, GenerationPath) else None,
            federated_bundle=federated_bundle,
            federated_plan=snap.get("federated_plan"),
        )
        cols_raw = snap.get("columns")
        cols_tuple: tuple[str, ...] | None = None
        if federated_bundle is not None and getattr(federated_bundle, "column_names", None):
            cols_tuple = tuple(str(c) for c in federated_bundle.column_names)
        elif isinstance(cols_raw, list) and cols_raw and all(isinstance(x, str) for x in cols_raw):
            cols_tuple = tuple(str(x) for x in cols_raw)
        elif rows_tuple:
            sql_for_headers = sql_out if isinstance(sql_out, str) else None
            cols_tuple = result_columns_for_session(
                sql_for_headers, list(rows_tuple), **MainInitOps.federation_contract_kwargs_from_snap(snap)
            )
        data_out: pandas.DataFrame | None = None
        if rows_tuple and raw_outcome == "success":
            sql_for_headers = sql_out if isinstance(sql_out, str) else None
            cols_use = (
                list(cols_tuple)
                if cols_tuple
                else result_columns_for_session(
                    sql_for_headers, list(rows_tuple), **MainInitOps.federation_contract_kwargs_from_snap(snap)
                )
            )
            if cols_use:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple], columns=list(cols_use))
            else:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple])
        session_error = session_error_from_turn_snap(snap) if raw_outcome != "success" else None
        if session_error is not None:
            sql_out = None
            data_out = None
        ri = snap.get("intent")
        isum_res: IntentSummary | None = None
        if isinstance(ri, RuntimeIntent) and session_error is None:
            isum_res = MainInteractiveOps.build_intent_summary(ri)
        parameters = (
            self._parameters_for_completed_turn(snap, qtxt) if raw_outcome == "success" and sql_out is not None else ()
        )
        matched_tmpl = snap.get("matched_template")
        template_id_out: str | None = None
        if raw_outcome == "success" and sql_out is not None:
            templates_for_id: dict[str, Any] | None = None
            store_for_id: dict[str, Any] | TemplateStoreView | None = None
            if not isinstance(matched_tmpl, Template):
                _, store_for_id, templates_for_id, _, _ = self._resources()
            template_id_out = self._resolve_session_step_template_id(
                matched_tmpl if isinstance(matched_tmpl, Template) else None,
                templates=templates_for_id,
                store=store_for_id,
                q_norm=qtxt,
            )
        turn_diagnostics = self._emit_turn_llm_usage(question=qtxt, diagnostics=())
        if raw_outcome == "federation_partial_failure":
            fed_source = snap.get("federation_source_id")
            fed_phase = snap.get("federation_phase")
            fed_succeeded = snap.get("federation_succeeded") or ()
            if fed_source or fed_phase or fed_succeeded:
                partial_source = str(fed_source or "composite")
                partial_phase = str(fed_phase or "execution")
                partial_details: list[tuple[str, str]] = [
                    ("source_id", partial_source),
                    ("phase", partial_phase),
                ]
                if fed_succeeded:
                    partial_details.append(("succeeded", ",".join(item[0] for item in fed_succeeded)))
                turn_diagnostics = turn_diagnostics + (
                    Diagnostic(
                        stage="execution",
                        level="error",
                        code=DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
                        message="",
                        details=tuple(partial_details),
                        source_id=partial_source,
                        phase=partial_phase,
                    ),
                )
        terminal_diagnostics = self._terminal_turn_diagnostics(turn_diagnostics)
        if self._session_schema_role() == SchemaRole.CONSUMER:
            terminal_diagnostics = sanitize_federation_diagnostics_for_egress(terminal_diagnostics)
        audit_details: list[tuple[str, str]] = [("outcome", raw_outcome)]
        terminal_kind = SESSION_KIND_RESULT
        if raw_outcome != "success":
            terminal_kind = SESSION_KIND_ERROR
        audit_details.append(("kind", terminal_kind))
        if cols_tuple:
            audit_details.append(("result_columns", ",".join(cols_tuple)))
        if federated_bundle is not None and self._session_schema_role() != SchemaRole.CONSUMER:
            source_ids = sorted(
                {
                    str(getattr(rec, "source_id", "") or "")
                    for rec in getattr(federated_bundle, "statements", ()) or ()
                    if getattr(rec, "phase", "") == "member" and str(getattr(rec, "source_id", "") or "")
                }
            )
            if source_ids:
                sources_text = ",".join(source_ids)
                audit_details.append(("sources_queried", sources_text))
                turn_diagnostics = turn_diagnostics + (
                    Diagnostic(
                        stage="execution",
                        level="info",
                        code=DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED,
                        message=f"Federated turn queried sources: {sources_text}",
                        details=(("phase", "execution"), ("sources_queried", sources_text)),
                        source_id="composite",
                        phase="execution",
                    ),
                )
        data_out, data_truncated = self._apply_data_row_cap(data_out)
        refusal_timing_kw: dict[str, int] = {}
        if session_error is not None:
            refusal_timing_kw["elapsed_ms"] = apply_refusal_timing_floor(turn_elapsed_ms())
        step = self._mk_step(
            done=True,
            prompt=None,
            kind=terminal_kind,
            sql=sql_out,
            data=data_out,
            error=session_error,
            reply_shape=None,
            semantic_warnings=(),
            parameters=parameters,
            diagnostics=terminal_diagnostics,
            data_truncated=data_truncated,
            template_id=template_id_out,
            intent_summary=isum_res,
            **refusal_timing_kw,
        )
        self._audit_ask_emit("ask_done", question=qtxt, details=tuple(audit_details))
        self._reset_after_turn()
        self._release_session_turn()
        return step

    def _terminal_cancelled_step(self, message: str) -> SessionStep:
        """Build a terminal :class:`SessionStep` for a cooperatively cancelled turn."""
        turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=())
        self._audit_ask_emit(
            AUDIT_EVENT_ASK_CANCELLED,
            question=self._turn_question,
            details=(("message", message),),
        )
        self.reset()
        st = self._mk_step(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            error=SessionError(code=SessionOutcome.CANCELLED),
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
        )
        return st

    def _terminal_error_step(self, message: str, *, exc: BaseException | None = None) -> SessionStep:
        """Build a terminal error :class:`SessionStep`."""
        fed_fields = MainInteractiveOps.federation_error_step_fields(exc) if exc is not None else {}
        fed_diag = MainInteractiveOps.federation_error_diagnostics(exc) if exc is not None else ()
        turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=fed_diag)
        audit_details: list[tuple[str, str]] = [("message", message)]
        if self._session_schema_role() != SchemaRole.CONSUMER:
            if fed_fields.get("federation_source_id"):
                audit_details.append(("source_id", str(fed_fields["federation_source_id"])))
        if fed_fields.get("federation_phase"):
            audit_details.append(("phase", str(fed_fields["federation_phase"])))
        if fed_fields.get("federation_limit_key"):
            audit_details.append(("limit_key", str(fed_fields["federation_limit_key"])))
        self._audit_ask_emit(
            "ask_error",
            question=self._turn_question,
            details=tuple(audit_details),
        )
        self.reset()
        return self._mk_step(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            error=session_error_from_terminal_message(message, federation_fields=fed_fields, exc=exc),
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
        )

    def _terminal_error_from_exception(self, exc: Exception) -> SessionStep:
        """Build a terminal error step, preserving federation attribution when present."""
        if isinstance(
            exc,
            (
                FederationMemberExecutionError,
                FederationPartialFailureError,
                FederationCapExceededError,
                FederationMemberProbeError,
            ),
        ):
            return self._terminal_error_step(federation_user_facing_error_message(exc), exc=exc)
        return self._terminal_error_step(str(exc))

    def _parameters_for_completed_turn(self, snap: dict[str, Any], qtxt: str) -> tuple[ParameterBinding, ...]:
        """Build parameter bindings for a successful terminal step."""
        if str(snap.get("outcome") or "") != "success":
            return ()
        schema, store, templates, _, _ = self._resources()
        tmpl_raw = snap.get("matched_template")
        hist_idx = snap.get("template_history_index")
        tmpl: Template | None = tmpl_raw if isinstance(tmpl_raw, Template) else None
        if tmpl is None and qtxt.strip():
            resolved = TemplateOps.resolve_template_for_question(qtxt, templates, template_store=store)
            if resolved is None:
                return ()
            tmpl, hist_idx = resolved
        if tmpl is None:
            return ()
        row_idx = int(hist_idx) if hist_idx is not None else 0
        intent_raw = snap.get("intent")
        override: dict[str, Any] | None = None
        nl = qtxt
        if isinstance(intent_raw, RuntimeIntent):
            if intent_raw.param_values:
                override = dict(intent_raw.param_values)
            if intent_raw.natural_language:
                nl = intent_raw.natural_language
        tmpl_map = templates if isinstance(templates, dict) else TemplateOps.store_to_templates(store)
        persist_labels = (
            self._session_mode == "writer" and MainInteractiveOps.persist_template_learning_for_pipeline_session(self)
        )
        return TemplateOps.build_parameter_bindings(
            tmpl,
            history_index=row_idx,
            schema=schema,
            question_nl=nl,
            persist_display_names=persist_labels,
            store=store,
            templates=tmpl_map,
            param_values_override=override,
        )

    def _resolve_session_step_template_id(
        self,
        matched: Template | None,
        *,
        templates: Mapping[str, Template] | dict[str, Any] | None = None,
        store: dict[str, Any] | TemplateStoreView | None = None,
        q_norm: str = "",
    ) -> str | None:
        """Return a known template id for SQL-bearing steps, with pending then resolve fallback."""
        if matched is not None:
            tid = str(getattr(matched, "id", "") or "").strip()
            if tid:
                return tid
        q = normalize_question(q_norm) if q_norm else ""
        if not q:
            return None
        tmpl_map: Mapping[str, Template] | dict[str, Any] | None = templates
        if tmpl_map is None and store is not None:
            tmpl_map = TemplateOps.store_to_templates(store)
        if tmpl_map is None:
            return None
        pending = TemplateOps.find_pending_template_for_question(tmpl_map, q)
        if pending is not None:
            return str(pending.id)
        resolved = TemplateOps.resolve_template_for_question(q, tmpl_map, template_store=store)
        if resolved is not None:
            return str(resolved[0].id)
        return None

    def _parameters_for_sql_bearing_step(
        self,
        *,
        sql: str | dict[str, str] | None,
        matched_template: Template | None,
        intent: RuntimeIntent | None,
        question_nl: str = "",
        persist_display_names: bool = False,
    ) -> tuple[ParameterBinding, ...]:
        """Project p-param bindings whenever *sql* is present on a session step."""
        if sql is None or matched_template is None:
            return ()
        schema, store, templates, _, _ = self._resources()
        override: dict[str, Any] | None = None
        nl = question_nl
        if intent is not None:
            if intent.param_values:
                override = dict(intent.param_values)
            if intent.natural_language:
                nl = intent.natural_language
        tmpl_map = templates if isinstance(templates, dict) else TemplateOps.store_to_templates(store)
        return TemplateOps.build_parameter_bindings(
            matched_template,
            history_index=0,
            schema=schema,
            question_nl=nl,
            persist_display_names=persist_display_names,
            store=store,
            templates=tmpl_map,
            param_values_override=override,
        )

    def _drive_question_turn(self, raw: str) -> SessionStep:
        """Run :func:`interactive_run_once` until suspend or completion."""
        self._reset_after_turn()
        self._turn_cancel_event = threading.Event()
        self._turn_accumulated_diagnostics = []
        self._turn_question = raw
        self._turn_llm_usage_start = len(snapshot_llm_usage_records())
        self._turn_llm_scope_tok = set_turn_llm_scope("question")
        turn_id = mint_turn_id()
        turn_id_token = push_turn_id(turn_id)
        turn_timing_tokens = push_turn_timing()
        owner_audit = getattr(self._owner, "_audit_emit", None)
        audit_token = push_audit_emit(owner_audit if callable(owner_audit) else None)
        self._audit_ask_emit("ask_begin", question=raw, details=())
        art = getattr(self._owner, "_artifacts_dir", None)
        adir = ""
        if art is not None:
            try:
                adir = os.path.abspath(os.fspath(art))
            except (TypeError, OSError, ValueError):
                adir = ""
        schema, store, templates, rejected, schema_terms = self._resources()
        cancel_token = push_session_turn_cancel(self._turn_cancel_event)
        ask_phase_token = push_ask_phase_callback(getattr(self._owner, "_phase_callback", None))

        def _run_turn() -> SessionStep:
            identity = self._owner_engine_identity()
            identity_token = push_engine_identity(identity)
            sandbox_runtime = getattr(self._owner, "_sandbox_runtime", None)
            sandbox_runtime_token = (
                SandboxRuntimeState.bind_sandbox_runtime(sandbox_runtime) if sandbox_runtime is not None else None
            )
            with owner_limits_scope(self._owner):
                with llm_usage_session_scope():
                    with llm_execution_scope(self._owner._runtime_config.llm_execution):
                        try:
                            return _run_turn_inner()
                        finally:
                            if sandbox_runtime_token is not None:
                                SandboxRuntimeState.reset_sandbox_runtime(sandbox_runtime_token)
                            pop_engine_identity(identity_token)

        def _run_turn_inner() -> SessionStep:
            try:
                if owner_is_aether_federation(self._owner):
                    members = getattr(self._owner, "_members", None)
                    if isinstance(members, dict) and members:
                        probe_federation_member_liveness(members)
                MainInitOps.interactive_run_once(
                    schema, store, templates, rejected, schema_terms, question=raw, pipeline_session=self
                )
            except PipelineSuspended as ex:
                if ex.state_id in ("empty_choice_queue", "choice_queue_mismatch"):
                    turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=())
                    self._audit_ask_emit(
                        "ask_error",
                        question=self._turn_question,
                        details=(("message", ex.message_for_caller),),
                    )
                    self.reset()
                    st_e = self._mk_step(
                        done=True,
                        prompt=None,
                        kind=SESSION_KIND_ERROR,
                        error=session_error_from_terminal_message(ex.message_for_caller),
                        diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
                    )
                    return st_e
                self._suspended = ex
                return self._suspend_to_step(ex)
            except SessionTurnCancelledError as exc:
                debug(f"[main_execution.PipelineSession._drive_question_turn] turn cancelled: {exc!r}")
                self._reset_after_turn()
                self._release_session_turn()
                return self._terminal_cancelled_step(str(exc))
            except Exception as exc:
                debug(f"[main_execution.PipelineSession._drive_question_turn] unexpected error: {exc!r}")
                self._reset_after_turn()
                self._release_session_turn()
                return self._terminal_error_from_exception(exc)
            pending = self._pending_terminal_step
            if pending is not None:
                self._pending_terminal_step = None
                return self._finalize_pending_meta_step(pending)
            return self._completed_step()

        try:
            with MainSessionSerdeOps._session_domain_knowledge_scope(self):
                with prompt_cache_schema_scope(schema_prompt_cache_id(schema)):
                    lock = getattr(self._owner, "_pipeline_writer_lock", None)
                    if self._session_mode == "reader":
                        if adir:
                            MainSpaceOps.reload_reader_learning_if_manifest_drift(self._owner)
                        return _run_turn()
                    consumer_writer = self._session_schema_role() == SchemaRole.CONSUMER
                    if lock is not None and adir:
                        with lock:
                            MainSpaceOps.drain_write_queue(
                                self._owner,
                                adir,
                                space_uid=self._space_name if consumer_writer else None,
                                consumer_writer=consumer_writer,
                            )
                    return _run_turn()
        finally:
            pop_turn_timing(*turn_timing_tokens)
            pop_turn_id(turn_id_token)
            pop_audit_emit(audit_token)
            pop_ask_phase_callback(ask_phase_token)
            pop_session_turn_cancel(cancel_token)

    def _resume_from_suspend(self) -> SessionStep:
        """Continue execution after enqueueing a programmatic answer."""
        if self._suspended is None:
            self._release_session_turn()
            return self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error=SessionError(code=SessionOutcome.INTERNAL_ERROR),
            )
        ex = self._suspended
        self._suspended = None
        self._resume_choice_stage_id = ex.state_id

        def _resume_work() -> None:
            identity = self._owner_engine_identity()
            identity_token = push_engine_identity(identity)
            sandbox_runtime = getattr(self._owner, "_sandbox_runtime", None)
            sandbox_runtime_token = (
                SandboxRuntimeState.bind_sandbox_runtime(sandbox_runtime) if sandbox_runtime is not None else None
            )
            try:
                with owner_limits_scope(self._owner):
                    with llm_execution_scope(self._owner._runtime_config.llm_execution):
                        MainInteractiveOps.dispatch_pipeline_resume(self, ex)
            finally:
                if sandbox_runtime_token is not None:
                    SandboxRuntimeState.reset_sandbox_runtime(sandbox_runtime_token)
                pop_engine_identity(identity_token)

        cancel_token = push_session_turn_cancel(self._turn_cancel_event)
        ask_phase_token = push_ask_phase_callback(getattr(self._owner, "_phase_callback", None))
        try:
            with MainSessionSerdeOps._session_domain_knowledge_scope(self):
                lock = getattr(self._owner, "_pipeline_writer_lock", None)
                art = getattr(self._owner, "_artifacts_dir", None)
                adir = ""
                if art is not None:
                    try:
                        adir = os.path.abspath(os.fspath(art))
                    except (TypeError, OSError, ValueError):
                        adir = ""
                if self._session_mode == "writer" and lock is not None:
                    with lock:
                        if adir:
                            consumer_writer = self._session_schema_role() == SchemaRole.CONSUMER
                            MainSpaceOps.drain_write_queue(
                                self._owner,
                                adir,
                                space_uid=self._space_name if consumer_writer else None,
                                consumer_writer=consumer_writer,
                            )
                with llm_usage_session_scope():
                    _resume_work()
        except PipelineSuspended as ex2:
            if ex2.state_id in ("empty_choice_queue", "choice_queue_mismatch"):
                self.reset()
                return self._terminal_error_step(ex2.message_for_caller)
            self._suspended = ex2
            return self._suspend_to_step(ex2)
        except SessionTurnCancelledError as exc:
            debug(f"[main_execution.PipelineSession._resume_from_suspend] turn cancelled: {exc!r}")
            self._reset_after_turn()
            self._release_session_turn()
            return self._terminal_cancelled_step(str(exc))
        except Exception as exc:
            debug(f"[main_execution.PipelineSession._resume_from_suspend] unexpected error: {exc!r}")
            self._reset_after_turn()
            self._release_session_turn()
            return self._terminal_error_from_exception(exc)
        finally:
            self._resume_choice_stage_id = None
            pop_ask_phase_callback(ask_phase_token)
            pop_session_turn_cancel(cancel_token)
        return self._completed_step()
