"""
Entry points for seed warmup, QSim, interactive runs, programmatic PipelineSession, and artifact helpers.

Optional ``pyspark.sql.SparkSession`` is imported at module load when available for engine reachability checks.
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import uuid
import sys
import tomllib
import zipfile
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas
from platformdirs import user_data_dir

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

from . import _core_utils
from ._config import (
    ARTIFACT_DIRECTORY_SEGMENT,
    SCHEMA_OVERRIDES_VERSION,
    AZURE_OPENAI_ENV_REQUIRED,
    DATABRICKS_ENV_CATALOG,
    DATABRICKS_ENV_HTTP_PATH,
    DATABRICKS_ENV_SCHEMA,
    DATABRICKS_ENV_SERVER_HOSTNAME,
    DATABRICKS_ENV_TOKEN,
    DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
    DIAGNOSTIC_CODE_LOW_CONFIDENCE,
    AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD,
    AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT,
    ENGINE_STORAGE_SLUG_MAX_CHARS,
    INTERACTIVE_STAGE_SQL_FEEDBACK,
    JSON_COMPACT_SEPARATORS,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_FILENAME,
    NORMALIZED_SEEDS_TXT,
    OPENAI_ENV_REQUIRED,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_SQL,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    POSTGRES_ENV_DATABASE,
    POSTGRES_ENV_HOST,
    POSTGRES_ENV_PASSWORD,
    POSTGRES_ENV_PORT,
    POSTGRES_ENV_SCHEMA,
    POSTGRES_ENV_USER,
    QSIM_QUESTIONS_PATTERN,
    SCHEMA_CONTEXT_CACHE_NAME,
    SCHEMA_CONTEXT_CACHE_VERSION,
    SCHEMA_CONTEXT_CACHED_DDL,
    SCHEMA_CONTEXT_CACHED_NOTES,
    SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT,
    SECRET_ENV_KEYS,
    SEED_NORMALIZATION_JSON,
    SESSION_KIND_ERROR,
    SESSION_KIND_IDLE,
    SESSION_KIND_RESULT,
    SIMULATION_CACHE_EXACT_FILENAMES,
    SIMULATION_CACHE_GLOB_PATTERNS,
    SUPPORTED_ENGINES,
    SUSPEND_ID_TO_SESSION_KIND,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_SEGMENT,
    WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS,
    WRITE_QUEUE_FILENAME,
    WRITE_QUEUE_MAX_BYTES_PER_DRAIN,
    DatabricksRuntimeConfig,
    EngineConfig,
    GenerationPath,
    PolicyConfig,
    PostgresRuntimeConfig,
    QSimConfig,
    SeedWarmupConfig,
    diagnostic_debug_enabled,
    load_runtime_config,
)
from ._contracts_base import (
    AccessError,
    ConfigError,
    ConnectionError,
    Diagnostic,
    FailureCategory,
    IntentSummary,
    LLMConfig,
    MigrationPendingError,
    MigrationReport,
    MigrationTier,
    PipelineSuspended,
    QSimSummary,
    RuntimeConfig,
    SchemaContext,
    SchemaGraph,
    SeedWarmupSummary,
    SessionActiveError,
    SessionStep,
    WriteQueueEvent,
)
from ._contracts_core import (
    DirectReuseSuspendContext,
    FeedbackKind,
    InteractiveTailSnapshot,
    QuestionFeedbackEntry,
    QuestionFormStorage,
    RefinementContext,
    RefinementRetry,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
    UserFeedbackRejectSuspendContext,
    classify_seed_warmup_intent_complexity,
)
from ._core_utils import (
    InteractiveChoicePort,
    RephraseHint,
    artifact_lock,
    _wipe_filenames,
    _wipe_globs,
    classify_migration_tier,
    clear_llm_clients,
    debug,
    detect_legacy_artifacts,
    diagnostic_segment,
    drain_diagnostic_collector,
    interactive_yes_no,
    log,
    normalize_question,
    notify,
    print_rephrase_hint,
    read_artifact_manifest,
    reset_diagnostic_collector,
    set_diagnostic_collector,
    take_and_clear_orphan_diagnostics,
    try_rename_migration_plan,
    USER_REJECTED_RESULT_BUCKET_TIPS,
    wipe_versioned_artifacts,
)
from ._dialect import DatabricksDialect, PostgresDialect, get_dialect
from ._expansion_ops import expand_gold_intents
from ._intent_process import (
    collect_structural_match_templates,
    list_union_match_candidates,
    match_template_for_union,
    reconcile_template_store_until_stable,
    structural_compare,
)
from ._pipeline import (
    _refinement_retry_available,
    best_accepted_template_similarity,
    build_interactive_tail_snapshot,
    build_result_dataframe,
    complete_direct_sql_reuse_user_choice,
    complete_user_feedback_reject,
    compose_intent_confirm_session_message,
    compute_final_metrics,
    confirm_intent_with_user,
    display_final_results_to_stdout,
    extract_column_headers,
    generate_and_validate_sql,
    handle_direct_sql_reuse,
    handle_user_feedback,
    load_pipeline_resources,
    match_question_level_template_reuse,
    parse_intent_via_llm,
    prepare_union_match_join_phase,
    save_result_csv,
)
from ._qsim import instantiate_all
from ._qsim_ops import generate_all_intents, generate_all_questions
from ._schema import (
    _column_names_lower_index,
    _finalize_with_overrides,
    apply_schema_overrides_to_graph,
    build_schema_graph_with_diff,
    compute_schema_limits,
    destructive_migration_execute,
    load_schema_graph_snapshot,
)
from ._seed_warmup import (
    accepted_template_instance_keys,
    get_next_seed_warmup_version,
    get_next_warmup_preflight_version,
    open_seed_warmup_cache_session,
    resolve_joins_for_table_set,
    run_gold_intent_generation,
    run_seed_warmup_execution,
    save_seed_warmup_cache_zip,
    save_seed_warmup_report,
    warmup_pool_operator_feature_stats,
)
from ._sql_to_intent import (
    compute_sql_history_content_hash,
    convert_sql_to_intent,
    dedup_runtime_intents,
    fetch_query_log,
    load_sql_history_statements,
    seed_warmup_intent_from_runtime_intent,
)
from ._templates import (
    TemplateStoreView,
    _apply_schema_migration_map,
    _load_schema_migration_map,
    _validate_schema_migration_map,
    apply_migration_policy,
    export_schema_migration_map_skeleton,
    has_any_rejection_history_for_question,
    load_template_store,
    record_question_feedback,
    save_template_store,
    should_auto_accept_for_question,
    store_to_templates,
    summarize_failure_for_memory,
    templates_to_store,
)
from ._utils import (
    body_similarity_key,
    flatten_param_values,
    intent_key,
    normalize_question_via_llm,
    validate_question,
)

SESSION_PROMPT_YESNO: str = "Is this correct? (y/n): "
SESSION_PROMPT_REASON: str = "Please provide a reason: "

MIGRATION_HEADER_BY_TIER: dict[str, str] = {
    "soft_refresh": "Refreshing cached metadata. Existing learning is kept.",
    "remap": "Schema renames detected. Mapping existing learning to the new names.",
    "destructive": "Learning reset: cache rebuilt from scratch (schema changed in ways that cannot be remapped).",
}

MIGRATION_DECLINED_LINE: str = "Migration declined. Provide a fresh artifacts_dir to continue."

SAVED_LINE: str = "Saved."

FEEDBACK_NOTED_LINE: str = "Feedback noted. Try rephrasing your question for a better match."


def _failure_category_for_terminal_step(step: SessionStep) -> str | None:
    """Map a terminal error :class:`SessionStep` to a coarse failure category string."""

    if step.kind != SESSION_KIND_ERROR:
        return None
    err = (step.error or "").strip()
    if not err:
        return None
    for d in step.diagnostics:
        code_u = (d.code or "").upper()
        if code_u in {"EXPLAIN_COST_EXCEEDED", "EXPLAIN_COST"} or "explain_cost_exceeded" in (d.code or "").lower():
            return FailureCategory.EXECUTION_COST_EXCEEDED.value
    blob = " ".join([step.error or "", *[x.message for x in step.diagnostics]]).lower()
    if ("cost" in blob or "explain_cost" in blob) and ("exceed" in blob or "cap" in blob):
        return FailureCategory.EXECUTION_COST_EXCEEDED.value
    if "timeout" in blob or "statement_timeout" in blob:
        return FailureCategory.EXECUTION_TIMEOUT.value
    return FailureCategory.EXECUTION_OTHER_ERROR.value


def _interactive_attach_refinement_ctx(
    choice_port: InteractiveChoicePort | None,
    refinement_ctx: RefinementContext,
) -> None:
    """Bind turn-local refinement state to an interactive session when supported."""

    if choice_port is None:
        return
    attach = getattr(choice_port, "_attach_refinement_ctx", None)
    if callable(attach):
        attach(refinement_ctx)


def _persist_template_learning_for_pipeline_session(port: Any | None) -> bool:
    """Return whether template-store and question-feedback mutations may be written for this choice-port session."""

    if port is None:
        return True
    return getattr(port, "_session_mode", "writer") == "writer"


def emit_write_queue_event(artifacts_dir: str, event: WriteQueueEvent) -> None:
    """Append one JSON line representing a deferred writer event."""

    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    os.makedirs(artifacts_dir, exist_ok=True)
    obj = {
        "kind": event.kind,
        "schema_hash": event.schema_hash,
        "produced_at": event.produced_at,
        "payload": [list(pair) for pair in event.payload],
    }
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def _decode_write_queue_event(obj: dict[str, Any]) -> WriteQueueEvent | None:
    kinds = {
        "template_accept",
        "template_reject",
        "paraphrase_emit",
        "override_proposal",
        "feedback_record",
    }
    kind = str(obj.get("kind") or "")
    if kind not in kinds:
        return None
    schema_hash = str(obj.get("schema_hash") or "")
    produced_at = str(obj.get("produced_at") or "")
    raw_pl = obj.get("payload")
    if not isinstance(raw_pl, list):
        return None
    pairs: list[tuple[str, str]] = []
    for row in raw_pl:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        pairs.append((str(row[0]), str(row[1])))
    return WriteQueueEvent(
        kind=kind,  # type: ignore[arg-type]
        schema_hash=schema_hash,
        produced_at=produced_at,
        payload=tuple(pairs),
    )


def _reload_reader_learning_if_manifest_drift(owner: Any) -> None:
    """Reload partitioned template store and replay overrides when disk manifest drifts from the live graph."""

    manifest = read_artifact_manifest(str(owner._artifacts_dir))
    if manifest is None:
        return
    live = str(getattr(owner._schema_graph, "effective_structural_hash", "") or "")
    man = str(manifest.effective_structural_hash or "")
    if not man or man == live:
        return
    store = load_template_store(live, owner._schema_graph)
    templates = store_to_templates(store)
    owner._store = store
    owner._templates = templates
    _finalize_with_overrides(
        owner._schema_graph,
        EngineConfig.SCHEMA_JSON_PATH,
        dialect=getattr(owner, "_dialect", None),
    )


def _emit_write_queue_audit(owner: Any, event_type: str, details: tuple[tuple[str, str], ...]) -> None:
    """Forward write-queue drain outcomes to ``owner._audit_emit`` when an audit sink is configured."""

    fn = getattr(owner, "_audit_emit", None)
    if not callable(fn):
        return
    sg = getattr(owner, "_schema_graph", None)
    sh = str(getattr(sg, "effective_structural_hash", "") or "") or None
    fn(
        event_type,
        schema_hash=sh,
        details=details,
    )


def _drain_dispatch_write_queue_event(owner: Any, event: WriteQueueEvent) -> bool:
    """Apply one queue event to *owner*'s live stores. Returns True when the template store should be saved."""

    live = str(getattr(owner._schema_graph, "effective_structural_hash", "") or "")
    if event.schema_hash != live:
        return False
    store = owner._store
    templates: dict[str, Any] = owner._templates
    rejected: dict[str, Any] = owner._rejected
    schema = owner._schema_graph
    dialect = getattr(owner, "_dialect", None)
    pl = dict(event.payload)

    if event.kind == "feedback_record":
        q_norm = str(pl.get("q_norm") or "")
        raw_entry = pl.get("entry_json") or "{}"
        try:
            entry_doc = json.loads(raw_entry)
        except json.JSONDecodeError:
            notify("write_queue: malformed entry_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
            return False
        if not isinstance(entry_doc, dict):
            return False
        entry = QuestionFeedbackEntry.from_dict(entry_doc)
        record_question_feedback(store, q_norm, entry)
        _emit_write_queue_audit(
            owner,
            AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD,
            (("kind", "feedback_record"), ("q_norm", q_norm)),
        )
        return True

    if event.kind == "template_reject":
        raw_ctx = pl.get("ctx_json") or "{}"
        try:
            ctx_doc = json.loads(raw_ctx)
        except json.JSONDecodeError:
            notify("write_queue: malformed ctx_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
            return False
        if not isinstance(ctx_doc, dict):
            return False
        intent = RuntimeIntent.from_dict(ctx_doc.get("intent") or {})
        mid = str(ctx_doc.get("matched_template_id") or "")
        mrej_id = str(ctx_doc.get("matched_rejected_template_id") or "")
        mt = templates.get(mid) if mid else None
        mrej = rejected.get(mrej_id) if mrej_id else None
        try:
            gpath = GenerationPath.parse(str(ctx_doc.get("generation_path") or ""))
        except (KeyError, ValueError, TypeError):
            gpath = GenerationPath.FRESH
        ctx = UserFeedbackRejectSuspendContext(
            intent=intent,
            sql=str(ctx_doc.get("sql") or ""),
            schema=schema,
            store=store,
            templates=templates,
            rejected=rejected,
            q_norm=str(ctx_doc.get("q_norm") or ""),
            generation_path=gpath,
            matched_template=mt,
            matched_rejected_template=mrej,
            dialect=dialect,
            structural_match_templates=None,
        )
        complete_user_feedback_reject(
            ctx,
            needs_reason=bool(ctx_doc.get("needs_reason")),
            reject_reason=str(ctx_doc.get("reject_reason") or ""),
            choice_port=None,
            persist_template_learning=True,
        )
        _emit_write_queue_audit(
            owner,
            AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT,
            (("kind", "template_reject"), ("q_norm", str(ctx_doc.get("q_norm") or ""))),
        )
        return False

    if event.kind == "template_accept":
        raw = pl.get("replay_json") or "{}"
        try:
            rep = json.loads(raw)
        except json.JSONDecodeError:
            notify("write_queue: malformed replay_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
            return False
        if not isinstance(rep, dict):
            return False
        intent = RuntimeIntent.from_dict(rep.get("intent") or {})
        sql = str(rep.get("sql") or "")
        q_norm = str(rep.get("q_norm") or "")
        try:
            gpath = GenerationPath.parse(str(rep.get("generation_path") or ""))
        except (KeyError, ValueError, TypeError):
            gpath = GenerationPath.FRESH
        mid = str(rep.get("matched_template_id") or "")
        mrej_id = str(rep.get("matched_rejected_id") or "")
        mt = templates.get(mid) if mid else None
        mrej = rejected.get(mrej_id) if mrej_id else None
        join_matches = bool(rep.get("join_matches", True))
        sm_ids = [x for x in str(rep.get("structural_ids") or "").split(",") if x]
        sm_list = [templates[x] for x in sm_ids if x in templates]
        fs_raw = rep.get("form_storage")
        fs: QuestionFormStorage | None = None
        if isinstance(fs_raw, dict):
            fs = QuestionFormStorage(
                corrected=str(fs_raw.get("corrected") or ""),
                normalized_optional=fs_raw.get("normalized_optional"),
                normalized_negative_memory_dropped=bool(fs_raw.get("normalized_negative_memory_dropped")),
                accept_via_normalized_lookup_only=bool(fs_raw.get("accept_via_normalized_lookup_only")),
            )
        handle_user_feedback(
            "y",
            intent,
            sql,
            schema,
            store,
            templates,
            rejected,
            q_norm,
            gpath,
            mt,
            mrej,
            dialect=dialect,
            structural_match_templates=sm_list or None,
            choice_port=None,
            join_matches_template=join_matches,
            form_storage=fs,
            persist_template_learning=True,
        )
        _emit_write_queue_audit(
            owner,
            AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT,
            (("kind", "template_accept"), ("q_norm", q_norm)),
        )
        return False

    if event.kind == "override_proposal":
        raw_doc = pl.get("document_json") or "{}"
        try:
            document = json.loads(raw_doc)
        except json.JSONDecodeError:
            notify("write_queue: malformed document_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
            return False
        if not isinstance(document, dict):
            return False
        document.setdefault("version", SCHEMA_OVERRIDES_VERSION)
        try:
            apply_schema_overrides_to_graph(schema, document)
        except Exception as exc:  # noqa: BLE001 — best-effort replay
            notify(
                f"write_queue: override_proposal apply failed: {exc}",
                stage="pipeline",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
            )
            return False
        _emit_write_queue_audit(
            owner,
            AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL,
            (("kind", "override_proposal"),),
        )
        return False

    if event.kind == "paraphrase_emit":
        notify(
            "write_queue: paraphrase_emit is reserved; line skipped",
            stage="pipeline",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )
        return False

    return False


def drain_write_queue(owner: Any, artifacts_dir: str) -> int:
    """Drain deferred reader events under the artifact lock; returns the number of events applied."""

    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    applied = 0
    with artifact_lock(artifacts_dir, timeout=WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return 0
        with open(path, "rb") as fh:
            body = fh.read()
        if not body:
            return 0
        limit = WRITE_QUEUE_MAX_BYTES_PER_DRAIN
        if len(body) > limit:
            head = body[:limit]
            cut = head.rfind(b"\n")
            if cut == -1:
                return 0
            to_process = head[: cut + 1]
            tail = head[cut + 1 :] + body[limit:]
        else:
            to_process = body
            tail = b""
        text = to_process.decode("utf-8")
        should_save = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                notify("write_queue: malformed line skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                continue
            if not isinstance(doc, dict):
                continue
            evt = _decode_write_queue_event(doc)
            if evt is None:
                notify("write_queue: unknown event skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                continue
            if _drain_dispatch_write_queue_event(owner, evt):
                should_save = True
            applied += 1
        with open(path, "wb") as out:
            out.write(tail)
    if should_save and isinstance(owner._store, TemplateStoreView):
        save_template_store(owner._store)
    return applied


def _interactive_run_intent_pass(
    *,
    corrected_text: str,
    q_norm: str,
    dialect: Any,
    schema: SchemaGraph,
    store: dict[str, Any],
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema_terms: set[str],
    choice_port: InteractiveChoicePort | None,
    form_storage: QuestionFormStorage | None,
    refinement_ctx: RefinementContext,
    persist_template_learning: bool,
) -> bool:
    """Parse intent once and continue through confirmation and SQL feedback."""

    if refinement_ctx.pending_retry:
        refinement_ctx.pending_retry = False
        if refinement_ctx.skip_refinement_increment_once:
            refinement_ctx.skip_refinement_increment_once = False
        else:
            refinement_ctx.refinement_rounds_executed += 1
    msg = "Refining intent..." if refinement_ctx.accumulated_reasons else "Processing intent..."
    _core_utils.progress(msg)
    parsed_intent, semantic_warnings, _ = parse_intent_via_llm(
        corrected_text,
        schema,
        templates,
        store,
        choice_port=choice_port,
        refinement_ctx=refinement_ctx,
        persist_template_learning=persist_template_learning,
    )
    if parsed_intent is None:
        _note_interactive_turn(choice_port, outcome="parse_failed", error="Intent parse failed.")
        return False
    ik = intent_key(parsed_intent)
    if refinement_ctx.refinement_rounds_executed > 0 and ik == refinement_ctx.last_intent_key:
        refinement_ctx.block_further_refinement = True
    refinement_ctx.last_intent_key = ik
    _run_interactive_post_intent_parse(
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
    )
    return True


def _result_columns_for_session(
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


def _build_intent_summary(intent: RuntimeIntent) -> IntentSummary:
    """Project a :class:`RuntimeIntent` into a compact :class:`IntentSummary` for session steps."""

    sel = tuple(sc.expr.signature_key for sc in intent.select_cols or [])
    flt = tuple(fp.signature_key for fp in intent.filters_param or [])
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


def _note_interactive_turn(
    choice_port: InteractiveChoicePort | None,
    *,
    outcome: str,
    error: str | None = None,
    sql: str | None = None,
    rows: list[tuple[Any, ...]] | None = None,
    columns: tuple[str, ...] | None = None,
    rejection_bucket: str | None = None,
    intent: RuntimeIntent | None = None,
) -> None:
    """Record turn outcome on *choice_port* when it implements ``note_turn_outcome``."""

    fn = getattr(choice_port, "note_turn_outcome", None)
    if callable(fn):
        fn(
            outcome=outcome,
            error=error,
            sql=sql,
            rows=rows,
            columns=columns,
            rejection_bucket=rejection_bucket,
            intent=intent,
        )


def _gold_intent_store_path_41_42_blocks_warmup(
    si: SeedWarmupIntent,
    templates: dict[str, Template],
) -> bool:
    """Return True when a gold row matches the store only via disallowed warmup subpaths 4.1 / 4.2."""

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


def get_next_qsim_version(artifacts_dir: str) -> int:
    """
    Return the next monotonic QSim version for an artifacts directory.

    Args:

        artifacts_dir: Directory containing `qsim_questions_v*.txt` and optionally `qsim_summary.json`.

    Returns:

        One greater than the highest existing version, or ``1`` when none exist.
    """
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


def _format_qsim_summary_line(s: QSimSummary) -> str:
    """Single-line human summary for one QSim run."""

    return f"  v{s.version}: intents={s.num_intents}  questions={s.num_questions}  seed={s.seed}"


def _format_seed_warmup_summary(s: SeedWarmupSummary) -> str:
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
        ),
    )


def print_qsim_range(start: int, end: int, artifacts_dir: str) -> None:
    """Emit QSim summaries from *start* through *end* plus the latest overall entry via :func:`notify`."""

    summaries = load_qsim_summaries(artifacts_dir)
    picked = [s for s in summaries if start <= int(s.version) <= end]
    lines = [f"QSim range ({len(picked)} runs):"]
    for s in picked:
        lines.append(_format_qsim_summary_line(s))
    if summaries:
        latest = max(summaries, key=lambda x: int(x.version))
        lines.append(
            f"Latest: v{latest.version}  intents={latest.num_intents}  "
            f"questions={latest.num_questions}  seed={latest.seed}",
        )
    notify("\n".join(lines), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def print_latest_seed_warmup_summary(artifacts_dir: str) -> None:
    """Emit the newest seed warmup summary under *artifacts_dir*, if any, via :func:`notify`."""

    s = find_latest_seed_warmup_summary(artifacts_dir)
    if s is None:
        notify("Seed warmup summary: none found.", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
        return
    notify(_format_seed_warmup_summary(s), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def _print_migration_applied(report: MigrationReport, sink: Callable[[str], None]) -> None:
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


def describe_runtime_config(runtime: RuntimeConfig, llm: LLMConfig) -> str:
    """Build a redacted multi-line snapshot of engine, schema scope, DB, and LLM settings."""

    lines: list[str] = []
    lines.append(f"Engine:          {runtime.engine}")
    lines.append(f"Artifacts dir:   {os.path.abspath(runtime.artifacts_dir)}")
    deny_list = sorted(runtime.schema_context.deny_columns)
    lines.append(f"Schema context:  deny_columns={deny_list!r}")
    if runtime.engine == "postgresql":
        lines.append("Postgres:")
        lines.append(f"  host:     {PostgresRuntimeConfig.HOST}")
        lines.append(f"  port:     {PostgresRuntimeConfig.PORT}")
        lines.append(f"  database: {PostgresRuntimeConfig.DATABASE}")
        lines.append(f"  user:     {PostgresRuntimeConfig.USER}")
        lines.append(f"  password: {_redact_display_value('password', PostgresRuntimeConfig.PASSWORD)}")
    else:
        lines.append("Databricks:")
        lines.append(f"  server_hostname: {DatabricksRuntimeConfig.SERVER_HOSTNAME}")
        lines.append(f"  http_path:       {DatabricksRuntimeConfig.HTTP_PATH}")
        lines.append(f"  catalog:         {DatabricksRuntimeConfig.CATALOG}")
        lines.append(f"  schema:          {DatabricksRuntimeConfig.SCHEMA}")
        lines.append(
            f"  access_token:    {_redact_display_value('access_token', DatabricksRuntimeConfig.ACCESS_TOKEN)}"
        )
    lines.append("LLM:")
    lines.append(f"  provider:   {llm.provider}")
    if llm.provider == "azure":
        base = EngineConfig.azure_base_url() or ""
        lines.append(f"  base_url:   {base}")
        lines.append(f"  api_key:    {_redact_display_value('api_key', EngineConfig.AZURE_API_TOKEN)}")
    else:
        lines.append(f"  base_url:   {EngineConfig.OPENAI_BASE_URL or ''}")
        lines.append(f"  api_key:    {_redact_display_value('api_key', EngineConfig.API_TOKEN)}")
    return "\n".join(lines)


def qsim_run_once(
    num_intents: int | None = None,
    num_questions: int | None = None,
    seed: int | None = None,
    artifacts_dir: str | None = None,
    schema: SchemaGraph | None = None,
) -> None:
    """
    Run full QSim (intents, values, NL questions) and write versioned question text plus summary.

    Args:

        num_intents: Distinct intent count; default `QSimConfig.INTENT_TYPES`.

        num_questions: Total NL questions; default `QSimConfig.QUESTIONS_COUNT`.

        seed: RNG seed; default `QSimConfig.RANDOM_SEED`.

        artifacts_dir: Output directory, or cwd when `None`.

        schema: Profiled `SchemaGraph` (required).

    Returns:

        ``None``; the latest run line is printed to stdout.

    Raises:

        RuntimeError: If `schema` is `None` or column roles are missing.
    """
    if num_intents is None:
        num_intents = QSimConfig.INTENT_TYPES
    if num_questions is None:
        num_questions = QSimConfig.QUESTIONS_COUNT
    if seed is None:
        seed = QSimConfig.RANDOM_SEED

    random.seed(seed)

    log(f"Starting question simulation: {num_intents} intent types, {num_questions} questions, seed={seed}")

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
    log(f"  Loaded {len(schema.tables)} tables, {total_cols} columns with metadata")

    column_roles: dict[str, str] = {}
    for table_name, table_meta in schema.tables.items():
        for col_name, col_meta in table_meta.columns.items():
            if col_meta.role:
                column_roles[f"{table_name}.{col_name}"] = col_meta.role

    base_dir = artifacts_dir or "."
    os.makedirs(base_dir, exist_ok=True)
    version = get_next_qsim_version(base_dir)

    log("Generating QSimIntent structures...")
    intents = generate_all_intents(schema, column_roles, num_intents, rng_seed=seed)
    log(f"  Generated {len(intents)} QSimIntent structures")

    log("Instantiating QSimIntents with values...")
    instantiated = instantiate_all(intents, schema, num_questions, rng_seed=seed)
    log(f"  Created {len(instantiated)} QSimIntent variants with values")

    log("Generating NL questions via LLM...")
    results = generate_all_questions(instantiated, schema)
    log(f"  Generated {len(results)} QSimIntents with questions")

    parent_ids = [
        (intent.intent_id.rsplit("_v", 1)[0] if "_v" in intent.intent_id else intent.intent_id) for intent in results
    ]
    intent_counts = Counter(parent_ids)
    log(f"  Questions per intent type: {dict(intent_counts)}")

    qname = QSIM_QUESTIONS_PATTERN.format(version=version)
    qsim_questions_path = os.path.join(base_dir, qname)
    qsim_summary_path = os.path.join(base_dir, "qsim_summary.json")

    log(f"Saving QSim questions to {qsim_questions_path}...")
    with open(qsim_questions_path, "w", encoding="utf-8") as f:
        for i, qintent in enumerate(results, 1):
            f.write(f"{i}. {qintent.question}\n")

    summary_entry = QSimSummary(
        version=version,
        num_intents=len(intents),
        num_questions=len(results),
        seed=seed,
    )

    summaries: list[Any] = []
    if os.path.exists(qsim_summary_path):
        with open(qsim_summary_path, encoding="utf-8") as f:
            summaries = json.load(f)
        if not isinstance(summaries, list):
            summaries = []
    summaries.append(summary_entry.to_dict())
    with open(qsim_summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS)

    log(f"Question simulation complete: {len(results)} questions saved")
    notify(f"QSim version: {version}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

    if results and diagnostic_debug_enabled():
        debug("[main_execution.qsim_run_once] samples:")
        for i, item in enumerate(results[:5]):
            debug(f"[main_execution.qsim_run_once]   {i + 1}. {item.question}")

    notify(_format_qsim_summary_line(summary_entry), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
    return None


def load_questions_from_qsim_txt(path: str) -> list[str]:
    """
    Load numbered natural-language questions from a QSim ``.txt`` artifact.

    Args:

        path: Path to ``qsim_questions_v{n}.txt``.

    Returns:

        Question strings in file order.
    """
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


def get_questions_only(questions: list[str], *, output_path: str) -> None:
    """
    Print and save a numbered list of natural-language questions.

    Args:

        questions: Question strings in display order.

        output_path: Destination file for the same numbered list written line by line.

    Returns:

        None.
    """
    for i, q in enumerate(questions, 1):
        notify(f"{i}. {q}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            f.write(f"{i}. {q}\n")


def print_questions_bundle(
    version: int,
    artifacts_dir: str,
) -> None:
    """Load QSim questions for a version, print them, and mirror lines to ``qsim_v{version}_questions.txt`` in the working directory."""

    path = resolve_qsim_path(version, artifacts_dir)
    questions = load_questions_from_qsim_txt(path)
    ver = int(version)
    out_path = str(Path.cwd() / f"qsim_v{ver}_questions.txt")
    get_questions_only(questions, output_path=out_path)


def seed_warmup_run_once(
    schema: SchemaGraph,
    dialect: Any,
    seed_filepath: str,
    output_dir: str,
    store: dict[str, Any] | None = None,
    templates: dict[str, Template] | None = None,
    interactive_gold: bool = True,
    seed: int | None = None,
    *,
    warmup_dry_run_only: bool = False,
) -> None:
    """
    Execute the seed warmup pipeline: gold build, expansion, execute, stratified sampling, and optional NL LLM.

    When *warmup_dry_run_only* is True, writes a preflight report and skips question LLM and template writes. Prints a final summary block and returns ``None``.
    """

    if seed is None:
        seed = SeedWarmupConfig.RANDOM_SEED
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    if warmup_dry_run_only:
        version = get_next_warmup_preflight_version(output_dir)
        report_name = SeedWarmupConfig.WARMUP_PREFLIGHT_REPORT_PATTERN.format(version=version)
    else:
        version = get_next_seed_warmup_version(output_dir)
        report_name = SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version)
    report_filepath = os.path.join(output_dir, report_name)

    log(f"Starting seed warmup {'preflight' if warmup_dry_run_only else 'run'} version {version}")
    notify(
        f"Seed warmup {'preflight' if warmup_dry_run_only else 'run'} version: {version}",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    schema_stats = schema.ensure_schema_stats()
    limits = compute_schema_limits(schema_stats)
    log(
        f"Computed SchemaLimits: max_filters={limits.max_filters}, "
        f"max_groupby={limits.max_groupby}, "
        f"max_tables={limits.max_tables}"
    )

    log("[P1] Gold build: seed normalization and gold intent generation")
    gold_intents_raw, gold_funnel, failure_trace_body, seed_norm_bundle = run_gold_intent_generation(
        schema,
        seed_filepath,
        interactive=interactive_gold,
        seed_warmup_version=version,
    )
    gold_warmup_intents = [SeedWarmupIntent.from_dict(d) if isinstance(d, dict) else d for d in gold_intents_raw]
    for row in gold_warmup_intents:
        row.source = "gold"
    log(f"Gold intents: {len(gold_warmup_intents)}")
    seed_questions_loaded = int(gold_funnel.get("seed_questions_loaded", 0))
    gold_failed_count = int(gold_funnel.get("gold_failed", 0))
    gold_user_rejected_count = int(gold_funnel.get("gold_user_rejected", 0))
    notify(
        "Phase A complete: seed normalization and gold intent generation "
        f"(seed_questions={seed_questions_loaded}, "
        f"gold_intents={len(gold_warmup_intents)}, "
        f"parse_failed={gold_failed_count}, "
        f"user_rejected={gold_user_rejected_count}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    log("[P2] Expansion: deterministic multi-depth expand_gold_intents")
    expanded_only = expand_gold_intents(
        gold_warmup_intents,
        schema,
        limits,
    )
    full_pool: list[SeedWarmupIntent] = list(gold_warmup_intents) + expanded_only
    pool_body_tier: set[tuple[str, str]] = set()
    deduped_pool: list[SeedWarmupIntent] = []
    for pool_intent in full_pool:
        bk = body_similarity_key(pool_intent.to_runtime_intent())
        tier = classify_seed_warmup_intent_complexity(pool_intent).value
        key = (bk, tier)
        if key in pool_body_tier:
            continue
        pool_body_tier.add(key)
        deduped_pool.append(pool_intent)

    log(f"[P3] Pool union and body dedupe (body_key,tier): {len(deduped_pool)} unique rows")

    tmpl_map: dict[str, Template] = templates if templates is not None else {}
    blocked_gold_rows = [
        row
        for row in deduped_pool
        if (row.source or "gold") == "gold" and _gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
    ]
    gold_warmup_blocked_path41_or_42 = len(blocked_gold_rows)
    warmup_queue = [row for row in deduped_pool if not _gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)]
    log(
        f"[P4] Gold vs store classification: gold_warmup_blocked_path41_or_42={gold_warmup_blocked_path41_or_42}; "
        f"queue {len(warmup_queue)} (expanded children keep distinct (body_key,tier))"
    )
    log("[P5] Synthetic rows filtered by template_instance_key / ledger inside execute loop")

    notify(
        "Phase B-C complete: expansion, pool dedupe, gold vs store classification "
        f"(expanded_synthetics={len(expanded_only)}, "
        f"unique_body_tier_rows={len(deduped_pool)}, "
        f"blocked_path_41_42={gold_warmup_blocked_path41_or_42}, "
        f"queued_for_warmup={len(warmup_queue)}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    log("Join resolution (cached per table-set)")
    join_cache: dict[frozenset[str], Any] = {}
    for gold in gold_warmup_intents:
        resolve_joins_for_table_set(
            gold.tables or [],
            schema,
            gold.intent_id or "gold",
            join_cache,
        )
    log(f"Join cache seeded with {len(join_cache)} table-set entries")
    log("[P7] Join cache seeded from gold table sets (reuse across pool)")
    notify(
        f"Phase D complete: join cache seeded (table_sets={len(join_cache)}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    with open(seed_filepath, "rb") as _seed_f:
        seed_content_sha256 = hashlib.sha256(_seed_f.read()).hexdigest()
    warmup_cache_session = open_seed_warmup_cache_session(
        output_dir,
        schema,
        seed_content_sha256,
    )
    log("[P0] Seed warmup cache manifest aligned to schema_hash and seed_content_hash")

    log(
        "[P8–P9] Execute and validate; stratified sampling after successes; "
        "[P10–P12] question LLM, realism, templates only on full run",
    )
    next_id = int(store.get("next_id", 1)) if store else 1
    join_intent_index = {row.intent_id: row for row in deduped_pool if getattr(row, "intent_id", None)}
    store_keys = accepted_template_instance_keys(tmpl_map)
    results, new_templates, updated_next_id, warmup_funnel = run_seed_warmup_execution(
        warmup_queue,
        schema,
        dialect,
        next_id,
        join_cache=join_cache,
        join_resolver_intent_index=join_intent_index,
        store_instance_keys=store_keys,
        accepted_templates=tmpl_map,
        warmup_run_mode="preflight" if warmup_dry_run_only else "full",
        warmup_cache=warmup_cache_session,
        warmup_report_version=version,
        warmup_dry_run_session=warmup_dry_run_only,
        warmup_lattice_root=output_dir,
    )
    save_seed_warmup_cache_zip(
        output_dir,
        warmup_cache_session.manifest,
        warmup_cache_session.work_units,
        gold_intent_dicts=[g.to_dict() for g in gold_warmup_intents],
    )
    for blocked in blocked_gold_rows:
        blocked_result = SeedWarmupResult(blocked.to_runtime_intent(), "")
        blocked_result.failure_code = "blocked_by_store_path_41_or_42"
        blocked_result.failure_stage = "gold_store_classification"
        blocked_result.drop_reason_category = "gold_store_classification"
        blocked_result.error = (
            "Gold intent skipped: existing store template covers it only via disallowed warmup paths 4.1 / 4.2."
        )
        results.append(blocked_result)
    log(f"Seed warmup execution results: {len(results)} rows, templates: {len(new_templates)}")
    exec_validation_drop = int(warmup_funnel.get("validation_drop", 0))
    exec_realism_drop = int(warmup_funnel.get("realism_drop", 0))
    exec_question_gen_failed = int(warmup_funnel.get("question_generation_failed", 0))
    exec_early_failed = int(warmup_funnel.get("early_pipeline_failed", 0))
    exec_preflight_ok = int(warmup_funnel.get("dry_run_execute_ok_count", 0))
    exec_total = len(results)
    exec_success = exec_preflight_ok if warmup_dry_run_only else sum(1 for r in results if r.success)
    notify(
        "Phase E complete: per-intent SQL build, validation, execution, realism gate "
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

    if store is not None and templates is not None and not warmup_dry_run_only:
        for tmpl in new_templates:
            dedupe_key = (
                tmpl.intent_key,
                getattr(
                    tmpl.intent_signature,
                    "chosen_join_candidate_id",
                    "",
                ),
                tmpl.sql_fp,
            )
            found = False
            for existing in templates.values():
                k = (
                    existing.intent_key,
                    getattr(
                        existing.intent_signature,
                        "chosen_join_candidate_id",
                        "",
                    ),
                    existing.sql_fp,
                )
                if k == dedupe_key:
                    found = True
                    for i, q in enumerate(tmpl.value_history.questions):
                        pv = tmpl.value_history.param_values[i] if i < len(tmpl.value_history.param_values) else {}
                        nl = (
                            tmpl.value_history.natural_language[i]
                            if i < len(tmpl.value_history.natural_language)
                            else ""
                        )
                        existing.value_history.add(pv, q, nl)
                    break
            if not found:
                templates[tmpl.id] = tmpl
        reconcile_template_store_until_stable(
            templates,
            template_store_view=store if isinstance(store, TemplateStoreView) else None,
        )
        store["next_id"] = updated_next_id
        store = templates_to_store(store, templates)
        save_template_store(store)

    templates_added = (
        len(new_templates) if store is not None and templates is not None and not warmup_dry_run_only else 0
    )

    exec_ok_ct = int(warmup_funnel.get("dry_run_execute_ok_count", 0))
    run_mode = "preflight" if warmup_dry_run_only else "full"
    registry_snapshot = {
        "run_mode": run_mode,
        "schema_hash": schema.schema_hash,
        "seed_content_hash": seed_content_sha256,
        "policy_version": warmup_cache_session.manifest.get("policy_version"),
        "code_version": warmup_cache_session.manifest.get("code_version"),
        "template_store_size_at_start": len(tmpl_map),
        "template_store_next_id_at_start": next_id,
        "template_store_next_id_at_end": updated_next_id,
        "work_units_total": len(warmup_cache_session.work_units),
        "work_units_touched_this_run": len(warmup_cache_session.touched_work_unit_ids),
    }
    save_seed_warmup_report(
        results,
        report_filepath,
        funnel={
            "seed_warmup_version": version,
            "warmup_dry_run_only": warmup_dry_run_only,
            "registry_snapshot": registry_snapshot,
            **gold_funnel,
            "synthetic_unique_body_keys": len(deduped_pool),
            "synthetic_runnable_count": len(warmup_queue),
            **warmup_pool_operator_feature_stats(warmup_queue),
            "gold_prompts_count": seed_questions_loaded,
            "templates_added": templates_added,
            "dry_run_execute_ok_count": exec_ok_ct,
            **warmup_funnel,
            "gold_warmup_blocked_path41_or_42": gold_warmup_blocked_path41_or_42,
        },
    )

    norm_json: str | None = None
    norm_txt: str | None = None
    if seed_norm_bundle is not None:
        norm_json, norm_txt = seed_norm_bundle
    if not warmup_dry_run_only:
        bundle_path = os.path.join(
            output_dir,
            SeedWarmupConfig.SEED_WARMUP_BUNDLE_PATTERN.format(version=version),
        )
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if norm_json is not None:
                zf.writestr(SEED_NORMALIZATION_JSON, norm_json)
            if norm_txt is not None:
                zf.writestr(NORMALIZED_SEEDS_TXT, norm_txt)
            if failure_trace_body:
                zf.writestr("gold_intent_failures_trace.txt", failure_trace_body)

    notify(
        "Phase F complete: seed warmup report"
        + (" and bundle zip" if not warmup_dry_run_only else "")
        + (
            ", template store updated"
            if store is not None and templates is not None and not warmup_dry_run_only
            else ""
        )
        + f" (templates_added={templates_added}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    total = len(results)
    if warmup_dry_run_only:
        success = exec_ok_ct
        failed = total - success
    else:
        success = sum(1 for r in results if r.success)
        failed = total - success
    success_rate = round(success / total, 3) if total > 0 else 0.0

    log(f"SEED WARMUP COMPLETE: {len(new_templates)} synthetic templates created")
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
    notify(_format_seed_warmup_summary(summary), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
    return None


def _raw_db_connection_for_query_log(dialect: Any) -> Any:
    """Best-effort DBAPI handle for query-log probes."""

    eng = getattr(dialect, "engine", None)
    if eng is not None:
        try:
            return eng.raw_connection()
        except Exception:
            pass
    return getattr(dialect, "connection", None)


def _dialect_name_for_query_log(dialect: Any) -> str:
    """Return ``postgresql`` or ``databricks`` for query-log dispatch."""

    if isinstance(dialect, PostgresDialect):
        return "postgresql"
    if isinstance(dialect, DatabricksDialect):
        return "databricks"
    return "postgresql"


def _run_seed_warmup_sql_history_pipeline(
    *,
    schema: SchemaGraph,
    dialect: Any,
    output_dir: str,
    store: dict[str, Any] | None,
    templates: dict[str, Template] | None,
    sql_texts: list[str],
    sql_history_content_hash: str,
    warmup_dry_run_only: bool,
    seed: int | None,
) -> None:
    """Execute seed warmup over converted SQL-history intents sharing cache keyed by *sql_history_content_hash*."""

    if seed is None:
        seed = SeedWarmupConfig.RANDOM_SEED
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    if warmup_dry_run_only:
        version = get_next_warmup_preflight_version(output_dir)
        report_name = SeedWarmupConfig.WARMUP_PREFLIGHT_REPORT_PATTERN.format(version=version)
    else:
        version = get_next_seed_warmup_version(output_dir)
        report_name = SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version)
    report_filepath = os.path.join(output_dir, report_name)

    log(f"Starting SQL-history seed warmup {'preflight' if warmup_dry_run_only else 'run'} version {version}")
    notify(
        f"SQL-history seed warmup {'preflight' if warmup_dry_run_only else 'run'} version: {version}",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    schema_stats = schema.ensure_schema_stats()
    limits = compute_schema_limits(schema_stats)
    log(
        f"Computed SchemaLimits: max_filters={limits.max_filters}, "
        f"max_groupby={limits.max_groupby}, "
        f"max_tables={limits.max_tables}",
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
            seed_warmup_intent_from_runtime_intent(
                rt,
                intent_id=f"sqlhist_{idx}_{bk[:24]}",
                seed_index=idx,
            ),
        )

    seed_questions_loaded = len(sql_texts)
    gold_warmup_intents = warmup_queue
    deduped_pool = warmup_queue
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
        "Phase A complete: SQL history conversion "
        f"(lines={seed_questions_loaded}, "
        f"converted_intents={len(warmup_queue)}, "
        f"conversion_failed={gold_failed_count}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    tmpl_map: dict[str, Template] = templates if templates is not None else {}
    gold_warmup_blocked_path41_or_42 = 0

    join_cache: dict[frozenset[str], Any] = {}
    for row in warmup_queue:
        resolve_joins_for_table_set(
            row.tables or [],
            schema,
            row.intent_id or "sqlhist",
            join_cache,
        )
    log(f"Join cache seeded with {len(join_cache)} table-set entries")

    warmup_cache_session = open_seed_warmup_cache_session(
        output_dir,
        schema,
        sql_history_content_sha256=sql_history_content_hash,
    )
    log("[P0] Seed warmup cache manifest aligned to schema_hash and sql_history_content_hash")

    next_id = int(store.get("next_id", 1)) if store else 1
    join_intent_index = {row.intent_id: row for row in warmup_queue if getattr(row, "intent_id", None)}
    store_keys = accepted_template_instance_keys(tmpl_map)
    results, new_templates, updated_next_id, warmup_funnel = run_seed_warmup_execution(
        warmup_queue,
        schema,
        dialect,
        next_id,
        join_cache=join_cache,
        join_resolver_intent_index=join_intent_index,
        store_instance_keys=store_keys,
        accepted_templates=tmpl_map,
        warmup_run_mode="preflight" if warmup_dry_run_only else "full",
        warmup_cache=warmup_cache_session,
        warmup_report_version=version,
        warmup_dry_run_session=warmup_dry_run_only,
        warmup_lattice_root=output_dir,
    )
    save_seed_warmup_cache_zip(
        output_dir,
        warmup_cache_session.manifest,
        warmup_cache_session.work_units,
        gold_intent_dicts=[g.to_dict() for g in gold_warmup_intents],
    )

    log(f"SQL-history seed warmup execution results: {len(results)} rows, templates: {len(new_templates)}")
    exec_validation_drop = int(warmup_funnel.get("validation_drop", 0))
    exec_realism_drop = int(warmup_funnel.get("realism_drop", 0))
    exec_question_gen_failed = int(warmup_funnel.get("question_generation_failed", 0))
    exec_early_failed = int(warmup_funnel.get("early_pipeline_failed", 0))
    exec_preflight_ok = int(warmup_funnel.get("dry_run_execute_ok_count", 0))
    exec_total = len(results)
    exec_success = exec_preflight_ok if warmup_dry_run_only else sum(1 for r in results if r.success)
    notify(
        "Phase E complete: per-intent SQL build, validation, execution "
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

    if store is not None and templates is not None and not warmup_dry_run_only:
        for tmpl in new_templates:
            dedupe_key = (
                tmpl.intent_key,
                getattr(
                    tmpl.intent_signature,
                    "chosen_join_candidate_id",
                    "",
                ),
                tmpl.sql_fp,
            )
            found = False
            for existing in templates.values():
                k = (
                    existing.intent_key,
                    getattr(
                        existing.intent_signature,
                        "chosen_join_candidate_id",
                        "",
                    ),
                    existing.sql_fp,
                )
                if k == dedupe_key:
                    found = True
                    for i, q in enumerate(tmpl.value_history.questions):
                        pv = tmpl.value_history.param_values[i] if i < len(tmpl.value_history.param_values) else {}
                        nl = (
                            tmpl.value_history.natural_language[i]
                            if i < len(tmpl.value_history.natural_language)
                            else ""
                        )
                        existing.value_history.add(pv, q, nl)
                    break
            if not found:
                templates[tmpl.id] = tmpl
        reconcile_template_store_until_stable(
            templates,
            template_store_view=store if isinstance(store, TemplateStoreView) else None,
        )
        store["next_id"] = updated_next_id
        store = templates_to_store(store, templates)
        save_template_store(store)

    templates_added = (
        len(new_templates) if store is not None and templates is not None and not warmup_dry_run_only else 0
    )

    exec_ok_ct = int(warmup_funnel.get("dry_run_execute_ok_count", 0))
    run_mode = "preflight" if warmup_dry_run_only else "full"
    registry_snapshot = {
        "run_mode": run_mode,
        "schema_hash": schema.schema_hash,
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
    save_seed_warmup_report(
        results,
        report_filepath,
        funnel={
            "seed_warmup_version": version,
            "warmup_dry_run_only": warmup_dry_run_only,
            "registry_snapshot": registry_snapshot,
            **gold_funnel,
            "sql_history_conversion_failures": len(fail_by_hash),
            "synthetic_unique_body_keys": len(deduped_pool),
            "synthetic_runnable_count": len(warmup_queue),
            **warmup_pool_operator_feature_stats(warmup_queue),
            "gold_prompts_count": seed_questions_loaded,
            "templates_added": templates_added,
            "dry_run_execute_ok_count": exec_ok_ct,
            **warmup_funnel,
            "gold_warmup_blocked_path41_or_42": gold_warmup_blocked_path41_or_42,
        },
    )

    notify(
        "Phase F complete: seed warmup report"
        + (" and bundle zip" if not warmup_dry_run_only else "")
        + (
            ", template store updated"
            if store is not None and templates is not None and not warmup_dry_run_only
            else ""
        )
        + f" (templates_added={templates_added}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    total = len(results)
    if warmup_dry_run_only:
        success = exec_ok_ct
        failed = total - success
    else:
        success = sum(1 for r in results if r.success)
        failed = total - success
    success_rate = round(success / total, 3) if total > 0 else 0.0

    log(f"SQL-HISTORY SEED WARMUP COMPLETE: {len(new_templates)} synthetic templates created")
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
    notify(_format_seed_warmup_summary(summary), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def run_seed_warmup_from_history_execution(
    self_text2sql: Any,
    sql_history_filepath: str,
    *,
    warmup_dry_run_only: bool = False,
    seed: int | None = None,
) -> None:
    """Drive :func:`run_seed_warmup_execution` from a newline-oriented SQL history file."""

    schema = self_text2sql._schema_graph
    dialect = self_text2sql._dialect
    output_dir = str(self_text2sql._artifacts_dir)
    store = self_text2sql._store
    templates = self_text2sql._templates
    statements = load_sql_history_statements(sql_history_filepath)
    content_hash = compute_sql_history_content_hash(statements)
    _run_seed_warmup_sql_history_pipeline(
        schema=schema,
        dialect=dialect,
        output_dir=output_dir,
        store=store,
        templates=templates,
        sql_texts=statements,
        sql_history_content_hash=content_hash,
        warmup_dry_run_only=warmup_dry_run_only,
        seed=seed,
    )


def run_seed_warmup_from_query_log_execution(
    self_text2sql: Any,
    *,
    lookback_days: int = 730,
    max_queries: int = 5000,
    min_runs: int = 1,
    user_filter: str | None = None,
    warmup_dry_run_only: bool = False,
    seed: int | None = None,
) -> None:
    """Drive :func:`run_seed_warmup_execution` from the engine query log."""

    schema = self_text2sql._schema_graph
    dialect = self_text2sql._dialect
    output_dir = str(self_text2sql._artifacts_dir)
    store = self_text2sql._store
    templates = self_text2sql._templates
    conn = _raw_db_connection_for_query_log(dialect)
    dialect_name = _dialect_name_for_query_log(dialect)
    sql_texts = fetch_query_log(
        dialect_name,
        conn,
        lookback_days=lookback_days,
        max_queries=max_queries,
        min_runs=min_runs,
        user_filter=user_filter,
    )
    content_hash = compute_sql_history_content_hash(sql_texts)
    _run_seed_warmup_sql_history_pipeline(
        schema=schema,
        dialect=dialect,
        output_dir=output_dir,
        store=store,
        templates=templates,
        sql_texts=sql_texts,
        sql_history_content_hash=content_hash,
        warmup_dry_run_only=warmup_dry_run_only,
        seed=seed,
    )


def _sql_feedback_suspend_context(
    snap_post: InteractiveTailSnapshot,
    sql: str,
    rows: list[tuple[Any, ...]],
    conf: float,
    tmpl_sd: dict[str, Any] | None,
    gen_out: SqlGenerationOutcome,
    matched_rejected_template: Any,
    force_feedback: bool,
    execution_intent: Any,
) -> SqlFeedbackSuspendContext:
    """Build a frozen payload for deferred SQL accept/reject."""

    return SqlFeedbackSuspendContext(
        tail=snap_post,
        execution_intent=execution_intent,
        sql=sql,
        rows=tuple(tuple(r) for r in rows),
        conf=conf,
        tmpl_sd=tmpl_sd,
        gen_out=gen_out,
        matched_rejected_template=matched_rejected_template,
        force_feedback=force_feedback,
    )


def _run_interactive_join_through_feedback(
    q_norm: str,
    intent: Any,
    semantic_warnings: list[dict[str, Any]],
    dialect: Any,
    schema: SchemaGraph,
    store: dict[str, Any],
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

    (
        matched_template,
        union_select_cols,
        cols_changed,
        union_sql_path,
        has_union_match,
        join_candidates,
        cmap,
        cte_join_hints,
    ) = prepare_union_match_join_phase(q_norm, intent, schema, dialect, templates, store=store)

    union_cand_ids = [c.template.id for c in list_union_match_candidates(intent, templates)]
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

    _run_sql_phase_after_intent_confirm(
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


def _run_sql_phase_after_intent_confirm(
    *,
    q_norm: str,
    intent: Any,
    schema: SchemaGraph,
    store: dict[str, Any],
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
    )
    if not gen_out.success:
        _note_interactive_turn(
            choice_port,
            outcome="validation_failed",
            error=gen_out.sql_validation_error,
        )
        if persist_template_learning:
            save_template_store(store)
        return None

    sql = gen_out.sql

    tmpl_sd = getattr(gen_out.matched_template, "structural_defaults", None) if gen_out.matched_template else None

    log("executing SQL")
    exec_sql = dialect.finalize_render(
        intent.sql_param or "",
        dict(flatten_param_values(intent)),
        schema=schema,
        intent=intent,
        execution_sql_override=None,
        structural_defaults=tmpl_sd,
    )
    _core_utils.progress("Executing SQL...")
    try:
        rows = dialect.execute(exec_sql)
        if len(rows) > int(PolicyConfig.RESULT_ROW_COUNT_SOFT_WARNING):
            _core_utils.notify(
                f"Query result row count {len(rows)} exceeds the soft warning threshold.",
                stage="execution",
                code=DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
            )
    except AccessError as exc:
        _note_interactive_turn(
            choice_port,
            outcome="permission_denied",
            error=str(exc),
        )
        if persist_template_learning:
            save_template_store(store)
        return None

    conf = compute_final_metrics(
        sql,
        intent,
        schema,
        templates,
        join_candidates,
        store,
        q_norm=snap_post.q_norm,
        explain_soft_diagnostics=getattr(gen_out, "explain_soft_diagnostics", 0),
    )

    if conf < PolicyConfig.FINAL_SQL_AUTO_ACCEPT_THRESHOLD:
        _core_utils.notify(
            f"SQL confidence {conf:.3f} is below the auto-accept threshold.",
            stage="pipeline",
            code=DIAGNOSTIC_CODE_LOW_CONFIDENCE,
        )

    force_feedback = has_any_rejection_history_for_question(store, snap_post.q_norm)
    if gen_out.matched_template is not None and not should_auto_accept_for_question(
        gen_out.matched_template, snap_post.q_norm
    ):
        force_feedback = True
    need_sql_feedback_prompt = force_feedback or conf < PolicyConfig.FINAL_SQL_AUTO_ACCEPT_THRESHOLD
    is_session = choice_port is not None and isinstance(choice_port, PipelineSession)
    if not is_session:
        display_final_results_to_stdout(
            q_norm,
            intent,
            sql,
            rows,
            structural_defaults=tmpl_sd,
            template_display_alias_map=(
                getattr(gen_out.matched_template, "display_alias_map", None)
                if gen_out.matched_template
                else None
            ),
        )

    sql_prompt = "Is this correct?"
    if need_sql_feedback_prompt:
        if choice_port is not None and not choice_port.has_pending_choice():
            raise PipelineSuspended(
                PIPELINE_SUSPEND_ID_SQL,
                sql_prompt,
                _sql_feedback_suspend_context(
                    snap_post,
                    sql,
                    rows,
                    conf,
                    tmpl_sd,
                    gen_out,
                    matched_rejected_template,
                    force_feedback,
                    intent,
                ),
            )
        choice = interactive_yes_no(
            INTERACTIVE_STAGE_SQL_FEEDBACK,
            sql_prompt,
            ["y", "n"],
            choice_port=choice_port,
        )
    else:
        log(f"[AUTO-ACCEPT] confidence={conf:.3f} >= {PolicyConfig.FINAL_SQL_AUTO_ACCEPT_THRESHOLD}")
        choice = "y"
    if choice is None:
        _note_interactive_turn(choice_port, outcome="user_declined", error="User cancelled SQL feedback.")
        if persist_template_learning:
            save_template_store(store)
        return None

    if choice == "y" and intent.grain != "scalar":
        df_full = build_result_dataframe(
            rows,
            intent,
            sql,
            structural_defaults=tmpl_sd,
            q_norm=q_norm,
            template_display_alias_map=(
                getattr(gen_out.matched_template, "display_alias_map", None)
                if gen_out.matched_template
                else None
            ),
        )
        if df_full is not None:
            save_result_csv(df_full)

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
    )
    row_tuples = [tuple(r) for r in rows]
    cols = _result_columns_for_session(sql, row_tuples)
    if choice == "n":
        rb: str | None = None
        if isinstance(feedback_result, dict):
            rb = str(feedback_result.get("category") or "").strip().upper() or None
        _note_interactive_turn(
            choice_port,
            outcome="intent_rejected",
            sql=sql,
            rows=row_tuples,
            columns=cols,
            intent=intent,
            rejection_bucket=rb,
        )
    else:
        _note_interactive_turn(choice_port, outcome="success", sql=sql, rows=row_tuples, columns=cols, intent=intent)


def _run_interactive_after_parsed_intent(
    q_norm: str,
    intent: Any,
    semantic_warnings: list[dict[str, Any]],
    dialect: Any,
    schema: SchemaGraph,
    store: dict[str, Any],
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema_terms: set[str],
    choice_port: InteractiveChoicePort | None,
    intent_already_confirmed: bool = False,
    form_storage: QuestionFormStorage | None = None,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
) -> None:
    """Match union templates, confirm intent with the user, then continue through SQL feedback."""

    ikey = intent_key(intent)
    debug(f"[main_execution._run_interactive_after_parsed_intent] intent_key: {ikey[:32]}")

    union_result = match_template_for_union(intent, templates)
    structural_match_templates = collect_structural_match_templates(intent, templates)
    matched_template = None
    union_select_cols = None
    cols_changed = False
    union_sql_path: GenerationPath | None = None
    has_union_match = union_result is not None
    if union_result is not None:
        matched_template, union_select_cols, cols_changed, union_sql_path = union_result

    intent_sim = best_accepted_template_similarity(intent, templates)
    union_cand_ids = [c.template.id for c in list_union_match_candidates(intent, templates)]
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
    )
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
        _note_interactive_turn(
            choice_port,
            outcome="user_declined",
            error="User declined intent confirmation.",
        )
        return None

    _run_interactive_join_through_feedback(
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


def _run_interactive_after_parsed_intent_from_tail(
    tail: InteractiveTailSnapshot,
    choice_port: InteractiveChoicePort | None,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
) -> None:
    """Resume after a deferred intent confirmation using a frozen tail snapshot."""

    _run_interactive_after_parsed_intent(
        tail.q_norm,
        tail.intent,
        list(tail.semantic_warnings),
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


def _run_interactive_post_intent_parse(
    q_norm: str,
    intent: Any,
    semantic_warnings: list[dict[str, Any]],
    dialect: Any,
    schema: SchemaGraph,
    store: dict[str, Any],
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema_terms: set[str],
    choice_port: InteractiveChoicePort | None,
    form_storage: QuestionFormStorage | None = None,
    refinement_ctx: RefinementContext | None = None,
    persist_template_learning: bool = True,
) -> None:
    """
    Continue the interactive pipeline after a parsed intent (joins through feedback).

    Args:

        q_norm: Normalised question text.

        intent: Parsed ``RuntimeIntent``.

        semantic_warnings: Parser warning payloads.

        dialect: Active dialect instance.

        schema: Schema graph.

        store: Template store.

        templates: Accepted template map.

        rejected: Rejected-template map.

        schema_terms: Schema vocabulary tokens.

        choice_port: Optional session port for deferred prompts.

    Returns:

        None.
    """

    _run_interactive_after_parsed_intent(
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
    )


def _complete_interactive_sql_feedback(
    ctx: SqlFeedbackSuspendContext,
    choice: str | None,
    *,
    choice_port: InteractiveChoicePort | None = None,
) -> None:
    """
    Apply accept or reject after a deferred final-SQL prompt.

    Args:

        ctx: Frozen suspend context from the SQL feedback stage.

        choice: Normalised ``"y"`` or ``"n"``, or ``None`` when cancelled.

    Returns:

        None.
    """

    tail = ctx.tail
    intent = ctx.execution_intent
    sql = ctx.sql
    rows = [tuple(r) for r in ctx.rows]
    tmpl_sd = ctx.tmpl_sd
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)
    if choice is None:
        if persist_tl:
            save_template_store(tail.store)
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
            )
            if df_full is not None:
                save_result_csv(df_full)
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
    )
    row_tuples = [tuple(r) for r in rows]
    cols = _result_columns_for_session(sql, row_tuples)
    if choice == "n":
        rb: str | None = None
        if isinstance(feedback_result, dict):
            rb = str(feedback_result.get("category") or "").strip().upper() or None
        _note_interactive_turn(
            choice_port,
            outcome="intent_rejected",
            sql=sql,
            rows=row_tuples,
            columns=cols,
            intent=intent,
            rejection_bucket=rb,
        )
    else:
        _note_interactive_turn(choice_port, outcome="success", sql=sql, rows=row_tuples, columns=cols, intent=intent)


def _complete_intent_rejection_feedback(
    tail: InteractiveTailSnapshot,
    feedback: str | None,
    choice_port: InteractiveChoicePort | None,
) -> None:
    """Persist free-text feedback after the user declines an intent."""

    body = (feedback or "").strip() or "user_declined_intent"
    entry = summarize_failure_for_memory(
        question=tail.q_norm,
        intent=tail.intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=tail.schema.effective_structural_hash,
        user_reason=body,
        sql=None,
    )
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)
    if persist_tl:
        record_question_feedback(tail.store, tail.q_norm, entry)
        save_template_store(tail.store)
    rb = entry.buckets[0].value if entry.buckets else None
    ctx_ref = getattr(choice_port, "_refinement_ctx", None)
    reason_line = body
    if ctx_ref is not None and _refinement_retry_available(ctx_ref):
        ctx_ref.accumulated_reasons.append(reason_line)
        ctx_ref.pending_retry = True
        raise RefinementRetry
    print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=rb)
    _note_interactive_turn(
        choice_port,
        outcome="user_declined",
        error="User declined intent confirmation.",
    )


def dispatch_pipeline_resume(session: Any, suspended: PipelineSuspended) -> None:
    """
    Drive the next pipeline segment after the caller enqueued a programmatic choice.

    Args:

        session: Interactive session exposing resources and the choice queue API.

        suspended: Prior ``PipelineSuspended`` instance holding ``state_id`` and payload.

    Returns:

        None.

    Raises:

        PipelineSuspended: When another prompt must be deferred.
    """

    sid = suspended.state_id
    payload = suspended.payload
    persist_tl = _persist_template_learning_for_pipeline_session(session)
    if sid == PIPELINE_SUSPEND_ID_DIRECT_REUSE:
        ch = session._consume_next_queued_choice()
        if not isinstance(payload, DirectReuseSuspendContext):
            raise TypeError("direct reuse resume expects DirectReuseSuspendContext")
        complete_direct_sql_reuse_user_choice(payload, ch, choice_port=session, persist_template_learning=persist_tl)
        return
    if sid == PIPELINE_SUSPEND_ID_INTENT_CONFIRM:
        ch = session._consume_next_queued_choice()
        if not isinstance(payload, InteractiveTailSnapshot):
            raise TypeError("intent resume expects InteractiveTailSnapshot")
        if ch is None:
            raise PipelineSuspended(
                "empty_choice_queue",
                "interactive choice queue is empty",
                None,
            )
        _run_interactive_after_parsed_intent_from_tail(
            payload,
            session,
            refinement_ctx=getattr(session, "_refinement_ctx", None),
            persist_template_learning=persist_tl,
        )
        return
    if sid == PIPELINE_SUSPEND_ID_SQL:
        ch = session._consume_next_queued_choice()
        if not isinstance(payload, SqlFeedbackSuspendContext):
            raise TypeError("SQL feedback resume expects SqlFeedbackSuspendContext")
        _complete_interactive_sql_feedback(payload, ch, choice_port=session)
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
        _complete_intent_rejection_feedback(payload, ch, session)
        return
    raise RuntimeError(f"unknown pipeline suspend id: {sid!r}")


def interactive_run_once(
    schema: SchemaGraph | None = None,
    store: Any | None = None,
    templates: list | None = None,
    rejected: list | None = None,
    schema_terms: Any | None = None,
    question: str | None = None,
    pipeline_session: Any | None = None,
) -> dict[str, Any] | None:
    """
    Execute a single interactive pipeline iteration.

    Reads a question from stdin or uses the supplied `question`, validates it, checks for template reuse, parses intent via LLM if needed, generates SQL, executes it, and handles user feedback.

    Args:

        schema: Pre-loaded `SchemaGraph`; raises when `None`.

        store: Template store dict; raises when `None`.

        templates: List of accepted `Template` objects; raises when `None`.

        rejected: List of `RejectedTemplate` objects; raises when `None`.

        schema_terms: Set of schema term tokens; raises when `None`.

        question: When provided, the pipeline uses this text instead of reading from stdin.

        pipeline_session: Optional session implementing the interactive choice port for suspend or resume.

    Returns:

        A dict with pipeline results on a full run, or `None` on early exit.
    """
    if question is None:
        notify("Enter question", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
        try:
            question = _core_utils.prompt("").strip()
        except (EOFError, KeyboardInterrupt):
            _core_utils.terminated()
            return None

    if not question:
        _core_utils.invalid_input()
        return None
    _core_utils.progress("\nValidating question...")

    raw_question = question

    dialect, schema, store, templates, rejected, schema_terms = load_pipeline_resources(
        schema, store, templates, rejected, schema_terms
    )
    choice_port: InteractiveChoicePort | None = pipeline_session
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)

    tmpl_pre = match_question_level_template_reuse(raw_question, templates, template_store=store)
    if tmpl_pre.reuse_type == "direct_reuse":
        log(f"direct SQL reuse via question match pre-validation (trust>=1, template='{tmpl_pre.best_template.id}')")
        debug("[main_execution.interactive_run_once] direct_reuse_pre: question_match")
        assert tmpl_pre.reuse_candidate_normalized is not None
        reuse_pre = handle_direct_sql_reuse(
            tmpl_pre.reuse_candidate_normalized,
            tmpl_pre.best_template,
            dialect,
            store,
            templates,
            rejected,
            schema,
            existing_nl=None,
            choice_port=choice_port,
            reuse_history_index=tmpl_pre.reuse_history_index,
            form_storage=QuestionFormStorage(corrected=raw_question.strip()),
            persist_template_learning=persist_tl,
        )
        if reuse_pre is not None and reuse_pre.success:
            return None

    valid, query_type, corrected = validate_question(raw_question)
    if not valid:
        if query_type == "restricted":
            print_rephrase_hint(RephraseHint.RESTRICTED_QUESTION)
            _note_interactive_turn(
                choice_port,
                outcome="parse_failed",
                error="Question rejected as restricted.",
            )
        else:
            print_rephrase_hint(RephraseHint.VAGUE_QUESTION)
            _note_interactive_turn(
                choice_port,
                outcome="parse_failed",
                error="Question failed validation.",
            )
        return None
    corrected_text = corrected
    if corrected_text != raw_question:
        debug(f"[main_execution.interactive_run_once] typo_corrected: '{raw_question}' -> '{corrected_text}'")

    tmpl_typo = match_question_level_template_reuse(corrected_text, templates, template_store=store)
    if tmpl_typo.reuse_type == "direct_reuse":
        log(f"direct SQL reuse via question match (trust>=1, template='{tmpl_typo.best_template.id}')")
        debug("[main_execution.interactive_run_once] direct_reuse: question_match")
        assert tmpl_typo.reuse_candidate_normalized is not None
        reuse_result = handle_direct_sql_reuse(
            tmpl_typo.reuse_candidate_normalized,
            tmpl_typo.best_template,
            dialect,
            store,
            templates,
            rejected,
            schema,
            existing_nl=None,
            choice_port=choice_port,
            reuse_history_index=tmpl_typo.reuse_history_index,
            form_storage=QuestionFormStorage(corrected=corrected_text),
            persist_template_learning=persist_tl,
        )
        if reuse_result is not None and reuse_result.success:
            return None

    neg_drop = False
    normalized_canonical = normalize_question_via_llm(corrected_text, raw_original=raw_question)
    if (
        normalized_canonical != corrected_text
        and has_any_rejection_history_for_question(store, corrected_text)
    ):
        debug(
            f"[main_execution.interactive_run_once] dropped_normalized_due_to_negative_memory "
            f"{normalized_canonical!r}"
        )
        neg_drop = True
        normalized_canonical = corrected_text

    tmpl_norm = None
    if normalized_canonical != corrected_text:
        tmpl_norm = match_question_level_template_reuse(normalized_canonical, templates, template_store=store)
        if tmpl_norm.reuse_type == "direct_reuse":
            log(
                f"direct SQL reuse via normalized question match (trust>=1, template='{tmpl_norm.best_template.id}')"
            )
            assert tmpl_norm.reuse_candidate_normalized is not None
            fs_norm = QuestionFormStorage(
                corrected=corrected_text,
                normalized_optional=normalized_canonical,
                normalized_negative_memory_dropped=neg_drop,
                accept_via_normalized_lookup_only=True,
            )
            reuse_norm = handle_direct_sql_reuse(
                tmpl_norm.reuse_candidate_normalized,
                tmpl_norm.best_template,
                dialect,
                store,
                templates,
                rejected,
                schema,
                existing_nl=None,
                choice_port=choice_port,
                reuse_history_index=tmpl_norm.reuse_history_index,
                form_storage=fs_norm,
                persist_template_learning=persist_tl,
            )
            if reuse_norm is not None and reuse_norm.success:
                return None

    norm_opt = normalized_canonical if normalized_canonical != corrected_text else None
    form_storage = QuestionFormStorage(
        corrected=corrected_text,
        normalized_optional=norm_opt,
        normalized_negative_memory_dropped=neg_drop,
        accept_via_normalized_lookup_only=False,
    )

    q_norm = normalize_question(corrected_text)
    debug(f"[main_execution.interactive_run_once] q_norm: {q_norm}")

    conv_hints: tuple[str, ...] = ()
    if choice_port is not None:
        raw_h = getattr(choice_port, "_pending_conversation_rejection_hints", None)
        if isinstance(raw_h, tuple):
            conv_hints = raw_h
            setattr(choice_port, "_pending_conversation_rejection_hints", ())

    refinement_ctx = RefinementContext(
        corrected_text,
        form_storage,
        conversation_rejection_hints=conv_hints,
    )
    _interactive_attach_refinement_ctx(choice_port, refinement_ctx)

    while True:
        try:
            completed = _interactive_run_intent_pass(
                corrected_text=corrected_text,
                q_norm=q_norm,
                dialect=dialect,
                schema=schema,
                store=store,
                templates=templates,
                rejected=rejected,
                schema_terms=schema_terms,
                choice_port=choice_port,
                form_storage=form_storage,
                refinement_ctx=refinement_ctx,
                persist_template_learning=persist_tl,
            )
            if not completed:
                return None
            break
        except RefinementRetry:
            continue


def get_seed_warmup_summary_from_dir(artifacts_dir: str, version: int) -> SeedWarmupSummary:
    """Build a ``SeedWarmupSummary`` from a persisted ``seed_warmup_report_v{version}.json`` file."""

    report_path = os.path.join(
        artifacts_dir,
        SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version),
    )
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Seed warmup report v{version} not found")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    total = report.get("total", 0)
    success = report.get("success", 0)
    failed = report.get("failed", 0)
    success_rate = round(success / total, 3) if total > 0 else 0.0

    return SeedWarmupSummary(
        version=version,
        total=total,
        success=success,
        failed=failed,
        success_rate=success_rate,
        seed_questions_loaded=int(report.get("seed_questions_loaded", 0)),
        gold_intents_total=int(report.get("gold_intents_total", 0)),
        unique_prompts=int(
            report.get(
                "unique_prompts",
                report.get("synthetic_runnable_count", report.get("unique_synthetic", 0)),
            ),
        ),
        gold_new=int(report.get("gold_new", 0)),
        gold_skipped=int(report.get("gold_skipped", 0)),
        gold_failed=int(report.get("gold_failed", 0)),
        gold_user_rejected=int(report.get("gold_user_rejected", 0)),
        deduped_prompts_count=int(
            report.get(
                "deduped_prompts_count",
                report.get(
                    "synthetic_unique_body_keys",
                    report.get("deduped_synthetic_count", 0),
                ),
            ),
        ),
        gold_prompts_count=int(
            report.get("gold_prompts_count", report.get("seed_questions_loaded", 0)),
        ),
        templates_added=int(report.get("templates_added", 0)),
        validation_drop=int(report.get("validation_drop", 0)),
        realism_drop=int(report.get("realism_drop", 0)),
        question_generation_failed=int(report.get("question_generation_failed", 0)),
        early_pipeline_failed=int(report.get("early_pipeline_failed", 0)),
    )


def _load_config_file(path: str | os.PathLike[str] | None) -> tuple[dict[str, str], frozenset[str]]:
    """
    Parse a TOML configuration file into flat environment-style string keys.

    Args:

        path: Optional filesystem path to a TOML file.

    Returns:

        A pair ``(values, claimed_env_keys)``. *values* maps canonical uppercase environment
        variable names to non-empty string values. *claimed_env_keys* lists every mapped env
        key that appears as a field in the TOML (including fields set to empty strings), so
        callers can treat the file as authoritative for those keys.

    Raises:

        ConfigError: When the file cannot be opened or parsed, or when the document root is not a table.
    """

    if path is None:
        return {}, frozenset()
    path_str = str(path).strip()
    if not path_str:
        return {}, frozenset()
    expanded = os.path.expanduser(path_str)
    try:
        with open(expanded, "rb") as file_handle:
            document = tomllib.load(file_handle)
    except OSError as exc:
        raise ConfigError(f"config_file cannot be opened: {expanded}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config_file TOML parse error in {expanded}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError(f"config_file root must be a table: {expanded}")
    output: dict[str, str] = {}
    claimed: set[str] = set()

    def _claim_put(block: dict[str, Any], subkey: str, target_key: str) -> None:
        if subkey not in block:
            return
        claimed.add(target_key)
        raw_value = block.get(subkey)
        if raw_value is None:
            return
        text = str(raw_value).strip()
        if text:
            output[target_key] = text

    openai_block = document.get("openai")
    if isinstance(openai_block, dict):
        _claim_put(openai_block, "api_key", "OPENAI_API_KEY")
        _claim_put(openai_block, "base_url", "OPENAI_BASE_URL")
    azure_block = document.get("azure_openai")
    if isinstance(azure_block, dict):
        _claim_put(azure_block, "endpoint", "AZURE_OPENAI_ENDPOINT")
        _claim_put(azure_block, "api_key", "AZURE_OPENAI_API_KEY")
        _claim_put(azure_block, "api_version", "AZURE_OPENAI_API_VERSION")
        _claim_put(azure_block, "base_url", "AZURE_OPENAI_BASE_URL")
        deployments_block = azure_block.get("deployments")
        if isinstance(deployments_block, dict):
            _claim_put(deployments_block, "light", "AZURE_OPENAI_DEPLOYMENT_LIGHT")
            _claim_put(deployments_block, "medium", "AZURE_OPENAI_DEPLOYMENT_MEDIUM")
            _claim_put(deployments_block, "heavy", "AZURE_OPENAI_DEPLOYMENT_HEAVY")
    postgres_block = document.get("postgresql")
    if isinstance(postgres_block, dict):
        _claim_put(postgres_block, "host", "POSTGRES_HOST")
        _claim_put(postgres_block, "port", "POSTGRES_PORT")
        _claim_put(postgres_block, "database", "POSTGRES_DB")
        _claim_put(postgres_block, "schema", "POSTGRES_SCHEMA")
        _claim_put(postgres_block, "user", "POSTGRES_USER")
        _claim_put(postgres_block, "password", "POSTGRES_PASSWORD")
    databricks_block = document.get("databricks")
    if isinstance(databricks_block, dict):
        _claim_put(databricks_block, "host", "DATABRICKS_HOST")
        _claim_put(databricks_block, "http_path", "DATABRICKS_HTTP_PATH")
        _claim_put(databricks_block, "access_token", "DATABRICKS_ACCESS_TOKEN")
        _claim_put(databricks_block, "catalog", "DATABRICKS_CATALOG")
        _claim_put(databricks_block, "schema", "DATABRICKS_SCHEMA")
        _claim_put(databricks_block, "cluster_id", "DATABRICKS_CLUSTER_ID")
    engine_block = document.get("engine")
    if isinstance(engine_block, dict):
        _claim_put(engine_block, "selected", "AETHERDIALECT_ENGINE")
    llm_block = document.get("llm")
    if isinstance(llm_block, dict):
        _claim_put(llm_block, "provider", "AETHERDIALECT_LLM_PROVIDER")
    execution_block = document.get("execution")
    if isinstance(execution_block, dict):
        _claim_put(execution_block, "max_query_cost_rows", "AETHERDIALECT_MAX_QUERY_COST_ROWS")
        _claim_put(execution_block, "max_query_cost_bytes", "AETHERDIALECT_MAX_QUERY_COST_BYTES")
        _claim_put(execution_block, "statement_timeout_ms", "AETHERDIALECT_STATEMENT_TIMEOUT_MS")
        _claim_put(execution_block, "llm_timeout_ms", "AETHERDIALECT_LLM_TIMEOUT_MS")
        _claim_put(execution_block, "profile_timeout_ms", "AETHERDIALECT_PROFILE_TIMEOUT_MS")
        _claim_put(execution_block, "explain_timeout_ms", "AETHERDIALECT_EXPLAIN_TIMEOUT_MS")
    return output, frozenset(claimed)


def _merge_configuration_environment(
    config_file_values: Mapping[str, str],
    *,
    toml_claimed_keys: frozenset[str] | None = None,
) -> tuple[dict[str, str], frozenset[str]]:
    """
    Build the effective environment mapping used for engine configuration reads.

    When *toml_claimed_keys* is ``None`` (no ``config_file`` in use), non-empty TOML values
    overlay ``os.environ`` for matching keys only.

    When *toml_claimed_keys* is provided (a ``config_file`` was loaded), the file is the single
    source of truth for every key in that set: non-empty flattened values replace ``os.environ``,
    and keys present in the file with empty or absent string values remove the variable from
    the effective mapping so environment defaults cannot leak past an explicit TOML field.

    This function never mutates ``os.environ``.

    Args:

        config_file_values: Flattened string keys from :func:`_load_config_file`.

        toml_claimed_keys: Keys the TOML document defines; ``None`` selects overlay-only behaviour.

    Returns:

        The merged mapping plus diagnostic keys whose effective value still equals the TOML-supplied value after the merge.
    """

    baseline = {str(k): str(v) for k, v in os.environ.items()}
    merged = dict(baseline)
    if toml_claimed_keys is None:
        config_effect_candidates: set[str] = set()
        for raw_key, raw_val in config_file_values.items():
            key = str(raw_key)
            value_string = str(raw_val).strip()
            if not value_string:
                continue
            baseline_value = str(baseline.get(key, "") or "").strip()
            if value_string != baseline_value:
                config_effect_candidates.add(key)
            merged[key] = value_string
        final_diag: set[str] = set()
        for key in config_effect_candidates:
            toml_value = str(config_file_values.get(key, "")).strip()
            if toml_value and merged.get(key) == toml_value:
                final_diag.add(key)
        return merged, frozenset(final_diag)

    for key in toml_claimed_keys:
        sk = str(key)
        if sk in config_file_values:
            value_string = str(config_file_values[sk]).strip()
            if value_string:
                merged[sk] = value_string
            else:
                merged.pop(sk, None)
        else:
            merged.pop(sk, None)

    config_effect_candidates = set()
    for sk in config_file_values:
        value_string = str(config_file_values[sk]).strip()
        if not value_string:
            continue
        baseline_value = str(baseline.get(sk, "") or "").strip()
        if value_string != baseline_value:
            config_effect_candidates.add(sk)
    final_diag_ssot: set[str] = set()
    for key in config_effect_candidates:
        toml_value = str(config_file_values.get(key, "")).strip()
        if toml_value and merged.get(key) == toml_value:
            final_diag_ssot.add(key)
    return merged, frozenset(final_diag_ssot)


def _engine_storage_slug_fragment(raw: str, *, fallback: str) -> str:
    """Return a filesystem-friendly lowercase token for a single slug component."""

    t = re.sub(r"[^0-9A-Za-z]+", "_", str(raw).strip()).strip("_").lower()
    return t if t else fallback


def compute_connection_storage_slug(engine: Literal["postgresql", "databricks"]) -> str:
    """
    Return a stable connection slug derived from the active engine runtime configuration.

    PostgreSQL uses host, port, database, and schema.

    Databricks uses the first hostname label (or ``pyspark`` when unset), catalog, and schema.

    When the composed slug is longer than :data:`ENGINE_STORAGE_SLUG_MAX_CHARS`, a deterministic hash suffix is used instead.
    """

    if engine == "postgresql":
        host = _engine_storage_slug_fragment(PostgresRuntimeConfig.HOST or "localhost", fallback="h")
        port = str(int(PostgresRuntimeConfig.PORT))
        db = _engine_storage_slug_fragment(PostgresRuntimeConfig.DATABASE or "db", fallback="d")
        sch = _engine_storage_slug_fragment(PostgresRuntimeConfig.SCHEMA or "public", fallback="s")
        slug = f"conn_postgresql_{host}_{port}_{db}_{sch}"
    else:
        host_raw = (DatabricksRuntimeConfig.SERVER_HOSTNAME or "").strip() or "pyspark"
        host = _engine_storage_slug_fragment(host_raw.split(".")[0], fallback="h")
        cat = _engine_storage_slug_fragment(DatabricksRuntimeConfig.CATALOG or "catalog", fallback="c")
        sch = _engine_storage_slug_fragment(DatabricksRuntimeConfig.SCHEMA or "schema", fallback="s")
        slug = f"conn_databricks_{host}_{cat}_{sch}"
    if len(slug) > int(ENGINE_STORAGE_SLUG_MAX_CHARS):
        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]
        return f"conn_{engine}_{digest}"
    return slug


def compute_engine_storage_dir(
    artifacts_root: str | None,
    engine: Literal["postgresql", "databricks"],
) -> str:
    """
    Return the absolute engine storage directory for persisted artifacts.

    When *artifacts_root* is ``None`` or blank, the parent directory is ``platformdirs.user_data_dir("aetherdialect")``.

    When *artifacts_root* is provided, the parent directory is its absolute expanded path.

    The final directory is always ``os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, connection_slug)``.
    """

    parent = (
        os.path.abspath(os.path.expanduser(str(artifacts_root)))
        if artifacts_root and str(artifacts_root).strip()
        else user_data_dir(appname="aetherdialect", appauthor=False)
    )
    slug = compute_connection_storage_slug(engine)
    return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, slug)


def _prepare_schema_context_for_init(
    schema_context: SchemaContext,
    engine_storage_dir: str,
    sink: Callable[[str], None],
) -> SchemaContext:
    """
    Merge an explicit ``SchemaContext`` with any compatible on-disk cache under *engine_storage_dir*.

    Returns:

        Possibly updated ``SchemaContext`` whose ``sql_file`` / ``notes_file`` fields may reuse cached materialised paths.
    """

    cached = load_schema_context_cache(engine_storage_dir)
    if cached is not None and (
        cached.include != schema_context.include
        or cached.allow_objects != schema_context.allow_objects
        or cached.deny_columns != schema_context.deny_columns
        or cached.allow_columns != schema_context.allow_columns
    ):
        sink("Schema scope changed since last run — caches will be rebuilt where needed.")
    notes_use = schema_context.notes_file
    sql_use = schema_context.sql_file
    if cached is not None:
        if notes_use is None and cached.notes_file:
            notes_use = cached.notes_file
            sink("  Schema context: reusing cached notes file.")
        if sql_use is None and cached.sql_file:
            sql_use = cached.sql_file
            sink("  Schema context: reusing cached SQL file.")
        cache_payload_path = os.path.join(engine_storage_dir, SCHEMA_CONTEXT_CACHE_NAME)
        try:
            with open(cache_payload_path, encoding="utf-8") as fh:
                prev_ctx = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            prev_ctx = None
        if isinstance(prev_ctx, dict):
            if schema_context.notes_file:
                old_notes = prev_ctx.get("notes_text")
                new_notes = _read_text_if_file(schema_context.notes_file)
                if isinstance(old_notes, str) and isinstance(new_notes, str) and new_notes != old_notes:
                    sink("  Schema context: notes file changed since last run.")
            if schema_context.sql_file:
                old_sql = prev_ctx.get("sql_text")
                new_sql = _read_text_if_file(schema_context.sql_file)
                if isinstance(old_sql, str) and isinstance(new_sql, str) and new_sql != old_sql:
                    sink("  Schema context: SQL file changed since last run.")
    if notes_use != schema_context.notes_file or sql_use != schema_context.sql_file:
        return SchemaContext(
            allow_objects=schema_context.allow_objects,
            include=schema_context.include,
            deny_columns=schema_context.deny_columns,
            allow_columns=schema_context.allow_columns,
            notes_file=notes_use,
            sql_file=sql_use,
        )
    return schema_context
def _env_all_non_empty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    """Return True when every key maps to a non-blank string."""

    return all(str(env.get(k, "") or "").strip() for k in keys)


def _env_first_nonempty(env: Mapping[str, str], *keys: str) -> str:
    """Return the first non-blank value among *keys*, else an empty string."""

    for k in keys:
        v = str(env.get(k, "") or "").strip()
        if v:
            return v
    return ""


def _env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    """True when at least one key maps to a non-blank string."""

    return any(str(env.get(k, "") or "").strip() for k in keys)


def _env_role_hint(label: str, keys: tuple[str, ...]) -> str:
    return f"{label}: {' or '.join(keys)}"


def _postgres_env_complete(env: Mapping[str, str]) -> bool:
    return (
        _env_any_nonempty(env, POSTGRES_ENV_DATABASE)
        and _env_any_nonempty(env, POSTGRES_ENV_USER)
        and _env_any_nonempty(env, POSTGRES_ENV_PASSWORD)
    )


def _databricks_uc_scope_complete(env: Mapping[str, str]) -> bool:
    return _env_any_nonempty(env, DATABRICKS_ENV_CATALOG) and _env_any_nonempty(env, DATABRICKS_ENV_SCHEMA)


def _databricks_sql_warehouse_complete(env: Mapping[str, str]) -> bool:
    return (
        _env_any_nonempty(env, DATABRICKS_ENV_SERVER_HOSTNAME)
        and _env_any_nonempty(env, DATABRICKS_ENV_HTTP_PATH)
        and _env_any_nonempty(env, DATABRICKS_ENV_TOKEN)
    )


def _pyspark_session_reachable() -> bool:
    if SparkSession is None:
        return False
    try:
        SparkSession.builder.getOrCreate()
    except Exception:
        return False
    return True


def _databricks_env_complete(env: Mapping[str, str]) -> bool:
    if not _databricks_uc_scope_complete(env):
        return False
    if _databricks_sql_warehouse_complete(env):
        return _package_importable("databricks.sql")
    return _pyspark_session_reachable()


def _openai_direct_env_complete(env: Mapping[str, str]) -> bool:
    return _env_any_nonempty(env, ("OPENAI_API_KEY",))


def _package_importable(name: str) -> bool:
    """Return True when *name* can be imported as a top-level module."""

    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _select_engine_name(env: Mapping[str, str]) -> Literal["postgresql", "databricks"]:
    pg_drv = _package_importable("psycopg2") or _package_importable("psycopg")
    dbx_drv = _package_importable("databricks.sql")
    pg_env = _postgres_env_complete(env)
    dbx_env = _databricks_env_complete(env)
    explicit = str(env.get("AETHERDIALECT_ENGINE", "") or "").strip().lower()
    if explicit:
        if explicit not in ("postgresql", "databricks"):
            raise ConfigError(
                f"Unsupported AETHERDIALECT_ENGINE: {explicit!r}. Expected 'postgresql' or 'databricks'.",
            )
        if explicit == "postgresql":
            if pg_drv and pg_env:
                return "postgresql"
            missing_pg: list[str] = []
            if not pg_drv:
                missing_pg.append("PostgreSQL driver (psycopg or psycopg2)")
            if pg_drv and not pg_env:
                missing_pg.append(
                    "PostgreSQL env (set one name from each required group): "
                    + _env_role_hint("database", POSTGRES_ENV_DATABASE)
                    + "; "
                    + _env_role_hint("user", POSTGRES_ENV_USER)
                    + "; "
                    + _env_role_hint("password", POSTGRES_ENV_PASSWORD)
                    + "; optional "
                    + _env_role_hint("host", POSTGRES_ENV_HOST)
                    + "; "
                    + _env_role_hint("port", POSTGRES_ENV_PORT)
                    + "; "
                    + _env_role_hint("schema", POSTGRES_ENV_SCHEMA),
                )
            raise ConfigError("Cannot select postgresql engine: " + "; ".join(missing_pg))
        if dbx_env:
            return "databricks"
        missing_dbx: list[str] = []
        if (
            not dbx_env
            and _databricks_uc_scope_complete(env)
            and _databricks_sql_warehouse_complete(env)
            and not dbx_drv
        ):
            missing_dbx.append(
                "Databricks SQL warehouse variables are set but the databricks-sql-connector package is not installed.",
            )
        elif not dbx_env:
            missing_dbx.append(
                "Databricks env: "
                + _env_role_hint("catalog", DATABRICKS_ENV_CATALOG)
                + "; "
                + _env_role_hint("schema", DATABRICKS_ENV_SCHEMA)
                + "; then either all of "
                + _env_role_hint("server hostname", DATABRICKS_ENV_SERVER_HOSTNAME)
                + ", "
                + _env_role_hint("SQL warehouse HTTP path", DATABRICKS_ENV_HTTP_PATH)
                + ", "
                + _env_role_hint("access token", DATABRICKS_ENV_TOKEN)
                + " (with databricks-sql-connector installed), or an active PySpark session.",
            )
        raise ConfigError("Cannot select databricks engine: " + "; ".join(missing_dbx))
    if pg_drv and pg_env and dbx_env:
        raise ConfigError(
            "Both PostgreSQL and Databricks are configured and available; set AETHERDIALECT_ENGINE "
            "or [engine] selected in the config file to 'postgresql' or 'databricks'.",
        )
    if pg_drv and pg_env:
        return "postgresql"
    if dbx_env:
        return "databricks"
    missing: list[str] = []
    if not pg_drv:
        missing.append("PostgreSQL driver (psycopg or psycopg2)")
    if pg_drv and not pg_env:
        missing.append(
            "PostgreSQL env (set one name from each required group): "
            + _env_role_hint("database", POSTGRES_ENV_DATABASE)
            + "; "
            + _env_role_hint("user", POSTGRES_ENV_USER)
            + "; "
            + _env_role_hint("password", POSTGRES_ENV_PASSWORD)
            + "; optional "
            + _env_role_hint("host", POSTGRES_ENV_HOST)
            + "; "
            + _env_role_hint("port", POSTGRES_ENV_PORT)
            + "; "
            + _env_role_hint("schema", POSTGRES_ENV_SCHEMA),
        )
    if (
        not dbx_env
        and _databricks_uc_scope_complete(env)
        and _databricks_sql_warehouse_complete(env)
        and not dbx_drv
    ):
        missing.append(
            "Databricks SQL warehouse variables are set but the databricks-sql-connector package is not installed.",
        )
    elif not dbx_env:
        missing.append(
            "Databricks env: "
            + _env_role_hint("catalog", DATABRICKS_ENV_CATALOG)
            + "; "
            + _env_role_hint("schema", DATABRICKS_ENV_SCHEMA)
            + "; then either all of "
            + _env_role_hint("server hostname", DATABRICKS_ENV_SERVER_HOSTNAME)
            + ", "
            + _env_role_hint("SQL warehouse HTTP path", DATABRICKS_ENV_HTTP_PATH)
            + ", "
            + _env_role_hint("access token", DATABRICKS_ENV_TOKEN)
            + " (with databricks-sql-connector installed), or an active PySpark session.",
        )
    raise ConfigError("Cannot select database engine: " + "; ".join(missing))


def _apply_postgres_env(env: Mapping[str, str]) -> None:
    """Copy PostgreSQL connection env into ``PostgresRuntimeConfig`` (``PG*`` / ``POSTGRES_*`` aliases) without toggling :class:`EngineConfig`."""

    host = _env_first_nonempty(env, *POSTGRES_ENV_HOST)
    PostgresRuntimeConfig.HOST = host or "localhost"
    port_raw = _env_first_nonempty(env, *POSTGRES_ENV_PORT)
    PostgresRuntimeConfig.PORT = int(port_raw) if port_raw else 5432
    PostgresRuntimeConfig.USER = _env_first_nonempty(env, *POSTGRES_ENV_USER)
    PostgresRuntimeConfig.PASSWORD = _env_first_nonempty(env, *POSTGRES_ENV_PASSWORD)
    PostgresRuntimeConfig.DATABASE = _env_first_nonempty(env, *POSTGRES_ENV_DATABASE)
    sch = _env_first_nonempty(env, *POSTGRES_ENV_SCHEMA) or "public"
    PostgresRuntimeConfig.SCHEMA = sch


def _apply_databricks_env(env: Mapping[str, str]) -> None:
    """Copy Databricks connection env into ``DatabricksRuntimeConfig`` (SDK / notebook aliases) without toggling :class:`EngineConfig`."""

    DatabricksRuntimeConfig.SERVER_HOSTNAME = _env_first_nonempty(env, *DATABRICKS_ENV_SERVER_HOSTNAME)
    DatabricksRuntimeConfig.HTTP_PATH = _env_first_nonempty(env, *DATABRICKS_ENV_HTTP_PATH)
    DatabricksRuntimeConfig.ACCESS_TOKEN = _env_first_nonempty(env, *DATABRICKS_ENV_TOKEN)
    DatabricksRuntimeConfig.CATALOG = _env_first_nonempty(env, *DATABRICKS_ENV_CATALOG)
    DatabricksRuntimeConfig.SCHEMA = _env_first_nonempty(env, *DATABRICKS_ENV_SCHEMA)
    DatabricksRuntimeConfig.validate()


def _activate_engine(name: Literal["postgresql", "databricks"]) -> None:
    """
    Bind :attr:`EngineConfig.TYPE` and :attr:`EngineConfig.RUNTIME` to the chosen engine.

    Pre-condition: the corresponding ``_apply_*_env`` loader has already populated the runtime config. This function performs no env reads — it is a pure switch over already-loaded credentials so callers can toggle engines mid-process without re-reading the environment.
    """

    if name == "postgresql":
        EngineConfig.TYPE = "postgresql"
        EngineConfig.RUNTIME = PostgresRuntimeConfig
        return
    if name == "databricks":
        EngineConfig.TYPE = "databricks"
        EngineConfig.RUNTIME = DatabricksRuntimeConfig
        return
    raise ConfigError(f"Unsupported engine activation: {name!r}. Expected 'postgresql' or 'databricks'.")


def _configure_openai_from_environment(env: Mapping[str, str]) -> None:
    """Populate :class:`EngineConfig` with OpenAI credentials and clear Azure fields."""

    EngineConfig.LLM_PROVIDER = "openai"
    EngineConfig.API_TOKEN = str(env["OPENAI_API_KEY"]).strip()
    EngineConfig.AZURE_API_TOKEN = None
    EngineConfig.OPENAI_MODEL = "gpt-4o-mini"
    EngineConfig.OPENAI_MODEL_INTENT = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_JOIN = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_SCHEMA = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_SCHEMA_BASE = "gpt-4.1-mini"
    EngineConfig.OPENAI_MODEL_DDL = "gpt-4.1-mini"
    bu = str(env.get("OPENAI_BASE_URL", "") or "").strip()
    EngineConfig.OPENAI_BASE_URL = bu or "https://api.openai.com/v1"


def _configure_azure_from_environment(env: Mapping[str, str]) -> None:
    """Populate :class:`EngineConfig` with Azure OpenAI credentials and clear OpenAI token."""

    EngineConfig.LLM_PROVIDER = "azure"
    EngineConfig.AZURE_API_TOKEN = str(env["AZURE_OPENAI_API_KEY"]).strip()
    EngineConfig.API_TOKEN = None
    EngineConfig.AZURE_OPENAI_ENDPOINT = str(env["AZURE_OPENAI_ENDPOINT"]).strip()
    EngineConfig.AZURE_OPENAI_API_VERSION = str(env["AZURE_OPENAI_API_VERSION"]).strip()
    base = str(env.get("AZURE_OPENAI_BASE_URL", "") or "").strip()
    EngineConfig.AZURE_OPENAI_BASE_URL = base or None
    EngineConfig.OPENAI_MODEL = "gpt-4o-mini"
    EngineConfig.OPENAI_MODEL_INTENT = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_JOIN = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_SCHEMA = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_SCHEMA_BASE = "gpt-4.1-mini"
    EngineConfig.OPENAI_MODEL_DDL = "gpt-4.1-mini"


def _configure_llm_from_environment(env: Mapping[str, str]) -> None:
    openai_ready = _openai_direct_env_complete(env)
    azure_ready = _env_all_non_empty(env, AZURE_OPENAI_ENV_REQUIRED)
    if not (openai_ready or azure_ready):
        raise ConfigError(
            "LLM is not configured. Set "
            + ", ".join(OPENAI_ENV_REQUIRED)
            + " for OpenAI, or "
            + ", ".join(AZURE_OPENAI_ENV_REQUIRED)
            + " for Azure OpenAI.",
        )
    explicit = str(env.get("AETHERDIALECT_LLM_PROVIDER", "") or "").strip().lower()
    if explicit:
        if explicit not in ("openai", "azure"):
            raise ConfigError(
                f"Unsupported AETHERDIALECT_LLM_PROVIDER: {explicit!r}. Expected 'openai' or 'azure'.",
            )
        if explicit == "openai":
            if not openai_ready:
                raise ConfigError("AETHERDIALECT_LLM_PROVIDER is 'openai' but the OpenAI environment is incomplete.")
            _configure_openai_from_environment(env)
        else:
            if not azure_ready:
                raise ConfigError("AETHERDIALECT_LLM_PROVIDER is 'azure' but the Azure OpenAI environment is incomplete.")
            _configure_azure_from_environment(env)
        clear_llm_clients()
        return
    if openai_ready and azure_ready:
        raise ConfigError(
            "Both OpenAI and Azure OpenAI credentials are available; set AETHERDIALECT_LLM_PROVIDER "
            "or [llm] provider in the config file to 'openai' or 'azure'.",
        )
    if openai_ready:
        _configure_openai_from_environment(env)
        clear_llm_clients()
        return
    if azure_ready:
        _configure_azure_from_environment(env)
        clear_llm_clients()
        return
    raise ConfigError("LLM is not configured.")


def configure_runtime_from_environment(
    schema_context: SchemaContext,
    merged_env: Mapping[str, str],
) -> Literal["postgresql", "databricks"]:
    clear_llm_clients()
    env: dict[str, str] = dict(merged_env)
    selected = _select_engine_name(env)
    if _postgres_env_complete(env):
        _apply_postgres_env(env)
    if _databricks_uc_scope_complete(env):
        _apply_databricks_env(env)
    _activate_engine(selected)
    if selected == "databricks" and not DatabricksRuntimeConfig.has_native_connection():
        if not _pyspark_session_reachable():
            raise ConfigError(
                "Databricks requires either all SQL warehouse connection variables or an active PySpark session.",
            )
    _configure_llm_from_environment(env)
    sql_path = schema_context.sql_file
    if sql_path:
        expanded = os.path.expanduser(str(sql_path))
        if selected == "postgresql":
            PostgresRuntimeConfig.SQL_FILE_PATH = expanded
        else:
            DatabricksRuntimeConfig.SQL_FILE_PATH = expanded
    if selected not in SUPPORTED_ENGINES:
        raise ConfigError(f"Unsupported engine resolved: {selected!r}")
    return selected


@dataclass
class Text2SQLInitResult:
    """Mutable template bundle and graph produced by :func:`initialize_text2sql`."""

    runtime_config: RuntimeConfig
    llm_config: LLMConfig
    schema_graph: SchemaGraph
    dialect: Any
    artifacts_dir: str
    store: TemplateStoreView
    templates: dict[str, Any]
    rejected: dict[str, Any]
    schema_terms: set[str]
    schema_stats: dict[str, Any]


def _read_text_if_file(path: str | None) -> str | None:
    """Return the text content of *path* if it exists and is a regular file, else None."""

    if not path:
        return None
    expanded = os.path.expanduser(str(path))
    if not os.path.isfile(expanded):
        return None
    with open(expanded, encoding="utf-8") as fh:
        return fh.read()


def write_schema_context_cache(artifacts_dir: str, schema_context: SchemaContext) -> str:
    """
    Persist *schema_context* (with sql_file/notes_file text inlined) to *artifacts_dir*.

    Returns the path of the written cache file.
    """

    payload: dict[str, Any] = {
        "version": SCHEMA_CONTEXT_CACHE_VERSION,
        "include": schema_context.include,
        "allow_objects": sorted(schema_context.allow_objects),
        "deny_columns": sorted(schema_context.deny_columns),
        "allow_columns": sorted(schema_context.allow_columns),
        "sql_file_original": schema_context.sql_file,
        "notes_file_original": schema_context.notes_file,
        "sql_text": _read_text_if_file(schema_context.sql_file),
        "notes_text": _read_text_if_file(schema_context.notes_file),
    }
    os.makedirs(artifacts_dir, exist_ok=True)
    cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return cache_path


def load_schema_context_cache(artifacts_dir: str) -> SchemaContext | None:
    """
    Reload a persisted ``SchemaContext`` from *artifacts_dir*.

    Returns ``None`` when no cache exists, the file is unreadable, or its version is unsupported. Inlined ``sql_text`` / ``notes_text`` are materialised back to disk inside *artifacts_dir* so downstream consumers that expect file paths continue to work.
    """

    cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != SCHEMA_CONTEXT_CACHE_VERSION:
        return None
    sql_text = payload.get("sql_text")
    notes_text = payload.get("notes_text")
    sql_file: str | None = None
    notes_file: str | None = None
    if isinstance(sql_text, str):
        sql_file = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHED_DDL)
        with open(sql_file, "w", encoding="utf-8") as fh:
            fh.write(sql_text)
    if isinstance(notes_text, str):
        notes_file = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHED_NOTES)
        with open(notes_file, "w", encoding="utf-8") as fh:
            fh.write(notes_text)
    include_raw = payload.get("include", "tables")
    if include_raw not in ("tables", "views", "both"):
        include_raw = "tables"
    return SchemaContext(
        allow_objects=frozenset(payload.get("allow_objects") or ()),
        include=include_raw,
        deny_columns=frozenset(payload.get("deny_columns") or ()),
        allow_columns=frozenset(payload.get("allow_columns") or ()),
        sql_file=sql_file,
        notes_file=notes_file,
    )


def _purge_schema_context_cache(artifacts_dir: str) -> None:
    """
    Remove the persisted ``schema_context.json`` and any materialised cache files.

    Used during legacy-artifact cleanup so a stale schema context cannot be silently reloaded after a learning-reset rebuild.
    """

    for name in (
        SCHEMA_CONTEXT_CACHE_NAME,
        SCHEMA_CONTEXT_CACHED_DDL,
        SCHEMA_CONTEXT_CACHED_NOTES,
    ):
        fp = os.path.join(artifacts_dir, name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError as exc:
                debug(f"[main_execution._purge_schema_context_cache] {fp}: {exc}")


def _notify_schema_context_warnings(schema_context: SchemaContext, sink: Callable[[str], None]) -> None:
    """Emit non-fatal notices for ambiguous or ineffective scope entries."""

    overlap = schema_context.allow_columns & schema_context.deny_columns
    if overlap:
        n = len(overlap)
        sink(
            f"  Schema scope: allow_columns ∩ deny_columns has {n} duplicate "
            f"entr{'ies' if n != 1 else 'y'}; deny_columns wins for those keys."
        )
    allow = schema_context.allow_objects
    for spec in sorted(schema_context.deny_columns):
        if "." not in spec:
            continue
        tbl, _, _rest = spec.partition(".")
        if tbl == "*":
            continue
        if allow and tbl not in allow:
            sink(
                f"  Schema scope: deny_columns entry {spec!r} references table {tbl!r} "
                "outside allow_objects; it never applies under the current scope."
            )


def _emit_runtime_config_override_diagnostics(overridden: frozenset[str]) -> None:
    """Emit one diagnostic per runtime-config field whose effective value came from the TOML file over env."""

    for key in sorted(overridden):
        _core_utils.notify(
            f"Runtime config file overrides environment for {key}",
            stage="config",
            code=DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
            details=(("key", key),),
        )


def initialize_text2sql(
    schema_context: SchemaContext | None = None,
    *,
    artifacts_dir: str | None = None,
    config_file: str | os.PathLike[str] | None = None,
    log_sink: Callable[[str], None] | None = None,
    execution_engine: Any | None = None,
) -> Text2SQLInitResult:
    """Configure the process environment, build the schema graph, migrate templates, and load stores."""

    sink: Callable[[str], None] = log_sink if log_sink is not None else _core_utils.notify
    sink("Initialising Text2SQL.")
    config_file_values, toml_claimed_keys = _load_config_file(config_file)
    ssot = config_file is not None and bool(str(config_file).strip())
    merged, toml_diagnostic_keys = _merge_configuration_environment(
        config_file_values,
        toml_claimed_keys=toml_claimed_keys if ssot else None,
    )
    selected_preview = _select_engine_name(merged)
    if _postgres_env_complete(merged):
        _apply_postgres_env(merged)
    if _databricks_uc_scope_complete(merged):
        _apply_databricks_env(merged)
    adir = compute_engine_storage_dir(artifacts_dir, selected_preview)
    if schema_context is None:
        cached = load_schema_context_cache(adir)
        if cached is None:
            raise ConfigError(
                "schema_context is required on first initialisation. No cached "
                f"schema_context.json was found in {adir!r}. Pass an explicit "
                "SchemaContext (use SchemaContext() to scope to the whole database).",
            )
        schema_context = cached
    else:
        schema_context = _prepare_schema_context_for_init(schema_context, adir, sink)
    _notify_schema_context_warnings(schema_context, sink)
    active_engine = configure_runtime_from_environment(schema_context, merged)
    try:
        llm_exec = load_runtime_config(merged_env=merged)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _emit_runtime_config_override_diagnostics(toml_diagnostic_keys)
    if EngineConfig.LLM_PROVIDER == "azure":
        missing = [
            n
            for n, v in (
                ("azure_endpoint", llm_exec.azure_endpoint),
                ("azure_api_key", llm_exec.azure_api_key),
                ("azure_api_version", llm_exec.azure_api_version),
                ("deployment_light", llm_exec.deployment_light),
                ("deployment_medium", llm_exec.deployment_medium),
                ("deployment_heavy", llm_exec.deployment_heavy),
            )
            if not (isinstance(v, str) and v.strip())
        ]
        if missing:
            raise ConfigError(
                "Azure OpenAI requires non-empty runtime configuration for: " + ", ".join(missing),
            )
    _rt = EngineConfig.RUNTIME
    _rt_name = (getattr(_rt, "__name__", None) or str(_rt) or "default").lower()
    if _rt_name.endswith("runtimeconfig"):
        _rt_name = _rt_name[: -len("runtimeconfig")]
    runtime_label = _rt_name or "default"
    sink(f"  Engine: {active_engine} ({runtime_label}).")
    os.makedirs(adir, exist_ok=True)
    legacy_files = detect_legacy_artifacts(adir)
    if legacy_files:
        sink(
            f"  Detected legacy artifacts (no manifest): {', '.join(legacy_files)}. Rebuilding caches.",
        )
        wipe_versioned_artifacts(adir)
        _purge_schema_context_cache(adir)
    EngineConfig.SCHEMA_JSON_PATH = os.path.join(adir, "schema_graph.json.gz")
    EngineConfig.TEMPLATE_STORE_DIR = os.path.join(adir, TEMPLATE_STORE_SEGMENT)
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(adir, "qsim_skeletons.json.gz")
    try:
        dialect = get_dialect(EngineConfig.TYPE, EngineConfig.RUNTIME, sqlalchemy_engine=execution_engine)
    except ConnectionError:
        raise
    except Exception as exc:
        raise ConnectionError(str(exc)) from exc
    notes_content: str | None = None
    if schema_context.notes_file:
        nf_path = os.path.expanduser(str(schema_context.notes_file))
        if os.path.isfile(nf_path):
            with open(nf_path, encoding="utf-8") as nf:
                notes_content = nf.read()
    previous_schema = load_schema_graph_snapshot(EngineConfig.SCHEMA_JSON_PATH)
    cwd_root = Path.cwd().resolve()
    map_path = cwd_root / MIGRATION_MAP_FILENAME
    pending_migration_map = _load_schema_migration_map(cwd_root) if map_path.is_file() else None
    schema_graph, schema_diff = build_schema_graph_with_diff(
        dialect,
        schema_context,
        notes_content=notes_content,
        log_sink=sink,
        refresh_existing_descriptions_on_addition=(
            pending_migration_map.refresh_existing_descriptions_on_addition
            if pending_migration_map is not None
            else False
        ),
    )
    stored = read_artifact_manifest(adir)
    if map_path.is_file():
        loaded = pending_migration_map if pending_migration_map is not None else _load_schema_migration_map(cwd_root)
        if loaded is not None:
            try:
                _validate_schema_migration_map(loaded, previous_schema, schema_graph)
            except MigrationPendingError as exc:
                msg = str(exc)
                if msg.startswith("STALE_MAP:"):
                    try:
                        map_path.unlink()
                    except OSError:
                        pass
                    sink("  Removed stale schema_migration_map.json for this snapshot.")
                else:
                    raise
            else:
                if loaded.action == MIGRATION_MAP_ACTION_ABORT:
                    try:
                        map_path.unlink()
                    except OSError:
                        pass
                    raise MigrationPendingError("user aborted via migration map")
                _apply_schema_migration_map(loaded, adir, schema_graph, Path(EngineConfig.SCHEMA_JSON_PATH))
                ts = datetime.now(timezone.utc).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
                applied_map = map_path.with_name(map_path.stem + ".applied.json")
                try:
                    if applied_map.is_file():
                        archive = applied_map.with_name(applied_map.stem + f".{ts}" + applied_map.suffix)
                        applied_map.rename(archive)
                    map_path.rename(applied_map)
                except OSError as exc:
                    debug(f"[main_execution.initialize_text2sql] could not archive migration map: {exc}")
                previous_schema = load_schema_graph_snapshot(EngineConfig.SCHEMA_JSON_PATH)
                pending_migration_map = None
                schema_graph, schema_diff = build_schema_graph_with_diff(
                    dialect,
                    schema_context,
                    notes_content=notes_content,
                    log_sink=sink,
                    refresh_existing_descriptions_on_addition=False,
                )
                stored = read_artifact_manifest(adir)
    tier_preview = classify_migration_tier(
        stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff
    )
    if tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
        rename_plan = try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
        skel_path = export_schema_migration_map_skeleton(
            cwd_root,
            tier=tier_preview,
            schema_diff=schema_diff,
            rename_plan=rename_plan,
        )
        raise MigrationPendingError(f"Schema migration required: edit {skel_path} and restart init.")
    migration_report = apply_migration_policy(
        adir,
        schema_graph,
        allow_destructive=True,
        previous_schema=previous_schema,
        schema_diff=schema_diff,
    )
    if migration_report.tier != MigrationTier.NO_CHANGE:
        _print_migration_applied(migration_report, sink)
    store = load_template_store(schema_graph.effective_structural_hash, schema_graph)
    templates = store_to_templates(store)
    rejected = {}
    sink(
        f"  Templates: {len(templates)} reusable, {len(rejected)} rejected.",
    )
    schema_terms: set[str] = set(schema_graph.tables.keys())
    for tinfo in schema_graph.tables.values():
        schema_terms.update(tinfo.columns)
        for col in tinfo.columns:
            schema_terms.add(col.lower())
    schema_stats = schema_graph.schema_stats or {}
    prov: Literal["openai", "azure"] = "azure" if EngineConfig.LLM_PROVIDER == "azure" else "openai"
    llm_config = LLMConfig(provider=prov)
    runtime_config = RuntimeConfig(
        engine=active_engine,
        artifacts_dir=adir,
        schema_context=schema_context,
        llm_execution=llm_exec,
    )
    try:
        write_schema_context_cache(adir, schema_context)
    except OSError as exc:
        debug(f"[main_execution.initialize_text2sql] schema_context cache write failed: {exc}")
    sink("Ready.")
    return Text2SQLInitResult(
        runtime_config=runtime_config,
        llm_config=llm_config,
        schema_graph=schema_graph,
        dialect=dialect,
        artifacts_dir=adir,
        store=store,
        templates=templates,
        rejected=rejected,
        schema_terms=schema_terms,
        schema_stats=schema_stats,
    )


def apply_confirmed_destructive_migration(
    artifacts_dir: str,
    schema_graph: SchemaGraph,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run deferred learning-reset migration and return refreshed template store maps."""

    destructive_migration_execute(artifacts_dir, schema_graph)
    store = load_template_store(schema_graph.effective_structural_hash, schema_graph)
    return store, store_to_templates(store), {}


def clear_template_store_only(artifacts_dir: str, schema_graph: SchemaGraph) -> bool:
    """Remove the partitioned template store directory and legacy monolithic file when present."""

    assert isinstance(schema_graph, SchemaGraph)
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    legacy = os.path.join(artifacts_dir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)
    existed = os.path.isdir(store_dir) or os.path.isfile(legacy)
    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)
    _wipe_filenames(artifacts_dir, (TEMPLATE_STORE_LEGACY_SINGLE_FILE,))
    return existed


def clear_simulation_caches_only(artifacts_dir: str) -> int:
    """Remove QSim and seed-warmup simulation artifacts; return count of files removed."""

    count = _wipe_filenames(artifacts_dir, SIMULATION_CACHE_EXACT_FILENAMES)
    count += _wipe_globs(artifacts_dir, SIMULATION_CACHE_GLOB_PATTERNS)
    return count


def resolve_qsim_path(version_or_result: int | QSimSummary, artifacts_dir: str) -> str:
    """
    Resolve the full file path for a QSim questions text artifact.

    Args:

        version_or_result: Integer run version or a `QSimSummary` instance.

        artifacts_dir: Directory where QSim output files are stored.

    Returns:

        Absolute path to `qsim_questions_v{version}.txt`.
    """
    if isinstance(version_or_result, QSimSummary):
        ver = version_or_result.version
    else:
        ver = int(version_or_result)
    return os.path.join(artifacts_dir, QSIM_QUESTIONS_PATTERN.format(version=ver))


def load_qsim_summaries(artifacts_dir: str) -> list[QSimSummary]:
    """Load every ``QSimSummary`` from ``qsim_summary.json`` under *artifacts_dir*, oldest first."""

    qsim_summary_path = os.path.join(artifacts_dir, "qsim_summary.json")
    if not os.path.exists(qsim_summary_path):
        return []
    with open(qsim_summary_path, encoding="utf-8") as f:
        summaries = json.load(f)
    if not isinstance(summaries, list):
        return []
    return [QSimSummary.from_dict(s) for s in summaries if isinstance(s, dict)]


def echo_choice_line_after_input(label: str, normalized: str | None, *, silent_no: bool = False) -> None:
    """Echo a yes/no answer on the same terminal line as *label* when possible."""

    if normalized is None:
        return
    suffix = "Yes" if normalized == "y" else "No"
    block = f"{label}{suffix}\n"
    if sys.stdout.isatty():
        sys.stdout.write(f"\033[1A\033[K{block}")
    else:
        sys.stdout.write(block)
    sys.stdout.flush()
    if normalized == "n" and not silent_no:
        _core_utils.terminated()


def normalise_yes_no(raw: str, options: list[str]) -> str | None:
    """Map free text to ``y`` or ``n`` when present in *options*."""

    token = raw.strip().lower()
    if token in ("y", "yes") and "y" in options:
        return "y"
    if token in ("n", "no") and "n" in options:
        return "n"
    return None


def find_latest_seed_warmup_summary(artifacts_dir: str) -> SeedWarmupSummary | None:
    """Return the newest ``SeedWarmupSummary`` under *artifacts_dir*, or ``None`` when absent."""

    if not os.path.isdir(artifacts_dir):
        return None
    best_ver = -1
    for name in os.listdir(artifacts_dir):
        if not name.startswith("seed_warmup_report_v") or not name.endswith(".json"):
            continue
        mid = name[len("seed_warmup_report_v") : -len(".json")]
        if not mid.isdigit():
            continue
        best_ver = max(best_ver, int(mid))
    if best_ver < 0:
        return None
    return get_seed_warmup_summary_from_dir(artifacts_dir, best_ver)


_SESSION_USER_FEEDBACK_BODY = (
    "What was wrong?\n"
    "Tip: a single sentence is enough — for example 'wrong table', "
    "'missing date filter', or 'should aggregate by month'."
)
_SESSION_INTENT_FEEDBACK_BODY = (
    "What should change about this interpretation?\n"
    "Tip: a single sentence is enough — for example 'wrong table', "
    "'missing date filter', or 'should aggregate by month'."
)


class PipelineSession(InteractiveChoicePort):
    """
    Programmatic driver for one interactive turn at a time via ask and step.

    When used as the interactive choice port, the internal pipeline calls :meth:`has_pending_choice` and :meth:`take_yes_no`. :meth:`note_turn_outcome` records the latest turn for :meth:`step` consumers. Builtin ``dir`` on this class lists only ask, ask_until_done, awaiting_prompt, reset, and step.
    """

    __slots__ = (
        "_owner",
        "_choice_queue",
        "_suspended",
        "_resume_choice_stage_id",
        "_last_turn_outcome",
        "_session_busy",
        "_refinement_ctx",
        "_session_mode",
        "_turn_question",
        "_pending_conversation_rejection_hints",
    )

    def __init__(self, owner: Any, *, mode: Literal["reader", "writer"] = "writer") -> None:
        self._owner = owner
        self._session_mode = mode
        self._choice_queue: deque[tuple[str, str]] = deque()
        self._suspended: PipelineSuspended | None = None
        self._resume_choice_stage_id: str | None = None
        self._last_turn_outcome: dict[str, Any] | None = None
        self._session_busy = False
        self._refinement_ctx: RefinementContext | None = None
        self._turn_question: str | None = None
        self._pending_conversation_rejection_hints: tuple[str, ...] = ()

    def _audit_ask_emit(
        self,
        event_type: str,
        *,
        question: str | None = None,
        details: tuple[tuple[str, str], ...] = (),
    ) -> None:
        fn = getattr(self._owner, "_audit_emit", None)
        if not callable(fn):
            return
        owner_schema = getattr(self._owner, "_schema_graph", None)
        schema_hash_val: str | None = None
        if owner_schema is not None:
            schema_hash_val = getattr(owner_schema, "effective_structural_hash", None)
        fn(
            event_type,
            question=question,
            schema_hash=schema_hash_val,
            details=details,
        )

    def _mk_step(self, *, diagnostics: tuple[Diagnostic, ...] = (), **kw: Any) -> SessionStep:
        """Build a :class:`SessionStep`, attaching drained pipeline diagnostics."""

        drained = drain_diagnostic_collector()
        merged = diagnostics + drained
        merged_kw = dict(kw)
        merged_kw["diagnostics"] = merged
        return SessionStep(**merged_kw)

    def _attach_refinement_ctx(self, ctx: RefinementContext | None) -> None:
        """Bind :class:`RefinementContext` for silent in-turn retries after user rejection."""

        self._refinement_ctx = ctx

    def _continue_after_refinement_retry(self) -> SessionStep:
        """Run additional intent passes until completion, another suspend, or terminal failure."""

        ctx = self._refinement_ctx
        if ctx is None:
            self._reset_after_turn()
            self._session_busy = False
            st = self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error="Refinement context missing.",
            )
            return replace(st, status=_failure_category_for_terminal_step(st))
        dialect = self._owner._dialect
        schema, store, templates, rejected, schema_terms = self._resources()
        corrected = ctx.corrected_question
        q_norm = normalize_question(corrected)
        while True:
            try:
                with _core_utils.llm_execution_scope(self._owner._runtime_config.llm_execution):
                    ok = _interactive_run_intent_pass(
                        corrected_text=corrected,
                        q_norm=q_norm,
                        dialect=dialect,
                        schema=schema,
                        store=store,
                        templates=templates,
                        rejected=rejected,
                        schema_terms=schema_terms,
                        choice_port=self,
                        form_storage=ctx.form_storage,
                        refinement_ctx=ctx,
                        persist_template_learning=_persist_template_learning_for_pipeline_session(self),
                    )
                    if not ok:
                        self._reset_after_turn()
                        self._session_busy = False
                        st = self._mk_step(
                            done=True,
                            prompt=None,
                            kind=SESSION_KIND_ERROR,
                            error="Intent parse failed.",
                        )
                        return replace(st, status=_failure_category_for_terminal_step(st))
                    return self._completed_step()
            except RefinementRetry:
                continue
            except PipelineSuspended as ex2:
                if ex2.state_id in ("empty_choice_queue", "choice_queue_mismatch"):
                    self.reset()
                    return self._terminal_error_step(ex2.message_for_caller)
                self._suspended = ex2
                return self._suspend_to_step(ex2)

    def _resources(
        self,
    ) -> tuple[SchemaGraph, dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
        """Return the schema graph and template backing structures from the owning facade."""

        owner = self._owner
        return (
            owner._schema_graph,
            owner._store,
            owner._templates,
            owner._rejected,
            owner._schema_terms,
        )

    def __dir__(self) -> list[str]:
        """Return names intended for interactive discovery."""

        return sorted(("ask", "ask_until_done", "awaiting_prompt", "reset", "step"))

    def __enter__(self) -> PipelineSession:
        """Return *self* for ``with`` blocks."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        """Reset partial turn state when leaving a ``with`` block."""

        self.reset()
        return False

    def _reset_after_turn(self) -> None:
        """Clear partial turn state after a completed or abandoned interactive pass."""

        self._choice_queue.clear()
        self._suspended = None
        self._resume_choice_stage_id = None
        self._last_turn_outcome = None
        self._refinement_ctx = None
        self._turn_question = None
        self._pending_conversation_rejection_hints = ()

    def reset(self) -> None:
        """Clear suspend state, queued programmatic answers, and partial turn state."""

        self._reset_after_turn()
        self._session_busy = False

    def note_turn_outcome(
        self,
        *,
        outcome: str,
        error: str | None = None,
        sql: str | None = None,
        rows: list[tuple[Any, ...]] | None = None,
        columns: tuple[str, ...] | None = None,
        rejection_bucket: str | None = None,
        intent: RuntimeIntent | None = None,
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
        }

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""

        return len(self._choice_queue) > 0

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Pop and normalise the next queued token against *options*."""

        if not self._choice_queue:
            raise PipelineSuspended(
                "empty_choice_queue",
                "interactive choice queue is empty",
                None,
            )
        sid_token, raw = self._choice_queue.popleft()
        expected = self._resume_choice_stage_id
        if expected is not None and sid_token != expected:
            raise PipelineSuspended(
                "choice_queue_mismatch",
                f"queued answer targeted {sid_token!r} but expected {expected!r}",
                None,
            )
        return normalise_yes_no(raw, options)

    def _consume_next_queued_choice(self) -> str | None:
        """Remove one raw queued token for resume paths that do not use :meth:`take_yes_no`."""

        if not self._choice_queue:
            return None
        sid_token, raw = self._choice_queue.popleft()
        expected = self._resume_choice_stage_id
        if expected is not None and sid_token != expected:
            raise PipelineSuspended(
                "choice_queue_mismatch",
                f"queued answer targeted {sid_token!r} but expected {expected!r}",
                None,
            )
        return raw

    def awaiting_prompt(self) -> bool:
        """Return True when the session is waiting on :meth:`step` input."""

        return self._suspended is not None

    def ask(self, question: str) -> SessionStep:
        """Start a new NL turn and return the first :class:`SessionStep` (prompt, result, or error)."""

        if not isinstance(question, str):
            self._audit_ask_emit(
                "ask_blocked",
                details=(("reason", "question_not_str"),),
            )
            raise TypeError("question must be str")
        if self._session_busy:
            self._audit_ask_emit(
                "ask_blocked",
                question=question,
                details=(("reason", "session_active"),),
            )
            raise SessionActiveError("Cannot start a new question while a turn is in progress.")
        buf = diagnostic_segment()
        for _orph in take_and_clear_orphan_diagnostics():
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        self._pending_conversation_rejection_hints = ()
        try:
            return self._drive_question_turn(question)
        finally:
            reset_diagnostic_collector(tok)

    def ask_until_done(self, question: str, *, on_confirm: Literal["y", "n"] = "y") -> SessionStep:
        """Run ``ask`` then auto-answer yes or no suspends with *on_confirm* until the turn ends.

        When the user declines executed SQL on the final yes or no prompt, the terminal :class:`SessionStep` carries ``status`` ``FailureCategory.RESULT_OKAY_INTENT_WRONG`` so programmatic callers can distinguish validated-but-rejected runs from unconditional success.
        """

        if not isinstance(question, str):
            raise TypeError("question must be str")
        step = self.ask(question)
        while not step.done:
            if step.reply_shape != "yes_no":
                raise SessionActiveError(f"free-text suspend at kind={step.kind}; ask_until_done cannot answer")
            step = self.step(on_confirm)
        return step

    def step(self, response: str | None = None) -> SessionStep:
        """Supply the next user answer for a suspended prompt."""

        buf = diagnostic_segment()
        for _orph in take_and_clear_orphan_diagnostics():
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        try:
            if self._suspended is not None:
                return self._step_pipeline_suspend(response or "")
            if not self._session_busy:
                return self._mk_step(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_IDLE,
                    error="No active turn; call ask() first.",
                )
            return self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error="No suspended prompt to answer.",
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
                    kind=SUSPEND_ID_TO_SESSION_KIND.get(
                        PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
                        SESSION_KIND_ERROR,
                    ),
                    message=(
                        "Reject reason cannot be empty.\n\n"
                        + _SESSION_USER_FEEDBACK_BODY
                    ),
                )
            self._choice_queue.append((PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT, text))
            return self._resume_from_suspend()
        if self._suspended is not None and self._suspended.state_id == PIPELINE_SUSPEND_ID_INTENT_FEEDBACK:
            text = (raw or "").strip()
            if not text:
                return self._mk_step(
                    done=False,
                    prompt=SESSION_PROMPT_REASON,
                    kind=SUSPEND_ID_TO_SESSION_KIND.get(
                        PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
                        SESSION_KIND_ERROR,
                    ),
                    message=(
                        "Feedback text cannot be empty.\n\n"
                        + _SESSION_INTENT_FEEDBACK_BODY
                    ),
                )
            self._choice_queue.append((PIPELINE_SUSPEND_ID_INTENT_FEEDBACK, text))
            return self._resume_from_suspend()
        normalised = normalise_yes_no(raw, ["y", "n"])
        if normalised is None:
            kind = SUSPEND_ID_TO_SESSION_KIND.get(self._suspended.state_id, SESSION_KIND_ERROR)
            return self._mk_step(
                done=False,
                prompt=SESSION_PROMPT_YESNO,
                kind=kind,
                message="Invalid choice — please answer y or n.",
                reply_shape="yes_no",
            )
        sid = self._suspended.state_id
        self._choice_queue.append((sid, normalised))
        return self._resume_from_suspend()

    def _suspend_to_step(self, ex: PipelineSuspended) -> SessionStep:
        """Build a :class:`SessionStep` describing a deferred pipeline prompt."""

        kind = SUSPEND_ID_TO_SESSION_KIND.get(ex.state_id, SESSION_KIND_ERROR)
        payload = ex.payload
        sql_out: str | None = None
        data_out: pandas.DataFrame | None = None
        body: str | None = None
        prompt_out = SESSION_PROMPT_YESNO
        isum: IntentSummary | None = None
        reply_shape: Literal["yes_no", "free_text"] | None = "yes_no"
        sem_w: tuple[str, ...] = ()

        if ex.state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM and isinstance(payload, InteractiveTailSnapshot):
            body, sem_w = compose_intent_confirm_session_message(payload.intent, list(payload.semantic_warnings))
            prompt_out = SESSION_PROMPT_YESNO
            isum = _build_intent_summary(payload.intent)
            reply_shape = "yes_no"
        elif ex.state_id == PIPELINE_SUSPEND_ID_SQL and isinstance(payload, SqlFeedbackSuspendContext):
            ctxp = payload
            body = ""
            sql_out = ctxp.sql
            full_df = build_result_dataframe(
                list(ctxp.rows),
                ctxp.execution_intent,
                ctxp.sql,
                structural_defaults=ctxp.tmpl_sd,
                q_norm=ctxp.tail.q_norm,
                template_display_alias_map=(
                    getattr(ctxp.gen_out.matched_template, "display_alias_map", None)
                    if ctxp.gen_out.matched_template
                    else None
                ),
            )
            if full_df is not None and not full_df.empty:
                data_out = full_df.head(5)
            prompt_out = SESSION_PROMPT_YESNO
            isum = _build_intent_summary(ctxp.execution_intent)
            reply_shape = "yes_no"
            sem_w = ()
        elif ex.state_id == PIPELINE_SUSPEND_ID_DIRECT_REUSE and isinstance(payload, DirectReuseSuspendContext):
            ctx = payload
            sql_out = ctx.display_sql
            rows_list = list(ctx.rows)
            hdr = list(ctx.headers) if ctx.headers else None
            if rows_list:
                if hdr and len(hdr) == len(rows_list[0]):
                    data_out = pandas.DataFrame([list(r) for r in rows_list], columns=hdr).head(5)
                else:
                    data_out = pandas.DataFrame([list(r) for r in rows_list]).head(5)
            body = ""
            prompt_out = SESSION_PROMPT_YESNO
            isum = _build_intent_summary(ctx.intent)
            reply_shape = "yes_no"
            sem_w = ()
        elif ex.state_id == PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT:
            body = _SESSION_USER_FEEDBACK_BODY
            prompt_out = SESSION_PROMPT_REASON
            reply_shape = "free_text"
            sem_w = ()
        elif ex.state_id == PIPELINE_SUSPEND_ID_INTENT_FEEDBACK:
            body = (ex.message_for_caller or "").strip() or _SESSION_INTENT_FEEDBACK_BODY
            prompt_out = SESSION_PROMPT_REASON
            reply_shape = "free_text"
            sem_w = ()
        else:
            body = ex.message_for_caller
            prompt_out = SESSION_PROMPT_YESNO
            reply_shape = "yes_no"
            sem_w = ()

        return self._mk_step(
            done=False,
            prompt=prompt_out,
            kind=kind,
            sql=sql_out,
            data=data_out,
            message=body,
            intent_summary=isum,
            reply_shape=reply_shape,
            semantic_warnings=sem_w,
        )

    def _completed_step(self) -> SessionStep:
        """Build a terminal :class:`SessionStep` after a full successful pipeline pass."""

        snap = self._last_turn_outcome or {}
        qtxt = self._turn_question or ""
        rows_raw = snap.get("rows")
        rows_tuple: tuple[tuple[Any, ...], ...] | None = None
        if isinstance(rows_raw, list):
            rows_tuple = tuple(tuple(r) for r in rows_raw)
        sql_val = snap.get("sql")
        sql_out = str(sql_val) if sql_val is not None else None
        cols_raw = snap.get("columns")
        cols_tuple: tuple[str, ...] | None = None
        if isinstance(cols_raw, list) and cols_raw and all(isinstance(x, str) for x in cols_raw):
            cols_tuple = tuple(str(x) for x in cols_raw)
        elif rows_tuple:
            cols_tuple = _result_columns_for_session(sql_out, list(rows_tuple))
        data_out: pandas.DataFrame | None = None
        if rows_tuple:
            cols_use = list(cols_tuple) if cols_tuple else _result_columns_for_session(sql_out, list(rows_tuple))
            if cols_use:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple], columns=list(cols_use))
            else:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple])
        raw_outcome = str(snap.get("outcome") or "success")
        if raw_outcome == "success":
            terminal_message = SAVED_LINE
            terminal_status: FailureCategory | None = None
        elif raw_outcome == "user_declined":
            bucket_key = str(snap.get("rejection_bucket") or "OTHER").strip().upper()
            tip = USER_REJECTED_RESULT_BUCKET_TIPS.get(bucket_key, USER_REJECTED_RESULT_BUCKET_TIPS["OTHER"])
            terminal_message = f"{FEEDBACK_NOTED_LINE}\n{tip}"
            terminal_status = None
        elif raw_outcome == "intent_rejected":
            bucket_key = str(snap.get("rejection_bucket") or "OTHER").strip().upper()
            tip = USER_REJECTED_RESULT_BUCKET_TIPS.get(bucket_key, USER_REJECTED_RESULT_BUCKET_TIPS["OTHER"])
            terminal_message = f"{FEEDBACK_NOTED_LINE}\n{tip}"
            terminal_status = FailureCategory.RESULT_OKAY_INTENT_WRONG.value
        else:
            terminal_message = str(snap.get("error") or raw_outcome)
            terminal_status = None
        err_snap = snap.get("error")
        ri = snap.get("intent")
        isum_res: IntentSummary | None = None
        if isinstance(ri, RuntimeIntent):
            isum_res = _build_intent_summary(ri)
        step = self._mk_step(
            done=True,
            prompt=None,
            kind=SESSION_KIND_RESULT,
            sql=sql_out,
            data=data_out,
            message=terminal_message,
            error=err_snap,
            intent_summary=isum_res,
            reply_shape=None,
            semantic_warnings=(),
            status=terminal_status,
        )
        self._audit_ask_emit(
            "ask_done",
            question=qtxt,
            details=(("outcome", raw_outcome), ("kind", step.kind)),
        )
        self._reset_after_turn()
        self._session_busy = False
        return step

    def _terminal_error_step(self, message: str) -> SessionStep:
        """Build a terminal error :class:`SessionStep`."""

        self._audit_ask_emit(
            "ask_error",
            question=self._turn_question,
            details=(("message", message),),
        )
        self.reset()
        st = self._mk_step(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            error=message,
        )
        return replace(st, status=_failure_category_for_terminal_step(st))

    def _drive_question_turn(self, raw: str) -> SessionStep:
        """Run :func:`interactive_run_once` until suspend or completion."""

        self._reset_after_turn()
        self._turn_question = raw
        self._session_busy = True
        self._audit_ask_emit("ask_begin", question=raw, details=())
        art = getattr(self._owner, "_artifacts_dir", None)
        adir = ""
        if art is not None:
            try:
                adir = os.path.abspath(os.fspath(art))
            except (TypeError, OSError, ValueError):
                adir = ""
        if adir and self._session_mode == "reader":
            _reload_reader_learning_if_manifest_drift(self._owner)
        schema, store, templates, rejected, schema_terms = self._resources()

        def _run_turn() -> SessionStep:
            with _core_utils.llm_execution_scope(self._owner._runtime_config.llm_execution):
                try:
                    interactive_run_once(
                        schema,
                        store,
                        templates,
                        rejected,
                        schema_terms,
                        question=raw,
                        pipeline_session=self,
                    )
                except PipelineSuspended as ex:
                    if ex.state_id in ("empty_choice_queue", "choice_queue_mismatch"):
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
                            error=ex.message_for_caller,
                        )
                        return replace(st_e, status=_failure_category_for_terminal_step(st_e))
                    self._suspended = ex
                    return self._suspend_to_step(ex)
                except Exception as exc:
                    debug(f"[main_execution.PipelineSession._drive_question_turn] unexpected error: {exc!r}")
                    self._audit_ask_emit(
                        "ask_error",
                        question=self._turn_question,
                        details=(("message", str(exc)),),
                    )
                    self._reset_after_turn()
                    self._session_busy = False
                    st_x = self._mk_step(
                        done=True,
                        prompt=None,
                        kind=SESSION_KIND_ERROR,
                        error=str(exc),
                    )
                    return replace(st_x, status=_failure_category_for_terminal_step(st_x))
                return self._completed_step()

        lock = getattr(self._owner, "_pipeline_writer_lock", None)
        if lock is not None and self._session_mode == "writer":
            with lock:
                if adir:
                    drain_write_queue(self._owner, adir)
                return _run_turn()
        return _run_turn()

    def _resume_from_suspend(self) -> SessionStep:
        """Continue execution after enqueueing a programmatic answer."""

        if self._suspended is None:
            self._session_busy = False
            st0 = self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error="No pending prompt.",
            )
            return replace(st0, status=_failure_category_for_terminal_step(st0))
        ex = self._suspended
        self._suspended = None
        self._resume_choice_stage_id = ex.state_id

        def _resume_work() -> None:
            with _core_utils.llm_execution_scope(self._owner._runtime_config.llm_execution):
                dispatch_pipeline_resume(self, ex)

        try:
            lock = getattr(self._owner, "_pipeline_writer_lock", None)
            if lock is not None and self._session_mode == "writer":
                with lock:
                    _resume_work()
            else:
                _resume_work()
        except RefinementRetry:
            self._resume_choice_stage_id = None
            return self._continue_after_refinement_retry()
        except PipelineSuspended as ex2:
            if ex2.state_id in ("empty_choice_queue", "choice_queue_mismatch"):
                self.reset()
                return self._terminal_error_step(ex2.message_for_caller)
            self._suspended = ex2
            return self._suspend_to_step(ex2)
        except Exception as exc:
            debug(f"[main_execution.PipelineSession._resume_from_suspend] unexpected error: {exc!r}")
            self._reset_after_turn()
            self._session_busy = False
            st_r = self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error=str(exc),
            )
            return replace(st_r, status=_failure_category_for_terminal_step(st_r))
        finally:
            self._resume_choice_stage_id = None
        return self._completed_step()
