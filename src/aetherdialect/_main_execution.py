"""Entry points for seed warmup, QSim, interactive runs, programmatic PipelineSession, and artifact helpers. Optional ``pyspark.sql.SparkSession`` is imported at module load when available for engine reachability checks. Per-engine dialect modules are imported via ``_dialect_postgres`` and ``_dialect_sqlglot_engines`` so ``register_dialect`` runs before ``list_engines()`` is used."""

from __future__ import annotations

import contextlib
import copy
import glob
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
import zipfile
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pandas
import tomllib
from platformdirs import user_data_dir

from ._config import (
    ConfigError,
    CsvRuntimeConfig,
    DatabricksRuntimeConfig,
    DuckDBRuntimeConfig,
    EngineConfig,
    EngineRuntimeConfig,
    PolicyConfig,
    QSimConfig,
    SeedWarmupConfig,
    llm_credentials_configured,
)
from ._constants import (
    AETHERSPACE_ARTIFACT_VERSION,
    AETHERSPACES_SEGMENT,
    APPLIED_MAP_ARCHIVE_RETENTION_COUNT,
    APPLIED_MAP_ARCHIVE_TIMESTAMP_RE,
    ARTIFACT_DIRECTORY_SEGMENT,
    AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD,
    AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT,
    AZURE_OPENAI_ENV_REQUIRED,
    DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED,
    DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE,
    DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
    DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
    DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED,
    DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
    DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED,
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION,
    ENGINE_STORAGE_SLUG_MAX_CHARS,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_MIGRATION_MAP_FILENAME,
    FEEDBACK_NOTED_LINE,
    INTERACTIVE_STAGE_SQL_FEEDBACK,
    JSON_COMPACT_SEPARATORS,
    MASTER_AETHERSPACE_NAME,
    MIGRATION_HEADER_BY_TIER,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_FILENAME,
    NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
    NAMED_SCHEMA_CONTEXT_PREFIX,
    NORMALIZED_SEEDS_TXT,
    OPENAI_ENV_REQUIRED,
    PERMISSION_DENIED_USER_MESSAGE,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_EXECUTE,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_SQL,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT,
    QSIM_QUESTIONS_PATTERN,
    REPHRASE_HINT_MESSAGES,
    SAVED_LINE,
    SCHEMA_CONTEXT_CACHE_NAME,
    SCHEMA_CONTEXT_CACHE_VERSION,
    SCHEMA_CONTEXT_CACHED_DDL,
    SCHEMA_CONTEXT_CACHED_NOTES,
    SCHEMA_CONTEXT_NAMED_SPEC_GLOB,
    SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT,
    SCHEMA_OVERRIDES_VERSION,
    SECRET_ENV_KEYS,
    SEED_NORMALIZATION_JSON,
    SESSION_INTENT_FEEDBACK_BODY,
    SESSION_KIND_ERROR,
    SESSION_KIND_IDLE,
    SESSION_KIND_RESULT,
    SESSION_PERSISTENCE_FORMAT_VERSION,
    SESSION_PROMPT_REASON,
    SESSION_PROMPT_YESNO,
    SESSION_USER_FEEDBACK_BODY,
    SIMULATION_CACHE_EXACT_FILENAMES,
    SIMULATION_CACHE_GLOB_PATTERNS,
    SUSPEND_ID_TO_SESSION_KIND,
    TABLE_PREVIEW_DEFAULT_LIMIT,
    TABLE_PREVIEW_MAX_LIMIT,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_PARTITION_PREFIX,
    TEMPLATE_STORE_SEGMENT,
    TEMPLATE_STORE_SPACES_SEGMENT,
    TOML_ENGINE_FIELD_MAPS,
    TOML_SECTION_TO_ENGINE,
    TRUST_AUTO_ACCEPT_THRESHOLD,
    USER_REJECTED_RESULT_BUCKET_TIPS,
    WARMUP_PHASE_B,
    WARMUP_PHASE_C,
    WARMUP_PHASE_D,
    WARMUP_PHASE_E,
    WARMUP_PHASE_F,
    WARMUP_PHASE_G,
    WARMUP_PHASE_I,
    WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS,
    WRITE_QUEUE_FILENAME,
    WRITE_QUEUE_MAX_BYTES_PER_DRAIN,
    env_any_nonempty,
    env_first_nonempty,
    env_role_hint,
    is_file_engine,
)
from ._core_utils import (
    BusinessKnowledgeHolder,
    InteractiveChoicePort,
    RephraseHint,
    active_engine_identity,
    artifact_lock,
    artifact_manifest_incompatible_with_package,
    business_knowledge_scope,
    debug,
    decode_write_queue_event,
    detect_legacy_artifacts,
    diagnostic_debug_enabled,
    diagnostic_segment,
    drain_diagnostic_collector,
    drain_llm_usage_records,
    emit_llm_usage_summary_diagnostics,
    interactive_yes_no,
    invalid_input,
    llm_call_audit_details,
    llm_execution_scope,
    llm_turn_audit_details,
    llm_turn_cost_diagnostic,
    load_runtime_config,
    manifest_matches_schema,
    normalize_question,
    note_interactive_turn,
    notify,
    permission_denied_detail_logging_enabled,
    pop_ask_phase_callback,
    pop_engine_identity,
    pop_session_turn_cancel,
    print_rephrase_hint,
    progress,
    prompt,
    push_ask_phase_callback,
    push_engine_identity,
    push_session_turn_cancel,
    read_artifact_manifest,
    reconcile_execute_bind_params,
    register_structural_migration_handler,
    reset_diagnostic_collector,
    reset_turn_llm_scope,
    session_turn_cancelled,
    set_diagnostic_collector,
    set_turn_llm_scope,
    snapshot_llm_usage_records,
    take_and_clear_orphan_diagnostics,
    terminated,
    try_rename_migration_plan,
    wipe_filenames,
    wipe_globs,
    wipe_versioned_artifacts,
)

try:
    from pyspark.sql import SparkSession as _SparkSession
except ImportError:
    SparkSession: Any = None
else:
    SparkSession = _SparkSession

from ._contracts_base import (
    AccessError,
    AetherEngineInitResult,
    AetherFederationInitResult,
    AetherSpace,
    ConnectionError,
    DataQualityReport,
    Diagnostic,
    EngineContext,
    EngineIdentity,
    FailureCategory,
    FederationCapExceededError,
    FederationContext,
    FederationCoordinatorConfig,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationManifest,
    FederationMappings,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationPartialFailureError,
    FederationPlanTemplate,
    FederationTurnCancelledError,
    IntentInterpretation,
    IntentSummary,
    LLMConfig,
    MigrationPendingError,
    MigrationPreview,
    MigrationReport,
    MigrationTier,
    OwnerOnlyOperationError,
    ParameterBinding,
    ParamValue,
    PipelineSuspended,
    RetryableError,
    RuntimeConfig,
    SchemaRole,
    SensitivityClassification,
    SessionActiveError,
    SessionNotice,
    SessionStep,
    SessionTurnCancelledError,
    SourceRuntime,
    SpaceContext,
    TablePreviewResult,
    WriteQueueEvent,
    norm_schema_identifier,
    resolve_federation_qualified_ref,
    where_leaves,
)
from ._contracts_core import (
    DirectReuseSuspendContext,
    FederatedPlan,
    FederatedPrepareOutcome,
    FederatedSqlBundle,
    FederationExecutionContext,
    FeedbackKind,
    GenerationPath,
    InteractiveTailSnapshot,
    InterpretPlan,
    QuestionFeedbackEntry,
    QuestionFormStorage,
    RefinementContext,
    RefinementRetry,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    SqlExecuteSuspendContext,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
    TurnPolicySnapshot,
    UserFeedbackRejectSuspendContext,
    classify_seed_warmup_intent_complexity,
)
from ._contracts_schema import QSimSummary, SchemaGraph, SeedWarmupSummary, TableMetadata
from ._data_quality import parse_source_selections, validate_upload_sources
from ._dialect import (
    extra_where_ops_for_engine,
    get_dialect,
    get_runtime_config_class,
    is_permission_denied_error,
    list_engines,
    sqlglot_dialect_for_engine,
)
from ._expansion_ops import expand_gold_intents
from ._federation import (
    FederationConfigError,
    FederationInvariantError,
    apply_federation_migration_map,
    archive_federation_migration_map_file,
    assert_federation_member_graph_roster_complete,
    assert_federation_sql_history_warmup_allowed,
    assert_query_log_warmup_allowed,
    build_federation_manifest_from_members,
    cached_or_suggest_cross_source_mappings,
    check_federation_member_drift_at_turn_start,
    cleanup_abandoned_federation_spill_directories,
    clear_federated_turn_state,
    clear_federation_plan_templates,
    compose_composite_graph,
    composite_schema_payload_counts,
    compute_federation_storage_dir,
    detect_broken_cross_source_joins,
    detect_federation_topology_change,
    export_federation_migration_map_skeleton,
    federation_artifact_paths,
    federation_composite_migration_tier,
    federation_ineligible_answerable_hint,
    federation_plan_combine_hash,
    federation_plan_matches_template,
    federation_plan_residual_hash,
    federation_plan_step_fingerprints,
    federation_plan_topology_identity,
    federation_residual_column_headers,
    federation_source_artifacts_dir,
    federation_user_facing_error_message,
    intersect_member_where_ops,
    load_federation_composite_graph,
    load_federation_declaration_from_path,
    load_federation_member_graphs,
    load_federation_migration_map,
    lookup_federation_plan_template,
    mappings_replay_matches,
    member_graphs_from_engines,
    owner_is_aether_federation,
    persist_federation_tree,
    plan_federated_intent,
    probe_federation_member_connections,
    probe_federation_member_liveness,
    prune_cross_source_joins,
    prune_federation_aliases,
    prune_federation_mappings,
    prune_federation_plan_templates_on_drift,
    qsim_intent_eligible_on_federation,
    raise_if_descriptions_name_federation_sources,
    raise_if_member_notes_name_federation_sources,
    reconcile_authored_declaration_for_members,
    reconcile_federation_member_graphs,
    reconcile_federation_topology,
    recorded_federation_source_ids,
    resolve_anchored_temporal_bind,
    resolve_federated_combine,
    resolve_federated_member_schema,
    resolve_federation_preview_target,
    source_ids_for_intent,
    stamp_federation_member_graph,
    validate_cross_source_keys_on_graph,
    validate_federation_file_members,
    validate_federation_migration_map,
    validate_manifest_cross_source_joins,
)
from ._intent_process import (
    collect_structural_match_templates,
    list_union_match_candidates,
    match_template_for_union,
    reconcile_template_store_until_stable,
    structural_compare,
)
from ._llm_provider import bind_sandbox_runtime, clear_llm_clients, reset_mock_provider, reset_sandbox_runtime
from ._pipeline import (
    best_accepted_template_similarity,
    build_interactive_tail_snapshot,
    build_result_dataframe,
    clear_planner_schema_invalid_after_user_accept,
    complete_direct_sql_reuse_user_choice,
    complete_user_feedback_reject,
    compose_intent_confirm_session_message,
    confirm_intent_with_user,
    display_final_results_to_stdout,
    emit_explain_soft_diagnostics,
    execute_federated_prepare,
    force_reuse_saved_question,
    generate_and_validate_sql,
    handle_direct_sql_reuse,
    handle_user_feedback,
    load_pipeline_resources,
    match_question_level_template_reuse,
    parse_intent_via_llm,
    persist_federated_member_stores,
    prepare_federated_sql_plan,
    prepare_union_match_join_phase,
    refinement_retry_available,
    replay_federated_prepare_from_plan_template,
    result_columns_for_session,
    results_csv_output_path,
    save_result_csv,
    stamp_sql_shape,
    try_federation_plan_intake_reuse,
)
from ._qsim import generate_all_intents, generate_all_questions, instantiate_all
from ._refusal_diagnostics import (
    emit_session_refusal_diagnostic,
    refusal_diagnostic_code_for_federation_reason,
)
from ._schema_catalog import emit_description_enrichment_failed, emit_description_enrichment_noop, llm_classify_schema
from ._schema_graph import (
    assign_schema_graph_hashes,
    classify_migration_tier,
    compute_schema_limits,
    consumer_graph_is_permission_subset,
    diff_schemas,
    intersect_member_database_feature_capabilities,
    load_schema_graph_snapshot,
    raise_if_schema_unusable,
    upgrade_artifacts_schema_graph_id,
    validate_scope_against_graph,
)
from ._schema_overrides import apply_overrides_and_persist, build_schema_graph_with_diff, finalize_with_overrides
from ._seed_warmup import (
    accepted_template_instance_keys,
    get_next_seed_warmup_version,
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
    apply_federation_composite_migration_policy,
    apply_migration_policy,
    apply_schema_migration_map,
    build_parameter_bindings,
    ensure_template_store_space_layout,
    export_schema_migration_map_skeleton,
    has_any_rejection_history_for_question,
    load_schema_migration_map,
    load_template_store,
    merge_seed_warmup_templates_into_store,
    record_question_feedback,
    resolve_template_for_question,
    save_template_store,
    should_prompt_sql_feedback,
    store_to_templates,
    summarize_failure_for_memory,
    template_store_base_dir,
    template_store_dir_for_space,
    templates_to_store,
    validate_schema_migration_map,
)
from ._utils import (
    body_similarity_key,
    enumerate_zero_row_equality_where,
    flatten_param_values,
    intent_key,
    normalize_question_via_llm,
    patch_where_literal_on_intent,
    validate_question,
    zero_row_where_remediation_candidates,
    zero_row_where_suggestions,
)
from ._validation_execute import execute_guarded_sql, validate_sql


def _sanitize_tenant_slug(tenant_slug: str) -> str:
    """Return a filesystem-safe tenant segment for artifact storage paths."""
    safe = re.sub(r"[^a-z0-9_-]+", "-", str(tenant_slug).strip().lower()).strip("-")
    if not safe:
        raise ValueError("tenant_slug must contain at least one alphanumeric character after sanitization")
    return safe


def _remove_empty_template_shard_files(artifacts_dir: str) -> None:
    spaces_root = os.path.join(template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT)
    if not os.path.isdir(spaces_root):
        return
    for root, _dirs, files in os.walk(spaces_root):
        for name in files:
            if not name.startswith(TEMPLATE_STORE_PARTITION_PREFIX):
                continue
            path = os.path.join(root, name)
            if os.path.isfile(path) and os.path.getsize(path) == 0:
                try:
                    os.remove(path)
                except OSError:
                    pass


def _prune_applied_map_archives(artifacts_dir: str, *, keep: int = APPLIED_MAP_ARCHIVE_RETENTION_COUNT) -> None:
    archives: list[tuple[str, str]] = []
    try:
        names = os.listdir(artifacts_dir)
    except OSError:
        return
    for name in names:
        if APPLIED_MAP_ARCHIVE_TIMESTAMP_RE.search(name):
            ts = name.split(".applied.", 1)[1].rsplit(".json", 1)[0]
            archives.append((name, ts))
    archives.sort(key=lambda item: item[1])
    for name, _ts in archives[:-keep]:
        try:
            os.remove(os.path.join(artifacts_dir, name))
        except OSError:
            pass


def _clear_stale_write_queue(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
    manifest = read_artifact_manifest(artifacts_dir)
    if manifest is None:
        return
    if str(manifest.schema_graph_id or "") == str(active_schema_graph_id or ""):
        return
    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def prune_stale_artifact_auxiliaries(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
    """Prune empty template shards, old applied-map archives, and stale write queues."""
    _remove_empty_template_shard_files(artifacts_dir)
    _prune_applied_map_archives(artifacts_dir)
    _clear_stale_write_queue(artifacts_dir, active_schema_graph_id=active_schema_graph_id)


def _aetherspace_dir(engine_dir: str) -> str:
    return os.path.join(engine_dir, AETHERSPACES_SEGMENT)


def _aetherspace_path(engine_dir: str, name: str) -> str:
    safe = str(name).strip()
    if not safe or safe != safe.strip() or "/" in safe or "\\" in safe:
        raise ConfigError(f"invalid aetherspace name: {name!r}")
    return os.path.join(_aetherspace_dir(engine_dir), f"{safe}.json")


def _write_json_atomic(path: str, obj: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json.tmp", prefix=".aetherspace_", dir=directory, delete=False
        ) as tf:
            tmp_path = tf.name
            json.dump(obj, tf, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=True)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _write_jsonl_atomic(path: str, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows atomically."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl.tmp", prefix=".aetherdialect_", dir=directory, delete=False
        ) as tf:
            tmp_path = tf.name
            for row in rows:
                tf.write(json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n")
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def validate_space_context_against_graph(
    space_context: SpaceContext, schema_graph: SchemaGraph, *, federation_manifest: FederationManifest | None = None
) -> SpaceContext:
    """Normalize *space_context* and verify every table/column exists in *schema_graph*."""
    graph_tables = set(schema_graph.tables.keys())
    if space_context.tables:
        for tbl in space_context.tables:
            if tbl not in graph_tables:
                raise ConfigError(f"SpaceContext tables entry {tbl!r} is not in the schema graph")
    scope_tables = frozenset(space_context.tables) if space_context.tables else frozenset(graph_tables)
    resolve_manifest = federation_manifest or FederationManifest(
        federation_id="",
        sources=(),
        table_namespace={},
        cross_source_joins=(),
        coordinator=FederationCoordinatorConfig(),
    )
    if space_context.columns:
        normalized_cols: set[str] = set()
        for qc in space_context.columns:
            resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=schema_graph)
            tbl = resolved.table
            col = resolved.column
            normalized = resolved.qualified
            if tbl not in graph_tables:
                raise ConfigError(f"SpaceContext columns entry {qc!r} references unknown table {tbl!r}")
            if tbl not in scope_tables:
                raise ConfigError(f"SpaceContext columns entry {qc!r} references table {tbl!r} outside tables scope")
            tm = schema_graph.tables.get(tbl)
            if tm is None or col not in tm.columns:
                raise ConfigError(f"SpaceContext columns entry {qc!r} is not in the schema graph")
            normalized_cols.add(normalized)
        return SpaceContext(
            tables=space_context.tables,
            columns=frozenset(normalized_cols),
            deny_objects=space_context.deny_objects,
            deny_columns=space_context.deny_columns,
            notes_file=space_context.notes_file,
        )
    return space_context


def build_master_space_descriptor(schema_graph: SchemaGraph) -> AetherSpace:
    """Return the implicit full-scope ``master`` descriptor derived from *schema_graph*."""
    tables = tuple(sorted(schema_graph.tables.keys()))
    columns: list[str] = []
    for tname in tables:
        tm = schema_graph.tables[tname]
        for col_name in sorted(tm.columns.keys()):
            columns.append(f"{tname}.{col_name}")
    return AetherSpace(name=MASTER_AETHERSPACE_NAME, _scope={"tables": tables, "columns": tuple(columns)}, notes=None)


def _space_column_resolve_manifest(federation_manifest: FederationManifest | None) -> FederationManifest:
    return federation_manifest or FederationManifest(
        federation_id="",
        sources=(),
        table_namespace={},
        cross_source_joins=(),
        coordinator=FederationCoordinatorConfig(),
    )


def subset_graph_for_space(
    master_graph: SchemaGraph, space_context: SpaceContext, *, federation_manifest: FederationManifest | None = None
) -> dict[str, Any]:
    """Build a versioned snapshot dict for persistence from *master_graph* and *space_context*."""
    validated = validate_space_context_against_graph(
        space_context, master_graph, federation_manifest=federation_manifest
    )
    resolve_manifest = _space_column_resolve_manifest(federation_manifest)
    graph_tables = set(master_graph.tables.keys())
    if validated.tables:
        scope_tables = sorted(validated.tables)
    elif validated.deny_objects or validated.deny_columns:
        scope_tables = []
    else:
        scope_tables = sorted(graph_tables)
    frozenset(scope_tables)
    scope_columns: list[str] = []
    table_descriptions: dict[str, str] = {}
    column_meta: dict[str, dict[str, Any]] = {}
    for tname in scope_tables:
        tm = master_graph.tables[tname]
        desc = (tm.description or "").strip()
        if desc:
            table_descriptions[tname] = desc
        allowed_cols = sorted(tm.columns.keys())
        if validated.columns:
            scoped_cols: list[str] = []
            for qc in validated.columns:
                resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=master_graph)
                if resolved.table == tname:
                    scoped_cols.append(resolved.column)
            allowed_cols = sorted(scoped_cols)
        for col_name in allowed_cols:
            col = tm.columns[col_name]
            qc = f"{tname}.{col_name}"
            scope_columns.append(qc)
            entry: dict[str, Any] = {}
            cdesc = (col.description or "").strip()
            if cdesc:
                entry["description"] = cdesc
            sens = getattr(col, "sensitivity", None)
            if sens is not None and str(getattr(sens, "value", sens)) != "none":
                entry["sensitivity"] = str(getattr(sens, "value", sens))
            if entry:
                column_meta[qc] = entry
    return {
        "version": AETHERSPACE_ARTIFACT_VERSION,
        "tables": scope_tables,
        "columns": sorted(scope_columns),
        "deny_objects": sorted(validated.deny_objects),
        "deny_columns": sorted(validated.deny_columns),
        "table_descriptions": table_descriptions,
        "column_meta": column_meta,
        "notes": None,
        "notes_hash": "",
    }


def load_aetherspace_snapshot(engine_dir: str, name: str) -> dict[str, Any] | None:
    """
    Load a persisted space snapshot.

    Returns:
        The snapshot dict, or ``None`` when the file is absent, unreadable, or
        structurally invalid (non-version failures).

    Raises:

        ConfigError: When the file exists but its ``version`` does not match
        :data:`AETHERSPACE_ARTIFACT_VERSION`. Delete the snapshot and redefine
        the aetherspace; there is no migration path.
    """
    path = _aetherspace_path(engine_dir, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    found = payload.get("version")
    if found != AETHERSPACE_ARTIFACT_VERSION:
        raise ConfigError(
            f"aetherspace snapshot at {path!r} has version {found!r}; "
            f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
            f"Delete {path!r} and redefine the aetherspace so it is rewritten "
            f"at the current version."
        )
    if not _aetherspace_snapshot_payload_valid(payload):
        return None
    return payload


def _aetherspace_snapshot_payload_valid(payload: dict[str, Any]) -> bool:
    """Return True when *payload* has the expected structural shape (version checked by the loader)."""
    for key in ("tables", "columns"):
        raw = payload.get(key)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)):
            return False
        if not all(isinstance(x, str) for x in raw):
            return False
    table_descriptions = payload.get("table_descriptions")
    if table_descriptions is not None and not isinstance(table_descriptions, dict):
        return False
    column_meta = payload.get("column_meta")
    if column_meta is not None and not isinstance(column_meta, dict):
        return False
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        return False
    notes_hash = payload.get("notes_hash")
    if notes_hash is not None and not isinstance(notes_hash, str):
        return False
    return True


def save_aetherspace_snapshot(engine_dir: str, name: str, snapshot: dict[str, Any]) -> str:
    """Persist *snapshot* atomically; return the written path."""
    path = _aetherspace_path(engine_dir, name)
    _write_json_atomic(path, snapshot)
    return path


def list_saved_aetherspace_names(engine_dir: str) -> tuple[str, ...]:
    """Return sorted saved space names (excluding ``master``)."""
    root = _aetherspace_dir(engine_dir)
    if not os.path.isdir(root):
        return ()
    names: list[str] = []
    for entry in os.listdir(root):
        if not entry.endswith(".json"):
            continue
        stem = entry[: -len(".json")]
        if stem and stem != MASTER_AETHERSPACE_NAME:
            names.append(stem)
    return tuple(sorted(names))


def _aetherspace_export_path(engine_dir: str, name: str) -> str:
    safe = str(name).strip()
    if not safe or safe != safe.strip() or "/" in safe or "\\" in safe:
        raise ConfigError(f"invalid aetherspace name: {name!r}")
    return os.path.join(engine_dir, AETHERSPACES_SEGMENT, "_exports", f"{safe}.export.json")


def _parse_aetherspace_export_payload(payload: Any, *, source_path: str) -> dict[str, Any]:
    """Validate an exported aetherspace JSON document and return a persistable snapshot dict."""
    if not isinstance(payload, dict):
        raise ConfigError(f"malformed aetherspace export at {source_path!r}: expected a JSON object")
    found = payload.get("version")
    if found != AETHERSPACE_ARTIFACT_VERSION:
        raise ConfigError(
            f"aetherspace export at {source_path!r} has version {found!r}; "
            f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
            f"Delete the export file and re-export at the current version."
        )
    snap = {key: value for key, value in payload.items() if key != "name"}
    if not _aetherspace_snapshot_payload_valid(snap):
        raise ConfigError(f"malformed aetherspace export at {source_path!r}")
    return snap


def validate_aetherspace_snapshot_against_graph(
    snapshot: dict[str, Any],
    schema_graph: SchemaGraph,
    *,
    federation_manifest: FederationManifest | None = None,
) -> None:
    """Raise :class:`ConfigError` when *snapshot* scope references objects outside *schema_graph*."""
    space_context = SpaceContext(
        tables=frozenset(str(t) for t in (snapshot.get("tables") or ())),
        columns=frozenset(str(c) for c in (snapshot.get("columns") or ())),
        deny_objects=frozenset(str(t) for t in (snapshot.get("deny_objects") or ())),
        deny_columns=frozenset(str(c) for c in (snapshot.get("deny_columns") or ())),
    )
    validate_space_context_against_graph(
        space_context,
        schema_graph,
        federation_manifest=federation_manifest,
    )


def export_aetherspace_json(engine_dir: str, name: str, master_graph: SchemaGraph) -> Path:
    """Write a JSON export for *name* and return its path (pair with :func:`apply_aetherspace_json`)."""
    if name == MASTER_AETHERSPACE_NAME:
        snap = subset_graph_for_space(master_graph, SpaceContext(tables=frozenset(), columns=frozenset()))
        snap["name"] = MASTER_AETHERSPACE_NAME
    else:
        loaded = load_aetherspace_snapshot(engine_dir, name)
        if loaded is None:
            raise ConfigError(f"unknown aetherspace {name!r}")
        snap = dict(loaded)
        snap["name"] = name
    export_dir = os.path.join(engine_dir, AETHERSPACES_SEGMENT, "_exports")
    os.makedirs(export_dir, exist_ok=True)
    out_path = _aetherspace_export_path(engine_dir, name)
    _write_json_atomic(out_path, snap)
    return Path(out_path)


def apply_aetherspace_json(
    engine_dir: str,
    name: str,
    master_graph: SchemaGraph,
    *,
    source: str | os.PathLike[str] | None = None,
    federation_manifest: FederationManifest | None = None,
) -> AetherSpace:
    """Persist one named aetherspace from an exported JSON document."""
    norm = str(name).strip().lower()
    if not norm:
        raise ConfigError("aetherspace name must be non-empty")
    if norm == MASTER_AETHERSPACE_NAME:
        raise ConfigError(
            "master is the implicit full-scope space; it cannot be created or overwritten",
        )
    source_path = os.fspath(source) if source is not None else _aetherspace_export_path(engine_dir, norm)
    if not os.path.isfile(source_path):
        raise ConfigError(f"aetherspace export file not found: {source_path}")
    try:
        with open(source_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"could not read aetherspace export at {source_path!r}: {exc}") from exc
    snap = _parse_aetherspace_export_payload(payload, source_path=source_path)
    validate_aetherspace_snapshot_against_graph(
        snap,
        master_graph,
        federation_manifest=federation_manifest,
    )
    save_aetherspace_snapshot(engine_dir, norm, snap)
    return aetherspace_descriptor_from_snapshot(norm, snap)


def delete_aetherspace_snapshot(engine_dir: str, name: str) -> bool:
    """Delete one persisted named aetherspace snapshot. Returns ``True`` when a file was removed."""
    norm = str(name).strip().lower()
    if not norm:
        raise ConfigError("aetherspace name must be non-empty")
    if norm == MASTER_AETHERSPACE_NAME:
        raise ConfigError("master is the implicit full-scope space and cannot be deleted")
    path = _aetherspace_path(engine_dir, norm)
    if not os.path.isfile(path):
        raise ConfigError(f"unknown aetherspace {name!r}")
    os.unlink(path)
    return True


def aetherspace_descriptor_from_snapshot(name: str, snapshot: dict[str, Any]) -> AetherSpace:
    """Build an :class:`AetherSpace` read-only view from a stored snapshot dict."""
    tables_raw = snapshot.get("tables") or ()
    cols_raw = snapshot.get("columns") or ()
    tables = tuple(str(t) for t in tables_raw)
    columns = tuple(str(c) for c in cols_raw)
    notes_raw = snapshot.get("notes")
    notes = str(notes_raw).strip() if isinstance(notes_raw, str) and notes_raw.strip() else None
    return AetherSpace(name=name, _scope={"tables": tables, "columns": columns}, notes=notes)


def space_allowed_sets_from_snapshot(snapshot: dict[str, Any] | None) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(allowed_tables, allowed_columns)`` for enforcement; empty frozensets mean unrestricted."""
    if snapshot is None:
        return frozenset(), frozenset()
    tables_raw = snapshot.get("tables") or ()
    cols_raw = snapshot.get("columns") or ()
    tables = frozenset(norm_schema_identifier(str(t), what="aetherspace table") for t in tables_raw)
    columns: set[str] = set()
    for spec in cols_raw:
        raw = str(spec).strip()
        if raw.count(".") != 1:
            continue
        tbl, col = raw.split(".", 1)
        columns.add(
            f"{norm_schema_identifier(tbl, what='aetherspace table')}.{norm_schema_identifier(col, what='aetherspace column')}"
        )
    return tables, frozenset(columns)


def space_deny_sets_from_snapshot(snapshot: dict[str, Any] | None) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(deny_objects, deny_columns)`` from a persisted aetherspace snapshot."""
    if snapshot is None:
        return frozenset(), frozenset()
    deny_obj_raw = snapshot.get("deny_objects") or ()
    deny_col_raw = snapshot.get("deny_columns") or ()
    deny_objects = frozenset(norm_schema_identifier(str(t), what="aetherspace deny_objects") for t in deny_obj_raw)
    deny_columns: set[str] = set()
    for spec in deny_col_raw:
        raw = str(spec).strip()
        if raw.count(".") != 1:
            continue
        tbl, col = raw.split(".", 1)
        deny_columns.add(
            f"{norm_schema_identifier(tbl, what='aetherspace deny table')}.{norm_schema_identifier(col, what='aetherspace deny column')}"
        )
    return deny_objects, frozenset(deny_columns)


def intersect_space_scope(
    base_tables: frozenset[str],
    base_columns: frozenset[str],
    base_deny_objects: frozenset[str],
    base_deny_columns: frozenset[str],
    ephemeral: SpaceContext | None,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """Intersect ephemeral session scope with base aetherspace scope. Ephemeral allows never widen base allows. Deny lists from both layers are unioned. When either allow side is empty it is treated as unrestricted at that layer, matching engine-context composition rules."""
    if ephemeral is None or not (
        ephemeral.tables or ephemeral.columns or ephemeral.deny_objects or ephemeral.deny_columns
    ):
        return base_tables, base_columns, base_deny_objects, base_deny_columns

    if base_tables:
        if ephemeral.tables:
            tables = frozenset(t for t in ephemeral.tables if t in base_tables)
        else:
            tables = base_tables
    else:
        tables = ephemeral.tables

    if base_columns:
        if ephemeral.columns:
            columns = frozenset(c for c in ephemeral.columns if c in base_columns)
        else:
            columns = base_columns
    else:
        columns = ephemeral.columns

    deny_objects = base_deny_objects | ephemeral.deny_objects
    deny_columns = base_deny_columns | ephemeral.deny_columns

    overlap_obj = tables & deny_objects
    if overlap_obj:
        raise ConfigError(f"SpaceContext tables and deny_objects overlap: {sorted(overlap_obj)!r}")

    for table_name in deny_objects:
        for spec in deny_columns:
            denied_table, _, _rest = spec.partition(".")
            if denied_table != "*" and denied_table == table_name:
                raise ConfigError(f"deny_objects entry {table_name!r} conflicts with deny_columns entry {spec!r}")

    return tables, columns, deny_objects, deny_columns


def _normalize_preview_limit(limit: int) -> int:
    if limit < 1:
        raise ConfigError(f"preview limit must be positive, got {limit}")
    return min(int(limit), TABLE_PREVIEW_MAX_LIMIT)


def _resolve_preview_scope_context(owner: Any) -> EngineContext | FederationContext:
    runtime_cfg = getattr(owner, "_runtime_config", None)
    execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
    if execution_context is not None:
        return execution_context
    if runtime_cfg is not None:
        ctx = getattr(runtime_cfg, "engine_context", None)
        if ctx is not None:
            return ctx
    return EngineContext()


def _table_allowed_in_preview_scope(
    table_name: str,
    schema_graph: SchemaGraph,
    scope_ctx: EngineContext | FederationContext,
    visible_objects: frozenset[str] | None,
) -> bool:
    allowed = set(schema_graph.tables.keys())
    if scope_ctx.allow_objects:
        allowed &= set(scope_ctx.allow_objects)
    if scope_ctx.deny_objects:
        allowed -= set(scope_ctx.deny_objects)
    if visible_objects is not None:
        allowed &= set(visible_objects)
    return table_name in allowed


def _preview_column_in_scope(
    table_name: str,
    col_name: str,
    scope_ctx: EngineContext | FederationContext,
) -> bool:
    if (table_name, col_name) in scope_ctx.qualified_denies():
        return False
    if col_name in scope_ctx.glob_column_denies():
        return False
    if not scope_ctx.allow_columns:
        return True
    if (table_name, col_name) in scope_ctx.qualified_allows():
        return True
    return col_name in scope_ctx.glob_column_allows()


def _preview_columns_for_table(
    table: TableMetadata,
    table_name: str,
    scope_ctx: EngineContext | FederationContext,
    schema_graph: SchemaGraph,
) -> list[tuple[str, bool]]:
    """Return ordered ``(column_name, redact_values)`` pairs for preview projection."""
    deny_set = (schema_graph.deny_columns or {}).get(table_name, set())
    disallowed = (schema_graph.disallowed_columns or {}).get(table_name, set())
    out: list[tuple[str, bool]] = []
    for col_name in sorted(table.columns.keys()):
        col = table.columns[col_name]
        if col.sensitivity == SensitivityClassification.HIDDEN:
            continue
        if col_name in deny_set or col_name in disallowed or col.is_denied:
            continue
        if not _preview_column_in_scope(table_name, col_name, scope_ctx):
            continue
        redact = col.sensitivity == SensitivityClassification.RESTRICTED or not col.is_selectable
        out.append((col_name, redact))
    return out


def _build_preview_sql(
    dialect: Any,
    physical_table: str,
    select_specs: list[tuple[str, str]],
    limit: int,
) -> str:
    quote = dialect.quote_identifier
    if not select_specs:
        cols_sql = "1"
    else:
        parts = [f"{quote(phys)} AS {quote(out_name)}" for out_name, phys in select_specs]
        cols_sql = ", ".join(parts)
    return f"SELECT {cols_sql} FROM {quote(physical_table)} LIMIT {int(limit)}"


def preview_scoped_table(
    *,
    table_name: str,
    schema_graph: SchemaGraph,
    dialect: Any,
    scope_ctx: EngineContext | FederationContext,
    schema_role: str,
    visible_objects: frozenset[str] | None,
    limit: int = TABLE_PREVIEW_DEFAULT_LIMIT,
    physical_table: str | None = None,
    column_physical_names: Mapping[str, str] | None = None,
    member_schema_graph: SchemaGraph | None = None,
) -> TablePreviewResult:
    """Return a bounded table sample through scope and sensitivity gates."""
    bounded_limit = _normalize_preview_limit(limit)
    norm_table = str(table_name).strip()
    if not norm_table:
        raise ConfigError("table_name must be non-empty")
    if norm_table not in schema_graph.tables:
        raise ConfigError(f"unknown table {table_name!r}")
    if not _table_allowed_in_preview_scope(norm_table, schema_graph, scope_ctx, visible_objects):
        raise AccessError("preview_table", PERMISSION_DENIED_USER_MESSAGE)

    table = schema_graph.tables[norm_table]
    preview_columns = _preview_columns_for_table(table, norm_table, scope_ctx, schema_graph)
    phys_table = physical_table or norm_table
    phys_map = dict(column_physical_names or ())
    select_specs: list[tuple[str, str]] = []
    fetch_columns: list[tuple[str, bool]] = []
    for logical_name, redact in preview_columns:
        if redact:
            fetch_columns.append((logical_name, True))
            continue
        phys_name = phys_map.get(logical_name, logical_name)
        select_specs.append((logical_name, phys_name))
        fetch_columns.append((logical_name, False))

    sql = _build_preview_sql(dialect, phys_table, select_specs, bounded_limit)
    if member_schema_graph is not None:
        ok, err, _cat, _diags = validate_sql(dialect, sql, schema=member_schema_graph)
        if not ok:
            raise ValueError(err or "sql validation failed")
        raw_rows = list(dialect.execute(sql, {}))
    else:
        raw_rows = list(
            execute_guarded_sql(
                dialect,
                sql,
                schema=schema_graph,
                schema_role=schema_role,
                schema_context=scope_ctx,
                visible_objects=visible_objects,
            )
        )

    fetched_by_name = {name: idx for idx, (name, _redact) in enumerate(fetch_columns) if not _redact}
    aligned_rows: list[tuple[Any, ...]] = []
    for raw in raw_rows[:bounded_limit]:
        row_cells: list[Any] = []
        for logical_name, redact in fetch_columns:
            if redact:
                row_cells.append(None)
                continue
            idx = fetched_by_name[logical_name]
            row_cells.append(raw[idx] if idx < len(raw) else None)
        aligned_rows.append(tuple(row_cells))

    column_names = tuple(name for name, _redact in fetch_columns)
    return TablePreviewResult(columns=column_names, rows=tuple(aligned_rows))


def preview_table_on_engine(
    engine: Any,
    table_name: str,
    *,
    limit: int = TABLE_PREVIEW_DEFAULT_LIMIT,
) -> TablePreviewResult:
    """Preview one table on a member engine through active scope and redaction."""
    return preview_scoped_table(
        table_name=table_name,
        schema_graph=engine._schema_graph,
        dialect=engine._dialect,
        scope_ctx=_resolve_preview_scope_context(engine),
        schema_role=str(getattr(engine, "_schema_role", "owner")),
        visible_objects=getattr(engine, "_consumer_visible_objects", None),
        limit=limit,
    )


def preview_table_on_federation(
    federation: Any,
    table_name: str,
    *,
    limit: int = TABLE_PREVIEW_DEFAULT_LIMIT,
) -> TablePreviewResult:
    """Preview one composite table through federation scope on the owning member."""
    member, physical_table, col_map = resolve_federation_preview_target(
        table_name,
        schema=federation._schema_graph,
        manifest=federation._federation_manifest,
        mappings=federation._federation_mappings,
        members=federation._members,
    )
    member_graphs = getattr(federation, "_federation_member_graphs", None) or {}
    source_id = ""
    for sid, eng in federation._members.items():
        if eng is member:
            source_id = sid
            break
    member_schema = member_graphs.get(source_id, member._schema_graph)
    return preview_scoped_table(
        table_name=table_name,
        schema_graph=federation._schema_graph,
        dialect=member._dialect,
        scope_ctx=_resolve_preview_scope_context(federation),
        schema_role=str(getattr(federation, "_schema_role", "owner")),
        visible_objects=getattr(federation, "_consumer_visible_objects", None),
        limit=limit,
        physical_table=physical_table,
        column_physical_names=col_map,
        member_schema_graph=member_schema,
    )


def build_subset_schema_for_space_notes(
    master_graph: SchemaGraph, space_context: SpaceContext, *, federation_manifest: FederationManifest | None = None
) -> SchemaGraph:
    """Return a deep-copied in-scope schema graph for notes-aware LLM classification."""
    validated = validate_space_context_against_graph(
        space_context, master_graph, federation_manifest=federation_manifest
    )
    resolve_manifest = _space_column_resolve_manifest(federation_manifest)
    graph_tables = set(master_graph.tables.keys())
    scope_tables = sorted(validated.tables) if validated.tables else sorted(graph_tables)
    subset_tables: dict[str, Any] = {}
    for tname in scope_tables:
        tm = copy.deepcopy(master_graph.tables[tname])
        if validated.columns:
            allowed: set[str] = set()
            for qc in validated.columns:
                resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=master_graph)
                if resolved.table == tname:
                    allowed.add(resolved.column)
            for col_name in list(tm.columns.keys()):
                if col_name not in allowed:
                    del tm.columns[col_name]
        subset_tables[tname] = tm
    return SchemaGraph(tables=subset_tables, join_paths_multi={})


def enrich_space_snapshot_with_notes(
    snapshot: dict[str, Any], master_graph: SchemaGraph, space_context: SpaceContext, notes_file: str
) -> dict[str, Any]:
    """Bake notes text and content hash into *snapshot*, optionally refining descriptions via LLM."""
    path = os.path.expanduser(str(notes_file).strip())
    if not os.path.isfile(path):
        raise ConfigError(f"notes_file not found: {notes_file!r}")
    with open(path, encoding="utf-8") as fh:
        notes_content = fh.read()
    notes_text = notes_content.strip() if notes_content.strip() else None
    notes_hash = hashlib.sha256(notes_content.encode("utf-8")).hexdigest()
    out = dict(snapshot)
    out["notes"] = notes_text
    out["notes_hash"] = notes_hash
    if not llm_credentials_configured():
        return out
    subset_sg = build_subset_schema_for_space_notes(master_graph, space_context)
    try:
        classifications = llm_classify_schema(subset_sg, notes_content)
    except Exception as exc:
        debug(f"[aetherspace.enrich_space_snapshot_with_notes] LLM classify skipped: {exc!r}")
        emit_description_enrichment_failed("aetherspace_notes", exc)
        return out
    table_descriptions = dict(out.get("table_descriptions") or {})
    column_meta = dict(out.get("column_meta") or {})
    scope_cols = frozenset(str(c) for c in (out.get("columns") or ()))
    enriched_any = False
    for tname in out.get("tables") or ():
        if tname not in classifications:
            continue
        _table_role, desc, col_classes = classifications[tname]
        if desc and str(desc).strip():
            table_descriptions[str(tname)] = str(desc).strip()
            enriched_any = True
        tm = subset_sg.tables.get(str(tname))
        if tm is None:
            continue
        for col_name, (_col_role, col_description, sensitivity) in col_classes.items():
            if col_name not in tm.columns:
                continue
            qc = f"{tname}.{col_name}"
            if scope_cols and qc not in scope_cols:
                continue
            entry = dict(column_meta.get(qc) or {})
            if col_description and str(col_description).strip():
                entry["description"] = str(col_description).strip()
                enriched_any = True
            if sensitivity is not None and str(sensitivity) not in ("", "none"):
                entry["sensitivity"] = str(sensitivity)
            if entry:
                column_meta[qc] = entry
    out["table_descriptions"] = table_descriptions
    out["column_meta"] = column_meta
    if not enriched_any:
        emit_description_enrichment_noop("aetherspace_notes")
    return out


def _remap_qualified_column(spec: str, tmap: Mapping[str, str], colmaps: Mapping[str, Mapping[str, str]]) -> str | None:
    raw = str(spec).strip()
    if raw.count(".") != 1:
        return None
    tbl, col = raw.split(".", 1)
    nt = tmap.get(tbl, tbl)
    nc = colmaps.get(tbl, {}).get(col, colmaps.get(nt, {}).get(col, col))
    return f"{nt}.{nc}"


def _prune_remap_string_list(
    values: list[str],
    *,
    tmap: Mapping[str, str],
    colmaps: Mapping[str, Mapping[str, str]],
    drop_tables: frozenset[str],
    drop_columns: frozenset[str],
    column_specs: bool,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        spec = str(raw).strip()
        if not spec:
            continue
        if column_specs:
            if spec.count(".") != 1:
                continue
            tbl, col = spec.split(".", 1)
            if tbl in drop_tables or spec in drop_columns or f"{tbl}.{col}" in drop_columns:
                continue
            remapped = _remap_qualified_column(spec, tmap, colmaps)
            if remapped is None:
                continue
            spec = remapped
        else:
            if spec in drop_tables:
                continue
            spec = tmap.get(spec, spec)
        if spec not in seen:
            seen.add(spec)
            out.append(spec)
    return sorted(out)


def _apply_structural_edit_to_aetherspace_snapshot(
    snapshot: dict[str, Any],
    *,
    tmap: Mapping[str, str],
    colmaps: Mapping[str, Mapping[str, str]],
    drop_tables: frozenset[str],
    drop_columns: frozenset[str],
    column_retypes: tuple[tuple[str, str, str], ...] = (),
) -> dict[str, Any]:
    out = dict(snapshot)
    tables = _prune_remap_string_list(
        [str(t) for t in (out.get("tables") or ())],
        tmap=tmap,
        colmaps=colmaps,
        drop_tables=drop_tables,
        drop_columns=drop_columns,
        column_specs=False,
    )
    columns = _prune_remap_string_list(
        [str(c) for c in (out.get("columns") or ())],
        tmap=tmap,
        colmaps=colmaps,
        drop_tables=drop_tables,
        drop_columns=drop_columns,
        column_specs=True,
    )
    table_descriptions: dict[str, str] = {}
    for key, val in dict(out.get("table_descriptions") or {}).items():
        tbl = str(key).strip()
        if tbl in drop_tables:
            continue
        nt = tmap.get(tbl, tbl)
        if isinstance(val, str) and val.strip():
            table_descriptions[nt] = val.strip()
    column_meta: dict[str, dict[str, Any]] = {}
    for key, meta in dict(out.get("column_meta") or {}).items():
        remapped = _remap_qualified_column(str(key), tmap, colmaps)
        if remapped is None:
            continue
        tbl = remapped.split(".", 1)[0]
        if tbl in drop_tables or remapped in drop_columns:
            continue
        if isinstance(meta, dict):
            entry = dict(meta)
            if column_retypes:
                tbl, col = remapped.split(".", 1)
                for rt_tbl, rt_col, new_dt in column_retypes:
                    if rt_tbl == tbl and rt_col == col:
                        entry["value_type"] = new_dt
            column_meta[remapped] = entry
    out["tables"] = tables
    out["columns"] = columns
    out["table_descriptions"] = table_descriptions
    out["column_meta"] = column_meta
    out["deny_objects"] = _prune_remap_string_list(
        [str(t) for t in (out.get("deny_objects") or ())],
        tmap=tmap,
        colmaps=colmaps,
        drop_tables=drop_tables,
        drop_columns=drop_columns,
        column_specs=False,
    )
    out["deny_columns"] = _prune_remap_string_list(
        [str(c) for c in (out.get("deny_columns") or ())],
        tmap=tmap,
        colmaps=colmaps,
        drop_tables=drop_tables,
        drop_columns=drop_columns,
        column_specs=True,
    )
    return out


def apply_structural_migration_to_aetherspace_snapshots(
    engine_dir: str,
    *,
    dropped_tables: tuple[str, ...] = (),
    dropped_columns: tuple[str, ...] = (),
    table_renames: tuple[tuple[str, str], ...] = (),
    column_renames: tuple[tuple[str, str, str], ...] = (),
    column_retypes: tuple[tuple[str, str, str], ...] = (),
) -> int:
    """Prune or remap table/column references inside every persisted aetherspace snapshot."""
    tmap = {old: new for old, new in table_renames if old and new and old != new}
    colmaps: dict[str, dict[str, str]] = defaultdict(dict)
    for ot, oc, nc in column_renames:
        colmaps[ot][oc] = nc
        nt = tmap.get(ot, ot)
        if nt != ot:
            colmaps.setdefault(nt, {})[oc] = nc
    drop_tables = frozenset(dropped_tables)
    drop_columns = frozenset(dropped_columns)
    if not (drop_tables or drop_columns or tmap or colmaps or column_retypes):
        return 0
    updated = 0
    root = _aetherspace_dir(engine_dir)
    if not os.path.isdir(root):
        return 0
    for entry in os.listdir(root):
        if not entry.endswith(".json"):
            continue
        stem = entry[: -len(".json")]
        if not stem or stem == MASTER_AETHERSPACE_NAME:
            continue
        path = os.path.join(root, entry)
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        edited = _apply_structural_edit_to_aetherspace_snapshot(
            payload,
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_retypes=column_retypes,
        )
        if edited != payload:
            _write_json_atomic(path, edited)
            updated += 1
    return updated


def apply_structural_migration_to_named_context_specs(
    engine_dir: str,
    *,
    dropped_tables: tuple[str, ...] = (),
    dropped_columns: tuple[str, ...] = (),
    table_renames: tuple[tuple[str, str], ...] = (),
    column_renames: tuple[tuple[str, str, str], ...] = (),
) -> int:
    """Prune or remap allow/deny lists inside named ``schema_context.<name>.json`` specs."""
    tmap = {old: new for old, new in table_renames if old and new and old != new}
    colmaps: dict[str, dict[str, str]] = defaultdict(dict)
    for ot, oc, nc in column_renames:
        colmaps[ot][oc] = nc
        nt = tmap.get(ot, ot)
        if nt != ot:
            colmaps.setdefault(nt, {})[oc] = nc
    drop_tables = frozenset(dropped_tables)
    drop_columns = frozenset(dropped_columns)
    if not (drop_tables or drop_columns or tmap or colmaps):
        return 0
    updated = 0
    for path in glob.glob(os.path.join(engine_dir, SCHEMA_CONTEXT_NAMED_SPEC_GLOB)):
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        edited = dict(payload)
        for key, column_specs in (("allow_columns", True), ("deny_columns", True), ("allow_objects", False)):
            raw_vals = payload.get(key)
            if not isinstance(raw_vals, list):
                continue
            edited[key] = _prune_remap_string_list(
                [str(v) for v in raw_vals],
                tmap=tmap,
                colmaps=colmaps,
                drop_tables=drop_tables,
                drop_columns=drop_columns,
                column_specs=column_specs,
            )
        if edited != payload:
            _write_json_atomic(path, edited)
            updated += 1
    return updated


def apply_structural_migration_to_persisted_scopes(
    engine_dir: str,
    *,
    dropped_tables: tuple[str, ...] = (),
    dropped_columns: tuple[str, ...] = (),
    table_renames: tuple[tuple[str, str], ...] = (),
    column_renames: tuple[tuple[str, str, str], ...] = (),
    column_retypes: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Apply table/column delete/remap migration to aetherspace snapshots and named context specs."""
    apply_structural_migration_to_aetherspace_snapshots(
        engine_dir,
        dropped_tables=dropped_tables,
        dropped_columns=dropped_columns,
        table_renames=table_renames,
        column_renames=column_renames,
        column_retypes=column_retypes,
    )
    apply_structural_migration_to_named_context_specs(
        engine_dir,
        dropped_tables=dropped_tables,
        dropped_columns=dropped_columns,
        table_renames=table_renames,
        column_renames=column_renames,
    )


def _normalize_context_name(name: str) -> str:
    norm = str(name).strip().lower()
    if not norm:
        raise ConfigError("engine context name must be non-empty")
    if "/" in norm or "\\" in norm:
        raise ConfigError(f"invalid engine context name: {name!r}")
    return norm


def _named_schema_context_path(engine_dir: str, name: str) -> str:
    safe = _normalize_context_name(name)
    return os.path.join(engine_dir, f"{NAMED_SCHEMA_CONTEXT_PREFIX}{safe}.json")


def _validate_scope_list_fields(payload: dict[str, Any]) -> None:
    for key in ("allow_objects", "deny_objects", "deny_columns", "allow_columns"):
        if key not in payload:
            continue
        val = payload[key]
        if val is None:
            continue
        if not isinstance(val, list):
            raise ConfigError(f"{key} must be a list or null, got {type(val).__name__}")


def _schema_context_from_named_payload(payload: dict[str, Any]) -> EngineContext:
    """Reconstruct a named :class:`EngineContext` from a persisted sidecar."""
    _validate_scope_list_fields(payload)
    return EngineContext(
        allow_objects=frozenset(str(x) for x in (payload.get("allow_objects") or ())),
        deny_objects=frozenset(str(x) for x in (payload.get("deny_objects") or ())),
        deny_columns=frozenset(str(x) for x in (payload.get("deny_columns") or ())),
        allow_columns=frozenset(str(x) for x in (payload.get("allow_columns") or ())),
    )


def load_named_schema_context(engine_dir: str, name: str) -> EngineContext | None:
    """Load a persisted named context spec, or ``None`` when absent."""
    if _normalize_context_name(name) == MASTER_AETHERSPACE_NAME:
        return None
    path = _named_schema_context_path(engine_dir, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION:
        return None
    return _schema_context_from_named_payload(payload)


def save_named_schema_context(engine_dir: str, name: str, ctx: EngineContext) -> str:
    """Persist a named allow/deny spec atomically; return the written path."""
    norm = _normalize_context_name(name)
    if norm == MASTER_AETHERSPACE_NAME:
        raise ConfigError("master engine context is derived live and is not persisted as a named sidecar")
    payload: dict[str, Any] = {
        "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
        "allow_objects": sorted(ctx.allow_objects),
        "deny_objects": sorted(ctx.deny_objects),
        "deny_columns": sorted(ctx.deny_columns),
        "allow_columns": sorted(ctx.allow_columns),
    }
    path = _named_schema_context_path(engine_dir, norm)
    _write_json_atomic(path, payload)
    return path


def list_named_schema_context_names(engine_dir: str) -> tuple[str, ...]:
    """Return sorted saved named engine-context names (excluding ``master``)."""
    if not os.path.isdir(engine_dir):
        return ()
    prefix = NAMED_SCHEMA_CONTEXT_PREFIX
    suffix = ".json"
    names: list[str] = []
    for entry in os.listdir(engine_dir):
        if not entry.startswith(prefix) or not entry.endswith(suffix):
            continue
        stem = entry[len(prefix) : -len(suffix)]
        if stem and stem != MASTER_AETHERSPACE_NAME:
            names.append(stem)
    return tuple(sorted(names))


def export_named_schema_context_json(engine_dir: str, name: str, master_context: EngineContext) -> Path:
    """Write a read-only JSON export for one engine context and return its path."""
    norm = _normalize_context_name(name)
    if norm == MASTER_AETHERSPACE_NAME:
        snap: dict[str, Any] = {
            "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
            "name": MASTER_AETHERSPACE_NAME,
            "allow_objects": sorted(master_context.allow_objects),
            "deny_columns": sorted(master_context.deny_columns),
            "allow_columns": sorted(master_context.allow_columns),
        }
    else:
        loaded = load_named_schema_context(engine_dir, norm)
        if loaded is None:
            raise ConfigError(f"unknown engine context {name!r}")
        snap = {
            "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
            "name": norm,
            "allow_objects": sorted(loaded.allow_objects),
            "deny_columns": sorted(loaded.deny_columns),
            "allow_columns": sorted(loaded.allow_columns),
        }
    export_dir = os.path.join(engine_dir, "_exports")
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f"schema_context.{norm}.export.json")
    _write_json_atomic(out_path, snap)
    return Path(out_path)


def validate_named_engine_context_spec(ctx: EngineContext) -> None:
    """Reject master-only fields on a named engine-context registration spec."""
    if ctx.sql_file is not None:
        raise ConfigError("named engine context cannot set sql_file; only master defines DDL")
    if ctx.notes_file is not None:
        raise ConfigError("named engine context cannot set notes_file; only master defines notes")
    if ctx.include != "tables":
        raise ConfigError("named engine context cannot set include; only master defines include mode")


def validate_named_context_subset(master: EngineContext, named: EngineContext, schema_graph: SchemaGraph) -> None:
    """Ensure *named* is a subset-only refinement of *master* over *schema_graph*."""
    if named.include != master.include:
        raise ConfigError("named EngineContext cannot set include; only master defines include mode")
    graph_tables = set(schema_graph.tables.keys())
    if named.allow_objects:
        if master.allow_objects:
            extra = named.allow_objects - master.allow_objects
            if extra:
                raise ConfigError(f"named context allow_objects widens master scope: {sorted(extra)!r}")
        else:
            extra = named.allow_objects - graph_tables
            if extra:
                raise ConfigError(f"named context allow_objects references unknown tables: {sorted(extra)!r}")
    if not named.deny_columns.issuperset(master.deny_columns):
        missing = master.deny_columns - named.deny_columns
        raise ConfigError(f"named context must inherit all master deny_columns; missing {sorted(missing)!r}")
    if named.allow_columns:
        if master.allow_columns:
            extra_cols = named.allow_columns - master.allow_columns
            if extra_cols:
                raise ConfigError(f"named context allow_columns widens master scope: {sorted(extra_cols)!r}")
    validate_scope_against_graph(schema_graph, named)


def _federation_execution_allow_objects(
    master_ctx: FederationContext, composite_tables: frozenset[str]
) -> frozenset[str]:
    """Intersect federation master allow_objects with composite catalog tables."""
    if master_ctx.allow_objects:
        if composite_tables:
            return frozenset(t for t in master_ctx.allow_objects if t in composite_tables)
        return master_ctx.allow_objects
    return composite_tables


def _effective_execution_context(master: EngineContext, active: EngineContext, active_name: str) -> EngineContext:
    """Combine master and active named context into the execution-time RBAC scope."""
    if _normalize_context_name(active_name) == MASTER_AETHERSPACE_NAME:
        return EngineContext(
            allow_objects=master.allow_objects,
            include=master.include,
            deny_objects=master.deny_objects,
            deny_columns=master.deny_columns,
            allow_columns=master.allow_columns,
            notes_file=master.notes_file,
            sql_file=master.sql_file,
        )
    if master.allow_objects:
        if active.allow_objects:
            eff_allow = frozenset(t for t in active.allow_objects if t in master.allow_objects)
        else:
            eff_allow = master.allow_objects
    else:
        eff_allow = active.allow_objects
    eff_deny = master.deny_columns | active.deny_columns
    eff_deny_obj = master.deny_objects | active.deny_objects
    if master.allow_columns:
        if active.allow_columns:
            eff_allow_cols = frozenset(c for c in active.allow_columns if c in master.allow_columns)
        else:
            eff_allow_cols = master.allow_columns
    else:
        eff_allow_cols = active.allow_columns
    return EngineContext(
        allow_objects=eff_allow,
        include=master.include,
        deny_objects=eff_deny_obj,
        deny_columns=eff_deny,
        allow_columns=eff_allow_cols,
    )


def context_allowed_table_set(
    ctx: EngineContext | FederationContext, schema_graph: SchemaGraph, *, mappings: FederationMappings | None = None
) -> frozenset[str]:
    """Return tables visible under *ctx* against *schema_graph*."""
    tables = set(schema_graph.tables.keys())
    if ctx.allow_objects:
        tables &= set(ctx.allow_objects)
    if ctx.deny_objects:
        if mappings is not None and isinstance(ctx, FederationContext):
            denied = set(ctx.deny_objects)
            for table_map in mappings.logical_tables:
                if table_map.semantics not in ("union", "replica"):
                    continue
                member_tables = {member.table for member in table_map.members}
                if member_tables & denied == member_tables:
                    denied.discard(table_map.logical)
            tables -= denied
        else:
            tables -= set(ctx.deny_objects)
    return frozenset(tables)


def validate_space_subset_of_execution_context(
    space_tables: frozenset[str],
    space_columns: frozenset[str],
    execution_ctx: EngineContext | FederationContext,
    schema_graph: SchemaGraph,
    *,
    mappings: FederationMappings | None = None,
    federation_manifest: FederationManifest | None = None,
) -> None:
    """Raise :class:`ConfigError` when an aetherspace exceeds the active execution context."""
    if not space_tables and not space_columns:
        return
    resolve_manifest = _space_column_resolve_manifest(federation_manifest)
    allowed_tables = context_allowed_table_set(execution_ctx, schema_graph, mappings=mappings)
    if space_tables:
        extra = space_tables - allowed_tables
        if extra:
            raise ConfigError(f"aetherspace tables {sorted(extra)!r} exceed the active engine context scope")
    if space_columns:
        for qc in space_columns:
            resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=schema_graph)
            tbl, col = resolved.table, resolved.column
            if tbl not in allowed_tables:
                raise ConfigError(
                    f"aetherspace column {qc!r} references table {tbl!r} outside the active engine context"
                )
            tm = schema_graph.tables.get(tbl)
            if tm is None or col not in tm.columns:
                continue
            trial = EngineContext(
                allow_objects=execution_ctx.allow_objects,
                include=execution_ctx.include,
                deny_objects=execution_ctx.deny_objects,
                deny_columns=execution_ctx.deny_columns,
                allow_columns=execution_ctx.allow_columns | frozenset({qc}),
            )
            if not _column_allowed_in_context(tbl, col, trial, schema_graph):
                raise ConfigError(f"aetherspace column {qc!r} is outside the active engine context scope")


def _column_allowed_in_context(
    table_name: str, col_name: str, ctx: EngineContext | FederationContext, schema_graph: SchemaGraph
) -> bool:
    if (table_name, col_name) in ctx.qualified_denies():
        return False
    if col_name in ctx.glob_column_denies():
        return False
    if not ctx.allow_columns:
        return True
    if (table_name, col_name) in ctx.qualified_allows():
        return True
    return col_name in ctx.glob_column_allows()


def resolve_engine_context_plan(
    engine_context: EngineContext | str | None,
    engine_dir: str,
    *,
    schema_role: SchemaRole,
    load_master: EngineContext | None,
    prepare_master: EngineContext | None,
) -> tuple[EngineContext, EngineContext, str]:
    """Resolve construction input into master, active, and registration name. *load_master* is the on-disk master cache (may be ``None``). *prepare_master* is an explicit master object after ``_prepare_schema_context_for_init`` (may be ``None``)."""
    if isinstance(engine_context, str):
        name = _normalize_context_name(engine_context)
        if schema_role == "consumer" and name != MASTER_AETHERSPACE_NAME:
            pass
        master = load_master
        if master is None:
            raise ConfigError("create master engine context first; no cached schema_context.json was found")
        if name == MASTER_AETHERSPACE_NAME:
            return master, master, MASTER_AETHERSPACE_NAME
        named = load_named_schema_context(engine_dir, name)
        if named is None:
            raise ConfigError(f"unknown engine context {engine_context!r}")
        return master, named, name

    if engine_context is None:
        master = load_master
        if master is None:
            raise ConfigError(
                "schema_context is required on first initialisation. No cached "
                f"schema_context.json was found in {engine_dir!r}. Pass an explicit "
                "EngineContext (use EngineContext() to scope to the whole database)."
            )
        return master, master, MASTER_AETHERSPACE_NAME

    if schema_role == "consumer":
        raise OwnerOnlyOperationError("EngineContext(engine context definition)")
    master = prepare_master if prepare_master is not None else engine_context
    return master, master, MASTER_AETHERSPACE_NAME


def _sync_owner_template_cache(owner: Any, store: Any) -> None:
    """Keep the facade template cache aligned with the in-memory store view."""
    owner._store = store
    owner._templates = store_to_templates(store)


def _persist_template_store(owner: Any | None, store: Any) -> None:
    """Flush *store* to disk and refresh *owner*'s cached template map when present."""
    save_template_store(store)
    if owner is not None:
        _sync_owner_template_cache(owner, store)


def _owner_from_choice_port(choice_port: InteractiveChoicePort | None) -> Any | None:
    if choice_port is None:
        return None
    return getattr(choice_port, "_owner", None)


def _failure_category_for_terminal_step(step: SessionStep) -> str | None:
    """Map a terminal error :class:`SessionStep` to a coarse failure category string."""
    if step.kind != SESSION_KIND_ERROR:
        return None
    err = (step.error or "").strip()
    if not err:
        return None
    if step.federation_limit_key:
        if step.federation_limit_key == "timeout_ms":
            return FailureCategory.EXECUTION_TIMEOUT.value
        return FailureCategory.EXECUTION_OTHER_ERROR.value
    if step.federation_phase or step.federation_source_id:
        return FailureCategory.EXECUTION_OTHER_ERROR.value
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


def _federation_error_step_fields(exc: BaseException) -> dict[str, Any]:
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


def _federation_error_diagnostics(exc: BaseException) -> tuple[Diagnostic, ...]:
    """Build turn diagnostics for a structured federation terminal error."""
    fields = _federation_error_step_fields(exc)
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
            ),
        )
    if isinstance(exc, FederationMemberProbeError):
        return (
            Diagnostic(
                stage="execution",
                level="error",
                code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
                message=user_message,
                details=detail_tuple,
                source_id=source_id,
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
        ),
    )


def _interactive_attach_refinement_ctx(
    choice_port: InteractiveChoicePort | None, refinement_ctx: RefinementContext
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


def _reload_reader_learning_if_manifest_drift(owner: Any) -> None:
    """Reload partitioned template store and replay overrides when disk manifest drifts from the live graph."""
    manifest = read_artifact_manifest(str(owner._artifacts_dir))
    if manifest is None:
        return
    live_graph = getattr(owner, "_schema_graph", None)
    if not isinstance(live_graph, SchemaGraph):
        return
    if manifest_matches_schema(manifest, live_graph):
        return
    store = load_template_store(live_graph.schema_graph_id, live_graph)
    templates = store_to_templates(store)
    owner._store = store
    owner._templates = templates
    finalize_with_overrides(
        owner._schema_graph, EngineConfig.SCHEMA_JSON_PATH, dialect=getattr(owner, "_dialect", None)
    )


def _emit_write_queue_audit(owner: Any, event_type: str, details: tuple[tuple[str, str], ...]) -> None:
    """Forward write-queue drain outcomes to ``owner._audit_emit`` when an audit sink is configured."""
    fn = getattr(owner, "_audit_emit", None)
    if not callable(fn):
        return
    sg = getattr(owner, "_schema_graph", None)
    sh = str(getattr(sg, "effective_structural_hash", "") or "") or None
    fn(event_type, schema_hash=sh, details=details)


@dataclass
class _WriteQueueDrainTarget:
    """Live graph and template state for one write-queue drain pass."""

    schema_graph: SchemaGraph
    store: dict[str, Any] | TemplateStoreView
    templates: dict[str, Any]
    rejected: dict[str, Any]
    dialect: Any


def _owner_write_queue_drain_target(owner: Any) -> _WriteQueueDrainTarget:
    return _WriteQueueDrainTarget(
        schema_graph=owner._schema_graph,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        dialect=getattr(owner, "_dialect", None),
    )


def _federation_member_write_queue_targets(owner: Any) -> list[tuple[str, _WriteQueueDrainTarget]]:
    """Return per-member artifact dirs and drain targets for a federation owner."""
    if not getattr(owner, "_is_aether_federation", False):
        return []
    runtimes = getattr(owner, "_federation_source_runtimes", None) or {}
    member_graphs = getattr(owner, "_federation_member_graphs", None) or {}
    if not isinstance(runtimes, dict) or not isinstance(member_graphs, dict):
        return []
    targets: list[tuple[str, _WriteQueueDrainTarget]] = []
    for source_id in sorted(runtimes):
        runtime = runtimes.get(source_id)
        if runtime is None or not getattr(runtime, "artifacts_dir", None):
            raise FederationConfigError(
                f"federation member store missing for source_id {source_id!r}; "
                "each member must have its own artifact tree"
            )
        artifacts_dir = getattr(runtime, "artifacts_dir", None)
        graph = member_graphs.get(source_id)
        if graph is None:
            raise FederationConfigError(
                f"federation member store missing for source_id {source_id!r}; "
                "each member must have its own artifact tree"
            )
        graph_id = str(graph.schema_graph_id or "")
        store = load_template_store(graph_id, graph, artifacts_dir=str(artifacts_dir))
        targets.append(
            (
                str(artifacts_dir),
                _WriteQueueDrainTarget(
                    schema_graph=graph,
                    store=store,
                    templates=store_to_templates(store),
                    rejected={},
                    dialect=getattr(runtime, "dialect", None),
                ),
            )
        )
    return targets


def _drain_dispatch_write_queue_event(
    owner: Any, event: WriteQueueEvent, *, target: _WriteQueueDrainTarget | None = None
) -> bool:
    """Apply one queue event to *target* stores. Returns True when the template store should be saved."""
    tgt = target or _owner_write_queue_drain_target(owner)
    live = str(getattr(tgt.schema_graph, "schema_graph_id", "") or "")
    if not live or event.schema_graph_id != live:
        return False
    store = tgt.store
    templates: dict[str, Any] = tgt.templates
    rejected: dict[str, Any] = tgt.rejected
    schema = tgt.schema_graph
    dialect = tgt.dialect
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
            owner, AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD, (("kind", "feedback_record"), ("q_norm", q_norm))
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
        try:
            intent = RuntimeIntent.from_dict(ctx_doc.get("intent") or {})
        except (KeyError, TypeError, ValueError):
            notify("write_queue: malformed intent in ctx_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
            return False
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
            store=cast(dict[str, Any], store),
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
        try:
            intent = RuntimeIntent.from_dict(rep.get("intent") or {})
        except (KeyError, TypeError, ValueError):
            notify("write_queue: malformed intent in replay_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
            return False
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
            owner, AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT, (("kind", "template_accept"), ("q_norm", q_norm))
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
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as tf:
                json.dump(document, tf, ensure_ascii=False)
                tmp_path = tf.name
            apply_overrides_and_persist(schema, tmp_path, schema_json_path=EngineConfig.SCHEMA_JSON_PATH)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            notify(
                f"write_queue: override_proposal apply failed: {exc}",
                stage="pipeline",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
            )
            return False
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        _emit_write_queue_audit(
            owner,
            AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL,
            (("kind", "override_proposal"),),
        )
        return False

    if event.kind == "paraphrase_emit":
        notify(
            "write_queue: paraphrase_emit is reserved; line skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO
        )
        return False


def _drain_write_queue_at_path(
    owner: Any, artifacts_dir: str, *, target: _WriteQueueDrainTarget | None = None
) -> tuple[int, set[dict[str, Any] | TemplateStoreView]]:
    """Drain one artifact tree's write queue; return applied count and stores needing save."""
    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    applied = 0
    stores_to_save: set[dict[str, Any] | TemplateStoreView] = set()
    with artifact_lock(artifacts_dir, timeout=WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return 0, stores_to_save
        with open(path, "rb") as fh:
            body = fh.read()
        if not body:
            return 0, stores_to_save
        limit = WRITE_QUEUE_MAX_BYTES_PER_DRAIN
        if len(body) > limit:
            head = body[:limit]
            cut = head.rfind(b"\n")
            if cut == -1:
                return 0, stores_to_save
            to_process = head[: cut + 1]
            tail = head[cut + 1 :] + body[limit:]
        else:
            to_process = body
            tail = b""
        text = to_process.decode("utf-8", errors="replace")
        tgt = target or _owner_write_queue_drain_target(owner)
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
            evt = decode_write_queue_event(doc)
            if evt is None:
                notify("write_queue: unknown event skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                continue
            try:
                if _drain_dispatch_write_queue_event(owner, evt, target=tgt):
                    stores_to_save.add(tgt.store)
            except Exception as exc:
                notify(
                    f"write_queue: event dispatch failed: {exc!r}", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO
                )
            applied += 1
        with open(path, "wb") as out:
            out.write(tail)
    return applied, stores_to_save


def drain_write_queue(owner: Any, artifacts_dir: str) -> int:
    """Drain deferred reader events under the artifact lock; returns the number of events applied."""
    applied, stores_to_save = _drain_write_queue_at_path(owner, artifacts_dir)
    seen_dirs = {os.path.abspath(os.fspath(artifacts_dir))}
    for member_dir, member_target in _federation_member_write_queue_targets(owner):
        member_abs = os.path.abspath(os.fspath(member_dir))
        if member_abs in seen_dirs:
            continue
        seen_dirs.add(member_abs)
        member_applied, member_stores = _drain_write_queue_at_path(owner, member_dir, target=member_target)
        applied += member_applied
        stores_to_save.update(member_stores)
    for store in stores_to_save:
        if isinstance(store, TemplateStoreView):
            if store is getattr(owner, "_store", None):
                _persist_template_store(owner, store)
            else:
                save_template_store(store)
    return applied


def _interactive_run_intent_pass(
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
    _raise_if_session_turn_cancelled()
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
        note_interactive_turn(choice_port, outcome="parse_failed", error="Intent parse failed.")
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
        interpret_plan=interpret_plan,
    )
    return True


def _intent_interpretation_from_plan(plan: InterpretPlan | None) -> IntentInterpretation | None:
    """Project an :class:`InterpretPlan` into session-step traceability."""
    if plan is None:
        return None
    return IntentInterpretation(approach=plan.approach, grounding=plan.grounding)


def _build_intent_summary(intent: RuntimeIntent) -> IntentSummary:
    """Project a :class:`RuntimeIntent` into a compact :class:`IntentSummary` for session steps."""
    sel = tuple(sc.expr.signature_key for sc in intent.select_cols or [])
    flt = tuple(fp.signature_key for fp in where_leaves(intent.where) or [])
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


def _gold_intent_store_path_41_42_blocks_warmup(si: SeedWarmupIntent, templates: dict[str, Template]) -> bool:
    """Return True when a gold row matches the store only via disallowed warmup subpaths 4.1 / 4.2."""
    if (si.source or "gold") != "gold":
        return False
    rt = si.to_runtime_intent()
    for tmpl in templates.values():
        if tmpl.trust_level < 1:
            continue
        cr = structural_compare(rt, tmpl, mode="warmup_gold_store_check")
        if cr.union_sql_path in (GenerationPath.UNION_TEMPLATE_WIDEN, GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN):
            return True
    return False


def _get_next_qsim_version(artifacts_dir: str) -> int:
    """Return the next monotonic QSim version for an artifacts. directory."""
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


def format_qsim_summary_line(s: QSimSummary) -> str:
    """Single-line human summary for one QSim run."""
    return f"  v{s.version}: intents={s.num_intents}  questions={s.num_questions}  seed={s.seed}"


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
    deny_list = sorted(runtime.engine_context.deny_columns)
    lines.append(f"Schema context:  deny_columns={deny_list!r}")
    runtime_cls = cast(type[EngineRuntimeConfig], get_runtime_config_class(runtime.engine))
    fields = runtime_cls.connection_slug_fields()
    redacted = runtime_cls.redacted_fields()
    lines.append(f"{runtime.engine}:")
    for key, value in fields.items():
        display = _redact_display_value(key, value) if key in redacted else value
        lines.append(f"  {key}: {display}")
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
    *,
    federation_manifest: FederationManifest | None = None,
    federation_mappings: FederationMappings | None = None,
) -> None:
    """Run full QSim (intents, values, NL questions) and write. versioned. question text plus summary."""
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
    version = _get_next_qsim_version(base_dir)
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
            if qsim_intent_eligible_on_federation(intent.tables or [], schema, federation_manifest, federation_mappings)
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
        (intent.intent_id.rsplit("_v", 1)[0] if "_v" in intent.intent_id else intent.intent_id) for intent in results
    ]
    intent_counts = Counter(parent_ids)
    debug(f"  Questions per intent type: {dict(intent_counts)}")

    qname = QSIM_QUESTIONS_PATTERN.format(version=version)
    qsim_questions_path = os.path.join(base_dir, qname)
    qsim_summary_path = os.path.join(base_dir, "qsim_summary.json")

    debug(f"Saving QSim questions to {qsim_questions_path}...")
    with open(qsim_questions_path, "w", encoding="utf-8") as f:
        for i, qintent in enumerate(results, 1):
            f.write(f"{i}. {qintent.question}\n")

    summary_entry = QSimSummary(version=version, num_intents=len(intents), num_questions=len(results), seed=seed)

    summaries: list[Any] = []
    if os.path.exists(qsim_summary_path):
        with open(qsim_summary_path, encoding="utf-8") as f:
            loaded_summaries: Any = json.load(f)
        if not isinstance(loaded_summaries, list):
            summaries = []
        else:
            summaries = loaded_summaries
    summaries.append(summary_entry.to_dict())
    with open(qsim_summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS)

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
        "summary_artifact": os.path.basename(qsim_summary_path),
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
    _write_json_atomic(qsim_trace_path, qsim_trace_payload)
    _write_jsonl_atomic(qsim_trace_rows_path, intent_trace_rows + instantiation_trace_rows + question_trace_rows)

    debug(f"Question simulation complete: {len(results)} questions saved")
    notify(f"QSim version: {version}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

    if results and diagnostic_debug_enabled():
        debug("[main_execution.qsim_run_once] samples:")
        for i, item in enumerate(results[:5]):
            debug(f"[main_execution.qsim_run_once]   {i + 1}. {item.question}")

    notify(format_qsim_summary_line(summary_entry), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
    return None


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


def _get_questions_only(questions: list[str], *, output_path: str) -> None:
    """Print and save a numbered list of natural-language questions."""
    for i, q in enumerate(questions, 1):
        notify(f"{i}. {q}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            f.write(f"{i}. {q}\n")


def print_questions_bundle(version: int, artifacts_dir: str) -> None:
    """Load QSim questions for a version, print them, and mirror lines to ``qsim_v{version}_questions.txt`` in the working directory."""
    path = resolve_qsim_path(version, artifacts_dir)
    questions = _load_questions_from_qsim_txt(path)
    ver = int(version)
    out_path = os.path.join(artifacts_dir, f"qsim_v{ver}_questions.txt")
    _get_questions_only(questions, output_path=out_path)


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
    version = get_next_seed_warmup_version(output_dir)
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
    gold_intents_raw, gold_funnel, failure_trace_body, seed_norm_bundle = run_gold_intent_generation(
        schema, seed_filepath, interactive=interactive_gold, seed_warmup_version=version
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
    expanded_only = expand_gold_intents(gold_warmup_intents, schema, limits)
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

    debug(f"[{WARMUP_PHASE_D}] Pool union and body dedupe (body_key,tier): {len(deduped_pool)} unique rows")

    tmpl_map: dict[str, Template] = templates if templates is not None else {}
    blocked_gold_rows = [
        row
        for row in deduped_pool
        if (row.source or "gold") == "gold" and _gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
    ]
    gold_warmup_blocked_path41_or_42 = len(blocked_gold_rows)
    warmup_queue = [row for row in deduped_pool if not _gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)]
    debug(
        f"[{WARMUP_PHASE_D}] Gold vs store classification: gold_warmup_blocked_path41_or_42={gold_warmup_blocked_path41_or_42}; "
        f"queue {len(warmup_queue)} (expanded children keep distinct (body_key,tier))"
    )
    debug(f"[{WARMUP_PHASE_G}] Synthetic rows filtered by template_instance_key / ledger inside execute loop")

    notify(
        "Expansion and deduplication complete: pool size vs store classification "
        f"(expanded_synthetics={len(expanded_only)}, "
        f"unique_body_tier_rows={len(deduped_pool)}, "
        f"blocked_path_41_42={gold_warmup_blocked_path41_or_42}, "
        f"queued_for_warmup={len(warmup_queue)}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    debug("Join resolution (cached per table-set)")
    join_cache: dict[frozenset[str], Any] = {}
    for gold in gold_warmup_intents:
        resolve_joins_for_table_set(gold.tables or [], schema, gold.intent_id or "gold", join_cache)
    debug(f"Join cache seeded with {len(join_cache)} table-set entries")
    debug(f"[{WARMUP_PHASE_E}] Join cache seeded from gold table sets (reuse across pool)")
    notify(
        f"Stage D complete: join cache seeded (table_sets={len(join_cache)}).",
        stage="cli",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )

    with open(seed_filepath, "rb") as _seed_f:
        seed_content_sha256 = hashlib.sha256(_seed_f.read()).hexdigest()
    warmup_cache_session = open_seed_warmup_cache_session(output_dir, schema, seed_content_sha256)
    debug(f"[{WARMUP_PHASE_F}] Seed warmup cache manifest aligned to schema_hash and seed_content_hash")

    debug(
        f"[{WARMUP_PHASE_G}] Execute and validate; stratified sampling after successes; "
        f"[{WARMUP_PHASE_I}] question LLM, realism, templates only on full run"
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
    debug(f"Seed warmup execution results: {len(results)} rows, templates: {len(new_templates)}")
    exec_validation_drop = int(warmup_funnel.get("validation_drop", 0))
    exec_realism_drop = int(warmup_funnel.get("realism_drop", 0))
    exec_question_gen_failed = int(warmup_funnel.get("question_generation_failed", 0))
    exec_early_failed = int(warmup_funnel.get("early_pipeline_failed", 0))
    exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
    exec_total = len(results)
    exec_success = sum(1 for r in results if r.success)
    notify(
        "Stage E complete: per-intent SQL build, validation, execution, realism gate "
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
            merge_seed_warmup_templates_into_store(templates, [tmpl])
        template_store_view = store if isinstance(store, TemplateStoreView) else None
        writable_store = store if isinstance(store, dict) else {}
        reconcile_template_store_until_stable(templates, template_store_view=template_store_view)
        writable_store["next_id"] = updated_next_id
        saved_store = templates_to_store(writable_store, templates)
        save_template_store(saved_store)

    templates_added = (
        len(new_templates) if (federation_manifest is not None or (store is not None and templates is not None)) else 0
    )

    exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
    run_mode = "full"
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
            "registry_snapshot": registry_snapshot,
            **gold_funnel,
            "synthetic_unique_body_keys": len(deduped_pool),
            "synthetic_runnable_count": len(warmup_queue),
            **warmup_pool_operator_feature_stats(warmup_queue),
            "gold_prompts_count": seed_questions_loaded,
            "templates_added": templates_added,
            "execute_ok_count": exec_ok_ct,
            **warmup_funnel,
            "gold_warmup_blocked_path41_or_42": gold_warmup_blocked_path41_or_42,
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
        "Stage F complete: seed warmup report and bundle zip"
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
    notify(format_seed_warmup_summary(summary), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
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
    """Return the dialect label used for query-log dispatch."""
    label = getattr(dialect, "dialect_label", None)
    if isinstance(label, str) and label.strip():
        return label.strip().lower()
    return str(getattr(dialect, "name", "postgresql")).strip().lower()


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
    version = get_next_seed_warmup_version(output_dir)
    report_name = SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version)
    report_filepath = os.path.join(output_dir, report_name)

    debug(f"Starting SQL-history seed warmup run version {version}")
    notify(
        f"SQL-history seed warmup run version: {version}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info"
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
    gold_warmup_blocked_path41_or_42 = 0
    if expand:
        debug(f"[{WARMUP_PHASE_C}] Expansion: deterministic multi-depth expand_gold_intents")
        expanded_only = expand_gold_intents(gold_warmup_intents, schema, limits)
        full_pool: list[SeedWarmupIntent] = list(gold_warmup_intents) + expanded_only
        pool_body_tier: set[tuple[str, str]] = set()
        deduped_pool = []
        for pool_intent in full_pool:
            bk = body_similarity_key(pool_intent.to_runtime_intent())
            tier = classify_seed_warmup_intent_complexity(pool_intent).value
            key = (bk, tier)
            if key in pool_body_tier:
                continue
            pool_body_tier.add(key)
            deduped_pool.append(pool_intent)
        debug(f"[{WARMUP_PHASE_D}] Pool union and body dedupe (body_key,tier): {len(deduped_pool)} unique rows")
        blocked_gold_rows = [
            row
            for row in deduped_pool
            if (row.source or "gold") == "gold" and _gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
        ]
        gold_warmup_blocked_path41_or_42 = len(blocked_gold_rows)
        warmup_queue = [row for row in deduped_pool if not _gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)]
        debug(
            f"[{WARMUP_PHASE_D}] Gold vs store classification: gold_warmup_blocked_path41_or_42={gold_warmup_blocked_path41_or_42}; "
            f"queue {len(warmup_queue)} (expanded children keep distinct (body_key,tier))"
        )
        notify(
            "Expansion and deduplication complete: pool size vs store classification "
            f"(expanded_synthetics={len(expanded_only)}, "
            f"unique_body_tier_rows={len(deduped_pool)}, "
            f"blocked_path_41_42={gold_warmup_blocked_path41_or_42}, "
            f"queued_for_warmup={len(warmup_queue)}).",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

    join_cache: dict[frozenset[str], Any] = {}
    join_seed_rows = gold_warmup_intents if expand else warmup_queue
    for row in join_seed_rows:
        resolve_joins_for_table_set(row.tables or [], schema, row.intent_id or "sqlhist", join_cache)
    debug(f"Join cache seeded with {len(join_cache)} table-set entries")

    warmup_cache_session = open_seed_warmup_cache_session(
        output_dir, schema, sql_history_content_sha256=sql_history_content_hash
    )
    debug(f"[{WARMUP_PHASE_F}] Seed warmup cache manifest aligned to schema_hash and sql_history_content_hash")

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
        warmup_cache=warmup_cache_session,
        warmup_report_version=version,
        warmup_lattice_root=output_dir,
        max_kept_intents=max_kept_intents,
    )
    save_seed_warmup_cache_zip(
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
        "Stage E complete: per-intent SQL build, validation, execution "
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
            merge_seed_warmup_templates_into_store(templates, [tmpl])
        reconcile_template_store_until_stable(templates, template_store_view=template_store_view)
        writable_store["next_id"] = updated_next_id
        saved_store = templates_to_store(writable_store, templates)
        save_template_store(saved_store)

    templates_added = len(new_templates) if store is not None and templates is not None else 0

    exec_ok_ct = int(warmup_funnel.get("execute_ok_count", 0))
    run_mode = "full"
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
            "registry_snapshot": registry_snapshot,
            **gold_funnel,
            "sql_history_conversion_failures": len(fail_by_hash),
            "synthetic_unique_body_keys": len(deduped_pool),
            "synthetic_runnable_count": len(warmup_queue),
            **warmup_pool_operator_feature_stats(warmup_queue),
            "gold_prompts_count": seed_questions_loaded,
            "templates_added": templates_added,
            "execute_ok_count": exec_ok_ct,
            **warmup_funnel,
            "gold_warmup_blocked_path41_or_42": gold_warmup_blocked_path41_or_42,
        },
    )

    notify(
        "Stage F complete: seed warmup report"
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
    notify(format_seed_warmup_summary(summary), stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def run_seed_warmup_from_history_execution(
    self_engine: Any,
    sql_history_filepath: str,
    *,
    expand: bool = False,
    max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    seed: int | None = None,
) -> None:
    """Drive :func:`run_seed_warmup_execution` from a newline-oriented SQL history file."""
    assert_federation_sql_history_warmup_allowed(self_engine)
    schema = self_engine._schema_graph
    dialect = self_engine._dialect
    output_dir = str(self_engine._artifacts_dir)
    store = self_engine._store
    templates = self_engine._templates
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
        seed=seed,
        expand=expand,
        max_kept_intents=max_kept_intents,
    )


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
    """Drive :func:`run_seed_warmup_execution` from the engine query log."""
    assert_query_log_warmup_allowed(self_engine)
    schema = self_engine._schema_graph
    dialect = self_engine._dialect
    output_dir = str(self_engine._artifacts_dir)
    store = self_engine._store
    templates = self_engine._templates
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
        seed=seed,
        expand=expand,
        max_kept_intents=max_kept_intents,
    )


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
        rows=tuple(tuple(r) for r in rows),
        tmpl_sd=tmpl_sd,
        gen_out=gen_out,
        matched_rejected_template=matched_rejected_template,
        force_feedback=force_feedback,
        federated_prepare=federated_prepare,
        federated_bundle=federated_bundle,
    )


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


def snapshot_turn_policy() -> TurnPolicySnapshot:
    """Freeze per-turn policy knobs at suspend for federation and execute resume."""
    return TurnPolicySnapshot(
        max_compose_repairs=PolicyConfig.MAX_ASK_COMPOSE_REPAIRS,
        max_interpret_ground_retries=PolicyConfig.MAX_ASK_INTERPRET_GROUND_RETRIES,
        trust_auto_accept_threshold=TRUST_AUTO_ACCEPT_THRESHOLD,
    )


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
        rows=rows,
        federated_prepare=federated_prepare,
        federation_plan_id=str(federation_plan_id or gen_out.federation_plan_id or ""),
        federation_exec_context=exec_ctx_pairs,
        turn_policy=turn_policy if turn_policy is not None else snapshot_turn_policy(),
    )


def _federation_exec_context_from_pairs(
    pairs: tuple[tuple[str, Any], ...] | Sequence[tuple[str, Any]] | None,
) -> dict[str, Any]:
    if not pairs:
        return {}
    return {str(k): v for k, v in pairs}


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


def _verify_federation_execute_resume(ctx: SqlExecuteSuspendContext) -> None:
    """Ensure the federated plan approved at suspend matches the resume payload."""
    expected = str(ctx.federation_plan_id or ctx.gen_out.federation_plan_id or "")
    actual = str(ctx.gen_out.federation_plan_id or "")
    if expected and actual and expected != actual:
        raise FederationInvariantError(
            f"federation plan id mismatch on execute resume: expected {expected!r}, got {actual!r}"
        )


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


def _session_step_federation_fields_from_snap(snap: Mapping[str, Any], raw_outcome: str) -> dict[str, Any]:
    """Copy federation attribution from a stored turn outcome onto SessionStep fields."""
    fed_source = str(snap.get("federation_source_id") or "") or None
    fed_phase = str(snap.get("federation_phase") or "") or None
    fed_succeeded = tuple(snap.get("federation_succeeded") or ())
    if raw_outcome == "federation_partial_failure" or fed_source or fed_phase or fed_succeeded:
        return {
            "federation_source_id": fed_source,
            "federation_phase": fed_phase,
            "federation_succeeded": fed_succeeded,
            "retryable": bool(snap.get("retryable")) if raw_outcome == "federation_partial_failure" else False,
        }
    return {
        "federation_source_id": None,
        "federation_phase": None,
        "federation_succeeded": (),
        "retryable": False,
    }


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
        gate_kwargs = _consumer_sql_gate_kwargs(choice_port)
        exec_ctx = dict(federation_exec_context or {})
        progress("Executing federated SQL...")
        turn_session = choice_port if isinstance(choice_port, PipelineSession) else None
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
                _federation_gate_kwargs_by_source(
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
            and _persist_template_learning_for_pipeline_session(choice_port)
            and isinstance(fed_manifest, FederationManifest)
        ):
            member_graphs = getattr(owner, "_federation_member_graphs", None)
            if isinstance(member_graphs, dict) and member_graphs:
                member_stores = federation_stores_by_source(
                    owner, member_graphs, space_name=_session_space_name_for_federation(owner, choice_port)
                )
                if member_stores:
                    persist_federated_member_stores(
                        fed_prep.plan,
                        store=store or {},
                        stores_by_source=member_stores,
                    )
        federated_bundle = exec_outcome.bundle
        return list(exec_outcome.rows), federated_bundle
    progress("Executing SQL...")
    return _run_pipeline_sql_rows(intent=intent, schema=exec_schema, dialect=exec_dialect, tmpl_sd=tmpl_sd), None


def _run_pipeline_sql_rows(
    *, intent: Any, schema: SchemaGraph, dialect: Any, tmpl_sd: dict[str, Any] | None
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
    exec_bind = reconcile_execute_bind_params(exec_sql, exec_params)
    return list(dialect.execute(exec_sql, exec_bind))


def try_zero_row_where_remediation(
    intent: Any, schema: SchemaGraph, dialect: Any, tmpl_sd: dict[str, Any] | None
) -> tuple[Any, list[tuple[Any, ...]] | None]:
    """Attempt filter literal auto-fix after a zero-row execute using cached distinct values."""
    if not PolicyConfig.ZERO_ROW_WHERE_AUTO_FIX_ENABLED:
        return intent, None
    for where_param, literal, column, cached in enumerate_zero_row_equality_where(intent, schema):
        for candidate in zero_row_where_remediation_candidates(literal, cached):
            trial_intent = patch_where_literal_on_intent(intent, where_param, candidate)
            try:
                trial_rows = _run_pipeline_sql_rows(
                    intent=trial_intent, schema=schema, dialect=dialect, tmpl_sd=tmpl_sd
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
    if len(rows) > int(PolicyConfig.RESULT_ROW_COUNT_SOFT_WARNING):
        notify(
            f"Query result row count {len(rows)} exceeds the soft warning threshold.",
            stage="execution",
            code=DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
        )

    if len(rows) == 0:
        for suggestion in zero_row_where_suggestions(intent, schema):
            notify(suggestion, stage="execution", code=DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION)

    need_sql_feedback_prompt = force_feedback or should_prompt_sql_feedback(store, q_norm, gen_out.matched_template)
    is_session = choice_port is not None and isinstance(choice_port, PipelineSession)
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
            **_federation_result_contract_kwargs(
                gen_out, federated_prepare=federated_prepare, federated_bundle=federated_bundle
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
            _persist_template_store(_owner_from_choice_port(choice_port), store)
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
            **_federation_result_contract_kwargs(
                gen_out, federated_prepare=federated_prepare, federated_bundle=federated_bundle
            ),
        )
        if df_full is not None:
            art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
            artifacts_dir = str(art) if art is not None else None
            save_result_csv(
                df_full,
                output_path=results_csv_output_path(store, artifacts_dir=artifacts_dir),
            )

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
        **_federation_feedback_kwargs(owner, gen_out, choice_port=choice_port, federated_prepare=federated_prepare),
    )
    emit_llm_usage_summary_diagnostics(drain_llm_usage_records())
    row_tuples = [tuple(r) for r in rows]
    cols = result_columns_for_session(
        sql,
        row_tuples,
        intent=intent,
        **_federation_result_contract_kwargs(
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
            sql=_resolved_session_step_sql(
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
            sql=_resolved_session_step_sql(
                sql,
                gen_out=gen_out,
                federated_bundle=federated_bundle,
                federated_plan=federated_prepare.plan if federated_prepare is not None else None,
                generation_path=gen_out.generation_path,
            ),
            rows=row_tuples,
            columns=cols,
            intent=intent,
            federated_bundle=federated_bundle,
            federated_plan=federated_prepare.plan if federated_prepare is not None else None,
            generation_path=gen_out.generation_path,
        )


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


def _handle_federation_ineligible_plan(
    plan: FederatedPlan,
    *,
    choice_port: InteractiveChoicePort | None,
    store: dict[str, Any] | TemplateStoreView,
    owner: Any | None,
    persist_template_learning: bool,
) -> None:
    ineligible_reason = str(plan.ineligible_reason or "")
    refusal_code = refusal_diagnostic_code_for_federation_reason(ineligible_reason)
    if refusal_code:
        emit_session_refusal_diagnostic(
            refusal_code,
            ineligible_reason,
            stage="validation",
            source_id="composite",
            details=(("phase", "prepare"), ("reason", ineligible_reason)),
        )
    notify(
        ineligible_reason,
        stage="validation",
        code=DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE,
        source_id="composite",
        details=(("phase", "prepare"), ("reason", ineligible_reason)),
    )
    answerable = federation_ineligible_answerable_hint(plan.ineligible_reason)
    if answerable:
        notify(answerable, stage="rephrase_hint", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
    print_rephrase_hint(RephraseHint.FEDERATION_INELIGIBLE)
    note_interactive_turn(
        choice_port,
        outcome="validation_failed",
        error=plan.ineligible_reason,
        failure_kind=FailureCategory.DENIED_REFERENCE.value,
        refusal_diagnostic_code=refusal_code,
    )
    clear_federated_turn_state(choice_port)


def _check_federation_eligibility_before_confirm(
    intent: Any,
    schema: SchemaGraph,
    store: dict[str, Any] | TemplateStoreView,
    choice_port: InteractiveChoicePort | None,
    *,
    persist_template_learning: bool = True,
) -> bool:
    """Return False when a federated plan is ineligible and the turn should stop."""
    owner = _owner_from_choice_port(choice_port)
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
        space=_federation_space_for_choice_port(choice_port),
        member_graphs=(
            getattr(owner, "_federation_member_graphs", None)
            if owner is not None and isinstance(getattr(owner, "_federation_member_graphs", None), dict)
            else None
        ),
    )
    if not plan.ineligible_reason:
        return True
    _handle_federation_ineligible_plan(
        plan, choice_port=choice_port, store=store, owner=owner, persist_template_learning=persist_template_learning
    )
    return False


def _consumer_sql_gate_kwargs(choice_port: InteractiveChoicePort | None) -> dict[str, Any]:
    """Collect execution-scope parameters from the active programmatic session owner."""
    owner = getattr(choice_port, "_owner", None)
    schema_role = str(getattr(owner, "_schema_role", "owner") or "owner")
    execution_visible_objects = getattr(choice_port, "execution_visible_objects", None)
    runtime_cfg = getattr(owner, "_runtime_config", None)
    master_context = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
    execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
    scope_ctx = execution_context if execution_context is not None else master_context
    space_tables = getattr(choice_port, "space_tables", None)
    space_columns = getattr(choice_port, "space_columns", None)
    context_name = str(getattr(owner, "_context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
    return {
        "schema_role": schema_role,
        "visible_objects": execution_visible_objects,
        "schema_context": scope_ctx,
        "context_name": context_name,
        "space_allowed_tables": space_tables,
        "space_allowed_columns": space_columns,
    }


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
    gate_kwargs = _consumer_sql_gate_kwargs(choice_port)
    owner = _owner_from_choice_port(choice_port)
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
        fed_space = _federation_space_for_choice_port(choice_port)
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
            _handle_federation_ineligible_plan(
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
            )
            plan_cache_hit = cached_plan_template is not None and federation_plan_matches_template(
                plan,
                cached_plan_template,
                step_fingerprints=step_fps,
                manifest_hash_value=manifest_hash_value,
                member_tuple_hash_value=member_tuple_hash_value,
            )
            member_stores = (
                federation_stores_by_source(
                    owner, member_graphs or {}, space_name=_session_space_name_for_federation(owner, choice_port)
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
                        _federation_gate_kwargs_by_source(
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
        fed_attr = _federation_failure_attribution(fed_prep_outcome)
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
                _persist_template_store(_owner_from_choice_port(choice_port), store)
            clear_federated_turn_state(choice_port)
            return None
        perm_denied = ek == "explain_permission_denied" or is_permission_denied_error(err_text)
        scope_denied = ek in (FailureCategory.ACCESS_POLICY.value, FailureCategory.DENIED_REFERENCE.value)
        if scope_denied or (schema_role == "consumer" and perm_denied):
            note_interactive_turn(
                choice_port, outcome="permission_denied", error=None, sql=None, intent=None, **fed_attr
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
            _persist_template_store(_owner_from_choice_port(choice_port), store)
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
    emit_explain_soft_diagnostics(getattr(gen_out, "explain_soft_diagnostics", 0))
    force_feedback = should_prompt_sql_feedback(store, snap_post.q_norm, gen_out.matched_template)
    exec_dialect = dialect
    exec_schema = schema
    snap_for_exec = snap_post
    is_session = choice_port is not None and isinstance(choice_port, PipelineSession)
    if is_session and choice_port is not None and not choice_port.has_pending_choice():
        raise PipelineSuspended(
            PIPELINE_SUSPEND_ID_EXECUTE,
            _federation_execute_confirm_prompt(gen_out, fed_prep_outcome, fed_manifest),
            _sql_execute_suspend_context(
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
        rows, federated_bundle = _run_sql_execution_for_gen_out(
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
        _handle_federation_partial_failure_interactive(choice_port, owner, exc)
        return None
    except FederationTurnCancelledError as exc:
        _handle_federation_turn_cancelled_interactive(choice_port, owner, exc)
        return None
    except AccessError as exc:
        if permission_denied_detail_logging_enabled():
            debug(f"[main_execution._run_sql_phase_after_intent_confirm] permission denied detail: {exc!r}")
        note_interactive_turn(choice_port, outcome="permission_denied", error=None, sql=None, intent=None)
        clear_federated_turn_state(choice_port)
        return None
    if len(rows) == 0:
        fixed_intent, fixed_rows = try_zero_row_where_remediation(intent, exec_schema, exec_dialect, tmpl_sd)
        if fixed_rows is not None:
            intent = fixed_intent
            rows = fixed_rows
    _offer_sql_feedback_after_execute(
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
        interpretation=interpret_plan,
    )
    if not _check_federation_eligibility_before_confirm(
        intent, schema, store, choice_port, persist_template_learning=persist_template_learning
    ):
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
    """Continue the interactive pipeline after a parsed intent (joins. through feedback)."""
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
        interpret_plan=interpret_plan,
    )


def _complete_interactive_execute(
    ctx: SqlExecuteSuspendContext, choice: str | None, *, choice_port: InteractiveChoicePort | None = None
) -> None:
    """Run deferred execution after the separated execute step, then continue to SQL feedback."""
    tail = ctx.tail
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)
    if choice is None or choice != "y":
        note_interactive_turn(choice_port, outcome="user_declined", error="User declined SQL execution.")
        if persist_tl:
            _persist_template_store(_owner_from_choice_port(choice_port), tail.store)
        clear_federated_turn_state(choice_port)
        return None
    rows = [tuple(r) for r in ctx.rows]
    execution_intent = ctx.execution_intent
    owner = _owner_from_choice_port(choice_port)
    _verify_federation_execute_resume(ctx)
    fed_prep = ctx.federated_prepare
    exec_ctx = _federation_exec_context_from_pairs(ctx.federation_exec_context)
    federated_bundle: FederatedSqlBundle | None = None
    if fed_prep is not None and not rows:
        try:
            rows, federated_bundle = _run_sql_execution_for_gen_out(
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
            _handle_federation_partial_failure_interactive(choice_port, owner, exc)
            return None
        except FederationTurnCancelledError as exc:
            _handle_federation_turn_cancelled_interactive(choice_port, owner, exc)
            return None
        except AccessError as exc:
            if permission_denied_detail_logging_enabled():
                debug(f"[main_execution._complete_interactive_execute] permission denied detail: {exc!r}")
            note_interactive_turn(choice_port, outcome="permission_denied", error=None, sql=None, intent=None)
            clear_federated_turn_state(choice_port)
            return None
    elif not rows:
        if ctx.gen_out.generation_path is GenerationPath.FEDERATION_PLAN:
            note_interactive_turn(
                choice_port, outcome="error", error="Federated prepare outcome missing; cannot execute federation plan."
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
            single_source = _federation_single_source_sql_context(
                owner, execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
            )
            if single_source is not None:
                exec_dialect, exec_schema = single_source
        try:
            rows = _run_pipeline_sql_rows(
                intent=execution_intent, schema=exec_schema, dialect=exec_dialect, tmpl_sd=ctx.tmpl_sd
            )
        except AccessError as exc:
            if permission_denied_detail_logging_enabled():
                debug(f"[main_execution._complete_interactive_execute] permission denied detail: {exc!r}")
            note_interactive_turn(choice_port, outcome="permission_denied", error=None, sql=None, intent=None)
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
            single_source = _federation_single_source_sql_context(
                owner, execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
            )
            if single_source is not None:
                exec_dialect, exec_schema = single_source
        fixed_intent, fixed_rows = try_zero_row_where_remediation(
            execution_intent, exec_schema, exec_dialect, ctx.tmpl_sd
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
    emit_explain_soft_diagnostics(getattr(ctx.gen_out, "explain_soft_diagnostics", 0))
    owner = _owner_from_choice_port(choice_port)
    _offer_sql_feedback_after_execute(
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


def _complete_interactive_sql_feedback(
    ctx: SqlFeedbackSuspendContext, choice: str | None, *, choice_port: InteractiveChoicePort | None = None
) -> None:
    """Apply accept or reject after a deferred final-SQL prompt."""
    tail = ctx.tail
    intent = ctx.execution_intent
    sql = ctx.sql
    rows = [tuple(r) for r in ctx.rows]
    tmpl_sd = ctx.tmpl_sd
    federated_bundle = ctx.federated_bundle
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)
    if choice is None:
        if persist_tl:
            _persist_template_store(_owner_from_choice_port(choice_port), tail.store)
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
                **_federation_result_contract_kwargs(
                    ctx.gen_out, federated_prepare=ctx.federated_prepare, federated_bundle=federated_bundle
                ),
            )
            if df_full is not None:
                owner = _owner_from_choice_port(choice_port)
                art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
                artifacts_dir = str(art) if art is not None else None
                save_result_csv(
                    df_full,
                    output_path=results_csv_output_path(tail.store, artifacts_dir=artifacts_dir),
                )
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
        **_federation_feedback_kwargs(
            _owner_from_choice_port(choice_port),
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
        **_federation_result_contract_kwargs(
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
            sql=_resolved_session_step_sql(
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
            sql=_resolved_session_step_sql(
                sql,
                gen_out=ctx.gen_out,
                federated_bundle=federated_bundle,
                federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
                generation_path=ctx.gen_out.generation_path,
            ),
            rows=row_tuples,
            columns=cols,
            intent=intent,
            federated_bundle=federated_bundle,
            federated_plan=ctx.federated_prepare.plan if ctx.federated_prepare is not None else None,
            generation_path=ctx.gen_out.generation_path,
        )


def _complete_intent_rejection_feedback(
    tail: InteractiveTailSnapshot, feedback: str | None, choice_port: InteractiveChoicePort | None
) -> None:
    """Persist free-text feedback after the user declines an intent."""
    body = (feedback or "").strip() or "user_declined_intent"
    entry = summarize_failure_for_memory(
        question=tail.q_norm,
        intent=tail.intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=tail.schema.effective_structural_hash,
        user_reason=body,
    )
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)
    if persist_tl:
        record_question_feedback(tail.store, tail.q_norm, entry)
        _persist_template_store(_owner_from_choice_port(choice_port), tail.store)
    rb = entry.buckets[0].value if entry.buckets else None
    ctx_ref = getattr(choice_port, "_refinement_ctx", None)
    reason_line = body
    if ctx_ref is not None and refinement_retry_available(ctx_ref):
        ctx_ref.accumulated_reasons.append(reason_line)
        ctx_ref.pending_retry = True
        raise RefinementRetry
    print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=rb)
    note_interactive_turn(choice_port, outcome="user_declined", error="User declined intent confirmation.")


def dispatch_pipeline_resume(session: Any, suspended: PipelineSuspended) -> None:
    """Drive the next pipeline segment after the caller enqueued a. programmatic choice."""
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
            raise PipelineSuspended("empty_choice_queue", "interactive choice queue is empty", None)
        if ch != "y":
            if getattr(payload.intent, "schema_invalid", False):
                print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=None)
            note_interactive_turn(session, outcome="user_declined", error="User declined intent confirmation.")
            if persist_tl:
                _persist_template_store(_owner_from_choice_port(session), payload.store)
            return
        clear_planner_schema_invalid_after_user_accept(payload.intent)
        _run_interactive_after_parsed_intent_from_tail(
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
        _complete_interactive_execute(payload, ch, choice_port=session)
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


def _owner_has_federation(owner: Any | None) -> bool:
    return owner is not None and getattr(owner, "_federation_manifest", None) is not None


def _raise_if_session_turn_cancelled() -> None:
    if session_turn_cancelled():
        raise SessionTurnCancelledError("Turn cancelled.")


def interactive_run_once(
    schema: SchemaGraph | None = None,
    store: dict[str, Any] | TemplateStoreView | None = None,
    templates: dict[str, Any] | None = None,
    rejected: dict[str, Any] | None = None,
    schema_terms: Any | None = None,
    question: str | None = None,
    pipeline_session: Any | None = None,
) -> dict[str, Any] | None:
    """Execute a single interactive pipeline iteration. Reads a. question. from stdin or uses the supplied `question`, validates it, checks for template reuse, parses intent via LLM if needed, generates SQL, executes it, and handles user feedback."""
    if question is None:
        notify("Enter question", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
        try:
            question = prompt("").strip()
        except (EOFError, KeyboardInterrupt):
            terminated()
            return None

    if not question:
        if pipeline_session is not None:
            note_interactive_turn(
                choice_port=pipeline_session, outcome="parse_failed", error="Question must not be empty."
            )
            return None
        invalid_input()
        return None
    _raise_if_session_turn_cancelled()
    progress("\nValidating question...")

    raw_question = question

    owner_dialect = None
    owner = getattr(pipeline_session, "_owner", None) if pipeline_session is not None else None
    if owner is not None:
        owner_dialect = getattr(owner, "_dialect", None)
    fed_reuse_kwargs = _federation_reuse_kwargs(owner, pipeline_session)

    dialect, schema, store, templates, rejected, schema_terms = load_pipeline_resources(
        schema, store, templates, rejected, schema_terms, dialect=owner_dialect
    )
    choice_port: InteractiveChoicePort | None = pipeline_session
    persist_tl = _persist_template_learning_for_pipeline_session(choice_port)
    gate_kwargs = _consumer_sql_gate_kwargs(choice_port)

    tmpl_pre = match_question_level_template_reuse(raw_question, templates, template_store=store)
    if tmpl_pre.reuse_type == "direct_reuse" and not _owner_has_federation(owner):
        best_template_pre = tmpl_pre.best_template
        if best_template_pre is None:
            return None
        debug(f"direct SQL reuse via question match pre-validation (trust>=1, template='{best_template_pre.id}')")
        debug("[main_execution.interactive_run_once] direct_reuse_pre: question_match")
        assert tmpl_pre.reuse_candidate_normalized is not None
        reuse_pre = handle_direct_sql_reuse(
            tmpl_pre.reuse_candidate_normalized,
            best_template_pre,
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
            **gate_kwargs,
            **fed_reuse_kwargs,
        )
        if reuse_pre is not None and reuse_pre.success:
            return None

    valid, query_type, corrected = validate_question(raw_question)
    if not valid:
        if query_type == "restricted":
            print_rephrase_hint(RephraseHint.RESTRICTED_QUESTION)
            note_interactive_turn(choice_port, outcome="restricted", error="Question rejected as restricted.")
        else:
            print_rephrase_hint(RephraseHint.VAGUE_QUESTION)
            note_interactive_turn(choice_port, outcome="invalid_question", error="Question failed validation.")
        return None
    corrected_text = corrected
    if corrected_text != raw_question:
        debug(f"[main_execution.interactive_run_once] typo_corrected: '{raw_question}' -> '{corrected_text}'")

    tmpl_typo = match_question_level_template_reuse(corrected_text, templates, template_store=store)
    if tmpl_typo.reuse_type == "direct_reuse" and not _owner_has_federation(owner):
        best_template_typo = tmpl_typo.best_template
        if best_template_typo is None:
            return None
        debug(f"direct SQL reuse via question match (trust>=1, template='{best_template_typo.id}')")
        debug("[main_execution.interactive_run_once] direct_reuse: question_match")
        assert tmpl_typo.reuse_candidate_normalized is not None
        reuse_result = handle_direct_sql_reuse(
            tmpl_typo.reuse_candidate_normalized,
            best_template_typo,
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
            **gate_kwargs,
            **fed_reuse_kwargs,
        )
        if reuse_result is not None and reuse_result.success:
            return None

    neg_drop = False
    normalized_canonical = normalize_question_via_llm(corrected_text, raw_original=raw_question)
    if normalized_canonical != corrected_text and has_any_rejection_history_for_question(store, corrected_text):
        debug(
            f"[main_execution.interactive_run_once] dropped_normalized_due_to_negative_memory {normalized_canonical!r}"
        )
        neg_drop = True
        normalized_canonical = corrected_text

    tmpl_norm = None
    if normalized_canonical != corrected_text:
        tmpl_norm = match_question_level_template_reuse(normalized_canonical, templates, template_store=store)
        if tmpl_norm.reuse_type == "direct_reuse" and not _owner_has_federation(owner):
            best_template_norm = tmpl_norm.best_template
            if best_template_norm is None:
                return None
            debug(f"direct SQL reuse via normalized question match (trust>=1, template='{best_template_norm.id}')")
            assert tmpl_norm.reuse_candidate_normalized is not None
            fs_norm = QuestionFormStorage(
                corrected=corrected_text,
                normalized_optional=normalized_canonical,
                normalized_negative_memory_dropped=neg_drop,
                accept_via_normalized_lookup_only=True,
            )
            reuse_norm = handle_direct_sql_reuse(
                tmpl_norm.reuse_candidate_normalized,
                best_template_norm,
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
                **gate_kwargs,
                **fed_reuse_kwargs,
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

    if _owner_has_federation(owner):
        fed_kwargs = _federation_reuse_kwargs(owner, choice_port)
        intake_reuse = try_federation_plan_intake_reuse(
            q_norm,
            schema,
            dialect,
            federation_dir=fed_kwargs.get("federation_dir"),
            federation_manifest=fed_kwargs.get("federation_manifest"),
            federation_mappings=fed_kwargs.get("federation_mappings"),
            stores_by_source=fed_kwargs.get("stores_by_source"),
            dialects_by_source=fed_kwargs.get("dialects_by_source"),
            source_runtimes=fed_kwargs.get("source_runtimes"),
            member_graphs=fed_kwargs.get("member_graphs"),
            gate_kwargs_by_source=fed_kwargs.get("gate_kwargs_by_source"),
        )
        if intake_reuse is not None and intake_reuse.success:
            return None

    conv_hints: tuple[str, ...] = ()
    if pipeline_session is not None:
        raw_h = getattr(pipeline_session, "_pending_conversation_rejection_hints", None)
        if isinstance(raw_h, tuple):
            conv_hints = raw_h
            pipeline_session._pending_conversation_rejection_hints = ()

    refinement_ctx = RefinementContext(corrected_text, form_storage, conversation_rejection_hints=conv_hints)
    _interactive_attach_refinement_ctx(choice_port, refinement_ctx)

    while True:
        _raise_if_session_turn_cancelled()
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
    return None


def get_seed_warmup_summary_from_dir(artifacts_dir: str, version: int) -> SeedWarmupSummary:
    """Build a ``SeedWarmupSummary`` from a persisted ``seed_warmup_report_v{version}.json`` file."""
    report_path = os.path.join(artifacts_dir, SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version))
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
            report.get("unique_prompts", report.get("synthetic_runnable_count", report.get("unique_synthetic", 0)))
        ),
        gold_new=int(report.get("gold_new", 0)),
        gold_skipped=int(report.get("gold_skipped", 0)),
        gold_failed=int(report.get("gold_failed", 0)),
        gold_user_rejected=int(report.get("gold_user_rejected", 0)),
        deduped_prompts_count=int(
            report.get(
                "deduped_prompts_count",
                report.get("synthetic_unique_body_keys", report.get("deduped_synthetic_count", 0)),
            )
        ),
        gold_prompts_count=int(report.get("gold_prompts_count", report.get("seed_questions_loaded", 0))),
        templates_added=int(report.get("templates_added", 0)),
        validation_drop=int(report.get("validation_drop", 0)),
        realism_drop=int(report.get("realism_drop", 0)),
        question_generation_failed=int(report.get("question_generation_failed", 0)),
        early_pipeline_failed=int(report.get("early_pipeline_failed", 0)),
    )


def _toml_claim_put_scalar(
    block: dict[str, Any], subkey: str, target_key: str, output: dict[str, str], claimed: set[str]
) -> None:
    if subkey not in block:
        return
    claimed.add(target_key)
    raw_value = block.get(subkey)
    if raw_value is None:
        return
    text = str(raw_value).strip()
    if text:
        output[target_key] = text


def _toml_claim_put_csv_files(block: dict[str, Any], output: dict[str, str], claimed: set[str]) -> None:
    files_raw = block.get("files")
    if files_raw is None:
        return
    claimed.add("CSV_FILES")
    if isinstance(files_raw, list):
        parts = [str(item).strip() for item in files_raw if str(item).strip()]
        if parts:
            output["CSV_FILES"] = ",".join(parts)
    else:
        text = str(files_raw).strip()
        if text:
            output["CSV_FILES"] = text


def _flatten_scalar_engine_fields(
    block: dict[str, Any],
    field_specs: tuple[tuple[str, str], ...],
    output: dict[str, str],
    claimed: set[str],
    *,
    section_name: str,
) -> None:
    for subkey, target_key in field_specs:
        _toml_claim_put_scalar(block, subkey, target_key, output, claimed)
    if section_name in {"csv", "excel"}:
        _toml_claim_put_csv_files(block, output, claimed)


def _flatten_engine_block(
    section_name: str,
    block: dict[str, Any],
    field_specs: tuple[tuple[str, str], ...],
    connection_name: str | None = None,
) -> tuple[dict[str, str], set[str], frozenset[str]]:
    """Flatten one engine TOML block to env-style keys. Scalar keys define a single unnamed connection. Nested dicts define named connections; when only sub-tables are present there is no unnamed default."""
    named_blocks = {key: value for key, value in block.items() if isinstance(value, dict)}
    scalar_keys = {key for key in block if not isinstance(block.get(key), dict)}
    if named_blocks and scalar_keys:
        raise ConfigError(
            f"config_file [{section_name}] mixes scalar keys with named connection sub-tables; "
            "use either a flat block or named sub-tables, not both."
        )
    output: dict[str, str] = {}
    claimed: set[str] = set()
    if not named_blocks:
        _flatten_scalar_engine_fields(block, field_specs, output, claimed, section_name=section_name)
        return output, claimed, frozenset()
    connection_names = frozenset(str(name) for name in named_blocks)
    selected = connection_name
    if selected is None and len(named_blocks) == 1:
        selected = next(iter(named_blocks))
    if selected is None:
        return output, claimed, connection_names
    if selected not in named_blocks:
        options = ", ".join(sorted(connection_names))
        raise ConfigError(f"config_file [{section_name}] has no connection {selected!r}; expected one of: {options}.")
    _flatten_scalar_engine_fields(named_blocks[selected], field_specs, output, claimed, section_name=section_name)
    return output, claimed, connection_names


def _select_connection_name(
    env: Mapping[str, str],
    named_connections_by_engine: Mapping[str, frozenset[str]],
    engine: str,
    *,
    explicit_connection: str | None = None,
) -> str | None:
    """Resolve the named connection handle for *engine*, if any."""
    names = named_connections_by_engine.get(engine, frozenset())
    if not names:
        return None
    explicit = str(explicit_connection or env.get("AETHERDIALECT_CONNECTION", "") or "").strip()
    if explicit:
        if explicit not in names:
            options = ", ".join(sorted(names))
            raise ConfigError(
                f"Unknown AETHERDIALECT_CONNECTION {explicit!r} for {engine}; expected one of: {options}."
            )
        return explicit
    if len(names) == 1:
        return next(iter(names))
    options = ", ".join(sorted(names))
    raise ConfigError(
        f"Multiple named connections configured for {engine} ({options}); "
        "set AETHERDIALECT_CONNECTION or pass connection= to AetherEngine."
    )


def _load_config_file(
    path: str | os.PathLike[str] | None, *, connection: str | None = None
) -> tuple[dict[str, str], frozenset[str], dict[str, frozenset[str]]]:
    """Parse a TOML configuration file into flat environment-style string keys."""
    if path is None:
        return {}, frozenset(), {}
    path_str = str(path).strip()
    if not path_str:
        return {}, frozenset(), {}
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
    named_connections_by_engine: dict[str, frozenset[str]] = {}

    def _claim_put(block: dict[str, Any], subkey: str, target_key: str) -> None:
        _toml_claim_put_scalar(block, subkey, target_key, output, claimed)

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
            _claim_put(deployments_block, "heavy", "AZURE_OPENAI_DEPLOYMENT_HEAVY")
    for section_name, field_specs in TOML_ENGINE_FIELD_MAPS.items():
        engine_block = document.get(section_name)
        if not isinstance(engine_block, dict):
            continue
        if section_name == "excel":
            field_specs = TOML_ENGINE_FIELD_MAPS["csv"]
        flat, section_claimed, named = _flatten_engine_block(section_name, engine_block, field_specs, connection)
        output.update(flat)
        claimed.update(section_claimed)
        if named:
            engine_name = TOML_SECTION_TO_ENGINE[section_name]
            existing = named_connections_by_engine.get(engine_name, frozenset())
            named_connections_by_engine[engine_name] = existing | named
    engine_block = document.get("engine")
    if isinstance(engine_block, dict):
        _claim_put(engine_block, "selected", "AETHERDIALECT_ENGINE")
        _claim_put(engine_block, "connection", "AETHERDIALECT_CONNECTION")
    llm_block = document.get("llm")
    if isinstance(llm_block, dict):
        _claim_put(llm_block, "provider", "AETHERDIALECT_LLM_PROVIDER")
    mock_block = document.get("mock")
    if isinstance(mock_block, dict):
        _claim_put(mock_block, "fixtures_file", "AETHERDIALECT_MOCK_FIXTURES_FILE")
    execution_block = document.get("execution")
    if isinstance(execution_block, dict):
        _claim_put(execution_block, "max_query_cost_rows", "AETHERDIALECT_MAX_QUERY_COST_ROWS")
        _claim_put(execution_block, "max_query_cost_bytes", "AETHERDIALECT_MAX_QUERY_COST_BYTES")
        _claim_put(execution_block, "statement_timeout_ms", "AETHERDIALECT_STATEMENT_TIMEOUT_MS")
        _claim_put(execution_block, "llm_timeout_ms", "AETHERDIALECT_LLM_TIMEOUT_MS")
        _claim_put(execution_block, "profile_timeout_ms", "AETHERDIALECT_PROFILE_TIMEOUT_MS")
        _claim_put(execution_block, "explain_timeout_ms", "AETHERDIALECT_EXPLAIN_TIMEOUT_MS")
    return output, frozenset(claimed), named_connections_by_engine


def _merge_configuration_environment(
    config_file_values: Mapping[str, str], *, toml_claimed_keys: frozenset[str] | None = None
) -> tuple[dict[str, str], frozenset[str]]:
    """Build the effective environment mapping used for engine. configuration reads. When *toml_claimed_keys* is ``None`` (no ``config_file`` in use), non-empty TOML values overlay ``os.environ`` for matching keys only. When *toml_claimed_keys* is provided (a ``config_file`` was loaded), the file is the single source of truth for every key in that set: non-empty flattened values replace ``os.environ``, and keys present in the file with empty or absent string values remove the variable from the effective mapping so environment defaults cannot leak past an explicit TOML field. This function never mutates ``os.environ``."""
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


def compute_connection_storage_slug(engine: str) -> str:
    """Return a stable connection slug derived from the active engine runtime configuration. When the composed slug is longer than :data:`ENGINE_STORAGE_SLUG_MAX_CHARS`, a deterministic hash suffix is used instead."""
    runtime_cls = cast(type[EngineRuntimeConfig], get_runtime_config_class(engine))
    fields = runtime_cls.connection_slug_fields()
    parts = [_engine_storage_slug_fragment(fields[key], fallback=key[0]) for key in runtime_cls.connection_slug_keys()]
    slug = f"conn_{engine}_" + "_".join(parts)
    if len(slug) > int(ENGINE_STORAGE_SLUG_MAX_CHARS):
        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]
        return f"conn_{engine}_{digest}"
    return slug


def compute_engine_storage_dir(artifacts_root: str | None, engine: str, *, tenant_slug: str | None = None) -> str:
    """Return the absolute engine storage directory for persisted artifacts. When *artifacts_root* is ``None`` or blank, the parent directory is ``platformdirs.user_data_dir("aetherdialect")``. When *artifacts_root* is provided, the parent directory is its absolute expanded path. The final directory is ``os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, connection_slug)``; when *tenant_slug* is provided, a sanitized tenant segment is inserted before *connection_slug*."""
    parent = (
        os.path.abspath(os.path.expanduser(str(artifacts_root)))
        if artifacts_root and str(artifacts_root).strip()
        else user_data_dir(appname="aetherdialect", appauthor=False)
    )
    slug = compute_connection_storage_slug(engine)
    if tenant_slug is not None and str(tenant_slug).strip():
        tenant_segment = _sanitize_tenant_slug(tenant_slug)
        return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, tenant_segment, slug)
    return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, slug)


def _prepare_schema_context_for_init(
    schema_context: EngineContext, engine_storage_dir: str, sink: Callable[[str], None]
) -> EngineContext:
    """Merge an explicit ``EngineContext`` with any compatible on-disk. cache under *engine_storage_dir*."""
    try:
        cached = load_schema_context_cache(engine_storage_dir)
    except ConfigError as exc:
        sink(str(exc))
        cached = None
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
        return EngineContext(
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
    return env_first_nonempty(env, *keys)


def _env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    """True when at least one key maps to a non-blank string."""
    return env_any_nonempty(env, keys)


def _env_role_hint(label: str, keys: tuple[str, ...]) -> str:
    return env_role_hint(label, keys)


def _runtime_config_for_engine(engine: str) -> type[EngineRuntimeConfig]:
    return cast(type[EngineRuntimeConfig], get_runtime_config_class(engine))


def _apply_runtime_environments(env: Mapping[str, str]) -> None:
    """Load every registered runtime config whose partial env scope is present."""
    PolicyConfig.apply_environment(env)
    for engine in list_engines():
        runtime_cls = _runtime_config_for_engine(engine)
        if runtime_cls.should_apply_environment(env):
            runtime_cls.apply_environment(env)


def _select_engine_name(
    env: Mapping[str, str], named_connections_by_engine: Mapping[str, frozenset[str]] | None = None
) -> str:
    named = named_connections_by_engine or {}
    engines = list_engines()
    explicit = str(env.get("AETHERDIALECT_ENGINE", "") or "").strip().lower()
    if explicit:
        if explicit not in engines:
            raise ConfigError(f"Unsupported AETHERDIALECT_ENGINE: {explicit!r}. Expected one of {engines}.")
        blockers = _runtime_config_for_engine(explicit).selection_blockers(env)
        if blockers and not named.get(explicit):
            raise ConfigError(f"Cannot select {explicit} engine: {'; '.join(blockers)}")
        return explicit
    ready: list[str] = []
    for engine in engines:
        if not _runtime_config_for_engine(engine).selection_blockers(env):
            ready.append(engine)
        elif named.get(engine):
            ready.append(engine)
    if len(ready) > 1:
        labels = ", ".join(ready)
        raise ConfigError(
            f"Multiple database engines are configured and available ({labels}); set AETHERDIALECT_ENGINE "
            "or [engine] selected in the config file to one of them."
        )
    if len(ready) == 1:
        return ready[0]
    missing: list[str] = []
    for engine in engines:
        missing.extend(_runtime_config_for_engine(engine).selection_blockers(env))
    raise ConfigError("Cannot select database engine: " + "; ".join(missing))


def _activate_engine(name: str) -> None:
    """Bind :attr:`EngineConfig.TYPE` and :attr:`EngineConfig.RUNTIME` to the chosen engine. Pre-condition: the corresponding runtime loader has already populated the runtime config. This function performs no env reads."""
    if name not in list_engines():
        raise ConfigError(f"Unsupported engine activation: {name!r}.")
    EngineConfig.TYPE = name
    EngineConfig.RUNTIME = _runtime_config_for_engine(name)


def configure_runtime_from_environment(engine_context: EngineContext, merged_env: Mapping[str, str]) -> str:
    env: dict[str, str] = dict(merged_env)
    selected = _select_engine_name(env)
    _apply_runtime_environments(env)
    _activate_engine(selected)
    if selected == "databricks" and not DatabricksRuntimeConfig.has_native_connection():
        if not DatabricksRuntimeConfig.pyspark_session_reachable():
            raise ConfigError(
                "Databricks requires either all SQL warehouse connection variables or an active PySpark session."
            )
    _configure_llm_from_environment(env)
    if selected not in list_engines():
        raise ConfigError(f"Unsupported engine resolved: {selected!r}")
    return selected


def _configure_openai_from_environment(env: Mapping[str, str]) -> None:
    """Populate :class:`EngineConfig` with OpenAI credentials and clear Azure fields."""
    EngineConfig.LLM_PROVIDER = "openai"
    EngineConfig.API_TOKEN = str(env["OPENAI_API_KEY"]).strip()
    EngineConfig.AZURE_API_TOKEN = None
    EngineConfig.OPENAI_MODEL = "gpt-4.1-nano"
    EngineConfig.OPENAI_MODEL_INTENT = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_JOIN = "gpt-5.4-nano"
    EngineConfig.OPENAI_MODEL_SCHEMA = "gpt-5-mini"
    EngineConfig.OPENAI_MODEL_SCHEMA_BASE = "gpt-4.1-mini"
    EngineConfig.OPENAI_MODEL_DDL = "gpt-4.1-nano"
    EngineConfig.OPENAI_MODEL_SYNTH = "gpt-5-mini"
    EngineConfig.OPENAI_MODEL_SYNTH_VARIETY = "gpt-5-nano"
    EngineConfig.OPENAI_MODEL_INTENT_FORMAT = "gpt-4.1-nano"
    EngineConfig.OPENAI_MODEL_INTENT_SCHEMA_REPAIR = "gpt-5.4-nano"
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
    EngineConfig.OPENAI_MODEL = "gpt-4.1-nano"
    EngineConfig.OPENAI_MODEL_INTENT = "gpt-5.4-mini"
    EngineConfig.OPENAI_MODEL_JOIN = "gpt-5.4-nano"
    EngineConfig.OPENAI_MODEL_SCHEMA = "gpt-5-mini"
    EngineConfig.OPENAI_MODEL_SCHEMA_BASE = "gpt-4.1-mini"
    EngineConfig.OPENAI_MODEL_DDL = "gpt-4.1-nano"
    EngineConfig.OPENAI_MODEL_SYNTH = "gpt-5-mini"
    EngineConfig.OPENAI_MODEL_SYNTH_VARIETY = "gpt-5-nano"
    EngineConfig.OPENAI_MODEL_INTENT_FORMAT = "gpt-4.1-nano"
    EngineConfig.OPENAI_MODEL_INTENT_SCHEMA_REPAIR = "gpt-5.4-nano"


def _openai_direct_env_complete(env: Mapping[str, str]) -> bool:
    return _env_any_nonempty(env, ("OPENAI_API_KEY",))


def _configure_mock_from_environment(env: Mapping[str, str]) -> None:
    """Bind mock LLM replay from a fixtures JSON file."""
    path = str(env.get("AETHERDIALECT_MOCK_FIXTURES_FILE", "") or "").strip()
    if not path:
        raise ConfigError(
            "Mock LLM requires AETHERDIALECT_MOCK_FIXTURES_FILE or [mock] fixtures_file in the config file."
        )
    EngineConfig.LLM_PROVIDER = "mock"
    EngineConfig.MOCK_FIXTURES_FILE = path
    EngineConfig.API_TOKEN = None
    EngineConfig.AZURE_API_TOKEN = None


def _configure_llm_from_environment(env: Mapping[str, str]) -> None:
    explicit = str(env.get("AETHERDIALECT_LLM_PROVIDER", "") or "").strip().lower()
    if explicit == "mock":
        _configure_mock_from_environment(env)
        clear_llm_clients()
        reset_mock_provider()
        return
    openai_ready = _openai_direct_env_complete(env)
    azure_ready = _env_all_non_empty(env, AZURE_OPENAI_ENV_REQUIRED)
    if not (openai_ready or azure_ready):
        raise ConfigError(
            "LLM is not configured. Set "
            + ", ".join(OPENAI_ENV_REQUIRED)
            + " for OpenAI, or "
            + ", ".join(AZURE_OPENAI_ENV_REQUIRED)
            + " for Azure OpenAI."
        )
    if explicit:
        if explicit not in ("openai", "azure"):
            raise ConfigError(
                f"Unsupported AETHERDIALECT_LLM_PROVIDER: {explicit!r}. Expected 'openai', 'azure', or 'mock'."
            )
        if explicit == "openai":
            if not openai_ready:
                raise ConfigError("AETHERDIALECT_LLM_PROVIDER is 'openai' but the OpenAI environment is incomplete.")
            _configure_openai_from_environment(env)
        else:
            if not azure_ready:
                raise ConfigError(
                    "AETHERDIALECT_LLM_PROVIDER is 'azure' but the Azure OpenAI environment is incomplete."
                )
            _configure_azure_from_environment(env)
        return
    if openai_ready and azure_ready:
        raise ConfigError(
            "Both OpenAI and Azure OpenAI credentials are available; set AETHERDIALECT_LLM_PROVIDER "
            "or [llm] provider in the config file to 'openai' or 'azure'."
        )
    if openai_ready:
        _configure_openai_from_environment(env)
        return
    if azure_ready:
        _configure_azure_from_environment(env)
        return
    raise ConfigError("LLM is not configured.")


def _federation_artifacts_root(owner: Any) -> str | None:
    """Resolve the artifacts parent directory that holds member ``conn_*`` trees."""
    root = getattr(owner, "_artifacts_root", None)
    if root is not None:
        return str(root)
    fed_dir = getattr(owner, "_federation_storage_dir", None)
    if fed_dir:
        return str(Path(fed_dir).parent)
    adir = getattr(owner, "_artifacts_dir", None)
    if adir is not None:
        return str(Path(adir).parent)
    return None


def _session_space_name_for_federation(owner: Any, choice_port: InteractiveChoicePort | None) -> str:
    """Return the active AetherSpace name for federated per-source template learning."""
    if choice_port is not None:
        sn = getattr(choice_port, "space_name", None)
        if callable(sn):
            value = sn()
            if value:
                return str(value)
        elif isinstance(sn, str) and sn.strip():
            return sn.strip()
    return str(getattr(owner, "_context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)


def _federation_gate_kwargs_by_source(
    owner: Any,
    choice_port: InteractiveChoicePort | None,
    manifest: FederationManifest,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compose per-source execution gate kwargs from manifest bindings."""
    engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
    allowed_where_ops = intersect_member_where_ops(dialects_by_source, engine_types_by_source=engine_types)
    gates: dict[str, dict[str, Any]] = {}
    runtimes = getattr(owner, "_federation_source_runtimes", None) or {}
    artifacts_root = _federation_artifacts_root(owner)
    for binding in manifest.sources:
        base = dict(_consumer_sql_gate_kwargs(choice_port))
        base["schema_role"] = binding.role
        member_ops = extra_where_ops_for_engine(binding.engine)
        base["allowed_where_ops"] = allowed_where_ops & (member_ops | set(FEDERATION_BASE_WHERE_OPS))
        if binding.context and binding.context != "master":
            runtime = runtimes.get(binding.source_id)
            member_dir = (
                str(runtime.artifacts_dir)
                if runtime is not None and runtime.artifacts_dir
                else federation_source_artifacts_dir(artifacts_root, binding)
            )
            named = load_named_schema_context(member_dir, binding.context)
            if named is None:
                raise ConfigError(
                    f"federation source {binding.source_id!r} declared context {binding.context!r} "
                    f"not found in member artifacts at {member_dir}"
                )
            base["schema_context"] = named
            base["context_name"] = binding.context
        else:
            base["schema_context"] = EngineContext()
            base["context_name"] = MASTER_AETHERSPACE_NAME
        gates[binding.source_id] = base
    return gates


def _federation_reuse_kwargs(owner: Any | None, choice_port: InteractiveChoicePort | None) -> dict[str, Any]:
    """Optional federation context for question-level reuse paths."""
    if owner is None or getattr(owner, "_federation_manifest", None) is None:
        return {}
    manifest = getattr(owner, "_federation_manifest", None)
    member_graphs = getattr(owner, "_federation_member_graphs", None)
    stores_by_source: dict[str, TemplateStoreView] = {}
    gate_kwargs_by_source: dict[str, dict[str, Any]] | None = None
    if isinstance(member_graphs, dict) and member_graphs:
        stores_by_source = federation_stores_by_source(
            owner, member_graphs, space_name=_session_space_name_for_federation(owner, choice_port)
        )
        if manifest is not None:
            gate_kwargs_by_source = _federation_gate_kwargs_by_source(
                owner, choice_port, manifest, getattr(owner, "_federation_dialects", None)
            )
    return {
        "federation_dir": getattr(owner, "_federation_storage_dir", None),
        "federation_manifest": manifest,
        "federation_mappings": getattr(owner, "_federation_mappings", None),
        "stores_by_source": stores_by_source or None,
        "dialects_by_source": getattr(owner, "_federation_dialects", None),
        "source_runtimes": getattr(owner, "_federation_source_runtimes", None),
        "member_graphs": member_graphs if isinstance(member_graphs, dict) else None,
        "gate_kwargs_by_source": gate_kwargs_by_source,
    }


def _federation_turn_active(
    *,
    gen_out: SqlGenerationOutcome | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
    federated_plan: FederatedPlan | None = None,
    generation_path: GenerationPath | None = None,
) -> bool:
    return (
        generation_path is GenerationPath.FEDERATION_PLAN
        or (gen_out is not None and gen_out.generation_path is GenerationPath.FEDERATION_PLAN)
        or federated_bundle is not None
        or federated_plan is not None
    )


def _federation_session_step_sql(
    gen_out: SqlGenerationOutcome | None = None,
    *,
    federated_bundle: FederatedSqlBundle | None = None,
    federated_plan: FederatedPlan | None = None,
    generation_path: GenerationPath | None = None,
) -> str | None:
    """Return the single member SQL for a federated turn, or None when not singular."""
    if not _federation_turn_active(
        gen_out=gen_out,
        federated_bundle=federated_bundle,
        federated_plan=federated_plan,
        generation_path=generation_path,
    ):
        return None
    if federated_bundle is not None and len(federated_bundle.statements) == 1:
        statement = str(federated_bundle.statements[0].statement or "").strip()
        return statement or None
    return None


def _resolved_session_step_sql(
    sql: str | None,
    *,
    gen_out: SqlGenerationOutcome | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
    federated_plan: FederatedPlan | None = None,
    generation_path: GenerationPath | None = None,
) -> str | None:
    """Resolve ``SessionStep.sql`` for single-engine and federated turns."""
    fed_sql = _federation_session_step_sql(
        gen_out,
        federated_bundle=federated_bundle,
        federated_plan=federated_plan,
        generation_path=generation_path,
    )
    if fed_sql is not None:
        return fed_sql
    if _federation_turn_active(
        gen_out=gen_out,
        federated_bundle=federated_bundle,
        federated_plan=federated_plan,
        generation_path=generation_path,
    ):
        return None
    return sql


def _federation_result_contract_kwargs(
    gen_out: SqlGenerationOutcome,
    *,
    federated_prepare: FederatedPrepareOutcome | None = None,
    federated_bundle: FederatedSqlBundle | None = None,
) -> dict[str, Any]:
    """Column and shape kwargs derived from a federated plan rather than display SQL."""
    if gen_out.generation_path is not GenerationPath.FEDERATION_PLAN:
        return {}
    prep = federated_prepare
    plan = prep.plan if prep is not None else None
    kwargs: dict[str, Any] = {
        "generation_path": gen_out.generation_path,
    }
    if plan is not None:
        kwargs["federated_plan"] = plan
    if federated_bundle is not None:
        kwargs["federated_bundle"] = federated_bundle
    column_names: Sequence[str] | None = None
    if federated_bundle is not None and federated_bundle.column_names:
        column_names = federated_bundle.column_names
    elif prep is not None and prep.bundle is not None and prep.bundle.column_names:
        column_names = prep.bundle.column_names
    elif plan is not None:
        residual = federation_residual_column_headers(plan)
        if residual:
            column_names = residual
    if column_names:
        kwargs["column_names"] = column_names
    return kwargs


def _federation_contract_kwargs_from_snap(snap: Mapping[str, Any]) -> dict[str, Any]:
    """Derive federation column contract kwargs stored on a completed turn snapshot."""
    federated_bundle = snap.get("federated_bundle")
    federated_plan = snap.get("federated_plan")
    generation_path = snap.get("generation_path")
    if federated_bundle is None and federated_plan is None and generation_path is not GenerationPath.FEDERATION_PLAN:
        return {}
    kwargs: dict[str, Any] = {"generation_path": GenerationPath.FEDERATION_PLAN}
    if federated_plan is not None:
        kwargs["federated_plan"] = federated_plan
    if federated_bundle is not None:
        kwargs["federated_bundle"] = federated_bundle
    column_names: Sequence[str] | None = None
    if federated_bundle is not None and getattr(federated_bundle, "column_names", None):
        column_names = federated_bundle.column_names
    elif federated_plan is not None:
        residual = federation_residual_column_headers(federated_plan)
        if residual:
            column_names = residual
    if column_names:
        kwargs["column_names"] = column_names
    return kwargs


def _federation_feedback_kwargs(
    owner: Any | None,
    gen_out: SqlGenerationOutcome,
    choice_port: InteractiveChoicePort | None = None,
    *,
    federated_prepare: FederatedPrepareOutcome | None = None,
) -> dict[str, Any]:
    """Build optional federation accept kwargs for :func:`handle_user_feedback`."""
    if gen_out.generation_path is not GenerationPath.FEDERATION_PLAN:
        return {}
    member_graphs = getattr(owner, "_federation_member_graphs", None) if owner is not None else None
    stores_by_source: dict[str, TemplateStoreView] = {}
    schemas_by_source: dict[str, SchemaGraph] = {}
    if owner is not None and isinstance(member_graphs, dict):
        stores_by_source = federation_stores_by_source(
            owner, member_graphs, space_name=_session_space_name_for_federation(owner, choice_port)
        )
        schemas_by_source = dict(member_graphs)
    federated_plan = federated_prepare.plan if federated_prepare is not None else None
    pending_plan_template = None
    if choice_port is not None:
        pending_plan_template = getattr(choice_port, "_pending_federation_plan_template", None)
    return {
        "federated_steps": tuple(gen_out.federated_steps),
        "federation_dir": gen_out.federation_dir,
        "federation_plan_id": gen_out.federation_plan_id,
        "stores_by_source": stores_by_source,
        "schemas_by_source": schemas_by_source,
        "federated_plan": federated_plan,
        "pending_plan_template": pending_plan_template,
    }


def federation_stores_by_source(
    owner: Any, member_graphs: Mapping[str, SchemaGraph], *, space_name: str = MASTER_AETHERSPACE_NAME
) -> dict[str, TemplateStoreView]:
    """Load per-source template stores from federation member artifact trees."""
    runtimes = getattr(owner, "_federation_source_runtimes", None) or {}
    stores: dict[str, TemplateStoreView] = {}
    for source_id, graph in member_graphs.items():
        runtime = runtimes.get(source_id)
        if runtime is None or not getattr(runtime, "artifacts_dir", None):
            raise FederationConfigError(
                f"federation member store missing for source_id {source_id!r}; "
                "each member must have its own artifact tree"
            )
        artifacts_dir = str(runtime.artifacts_dir)
        graph_id = str(graph.schema_graph_id or "")
        stores[source_id] = load_template_store(graph_id, graph, artifacts_dir=artifacts_dir, space_name=space_name)
    return stores


def _federation_duckdb_schema_for_connection(connection: str) -> str:
    """Map a federation source connection label to the DuckDB schema used for qualification."""
    conn = str(connection or "").strip().lower()
    if conn in {"", "memory", "main", "storefront"}:
        return "main"
    return conn


def _duckdb_runtime_config_for_schema(base_cls: type[EngineRuntimeConfig], schema: str) -> type[EngineRuntimeConfig]:
    """Return a DuckDB runtime config class with ``SCHEMA`` pinned to *schema*."""
    if schema == "main":
        return base_cls
    return cast(type[EngineRuntimeConfig], type(f"_FederationDuckDBSchema_{schema}", (base_cls,), {"SCHEMA": schema}))


def _build_federation_source_runtimes(
    manifest: FederationManifest,
    artifacts_root: str | None,
    default_dialect: Any,
    *,
    default_identity: EngineIdentity | None = None,
    native_connection: Any = None,
    sqlalchemy_engine: Any = None,
    engines_by_source: Mapping[str, Any] | None = None,
    native_connections_by_source: Mapping[str, Any] | None = None,
    existing_runtimes: Mapping[str, SourceRuntime] | None = None,
    members_by_source: Mapping[str, Any] | None = None,
) -> dict[str, SourceRuntime]:
    """Bind per-source dialect handles for federated SQL generation and execution."""
    runtimes: dict[str, SourceRuntime] = {}
    fallback_identity = default_identity or active_engine_identity()
    sa_by_source = dict(engines_by_source or {})
    native_by_source = dict(native_connections_by_source or {})
    prior_runtimes = dict(existing_runtimes or {})
    members = dict(members_by_source or {})
    for binding in manifest.sources:
        adir = federation_source_artifacts_dir(artifacts_root, binding)
        engine_type = str(binding.engine or fallback_identity.engine_type).strip().lower()
        member_engine = members.get(binding.source_id)
        member_dialect = getattr(member_engine, "_dialect", None) if member_engine is not None else None
        if member_dialect is not None:
            schema_path = os.path.join(adir, "schema_graph.json.gz")
            if hasattr(member_dialect, "_schema_json_path"):
                member_dialect._schema_json_path = schema_path
            runtimes[binding.source_id] = SourceRuntime(
                source_id=binding.source_id,
                engine=engine_type,
                connection=str(binding.connection or ""),
                artifacts_dir=adir,
                dialect=member_dialect,
                sqlglot_dialect=sqlglot_dialect_for_engine(engine_type),
                native_connection=native_by_source.get(binding.source_id),
                sqlalchemy_engine=sa_by_source.get(binding.source_id),
            )
            continue
        try:
            runtime_cls = _runtime_config_for_engine(engine_type)
        except Exception:
            runtime_cls = fallback_identity.runtime_config
        if engine_type == "duckdb":
            runtime_cls = _duckdb_runtime_config_for_schema(
                runtime_cls, _federation_duckdb_schema_for_connection(str(binding.connection or ""))
            )
        source_sa = sa_by_source.get(binding.source_id, sqlalchemy_engine)
        source_native = native_by_source.get(binding.source_id, native_connection)
        prior = prior_runtimes.get(binding.source_id)
        if prior is not None and prior.engine == engine_type and prior.connection == str(binding.connection or ""):
            if prior.native_connection is not None:
                source_native = prior.native_connection
            if prior.sqlalchemy_engine is not None:
                source_sa = prior.sqlalchemy_engine
        try:
            bound_dialect = get_dialect(
                engine_type, runtime_cls, sqlalchemy_engine=source_sa, native_connection=source_native
            )
        except Exception:
            bound_dialect = default_dialect
        runtimes[binding.source_id] = SourceRuntime(
            source_id=binding.source_id,
            engine=engine_type,
            connection=str(binding.connection or ""),
            artifacts_dir=adir,
            dialect=bound_dialect,
            sqlglot_dialect=sqlglot_dialect_for_engine(engine_type),
            native_connection=source_native,
            sqlalchemy_engine=source_sa,
        )
    return runtimes


def _federation_single_source_sql_context(
    owner: Any,
    intent: Any,
    schema: SchemaGraph,
    fed_manifest: FederationManifest,
    fed_mappings: FederationMappings | None,
    default_dialect: Any,
) -> tuple[Any, SchemaGraph] | None:
    """Return per-source dialect and schema when *intent* references exactly one federation source."""
    source_ids = source_ids_for_intent(intent, schema, fed_mappings, fed_manifest)
    if len(source_ids) != 1:
        return None
    source_id = next(iter(source_ids))
    member_graphs = getattr(owner, "_federation_member_graphs", None) if owner is not None else None
    member_schema = resolve_federated_member_schema(
        source_id,
        schema,
        manifest=fed_manifest,
        member_graphs=member_graphs if isinstance(member_graphs, dict) else None,
    )
    dialects_by_source = getattr(owner, "_federation_dialects", None) if owner is not None else None
    source_dialect = (
        dialects_by_source.get(source_id)
        if isinstance(dialects_by_source, dict) and source_id in dialects_by_source
        else default_dialect
    )
    return source_dialect, member_schema


def _read_text_if_file(path: str | None) -> str | None:
    """Return the text content of *path* if it exists and is a regular file, else None."""
    if not path:
        return None
    expanded = os.path.expanduser(str(path))
    if not os.path.isfile(expanded):
        return None
    with open(expanded, encoding="utf-8") as fh:
        return fh.read()


def write_schema_context_cache(artifacts_dir: str, schema_context: EngineContext) -> str:
    """Persist *schema_context* (with sql_file/notes_file text inlined) to *artifacts_dir*. Returns the path of the written cache file."""
    payload: dict[str, Any] = {
        "version": SCHEMA_CONTEXT_CACHE_VERSION,
        "include": schema_context.include,
        "allow_objects": sorted(schema_context.allow_objects),
        "deny_objects": sorted(schema_context.deny_objects),
        "deny_columns": sorted(schema_context.deny_columns),
        "allow_columns": sorted(schema_context.allow_columns),
        "sql_file_original": schema_context.sql_file,
        "notes_file_original": schema_context.notes_file,
        "sql_text": _read_text_if_file(schema_context.sql_file),
        "notes_text": _read_text_if_file(schema_context.notes_file),
    }
    os.makedirs(artifacts_dir, exist_ok=True)
    cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
    _write_json_atomic(cache_path, payload)
    return cache_path


def load_schema_context_cache(artifacts_dir: str) -> EngineContext | None:
    """Reload a persisted ``EngineContext`` from *artifacts_dir*. Inlined ``sql_text`` / ``notes_text`` are materialised back to disk inside *artifacts_dir* so downstream consumers that expect file paths continue to work. Returns: The restored ``EngineContext``, or ``None`` when no cache file exists or the file is unreadable / not a JSON object. Raises: ConfigError: When the cache file exists but its ``version`` is not :data:`SCHEMA_CONTEXT_CACHE_VERSION` (including legacy version 3). Delete the cache file (or the engine artifacts directory) and re-run initialization so the cache is rewritten; there is no migration path."""
    cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cache_version = payload.get("version")
    if cache_version != SCHEMA_CONTEXT_CACHE_VERSION:
        raise ConfigError(
            f"schema context cache at {cache_path!r} has version {cache_version!r}; "
            f"this build expects {SCHEMA_CONTEXT_CACHE_VERSION}. "
            f"Delete {cache_path!r} (or the engine artifacts directory) and re-run "
            f"initialize_aether_engine so the cache is rebuilt from scratch."
        )
    _validate_scope_list_fields(payload)
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
    if include_raw not in ("tables", "views"):
        raise ConfigError(f"schema context cache include must be tables or views; got {include_raw!r}")
    return EngineContext(
        allow_objects=frozenset(payload.get("allow_objects") or ()),
        include=include_raw,
        deny_objects=frozenset(payload.get("deny_objects") or ()),
        deny_columns=frozenset(payload.get("deny_columns") or ()),
        allow_columns=frozenset(payload.get("allow_columns") or ()),
        sql_file=sql_file,
        notes_file=notes_file,
    )


def _purge_schema_context_cache(artifacts_dir: str) -> None:
    """Remove the persisted ``schema_context.json`` and any materialised cache files. Used during legacy-artifact cleanup so a stale schema context cannot be silently reloaded after a learning-reset rebuild."""
    for name in (SCHEMA_CONTEXT_CACHE_NAME, SCHEMA_CONTEXT_CACHED_DDL, SCHEMA_CONTEXT_CACHED_NOTES):
        fp = os.path.join(artifacts_dir, name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError as exc:
                debug(f"[main_execution._purge_schema_context_cache] {fp}: {exc}")


def _notify_schema_context_warnings(schema_context: EngineContext, sink: Callable[[str], None]) -> None:
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


def _upload_validation_config_error(message: str, data_quality_report: object) -> ConfigError:
    """Attach upload validation context to a configuration error."""
    exc = ConfigError(message)
    exc.data_quality_report = data_quality_report
    return exc


def _emit_runtime_config_override_diagnostics(overridden: frozenset[str]) -> None:
    """Emit one diagnostic per runtime-config field whose effective value came from the TOML file over env."""
    for key in sorted(overridden):
        notify(
            f"Runtime config file overrides environment for {key}",
            stage="config",
            code=DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
            details=(("key", key),),
        )


def migration_report_for_init(
    artifacts_dir: str,
    prompt_schema: SchemaGraph,
    *,
    schema_role: SchemaRole,
    previous_schema: SchemaGraph | None,
    schema_diff: object | None,
) -> MigrationReport:
    """Resolve template migration during single-engine init; consumers never mutate artifacts."""
    if schema_role == "consumer":
        return MigrationReport(tier=MigrationTier.NO_CHANGE)
    return apply_migration_policy(
        artifacts_dir,
        prompt_schema,
        allow_destructive=True,
        previous_schema=previous_schema,
        schema_diff=schema_diff,
    )


def preview_schema_migration(
    *,
    artifacts_dir: str | os.PathLike[str],
    schema_graph: Any,
) -> MigrationPreview:
    """Return a read-only migration preview for the live schema graph against stored artifacts."""
    adir = Path(os.fspath(artifacts_dir))
    schema_path = adir / "schema_graph.json.gz"
    previous_schema = load_schema_graph_snapshot(str(schema_path)) if schema_path.is_file() else None
    schema_diff = diff_schemas(previous_schema, schema_graph) if previous_schema is not None else None
    stored = read_artifact_manifest(str(adir))
    tier = classify_migration_tier(stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff)
    if tier in (
        MigrationTier.NO_CHANGE,
        MigrationTier.ADDITIVE,
        MigrationTier.SOFT_REFRESH,
        MigrationTier.PERMISSION_FILTERED,
    ):
        preview_tier: Literal["compatible", "remap", "destructive"] = "compatible"
    elif tier == MigrationTier.REMAP:
        preview_tier = "remap"
    else:
        preview_tier = "destructive"
    affected_tables: tuple[str, ...] = ()
    affected_columns: tuple[tuple[str, str], ...] = ()
    if schema_diff is not None:
        affected_tables = tuple(sorted(set(schema_diff.dropped_tables) | set(schema_diff.added_tables)))
        column_pairs: list[tuple[str, str]] = []
        for table_name, table_diff in schema_diff.per_table.items():
            for column_name in table_diff.dropped_columns:
                column_pairs.append((table_name, column_name))
            for column_name in table_diff.added_columns:
                column_pairs.append((table_name, column_name))
        affected_columns = tuple(sorted(column_pairs))
    return MigrationPreview(
        tier=preview_tier,
        affected_tables=affected_tables,
        affected_columns=affected_columns,
        skeleton_path=str(adir / MIGRATION_MAP_FILENAME),
    )


def initialize_aether_engine(
    engine_context: EngineContext | str | None = None,
    *,
    artifacts_dir: str | None = None,
    tenant_slug: str | None = None,
    config_file: str | os.PathLike[str] | None = None,
    connection: str | None = None,
    log_sink: Callable[[str], None] | None = None,
    execution_engine: Any | None = None,
    native_connection: Any | None = None,
    schema_role: SchemaRole = "owner",
    source_selections: Mapping[str, Mapping[str, Any]] | None = None,
    trust_bundled_baseline: bool = False,
    token_provider: Callable[[], str | Mapping[str, str]] | None = None,
) -> AetherEngineInitResult:
    """Configure the process environment, build the schema graph, migrate templates, and load stores."""
    sink: Callable[[str], None] = log_sink if log_sink is not None else notify
    sink("Initialising AetherEngine.")
    config_file_values, toml_claimed_keys, named_by_engine = _load_config_file(config_file)
    ssot = config_file is not None and bool(str(config_file).strip())
    merged, toml_diagnostic_keys = _merge_configuration_environment(
        config_file_values, toml_claimed_keys=toml_claimed_keys if ssot else None
    )
    selected_preview = _select_engine_name(merged, named_by_engine)
    resolved_connection = _select_connection_name(
        merged, named_by_engine, selected_preview, explicit_connection=connection
    )
    if resolved_connection and named_by_engine.get(selected_preview):
        connection_values, connection_claimed, _ = _load_config_file(config_file, connection=resolved_connection)
        config_file_values.update(connection_values)
        toml_claimed_keys = toml_claimed_keys | connection_claimed
        merged, toml_diagnostic_keys = _merge_configuration_environment(
            config_file_values, toml_claimed_keys=toml_claimed_keys if ssot else None
        )
        merged["AETHERDIALECT_CONNECTION"] = resolved_connection
    _apply_runtime_environments(merged)
    adir = compute_engine_storage_dir(artifacts_dir, selected_preview, tenant_slug=tenant_slug)
    try:
        cached_master = load_schema_context_cache(adir)
    except ConfigError as exc:
        sink(str(exc))
        cached_master = None
    prepare_master: EngineContext | None = None
    if isinstance(engine_context, FederationContext):
        raise ConfigError("initialize_aether_engine does not accept FederationContext; use AetherFederation instead")
    if isinstance(engine_context, EngineContext):
        prepare_master = _prepare_schema_context_for_init(engine_context, adir, sink)
    master_ctx, active_ctx, context_name = resolve_engine_context_plan(
        engine_context, adir, schema_role=schema_role, load_master=cached_master, prepare_master=prepare_master
    )
    _notify_schema_context_warnings(master_ctx, sink)
    active_engine = configure_runtime_from_environment(master_ctx, merged)
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
                ("deployment_heavy", llm_exec.deployment_heavy),
            )
            if not (isinstance(v, str) and v.strip())
        ]
        if missing:
            raise ConfigError("Azure OpenAI requires non-empty runtime configuration for: " + ", ".join(missing))
    _rt = EngineConfig.RUNTIME
    _rt_name = (getattr(_rt, "__name__", None) or str(_rt) or "default").lower()
    if _rt_name.endswith("runtimeconfig"):
        _rt_name = _rt_name[: -len("runtimeconfig")]
    runtime_label = _rt_name or "default"
    sink(f"  Engine: {active_engine} ({runtime_label}).")
    os.makedirs(adir, exist_ok=True)
    legacy_files = detect_legacy_artifacts(adir)
    if legacy_files:
        sink(f"  Detected legacy artifacts (no manifest): {', '.join(legacy_files)}. Rebuilding caches.")
        wipe_versioned_artifacts(adir)
        _purge_schema_context_cache(adir)
    EngineConfig.SCHEMA_JSON_PATH = os.path.join(adir, "schema_graph.json.gz")
    ensure_template_store_space_layout(adir)
    EngineConfig.TEMPLATE_STORE_DIR = template_store_dir_for_space(adir, MASTER_AETHERSPACE_NAME)
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(adir, "qsim_skeletons.json.gz")
    data_quality_report: DataQualityReport | None = None
    if is_file_engine(active_engine):
        upload_paths = CsvRuntimeConfig.resolve_source_files()
        selections = parse_source_selections(source_selections or CsvRuntimeConfig.SOURCE_SELECTIONS)
        data_quality_report = validate_upload_sources(upload_paths, log_sink=sink, source_selections=selections)
        if data_quality_report.requires_review and not selections:
            raise _upload_validation_config_error(
                f"{data_quality_report.narrative} "
                "Call inspect_tabular_upload and pass source_selections with the accepted interpretation.",
                data_quality_report,
            )
        if not data_quality_report.ok:
            raise _upload_validation_config_error(data_quality_report.narrative, data_quality_report)
        if source_selections:
            CsvRuntimeConfig.set_source_selections(source_selections)
            data_quality_report = DataQualityReport(
                ok=data_quality_report.ok,
                issues=data_quality_report.issues,
                narrative=data_quality_report.narrative,
                suggested_selections=data_quality_report.suggested_selections,
                confirmed_selections=dict(source_selections),
            )
    if token_provider is not None:
        apply_connection_credentials_for_engine(
            active_engine,
            resolve_connection_credentials(None, token_provider),
        )
    try:
        dialect = get_dialect(
            EngineConfig.TYPE,
            EngineConfig.RUNTIME,
            sqlalchemy_engine=execution_engine,
            native_connection=native_connection,
        )
    except ConnectionError:
        raise
    except Exception as exc:
        raise ConnectionError(str(exc)) from exc
    engine_identity = EngineIdentity(engine_type=EngineConfig.TYPE, runtime_config=EngineConfig.RUNTIME)
    notes_content: str | None = None
    if master_ctx.notes_file:
        nf_path = os.path.expanduser(str(master_ctx.notes_file))
        if os.path.isfile(nf_path):
            with open(nf_path, encoding="utf-8") as nf:
                notes_content = nf.read()
    previous_schema = load_schema_graph_snapshot(EngineConfig.SCHEMA_JSON_PATH)
    artifacts_root = Path(adir)
    map_path = artifacts_root / MIGRATION_MAP_FILENAME
    pending_migration_map = (
        load_schema_migration_map(artifacts_root) if map_path.is_file() and schema_role == "owner" else None
    )
    schema_graph, schema_diff = build_schema_graph_with_diff(
        dialect,
        master_ctx,
        notes_content=notes_content,
        log_sink=sink,
        refresh_existing_descriptions_on_addition=(
            pending_migration_map.refresh_existing_descriptions_on_addition
            if pending_migration_map is not None
            else False
        ),
        force_live_schema_reflect=pending_migration_map is not None,
        trust_bundled_baseline=trust_bundled_baseline,
    )
    stored = read_artifact_manifest(adir)
    if map_path.is_file() and schema_role == "owner":
        loaded = (
            pending_migration_map if pending_migration_map is not None else load_schema_migration_map(artifacts_root)
        )
        if loaded is not None:
            try:
                validate_schema_migration_map(loaded, previous_schema, schema_graph)
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
                apply_schema_migration_map(loaded, adir, schema_graph, Path(EngineConfig.SCHEMA_JSON_PATH))
                ts = datetime.now(timezone.utc).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
                applied_map = map_path.with_name(map_path.stem + ".applied.json")
                try:
                    if applied_map.is_file():
                        archive = applied_map.with_name(applied_map.stem + f".{ts}" + applied_map.suffix)
                        applied_map.rename(archive)
                    map_path.rename(applied_map)
                except OSError as exc:
                    debug(f"[main_execution.initialize_aether_engine] could not archive migration map: {exc}")
                previous_schema = load_schema_graph_snapshot(EngineConfig.SCHEMA_JSON_PATH)
                pending_migration_map = None
                schema_graph, schema_diff = build_schema_graph_with_diff(
                    dialect,
                    master_ctx,
                    notes_content=notes_content,
                    log_sink=sink,
                    refresh_existing_descriptions_on_addition=False,
                    force_live_schema_reflect=True,
                    trust_bundled_baseline=trust_bundled_baseline,
                )
                stored = read_artifact_manifest(adir)
    owner_snapshot = previous_schema
    stored = read_artifact_manifest(adir)
    if schema_role == "owner" and stored is not None and not stored.schema_graph_id:
        upgrade_artifacts_schema_graph_id(adir)
        stored = read_artifact_manifest(adir)
    pinned_id = None
    if owner_snapshot is not None:
        pinned_id = str(owner_snapshot.schema_graph_id or "") or None
    if pinned_id is None and stored is not None:
        pinned_id = str(stored.schema_graph_id or "") or None
    assign_schema_graph_hashes(
        schema_graph,
        master_ctx,
        str(getattr(schema_graph, "notes_sha256", "") or ""),
        schema_role=schema_role,
        pinned_schema_graph_id=pinned_id if schema_role == "consumer" else None,
    )
    consumer_visible: frozenset[str] | None = None
    prompt_schema = schema_graph
    tier_preview = classify_migration_tier(
        stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff
    )
    if (
        schema_role == "consumer"
        and owner_snapshot is not None
        and consumer_graph_is_permission_subset(owner_snapshot, schema_graph)
        and stored is not None
        and str(stored.schema_graph_id or "") == str(schema_graph.schema_graph_id or "")
    ):
        tier_preview = MigrationTier.PERMISSION_FILTERED
    if tier_preview == MigrationTier.PERMISSION_FILTERED and owner_snapshot is not None:
        consumer_visible = frozenset(schema_graph.tables.keys())
        prompt_schema = copy.deepcopy(owner_snapshot)
        schema_graph = prompt_schema
    if schema_role == "consumer" and stored is not None and artifact_manifest_incompatible_with_package(stored):
        raise ConfigError(
            "Artifact manifest is incompatible with this package version; "
            "an owner must refresh artifacts before consumer init can proceed."
        )
    if schema_role == "consumer" and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
        raise ConfigError(
            "Schema has drifted since artifacts were published; "
            "an owner must refresh artifacts before consumer init can proceed."
        )
    if schema_role == "owner" and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
        rename_plan = try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
        skel_path = export_schema_migration_map_skeleton(
            artifacts_root, tier=tier_preview, schema_diff=schema_diff, rename_plan=rename_plan
        )
        raise MigrationPendingError(f"Schema migration required: edit {skel_path} and restart init.")
    migration_report = migration_report_for_init(
        adir,
        prompt_schema,
        schema_role=schema_role,
        previous_schema=previous_schema,
        schema_diff=schema_diff,
    )
    if migration_report.tier != MigrationTier.NO_CHANGE:
        _print_migration_applied(migration_report, sink)
    if schema_role == "owner":
        prune_stale_artifact_auxiliaries(adir, active_schema_graph_id=str(prompt_schema.schema_graph_id))
    store = load_template_store(
        prompt_schema.schema_graph_id, prompt_schema, space_name=MASTER_AETHERSPACE_NAME, artifacts_dir=adir
    )
    templates = store_to_templates(store)
    rejected: dict[str, Any] = {}
    sink(f"  Templates: {len(templates)} reusable, {len(rejected)} rejected.")
    schema_terms: set[str] = set(schema_graph.tables.keys())
    for tinfo in schema_graph.tables.values():
        schema_terms.update(tinfo.columns)
        for col in tinfo.columns:
            schema_terms.add(col.lower())
    schema_stats = schema_graph.schema_stats or {}
    if EngineConfig.LLM_PROVIDER == "azure":
        prov: Literal["openai", "azure", "mock"] = "azure"
    elif EngineConfig.LLM_PROVIDER == "mock":
        prov = "mock"
    else:
        prov = "openai"
    llm_config = LLMConfig(provider=prov)
    if context_name != MASTER_AETHERSPACE_NAME:
        validate_named_context_subset(master_ctx, active_ctx, schema_graph)
    execution_ctx = _effective_execution_context(master_ctx, active_ctx, context_name)
    if schema_role == "consumer" and consumer_visible is None and execution_ctx.allow_objects:
        consumer_visible = frozenset(execution_ctx.allow_objects)
    runtime_config = RuntimeConfig(
        engine=active_engine,
        artifacts_dir=adir,
        engine_context=master_ctx,
        llm_execution=llm_exec,
        execution_context=execution_ctx,
    )
    if schema_role == "owner":
        try:
            write_schema_context_cache(adir, master_ctx)
        except OSError as exc:
            debug(f"[main_execution.initialize_aether_engine] schema_context cache write failed: {exc}")
    sink("Ready.")
    return AetherEngineInitResult(
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
        schema_role=schema_role,
        consumer_visible_objects=consumer_visible,
        context_name=context_name,
        execution_context=execution_ctx,
        data_quality_report=data_quality_report,
        federation_manifest=None,
        federation_mappings=None,
        federation_member_graphs=None,
        federation_storage_dir=None,
        federation_source_runtimes=None,
        federation_mapping_suggestions=(),
        federation_dialects_by_source=None,
        engine_identity=engine_identity,
    )


def initialize_aether_federation(
    name: str,
    *,
    members: Mapping[str, Any],
    declaration_file: str,
    declaration: tuple[FederationManifest, FederationMappings] | None = None,
    artifacts_dir: str | None = None,
    schema_role: SchemaRole = "owner",
    master_context: FederationContext | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> AetherFederationInitResult:
    """Compose a federated schema graph from member engines and persist the federation tree."""
    sink: Callable[[str], None] = log_sink if log_sink is not None else notify
    sink(f"Initialising AetherFederation {name!r}.")
    cleanup_abandoned_federation_spill_directories()
    member_dict = {str(connection_name): engine for connection_name, engine in members.items()}
    if not member_dict:
        raise ConfigError("AetherFederation requires at least one member engine")
    validate_federation_file_members(member_dict)
    if declaration is not None:
        authored_manifest, fed_mappings = declaration
    else:
        authored_manifest, fed_mappings = load_federation_declaration_from_path(declaration_file)
    probe_federation_member_connections(member_dict, manifest=authored_manifest, mappings=fed_mappings)
    fed_id = str(name).strip()
    if not fed_id:
        raise ConfigError("AetherFederation name must be non-empty")
    if authored_manifest.federation_id != fed_id:
        raise ConfigError(
            f"federation name {fed_id!r} disagrees with manifest federation_id {authored_manifest.federation_id!r}"
        )
    fed_member_graphs_dict = member_graphs_from_engines(member_dict)
    member_source_ids = set(member_dict)
    authored_manifest, fed_mappings = reconcile_authored_declaration_for_members(
        authored_manifest,
        fed_mappings,
        active_source_ids=member_source_ids,
    )
    fed_manifest = build_federation_manifest_from_members(
        member_dict,
        declaration=authored_manifest,
        member_graphs=fed_member_graphs_dict,
        mappings=fed_mappings,
    )
    active_source_ids = {binding.source_id for binding in fed_manifest.sources}
    fed_manifest = prune_federation_aliases(fed_manifest, active_source_ids=active_source_ids)
    fed_manifest = prune_cross_source_joins(fed_manifest, active_source_ids=active_source_ids)
    fed_mappings = prune_federation_mappings(fed_mappings, fed_manifest, active_source_ids=active_source_ids)
    validate_manifest_cross_source_joins(fed_manifest)
    fed_storage_dir = compute_federation_storage_dir(artifacts_dir, fed_manifest.federation_id)
    if artifacts_dir:
        os.makedirs(fed_storage_dir, exist_ok=True)
    loaded_member_graphs = load_federation_member_graphs(artifacts_dir, fed_manifest)
    if loaded_member_graphs:
        recorded_ids = recorded_federation_source_ids(fed_storage_dir)
        topology_change = detect_federation_topology_change(recorded_ids, fed_manifest) if recorded_ids else "none"
        if topology_change != "add":
            assert_federation_member_graph_roster_complete(fed_manifest, loaded_member_graphs)
    fed_member_graphs_dict = reconcile_federation_member_graphs(
        fed_member_graphs_dict, loaded_member_graphs, fed_manifest
    )
    for source_id, member_graph in fed_member_graphs_dict.items():
        member_engine = member_dict.get(source_id)
        member_ctx = EngineContext()
        if member_engine is not None:
            runtime_cfg = getattr(member_engine, "_runtime_config", None)
            ctx = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
            if isinstance(ctx, EngineContext):
                member_ctx = ctx
        raise_if_schema_unusable(member_graph, member_ctx, federation_composite=False)
    try:
        coord_dialect = get_dialect("duckdb", DuckDBRuntimeConfig)
    except Exception as exc:
        raise FederationConfigError(
            f"federation coordinator dialect resolution failed for engine 'duckdb': {exc}"
        ) from exc
    llm_exec = load_runtime_config(merged_env=dict(os.environ))
    if EngineConfig.LLM_PROVIDER == "azure":
        prov: Literal["openai", "azure", "mock"] = "azure"
    elif EngineConfig.LLM_PROVIDER == "mock":
        prov = "mock"
    else:
        prov = "openai"
    llm_config = LLMConfig(provider=prov)
    fed_master_ctx = master_context or FederationContext()
    master_ctx = fed_master_ctx
    execution_ctx = fed_master_ctx
    context_name = MASTER_AETHERSPACE_NAME
    engine_identity = active_engine_identity()
    notes_content: str | None = None
    if fed_master_ctx.notes_file:
        notes_content = _read_text_if_file(fed_master_ctx.notes_file)
    if notes_content:
        for binding in fed_manifest.sources:
            token = str(binding.source_id or "").strip()
            if token and token in notes_content:
                raise ConfigError(f"federation notes must not name a source or member; found {token!r}")
    raise_if_descriptions_name_federation_sources(
        fed_member_graphs_dict,
        [binding.source_id for binding in fed_manifest.sources],
    )
    source_ids = [binding.source_id for binding in fed_manifest.sources]
    for _, member_engine in member_dict.items():
        member_ctx = EngineContext()
        runtime_cfg = getattr(member_engine, "_runtime_config", None)
        ctx = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
        if isinstance(ctx, EngineContext):
            member_ctx = ctx
        raise_if_member_notes_name_federation_sources(member_ctx.notes_file, source_ids)
    federation_artifacts_root = Path(fed_storage_dir)
    recorded_source_ids = recorded_federation_source_ids(fed_storage_dir)
    if recorded_source_ids:
        fed_manifest, fed_mappings, topo_report = reconcile_federation_topology(
            fed_manifest, fed_mappings, recorded_source_ids, federation_dir=fed_storage_dir
        )
        if topo_report.change != "none":
            added = ", ".join(topo_report.added_source_ids) or "none"
            removed = ", ".join(topo_report.removed_source_ids) or "none"
            sink(
                "  Federation topology change "
                f"{topo_report.change!r}: added=[{added}] removed=[{removed}] "
                f"plan_templates_invalidated={topo_report.plan_templates_invalidated}"
            )
    fed_map_path = federation_artifacts_root / FEDERATION_MIGRATION_MAP_FILENAME
    pending_fed_map_archive: Path | None = None
    if schema_role == "owner" and fed_map_path.is_file():
        fed_loaded = load_federation_migration_map(str(fed_map_path))
        if fed_loaded is not None:
            try:
                validate_federation_migration_map(
                    fed_loaded,
                    cached_member_graphs=loaded_member_graphs,
                    live_member_graphs=fed_member_graphs_dict,
                    manifest=fed_manifest,
                )
            except MigrationPendingError as exc:
                msg = str(exc)
                if msg.startswith("STALE_MAP:"):
                    try:
                        fed_map_path.unlink()
                    except OSError:
                        pass
                    sink("  Removed stale federation_migration_map.json for this snapshot.")
                else:
                    raise
            else:
                if fed_loaded.action == MIGRATION_MAP_ACTION_ABORT:
                    try:
                        fed_map_path.unlink()
                    except OSError:
                        pass
                    raise MigrationPendingError("user aborted via federation migration map")
                fed_manifest, fed_mappings = apply_federation_migration_map(
                    fed_loaded, fed_manifest, fed_mappings, fed_storage_dir
                )
                pending_fed_map_archive = fed_map_path
    fed_mapping_suggestions = cached_or_suggest_cross_source_mappings(
        fed_member_graphs_dict, fed_manifest, fed_storage_dir, existing_mappings=fed_mappings
    )
    sa_by_source = {source_id: getattr(engine, "_execution_engine", None) for source_id, engine in member_dict.items()}
    native_by_source = {
        source_id: getattr(engine, "_native_connection", None) for source_id, engine in member_dict.items()
    }
    default_dialect = coord_dialect
    fed_source_runtimes = _build_federation_source_runtimes(
        fed_manifest,
        artifacts_dir,
        default_dialect,
        default_identity=engine_identity,
        engines_by_source=sa_by_source,
        native_connections_by_source=native_by_source,
        members_by_source=member_dict,
    )
    fed_dialects_by_source = {source_id: runtime.dialect for source_id, runtime in fed_source_runtimes.items()}
    llm_classify = llm_classify_schema if notes_content else None
    federation_format_stale = False
    try:
        replay_ok = mappings_replay_matches(fed_storage_dir, fed_member_graphs_dict, fed_manifest, fed_mappings)
    except FederationConfigError as exc:
        sink(str(exc))
        replay_ok = False
        federation_format_stale = True
    broken_joins = detect_broken_cross_source_joins(fed_member_graphs_dict, fed_manifest)
    has_persisted_federation = os.path.isfile(federation_artifact_paths(fed_storage_dir)["artifact_manifest"])
    if broken_joins and has_persisted_federation:
        skel_path = ""
        if schema_role == "owner":
            skel_path = export_federation_migration_map_skeleton(
                str(federation_artifacts_root), dropped_joins=broken_joins
            )
            sink(f"  Federation cross-source join columns missing; migration skeleton written to {skel_path!r}.")
        if schema_role == "owner":
            raise MigrationPendingError(f"Federation migration required: edit {skel_path} and restart init.")
        raise ConfigError(
            "Federation cross-source join columns are missing; "
            "an owner must refresh artifacts before consumer init can proceed."
        )
    if not replay_ok and not federation_format_stale and has_persisted_federation:
        prune_federation_plan_templates_on_drift(fed_storage_dir, fed_member_graphs_dict, fed_manifest, fed_mappings)
        if schema_role == "owner":
            skel_path = export_federation_migration_map_skeleton(str(federation_artifacts_root))
            sink(f"  Federation drift detected; migration skeleton written to {skel_path!r}.")
            raise MigrationPendingError(f"Federation migration required: edit {skel_path} and restart init.")
        raise ConfigError(
            "Federation member graphs have drifted; an owner must refresh artifacts before consumer init can proceed."
        )
    schema_graph = compose_composite_graph(
        fed_member_graphs_dict,
        fed_manifest,
        fed_mappings,
        notes_content=notes_content,
        llm_classify=llm_classify,
        master_context=fed_master_ctx,
    )
    validate_cross_source_keys_on_graph(schema_graph, fed_manifest, fed_mappings)
    for source_id, member_graph in fed_member_graphs_dict.items():
        engine = ""
        for binding in fed_manifest.sources:
            if binding.source_id == source_id:
                engine = str(binding.engine or "").strip().lower()
                break
        stamp_federation_member_graph(
            member_graph,
            federation_id=fed_manifest.federation_id,
            source_id=source_id,
            engine=engine,
        )
    object.__setattr__(
        schema_graph,
        "_database_feature_capability_cache",
        intersect_member_database_feature_capabilities(fed_member_graphs_dict),
    )
    notes_sha = str(getattr(schema_graph, "notes_sha256", "") or "")
    assign_schema_graph_hashes(
        schema_graph,
        master_ctx,
        notes_sha,
        schema_role=schema_role,
        federation_scope_hash=schema_graph.scope_hash or None,
    )
    stored = read_artifact_manifest(fed_storage_dir)
    previous_composite = load_federation_composite_graph(fed_storage_dir) if stored is not None else None
    tier_preview = federation_composite_migration_tier(
        fed_storage_dir, schema_graph, previous_composite=previous_composite
    )
    if schema_role == "consumer" and stored is not None and artifact_manifest_incompatible_with_package(stored):
        raise ConfigError(
            "Federation artifact manifest is incompatible with this package version; "
            "an owner must refresh artifacts before consumer init can proceed."
        )
    if schema_role == "consumer" and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
        raise ConfigError(
            "Federation artifacts have drifted; an owner must refresh artifacts before consumer init can proceed."
        )
    if schema_role == "owner" and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
        skel_path = export_federation_migration_map_skeleton(str(federation_artifacts_root))
        sink(f"  Federation composite drift ({tier_preview.value}); migration skeleton written to {skel_path!r}.")
        raise MigrationPendingError(f"Federation migration required: edit {skel_path} and restart init.")
    migration_report = apply_federation_composite_migration_policy(
        fed_storage_dir,
        schema_graph,
        allow_destructive=schema_role == "owner",
        previous_composite=previous_composite,
    )
    if migration_report.tier != MigrationTier.NO_CHANGE:
        _print_migration_applied(migration_report, sink)
    if schema_role == "owner":
        prune_stale_artifact_auxiliaries(fed_storage_dir, active_schema_graph_id=str(schema_graph.schema_graph_id))
    if schema_role == "owner":
        persist_federation_tree(
            fed_storage_dir,
            manifest=fed_manifest,
            mappings=fed_mappings,
            composite=schema_graph,
            member_graphs=fed_member_graphs_dict,
        )
        if pending_fed_map_archive is not None:
            archive_federation_migration_map_file(pending_fed_map_archive, archive_dir=fed_storage_dir)
    store = load_template_store(
        schema_graph.schema_graph_id, schema_graph, space_name=MASTER_AETHERSPACE_NAME, artifacts_dir=fed_storage_dir
    )
    templates = store_to_templates(store)
    rejected: dict[str, Any] = {}
    sink(f"  Templates: {len(templates)} reusable, {len(rejected)} rejected.")
    schema_terms: set[str] = set(schema_graph.tables.keys())
    for tinfo in schema_graph.tables.values():
        schema_terms.update(tinfo.columns)
        for col in tinfo.columns:
            schema_terms.add(col.lower())
    schema_stats = schema_graph.schema_stats or {}
    composite_tables = frozenset(schema_graph.tables.keys())
    if composite_tables:
        execution_ctx = replace(
            fed_master_ctx,
            allow_objects=_federation_execution_allow_objects(fed_master_ctx, composite_tables),
        )
    runtime_config = RuntimeConfig(
        engine="federation",
        artifacts_dir=fed_storage_dir,
        engine_context=master_ctx,
        llm_execution=llm_exec,
        execution_context=execution_ctx,
    )
    drain_owner = SimpleNamespace(
        _is_aether_federation=True,
        _schema_graph=schema_graph,
        _store=store,
        _templates=templates,
        _rejected=rejected,
        _dialect=coord_dialect,
        _federation_source_runtimes=fed_source_runtimes,
        _federation_member_graphs=fed_member_graphs_dict,
    )
    drain_write_queue(drain_owner, fed_storage_dir)
    sink(f"  Federation: {fed_manifest.federation_id} ({len(member_dict)} members).")
    payload_counts = composite_schema_payload_counts(schema_graph)
    sink(
        "  Composite schema payload: "
        f"{payload_counts['tables']} tables, "
        f"{payload_counts['columns']} columns, "
        f"{payload_counts['enum_types']} enum types "
        f"({payload_counts['enum_labels']} labels)."
    )
    sink("Ready.")
    return AetherFederationInitResult(
        runtime_config=runtime_config,
        llm_config=llm_config,
        schema_graph=schema_graph,
        dialect=coord_dialect,
        artifacts_dir=fed_storage_dir,
        store=store,
        templates=templates,
        rejected=rejected,
        schema_terms=schema_terms,
        schema_stats=schema_stats,
        schema_role=schema_role,
        consumer_visible_objects=None,
        context_name=context_name,
        execution_context=execution_ctx,
        data_quality_report=None,
        federation_manifest=fed_manifest,
        federation_mappings=fed_mappings,
        federation_member_graphs=fed_member_graphs_dict,
        federation_storage_dir=fed_storage_dir,
        federation_source_runtimes=fed_source_runtimes,
        federation_mapping_suggestions=fed_mapping_suggestions,
        federation_dialects_by_source=fed_dialects_by_source,
        engine_identity=engine_identity,
        members=member_dict,
    )


def clear_template_store_only(artifacts_dir: str, schema_graph: SchemaGraph) -> bool:
    """Remove the partitioned template store directory and legacy monolithic file when present."""
    assert isinstance(schema_graph, SchemaGraph)
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    legacy = os.path.join(artifacts_dir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)
    existed = os.path.isdir(store_dir) or os.path.isfile(legacy)
    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)
    wipe_filenames(artifacts_dir, (TEMPLATE_STORE_LEGACY_SINGLE_FILE,))
    return existed


def resolve_connection_credentials(
    credentials: str | Mapping[str, str] | None,
    token_provider: Callable[[], str | Mapping[str, str]] | None,
) -> str | Mapping[str, str]:
    """Return explicit credentials or consult *token_provider*."""
    if credentials is not None:
        return credentials
    if token_provider is not None:
        resolved = token_provider()
        if resolved is None or (isinstance(resolved, str) and not str(resolved).strip()):
            raise ConfigError("token_provider returned an empty credential value")
        return resolved
    raise ConfigError(
        "refresh_connection requires explicit credentials or a token_provider callable configured on the engine."
    )


def apply_connection_credentials_for_engine(
    engine_type: str,
    credentials: str | Mapping[str, str],
) -> None:
    """Apply rotatable secrets on the runtime config for *engine_type*."""
    runtime_cls = _runtime_config_for_engine(engine_type)
    try:
        runtime_cls.apply_connection_credentials(credentials)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def dispose_engine_dialect(
    dialect: Any,
    *,
    borrowed_execution_engine: Any | None = None,
    borrowed_native_connection: Any | None = None,
) -> None:
    """Release dialect-owned database handles without closing borrowed caller handles."""
    dispose_native = getattr(dialect, "dispose_native_connection", None)
    if callable(dispose_native):
        try:
            dispose_native()
        except Exception:
            pass
        return
    connection = getattr(dialect, "connection", None)
    if connection is not None and connection is not borrowed_native_connection:
        close = getattr(connection, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    sa_engine = getattr(dialect, "engine", None)
    if sa_engine is not None and sa_engine is not borrowed_execution_engine:
        dispose = getattr(sa_engine, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass


def refresh_engine_connection(
    *,
    engine_type: str,
    dialect: Any,
    credentials: str | Mapping[str, str] | None = None,
    token_provider: Callable[[], str | Mapping[str, str]] | None = None,
    execution_engine: Any | None = None,
    native_connection: Any | None = None,
) -> Any:
    """Dispose the live dialect, apply fresh credentials, and open a replacement handle."""
    resolved = resolve_connection_credentials(credentials, token_provider)
    dispose_engine_dialect(
        dialect,
        borrowed_execution_engine=execution_engine,
        borrowed_native_connection=native_connection,
    )
    apply_connection_credentials_for_engine(engine_type, resolved)
    runtime_cls = _runtime_config_for_engine(engine_type)
    try:
        return get_dialect(
            engine_type,
            runtime_cls,
            sqlalchemy_engine=execution_engine,
            native_connection=native_connection,
        )
    except ConnectionError:
        raise
    except Exception as exc:
        raise ConnectionError(str(exc)) from exc


def dispose_federation_source_runtimes(
    runtimes: Mapping[str, SourceRuntime] | None, *, member_engines: Mapping[str, Any] | None = None
) -> None:
    """Release dialect-owned resources for federation source runtimes without closing borrowed member handles."""
    if not runtimes:
        return
    borrowed_sa: set[int] = set()
    borrowed_native: set[int] = set()
    for engine in (member_engines or {}).values():
        sa = getattr(engine, "_execution_engine", None)
        if sa is not None:
            borrowed_sa.add(id(sa))
        native = getattr(engine, "_native_connection", None)
        if native is not None:
            borrowed_native.add(id(native))
    for runtime in runtimes.values():
        dialect = getattr(runtime, "dialect", None)
        dispose_dialect = getattr(dialect, "dispose_native_connection", None)
        if callable(dispose_dialect):
            try:
                dispose_dialect()
            except Exception:
                pass
        sa = getattr(runtime, "sqlalchemy_engine", None)
        if sa is not None and id(sa) not in borrowed_sa:
            dispose_sa = getattr(sa, "dispose", None)
            if callable(dispose_sa):
                try:
                    dispose_sa()
                except Exception:
                    pass
        native = getattr(runtime, "native_connection", None)
        if native is not None and id(native) not in borrowed_native:
            close_native = getattr(native, "close", None)
            if callable(close_native):
                try:
                    close_native()
                except Exception:
                    pass


def clear_federation_template_stores(
    federation_dir: str | None,
    composite_artifacts_dir: str,
    composite_graph: SchemaGraph,
    member_engines: Mapping[str, Any],
) -> bool:
    """Clear composite, plan-record, and member template stores for a federation."""
    existed = clear_template_store_only(composite_artifacts_dir, composite_graph)
    if federation_dir:
        clear_federation_plan_templates(federation_dir)
    for engine in member_engines.values():
        graph = getattr(engine, "_schema_graph", None)
        adir = getattr(engine, "_artifacts_dir", None)
        if graph is not None and adir is not None:
            existed = clear_template_store_only(str(adir), graph) or existed
    return existed


def describe_federation_config(
    federation_name: str,
    runtime: RuntimeConfig,
    llm: LLMConfig,
    *,
    members: Mapping[str, Any],
    federation_storage_dir: str | None = None,
) -> str:
    """Build a redacted config snapshot including federation topology."""
    lines = [describe_runtime_config(runtime, llm), "", "Federation:"]
    lines.append(f"  name:          {federation_name}")
    if federation_storage_dir:
        lines.append(f"  storage dir:   {os.path.abspath(federation_storage_dir)}")
    lines.append(f"  member count:  {len(members)}")
    for connection_name, engine in sorted(members.items()):
        member_engine = str(getattr(engine, "dialect", "") or "")
        member_dir = os.path.abspath(str(getattr(engine, "_artifacts_dir", "") or ""))
        lines.append(f"  {connection_name}: engine={member_engine!r} artifacts_dir={member_dir}")
    return "\n".join(lines)


def clear_simulation_caches_only(artifacts_dir: str) -> int:
    """Remove QSim and seed-warmup simulation artifacts; return count of files removed."""
    count = wipe_filenames(artifacts_dir, SIMULATION_CACHE_EXACT_FILENAMES)
    count += wipe_globs(artifacts_dir, SIMULATION_CACHE_GLOB_PATTERNS)
    return count


def resolve_qsim_path(version_or_result: int | QSimSummary, artifacts_dir: str) -> str:
    """Resolve the full file path for a QSim questions text artifact."""
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


def _validate_yes_no_reply_token(token: str, *, param: str) -> None:
    if token not in ("y", "n"):
        raise ValueError(f"{param} must be 'y' or 'n'")


def _normalise_yes_no(raw: str, options: list[str]) -> str | None:
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


def _check_session_persistence_format_version(payload: dict[str, Any]) -> None:
    stored = payload.get("format_version")
    try:
        found = int(stored) if stored is not None else None
    except (TypeError, ValueError):
        found = None
    if found != SESSION_PERSISTENCE_FORMAT_VERSION:
        raise ConfigError(
            f"session persistence payload has format_version {stored!r}; "
            f"this build expects {SESSION_PERSISTENCE_FORMAT_VERSION}."
        )


def _serialize_diagnostic(diag: Diagnostic) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stage": diag.stage,
        "level": diag.level,
        "code": diag.code,
        "message": diag.message,
        "details": [list(pair) for pair in diag.details],
    }
    if diag.duration_ms is not None:
        out["duration_ms"] = diag.duration_ms
    if diag.source_id is not None:
        out["source_id"] = diag.source_id
    return out


def _deserialize_diagnostic(raw: dict[str, Any]) -> Diagnostic:
    details_raw = raw.get("details") or []
    details = tuple(tuple(pair) for pair in details_raw)
    duration_ms = raw.get("duration_ms")
    if duration_ms is not None:
        duration_ms = int(duration_ms)
    source_id = raw.get("source_id")
    if source_id is not None:
        source_id = str(source_id)
    return Diagnostic(
        stage=str(raw["stage"]),
        level=str(raw["level"]),
        code=str(raw["code"]),
        message=str(raw["message"]),
        details=details,
        duration_ms=duration_ms,
        source_id=source_id,
    )


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


def _serialize_interpretation(interp: IntentInterpretation) -> dict[str, Any]:
    return {
        "approach": interp.approach,
        "grounding": [list(pair) for pair in interp.grounding],
    }


def _deserialize_interpretation(raw: dict[str, Any]) -> IntentInterpretation:
    grounding_raw = raw.get("grounding") or []
    return IntentInterpretation(
        approach=str(raw["approach"]),
        grounding=tuple(tuple(pair) for pair in grounding_raw),
    )


def _deserialize_param_value(raw: Any) -> ParamValue:
    if isinstance(raw, list):
        return [item for item in raw]
    if isinstance(raw, (str, int, float, bool)):
        return raw
    raise ValueError(f"unsupported parameter value type: {type(raw)!r}")


def _serialize_parameter_binding(binding: ParameterBinding) -> dict[str, Any]:
    out: dict[str, Any] = {
        "handle": binding.handle,
        "current_value": binding.current_value,
        "display_name": binding.display_name,
    }
    if binding.upper_handle:
        out["upper_handle"] = binding.upper_handle
    if binding.unit_handle:
        out["unit_handle"] = binding.unit_handle
    return out


def _serialize_session_notice(notice: SessionNotice) -> dict[str, Any]:
    return {"code": notice.code, "level": notice.level, "message": notice.message}


def _deserialize_session_notice(raw: dict[str, Any]) -> SessionNotice:
    return SessionNotice(
        code=str(raw["code"]),
        level=str(raw["level"]),
        message=str(raw["message"]),
    )


def _deserialize_parameter_binding(raw: dict[str, Any]) -> ParameterBinding:
    current_raw = raw.get("current_value")
    current_value = None if current_raw is None else _deserialize_param_value(current_raw)
    return ParameterBinding(
        handle=str(raw["handle"]),
        current_value=current_value,
        display_name=str(raw.get("display_name") or ""),
        upper_handle=str(raw.get("upper_handle") or ""),
        unit_handle=str(raw.get("unit_handle") or ""),
    )


def _serialize_dataframe(df: pandas.DataFrame) -> dict[str, Any]:
    return {
        "columns": list(df.columns),
        "records": df.to_dict(orient="records"),
    }


def _deserialize_dataframe(raw: dict[str, Any]) -> pandas.DataFrame:
    columns = list(raw.get("columns") or [])
    records = raw.get("records") or []
    if not columns:
        return pandas.DataFrame(records)
    return pandas.DataFrame.from_records(records, columns=columns)


def serialize_session_step(step: SessionStep) -> dict[str, Any]:
    """Return a JSON-serialisable dict for *step*."""
    payload: dict[str, Any] = {
        "format_version": SESSION_PERSISTENCE_FORMAT_VERSION,
        "done": step.done,
        "prompt": step.prompt,
        "kind": step.kind,
        "sql": step.sql,
        "message": step.message,
        "error": step.error,
        "status": step.status,
        "reply_shape": step.reply_shape,
        "semantic_warnings": list(step.semantic_warnings),
        "retryable": step.retryable,
        "diagnostics": [_serialize_diagnostic(d) for d in step.diagnostics],
        "parameters": [_serialize_parameter_binding(p) for p in step.parameters],
        "federation_source_id": step.federation_source_id,
        "federation_phase": step.federation_phase,
        "federation_limit_key": step.federation_limit_key,
        "federation_succeeded": [list(row) for row in step.federation_succeeded],
        "notices": [_serialize_session_notice(n) for n in step.notices],
        "data_truncated": step.data_truncated,
    }
    if step.data is not None:
        payload["data"] = _serialize_dataframe(step.data)
    if step.intent_summary is not None:
        payload["intent_summary"] = _serialize_intent_summary(step.intent_summary)
    if step.interpretation is not None:
        payload["interpretation"] = _serialize_interpretation(step.interpretation)
    return payload


def deserialize_session_step(payload: dict[str, Any]) -> SessionStep:
    """Rebuild a :class:`SessionStep` from *payload*, refusing on version mismatch."""
    _check_session_persistence_format_version(payload)
    data_out: pandas.DataFrame | None = None
    data_raw = payload.get("data")
    if data_raw is not None:
        data_out = _deserialize_dataframe(data_raw)
    intent_summary_raw = payload.get("intent_summary")
    intent_summary = _deserialize_intent_summary(intent_summary_raw) if intent_summary_raw is not None else None
    interpretation_raw = payload.get("interpretation")
    interpretation = _deserialize_interpretation(interpretation_raw) if interpretation_raw is not None else None
    diagnostics_raw = payload.get("diagnostics") or []
    diagnostics = tuple(_deserialize_diagnostic(d) for d in diagnostics_raw)
    parameters_raw = payload.get("parameters") or []
    parameters = tuple(_deserialize_parameter_binding(p) for p in parameters_raw)
    federation_succeeded_raw = payload.get("federation_succeeded") or []
    federation_succeeded = tuple(tuple(row) for row in federation_succeeded_raw)
    notices_raw = payload.get("notices") or []
    notices = tuple(_deserialize_session_notice(n) for n in notices_raw)
    reply_shape = payload.get("reply_shape")
    if reply_shape is not None and reply_shape not in ("yes_no", "free_text"):
        raise ValueError(f"invalid reply_shape: {reply_shape!r}")
    return SessionStep(
        done=bool(payload["done"]),
        prompt=payload.get("prompt"),
        kind=str(payload["kind"]),
        sql=payload.get("sql"),
        data=data_out,
        message=payload.get("message"),
        error=payload.get("error"),
        intent_summary=intent_summary,
        diagnostics=diagnostics,
        status=payload.get("status"),
        reply_shape=reply_shape,
        semantic_warnings=tuple(payload.get("semantic_warnings") or ()),
        interpretation=interpretation,
        parameters=parameters,
        federation_source_id=payload.get("federation_source_id"),
        federation_phase=payload.get("federation_phase"),
        federation_limit_key=payload.get("federation_limit_key"),
        federation_succeeded=federation_succeeded,
        retryable=bool(payload.get("retryable", False)),
        notices=notices,
        data_truncated=bool(payload.get("data_truncated", False)),
    )


def serialize_suspended_state(
    state_id: str,
    message: str,
    choice_queue: list[tuple[str, str]],
    turn_question: str | None,
    *,
    resume_choice_stage_id: str | None = None,
) -> dict[str, Any]:
    """Capture minimal suspended-session fields for later restoration."""
    payload: dict[str, Any] = {
        "format_version": SESSION_PERSISTENCE_FORMAT_VERSION,
        "state_id": state_id,
        "message": message,
        "choice_queue": [list(pair) for pair in choice_queue],
        "turn_question": turn_question,
    }
    if resume_choice_stage_id is not None:
        payload["resume_choice_stage_id"] = resume_choice_stage_id
    return payload


def deserialize_suspended_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild suspended-session fields from *payload*, refusing on version mismatch."""
    _check_session_persistence_format_version(payload)
    choice_queue_raw = payload.get("choice_queue") or []
    choice_queue = [tuple(pair) for pair in choice_queue_raw]
    turn_question = payload.get("turn_question")
    if turn_question is not None:
        turn_question = str(turn_question)
    resume_choice_stage_id = payload.get("resume_choice_stage_id")
    if resume_choice_stage_id is not None:
        resume_choice_stage_id = str(resume_choice_stage_id)
    return {
        "state_id": str(payload["state_id"]),
        "message": str(payload.get("message") or ""),
        "choice_queue": choice_queue,
        "turn_question": turn_question,
        "resume_choice_stage_id": resume_choice_stage_id,
    }


@contextlib.contextmanager
def _owner_business_knowledge_scope(owner: Any):
    """Bind the owner's stored business knowledge for nested pipeline work."""
    holder = getattr(owner, "_business_knowledge", None)
    if not isinstance(holder, BusinessKnowledgeHolder):
        yield
        return
    with business_knowledge_scope(**holder.scope_kwargs()):
        yield


class PipelineSession(InteractiveChoicePort):
    """Programmatic driver for one interactive turn at a time via ask and step. When used as the interactive choice port, the internal pipeline calls :meth:`has_pending_choice` and :meth:`take_yes_no`. :meth:`note_turn_outcome` records the latest turn for :meth:`step` consumers. Builtin ``dir`` on this class lists only ask, ask_until_done, awaiting_prompt, reset, and step. Writer and reader turns acquire the owner's ``_pipeline_writer_lock``; writers may drain the artifact write queue. Only one turn may be in flight per session instance at a time."""

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
        "_active_federation_execution_context",
        "_pending_federation_plan_template",
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
        self._resume_choice_stage_id: str | None = None
        self._last_turn_outcome: dict[str, Any] | None = None
        self._session_busy = False
        self._session_busy_lock = threading.Lock()
        self._refinement_ctx: RefinementContext | None = None
        self._turn_question: str | None = None
        self._pending_conversation_rejection_hints: tuple[str, ...] = ()
        self._turn_llm_usage_start = 0
        self._turn_llm_scope_tok: Any = None
        self._turn_accumulated_diagnostics: list[Diagnostic] = []
        self._turn_cancel_event = threading.Event()
        self._data_row_cap = int(data_row_cap) if data_row_cap is not None and int(data_row_cap) > 0 else None
        self._active_federation_execution_context: FederationExecutionContext | None = None
        self._pending_federation_plan_template: FederationPlanTemplate | None = None

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
        return serialize_suspended_state(
            self._suspended.state_id,
            self._suspended.message_for_caller,
            list(self._choice_queue),
            self._turn_question,
            resume_choice_stage_id=self._resume_choice_stage_id,
        )

    @classmethod
    def restore_serialized_state(cls, owner: Any, payload: dict[str, Any]) -> PipelineSession:
        """Rebuild a session from :meth:`export_serialized_state` output."""
        fields = deserialize_suspended_state(payload)
        sess = cls(
            owner,
            mode="writer",
            space_name=str(getattr(owner, "_active_space_name", "master") or "master"),
        )
        sess._suspended = PipelineSuspended(
            fields["state_id"],
            fields["message"],
            None,
        )
        sess._choice_queue = deque(fields["choice_queue"])
        sess._turn_question = fields["turn_question"]
        sess._resume_choice_stage_id = fields["resume_choice_stage_id"]
        sess._session_busy = True
        return sess

    def _audit_ask_emit(
        self, event_type: str, *, question: str | None = None, details: tuple[tuple[str, str], ...] = ()
    ) -> None:
        fn = getattr(self._owner, "_audit_emit", None)
        if not callable(fn):
            return
        owner_schema = getattr(self._owner, "_schema_graph", None)
        schema_hash_val: str | None = None
        if owner_schema is not None:
            schema_hash_val = getattr(owner_schema, "effective_structural_hash", None)
        fn(event_type, question=question, schema_hash=schema_hash_val, details=details)

    def _turn_llm_usage_records(self) -> tuple[Any, ...]:
        records = snapshot_llm_usage_records()
        if self._turn_llm_usage_start >= len(records):
            return ()
        return records[self._turn_llm_usage_start :]

    def _emit_turn_llm_usage(
        self, *, question: str | None, diagnostics: tuple[Diagnostic, ...] = ()
    ) -> tuple[Diagnostic, ...]:
        records = self._turn_llm_usage_records()
        if not records:
            return diagnostics
        provider_raw = str(getattr(getattr(self._owner, "_llm_config", None), "provider", "openai"))
        provider: Literal["openai", "azure", "mock"]
        if provider_raw in ("openai", "azure", "mock"):
            provider = cast(Literal["openai", "azure", "mock"], provider_raw)
        else:
            provider = "openai"
        for record in records:
            self._audit_ask_emit("llm_call", question=question, details=llm_call_audit_details(record))
        self._audit_ask_emit("llm_turn", question=question, details=llm_turn_audit_details(records, provider=provider))
        cost_diag = llm_turn_cost_diagnostic(records, provider=provider)
        if cost_diag is None:
            return diagnostics
        return diagnostics + (cost_diag,)

    def _extend_turn_accumulated_diagnostics(self, merged: tuple[Diagnostic, ...]) -> None:
        """Append *merged* to the active turn accumulator, deduping by (code, stage, message)."""
        seen = {(d.code, d.stage, d.message) for d in self._turn_accumulated_diagnostics}
        for d in merged:
            key = (d.code, d.stage, d.message)
            if key in seen:
                continue
            self._turn_accumulated_diagnostics.append(d)
            seen.add(key)

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
        return SessionStep(**merged_kw)

    def _attach_refinement_ctx(self, ctx: RefinementContext | None) -> None:
        """Bind :class:`RefinementContext` for silent in-turn retries after user rejection."""
        self._refinement_ctx = ctx

    def _continue_after_refinement_retry(self) -> SessionStep:
        """Run additional intent passes until completion, another suspend, or terminal failure."""
        ctx = self._refinement_ctx
        if ctx is None:
            turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=())
            self._reset_after_turn()
            self._release_session_turn()
            st = self._mk_step(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                error="Refinement context missing.",
                diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
            )
            return replace(st, status=_failure_category_for_terminal_step(st))
        dialect = self._owner._dialect
        schema, store, templates, rejected, schema_terms = self._resources()
        corrected = ctx.corrected_question
        q_norm = normalize_question(corrected)
        while True:
            try:
                with llm_execution_scope(self._owner._runtime_config.llm_execution):
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
                        turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=())
                        self._reset_after_turn()
                        self._release_session_turn()
                        st = self._mk_step(
                            done=True,
                            prompt=None,
                            kind=SESSION_KIND_ERROR,
                            error="Intent parse failed.",
                            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
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
    ) -> tuple[SchemaGraph, dict[str, Any] | TemplateStoreView, dict[str, Any], dict[str, Any], set[str]]:
        """Return the schema graph and template backing structures from the owning facade."""
        owner = self._owner
        cached_store = getattr(owner, "_store", None)
        store: dict[str, Any] | TemplateStoreView
        if isinstance(cached_store, TemplateStoreView):
            store = cached_store
        elif isinstance(cached_store, dict) and cached_store:
            store = cached_store
        else:
            graph_id = str(getattr(owner._schema_graph, "schema_graph_id", "") or "")
            raw_ad = getattr(owner, "_artifacts_dir", None)
            if isinstance(raw_ad, (str, Path)):
                store = load_template_store(
                    graph_id, owner._schema_graph, space_name=self._space_name, artifacts_dir=str(raw_ad)
                )
            else:
                store = load_template_store(graph_id, owner._schema_graph, space_name=self._space_name)
            _sync_owner_template_cache(owner, store)
        templates = getattr(owner, "_templates", None)
        if templates is None:
            templates = store_to_templates(store)
            owner._templates = templates
        return (owner._schema_graph, store, templates, owner._rejected, owner._schema_terms)

    def __dir__(self) -> list[str]:
        """Return names intended for interactive discovery."""
        return sorted(
            ("accept_until_done", "ask", "ask_until_done", "awaiting_prompt", "reset", "reuse_saved_question", "step")
        )

    def __enter__(self) -> PipelineSession:
        """Return *self* for ``with`` blocks."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        """Reset partial turn state when leaving a ``with`` block."""
        self.reset()
        return False

    def _reset_after_turn(self) -> None:
        """Clear partial turn state after a completed or abandoned interactive pass."""
        if self._turn_llm_scope_tok is not None:
            reset_turn_llm_scope(self._turn_llm_scope_tok)
            self._turn_llm_scope_tok = None
        self._turn_llm_usage_start = 0
        self._turn_accumulated_diagnostics = []
        self._choice_queue.clear()
        self._suspended = None
        self._resume_choice_stage_id = None
        self._last_turn_outcome = None
        self._refinement_ctx = None
        self._turn_question = None
        self._pending_conversation_rejection_hints = ()
        self._turn_cancel_event.clear()
        self._active_federation_execution_context = None
        self._pending_federation_plan_template = None

    def reset(self) -> None:
        """Clear suspend state, queued programmatic answers, and partial turn state."""
        self._reset_after_turn()
        self._release_session_turn()

    def cancel(self) -> bool:
        """Cancel the in-flight turn owned by this session when one is in progress. Safe to call from another thread. Cancellation is cooperative: federation workers observe it between member stages or batches; non-federated work observes it at pipeline checkpoints."""
        cancelled = False
        with self._session_busy_lock:
            busy = self._session_busy
        if busy:
            self._turn_cancel_event.set()
            cancelled = True
        ctx = self._active_federation_execution_context
        if ctx is not None:
            ctx.cancel()
            cancelled = True
        return cancelled

    def cancel_active_federation_turn(self) -> bool:
        """Cancel the federated turn owned by this session when one is in progress. Deprecated alias for :meth:`cancel`."""
        return self.cancel()

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
        return _normalise_yes_no(raw, options)

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

    def ask(self, question: str) -> SessionStep:
        """Start a new NL turn and return the first :class:`SessionStep` (prompt, result, or error)."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        if getattr(self._owner, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        if not question.strip():
            self._audit_ask_emit(
                "ask_blocked",
                question=question,
                details=(("reason", "empty_question"),),
            )
            st = self._mk_step(done=True, prompt=None, kind=SESSION_KIND_ERROR, error="Question must not be empty.")
            return replace(st, status=_failure_category_for_terminal_step(st))
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
        for _orph in take_and_clear_orphan_diagnostics():
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        self._pending_conversation_rejection_hints = ()
        try:
            return self._drive_question_turn(question)
        finally:
            reset_diagnostic_collector(tok)

    def ask_until_done(self, question: str, *, on_confirm: Literal["y", "n"] = "y") -> SessionStep:
        """Run ``ask`` then auto-answer yes or no suspends with *on_confirm* until the turn ends. When the user declines executed SQL on the final yes or no prompt, the terminal :class:`SessionStep` carries ``status`` ``FailureCategory.RESULT_OKAY_INTENT_WRONG`` so programmatic callers can distinguish validated-but-rejected runs from unconditional success."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        _validate_yes_no_reply_token(on_confirm, param="on_confirm")
        step = self.ask(question)
        while not step.done:
            if step.reply_shape != "yes_no":
                raise SessionActiveError(f"free-text suspend at kind={step.kind}; ask_until_done cannot answer")
            step = self.step(on_confirm)
        return step

    def accept_until_done(
        self, question: str, *, on_yes_no: Literal["y", "n"] = "y", on_free_text: str = "looks good"
    ) -> SessionStep:
        """Auto-answer yes-or-no and free-text suspends until the turn ends. Intended for sandbox tours and quick demos where every prompt can be confirmed automatically."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        _validate_yes_no_reply_token(on_yes_no, param="on_yes_no")
        step = self.ask(question)
        while not step.done:
            if step.reply_shape == "yes_no":
                step = self.step(on_yes_no)
            elif step.reply_shape == "free_text":
                step = self.step(on_free_text)
            else:
                break
        return step

    def reuse_saved_question(self, question_old: str, question_new: str, new_values: dict[str, Any]) -> SessionStep:
        """
        Re-execute a stored template with caller-supplied bind values.

        Args:

            question_old: Prior question text that identifies the stored template.
            question_new: New natural-language question recorded in value history.
            new_values: Changed bind values keyed by template handles (``p1``, ``s1``, …).

        Returns:

            Terminal :class:`SessionStep` with SQL, data, and parameter bindings.

        Raises:

            SessionActiveError: When another turn is already in flight.
            TypeError: When arguments are not the expected types.
            ConfigError: When no template matches or bind values are invalid.
        """
        if not isinstance(question_old, str) or not isinstance(question_new, str):
            raise TypeError("question_old and question_new must be str")
        if not isinstance(new_values, dict):
            raise TypeError("new_values must be a dict")
        with self._session_busy_lock:
            if self._session_busy:
                raise SessionActiveError("Cannot start a new turn while a prompt is pending.")
            self._session_busy = True
        self._reset_after_turn()
        self._turn_question = question_new
        self._turn_llm_usage_start = len(snapshot_llm_usage_records())
        self._turn_llm_scope_tok = set_turn_llm_scope("question")
        self._audit_ask_emit(
            "ask_begin",
            question=question_new,
            details=(("reuse_saved_question", question_old),),
        )
        owner = self._owner
        schema, store, templates, rejected, _schema_terms = self._resources()
        dialect = owner._dialect
        persist_tl = _persist_template_learning_for_pipeline_session(self)
        gate_kwargs = _consumer_sql_gate_kwargs(self)

        def _run_forced() -> SessionStep:
            with _owner_business_knowledge_scope(owner):
                with llm_execution_scope(owner._runtime_config.llm_execution):
                    try:
                        force_reuse_saved_question(
                            question_old,
                            question_new,
                            new_values,
                            dialect,
                            store,
                            templates,
                            rejected,
                            schema,
                            choice_port=self,
                            persist_template_learning=persist_tl,
                            **gate_kwargs,
                            **_federation_reuse_kwargs(owner, self),
                        )
                    except ConfigError as exc:
                        turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=())
                        self._reset_after_turn()
                        self._release_session_turn()
                        st_err = self._mk_step(
                            done=True,
                            prompt=None,
                            kind=SESSION_KIND_ERROR,
                            error=str(exc),
                            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
                        )
                        return replace(st_err, status=_failure_category_for_terminal_step(st_err))
                    except Exception as exc:
                        debug(f"[main_execution.PipelineSession.reuse_saved_question] unexpected error: {exc!r}")
                        self._reset_after_turn()
                        self._release_session_turn()
                        return self._terminal_error_from_exception(exc)
                    return self._completed_step()

        lock = getattr(owner, "_pipeline_writer_lock", None)
        art = getattr(owner, "_artifacts_dir", None)
        adir = ""
        if art is not None:
            try:
                adir = os.path.abspath(os.fspath(art))
            except (TypeError, OSError, ValueError):
                adir = ""
        if lock is not None:
            with lock:
                if adir and self._session_mode == "writer":
                    drain_write_queue(owner, adir)
                return _run_forced()
        return _run_forced()

    def step(self, response: str | None = None) -> SessionStep:
        """Supply the next user answer for a suspended prompt."""
        if getattr(self._owner, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        buf = diagnostic_segment()
        for _orph in take_and_clear_orphan_diagnostics():
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        try:
            if self._suspended is not None:
                return self._step_pipeline_suspend(response or "")
            if not self._session_busy:
                return self._mk_step(
                    done=True, prompt=None, kind=SESSION_KIND_IDLE, error="No active turn; call ask() first."
                )
            return self._mk_step(
                done=True, prompt=None, kind=SESSION_KIND_ERROR, error="No suspended prompt to answer."
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
                    message=("Reject reason cannot be empty.\n\n" + SESSION_USER_FEEDBACK_BODY),
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
                    message=("Feedback text cannot be empty.\n\n" + SESSION_INTENT_FEEDBACK_BODY),
                )
            self._choice_queue.append((PIPELINE_SUSPEND_ID_INTENT_FEEDBACK, text))
            return self._resume_from_suspend()
        suspended = self._suspended
        if suspended is None:
            return self._mk_step(
                done=True, prompt=None, kind=SESSION_KIND_ERROR, error="No suspended prompt to answer."
            )
        normalised = _normalise_yes_no(raw, ["y", "n"])
        if normalised is None:
            kind = SUSPEND_ID_TO_SESSION_KIND.get(suspended.state_id, SESSION_KIND_ERROR)
            return self._mk_step(
                done=False,
                prompt=SESSION_PROMPT_YESNO,
                kind=kind,
                message="Invalid choice — please answer y or n.",
                reply_shape="yes_no",
            )
        sid = suspended.state_id
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
        interpretation: IntentInterpretation | None = None

        if ex.state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM and isinstance(payload, InteractiveTailSnapshot):
            body, sem_w = compose_intent_confirm_session_message(payload.intent, list(payload.semantic_warnings))
            prompt_out = SESSION_PROMPT_YESNO
            isum = _build_intent_summary(payload.intent)
            interpretation = _intent_interpretation_from_plan(payload.interpretation)
            reply_shape = "yes_no"
        elif ex.state_id == PIPELINE_SUSPEND_ID_EXECUTE and isinstance(payload, SqlExecuteSuspendContext):
            ctx_exec = payload
            sql_out = ctx_exec.sql
            body = ""
            prompt_out = SESSION_PROMPT_YESNO
            isum = _build_intent_summary(ctx_exec.execution_intent)
            reply_shape = "yes_no"
            sem_w = ()
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
                **_federation_result_contract_kwargs(
                    ctxp.gen_out, federated_prepare=ctxp.federated_prepare, federated_bundle=ctxp.federated_bundle
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
            body = SESSION_USER_FEEDBACK_BODY
            prompt_out = SESSION_PROMPT_REASON
            reply_shape = "free_text"
            sem_w = ()
        elif ex.state_id == PIPELINE_SUSPEND_ID_INTENT_FEEDBACK:
            body = (ex.message_for_caller or "").strip() or SESSION_INTENT_FEEDBACK_BODY
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
            interpretation=interpretation,
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
        federated_bundle = snap.get("federated_bundle")
        generation_path = snap.get("generation_path")
        sql_out = _resolved_session_step_sql(
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
            cols_tuple = result_columns_for_session(
                sql_out, list(rows_tuple), **_federation_contract_kwargs_from_snap(snap)
            )
        data_out: pandas.DataFrame | None = None
        if rows_tuple:
            cols_use = (
                list(cols_tuple)
                if cols_tuple
                else result_columns_for_session(
                    sql_out, list(rows_tuple), **_federation_contract_kwargs_from_snap(snap)
                )
            )
            if cols_use:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple], columns=list(cols_use))
            else:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple])
        terminal_notices: tuple[SessionNotice, ...] = ()
        raw_outcome = str(snap.get("outcome") or "success")
        if raw_outcome == "success":
            terminal_message = None
            terminal_status: str | None = None
            terminal_notices = (SessionNotice(code="turn_saved", level="info", message=SAVED_LINE),)
        elif raw_outcome == "permission_denied":
            terminal_message = PERMISSION_DENIED_USER_MESSAGE
            terminal_status = "permission_denied"
            sql_out = None
            data_out = None
        elif raw_outcome == "restricted":
            terminal_message = REPHRASE_HINT_MESSAGES["restricted_question"]
            terminal_status = "restricted"
            sql_out = None
            data_out = None
        elif raw_outcome == "invalid_question":
            terminal_message = REPHRASE_HINT_MESSAGES["vague_question"]
            terminal_status = "invalid_question"
            sql_out = None
            data_out = None
        elif raw_outcome == "parse_failed":
            terminal_message = str(snap.get("error") or REPHRASE_HINT_MESSAGES["intent_parse_failed"])
            terminal_status = FailureCategory.INTENT_PARSE_FAILED.value
            sql_out = None
            data_out = None
        elif raw_outcome == "validation_failed":
            terminal_message = str(snap.get("error") or REPHRASE_HINT_MESSAGES["sql_validation_failed"])
            terminal_status = "validation_failed"
            sql_out = None
            data_out = None
        elif raw_outcome == "schema_invalid_declined":
            terminal_message = REPHRASE_HINT_MESSAGES["schema_invalid_declined"]
            terminal_status = "schema_invalid_declined"
            sql_out = None
            data_out = None
        elif raw_outcome == "federation_partial_failure":
            terminal_message = REPHRASE_HINT_MESSAGES["federation_partial_failure"]
            terminal_status = "federation_partial_failure"
            sql_out = None
            data_out = None
        elif raw_outcome == "user_declined":
            bucket_key = str(snap.get("rejection_bucket") or "OTHER").strip().upper()
            tip = USER_REJECTED_RESULT_BUCKET_TIPS.get(bucket_key, USER_REJECTED_RESULT_BUCKET_TIPS["OTHER"])
            terminal_message = tip
            terminal_status = None
            terminal_notices = (SessionNotice(code="feedback_noted", level="info", message=FEEDBACK_NOTED_LINE),)
        elif raw_outcome == "intent_rejected":
            bucket_key = str(snap.get("rejection_bucket") or "OTHER").strip().upper()
            tip = USER_REJECTED_RESULT_BUCKET_TIPS.get(bucket_key, USER_REJECTED_RESULT_BUCKET_TIPS["OTHER"])
            terminal_message = tip
            terminal_status = FailureCategory.RESULT_OKAY_INTENT_WRONG.value
            terminal_notices = (SessionNotice(code="feedback_noted", level="info", message=FEEDBACK_NOTED_LINE),)
        else:
            terminal_message = str(snap.get("error") or raw_outcome)
            terminal_status = None
        err_snap = snap.get("error")
        ri = snap.get("intent")
        isum_res: IntentSummary | None = None
        if raw_outcome != "permission_denied" and isinstance(ri, RuntimeIntent):
            isum_res = _build_intent_summary(ri)
        if raw_outcome == "permission_denied":
            err_snap = None
        elif raw_outcome == "federation_partial_failure":
            err_snap = None
        parameters = self._parameters_for_completed_turn(snap, qtxt) if raw_outcome == "success" else ()
        turn_diagnostics = self._emit_turn_llm_usage(question=qtxt, diagnostics=())
        refusal_code = snap.get("refusal_diagnostic_code")
        if refusal_code and raw_outcome == "validation_failed":
            refusal_msg = str(snap.get("error") or terminal_message or "")
            turn_diagnostics = turn_diagnostics + (
                Diagnostic(
                    stage="validation",
                    level="error",
                    code=str(refusal_code),
                    message=refusal_msg,
                ),
            )
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
                        message=terminal_message,
                        details=tuple(partial_details),
                        source_id=partial_source,
                    ),
                )
        audit_details: list[tuple[str, str]] = [("outcome", raw_outcome), ("kind", SESSION_KIND_RESULT)]
        if cols_tuple:
            audit_details.append(("result_columns", ",".join(cols_tuple)))
        if federated_bundle is not None:
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
                    ),
                )
        fed_step_fields = _session_step_federation_fields_from_snap(snap, raw_outcome)
        data_out, data_truncated = self._apply_data_row_cap(data_out)
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
            parameters=parameters,
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
            federated_bundle=federated_bundle,
            notices=terminal_notices,
            data_truncated=data_truncated,
            **fed_step_fields,
        )
        self._audit_ask_emit("ask_done", question=qtxt, details=tuple(audit_details))
        self._reset_after_turn()
        self._release_session_turn()
        return step

    def _terminal_error_step(self, message: str, *, exc: BaseException | None = None) -> SessionStep:
        """Build a terminal error :class:`SessionStep`."""
        fed_fields = _federation_error_step_fields(exc) if exc is not None else {}
        fed_diag = _federation_error_diagnostics(exc) if exc is not None else ()
        turn_diagnostics = self._emit_turn_llm_usage(question=self._turn_question, diagnostics=fed_diag)
        audit_details: list[tuple[str, str]] = [("message", message)]
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
        st = self._mk_step(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            error=message,
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
            **fed_fields,
        )
        return replace(st, status=_failure_category_for_terminal_step(st))

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
            resolved = resolve_template_for_question(qtxt, templates, template_store=store)
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
        tmpl_map = templates if isinstance(templates, dict) else store_to_templates(store)
        persist_labels = self._session_mode == "writer" and _persist_template_learning_for_pipeline_session(self)
        return build_parameter_bindings(
            tmpl,
            history_index=row_idx,
            schema=schema,
            question_nl=nl,
            persist_display_names=persist_labels,
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
        ask_phase_token = push_ask_phase_callback(getattr(self._owner, "_ask_phase_callback", None))

        def _run_turn() -> SessionStep:
            identity = getattr(self._owner, "_engine_identity", None)
            if identity is None:
                identity = EngineIdentity(engine_type=str(self._owner.dialect), runtime_config=EngineConfig.RUNTIME)
            identity_token = push_engine_identity(identity)
            sandbox_runtime = getattr(self._owner, "_sandbox_runtime", None)
            sandbox_runtime_token = bind_sandbox_runtime(sandbox_runtime) if sandbox_runtime is not None else None
            with llm_execution_scope(self._owner._runtime_config.llm_execution):
                try:
                    return _run_turn_inner()
                finally:
                    if sandbox_runtime_token is not None:
                        reset_sandbox_runtime(sandbox_runtime_token)
                    pop_engine_identity(identity_token)

        def _run_turn_inner() -> SessionStep:
            try:
                if owner_is_aether_federation(self._owner):
                    members = getattr(self._owner, "_members", None)
                    if isinstance(members, dict) and members:
                        probe_federation_member_liveness(members)
                interactive_run_once(
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
                        error=ex.message_for_caller,
                        diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
                    )
                    return replace(st_e, status=_failure_category_for_terminal_step(st_e))
                self._suspended = ex
                return self._suspend_to_step(ex)
            except Exception as exc:
                debug(f"[main_execution.PipelineSession._drive_question_turn] unexpected error: {exc!r}")
                self._reset_after_turn()
                self._release_session_turn()
                return self._terminal_error_from_exception(exc)
            return self._completed_step()

        try:
            with _owner_business_knowledge_scope(self._owner):
                lock = getattr(self._owner, "_pipeline_writer_lock", None)
                if lock is not None:
                    with lock:
                        if adir and self._session_mode == "reader":
                            _reload_reader_learning_if_manifest_drift(self._owner)
                        if adir and self._session_mode == "writer":
                            drain_write_queue(self._owner, adir)
                        return _run_turn()
                if adir and self._session_mode == "reader":
                    _reload_reader_learning_if_manifest_drift(self._owner)
                return _run_turn()
        finally:
            pop_ask_phase_callback(ask_phase_token)
            pop_session_turn_cancel(cancel_token)

    def _resume_from_suspend(self) -> SessionStep:
        """Continue execution after enqueueing a programmatic answer."""
        if self._suspended is None:
            self._release_session_turn()
            st0 = self._mk_step(done=True, prompt=None, kind=SESSION_KIND_ERROR, error="No pending prompt.")
            return replace(st0, status=_failure_category_for_terminal_step(st0))
        ex = self._suspended
        self._suspended = None
        self._resume_choice_stage_id = ex.state_id

        def _resume_work() -> None:
            with llm_execution_scope(self._owner._runtime_config.llm_execution):
                dispatch_pipeline_resume(self, ex)

        cancel_token = push_session_turn_cancel(self._turn_cancel_event)
        ask_phase_token = push_ask_phase_callback(getattr(self._owner, "_ask_phase_callback", None))
        refinement_retry = False
        try:
            with _owner_business_knowledge_scope(self._owner):
                lock = getattr(self._owner, "_pipeline_writer_lock", None)
                if lock is not None:
                    with lock:
                        _resume_work()
                else:
                    _resume_work()
        except RefinementRetry:
            refinement_retry = True
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
            self._release_session_turn()
            return self._terminal_error_from_exception(exc)
        finally:
            self._resume_choice_stage_id = None
            if not refinement_retry:
                pop_ask_phase_callback(ask_phase_token)
                pop_session_turn_cancel(cancel_token)
        return self._completed_step()


importlib.import_module("aetherdialect._dialect_postgres")
importlib.import_module("aetherdialect._dialect_sqlglot_engines")
register_structural_migration_handler(apply_structural_migration_to_persisted_scopes)
