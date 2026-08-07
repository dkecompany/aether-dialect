"""Entry points for seed warmup, QSim, interactive runs, programmatic PipelineSession, and artifact helpers. Optional ``pyspark.sql.SparkSession`` is imported at module load when available for engine reachability checks. Per-engine dialect modules are imported via ``_dialect_postgres`` and ``_dialect_sqlglot_engines`` so ``register_dialect`` runs before ``DialectRegistry.list_engines()`` is used."""

from __future__ import annotations

import base64
import contextlib
import copy
import glob
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
import tomllib
import zipfile
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import jsonschema
import pandas

from ._config import (
    CsvRuntimeConfig,
    DatabricksRuntimeConfig,
    DuckDBRuntimeConfig,
    EngineConfig,
    EngineLimits,
    EngineRuntimeConfig,
    PolicyConfig,
    QSimConfig,
    SeedWarmupConfig,
)
from ._constants import (
    AETHERSPACE_ARTIFACT_VERSION,
    AETHERSPACES_SEGMENT,
    APPLIED_MAP_ARCHIVE_RETENTION_COUNT,
    APPLIED_MAP_ARCHIVE_TIMESTAMP_RE,
    ARTIFACT_DIR_MODE,
    ARTIFACT_DIRECTORY_SEGMENT,
    ARTIFACT_FILE_MODE,
    ARTIFACT_MANIFEST_FILENAME,
    AUDIT_EVENT_ASK_CANCELLED,
    AUDIT_EVENT_ASK_SUSPEND,
    AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD,
    AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT,
    AZURE_OPENAI_ENV_REQUIRED,
    DIAGNOSTIC_CODE_ARTIFACT_GROWTH,
    DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR,
    DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
    DIAGNOSTIC_CODE_CONFIGURATION_KEY_IGNORED,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED,
    DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE,
    DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT,
    DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE,
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
    DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED,
    DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED,
    DIAGNOSTIC_CODE_LARGE_RESULT_WARNING,
    DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT,
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED,
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION,
    ENGINE_STORAGE_SLUG_MAX_CHARS,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_COMPOSITE_SCHEMA_FILENAME,
    FEDERATION_MIGRATION_MAP_FILENAME,
    FEDERATION_STORAGE_PREFIX,
    FEEDBACK_NOTED_LINE,
    FILE_ENGINE_NAMES,
    INTERACTIVE_STAGE_SQL_FEEDBACK,
    JSON_COMPACT_SEPARATORS,
    KNOWLEDGE_EXPORT_FORMAT_VERSION,
    MASTER_AETHERSPACE_NAME,
    META_ANSWER_FORMAT_VERSION,
    META_ANSWERS_FILENAME,
    META_BUSINESS_KNOWLEDGE_SYSTEM,
    META_DEFAULT_SOURCE_ID,
    META_EMPTY_BUSINESS_KNOWLEDGE_MESSAGE,
    META_KNOWLEDGE_ANSWER_SCHEMA,
    META_SCHEMA_ANSWER_SCHEMA,
    META_SCHEMA_CATALOG_SYSTEM,
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
    REMEDIATION_RESTRICTED_QUESTION,
    REMOVED_BEHAVIOUR_ENVIRONMENT_KEYS,
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
    SESSION_KIND_META,
    SESSION_KIND_RESULT,
    SESSION_PERSISTENCE_FORMAT_VERSION,
    SESSION_PROMPT_REASON,
    SESSION_PROMPT_YESNO,
    SESSION_USER_FEEDBACK_BODY,
    SIMULATION_CACHE_EXACT_FILENAMES,
    SIMULATION_CACHE_GLOB_PATTERNS,
    SUSPEND_ID_TO_SESSION_KIND,
    SUSPEND_STATE_FORMAT_VERSION,
    TABLE_PREVIEW_DEFAULT_LIMIT,
    TABLE_PREVIEW_MAX_LIMIT,
    TEMPLATE_STORE_HEADER_FILENAME,
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
)
from ._contracts_base import (
    AccessError,
    AetherEngineInitResult,
    AetherFederationInitResult,
    AetherSpace,
    BusinessKnowledgeEntry,
    BusinessKnowledgeHolder,
    BusinessKnowledgeState,
    ConfigError,
    DatabaseConnectionError,
    DataQualityReport,
    DescriptionOwner,
    Diagnostic,
    DiagnosticSeverity,
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
    FederationTopologyReport,
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
    PredicateGroup,
    QuestionRoute,
    RefreshReport,
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
    SuspendedSessionExpiredError,
    TablePreviewResult,
    WriteQueueEvent,
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
)
from ._contracts_schema import QSimSummary, SchemaGraph, SeedWarmupSummary, TableMetadata
from ._core_utils import (
    InteractiveChoicePort,
    RephraseHint,
    active_business_knowledge,
    active_business_knowledge_digest,
    active_engine_identity,
    active_engine_runtime_config,
    artifact_lock,
    artifact_manifest_incompatible_with_package,
    bind_construction_orphan_identity,
    business_knowledge_scope,
    debug,
    decode_write_queue_event,
    details_with_turn_id,
    detect_legacy_artifacts,
    diagnostic_debug_enabled,
    diagnostic_segment,
    drain_diagnostic_collector,
    drain_llm_usage_records,
    emit_llm_usage_summary_diagnostics,
    emit_session_refusal_diagnostic,
    failure_kind_is_permission_denied,
    format_versions_match,
    interactive_yes_no,
    invalid_input,
    llm_call_audit_details,
    llm_execution_scope,
    llm_turn_audit_details,
    llm_turn_cost_diagnostic,
    llm_usage_session_scope,
    load_runtime_config,
    manifest_matches_schema,
    mint_turn_id,
    norm_schema_identifier,
    normalize_question,
    note_interactive_turn,
    notes_content_from_context,
    notify,
    permission_denied_detail_logging_enabled,
    pop_ask_phase_callback,
    pop_engine_identity,
    pop_engine_limits,
    pop_session_turn_cancel,
    pop_turn_id,
    print_rephrase_hint,
    progress,
    prompt,
    push_ask_phase_callback,
    push_engine_identity,
    push_engine_limits,
    push_session_turn_cancel,
    push_turn_id,
    read_artifact_manifest,
    reconcile_execute_bind_params,
    refusal_diagnostic_code_for_federation_reason,
    refusal_diagnostic_code_for_outcome,
    refusal_user_text_for_code,
    register_structural_migration_handler,
    release_construction_orphan_identity,
    reset_diagnostic_collector,
    reset_turn_llm_scope,
    sanitize_tenant_slug,
    session_turn_cancelled,
    set_diagnostic_collector,
    set_turn_llm_scope,
    snapshot_llm_usage_records,
    stable_json,
    summarize_llm_turn_usage,
    take_and_clear_orphan_diagnostics,
    terminated,
    try_rename_migration_plan,
    unregister_dialect_live_handles,
    warn_if_artifacts_dir_not_local,
    wipe_filenames,
    wipe_globs,
    wipe_versioned_artifacts,
    write_queue_event_space_name,
)
from ._data_quality import parse_source_selections, validate_upload_sources
from ._dialect import (
    Dialect,
    DialectRegistry,
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
    federation_user_facing_ineligible_message,
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
    purge_departed_federation_member_trees,
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
    resolve_federation_qualified_ref,
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
from ._llm_provider import LLMProvider, MockProvider, SandboxRuntimeState
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
    save_result_csv_for_store,
    stamp_sql_shape,
    try_federation_plan_intake_reuse,
)
from ._qsim import generate_all_intents, generate_all_questions, instantiate_all
from ._schema_build import emit_materialized_view_answer_diagnostics
from ._schema_catalog import (
    emit_description_enrichment_noop,
    extract_business_knowledge_from_notes,
    llm_classify_schema,
)
from ._schema_graph import (
    apply_deny_objects_filter,
    assign_schema_graph_hashes,
    classify_migration_tier,
    compute_schema_limits,
    consumer_graph_is_permission_subset,
    diff_schemas,
    intersect_member_database_feature_capabilities,
    load_schema_graph_snapshot,
    raise_if_schema_unusable,
    strip_schema_context_denied_columns,
    upgrade_artifacts_schema_graph_id,
    validate_scope_against_graph,
)
from ._schema_overrides import apply_overrides_and_persist, build_schema_graph_with_diff, finalize_with_overrides
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
from ._templates import (
    LazyTemplateMapping,
    TemplateOps,
    TemplateRefs,
    TemplateStoreView,
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
from ._validation_schema import assert_execution_parameters_validated


@dataclass(frozen=True, slots=True)
class EngineArtifactState:
    """Per-engine artifact paths registered during construction."""

    schema_json_path: str
    template_store_dir: str


_ENGINE_ARTIFACT_STATES: dict[str, EngineArtifactState] = {}


@dataclass
class _WriteQueueDrainTarget:
    """Live graph and template state for one write-queue drain pass."""

    schema_graph: SchemaGraph
    store: dict[str, Any] | TemplateStoreView
    templates: dict[str, Template] | LazyTemplateMapping
    rejected: dict[str, Any]
    dialect: Any


class MainExecutionOps:
    """Main-execution helpers folded off the module top level (staticmethod-style). Identity-keyed artifacts (see :func:`TemplateOps.orphan_superseded_identity_artifacts`): +----------------------+----------------------------------------------------------+ | Kind                 | Location                                                 | +======================+==========================================================+ | template_shards      | ``intent_templates/spaces/<space>/header.json.gz`` and   | |                      | ``partition_*.json.gz``                                  | | feedback_shards      | ``intent_templates/spaces/<space>/feedback/partition_*`` | | join_feedback        | feedback shard bodies via ``feedback_shard_index``       | | warmup_lattices      | ``anchor_lattice/lattice_<schema_graph_id>_v*.json``     | | qsim_skeletons       | ``qsim_skeletons.json.gz`` (``schema_graph_id`` field)   | | schema_context_cache | ``schema_context.json``                                  | | prompt_cache_refs    | ``schema_context.json`` scope for provider cache keys    | | write_queue          | ``write_queue.jsonl`` (cleared on manifest mismatch)     | | federation_pins      | federation composite manifest ``schema_graph_id``        | +----------------------+----------------------------------------------------------+"""

    @staticmethod
    def register_engine_artifact_state(
        artifacts_dir: str,
        *,
        schema_json_path: str,
        template_store_dir: str,
    ) -> None:
        """Record artifact paths for *artifacts_dir* without mutating global EngineConfig."""
        _ENGINE_ARTIFACT_STATES[os.path.abspath(artifacts_dir)] = EngineArtifactState(
            schema_json_path=schema_json_path,
            template_store_dir=template_store_dir,
        )

    @staticmethod
    def engine_schema_json_path(artifacts_dir: str) -> str:
        """Return the schema graph path for a constructed engine directory."""
        state = _ENGINE_ARTIFACT_STATES.get(os.path.abspath(artifacts_dir))
        if state is not None:
            return state.schema_json_path
        return os.path.join(artifacts_dir, "schema_graph.json.gz")

    @staticmethod
    def engine_template_store_dir(artifacts_dir: str) -> str:
        """Return the template store path for a constructed engine directory."""
        state = _ENGINE_ARTIFACT_STATES.get(os.path.abspath(artifacts_dir))
        if state is not None:
            return state.template_store_dir
        return TemplateOps.template_store_dir_for_space(artifacts_dir, MASTER_AETHERSPACE_NAME)

    @staticmethod
    def _remove_empty_template_shard_files(artifacts_dir: str) -> None:
        spaces_root = os.path.join(TemplateOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT)
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

    @staticmethod
    def _prune_stale_template_shards(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        import gzip

        spaces_root = os.path.join(TemplateOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT)
        if not os.path.isdir(spaces_root):
            return
        active = str(active_schema_graph_id or "")
        if not active:
            return
        for root, _dirs, files in os.walk(spaces_root):
            header_path = os.path.join(root, TEMPLATE_STORE_HEADER_FILENAME)
            if TEMPLATE_STORE_HEADER_FILENAME not in files:
                continue
            header_id = ""
            try:
                with gzip.open(header_path, "rt", encoding="utf-8") as fh:
                    hdr = json.load(fh)
                if isinstance(hdr, dict):
                    header_id = str(hdr.get("schema_graph_id", "") or "")
            except (OSError, json.JSONDecodeError):
                header_id = ""
            if header_id and header_id != active:
                for name in files:
                    if name.startswith(TEMPLATE_STORE_PARTITION_PREFIX):
                        try:
                            os.remove(os.path.join(root, name))
                        except OSError:
                            pass
                try:
                    os.remove(header_path)
                except OSError:
                    pass

    @staticmethod
    def _prune_stale_warmup_lattices(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        lattice_root = os.path.join(artifacts_dir, SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR)
        if not os.path.isdir(lattice_root):
            return
        active = str(active_schema_graph_id or "")
        if not active:
            return
        prefix = "lattice_"
        for name in os.listdir(lattice_root):
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            body = name[len(prefix) :]
            graph_id = body.split("_v", 1)[0] if "_v" in body else ""
            if graph_id and graph_id != active:
                try:
                    os.remove(os.path.join(lattice_root, name))
                except OSError:
                    pass

    @staticmethod
    def _prune_orphaned_federation_trees(federation_parent_dir: str, *, active_fed_dir: str) -> None:
        keep = os.path.abspath(active_fed_dir)
        parent = os.path.abspath(federation_parent_dir)
        if not os.path.isdir(parent):
            return
        for name in os.listdir(parent):
            if not name.startswith(FEDERATION_STORAGE_PREFIX):
                continue
            path = os.path.join(parent, name)
            if not os.path.isdir(path) or os.path.abspath(path) == keep:
                continue
            composite = os.path.join(path, FEDERATION_COMPOSITE_SCHEMA_FILENAME)
            manifest = os.path.join(path, ARTIFACT_MANIFEST_FILENAME)
            if os.path.isfile(composite) or os.path.isfile(manifest):
                continue
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def prune_stale_artifact_auxiliaries(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        """Prune stale template shards, warmup lattices, old applied-map archives, and stale write queues."""
        MainExecutionOps._remove_empty_template_shard_files(artifacts_dir)
        MainExecutionOps._prune_stale_template_shards(artifacts_dir, active_schema_graph_id=active_schema_graph_id)
        MainExecutionOps._prune_stale_warmup_lattices(artifacts_dir, active_schema_graph_id=active_schema_graph_id)
        MainExecutionOps._prune_applied_map_archives(artifacts_dir)
        MainExecutionOps._clear_stale_write_queue(artifacts_dir, active_schema_graph_id=active_schema_graph_id)

    @staticmethod
    def orphan_superseded_identity_artifacts_on_rotation(
        artifacts_dir: str,
        *,
        previous_schema_graph_id: str,
        active_schema_graph_id: str,
    ) -> list[str]:
        """Move every artifact keyed to a superseded schema graph identity into ``orphaned/<id>/``."""
        return TemplateOps.orphan_superseded_identity_artifacts(
            artifacts_dir,
            previous_schema_graph_id=previous_schema_graph_id,
            active_schema_graph_id=active_schema_graph_id,
        )

    @staticmethod
    def _aetherspace_dir(engine_dir: str) -> str:
        return os.path.join(engine_dir, AETHERSPACES_SEGMENT)

    @staticmethod
    def _aetherspace_path(engine_dir: str, name: str) -> str:
        try:
            safe = TemplateOps.validate_space_name(name)
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace name: {name!r}") from exc
        return os.path.join(MainExecutionOps._aetherspace_dir(engine_dir), f"{safe}.json")

    @staticmethod
    def _write_json_atomic(path: str, obj: Any) -> None:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json.tmp", prefix=".aetherspace_", dir=directory, delete=False
            ) as tf:
                tmp_path = tf.name
                json.dump(obj, tf, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=True)
            os.replace(tmp_path, path)
            tmp_path = None
            try:
                os.chmod(path, ARTIFACT_FILE_MODE)
            except OSError:
                pass
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _write_jsonl_atomic(path: str, rows: list[dict[str, Any]]) -> None:
        """Write JSONL rows atomically."""
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
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
            try:
                os.chmod(path, ARTIFACT_FILE_MODE)
            except OSError:
                pass
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
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
                    raise ConfigError(
                        f"SpaceContext columns entry {qc!r} references table {tbl!r} outside tables scope"
                    )
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
                notes=space_context.notes,
            )
        return space_context

    @staticmethod
    def build_master_space_descriptor(schema_graph: SchemaGraph) -> AetherSpace:
        """Return the implicit full-scope ``master`` descriptor derived from *schema_graph*."""
        tables = tuple(sorted(schema_graph.tables.keys()))
        columns: list[str] = []
        for tname in tables:
            tm = schema_graph.tables[tname]
            for col_name in sorted(tm.columns.keys()):
                columns.append(f"{tname}.{col_name}")
        return AetherSpace(
            name=MASTER_AETHERSPACE_NAME, _scope={"tables": tables, "columns": tuple(columns)}, notes=None
        )

    @staticmethod
    def _space_column_resolve_manifest(federation_manifest: FederationManifest | None) -> FederationManifest:
        return federation_manifest or FederationManifest(
            federation_id="",
            sources=(),
            table_namespace={},
            cross_source_joins=(),
            coordinator=FederationCoordinatorConfig(),
        )

    @staticmethod
    def subset_graph_for_space(
        master_graph: SchemaGraph, space_context: SpaceContext, *, federation_manifest: FederationManifest | None = None
    ) -> dict[str, Any]:
        """Build a versioned snapshot dict for persistence from *master_graph* and *space_context*."""
        validated = MainExecutionOps.validate_space_context_against_graph(
            space_context, master_graph, federation_manifest=federation_manifest
        )
        resolve_manifest = MainExecutionOps._space_column_resolve_manifest(federation_manifest)
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

    @staticmethod
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
        path = MainExecutionOps._aetherspace_path(engine_dir, name)
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
        if not format_versions_match(found, AETHERSPACE_ARTIFACT_VERSION):
            raise ConfigError(
                f"aetherspace snapshot at {path!r} has version {found!r}; "
                f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
                f"Delete {path!r} and redefine the aetherspace so it is rewritten "
                f"at the current version."
            )
        if not MainExecutionOps._aetherspace_snapshot_payload_valid(payload):
            return None
        return payload

    @staticmethod
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

    @staticmethod
    def save_aetherspace_snapshot(engine_dir: str, name: str, snapshot: dict[str, Any]) -> str:
        """Persist *snapshot* atomically; return the written path."""
        path = MainExecutionOps._aetherspace_path(engine_dir, name)
        MainExecutionOps._write_json_atomic(path, snapshot)
        return path

    @staticmethod
    def list_saved_aetherspace_names(engine_dir: str) -> tuple[str, ...]:
        """Return sorted saved space names (excluding ``master``)."""
        root = MainExecutionOps._aetherspace_dir(engine_dir)
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

    @staticmethod
    def _aetherspace_export_path(engine_dir: str, name: str) -> str:
        safe = str(name).strip()
        if not safe or safe != safe.strip() or "/" in safe or "\\" in safe:
            raise ConfigError(f"invalid aetherspace name: {name!r}")
        return os.path.join(engine_dir, AETHERSPACES_SEGMENT, "_exports", f"{safe}.export.json")

    @staticmethod
    def _parse_aetherspace_export_payload(payload: Any, *, source_path: str) -> dict[str, Any]:
        """Validate an exported aetherspace JSON document and return a persistable snapshot dict."""
        if not isinstance(payload, dict):
            raise ConfigError(f"malformed aetherspace export at {source_path!r}: expected a JSON object")
        found = payload.get("version")
        if not format_versions_match(found, AETHERSPACE_ARTIFACT_VERSION):
            raise ConfigError(
                f"aetherspace export at {source_path!r} has version {found!r}; "
                f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
                f"Delete the export file and re-export at the current version."
            )
        snap = {key: value for key, value in payload.items() if key != "name"}
        if not MainExecutionOps._aetherspace_snapshot_payload_valid(snap):
            raise ConfigError(f"malformed aetherspace export at {source_path!r}")
        return snap

    @staticmethod
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
        MainExecutionOps.validate_space_context_against_graph(
            space_context,
            schema_graph,
            federation_manifest=federation_manifest,
        )

    @staticmethod
    def export_aetherspace_json(engine_dir: str, name: str, master_graph: SchemaGraph) -> Path:
        """Write a JSON export for *name* and return its path (pair with :func:`apply_aetherspace_json`)."""
        if name == MASTER_AETHERSPACE_NAME:
            snap = MainExecutionOps.subset_graph_for_space(
                master_graph, SpaceContext(tables=frozenset(), columns=frozenset())
            )
            snap["name"] = MASTER_AETHERSPACE_NAME
        else:
            loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, name)
            if loaded is None:
                raise ConfigError(f"unknown aetherspace {name!r}")
            snap = dict(loaded)
            snap["name"] = name
        export_dir = os.path.join(engine_dir, AETHERSPACES_SEGMENT, "_exports")
        os.makedirs(export_dir, exist_ok=True)
        out_path = MainExecutionOps._aetherspace_export_path(engine_dir, name)
        MainExecutionOps._write_json_atomic(out_path, snap)
        return Path(out_path)

    @staticmethod
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
        source_path = (
            os.fspath(source) if source is not None else MainExecutionOps._aetherspace_export_path(engine_dir, norm)
        )
        if not os.path.isfile(source_path):
            raise ConfigError(f"aetherspace export file not found: {source_path}")
        try:
            with open(source_path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConfigError(f"could not read aetherspace export at {source_path!r}: {exc}") from exc
        snap = MainExecutionOps._parse_aetherspace_export_payload(payload, source_path=source_path)
        MainExecutionOps.validate_aetherspace_snapshot_against_graph(
            snap,
            master_graph,
            federation_manifest=federation_manifest,
        )
        MainExecutionOps.save_aetherspace_snapshot(engine_dir, norm, snap)
        return MainExecutionOps.aetherspace_descriptor_from_snapshot(norm, snap)

    @staticmethod
    def delete_aetherspace_snapshot(engine_dir: str, name: str) -> bool:
        """Delete one persisted named aetherspace snapshot. Returns ``True`` when a file was removed."""
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("aetherspace name must be non-empty")
        if norm == MASTER_AETHERSPACE_NAME:
            raise ConfigError("master is the implicit full-scope space and cannot be deleted")
        path = MainExecutionOps._aetherspace_path(engine_dir, norm)
        if not os.path.isfile(path):
            raise ConfigError(f"unknown aetherspace {name!r}")
        os.unlink(path)
        return True

    @staticmethod
    def aetherspace_descriptor_from_snapshot(name: str, snapshot: dict[str, Any]) -> AetherSpace:
        """Build an :class:`AetherSpace` read-only view from a stored snapshot dict."""
        tables_raw = snapshot.get("tables") or ()
        cols_raw = snapshot.get("columns") or ()
        tables = tuple(str(t) for t in tables_raw)
        columns = tuple(str(c) for c in cols_raw)
        notes_raw = snapshot.get("notes")
        notes = str(notes_raw).strip() if isinstance(notes_raw, str) and notes_raw.strip() else None
        return AetherSpace(name=name, _scope={"tables": tables, "columns": columns}, notes=notes)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _normalize_preview_limit(limit: int) -> int:
        if limit < 1:
            raise ConfigError(f"preview limit must be positive, got {limit}")
        return min(int(limit), TABLE_PREVIEW_MAX_LIMIT)

    @staticmethod
    def _resolve_preview_scope_context(owner: Any) -> EngineContext | FederationContext:
        runtime_cfg = getattr(owner, "_runtime_config", None)
        execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
        if execution_context is not None:
            return cast(EngineContext | FederationContext, execution_context)
        if runtime_cfg is not None:
            ctx = getattr(runtime_cfg, "engine_context", None)
            if ctx is not None:
                return cast(EngineContext | FederationContext, ctx)
        return EngineContext()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
            if not MainExecutionOps._preview_column_in_scope(table_name, col_name, scope_ctx):
                continue
            redact = col.sensitivity == SensitivityClassification.RESTRICTED or not col.is_selectable
            out.append((col_name, redact))
        return out

    @staticmethod
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

    @staticmethod
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
        bounded_limit = MainExecutionOps._normalize_preview_limit(limit)
        norm_table = str(table_name).strip()
        if not norm_table:
            raise ConfigError("table_name must be non-empty")
        if norm_table not in schema_graph.tables:
            if schema_role == "consumer":
                raise AccessError("preview_table", PERMISSION_DENIED_USER_MESSAGE, reason="scope")
            raise ConfigError(f"unknown table {table_name!r}")
        if not MainExecutionOps._table_allowed_in_preview_scope(norm_table, schema_graph, scope_ctx, visible_objects):
            raise AccessError("preview_table", PERMISSION_DENIED_USER_MESSAGE, reason="scope")

        table = schema_graph.tables[norm_table]
        preview_columns = MainExecutionOps._preview_columns_for_table(table, norm_table, scope_ctx, schema_graph)
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

        sql = MainExecutionOps._build_preview_sql(dialect, phys_table, select_specs, bounded_limit)
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

    @staticmethod
    def preview_table_on_engine(
        engine: Any,
        table_name: str,
        *,
        limit: int = TABLE_PREVIEW_DEFAULT_LIMIT,
    ) -> TablePreviewResult:
        """Preview one table on a member engine through active scope and redaction."""
        return MainExecutionOps.preview_scoped_table(
            table_name=table_name,
            schema_graph=engine._schema_graph,
            dialect=engine._dialect,
            scope_ctx=MainExecutionOps._resolve_preview_scope_context(engine),
            schema_role=str(getattr(engine, "_schema_role", "owner")),
            visible_objects=getattr(engine, "_consumer_visible_objects", None),
            limit=limit,
        )

    @staticmethod
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
        return MainExecutionOps.preview_scoped_table(
            table_name=table_name,
            schema_graph=federation._schema_graph,
            dialect=member._dialect,
            scope_ctx=MainExecutionOps._resolve_preview_scope_context(federation),
            schema_role=str(getattr(federation, "_schema_role", "owner")),
            visible_objects=getattr(federation, "_consumer_visible_objects", None),
            limit=limit,
            physical_table=physical_table,
            column_physical_names=col_map,
            member_schema_graph=member_schema,
        )

    @staticmethod
    def build_subset_schema_for_space_notes(
        master_graph: SchemaGraph, space_context: SpaceContext, *, federation_manifest: FederationManifest | None = None
    ) -> SchemaGraph:
        """Return a deep-copied in-scope schema graph for notes-aware LLM classification."""
        validated = MainExecutionOps.validate_space_context_against_graph(
            space_context, master_graph, federation_manifest=federation_manifest
        )
        resolve_manifest = MainExecutionOps._space_column_resolve_manifest(federation_manifest)
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
        subset = SchemaGraph(tables=subset_tables, join_paths_multi={})
        deny_ctx = EngineContext(
            deny_objects=frozenset(validated.deny_objects or ()),
            deny_columns=frozenset(validated.deny_columns or ()),
        )
        apply_deny_objects_filter(subset, deny_ctx)
        strip_schema_context_denied_columns(subset, deny_ctx)
        return subset

    @staticmethod
    def merge_business_knowledge(
        engine_entries: Sequence[BusinessKnowledgeEntry],
        space_entries: Sequence[BusinessKnowledgeEntry],
    ) -> tuple[BusinessKnowledgeEntry, ...]:
        """Merge space BK over engine BK by key; space replaces on collision."""
        merged: dict[str, BusinessKnowledgeEntry] = {}
        for entry in engine_entries:
            merged[entry.key] = entry
        for entry in space_entries:
            merged[entry.key] = entry
        return tuple(merged[k] for k in sorted(merged))

    @staticmethod
    def _entries_from_snapshot_business_knowledge(
        snapshot: Mapping[str, Any] | None,
    ) -> tuple[BusinessKnowledgeEntry, ...]:
        if not isinstance(snapshot, Mapping):
            return ()
        raw = snapshot.get("business_knowledge")
        if not isinstance(raw, list):
            return ()
        out: list[BusinessKnowledgeEntry] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            text = str(item.get("text") or "").strip()
            kind = str(item.get("kind") or "glossary").strip() or "glossary"
            if not key or not text or key in seen:
                continue
            try:
                entry = BusinessKnowledgeEntry.normalize(BusinessKnowledgeEntry(key=key, text=text, kind=kind))
            except ConfigError:
                continue
            seen.add(entry.key)
            out.append(entry)
        return tuple(out)

    @staticmethod
    def build_space_knowledge_export(
        *,
        engine_entries: Sequence[BusinessKnowledgeEntry],
        business_knowledge_version: int,
        space: str | None = None,
        space_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build per-space (or master) business-knowledge export (no table inventory)."""
        scope = MASTER_AETHERSPACE_NAME
        entries = tuple(engine_entries)
        if space is not None:
            norm = str(space).strip().lower()
            if not norm:
                raise ConfigError("space name must be non-empty")
            if norm == MASTER_AETHERSPACE_NAME:
                scope = MASTER_AETHERSPACE_NAME
            else:
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
                scope = norm
                space_bk = MainExecutionOps._entries_from_snapshot_business_knowledge(space_snapshot)
                entries = MainExecutionOps.merge_business_knowledge(engine_entries, space_bk)
        return {
            "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
            "scope": scope,
            "business_knowledge_version": int(business_knowledge_version),
            "business_knowledge": [{"key": e.key, "kind": e.kind, "text": e.text} for e in entries],
        }

    @staticmethod
    def build_knowledge_export(
        *,
        engine_entries: Sequence[BusinessKnowledgeEntry],
        business_knowledge_version: int,
        space_snapshots: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build the all-spaces business-knowledge wrapper export."""
        spaces_out: dict[str, dict[str, Any]] = {}
        for name in sorted(space_snapshots):
            space_bk = MainExecutionOps._entries_from_snapshot_business_knowledge(space_snapshots[name])
            spaces_out[name] = {
                "business_knowledge_version": int(business_knowledge_version),
                "business_knowledge": [{"key": e.key, "kind": e.kind, "text": e.text} for e in space_bk],
            }
        return {
            "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
            "engine": {
                "business_knowledge_version": int(business_knowledge_version),
                "business_knowledge": [{"key": e.key, "kind": e.kind, "text": e.text} for e in engine_entries],
            },
            "spaces": spaces_out,
        }

    @staticmethod
    def build_metadata_export(
        *,
        schema_graph: SchemaGraph,
        space: str | None = None,
        space_snapshot: Mapping[str, Any] | None = None,
        federation_members: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build deterministic table/column inventory (and optional federation member roster)."""
        table_desc_overlay: dict[str, str] = {}
        column_desc_overlay: dict[str, str] = {}
        allowed_tables: set[str] | None = None
        if space is not None:
            norm = str(space).strip().lower()
            if norm and norm != MASTER_AETHERSPACE_NAME:
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
                raw_tables = space_snapshot.get("tables")
                if isinstance(raw_tables, (list, tuple)):
                    allowed_tables = {str(t) for t in raw_tables}
                for tname, desc in dict(space_snapshot.get("table_descriptions") or {}).items():
                    if isinstance(desc, str) and desc.strip():
                        table_desc_overlay[str(tname)] = desc.strip()
                for qc, meta in dict(space_snapshot.get("column_meta") or {}).items():
                    if not isinstance(meta, dict):
                        continue
                    desc = meta.get("description")
                    if isinstance(desc, str) and desc.strip():
                        column_desc_overlay[str(qc)] = desc.strip()
        tables_out: list[dict[str, Any]] = []
        for tname in sorted(schema_graph.tables):
            if allowed_tables is not None and tname not in allowed_tables:
                continue
            tbl = schema_graph.tables[tname]
            t_desc = table_desc_overlay.get(tname)
            if t_desc is None:
                t_desc = str(tbl.description or "")
            columns_out: list[dict[str, Any]] = []
            for cname in sorted(tbl.columns):
                col = tbl.columns[cname]
                qc = f"{tname}.{cname}"
                c_desc = column_desc_overlay.get(qc)
                if c_desc is None:
                    c_desc = str(col.description or "")
                columns_out.append(
                    {
                        "name": cname,
                        "data_type": str(col.data_type or ""),
                        "role": str(col.role) if col.role is not None else "",
                        "description": c_desc,
                    }
                )
            tables_out.append({"name": tname, "description": t_desc, "columns": columns_out})
        out: dict[str, Any] = {
            "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
            "table_count": len(tables_out),
            "tables": tables_out,
        }
        if federation_members is not None:
            out["members"] = {
                str(sid): {
                    "tables": sorted(str(t) for t in (info if isinstance(info, (list, tuple, set, frozenset)) else []))
                }
                for sid, info in sorted(federation_members.items(), key=lambda kv: str(kv[0]))
            }
            out["member_count"] = len(out["members"])
        return out

    @staticmethod
    def build_knowledge_layers(
        *,
        engine_entries: Sequence[BusinessKnowledgeEntry],
        space_snapshots: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Deprecated alias — use :meth:`build_knowledge_export`."""
        return MainExecutionOps.build_knowledge_export(
            engine_entries=engine_entries,
            business_knowledge_version=0,
            space_snapshots=space_snapshots,
        )

    @staticmethod
    def enrich_space_snapshot_with_notes(
        snapshot: dict[str, Any],
        master_graph: SchemaGraph,
        space_context: SpaceContext,
        notes_file: str | None = None,
        *,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Bake notes text and content hash into *snapshot*, optionally refining descriptions via LLM."""
        if notes is not None and notes_file is not None:
            raise ConfigError("set at most one of notes and notes_file")
        if notes is not None:
            notes_content = str(notes)
        elif notes_file is not None and str(notes_file).strip():
            path = os.path.expanduser(str(notes_file).strip())
            if not os.path.isfile(path):
                raise ConfigError(f"notes_file not found: {notes_file!r}")
            with open(path, encoding="utf-8") as fh:
                notes_content = fh.read()
        else:
            raise ConfigError("enrich_space_snapshot_with_notes requires notes or notes_file")
        notes_text = notes_content.strip() if notes_content.strip() else None
        notes_hash = hashlib.sha256(notes_content.encode("utf-8")).hexdigest()
        out = dict(snapshot)
        out["notes"] = notes_text
        out["notes_hash"] = notes_hash
        space_bk: tuple[BusinessKnowledgeEntry, ...] = ()
        if notes_text:
            space_bk = extract_business_knowledge_from_notes(notes_content, master_graph)
        out["business_knowledge"] = [{"key": e.key, "kind": e.kind, "text": e.text} for e in space_bk]
        out["business_knowledge_digest"] = BusinessKnowledgeState.digest_for(space_bk)
        if not EngineConfig.llm_credentials_configured():
            return out
        subset_sg = MainExecutionOps.build_subset_schema_for_space_notes(master_graph, space_context)
        classifications = llm_classify_schema(subset_sg, notes_content)
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
                    entry["description_owner"] = DescriptionOwner.SPACE_NOTES.value
                    enriched_any = True
                if sensitivity is not None and str(sensitivity) not in ("", "none"):
                    entry["sensitivity"] = str(sensitivity)
                if entry:
                    column_meta[qc] = entry
        if table_descriptions:
            out["_table_description_owners"] = {str(t): DescriptionOwner.SPACE_NOTES.value for t in table_descriptions}
        out["table_descriptions"] = table_descriptions
        out["column_meta"] = column_meta
        if not enriched_any and notes_text:
            emit_description_enrichment_noop("aetherspace_notes")
        return out

    @staticmethod
    def _remap_qualified_column(
        spec: str, tmap: Mapping[str, str], colmaps: Mapping[str, Mapping[str, str]]
    ) -> str | None:
        raw = str(spec).strip()
        if raw.count(".") != 1:
            return None
        tbl, col = raw.split(".", 1)
        nt = tmap.get(tbl, tbl)
        nc = colmaps.get(tbl, {}).get(col, colmaps.get(nt, {}).get(col, col))
        return f"{nt}.{nc}"

    @staticmethod
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
                remapped = MainExecutionOps._remap_qualified_column(spec, tmap, colmaps)
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

    @staticmethod
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
        tables = MainExecutionOps._prune_remap_string_list(
            [str(t) for t in (out.get("tables") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=False,
        )
        columns = MainExecutionOps._prune_remap_string_list(
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
            remapped = MainExecutionOps._remap_qualified_column(str(key), tmap, colmaps)
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
        out["deny_objects"] = MainExecutionOps._prune_remap_string_list(
            [str(t) for t in (out.get("deny_objects") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=False,
        )
        out["deny_columns"] = MainExecutionOps._prune_remap_string_list(
            [str(c) for c in (out.get("deny_columns") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=True,
        )
        return out

    @staticmethod
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
        root = MainExecutionOps._aetherspace_dir(engine_dir)
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
            edited = MainExecutionOps._apply_structural_edit_to_aetherspace_snapshot(
                payload,
                tmap=tmap,
                colmaps=colmaps,
                drop_tables=drop_tables,
                drop_columns=drop_columns,
                column_retypes=column_retypes,
            )
            if edited != payload:
                MainExecutionOps._write_json_atomic(path, edited)
                updated += 1
        return updated

    @staticmethod
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
                edited[key] = MainExecutionOps._prune_remap_string_list(
                    [str(v) for v in raw_vals],
                    tmap=tmap,
                    colmaps=colmaps,
                    drop_tables=drop_tables,
                    drop_columns=drop_columns,
                    column_specs=column_specs,
                )
            if edited != payload:
                MainExecutionOps._write_json_atomic(path, edited)
                updated += 1
        return updated

    @staticmethod
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
        MainExecutionOps.apply_structural_migration_to_aetherspace_snapshots(
            engine_dir,
            dropped_tables=dropped_tables,
            dropped_columns=dropped_columns,
            table_renames=table_renames,
            column_renames=column_renames,
            column_retypes=column_retypes,
        )
        MainExecutionOps.apply_structural_migration_to_named_context_specs(
            engine_dir,
            dropped_tables=dropped_tables,
            dropped_columns=dropped_columns,
            table_renames=table_renames,
            column_renames=column_renames,
        )

    @staticmethod
    def _normalize_context_name(name: str) -> str:
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("engine context name must be non-empty")
        if "/" in norm or "\\" in norm:
            raise ConfigError(f"invalid engine context name: {name!r}")
        return norm

    @staticmethod
    def _named_schema_context_path(engine_dir: str, name: str) -> str:
        safe = MainExecutionOps._normalize_context_name(name)
        return os.path.join(engine_dir, f"{NAMED_SCHEMA_CONTEXT_PREFIX}{safe}.json")

    @staticmethod
    def _validate_scope_list_fields(payload: dict[str, Any]) -> None:
        for key in ("allow_objects", "deny_objects", "deny_columns", "allow_columns"):
            if key not in payload:
                continue
            val = payload[key]
            if val is None:
                continue
            if not isinstance(val, list):
                raise ConfigError(f"{key} must be a list or null, got {type(val).__name__}")

    @staticmethod
    def _schema_context_from_named_payload(payload: dict[str, Any]) -> EngineContext:
        """Reconstruct a named :class:`EngineContext` from a persisted sidecar."""
        MainExecutionOps._validate_scope_list_fields(payload)
        return EngineContext(
            allow_objects=frozenset(str(x) for x in (payload.get("allow_objects") or ())),
            deny_objects=frozenset(str(x) for x in (payload.get("deny_objects") or ())),
            deny_columns=frozenset(str(x) for x in (payload.get("deny_columns") or ())),
            allow_columns=frozenset(str(x) for x in (payload.get("allow_columns") or ())),
        )

    @staticmethod
    def load_named_schema_context(engine_dir: str, name: str) -> EngineContext | None:
        """
        Load a persisted named context spec, or ``None`` when absent.

        Raises:

            ConfigError: When the sidecar exists but its ``version`` is not
            :data:`NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION`. Delete the stale
            file and re-save the named context; there is no migration path.
        """
        if MainExecutionOps._normalize_context_name(name) == MASTER_AETHERSPACE_NAME:
            return None
        path = MainExecutionOps._named_schema_context_path(engine_dir, name)
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
        if not format_versions_match(found, NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION):
            raise ConfigError(
                f"named schema context at {path!r} has version {found!r}; "
                f"this build expects {NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION}. "
                f"Delete {path!r} and re-save the named context so it is rewritten "
                f"at the current version."
            )
        return MainExecutionOps._schema_context_from_named_payload(payload)

    @staticmethod
    def save_named_schema_context(engine_dir: str, name: str, ctx: EngineContext) -> str:
        """Persist a named allow/deny spec atomically; return the written path."""
        norm = MainExecutionOps._normalize_context_name(name)
        if norm == MASTER_AETHERSPACE_NAME:
            raise ConfigError("master engine context is derived live and is not persisted as a named sidecar")
        payload: dict[str, Any] = {
            "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
            "allow_objects": sorted(ctx.allow_objects),
            "deny_objects": sorted(ctx.deny_objects),
            "deny_columns": sorted(ctx.deny_columns),
            "allow_columns": sorted(ctx.allow_columns),
        }
        path = MainExecutionOps._named_schema_context_path(engine_dir, norm)
        MainExecutionOps._write_json_atomic(path, payload)
        return path

    @staticmethod
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

    @staticmethod
    def export_named_schema_context_json(engine_dir: str, name: str, master_context: EngineContext) -> Path:
        """Write a read-only JSON export for one engine context and return its path."""
        norm = MainExecutionOps._normalize_context_name(name)
        if norm == MASTER_AETHERSPACE_NAME:
            snap: dict[str, Any] = {
                "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
                "name": MASTER_AETHERSPACE_NAME,
                "allow_objects": sorted(master_context.allow_objects),
                "deny_columns": sorted(master_context.deny_columns),
                "allow_columns": sorted(master_context.allow_columns),
            }
        else:
            loaded = MainExecutionOps.load_named_schema_context(engine_dir, norm)
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
        MainExecutionOps._write_json_atomic(out_path, snap)
        return Path(out_path)

    @staticmethod
    def validate_named_engine_context_spec(ctx: EngineContext) -> None:
        """Reject master-only fields on a named engine-context registration spec."""
        if ctx.sql_file is not None:
            raise ConfigError("named engine context cannot set sql_file; only master defines DDL")
        if ctx.notes_file is not None:
            raise ConfigError("named engine context cannot set notes_file; only master defines notes")
        if getattr(ctx, "notes", None) is not None:
            raise ConfigError("named engine context cannot set notes; only master defines notes")
        if ctx.include != "tables":
            raise ConfigError("named engine context cannot set include; only master defines include mode")

    @staticmethod
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

    @staticmethod
    def _federation_execution_allow_objects(
        master_ctx: FederationContext, composite_tables: frozenset[str]
    ) -> frozenset[str]:
        """Intersect federation master allow_objects with composite catalog tables."""
        if master_ctx.allow_objects:
            if composite_tables:
                return frozenset(t for t in master_ctx.allow_objects if t in composite_tables)
            return master_ctx.allow_objects
        return composite_tables

    @staticmethod
    def _effective_execution_context(master: EngineContext, active: EngineContext, active_name: str) -> EngineContext:
        """Combine master and active named context into the execution- time RBAC scope."""
        if MainExecutionOps._normalize_context_name(active_name) == MASTER_AETHERSPACE_NAME:
            return EngineContext(
                allow_objects=master.allow_objects,
                include=master.include,
                deny_objects=master.deny_objects,
                deny_columns=master.deny_columns,
                allow_columns=master.allow_columns,
                notes_file=master.notes_file,
                notes=master.notes,
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

    @staticmethod
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

    @staticmethod
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
        resolve_manifest = MainExecutionOps._space_column_resolve_manifest(federation_manifest)
        allowed_tables = MainExecutionOps.context_allowed_table_set(execution_ctx, schema_graph, mappings=mappings)
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
                if not MainExecutionOps._column_allowed_in_context(tbl, col, trial, schema_graph):
                    raise ConfigError(f"aetherspace column {qc!r} is outside the active engine context scope")

    @staticmethod
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

    @staticmethod
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
            name = MainExecutionOps._normalize_context_name(engine_context)
            if schema_role == "consumer" and name != MASTER_AETHERSPACE_NAME:
                pass
            master = load_master
            if master is None:
                raise ConfigError("create master engine context first; no cached schema_context.json was found")
            if name == MASTER_AETHERSPACE_NAME:
                return master, master, MASTER_AETHERSPACE_NAME
            named = MainExecutionOps.load_named_schema_context(engine_dir, name)
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

    @staticmethod
    def _sync_owner_template_cache(owner: Any, store: Any, *, space_name: str = MASTER_AETHERSPACE_NAME) -> None:
        """Keep the facade template cache aligned with the in-memory store view per aetherspace."""
        norm_space = TemplateOps.validate_space_name(space_name)
        stores_by_space = getattr(owner, "_store_by_space", None)
        if not isinstance(stores_by_space, dict):
            stores_by_space = {}
            owner._store_by_space = stores_by_space
        templates_by_space = getattr(owner, "_templates_by_space", None)
        if not isinstance(templates_by_space, dict):
            templates_by_space = {}
            owner._templates_by_space = templates_by_space
        stores_by_space[norm_space] = store
        templates_by_space[norm_space] = TemplateOps.store_to_templates(store)
        if norm_space == MASTER_AETHERSPACE_NAME:
            owner._store = store
            owner._templates = templates_by_space[norm_space]

    @staticmethod
    def _owner_template_store_for_space(owner: Any, space_name: str) -> Any | None:
        """Return a cached template store for *space_name*, if one is loaded on *owner*."""
        norm_space = TemplateOps.validate_space_name(space_name)
        stores_by_space = getattr(owner, "_store_by_space", None)
        if isinstance(stores_by_space, dict):
            cached = stores_by_space.get(norm_space)
            if cached is not None:
                return cached
        if norm_space == MASTER_AETHERSPACE_NAME:
            return getattr(owner, "_store", None)
        return None

    @staticmethod
    def _persist_template_store(owner: Any | None, store: Any, *, space_name: str = MASTER_AETHERSPACE_NAME) -> None:
        """Flush *store* to disk and refresh *owner*'s cached template map when present."""
        TemplateOps.save_template_store(store)
        if owner is not None:
            MainExecutionOps._sync_owner_template_cache(owner, store, space_name=space_name)

    @staticmethod
    def _owner_from_choice_port(choice_port: InteractiveChoicePort | None) -> Any | None:
        if choice_port is None:
            return None
        return getattr(choice_port, "_owner", None)

    @staticmethod
    def _note_access_error_turn(choice_port: InteractiveChoicePort | None, exc: AccessError) -> None:
        """Map scope vs warehouse ``AccessError`` to the correct interactive outcome."""
        if permission_denied_detail_logging_enabled():
            debug(f"[main_execution] access error detail: {exc!r} reason={getattr(exc, 'reason', None)!r}")
        reason = getattr(exc, "reason", "warehouse")
        owner = MainExecutionOps._owner_from_choice_port(choice_port)
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

    @staticmethod
    def _federation_error_diagnostics(exc: BaseException) -> tuple[Diagnostic, ...]:
        """Build turn diagnostics for a structured federation terminal error."""
        fields = MainExecutionOps._federation_error_step_fields(exc)
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
    def _interactive_attach_refinement_ctx(
        choice_port: InteractiveChoicePort | None, refinement_ctx: RefinementContext
    ) -> None:
        """Bind turn-local refinement state to an interactive session when supported."""
        if choice_port is None:
            return
        attach = getattr(choice_port, "_attach_refinement_ctx", None)
        if callable(attach):
            attach(refinement_ctx)

    @staticmethod
    def _persist_template_learning_for_pipeline_session(port: Any | None) -> bool:
        """Return whether template-store and question-feedback mutations may be written for this choice-port session."""
        if port is None:
            return True
        return getattr(port, "_session_mode", "writer") == "writer"

    @staticmethod
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
        store = TemplateOps.load_template_store(live_graph.schema_graph_id, live_graph)
        templates = TemplateOps.store_to_templates(store)
        owner._store = store
        owner._templates = templates
        finalize_with_overrides(
            owner._schema_graph,
            MainExecutionOps.engine_schema_json_path(str(owner._artifacts_dir)),
            dialect=getattr(owner, "_dialect", None),
        )

    @staticmethod
    def _emit_write_queue_audit(owner: Any, event_type: str, details: tuple[tuple[str, str], ...]) -> None:
        """Forward write-queue drain outcomes to ``owner._audit_emit`` when an audit sink is configured."""
        fn = getattr(owner, "_audit_emit", None)
        if not callable(fn):
            return
        sg = getattr(owner, "_schema_graph", None)
        sh = str(getattr(sg, "effective_structural_hash", "") or "") or None
        fn(event_type, schema_hash=sh, details=details)

    @staticmethod
    def _owner_write_queue_drain_target(owner: Any) -> _WriteQueueDrainTarget:
        return _WriteQueueDrainTarget(
            schema_graph=owner._schema_graph,
            store=owner._store,
            templates=owner._templates,
            rejected=owner._rejected,
            dialect=getattr(owner, "_dialect", None),
        )

    @staticmethod
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
            store = TemplateOps.load_template_store(graph_id, graph, artifacts_dir=str(artifacts_dir))
            targets.append(
                (
                    str(artifacts_dir),
                    _WriteQueueDrainTarget(
                        schema_graph=graph,
                        store=store,
                        templates=TemplateOps.store_to_templates(store),
                        rejected={},
                        dialect=getattr(runtime, "dialect", None),
                    ),
                )
            )
        return targets

    @staticmethod
    def _drain_dispatch_write_queue_event(
        owner: Any, event: WriteQueueEvent, *, target: _WriteQueueDrainTarget | None = None
    ) -> bool:
        """Apply one queue event to *target* stores. Returns True when the template store should be saved."""
        tgt = target or MainExecutionOps._owner_write_queue_drain_target(owner)
        live = str(getattr(tgt.schema_graph, "schema_graph_id", "") or "")
        if not live or event.schema_graph_id != live:
            return False
        store = tgt.store
        templates: dict[str, Template] | LazyTemplateMapping = tgt.templates
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
            TemplateOps.record_question_feedback(store, q_norm, entry)
            MainExecutionOps._emit_write_queue_audit(
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
                templates=cast(dict[str, Any], templates),
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
            MainExecutionOps._emit_write_queue_audit(
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
                notify(
                    "write_queue: malformed intent in replay_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO
                )
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
                cast(dict[str, Any], templates),
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
            MainExecutionOps._emit_write_queue_audit(
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
                apply_overrides_and_persist(
                    schema,
                    tmp_path,
                    schema_json_path=MainExecutionOps.engine_schema_json_path(str(owner._artifacts_dir)),
                )
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
            MainExecutionOps._emit_write_queue_audit(
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

    @staticmethod
    def _write_queue_store_checkpoint(store: dict[str, Any] | TemplateStoreView) -> Any:
        """Capture mutable template-store state before write-queue drain mutations."""
        if isinstance(store, TemplateStoreView):
            return (
                copy.deepcopy(store.feedback_shard_index),
                {pid: copy.deepcopy(part) for pid, part in store._feedback_partition_cache.items()},
                set(store._dirty_feedback_partitions),
                copy.deepcopy(store.partition_map),
                {pid: copy.deepcopy(part) for pid, part in store._partition_cache.items()},
                set(store._dirty_partitions),
                copy.deepcopy(store._indexes),
                int(store.next_id),
            )
        return copy.deepcopy(store)

    @staticmethod
    def _write_queue_store_restore(store: dict[str, Any] | TemplateStoreView, checkpoint: Any) -> None:
        """Restore *store* from a checkpoint produced by :meth:`_write_queue_store_checkpoint`."""
        if isinstance(store, TemplateStoreView):
            (
                feedback_shard_index,
                feedback_cache,
                dirty_feedback,
                partition_map,
                partition_cache,
                dirty_partitions,
                indexes,
                next_id,
            ) = checkpoint
            store.feedback_shard_index = feedback_shard_index
            store._feedback_partition_cache = OrderedDict(feedback_cache)
            store._dirty_feedback_partitions = dirty_feedback
            store.partition_map = partition_map
            store._partition_cache = OrderedDict(partition_cache)
            store._dirty_partitions = dirty_partitions
            store._indexes = indexes
            store.next_id = next_id
            return
        store.clear()
        if isinstance(checkpoint, dict):
            store.update(checkpoint)

    @staticmethod
    def _persist_write_queue_stores(
        owner: Any,
        stores: set[dict[str, Any] | TemplateStoreView],
        *,
        store_spaces: Mapping[int, str] | None = None,
    ) -> None:
        """Flush dirty template stores after a successful write-queue drain batch."""
        for store in stores:
            if isinstance(store, TemplateStoreView):
                space_name = (store_spaces or {}).get(id(store), MASTER_AETHERSPACE_NAME)
                MainExecutionOps._persist_template_store(owner, store, space_name=space_name)
            else:
                TemplateOps.save_template_store(store)
                if store is getattr(owner, "_store", None):
                    MainExecutionOps._sync_owner_template_cache(owner, store)

    @staticmethod
    def _archive_corrupt_write_queue(artifacts_dir: str, path: str) -> str:
        """Move an unparseable write queue aside and return the archive path."""
        ts = datetime.now(UTC).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
        corrupt_name = f"write_queue.corrupt.{ts}.jsonl"
        corrupt_path = os.path.join(artifacts_dir, corrupt_name)
        os.replace(path, corrupt_path)
        return corrupt_path

    @staticmethod
    def _drain_write_queue_at_path(
        owner: Any, artifacts_dir: str, *, target: _WriteQueueDrainTarget | None = None
    ) -> int:
        """Drain one artifact tree's write queue under the artifact lock."""
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
                    corrupt_path = MainExecutionOps._archive_corrupt_write_queue(artifacts_dir, path)
                    notify(
                        f"write queue archived as corrupt: {corrupt_path}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT,
                        details=(("path", corrupt_path),),
                    )
                    return 0
                to_process = head[: cut + 1]
                tail = head[cut + 1 :] + body[limit:]
            else:
                to_process = body
                tail = b""
            text = to_process.decode("utf-8", errors="replace")
            tgt = target or MainExecutionOps._owner_write_queue_drain_target(owner)
            raw_lines = text.splitlines(keepends=True)
            stores_to_save: set[dict[str, Any] | TemplateStoreView] = set()
            store_checkpoints: dict[int, Any] = {}
            store_spaces: dict[int, str] = {}
            pending_suffix: list[str] = []
            for idx, raw_line in enumerate(raw_lines):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    doc = json.loads(stripped)
                except json.JSONDecodeError:
                    notify("write_queue: malformed line skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                    continue
                if not isinstance(doc, dict):
                    continue
                evt = decode_write_queue_event(doc)
                if evt is None:
                    notify("write_queue: unknown event skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                    continue
                explicit_space = write_queue_event_space_name(doc)
                if target is not None and explicit_space is None:
                    event_space = MASTER_AETHERSPACE_NAME
                    event_store: dict[str, Any] | TemplateStoreView = target.store
                else:
                    event_space = explicit_space or MASTER_AETHERSPACE_NAME
                    cached_event_store = MainExecutionOps._owner_template_store_for_space(owner, event_space)
                    if cached_event_store is None:
                        graph = tgt.schema_graph
                        graph_id = str(getattr(graph, "schema_graph_id", "") or "")
                        event_store = TemplateOps.load_template_store(
                            graph_id, graph, artifacts_dir=artifacts_dir, space_name=event_space
                        )
                        MainExecutionOps._sync_owner_template_cache(owner, event_store, space_name=event_space)
                    else:
                        event_store = cached_event_store
                event_templates = TemplateOps.store_to_templates(event_store)
                event_target = _WriteQueueDrainTarget(
                    schema_graph=tgt.schema_graph,
                    store=event_store,
                    templates=event_templates,
                    rejected=tgt.rejected,
                    dialect=tgt.dialect,
                )
                store = event_target.store
                store_key = id(store)
                store_spaces[store_key] = event_space
                if store_key not in store_checkpoints:
                    store_checkpoints[store_key] = MainExecutionOps._write_queue_store_checkpoint(store)
                try:
                    if MainExecutionOps._drain_dispatch_write_queue_event(owner, evt, target=event_target):
                        stores_to_save.add(store)
                    applied += 1
                except Exception as exc:
                    notify(
                        f"write_queue: event dispatch failed: {exc!r}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    pending_suffix = list(raw_lines[idx:])
                    break
            if stores_to_save:
                try:
                    MainExecutionOps._persist_write_queue_stores(owner, stores_to_save, store_spaces=store_spaces)
                except Exception as exc:
                    notify(
                        f"write_queue: store persist failed: {exc!r}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    for store_key, checkpoint in store_checkpoints.items():
                        for store in stores_to_save:
                            if id(store) == store_key:
                                MainExecutionOps._write_queue_store_restore(store, checkpoint)
                                break
                    return 0
            with open(path, "wb") as out:
                if pending_suffix:
                    out.write("".join(pending_suffix).encode("utf-8"))
                out.write(tail)
        return applied

    @staticmethod
    def drain_write_queue(owner: Any, artifacts_dir: str) -> int:
        """Drain deferred reader events under the artifact lock; returns the number of events applied."""
        applied = MainExecutionOps._drain_write_queue_at_path(owner, artifacts_dir)
        seen_dirs = {os.path.abspath(os.fspath(artifacts_dir))}
        for member_dir, member_target in MainExecutionOps._federation_member_write_queue_targets(owner):
            member_abs = os.path.abspath(os.fspath(member_dir))
            if member_abs in seen_dirs:
                continue
            seen_dirs.add(member_abs)
            applied += MainExecutionOps._drain_write_queue_at_path(owner, member_dir, target=member_target)
        return applied

    @staticmethod
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
        MainExecutionOps._raise_if_session_turn_cancelled()
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
        MainExecutionOps._run_interactive_post_intent_parse(
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
    def _intent_interpretation_from_plan(plan: InterpretPlan | None) -> IntentInterpretation | None:
        """Project an :class:`InterpretPlan` into session-step traceability."""
        if plan is None:
            return None
        return IntentInterpretation(approach=plan.approach, grounding=plan.grounding)

    @staticmethod
    def _build_intent_summary(intent: RuntimeIntent) -> IntentSummary:
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
    def _gold_intent_store_path_41_42_blocks_warmup(si: SeedWarmupIntent, templates: dict[str, Template]) -> bool:
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

    @staticmethod
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
    def describe_runtime_config(runtime: RuntimeConfig, llm: LLMConfig) -> str:
        """Build a redacted multi-line snapshot of engine, schema scope, DB, and LLM settings."""
        lines: list[str] = []
        lines.append(f"Engine:          {runtime.engine}")
        lines.append(f"Artifacts dir:   {os.path.abspath(runtime.artifacts_dir)}")
        deny_list = sorted(runtime.engine_context.deny_columns)
        lines.append(f"Schema context:  deny_columns={deny_list!r}")
        runtime_cls = cast(type[EngineRuntimeConfig], DialectRegistry.get_runtime_config_class(runtime.engine))
        try:
            runtime_cfg = active_engine_runtime_config()
        except RuntimeError:
            runtime_cfg = runtime_cls()
        fields = runtime_cfg.connection_slug_fields()
        redacted = runtime_cls.redacted_fields()
        lines.append(f"{runtime.engine}:")
        for key, value in fields.items():
            display = MainExecutionOps._redact_display_value(key, value) if key in redacted else value
            lines.append(f"  {key}: {display}")
        lines.append("LLM:")
        lines.append(f"  provider:   {llm.provider}")
        if llm.provider == "azure":
            base = EngineConfig.azure_base_url() or ""
            lines.append(f"  base_url:   {base}")
            lines.append(
                f"  api_key:    {MainExecutionOps._redact_display_value('api_key', EngineConfig.AZURE_API_TOKEN)}"
            )
        else:
            lines.append(f"  base_url:   {EngineConfig.OPENAI_BASE_URL or ''}")
            lines.append(f"  api_key:    {MainExecutionOps._redact_display_value('api_key', EngineConfig.API_TOKEN)}")
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
        version = MainExecutionOps._get_next_qsim_version(base_dir)
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
        MainExecutionOps._write_json_atomic(qsim_summary_path, summary_entry.to_dict())
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
        MainExecutionOps._write_json_atomic(qsim_trace_path, qsim_trace_payload)
        MainExecutionOps._write_jsonl_atomic(
            qsim_trace_rows_path, intent_trace_rows + instantiation_trace_rows + question_trace_rows
        )

        debug(f"Question simulation complete: {len(results)} questions saved")
        notify(f"QSim version: {version}", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")

        if results and diagnostic_debug_enabled():
            debug("[main_execution.qsim_run_once] samples:")
            for i, item in enumerate(results[:5]):
                debug(f"[main_execution.qsim_run_once]   {i + 1}. {item.question}")

        notify(
            MainExecutionOps.format_qsim_summary_line(summary_entry),
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
        path = MainExecutionOps.resolve_qsim_path(version, artifacts_dir)
        questions = MainExecutionOps._load_questions_from_qsim_txt(path)
        ver = int(version)
        out_path = os.path.join(artifacts_dir, f"qsim_v{ver}_questions.txt")
        MainExecutionOps._get_questions_only(questions, output_path=out_path)

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
            and MainExecutionOps._gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
        ]
        gold_warmup_blocked_path41_or_42 = len(blocked_gold_rows)
        warmup_queue = [
            row
            for row in deduped_pool
            if not MainExecutionOps._gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
        ]
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
        join_cache: dict[JoinCacheKey, JoinCacheEntry] = {}
        for gold in gold_warmup_intents:
            SeedWarmupCacheSession.resolve_joins_for_table_set(
                gold.tables or [], schema, gold.intent_id or "gold", join_cache
            )
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
        notify(
            MainExecutionOps.format_seed_warmup_summary(summary),
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
        gold_warmup_blocked_path41_or_42 = 0
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
                and MainExecutionOps._gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
            ]
            gold_warmup_blocked_path41_or_42 = len(blocked_gold_rows)
            warmup_queue = [
                row
                for row in deduped_pool
                if not MainExecutionOps._gold_intent_store_path_41_42_blocks_warmup(row, tmpl_map)
            ]
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
        notify(
            MainExecutionOps.format_seed_warmup_summary(summary),
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
        MainExecutionOps._run_seed_warmup_sql_history_pipeline(
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
        conn = MainExecutionOps._raw_db_connection_for_query_log(dialect)
        dialect_name = MainExecutionOps._dialect_name_for_query_log(dialect)
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
        MainExecutionOps._run_seed_warmup_sql_history_pipeline(
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
        return tuple(tuple(r) for r in rows[:5])

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
            preview_rows=MainExecutionOps._suspend_preview_rows(rows),
            sql_parameters=MainExecutionOps._freeze_sql_parameters(execution_intent),
            suspended_at=MainExecutionOps._suspend_now(),
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
            preview_rows=MainExecutionOps._suspend_preview_rows(rows),
            sql_parameters=MainExecutionOps._freeze_sql_parameters(execution_intent),
            suspended_at=MainExecutionOps._suspend_now(),
            federated_prepare=federated_prepare,
            federation_plan_id=str(federation_plan_id or gen_out.federation_plan_id or ""),
            federation_exec_context=exec_ctx_pairs,
            turn_policy=turn_policy if turn_policy is not None else MainExecutionOps.snapshot_turn_policy(),
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
            gate_kwargs = MainExecutionOps._consumer_sql_gate_kwargs(choice_port)
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
                    MainExecutionOps._federation_gate_kwargs_by_source(
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
                and MainExecutionOps._persist_template_learning_for_pipeline_session(choice_port)
                and isinstance(fed_manifest, FederationManifest)
            ):
                member_graphs = getattr(owner, "_federation_member_graphs", None)
                if isinstance(member_graphs, dict) and member_graphs:
                    member_stores = MainExecutionOps.federation_stores_by_source(
                        owner,
                        member_graphs,
                        space_name=MainExecutionOps._session_space_name_for_federation(owner, choice_port),
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
        return MainExecutionOps._run_pipeline_sql_rows(
            intent=intent, schema=exec_schema, dialect=exec_dialect, tmpl_sd=tmpl_sd
        ), None

    @staticmethod
    def _run_pipeline_sql_rows(
        *, intent: Any, schema: SchemaGraph, dialect: Any, tmpl_sd: dict[str, Any] | None
    ) -> list[tuple[Any, ...]]:
        """Finalize and execute pipeline SQL, returning row tuples."""
        assert_execution_parameters_validated(intent, schema)
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

    @staticmethod
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
                    trial_rows = MainExecutionOps._run_pipeline_sql_rows(
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
                **MainExecutionOps._federation_result_contract_kwargs(
                    gen_out, federated_prepare=federated_prepare, federated_bundle=federated_bundle
                ),
            )

        sql_prompt = "Is this correct?"
        if need_sql_feedback_prompt:
            if choice_port is not None and not choice_port.has_pending_choice():
                raise PipelineSuspended(
                    PIPELINE_SUSPEND_ID_SQL,
                    sql_prompt,
                    MainExecutionOps._sql_feedback_suspend_context(
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
                MainExecutionOps._persist_template_store(MainExecutionOps._owner_from_choice_port(choice_port), store)
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
                **MainExecutionOps._federation_result_contract_kwargs(
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
            **MainExecutionOps._federation_feedback_kwargs(
                owner, gen_out, choice_port=choice_port, federated_prepare=federated_prepare
            ),
        )
        emit_llm_usage_summary_diagnostics(drain_llm_usage_records())
        row_tuples = [tuple(r) for r in rows]
        cols = result_columns_for_session(
            sql,
            row_tuples,
            intent=intent,
            **MainExecutionOps._federation_result_contract_kwargs(
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
                sql=MainExecutionOps._resolved_session_step_sql(
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
                sql=MainExecutionOps._resolved_session_step_sql(
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

        MainExecutionOps._run_sql_phase_after_intent_confirm(
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
        owner = MainExecutionOps._owner_from_choice_port(choice_port)
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
            space=MainExecutionOps._federation_space_for_choice_port(choice_port),
            member_graphs=(
                getattr(owner, "_federation_member_graphs", None)
                if owner is not None and isinstance(getattr(owner, "_federation_member_graphs", None), dict)
                else None
            ),
        )
        if not plan.ineligible_reason:
            return True
        MainExecutionOps._handle_federation_ineligible_plan(
            plan, choice_port=choice_port, store=store, owner=owner, persist_template_learning=persist_template_learning
        )
        return False

    @staticmethod
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
        space_deny_tables = getattr(choice_port, "space_deny_objects", None)
        space_deny_columns = getattr(choice_port, "space_deny_columns", None)
        context_name = str(getattr(owner, "_context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
        return {
            "schema_role": schema_role,
            "visible_objects": execution_visible_objects,
            "schema_context": scope_ctx,
            "context_name": context_name,
            "space_allowed_tables": space_tables,
            "space_allowed_columns": space_columns,
            "space_deny_tables": space_deny_tables,
            "space_deny_columns": space_deny_columns,
        }

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
        gate_kwargs = MainExecutionOps._consumer_sql_gate_kwargs(choice_port)
        owner = MainExecutionOps._owner_from_choice_port(choice_port)
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
            fed_space = MainExecutionOps._federation_space_for_choice_port(choice_port)
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
                MainExecutionOps._handle_federation_ineligible_plan(
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
                    MainExecutionOps.federation_stores_by_source(
                        owner,
                        member_graphs or {},
                        space_name=MainExecutionOps._session_space_name_for_federation(owner, choice_port),
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
                            MainExecutionOps._federation_gate_kwargs_by_source(
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
            fed_attr = MainExecutionOps._federation_failure_attribution(fed_prep_outcome)
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
                    MainExecutionOps._persist_template_store(
                        MainExecutionOps._owner_from_choice_port(choice_port), store
                    )
                clear_federated_turn_state(choice_port)
                return None
            perm_denied = ek == "explain_permission_denied" or Dialect.is_permission_denied_error(err_text)
            scope_denied = failure_kind_is_permission_denied(ek, err_text)
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
                MainExecutionOps._persist_template_store(MainExecutionOps._owner_from_choice_port(choice_port), store)
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
        is_session = choice_port is not None and isinstance(choice_port, PipelineSession)
        if is_session and choice_port is not None and not choice_port.has_pending_choice():
            raise PipelineSuspended(
                PIPELINE_SUSPEND_ID_EXECUTE,
                MainExecutionOps._federation_execute_confirm_prompt(gen_out, fed_prep_outcome, fed_manifest),
                MainExecutionOps._sql_execute_suspend_context(
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
            rows, federated_bundle = MainExecutionOps._run_sql_execution_for_gen_out(
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
            MainExecutionOps._handle_federation_partial_failure_interactive(choice_port, owner, exc)
            return None
        except FederationTurnCancelledError as exc:
            MainExecutionOps._handle_federation_turn_cancelled_interactive(choice_port, owner, exc)
            return None
        except AccessError as exc:
            MainExecutionOps._note_access_error_turn(choice_port, exc)
            clear_federated_turn_state(choice_port)
            return None
        if len(rows) == 0:
            fixed_intent, fixed_rows = MainExecutionOps.try_zero_row_where_remediation(
                intent, exec_schema, exec_dialect, tmpl_sd
            )
            if fixed_rows is not None:
                intent = fixed_intent
                rows = fixed_rows
        MainExecutionOps._offer_sql_feedback_after_execute(
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
        if not MainExecutionOps._check_federation_eligibility_before_confirm(
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

        MainExecutionOps._run_interactive_join_through_feedback(
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
        MainExecutionOps._run_interactive_after_parsed_intent(
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
        """Continue the interactive pipeline after a parsed intent (joins. through feedback)."""
        MainExecutionOps._run_interactive_after_parsed_intent(
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
        persist_tl = MainExecutionOps._persist_template_learning_for_pipeline_session(choice_port)
        if choice is None or choice != "y":
            note_interactive_turn(choice_port, outcome="user_declined", error="User declined SQL execution.")
            if persist_tl:
                MainExecutionOps._persist_template_store(
                    MainExecutionOps._owner_from_choice_port(choice_port), tail.store
                )
            clear_federated_turn_state(choice_port)
            return None
        execution_intent = ctx.execution_intent
        owner = MainExecutionOps._owner_from_choice_port(choice_port)
        MainExecutionOps._verify_federation_execute_resume(ctx)
        fed_prep = ctx.federated_prepare
        exec_ctx = MainExecutionOps._federation_exec_context_from_pairs(ctx.federation_exec_context)
        federated_bundle: FederatedSqlBundle | None = None
        if fed_prep is not None:
            try:
                rows, federated_bundle = MainExecutionOps._run_sql_execution_for_gen_out(
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
                MainExecutionOps._handle_federation_partial_failure_interactive(choice_port, owner, exc)
                return None
            except FederationTurnCancelledError as exc:
                MainExecutionOps._handle_federation_turn_cancelled_interactive(choice_port, owner, exc)
                return None
            except AccessError as exc:
                MainExecutionOps._note_access_error_turn(choice_port, exc)
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
                single_source = MainExecutionOps._federation_single_source_sql_context(
                    owner, execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
                )
                if single_source is not None:
                    exec_dialect, exec_schema = single_source
            try:
                rows = MainExecutionOps._run_pipeline_sql_rows(
                    intent=execution_intent, schema=exec_schema, dialect=exec_dialect, tmpl_sd=ctx.tmpl_sd
                )
            except AccessError as exc:
                MainExecutionOps._note_access_error_turn(choice_port, exc)
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
                single_source = MainExecutionOps._federation_single_source_sql_context(
                    owner, execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
                )
                if single_source is not None:
                    exec_dialect, exec_schema = single_source
            fixed_intent, fixed_rows = MainExecutionOps.try_zero_row_where_remediation(
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
        emit_explain_soft_diagnostics(getattr(ctx.gen_out, "explain_soft_findings", ()))
        owner = MainExecutionOps._owner_from_choice_port(choice_port)
        MainExecutionOps._offer_sql_feedback_after_execute(
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
        owner = MainExecutionOps._owner_from_choice_port(choice_port)
        fed_prep = ctx.federated_prepare
        federated_bundle: FederatedSqlBundle | None = (
            ctx.federated_bundle if isinstance(ctx, SqlFeedbackSuspendContext) else None
        )
        exec_ctx: dict[str, Any] = {}
        if isinstance(ctx, SqlExecuteSuspendContext):
            exec_ctx = MainExecutionOps._federation_exec_context_from_pairs(ctx.federation_exec_context)
        if fed_prep is not None:
            rows, federated_bundle = MainExecutionOps._run_sql_execution_for_gen_out(
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
            single_source = MainExecutionOps._federation_single_source_sql_context(
                owner, ctx.execution_intent, tail.schema, fed_manifest, fed_mappings, tail.dialect
            )
            if single_source is not None:
                exec_dialect, exec_schema = single_source
        rows = MainExecutionOps._run_pipeline_sql_rows(
            intent=ctx.execution_intent, schema=exec_schema, dialect=exec_dialect, tmpl_sd=ctx.tmpl_sd
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
        persist_tl = MainExecutionOps._persist_template_learning_for_pipeline_session(choice_port)
        if choice is None:
            if persist_tl:
                MainExecutionOps._persist_template_store(
                    MainExecutionOps._owner_from_choice_port(choice_port), tail.store
                )
            return None
        try:
            rows, federated_bundle = MainExecutionOps._reexecute_suspend_sql_rows(ctx, choice_port=choice_port)
        except AccessError as exc:
            MainExecutionOps._note_access_error_turn(choice_port, exc)
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
                    **MainExecutionOps._federation_result_contract_kwargs(
                        ctx.gen_out, federated_prepare=ctx.federated_prepare, federated_bundle=federated_bundle
                    ),
                )
                if df_full is not None:
                    owner = MainExecutionOps._owner_from_choice_port(choice_port)
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
            **MainExecutionOps._federation_feedback_kwargs(
                MainExecutionOps._owner_from_choice_port(choice_port),
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
            **MainExecutionOps._federation_result_contract_kwargs(
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
                sql=MainExecutionOps._resolved_session_step_sql(
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
                sql=MainExecutionOps._resolved_session_step_sql(
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
        persist_tl = MainExecutionOps._persist_template_learning_for_pipeline_session(choice_port)
        if persist_tl:
            TemplateOps.record_question_feedback(tail.store, tail.q_norm, entry)
            MainExecutionOps._persist_template_store(MainExecutionOps._owner_from_choice_port(choice_port), tail.store)
        rb = entry.buckets[0].value if entry.buckets else None
        ctx_ref = getattr(choice_port, "_refinement_ctx", None)
        reason_line = body
        if ctx_ref is not None and refinement_retry_available(ctx_ref):
            ctx_ref.accumulated_reasons.append(reason_line)
            ctx_ref.pending_retry = True
            raise RefinementRetry
        print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT, rejection_bucket=rb)
        note_interactive_turn(choice_port, outcome="user_declined", error="User declined intent confirmation.")

    @staticmethod
    def dispatch_pipeline_resume(session: Any, suspended: PipelineSuspended) -> None:
        """Drive the next pipeline segment after the caller enqueued a. programmatic choice."""
        sid = suspended.state_id
        payload = suspended.payload
        persist_tl = MainExecutionOps._persist_template_learning_for_pipeline_session(session)
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
                    MainExecutionOps._persist_template_store(
                        MainExecutionOps._owner_from_choice_port(session), payload.store
                    )
                return
            clear_planner_schema_invalid_after_user_accept(payload.intent)
            MainExecutionOps._run_interactive_after_parsed_intent_from_tail(
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
            MainExecutionOps._complete_interactive_execute(payload, ch, choice_port=session)
            return
        if sid == PIPELINE_SUSPEND_ID_SQL:
            ch = session._consume_next_queued_choice()
            if not isinstance(payload, SqlFeedbackSuspendContext):
                raise TypeError("SQL feedback resume expects SqlFeedbackSuspendContext")
            MainExecutionOps._complete_interactive_sql_feedback(payload, ch, choice_port=session)
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
            MainExecutionOps._complete_intent_rejection_feedback(payload, ch, session)
            return
        raise RuntimeError(f"unknown pipeline suspend id: {sid!r}")

    @staticmethod
    def _owner_has_federation(owner: Any | None) -> bool:
        return owner is not None and getattr(owner, "_federation_manifest", None) is not None

    @staticmethod
    def _raise_if_session_turn_cancelled() -> None:
        if session_turn_cancelled():
            raise SessionTurnCancelledError("Turn cancelled.")

    @staticmethod
    def build_meta_schema_dump(schema: SchemaGraph) -> dict[str, Any]:
        """Build a filtered schema grounding dump for schema_catalog answers."""
        tables_out: list[dict[str, Any]] = []
        columns_per_table: dict[str, int] = {}
        tables_per_member: dict[str, int] = {}
        member_table_counts: dict[str, int] = {}
        relationships: list[dict[str, str]] = []
        seen_rel: set[tuple[str, str, str]] = set()
        total_columns = 0
        for table_name in sorted(schema.tables):
            tbl = schema.tables[table_name]
            source_id = str(tbl.source_id or "").strip() or META_DEFAULT_SOURCE_ID
            cols_out: list[dict[str, Any]] = []
            for col_name in sorted(tbl.columns):
                col = tbl.columns[col_name]
                if col.is_denied:
                    continue
                if col.sensitivity == SensitivityClassification.HIDDEN:
                    continue
                fk_target = None
                if col.is_foreign_key and col.fk_target is not None:
                    fk_target = f"{col.fk_target[0]}.{col.fk_target[1]}"
                cols_out.append(
                    {
                        "name": col_name,
                        "data_type": str(col.data_type or ""),
                        "value_type": str(col.value_type or ""),
                        "role": str(col.role or ""),
                        "description": str(col.description or ""),
                        "is_primary_key": bool(col.is_primary_key),
                        "is_foreign_key": bool(col.is_foreign_key),
                        "fk_target": fk_target,
                    }
                )
            if not cols_out and not tbl.columns:
                continue
            if not cols_out:
                continue
            tables_out.append(
                {
                    "name": table_name,
                    "source_id": source_id,
                    "description": str(tbl.description or ""),
                    "columns": cols_out,
                    "primary_key": list(tbl.primary_key or []),
                    "foreign_keys": [
                        {
                            "src_cols": list(fk.src_cols),
                            "dst_table": fk.dst_table,
                            "dst_cols": list(fk.dst_cols),
                        }
                        for fk in (tbl.foreign_keys or [])
                    ],
                }
            )
            columns_per_table[table_name] = len(cols_out)
            total_columns += len(cols_out)
            member_table_counts[source_id] = member_table_counts.get(source_id, 0) + 1
            for fk in tbl.foreign_keys or []:
                for src_c, dst_c in zip(fk.src_cols, fk.dst_cols, strict=False):
                    left = f"{fk.src_table}.{src_c}"
                    right = f"{fk.dst_table}.{dst_c}"
                    kind = "semantic" if str(fk.join_kind or "").lower() == "semantic" else "fk"
                    key = (left, right, kind)
                    if key in seen_rel:
                        continue
                    seen_rel.add(key)
                    relationships.append({"left": left, "right": right, "kind": kind})
            for col_name, col in tbl.columns.items():
                if col.is_denied or col.sensitivity == SensitivityClassification.HIDDEN:
                    continue
                for nb_table, nb_col in col.semantic_join_neighbors or ():
                    left = f"{table_name}.{col_name}"
                    right = f"{nb_table}.{nb_col}"
                    key = (left, right, "semantic")
                    if key in seen_rel:
                        continue
                    seen_rel.add(key)
                    relationships.append({"left": left, "right": right, "kind": "semantic"})
        tables_per_member = dict(sorted(member_table_counts.items()))
        members = [{"source_id": sid, "table_count": ct} for sid, ct in tables_per_member.items()]
        if not members:
            members = [{"source_id": META_DEFAULT_SOURCE_ID, "table_count": 0}]
            tables_per_member = {META_DEFAULT_SOURCE_ID: 0}
        return {
            "inventory": {
                "table_count": len(tables_out),
                "column_count": total_columns,
                "member_count": len(tables_per_member),
                "columns_per_table": columns_per_table,
                "tables_per_member": tables_per_member,
            },
            "members": members,
            "tables": tables_out,
            "relationships": relationships,
        }

    @staticmethod
    def validate_meta_schema_answer(answer: dict[str, Any], dump: dict[str, Any]) -> None:
        """Validate a schema_catalog LLM answer against JSON Schema and dump grounding rules."""
        try:
            jsonschema.validate(instance=answer, schema=META_SCHEMA_ANSWER_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"meta schema answer failed JSON Schema: {exc.message}") from exc
        if answer.get("response_kind") != "schema_catalog":
            raise ValueError("response_kind must be schema_catalog")
        inventory = dump.get("inventory") or {}
        dump_tables = {str(t["name"]): t for t in dump.get("tables") or [] if isinstance(t, dict)}
        dump_members = {str(m["source_id"]) for m in dump.get("members") or [] if isinstance(m, dict)}
        dump_rels = {
            (str(r.get("left")), str(r.get("right")), str(r.get("kind")))
            for r in dump.get("relationships") or []
            if isinstance(r, dict)
        }
        for tbl in answer.get("tables") or []:
            name = str(tbl.get("name") or "")
            if name not in dump_tables:
                raise ValueError(f"invented table not in schema dump: {name}")
            dump_tbl = dump_tables[name]
            if str(tbl.get("source_id") or "") != str(dump_tbl.get("source_id") or ""):
                raise ValueError(f"source_id mismatch for table {name}")
            dump_cols = {str(c["name"]): c for c in dump_tbl.get("columns") or []}
            for col in tbl.get("columns") or []:
                cname = str(col.get("name") or "")
                if cname not in dump_cols:
                    raise ValueError(f"invented column not in schema dump: {name}.{cname}")
        for rel in answer.get("relationships") or []:
            kind = str(rel.get("kind") or "")
            if kind not in ("fk", "semantic"):
                raise ValueError(f"invalid relationship kind: {kind}")
            left = str(rel.get("left") or "")
            right = str(rel.get("right") or "")
            if (left, right, kind) not in dump_rels and (right, left, kind) not in dump_rels:
                ok_endpoints = False
                for dl, dr, dk in dump_rels:
                    if dk != kind:
                        continue
                    if {dl, dr} == {left, right}:
                        ok_endpoints = True
                        break
                if not ok_endpoints:
                    raise ValueError(f"invented relationship not in schema dump: {left}->{right}")
        counts = answer.get("counts") or {}
        if counts.get("tables") is not None and int(counts["tables"]) != int(inventory.get("table_count") or 0):
            raise ValueError("counts.tables must equal inventory.table_count")
        if counts.get("columns") is not None and int(counts["columns"]) != int(inventory.get("column_count") or 0):
            raise ValueError("counts.columns must equal inventory.column_count")
        if counts.get("members") is not None and int(counts["members"]) != int(inventory.get("member_count") or 0):
            raise ValueError("counts.members must equal inventory.member_count")
        cit = counts.get("columns_in_table")
        if isinstance(cit, dict):
            tname = str(cit.get("table") or "")
            if tname not in (inventory.get("columns_per_table") or {}):
                raise ValueError(f"columns_in_table.table not in dump: {tname}")
            if int(cit.get("columns") or -1) != int((inventory.get("columns_per_table") or {})[tname]):
                raise ValueError("columns_in_table.columns must equal inventory.columns_per_table")
        tim = counts.get("tables_in_member")
        if isinstance(tim, dict):
            sid = str(tim.get("source_id") or "")
            if sid not in dump_members:
                raise ValueError(f"tables_in_member.source_id not in dump: {sid}")
            if int(tim.get("tables") or -1) != int((inventory.get("tables_per_member") or {}).get(sid, -2)):
                raise ValueError("tables_in_member.tables must equal inventory.tables_per_member")

    @staticmethod
    def format_meta_schema_message(answer: dict[str, Any]) -> str:
        """Render a schema_catalog answer as deterministic plain text."""
        lines: list[str] = [str(answer.get("headline") or "").strip()]
        counts = answer.get("counts") or {}
        count_lines: list[str] = []
        if counts.get("tables") is not None:
            count_lines.append(f"tables: {counts['tables']}")
        if counts.get("columns") is not None:
            count_lines.append(f"columns: {counts['columns']}")
        if counts.get("members") is not None:
            count_lines.append(f"members: {counts['members']}")
        cit = counts.get("columns_in_table")
        if isinstance(cit, dict):
            count_lines.append(f"columns in {cit.get('table')}: {cit.get('columns')}")
        tim = counts.get("tables_in_member")
        if isinstance(tim, dict):
            count_lines.append(f"tables in {tim.get('source_id')}: {tim.get('tables')}")
        if count_lines:
            lines.append("")
            lines.extend(count_lines)
        tables = answer.get("tables") or []
        if tables:
            lines.append("")
            lines.append("tables:")
            for tbl in tables:
                lines.append(f"- {tbl.get('name')} ({tbl.get('source_id')}): {tbl.get('description') or ''}".rstrip())
                for col in tbl.get("columns") or []:
                    lines.append(
                        f"  - {col.get('name')} {col.get('data_type')} "
                        f"[{col.get('role')}] {col.get('description') or ''}".rstrip()
                    )
        rels = answer.get("relationships") or []
        if rels:
            lines.append("")
            lines.append("relationships:")
            for rel in rels:
                lines.append(f"- {rel.get('left')} -> {rel.get('right')} ({rel.get('kind')})")
        notes = [str(n) for n in (answer.get("notes") or []) if str(n).strip()]
        if notes:
            lines.append("")
            lines.append("notes:")
            for note in notes:
                lines.append(f"- {note}")
        return "\n".join(lines).strip()

    @staticmethod
    def _meta_cache_space_name(space_overlay: Any) -> str:
        if space_overlay is None:
            return ""
        if isinstance(space_overlay, str):
            return space_overlay.strip()
        name = getattr(space_overlay, "name", None) or getattr(space_overlay, "space_name", None)
        return str(name or "").strip()

    @staticmethod
    def _meta_cache_federation_id(owner: Any, schema: SchemaGraph | None) -> str:
        if schema is not None and isinstance(schema.federation_membership, dict):
            fed = str(schema.federation_membership.get("federation_id") or "").strip()
            if fed:
                return fed
        manifest = getattr(owner, "_federation_manifest", None) if owner is not None else None
        if manifest is not None:
            fed = str(getattr(manifest, "federation_id", "") or "").strip()
            if fed:
                return fed
        return ""

    @staticmethod
    def _meta_cache_schema_graph_id(schema: SchemaGraph | None) -> str:
        if schema is None:
            return ""
        return str(getattr(schema, "schema_graph_id", "") or "").strip()

    @staticmethod
    def _meta_cache_bk_digest(owner: Any) -> str:
        digest = active_business_knowledge_digest()
        if digest:
            return digest
        holder = getattr(owner, "_business_knowledge", None) if owner is not None else None
        if isinstance(holder, BusinessKnowledgeHolder):
            return str(holder.digest() or "").strip()
        return ""

    @staticmethod
    def meta_answer_cache_key(
        *,
        schema_graph_id: str,
        federation_id: str,
        space_name: str,
        business_knowledge_digest: str,
        corrected_question: str,
        route: str,
    ) -> str:
        """Return the sha256 hex cache key for a metadata answer."""
        material = "|".join(
            (
                str(schema_graph_id or ""),
                str(federation_id or ""),
                str(space_name or ""),
                str(business_knowledge_digest or ""),
                str(corrected_question or ""),
                str(route or ""),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _meta_answers_path(artifacts_dir: str) -> str:
        return os.path.join(os.path.abspath(artifacts_dir), META_ANSWERS_FILENAME)

    @staticmethod
    def load_meta_answer_cache(artifacts_dir: str | None) -> dict[str, Any]:
        """Load ``meta_answers.json`` or return an empty versioned document."""
        empty: dict[str, Any] = {"meta_answer_format_version": META_ANSWER_FORMAT_VERSION, "entries": {}}
        if not artifacts_dir:
            return empty
        path = MainExecutionOps._meta_answers_path(artifacts_dir)
        if not os.path.isfile(path):
            return empty
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return empty
        if not isinstance(payload, dict):
            return empty
        if not format_versions_match(payload.get("meta_answer_format_version"), META_ANSWER_FORMAT_VERSION):
            return empty
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return empty
        return {"meta_answer_format_version": META_ANSWER_FORMAT_VERSION, "entries": dict(entries)}

    @staticmethod
    def save_meta_answer_cache(artifacts_dir: str | None, cache: dict[str, Any]) -> None:
        """Persist ``meta_answers.json`` when *artifacts_dir* is set."""
        if not artifacts_dir:
            return
        payload = {
            "meta_answer_format_version": META_ANSWER_FORMAT_VERSION,
            "entries": dict(cache.get("entries") or {}),
        }
        MainExecutionOps._write_json_atomic(MainExecutionOps._meta_answers_path(artifacts_dir), payload)

    @staticmethod
    def _lookup_meta_answer_cache(
        artifacts_dir: str | None,
        *,
        schema: SchemaGraph | None,
        owner: Any,
        space_overlay: Any,
        corrected: str,
        route: QuestionRoute,
    ) -> SessionStep | None:
        if not artifacts_dir:
            return None
        key = MainExecutionOps.meta_answer_cache_key(
            schema_graph_id=MainExecutionOps._meta_cache_schema_graph_id(schema),
            federation_id=MainExecutionOps._meta_cache_federation_id(owner, schema),
            space_name=MainExecutionOps._meta_cache_space_name(space_overlay),
            business_knowledge_digest=MainExecutionOps._meta_cache_bk_digest(owner),
            corrected_question=corrected,
            route=route.value,
        )
        cache = MainExecutionOps.load_meta_answer_cache(artifacts_dir)
        entry = (cache.get("entries") or {}).get(key)
        if not isinstance(entry, dict):
            return None
        message = entry.get("message")
        kind = str(entry.get("kind") or SESSION_KIND_META)
        meta_payload = entry.get("meta_payload")
        if not isinstance(message, str) or not message.strip():
            return None
        if meta_payload is not None and not isinstance(meta_payload, dict):
            return None
        notify("Metadata cache hit", stage="meta", code="meta.cache.hit", level="info")
        return SessionStep(
            done=True,
            prompt=None,
            kind=kind,
            sql=None,
            data=None,
            message=message,
            meta_payload=dict(meta_payload) if isinstance(meta_payload, dict) else None,
            diagnostics=(),
            notices=(),
            intent_summary=None,
            semantic_warnings=(),
            status=None,
            error=None,
            federated_bundle=None,
            parameters=(),
        )

    @staticmethod
    def _store_meta_answer_cache(
        artifacts_dir: str | None,
        *,
        schema: SchemaGraph | None,
        owner: Any,
        space_overlay: Any,
        corrected: str,
        route: QuestionRoute,
        step: SessionStep,
    ) -> None:
        if not artifacts_dir or step.kind != SESSION_KIND_META or step.error is not None:
            return
        if not isinstance(step.message, str) or not step.message.strip():
            return
        key = MainExecutionOps.meta_answer_cache_key(
            schema_graph_id=MainExecutionOps._meta_cache_schema_graph_id(schema),
            federation_id=MainExecutionOps._meta_cache_federation_id(owner, schema),
            space_name=MainExecutionOps._meta_cache_space_name(space_overlay),
            business_knowledge_digest=MainExecutionOps._meta_cache_bk_digest(owner),
            corrected_question=corrected,
            route=route.value,
        )
        cache = MainExecutionOps.load_meta_answer_cache(artifacts_dir)
        entries = dict(cache.get("entries") or {})
        entries[key] = {
            "message": step.message,
            "meta_payload": dict(step.meta_payload) if isinstance(step.meta_payload, dict) else None,
            "kind": step.kind,
        }
        cache["entries"] = entries
        MainExecutionOps.save_meta_answer_cache(artifacts_dir, cache)

    @staticmethod
    def _resolve_active_business_knowledge_entries(owner: Any) -> tuple[BusinessKnowledgeEntry, ...]:
        """Return scoped business knowledge, falling back to the owner's holder."""
        active = active_business_knowledge()
        if active:
            return active
        holder = getattr(owner, "_business_knowledge", None) if owner is not None else None
        if isinstance(holder, BusinessKnowledgeHolder):
            return holder.entries()
        return ()

    @staticmethod
    def _answer_business_knowledge_question(
        owner: Any,
        corrected: str,
        *,
        schema: SchemaGraph | None = None,
        space_overlay: Any = None,
        artifacts_dir: str | None = None,
    ) -> SessionStep:
        """Answer a business_knowledge route from the active knowledge list."""
        route = QuestionRoute.BUSINESS_KNOWLEDGE
        cached = MainExecutionOps._lookup_meta_answer_cache(
            artifacts_dir,
            schema=schema,
            owner=owner,
            space_overlay=space_overlay,
            corrected=corrected,
            route=route,
        )
        if cached is not None:
            return cached
        entries = MainExecutionOps._resolve_active_business_knowledge_entries(owner)
        payload_entries = [{"key": e.key, "kind": e.kind, "text": e.text} for e in entries]
        if not payload_entries:
            step = SessionStep(
                done=True,
                prompt=None,
                kind=SESSION_KIND_META,
                sql=None,
                data=None,
                message=META_EMPTY_BUSINESS_KNOWLEDGE_MESSAGE,
                meta_payload={"response_kind": "business_knowledge"},
                diagnostics=(),
                notices=(),
                intent_summary=None,
                semantic_warnings=(),
                status=None,
                error=None,
                federated_bundle=None,
                parameters=(),
            )
            MainExecutionOps._store_meta_answer_cache(
                artifacts_dir,
                schema=schema,
                owner=owner,
                space_overlay=space_overlay,
                corrected=corrected,
                route=route,
                step=step,
            )
            return step
        notify("Metadata cache miss", stage="meta", code="meta.cache.miss", level="info")
        user = stable_json({"question": corrected, "business_knowledge": payload_entries})
        answer: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    raw = LLMProvider.json(META_BUSINESS_KNOWLEDGE_SYSTEM, user, task="default")
                else:
                    notify("Metadata answer repair", stage="meta", code="meta.answer.repair", level="info")
                    repair_user = stable_json(
                        {
                            "question": corrected,
                            "business_knowledge": payload_entries,
                            "previous_answer": answer,
                            "error": last_error,
                        }
                    )
                    raw = LLMProvider.json(META_BUSINESS_KNOWLEDGE_SYSTEM, repair_user, task="default")
                if not isinstance(raw, dict):
                    raise ValueError("business knowledge answer must be a JSON object")
                answer = raw
                jsonschema.validate(instance=answer, schema=META_KNOWLEDGE_ANSWER_SCHEMA)
                if answer.get("response_kind") != "business_knowledge":
                    raise ValueError("response_kind must be business_knowledge")
                message = str(answer.get("message") or "").strip()
                if not message:
                    raise ValueError("business knowledge message must be non-empty")
                notify("Metadata answer validated", stage="meta", code="meta.answer.validated", level="info")
                step = SessionStep(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_META,
                    sql=None,
                    data=None,
                    message=message,
                    meta_payload={"response_kind": "business_knowledge"},
                    diagnostics=(),
                    notices=(),
                    intent_summary=None,
                    semantic_warnings=(),
                    status=None,
                    error=None,
                    federated_bundle=None,
                    parameters=(),
                )
                MainExecutionOps._store_meta_answer_cache(
                    artifacts_dir,
                    schema=schema,
                    owner=owner,
                    space_overlay=space_overlay,
                    corrected=corrected,
                    route=route,
                    step=step,
                )
                return step
            except (ValueError, TypeError, jsonschema.ValidationError) as exc:
                last_error = str(exc)
                continue
        notify(
            f"Metadata answer failed: {last_error or 'unknown'}",
            stage="meta",
            code="meta.answer.failed",
            level="error",
        )
        return SessionStep(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            sql=None,
            message=None,
            error=last_error or "metadata answer failed",
            status=FailureCategory.META_ERROR.value,
            meta_payload=None,
            diagnostics=(
                Diagnostic(
                    stage="meta",
                    level=DiagnosticSeverity.ERROR,
                    code="meta.answer.failed",
                    message=last_error or "metadata answer failed",
                    phase="meta",
                ),
            ),
        )

    @staticmethod
    def answer_metadata_question(
        owner: Any,
        corrected: str,
        route: QuestionRoute | str,
        schema: SchemaGraph | None,
        space_overlay: Any = None,
        artifacts_dir: str | None = None,
    ) -> SessionStep:
        """Answer a schema_catalog or business_knowledge question without SQL generation."""
        route_enum = route if isinstance(route, QuestionRoute) else QuestionRoute(str(route))
        if route_enum == QuestionRoute.BUSINESS_KNOWLEDGE:
            return MainExecutionOps._answer_business_knowledge_question(
                owner,
                corrected,
                schema=schema,
                space_overlay=space_overlay,
                artifacts_dir=artifacts_dir,
            )
        if schema is None:
            notify("Metadata answer failed: schema missing", stage="meta", code="meta.answer.failed", level="error")
            return SessionStep(
                done=True,
                prompt=None,
                kind=SESSION_KIND_ERROR,
                sql=None,
                message=None,
                error="schema missing for metadata answer",
                status=FailureCategory.META_ERROR.value,
                meta_payload=None,
                diagnostics=(
                    Diagnostic(
                        stage="meta",
                        level=DiagnosticSeverity.ERROR,
                        code="meta.answer.failed",
                        message="schema missing for metadata answer",
                        phase="meta",
                    ),
                ),
            )
        cached = MainExecutionOps._lookup_meta_answer_cache(
            artifacts_dir,
            schema=schema,
            owner=owner,
            space_overlay=space_overlay,
            corrected=corrected,
            route=route_enum,
        )
        if cached is not None:
            return cached
        dump = MainExecutionOps.build_meta_schema_dump(schema)
        notify("Metadata cache miss", stage="meta", code="meta.cache.miss", level="info")
        user_payload = {"question": corrected, "schema": dump}
        user = stable_json(user_payload)
        answer: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    raw = LLMProvider.json(META_SCHEMA_CATALOG_SYSTEM, user, task="default")
                else:
                    notify("Metadata answer repair", stage="meta", code="meta.answer.repair", level="info")
                    repair_user = stable_json(
                        {
                            "question": corrected,
                            "schema": dump,
                            "previous_answer": answer,
                            "error": last_error,
                        }
                    )
                    raw = LLMProvider.json(META_SCHEMA_CATALOG_SYSTEM, repair_user, task="default")
                if not isinstance(raw, dict):
                    raise ValueError("metadata answer must be a JSON object")
                answer = raw
                MainExecutionOps.validate_meta_schema_answer(answer, dump)
                notify("Metadata answer validated", stage="meta", code="meta.answer.validated", level="info")
                message = MainExecutionOps.format_meta_schema_message(answer)
                step = SessionStep(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_META,
                    sql=None,
                    data=None,
                    message=message,
                    meta_payload=dict(answer),
                    diagnostics=(),
                    notices=(),
                    intent_summary=None,
                    semantic_warnings=(),
                    status=None,
                    error=None,
                    federated_bundle=None,
                    parameters=(),
                )
                MainExecutionOps._store_meta_answer_cache(
                    artifacts_dir,
                    schema=schema,
                    owner=owner,
                    space_overlay=space_overlay,
                    corrected=corrected,
                    route=route_enum,
                    step=step,
                )
                return step
            except (ValueError, TypeError, jsonschema.ValidationError) as exc:
                last_error = str(exc)
                continue
        notify(
            f"Metadata answer failed: {last_error or 'unknown'}",
            stage="meta",
            code="meta.answer.failed",
            level="error",
        )
        return SessionStep(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            sql=None,
            message=None,
            error=last_error or "metadata answer failed",
            status=FailureCategory.META_ERROR.value,
            meta_payload=None,
            diagnostics=(
                Diagnostic(
                    stage="meta",
                    level=DiagnosticSeverity.ERROR,
                    code="meta.answer.failed",
                    message=last_error or "metadata answer failed",
                    phase="meta",
                ),
            ),
        )

    @staticmethod
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
        MainExecutionOps._raise_if_session_turn_cancelled()
        progress("\nValidating question...")

        raw_question = question

        owner_dialect = None
        owner = getattr(pipeline_session, "_owner", None) if pipeline_session is not None else None
        if owner is not None:
            owner_dialect = getattr(owner, "_dialect", None)
        fed_reuse_kwargs = MainExecutionOps._federation_reuse_kwargs(owner, pipeline_session)

        dialect, schema, store, templates, rejected, schema_terms = load_pipeline_resources(
            schema, store, templates, rejected, schema_terms, dialect=owner_dialect
        )
        choice_port: InteractiveChoicePort | None = pipeline_session
        persist_tl = MainExecutionOps._persist_template_learning_for_pipeline_session(choice_port)
        gate_kwargs = MainExecutionOps._consumer_sql_gate_kwargs(choice_port)

        tmpl_pre = match_question_level_template_reuse(raw_question, templates, template_store=store)
        if tmpl_pre.reuse_type == "direct_reuse" and not MainExecutionOps._owner_has_federation(owner):
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

        validation = validate_question(raw_question)
        if not validation.accepted:
            if validation.route == QuestionRoute.RESTRICTED:
                restricted_code = refusal_diagnostic_code_for_outcome("restricted")
                restricted_message = refusal_user_text_for_code(restricted_code or "")
                notify(
                    f"\n{restricted_message}",
                    stage="rephrase_hint",
                    code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    level="info",
                    remediation=REMEDIATION_RESTRICTED_QUESTION,
                )
                note_interactive_turn(choice_port, outcome="restricted", error=restricted_message)
            else:
                print_rephrase_hint(RephraseHint.VAGUE_QUESTION)
                note_interactive_turn(choice_port, outcome="invalid_question", error="Question failed validation.")
            return None
        corrected_text = validation.corrected
        if corrected_text != raw_question:
            debug(f"[main_execution.interactive_run_once] typo_corrected: '{raw_question}' -> '{corrected_text}'")

        if validation.route in (QuestionRoute.SCHEMA_CATALOG, QuestionRoute.BUSINESS_KNOWLEDGE):
            route_code = f"meta.route.{validation.route.value}"
            notify(
                f"Metadata route: {validation.route.value}",
                stage="meta",
                code=route_code,
                level="info",
            )
            art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
            adir: str | None = None
            if art is not None:
                try:
                    adir = os.path.abspath(os.fspath(art))
                except (TypeError, OSError, ValueError):
                    adir = None
            space_overlay = getattr(pipeline_session, "_space_name", None) if pipeline_session is not None else None
            meta_step = MainExecutionOps.answer_metadata_question(
                owner,
                corrected_text,
                validation.route,
                schema,
                space_overlay,
                adir,
            )
            if pipeline_session is not None:
                pipeline_session._pending_terminal_step = meta_step
            return None

        tmpl_typo = match_question_level_template_reuse(corrected_text, templates, template_store=store)
        if tmpl_typo.reuse_type == "direct_reuse" and not MainExecutionOps._owner_has_federation(owner):
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
        if normalized_canonical != corrected_text and TemplateOps.has_any_rejection_history_for_question(
            store, corrected_text
        ):
            debug(
                f"[main_execution.interactive_run_once] dropped_normalized_due_to_negative_memory {normalized_canonical!r}"
            )
            neg_drop = True
            normalized_canonical = corrected_text

        tmpl_norm = None
        if normalized_canonical != corrected_text:
            tmpl_norm = match_question_level_template_reuse(normalized_canonical, templates, template_store=store)
            if tmpl_norm.reuse_type == "direct_reuse" and not MainExecutionOps._owner_has_federation(owner):
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

        if MainExecutionOps._owner_has_federation(owner):
            fed_kwargs = MainExecutionOps._federation_reuse_kwargs(owner, choice_port)
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
        MainExecutionOps._interactive_attach_refinement_ctx(choice_port, refinement_ctx)

        while True:
            MainExecutionOps._raise_if_session_turn_cancelled()
            try:
                completed = MainExecutionOps._interactive_run_intent_pass(
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _flatten_scalar_engine_fields(
        block: dict[str, Any],
        field_specs: tuple[tuple[str, str], ...],
        output: dict[str, str],
        claimed: set[str],
        *,
        section_name: str,
    ) -> None:
        for subkey, target_key in field_specs:
            MainExecutionOps._toml_claim_put_scalar(block, subkey, target_key, output, claimed)
        if section_name in {"csv", "excel"}:
            MainExecutionOps._toml_claim_put_csv_files(block, output, claimed)

    @staticmethod
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
            MainExecutionOps._flatten_scalar_engine_fields(
                block, field_specs, output, claimed, section_name=section_name
            )
            return output, claimed, frozenset()
        connection_names = frozenset(str(name) for name in named_blocks)
        selected = connection_name
        if selected is None and len(named_blocks) == 1:
            selected = next(iter(named_blocks))
        if selected is None:
            return output, claimed, connection_names
        if selected not in named_blocks:
            options = ", ".join(sorted(connection_names))
            raise ConfigError(
                f"config_file [{section_name}] has no connection {selected!r}; expected one of: {options}."
            )
        MainExecutionOps._flatten_scalar_engine_fields(
            named_blocks[selected], field_specs, output, claimed, section_name=section_name
        )
        return output, claimed, connection_names

    @staticmethod
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

    @staticmethod
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
            MainExecutionOps._toml_claim_put_scalar(block, subkey, target_key, output, claimed)

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
            flat, section_claimed, named = MainExecutionOps._flatten_engine_block(
                section_name, engine_block, field_specs, connection
            )
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
        return output, frozenset(claimed), named_connections_by_engine

    @staticmethod
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

    @staticmethod
    def _engine_storage_slug_fragment(raw: str, *, fallback: str) -> str:
        """Return a filesystem-friendly lowercase token for a single slug component."""
        t = re.sub(r"[^0-9A-Za-z]+", "_", str(raw).strip()).strip("_").lower()
        return t if t else fallback

    @staticmethod
    def compute_connection_storage_slug(engine: str, runtime: EngineRuntimeConfig | None = None) -> str:
        """Return a stable connection slug derived from the active engine runtime configuration. When the composed slug is longer than :data:`ENGINE_STORAGE_SLUG_MAX_CHARS`, a deterministic hash suffix is used instead."""
        runtime_cls = DialectRegistry.get_runtime_config_class(engine)
        runtime_cfg = runtime if runtime is not None else runtime_cls()
        fields = dict(runtime_cfg.connection_slug_fields())
        slug_keys = runtime_cfg.connection_slug_keys()
        if runtime is None:
            for key in slug_keys:
                attr = key.upper()
                if hasattr(runtime_cls, attr):
                    class_val = getattr(runtime_cls, attr)
                    if class_val is not None and str(class_val).strip():
                        fields[key] = str(class_val)
        parts = [MainExecutionOps._engine_storage_slug_fragment(fields[key], fallback=key[0]) for key in slug_keys]
        slug = f"conn_{engine}_" + "_".join(parts)
        if len(slug) > int(ENGINE_STORAGE_SLUG_MAX_CHARS):
            digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]
            return f"conn_{engine}_{digest}"
        return slug

    @staticmethod
    def compute_engine_storage_dir(
        artifacts_root: str | None,
        engine: str,
        *,
        tenant_slug: str | None = None,
        runtime: EngineRuntimeConfig | None = None,
    ) -> str:
        """Return the absolute engine storage directory for persisted artifacts. When *artifacts_root* is ``None`` or blank, the parent directory is :meth:`EngineConfig.default_artifacts_root`. When *artifacts_root* is provided, the parent directory is its absolute expanded path. The final directory is ``os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, connection_slug)``; when *tenant_slug* is provided, a sanitized tenant segment is inserted before *connection_slug*."""
        parent = (
            os.path.abspath(os.path.expanduser(str(artifacts_root)))
            if artifacts_root and str(artifacts_root).strip()
            else str(EngineConfig.default_artifacts_root())
        )
        slug = MainExecutionOps.compute_connection_storage_slug(engine, runtime=runtime)
        if tenant_slug is not None and str(tenant_slug).strip():
            tenant_segment = sanitize_tenant_slug(tenant_slug)
            return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, tenant_segment, slug)
        return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, slug)

    @staticmethod
    def _prepare_schema_context_for_init(
        schema_context: EngineContext, engine_storage_dir: str, sink: Callable[[str], None]
    ) -> EngineContext:
        """Merge an explicit ``EngineContext`` with any compatible on- disk. cache under *engine_storage_dir*."""
        try:
            cached = MainExecutionOps.load_schema_context_cache(engine_storage_dir)
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
                    new_notes = MainExecutionOps._read_text_if_file(schema_context.notes_file)
                    if isinstance(old_notes, str) and isinstance(new_notes, str) and new_notes != old_notes:
                        sink("  Schema context: notes file changed since last run.")
                if schema_context.sql_file:
                    old_sql = prev_ctx.get("sql_text")
                    new_sql = MainExecutionOps._read_text_if_file(schema_context.sql_file)
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

    @staticmethod
    def _env_all_non_empty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
        """Return True when every key maps to a non-blank string."""
        return all(str(env.get(k, "") or "").strip() for k in keys)

    @staticmethod
    def _env_first_nonempty(env: Mapping[str, str], *keys: str) -> str:
        """Return the first non-blank value among *keys*, else an empty string."""
        return EngineConfig.env_first_nonempty(env, *keys)

    @staticmethod
    def _env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
        """True when at least one key maps to a non-blank string."""
        return EngineConfig.env_any_nonempty(env, keys)

    @staticmethod
    def _env_role_hint(label: str, keys: tuple[str, ...]) -> str:
        return EngineConfig.env_role_hint(label, keys)

    @staticmethod
    def _runtime_config_for_engine(engine: str) -> type[EngineRuntimeConfig]:
        return cast(type[EngineRuntimeConfig], DialectRegistry.get_runtime_config_class(engine))

    @staticmethod
    def emit_ignored_behaviour_environment_diagnostics(env: Mapping[str, str]) -> None:
        """Emit ``CONFIGURATION_KEY_IGNORED`` for legacy behaviour env vars that are still set."""
        for env_name, replacement in REMOVED_BEHAVIOUR_ENVIRONMENT_KEYS.items():
            raw = str(env.get(env_name, "") or "").strip()
            if not raw:
                continue
            notify(
                f"Environment variable {env_name} is ignored; use {replacement} instead.",
                stage="config",
                code=DIAGNOSTIC_CODE_CONFIGURATION_KEY_IGNORED,
                details=(("key", env_name), ("replacement", replacement)),
            )

    @staticmethod
    def _apply_runtime_environments(env: Mapping[str, str]) -> None:
        """Load every registered runtime config whose partial env scope is present."""
        MainExecutionOps.emit_ignored_behaviour_environment_diagnostics(env)
        for engine in DialectRegistry.list_engines():
            runtime_cls = MainExecutionOps._runtime_config_for_engine(engine)
            if runtime_cls.should_apply_environment(env):
                runtime_cls.load_process_default_from_environment(env)

    @staticmethod
    def _select_engine_name(
        env: Mapping[str, str], named_connections_by_engine: Mapping[str, frozenset[str]] | None = None
    ) -> str:
        named = named_connections_by_engine or {}
        engines = DialectRegistry.list_engines()
        explicit = str(env.get("AETHERDIALECT_ENGINE", "") or "").strip().lower()
        if explicit:
            if explicit not in engines:
                raise ConfigError(f"Unsupported AETHERDIALECT_ENGINE: {explicit!r}. Expected one of {engines}.")
            blockers = MainExecutionOps._runtime_config_for_engine(explicit).selection_blockers(env)
            if blockers and not named.get(explicit):
                raise ConfigError(f"Cannot select {explicit} engine: {'; '.join(blockers)}")
            return explicit
        ready: list[str] = []
        for engine in engines:
            if not MainExecutionOps._runtime_config_for_engine(engine).selection_blockers(env):
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
            missing.extend(MainExecutionOps._runtime_config_for_engine(engine).selection_blockers(env))
        raise ConfigError("Cannot select database engine: " + "; ".join(missing))

    @staticmethod
    def _activate_engine(name: str) -> None:
        """Bind :attr:`EngineConfig.TYPE` and :attr:`EngineConfig.RUNTIME` to the chosen engine."""
        if name not in DialectRegistry.list_engines():
            raise ConfigError(f"Unsupported engine activation: {name!r}.")
        EngineConfig.TYPE = name
        EngineConfig.RUNTIME = MainExecutionOps._runtime_config_for_engine(name)

    @staticmethod
    def configure_runtime_from_environment(
        engine_context: EngineContext, merged_env: Mapping[str, str]
    ) -> tuple[str, EngineRuntimeConfig]:
        env: dict[str, str] = dict(merged_env)
        selected = MainExecutionOps._select_engine_name(env)
        MainExecutionOps._apply_runtime_environments(env)
        PolicyConfig.apply_environment(env)
        runtime = MainExecutionOps._runtime_config_for_engine(selected).from_environment(env)
        MainExecutionOps._activate_engine(selected)
        if selected == "databricks" and not cast(DatabricksRuntimeConfig, runtime).has_native_connection():
            if not DatabricksRuntimeConfig.pyspark_session_reachable():
                raise ConfigError(
                    "Databricks requires either all SQL warehouse connection variables or an active PySpark session."
                )
        MainExecutionOps._configure_llm_from_environment(env)
        if selected not in DialectRegistry.list_engines():
            raise ConfigError(f"Unsupported engine resolved: {selected!r}")
        return selected, runtime

    @staticmethod
    def validate_azure_llm_execution(llm_exec: Any) -> None:
        """Raise ``ConfigError`` when Azure provider is missing required deployment fields."""
        missing = [
            n
            for n, v in (
                ("azure_endpoint", getattr(llm_exec, "azure_endpoint", None)),
                ("azure_api_key", getattr(llm_exec, "azure_api_key", None)),
                ("azure_api_version", getattr(llm_exec, "azure_api_version", None)),
                ("deployment_light", getattr(llm_exec, "deployment_light", None)),
                ("deployment_heavy", getattr(llm_exec, "deployment_heavy", None)),
            )
            if not (isinstance(v, str) and v.strip())
        ]
        if missing:
            raise ConfigError("Azure OpenAI requires non-empty runtime configuration for: " + ", ".join(missing))

    @staticmethod
    def _apply_logical_model_env_overrides(env: Mapping[str, str]) -> None:
        """Override ``OPENAI_MODEL*`` ClassVars when matching environment keys are set."""
        for attr in (
            "OPENAI_MODEL",
            "OPENAI_MODEL_INTENT",
            "OPENAI_MODEL_JOIN",
            "OPENAI_MODEL_SCHEMA_BASE",
            "OPENAI_MODEL_DDL",
            "OPENAI_MODEL_SCHEMA",
            "OPENAI_MODEL_SYNTH",
            "OPENAI_MODEL_SYNTH_VARIETY",
            "OPENAI_MODEL_INTENT_FORMAT",
            "OPENAI_MODEL_INTENT_SCHEMA_REPAIR",
            "OPENAI_MODEL_UPLOAD_SUMMARY",
            "OPENAI_MODEL_UPLOAD_INTERPRET",
        ):
            raw = str(env.get(attr, "") or "").strip()
            if raw:
                setattr(EngineConfig, attr, raw)

    @staticmethod
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
        MainExecutionOps._apply_logical_model_env_overrides(env)

    @staticmethod
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
        MainExecutionOps._apply_logical_model_env_overrides(env)

    @staticmethod
    def _openai_direct_env_complete(env: Mapping[str, str]) -> bool:
        return MainExecutionOps._env_any_nonempty(env, ("OPENAI_API_KEY",))

    @staticmethod
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

    @staticmethod
    def _configure_llm_from_environment(env: Mapping[str, str]) -> None:
        explicit = str(env.get("AETHERDIALECT_LLM_PROVIDER", "") or "").strip().lower()
        if explicit == "mock":
            MainExecutionOps._configure_mock_from_environment(env)
            LLMProvider.clear_llm_clients()
            MockProvider.reset_mock_provider()
            return
        openai_ready = MainExecutionOps._openai_direct_env_complete(env)
        azure_ready = MainExecutionOps._env_all_non_empty(env, AZURE_OPENAI_ENV_REQUIRED)
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
                    raise ConfigError(
                        "AETHERDIALECT_LLM_PROVIDER is 'openai' but the OpenAI environment is incomplete."
                    )
                MainExecutionOps._configure_openai_from_environment(env)
            else:
                if not azure_ready:
                    raise ConfigError(
                        "AETHERDIALECT_LLM_PROVIDER is 'azure' but the Azure OpenAI environment is incomplete."
                    )
                MainExecutionOps._configure_azure_from_environment(env)
            return
        if openai_ready and azure_ready:
            raise ConfigError(
                "Both OpenAI and Azure OpenAI credentials are available; set AETHERDIALECT_LLM_PROVIDER "
                "or [llm] provider in the config file to 'openai' or 'azure'."
            )
        if openai_ready:
            MainExecutionOps._configure_openai_from_environment(env)
            return
        if azure_ready:
            MainExecutionOps._configure_azure_from_environment(env)
            return
        raise ConfigError("LLM is not configured.")

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        artifacts_root = MainExecutionOps._federation_artifacts_root(owner)
        for binding in manifest.sources:
            base = dict(MainExecutionOps._consumer_sql_gate_kwargs(choice_port))
            base["schema_role"] = binding.role
            member_ops = DialectRegistry.extra_where_ops_for_engine(binding.engine)
            base["allowed_where_ops"] = allowed_where_ops & (member_ops | set(FEDERATION_BASE_WHERE_OPS))
            if binding.context and binding.context != "master":
                runtime = runtimes.get(binding.source_id)
                member_dir = (
                    str(runtime.artifacts_dir)
                    if runtime is not None and runtime.artifacts_dir
                    else federation_source_artifacts_dir(artifacts_root, binding)
                )
                named = MainExecutionOps.load_named_schema_context(member_dir, binding.context)
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

    @staticmethod
    def _federation_reuse_kwargs(owner: Any | None, choice_port: InteractiveChoicePort | None) -> dict[str, Any]:
        """Optional federation context for question-level reuse paths."""
        if owner is None or getattr(owner, "_federation_manifest", None) is None:
            return {}
        manifest = getattr(owner, "_federation_manifest", None)
        member_graphs = getattr(owner, "_federation_member_graphs", None)
        stores_by_source: dict[str, TemplateStoreView] = {}
        gate_kwargs_by_source: dict[str, dict[str, Any]] | None = None
        if isinstance(member_graphs, dict) and member_graphs:
            stores_by_source = MainExecutionOps.federation_stores_by_source(
                owner, member_graphs, space_name=MainExecutionOps._session_space_name_for_federation(owner, choice_port)
            )
            if manifest is not None:
                gate_kwargs_by_source = MainExecutionOps._federation_gate_kwargs_by_source(
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

    @staticmethod
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

    @staticmethod
    def _federation_session_step_sql(
        gen_out: SqlGenerationOutcome | None = None,
        *,
        federated_bundle: FederatedSqlBundle | None = None,
        federated_plan: FederatedPlan | None = None,
        generation_path: GenerationPath | None = None,
    ) -> str | dict[str, str] | None:
        """Return member SQL for a federated turn: ``str`` for one member, ``dict`` for many."""
        if not MainExecutionOps._federation_turn_active(
            gen_out=gen_out,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
        ):
            return None
        if federated_bundle is None:
            return None
        member_statements = [
            rec
            for rec in federated_bundle.statements
            if str(getattr(rec, "phase", "member") or "member") == "member"
            and str(getattr(rec, "source_id", "") or "").strip()
            and str(getattr(rec, "statement", "") or "").strip()
        ]
        if not member_statements:
            return None
        if len(member_statements) == 1:
            return str(member_statements[0].statement).strip() or None
        mapping: dict[str, str] = {}
        for rec in member_statements:
            sid = str(rec.source_id).strip()
            statement = str(rec.statement).strip()
            if sid and statement and sid not in mapping:
                mapping[sid] = statement
        return mapping or None

    @staticmethod
    def _resolved_session_step_sql(
        sql: str | dict[str, str] | None,
        *,
        gen_out: SqlGenerationOutcome | None = None,
        federated_bundle: FederatedSqlBundle | None = None,
        federated_plan: FederatedPlan | None = None,
        generation_path: GenerationPath | None = None,
    ) -> str | dict[str, str] | None:
        """Resolve ``SessionStep.sql`` for single-engine and federated turns."""
        fed_sql = MainExecutionOps._federation_session_step_sql(
            gen_out,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
        )
        if fed_sql is not None:
            return fed_sql
        if MainExecutionOps._federation_turn_active(
            gen_out=gen_out,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
        ):
            return None
        return sql

    @staticmethod
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

    @staticmethod
    def _federation_contract_kwargs_from_snap(snap: Mapping[str, Any]) -> dict[str, Any]:
        """Derive federation column contract kwargs stored on a completed turn snapshot."""
        federated_bundle = snap.get("federated_bundle")
        federated_plan = snap.get("federated_plan")
        generation_path = snap.get("generation_path")
        if (
            federated_bundle is None
            and federated_plan is None
            and generation_path is not GenerationPath.FEDERATION_PLAN
        ):
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

    @staticmethod
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
            stores_by_source = MainExecutionOps.federation_stores_by_source(
                owner, member_graphs, space_name=MainExecutionOps._session_space_name_for_federation(owner, choice_port)
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

    @staticmethod
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
            stores[source_id] = TemplateOps.load_template_store(
                graph_id, graph, artifacts_dir=artifacts_dir, space_name=space_name
            )
        return stores

    @staticmethod
    def _federation_duckdb_schema_for_connection(connection: str) -> str:
        """Map a federation source connection label to the DuckDB schema used for qualification."""
        conn = str(connection or "").strip().lower()
        if conn in {"", "memory", "main", "storefront"}:
            return "main"
        return conn

    @staticmethod
    def _duckdb_runtime_config_for_schema(base_cls: type[EngineRuntimeConfig], schema: str) -> EngineRuntimeConfig:
        """Return a DuckDB runtime config with ``SCHEMA`` pinned to *schema*."""
        runtime_cfg = EngineRuntimeConfig.process_default_for_class(base_cls)
        if schema != "main":
            runtime_cfg = copy.copy(runtime_cfg)
            cast(DuckDBRuntimeConfig, runtime_cfg).SCHEMA = schema
        return runtime_cfg

    @staticmethod
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
            adir = federation_source_artifacts_dir(
                artifacts_root,
                binding,
                federation_id=str(manifest.federation_id or "") or None,
            )
            with artifact_lock(adir):
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
                        sqlglot_dialect=DialectRegistry.sqlglot_dialect_for_engine(engine_type),
                        native_connection=native_by_source.get(binding.source_id),
                        sqlalchemy_engine=sa_by_source.get(binding.source_id),
                    )
                    continue
                try:
                    runtime_cfg_cls = MainExecutionOps._runtime_config_for_engine(engine_type)
                    runtime_cfg = EngineRuntimeConfig.process_default_for_class(runtime_cfg_cls)
                except Exception:
                    identity_runtime = fallback_identity.runtime_config
                    if isinstance(identity_runtime, type):
                        runtime_cfg_cls = identity_runtime
                        runtime_cfg = EngineRuntimeConfig.process_default_for_class(runtime_cfg_cls)
                    else:
                        runtime_cfg = identity_runtime
                        runtime_cfg_cls = type(runtime_cfg)
                if engine_type == "duckdb":
                    runtime_cfg = MainExecutionOps._duckdb_runtime_config_for_schema(
                        runtime_cfg_cls,
                        MainExecutionOps._federation_duckdb_schema_for_connection(str(binding.connection or "")),
                    )
                source_sa = sa_by_source.get(binding.source_id, sqlalchemy_engine)
                source_native = native_by_source.get(binding.source_id, native_connection)
                prior = prior_runtimes.get(binding.source_id)
                if (
                    prior is not None
                    and prior.engine == engine_type
                    and prior.connection == str(binding.connection or "")
                ):
                    if prior.native_connection is not None:
                        source_native = prior.native_connection
                    if prior.sqlalchemy_engine is not None:
                        source_sa = prior.sqlalchemy_engine
                try:
                    bound_dialect = DialectRegistry.get_dialect(
                        engine_type, runtime_cfg, sqlalchemy_engine=source_sa, native_connection=source_native
                    )
                except Exception:
                    bound_dialect = default_dialect
                runtimes[binding.source_id] = SourceRuntime(
                    source_id=binding.source_id,
                    engine=engine_type,
                    connection=str(binding.connection or ""),
                    artifacts_dir=adir,
                    dialect=bound_dialect,
                    sqlglot_dialect=DialectRegistry.sqlglot_dialect_for_engine(engine_type),
                    native_connection=source_native,
                    sqlalchemy_engine=source_sa,
                )
        return runtimes

    @staticmethod
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

    @staticmethod
    def _read_text_if_file(path: str | None) -> str | None:
        """Return the text content of *path* if it exists and is a regular file, else None."""
        if not path:
            return None
        expanded = os.path.expanduser(str(path))
        if not os.path.isfile(expanded):
            return None
        with open(expanded, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
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
            "notes_inline": schema_context.notes is not None,
            "sql_text": MainExecutionOps._read_text_if_file(schema_context.sql_file),
            "notes_text": notes_content_from_context(schema_context),
        }
        os.makedirs(artifacts_dir, exist_ok=True)
        cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
        MainExecutionOps._write_json_atomic(cache_path, payload)
        return cache_path

    @staticmethod
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
        if not format_versions_match(cache_version, SCHEMA_CONTEXT_CACHE_VERSION):
            raise ConfigError(
                f"schema context cache at {cache_path!r} has version {cache_version!r}; "
                f"this build expects {SCHEMA_CONTEXT_CACHE_VERSION}. "
                f"Delete {cache_path!r} (or the engine artifacts directory) and re-run "
                f"initialize_aether_engine so the cache is rebuilt from scratch."
            )
        MainExecutionOps._validate_scope_list_fields(payload)
        sql_text = payload.get("sql_text")
        notes_text = payload.get("notes_text")
        sql_file: str | None = None
        notes_file: str | None = None
        notes_inline: str | None = None
        if isinstance(sql_text, str):
            sql_file = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHED_DDL)
            with open(sql_file, "w", encoding="utf-8") as fh:
                fh.write(sql_text)
        if isinstance(notes_text, str):
            if payload.get("notes_inline"):
                notes_inline = notes_text
            else:
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
            notes=notes_inline,
        )

    @staticmethod
    def _purge_schema_context_cache(artifacts_dir: str) -> None:
        """Remove the persisted ``schema_context.json`` and any materialised cache files. Used during legacy-artifact cleanup so a stale schema context cannot be silently reloaded after a learning-reset rebuild."""
        for name in (SCHEMA_CONTEXT_CACHE_NAME, SCHEMA_CONTEXT_CACHED_DDL, SCHEMA_CONTEXT_CACHED_NOTES):
            fp = os.path.join(artifacts_dir, name)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError as exc:
                    debug(f"[main_execution._purge_schema_context_cache] {fp}: {exc}")

    @staticmethod
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

    @staticmethod
    def _upload_validation_config_error(message: str, data_quality_report: object) -> ConfigError:
        """Attach upload validation context to a configuration error."""
        exc = ConfigError(message)
        cast(Any, exc).data_quality_report = data_quality_report
        return exc

    @staticmethod
    def _emit_runtime_config_override_diagnostics(overridden: frozenset[str]) -> None:
        """Emit one diagnostic per runtime-config field whose effective value came from the TOML file over env."""
        for key in sorted(overridden):
            notify(
                f"Runtime config file overrides environment for {key}",
                stage="config",
                code=DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
                details=(("key", key),),
            )

    @staticmethod
    def migration_report_for_init(
        artifacts_dir: str,
        prompt_schema: SchemaGraph,
        *,
        schema_role: SchemaRole,
        previous_schema: SchemaGraph | None,
        schema_diff: Any | None,
    ) -> MigrationReport:
        """Resolve template migration during single-engine init; consumers never mutate artifacts."""
        if schema_role == "consumer":
            return MigrationReport(tier=MigrationTier.NO_CHANGE)
        return TemplateOps.apply_migration_policy(
            artifacts_dir,
            prompt_schema,
            allow_destructive=True,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )

    @staticmethod
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

    @staticmethod
    def _emit_artifact_growth_diagnostics(artifacts_dir: str, limits: EngineLimits) -> list[Diagnostic]:
        """Emit growth snapshot and optional near-limit warnings for artifact storage."""
        artifact_bytes = TemplateOps.artifact_directory_byte_size(artifacts_dir)
        template_count, feedback_shard_count, orphan_count = TemplateOps.artifact_growth_counts(artifacts_dir)
        growth = Diagnostic(
            stage="artifact",
            level="info",
            code=DIAGNOSTIC_CODE_ARTIFACT_GROWTH,
            message="Artifact directory growth snapshot",
            details=(
                ("artifact_bytes", str(artifact_bytes)),
                ("template_count", str(template_count)),
                ("feedback_shard_count", str(feedback_shard_count)),
                ("orphan_count", str(orphan_count)),
            ),
            phase="artifact",
        )
        notify(
            growth.message,
            stage=growth.stage,
            code=growth.code,
            level=growth.level,
            details=growth.details,
        )
        diags: list[Diagnostic] = [growth]
        if limits.template_store_max_count is not None:
            cap = int(limits.template_store_max_count)
            if cap > 0 and template_count >= int(cap * 0.9):
                near = Diagnostic(
                    stage="artifact",
                    level=DiagnosticSeverity.WARNING,
                    code=DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR,
                    message="Template count is within ten percent of template_store_max_count",
                    details=(
                        ("limit", "template_store_max_count"),
                        ("cap", str(cap)),
                        ("current", str(template_count)),
                    ),
                    phase="artifact",
                )
                notify(near.message, stage=near.stage, code=near.code, level=near.level, details=near.details)
                diags.append(near)
        if limits.template_store_max_disk_bytes is not None:
            cap = int(limits.template_store_max_disk_bytes)
            if cap > 0 and artifact_bytes >= int(cap * 0.9):
                near = Diagnostic(
                    stage="artifact",
                    level=DiagnosticSeverity.WARNING,
                    code=DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR,
                    message="Artifact directory size is within ten percent of template_store_max_disk_bytes",
                    details=(
                        ("limit", "template_store_max_disk_bytes"),
                        ("cap", str(cap)),
                        ("current", str(artifact_bytes)),
                    ),
                    phase="artifact",
                )
                notify(near.message, stage=near.stage, code=near.code, level=near.level, details=near.details)
                diags.append(near)
        return diags

    @staticmethod
    def refresh_aether_engine(
        owner: Any,
        *,
        reflect: bool = True,
        log_sink: Callable[[str], None] | None = None,
    ) -> RefreshReport:
        """Re-run post-connection artifact reconciliation for an existing engine."""
        sink: Callable[[str], None] = log_sink if log_sink is not None else notify
        adir = str(owner._artifacts_dir)
        dialect = owner._dialect
        runtime_cfg = owner._runtime_config
        master_ctx = runtime_cfg.engine_context
        if not isinstance(master_ctx, EngineContext):
            raise ConfigError("refresh requires a single-engine context")
        schema_role = getattr(owner, "_schema_role", SchemaRole.OWNER)
        trust_bundled_baseline = getattr(owner, "_trust_bundled_baseline", False)
        limits = getattr(owner, "_limits", EngineLimits())
        schema_json_path = MainExecutionOps.engine_schema_json_path(adir)
        diagnostics: list[Diagnostic] = list(TemplateOps.collect_orphaned_migration_checkpoints(adir))
        notes_content: str | None = None
        if master_ctx.notes is not None or master_ctx.notes_file:
            notes_content = notes_content_from_context(master_ctx)
        previous_schema = load_schema_graph_snapshot(schema_json_path)
        artifacts_root = Path(adir)
        map_path = artifacts_root / MIGRATION_MAP_FILENAME
        pending_migration_map = None
        if reflect:
            pending_migration_map = (
                TemplateOps.load_schema_migration_map(artifacts_root)
                if map_path.is_file() and schema_role == SchemaRole.OWNER
                else None
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
                schema_json_path=schema_json_path,
            )
            if map_path.is_file() and schema_role == SchemaRole.OWNER:
                loaded = (
                    pending_migration_map
                    if pending_migration_map is not None
                    else TemplateOps.load_schema_migration_map(artifacts_root)
                )
                if loaded is not None:
                    try:
                        TemplateOps.validate_schema_migration_map(loaded, previous_schema, schema_graph)
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
                        TemplateOps.apply_schema_migration_map(loaded, adir, schema_graph, Path(schema_json_path))
                        ts = datetime.now(UTC).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
                        applied_map = map_path.with_name(map_path.stem + ".applied.json")
                        try:
                            if applied_map.is_file():
                                archive = applied_map.with_name(applied_map.stem + f".{ts}" + applied_map.suffix)
                                applied_map.rename(archive)
                            map_path.rename(applied_map)
                        except OSError as exc:
                            debug(f"[main_execution.refresh_aether_engine] could not archive migration map: {exc}")
                        previous_schema = load_schema_graph_snapshot(schema_json_path)
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
        else:
            loaded_graph = load_schema_graph_snapshot(schema_json_path)
            if loaded_graph is None:
                raise ConfigError("artifact-only refresh requires a cached schema graph")
            schema_graph = loaded_graph
            schema_diff = None
            finalize_with_overrides(schema_graph, schema_json_path, dialect=dialect)
        owner_snapshot = previous_schema
        stored = read_artifact_manifest(adir)
        if schema_role == SchemaRole.OWNER and stored is not None and not stored.schema_graph_id:
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
            pinned_schema_graph_id=pinned_id if schema_role == SchemaRole.CONSUMER else None,
        )
        consumer_visible: frozenset[str] | None = None
        prompt_schema = schema_graph
        tier_preview = classify_migration_tier(
            stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff
        )
        if (
            schema_role == SchemaRole.CONSUMER
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
        if (
            schema_role == SchemaRole.CONSUMER
            and stored is not None
            and artifact_manifest_incompatible_with_package(stored)
        ):
            raise ConfigError(
                "Artifact manifest is incompatible with this package version; "
                "an owner must refresh artifacts before consumer init can proceed."
            )
        if schema_role == SchemaRole.CONSUMER and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            raise ConfigError(
                "Schema has drifted since artifacts were published; "
                "an owner must refresh artifacts before consumer init can proceed."
            )
        if schema_role == SchemaRole.OWNER and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            rename_plan = (
                try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
            )
            skel_path = TemplateOps.export_schema_migration_map_skeleton(
                artifacts_root, tier=tier_preview, schema_diff=schema_diff, rename_plan=rename_plan
            )
            raise MigrationPendingError(f"Schema migration required: edit {skel_path} and restart init.")
        migration_report = MainExecutionOps.migration_report_for_init(
            adir,
            prompt_schema,
            schema_role=schema_role,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )
        if migration_report.tier != MigrationTier.NO_CHANGE:
            MainExecutionOps._print_migration_applied(migration_report, sink)
        previous_graph_id = ""
        if stored is not None:
            previous_graph_id = str(stored.schema_graph_id or "")
        elif owner_snapshot is not None:
            previous_graph_id = str(owner_snapshot.schema_graph_id or "")
        active_graph_id = str(prompt_schema.schema_graph_id or "")
        if (
            schema_role == SchemaRole.OWNER
            and previous_graph_id
            and active_graph_id
            and previous_graph_id != active_graph_id
        ):
            MainExecutionOps.orphan_superseded_identity_artifacts_on_rotation(
                adir,
                previous_schema_graph_id=previous_graph_id,
                active_schema_graph_id=active_graph_id,
            )
        if schema_role == SchemaRole.OWNER:
            MainExecutionOps.prune_stale_artifact_auxiliaries(adir, active_schema_graph_id=active_graph_id)
        store = TemplateOps.load_template_store(
            prompt_schema.schema_graph_id, prompt_schema, space_name=MASTER_AETHERSPACE_NAME, artifacts_dir=adir
        )
        reconcile_report = TemplateOps.reconcile_template_store(store, prompt_schema)
        if reconcile_report.dropped_template_ids:
            TemplateOps.save_template_store(store)
        templates = TemplateOps.store_to_templates(store)
        orphans_removed, bytes_reclaimed = TemplateOps.collect_expired_template_orphans(adir)
        diagnostics.extend(MainExecutionOps._emit_artifact_growth_diagnostics(adir, limits))
        if schema_role == SchemaRole.OWNER:
            try:
                MainExecutionOps.write_schema_context_cache(adir, master_ctx)
            except OSError as exc:
                debug(f"[main_execution.refresh_aether_engine] schema_context cache write failed: {exc}")
        owner._schema_graph = schema_graph
        owner._store = store
        owner._templates = templates
        if consumer_visible is not None:
            owner._consumer_visible_objects = consumer_visible
        schema_terms: set[str] = set(schema_graph.tables.keys())
        for tinfo in schema_graph.tables.values():
            schema_terms.update(tinfo.columns)
            for col in tinfo.columns:
                schema_terms.add(col.lower())
        owner._schema_terms = schema_terms
        owner._schema_stats = schema_graph.schema_stats or {}
        objects_added = tuple(sorted(set(migration_report.added_tables)))
        objects_removed = tuple(sorted(set(migration_report.dropped_tables)))
        schema_changed = schema_diff is not None and not schema_diff.is_empty
        return RefreshReport(
            migration_tier=migration_report.tier,
            schema_changed=schema_changed,
            objects_added=objects_added,
            objects_removed=objects_removed,
            templates_invalidated=len(reconcile_report.dropped_template_ids),
            orphans_removed=orphans_removed,
            bytes_reclaimed=bytes_reclaimed,
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def overlay_programmatic_connection(merged: dict[str, str], connection: Mapping[str, Any]) -> dict[str, str]:
        """Merge programmatic connection parameters into *merged* without writing ``os.environ``."""
        for raw_key, raw_val in connection.items():
            if raw_val is None:
                continue
            key = str(raw_key).strip()
            if not key:
                continue
            value = str(raw_val).strip()
            if not value:
                continue
            merged[key] = value
        return merged

    @staticmethod
    def _split_connection_argument(
        connection: str | Mapping[str, Any] | None,
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Return ``(named_connection, programmatic_mapping)`` for engine construction."""
        if connection is None:
            return None, None
        if isinstance(connection, Mapping):
            return None, connection
        named = str(connection).strip()
        return (named or None), None

    @staticmethod
    def initialize_aether_engine(
        engine_context: EngineContext | str | None = None,
        *,
        artifacts_dir: str | None = None,
        tenant_slug: str | None = None,
        config_file: str | os.PathLike[str] | None = None,
        connection: str | Mapping[str, Any] | None = None,
        log_sink: Callable[[str], None] | None = None,
        execution_engine: Any | None = None,
        native_connection: Any | None = None,
        schema_role: SchemaRole = SchemaRole.OWNER,
        source_selections: Mapping[str, Mapping[str, Any]] | None = None,
        trust_bundled_baseline: bool = False,
        token_provider: Callable[[], str | Mapping[str, str]] | None = None,
    ) -> AetherEngineInitResult:
        """Configure the process environment, build the schema graph, migrate templates, and load stores."""
        sink: Callable[[str], None] = log_sink if log_sink is not None else notify
        sink("Initialising AetherEngine.")
        named_connection, programmatic_connection = MainExecutionOps._split_connection_argument(connection)
        config_file_values, toml_claimed_keys, named_by_engine = MainExecutionOps._load_config_file(config_file)
        ssot = config_file is not None and bool(str(config_file).strip())
        merged, toml_diagnostic_keys = MainExecutionOps._merge_configuration_environment(
            config_file_values, toml_claimed_keys=toml_claimed_keys if ssot else None
        )
        if programmatic_connection is not None:
            MainExecutionOps.overlay_programmatic_connection(merged, programmatic_connection)
        selected_preview = MainExecutionOps._select_engine_name(merged, named_by_engine)
        resolved_connection = MainExecutionOps._select_connection_name(
            merged, named_by_engine, selected_preview, explicit_connection=named_connection
        )
        if resolved_connection and named_by_engine.get(selected_preview):
            connection_values, connection_claimed, _ = MainExecutionOps._load_config_file(
                config_file, connection=resolved_connection
            )
            config_file_values.update(connection_values)
            toml_claimed_keys = toml_claimed_keys | connection_claimed
            merged, toml_diagnostic_keys = MainExecutionOps._merge_configuration_environment(
                config_file_values, toml_claimed_keys=toml_claimed_keys if ssot else None
            )
            if programmatic_connection is not None:
                MainExecutionOps.overlay_programmatic_connection(merged, programmatic_connection)
            merged["AETHERDIALECT_CONNECTION"] = resolved_connection
        MainExecutionOps._apply_runtime_environments(merged)
        preview_runtime = MainExecutionOps._runtime_config_for_engine(selected_preview).from_environment(merged)
        adir = MainExecutionOps.compute_engine_storage_dir(
            artifacts_dir, selected_preview, tenant_slug=tenant_slug, runtime=preview_runtime
        )
        warn_if_artifacts_dir_not_local(adir)
        try:
            cached_master = MainExecutionOps.load_schema_context_cache(adir)
        except ConfigError as exc:
            sink(str(exc))
            cached_master = None
        prepare_master: EngineContext | None = None
        if isinstance(engine_context, FederationContext):
            raise ConfigError(
                "initialize_aether_engine does not accept FederationContext; use AetherFederation instead"
            )
        if isinstance(engine_context, EngineContext):
            prepare_master = MainExecutionOps._prepare_schema_context_for_init(engine_context, adir, sink)
        master_ctx, active_ctx, context_name = MainExecutionOps.resolve_engine_context_plan(
            engine_context, adir, schema_role=schema_role, load_master=cached_master, prepare_master=prepare_master
        )
        MainExecutionOps._notify_schema_context_warnings(master_ctx, sink)
        active_engine, active_runtime = MainExecutionOps.configure_runtime_from_environment(master_ctx, merged)
        engine_identity = EngineIdentity(engine_type=active_engine, runtime_config=active_runtime)
        construction_orphan_token = bind_construction_orphan_identity(engine_identity)
        try:
            llm_exec = load_runtime_config(merged_env=merged)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        MainExecutionOps._emit_runtime_config_override_diagnostics(toml_diagnostic_keys)
        if EngineConfig.LLM_PROVIDER == "azure":
            MainExecutionOps.validate_azure_llm_execution(llm_exec)
        _rt = active_runtime
        _rt_name = type(_rt).__name__.lower()
        if _rt_name.endswith("runtimeconfig"):
            _rt_name = _rt_name[: -len("runtimeconfig")]
        runtime_label = _rt_name or "default"
        sink(f"  Engine: {active_engine} ({runtime_label}).")
        os.makedirs(adir, exist_ok=True)
        legacy_files = detect_legacy_artifacts(adir)
        if legacy_files:
            sink(f"  Detected legacy artifacts (no manifest): {', '.join(legacy_files)}. Rebuilding caches.")
            wipe_versioned_artifacts(adir)
            MainExecutionOps._purge_schema_context_cache(adir)
        schema_json_path = os.path.join(adir, "schema_graph.json.gz")
        template_store_dir = TemplateOps.template_store_dir_for_space(adir, MASTER_AETHERSPACE_NAME)
        MainExecutionOps.register_engine_artifact_state(
            adir,
            schema_json_path=schema_json_path,
            template_store_dir=template_store_dir,
        )
        TemplateOps.ensure_template_store_space_layout(adir)
        QSimConfig.SKELETONS_JSON_PATH = os.path.join(adir, "qsim_skeletons.json.gz")
        data_quality_report: DataQualityReport | None = None
        if (active_engine or "").strip().lower() in FILE_ENGINE_NAMES:
            csv_runtime = cast(CsvRuntimeConfig, active_runtime)
            upload_paths = csv_runtime.resolve_source_files()
            selections = parse_source_selections(source_selections or csv_runtime.SOURCE_SELECTIONS)
            data_quality_report = validate_upload_sources(upload_paths, log_sink=sink, source_selections=selections)
            if data_quality_report.requires_review and not selections:
                raise MainExecutionOps._upload_validation_config_error(
                    f"{data_quality_report.narrative} "
                    "Call inspect_tabular_upload and pass source_selections with the accepted interpretation.",
                    data_quality_report,
                )
            if not data_quality_report.ok:
                raise MainExecutionOps._upload_validation_config_error(
                    data_quality_report.narrative, data_quality_report
                )
            if source_selections:
                csv_runtime.set_source_selections(source_selections)
                data_quality_report = DataQualityReport(
                    ok=data_quality_report.ok,
                    issues=data_quality_report.issues,
                    narrative=data_quality_report.narrative,
                    suggested_selections=data_quality_report.suggested_selections,
                    confirmed_selections=cast(Any, {k: dict(v) for k, v in source_selections.items()}),
                )
        if token_provider is not None:
            active_runtime = MainExecutionOps.apply_connection_credentials_for_engine(
                active_engine,
                MainExecutionOps.resolve_connection_credentials(None, token_provider),
                runtime=active_runtime,
            )
        try:
            dialect = DialectRegistry.get_dialect(
                active_engine,
                active_runtime,
                sqlalchemy_engine=execution_engine,
                native_connection=native_connection,
            )
        except DatabaseConnectionError:
            raise
        except Exception as exc:
            raise DatabaseConnectionError(str(exc)) from exc
        notes_content: str | None = None
        if master_ctx.notes is not None or master_ctx.notes_file:
            notes_content = notes_content_from_context(master_ctx)
        previous_schema = load_schema_graph_snapshot(schema_json_path)
        TemplateOps.restore_leftover_migration_checkpoints_on_init(adir, schema_json_path=Path(schema_json_path))
        artifacts_root = Path(adir)
        map_path = artifacts_root / MIGRATION_MAP_FILENAME
        pending_migration_map = (
            TemplateOps.load_schema_migration_map(artifacts_root)
            if map_path.is_file() and schema_role == "owner"
            else None
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
            schema_json_path=schema_json_path,
        )
        stored = read_artifact_manifest(adir)
        if map_path.is_file() and schema_role == "owner":
            loaded = (
                pending_migration_map
                if pending_migration_map is not None
                else TemplateOps.load_schema_migration_map(artifacts_root)
            )
            if loaded is not None:
                try:
                    TemplateOps.validate_schema_migration_map(loaded, previous_schema, schema_graph)
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
                    TemplateOps.apply_schema_migration_map(loaded, adir, schema_graph, Path(schema_json_path))
                    ts = datetime.now(UTC).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
                    applied_map = map_path.with_name(map_path.stem + ".applied.json")
                    try:
                        if applied_map.is_file():
                            archive = applied_map.with_name(applied_map.stem + f".{ts}" + applied_map.suffix)
                            applied_map.rename(archive)
                        map_path.rename(applied_map)
                    except OSError as exc:
                        debug(f"[main_execution.initialize_aether_engine] could not archive migration map: {exc}")
                    previous_schema = load_schema_graph_snapshot(schema_json_path)
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
            rename_plan = (
                try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
            )
            skel_path = TemplateOps.export_schema_migration_map_skeleton(
                artifacts_root, tier=tier_preview, schema_diff=schema_diff, rename_plan=rename_plan
            )
            raise MigrationPendingError(f"Schema migration required: edit {skel_path} and restart init.")
        migration_report = MainExecutionOps.migration_report_for_init(
            adir,
            prompt_schema,
            schema_role=schema_role,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )
        if migration_report.tier != MigrationTier.NO_CHANGE:
            MainExecutionOps._print_migration_applied(migration_report, sink)
        if schema_role == "owner":
            MainExecutionOps.prune_stale_artifact_auxiliaries(
                adir, active_schema_graph_id=str(prompt_schema.schema_graph_id)
            )
        store = TemplateOps.load_template_store(
            prompt_schema.schema_graph_id, prompt_schema, space_name=MASTER_AETHERSPACE_NAME, artifacts_dir=adir
        )
        templates = TemplateOps.store_to_templates(store)
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
            MainExecutionOps.validate_named_context_subset(master_ctx, active_ctx, schema_graph)
        execution_ctx = MainExecutionOps._effective_execution_context(master_ctx, active_ctx, context_name)
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
                MainExecutionOps.write_schema_context_cache(adir, master_ctx)
            except OSError as exc:
                debug(f"[main_execution.initialize_aether_engine] schema_context cache write failed: {exc}")
        sink("Ready.")
        release_construction_orphan_identity(construction_orphan_token)
        return AetherEngineInitResult(
            runtime_config=runtime_config,
            llm_config=llm_config,
            schema_graph=schema_graph,
            dialect=dialect,
            artifacts_dir=adir,
            store=store,
            templates=cast(dict[str, Any], templates),
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

    @staticmethod
    def initialize_aether_federation(
        name: str,
        *,
        members: Mapping[str, Any],
        declaration_file: str,
        declaration: tuple[FederationManifest, FederationMappings] | None = None,
        artifacts_dir: str | None = None,
        tenant_slug: str | None = None,
        schema_role: SchemaRole = SchemaRole.OWNER,
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
        fed_storage_dir = compute_federation_storage_dir(
            artifacts_dir,
            fed_manifest.federation_id,
            tenant_slug=tenant_slug,
        )
        if artifacts_dir:
            os.makedirs(fed_storage_dir, exist_ok=True)
            MainExecutionOps._prune_orphaned_federation_trees(
                os.path.dirname(fed_storage_dir), active_fed_dir=fed_storage_dir
            )
        with artifact_lock(fed_storage_dir):
            loaded_member_graphs = load_federation_member_graphs(artifacts_dir, fed_manifest)
            if loaded_member_graphs:
                recorded_ids = recorded_federation_source_ids(fed_storage_dir)
                topology_change = (
                    detect_federation_topology_change(recorded_ids, fed_manifest) if recorded_ids else "none"
                )
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
                coord_dialect = DialectRegistry.get_dialect("duckdb", DuckDBRuntimeConfig)
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
            if fed_master_ctx.notes is not None or fed_master_ctx.notes_file:
                notes_content = notes_content_from_context(fed_master_ctx)
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
                member_notes = notes_content_from_context(member_ctx)
                if member_notes:
                    for token in source_ids:
                        sid = str(token or "").strip()
                        if sid and sid in member_notes:
                            raise ConfigError(f"federation notes must not name a source or member; found {sid!r}")
            federation_artifacts_root = Path(fed_storage_dir)
            recorded_source_ids = recorded_federation_source_ids(fed_storage_dir)
            topo_report: FederationTopologyReport | None = None
            topology_shrink_only = False
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
                if topo_report.removed_source_ids:
                    purge_departed_federation_member_trees(
                        fed_storage_dir,
                        artifacts_root=artifacts_dir,
                        removed_source_ids=topo_report.removed_source_ids,
                    )
            topology_shrink_only = (
                topo_report is not None and topo_report.change == "remove" and not topo_report.added_source_ids
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
            sa_by_source = {
                source_id: getattr(engine, "_execution_engine", None) for source_id, engine in member_dict.items()
            }
            native_by_source = {
                source_id: getattr(engine, "_native_connection", None) for source_id, engine in member_dict.items()
            }
            default_dialect = coord_dialect
            fed_source_runtimes = MainExecutionOps._build_federation_source_runtimes(
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
                    sink(
                        f"  Federation cross-source join columns missing; migration skeleton written to {skel_path!r}."
                    )
                if schema_role == "owner":
                    raise MigrationPendingError(f"Federation migration required: edit {skel_path} and restart init.")
                raise ConfigError(
                    "Federation cross-source join columns are missing; "
                    "an owner must refresh artifacts before consumer init can proceed."
                )
            if not replay_ok and not federation_format_stale and has_persisted_federation:
                prune_federation_plan_templates_on_drift(
                    fed_storage_dir, fed_member_graphs_dict, fed_manifest, fed_mappings
                )
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
            if (
                schema_role == "owner"
                and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE)
                and not topology_shrink_only
            ):
                skel_path = export_federation_migration_map_skeleton(str(federation_artifacts_root))
                sink(
                    f"  Federation composite drift ({tier_preview.value}); migration skeleton written to {skel_path!r}."
                )
                raise MigrationPendingError(f"Federation migration required: edit {skel_path} and restart init.")
            migration_report = TemplateOps.apply_federation_composite_migration_policy(
                fed_storage_dir,
                schema_graph,
                allow_destructive=schema_role == "owner",
                previous_composite=previous_composite,
            )
            if migration_report.tier != MigrationTier.NO_CHANGE:
                MainExecutionOps._print_migration_applied(migration_report, sink)
            if schema_role == "owner":
                MainExecutionOps.prune_stale_artifact_auxiliaries(
                    fed_storage_dir, active_schema_graph_id=str(schema_graph.schema_graph_id)
                )
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
        store = TemplateOps.load_template_store(
            schema_graph.schema_graph_id,
            schema_graph,
            space_name=MASTER_AETHERSPACE_NAME,
            artifacts_dir=fed_storage_dir,
        )
        templates = TemplateOps.store_to_templates(store)
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
                allow_objects=MainExecutionOps._federation_execution_allow_objects(fed_master_ctx, composite_tables),
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
        MainExecutionOps.drain_write_queue(drain_owner, fed_storage_dir)
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
            templates=cast(dict[str, Any], templates),
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def apply_connection_credentials_for_engine(
        engine_type: str,
        credentials: str | Mapping[str, str],
        *,
        runtime: EngineRuntimeConfig | None = None,
    ) -> EngineRuntimeConfig:
        """Apply rotatable secrets on the runtime config for *engine_type*."""
        runtime_cfg = runtime
        if runtime_cfg is None:
            runtime_cfg = EngineRuntimeConfig.process_default_for_class(EngineConfig.RUNTIME)
        try:
            runtime_cfg.apply_connection_credentials(credentials)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return runtime_cfg

    @staticmethod
    def dispose_engine_dialect(
        dialect: Any,
        *,
        borrowed_execution_engine: Any | None = None,
        borrowed_native_connection: Any | None = None,
    ) -> None:
        """Release dialect-owned database handles without closing borrowed caller handles."""
        unregister_dialect_live_handles(
            dialect,
            borrowed_execution_engine=borrowed_execution_engine,
            borrowed_native_connection=borrowed_native_connection,
        )
        dispose_native = getattr(dialect, "dispose_native_connection", None)
        if callable(dispose_native):
            try:
                dispose_native()
            except (OSError, AttributeError, TypeError):
                pass
            return
        connection = getattr(dialect, "connection", None)
        if connection is not None and connection is not borrowed_native_connection:
            close = getattr(connection, "close", None)
            if callable(close):
                try:
                    close()
                except (OSError, AttributeError, TypeError):
                    pass
        sa_engine = getattr(dialect, "engine", None)
        if sa_engine is not None and sa_engine is not borrowed_execution_engine:
            dispose = getattr(sa_engine, "dispose", None)
            if callable(dispose):
                try:
                    dispose()
                except (OSError, AttributeError, TypeError):
                    pass

    @staticmethod
    def refresh_engine_connection(
        *,
        engine_type: str,
        dialect: Any,
        credentials: str | Mapping[str, str] | None = None,
        token_provider: Callable[[], str | Mapping[str, str]] | None = None,
        execution_engine: Any | None = None,
        native_connection: Any | None = None,
        runtime: EngineRuntimeConfig | None = None,
    ) -> Any:
        """Dispose the live dialect, apply fresh credentials, and open a replacement handle."""
        resolved = MainExecutionOps.resolve_connection_credentials(credentials, token_provider)
        MainExecutionOps.dispose_engine_dialect(
            dialect,
            borrowed_execution_engine=execution_engine,
            borrowed_native_connection=native_connection,
        )
        runtime_cfg = runtime or getattr(dialect, "config", None)
        runtime_cfg = MainExecutionOps.apply_connection_credentials_for_engine(
            engine_type, resolved, runtime=runtime_cfg
        )
        try:
            return DialectRegistry.get_dialect(
                engine_type,
                runtime_cfg,
                sqlalchemy_engine=execution_engine,
                native_connection=native_connection,
            )
        except DatabaseConnectionError:
            raise
        except Exception as exc:
            raise DatabaseConnectionError(str(exc)) from exc

    @staticmethod
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
                except (OSError, AttributeError, TypeError):
                    pass
            sa = getattr(runtime, "sqlalchemy_engine", None)
            if sa is not None and id(sa) not in borrowed_sa:
                dispose_sa = getattr(sa, "dispose", None)
                if callable(dispose_sa):
                    try:
                        dispose_sa()
                    except (OSError, AttributeError, TypeError):
                        pass
            native = getattr(runtime, "native_connection", None)
            if native is not None and id(native) not in borrowed_native:
                close_native = getattr(native, "close", None)
                if callable(close_native):
                    try:
                        close_native()
                    except (OSError, AttributeError, TypeError):
                        pass

    @staticmethod
    def clear_federation_template_stores(
        federation_dir: str | None,
        composite_artifacts_dir: str,
        composite_graph: SchemaGraph,
        member_engines: Mapping[str, Any],
    ) -> bool:
        """Clear composite, plan-record, and member template stores for a federation."""
        existed = MainExecutionOps.clear_template_store_only(composite_artifacts_dir, composite_graph)
        if federation_dir:
            clear_federation_plan_templates(federation_dir)
        for engine in member_engines.values():
            graph = getattr(engine, "_schema_graph", None)
            adir = getattr(engine, "_artifacts_dir", None)
            if graph is not None and adir is not None:
                existed = MainExecutionOps.clear_template_store_only(str(adir), graph) or existed
        return existed

    @staticmethod
    def describe_federation_config(
        federation_name: str,
        runtime: RuntimeConfig,
        llm: LLMConfig,
        *,
        members: Mapping[str, Any],
        federation_storage_dir: str | None = None,
    ) -> str:
        """Build a redacted config snapshot including federation topology."""
        lines = [MainExecutionOps.describe_runtime_config(runtime, llm), "", "Federation:"]
        lines.append(f"  name:          {federation_name}")
        if federation_storage_dir:
            lines.append(f"  storage dir:   {os.path.abspath(federation_storage_dir)}")
        lines.append(f"  member count:  {len(members)}")
        for connection_name, engine in sorted(members.items()):
            member_engine = str(getattr(engine, "dialect", "") or "")
            member_dir = os.path.abspath(str(getattr(engine, "_artifacts_dir", "") or ""))
            lines.append(f"  {connection_name}: engine={member_engine!r} artifacts_dir={member_dir}")
        return "\n".join(lines)

    @staticmethod
    def clear_simulation_caches_only(artifacts_dir: str) -> int:
        """Remove QSim and seed-warmup simulation artifacts; return count of files removed."""
        count = wipe_filenames(artifacts_dir, SIMULATION_CACHE_EXACT_FILENAMES)
        count += wipe_globs(artifacts_dir, SIMULATION_CACHE_GLOB_PATTERNS)
        return count

    @staticmethod
    def resolve_qsim_path(version_or_result: int | QSimSummary, artifacts_dir: str) -> str:
        """Resolve the full file path for a QSim questions text artifact."""
        if isinstance(version_or_result, QSimSummary):
            ver = version_or_result.version
        else:
            ver = int(version_or_result)
        return os.path.join(artifacts_dir, QSIM_QUESTIONS_PATTERN.format(version=ver))

    @staticmethod
    def load_qsim_summaries(artifacts_dir: str) -> list[QSimSummary]:
        """Load every ``QSimSummary`` from per-run files under ``qsim/``, oldest first."""
        qsim_dir = os.path.join(artifacts_dir, "qsim")
        index_path = os.path.join(qsim_dir, "index.jsonl")
        if os.path.isfile(index_path):
            summaries: list[QSimSummary] = []
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    run_id = str(row.get("run_id") or row.get("version") or "").strip()
                    if not run_id:
                        continue
                    summary_path = os.path.join(qsim_dir, f"summary_{run_id}.json")
                    if not os.path.isfile(summary_path):
                        continue
                    try:
                        with open(summary_path, encoding="utf-8") as sf:
                            payload = json.load(sf)
                    except (json.JSONDecodeError, OSError):
                        continue
                    if isinstance(payload, dict):
                        summaries.append(QSimSummary.from_dict(payload))
            return summaries
        qsim_summary_path = os.path.join(artifacts_dir, "qsim_summary.json")
        if not os.path.exists(qsim_summary_path):
            return []
        with open(qsim_summary_path, encoding="utf-8") as f:
            summaries_raw: Any = json.load(f)
        if not isinstance(summaries_raw, list):
            return []
        return [QSimSummary.from_dict(s) for s in summaries_raw if isinstance(s, dict)]

    @staticmethod
    def _validate_yes_no_reply_token(token: str, *, param: str) -> None:
        if token not in ("y", "n"):
            raise ValueError(f"{param} must be 'y' or 'n'")

    @staticmethod
    def _normalise_yes_no(raw: str, options: list[str]) -> str | None:
        """Map free text to ``y`` or ``n`` when present in *options*."""
        token = raw.strip().lower()
        if token in ("y", "yes") and "y" in options:
            return "y"
        if token in ("n", "no") and "n" in options:
            return "n"
        return None

    @staticmethod
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
        return MainExecutionOps.get_seed_warmup_summary_from_dir(artifacts_dir, best_ver)

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
    def _serialize_interpretation(interp: IntentInterpretation) -> dict[str, Any]:
        return {
            "approach": interp.approach,
            "grounding": [list(pair) for pair in interp.grounding],
        }

    @staticmethod
    def _deserialize_interpretation(raw: dict[str, Any]) -> IntentInterpretation:
        grounding_raw = raw.get("grounding") or []
        return IntentInterpretation(
            approach=str(raw["approach"]),
            grounding=tuple(tuple(pair) for pair in grounding_raw),
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
        out: dict[str, Any] = {
            "handle": binding.handle,
            "current_value": binding.current_value,
            "display_name": binding.display_name,
            "column_expr": binding.column_expr,
        }
        if binding.upper_handle:
            out["upper_handle"] = binding.upper_handle
        if binding.unit_handle:
            out["unit_handle"] = binding.unit_handle
        return out

    @staticmethod
    def _serialize_session_notice(notice: SessionNotice) -> dict[str, Any]:
        return {"code": notice.code, "level": notice.level, "message": notice.message}

    @staticmethod
    def _deserialize_session_notice(raw: dict[str, Any]) -> SessionNotice:
        level_raw = str(raw["level"])
        level: Literal["info", "warning", "error"] = (
            cast(Literal["info", "warning", "error"], level_raw)
            if level_raw in ("info", "warning", "error")
            else "info"
        )
        return SessionNotice(
            code=str(raw["code"]),
            level=level,
            message=str(raw["message"]),
        )

    @staticmethod
    def _deserialize_parameter_binding(raw: dict[str, Any]) -> ParameterBinding:
        current_raw = raw.get("current_value")
        current_value = None if current_raw is None else MainExecutionOps._deserialize_param_value(current_raw)
        return ParameterBinding(
            handle=str(raw["handle"]),
            current_value=current_value,
            display_name=str(raw.get("display_name") or ""),
            column_expr=str(raw.get("column_expr") or ""),
            upper_handle=str(raw.get("upper_handle") or ""),
            unit_handle=str(raw.get("unit_handle") or ""),
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
            {key: MainExecutionOps._json_encode_session_cell(val) for key, val in row.items()}
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
            "message": step.message,
            "error": step.error,
            "status": step.status,
            "reply_shape": step.reply_shape,
            "semantic_warnings": list(step.semantic_warnings),
            "retryable": step.retryable,
            "diagnostics": [MainExecutionOps._serialize_diagnostic(d) for d in step.diagnostics],
            "parameters": [MainExecutionOps._serialize_parameter_binding(p) for p in step.parameters],
            "federation_source_id": step.federation_source_id,
            "federation_phase": step.federation_phase,
            "federation_limit_key": step.federation_limit_key,
            "federation_succeeded": [list(row) for row in step.federation_succeeded],
            "notices": [MainExecutionOps._serialize_session_notice(n) for n in step.notices],
            "data_truncated": step.data_truncated,
            "refusal_code": step.refusal_code,
            "refusal_diagnostic_code": step.refusal_diagnostic_code,
            "template_id": step.template_id,
            "meta_payload": step.meta_payload,
        }
        if step.data is not None:
            payload["data"] = MainExecutionOps._serialize_dataframe(step.data)
        if step.intent_summary is not None:
            payload["intent_summary"] = MainExecutionOps._serialize_intent_summary(step.intent_summary)
        if step.interpretation is not None:
            payload["interpretation"] = MainExecutionOps._serialize_interpretation(step.interpretation)
        return payload

    @staticmethod
    def deserialize_session_step(payload: dict[str, Any]) -> SessionStep:
        """Rebuild a :class:`SessionStep` from *payload*, refusing on version mismatch."""
        MainExecutionOps._check_session_persistence_format_version(payload)
        data_out: pandas.DataFrame | None = None
        data_raw = payload.get("data")
        if data_raw is not None:
            data_out = MainExecutionOps._deserialize_dataframe(data_raw)
        intent_summary_raw = payload.get("intent_summary")
        intent_summary = (
            MainExecutionOps._deserialize_intent_summary(intent_summary_raw) if intent_summary_raw is not None else None
        )
        interpretation_raw = payload.get("interpretation")
        interpretation = (
            MainExecutionOps._deserialize_interpretation(interpretation_raw) if interpretation_raw is not None else None
        )
        diagnostics_raw = payload.get("diagnostics") or []
        diagnostics = tuple(MainExecutionOps._deserialize_diagnostic(d) for d in diagnostics_raw)
        parameters_raw = payload.get("parameters") or []
        parameters = tuple(MainExecutionOps._deserialize_parameter_binding(p) for p in parameters_raw)
        federation_succeeded_raw = payload.get("federation_succeeded") or []
        federation_succeeded = tuple(tuple(row) for row in federation_succeeded_raw)
        notices_raw = payload.get("notices") or []
        notices = tuple(MainExecutionOps._deserialize_session_notice(n) for n in notices_raw)
        reply_shape = payload.get("reply_shape")
        if reply_shape is not None and reply_shape not in ("yes_no", "free_text"):
            raise ValueError(f"invalid reply_shape: {reply_shape!r}")
        sql_raw = payload.get("sql")
        if sql_raw is not None and not isinstance(sql_raw, (str, dict)):
            raise ValueError(f"invalid sql payload type: {type(sql_raw)!r}")
        meta_raw = payload.get("meta_payload")
        if meta_raw is not None and not isinstance(meta_raw, dict):
            raise ValueError(f"invalid meta_payload type: {type(meta_raw)!r}")
        return SessionStep(
            done=bool(payload["done"]),
            prompt=payload.get("prompt"),
            kind=str(payload["kind"]),
            sql=sql_raw,
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
            refusal_code=payload.get("refusal_code"),
            refusal_diagnostic_code=payload.get("refusal_diagnostic_code"),
            template_id=str(payload["template_id"]) if payload.get("template_id") is not None else None,
            meta_payload=meta_raw,
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
            "matched_template": MainExecutionOps._serialize_template_ref(gen_out.matched_template),
            "join_matches_template": gen_out.join_matches_template,
            "error_kind": gen_out.error_kind,
            "refusal_diagnostic_code": gen_out.refusal_diagnostic_code,
            "federation_plan_id": gen_out.federation_plan_id,
        }

    @staticmethod
    def _deserialize_sql_generation_outcome(raw: dict[str, Any]) -> SqlGenerationOutcome:
        matched = MainExecutionOps._deserialize_template_ref(raw.get("matched_template"))
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
            "matched_template": MainExecutionOps._serialize_template_ref(tail.matched_template),
            "union_select_cols": list(tail.union_select_cols) if tail.union_select_cols is not None else None,
            "structural_match_templates": [
                MainExecutionOps._serialize_template_ref(tmpl)
                for tmpl in tail.structural_match_templates
                if tmpl is not None
            ],
            "ikey": tail.ikey,
            "intent_sim": tail.intent_sim,
            "union_sql_path": union_sql_path,
            "union_candidate_template_ids": list(tail.union_candidate_template_ids),
            "form_storage": MainExecutionOps._serialize_question_form_storage(tail.form_storage),
            "interpretation": MainExecutionOps._serialize_interpret_plan(tail.interpretation),
        }

    @staticmethod
    def _deserialize_interactive_tail_snapshot(raw: dict[str, Any], *, owner: Any | None) -> InteractiveTailSnapshot:
        refs = MainExecutionOps._owner_learning_refs(owner)
        intent = RuntimeIntent.from_dict(raw.get("intent") or {})
        matched_template = MainExecutionOps._deserialize_template_ref(raw.get("matched_template"))
        structural_raw = raw.get("structural_match_templates") or []
        structural_match_templates = tuple(
            tmpl
            for tmpl in (MainExecutionOps._deserialize_template_ref(item) for item in structural_raw)
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
            form_storage=MainExecutionOps._deserialize_question_form_storage(raw.get("form_storage")),
            interpretation=MainExecutionOps._deserialize_interpret_plan(raw.get("interpretation")),
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
                "tail": MainExecutionOps._serialize_interactive_tail_snapshot(payload.tail),
                "gen_out": MainExecutionOps._serialize_sql_generation_outcome(payload.gen_out),
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
                "ref_tmpl": MainExecutionOps._serialize_template_ref(payload.ref_tmpl),
                "form_storage": MainExecutionOps._serialize_question_form_storage(payload.form_storage),
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
                "tail": MainExecutionOps._serialize_interactive_tail_snapshot(payload.tail),
                "gen_out": MainExecutionOps._serialize_sql_generation_outcome(payload.gen_out),
                "federation_plan_id": payload.federation_plan_id,
            }
        if state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM and isinstance(payload, InteractiveTailSnapshot):
            return {
                "type": "intent_confirm",
                "tail": MainExecutionOps._serialize_interactive_tail_snapshot(payload),
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
            tail = MainExecutionOps._deserialize_interactive_tail_snapshot(raw.get("tail") or {}, owner=owner)
            execution_intent = RuntimeIntent.from_dict(raw.get("execution_intent") or tail.intent.to_dict())
            return SqlFeedbackSuspendContext(
                tail=tail,
                execution_intent=execution_intent,
                sql=str(raw.get("sql") or ""),
                preview_rows=tuple(tuple(row) for row in (raw.get("preview_rows") or [])),
                sql_parameters=tuple((str(k), v) for k, v in (raw.get("sql_parameters") or [])),
                suspended_at=suspended_dt,
                tmpl_sd=raw.get("tmpl_sd"),
                gen_out=MainExecutionOps._deserialize_sql_generation_outcome(raw.get("gen_out") or {}),
                matched_rejected_template=None,
                force_feedback=bool(raw.get("force_feedback", False)),
            )
        if payload_type == "direct_reuse":
            refs = MainExecutionOps._owner_learning_refs(owner)
            ref_tmpl = MainExecutionOps._deserialize_template_ref(raw.get("ref_tmpl"))
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
                form_storage=MainExecutionOps._deserialize_question_form_storage(raw.get("form_storage")),
            )
        if payload_type == "sql_execute":
            suspended_at = raw.get("suspended_at")
            suspended_dt = datetime.fromisoformat(suspended_at) if isinstance(suspended_at, str) else None
            tail = MainExecutionOps._deserialize_interactive_tail_snapshot(raw.get("tail") or {}, owner=owner)
            execution_intent = RuntimeIntent.from_dict(raw.get("execution_intent") or tail.intent.to_dict())
            return SqlExecuteSuspendContext(
                tail=tail,
                execution_intent=execution_intent,
                sql=str(raw.get("sql") or ""),
                gen_out=MainExecutionOps._deserialize_sql_generation_outcome(raw.get("gen_out") or {}),
                matched_rejected_template=None,
                force_feedback=bool(raw.get("force_feedback", False)),
                tmpl_sd=raw.get("tmpl_sd"),
                preview_rows=tuple(tuple(row) for row in (raw.get("preview_rows") or [])),
                sql_parameters=tuple((str(k), v) for k, v in (raw.get("sql_parameters") or [])),
                suspended_at=suspended_dt,
                federation_plan_id=str(raw.get("federation_plan_id") or ""),
            )
        if payload_type == "intent_confirm":
            return MainExecutionOps._deserialize_interactive_tail_snapshot(raw.get("tail") or {}, owner=owner)
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
            serialized_payload = MainExecutionOps._serialize_pipeline_suspend_payload(state_id, suspend_payload) or {}
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
        MainExecutionOps._check_suspend_state_format_version(payload)
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
            suspend_payload = MainExecutionOps._deserialize_pipeline_suspend_payload(
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
    def _owner_business_knowledge_scope(owner: Any) -> Any:
        """Bind the owner's stored business knowledge for nested pipeline work."""
        holder = getattr(owner, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            yield
            return
        with business_knowledge_scope(**holder.scope_kwargs()):
            yield


class PipelineSession(InteractiveChoicePort):
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
        "_active_federation_execution_context",
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
        self._active_federation_execution_context: FederationExecutionContext | None = None
        self._pending_federation_plan_template: FederationPlanTemplate | None = None
        self._pending_terminal_step: SessionStep | None = None

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
        return MainExecutionOps.serialize_suspended_state(
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
        fields = MainExecutionOps.deserialize_suspended_state(payload, owner=owner)
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
        owner_schema = getattr(self._owner, "_schema_graph", None)
        schema_hash_val: str | None = None
        if owner_schema is not None:
            schema_hash_val = getattr(owner_schema, "effective_structural_hash", None)
        fn(event_type, question=question, schema_hash=schema_hash_val, details=details_with_turn_id(details))

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
            self._turn_llm_usage_summary = None
            return diagnostics
        provider_raw = str(getattr(getattr(self._owner, "_llm_config", None), "provider", "openai"))
        provider: Literal["openai", "azure", "mock"]
        if provider_raw in ("openai", "azure", "mock"):
            provider = cast(Literal["openai", "azure", "mock"], provider_raw)
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
        drain_llm_usage_records()
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
            return replace(st, status=MainExecutionOps._failure_category_for_terminal_step(st))
        dialect = self._owner._dialect
        schema, store, templates, rejected, schema_terms = self._resources()
        corrected = ctx.corrected_question
        q_norm = normalize_question(corrected)
        while True:
            try:
                with llm_execution_scope(self._owner._runtime_config.llm_execution):
                    ok = MainExecutionOps._interactive_run_intent_pass(
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
                        persist_template_learning=MainExecutionOps._persist_template_learning_for_pipeline_session(
                            self
                        ),
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
                        return replace(st, status=MainExecutionOps._failure_category_for_terminal_step(st))
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
        space_name = TemplateOps.validate_space_name(self._space_name)
        cached_store = MainExecutionOps._owner_template_store_for_space(owner, space_name)
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
            MainExecutionOps._sync_owner_template_cache(owner, store, space_name=space_name)
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
        return sorted(
            ("accept_until_done", "ask", "ask_until_done", "awaiting_prompt", "reset", "reuse_saved_question", "step")
        )

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
        self._active_federation_execution_context = None
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
            ctx = self._active_federation_execution_context
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
        return MainExecutionOps._normalise_yes_no(raw, options)

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
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        if not question.strip():
            self._audit_ask_emit(
                "ask_blocked",
                question=question,
                details=(("reason", "empty_question"),),
            )
            st = self._mk_step(done=True, prompt=None, kind=SESSION_KIND_ERROR, error="Question must not be empty.")
            return replace(st, status=MainExecutionOps._failure_category_for_terminal_step(st))
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
            return self._drive_question_turn(question)
        finally:
            reset_diagnostic_collector(tok)

    def ask_until_done(self, question: str, *, on_confirm: Literal["y", "n"] = "y") -> SessionStep:
        """Run ``ask`` then auto-answer yes or no suspends with *on_confirm* until the turn ends. When the user declines executed SQL on the final yes or no prompt, the terminal :class:`SessionStep` carries ``status`` ``FailureCategory.RESULT_OKAY_INTENT_WRONG`` so programmatic callers can distinguish validated-but-rejected runs from unconditional success."""
        if not isinstance(question, str):
            raise TypeError("question must be str")
        MainExecutionOps._validate_yes_no_reply_token(on_confirm, param="on_confirm")
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
        MainExecutionOps._validate_yes_no_reply_token(on_yes_no, param="on_yes_no")
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
        persist_tl = MainExecutionOps._persist_template_learning_for_pipeline_session(self)
        gate_kwargs = MainExecutionOps._consumer_sql_gate_kwargs(self)

        def _run_forced() -> SessionStep:
            identity = self._owner_engine_identity()
            identity_token = push_engine_identity(identity)
            limits_token = push_engine_limits(getattr(owner, "limits", EngineLimits()))
            sandbox_runtime = getattr(owner, "_sandbox_runtime", None)
            sandbox_runtime_token = (
                SandboxRuntimeState.bind_sandbox_runtime(sandbox_runtime) if sandbox_runtime is not None else None
            )
            try:
                with MainExecutionOps._owner_business_knowledge_scope(owner):
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
                                **MainExecutionOps._federation_reuse_kwargs(owner, self),
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
                            return replace(st_err, status=MainExecutionOps._failure_category_for_terminal_step(st_err))
                        except Exception as exc:
                            debug(f"[main_execution.PipelineSession.reuse_saved_question] unexpected error: {exc!r}")
                            self._reset_after_turn()
                            self._release_session_turn()
                            return self._terminal_error_from_exception(exc)
                        return self._completed_step()
            finally:
                if sandbox_runtime_token is not None:
                    SandboxRuntimeState.reset_sandbox_runtime(sandbox_runtime_token)
                pop_engine_limits(limits_token)
                pop_engine_identity(identity_token)

        lock = getattr(owner, "_pipeline_writer_lock", None)
        art = getattr(owner, "_artifacts_dir", None)
        adir = ""
        if art is not None:
            try:
                adir = os.path.abspath(os.fspath(art))
            except (TypeError, OSError, ValueError):
                adir = ""
        if self._session_mode == "writer" and lock is not None and adir:
            with lock:
                MainExecutionOps.drain_write_queue(owner, adir)
        with llm_usage_session_scope():
            return _run_forced()

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
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        self._enforce_suspended_session_ttl()
        buf = diagnostic_segment()
        for _orph in take_and_clear_orphan_diagnostics(self._owner_engine_identity()):
            buf.append(_orph)
        tok = set_diagnostic_collector(buf)
        try:
            pending = self._pending_terminal_step
            if pending is not None:
                self._pending_terminal_step = None
                return pending
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
        normalised = MainExecutionOps._normalise_yes_no(raw, ["y", "n"])
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
        sql_out: str | dict[str, str] | None = None
        data_out: pandas.DataFrame | None = None
        body: str | None = None
        prompt_out = SESSION_PROMPT_YESNO
        isum: IntentSummary | None = None
        reply_shape: Literal["yes_no", "free_text"] | None = "yes_no"
        sem_w: tuple[str, ...] = ()
        interpretation: IntentInterpretation | None = None
        parameters: tuple[ParameterBinding, ...] = ()
        template_id_out: str | None = None
        matched_for_params: Template | None = None
        intent_for_params: RuntimeIntent | None = None

        if ex.state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM and isinstance(payload, InteractiveTailSnapshot):
            body, sem_w = compose_intent_confirm_session_message(payload.intent, list(payload.semantic_warnings))
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainExecutionOps._build_intent_summary(payload.intent)
            interpretation = MainExecutionOps._intent_interpretation_from_plan(payload.interpretation)
            reply_shape = "yes_no"
        elif ex.state_id == PIPELINE_SUSPEND_ID_EXECUTE and isinstance(payload, SqlExecuteSuspendContext):
            ctx_exec = payload
            sql_out = MainExecutionOps._resolved_session_step_sql(
                ctx_exec.sql,
                gen_out=ctx_exec.gen_out,
                federated_bundle=getattr(ctx_exec, "federated_bundle", None),
                federated_plan=(ctx_exec.federated_prepare.plan if ctx_exec.federated_prepare is not None else None),
                generation_path=ctx_exec.gen_out.generation_path,
            )
            if sql_out is None:
                sql_out = ctx_exec.sql
            body = ""
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainExecutionOps._build_intent_summary(ctx_exec.execution_intent)
            reply_shape = "yes_no"
            sem_w = ()
            matched_for_params = ctx_exec.gen_out.matched_template
            intent_for_params = ctx_exec.execution_intent
        elif ex.state_id == PIPELINE_SUSPEND_ID_SQL and isinstance(payload, SqlFeedbackSuspendContext):
            ctxp = payload
            body = ""
            sql_out = MainExecutionOps._resolved_session_step_sql(
                ctxp.sql,
                gen_out=ctxp.gen_out,
                federated_bundle=ctxp.federated_bundle,
                federated_plan=ctxp.federated_prepare.plan if ctxp.federated_prepare is not None else None,
                generation_path=ctxp.gen_out.generation_path,
            )
            if sql_out is None:
                sql_out = ctxp.sql
            full_df = build_result_dataframe(
                list(ctxp.preview_rows),
                ctxp.execution_intent,
                ctxp.sql if isinstance(ctxp.sql, str) else "",
                structural_defaults=ctxp.tmpl_sd,
                q_norm=ctxp.tail.q_norm,
                template_display_alias_map=(
                    getattr(ctxp.gen_out.matched_template, "display_alias_map", None)
                    if ctxp.gen_out.matched_template
                    else None
                ),
                **MainExecutionOps._federation_result_contract_kwargs(
                    ctxp.gen_out, federated_prepare=ctxp.federated_prepare, federated_bundle=ctxp.federated_bundle
                ),
            )
            if full_df is not None and not full_df.empty:
                data_out = full_df.head(5)
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainExecutionOps._build_intent_summary(ctxp.execution_intent)
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
                    data_out = pandas.DataFrame([list(r) for r in rows_list], columns=hdr).head(5)
                else:
                    data_out = pandas.DataFrame([list(r) for r in rows_list]).head(5)
            body = ""
            prompt_out = SESSION_PROMPT_YESNO
            isum = MainExecutionOps._build_intent_summary(ctx.intent)
            reply_shape = "yes_no"
            sem_w = ()
            matched_for_params = ctx.ref_tmpl if isinstance(ctx.ref_tmpl, Template) else None
            intent_for_params = ctx.intent
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

        if sql_out is not None:
            parameters = self._parameters_for_sql_bearing_step(
                sql=sql_out,
                matched_template=matched_for_params,
                intent=intent_for_params,
                question_nl=self._turn_question or "",
                persist_display_names=False,
            )
            if matched_for_params is not None:
                template_id_out = str(matched_for_params.id)

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
            message=body,
            intent_summary=isum,
            reply_shape=reply_shape,
            semantic_warnings=sem_w,
            interpretation=interpretation,
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
            message=pending.message,
            error=pending.error,
            status=pending.status,
            meta_payload=pending.meta_payload,
            notices=tuple(pending.notices or ()),
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
            reply_shape=None,
            intent_summary=None,
            semantic_warnings=(),
            parameters=(),
            federated_bundle=None,
        )
        self._reset_after_turn()
        self._release_session_turn()
        return step

    def _completed_step(self) -> SessionStep:
        """Build a terminal :class:`SessionStep` after a full successful pipeline pass."""
        snap = self._last_turn_outcome or {}
        qtxt = self._turn_question or ""
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
        sql_out = MainExecutionOps._resolved_session_step_sql(
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
                sql_for_headers, list(rows_tuple), **MainExecutionOps._federation_contract_kwargs_from_snap(snap)
            )
        data_out: pandas.DataFrame | None = None
        if rows_tuple:
            sql_for_headers = sql_out if isinstance(sql_out, str) else None
            cols_use = (
                list(cols_tuple)
                if cols_tuple
                else result_columns_for_session(
                    sql_for_headers, list(rows_tuple), **MainExecutionOps._federation_contract_kwargs_from_snap(snap)
                )
            )
            if cols_use:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple], columns=list(cols_use))
            else:
                data_out = pandas.DataFrame([list(r) for r in rows_tuple])
        terminal_notices: tuple[SessionNotice, ...] = ()
        terminal_refusal_code: str | None = None
        raw_outcome = str(snap.get("outcome") or "success")
        if raw_outcome == "success":
            terminal_message = None
            terminal_status: str | None = None
            terminal_notices = (SessionNotice(code="turn_saved", level="info", message=SAVED_LINE),)
        elif raw_outcome == "permission_denied":
            terminal_message = refusal_user_text_for_code(
                refusal_diagnostic_code_for_outcome("permission_denied") or ""
            )
            terminal_status = "permission_denied"
            terminal_refusal_code = refusal_diagnostic_code_for_outcome("permission_denied")
            sql_out = None
            data_out = None
        elif raw_outcome == "restricted":
            terminal_message = refusal_user_text_for_code(refusal_diagnostic_code_for_outcome("restricted") or "")
            terminal_status = "restricted"
            terminal_refusal_code = refusal_diagnostic_code_for_outcome("restricted")
            sql_out = None
            data_out = None
        elif raw_outcome == "invalid_question":
            terminal_message = refusal_user_text_for_code(refusal_diagnostic_code_for_outcome("invalid_question") or "")
            terminal_status = "invalid_question"
            terminal_refusal_code = refusal_diagnostic_code_for_outcome("invalid_question")
            sql_out = None
            data_out = None
        elif raw_outcome == "parse_failed":
            parse_code = refusal_diagnostic_code_for_outcome("parse_failed")
            terminal_message = str(snap.get("error") or refusal_user_text_for_code(parse_code or ""))
            terminal_status = FailureCategory.INTENT_PARSE_FAILED.value
            terminal_refusal_code = parse_code
            sql_out = None
            data_out = None
        elif raw_outcome == "validation_failed":
            terminal_message = str(snap.get("error") or REPHRASE_HINT_MESSAGES["sql_validation_failed"])
            terminal_status = "validation_failed"
            sql_out = None
            data_out = None
        elif raw_outcome == "schema_invalid_declined":
            declined_code = refusal_diagnostic_code_for_outcome("schema_invalid_declined")
            terminal_message = refusal_user_text_for_code(declined_code or "")
            terminal_status = "schema_invalid_declined"
            terminal_refusal_code = declined_code
        elif raw_outcome == "federation_partial_failure":
            terminal_message = REPHRASE_HINT_MESSAGES["federation_partial_failure"]
            terminal_status = "federation_partial_failure"
            sql_out = None
            data_out = None
        elif raw_outcome == "federation_turn_cancelled":
            terminal_message = REPHRASE_HINT_MESSAGES["federation_turn_cancelled"]
            terminal_status = FailureCategory.FEDERATION_TURN_CANCELLED.value
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
            isum_res = MainExecutionOps._build_intent_summary(ri)
        if raw_outcome == "permission_denied":
            err_snap = None
        elif raw_outcome == "federation_partial_failure":
            err_snap = None
        elif raw_outcome == "federation_turn_cancelled":
            err_snap = None
        parameters = (
            self._parameters_for_completed_turn(snap, qtxt) if raw_outcome == "success" and sql_out is not None else ()
        )
        matched_tmpl = snap.get("matched_template")
        template_id_out: str | None = None
        if raw_outcome == "success" and isinstance(matched_tmpl, Template):
            template_id_out = str(matched_tmpl.id)
        elif raw_outcome == "success" and sql_out is not None and qtxt.strip():
            schema, store, templates, _, _ = self._resources()
            resolved = TemplateOps.resolve_template_for_question(qtxt, templates, template_store=store)
            if resolved is not None:
                template_id_out = str(resolved[0].id)
        turn_diagnostics = self._emit_turn_llm_usage(question=qtxt, diagnostics=())
        refusal_diagnostic_code = snap.get("refusal_diagnostic_code")
        if refusal_diagnostic_code and raw_outcome == "validation_failed":
            refusal_diagnostic_code = str(refusal_diagnostic_code)
            terminal_refusal_code = refusal_diagnostic_code
            refusal_msg = str(snap.get("error") or terminal_message or "")
            turn_diagnostics = turn_diagnostics + (
                Diagnostic(
                    stage="validation",
                    level="error",
                    code=refusal_diagnostic_code,
                    message=refusal_msg,
                    phase="validation",
                    details=details_with_turn_id(),
                ),
            )
        elif terminal_refusal_code is not None:
            refusal_diagnostic_code = terminal_refusal_code
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
                        message=terminal_message or "",
                        details=tuple(partial_details),
                        source_id=partial_source,
                        phase=partial_phase,
                    ),
                )
        if raw_outcome == "restricted" and terminal_message:
            turn_diagnostics = turn_diagnostics + (
                Diagnostic(
                    stage="rephrase_hint",
                    level="info",
                    code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    message=terminal_message,
                    remediation=REMEDIATION_RESTRICTED_QUESTION,
                    phase="rephrase_hint",
                ),
            )
        audit_details: list[tuple[str, str]] = [("outcome", raw_outcome)]
        terminal_kind = SESSION_KIND_RESULT
        if raw_outcome == "federation_turn_cancelled":
            terminal_kind = SESSION_KIND_ERROR
        audit_details.append(("kind", terminal_kind))
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
                        phase="execution",
                    ),
                )
        fed_step_fields = MainExecutionOps._session_step_federation_fields_from_snap(snap, raw_outcome)
        data_out, data_truncated = self._apply_data_row_cap(data_out)
        step = self._mk_step(
            done=True,
            prompt=None,
            kind=terminal_kind,
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
            refusal_code=terminal_refusal_code,
            refusal_diagnostic_code=refusal_diagnostic_code,
            template_id=template_id_out,
            **fed_step_fields,
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
            error=message,
            diagnostics=self._terminal_turn_diagnostics(turn_diagnostics),
        )
        return replace(st, status=FailureCategory.TURN_CANCELLED.value)

    def _terminal_error_step(self, message: str, *, exc: BaseException | None = None) -> SessionStep:
        """Build a terminal error :class:`SessionStep`."""
        fed_fields = MainExecutionOps._federation_error_step_fields(exc) if exc is not None else {}
        fed_diag = MainExecutionOps._federation_error_diagnostics(exc) if exc is not None else ()
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
        return replace(st, status=MainExecutionOps._failure_category_for_terminal_step(st))

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
            self._session_mode == "writer" and MainExecutionOps._persist_template_learning_for_pipeline_session(self)
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
            identity = self._owner_engine_identity()
            identity_token = push_engine_identity(identity)
            limits_token = push_engine_limits(getattr(self._owner, "limits", EngineLimits()))
            sandbox_runtime = getattr(self._owner, "_sandbox_runtime", None)
            sandbox_runtime_token = (
                SandboxRuntimeState.bind_sandbox_runtime(sandbox_runtime) if sandbox_runtime is not None else None
            )
            with llm_usage_session_scope():
                with llm_execution_scope(self._owner._runtime_config.llm_execution):
                    try:
                        return _run_turn_inner()
                    finally:
                        if sandbox_runtime_token is not None:
                            SandboxRuntimeState.reset_sandbox_runtime(sandbox_runtime_token)
                        pop_engine_limits(limits_token)
                        pop_engine_identity(identity_token)

        def _run_turn_inner() -> SessionStep:
            try:
                if owner_is_aether_federation(self._owner):
                    members = getattr(self._owner, "_members", None)
                    if isinstance(members, dict) and members:
                        probe_federation_member_liveness(members)
                MainExecutionOps.interactive_run_once(
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
                    return replace(st_e, status=MainExecutionOps._failure_category_for_terminal_step(st_e))
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
            with MainExecutionOps._owner_business_knowledge_scope(self._owner):
                lock = getattr(self._owner, "_pipeline_writer_lock", None)
                if self._session_mode == "reader":
                    if adir:
                        MainExecutionOps._reload_reader_learning_if_manifest_drift(self._owner)
                    return _run_turn()
                if lock is not None and adir:
                    with lock:
                        MainExecutionOps.drain_write_queue(self._owner, adir)
                return _run_turn()
        finally:
            pop_turn_id(turn_id_token)
            pop_ask_phase_callback(ask_phase_token)
            pop_session_turn_cancel(cancel_token)

    def _resume_from_suspend(self) -> SessionStep:
        """Continue execution after enqueueing a programmatic answer."""
        if self._suspended is None:
            self._release_session_turn()
            st0 = self._mk_step(done=True, prompt=None, kind=SESSION_KIND_ERROR, error="No pending prompt.")
            return replace(st0, status=MainExecutionOps._failure_category_for_terminal_step(st0))
        ex = self._suspended
        self._suspended = None
        self._resume_choice_stage_id = ex.state_id

        def _resume_work() -> None:
            identity = self._owner_engine_identity()
            identity_token = push_engine_identity(identity)
            limits_token = push_engine_limits(getattr(self._owner, "limits", EngineLimits()))
            sandbox_runtime = getattr(self._owner, "_sandbox_runtime", None)
            sandbox_runtime_token = (
                SandboxRuntimeState.bind_sandbox_runtime(sandbox_runtime) if sandbox_runtime is not None else None
            )
            try:
                with llm_execution_scope(self._owner._runtime_config.llm_execution):
                    MainExecutionOps.dispatch_pipeline_resume(self, ex)
            finally:
                if sandbox_runtime_token is not None:
                    SandboxRuntimeState.reset_sandbox_runtime(sandbox_runtime_token)
                pop_engine_limits(limits_token)
                pop_engine_identity(identity_token)

        cancel_token = push_session_turn_cancel(self._turn_cancel_event)
        ask_phase_token = push_ask_phase_callback(getattr(self._owner, "_ask_phase_callback", None))
        refinement_retry = False
        try:
            with MainExecutionOps._owner_business_knowledge_scope(self._owner):
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
                            MainExecutionOps.drain_write_queue(self._owner, adir)
                with llm_usage_session_scope():
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
            if not refinement_retry:
                pop_ask_phase_callback(ask_phase_token)
                pop_session_turn_cancel(cancel_token)
        return self._completed_step()


register_structural_migration_handler(MainExecutionOps.apply_structural_migration_to_persisted_scopes)
