"""Public AetherEngine facade delegating construction and runners to main_execution. Attributes on AetherEngine whose names start with a single underscore are private implementation details and are not part of the public stability contract."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import pandas

from ._config import (
    EngineConfig,
    EngineLimits,
    FederationLimits,
    SeedWarmupConfig,
)
from ._constants import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    FEDERATION_DECLARATION_FILENAME,
    FEDERATION_DECLARATION_VERSION,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_MIGRATION_MAP_FILENAME,
    MASTER_AETHERSPACE_NAME,
    MASTER_AETHERSPACE_UID,
    MIGRATION_MAP_FILENAME,
    WRITE_QUEUE_FILENAME,
    YES_NO_SESSION_KINDS,
)
from ._constants_runtime import (
    FEDERATION_METHOD_SEMANTICS,
    FEDERATION_WARMUP_UNSUPPORTED_MESSAGE,
)
from ._contracts_base import (
    AetherError,
    AetherSpace,
    AetherspaceDeleteResult,
    ArtifactLockTimeoutError,
    ConfigError,
    DatabaseConnectionError,
    DatabaseExecutionError,
    DatabasePingFailed,
    DataQualityReport,
    Diagnostic,
    DomainKnowledgeEntry,
    DomainKnowledgeHolder,
    EngineContext,
    EngineIdentity,
    FederationCapExceededError,
    FederationConfigError,
    FederationContext,
    FederationDeclarationError,
    FederationIneligibleError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMappingsAppliedSidecarError,
    FederationMemberEngine,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationMemberUnprofilableError,
    FederationMethodScope,
    FederationPartialFailureError,
    FederationRuntimeError,
    FederationTurnCancelledError,
    LlmTransientFailure,
    MigrationPendingError,
    MigrationTier,
    MockFixtureMissingError,
    OwnerOnlyOperationError,
    RefreshReport,
    ResultCapExceededError,
    RetryableDatabaseExecutionError,
    RetryableError,
    RetryableFederationPartialFailureError,
    SchemaAccessError,
    SchemaInvariantError,
    SchemaRole,
    SessionActiveError,
    SessionTurnCancelledError,
    SpaceContext,
    StatementTimeoutError,
    StructureReport,
    SuspendedSessionExpiredError,
)
from ._contracts_core import (
    AccessError,
    AetherEngineInitResult,
    AetherFederationInitResult,
    AggregateJoinFanOutError,
    AmbiguousDateLiteralError,
    AuditEvent,
    ClauseWidenedRowsetError,
    ComparisonJoinScopeExceededError,
    ConfigSnapshot,
    FederatedPrepareOutcome,
    JoinCandidateCapExceededError,
    JoinColumnCountMismatchError,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    JoinPathKeyTypeError,
    JoinPathTieCapExceededError,
    JoinProbeEdgeKindMismatchError,
    LlmJsonExhausted,
    MigrationPreview,
    NoJoinPathError,
    NullInNegatedListError,
    PhaseProgressEvent,
    PipelineSuspended,
    ProbeCtePlacementError,
    QSimSummarySnapshot,
    RefinementRetry,
    RegistryRenderError,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    SessionError,
    SessionOutcome,
    SessionStep,
    StoredTemplateDetail,
    StoredTemplateSummary,
    SubdayDateWindowOnDateColumnError,
    Template,
    TemplateExecutionResult,
)
from ._contracts_schema import (
    CsvSourceSelection,
    FederationManifest,
    FederationMappings,
    FederationMappingSuggestion,
    PersistedFederationInspection,
    UploadIngestResult,
)
from ._data_quality import inspect_tabular_upload
from ._dialect_sqlglot_engines import CsvDialect
from ._expansion_ops import clear_expansion_subtree_pool
from ._federation_compose import collapsed_member_physical_table_names, validate_federation_context_against_mappings
from ._federation_execute import (
    federation_member_artifacts_dir_for_purge,
    finalize_federation_composite_overrides,
    inspect_persisted_federation,
    parse_federation_migration_map,
    prune_federation_plan_templates_for_sources,
    prune_federation_plan_templates_on_drift,
    purge_federation_member_artifacts,
    reconcile_authored_declaration_for_members,
)
from ._federation_manifest import (
    binding_from_member_engine,
    export_federation_declaration,
    federation_declaration_document,
    federation_members_mapping,
    load_federation_declaration_from_path,
    member_connection_name_from_engine,
    parse_federation_declaration,
)
from ._knowledge_staleness import knowledge_artifact_save_stamps
from ._llm_provider import LLMProvider
from ._main_execution import MainExecutionOps
from ._main_session import PipelineSession
from ._pipeline_execute import execute_stored_template_by_ref
from ._qsim import (
    clear_engine_skeleton_cache,
    drop_engine_skeleton_cache_owner,
    pop_qsim_engine_owner,
    pop_simulation_artifact_partition,
    push_qsim_engine_owner,
    push_simulation_artifact_scope_from_owner,
    register_engine_skeleton_cache_owner,
)
from ._sandbox import (
    Sandbox,
)
from ._schema_finalize import (
    apply_structure_document,
    build_public_structure_document,
    delete_persisted_structure_artifacts,
    dump_structure_edits,
)
from ._schema_graph import load_schema_graph_snapshot
from ._schema_profile import extract_domain_knowledge_from_notes, filter_schema_anchored_domain_knowledge
from ._templates import LazyTemplateMapping
from ._templates_ops import TemplateOps
from ._utils import (
    apply_federation_member_defaults,
    dataframe_to_row_tuples,
    delete_domain_knowledge_artifact,
    diagnostic_print_listener,
    echo_user_text,
    echo_yes_no_answer,
    error,
    load_domain_knowledge_artifact,
    notes_content_from_context,
    notify,
    owner_limits_scope,
    pop_construction_phase_callback,
    pop_diagnostic_sink,
    pop_engine_identity,
    print_query_result,
    progress_enabled,
    push_construction_phase_callback,
    push_diagnostic_sink,
    push_engine_identity,
    require_driver,
    stable_json,
    terminated,
    validate_federation_pool_capacity,
)
from ._utils_artifacts import (
    register_dialect_live_handles,
    release_close_resources,
    save_domain_knowledge_artifact,
)

aetherspace_descriptor_from_snapshot = MainExecutionOps.aetherspace_descriptor_from_snapshot
allocate_aetherspace_uid = MainExecutionOps.allocate_aetherspace_uid
build_master_space_descriptor = MainExecutionOps.build_master_space_descriptor
clear_federation_template_stores = MainExecutionOps.clear_federation_template_stores
clear_simulation_caches_only = MainExecutionOps.clear_simulation_caches_only
clear_template_store_only = MainExecutionOps.clear_template_store_only
delete_aetherspace = MainExecutionOps.delete_aetherspace
delete_aetherspace_snapshot = MainExecutionOps.delete_aetherspace_snapshot
describe_federation_config = MainExecutionOps.describe_federation_config
describe_runtime_config = MainExecutionOps.describe_runtime_config
dispose_engine_dialect = MainExecutionOps.dispose_engine_dialect
dispose_federation_source_runtimes = MainExecutionOps.dispose_federation_source_runtimes
drain_write_queue = MainExecutionOps.drain_write_queue
enrich_space_snapshot_with_notes = MainExecutionOps.enrich_space_snapshot_with_notes
engine_schema_json_path = MainExecutionOps.engine_schema_json_path
engine_template_store_dir = MainExecutionOps.engine_template_store_dir
consumer_safe_scope_context_fields = MainExecutionOps.consumer_safe_scope_context_fields
engine_context_references_out_of_scope = MainExecutionOps.engine_context_references_out_of_scope
build_named_schema_context_export = MainExecutionOps.build_named_schema_context_export
federation_stores_by_source = MainExecutionOps.federation_stores_by_source
find_latest_seed_warmup_summary = MainExecutionOps.find_latest_seed_warmup_summary
format_qsim_summary_line = MainExecutionOps.format_qsim_summary_line
format_seed_warmup_summary = MainExecutionOps.format_seed_warmup_summary
initialize_aether_engine = MainExecutionOps.initialize_aether_engine
initialize_aether_federation = MainExecutionOps.initialize_aether_federation
intersect_space_scope = MainExecutionOps.intersect_space_scope
list_named_schema_context_names = MainExecutionOps.list_named_schema_context_names
list_saved_aetherspace_entries = MainExecutionOps.list_saved_aetherspace_entries
list_saved_aetherspace_names = MainExecutionOps.list_saved_aetherspace_names
load_aetherspace_snapshot = MainExecutionOps.load_aetherspace_snapshot
load_named_schema_context = MainExecutionOps.load_named_schema_context
load_qsim_summaries = MainExecutionOps.load_qsim_summaries
preview_schema_migration = MainExecutionOps.preview_schema_migration
print_questions_bundle = MainExecutionOps.print_questions_bundle
qsim_run_once = MainExecutionOps.qsim_run_once
refresh_engine_connection = MainExecutionOps.refresh_engine_connection
resolve_aetherspace_identity = MainExecutionOps.resolve_aetherspace_identity
resolve_qsim_path = MainExecutionOps.resolve_qsim_path
run_seed_warmup_from_history_execution = MainExecutionOps.run_seed_warmup_from_history_execution
run_seed_warmup_from_query_log_execution = MainExecutionOps.run_seed_warmup_from_query_log_execution
save_aetherspace_snapshot = MainExecutionOps.save_aetherspace_snapshot
seed_warmup_run_once = MainExecutionOps.seed_warmup_run_once
space_allowed_sets_from_snapshot = MainExecutionOps.space_allowed_sets_from_snapshot
space_deny_sets_from_snapshot = MainExecutionOps.space_deny_sets_from_snapshot
subset_graph_for_space = MainExecutionOps.subset_graph_for_space
validate_space_context_against_graph = MainExecutionOps.validate_space_context_against_graph
filter_space_snapshot_sensitive_columns = MainExecutionOps.filter_space_snapshot_sensitive_columns
aetherspace_within_effective_visibility = MainExecutionOps.aetherspace_within_effective_visibility
validate_aetherspace_define_within_visibility = MainExecutionOps.validate_aetherspace_define_within_visibility
effective_visible_tables = MainExecutionOps.effective_visible_tables
effective_visible_columns = MainExecutionOps.effective_visible_columns

build_stored_template_detail = TemplateOps.build_stored_template_detail
list_stored_template_summaries = TemplateOps.list_stored_template_summaries
load_template_store = TemplateOps.load_template_store
primary_template_q_norm = TemplateOps.primary_template_q_norm
resolve_template_ref = TemplateOps.resolve_template_ref
store_to_templates = TemplateOps.store_to_templates
template_visible_to_callers = TemplateOps.template_visible_to_callers

__all__ = [
    "FEDERATION_METHOD_SEMANTICS",
    "FederationMethodScope",
    "MUTATING_ENGINE_METHODS",
    "MUTATING_FEDERATION_METHODS",
    "guarded_by_writer_lock",
]

MUTATING_ENGINE_METHODS: tuple[str, ...] = (
    "aetherspace",
    "apply_knowledge",
    "apply_migration_map",
    "apply_structure",
    "clear_all_learning",
    "clear_simulation_caches",
    "clear_template_store",
    "close",
    "delete_aetherspace",
    "execute_template",
    "ingest_upload_sources",
    "refresh",
    "run_interactive",
    "run_qsim",
    "run_seed_warmup",
    "run_seed_warmup_from_history",
    "run_seed_warmup_from_query_log",
)

MUTATING_FEDERATION_METHODS: tuple[str, ...] = (
    "add_engine",
    "aetherspace",
    "apply_federation",
    "apply_knowledge",
    "apply_migration_map",
    "apply_structure",
    "clear_all_learning",
    "clear_simulation_caches",
    "clear_template_store",
    "close",
    "delete_aetherspace",
    "execute_template",
    "remove_engine",
    "run_interactive",
    "run_qsim",
    "run_seed_warmup",
    "run_seed_warmup_from_history",
    "run_seed_warmup_from_query_log",
)


def guarded_by_writer_lock(method: Callable[..., Any]) -> bool:
    """Return whether *method* is wrapped by :func:`_writer_lock_guard`."""
    return bool(getattr(method, "_guarded_by_writer_lock", False))


def _writer_lock_guard(
    method: Callable[..., Any] | None = None,
    *,
    before_acquire: Callable[[Any, str], None] | None = None,
) -> Callable[..., Any]:
    """Serialize mutating facade entry points on ``_pipeline_writer_lock``."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if before_acquire is not None:
                before_acquire(self, fn.__name__)
            inst_dict = getattr(self, "__dict__", None)
            if isinstance(inst_dict, dict) and "_pipeline_writer_lock" in inst_dict:
                lock = inst_dict["_pipeline_writer_lock"]
            else:
                lock = getattr(self, "_pipeline_writer_lock", None)
            if lock is None:
                lock = self._pipeline_writer_lock = threading.Lock()
            with lock:
                return fn(self, *args, **kwargs)

        guarded: Any = wrapper
        guarded._guarded_by_writer_lock = True
        return wrapper

    if method is not None:
        return decorate(method)
    return decorate


def _init_log_sink(line: str) -> None:
    notify(line, stage="init", code=DIAGNOSTIC_CODE_ENGINE_INFO)


def _sql_text_for_print(sql: str | dict[str, str]) -> str:
    """Format session SQL for :func:`print_query_result` when federated bundles use dict form."""
    if isinstance(sql, str):
        return sql
    return json.dumps(sql, ensure_ascii=False)


class AsyncPipelineSession:
    """Async façade over :class:`PipelineSession` using worker threads."""

    __slots__ = ("_inner",)

    def __init__(self, inner: PipelineSession) -> None:
        self._inner = inner

    async def _to_thread(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run *fn* on a worker thread with the caller's ContextVar bindings."""
        ctx = contextvars.copy_context()
        if kwargs:
            return await asyncio.to_thread(ctx.run, functools.partial(fn, *args, **kwargs))
        return await asyncio.to_thread(ctx.run, fn, *args)

    async def ask(self, question: str) -> SessionStep:
        return cast(SessionStep, await self._to_thread(self._inner.ask, question))

    async def step(self, response: str | None = None) -> SessionStep:
        return cast(SessionStep, await self._to_thread(self._inner.step, response))

    async def reset(self) -> None:
        await self._to_thread(self._inner.reset)

    async def awaiting_prompt(self) -> bool:
        return cast(bool, await self._to_thread(self._inner.awaiting_prompt))

    async def ask_until_done(self, question: str, *, on_confirm: Literal["y", "n"] = "y") -> SessionStep:
        """Async wrapper around :meth:`PipelineSession.ask_until_done` including the same terminal-status semantics for final SQL rejection."""
        return cast(
            SessionStep,
            await self._to_thread(self._inner.ask_until_done, question, on_confirm=on_confirm),
        )

    async def accept_until_done(
        self,
        question: str,
        *,
        on_yes_no: Literal["y", "n"] = "y",
        on_free_text: str = "looks good",
    ) -> SessionStep:
        """Async wrapper around :meth:`PipelineSession.accept_until_done`."""
        return cast(
            SessionStep,
            await self._to_thread(
                self._inner.accept_until_done,
                question,
                on_yes_no=on_yes_no,
                on_free_text=on_free_text,
            ),
        )

    async def cancel(self) -> bool:
        """Cancel an in-flight turn on the underlying session."""
        return cast(bool, await self._to_thread(self._inner.cancel))

    async def __aenter__(self) -> AsyncPipelineSession:
        """Enter the underlying synchronous session context on a worker thread."""
        await self._to_thread(self._inner.__enter__)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        return cast(Literal[False], await self._to_thread(self._inner.__exit__, exc_type, exc_val, exc_tb))


def _render_interactive_suspend_step(step: SessionStep) -> None:
    """Emit suspend-phase notification and optional SQL preview via :func:`print_query_result`."""
    if step.prompt:
        notify(step.prompt, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    if step.sql is not None:
        hdr = list(step.data.columns) if step.data is not None else None
        rows = dataframe_to_row_tuples(step.data)
        print_query_result(rows, _sql_text_for_print(step.sql), headers=hdr)


def _render_interactive_terminal_step(step: SessionStep) -> None:
    """Emit terminal errors, answers, and optional final SQL preview."""
    if step.error is not None:
        error(f"{step.error.code.value}" + (f" ({step.error.detail_code})" if step.error.detail_code else ""))
        return
    if step.answer:
        notify(step.answer, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    if step.sql is not None:
        hdr = list(step.data.columns) if step.data is not None else None
        rows = dataframe_to_row_tuples(step.data)
        print_query_result(rows, _sql_text_for_print(step.sql), headers=hdr)


class AetherEngine:
    """Facade for environment-driven database setup, schema graph, and mode runners. Concurrency contract: reader-mode :class:`PipelineSession` turns may overlap; every mutating facade entry point takes ``_pipeline_writer_lock`` so store, artifact, and write-queue writes are serialized on one instance."""

    _connection_mapping: dict[str, str] | None

    __slots__ = (
        "_runtime_config",
        "_llm_config",
        "_schema_graph",
        "_dialect",
        "_artifacts_dir",
        "_store",
        "_templates",
        "_rejected",
        "_schema_terms",
        "_schema_stats",
        "_execution_engine",
        "_native_connection",
        "_named_connection",
        "_connection_mapping",
        "_connection",
        "_data_quality_report",
        "_audit_sink",
        "_phase_callback",
        "_diagnostic_sink",
        "_diagnostic_sink_token",
        "_pipeline_writer_lock",
        "_config_file",
        "_schema_role",
        "_consumer_visible_objects",
        "_credential_default_space_uid",
        "_context_name",
        "_sandbox_mode",
        "_sandbox_runtime",
        "_sandbox_extract_path",
        "_sandbox_closed",
        "_session_timezone",
        "_token_provider",
        "_domain_knowledge",
        "_schema_json_path",
        "_template_store_dir",
        "_limits",
        "_limits_explicit",
        "_closed",
        "_engine_identity",
        "_store_by_space",
        "_templates_by_space",
    )

    def __dir__(self) -> list[str]:
        """Return names intended for interactive discovery."""
        return sorted(
            (
                "close",
                "session",
                "asession",
                "run_interactive",
                "run_seed_warmup",
                "run_seed_warmup_from_history",
                "run_seed_warmup_from_query_log",
                "run_qsim",
                "get_qsim_summary",
                "get_questions_only",
                "get_seed_warmup_summary",
                "export_structure",
                "apply_structure",
                "export_knowledge",
                "apply_knowledge",
                "aetherspace",
                "delete_aetherspace",
                "export_context",
                "list_aetherspaces",
                "list_contexts",
                "clear_template_store",
                "clear_simulation_caches",
                "clear_all_learning",
                "list_templates",
                "fetch_template",
                "execute_template",
                "apply_migration_map",
                "refresh",
            ),
        )

    def __init__(
        self,
        engine_context: EngineContext | str | None = None,
        *,
        artifacts_dir: str | None = None,
        config_file: str | os.PathLike[str] | None = None,
        connection: str | Mapping[str, Any] | None = None,
        execution_engine: Any = None,
        native_connection: Any = None,
        source_selections: Mapping[str, Mapping[str, Any]] | None = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
        phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
        diagnostic_sink: Callable[[Diagnostic], None] | None = None,
        role: SchemaRole = SchemaRole.OWNER,
        token_provider: Callable[[], str | Mapping[str, str]] | None = None,
        limits: EngineLimits | None = None,
        _trust_bundled_baseline: bool = False,
        _storage_dir: str | None = None,
    ) -> None:
        """
        Initialise engine configuration from the environment, build the schema graph, and load templates.

        Args:

            engine_context: Master or named engine context. Pass a :class:`EngineContext` to define
            scope (owner persists; ``name='master'`` builds the graph, other names define subset
            specs over the existing master graph). Pass a ``str`` name to consume a stored context
            (consumers must use a name; owners may use ``None`` or ``'master'`` for the default).
            When omitted, a persisted master ``EngineContext`` is loaded from ``artifacts_dir``;
            if no cached context exists, a ``ConfigError`` is raised.
            artifacts_dir: Optional directory root; engine files are stored under ``<root>/aetherdialect/<connection_slug>``.
            config_file: Path to a TOML file. When set, every mapped field that appears in the file is authoritative for the corresponding process environment key (empty TOML values clear any inherited environment value for that key); fields omitted from the file are still read from ``os.environ``. When omitted, settings are read from ``os.environ`` only.
            connection: Named TOML connection sub-block when multiple databases of the same engine type share one config file, or a mapping of connection environment keys for this instance only (never written to ``os.environ``). String form selects credentials and the artifact slug; it is unrelated to any federation ``source_id``.
            execution_engine: Optional SQLAlchemy engine for query execution (caller-owned pool / read replica).
            native_connection: Optional native duckdb or sqlite3 connection for embedded engines.
            For DuckDB and SQLite, ``native_connection`` ensures reflection and execution share one in-memory or file-backed database.
            ``execution_engine`` is honored when it wraps the same ``StaticPool`` connection.
            source_selections: CSV file engine only: per-filename interpretation accepted after :func:`inspect_tabular_upload` (``header_row``, ``table_range``, ``append_regions``, etc.).
            audit_sink: Optional callback receiving :class:`AuditEvent` records at lifecycle boundaries.
            phase_callback: Optional callback receiving :class:`PhaseProgressEvent` during construction and ask turns.
            diagnostic_sink: Optional callback receiving :class:`Diagnostic` records from the diagnostic channel.
            role: Schema identity role; ``owner`` may mutate shared artifacts, ``consumer`` pins the owner snapshot id.
            token_provider: Optional callable returning a fresh secret string or credential field mapping consulted when opening the database connection (initial construction and :meth:`refresh`).

        Raises:

            ConfigError, DatabaseConnectionError, MigrationPendingError: Same as :func:`initialize_aether_engine`.
        """
        self._config_file = os.path.expanduser(str(config_file)) if config_file is not None else None
        self._execution_engine = execution_engine
        self._native_connection = native_connection
        if isinstance(connection, Mapping):
            self._named_connection = None
            self._connection_mapping = {str(k): str(v) for k, v in connection.items()}
        else:
            self._named_connection = (
                str(connection).strip() if connection is not None and str(connection).strip() else None
            )
            self._connection_mapping = None
        self._connection = None
        self._data_quality_report: DataQualityReport | None = None
        self._audit_sink = audit_sink
        self._phase_callback = phase_callback
        self._diagnostic_sink = diagnostic_sink
        self._diagnostic_sink_token = push_diagnostic_sink(diagnostic_sink)
        self._pipeline_writer_lock = threading.Lock()
        self._schema_role = role
        self._consumer_visible_objects: frozenset[str] | None = None
        self._credential_default_space_uid: str | None = None
        self._context_name = MASTER_AETHERSPACE_NAME
        self._sandbox_mode = False
        self._sandbox_runtime = None
        self._sandbox_extract_path = None
        self._sandbox_closed = False
        self._session_timezone = None
        self._token_provider = token_provider
        self._limits_explicit = limits is not None
        self._limits = limits if limits is not None else EngineLimits()
        self._closed = False
        bundle = self._initialize_engine_bundle(
            engine_context,
            artifacts_dir=artifacts_dir,
            config_file=config_file,
            connection=connection,
            log_sink=_init_log_sink,
            execution_engine=self._execution_engine,
            native_connection=self._native_connection,
            schema_role=role,
            source_selections=source_selections,
            trust_bundled_baseline=_trust_bundled_baseline,
            token_provider=token_provider,
            limits=self._limits,
            storage_dir=_storage_dir,
        )
        self._apply_init_bundle(bundle)
        self._domain_knowledge = DomainKnowledgeHolder()
        if not self._load_persisted_domain_knowledge():
            self._ingest_notes_domain_knowledge()
        self._audit_emit(
            "init",
            question=None,
            schema_hash=None,
            details=(("engine", self.dialect),),
        )
        if bundle.data_quality_report is not None:
            report = bundle.data_quality_report
            self._audit_emit(
                "data_quality",
                details=(
                    ("ok", "yes" if report.ok else "no"),
                    ("issue_count", str(len(report.issues))),
                    ("issues_json", stable_json(report.to_json_dict()["issues"])),
                ),
            )

    def _reinit_bundle_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`initialize_aether_engine` when reloading state."""
        return {
            "artifacts_dir": str(self._artifacts_dir),
            "log_sink": _init_log_sink,
            "execution_engine": self._execution_engine,
            "native_connection": getattr(self, "_native_connection", None),
            "config_file": self._config_file,
            "schema_role": self._schema_role,
            "limits": self._limits,
            "storage_dir": str(self._artifacts_dir),
        }

    def _initialize_engine_bundle(
        self,
        engine_context: EngineContext | str | None,
        **kwargs: Any,
    ) -> AetherEngineInitResult:
        """Run :func:`initialize_aether_engine` with construction-phase progress wired."""
        token = push_construction_phase_callback(self._phase_callback)
        try:
            return initialize_aether_engine(engine_context, **kwargs)
        finally:
            pop_construction_phase_callback(token)

    def _single_engine_context(self) -> EngineContext:
        ctx = self._runtime_config.engine_context
        if not isinstance(ctx, EngineContext):
            raise ConfigError("this operation requires a single-engine context")
        return ctx

    def _require_owner(self, operation: str) -> None:
        if self._schema_role != SchemaRole.OWNER:
            raise OwnerOnlyOperationError(operation)

    def _caller_visibility(self) -> tuple[EngineContext | FederationContext, frozenset[str] | None]:
        """Return scope context and caller-visible objects; consumers fail closed to an empty set when visibility is unset."""
        scope_ctx = MainExecutionOps.resolve_preview_scope_context(self)
        visible = getattr(self, "_consumer_visible_objects", None)
        if self._schema_role == SchemaRole.CONSUMER and visible is None:
            visible = frozenset()
        if visible is not None and not isinstance(visible, frozenset):
            visible = frozenset(visible)
        return scope_ctx, visible

    def _resolve_session_space(self, space: str | None) -> str:
        if space is None:
            return self.default_space_uid
        return str(space)

    def _resolve_aetherspace(
        self, token: str
    ) -> tuple[AetherSpace, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        raw = str(token).strip()
        if not raw:
            raise ConfigError("aetherspace identity must be non-empty")
        lower = raw.lower()
        if lower == MASTER_AETHERSPACE_NAME:
            desc = build_master_space_descriptor(self._schema_graph)
            return desc, frozenset(), frozenset(), frozenset(), frozenset()
        uid: str | None = None
        snap: dict[str, Any] | None = None
        try:
            candidate = MainExecutionOps.validate_space_uid(raw)
        except ValueError:
            candidate = None
        if candidate is not None and candidate != MASTER_AETHERSPACE_UID:
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), candidate)
            if snap is not None:
                uid = candidate
        if uid is None:
            uid = resolve_aetherspace_identity(str(self._artifacts_dir), lower)
            if uid == MASTER_AETHERSPACE_UID:
                desc = build_master_space_descriptor(self._schema_graph)
                return desc, frozenset(), frozenset(), frozenset(), frozenset()
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), uid)
        if snap is None:
            raise ConfigError(f"unknown aetherspace {token!r}")
        desc = aetherspace_descriptor_from_snapshot(uid, snap)
        tables, columns = space_allowed_sets_from_snapshot(snap)
        deny_objects, deny_columns = space_deny_sets_from_snapshot(snap)
        scope_ctx, visible = self._caller_visibility()
        if not aetherspace_within_effective_visibility(
            tables,
            columns,
            self._schema_graph,
            scope_ctx,
            visible,
        ):
            raise ConfigError(f"unknown aetherspace {token!r}")
        tables_raw = snap.get("tables")
        if isinstance(tables_raw, (list, tuple)) and len(tables_raw) == 0:
            raise ConfigError("space empty after schema migration; redefine")
        if getattr(self, "_sandbox_mode", False):
            Sandbox.require_sandbox_space_lock(desc.name, tables)
        return desc, tables, columns, deny_objects, deny_columns

    def _resolve_aetherspace_visible_by_name(self, name: str) -> AetherSpace:
        """Resolve display *name* among spaces visible to the caller (unique match required)."""
        try:
            norm = TemplateOps.validate_space_name(str(name).strip().lower())
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace name: {name!r}") from exc
        if norm == MASTER_AETHERSPACE_NAME:
            return build_master_space_descriptor(self._schema_graph)
        scope_ctx, visible = self._caller_visibility()
        matches: list[AetherSpace] = []
        for uid, label in list_saved_aetherspace_entries(str(self._artifacts_dir)):
            if label != norm:
                continue
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), uid)
            if snap is None:
                continue
            tables, columns = space_allowed_sets_from_snapshot(snap)
            if not aetherspace_within_effective_visibility(
                tables,
                columns,
                self._schema_graph,
                scope_ctx,
                visible,
            ):
                continue
            matches.append(aetherspace_descriptor_from_snapshot(uid, snap))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            uids = ", ".join(m.uid for m in matches)
            raise ConfigError(f"ambiguous aetherspace name {norm!r}; matches uids {uids}")
        raise ConfigError(f"unknown aetherspace {name!r}")

    def _require_production_api(self, operation: str) -> None:
        if getattr(self, "_sandbox_mode", False):
            raise ConfigError(
                f"{operation} is not available inside the offline sandbox. "
                "See docs/SANDBOX.md#exit-the-sandbox for production setup.",
            )

    def _audit_emit(
        self,
        event_type: str,
        *,
        question: str | None = None,
        schema_hash: str | None = None,
        details: tuple[tuple[str, str], ...] = (),
        turn_id: str | None = None,
    ) -> None:
        sink = getattr(self, "_audit_sink", None)
        if sink is None:
            return
        ev = AuditEvent(
            event_type=event_type,
            timestamp_iso=datetime.now(UTC).isoformat(),
            question=question,
            schema_hash=schema_hash,
            provider=self._llm_config.provider,
            details=details,
            turn_id=turn_id,
        )
        sink(ev)

    def _apply_init_bundle(self, bundle: AetherEngineInitResult) -> None:
        """Assign fields from a fresh :class:`AetherEngineInitResult` (also used after cache-clear reloads)."""
        self._runtime_config = bundle.runtime_config
        self._llm_config = bundle.llm_config
        self._schema_graph = bundle.schema_graph
        self._dialect = bundle.dialect
        self._artifacts_dir = Path(bundle.artifacts_dir)
        self._store = bundle.store
        self._templates = bundle.templates
        self._rejected = bundle.rejected
        self._schema_terms = bundle.schema_terms
        self._schema_stats = bundle.schema_stats
        self._schema_role = bundle.schema_role
        self._consumer_visible_objects = bundle.consumer_visible_objects
        self._context_name = getattr(bundle, "context_name", MASTER_AETHERSPACE_NAME)
        self._data_quality_report = bundle.data_quality_report
        self._schema_json_path = engine_schema_json_path(str(self._artifacts_dir))
        self._template_store_dir = engine_template_store_dir(str(self._artifacts_dir))
        self._engine_identity = getattr(bundle, "engine_identity", None)
        register_dialect_live_handles(self._dialect, owner=self)
        register_engine_skeleton_cache_owner(self)
        clear_engine_skeleton_cache(self)
        self._credential_default_space_uid = None
        if self._schema_role == SchemaRole.CONSUMER:
            dk = None
            holder = getattr(self, "_domain_knowledge", None)
            if holder is not None:
                try:
                    dk = holder.entries()
                except Exception:
                    dk = None
            self._credential_default_space_uid = MainExecutionOps.ensure_credential_default_aetherspace(
                str(self._artifacts_dir),
                self._schema_graph,
                self._consumer_visible_objects,
                engine_domain_knowledge=dk,
            )
        MainExecutionOps.bind_owner_default_template_store(
            self,
            self._schema_graph,
            str(self._artifacts_dir),
            schema_role=self._schema_role,
        )

    @property
    def data_quality_report(self) -> DataQualityReport | None:
        """Upload validation report from the most recent successful construction, when applicable."""
        return self._data_quality_report

    @_writer_lock_guard
    def ingest_upload_sources(
        self,
        paths: Sequence[str | os.PathLike[str]],
        *,
        source_selections: Mapping[str, CsvSourceSelection | Mapping[str, Any]] | None = None,
        relation_names: Mapping[str, str] | None = None,
        log_sink: Callable[[str], None] | None = None,
    ) -> UploadIngestResult:
        """Validate uploads and materialise accepted relations into this embedded member."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        self._require_owner("ingest_upload_sources")
        return CsvDialect.ingest_upload_sources_into_engine(
            self,
            paths,
            source_selections=source_selections,
            relation_names=relation_names,
            log_sink=log_sink,
        )

    def preview_migration_map(self) -> MigrationPreview:
        """Return a read-only preview of schema migration impact against stored artifacts."""
        return preview_schema_migration(artifacts_dir=self._artifacts_dir, schema_graph=self._schema_graph)

    def _replace_domain_knowledge(self, entries: Sequence[DomainKnowledgeEntry]) -> None:
        """Replace prompt-time domain knowledge (internal; public path is apply_knowledge)."""
        holder = getattr(self, "_domain_knowledge", None)
        if holder is None:
            self._domain_knowledge = DomainKnowledgeHolder()
            holder = self._domain_knowledge
        holder.set(entries, self._schema_graph)
        self._persist_domain_knowledge()

    def _refresh_aetherspace_snapshots_after_master_knowledge_change(self) -> None:
        """Re-derive prose on note-bearing space snapshots when master DK or descriptions change."""
        holder = getattr(self, "_domain_knowledge", None)
        dk = holder.entries() if isinstance(holder, DomainKnowledgeHolder) else ()
        try:
            MainExecutionOps.reenrich_aetherspace_snapshots_with_notes(
                str(self._artifacts_dir),
                self._schema_graph,
                engine_domain_knowledge=dk,
            )
        except OSError:
            pass

    def _persist_domain_knowledge(self) -> None:
        """Write master domain knowledge beside the schema cache for reopen / prompt reload."""
        holder = getattr(self, "_domain_knowledge", None)
        if not isinstance(holder, DomainKnowledgeHolder):
            return
        try:
            stamps = knowledge_artifact_save_stamps(self._schema_graph)
            save_domain_knowledge_artifact(self._artifacts_dir, holder.entries(), **stamps)
            refresh = getattr(self, "_refresh_aetherspace_snapshots_after_master_knowledge_change", None)
            if callable(refresh):
                refresh()
        except OSError:
            pass

    def _clear_notes_domain_knowledge(self) -> None:
        """Drop in-memory and on-disk notes-derived domain knowledge when notes are absent."""
        holder = getattr(self, "_domain_knowledge", None)
        if holder is None:
            self._domain_knowledge = DomainKnowledgeHolder()
            holder = self._domain_knowledge
        holder.set((), self._schema_graph)
        try:
            delete_domain_knowledge_artifact(self._artifacts_dir)
        except OSError:
            pass

    def _load_persisted_domain_knowledge(self) -> bool:
        """Load master domain knowledge from artifacts when present. Returns True when applied."""
        loaded = load_domain_knowledge_artifact(self._artifacts_dir, self._schema_graph)
        if loaded is None:
            return False
        holder = getattr(self, "_domain_knowledge", None)
        if holder is None:
            self._domain_knowledge = DomainKnowledgeHolder()
            holder = self._domain_knowledge
        holder.set(loaded, self._schema_graph)
        self._audit_emit(
            "domain_knowledge_ingest",
            details=(
                ("status", "loaded_artifact"),
                ("kept", str(len(loaded))),
                ("dropped", "0"),
                ("keys", ",".join(e.key for e in loaded)),
            ),
        )
        return True

    def _ingest_notes_domain_knowledge(self) -> None:
        """Extract domain knowledge from construction notes when present. Persists an empty artifact only when extract intentionally returned nothing with notes present (no filter drops)."""
        ctx = getattr(getattr(self, "_runtime_config", None), "engine_context", None)
        notes_content = notes_content_from_context(ctx) if ctx is not None else None
        if notes_content is None or not str(notes_content).strip():
            self._clear_notes_domain_knowledge()
            self._audit_emit(
                "domain_knowledge_ingest",
                details=(("status", "skipped_no_notes"), ("kept", "0")),
            )
            return
        try:
            entries = extract_domain_knowledge_from_notes(notes_content, self._schema_graph)
        except MockFixtureMissingError:
            if EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
                self._audit_emit(
                    "domain_knowledge_ingest",
                    details=(("status", "skipped_sandbox_fixture"), ("kept", "0")),
                )
                return
            raise
        safe = filter_schema_anchored_domain_knowledge(entries, self._schema_graph) if entries else ()
        dropped = len(entries) - len(safe)
        if safe:
            self._replace_domain_knowledge(safe)
        elif not safe and not dropped:
            self._persist_domain_knowledge()
        self._audit_emit(
            "domain_knowledge_ingest",
            details=(
                (
                    "status",
                    "ok" if safe else ("empty_after_filter" if dropped else "empty"),
                ),
                ("kept", str(len(safe))),
                ("dropped", str(dropped)),
                ("notes_chars", str(len(notes_content))),
                ("keys", ",".join(e.key for e in safe)),
            ),
        )

    def _domain_knowledge_entries(self) -> tuple[DomainKnowledgeEntry, ...]:
        """Return active domain knowledge entries (internal; public path is export_knowledge)."""
        holder = getattr(self, "_domain_knowledge", None)
        if not isinstance(holder, DomainKnowledgeHolder):
            return ()
        return holder.entries()

    def _resolve_space_knowledge_export_target(self, space: str | None) -> tuple[str | None, dict[str, Any] | None]:
        """Resolve export target for one space; ``None`` selects :attr:`default_space_uid`."""
        resolved = self._resolve_session_space(space)
        norm = str(resolved).strip().lower()
        if not norm:
            raise ConfigError("space identity must be non-empty")
        if norm in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            self._require_owner("export_knowledge")
            return MASTER_AETHERSPACE_UID, None
        desc = self._resolve_aetherspace(norm)[0]
        space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), desc.uid)
        if space_snapshot is None:
            raise ConfigError(f"unknown aetherspace {resolved!r}")
        return desc.uid, space_snapshot

    def export_knowledge(self, space: str | None = None) -> dict[str, Any]:
        """Return space domain knowledge and description overlays for the default or one named space."""
        space_token, space_snapshot = self._resolve_space_knowledge_export_target(space)
        scope_ctx, visible = self._caller_visibility()
        payload = MainExecutionOps.build_space_knowledge_export(
            engine_entries=self._domain_knowledge_entries(),
            space=space_token,
            space_snapshot=space_snapshot,
            schema_graph=self._schema_graph,
            scope_ctx=scope_ctx,
            visible_objects=visible,
        )
        self._audit_emit(
            "export_knowledge",
            schema_hash=self._effective_structural_hash or None,
            details=(("space", str(space_token or "default")),),
        )
        payload.pop("format_version", None)
        return payload

    def export_structure(self, space: str | None = None) -> dict[str, Any]:
        """Return structural inventory merged with editable overrides for the default or one named space."""
        resolved_space = self._resolve_session_space(space)
        space_snapshot = None
        space_token = resolved_space
        norm = str(resolved_space).strip().lower()
        if norm and norm not in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            desc = self._resolve_aetherspace(norm)[0]
            space_token = desc.uid
            space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), desc.uid)
            if space_snapshot is None:
                raise ConfigError(f"unknown aetherspace {resolved_space!r}")
        scope_ctx, visible = self._caller_visibility()
        inventory = MainExecutionOps.build_structure_export(
            schema_graph=self._schema_graph,
            space=space_token,
            space_snapshot=space_snapshot,
            scope_ctx=scope_ctx,
            visible_objects=visible,
        )
        overrides_raw = dump_structure_edits(self._schema_graph)
        payload = build_public_structure_document(
            inventory=inventory,
            overrides={
                "tables": overrides_raw.get("tables") or {},
                "foreign_keys_add": overrides_raw.get("foreign_keys_add") or [],
                "foreign_keys_remove": overrides_raw.get("foreign_keys_remove") or [],
                "primary_keys_add": overrides_raw.get("primary_keys_add") or [],
                "primary_keys_remove": overrides_raw.get("primary_keys_remove") or [],
            },
        )
        self._audit_emit(
            "export_structure",
            schema_hash=self._effective_structural_hash or None,
            details=(("space", str(space_token if space_token is not None else "default")),),
        )
        return payload

    @_writer_lock_guard
    def apply_knowledge(self, space: str, document: Mapping[str, Any]) -> None:
        """Replace space domain knowledge and description overlays from one exported document."""
        fields = MainExecutionOps.knowledge_document_apply_fields(document)
        self._apply_knowledge_impl(
            space,
            domain_knowledge=fields.get("domain_knowledge"),
            table_descriptions=fields.get("table_descriptions"),
            column_descriptions=fields.get("column_descriptions"),
        )

    @_writer_lock_guard
    def apply_structure(self, document: Mapping[str, Any]) -> None:
        """Apply a structural document declaratively; the document becomes the truth."""
        self._require_owner("apply_structure")
        schema_json_path = str(self._artifacts_dir / "schema_graph.json.gz")
        report = apply_structure_document(
            self._schema_graph,
            document,
            schema_json_path=schema_json_path,
            dialect=self._dialect,
            domain_knowledge=self._domain_knowledge_entries(),
        )
        if report.domain_knowledge_entries is not None:
            self._replace_domain_knowledge(report.domain_knowledge_entries)
        if (
            report.table_edits
            or report.column_edits
            or report.fks_added
            or report.fks_removed
            or report.pks_added
            or report.pks_endorsed
            or report.pks_blocked
            or report.coerced_columns
            or report.collapsed_inferences
        ):
            self._schema_stats = self._schema_graph.refresh_schema_stats()
        sh = getattr(self._schema_graph, "effective_structural_hash", None)
        self._audit_emit(
            "apply_structure",
            schema_hash=str(sh) if sh is not None else None,
            details=(
                ("table_edits", str(report.table_edits)),
                ("column_edits", str(report.column_edits)),
            ),
        )

    def _apply_knowledge_impl(
        self,
        space: str,
        *,
        domain_knowledge: Sequence[DomainKnowledgeEntry | Mapping[str, Any]] | None = None,
        table_descriptions: Mapping[str, str] | None = None,
        column_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Internal apply path for space knowledge overlays."""
        self._require_owner("apply_knowledge")
        norm = str(space).strip().lower()
        if not norm:
            raise ConfigError("space identity must be non-empty")
        if norm in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            if domain_knowledge is None and table_descriptions is None and column_descriptions is None:
                raise ConfigError(
                    "apply_knowledge requires domain_knowledge and/or table_descriptions and/or column_descriptions"
                )
            if domain_knowledge is not None:
                self._replace_domain_knowledge(MainExecutionOps.normalize_domain_knowledge_entries(domain_knowledge))
            if table_descriptions is not None or column_descriptions is not None:
                MainExecutionOps.apply_master_space_knowledge_to_graph(
                    self._schema_graph,
                    schema_json_path=str(self._artifacts_dir / "schema_graph.json.gz"),
                    table_descriptions=table_descriptions,
                    column_descriptions=column_descriptions,
                )
                self._refresh_aetherspace_snapshots_after_master_knowledge_change()
            return
        desc = self._resolve_aetherspace(norm)[0]
        snap = load_aetherspace_snapshot(str(self._artifacts_dir), desc.uid)
        if snap is None:
            raise ConfigError(f"unknown aetherspace {space!r}")
        updated = MainExecutionOps.apply_knowledge_to_snapshot(
            snap,
            domain_knowledge=domain_knowledge,
            table_descriptions=table_descriptions,
            column_descriptions=column_descriptions,
            schema_graph=self._schema_graph,
        )
        updated = filter_space_snapshot_sensitive_columns(updated, self._schema_graph)
        save_aetherspace_snapshot(str(self._artifacts_dir), desc.uid, updated)

    @property
    def _schema_graph_id(self) -> str:
        """Stable schema-graph identity for template store and write- queue matching."""
        return str(getattr(self._schema_graph, "schema_graph_id", "") or "")

    @property
    def dialect(self) -> str:
        """Registered engine name from ``list_engines()``; see ``docs/SUPPORT_MATRIX.md``."""
        return str(self._runtime_config.engine)

    @property
    def limits(self) -> EngineLimits:
        """Read-only behavioural limits for this engine instance."""
        return self._limits

    def _require_open(self, operation: str) -> None:
        if getattr(self, "_closed", False):
            raise RuntimeError(f"AetherEngine is closed; create a new instance for {operation!r}.")

    @_writer_lock_guard
    def close(self) -> None:
        """Release database handles, drain deferred writes, and mark this instance closed."""
        if getattr(self, "_closed", False):
            return
        drain_write_queue(self, str(self._artifacts_dir))
        release_close_resources(self)
        dispose_engine_dialect(
            self._dialect,
            borrowed_execution_engine=self._execution_engine,
            borrowed_native_connection=getattr(self, "_native_connection", None),
        )
        LLMProvider.clear_llm_clients()
        drop_engine_skeleton_cache_owner(self)
        clear_expansion_subtree_pool(str(self._artifacts_dir))
        sink_token = getattr(self, "_diagnostic_sink_token", None)
        if sink_token is not None:
            pop_diagnostic_sink(sink_token)
            self._diagnostic_sink_token = cast(Any, None)
        self._closed = True
        self._audit_emit("close", schema_hash=self._effective_structural_hash or None)

    def __enter__(self) -> AetherEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        self.close()
        return False

    @_writer_lock_guard
    def refresh(
        self,
        *,
        reflect: bool = True,
        credentials: str | Mapping[str, str] | None = None,
    ) -> RefreshReport:
        """Re-resolve credentials, reopen the connection, and reconcile artifacts against the live schema."""
        self._require_owner("refresh")
        self._require_open("refresh")
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        engine_type = self.dialect
        self._dialect = refresh_engine_connection(
            engine_type=engine_type,
            dialect=self._dialect,
            credentials=credentials,
            token_provider=getattr(self, "_token_provider", None),
            execution_engine=self._execution_engine,
            native_connection=getattr(self, "_native_connection", None),
            runtime=cast(Any, self._runtime_config),
        )
        report = MainExecutionOps.refresh_aether_engine(self, reflect=reflect)
        self._audit_emit(
            "refresh",
            schema_hash=self._effective_structural_hash or None,
            details=(
                ("migration_tier", report.migration_tier.value),
                ("schema_changed", "yes" if report.schema_changed else "no"),
                ("orphans_removed", str(report.orphans_removed)),
                ("bytes_reclaimed", str(report.bytes_reclaimed)),
            ),
        )
        return report

    @property
    def _write_queue_path(self) -> Path:
        """Internal path to ``write_queue.jsonl`` under the engine storage directory."""
        return self._artifacts_dir / WRITE_QUEUE_FILENAME

    @property
    def _effective_structural_hash(self) -> str:
        """Effective structural fingerprint of the live schema graph (used for manifest and write-queue matching)."""
        return str(getattr(self._schema_graph, "effective_structural_hash", "") or "")

    def _ensure_llm(self) -> None:
        """Raise when no LLM credentials are available on ``EngineConfig``."""
        if not EngineConfig.llm_credentials_configured():
            raise ConfigError(
                "LLM is not configured. Set OpenAI or Azure OpenAI variables documented in API_REFERENCE.md.",
            )

    def _compute_num_intents_range(self) -> tuple[int, int]:
        """Return schema-adaptive ``(min_intents, max_intents)``."""
        table_count = int(self._schema_stats.get("table_count", 1))
        min_intents = max(5, table_count)
        max_intents = min(200, table_count * 10)
        return (min_intents, max_intents)

    def _compute_num_questions_range(self) -> tuple[int, int]:
        """Return schema-adaptive ``(min_questions, max_questions)``."""
        min_intents, _ = self._compute_num_intents_range()
        total_filterable = int(self._schema_stats.get("total_filterable", 1))
        min_questions = max(min_intents, 10)
        max_questions = min(2000, total_filterable * 20)
        return (min_questions, max_questions)

    def session(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str | None = None,
        ephemeral_scope: SpaceContext | None = None,
        data_row_cap: int | None = None,
    ) -> PipelineSession:
        """Return a programmatic session sharing this instance's schema graph and template store. ``writer`` mode may mutate artifacts and takes ``_pipeline_writer_lock`` only around store and write-queue mutations; ``reader`` mode is read-only and never takes that lock."""
        self._require_open("session")
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        Sandbox.require_sandbox_adoption(self)
        if self._schema_role == SchemaRole.CONSUMER and mode == "writer":
            raise OwnerOnlyOperationError("PipelineSession(mode='writer')")
        desc, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(
            self._resolve_session_space(space)
        )
        if ephemeral_scope is not None and (
            ephemeral_scope.tables
            or ephemeral_scope.columns
            or ephemeral_scope.deny_objects
            or ephemeral_scope.deny_columns
        ):
            ephemeral_scope = validate_space_context_against_graph(ephemeral_scope, self._schema_graph)
            scope_ctx, visible = self._caller_visibility()
            validate_aetherspace_define_within_visibility(
                ephemeral_scope.tables,
                ephemeral_scope.columns,
                self._schema_graph,
                scope_ctx,
                visible,
            )
        space_tables, space_columns, space_deny_objects, space_deny_columns = intersect_space_scope(
            space_tables,
            space_columns,
            space_deny_objects,
            space_deny_columns,
            ephemeral_scope,
        )
        space_description_overlay: dict[str, Any] | None = None
        space_uid = desc.uid
        if space_uid != MASTER_AETHERSPACE_UID:
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), space_uid)
            if isinstance(snap, dict):
                table_descriptions = snap.get("table_descriptions")
                column_meta = snap.get("column_meta")
                if isinstance(table_descriptions, dict) or isinstance(column_meta, dict):
                    space_description_overlay = {
                        "table_descriptions": dict(table_descriptions or {}),
                        "column_meta": dict(column_meta or {}),
                    }
        payload_visible = space_tables if space_tables else None
        MainExecutionOps.bind_template_store_for_space(self, space_uid)
        return PipelineSession(
            self,
            mode=mode,
            visible_objects=payload_visible,
            execution_visible_objects=self._consumer_visible_objects,
            space_name=space_uid,
            space_tables=space_tables,
            space_columns=space_columns,
            space_deny_objects=space_deny_objects,
            space_deny_columns=space_deny_columns,
            space_description_overlay=space_description_overlay,
            data_row_cap=data_row_cap,
        )

    def asession(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str | None = None,
        ephemeral_scope: SpaceContext | None = None,
        data_row_cap: int | None = None,
    ) -> AsyncPipelineSession:
        """Async wrapper around :meth:`session` (uses threads; underlying API remains synchronous)."""
        return AsyncPipelineSession(
            self.session(mode=mode, space=space, ephemeral_scope=ephemeral_scope, data_row_cap=data_row_cap)
        )

    @_writer_lock_guard
    def aetherspace(
        self,
        name: str | None = None,
        space_context: SpaceContext | None = None,
        *,
        uid: str | None = None,
        notes_file: str | None = None,
        notes: str | None = None,
    ) -> AetherSpace:
        """Create, update, or read an aetherspace (uid is durable identity; name is a label)."""
        if notes_file is not None and notes is not None:
            raise ConfigError("set at most one of notes and notes_file")
        try:
            uid_norm = MainExecutionOps.validate_space_uid(str(uid).strip()) if uid is not None else None
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace uid: {uid!r}") from exc
        name_norm = str(name).strip().lower() if name is not None else None
        if space_context is None:
            if uid_norm and name_norm:
                raise ConfigError("read aetherspace with uid or name, not both")
            if uid_norm:
                return self._resolve_aetherspace(uid_norm)[0]
            if name_norm:
                return self._resolve_aetherspace_visible_by_name(name_norm)
            raise ConfigError("aetherspace read requires uid or name")
        self._require_owner("aetherspace")
        if uid_norm:
            if uid_norm == MASTER_AETHERSPACE_UID:
                raise ConfigError(
                    "master is the implicit full-scope space; it cannot be created or overwritten",
                )
            desc = self._resolve_aetherspace(uid_norm)[0]
            display = name_norm if name_norm else desc.name
        else:
            if not name_norm:
                raise ConfigError("aetherspace create requires name")
            if name_norm == MASTER_AETHERSPACE_NAME:
                raise ConfigError(
                    "master is the implicit full-scope space; it cannot be created or overwritten",
                )
            try:
                display = TemplateOps.validate_space_name(name_norm)
            except ValueError as exc:
                raise ConfigError(f"invalid aetherspace name: {name!r}") from exc
            uid_norm = allocate_aetherspace_uid(str(self._artifacts_dir))
        scope_ctx, visible = self._caller_visibility()
        validate_aetherspace_define_within_visibility(
            space_context.tables,
            space_context.columns,
            self._schema_graph,
            scope_ctx,
            visible,
        )
        validated = validate_space_context_against_graph(space_context, self._schema_graph)
        if getattr(self, "_sandbox_mode", False):
            Sandbox.require_sandbox_space_lock(display, validated.tables)
        snapshot = subset_graph_for_space(self._schema_graph, validated)
        snapshot["uid"] = uid_norm
        snapshot["name"] = display
        if notes_file is not None:
            notes_path, notes_inline = notes_file, None
        elif notes is not None:
            notes_path, notes_inline = None, notes
        else:
            notes_path, notes_inline = validated.notes_file, validated.notes
        if notes_path is not None and str(notes_path).strip() and getattr(self, "_sandbox_mode", False):
            connection = getattr(self, "_native_connection", None)
            host = Sandbox.sandbox_host_for_connection(connection) if connection is not None else None
            if host is not None:
                notes_path = Sandbox.validate_sandbox_aetherspace_notes_pairing(
                    display,
                    notes_path,
                    extract_path=host._extract_path,
                )
        if notes_inline is not None and str(notes_inline).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes=str(notes_inline),
                engine_domain_knowledge=self._domain_knowledge_entries(),
            )
        elif notes_path is not None and str(notes_path).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes_path,
                engine_domain_knowledge=self._domain_knowledge_entries(),
            )
        else:
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                engine_domain_knowledge=self._domain_knowledge_entries(),
            )
        snapshot["uid"] = uid_norm
        snapshot["name"] = display
        snapshot = filter_space_snapshot_sensitive_columns(snapshot, self._schema_graph)
        save_aetherspace_snapshot(str(self._artifacts_dir), uid_norm, snapshot)
        return aetherspace_descriptor_from_snapshot(uid_norm, snapshot)

    @_writer_lock_guard
    def delete_aetherspace(
        self,
        name: str | None = None,
        *,
        uid: str | None = None,
        persist_learning: bool = True,
    ) -> AetherspaceDeleteResult:
        """Delete one persisted aetherspace snapshot and its learning partition."""
        self._require_owner("delete_aetherspace")
        token = uid if uid is not None else name
        if token is None:
            raise ConfigError("delete_aetherspace requires uid or name")
        desc = self._resolve_aetherspace(str(token))[0]
        return delete_aetherspace(
            str(self._artifacts_dir),
            desc.uid,
            persist_learning=persist_learning,
            schema_graph=self._schema_graph,
        )

    def list_aetherspaces(self, *, include_system: bool = False) -> tuple[AetherSpace, ...]:
        """Return aetherspace descriptors visible to the caller. Owners include the implicit ``master`` space. Consumers omit ``master``. System credential-default spaces are omitted unless *include_system* is True."""
        scope_ctx, visible = self._caller_visibility()
        out: list[AetherSpace] = []
        if self._schema_role != SchemaRole.CONSUMER:
            out.append(build_master_space_descriptor(self._schema_graph))
        for space_uid, _label in list_saved_aetherspace_entries(str(self._artifacts_dir)):
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), space_uid)
            if snap is None:
                continue
            if not include_system and MainExecutionOps.is_credential_default_snapshot(snap):
                continue
            tables, columns = space_allowed_sets_from_snapshot(snap)
            if aetherspace_within_effective_visibility(
                tables,
                columns,
                self._schema_graph,
                scope_ctx,
                visible,
            ):
                out.append(aetherspace_descriptor_from_snapshot(space_uid, snap))
        return tuple(out)

    @property
    def default_space_uid(self) -> str:
        """The default aetherspace for this engine: the master space for an owner, the visibility-keyed default for a consumer."""
        if self._schema_role == SchemaRole.CONSUMER:
            uid = getattr(self, "_credential_default_space_uid", None)
            if uid:
                return str(uid)
        return MASTER_AETHERSPACE_UID

    def export_context(self, name: str) -> dict[str, Any]:
        """Return a read-only export document for one named engine context. Owner-only."""
        self._require_owner("export_context")
        norm = str(name).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME and load_named_schema_context(str(self._artifacts_dir), norm) is None:
            raise ConfigError(f"unknown engine context {name!r}")
        master_ctx = self._runtime_config.engine_context
        if not isinstance(master_ctx, EngineContext):
            raise ConfigError("export_context requires a single-engine context")
        return build_named_schema_context_export(
            str(self._artifacts_dir),
            norm,
            master_ctx,
            schema_graph=self._schema_graph,
            schema_role=self._schema_role,
        )

    def list_contexts(self) -> tuple[str, ...]:
        """Return saved engine-context names plus the implicit ``master`` context. Owner-only."""
        self._require_owner("list_contexts")
        saved = list_named_schema_context_names(str(self._artifacts_dir))
        return (MASTER_AETHERSPACE_NAME,) + saved

    def _templates_for_space(self, space: str | None) -> dict[str, Template] | LazyTemplateMapping:
        """Load the in-memory template map for one aetherspace namespace."""
        norm = str(self._resolve_session_space(space)).strip().lower()
        self._resolve_aetherspace(norm)
        context_name = getattr(self, "_context_name", MASTER_AETHERSPACE_NAME)
        if norm == MASTER_AETHERSPACE_NAME and context_name == MASTER_AETHERSPACE_NAME:
            return dict(self._templates)
        store = load_template_store(
            self._schema_graph_id,
            self._schema_graph,
            space_name=norm,
            artifacts_dir=str(self._artifacts_dir),
        )
        return store_to_templates(store)

    def list_templates(self, *, space: str | None = None) -> tuple[StoredTemplateSummary, ...]:
        """Enumerate caller-visible stored templates for one aetherspace namespace."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        scope_ctx, visible = self._caller_visibility()
        visible_tables = effective_visible_tables(self._schema_graph, scope_ctx, visible)
        templates = self._templates_for_space(space)
        return list_stored_template_summaries(
            templates,
            space=space or MASTER_AETHERSPACE_NAME,
            dialect=self._dialect,
            visible_tables=visible_tables,
        )

    def fetch_template(self, template_ref: str, *, space: str | None = None) -> StoredTemplateDetail:
        """Fetch one stored template by id or ``sql_fp`` hash."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        scope_ctx, visible = self._caller_visibility()
        visible_tables = effective_visible_tables(self._schema_graph, scope_ctx, visible)
        templates = self._templates_for_space(space)
        tmpl = resolve_template_ref(template_ref, templates)
        if tmpl is None or not TemplateOps.template_enumerable_by_caller(tmpl, visible_tables=visible_tables):
            raise ConfigError(f"unknown template ref {template_ref!r}")
        vh = tmpl.value_history
        hist_idx = 0
        if vh.questions:
            primary = primary_template_q_norm(tmpl)
            hist_idx = vh.questions.index(primary) if primary in vh.questions else 0
        return build_stored_template_detail(
            tmpl,
            space=space or MASTER_AETHERSPACE_NAME,
            schema=self._schema_graph,
            dialect=self._dialect,
            history_index=hist_idx,
            schema_context=scope_ctx,
            visible_objects=visible,
        )

    @_writer_lock_guard
    def execute_template(
        self,
        template_ref: str,
        params: dict[str, Any] | None = None,
        *,
        question: str | None = None,
        space: str | None = None,
        as_dataframe: bool = False,
    ) -> TemplateExecutionResult | pandas.DataFrame:
        """Execute one stored template by id or ``sql_fp`` with caller- supplied bind values."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dict or None")
        bind = dict(params or ())
        templates = self._templates_for_space(space)
        scope_ctx, visible = self._caller_visibility()
        visible_tables = effective_visible_tables(self._schema_graph, scope_ctx, visible)
        identity = getattr(self, "_engine_identity", None)
        if not isinstance(identity, EngineIdentity):
            runtime_cfg = getattr(self, "_runtime_config", None)
            dialect_obj = self._dialect
            engine_type = str(getattr(dialect_obj, "name", getattr(self, "dialect", "")) or "")
            identity = EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
        identity_token = push_engine_identity(identity)
        with owner_limits_scope(self):
            try:
                result = execute_stored_template_by_ref(
                    template_ref,
                    bind,
                    question=question,
                    dialect=self._dialect,
                    store=self._store,
                    templates=cast(dict[str, Any], templates),
                    rejected=self._rejected,
                    schema=self._schema_graph,
                    schema_context=scope_ctx,
                    visible_objects=visible,
                    visible_tables=visible_tables,
                    schema_role=self._schema_role,
                    persist_template_learning=False,
                )
            finally:
                pop_engine_identity(identity_token)
        if as_dataframe:
            return pandas.DataFrame([list(r) for r in result.rows], columns=list(result.columns) or None)
        return result

    @_writer_lock_guard
    def apply_migration_map(self, document: Mapping[str, Any]) -> None:
        """Validate and persist a schema migration map, then reconcile artifacts."""
        self._require_owner("apply_migration_map")
        self._require_open("apply_migration_map")
        if not isinstance(document, Mapping):
            raise ConfigError("migration map must be a JSON object")
        payload = dict(document)
        if "version" not in payload:
            payload["version"] = 1
        map_obj = TemplateOps.parse_schema_migration_map_payload(payload)
        map_path = self._artifacts_dir / MIGRATION_MAP_FILENAME
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        TemplateOps.validate_schema_migration_map(
            map_obj,
            load_schema_graph_snapshot(str(self._artifacts_dir / "schema_graph.json.gz")),
            self._schema_graph,
        )
        self.refresh()

    def get_seed_warmup_summary(self) -> SeedWarmupSummarySnapshot:
        """Return the newest seed-warmup summary text if present."""
        self._require_production_api("get_seed_warmup_summary")
        s = find_latest_seed_warmup_summary(str(self._artifacts_dir))
        if s is None:
            return SeedWarmupSummarySnapshot(text="Seed warmup summary: none found.")
        return SeedWarmupSummarySnapshot(text=format_seed_warmup_summary(s))

    def get_qsim_summary(self, start: int, end: int) -> QSimSummarySnapshot:
        """Return QSim summary lines for versions ``start`` through ``end`` inclusive."""
        self._require_production_api("get_qsim_summary")
        summaries = load_qsim_summaries(str(self._artifacts_dir))
        if not summaries:
            raise ConfigError("QSim summary not found; run run_qsim first")
        picked = [s for s in summaries if start <= int(s.version) <= end]
        lines: list[str] = [f"QSim range ({len(picked)} runs):"]
        for s in picked:
            lines.append(format_qsim_summary_line(s))
        if summaries:
            latest = max(summaries, key=lambda x: int(x.version))
            lines.append(
                f"Latest: v{latest.version}  intents={latest.num_intents}  "
                f"questions={latest.num_questions}  seed={latest.seed}",
            )
        return QSimSummarySnapshot(lines=tuple(lines))

    def _validate_num_intents(self, value: int) -> None:
        """Raise ``ValueError`` when *value* is outside the adaptive ``num_intents`` range."""
        min_intents, max_intents = self._compute_num_intents_range()
        if not (min_intents <= value <= max_intents):
            raise ValueError(
                f"num_intents must be {min_intents}-{max_intents} for this schema "
                f"({self._schema_stats['table_count']} tables)",
            )

    def _validate_num_questions(self, value: int) -> None:
        """Raise ``ValueError`` when *value* is outside the adaptive ``num_questions`` range."""
        min_questions, max_questions = self._compute_num_questions_range()
        if not (min_questions <= value <= max_questions):
            raise ValueError(
                f"num_questions must be {min_questions}-{max_questions} for this schema "
                f"({self._schema_stats['total_filterable']} total filterable columns)",
            )

    @_writer_lock_guard
    def run_interactive(self, *, space: str | None = None) -> None:
        """Prompt once for a natural-language question, resolve it through the interactive prompt cycle, then return. An empty line at the question prompt warns once; a second empty line terminates with ``User terminated.``. There is no outer REPL loop; call ``run_interactive`` again for another question."""
        self._ensure_llm()
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(
            self._resolve_session_space(space)
        )
        payload_visible = space_tables if space_tables else None
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            notify(
                "\nInteractive mode",
                stage="interactive",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
            )
            empty_streak = 0
            question = ""
            while True:
                print("Enter question (empty line to quit): ", end="", flush=True)
                try:
                    raw = input()
                except (EOFError, KeyboardInterrupt):
                    terminated()
                    return
                echo_user_text(raw)
                if raw.strip() == "":
                    empty_streak += 1
                    if empty_streak >= 2:
                        terminated()
                        return
                    notify(
                        "Press Enter again to quit.",
                        stage="interactive",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    continue
                question = raw.strip()
                break

            with PipelineSession(
                self,
                visible_objects=payload_visible,
                execution_visible_objects=self._consumer_visible_objects,
                space_name=str(space).strip().lower(),
                space_tables=space_tables,
                space_columns=space_columns,
                space_deny_objects=space_deny_objects,
                space_deny_columns=space_deny_columns,
            ) as session:
                try:
                    with progress_enabled():
                        step = session.ask(question)
                        while not step.done:
                            _render_interactive_suspend_step(step)
                            print(step.prompt or "", end="", flush=True)
                            try:
                                ans = input()
                            except (EOFError, KeyboardInterrupt):
                                terminated()
                                return
                            if step.kind in YES_NO_SESSION_KINDS:
                                echo_yes_no_answer(ans)
                            else:
                                echo_user_text(ans)
                            step = session.step(ans)
                        _render_interactive_terminal_step(step)
                except (EOFError, KeyboardInterrupt):
                    terminated()
                    return
                except Exception as exc:
                    error(f"{exc.__class__.__name__}: {exc}")
                    return

    @_writer_lock_guard
    def run_seed_warmup(
        self,
        seed_filepath: str,
        interactive_gold: bool = True,
        *,
        abort_on_gold_failure: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> None:
        """Run seed warmup execution, stratified sampling, and template writes."""
        self._require_owner("run_seed_warmup")
        self._require_production_api("run_seed_warmup")
        self._ensure_llm()
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            owner_tok = push_qsim_engine_owner(self)
            scope_tok = push_simulation_artifact_scope_from_owner(self)
            try:
                seed_warmup_run_once(
                    schema=self._schema_graph,
                    dialect=self._dialect,
                    seed_filepath=seed_filepath,
                    output_dir=str(self._artifacts_dir),
                    store=self._store,
                    templates=self._templates,
                    interactive_gold=interactive_gold,
                    abort_on_gold_failure=abort_on_gold_failure,
                    max_kept_intents=max_kept_intents,
                )
            finally:
                pop_simulation_artifact_partition(scope_tok)
                pop_qsim_engine_owner(owner_tok)

    @_writer_lock_guard
    def run_seed_warmup_from_history(
        self,
        sql_history_filepath: str,
        *,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> None:
        """
        Reverse-engineer SQL history into intents and run seed warmup over them.

        Args:

            sql_history_filepath: Newline-delimited historical ``SELECT`` statements.
            expand: When ``True``, apply the same deterministic expansion used by seed-question warmup.
            max_kept_intents: Cap on kept intents after sampling; ``None`` keeps every intent that
            passes quality and dedup checks (no budget cap).
        """
        self._require_owner("run_seed_warmup_from_history")
        self._require_production_api("run_seed_warmup_from_history")
        self._ensure_llm()
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            run_seed_warmup_from_history_execution(
                self,
                sql_history_filepath,
                expand=expand,
                max_kept_intents=max_kept_intents,
            )

    @_writer_lock_guard
    def run_seed_warmup_from_query_log(
        self,
        *,
        lookback_days: int = 730,
        max_queries: int = 5000,
        min_runs: int = 1,
        user_filter: str | None = None,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> None:
        """
        Fetch SQL history from engine system catalogs when available and run seed warmup.

        Args:

            lookback_days: How far back to read warehouse query history.
            max_queries: Maximum number of historical statements to ingest.
            min_runs: Minimum execution count required for a logged statement.
            user_filter: Optional warehouse user or role filter when supported.
            expand: When ``True``, apply the same deterministic expansion used by seed-question warmup.
            max_kept_intents: Cap on kept intents after sampling; ``None`` keeps every intent that
            passes quality and dedup checks (no budget cap).
        """
        self._require_production_api("run_seed_warmup_from_query_log")
        self._ensure_llm()
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            run_seed_warmup_from_query_log_execution(
                self,
                lookback_days=lookback_days,
                max_queries=max_queries,
                min_runs=min_runs,
                user_filter=user_filter,
                expand=expand,
                max_kept_intents=max_kept_intents,
            )

    @_writer_lock_guard
    def run_qsim(
        self,
        num_intents: int = 20,
        num_questions: int = 100,
        seed: int | None = None,
    ) -> None:
        """Generate synthetic NL questions from schema-derived intent skeletons."""
        self._require_owner("run_qsim")
        self._require_production_api("run_qsim")
        self._ensure_llm()
        self._validate_num_intents(num_intents)
        self._validate_num_questions(num_questions)
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            owner_tok = push_qsim_engine_owner(self)
            scope_tok = push_simulation_artifact_scope_from_owner(self)
            try:
                qsim_run_once(
                    num_intents=num_intents,
                    num_questions=num_questions,
                    seed=seed,
                    artifacts_dir=str(self._artifacts_dir),
                    schema=self._schema_graph,
                )
            finally:
                pop_simulation_artifact_partition(scope_tok)
                pop_qsim_engine_owner(owner_tok)

    def get_questions_only(self, version: int) -> None:
        """Print NL questions from a QSim artifact and write them to ``qsim_v{version}_questions.txt`` in the process working directory."""
        path = resolve_qsim_path(version, str(self._artifacts_dir))
        if not os.path.isfile(path):
            raise ConfigError(f"QSim questions file not found for version {version}: {path}")
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            print_questions_bundle(version, str(self._artifacts_dir))

    def _resolve_learning_clear_space(self, space: str | None) -> str | None:
        """Return None for all spaces, else a template-store partition key (``master`` or space uid)."""
        if space is None:
            return None
        token = str(space).strip().lower()
        if not token or token == "all":
            return None
        if token in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            return MASTER_AETHERSPACE_NAME
        desc = self._resolve_aetherspace(token)[0]
        return str(desc.uid).strip().lower()

    @_writer_lock_guard
    def clear_template_store(self, *, space: str | None = None) -> bool:
        """Owner-only: remove template learning then reload. ``space=None``/``"all"`` clears every partition; otherwise one space (uid or master)."""
        self._require_owner("clear_template_store")
        space_key = self._resolve_learning_clear_space(space)
        existed = clear_template_store_only(str(self._artifacts_dir), self._schema_graph, space=space_key)
        bundle = self._initialize_engine_bundle(
            self._single_engine_context(),
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_template_store",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("existed", str(existed)), ("space", space_key or "all")),
        )
        return existed

    @_writer_lock_guard
    def clear_simulation_caches(self) -> int:
        """Remove QSim and seed-warmup artifact files, then reload engine initialization state."""
        self._require_owner("clear_simulation_caches")
        count = clear_simulation_caches_only(str(self._artifacts_dir))
        bundle = self._initialize_engine_bundle(
            self._single_engine_context(),
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_simulation_caches",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("removed_files", str(count)),),
        )
        return count

    @_writer_lock_guard
    def clear_all_learning(self, *, keep_structure: bool = True, space: str | None = None) -> None:
        """Owner-only: clear learning then reload. With ``space`` set, only that template partition is cleared (sim caches / structural overrides stay). Without ``space`` (or ``space="all"``), clears all templates, simulation caches, and optionally structural overrides. Domain knowledge is not cleared here — export, edit, and ``apply_knowledge``."""
        self._require_owner("clear_all_learning")
        space_key = self._resolve_learning_clear_space(space)
        clear_template_store_only(str(self._artifacts_dir), self._schema_graph, space=space_key)
        if space_key is None:
            clear_simulation_caches_only(str(self._artifacts_dir))
            if not keep_structure:
                delete_persisted_structure_artifacts(str(self._artifacts_dir / "schema_graph.json.gz"))
        bundle = self._initialize_engine_bundle(
            self._single_engine_context(),
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_all_learning",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None) or None,
            details=(("keep_structure", str(keep_structure)), ("space", space_key or "all")),
        )


class AetherFederation:
    """Federated scope over named member engines with a composed schema graph."""

    _is_aether_federation = True

    __slots__ = (
        "_name",
        "_members",
        "_declaration_path",
        "_declaration_parsed",
        "_master_context",
        "_mappings",
        "_artifacts_root",
        "_runtime_config",
        "_llm_config",
        "_schema_graph",
        "_dialect",
        "_artifacts_dir",
        "_store",
        "_templates",
        "_rejected",
        "_schema_terms",
        "_schema_stats",
        "_audit_sink",
        "_phase_callback",
        "_diagnostic_sink",
        "_diagnostic_sink_token",
        "_pipeline_writer_lock",
        "_schema_role",
        "_consumer_visible_objects",
        "_credential_default_space_uid",
        "_context_name",
        "_sandbox_mode",
        "_sandbox_runtime",
        "_sandbox_extract_path",
        "_sandbox_closed",
        "_closed",
        "_federation_manifest",
        "_federation_mappings",
        "_federation_member_graphs",
        "_federation_storage_dir",
        "_federation_dialects",
        "_federation_source_runtimes",
        "_federation_mapping_suggestions",
        "_engine_identity",
        "_domain_knowledge",
        "_limits",
        "_store_by_space",
        "_templates_by_space",
    )

    def __init__(
        self,
        name: str,
        *,
        members: Sequence[AetherEngine],
        declaration: str | os.PathLike[str] | Mapping[str, Any],
        context: FederationContext | None = None,
        artifacts_dir: str | None = None,
        role: SchemaRole = SchemaRole.OWNER,
        limits: FederationLimits | None = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
        phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
        diagnostic_sink: Callable[[Diagnostic], None] | None = None,
    ) -> None:
        """Construct a federation over member engines. Each member's ``source_id`` is its connection name (TOML sub- block or mapping ``name`` key)."""
        self._name = str(name).strip()
        if not self._name:
            raise ConfigError("AetherFederation name must be non-empty")
        try:
            require_driver("duckdb")
        except ConfigError as exc:
            raise ConfigError("pip install aetherdialect[federation]") from exc
        self._members = federation_members_mapping(cast(Sequence[FederationMemberEngine], members))
        self._declaration_parsed: tuple[FederationManifest, FederationMappings] | None
        if isinstance(declaration, Mapping):
            self._declaration_parsed = parse_federation_declaration(declaration)
            self._declaration_path = None
        else:
            self._declaration_path = str(declaration)
            self._declaration_parsed = None
        self._master_context = context
        self._mappings: FederationMappings | None = None
        self._artifacts_root = Path(artifacts_dir) if artifacts_dir is not None else None
        self._audit_sink = audit_sink
        self._phase_callback = phase_callback
        self._diagnostic_sink = diagnostic_sink
        self._diagnostic_sink_token = push_diagnostic_sink(diagnostic_sink)
        self._pipeline_writer_lock = threading.Lock()
        self._schema_role = role
        self._consumer_visible_objects: frozenset[str] | None = None
        self._credential_default_space_uid: str | None = None
        self._context_name = MASTER_AETHERSPACE_NAME
        self._sandbox_mode = False
        self._sandbox_runtime = None
        self._sandbox_extract_path = None
        self._sandbox_closed = False
        self._closed = False
        self._limits = limits if limits is not None else FederationLimits()
        apply_federation_member_defaults(self._members, self._limits)
        validate_federation_pool_capacity(self._members, self._limits)
        construction_token = push_construction_phase_callback(phase_callback)
        try:
            bundle = initialize_aether_federation(
                self._name,
                members=self._members,
                declaration=self._declaration_parsed,
                declaration_file=self._declaration_path,
                artifacts_dir=str(self._artifacts_root) if self._artifacts_root is not None else None,
                schema_role=role,
                master_context=self._master_context,
                log_sink=_init_log_sink,
            )
        finally:
            pop_construction_phase_callback(construction_token)
        self._apply_init_bundle(bundle)
        if isinstance(bundle.members, Mapping):
            self._members = federation_members_mapping(bundle.members)
        if bundle.federation_mappings is not None:
            self._mappings = bundle.federation_mappings
        self._domain_knowledge = DomainKnowledgeHolder()
        if not self._load_persisted_domain_knowledge():
            self._ingest_notes_domain_knowledge()
        if self._schema_role == SchemaRole.CONSUMER:
            holder = getattr(self, "_domain_knowledge", None)
            engine_dk = holder.entries() if isinstance(holder, DomainKnowledgeHolder) else ()
            self._credential_default_space_uid = MainExecutionOps.ensure_credential_default_aetherspace(
                str(self._artifacts_dir),
                self._schema_graph,
                self._consumer_visible_objects,
                engine_domain_knowledge=engine_dk,
            )
            MainExecutionOps.bind_owner_default_template_store(
                self,
                self._schema_graph,
                str(self._artifacts_dir),
                schema_role=self._schema_role,
            )
        self._replay_composite_overrides()
        self._audit_emit(
            "init",
            question=None,
            schema_hash=None,
            details=(("federation", self._name), ("members", str(len(self._members)))),
        )

    @classmethod
    def inspect_persisted(
        cls,
        federation_id: str,
        *,
        artifacts_dir: str,
        schema_role: SchemaRole = SchemaRole.OWNER,
    ) -> PersistedFederationInspection:
        """Load declaration and roster from a persisted ``fed_<id>`` tree without member engines."""
        return inspect_persisted_federation(artifacts_dir, federation_id, schema_role=schema_role)

    def _authored_federation_declaration(self) -> tuple[FederationManifest, FederationMappings]:
        parsed = self._declaration_parsed
        if parsed is not None:
            return parsed
        path = self._declaration_path
        if path:
            return load_federation_declaration_from_path(str(path))
        raise ConfigError("federation declaration is not configured")

    def _federation_declaration_export_path(self) -> str:
        path = getattr(self, "_declaration_path", None)
        if path:
            return str(path)
        return str(self._artifacts_dir / FEDERATION_DECLARATION_FILENAME)

    def _recompose(self) -> None:
        manifest_decl, file_mappings = self._authored_federation_declaration()
        active_source_ids = set(self._members)
        base_mappings = self._mappings if self._mappings is not None else file_mappings
        manifest_decl, reconciled_mappings = reconcile_authored_declaration_for_members(
            manifest_decl,
            base_mappings,
            active_source_ids=active_source_ids,
        )
        if self._mappings is not None and reconciled_mappings == self._mappings:
            mappings = self._mappings
        else:
            mappings = reconciled_mappings
        bundle = initialize_aether_federation(
            self._name,
            members=self._members,
            declaration=(manifest_decl, mappings),
            declaration_file=self._declaration_path,
            artifacts_dir=str(self._artifacts_root) if self._artifacts_root is not None else None,
            schema_role=self._schema_role,
            master_context=self._master_context,
            log_sink=_init_log_sink,
        )
        self._apply_init_bundle(bundle)
        if isinstance(bundle.members, Mapping):
            self._members = federation_members_mapping(bundle.members)
        if bundle.federation_mappings is not None:
            self._mappings = bundle.federation_mappings
        self._replay_composite_overrides()
        if self._federation_manifest is not None and self._mappings is not None:
            export_federation_declaration(
                self._federation_manifest,
                self._mappings,
                self._federation_declaration_export_path(),
            )

    def _composite_federation_dir(self) -> str:
        fed_dir = getattr(self, "_federation_storage_dir", None)
        if not fed_dir:
            raise ConfigError("federation storage directory is unset")
        return str(fed_dir)

    def _replay_composite_overrides(self) -> None:
        fed_dir = getattr(self, "_federation_storage_dir", None)
        if not fed_dir or self._schema_graph is None:
            return
        if finalize_federation_composite_overrides(self._schema_graph, str(fed_dir), dialect=self._dialect):
            self._schema_stats = self._schema_graph.refresh_schema_stats()

    def _require_no_active_session_turn(self, operation: str) -> None:
        if self._pipeline_writer_lock.locked():
            raise RuntimeError(f"cannot {operation} while a session turn is in progress")

    @_writer_lock_guard(before_acquire=lambda self, op: self._require_no_active_session_turn(op))
    def add_engine(self, engine: AetherEngine) -> None:
        """Register a member engine, recompose the composite graph, and persist the federation tree."""
        self._require_owner("add_engine")
        self._require_open("add_engine")
        source_id = member_connection_name_from_engine(cast(FederationMemberEngine, engine))
        self._members[source_id] = engine
        self._recompose()

    @_writer_lock_guard(before_acquire=lambda self, op: self._require_no_active_session_turn(op))
    def remove_engine(self, connection_name: str) -> None:
        """Remove a member engine, prune dependent plan templates, recompose, and persist."""
        self._require_owner("remove_engine")
        self._require_open("remove_engine")
        key = str(connection_name).strip()
        if key not in self._members:
            raise ConfigError(f"unknown federation member: {connection_name!r}")
        member_engine = self._members[key]
        del self._members[key]
        fed_dir = getattr(self, "_federation_storage_dir", None)
        if fed_dir:
            prune_federation_plan_templates_for_sources(str(fed_dir), {key})
            binding = binding_from_member_engine(cast(FederationMemberEngine, member_engine))
            fed_manifest = self._federation_manifest
            federation_id = str(fed_manifest.federation_id) if fed_manifest is not None else None
            raw_member_dir = getattr(member_engine, "_artifacts_dir", None)
            member_dir = federation_member_artifacts_dir_for_purge(
                str(self._artifacts_dir),
                binding,
                federation_id=federation_id,
                member_artifacts_dir=raw_member_dir if isinstance(raw_member_dir, str) else None,
            )
            purge_federation_member_artifacts(
                str(fed_dir),
                member_artifacts_dir=member_dir,
                artifacts_root=str(self._artifacts_dir),
                source_id=key,
                member_engine=member_engine,
                manifest=fed_manifest,
                federation_id=federation_id,
            )
        self._recompose()

    def export_federation(self) -> dict[str, Any]:
        """Return the federation declaration document (topology and mappings)."""
        self._require_owner("export_federation")
        manifest = self._federation_manifest
        if manifest is None:
            raise ConfigError("federation manifest not loaded")
        mappings = self._mappings or self._federation_mappings
        if mappings is None:
            mappings = FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        document = federation_declaration_document(manifest, mappings)
        document.pop("version", None)
        self._audit_emit("export_federation", schema_hash=self._effective_structural_hash or None)
        return document

    @_writer_lock_guard
    def apply_federation(self, document: Mapping[str, Any]) -> None:
        """Apply a federation declaration document and recompose."""
        self._require_owner("apply_federation")
        payload = dict(document)
        if "version" not in payload:
            payload["version"] = FEDERATION_DECLARATION_VERSION
        try:
            manifest, mappings = parse_federation_declaration(payload)
        except FederationConfigError as exc:
            raise FederationConfigError(f"malformed federation declaration: {exc}") from exc
        self._declaration_parsed = (manifest, mappings)
        self._mappings = mappings
        self._recompose()

    @_writer_lock_guard
    def apply_migration_map(self, document: Mapping[str, Any]) -> None:
        """Validate and persist a federation migration map, then recompose."""
        self._require_owner("apply_migration_map")
        self._require_open("apply_migration_map")
        if not isinstance(document, Mapping):
            raise ConfigError("federation migration map must be a JSON object")
        payload = dict(document)
        if "version" not in payload:
            payload["version"] = 1
        parse_federation_migration_map(payload)
        dst = self._artifacts_dir / FEDERATION_MIGRATION_MAP_FILENAME
        dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._recompose()

    def _require_owner(self, operation: str) -> None:
        if self._schema_role != SchemaRole.OWNER:
            raise OwnerOnlyOperationError(operation)

    def _require_open(self, operation: str) -> None:
        if getattr(self, "_closed", False):
            raise RuntimeError(f"AetherFederation is closed; create a new instance for {operation!r}.")

    def _require_production_api(self, operation: str) -> None:
        if getattr(self, "_sandbox_mode", False):
            raise ConfigError(
                f"{operation} is not available inside the offline sandbox. "
                "See docs/SANDBOX.md#exit-the-sandbox for production setup.",
            )

    def _resolve_member(self, connection_name: str, operation: str) -> AetherEngine:
        self._require_open(operation)
        key = str(connection_name).strip()
        if not key:
            raise ConfigError("connection_name must be non-empty")
        member = self._members.get(key)
        if member is None:
            raise ConfigError(f"unknown federation member: {connection_name!r}")
        return cast(AetherEngine, member)

    def _federation_unsupported(self, operation: str) -> NoReturn:
        raise ConfigError(
            f"{operation} is not available on AetherFederation; "
            "use the member engine directly or the federation session API.",
        )

    def _audit_emit(
        self,
        event_type: str,
        *,
        question: str | None = None,
        schema_hash: str | None = None,
        details: tuple[tuple[str, str], ...] = (),
        turn_id: str | None = None,
    ) -> None:
        sink = getattr(self, "_audit_sink", None)
        if sink is None:
            return
        ev = AuditEvent(
            event_type=event_type,
            timestamp_iso=datetime.now(UTC).isoformat(),
            question=question,
            schema_hash=schema_hash,
            provider=self._llm_config.provider,
            details=details,
            turn_id=turn_id,
        )
        sink(ev)

    @property
    def _effective_structural_hash(self) -> str:
        """Effective structural fingerprint of the composed federation schema graph."""
        return str(getattr(self._schema_graph, "effective_structural_hash", "") or "")

    @property
    def limits(self) -> FederationLimits:
        """Read-only federation coordination limits."""
        return self._limits

    def _apply_init_bundle(self, bundle: AetherFederationInitResult) -> None:
        self._runtime_config = bundle.runtime_config
        self._llm_config = bundle.llm_config
        self._schema_graph = bundle.schema_graph
        self._dialect = bundle.dialect
        self._artifacts_dir = Path(bundle.artifacts_dir)
        self._store = bundle.store
        self._templates = bundle.templates
        self._rejected = bundle.rejected
        self._schema_terms = bundle.schema_terms
        self._schema_stats = bundle.schema_stats
        self._schema_role = bundle.schema_role
        self._consumer_visible_objects = bundle.consumer_visible_objects
        self._credential_default_space_uid = None
        self._context_name = getattr(bundle, "context_name", MASTER_AETHERSPACE_NAME)
        self._federation_manifest = getattr(bundle, "federation_manifest", None)
        self._federation_mappings = getattr(bundle, "federation_mappings", None)
        self._federation_member_graphs = getattr(bundle, "federation_member_graphs", None)
        self._federation_storage_dir = getattr(bundle, "federation_storage_dir", None)
        self._federation_dialects = getattr(bundle, "federation_dialects_by_source", None)
        self._federation_source_runtimes = getattr(bundle, "federation_source_runtimes", None)
        self._federation_mapping_suggestions = getattr(bundle, "federation_mapping_suggestions", ())
        self._engine_identity = getattr(bundle, "engine_identity", None)
        self._store_by_space = {MASTER_AETHERSPACE_NAME: bundle.store}
        self._templates_by_space = {MASTER_AETHERSPACE_NAME: bundle.templates}
        register_engine_skeleton_cache_owner(self)
        clear_engine_skeleton_cache(self)

    @property
    def dialect(self) -> str:
        return str(self._runtime_config.engine)

    def _replace_domain_knowledge(self, entries: Sequence[DomainKnowledgeEntry]) -> None:
        """Replace prompt-time domain knowledge (internal; public path is apply_knowledge)."""
        holder = getattr(self, "_domain_knowledge", None)
        if holder is None:
            self._domain_knowledge = DomainKnowledgeHolder()
            holder = self._domain_knowledge
        holder.set(entries, self._schema_graph)
        self._persist_domain_knowledge()

    def _persist_domain_knowledge(self) -> None:
        """Write federation-level domain knowledge beside federation artifacts."""
        holder = getattr(self, "_domain_knowledge", None)
        if not isinstance(holder, DomainKnowledgeHolder):
            return
        try:
            stamps = knowledge_artifact_save_stamps(self._schema_graph)
            save_domain_knowledge_artifact(self._artifacts_dir, holder.entries(), **stamps)
        except OSError:
            pass

    def _load_persisted_domain_knowledge(self) -> bool:
        """Load federation domain knowledge from artifacts when present. Returns True when applied."""
        loaded = load_domain_knowledge_artifact(self._artifacts_dir, self._schema_graph)
        if loaded is None:
            return False
        holder = getattr(self, "_domain_knowledge", None)
        if holder is None:
            self._domain_knowledge = DomainKnowledgeHolder()
            holder = self._domain_knowledge
        holder.set(loaded, self._schema_graph)
        self._audit_emit(
            "domain_knowledge_ingest",
            details=(
                ("status", "loaded_artifact"),
                ("kept", str(len(loaded))),
                ("dropped", "0"),
                ("keys", ",".join(e.key for e in loaded)),
            ),
        )
        return True

    def _ingest_notes_domain_knowledge(self) -> None:
        """Merge member + federation-notes knowledge and enrich composite descriptions."""
        ctx = getattr(self, "_master_context", None)
        notes_content = notes_content_from_context(ctx) if ctx is not None else None
        member_dk: list[tuple[str, tuple[DomainKnowledgeEntry, ...]]] = []
        member_structural: list[tuple[str, tuple[Any, ...]]] = []
        member_table_universe: set[str] = set()
        for source_id, eng in (getattr(self, "_members", None) or {}).items():
            entries: tuple[DomainKnowledgeEntry, ...] = ()
            try:
                entries = tuple(eng._domain_knowledge_entries())
            except (AttributeError, TypeError, ConfigError):
                entries = ()
            member_dk.append((str(source_id), entries))
            sg = getattr(eng, "_schema_graph", None)
            facts = tuple(getattr(sg, "structural_knowledge", ()) or ()) if sg is not None else ()
            member_structural.append((str(source_id), facts))
            if sg is not None:
                member_table_universe.update(str(name) for name in sg.tables.keys())
        for member_graph in (getattr(self, "_federation_member_graphs", None) or {}).values():
            if member_graph is not None:
                member_table_universe.update(str(name) for name in member_graph.tables.keys())
        try:
            final_dk = MainExecutionOps.enrich_federation_composite_knowledge(
                self._schema_graph,
                member_domain_knowledge=member_dk,
                member_structural_knowledge=member_structural,
                notes_content=notes_content,
                all_schema_table_names=member_table_universe,
            )
        except MockFixtureMissingError:
            if EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
                self._audit_emit(
                    "domain_knowledge_ingest",
                    details=(("status", "skipped_sandbox_fixture"), ("kept", "0")),
                )
                return
            raise
        if final_dk:
            self._replace_domain_knowledge(final_dk)
        self._audit_emit(
            "domain_knowledge_ingest",
            details=(
                (
                    "status",
                    "ok" if final_dk else ("skipped_no_notes" if not (notes_content or "").strip() else "empty"),
                ),
                ("kept", str(len(final_dk))),
                ("members", str(len(member_dk))),
                ("notes_chars", str(len(notes_content or ""))),
                ("keys", ",".join(e.key for e in final_dk)),
            ),
        )

    def _domain_knowledge_entries(self) -> tuple[DomainKnowledgeEntry, ...]:
        """Return active domain knowledge entries (internal; public path is export_knowledge)."""
        holder = getattr(self, "_domain_knowledge", None)
        if not isinstance(holder, DomainKnowledgeHolder):
            return ()
        return holder.entries()

    def _resolve_space_knowledge_export_target(self, space: str | None) -> tuple[str | None, dict[str, Any] | None]:
        """Resolve export target for one space; ``None`` selects :attr:`default_space_uid`."""
        resolved = self._resolve_session_space(space)
        norm = str(resolved).strip().lower()
        if not norm:
            raise ConfigError("space identity must be non-empty")
        if norm in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            self._require_owner("export_knowledge")
            return MASTER_AETHERSPACE_UID, None
        desc = self._resolve_aetherspace(norm)[0]
        space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), desc.uid)
        if space_snapshot is None:
            raise ConfigError(f"unknown aetherspace {resolved!r}")
        return desc.uid, space_snapshot

    def export_knowledge(self, space: str | None = None) -> dict[str, Any]:
        """Return space domain knowledge and description overlays for the default or one named space."""
        space_token, space_snapshot = self._resolve_space_knowledge_export_target(space)
        scope_ctx, visible = self._caller_visibility()
        payload = MainExecutionOps.build_space_knowledge_export(
            engine_entries=self._domain_knowledge_entries(),
            space=space_token,
            space_snapshot=space_snapshot,
            schema_graph=self._schema_graph,
            scope_ctx=scope_ctx,
            visible_objects=visible,
        )
        self._audit_emit(
            "export_knowledge",
            schema_hash=self._effective_structural_hash or None,
            details=(("space", str(space_token or "default")),),
        )
        payload.pop("format_version", None)
        return payload

    def export_structure(self, space: str | None = None) -> dict[str, Any]:
        """Return structural inventory merged with editable overrides for the default or one named space."""
        resolved_space = self._resolve_session_space(space)
        space_snapshot = None
        space_token = resolved_space
        norm = str(resolved_space).strip().lower()
        if norm and norm not in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            desc = self._resolve_aetherspace(norm)[0]
            space_token = desc.uid
            space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), desc.uid)
            if space_snapshot is None:
                raise ConfigError(f"unknown aetherspace {resolved_space!r}")
        members = getattr(self, "_member_table_roster", None)
        if self._schema_role == SchemaRole.CONSUMER:
            members = None
        scope_ctx, visible = self._caller_visibility()
        inventory = MainExecutionOps.build_structure_export(
            schema_graph=self._schema_graph,
            space=space_token,
            space_snapshot=space_snapshot,
            federation_members=members,
            scope_ctx=scope_ctx,
            visible_objects=visible,
        )
        overrides_raw = dump_structure_edits(self._schema_graph)
        payload = build_public_structure_document(
            inventory=inventory,
            overrides={
                "tables": overrides_raw.get("tables") or {},
                "foreign_keys_add": overrides_raw.get("foreign_keys_add") or [],
                "foreign_keys_remove": overrides_raw.get("foreign_keys_remove") or [],
                "primary_keys_add": overrides_raw.get("primary_keys_add") or [],
                "primary_keys_remove": overrides_raw.get("primary_keys_remove") or [],
            },
        )
        self._audit_emit(
            "export_structure",
            schema_hash=self._effective_structural_hash or None,
            details=(("space", str(space_token if space_token is not None else "default")),),
        )
        return payload

    @_writer_lock_guard
    def apply_knowledge(self, space: str, document: Mapping[str, Any]) -> None:
        """Replace space domain knowledge and description overlays from one exported document."""
        fields = MainExecutionOps.knowledge_document_apply_fields(document)
        self._apply_knowledge_impl(
            space,
            domain_knowledge=fields.get("domain_knowledge"),
            table_descriptions=fields.get("table_descriptions"),
            column_descriptions=fields.get("column_descriptions"),
        )

    @_writer_lock_guard
    def apply_structure(self, document: Mapping[str, Any]) -> None:
        """Apply a structural document to the composite federation graph declaratively."""
        self._require_owner("apply_structure")
        fed_dir = self._composite_federation_dir()
        schema_json_path = str(Path(fed_dir) / "schema_graph.json.gz")
        report = apply_structure_document(
            self._schema_graph,
            document,
            schema_json_path=schema_json_path,
            dialect=self._dialect,
            domain_knowledge=self._domain_knowledge_entries(),
        )
        if report.domain_knowledge_entries is not None:
            self._replace_domain_knowledge(report.domain_knowledge_entries)
        if (
            report.table_edits
            or report.column_edits
            or report.fks_added
            or report.fks_removed
            or report.pks_added
            or report.pks_endorsed
            or report.pks_blocked
            or report.coerced_columns
            or report.collapsed_inferences
        ):
            self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._recompose()
        sh = getattr(self._schema_graph, "effective_structural_hash", None)
        self._audit_emit(
            "apply_structure",
            schema_hash=str(sh) if sh is not None else None,
            details=(
                ("scope", "composite"),
                ("table_edits", str(report.table_edits)),
                ("column_edits", str(report.column_edits)),
            ),
        )

    def _apply_knowledge_impl(
        self,
        space: str,
        *,
        domain_knowledge: Sequence[DomainKnowledgeEntry | Mapping[str, Any]] | None = None,
        table_descriptions: Mapping[str, str] | None = None,
        column_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Internal apply path for space knowledge overlays."""
        self._require_owner("apply_knowledge")
        norm = str(space).strip().lower()
        if not norm:
            raise ConfigError("space identity must be non-empty")
        if norm in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            if domain_knowledge is None and table_descriptions is None and column_descriptions is None:
                raise ConfigError(
                    "apply_knowledge requires domain_knowledge and/or table_descriptions and/or column_descriptions"
                )
            if domain_knowledge is not None:
                self._replace_domain_knowledge(MainExecutionOps.normalize_domain_knowledge_entries(domain_knowledge))
            if table_descriptions is not None or column_descriptions is not None:
                MainExecutionOps.apply_master_space_knowledge_to_graph(
                    self._schema_graph,
                    schema_json_path=str(self._artifacts_dir / "schema_graph.json.gz"),
                    table_descriptions=table_descriptions,
                    column_descriptions=column_descriptions,
                )
            return
        desc = self._resolve_aetherspace(norm)[0]
        snap = load_aetherspace_snapshot(str(self._artifacts_dir), desc.uid)
        if snap is None:
            raise ConfigError(f"unknown aetherspace {space!r}")
        updated = MainExecutionOps.apply_knowledge_to_snapshot(
            snap,
            domain_knowledge=domain_knowledge,
            table_descriptions=table_descriptions,
            column_descriptions=column_descriptions,
            schema_graph=self._schema_graph,
        )
        updated = filter_space_snapshot_sensitive_columns(updated, self._schema_graph)
        save_aetherspace_snapshot(str(self._artifacts_dir), desc.uid, updated)

    def _caller_visibility(self) -> tuple[EngineContext | FederationContext, frozenset[str] | None]:
        """Return scope context and caller-visible objects; consumers fail closed to an empty set when visibility is unset."""
        scope_ctx = MainExecutionOps.resolve_preview_scope_context(self)
        visible = getattr(self, "_consumer_visible_objects", None)
        if self._schema_role == SchemaRole.CONSUMER and visible is None:
            visible = frozenset()
        if visible is not None and not isinstance(visible, frozenset):
            visible = frozenset(visible)
        return scope_ctx, visible

    def _resolve_session_space(self, space: str | None) -> str:
        if space is None:
            return self.default_space_uid
        return str(space)

    def _resolve_aetherspace(
        self, token: str
    ) -> tuple[AetherSpace, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        raw = str(token).strip()
        if not raw:
            raise ConfigError("aetherspace identity must be non-empty")
        lower = raw.lower()
        if lower == MASTER_AETHERSPACE_NAME:
            desc = build_master_space_descriptor(self._schema_graph)
            return desc, frozenset(), frozenset(), frozenset(), frozenset()
        uid: str | None = None
        snap: dict[str, Any] | None = None
        try:
            candidate = MainExecutionOps.validate_space_uid(raw)
        except ValueError:
            candidate = None
        if candidate is not None and candidate != MASTER_AETHERSPACE_UID:
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), candidate)
            if snap is not None:
                uid = candidate
        if uid is None:
            uid = resolve_aetherspace_identity(str(self._artifacts_dir), lower)
            if uid == MASTER_AETHERSPACE_UID:
                desc = build_master_space_descriptor(self._schema_graph)
                return desc, frozenset(), frozenset(), frozenset(), frozenset()
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), uid)
        if snap is None:
            raise ConfigError(f"unknown aetherspace {token!r}")
        desc = aetherspace_descriptor_from_snapshot(uid, snap)
        tables, columns = space_allowed_sets_from_snapshot(snap)
        deny_objects, deny_columns = space_deny_sets_from_snapshot(snap)
        mappings = self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        collapsed = collapsed_member_physical_table_names(mappings)
        for table_name in tables | deny_objects:
            logical = collapsed.get(table_name)
            if logical is not None:
                raise ConfigError(
                    f"aetherspace {token!r} names collapsed member table {table_name!r}; "
                    f"use logical table {logical!r} instead",
                )
        validate_federation_context_against_mappings(
            FederationContext(
                allow_objects=tables,
                deny_objects=deny_objects,
                allow_columns=columns,
                deny_columns=deny_columns,
            ),
            mappings,
        )
        scope_ctx, visible = self._caller_visibility()
        if not aetherspace_within_effective_visibility(
            tables,
            columns,
            self._schema_graph,
            scope_ctx,
            visible,
            mappings=mappings,
            federation_manifest=self._federation_manifest,
        ):
            raise ConfigError(f"unknown aetherspace {token!r}")
        tables_raw = snap.get("tables")
        if isinstance(tables_raw, (list, tuple)) and len(tables_raw) == 0:
            raise ConfigError("space empty after schema migration; redefine")
        if getattr(self, "_sandbox_mode", False):
            Sandbox.require_sandbox_space_lock(desc.name, tables)
        return desc, tables, columns, deny_objects, deny_columns

    def _resolve_aetherspace_visible_by_name(self, name: str) -> AetherSpace:
        """Resolve display *name* among spaces visible to the caller (unique match required)."""
        try:
            norm = TemplateOps.validate_space_name(str(name).strip().lower())
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace name: {name!r}") from exc
        if norm == MASTER_AETHERSPACE_NAME:
            return build_master_space_descriptor(self._schema_graph)
        scope_ctx, visible = self._caller_visibility()
        mappings = self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        matches: list[AetherSpace] = []
        for space_uid, label in list_saved_aetherspace_entries(str(self._artifacts_dir)):
            if label != norm:
                continue
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), space_uid)
            if snap is None:
                continue
            tables, columns = space_allowed_sets_from_snapshot(snap)
            if not aetherspace_within_effective_visibility(
                tables,
                columns,
                self._schema_graph,
                scope_ctx,
                visible,
                mappings=mappings,
                federation_manifest=self._federation_manifest,
            ):
                continue
            matches.append(aetherspace_descriptor_from_snapshot(space_uid, snap))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            uids = ", ".join(m.uid for m in matches)
            raise ConfigError(f"ambiguous aetherspace name {norm!r}; matches uids {uids}")
        raise ConfigError(f"unknown aetherspace {name!r}")

    def _ensure_llm(self) -> None:
        if not EngineConfig.llm_credentials_configured():
            raise ConfigError(
                "LLM is not configured. Set OpenAI or Azure OpenAI variables documented in API_REFERENCE.md.",
            )

    def _compute_num_intents_range(self) -> tuple[int, int]:
        table_count = int(self._schema_stats.get("table_count", 1))
        min_intents = max(5, table_count)
        max_intents = min(200, table_count * 10)
        return (min_intents, max_intents)

    def _compute_num_questions_range(self) -> tuple[int, int]:
        min_intents, _ = self._compute_num_intents_range()
        total_filterable = int(self._schema_stats.get("total_filterable", 1))
        min_questions = max(min_intents, 10)
        max_questions = min(2000, total_filterable * 20)
        return (min_questions, max_questions)

    def _validate_num_intents(self, value: int) -> None:
        min_intents, max_intents = self._compute_num_intents_range()
        if not (min_intents <= value <= max_intents):
            raise ValueError(
                f"num_intents must be {min_intents}-{max_intents} for this schema "
                f"({self._schema_stats['table_count']} tables)",
            )

    def _validate_num_questions(self, value: int) -> None:
        min_questions, max_questions = self._compute_num_questions_range()
        if not (min_questions <= value <= max_questions):
            raise ValueError(
                f"num_questions must be {min_questions}-{max_questions} for this schema "
                f"({self._schema_stats['total_filterable']} total filterable columns)",
            )

    def session(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str | None = None,
        ephemeral_scope: SpaceContext | None = None,
        data_row_cap: int | None = None,
    ) -> PipelineSession:
        self._require_open("session")
        if self._sandbox_closed:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        if self._schema_role == SchemaRole.CONSUMER and mode == "writer":
            raise OwnerOnlyOperationError("PipelineSession(mode='writer')")
        desc, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(
            self._resolve_session_space(space)
        )
        if ephemeral_scope is not None and (
            ephemeral_scope.tables
            or ephemeral_scope.columns
            or ephemeral_scope.deny_objects
            or ephemeral_scope.deny_columns
        ):
            ephemeral_scope = validate_space_context_against_graph(
                ephemeral_scope,
                self._schema_graph,
                federation_manifest=self._federation_manifest,
            )
            scope_ctx, visible = self._caller_visibility()
            validate_aetherspace_define_within_visibility(
                ephemeral_scope.tables,
                ephemeral_scope.columns,
                self._schema_graph,
                scope_ctx,
                visible,
                mappings=self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION),
                federation_manifest=self._federation_manifest,
            )
        space_tables, space_columns, space_deny_objects, space_deny_columns = intersect_space_scope(
            space_tables,
            space_columns,
            space_deny_objects,
            space_deny_columns,
            ephemeral_scope,
        )
        space_description_overlay: dict[str, Any] | None = None
        space_uid = desc.uid
        if space_uid != MASTER_AETHERSPACE_UID:
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), space_uid)
            if isinstance(snap, dict):
                table_descriptions = snap.get("table_descriptions")
                column_meta = snap.get("column_meta")
                if isinstance(table_descriptions, dict) or isinstance(column_meta, dict):
                    space_description_overlay = {
                        "table_descriptions": dict(table_descriptions or {}),
                        "column_meta": dict(column_meta or {}),
                    }
        payload_visible = space_tables if space_tables else None
        MainExecutionOps.bind_template_store_for_space(self, space_uid)
        return PipelineSession(
            self,
            mode=mode,
            visible_objects=payload_visible,
            execution_visible_objects=self._consumer_visible_objects,
            space_name=space_uid,
            space_tables=space_tables,
            space_columns=space_columns,
            space_deny_objects=space_deny_objects,
            space_deny_columns=space_deny_columns,
            space_description_overlay=space_description_overlay,
            data_row_cap=data_row_cap,
        )

    def list_templates(self, *, space: str | None = None) -> tuple[StoredTemplateSummary, ...]:
        """Enumerate caller-visible federation plan templates."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        del space
        scope_ctx, visible = self._caller_visibility()
        visible_tables = effective_visible_tables(self._schema_graph, scope_ctx, visible)
        return list_stored_template_summaries(
            self._templates,
            space="master",
            dialect=self._dialect,
            visible_tables=visible_tables,
        )

    def fetch_template(self, template_ref: str, *, space: str | None = None) -> StoredTemplateDetail:
        """Fetch one federation plan template by id or ``sql_fp`` hash."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        del space
        scope_ctx, visible = self._caller_visibility()
        visible_tables = effective_visible_tables(self._schema_graph, scope_ctx, visible)
        tmpl = resolve_template_ref(template_ref, self._templates)
        if tmpl is None or not TemplateOps.template_enumerable_by_caller(tmpl, visible_tables=visible_tables):
            raise ConfigError(f"unknown template ref {template_ref!r}")
        vh = tmpl.value_history
        hist_idx = 0
        if vh.questions:
            primary = primary_template_q_norm(tmpl)
            hist_idx = vh.questions.index(primary) if primary in vh.questions else 0
        return build_stored_template_detail(
            tmpl,
            space="master",
            schema=self._schema_graph,
            dialect=self._dialect,
            history_index=hist_idx,
            schema_context=scope_ctx,
            visible_objects=visible,
        )

    @_writer_lock_guard
    def execute_template(
        self,
        template_ref: str,
        params: dict[str, Any] | None = None,
        *,
        question: str | None = None,
        space: str | None = None,
        as_dataframe: bool = False,
    ) -> TemplateExecutionResult | pandas.DataFrame:
        """Execute one federation-stored template by id or ``sql_fp`` with p-param binds."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new Sandbox instance.")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dict or None")
        del space
        bind = dict(params or ())
        scope_ctx, visible = self._caller_visibility()
        visible_tables = effective_visible_tables(self._schema_graph, scope_ctx, visible)
        identity = getattr(self, "_engine_identity", None)
        if not isinstance(identity, EngineIdentity):
            runtime_cfg = getattr(self, "_runtime_config", None)
            dialect_obj = self._dialect
            engine_type = str(getattr(dialect_obj, "name", "duckdb") or "duckdb")
            identity = EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
        identity_token = push_engine_identity(identity)
        with owner_limits_scope(self):
            try:
                result = execute_stored_template_by_ref(
                    template_ref,
                    bind,
                    question=question,
                    dialect=self._dialect,
                    store=self._store,
                    templates=self._templates,
                    rejected=self._rejected,
                    schema=self._schema_graph,
                    schema_context=scope_ctx,
                    visible_objects=visible,
                    visible_tables=visible_tables,
                    schema_role=self._schema_role,
                    persist_template_learning=False,
                )
            finally:
                pop_engine_identity(identity_token)
        if as_dataframe:
            return pandas.DataFrame([list(r) for r in result.rows], columns=list(result.columns) or None)
        return result

    def asession(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str | None = None,
        ephemeral_scope: SpaceContext | None = None,
        data_row_cap: int | None = None,
    ) -> AsyncPipelineSession:
        return AsyncPipelineSession(
            self.session(mode=mode, space=space, ephemeral_scope=ephemeral_scope, data_row_cap=data_row_cap)
        )

    @_writer_lock_guard
    def aetherspace(
        self,
        name: str | None = None,
        space_context: SpaceContext | None = None,
        *,
        uid: str | None = None,
        notes_file: str | None = None,
        notes: str | None = None,
    ) -> AetherSpace:
        """Create, update, or read an aetherspace on the composite graph."""
        if notes_file is not None and notes is not None:
            raise ConfigError("set at most one of notes and notes_file")
        try:
            uid_norm = MainExecutionOps.validate_space_uid(str(uid).strip()) if uid is not None else None
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace uid: {uid!r}") from exc
        name_norm = str(name).strip().lower() if name is not None else None
        if space_context is None:
            if uid_norm and name_norm:
                raise ConfigError("read aetherspace with uid or name, not both")
            if uid_norm:
                return self._resolve_aetherspace(uid_norm)[0]
            if name_norm:
                return self._resolve_aetherspace_visible_by_name(name_norm)
            raise ConfigError("aetherspace read requires uid or name")
        self._require_owner("aetherspace")
        if uid_norm:
            if uid_norm == MASTER_AETHERSPACE_UID:
                raise ConfigError(
                    "master is the implicit full-scope space; it cannot be created or overwritten",
                )
            desc = self._resolve_aetherspace(uid_norm)[0]
            display = name_norm if name_norm else desc.name
        else:
            if not name_norm:
                raise ConfigError("aetherspace create requires name")
            if name_norm == MASTER_AETHERSPACE_NAME:
                raise ConfigError(
                    "master is the implicit full-scope space; it cannot be created or overwritten",
                )
            try:
                display = TemplateOps.validate_space_name(name_norm)
            except ValueError as exc:
                raise ConfigError(f"invalid aetherspace name: {name!r}") from exc
            uid_norm = allocate_aetherspace_uid(str(self._artifacts_dir))
        mappings = self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        scope_ctx, visible = self._caller_visibility()
        validate_aetherspace_define_within_visibility(
            space_context.tables,
            space_context.columns,
            self._schema_graph,
            scope_ctx,
            visible,
            mappings=mappings,
            federation_manifest=self._federation_manifest,
        )
        validated = validate_space_context_against_graph(
            space_context,
            self._schema_graph,
            federation_manifest=self._federation_manifest,
        )
        if getattr(self, "_sandbox_mode", False):
            Sandbox.require_sandbox_space_lock(display, validated.tables)
        collapsed = collapsed_member_physical_table_names(mappings)
        for table_name in validated.tables | validated.deny_objects:
            logical = collapsed.get(table_name)
            if logical is not None:
                raise ConfigError(
                    f"SpaceContext names collapsed member table {table_name!r}; use logical table {logical!r} instead",
                )
        validate_federation_context_against_mappings(
            FederationContext(
                allow_objects=validated.tables,
                deny_objects=validated.deny_objects,
                allow_columns=validated.columns,
                deny_columns=validated.deny_columns,
            ),
            mappings,
        )
        snapshot = subset_graph_for_space(
            self._schema_graph,
            validated,
            federation_manifest=self._federation_manifest,
        )
        snapshot["uid"] = uid_norm
        snapshot["name"] = display
        if notes_file is not None:
            notes_path, notes_inline = notes_file, None
        elif notes is not None:
            notes_path, notes_inline = None, notes
        else:
            notes_path, notes_inline = validated.notes_file, validated.notes
        if notes_path is not None and str(notes_path).strip() and getattr(self, "_sandbox_mode", False):
            connection = getattr(self, "_native_connection", None)
            host = Sandbox.sandbox_host_for_connection(connection) if connection is not None else None
            if host is not None:
                notes_path = Sandbox.validate_sandbox_aetherspace_notes_pairing(
                    display,
                    notes_path,
                    extract_path=host._extract_path,
                )
        if notes_inline is not None and str(notes_inline).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes=str(notes_inline),
                engine_domain_knowledge=self._domain_knowledge_entries(),
            )
        elif notes_path is not None and str(notes_path).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes_path,
                engine_domain_knowledge=self._domain_knowledge_entries(),
            )
        else:
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                engine_domain_knowledge=self._domain_knowledge_entries(),
            )
        snapshot["uid"] = uid_norm
        snapshot["name"] = display
        snapshot = filter_space_snapshot_sensitive_columns(
            snapshot,
            self._schema_graph,
            federation_manifest=self._federation_manifest,
        )
        save_aetherspace_snapshot(str(self._artifacts_dir), uid_norm, snapshot)
        return aetherspace_descriptor_from_snapshot(uid_norm, snapshot)

    @_writer_lock_guard
    def delete_aetherspace(
        self,
        name: str | None = None,
        *,
        uid: str | None = None,
        persist_learning: bool = True,
    ) -> AetherspaceDeleteResult:
        """Delete one persisted aetherspace snapshot and its learning partition."""
        self._require_owner("delete_aetherspace")
        token = uid if uid is not None else name
        if token is None:
            raise ConfigError("delete_aetherspace requires uid or name")
        desc = self._resolve_aetherspace(str(token))[0]
        return delete_aetherspace(
            str(self._artifacts_dir),
            desc.uid,
            persist_learning=persist_learning,
            schema_graph=self._schema_graph,
        )

    def list_aetherspaces(self, *, include_system: bool = False) -> tuple[AetherSpace, ...]:
        """Return aetherspace descriptors visible to the caller. Owners include the implicit ``master`` space. Consumers omit ``master``. System credential-default spaces are omitted unless *include_system* is True."""
        scope_ctx, visible = self._caller_visibility()
        mappings = self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        out: list[AetherSpace] = []
        if self._schema_role != SchemaRole.CONSUMER:
            out.append(build_master_space_descriptor(self._schema_graph))
        for space_uid, _label in list_saved_aetherspace_entries(str(self._artifacts_dir)):
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), space_uid)
            if snap is None:
                continue
            if not include_system and MainExecutionOps.is_credential_default_snapshot(snap):
                continue
            tables, columns = space_allowed_sets_from_snapshot(snap)
            if aetherspace_within_effective_visibility(
                tables,
                columns,
                self._schema_graph,
                scope_ctx,
                visible,
                mappings=mappings,
                federation_manifest=self._federation_manifest,
            ):
                out.append(aetherspace_descriptor_from_snapshot(space_uid, snap))
        return tuple(out)

    @property
    def default_space_uid(self) -> str:
        """The default aetherspace for this federation: the master space for an owner, the visibility-keyed default for a consumer."""
        if self._schema_role == SchemaRole.CONSUMER:
            uid = getattr(self, "_credential_default_space_uid", None)
            if uid:
                return str(uid)
        return MASTER_AETHERSPACE_UID

    def export_context(self, name: str) -> dict[str, Any]:
        """Return a read-only export document for one named federation context preset. Owner-only."""
        self._require_owner("export_context")
        norm = str(name).strip().lower()
        master_ctx = self._runtime_config.engine_context
        if not isinstance(master_ctx, FederationContext):
            raise ConfigError("export_context on AetherFederation requires a FederationContext master")
        path = Path(str(self._artifacts_dir)) / f"federation_context.{norm}.json"
        if norm != MASTER_AETHERSPACE_NAME and not path.is_file():
            if load_named_schema_context(str(self._artifacts_dir), norm) is None:
                raise ConfigError(f"unknown federation context {name!r}")
        if norm == MASTER_AETHERSPACE_NAME:
            context_fields = {
                "allow_objects": sorted(master_ctx.allow_objects),
                "include": str(master_ctx.include),
                "deny_objects": sorted(master_ctx.deny_objects),
                "deny_columns": sorted(master_ctx.deny_columns),
                "allow_columns": sorted(getattr(master_ctx, "allow_columns", frozenset())),
                "notes_file": master_ctx.notes_file,
                "notes": getattr(master_ctx, "notes", None),
            }
        else:
            row = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            raw_context = row.get("context") if isinstance(row, dict) else None
            if not isinstance(raw_context, dict):
                raise ConfigError(f"unknown federation context {name!r}")
            context_fields = cast(dict[str, Any], raw_context)
        return {"name": norm, "context": context_fields}

    def list_contexts(self) -> tuple[str, ...]:
        """Return saved federation-context names plus the implicit ``master`` context. Owner-only."""
        self._require_owner("list_contexts")
        root = Path(str(self._artifacts_dir))
        names: list[str] = []
        for p in sorted(root.glob("federation_context.*.json")):
            stem = p.name[len("federation_context.") : -len(".json")]
            if not stem or stem == MASTER_AETHERSPACE_NAME:
                continue
            names.append(stem)
        return (MASTER_AETHERSPACE_NAME,) + tuple(names)

    def prepared_federated_outcome(self) -> FederatedPrepareOutcome | None:
        """Return the staged federated prepare outcome from an in-flight turn, if any."""
        return None

    def mapping_suggestions(self) -> tuple[FederationMappingSuggestion, ...]:
        """Return cross-source mapping suggestions computed at federation composition time."""
        self._require_owner("mapping_suggestions")
        self._require_open("mapping_suggestions")
        return tuple(getattr(self, "_federation_mapping_suggestions", ()) or ())

    @_writer_lock_guard
    def run_interactive(self, *, space: str | None = None) -> None:
        self._ensure_llm()
        resolved_space = self._resolve_session_space(space)
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(
            resolved_space
        )
        space_description_overlay: dict[str, Any] | None = None
        norm = str(resolved_space).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME:
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
            if isinstance(snap, dict):
                table_descriptions = snap.get("table_descriptions")
                column_meta = snap.get("column_meta")
                if isinstance(table_descriptions, dict) or isinstance(column_meta, dict):
                    space_description_overlay = {
                        "table_descriptions": dict(table_descriptions or {}),
                        "column_meta": dict(column_meta or {}),
                    }
        payload_visible = space_tables if space_tables else None
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            notify(
                "\nInteractive mode",
                stage="interactive",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
            )
            empty_streak = 0
            question = ""
            while True:
                print("Enter question (empty line to quit): ", end="", flush=True)
                try:
                    raw = input()
                except (EOFError, KeyboardInterrupt):
                    terminated()
                    return
                echo_user_text(raw)
                if raw.strip() == "":
                    empty_streak += 1
                    if empty_streak >= 2:
                        terminated()
                        return
                    notify(
                        "Press Enter again to quit.",
                        stage="interactive",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    continue
                question = raw.strip()
                break

            with PipelineSession(
                self,
                visible_objects=payload_visible,
                execution_visible_objects=self._consumer_visible_objects,
                space_name=str(space).strip().lower(),
                space_tables=space_tables,
                space_columns=space_columns,
                space_deny_objects=space_deny_objects,
                space_deny_columns=space_deny_columns,
                space_description_overlay=space_description_overlay,
            ) as session:
                try:
                    with progress_enabled():
                        step = session.ask(question)
                        while not step.done:
                            _render_interactive_suspend_step(step)
                            print(step.prompt or "", end="", flush=True)
                            try:
                                ans = input()
                            except (EOFError, KeyboardInterrupt):
                                terminated()
                                return
                            if step.kind in YES_NO_SESSION_KINDS:
                                echo_yes_no_answer(ans)
                            else:
                                echo_user_text(ans)
                            step = session.step(ans)
                        _render_interactive_terminal_step(step)
                except (EOFError, KeyboardInterrupt):
                    terminated()
                    return
                except Exception as exc:
                    error(f"{exc.__class__.__name__}: {exc}")
                    return

    @_writer_lock_guard
    def run_seed_warmup(
        self,
        seed_filepath: str,
        interactive_gold: bool = True,
        *,
        abort_on_gold_failure: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> NoReturn:
        """Refuse seed warmup on the federation façade; run it on each member engine instead."""
        raise ConfigError(FEDERATION_WARMUP_UNSUPPORTED_MESSAGE)

    @_writer_lock_guard
    def run_seed_warmup_from_history(
        self,
        sql_history_filepath: str,
        *,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> NoReturn:
        """Refuse SQL-history warmup on the federation façade; run it on each member engine instead."""
        raise ConfigError(FEDERATION_WARMUP_UNSUPPORTED_MESSAGE)

    @_writer_lock_guard
    def run_seed_warmup_from_query_log(
        self,
        *,
        lookback_days: int = 730,
        max_queries: int = 5000,
        min_runs: int = 1,
        user_filter: str | None = None,
        expand: bool = False,
        max_kept_intents: int | None = SeedWarmupConfig.WARMUP_TARGET_CAP,
    ) -> NoReturn:
        """Refuse query-log warmup on the federation façade; run it on each member engine instead."""
        raise ConfigError(FEDERATION_WARMUP_UNSUPPORTED_MESSAGE)

    @_writer_lock_guard
    def run_qsim(
        self,
        num_intents: int = 20,
        num_questions: int = 100,
        seed: int | None = None,
    ) -> None:
        """Generate synthetic NL questions from the composite schema graph."""
        self._require_production_api("run_qsim")
        self._require_open("run_qsim")
        self._ensure_llm()
        self._validate_num_intents(num_intents)
        self._validate_num_questions(num_questions)
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            owner_tok = push_qsim_engine_owner(self)
            scope_tok = push_simulation_artifact_scope_from_owner(self)
            try:
                qsim_run_once(
                    num_intents=num_intents,
                    num_questions=num_questions,
                    seed=seed,
                    artifacts_dir=str(self._artifacts_dir),
                    schema=self._schema_graph,
                    federation_manifest=self._federation_manifest,
                    federation_mappings=self._federation_mappings,
                )
            finally:
                pop_simulation_artifact_partition(scope_tok)
                pop_qsim_engine_owner(owner_tok)

    def get_questions_only(self, version: int) -> None:
        """Print NL questions from a QSim artifact."""
        path = resolve_qsim_path(version, str(self._artifacts_dir))
        if not os.path.isfile(path):
            raise ConfigError(f"QSim questions file not found for version {version}: {path}")
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            print_questions_bundle(version, str(self._artifacts_dir))

    def get_seed_warmup_summary(self) -> SeedWarmupSummarySnapshot:
        """Return the newest seed-warmup summary text if present."""
        self._require_production_api("get_seed_warmup_summary")
        s = find_latest_seed_warmup_summary(str(self._artifacts_dir))
        if s is None:
            return SeedWarmupSummarySnapshot(text="Seed warmup summary: none found.")
        return SeedWarmupSummarySnapshot(text=format_seed_warmup_summary(s))

    def get_qsim_summary(self, start: int, end: int) -> QSimSummarySnapshot:
        """Return QSim summary lines for versions ``start`` through ``end`` inclusive."""
        self._require_production_api("get_qsim_summary")
        summaries = load_qsim_summaries(str(self._artifacts_dir))
        if not summaries:
            raise ConfigError("QSim summary not found; run run_qsim first")
        picked = [s for s in summaries if start <= int(s.version) <= end]
        lines: list[str] = [f"QSim range ({len(picked)} runs):"]
        for s in picked:
            lines.append(format_qsim_summary_line(s))
        if summaries:
            latest = max(summaries, key=lambda x: int(x.version))
            lines.append(
                f"Latest: v{latest.version}  intents={latest.num_intents}  "
                f"questions={latest.num_questions}  seed={latest.seed}",
            )
        return QSimSummarySnapshot(lines=tuple(lines))

    def _resolve_learning_clear_space(self, space: str | None) -> str | None:
        """Return None for all spaces, else a template-store partition key (``master`` or space uid)."""
        if space is None:
            return None
        token = str(space).strip().lower()
        if not token or token == "all":
            return None
        if token in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
            return MASTER_AETHERSPACE_NAME
        desc = self._resolve_aetherspace(token)[0]
        return str(desc.uid).strip().lower()

    @_writer_lock_guard
    def clear_template_store(self, *, space: str | None = None) -> bool:
        """Owner-only: clear composite/member template learning then recompose. ``space=None``/``"all"`` clears every partition (and federation plan templates); otherwise one space partition."""
        self._require_owner("clear_template_store")
        space_key = self._resolve_learning_clear_space(space)
        drain_write_queue(self, str(self._artifacts_dir))
        existed = clear_federation_template_stores(
            str(self._federation_storage_dir) if self._federation_storage_dir else None,
            str(self._artifacts_dir),
            self._schema_graph,
            self._members,
            space=space_key,
        )
        self._recompose()
        self._audit_emit(
            "clear_template_store",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("existed", str(existed)), ("space", space_key or "all")),
        )
        return existed

    @_writer_lock_guard
    def clear_simulation_caches(self) -> int:
        """Remove QSim and seed-warmup artifacts from federation and member trees."""
        self._require_owner("clear_simulation_caches")
        count = clear_simulation_caches_only(str(self._artifacts_dir))
        for engine in self._members.values():
            adir = getattr(engine, "_artifacts_dir", None)
            if adir is not None:
                count += clear_simulation_caches_only(str(adir))
        self._recompose()
        self._audit_emit(
            "clear_simulation_caches",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("removed_files", str(count)),),
        )
        return count

    @_writer_lock_guard
    def clear_all_learning(self, *, keep_structure: bool = True, space: str | None = None) -> None:
        """Owner-only: clear federation learning then recompose. With ``space`` set, only that template partition is cleared across composite/members. Without ``space`` (or ``space="all"``), clears all templates, simulation caches, and optionally member structural overrides."""
        self._require_owner("clear_all_learning")
        space_key = self._resolve_learning_clear_space(space)
        drain_write_queue(self, str(self._artifacts_dir))
        clear_federation_template_stores(
            str(self._federation_storage_dir) if self._federation_storage_dir else None,
            str(self._artifacts_dir),
            self._schema_graph,
            self._members,
            space=space_key,
        )
        count = 0
        if space_key is None:
            count = clear_simulation_caches_only(str(self._artifacts_dir))
            for engine in self._members.values():
                adir = getattr(engine, "_artifacts_dir", None)
                if adir is not None:
                    count += clear_simulation_caches_only(str(adir))
            if not keep_structure:
                for connection_name in self._members:
                    member = self._resolve_member(connection_name, "clear_all_learning")
                    delete_persisted_structure_artifacts(str(member._artifacts_dir / "schema_graph.json.gz"))
        self._recompose()
        self._audit_emit(
            "clear_all_learning",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None) or None,
            details=(
                ("keep_structure", str(keep_structure)),
                ("removed_files", str(count)),
                ("space", space_key or "all"),
            ),
        )

    @_writer_lock_guard(before_acquire=lambda self, op: self._require_no_active_session_turn(op))
    def refresh(self, *, reflect: bool = True) -> RefreshReport:
        """Refresh each member engine, then re-run composite drift checks and plan-template pruning."""
        self._require_open("refresh")
        self._require_owner("refresh")
        member_reports = [member.refresh(reflect=reflect) for member in self._members.values()]
        self._recompose()
        fed_dir = self._composite_federation_dir()
        if self._federation_manifest is not None:
            prune_federation_plan_templates_on_drift(
                fed_dir,
                self._federation_member_graphs or {},
                self._federation_manifest,
                self._mappings,
            )
        self._replay_composite_overrides()
        migration_tier = MigrationTier.NO_CHANGE
        for report in member_reports:
            if report.migration_tier != MigrationTier.NO_CHANGE:
                migration_tier = report.migration_tier
        diagnostics: list[Diagnostic] = []
        for report in member_reports:
            diagnostics.extend(report.diagnostics)
        tables_added = tuple(sorted({table for report in member_reports for table in report.tables_added}))
        tables_removed = tuple(sorted({table for report in member_reports for table in report.tables_removed}))
        columns_added = tuple(sorted({pair for report in member_reports for pair in report.columns_added}))
        columns_removed = tuple(sorted({pair for report in member_reports for pair in report.columns_removed}))
        return RefreshReport(
            migration_tier=migration_tier,
            schema_changed=any(report.schema_changed for report in member_reports),
            tables_added=tables_added,
            tables_removed=tables_removed,
            columns_added=columns_added,
            columns_removed=columns_removed,
            templates_invalidated=sum(report.templates_invalidated for report in member_reports),
            orphans_removed=sum(report.orphans_removed for report in member_reports),
            bytes_reclaimed=sum(report.bytes_reclaimed for report in member_reports),
            diagnostics=tuple(diagnostics),
        )

    @_writer_lock_guard
    def close(self) -> None:
        """Dispose federation-owned source runtimes and the coordinator dialect. Idempotent."""
        if getattr(self, "_closed", False):
            return
        drain_write_queue(self, str(self._artifacts_dir))
        dispose_engine_dialect(self._dialect)
        runtimes = self._federation_source_runtimes
        if runtimes is not None:
            dispose_federation_source_runtimes(runtimes, member_engines=self._members)
            self._federation_source_runtimes = None
        release_close_resources(self)
        drop_engine_skeleton_cache_owner(self)
        clear_expansion_subtree_pool(str(self._artifacts_dir))
        sink_token = getattr(self, "_diagnostic_sink_token", None)
        if sink_token is not None:
            pop_diagnostic_sink(sink_token)
            self._diagnostic_sink_token = cast(Any, None)
        self._closed = True
        self._audit_emit(
            "close",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
        )

    def __enter__(self) -> AetherFederation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        self.close()
        return False


def _print_structure_summary(report: StructureReport) -> None:
    """Emit a fixed-template summary of a ``StructureReport`` through the notify channel."""
    notify("Schema overrides applied:", stage="structure", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    notify(
        f"  Tables updated:           {report.table_edits}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Columns updated:          {report.column_edits}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  FK edges added:           {report.fks_added}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  FK edges endorsed (user): {report.fks_endorsed}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  FK edges removed:         {report.fks_removed}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  PKs endorsed (user):     {report.pks_endorsed}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Inferred PKs cleared:     {report.pks_blocked}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  PK/FK roles coerced:      {report.coerced_columns}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Redundant inferences:     {report.collapsed_inferences}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Descriptions refined:     {report.descriptions_refined}",
        stage="structure",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    if report.skipped:
        notify(
            f"  Soft skips ({len(report.skipped)}):",
            stage="structure",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )
        for skip in report.skipped:
            notify(
                f"    {skip.path}  -  {skip.reason}",
                stage="structure",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                details=(("path", skip.path), ("reason", skip.reason)),
            )
    else:
        notify(
            "  Soft skips:               none",
            stage="structure",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )


try:
    __version__ = version("aetherdialect")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

_PUBLIC_API = (
    AccessError,
    AggregateJoinFanOutError,
    AetherEngine,
    AetherError,
    AetherFederation,
    AetherSpace,
    AmbiguousDateLiteralError,
    ArtifactLockTimeoutError,
    AsyncPipelineSession,
    AuditEvent,
    DomainKnowledgeEntry,
    ClauseWidenedRowsetError,
    ComparisonJoinScopeExceededError,
    ConfigError,
    ConfigSnapshot,
    DatabaseConnectionError,
    DatabaseExecutionError,
    DatabasePingFailed,
    DataQualityReport,
    Diagnostic,
    EngineContext,
    EngineLimits,
    FederationCapExceededError,
    FederationConfigError,
    FederationContext,
    FederationDeclarationError,
    FederationIneligibleError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationLimits,
    FederationMalformedMemberAnswerError,
    FederationMappingsAppliedSidecarError,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationMemberUnprofilableError,
    FederationPartialFailureError,
    FederationRuntimeError,
    FederationTurnCancelledError,
    JoinCandidateCapExceededError,
    JoinColumnCountMismatchError,
    JoinInjectionAlignmentError,
    JoinInjectionFailedError,
    JoinPathKeyTypeError,
    JoinPathTieCapExceededError,
    JoinProbeEdgeKindMismatchError,
    LlmJsonExhausted,
    LlmTransientFailure,
    MigrationPendingError,
    MigrationPreview,
    MockFixtureMissingError,
    NoJoinPathError,
    NullInNegatedListError,
    OwnerOnlyOperationError,
    PersistedFederationInspection,
    PhaseProgressEvent,
    PipelineSession,
    PipelineSuspended,
    ProbeCtePlacementError,
    QSimSummarySnapshot,
    RefinementRetry,
    RegistryRenderError,
    ResultCapExceededError,
    RetryableDatabaseExecutionError,
    RetryableError,
    RetryableFederationPartialFailureError,
    Sandbox,
    SchemaAccessError,
    SchemaInvariantError,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    SessionActiveError,
    SessionError,
    SessionOutcome,
    SessionStep,
    SessionTurnCancelledError,
    SpaceContext,
    StatementTimeoutError,
    SubdayDateWindowOnDateColumnError,
    SuspendedSessionExpiredError,
    UploadIngestResult,
    __version__,
    inspect_tabular_upload,
)
