"""Public AetherEngine facade delegating construction and runners to main_execution. Attributes on AetherEngine whose names start with a single underscore are private implementation details and are not part of the public stability contract."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import shutil
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
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_METHOD_SEMANTICS,
    FEDERATION_MIGRATION_MAP_FILENAME,
    FEDERATION_WARMUP_UNSUPPORTED_MESSAGE,
    MASTER_AETHERSPACE_NAME,
    MIGRATION_MAP_FILENAME,
    NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
    SCHEMA_OVERRIDES_APPLIED_SUFFIX,
    SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT,
    SCHEMA_OVERRIDES_DEFAULT_FILENAME,
    TABLE_PREVIEW_DEFAULT_LIMIT,
    WRITE_QUEUE_FILENAME,
    YES_NO_SESSION_KINDS,
)
from ._contracts_base import (
    AccessError,
    AetherEngineInitResult,
    AetherError,
    AetherFederationInitResult,
    AetherSpace,
    AggregateJoinFanOutError,
    AmbiguousDateLiteralError,
    ArtifactLockTimeoutError,
    AuditEvent,
    BusinessKnowledgeEntry,
    BusinessKnowledgeHolder,
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
    EngineIdentity,
    FederationCapExceededError,
    FederationConfigError,
    FederationContext,
    FederationDeclarationError,
    FederationIneligibleError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMappings,
    FederationMappingsAppliedSidecarError,
    FederationMappingSuggestion,
    FederationMemberEngine,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationMemberUnprofilableError,
    FederationMethodScope,
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
    MigrationTier,
    MockFixtureMissingError,
    NoJoinPathError,
    NullInNegatedListError,
    OverrideReport,
    OverrideSkip,
    OwnerOnlyOperationError,
    PersistedFederationInspection,
    PhaseProgressEvent,
    PipelineSuspended,
    PlanPreviewResult,
    ProbeCtePlacementError,
    QSimSummarySnapshot,
    RefreshReport,
    RegistryRenderError,
    ResultCapExceededError,
    RetryableDatabaseExecutionError,
    RetryableError,
    RetryableFederationPartialFailureError,
    SchemaAccessError,
    SchemaInclude,
    SchemaInvariantError,
    SchemaRole,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    SessionActiveError,
    SessionStep,
    SessionTurnCancelledError,
    SpaceContext,
    StatementTimeoutError,
    StoredTemplateDetail,
    StoredTemplateSummary,
    SubdayDateWindowOnDateColumnError,
    SuspendedSessionExpiredError,
    TablePreviewResult,
    TemplateExecutionResult,
    WriteQueueEvent,
)
from ._contracts_core import FederatedPrepareOutcome, Template
from ._contracts_schema import CsvSourceSelection, UploadIngestResult
from ._core_utils import (
    dataframe_to_row_tuples,
    diagnostic_print_listener,
    echo_user_text,
    echo_yes_no_answer,
    emit_write_queue_event,
    error,
    notify,
    pop_construction_phase_callback,
    pop_engine_identity,
    pop_engine_limits,
    print_query_result,
    progress_enabled,
    push_construction_phase_callback,
    push_engine_identity,
    push_engine_limits,
    register_dialect_live_handles,
    release_close_resources,
    require_driver,
    stable_json,
    terminated,
    validate_federation_pool_capacity,
)
from ._data_quality import inspect_tabular_upload
from ._dialect_sqlglot_engines import CsvDialect
from ._expansion_ops import clear_expansion_subtree_pool
from ._federation import (
    apply_federation_composite_overrides,
    archive_federation_editor_file,
    binding_from_member_engine,
    collapsed_member_physical_table_names,
    export_federation_composite_overrides,
    export_federation_declaration,
    federation_member_artifacts_dir_for_purge,
    finalize_federation_composite_overrides,
    inspect_persisted_federation,
    load_federation_declaration_from_path,
    parse_federation_declaration,
    prune_federation_plan_templates_for_sources,
    prune_federation_plan_templates_on_drift,
    purge_federation_member_artifacts,
    reconcile_authored_declaration_for_members,
    validate_federation_context_against_mappings,
)
from ._llm_provider import LLMProvider
from ._main_execution import (
    MainExecutionOps,
    PipelineSession,
)
from ._pipeline import (
    execute_stored_template_by_ref,
    preview_plan_on_engine,
    preview_plan_on_federation,
)
from ._qsim import (
    clear_engine_skeleton_cache,
    drop_engine_skeleton_cache_owner,
    pop_qsim_engine_owner,
    push_qsim_engine_owner,
    register_engine_skeleton_cache_owner,
)
from ._sandbox import (
    Sandbox,
    SandboxHandle,
)
from ._schema_catalog import extract_business_knowledge_from_notes
from ._schema_overrides import (
    apply_overrides_and_persist,
    clear_persisted_overrides,
    dump_schema_overrides_to_path,
)
from ._templates import (
    LazyTemplateMapping,
    TemplateOps,
)

aetherspace_descriptor_from_snapshot = MainExecutionOps.aetherspace_descriptor_from_snapshot
apply_aetherspace_json = MainExecutionOps.apply_aetherspace_json
build_master_space_descriptor = MainExecutionOps.build_master_space_descriptor
clear_federation_template_stores = MainExecutionOps.clear_federation_template_stores
clear_simulation_caches_only = MainExecutionOps.clear_simulation_caches_only
clear_template_store_only = MainExecutionOps.clear_template_store_only
delete_aetherspace_snapshot = MainExecutionOps.delete_aetherspace_snapshot
describe_federation_config = MainExecutionOps.describe_federation_config
describe_runtime_config = MainExecutionOps.describe_runtime_config
dispose_engine_dialect = MainExecutionOps.dispose_engine_dialect
dispose_federation_source_runtimes = MainExecutionOps.dispose_federation_source_runtimes
drain_write_queue = MainExecutionOps.drain_write_queue
enrich_space_snapshot_with_notes = MainExecutionOps.enrich_space_snapshot_with_notes
engine_schema_json_path = MainExecutionOps.engine_schema_json_path
engine_template_store_dir = MainExecutionOps.engine_template_store_dir
export_aetherspace_json = MainExecutionOps.export_aetherspace_json
export_named_schema_context_json = MainExecutionOps.export_named_schema_context_json
federation_stores_by_source = MainExecutionOps.federation_stores_by_source
find_latest_seed_warmup_summary = MainExecutionOps.find_latest_seed_warmup_summary
format_qsim_summary_line = MainExecutionOps.format_qsim_summary_line
format_seed_warmup_summary = MainExecutionOps.format_seed_warmup_summary
initialize_aether_engine = MainExecutionOps.initialize_aether_engine
initialize_aether_federation = MainExecutionOps.initialize_aether_federation
intersect_space_scope = MainExecutionOps.intersect_space_scope
list_named_schema_context_names = MainExecutionOps.list_named_schema_context_names
list_saved_aetherspace_names = MainExecutionOps.list_saved_aetherspace_names
load_aetherspace_snapshot = MainExecutionOps.load_aetherspace_snapshot
load_named_schema_context = MainExecutionOps.load_named_schema_context
load_qsim_summaries = MainExecutionOps.load_qsim_summaries
preview_schema_migration = MainExecutionOps.preview_schema_migration
preview_table_on_engine = MainExecutionOps.preview_table_on_engine
preview_table_on_federation = MainExecutionOps.preview_table_on_federation
print_questions_bundle = MainExecutionOps.print_questions_bundle
qsim_run_once = MainExecutionOps.qsim_run_once
refresh_engine_connection = MainExecutionOps.refresh_engine_connection
resolve_qsim_path = MainExecutionOps.resolve_qsim_path
run_seed_warmup_from_history_execution = MainExecutionOps.run_seed_warmup_from_history_execution
run_seed_warmup_from_query_log_execution = MainExecutionOps.run_seed_warmup_from_query_log_execution
save_aetherspace_snapshot = MainExecutionOps.save_aetherspace_snapshot
seed_warmup_run_once = MainExecutionOps.seed_warmup_run_once
space_allowed_sets_from_snapshot = MainExecutionOps.space_allowed_sets_from_snapshot
space_deny_sets_from_snapshot = MainExecutionOps.space_deny_sets_from_snapshot
subset_graph_for_space = MainExecutionOps.subset_graph_for_space
validate_space_context_against_graph = MainExecutionOps.validate_space_context_against_graph
validate_space_subset_of_execution_context = MainExecutionOps.validate_space_subset_of_execution_context

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
    "apply_aetherspace",
    "apply_overrides",
    "clear_all_learning",
    "clear_persisted_overrides",
    "clear_simulation_caches",
    "clear_template_store",
    "close",
    "delete_aetherspace",
    "execute_template",
    "export_aetherspace",
    "export_context",
    "export_overrides",
    "ingest_upload_sources",
    "refresh_connection",
    "run_interactive",
    "run_qsim",
    "run_seed_warmup",
    "run_seed_warmup_from_history",
    "run_seed_warmup_from_query_log",
)

MUTATING_FEDERATION_METHODS: tuple[str, ...] = (
    "add_engine",
    "aetherspace",
    "apply_aetherspace",
    "apply_federation_declaration",
    "apply_migration_map",
    "apply_overrides",
    "clear_all_learning",
    "clear_persisted_overrides",
    "clear_simulation_caches",
    "clear_template_store",
    "close",
    "delete_aetherspace",
    "execute_template",
    "export_aetherspace",
    "export_federation_declaration",
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

    async def cancel_active_federation_turn(self) -> bool:
        """Cancel an in-flight federated turn on the underlying session."""
        return cast(bool, await self._to_thread(self._inner.cancel_active_federation_turn))

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
    if step.message:
        notify(step.message, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    if step.sql is not None:
        hdr = list(step.data.columns) if step.data is not None else None
        rows = dataframe_to_row_tuples(step.data)
        print_query_result(rows, _sql_text_for_print(step.sql), headers=hdr)


def _render_interactive_terminal_step(step: SessionStep) -> None:
    """Emit terminal errors, messages, and optional final SQL preview."""
    if step.error:
        error(step.error)
        return
    if step.message:
        notify(step.message, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
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
        "_construction_phase_callback",
        "_ask_phase_callback",
        "_pipeline_writer_lock",
        "_config_file",
        "_schema_role",
        "_consumer_visible_objects",
        "_context_name",
        "_sandbox_mode",
        "_sandbox_closed",
        "_trust_bundled_baseline",
        "_init_notices",
        "_token_provider",
        "_business_knowledge",
        "_schema_json_path",
        "_template_store_dir",
        "_tenant_slug",
        "_limits",
        "_closed",
        "_engine_identity",
    )

    def __dir__(self) -> list[str]:
        """Return names intended for interactive discovery."""
        return sorted(
            (
                "show_config",
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
                "export_overrides",
                "apply_overrides",
                "export_schema_overrides",
                "apply_schema_overrides",
                "aetherspace",
                "apply_aetherspace",
                "delete_aetherspace",
                "export_aetherspace",
                "export_context",
                "export_engine_context",
                "list_aetherspaces",
                "list_contexts",
                "list_engine_contexts",
                "clear_persisted_overrides",
                "clear_template_store",
                "clear_simulation_caches",
                "clear_all_learning",
                "list_templates",
                "fetch_template",
                "execute_template",
                "preview_table",
                "preview_plan",
                "apply_migration_map",
                "refresh_connection",
                "refresh",
                "offline_sandbox",
                "sandbox_questions",
                "sandbox_catalog",
                "sandbox_paraphrase_pairs",
                "sandbox_validation_failure_demo",
                "sandbox_feedback_demo",
            ),
        )

    def __init__(
        self,
        engine_context: EngineContext | str | None = None,
        *,
        artifacts_dir: str | None = None,
        tenant_slug: str | None = None,
        config_file: str | os.PathLike[str] | None = None,
        connection: str | Mapping[str, Any] | None = None,
        execution_engine: Any = None,
        native_connection: Any = None,
        source_selections: Mapping[str, Mapping[str, Any]] | None = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
        construction_phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
        ask_phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
        role: SchemaRole = SchemaRole.OWNER,
        trust_bundled_baseline: bool = False,
        init_notices: tuple[str, ...] = (),
        token_provider: Callable[[], str | Mapping[str, str]] | None = None,
        limits: EngineLimits | None = None,
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
            tenant_slug: Optional tenant segment inserted before the connection slug to isolate artifact trees.
            config_file: Path to a TOML file. When set, every mapped field that appears in the file is authoritative for the corresponding process environment key (empty TOML values clear any inherited environment value for that key); fields omitted from the file are still read from ``os.environ``. When omitted, settings are read from ``os.environ`` only.
            connection: Named TOML connection sub-block when multiple databases of the same engine type share one config file, or a mapping of connection environment keys for this instance only (never written to ``os.environ``). String form selects credentials and the artifact slug; it is unrelated to any federation ``source_id``.
            execution_engine: Optional SQLAlchemy engine for query execution (caller-owned pool / read replica).
            native_connection: Optional native duckdb or sqlite3 connection for embedded engines.
            For DuckDB and SQLite, ``native_connection`` or ``DuckDBRuntimeConfig.attach_connection`` ensures reflection and execution share one in-memory or file-backed database.
            ``execution_engine`` is honored when it wraps the same ``StaticPool`` connection.
            source_selections: CSV file engine only: per-filename interpretation accepted after :func:`inspect_tabular_upload` (``header_row``, ``table_range``, ``append_regions``, etc.).
            audit_sink: Optional callback receiving :class:`AuditEvent` records at lifecycle boundaries.
            role: Schema identity role; ``owner`` may mutate shared artifacts, ``consumer`` pins the owner snapshot id.
            token_provider: Optional callable returning a fresh secret string or credential field mapping consulted when opening the database connection (initial construction and :meth:`refresh_connection`).

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
        self._construction_phase_callback = construction_phase_callback
        self._ask_phase_callback = ask_phase_callback
        self._pipeline_writer_lock = threading.Lock()
        self._schema_role = role
        self._consumer_visible_objects: frozenset[str] | None = None
        self._context_name = MASTER_AETHERSPACE_NAME
        self._sandbox_mode = False
        self._sandbox_closed = False
        self._trust_bundled_baseline = bool(trust_bundled_baseline)
        self._init_notices = tuple(init_notices)
        self._token_provider = token_provider
        self._limits = limits if limits is not None else EngineLimits()
        self._closed = False
        self._tenant_slug = str(tenant_slug).strip() if tenant_slug is not None and str(tenant_slug).strip() else None
        bundle = self._initialize_engine_bundle(
            engine_context,
            artifacts_dir=artifacts_dir,
            tenant_slug=self._tenant_slug,
            config_file=config_file,
            connection=connection,
            log_sink=_init_log_sink,
            execution_engine=self._execution_engine,
            native_connection=self._native_connection,
            schema_role=role,
            source_selections=source_selections,
            trust_bundled_baseline=self._trust_bundled_baseline,
            token_provider=token_provider,
        )
        self._apply_init_bundle(bundle)
        self._business_knowledge = BusinessKnowledgeHolder()
        self._ingest_notes_business_knowledge()
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
            "tenant_slug": self._tenant_slug,
            "log_sink": _init_log_sink,
            "execution_engine": self._execution_engine,
            "native_connection": getattr(self, "_native_connection", None),
            "config_file": self._config_file,
            "schema_role": self._schema_role,
            "trust_bundled_baseline": getattr(self, "_trust_bundled_baseline", False),
        }

    def _initialize_engine_bundle(
        self,
        engine_context: EngineContext | str | None,
        **kwargs: Any,
    ) -> AetherEngineInitResult:
        """Run :func:`initialize_aether_engine` with construction-phase progress wired."""
        token = push_construction_phase_callback(self._construction_phase_callback)
        try:
            return initialize_aether_engine(engine_context, **kwargs)
        finally:
            pop_construction_phase_callback(token)

    @property
    def init_notices(self) -> tuple[str, ...]:
        """Fixture-resolution notices emitted during engine construction."""
        return getattr(self, "_init_notices", ())

    def _single_engine_context(self) -> EngineContext:
        ctx = self._runtime_config.engine_context
        if not isinstance(ctx, EngineContext):
            raise ConfigError("this operation requires a single-engine context")
        return ctx

    def _require_owner(self, operation: str) -> None:
        if self._schema_role != SchemaRole.OWNER:
            raise OwnerOnlyOperationError(operation)

    def _require_master_context(self, operation: str) -> None:
        if getattr(self, "_context_name", MASTER_AETHERSPACE_NAME) != MASTER_AETHERSPACE_NAME:
            raise ConfigError(
                f"Operation {operation!r} requires the master engine context; "
                "this instance is bound to a non-master context.",
            )

    def _resolve_aetherspace(
        self, name: str
    ) -> tuple[AetherSpace, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("aetherspace name must be non-empty")
        if norm == MASTER_AETHERSPACE_NAME:
            desc = build_master_space_descriptor(self._schema_graph)
            return desc, frozenset(), frozenset(), frozenset(), frozenset()
        snap = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
        if snap is None:
            raise ConfigError(f"unknown aetherspace {name!r}")
        tables_raw = snap.get("tables")
        if isinstance(tables_raw, (list, tuple)) and len(tables_raw) == 0:
            raise ConfigError("space empty after schema migration; redefine")
        desc = aetherspace_descriptor_from_snapshot(norm, snap)
        tables, columns = space_allowed_sets_from_snapshot(snap)
        deny_objects, deny_columns = space_deny_sets_from_snapshot(snap)
        return desc, tables, columns, deny_objects, deny_columns

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
    ) -> None:
        sink = self._audit_sink
        if sink is None:
            return
        ev = AuditEvent(
            event_type=event_type,
            timestamp_iso=datetime.now(UTC).isoformat(),
            question=question,
            schema_hash=schema_hash,
            provider=self._llm_config.provider,
            details=details,
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
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
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

    def set_business_knowledge(self, entries: Sequence[BusinessKnowledgeEntry]) -> int:
        """Replace prompt-time business knowledge and return the new monotonic version."""
        holder = getattr(self, "_business_knowledge", None)
        if holder is None:
            self._business_knowledge = BusinessKnowledgeHolder()
            holder = self._business_knowledge
        return holder.set(entries, self._schema_graph)

    def _ingest_notes_business_knowledge(self) -> None:
        """Pass B: extract business knowledge from construction notes when present."""
        ctx = getattr(getattr(self, "_runtime_config", None), "engine_context", None)
        notes_path = getattr(ctx, "notes_file", None) if ctx is not None else None
        notes_inline = getattr(ctx, "notes", None) if ctx is not None else None
        notes_content: str | None = None
        if notes_inline is not None and str(notes_inline).strip():
            notes_content = str(notes_inline)
        elif notes_path is not None and str(notes_path).strip():
            path = os.path.expanduser(str(notes_path).strip())
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    notes_content = fh.read()
        entries = extract_business_knowledge_from_notes(notes_content, self._schema_graph)
        if entries:
            self.set_business_knowledge(entries)

    def business_knowledge(self) -> tuple[BusinessKnowledgeEntry, ...]:
        """Return the active business knowledge entries."""
        holder = getattr(self, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            return ()
        return holder.entries()

    def business_knowledge_digest(self) -> str:
        """Return a stable digest of the active business knowledge."""
        holder = getattr(self, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            return ""
        return holder.digest()

    def business_knowledge_version(self) -> int:
        """Return the monotonic version counter for business knowledge updates."""
        holder = getattr(self, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            return 0
        return holder.version()

    def export_space_knowledge(self, space: str | None = None) -> dict[str, Any]:
        """Return business knowledge for master or one named space (key/kind/text only)."""
        space_snapshot = None
        if space is not None:
            norm = str(space).strip().lower()
            if norm and norm != MASTER_AETHERSPACE_NAME:
                space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
        return MainExecutionOps.build_space_knowledge_export(
            engine_entries=self.business_knowledge(),
            business_knowledge_version=self.business_knowledge_version(),
            space=space,
            space_snapshot=space_snapshot,
        )

    def export_knowledge(self) -> dict[str, Any]:
        """Return engine plus per-space business knowledge wrapper for agents and metadata review."""
        snapshots: dict[str, Any] = {}
        for name in list_saved_aetherspace_names(str(self._artifacts_dir)):
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), name)
            if snap is not None:
                snapshots[name] = snap
        return MainExecutionOps.build_knowledge_export(
            engine_entries=self.business_knowledge(),
            business_knowledge_version=self.business_knowledge_version(),
            space_snapshots=snapshots,
        )

    def export_metadata(self, space: str | None = None) -> dict[str, Any]:
        """Return deterministic table/column inventory for master or one named space."""
        space_snapshot = None
        if space is not None:
            norm = str(space).strip().lower()
            if norm and norm != MASTER_AETHERSPACE_NAME:
                space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
        return MainExecutionOps.build_metadata_export(
            schema_graph=self._schema_graph,
            space=space,
            space_snapshot=space_snapshot,
        )

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

    @property
    def live_connection_handle(self) -> Any:
        """Return the SQLAlchemy engine or native connection used for execution."""
        if self._execution_engine is not None:
            return self._execution_engine
        native = getattr(self, "_native_connection", None)
        if native is not None:
            return native
        sa_engine = getattr(self._dialect, "engine", None)
        if sa_engine is not None:
            return sa_engine
        return getattr(self._dialect, "connection", None)

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
    def refresh_connection(
        self,
        credentials: str | Mapping[str, str] | None = None,
    ) -> None:
        """Replace database credentials and reopen the live connection without rebuilding schema artifacts."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        engine_type = self.dialect
        self._dialect = refresh_engine_connection(
            engine_type=engine_type,
            dialect=self._dialect,
            credentials=credentials,
            token_provider=getattr(self, "_token_provider", None),
            execution_engine=self._execution_engine,
            native_connection=getattr(self, "_native_connection", None),
        )
        self._audit_emit(
            "connection_refresh",
            schema_hash=self._effective_structural_hash or None,
            details=(("engine", engine_type),),
        )

    @_writer_lock_guard
    def refresh(self, *, reflect: bool = True) -> RefreshReport:
        """Reconcile artifacts against the live or cached schema graph without reconstructing the engine."""
        self._require_open("refresh")
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
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
    def last_overrides_skipped(self) -> tuple[OverrideSkip, ...]:
        """Per-entry skips recorded by the most recent overrides replay (empty before any sidecar applies)."""
        return tuple(getattr(self._schema_graph, "_last_overrides_skipped", ()) or ())

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
        space: str = "master",
        ephemeral_scope: SpaceContext | None = None,
        data_row_cap: int | None = None,
    ) -> PipelineSession:
        """Return a programmatic session sharing this instance's schema graph and template store. ``writer`` mode may mutate artifacts and takes ``_pipeline_writer_lock`` only around store and write-queue mutations; ``reader`` mode is read-only and never takes that lock."""
        self._require_open("session")
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        Sandbox.require_sandbox_adoption(self)
        if self._schema_role == SchemaRole.CONSUMER and mode == "writer":
            raise OwnerOnlyOperationError("PipelineSession(mode='writer')")
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(space)
        space_tables, space_columns, space_deny_objects, space_deny_columns = intersect_space_scope(
            space_tables,
            space_columns,
            space_deny_objects,
            space_deny_columns,
            ephemeral_scope,
        )
        space_description_overlay: dict[str, Any] | None = None
        norm = str(space).strip().lower()
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
        exec_ctx = getattr(self._runtime_config, "execution_context", None)
        if exec_ctx is None:
            exec_ctx = self._runtime_config.engine_context
        validate_space_subset_of_execution_context(
            space_tables,
            space_columns,
            exec_ctx,
            self._schema_graph,
        )
        payload_visible = space_tables if space_tables else None
        return PipelineSession(
            self,
            mode=mode,
            visible_objects=payload_visible,
            execution_visible_objects=self._consumer_visible_objects,
            space_name=str(space).strip().lower(),
            space_tables=space_tables,
            space_columns=space_columns,
            space_deny_objects=space_deny_objects,
            space_deny_columns=space_deny_columns,
            space_description_overlay=space_description_overlay,
            data_row_cap=data_row_cap,
        )

    def preview_table(self, table_name: str, *, limit: int = TABLE_PREVIEW_DEFAULT_LIMIT) -> TablePreviewResult:
        """Return the first rows of *table_name* through scope and sensitivity gates."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        return preview_table_on_engine(self, table_name, limit=limit)

    def preview_plan(self, question: str) -> PlanPreviewResult:
        """Return what a turn would run for *question* without generating or executing SQL."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        return preview_plan_on_engine(
            self,
            question,
            visible_objects=self._consumer_visible_objects,
            execution_visible_objects=self._consumer_visible_objects,
        )

    def asession(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str = "master",
    ) -> AsyncPipelineSession:
        """Async wrapper around :meth:`session` (uses threads; underlying API remains synchronous)."""
        return AsyncPipelineSession(self.session(mode=mode, space=space))

    @_writer_lock_guard
    def aetherspace(
        self,
        name: str,
        space_context: SpaceContext | None = None,
        *,
        notes_file: str | None = None,
        notes: str | None = None,
    ) -> AetherSpace:
        """Check or define a named aetherspace scope snapshot."""
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("aetherspace name must be non-empty")
        if space_context is None:
            return self._resolve_aetherspace(norm)[0]
        self._require_owner("aetherspace")
        self._require_master_context("aetherspace")
        if norm == MASTER_AETHERSPACE_NAME:
            raise ConfigError(
                "master is the implicit full-scope space; it cannot be created or overwritten",
            )
        if notes_file is not None and notes is not None:
            raise ConfigError("set at most one of notes and notes_file")
        validated = validate_space_context_against_graph(space_context, self._schema_graph)
        snapshot = subset_graph_for_space(self._schema_graph, validated)
        if notes_file is not None:
            notes_path, notes_inline = notes_file, None
        elif notes is not None:
            notes_path, notes_inline = None, notes
        else:
            notes_path, notes_inline = validated.notes_file, validated.notes
        if notes_inline is not None and str(notes_inline).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes=str(notes_inline),
            )
        elif notes_path is not None and str(notes_path).strip():
            if getattr(self, "_sandbox_mode", False):
                connection = getattr(self, "_native_connection", None)
                host = Sandbox.sandbox_host_for_connection(connection) if connection is not None else None
                if host is not None:
                    Sandbox.validate_sandbox_aetherspace_notes_pairing(
                        validated,
                        notes_path,
                        extract_path=host._extract_path,
                    )
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes_path,
            )
        save_aetherspace_snapshot(str(self._artifacts_dir), norm, snapshot)
        return aetherspace_descriptor_from_snapshot(norm, snapshot)

    @_writer_lock_guard
    def export_aetherspace(self, name: str) -> Path:
        """Export a JSON snapshot of one named aetherspace for review or apply."""
        self._require_master_context("export_aetherspace")
        norm = str(name).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME and load_aetherspace_snapshot(str(self._artifacts_dir), norm) is None:
            raise ConfigError(f"unknown aetherspace {name!r}")
        return export_aetherspace_json(str(self._artifacts_dir), norm, self._schema_graph)

    @_writer_lock_guard
    def apply_aetherspace(self, name: str, *, source: str | os.PathLike[str] | None = None) -> AetherSpace:
        """Apply an exported aetherspace JSON document and persist it under *name*."""
        self._require_owner("apply_aetherspace")
        self._require_master_context("apply_aetherspace")
        norm = str(name).strip().lower()
        return apply_aetherspace_json(
            str(self._artifacts_dir),
            norm,
            self._schema_graph,
            source=source,
        )

    @_writer_lock_guard
    def delete_aetherspace(self, name: str) -> bool:
        """Delete one persisted named aetherspace snapshot."""
        self._require_owner("delete_aetherspace")
        self._require_master_context("delete_aetherspace")
        norm = str(name).strip().lower()
        return delete_aetherspace_snapshot(str(self._artifacts_dir), norm)

    def list_aetherspaces(self) -> tuple[str, ...]:
        """Return saved aetherspace names plus the implicit ``master`` space."""
        self._require_master_context("list_aetherspaces")
        saved = list_saved_aetherspace_names(str(self._artifacts_dir))
        return (MASTER_AETHERSPACE_NAME,) + saved

    @_writer_lock_guard
    def export_context(self, name: str) -> Path:
        """Export a read-only JSON snapshot of one named engine context."""
        self._require_master_context("export_context")
        norm = str(name).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME and load_named_schema_context(str(self._artifacts_dir), norm) is None:
            raise ConfigError(f"unknown engine context {name!r}")
        master_ctx = self._runtime_config.engine_context
        if not isinstance(master_ctx, EngineContext):
            raise ConfigError("export_context requires a single-engine context")
        return export_named_schema_context_json(
            str(self._artifacts_dir),
            norm,
            master_ctx,
        )

    def list_contexts(self) -> tuple[str, ...]:
        """Return saved engine-context names plus the implicit ``master`` context."""
        self._require_master_context("list_contexts")
        saved = list_named_schema_context_names(str(self._artifacts_dir))
        return (MASTER_AETHERSPACE_NAME,) + saved

    def export_engine_context(self, name: str) -> Path:
        """Deprecated alias for :meth:`export_context`."""
        return cast(Path, self.export_context(name))

    def list_engine_contexts(self) -> tuple[str, ...]:
        """Deprecated alias for :meth:`list_contexts`."""
        return self.list_contexts()

    def _templates_for_space(self, space: str) -> dict[str, Template] | LazyTemplateMapping:
        """Load the in-memory template map for one aetherspace namespace."""
        norm = str(space).strip().lower()
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

    def list_templates(self, *, space: str = "master") -> tuple[StoredTemplateSummary, ...]:
        """Enumerate caller-visible stored templates for one aetherspace namespace."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        templates = self._templates_for_space(space)
        return list_stored_template_summaries(templates, space=space, dialect=self._dialect)

    def fetch_template(self, template_ref: str, *, space: str = "master") -> StoredTemplateDetail:
        """Fetch one stored template by id or ``sql_fp`` hash."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        templates = self._templates_for_space(space)
        tmpl = resolve_template_ref(template_ref, templates)
        if tmpl is None or not template_visible_to_callers(tmpl):
            raise ConfigError(f"unknown template ref {template_ref!r}")
        vh = tmpl.value_history
        hist_idx = 0
        if vh.questions:
            primary = primary_template_q_norm(tmpl)
            hist_idx = vh.questions.index(primary) if primary in vh.questions else 0
        return build_stored_template_detail(
            tmpl,
            space=space,
            schema=self._schema_graph,
            dialect=self._dialect,
            history_index=hist_idx,
        )

    @_writer_lock_guard
    def execute_template(
        self,
        template_ref: str,
        params: dict[str, Any] | None = None,
        *,
        question: str | None = None,
        space: str = "master",
        as_dataframe: bool = False,
    ) -> TemplateExecutionResult | pandas.DataFrame:
        """Execute one stored template by id or ``sql_fp`` with caller- supplied bind values."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dict or None")
        bind = dict(params or ())
        templates = self._templates_for_space(space)
        runtime_cfg = getattr(self, "_runtime_config", None)
        execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
        scope_ctx = execution_context
        if scope_ctx is None and runtime_cfg is not None:
            scope_ctx = getattr(runtime_cfg, "engine_context", None)
        identity = getattr(self, "_engine_identity", None)
        if not isinstance(identity, EngineIdentity):
            dialect_obj = self._dialect
            engine_type = str(getattr(dialect_obj, "name", getattr(self, "dialect", "")) or "")
            identity = EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
        identity_token = push_engine_identity(identity)
        limits_token = push_engine_limits(getattr(self, "limits", EngineLimits()))
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
                visible_objects=self._consumer_visible_objects,
                schema_role=self._schema_role,
                persist_template_learning=False,
            )
        finally:
            pop_engine_limits(limits_token)
            pop_engine_identity(identity_token)
        if as_dataframe:
            return pandas.DataFrame([list(r) for r in result.rows], columns=list(result.columns) or None)
        return result

    @classmethod
    def apply_migration_map(
        cls,
        path: str = "schema_migration_map.json",
        *,
        config_file: str | os.PathLike[str] | None = None,
        engine_context: EngineContext,
        artifacts_dir: str,
        execution_engine: Any = None,
        native_connection: Any = None,
        role: SchemaRole = SchemaRole.OWNER,
    ) -> AetherEngine:
        """Copy a validated migration map into the working directory and construct ``AetherEngine``."""
        if role != SchemaRole.OWNER:
            raise OwnerOnlyOperationError("apply_migration_map")
        src = Path(os.path.expanduser(str(path))).resolve()
        dst = Path(artifacts_dir) / MIGRATION_MAP_FILENAME
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return cls(
            engine_context,
            artifacts_dir=artifacts_dir,
            config_file=config_file,
            execution_engine=execution_engine,
            native_connection=native_connection,
            role=role,
        )

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
    def run_interactive(self, *, space: str = "master") -> None:
        """Prompt once for a natural-language question, resolve it through the interactive prompt cycle, then return. An empty line at the question prompt warns once; a second empty line terminates with ``User terminated.``. There is no outer REPL loop; call ``run_interactive`` again for another question."""
        self._ensure_llm()
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(space)
        exec_ctx = getattr(self._runtime_config, "execution_context", None)
        if exec_ctx is None:
            exec_ctx = self._runtime_config.engine_context
        validate_space_subset_of_execution_context(
            space_tables,
            space_columns,
            exec_ctx,
            self._schema_graph,
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
        self._require_production_api("run_seed_warmup")
        self._ensure_llm()
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
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
        self._require_production_api("run_qsim")
        self._ensure_llm()
        self._validate_num_intents(num_intents)
        self._validate_num_questions(num_questions)
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            owner_tok = push_qsim_engine_owner(self)
            try:
                qsim_run_once(
                    num_intents=num_intents,
                    num_questions=num_questions,
                    seed=seed,
                    artifacts_dir=str(self._artifacts_dir),
                    schema=self._schema_graph,
                )
            finally:
                pop_qsim_engine_owner(owner_tok)

    def get_questions_only(self, version: int) -> None:
        """Print NL questions from a QSim artifact and write them to ``qsim_v{version}_questions.txt`` in the process working directory."""
        path = resolve_qsim_path(version, str(self._artifacts_dir))
        if not os.path.isfile(path):
            raise ConfigError(f"QSim questions file not found for version {version}: {path}")
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            print_questions_bundle(version, str(self._artifacts_dir))

    def show_config(self) -> ConfigSnapshot:
        """Return a redacted snapshot of engine, schema scope, database, and LLM settings."""
        return ConfigSnapshot(text=describe_runtime_config(self._runtime_config, self._llm_config))

    @_writer_lock_guard
    def export_overrides(self) -> Path:
        """Write ``schema_overrides.json`` in the process working directory and return its path."""
        self._require_owner("export_overrides")
        target = self._artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        return dump_schema_overrides_to_path(
            self._schema_graph,
            target,
            business_knowledge=self.business_knowledge(),
        )

    def export_schema_overrides(self) -> Path:
        """Deprecated alias for :meth:`export_overrides`."""
        return cast(Path, self.export_overrides())

    @_writer_lock_guard
    def apply_overrides(self) -> None:
        """Apply ``schema_overrides.json`` from the working directory."""
        self._apply_schema_overrides_impl()

    def apply_schema_overrides(self) -> None:
        """Deprecated alias for :meth:`apply_overrides`."""
        self.apply_overrides()

    def _apply_schema_overrides_impl(self) -> None:
        """Apply ``schema_overrides.json`` from the working directory to the in-memory schema graph, re-stamp the cached graph artifact, print a summary, then rename editor JSON files to ``*.applied.json`` (archiving any prior applied copy)."""
        source = self._artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        if not source.is_file():
            raise ConfigError(f"schema overrides file not found: {source}")
        if self._schema_role == SchemaRole.CONSUMER:
            with open(source, encoding="utf-8") as fh:
                document = json.load(fh)
            ev = WriteQueueEvent(
                kind="override_proposal",
                schema_graph_id=self._schema_graph_id,
                schema_hash=str(self._schema_graph.effective_structural_hash or ""),
                produced_at=datetime.now(UTC).isoformat(),
                payload=(("document_json", json.dumps(document, ensure_ascii=False)),),
            )
            emit_write_queue_event(str(self._artifacts_dir), ev)
            return
        self._require_owner("apply_overrides")
        schema_json_path = str(self._artifacts_dir / "schema_graph.json.gz")
        report = apply_overrides_and_persist(
            self._schema_graph,
            source,
            schema_json_path=schema_json_path,
            dialect=self._dialect,
        )
        if report.business_knowledge_entries is not None:
            self.set_business_knowledge(report.business_knowledge_entries)
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
        _print_override_summary(report)
        editor = self._artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        companion = editor.with_name(editor.stem + ".schema.json")
        ts = datetime.now(UTC).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
        stem = Path(SCHEMA_OVERRIDES_DEFAULT_FILENAME).stem
        applied_main = editor.parent / f"{stem}{SCHEMA_OVERRIDES_APPLIED_SUFFIX}"
        applied_schema = editor.parent / f"{stem}.applied.schema.json"

        def _archive_and_rename(src: Path, dest: Path) -> None:
            if not src.is_file():
                return
            if dest.is_file():
                archive = dest.with_name(dest.stem + f".{ts}" + dest.suffix)
                try:
                    dest.rename(archive)
                except OSError:
                    pass
            try:
                src.rename(dest)
            except OSError:
                pass

        _archive_and_rename(editor, applied_main)
        _archive_and_rename(companion, applied_schema)
        sh = getattr(self._schema_graph, "effective_structural_hash", None)
        self._audit_emit(
            "apply_schema_overrides",
            schema_hash=str(sh) if sh is not None else None,
            details=(
                ("table_edits", str(report.table_edits)),
                ("column_edits", str(report.column_edits)),
            ),
        )

    @_writer_lock_guard
    def clear_persisted_overrides(self) -> bool:
        """Delete the persisted overrides sidecar and the cached schema graph, then rebuild the schema from catalog and inference layers. Returns True when a sidecar existed and was removed; False when no sidecar was present. After clearing, the schema graph is rebuilt from scratch (catalog reflection, profile-based PK inference, and FK inference) and the in-memory graph is published atomically; user-added FKs, user PK overrides, and the inference block lists are all discarded."""
        self._require_owner("clear_persisted_overrides")
        removed = clear_persisted_overrides(EngineConfig.SCHEMA_JSON_PATH)
        bundle = self._initialize_engine_bundle(
            self._single_engine_context(),
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_persisted_overrides",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("removed", str(removed)),),
        )
        return removed

    @_writer_lock_guard
    def clear_template_store(self) -> bool:
        """Remove persisted templates and question feedback, then reload engine initialization state."""
        self._require_owner("clear_template_store")
        existed = clear_template_store_only(str(self._artifacts_dir), self._schema_graph)
        bundle = self._initialize_engine_bundle(
            self._single_engine_context(),
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_template_store",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("existed", str(existed)),),
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
    def clear_all_learning(self, *, keep_overrides: bool = True) -> None:
        """Remove templates, simulation caches, and optionally persisted schema overrides, then reload."""
        self._require_owner("clear_all_learning")
        clear_template_store_only(str(self._artifacts_dir), self._schema_graph)
        clear_simulation_caches_only(str(self._artifacts_dir))
        if not keep_overrides:
            clear_persisted_overrides(EngineConfig.SCHEMA_JSON_PATH)
        bundle = self._initialize_engine_bundle(
            self._single_engine_context(),
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_all_learning",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None) or None,
            details=(("keep_overrides", str(keep_overrides)),),
        )

    @classmethod
    def offline_sandbox(
        cls,
        *,
        artifacts_dir: str | None = None,
        cleanup_artifacts: bool = True,
        deny_columns: frozenset[str] | None = None,
        include: SchemaInclude = SchemaInclude.TABLES,
        engine_context: EngineContext | None = None,
        notes_file: str | None = None,
        sql_file: str | None = None,
        llm_config: str | os.PathLike[str] | None = None,
        maintainer_access: bool = False,
        seed_sql: str | None = None,
        bundle_dir: str | None = None,
        connection: Any | None = None,
        owns_connection: bool | None = None,
    ) -> SandboxHandle:
        """Enter the offline practice environment (in-memory DuckDB and mock LLM fixtures). Pass ``include="views"`` to reflect bundled analytical views instead of base tables."""
        return Sandbox.create_offline_sandbox(
            cls,
            artifacts_dir=artifacts_dir,
            cleanup_artifacts=cleanup_artifacts,
            deny_columns=deny_columns,
            include=include,
            engine_context=engine_context,
            notes_file=notes_file,
            sql_file=sql_file,
            llm_config=llm_config,
            maintainer_access=maintainer_access,
            seed_sql=seed_sql,
            bundle_dir=bundle_dir,
            connection=connection,
            owns_connection=owns_connection,
        )

    @classmethod
    def sandbox_questions(cls) -> list[str]:
        """Return curated natural-language sandbox practice questions."""
        return Sandbox.sandbox_questions()

    @classmethod
    def sandbox_doctor(cls) -> list[str]:
        """Return human-readable problems; empty list means the sandbox bundle looks healthy."""
        return Sandbox.sandbox_doctor()

    @classmethod
    def sandbox_catalog(cls) -> dict[str, object]:
        """Return bundled sandbox discovery metadata (paraphrase pairs, demos)."""
        return Sandbox.sandbox_catalog()

    @classmethod
    def sandbox_paraphrase_pairs(cls) -> list[dict[str, object]]:
        """Return canonical→paraphrase wordings from the bundled sandbox catalog."""
        return Sandbox.sandbox_paraphrase_pairs()

    @classmethod
    def sandbox_validation_failure_demo(cls) -> list[dict[str, str]]:
        """Return example validation-failure questions and short descriptions."""
        return Sandbox.sandbox_validation_failure_demo()

    @classmethod
    def sandbox_feedback_demo(cls) -> dict[str, str]:
        """Return the scripted reject/retry feedback demo."""
        return Sandbox.sandbox_feedback_demo()

    @classmethod
    def assert_sandbox_complete(cls) -> None:
        """Validate the shipped sandbox corpus and raise when any slot fails."""
        Sandbox.assert_sandbox_complete(cls)


class AetherFederation:
    """Federated scope over named member engines with a composed schema graph."""

    _is_aether_federation = True

    __slots__ = (
        "_name",
        "_members",
        "_declaration_file",
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
        "_construction_phase_callback",
        "_ask_phase_callback",
        "_pipeline_writer_lock",
        "_schema_role",
        "_consumer_visible_objects",
        "_context_name",
        "_sandbox_mode",
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
        "_business_knowledge",
        "_tenant_slug",
        "_limits",
        "_federation_owned_members",
    )

    def __init__(
        self,
        name: str,
        *,
        members: Mapping[str, AetherEngine],
        declaration_file: str,
        context: FederationContext | None = None,
        artifacts_dir: str | None = None,
        tenant_slug: str | None = None,
        role: SchemaRole = SchemaRole.OWNER,
        limits: FederationLimits | None = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
        construction_phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
        ask_phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
    ) -> None:
        """Construct a federation over named member engines. The ``members`` mapping keys are federation ``source_id`` values: the names used in cross-source joins, logical tables, logical columns, aliases, diagnostics, and ``export_overrides(source_id)``. Choose these names to match your declaration. They are not the TOML ``connection=`` sub- block on each member engine, which only selects credentials and artifact slugs for that member's database."""
        self._name = str(name).strip()
        if not self._name:
            raise ConfigError("AetherFederation name must be non-empty")
        try:
            require_driver("duckdb")
        except ConfigError as exc:
            raise ConfigError("pip install aetherdialect[federation]") from exc
        if not members:
            raise ConfigError("AetherFederation requires at least one member engine")
        self._members = dict(members)
        self._declaration_file = str(declaration_file)
        self._master_context = context
        self._mappings: FederationMappings | None = None
        self._artifacts_root = Path(artifacts_dir) if artifacts_dir is not None else None
        self._tenant_slug = str(tenant_slug).strip() if tenant_slug is not None and str(tenant_slug).strip() else None
        self._audit_sink = audit_sink
        self._construction_phase_callback = construction_phase_callback
        self._ask_phase_callback = ask_phase_callback
        self._pipeline_writer_lock = threading.Lock()
        self._schema_role = role
        self._consumer_visible_objects: frozenset[str] | None = None
        self._context_name = MASTER_AETHERSPACE_NAME
        self._sandbox_mode = False
        self._sandbox_closed = False
        self._closed = False
        self._limits = limits if limits is not None else FederationLimits()
        self._federation_owned_members: set[str] = set()
        validate_federation_pool_capacity(self._members, self._limits)
        construction_token = push_construction_phase_callback(construction_phase_callback)
        try:
            bundle = initialize_aether_federation(
                self._name,
                members=self._members,
                declaration_file=self._declaration_file,
                artifacts_dir=str(self._artifacts_root) if self._artifacts_root is not None else None,
                tenant_slug=self._tenant_slug,
                schema_role=role,
                master_context=self._master_context,
                log_sink=_init_log_sink,
            )
        finally:
            pop_construction_phase_callback(construction_token)
        self._apply_init_bundle(bundle)
        self._business_knowledge = BusinessKnowledgeHolder()
        self._ingest_notes_business_knowledge()
        if bundle.members is not None:
            self._members = dict(bundle.members)
        if bundle.federation_mappings is not None:
            self._mappings = bundle.federation_mappings
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
    ) -> PersistedFederationInspection:
        """Load declaration and roster from a persisted ``fed_<id>`` tree without member engines."""
        return inspect_persisted_federation(artifacts_dir, federation_id)

    def _recompose(self) -> None:
        manifest_decl, file_mappings = load_federation_declaration_from_path(self._declaration_file)
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
            declaration_file=self._declaration_file,
            declaration=(manifest_decl, mappings),
            artifacts_dir=str(self._artifacts_root) if self._artifacts_root is not None else None,
            tenant_slug=self._tenant_slug,
            schema_role=self._schema_role,
            master_context=self._master_context,
            log_sink=_init_log_sink,
        )
        self._apply_init_bundle(bundle)
        if bundle.members is not None:
            self._members = dict(bundle.members)
        if bundle.federation_mappings is not None:
            self._mappings = bundle.federation_mappings
        self._replay_composite_overrides()
        if self._federation_manifest is not None and self._mappings is not None:
            export_federation_declaration(
                self._federation_manifest,
                self._mappings,
                self._declaration_file,
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
    def add_engine(self, connection_name: str, engine: AetherEngine) -> None:
        """Register a member engine, recompose the composite graph, and persist the federation tree."""
        self._require_owner("add_engine")
        self._require_open("add_engine")
        key = str(connection_name).strip()
        if not key:
            raise ConfigError("connection_name must be non-empty")
        self._members[key] = engine
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
            binding = binding_from_member_engine(key, cast(FederationMemberEngine, member_engine))
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

    @_writer_lock_guard
    def export_federation_declaration(self) -> Path:
        """Write ``federation_declaration.json`` in the working directory (authored shape)."""
        self._require_owner("export_federation_declaration")
        manifest = self._federation_manifest
        if manifest is None:
            raise ConfigError("federation manifest not loaded")
        mappings = self._mappings or self._federation_mappings
        if mappings is None:
            mappings = FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        target = self._artifacts_dir / FEDERATION_DECLARATION_FILENAME
        export_federation_declaration(manifest, mappings, target)
        return target

    @_writer_lock_guard
    def apply_federation_declaration(self) -> None:
        """Apply the authored federation declaration from the working directory and recompose."""
        self._require_owner("apply_federation_declaration")
        source = self._artifacts_dir / FEDERATION_DECLARATION_FILENAME
        if not source.is_file():
            raise ConfigError(f"federation declaration file not found: {source}")
        try:
            manifest, mappings = parse_federation_declaration(source.read_text(encoding="utf-8"))
        except FederationConfigError as exc:
            raise FederationConfigError(
                f"malformed federation declaration in declarations file {source!r}: {exc}"
            ) from exc
        export_federation_declaration(manifest, mappings, self._declaration_file)
        self._mappings = mappings
        archive_federation_editor_file(str(source))
        self._recompose()

    @_writer_lock_guard
    def apply_migration_map(self, path: str = "federation_migration_map.json") -> None:
        """Copy ``federation_migration_map.json`` into the working directory and recompose."""
        self._require_owner("apply_migration_map")
        self._require_open("apply_migration_map")
        src = Path(os.path.expanduser(str(path))).resolve()
        dst = self._artifacts_dir / FEDERATION_MIGRATION_MAP_FILENAME
        shutil.copyfile(src, dst)
        self._recompose()

    def _require_owner(self, operation: str) -> None:
        if self._schema_role != SchemaRole.OWNER:
            raise OwnerOnlyOperationError(operation)

    def _require_master_context(self, operation: str) -> None:
        if getattr(self, "_context_name", MASTER_AETHERSPACE_NAME) != MASTER_AETHERSPACE_NAME:
            raise ConfigError(
                f"Operation {operation!r} requires the master engine context; "
                "this instance is bound to a non-master context.",
            )

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
        return member

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
    ) -> None:
        sink = self._audit_sink
        if sink is None:
            return
        ev = AuditEvent(
            event_type=event_type,
            timestamp_iso=datetime.now(UTC).isoformat(),
            question=question,
            schema_hash=schema_hash,
            provider=self._llm_config.provider,
            details=details,
        )
        sink(ev)

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
        self._context_name = getattr(bundle, "context_name", MASTER_AETHERSPACE_NAME)
        self._federation_manifest = getattr(bundle, "federation_manifest", None)
        self._federation_mappings = getattr(bundle, "federation_mappings", None)
        self._federation_member_graphs = getattr(bundle, "federation_member_graphs", None)
        self._federation_storage_dir = getattr(bundle, "federation_storage_dir", None)
        self._federation_dialects = getattr(bundle, "federation_dialects_by_source", None)
        self._federation_source_runtimes = getattr(bundle, "federation_source_runtimes", None)
        self._federation_mapping_suggestions = getattr(bundle, "federation_mapping_suggestions", ())
        self._engine_identity = getattr(bundle, "engine_identity", None)
        register_engine_skeleton_cache_owner(self)
        clear_engine_skeleton_cache(self)

    @property
    def dialect(self) -> str:
        return str(self._runtime_config.engine)

    def set_business_knowledge(self, entries: Sequence[BusinessKnowledgeEntry]) -> int:
        """Replace prompt-time business knowledge and return the new monotonic version."""
        holder = getattr(self, "_business_knowledge", None)
        if holder is None:
            self._business_knowledge = BusinessKnowledgeHolder()
            holder = self._business_knowledge
        return holder.set(entries, self._schema_graph)

    def _ingest_notes_business_knowledge(self) -> None:
        """Pass B: extract business knowledge from federation construction notes when present."""
        ctx = getattr(self, "_master_context", None)
        notes_path = getattr(ctx, "notes_file", None) if ctx is not None else None
        notes_inline = getattr(ctx, "notes", None) if ctx is not None else None
        notes_content: str | None = None
        if notes_inline is not None and str(notes_inline).strip():
            notes_content = str(notes_inline)
        elif notes_path is not None and str(notes_path).strip():
            path = os.path.expanduser(str(notes_path).strip())
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    notes_content = fh.read()
        entries = extract_business_knowledge_from_notes(notes_content, self._schema_graph)
        if entries:
            self.set_business_knowledge(entries)

    def business_knowledge(self) -> tuple[BusinessKnowledgeEntry, ...]:
        """Return the active business knowledge entries."""
        holder = getattr(self, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            return ()
        return holder.entries()

    def business_knowledge_digest(self) -> str:
        """Return a stable digest of the active business knowledge."""
        holder = getattr(self, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            return ""
        return holder.digest()

    def business_knowledge_version(self) -> int:
        """Return the monotonic version counter for business knowledge updates."""
        holder = getattr(self, "_business_knowledge", None)
        if not isinstance(holder, BusinessKnowledgeHolder):
            return 0
        return holder.version()

    def export_space_knowledge(self, space: str | None = None) -> dict[str, Any]:
        """Return business knowledge for master or one named space (key/kind/text only)."""
        space_snapshot = None
        if space is not None:
            norm = str(space).strip().lower()
            if norm and norm != MASTER_AETHERSPACE_NAME:
                space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
        return MainExecutionOps.build_space_knowledge_export(
            engine_entries=self.business_knowledge(),
            business_knowledge_version=self.business_knowledge_version(),
            space=space,
            space_snapshot=space_snapshot,
        )

    def export_knowledge(self) -> dict[str, Any]:
        """Return engine plus per-space business knowledge wrapper."""
        snapshots: dict[str, Any] = {}
        for name in list_saved_aetherspace_names(str(self._artifacts_dir)):
            snap = load_aetherspace_snapshot(str(self._artifacts_dir), name)
            if snap is not None:
                snapshots[name] = snap
        return MainExecutionOps.build_knowledge_export(
            engine_entries=self.business_knowledge(),
            business_knowledge_version=self.business_knowledge_version(),
            space_snapshots=snapshots,
        )

    def export_metadata(self, space: str | None = None) -> dict[str, Any]:
        """Return deterministic table/column inventory for the composite graph."""
        space_snapshot = None
        if space is not None:
            norm = str(space).strip().lower()
            if norm and norm != MASTER_AETHERSPACE_NAME:
                space_snapshot = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
        members = getattr(self, "_member_table_roster", None)
        return MainExecutionOps.build_metadata_export(
            schema_graph=self._schema_graph,
            space=space,
            space_snapshot=space_snapshot,
            federation_members=members,
        )

    def _resolve_aetherspace(
        self, name: str
    ) -> tuple[AetherSpace, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("aetherspace name must be non-empty")
        if norm == MASTER_AETHERSPACE_NAME:
            desc = build_master_space_descriptor(self._schema_graph)
            return desc, frozenset(), frozenset(), frozenset(), frozenset()
        snap = load_aetherspace_snapshot(str(self._artifacts_dir), norm)
        if snap is None:
            raise ConfigError(f"unknown aetherspace {name!r}")
        tables_raw = snap.get("tables")
        if isinstance(tables_raw, (list, tuple)) and len(tables_raw) == 0:
            raise ConfigError("space empty after schema migration; redefine")
        desc = aetherspace_descriptor_from_snapshot(norm, snap)
        tables, columns = space_allowed_sets_from_snapshot(snap)
        deny_objects, deny_columns = space_deny_sets_from_snapshot(snap)
        mappings = self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
        collapsed = collapsed_member_physical_table_names(mappings)
        for table_name in tables | deny_objects:
            logical = collapsed.get(table_name)
            if logical is not None:
                raise ConfigError(
                    f"aetherspace {name!r} names collapsed member table {table_name!r}; "
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
        return desc, tables, columns, deny_objects, deny_columns

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
        space: str = "master",
        ephemeral_scope: SpaceContext | None = None,
        data_row_cap: int | None = None,
    ) -> PipelineSession:
        self._require_open("session")
        if self._sandbox_closed:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        if self._schema_role == SchemaRole.CONSUMER and mode == "writer":
            raise OwnerOnlyOperationError("PipelineSession(mode='writer')")
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(space)
        space_tables, space_columns, space_deny_objects, space_deny_columns = intersect_space_scope(
            space_tables,
            space_columns,
            space_deny_objects,
            space_deny_columns,
            ephemeral_scope,
        )
        space_description_overlay: dict[str, Any] | None = None
        norm = str(space).strip().lower()
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
        exec_ctx = getattr(self._runtime_config, "execution_context", None)
        if exec_ctx is None:
            exec_ctx = self._runtime_config.engine_context
        validate_space_subset_of_execution_context(
            space_tables,
            space_columns,
            exec_ctx,
            self._schema_graph,
        )
        payload_visible = space_tables if space_tables else None
        return PipelineSession(
            self,
            mode=mode,
            visible_objects=payload_visible,
            execution_visible_objects=self._consumer_visible_objects,
            space_name=str(space).strip().lower(),
            space_tables=space_tables,
            space_columns=space_columns,
            space_deny_objects=space_deny_objects,
            space_deny_columns=space_deny_columns,
            space_description_overlay=space_description_overlay,
            data_row_cap=data_row_cap,
        )

    def list_templates(self, *, space: str = "master") -> tuple[StoredTemplateSummary, ...]:
        """Enumerate caller-visible federation plan templates."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        del space
        return list_stored_template_summaries(self._templates, space="master", dialect=self._dialect)

    def fetch_template(self, template_ref: str, *, space: str = "master") -> StoredTemplateDetail:
        """Fetch one federation plan template by id or ``sql_fp`` hash."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        del space
        tmpl = resolve_template_ref(template_ref, self._templates)
        if tmpl is None or not template_visible_to_callers(tmpl):
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
        )

    @_writer_lock_guard
    def execute_template(
        self,
        template_ref: str,
        params: dict[str, Any] | None = None,
        *,
        question: str | None = None,
        space: str = "master",
        as_dataframe: bool = False,
    ) -> TemplateExecutionResult | pandas.DataFrame:
        """Execute one federation-stored template by id or ``sql_fp`` with p-param binds."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dict or None")
        del space
        bind = dict(params or ())
        runtime_cfg = getattr(self, "_runtime_config", None)
        execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
        scope_ctx = execution_context
        if scope_ctx is None and runtime_cfg is not None:
            scope_ctx = getattr(runtime_cfg, "engine_context", None)
        identity = getattr(self, "_engine_identity", None)
        if not isinstance(identity, EngineIdentity):
            dialect_obj = self._dialect
            engine_type = str(getattr(dialect_obj, "name", "duckdb") or "duckdb")
            identity = EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
        identity_token = push_engine_identity(identity)
        limits_token = push_engine_limits(getattr(self, "_limits", EngineLimits()))
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
                visible_objects=self._consumer_visible_objects,
                schema_role=self._schema_role,
                persist_template_learning=False,
            )
        finally:
            pop_engine_limits(limits_token)
            pop_engine_identity(identity_token)
        if as_dataframe:
            return pandas.DataFrame([list(r) for r in result.rows], columns=list(result.columns) or None)
        return result

    def asession(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str = "master",
    ) -> AsyncPipelineSession:
        return AsyncPipelineSession(self.session(mode=mode, space=space))

    @_writer_lock_guard
    def aetherspace(
        self,
        name: str,
        space_context: SpaceContext | None = None,
        *,
        notes_file: str | None = None,
        notes: str | None = None,
    ) -> AetherSpace:
        """Check or define a named aetherspace scope snapshot on the composite graph."""
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("aetherspace name must be non-empty")
        if space_context is None:
            return self._resolve_aetherspace(norm)[0]
        self._require_owner("aetherspace")
        if norm == MASTER_AETHERSPACE_NAME:
            raise ConfigError(
                "master is the implicit full-scope space; it cannot be created or overwritten",
            )
        if notes_file is not None and notes is not None:
            raise ConfigError("set at most one of notes and notes_file")
        validated = validate_space_context_against_graph(
            space_context,
            self._schema_graph,
            federation_manifest=self._federation_manifest,
        )
        mappings = self._federation_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
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
        if notes_file is not None:
            notes_path, notes_inline = notes_file, None
        elif notes is not None:
            notes_path, notes_inline = None, notes
        else:
            notes_path, notes_inline = validated.notes_file, validated.notes
        if notes_inline is not None and str(notes_inline).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes=str(notes_inline),
            )
        elif notes_path is not None and str(notes_path).strip():
            if getattr(self, "_sandbox_mode", False):
                connection = getattr(self, "_native_connection", None)
                host = Sandbox.sandbox_host_for_connection(connection) if connection is not None else None
                if host is not None:
                    Sandbox.validate_sandbox_aetherspace_notes_pairing(
                        validated,
                        notes_path,
                        extract_path=host._extract_path,
                    )
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes_path,
            )
        save_aetherspace_snapshot(str(self._artifacts_dir), norm, snapshot)
        return aetherspace_descriptor_from_snapshot(norm, snapshot)

    @_writer_lock_guard
    def export_aetherspace(self, name: str) -> Path:
        """Export a JSON snapshot of one named aetherspace for review or apply."""
        self._require_master_context("export_aetherspace")
        norm = str(name).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME and load_aetherspace_snapshot(str(self._artifacts_dir), norm) is None:
            raise ConfigError(f"unknown aetherspace {name!r}")
        return export_aetherspace_json(str(self._artifacts_dir), norm, self._schema_graph)

    @_writer_lock_guard
    def apply_aetherspace(self, name: str, *, source: str | os.PathLike[str] | None = None) -> AetherSpace:
        """Apply an exported aetherspace JSON document and persist it under *name*."""
        self._require_owner("apply_aetherspace")
        self._require_master_context("apply_aetherspace")
        norm = str(name).strip().lower()
        return apply_aetherspace_json(
            str(self._artifacts_dir),
            norm,
            self._schema_graph,
            source=source,
            federation_manifest=self._federation_manifest,
        )

    @_writer_lock_guard
    def delete_aetherspace(self, name: str) -> bool:
        """Delete one persisted named aetherspace snapshot."""
        self._require_owner("delete_aetherspace")
        self._require_master_context("delete_aetherspace")
        norm = str(name).strip().lower()
        return delete_aetherspace_snapshot(str(self._artifacts_dir), norm)

    def list_aetherspaces(self) -> tuple[str, ...]:
        """Return saved aetherspace names plus the implicit ``master`` space."""
        self._require_master_context("list_aetherspaces")
        saved = list_saved_aetherspace_names(str(self._artifacts_dir))
        return (MASTER_AETHERSPACE_NAME,) + saved

    def export_context(self, name: str) -> Path:
        """Export a read-only JSON snapshot of one named federation context preset."""
        self._require_master_context("export_context")
        norm = str(name).strip().lower()
        master_ctx = self._runtime_config.engine_context
        if not isinstance(master_ctx, FederationContext):
            raise ConfigError("export_context on AetherFederation requires a FederationContext master")
        path = Path(str(self._artifacts_dir)) / f"federation_context.{norm}.json"
        if norm != MASTER_AETHERSPACE_NAME and not path.is_file():
            if load_named_schema_context(str(self._artifacts_dir), norm) is None:
                raise ConfigError(f"unknown federation context {name!r}")
        payload = {
            "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
            "name": norm,
            "context": {
                "allow_objects": sorted(master_ctx.allow_objects),
                "include": str(master_ctx.include),
                "deny_objects": sorted(master_ctx.deny_objects),
                "deny_columns": sorted(master_ctx.deny_columns),
                "allow_columns": sorted(getattr(master_ctx, "allow_columns", frozenset())),
                "notes_file": master_ctx.notes_file,
                "notes": getattr(master_ctx, "notes", None),
            },
        }
        path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def list_contexts(self) -> tuple[str, ...]:
        """Return saved federation-context names plus the implicit ``master`` context."""
        self._require_master_context("list_contexts")
        root = Path(str(self._artifacts_dir))
        names = []
        for p in sorted(root.glob("federation_context.*.json")):
            stem = p.name[len("federation_context.") : -len(".json")]
            if stem and stem != MASTER_AETHERSPACE_NAME:
                names.append(stem)
        return (MASTER_AETHERSPACE_NAME,) + tuple(names)

    def prepared_federated_outcome(self) -> FederatedPrepareOutcome | None:
        """Return the staged federated prepare outcome from an in-flight turn, if any."""
        return None

    def preview_table(self, table_name: str, *, limit: int = TABLE_PREVIEW_DEFAULT_LIMIT) -> TablePreviewResult:
        """Return the first rows of a composite table through federation scope and sensitivity gates."""
        self._require_open("preview_table")
        return preview_table_on_federation(self, table_name, limit=limit)

    def mapping_suggestions(self) -> tuple[FederationMappingSuggestion, ...]:
        """Return cross-source mapping suggestions computed at federation composition time."""
        self._require_open("mapping_suggestions")
        return tuple(getattr(self, "_federation_mapping_suggestions", ()) or ())

    def preview_plan(self, question: str, *, space: str = "master") -> PlanPreviewResult:
        """Return what a federated turn would run for *question* without executing SQL."""
        self._require_open("preview_plan")
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(space)
        space_ctx = None
        if space_tables or space_columns or space_deny_objects or space_deny_columns:
            space_ctx = SpaceContext(
                tables=space_tables,
                columns=space_columns,
                deny_objects=space_deny_objects,
                deny_columns=space_deny_columns,
            )
        space_description_overlay: dict[str, Any] | None = None
        norm = str(space).strip().lower()
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
        return preview_plan_on_federation(
            self,
            question,
            space=space_ctx,
            visible_objects=payload_visible,
            execution_visible_objects=self._consumer_visible_objects,
            space_columns=space_columns,
            space_deny_objects=space_deny_objects,
            space_deny_columns=space_deny_columns,
            space_description_overlay=space_description_overlay,
        )

    @_writer_lock_guard
    def run_interactive(self, *, space: str = "master") -> None:
        self._ensure_llm()
        _, space_tables, space_columns, space_deny_objects, space_deny_columns = self._resolve_aetherspace(space)
        space_description_overlay: dict[str, Any] | None = None
        norm = str(space).strip().lower()
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
        exec_ctx = getattr(self._runtime_config, "execution_context", None)
        if exec_ctx is None:
            exec_ctx = self._runtime_config.engine_context
        validate_space_subset_of_execution_context(
            space_tables,
            space_columns,
            exec_ctx,
            self._schema_graph,
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

    def show_config(self) -> ConfigSnapshot:
        """Return a redacted snapshot of federation topology and LLM settings."""
        return ConfigSnapshot(
            text=describe_federation_config(
                self._name,
                self._runtime_config,
                self._llm_config,
                members=self._members,
                federation_storage_dir=str(self._federation_storage_dir or self._artifacts_dir),
            ),
        )

    def export_overrides(self, connection_name: str | None = None) -> Path:
        """Member-scoped override export."""
        return self._export_schema_overrides_impl(connection_name)

    def export_schema_overrides(self, connection_name: str | None = None) -> Path:
        """Deprecated alias for :meth:`export_overrides`."""
        return self.export_overrides(connection_name)

    def _export_schema_overrides_impl(self, connection_name: str | None = None) -> Path:
        """Export schema overrides for the composite graph or one member engine."""
        if connection_name is not None:
            member = self._resolve_member(connection_name, "export_overrides")
            return cast(Path, member.export_overrides())
        self._require_owner("export_overrides")
        target = self._artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        return export_federation_composite_overrides(
            self._schema_graph,
            target,
            business_knowledge=self.business_knowledge(),
        )

    @_writer_lock_guard
    def apply_overrides(self, connection_name: str | None = None) -> None:
        """Apply schema overrides to the composite graph or one member engine."""
        self._apply_schema_overrides_impl(connection_name)

    def apply_schema_overrides(self, connection_name: str | None = None) -> None:
        """Deprecated alias for :meth:`apply_overrides`."""
        self.apply_overrides(connection_name)

    def _apply_schema_overrides_impl(self, connection_name: str | None = None) -> None:
        """Apply schema overrides to the composite graph or one member engine."""
        if connection_name is not None:
            self._require_owner("apply_overrides")
            member = self._resolve_member(connection_name, "apply_overrides")
            member.apply_overrides()
            self._recompose()
            return
        source = self._artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        if not source.is_file():
            raise ConfigError(f"schema overrides file not found: {source}")
        self._require_owner("apply_overrides")
        report = apply_federation_composite_overrides(
            self._schema_graph,
            self._composite_federation_dir(),
            source,
            dialect=self._dialect,
        )
        if report.business_knowledge_entries is not None:
            self.set_business_knowledge(report.business_knowledge_entries)
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
        _print_override_summary(report)
        editor = self._artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        companion = editor.with_name(editor.stem + ".schema.json")
        ts = datetime.now(UTC).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
        stem = Path(SCHEMA_OVERRIDES_DEFAULT_FILENAME).stem
        applied_main = editor.parent / f"{stem}{SCHEMA_OVERRIDES_APPLIED_SUFFIX}"
        applied_schema = editor.parent / f"{stem}.applied.schema.json"

        def _archive_and_rename(src: Path, dest: Path) -> None:
            if not src.is_file():
                return
            if dest.is_file():
                archive = dest.with_name(dest.stem + f".{ts}" + dest.suffix)
                try:
                    dest.rename(archive)
                except OSError:
                    pass
            try:
                src.rename(dest)
            except OSError:
                pass

        _archive_and_rename(editor, applied_main)
        _archive_and_rename(companion, applied_schema)
        sh = getattr(self._schema_graph, "effective_structural_hash", None)
        self._audit_emit(
            "apply_schema_overrides",
            schema_hash=str(sh) if sh is not None else None,
            details=(
                ("scope", "composite"),
                ("table_edits", str(report.table_edits)),
                ("column_edits", str(report.column_edits)),
            ),
        )

    @_writer_lock_guard
    def clear_persisted_overrides(self, connection_name: str) -> bool:
        """Clear persisted overrides for one member engine, then recompose."""
        self._require_owner("clear_persisted_overrides")
        member = self._resolve_member(connection_name, "clear_persisted_overrides")
        removed = member.clear_persisted_overrides()
        self._recompose()
        self._audit_emit(
            "clear_persisted_overrides",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("connection", connection_name), ("removed", str(removed))),
        )
        return cast(bool, removed)

    @_writer_lock_guard
    def clear_template_store(self) -> bool:
        """Remove composite, plan-record, and member template stores, then recompose."""
        self._require_owner("clear_template_store")
        drain_write_queue(self, str(self._artifacts_dir))
        existed = clear_federation_template_stores(
            str(self._federation_storage_dir) if self._federation_storage_dir else None,
            str(self._artifacts_dir),
            self._schema_graph,
            self._members,
        )
        self._recompose()
        self._audit_emit(
            "clear_template_store",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None),
            details=(("existed", str(existed)),),
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
    def clear_all_learning(self, *, keep_overrides: bool = True) -> None:
        """Remove templates, simulation caches, and optionally member overrides, then recompose."""
        self._require_owner("clear_all_learning")
        drain_write_queue(self, str(self._artifacts_dir))
        clear_federation_template_stores(
            str(self._federation_storage_dir) if self._federation_storage_dir else None,
            str(self._artifacts_dir),
            self._schema_graph,
            self._members,
        )
        count = clear_simulation_caches_only(str(self._artifacts_dir))
        for engine in self._members.values():
            adir = getattr(engine, "_artifacts_dir", None)
            if adir is not None:
                count += clear_simulation_caches_only(str(adir))
        if not keep_overrides:
            for connection_name in self._members:
                self._resolve_member(connection_name, "clear_all_learning").clear_persisted_overrides()
        self._recompose()
        self._audit_emit(
            "clear_all_learning",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None) or None,
            details=(("keep_overrides", str(keep_overrides)), ("removed_files", str(count))),
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
        return RefreshReport(
            migration_tier=migration_tier,
            schema_changed=any(report.schema_changed for report in member_reports),
            objects_added=tuple(sorted({obj for report in member_reports for obj in report.objects_added})),
            objects_removed=tuple(sorted({obj for report in member_reports for obj in report.objects_removed})),
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
        for connection_name in sorted(self._federation_owned_members):
            member = self._members.get(connection_name)
            if member is not None and not getattr(member, "_closed", False):
                member.close()
        release_close_resources(self)
        drop_engine_skeleton_cache_owner(self)
        clear_expansion_subtree_pool(str(self._artifacts_dir))
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


def _print_override_summary(report: OverrideReport) -> None:
    """Emit a fixed-template summary of an ``OverrideReport`` through the notify channel."""
    notify("Schema overrides applied:", stage="overrides", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    notify(
        f"  Tables updated:           {report.table_edits}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Columns updated:          {report.column_edits}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  FK edges added:           {report.fks_added}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  FK edges endorsed (user): {report.fks_endorsed}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  FK edges removed:         {report.fks_removed}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  PKs endorsed (user):     {report.pks_endorsed}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Inferred PKs cleared:     {report.pks_blocked}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  PK/FK roles coerced:      {report.coerced_columns}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Redundant inferences:     {report.collapsed_inferences}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    notify(
        f"  Descriptions refined:     {report.descriptions_refined}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    if report.skipped:
        notify(
            f"  Soft skips ({len(report.skipped)}):",
            stage="overrides",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )
        for skip in report.skipped:
            notify(
                f"    {skip.path}  -  {skip.reason}",
                stage="overrides",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                details=(("path", skip.path), ("reason", skip.reason)),
            )
    else:
        notify(
            "  Soft skips:               none",
            stage="overrides",
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
    BusinessKnowledgeEntry,
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
    PlanPreviewResult,
    ProbeCtePlacementError,
    QSimSummarySnapshot,
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
    SessionStep,
    SessionTurnCancelledError,
    SpaceContext,
    StatementTimeoutError,
    SubdayDateWindowOnDateColumnError,
    SuspendedSessionExpiredError,
    TablePreviewResult,
    UploadIngestResult,
    __version__,
    inspect_tabular_upload,
)
