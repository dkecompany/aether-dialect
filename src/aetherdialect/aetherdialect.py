"""Public AetherEngine facade delegating construction and runners to main_execution. Attributes on AetherEngine whose names start with a single underscore are private implementation details and are not part of the public stability contract."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import pandas

from ._config import (
    ConfigError,
    EngineConfig,
    SeedWarmupConfig,
    llm_credentials_configured,
)
from ._constants import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    MASTER_AETHERSPACE_NAME,
    MIGRATION_MAP_FILENAME,
    PERMISSION_DENIED_USER_MESSAGE,
    SCHEMA_OVERRIDES_APPLIED_SUFFIX,
    SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT,
    SCHEMA_OVERRIDES_DEFAULT_FILENAME,
    WRITE_QUEUE_FILENAME,
    YES_NO_SESSION_KINDS,
)
from ._contracts_base import (
    AetherSpace,
    AuditEvent,
    ConfigSnapshot,
    ConnectionError,
    DatabasePingFailed,
    Diagnostic,
    EngineContext,
    LlmTransientFailure,
    MigrationPendingError,
    MigrationPreview,
    OverrideReport,
    OverrideSkip,
    OwnerOnlyOperationError,
    QSimSummarySnapshot,
    RetryableError,
    SchemaAccessError,
    SchemaRole,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    SessionActiveError,
    SessionStep,
    SpaceContext,
    StatementTimeoutError,
    WriteQueueEvent,
)
from ._core_utils import (
    dataframe_to_row_tuples,
    diagnostic_print_listener,
    echo_user_text,
    echo_yes_no_answer,
    emit_write_queue_event,
    error,
    notify,
    print_query_result,
    progress_enabled,
    terminated,
)
from ._llm_provider import MockFixtureMissingError
from ._main_execution import (
    AetherEngineInitResult,
    PipelineSession,
    aetherspace_descriptor_from_snapshot,
    build_master_space_descriptor,
    clear_simulation_caches_only,
    clear_template_store_only,
    describe_runtime_config,
    enrich_space_snapshot_with_notes,
    export_aetherspace_json,
    export_named_schema_context_json,
    find_latest_seed_warmup_summary,
    format_qsim_summary_line,
    format_seed_warmup_summary,
    initialize_aether_engine,
    list_named_schema_context_names,
    list_saved_aetherspace_names,
    load_aetherspace_snapshot,
    load_named_schema_context,
    load_qsim_summaries,
    print_questions_bundle,
    qsim_run_once,
    resolve_qsim_path,
    run_seed_warmup_from_history_execution,
    run_seed_warmup_from_query_log_execution,
    save_aetherspace_snapshot,
    seed_warmup_run_once,
    space_allowed_sets_from_snapshot,
    space_deny_sets_from_snapshot,
    subset_graph_for_space,
    validate_space_context_against_graph,
    validate_space_subset_of_execution_context,
)
from ._sandbox import (
    SandboxHandle,
    assert_sandbox_complete,
    create_offline_sandbox,
    sandbox_catalog,
    sandbox_doctor,
    sandbox_feedback_demo,
    sandbox_paraphrase_pairs,
    sandbox_questions,
    sandbox_validation_failure_demo,
)
from ._schema_overrides import (
    apply_overrides_and_persist,
    clear_persisted_overrides,
    dump_schema_overrides_to_path,
)
from ._validation_execute import execute_guarded_sql


def _init_log_sink(line: str) -> None:
    notify(line, stage="init", code=DIAGNOSTIC_CODE_ENGINE_INFO)


class AsyncPipelineSession:
    """Async façade over :class:`PipelineSession` using worker threads."""

    __slots__ = ("_inner",)

    def __init__(self, inner: PipelineSession) -> None:
        self._inner = inner

    async def ask(self, question: str) -> SessionStep:
        return await asyncio.to_thread(self._inner.ask, question)

    async def step(self, response: str | None = None) -> SessionStep:
        return await asyncio.to_thread(self._inner.step, response)

    async def reset(self) -> None:
        await asyncio.to_thread(self._inner.reset)

    async def awaiting_prompt(self) -> bool:
        return await asyncio.to_thread(self._inner.awaiting_prompt)

    async def ask_until_done(self, question: str, *, on_confirm: Literal["y", "n"] = "y") -> SessionStep:
        """Async wrapper around :meth:`PipelineSession.ask_until_done` including the same terminal-status semantics for final SQL rejection."""
        return await asyncio.to_thread(self._inner.ask_until_done, question, on_confirm=on_confirm)

    async def accept_until_done(
        self,
        question: str,
        *,
        on_yes_no: Literal["y", "n"] = "y",
        on_free_text: str = "looks good",
    ) -> SessionStep:
        """Async wrapper around :meth:`PipelineSession.accept_until_done`."""
        return await asyncio.to_thread(
            self._inner.accept_until_done,
            question,
            on_yes_no=on_yes_no,
            on_free_text=on_free_text,
        )

    async def __aenter__(self) -> AsyncPipelineSession:
        """Enter the underlying synchronous session context on a worker thread."""
        await asyncio.to_thread(self._inner.__enter__)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        return await asyncio.to_thread(self._inner.__exit__, exc_type, exc_val, exc_tb)


def _render_interactive_suspend_step(step: SessionStep) -> None:
    """Emit suspend-phase notification and optional SQL preview via :func:`print_query_result`."""
    if step.message:
        notify(step.message, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    if step.sql is not None:
        hdr = list(step.data.columns) if step.data is not None else None
        rows = dataframe_to_row_tuples(step.data)
        print_query_result(rows, step.sql, headers=hdr)


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
        print_query_result(rows, step.sql, headers=hdr)


class AetherEngine:
    """Facade for environment-driven database setup, schema graph, and mode runners. ``_pipeline_writer_lock`` serializes writer-mode :class:`PipelineSession` turns and write-queue drains on a single instance; reader-mode sessions do not take this lock."""

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
        "_audit_sink",
        "_pipeline_writer_lock",
        "_config_file",
        "_schema_role",
        "_consumer_visible_objects",
        "_context_name",
        "_sandbox_mode",
        "_sandbox_closed",
    )

    def __dir__(self) -> list[str]:
        """Return names intended for interactive discovery."""
        return sorted(
            (
                "show_config",
                "session",
                "asession",
                "run_interactive",
                "run_seed_warmup",
                "run_seed_warmup_from_history",
                "run_seed_warmup_from_query_log",
                "run_qsim",
                "get_qsim_summary",
                "get_questions_only",
                "get_schema_stats",
                "get_seed_warmup_summary",
                "export_schema_overrides",
                "apply_schema_overrides",
                "aetherspace",
                "export_aetherspace",
                "export_aetherengine",
                "list_aetherspaces",
                "list_aetherengines",
                "clear_persisted_overrides",
                "clear_template_store",
                "clear_simulation_caches",
                "clear_all_learning",
                "apply_migration_map",
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
        config_file: str | os.PathLike[str] | None = None,
        execution_engine: Any = None,
        native_connection: Any = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
        role: SchemaRole = "owner",
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
            execution_engine: Optional SQLAlchemy engine for query execution (caller-owned pool / read replica).
            native_connection: Optional native duckdb or sqlite3 connection for embedded engines.
            For DuckDB and SQLite, ``native_connection`` or ``DuckDBRuntimeConfig.attach_connection`` ensures reflection and execution share one in-memory or file-backed database.
            ``execution_engine`` is honored when it wraps the same ``StaticPool`` connection.
            audit_sink: Optional callback receiving :class:`AuditEvent` records at lifecycle boundaries.
            role: Schema identity role; ``owner`` may mutate shared artifacts, ``consumer`` pins the owner snapshot id.

        Raises:

            ConfigError, ConnectionError, MigrationPendingError: Same as :func:`initialize_aether_engine`.
        """
        self._config_file = os.path.expanduser(str(config_file)) if config_file is not None else None
        self._execution_engine = execution_engine
        self._native_connection = native_connection
        self._audit_sink = audit_sink
        self._pipeline_writer_lock = threading.Lock()
        self._schema_role = role
        self._consumer_visible_objects: frozenset[str] | None = None
        self._context_name = MASTER_AETHERSPACE_NAME
        self._sandbox_mode = False
        self._sandbox_closed = False
        bundle = initialize_aether_engine(
            engine_context,
            artifacts_dir=artifacts_dir,
            config_file=config_file,
            log_sink=_init_log_sink,
            execution_engine=self._execution_engine,
            native_connection=self._native_connection,
            schema_role=role,
        )
        self._apply_init_bundle(bundle)
        self._audit_emit(
            "init",
            question=None,
            schema_hash=None,
            details=(("engine", self.dialect),),
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
        }

    def _require_owner(self, operation: str) -> None:
        if self._schema_role != "owner":
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
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
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

    @property
    def _schema_graph_id(self) -> str:
        """Stable schema-graph identity for template store and write- queue matching."""
        return str(getattr(self._schema_graph, "schema_graph_id", "") or "")

    @property
    def dialect(self) -> str:
        """Registered engine name from ``list_engines()``; see ``docs/SUPPORT_MATRIX.md``."""
        return str(self._runtime_config.engine)

    @property
    def write_queue_path(self) -> Path:
        """Path to ``write_queue.jsonl`` under the engine storage directory."""
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
        if not llm_credentials_configured():
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
    ) -> PipelineSession:
        """Return a programmatic session sharing this instance's schema graph and template store. ``writer`` mode may mutate artifacts and takes ``_pipeline_writer_lock`` during turns; ``reader`` mode is read- only and shares the owner snapshot without that lock."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")
        if self._schema_role == "consumer" and mode == "writer":
            raise OwnerOnlyOperationError("PipelineSession(mode='writer')")
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
        )

    def execute_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        as_dataframe: bool = False,
    ) -> pandas.DataFrame | list[tuple[Any, ...]]:
        """Execute a validated SELECT through the active dialect and engine context."""
        if getattr(self, "_sandbox_closed", False) is True:
            raise RuntimeError("Sandbox handle is closed; create a new offline_sandbox() instance.")

        runtime_cfg = getattr(self, "_runtime_config", None)
        execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
        scope_ctx = execution_context
        if scope_ctx is None and runtime_cfg is not None:
            scope_ctx = getattr(runtime_cfg, "engine_context", None)
        rows = execute_guarded_sql(
            self._dialect,
            sql,
            params,
            schema=self._schema_graph,
            schema_role=self._schema_role,
            schema_context=scope_ctx,
            visible_objects=self._consumer_visible_objects,
        )
        if as_dataframe:
            return pandas.DataFrame([list(r) for r in rows])
        return rows

    def asession(
        self,
        *,
        mode: Literal["reader", "writer"] = "writer",
        space: str = "master",
    ) -> AsyncPipelineSession:
        """Async wrapper around :meth:`session` (uses threads; underlying API remains synchronous)."""
        return AsyncPipelineSession(self.session(mode=mode, space=space))

    def aetherspace(
        self,
        name: str,
        space_context: SpaceContext | None = None,
        *,
        notes_file: str | None = None,
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
        validated = validate_space_context_against_graph(space_context, self._schema_graph)
        snapshot = subset_graph_for_space(self._schema_graph, validated)
        if notes_file is not None and str(notes_file).strip():
            snapshot = enrich_space_snapshot_with_notes(
                snapshot,
                self._schema_graph,
                validated,
                notes_file,
            )
        save_aetherspace_snapshot(str(self._artifacts_dir), norm, snapshot)
        return aetherspace_descriptor_from_snapshot(norm, snapshot)

    def export_aetherspace(self, name: str) -> Path:
        """Export a read-only JSON snapshot of one named aetherspace."""
        self._require_master_context("export_aetherspace")
        norm = str(name).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME and load_aetherspace_snapshot(str(self._artifacts_dir), norm) is None:
            raise ConfigError(f"unknown aetherspace {name!r}")
        return export_aetherspace_json(str(self._artifacts_dir), norm, self._schema_graph)

    def list_aetherspaces(self) -> tuple[str, ...]:
        """Return saved aetherspace names plus the implicit ``master`` space."""
        self._require_master_context("list_aetherspaces")
        saved = list_saved_aetherspace_names(str(self._artifacts_dir))
        return (MASTER_AETHERSPACE_NAME,) + saved

    def export_aetherengine(self, name: str) -> Path:
        """Export a read-only JSON snapshot of one named engine context."""
        self._require_master_context("export_aetherengine")
        norm = str(name).strip().lower()
        if norm != MASTER_AETHERSPACE_NAME and load_named_schema_context(str(self._artifacts_dir), norm) is None:
            raise ConfigError(f"unknown engine context {name!r}")
        return export_named_schema_context_json(
            str(self._artifacts_dir),
            norm,
            self._runtime_config.engine_context,
        )

    def list_aetherengines(self) -> tuple[str, ...]:
        """Return saved engine-context names plus the implicit ``master`` context."""
        self._require_master_context("list_aetherengines")
        saved = list_named_schema_context_names(str(self._artifacts_dir))
        return (MASTER_AETHERSPACE_NAME,) + saved

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
        role: SchemaRole = "owner",
    ) -> AetherEngine:
        """Copy a validated migration map into the working directory and construct ``AetherEngine``."""
        if role != "owner":
            raise OwnerOnlyOperationError("apply_migration_map")
        src = Path(os.path.expanduser(str(path))).resolve()
        dst = Path.cwd() / MIGRATION_MAP_FILENAME
        shutil.copyfile(src, dst)
        return cls(
            engine_context,
            artifacts_dir=artifacts_dir,
            config_file=config_file,
            execution_engine=execution_engine,
            native_connection=native_connection,
            role=role,
        )

    def get_schema_stats(self) -> SchemaStatsSnapshot:
        """Return frozen schema statistics."""
        return SchemaStatsSnapshot(stats=dict(self._schema_stats))

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
            qsim_run_once(
                num_intents=num_intents,
                num_questions=num_questions,
                seed=seed,
                artifacts_dir=str(self._artifacts_dir),
                schema=self._schema_graph,
            )

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

    def export_schema_overrides(self) -> Path:
        """Write ``schema_overrides.json`` in the process working directory and return its path, replacing any existing file atomically."""
        self._require_owner("export_schema_overrides")
        target = Path.cwd() / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        return dump_schema_overrides_to_path(self._schema_graph, target)

    def apply_schema_overrides(self) -> None:
        """Apply ``schema_overrides.json`` from the working directory to the in-memory schema graph, re-stamp the cached graph artifact, print a summary, then rename editor JSON files to ``*.applied.json`` (archiving any prior applied copy)."""
        source = Path.cwd() / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        if not source.is_file():
            raise ConfigError(f"schema overrides file not found: {source}")
        if self._schema_role == "consumer":
            with open(source, encoding="utf-8") as fh:
                document = json.load(fh)
            ev = WriteQueueEvent(
                kind="override_proposal",
                schema_graph_id=self._schema_graph_id,
                schema_hash=str(self._schema_graph.effective_structural_hash or ""),
                produced_at=datetime.now(timezone.utc).isoformat(),
                payload=(("document_json", json.dumps(document, ensure_ascii=False)),),
            )
            emit_write_queue_event(str(self._artifacts_dir), ev)
            return
        self._require_owner("apply_schema_overrides")
        report = apply_overrides_and_persist(
            self._schema_graph,
            source,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            dialect=self._dialect,
        )
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
        editor = Path.cwd() / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        companion = editor.with_name(editor.stem + ".schema.json")
        ts = datetime.now(timezone.utc).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
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

    def clear_persisted_overrides(self) -> bool:
        """Delete the persisted overrides sidecar and the cached schema graph, then rebuild the schema from catalog and inference layers. Returns True when a sidecar existed and was removed; False when no sidecar was present. After clearing, the schema graph is rebuilt from scratch (catalog reflection, profile-based PK inference, and FK inference) and the in-memory graph is published atomically; user-added FKs, user PK overrides, and the inference block lists are all discarded."""
        self._require_owner("clear_persisted_overrides")
        removed = clear_persisted_overrides(EngineConfig.SCHEMA_JSON_PATH)
        bundle = initialize_aether_engine(
            self._runtime_config.engine_context,
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

    def clear_template_store(self) -> bool:
        """Remove persisted templates and question feedback, then reload engine initialization state."""
        self._require_owner("clear_template_store")
        existed = clear_template_store_only(str(self._artifacts_dir), self._schema_graph)
        bundle = initialize_aether_engine(
            self._runtime_config.engine_context,
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

    def clear_simulation_caches(self) -> int:
        """Remove QSim and seed-warmup artifact files, then reload engine initialization state."""
        self._require_owner("clear_simulation_caches")
        count = clear_simulation_caches_only(str(self._artifacts_dir))
        bundle = initialize_aether_engine(
            self._runtime_config.engine_context,
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

    def clear_all_learning(self, *, keep_overrides: bool = True) -> None:
        """Remove templates, simulation caches, and optionally persisted schema overrides, then reload."""
        self._require_owner("clear_all_learning")
        clear_template_store_only(str(self._artifacts_dir), self._schema_graph)
        clear_simulation_caches_only(str(self._artifacts_dir))
        if not keep_overrides:
            clear_persisted_overrides(EngineConfig.SCHEMA_JSON_PATH)
        bundle = initialize_aether_engine(
            self._runtime_config.engine_context,
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
    def offline_sandbox(cls, **kwargs: Any) -> SandboxHandle:
        """Enter the offline Rental Shop sandbox (in-memory DuckDB + mock LLM fixtures)."""
        return create_offline_sandbox(cls, **kwargs)

    @classmethod
    def sandbox_questions(cls) -> list[str]:
        """Return curated natural-language sandbox practice questions."""
        return sandbox_questions()

    @classmethod
    def sandbox_doctor(cls) -> list[str]:
        """Return human-readable problems; empty list means the sandbox bundle looks healthy."""
        return sandbox_doctor()

    @classmethod
    def sandbox_catalog(cls) -> dict[str, object]:
        """Return bundled sandbox discovery metadata (paraphrase pairs, demos)."""
        return sandbox_catalog()

    @classmethod
    def sandbox_paraphrase_pairs(cls) -> list[dict[str, object]]:
        """Return canonical→paraphrase wordings from the bundled sandbox catalog."""
        return sandbox_paraphrase_pairs()

    @classmethod
    def sandbox_validation_failure_demo(cls) -> list[dict[str, str]]:
        """Return example validation-failure questions and short descriptions."""
        return sandbox_validation_failure_demo()

    @classmethod
    def sandbox_feedback_demo(cls) -> dict[str, str]:
        """Return the scripted reject/retry feedback demo."""
        return sandbox_feedback_demo()

    @classmethod
    def assert_sandbox_complete(cls) -> None:
        """Validate the shipped sandbox corpus and raise when any slot fails."""
        assert_sandbox_complete(cls)


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
    AetherSpace,
    AsyncPipelineSession,
    AuditEvent,
    ConfigError,
    ConfigSnapshot,
    ConnectionError,
    DatabasePingFailed,
    Diagnostic,
    LlmTransientFailure,
    MigrationPendingError,
    MigrationPreview,
    MockFixtureMissingError,
    OwnerOnlyOperationError,
    PERMISSION_DENIED_USER_MESSAGE,
    PipelineSession,
    QSimSummarySnapshot,
    RetryableError,
    SchemaAccessError,
    EngineContext,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    SessionActiveError,
    SessionStep,
    SpaceContext,
    StatementTimeoutError,
    AetherEngine,
    __version__,
)
