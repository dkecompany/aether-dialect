"""
Public Text2SQL facade delegating construction and runners to main_execution.

Attributes on Text2SQL whose names start with a single underscore are private implementation details and are not part of the public stability contract.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import _core_utils
from ._config import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    MIGRATION_MAP_FILENAME,
    SCHEMA_OVERRIDES_APPLIED_SUFFIX,
    SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT,
    SCHEMA_OVERRIDES_DEFAULT_FILENAME,
    YES_NO_SESSION_KINDS,
    EngineConfig,
    llm_credentials_configured,
)
from ._contracts_base import (
    AuditEvent,
    ConfigError,
    ConfigSnapshot,
    OverrideReport,
    OverrideSkip,
    QSimSummarySnapshot,
    SchemaContext,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    SessionStep,
)
from ._core_utils import diagnostic_print_listener
from ._main_execution import (
    PipelineSession,
    Text2SQLInitResult,
    _format_qsim_summary_line,
    _format_seed_warmup_summary,
    clear_simulation_caches_only,
    clear_template_store_only,
    describe_runtime_config,
    find_latest_seed_warmup_summary,
    initialize_text2sql,
    load_qsim_summaries,
    print_questions_bundle,
    qsim_run_once,
    run_seed_warmup_from_history_execution,
    run_seed_warmup_from_query_log_execution,
    seed_warmup_run_once,
)
from ._schema import (
    apply_overrides_and_persist,
    dump_schema_overrides_to_path,
)
from ._schema import (
    clear_persisted_overrides as _clear_persisted_overrides,
)


def _init_log_sink(line: str) -> None:
    _core_utils.notify(line, stage="init", code=DIAGNOSTIC_CODE_ENGINE_INFO)


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
        _core_utils.notify(step.message, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    if step.sql is not None:
        hdr = list(step.data.columns) if step.data is not None else None
        rows = _core_utils.dataframe_to_row_tuples(step.data)
        _core_utils.print_query_result(rows, step.sql, headers=hdr)


def _render_interactive_terminal_step(step: SessionStep) -> None:
    """Emit terminal errors, messages, and optional final SQL preview."""

    if step.error:
        _core_utils.error(step.error)
        return
    if step.message:
        _core_utils.notify(step.message, stage="interactive", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    if step.sql is not None:
        hdr = list(step.data.columns) if step.data is not None else None
        rows = _core_utils.dataframe_to_row_tuples(step.data)
        _core_utils.print_query_result(rows, step.sql, headers=hdr)


class Text2SQL:
    """Facade for environment-driven database setup, schema graph, and mode runners."""

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
        "_audit_sink",
        "_pipeline_writer_lock",
        "_config_file",
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
                "dry_run_warmup",
                "run_qsim",
                "get_qsim_summary",
                "get_questions_only",
                "get_schema_stats",
                "get_seed_warmup_summary",
                "export_schema_overrides",
                "apply_schema_overrides",
                "clear_persisted_overrides",
                "clear_template_store",
                "clear_simulation_caches",
                "clear_all_learning",
                "apply_migration_map",
            ),
        )

    def __init__(
        self,
        schema_context: SchemaContext | None = None,
        *,
        artifacts_dir: str | None = None,
        config_file: str | os.PathLike[str] | None = None,
        execution_engine: Any = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
    ) -> None:
        """
        Initialise engine configuration from the environment, build the schema graph, and load templates.

        Args:

            schema_context: Schema scope, allow and deny lists, and optional notes or SQL file paths.
            When omitted, a persisted ``SchemaContext`` is loaded from ``artifacts_dir``;
            if no cached context exists, a ``ConfigError`` is raised.

            artifacts_dir: Optional directory root; engine files are stored under ``<root>/aetherdialect/<connection_slug>``.

            config_file: Path to a TOML file. When set, every mapped field that appears in the file is authoritative for the corresponding process environment key (empty TOML values clear any inherited environment value for that key); fields omitted from the file are still read from ``os.environ``. When omitted, settings are read from ``os.environ`` only.

            execution_engine: Optional SQLAlchemy engine for query execution (caller-owned pool / read replica).

            audit_sink: Optional callback receiving :class:`AuditEvent` records at lifecycle boundaries.

        Raises:

            ConfigError, ConnectionError, MigrationPendingError: Same as :func:`initialize_text2sql`.
        """

        self._config_file = os.path.expanduser(str(config_file)) if config_file is not None else None
        self._execution_engine = execution_engine
        self._audit_sink = audit_sink
        self._pipeline_writer_lock = threading.Lock()
        bundle = initialize_text2sql(
            schema_context,
            artifacts_dir=artifacts_dir,
            config_file=config_file,
            log_sink=_init_log_sink,
            execution_engine=self._execution_engine,
        )
        self._apply_init_bundle(bundle)
        self._audit_emit(
            "init",
            question=None,
            schema_hash=None,
            details=(("engine", self.dialect),),
        )

    def _reinit_bundle_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`initialize_text2sql` when reloading state."""

        return {
            "artifacts_dir": str(self._artifacts_dir),
            "log_sink": _init_log_sink,
            "execution_engine": self._execution_engine,
            "config_file": self._config_file,
        }

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

    def _apply_init_bundle(self, bundle: Text2SQLInitResult) -> None:
        """Assign fields from a fresh :class:`Text2SQLInitResult` (also used after cache-clear reloads)."""

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

    @property
    def dialect(self) -> str:
        """Registered engine name (``postgresql`` or ``databricks``)."""

        return str(self._runtime_config.engine)

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

    def session(self, *, mode: Literal["reader", "writer"] = "writer") -> PipelineSession:
        """Return a programmatic session sharing this instance's schema graph and template store."""

        return PipelineSession(self, mode=mode)

    def asession(self, *, mode: Literal["reader", "writer"] = "writer") -> AsyncPipelineSession:
        """Async wrapper around :meth:`session` (uses threads; underlying API remains synchronous)."""

        return AsyncPipelineSession(self.session(mode=mode))

    @classmethod
    def apply_migration_map(
        cls,
        path: str = "schema_migration_map.json",
        *,
        config_file: str | os.PathLike[str] | None = None,
        schema_context: SchemaContext,
        artifacts_dir: str,
    ) -> Text2SQL:
        """Copy a validated migration map into the working directory and construct ``Text2SQL``."""

        src = Path(os.path.expanduser(str(path))).resolve()
        dst = Path.cwd() / MIGRATION_MAP_FILENAME
        shutil.copyfile(src, dst)
        return cls(
            schema_context,
            artifacts_dir=artifacts_dir,
            config_file=config_file,
        )

    def get_schema_stats(self) -> SchemaStatsSnapshot:
        """Return frozen schema statistics."""

        return SchemaStatsSnapshot(stats=dict(self._schema_stats))

    def get_seed_warmup_summary(self) -> SeedWarmupSummarySnapshot:
        """Return the newest seed-warmup summary text if present."""

        s = find_latest_seed_warmup_summary(str(self._artifacts_dir))
        if s is None:
            return SeedWarmupSummarySnapshot(text="Seed warmup summary: none found.")
        return SeedWarmupSummarySnapshot(text=_format_seed_warmup_summary(s))

    def get_qsim_summary(self, start: int, end: int) -> QSimSummarySnapshot:
        """Return QSim summary lines for versions ``start`` through ``end`` inclusive."""

        summaries = load_qsim_summaries(str(self._artifacts_dir))
        picked = [s for s in summaries if start <= int(s.version) <= end]
        lines: list[str] = [f"QSim range ({len(picked)} runs):"]
        for s in picked:
            lines.append(_format_qsim_summary_line(s))
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

    def run_interactive(self) -> None:
        """
        Prompt once for a natural-language question, resolve it through the interactive prompt cycle, then return.

        An empty line at the question prompt warns once; a second empty line terminates with ``User terminated.``.
        There is no outer REPL loop; call ``run_interactive`` again for another question.
        """

        self._ensure_llm()
        with _core_utils.diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            _core_utils.notify(
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
                    _core_utils.terminated()
                    return
                _core_utils.echo_user_text(raw)
                if raw.strip() == "":
                    empty_streak += 1
                    if empty_streak >= 2:
                        _core_utils.terminated()
                        return
                    _core_utils.notify(
                        "Press Enter again to quit.",
                        stage="interactive",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    continue
                question = raw.strip()
                break

            with PipelineSession(self) as session:
                try:
                    with _core_utils.progress_enabled():
                        step = session.ask(question)
                        while not step.done:
                            _render_interactive_suspend_step(step)
                            print(step.prompt or "", end="", flush=True)
                            try:
                                ans = input()
                            except (EOFError, KeyboardInterrupt):
                                _core_utils.terminated()
                                return
                            if step.kind in YES_NO_SESSION_KINDS:
                                _core_utils.echo_yes_no_answer(ans)
                            else:
                                _core_utils.echo_user_text(ans)
                            step = session.step(ans)
                        _render_interactive_terminal_step(step)
                except (EOFError, KeyboardInterrupt):
                    _core_utils.terminated()
                    return
                except Exception as exc:
                    _core_utils.error(f"{exc.__class__.__name__}: {exc}")
                    return

    def run_seed_warmup(
        self,
        seed_filepath: str,
        interactive_gold: bool = True,
    ) -> None:
        """Run seed warmup execution, stratified sampling, and template writes."""

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
                warmup_dry_run_only=False,
            )

    def run_seed_warmup_from_history(self, sql_history_filepath: str) -> None:
        """Reverse-engineer SQL history into intents and run seed warmup over them."""

        self._ensure_llm()
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            run_seed_warmup_from_history_execution(self, sql_history_filepath)

    def run_seed_warmup_from_query_log(
        self,
        *,
        lookback_days: int = 730,
        max_queries: int = 5000,
        min_runs: int = 1,
        user_filter: str | None = None,
    ) -> None:
        """Fetch SQL history from engine system catalogs when available and run seed warmup."""

        self._ensure_llm()
        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            run_seed_warmup_from_query_log_execution(
                self,
                lookback_days=lookback_days,
                max_queries=max_queries,
                min_runs=min_runs,
                user_filter=user_filter,
            )

    def dry_run_warmup(
        self,
        seed_filepath: str,
        interactive_gold: bool = True,
    ) -> None:
        """Run seed warmup execution without question LLM or template persistence."""

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
                warmup_dry_run_only=True,
            )

    def run_qsim(
        self,
        num_intents: int = 20,
        num_questions: int = 100,
        seed: int | None = None,
    ) -> None:
        """Generate synthetic NL questions from schema-derived intent skeletons."""

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

        with diagnostic_print_listener(lambda m: print(m, file=sys.stdout, flush=True)):
            print_questions_bundle(version, str(self._artifacts_dir))

    def show_config(self) -> ConfigSnapshot:
        """Return a redacted snapshot of engine, schema scope, database, and LLM settings."""

        return ConfigSnapshot(text=describe_runtime_config(self._runtime_config, self._llm_config))

    def export_schema_overrides(self) -> Path:
        """Write ``schema_overrides.json`` in the process working directory and return its path, replacing any existing file atomically."""

        target = Path.cwd() / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        return dump_schema_overrides_to_path(self._schema_graph, target)

    def apply_schema_overrides(self) -> None:
        """Apply ``schema_overrides.json`` from the working directory to the in-memory schema graph, re-stamp the cached graph artifact, print a summary, then rename editor JSON files to ``*.applied.json`` (archiving any prior applied copy)."""

        source = Path.cwd() / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        report = apply_overrides_and_persist(
            self._schema_graph,
            source,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
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
        """
        Delete the persisted overrides sidecar and the cached schema graph, then rebuild the schema from catalog and inference layers.

        Returns True when a sidecar existed and was removed; False when no sidecar was present. After clearing, the schema graph is rebuilt from scratch (catalog reflection, profile-based PK inference, and FK inference) and the in-memory graph is published atomically; user-added FKs, user PK overrides, and the inference block lists are all discarded.
        """

        removed = _clear_persisted_overrides(EngineConfig.SCHEMA_JSON_PATH)
        bundle = initialize_text2sql(
            self._runtime_config.schema_context,
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

        existed = clear_template_store_only(str(self._artifacts_dir), self._schema_graph)
        bundle = initialize_text2sql(
            self._runtime_config.schema_context,
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

        count = clear_simulation_caches_only(str(self._artifacts_dir))
        bundle = initialize_text2sql(
            self._runtime_config.schema_context,
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

        clear_template_store_only(str(self._artifacts_dir), self._schema_graph)
        clear_simulation_caches_only(str(self._artifacts_dir))
        if not keep_overrides:
            _clear_persisted_overrides(EngineConfig.SCHEMA_JSON_PATH)
        bundle = initialize_text2sql(
            self._runtime_config.schema_context,
            **self._reinit_bundle_kwargs(),
        )
        self._apply_init_bundle(bundle)
        self._schema_stats = self._schema_graph.refresh_schema_stats()
        self._audit_emit(
            "clear_all_learning",
            schema_hash=str(getattr(self._schema_graph, "effective_structural_hash", "") or None) or None,
            details=(("keep_overrides", str(keep_overrides)),),
        )


def _print_override_summary(report: OverrideReport) -> None:
    """Emit a fixed-template summary of an ``OverrideReport`` through the notify channel."""

    _core_utils.notify("Schema overrides applied:", stage="overrides", code=DIAGNOSTIC_CODE_ENGINE_INFO)
    _core_utils.notify(
        f"  Tables updated:           {report.table_edits}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  Columns updated:          {report.column_edits}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  FK edges added:           {report.fks_added}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  FK edges endorsed (user): {report.fks_endorsed}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  FK edges removed:         {report.fks_removed}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  PKs endorsed (user):     {report.pks_endorsed}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  Inferred PKs cleared:     {report.pks_blocked}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  PK/FK roles coerced:      {report.coerced_columns}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  Redundant inferences:     {report.collapsed_inferences}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    _core_utils.notify(
        f"  Descriptions refined:     {report.descriptions_refined}",
        stage="overrides",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
    )
    if report.skipped:
        _core_utils.notify(
            f"  Soft skips ({len(report.skipped)}):",
            stage="overrides",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )
        for skip in report.skipped:
            _core_utils.notify(
                f"    {skip.path}  -  {skip.reason}",
                stage="overrides",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                details=(("path", skip.path), ("reason", skip.reason)),
            )
    else:
        _core_utils.notify(
            "  Soft skips:               none",
            stage="overrides",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
        )


__all__ = [
    "Text2SQL",
    "AsyncPipelineSession",
    "ConfigSnapshot",
    "SchemaStatsSnapshot",
    "SeedWarmupSummarySnapshot",
    "QSimSummarySnapshot",
]
