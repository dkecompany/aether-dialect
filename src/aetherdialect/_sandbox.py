"""Offline mock sandbox: zip-backed demo data bundle, fixture replay, tours."""

from __future__ import annotations

import gc
import importlib
import json
import os
import shutil
import stat
import tempfile
import time
import tomllib
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from aetherdialect._config import (
    DuckDBRuntimeConfig,
    SandboxBundlePolicy,
    SQLiteRuntimeConfig,
)
from aetherdialect._constants import (
    AETHERSPACES_SEGMENT,
    CONSUMER_ALLOW_OBJECTS,
    CONSUMER_EXAMPLE_NARROW_ALLOW_OBJECTS,
    FEDERATION_COMPOSITE_SCHEMA_FILENAME,
    FEDERATION_DECLARATION_FILENAME,
    FEDERATION_MANIFEST_FILENAME,
    FEDERATION_MAPPINGS_FILENAME,
    RENTAL_SHOP_VIEW_NAMES,
    SANDBOX_BASELINE_CACHE_FILES,
    SANDBOX_BUNDLED_DATASET_NAMES,
    SANDBOX_BUNDLED_MEMBER_SEEDS,
    SANDBOX_CATALOG_SPACE_TABLES,
    SANDBOX_CONNECTION_HOST_ATTR,
    SANDBOX_DEFAULT_DATASET_NAME,
    SANDBOX_DOCTOR_OPTIONAL_BASELINE_DIRS,
    SANDBOX_DOCTOR_OPTIONAL_BASELINE_MEMBERS,
    SANDBOX_DOCTOR_REQUIRED_MEMBERS,
    SANDBOX_FIXTURE_ALIASES,
    SANDBOX_INTERPRET_DOMAIN_FILENAME,
    SANDBOX_LEGACY_FAITHFULNESS_SPECS,
    SANDBOX_MIN_FIXTURE_COUNT,
    SANDBOX_MIN_INTENT_FIXTURE_COUNT,
    SANDBOX_RECIPES,
    SANDBOX_SCHEMA_LITERALS_FILENAME,
    SANDBOX_TOUR_EXPECT_NO_SQL,
    SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL,
    SANDBOX_VALIDATION_FAILURE_QUESTIONS,
    SCHEMA_OVERRIDES_DEFAULT_FILENAME,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_RESULT,
    WRITE_QUEUE_FILENAME,
)
from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    FederationContext,
    MigrationPendingError,
    MigrationPreview,
    OwnerOnlyOperationError,
    SandboxBuildSection,
    SandboxLlmMode,
    SandboxPreset,
    SchemaInclude,
    SchemaRole,
    SessionActiveError,
    SessionStep,
    SpaceContext,
    WhereParam,
)
from aetherdialect._contracts_core import (
    NormalizedExpr,
    PredicateGroup,
    RuntimeIntent,
)
from aetherdialect._core_utils import (
    append_failure_trace,
    build_session_step_trace,
    debug,
    pipeline_capture,
    require_driver,
)
from aetherdialect._dialect import Dialect
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._federation import (
    FederationSourceBinding,
    binding_from_member_engine,
    compute_federation_storage_dir,
    federation_source_artifacts_dir,
    federation_source_storage_slug,
    parse_federation_declaration,
)
from aetherdialect._llm_provider import (
    MockProvider,
    SandboxRuntimeState,
)
from aetherdialect._main_execution import MainExecutionOps, PipelineSession
from aetherdialect._templates import TemplateOps

_ARTIFACT_REFCOUNT: dict[str, int] = {}
_CONNECTION_SANDBOX_HOSTS: dict[int, Sandbox] = {}


@dataclass(frozen=True)
class _DataBundleAccess:
    path: Path
    owns_cleanup: bool


@dataclass
class _SandboxDataset:
    connection: Any
    owns_connection: bool


class _SandboxPendingMethods:
    """Temporary holder; methods are reassigned onto Sandbox below."""

    @staticmethod
    def _write_sandbox_toml(*, fixtures_file: str) -> str:
        lines = [
            "[engine]",
            'selected = "duckdb"',
            "",
            "[duckdb]",
            'path = ":memory:"',
            "",
            "[llm]",
            'provider = "mock"',
            "",
            "[mock]",
            f"fixtures_file = {json.dumps(fixtures_file)}",
            "",
        ]
        fd, path = tempfile.mkstemp(prefix="aetherdialect_sandbox_", suffix=".toml")
        os.close(fd)
        config_path = Path(path)
        config_path.write_text("\n".join(lines), encoding="utf-8")
        return str(config_path)

    @staticmethod
    def _schema_context_for_preset(
        preset: SandboxPreset | str,
        *,
        notes_file: str | None,
        sql_file: str | None,
        deny_columns: frozenset[str] | None,
        restricted_consumer: bool,
        include: SchemaInclude = SchemaInclude.TABLES,
        engine_context: EngineContext | None = None,
    ) -> EngineContext:
        if engine_context is not None:
            ctx = replace(
                engine_context,
                notes_file=notes_file,
                sql_file=sql_file,
                include=include,
            )
            if deny_columns:
                merged = ctx.deny_columns or frozenset()
                ctx = replace(ctx, deny_columns=merged | deny_columns)
            return ctx
        if preset == "owner_writer":
            ctx = Sandbox._owner_writer_schema_context(notes_file=notes_file, sql_file=sql_file)
        else:
            ctx = Sandbox._consumer_reader_schema_context(
                notes_file=notes_file,
                sql_file=sql_file,
            )
        ctx = replace(ctx, include=include)
        if deny_columns:
            ctx = replace(ctx, deny_columns=deny_columns)
        return ctx

    @staticmethod
    def _resolve_sandbox_fixture_path(
        extract_path: Path,
        user_path: str | None,
    ) -> tuple[str | None, str | None]:
        """Resolve a user path to a bundled fixture when the path is absent on disk."""
        if not user_path:
            return None, None
        expanded = Path(user_path).expanduser()
        if expanded.is_file():
            return str(expanded), None
        alias_target = SANDBOX_FIXTURE_ALIASES.get(Path(user_path).name)
        if alias_target:
            bundled = extract_path / alias_target
            if bundled.is_file():
                return (
                    str(bundled),
                    f"{user_path!r} resolved to bundled fixture {alias_target!r}",
                )
        return None, None

    @staticmethod
    def _resolve_sandbox_notes_and_sql(
        *,
        engine_context: EngineContext | None,
        notes_file: str | None,
        sql_file: str | None,
        bundled_notes: str | None,
        bundled_sql: str | None,
        extract_path: Path | None = None,
    ) -> tuple[str | None, str | None, tuple[str, ...]]:
        resolved_notes = notes_file
        if resolved_notes is None and engine_context is not None:
            resolved_notes = engine_context.notes_file
        notices: list[str] = []
        if extract_path is not None and resolved_notes:
            aliased_notes, notice = Sandbox._resolve_sandbox_fixture_path(extract_path, resolved_notes)
            if aliased_notes is not None:
                resolved_notes = aliased_notes
                if notice:
                    notices.append(notice)
        if resolved_notes is None:
            resolved_notes = bundled_notes

        resolved_sql = sql_file
        if resolved_sql is None and engine_context is not None:
            resolved_sql = engine_context.sql_file
        if extract_path is not None and resolved_sql:
            aliased_sql, notice = Sandbox._resolve_sandbox_fixture_path(extract_path, resolved_sql)
            if aliased_sql is not None:
                resolved_sql = aliased_sql
                if notice:
                    notices.append(notice)
        if resolved_sql is None:
            resolved_sql = bundled_sql
        return resolved_notes, resolved_sql, tuple(notices)

    @staticmethod
    def _sandbox_scope_signature(
        ctx: EngineContext,
    ) -> tuple[frozenset[str] | None, frozenset[str] | None, frozenset[str] | None, str]:
        allow = frozenset(ctx.allow_objects) if ctx.allow_objects else None
        deny_objects = frozenset(ctx.deny_objects) if ctx.deny_objects else None
        deny_columns = frozenset(ctx.deny_columns) if ctx.deny_columns else None
        return allow, deny_objects, deny_columns, ctx.include

    @staticmethod
    def _sandbox_trusts_bundled_baseline(
        *,
        preset: SandboxPreset | str,
        schema_context: EngineContext,
        bundled_notes: str | None,
        bundled_sql: str | None,
        deny_columns: frozenset[str] | None,
        restricted_consumer: bool,
        include: SchemaInclude,
        engine_context: EngineContext | None,
        notes_file: str | None,
        sql_file: str | None,
    ) -> bool:
        if engine_context is None and notes_file is None and sql_file is None:
            return True
        baseline_context = Sandbox._schema_context_for_preset(
            preset,
            notes_file=bundled_notes,
            sql_file=bundled_sql,
            deny_columns=deny_columns,
            restricted_consumer=restricted_consumer,
            include=include,
        )
        return Sandbox._sandbox_scope_signature(schema_context) == Sandbox._sandbox_scope_signature(baseline_context)

    @staticmethod
    def _role_for_preset(preset: SandboxPreset | str) -> SchemaRole:
        return SchemaRole.CONSUMER if preset == "consumer_reader" else SchemaRole.OWNER

    @staticmethod
    def _baseline_dir_for_preset(
        extract_path: Path,
        preset: SandboxPreset | str,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
    ) -> Path | None:
        """Return the bundled schema baseline directory for *preset*, or ``None`` when absent."""
        root = extract_path / "artifacts_baseline"
        if preset == "federation":
            fed = root / "federation"
            if fed.is_dir() and (fed / FEDERATION_COMPOSITE_SCHEMA_FILENAME).is_file():
                return fed
            if fed.is_dir() and (fed / "schema_graph.json.gz").is_file():
                return fed
        if include == "views":
            if preset == "consumer_reader":
                views = root / "consumer_views"
            else:
                views = root / "owner_views"
            if views.is_dir() and (views / "schema_graph.json.gz").is_file():
                return views
            return views if views.is_dir() else None
        if preset == "consumer_reader":
            consumer = root / "consumer"
            if consumer.is_dir() and (consumer / "schema_graph.json.gz").is_file():
                return consumer
        owner = root / "owner"
        if owner.is_dir() and (owner / "schema_graph.json.gz").is_file():
            return owner
        if (root / "schema_graph.json.gz").is_file():
            return root
        consumer = root / "consumer"
        if consumer.is_dir() and (consumer / "schema_graph.json.gz").is_file():
            return consumer
        if root.is_dir() and any(root.iterdir()):
            return root
        return None

    @staticmethod
    def _reset_sandbox_duckdb_runtime() -> None:
        DuckDBRuntimeConfig.DATABASE_PATH = ":memory:"
        DuckDBRuntimeConfig.SCHEMA = "main"

    @staticmethod
    def _sandbox_memory_engine_dir(artifacts_dir: str) -> Path:
        saved_path = DuckDBRuntimeConfig.DATABASE_PATH
        saved_schema = DuckDBRuntimeConfig.SCHEMA
        try:
            Sandbox._reset_sandbox_duckdb_runtime()
            return Path(MainExecutionOps.compute_engine_storage_dir(artifacts_dir, "duckdb"))
        finally:
            DuckDBRuntimeConfig.DATABASE_PATH = saved_path
            DuckDBRuntimeConfig.SCHEMA = saved_schema

    @staticmethod
    def _copy_baseline_cache_files(source: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for name in SANDBOX_BASELINE_CACHE_FILES:
            src = source / name
            if src.is_file():
                shutil.copy2(src, dest / name)
        composite = source / FEDERATION_COMPOSITE_SCHEMA_FILENAME
        schema_dest = dest / "schema_graph.json.gz"
        if composite.is_file() and not schema_dest.is_file():
            shutil.copy2(composite, schema_dest)
        for sidecar in source.glob("schema_context*.json"):
            if sidecar.name == "schema_context.json":
                continue
            dst = dest / sidecar.name
            if not dst.is_file():
                shutil.copy2(sidecar, dst)

    @staticmethod
    def _seed_bundled_aetherspaces(extract_path: Path, engine_dir: Path) -> None:
        src = extract_path / "artifacts_baseline" / AETHERSPACES_SEGMENT
        if not src.is_dir():
            return
        dest = engine_dir / AETHERSPACES_SEGMENT
        dest.mkdir(parents=True, exist_ok=True)
        for path in src.glob("*.json"):
            shutil.copy2(path, dest / path.name)

    @staticmethod
    def _aether_federation_cls() -> type[Any]:
        """Resolve ``AetherFederation`` without a circular import at ``_sandbox`` load time."""
        mod = importlib.import_module("aetherdialect.aetherdialect")
        return cast(type[Any], mod.AetherFederation)

    @staticmethod
    def _federation_partition_tables_from_bundle(extract_path: Path, source_id: str) -> frozenset[str]:
        """Return physical table names owned by *source_id* from the bundled partition map."""
        partition_path = extract_path / "federation_partition.json"
        if partition_path.is_file():
            payload = json.loads(partition_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw = payload.get(source_id, [])
                if isinstance(raw, list):
                    tables: set[str] = {str(name).strip() for name in raw if str(name).strip()}
                    declaration_path = extract_path / FEDERATION_DECLARATION_FILENAME
                    if declaration_path.is_file():
                        _, mappings = parse_federation_declaration(
                            json.loads(declaration_path.read_text(encoding="utf-8")),
                        )
                        for entry in mappings.logical_tables:
                            for member in entry.members:
                                if str(member.source).strip() == source_id:
                                    tables.add(str(member.table).strip())
                    return frozenset(name for name in tables if name)
        declaration_path = extract_path / FEDERATION_DECLARATION_FILENAME
        if not declaration_path.is_file():
            return frozenset()
        _, mappings = parse_federation_declaration(json.loads(declaration_path.read_text(encoding="utf-8")))
        tables = set()
        for entry in mappings.logical_tables:
            for member in entry.members:
                if str(member.source).strip() == source_id:
                    tables.add(str(member.table).strip())
        return frozenset(name for name in tables if name)

    @staticmethod
    def _seed_federation_member_baseline(
        *,
        baseline: Path,
        artifacts_dir: str,
        binding: Any,
    ) -> None:
        slug = federation_source_storage_slug(binding)
        member_src = baseline / slug
        if not member_src.is_dir():
            return
        member_dest = Path(federation_source_artifacts_dir(artifacts_dir, binding))
        if (member_dest / "schema_graph.json.gz").is_file():
            return
        Sandbox._copy_baseline_cache_files(member_src, member_dest)

    @staticmethod
    def _seed_federation_composite_baseline(*, baseline: Path, artifacts_dir: str, federation_id: str) -> None:
        fed_storage = Path(compute_federation_storage_dir(artifacts_dir, federation_id))
        graph_path = fed_storage / "schema_graph.json.gz"
        composite_path = fed_storage / FEDERATION_COMPOSITE_SCHEMA_FILENAME
        if graph_path.is_file() or composite_path.is_file():
            return
        fed_storage.mkdir(parents=True, exist_ok=True)
        Sandbox._copy_baseline_cache_files(baseline, fed_storage)
        for name in (
            FEDERATION_DECLARATION_FILENAME,
            FEDERATION_MANIFEST_FILENAME,
            FEDERATION_MAPPINGS_FILENAME,
            FEDERATION_COMPOSITE_SCHEMA_FILENAME,
        ):
            src = baseline / name
            if src.is_file() and not (fed_storage / name).is_file():
                shutil.copy2(src, fed_storage / name)

    @staticmethod
    def _create_federation_member_engine(
        engine_cls: type[Any],
        *,
        member_name: str,
        seed_path: str,
        extract_path: Path,
        artifacts_dir: str,
        config_path: str,
        notes_file: str | None,
        sql_file: str | None,
        allow_tables: frozenset[str],
        baseline: Path | None,
        binding: Any | None,
    ) -> tuple[Any, Any]:
        del member_name, extract_path
        connection = Sandbox._load_memory_connection(seed_path)
        execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
        schema_context = Sandbox._owner_writer_schema_context(notes_file=notes_file, sql_file=sql_file)
        if allow_tables:
            schema_context = replace(schema_context, allow_objects=allow_tables)
        if baseline is not None and binding is not None:
            Sandbox._seed_federation_member_baseline(baseline=baseline, artifacts_dir=artifacts_dir, binding=binding)
        engine = engine_cls(
            schema_context,
            artifacts_dir=artifacts_dir,
            config_file=config_path,
            execution_engine=execution_engine,
            native_connection=connection,
            role=SchemaRole.OWNER,
            trust_bundled_baseline=True,
        )
        engine._sandbox_mode = True
        return engine, connection

    @staticmethod
    def _default_federation_member_specs(extract_path: Path) -> tuple[tuple[str, str], ...]:
        return tuple(
            (member_name, str(extract_path / seed_name)) for member_name, seed_name in SANDBOX_BUNDLED_MEMBER_SEEDS
        )

    @staticmethod
    def _require_bundled_federation_seeds(extract_path: Path) -> None:
        missing = [
            seed_name
            for _member_name, seed_name in SANDBOX_BUNDLED_MEMBER_SEEDS
            if not (extract_path / seed_name).is_file()
        ]
        if missing:
            raise ConfigError(
                "federation preset requires bundled partition seeds: " + ", ".join(missing),
            )

    @staticmethod
    def _create_federation_offline_sandbox(
        engine_cls: type[Any],
        *,
        sandbox: Sandbox,
        owned_artifacts: bool,
        declaration_file: str | None = None,
        members: Mapping[str, str] | None = None,
        federation_context: FederationContext | None = None,
    ) -> SandboxHandle:
        del engine_cls
        extract_path = sandbox._extract_path
        if declaration_file is not None:
            declaration_path = Path(declaration_file)
        else:
            declaration_path = extract_path / FEDERATION_DECLARATION_FILENAME
        if members is None:
            Sandbox._require_bundled_federation_seeds(extract_path)
        if not declaration_path.is_file():
            raise ConfigError(f"federation preset requires bundled {FEDERATION_DECLARATION_FILENAME}")

        authored_manifest, _ = parse_federation_declaration(
            json.loads(declaration_path.read_text(encoding="utf-8")),
        )
        federation = sandbox.federation(
            authored_manifest.federation_id,
            declaration_file=str(declaration_path),
            members=members,
            context=federation_context,
        )
        member_connections = tuple(
            member._native_connection
            for member in federation._members.values()
            if getattr(member, "_native_connection", None) is not None
        )
        return SandboxHandle(
            federation,
            connection=None,
            artifacts_dir=sandbox.artifacts_dir,
            owned_artifacts=owned_artifacts,
            owns_connection=False,
            config_path=sandbox._config_path,
            extract_dir=sandbox._extract_dir,
            saved_embedded_runtime_state=sandbox._saved_embedded_runtime_state,
            member_connections=member_connections or None,
            sandbox=sandbox,
        )

    @staticmethod
    def _ensure_federation_catalog_aetherspace(engine: Any, extract_path: Path) -> None:
        notes_path = extract_path / "sandbox_space_catalog_notes.txt"
        notes_arg = str(notes_path) if notes_path.is_file() else None
        catalog = SpaceContext(
            tables=SANDBOX_CATALOG_SPACE_TABLES,
            columns=frozenset(),
            notes_file=notes_arg,
        )
        engine.aetherspace("catalog", space_context=catalog)

    @staticmethod
    def _apply_bundled_federation_mappings(
        engine: Any, extract_path: Path, *, handle: SandboxHandle | None = None
    ) -> None:
        source = extract_path / FEDERATION_DECLARATION_FILENAME
        if not source.is_file():
            return
        target = Path(engine._artifacts_dir) / FEDERATION_DECLARATION_FILENAME
        shutil.copyfile(source, target)
        if handle is not None:
            handle.register_cwd_sidecar(target)
        engine.apply_federation_declaration()

    @staticmethod
    @contextmanager
    def federation_scenario_session(
        engine: Any,
        question: str,
        *,
        mode: str | None = None,
    ) -> Iterator[PipelineSession]:
        """Yield a federation session configured for the sandbox scenario tied to *question*."""
        scenario = Sandbox._load_sandbox_scenarios_by_question().get(Sandbox._normalize_sandbox_question(question), {})
        mechanism = str(scenario.get("mechanism", ""))
        session_engine = engine
        if mechanism == "federation_aetherspace":
            bundle_access = Sandbox._open_data_bundle()
            try:
                Sandbox._ensure_federation_catalog_aetherspace(engine, bundle_access.path)
            finally:
                if bundle_access.owns_cleanup:
                    shutil.rmtree(bundle_access.path, ignore_errors=True)
        elif mechanism == "federation_mapping_confirm":
            bundle_access = Sandbox._open_data_bundle()
            try:
                Sandbox._apply_bundled_federation_mappings(engine, bundle_access.path)
            finally:
                if bundle_access.owns_cleanup:
                    shutil.rmtree(bundle_access.path, ignore_errors=True)
        with session_engine.session(mode=mode or "writer") as session:
            yield cast(PipelineSession, session)

    @staticmethod
    @contextmanager
    def _preset_offline_handle_cm(
        engine_cls: type[Any],
        *,
        preset: SandboxPreset | str,
        restricted_consumer: bool = False,
        include: SchemaInclude = SchemaInclude.TABLES,
        artifacts_dir: str | None = None,
        cleanup_artifacts: bool = True,
        declaration_file: str | None = None,
        members: Mapping[str, str] | None = None,
        federation_context: FederationContext | None = None,
    ) -> Iterator[SandboxHandle]:
        """Internal corpus entry: build a preset-scoped offline handle without public preset parameters."""
        self_created_artifacts = artifacts_dir is None
        owned_artifacts = cleanup_artifacts and self_created_artifacts
        if preset == "federation":
            sandbox = Sandbox(
                artifacts_dir=artifacts_dir,
                cleanup=False,
                auto_seed=False,
            )
            try:
                yield Sandbox._create_federation_offline_sandbox(
                    engine_cls,
                    sandbox=sandbox,
                    owned_artifacts=owned_artifacts,
                    declaration_file=declaration_file,
                    members=members,
                    federation_context=federation_context,
                )
            finally:
                sandbox.close()
            return

        reset_shared_engine_cache = not self_created_artifacts
        sandbox = Sandbox(
            artifacts_dir=artifacts_dir,
            cleanup=False,
            auto_seed=True,
        )
        try:
            connection = sandbox.connection(SANDBOX_DEFAULT_DATASET_NAME)
            preset_engine_context = (
                EngineContext(allow_objects=CONSUMER_EXAMPLE_NARROW_ALLOW_OBJECTS)
                if restricted_consumer and preset == "consumer_reader"
                else None
            )
            engine = sandbox.create_preset_engine(
                engine_cls,
                preset=preset,
                connection=connection,
                restricted_consumer=False,
                include=include,
                reset_shared_engine_cache=reset_shared_engine_cache,
                engine_context=preset_engine_context,
            )
            yield SandboxHandle(
                engine,
                connection=connection,
                artifacts_dir=sandbox.artifacts_dir,
                owned_artifacts=owned_artifacts,
                owns_connection=False,
                config_path=sandbox._config_path,
                extract_dir=sandbox._extract_dir,
                saved_embedded_runtime_state=sandbox._saved_embedded_runtime_state,
                sandbox=sandbox,
            )
        finally:
            sandbox.close()

    @staticmethod
    def create_offline_sandbox(
        engine_cls: type[Any],
        *,
        artifacts_dir: str | None = None,
        cleanup_artifacts: bool = True,
        deny_columns: frozenset[str] | None = None,
        seed_sql: str | None = None,
        bundle_dir: str | None = None,
        connection: Any | None = None,
        owns_connection: bool | None = None,
        include: SchemaInclude = SchemaInclude.TABLES,
        engine_context: EngineContext | None = None,
        notes_file: str | None = None,
        sql_file: str | None = None,
        llm_config: str | os.PathLike[str] | None = None,
        maintainer_access: bool = False,
    ) -> SandboxHandle:
        """Enter the offline sandbox: in-memory DuckDB with bundled seed data and mock LLM fixtures."""
        if bundle_dir is not None:
            Sandbox._require_maintainer_access(enabled=maintainer_access, hook="bundle_dir")
        if seed_sql is not None:
            Sandbox._require_maintainer_access(enabled=maintainer_access, hook="seed_sql")
        if connection is not None:
            Sandbox._require_maintainer_access(enabled=maintainer_access, hook="connection")
        if bundle_dir is not None or seed_sql is not None:
            MockProvider.reset_mock_provider(clear_literals=True)
            TemplateOps.clear_sandbox_paraphrase_source()
            MockProvider.clear_canonical_schema_literals_cache()

        self_created_artifacts = artifacts_dir is None
        owned_artifacts = cleanup_artifacts and self_created_artifacts
        needs_custom_seed = seed_sql is not None or connection is not None
        sandbox = Sandbox(
            bundle_dir=bundle_dir,
            artifacts_dir=artifacts_dir,
            cleanup=False,
            auto_seed=not needs_custom_seed,
            llm_config=llm_config,
            maintainer_access=maintainer_access,
        )
        if connection is None:
            if needs_custom_seed:
                sandbox.load_dataset(SANDBOX_DEFAULT_DATASET_NAME, seed_sql=seed_sql)
            connection = sandbox.connection(SANDBOX_DEFAULT_DATASET_NAME)
            resolved_owns_connection = False
        else:
            resolved_owns_connection = False if owns_connection is None else owns_connection

        reset_shared_engine_cache = not self_created_artifacts and bundle_dir is None and seed_sql is None
        engine = sandbox.create_preset_engine(
            engine_cls,
            preset="owner_writer",
            connection=connection,
            deny_columns=deny_columns,
            restricted_consumer=False,
            include=include,
            reset_shared_engine_cache=reset_shared_engine_cache,
            engine_context=engine_context,
            notes_file=notes_file,
            sql_file=sql_file,
        )
        return SandboxHandle(
            engine,
            connection=connection,
            artifacts_dir=sandbox.artifacts_dir,
            owned_artifacts=owned_artifacts,
            owns_connection=resolved_owns_connection,
            config_path=sandbox._config_path,
            extract_dir=sandbox._extract_dir,
            saved_embedded_runtime_state=sandbox._saved_embedded_runtime_state,
            sandbox=sandbox,
        )


class _SandboxNormalizeHelpers:
    @staticmethod
    def _normalize_sandbox_question(question: str) -> str:
        return " ".join(question.strip().lower().split())

    @staticmethod
    def _require_sandbox_runtime_for_state(hook: str) -> SandboxRuntimeState:
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is None:
            raise RuntimeError(
                f"{hook} requires an active Sandbox runtime; create Sandbox() or SandboxRuntimeState.bind_sandbox_runtime() first.",
            )
        return runtime


class _SandboxCorpusMethods:
    """Temporary holder; methods are reassigned onto Sandbox below."""

    @staticmethod
    def _read_sandbox_json_member(leaf: str, extract_path: Path | None = None) -> str | None:
        """Read a sandbox JSON member from *extract_path* or the shipped data bundle."""
        if extract_path is not None:
            path = extract_path / leaf
            return path.read_text(encoding="utf-8") if path.is_file() else None
        bundle = Sandbox.data_zip_path()
        if bundle.is_dir():
            path = bundle / leaf
            return path.read_text(encoding="utf-8") if path.is_file() else None
        if not bundle.is_file():
            return None
        with zipfile.ZipFile(bundle) as zf:
            names = set(zf.namelist())
            if not Sandbox._zip_contains_member(names, leaf):
                return None
            for name in zf.namelist():
                if name == leaf or name.endswith(f"/{leaf}"):
                    return zf.read(name).decode("utf-8")
        return None

    @staticmethod
    def _load_sandbox_expectations_catalog(extract_path: Path | None = None) -> SandboxExpectationsCatalog:
        """Load expectation rows keyed by slot_id and by (profile, tier, question)."""
        text = Sandbox._read_sandbox_json_member("sandbox_expectations.json", extract_path)
        if text is None:
            return SandboxExpectationsCatalog(by_slot_id={}, by_context={})
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return SandboxExpectationsCatalog(by_slot_id={}, by_context={})
        slots = payload.get("slots") if isinstance(payload, dict) else None
        if not isinstance(slots, list):
            return SandboxExpectationsCatalog(by_slot_id={}, by_context={})
        by_slot_id: dict[str, dict[str, object]] = {}
        by_context: dict[tuple[str, str, str], dict[str, object]] = {}
        for row in slots:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            expect = row.get("expect")
            if not question or not isinstance(expect, dict):
                continue
            slot_id = str(row.get("slot_id", "")).strip()
            profile = str(row.get("profile", "owner_writer")).strip() or "owner_writer"
            tier = str(row.get("tier", "questions")).strip() or "questions"
            if slot_id:
                by_slot_id[slot_id] = expect
            by_context[(profile, tier, Sandbox._normalize_sandbox_question(question))] = expect
        return SandboxExpectationsCatalog(by_slot_id=by_slot_id, by_context=by_context)

    @staticmethod
    def _ensure_expectations_catalog(extract_path: Path | None = None) -> SandboxExpectationsCatalog:
        global SANDBOX_EXPECTATIONS_CATALOG
        if SANDBOX_EXPECTATIONS_CATALOG is not None:
            return SANDBOX_EXPECTATIONS_CATALOG
        SANDBOX_EXPECTATIONS_CATALOG = Sandbox._load_sandbox_expectations_catalog(extract_path)
        return SANDBOX_EXPECTATIONS_CATALOG

    @staticmethod
    def _expectation_payload_for_context(
        question: str,
        *,
        slot_id: str | None = None,
        profile: str | None = None,
        tier: str | None = None,
        extract_path: Path | None = None,
    ) -> dict[str, object] | None:
        """Resolve an expectation row by slot_id, then (profile, tier, question)."""
        catalog = Sandbox._ensure_expectations_catalog(extract_path)
        if slot_id:
            expect = catalog.by_slot_id.get(slot_id)
            if isinstance(expect, dict):
                return expect
        norm = Sandbox._normalize_sandbox_question(question)
        if profile is not None and tier is not None:
            expect = catalog.by_context.get((profile, tier, norm))
            if isinstance(expect, dict):
                return expect
        return None

    @staticmethod
    def _load_sandbox_expectations_index(extract_path: Path | None = None) -> dict[str, dict[str, object]]:
        """Legacy question-only index (owner_writer + questions rows only)."""
        catalog = Sandbox._load_sandbox_expectations_catalog(extract_path)
        index: dict[str, dict[str, object]] = {}
        for (profile, tier, question_norm), expect in catalog.by_context.items():
            if profile == "owner_writer" and tier == "questions":
                index[question_norm] = expect
        return index

    @staticmethod
    def _faithfulness_from_expect(expect: dict[str, object]) -> SandboxFaithfulnessExpectation:
        must_tables = expect.get("must_tables")
        forbidden = expect.get("forbidden_tables")
        sql_contains = expect.get("sql_contains")
        sql_excludes = expect.get("forbidden_sql_tokens")
        status = expect.get("terminal_status")
        contains_join = expect.get("contains_join")
        status_str: str | None
        if status == "ok" or status is None:
            status_str = None
        else:
            status_str = str(status)
        return SandboxFaithfulnessExpectation(
            status=status_str,
            required_tables=frozenset(str(t) for t in must_tables) if isinstance(must_tables, list) else frozenset(),
            forbidden_tables=frozenset(str(t) for t in forbidden) if isinstance(forbidden, list) else frozenset(),
            sql_contains=tuple(str(t) for t in sql_contains) if isinstance(sql_contains, list) else (),
            sql_excludes=tuple(str(t) for t in sql_excludes) if isinstance(sql_excludes, list) else (),
            contains_join=contains_join if isinstance(contains_join, bool) else None,
        )

    @staticmethod
    def _ensure_faithfulness_index(
        extract_path: Path | None = None,
        *,
        runtime: SandboxRuntimeState | None = None,
    ) -> None:
        active_runtime = runtime or SandboxRuntimeState.current_sandbox_runtime()
        if active_runtime is None:
            return
        if active_runtime.faithfulness_loaded:
            return
        loaded = Sandbox._load_sandbox_expectations_index(extract_path)
        if loaded:
            for question_norm, expect in loaded.items():
                active_runtime.faithfulness_by_question[question_norm] = Sandbox._faithfulness_from_expect(expect)
        else:
            active_runtime.faithfulness_by_question.update(Sandbox._legacy_faithfulness_expectations())
        active_runtime.faithfulness_loaded = True

    @staticmethod
    def _load_sandbox_scenarios_by_question(extract_path: Path | None = None) -> dict[str, dict[str, object]]:
        text = Sandbox._read_sandbox_json_member("sandbox_scenarios.json", extract_path)
        if text is None:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        rows = payload.get("scenarios") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return {}
        index: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            if question:
                index[Sandbox._normalize_sandbox_question(question)] = row
        return index

    @staticmethod
    def _validation_failure_expect(question: str) -> dict[str, object] | None:
        """Expectation rows for validation_failures tier when not in the catalog."""
        norm = Sandbox._normalize_sandbox_question(question)
        known = {Sandbox._normalize_sandbox_question(q) for q in SANDBOX_VALIDATION_FAILURE_QUESTIONS}
        if norm not in known:
            return None
        return {
            "terminal_status": "error",
            "sql_required": False,
            "grain": "none",
            "must_tables": [],
            "must_where": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
            "validation_failure": True,
        }

    @staticmethod
    def _expectation_payload_for_question(
        question: str,
        *,
        slot_id: str | None = None,
        profile: str | None = None,
        tier: str | None = None,
    ) -> dict[str, object] | None:
        resolved = Sandbox._expectation_payload_for_context(
            question,
            slot_id=slot_id,
            profile=profile if profile is not None else "owner_writer",
            tier=tier if tier is not None else "questions",
        )
        if resolved is not None:
            return resolved
        if tier == "validation_failures":
            vf_expect = Sandbox._validation_failure_expect(question)
            if vf_expect is not None:
                return vf_expect
        return Sandbox._load_sandbox_expectations_index().get(Sandbox._normalize_sandbox_question(question))

    @staticmethod
    def _legacy_faithfulness_expectations() -> dict[str, SandboxFaithfulnessExpectation]:
        out: dict[str, SandboxFaithfulnessExpectation] = {}
        for question, spec in SANDBOX_LEGACY_FAITHFULNESS_SPECS.items():
            required_raw = spec.get("required_tables", ())
            required_tables = (
                frozenset(required_raw) if isinstance(required_raw, (frozenset, set, tuple, list)) else frozenset()
            )
            out[question] = SandboxFaithfulnessExpectation(
                status=cast(str | None, spec.get("status")),
                required_tables=required_tables,
                forbidden_tables=frozenset(),
                sql_contains=tuple(cast(tuple[str, ...], spec.get("sql_contains", ()))),
                sql_excludes=tuple(cast(tuple[str, ...], spec.get("sql_excludes", ()))),
                contains_join=cast(bool | None, spec.get("contains_join")),
            )
        return out

    @staticmethod
    def _faithfulness_sql_text(step: object) -> str:
        """Return SQL text used for faithfulness checks, including federation display SQL."""
        sql = str(getattr(step, "sql", "") or "")
        if sql:
            return sql
        bundle = getattr(step, "federated_bundle", None)
        if bundle is not None:
            display_sql = str(getattr(bundle, "display_sql", "") or "")
            if display_sql:
                return display_sql
        return ""

    @staticmethod
    def _faithfulness_table_names(step: object) -> set[str]:
        """Return lowercased base schema table names from intent and SQL for faithfulness checks."""
        sql = Sandbox._faithfulness_sql_text(step)
        intent = getattr(step, "intent", None)
        if isinstance(intent, RuntimeIntent):
            cte_steps = intent.cte_steps or []
            cte_aliases = {(s.cte_name or "").strip() for s in cte_steps if (s.cte_name or "").strip()}
            cte_aliases.update(str(name).strip() for name in (intent.planner_cte_names or []) if str(name).strip())
            base_from_ctes: set[str] = set()
            for cte in cte_steps:
                base_from_ctes.update(t for t in (cte.tables or []) if t and t not in cte_aliases)
            main_base = {t for t in (intent.tables or []) if t and t not in cte_aliases}
            names: set[str] = set(base_from_ctes) | set(main_base)
            if sql:
                names.update(
                    t
                    for t in Dialect.sql_tables_referenced(sql, sqlglot_dialect=Dialect.active_sqlglot_dialect())
                    if t and t not in cte_aliases
                )
            return {str(t).lower() for t in names if t}
        summary = getattr(step, "intent_summary", None)
        tables = list(getattr(intent, "tables", None) or getattr(summary, "tables", None) or [])
        names = {str(t).lower() for t in tables if t}
        if sql:
            names.update(
                str(t).lower()
                for t in Dialect.sql_tables_referenced(sql, sqlglot_dialect=Dialect.active_sqlglot_dialect())
                if t
            )
        return names

    @staticmethod
    def _faithfulness_mismatch(step: object, expectation: SandboxFaithfulnessExpectation) -> str | None:
        status = getattr(step, "status", None)
        if expectation.status is not None and status != expectation.status:
            return f"status expected {expectation.status!r}, got {status!r}"
        sql = Sandbox._faithfulness_sql_text(step)
        sql_lower = sql.lower()
        used_tables = Sandbox._faithfulness_table_names(step)
        missing = sorted(t for t in expectation.required_tables if t.lower() not in used_tables)
        if missing:
            return f"missing required tables: {', '.join(missing)}"
        forbidden_hit = sorted(t for t in expectation.forbidden_tables if t.lower() in used_tables)
        if forbidden_hit:
            return f"forbidden tables used: {', '.join(forbidden_hit)}"
        for token in expectation.sql_contains:
            if token.lower() not in sql_lower:
                return f"missing sql token {token!r}"
        for token in expectation.sql_excludes:
            if token.lower() in sql_lower:
                return f"forbidden sql token {token!r}"
        if expectation.contains_join is not None:
            has_join = " join " in f" {sql_lower} "
            if expectation.contains_join and not has_join:
                return "expected JOIN in SQL"
            if not expectation.contains_join and has_join:
                return "unexpected JOIN in SQL"
        return None

    @staticmethod
    def faithfulness_expectation_for_question(
        question: str,
        *,
        slot_id: str | None = None,
        profile: str | None = None,
        tier: str | None = None,
    ) -> SandboxFaithfulnessExpectation | None:
        """Return the deterministic faithfulness descriptor for *question*, if any."""
        expect = Sandbox._expectation_payload_for_question(
            question,
            slot_id=slot_id,
            profile=profile,
            tier=tier,
        )
        if expect is not None:
            return Sandbox._faithfulness_from_expect(expect)
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None and runtime.faithfulness_loaded:
            return runtime.faithfulness_by_question.get(Sandbox._normalize_sandbox_question(question))
        Sandbox._ensure_faithfulness_index(runtime=runtime)
        if runtime is not None:
            return runtime.faithfulness_by_question.get(Sandbox._normalize_sandbox_question(question))
        return None

    @staticmethod
    def check_sandbox_faithfulness(
        step: object,
        question: str,
        *,
        slot_id: str | None = None,
        profile: str | None = None,
        tier: str | None = None,
    ) -> str | None:
        """Return a mismatch detail when *step* fails the faithfulness descriptor for *question*."""
        expectation = Sandbox.faithfulness_expectation_for_question(
            question,
            slot_id=slot_id,
            profile=profile,
            tier=tier,
        )
        if expectation is None:
            return None
        return Sandbox._faithfulness_mismatch(step, expectation)

    @staticmethod
    def question_ok(
        step: object,
        question: str,
        *,
        slot_id: str | None = None,
        profile: str | None = None,
        tier: str | None = None,
    ) -> bool:
        """Return True when *step* satisfies sandbox expectations for *question*."""
        if not getattr(step, "done", False):
            return False
        expect = Sandbox._expectation_payload_for_question(
            question,
            slot_id=slot_id,
            profile=profile,
            tier=tier,
        )
        if expect is not None:
            terminal = expect.get("terminal_status")
            sql_required = expect.get("sql_required", True)
            if terminal == "invalid_question":
                return getattr(step, "status", None) == "invalid_question"
            if terminal == "permission_denied":
                return getattr(step, "status", None) == "permission_denied"
            if terminal == "error" or expect.get("validation_failure"):
                err = str(getattr(step, "error", "") or "").lower()
                msg = str(getattr(step, "message", "") or "").lower()
                combined = f"{err} {msg}"
                if "restricted" in combined or "rejected" in combined:
                    return True
                status = getattr(step, "status", None)
                if status in ("restricted", "permission_denied", "schema_invalid", "invalid_question"):
                    return True
                if "schema_invalid" in combined or "permission" in combined:
                    return True
                if not sql_required:
                    return getattr(step, "sql", None) is None or bool(getattr(step, "error", None))
            if not sql_required:
                return getattr(step, "sql", None) is None or bool(getattr(step, "error", None))
        if question in SANDBOX_TOUR_EXPECT_NO_SQL:
            return getattr(step, "sql", None) is None or bool(getattr(step, "error", None))
        if question in SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL:
            return getattr(step, "sql", None) is None or bool(getattr(step, "error", None))
        if question in SANDBOX_VALIDATION_FAILURE_QUESTIONS:
            err = str(getattr(step, "error", "") or "").lower()
            msg = str(getattr(step, "message", "") or "").lower()
            combined = f"{err} {msg}"
            if "restricted" in combined or "rejected" in combined:
                return True
            status = getattr(step, "status", None)
            if status in ("restricted", "permission_denied"):
                return True
            if "schema_invalid" in combined or "permission" in combined:
                return True
        expectation = Sandbox.faithfulness_expectation_for_question(
            question,
            slot_id=slot_id,
            profile=profile,
            tier=tier,
        )
        if expectation is not None and expectation.status == "invalid_question":
            return getattr(step, "status", None) == "invalid_question"
        status = getattr(step, "status", None)
        if status is not None and status != "ok":
            return False
        if not bool(getattr(step, "sql", None)) and not Sandbox._faithfulness_sql_text(step):
            return False
        if (
            Sandbox.check_sandbox_faithfulness(
                step,
                question,
                slot_id=slot_id,
                profile=profile,
                tier=tier,
            )
            is not None
        ):
            return False
        return True

    @staticmethod
    def _validate_trace_path() -> Path:
        return Path(tempfile.gettempdir()) / "aetherdialect_validate_trace.txt"

    @staticmethod
    def _reset_validate_trace_file() -> None:
        path = Sandbox._validate_trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    @staticmethod
    def _append_validate_trace_row(
        row: dict[str, str],
        step: object | None,
        *,
        captured_logs: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        scenario_id = f"{row.get('kind', '')}:{row.get('tier', '')}".strip(":")
        trace = build_session_step_trace(
            scenario_id=scenario_id,
            question=str(row.get("name", "")),
            step=step,
            error=error or str(row.get("detail", "") or ""),
            captured_logs=captured_logs,
        )
        append_failure_trace(trace, Sandbox._validate_trace_path())

    @staticmethod
    def _validate_question_slot(
        engine_cls: type[Any],
        question: str,
        *,
        tier: str,
        preset: SandboxPreset | str = SandboxPreset.OWNER_WRITER,
        mode: str | None = None,
        apply_overrides: bool = False,
        restricted_consumer: bool = False,
        slot_id: str | None = None,
        sandbox_factory: Callable[..., Any] | None = None,
        include: SchemaInclude = SchemaInclude.TABLES,
    ) -> dict[str, str] | None:
        """Run one offline slot and return a failure row when expectations are not met."""
        step: object | None = None
        captured_logs: list[str] = []
        try:
            with pipeline_capture(auto_responses=["y"]) as capture:
                if sandbox_factory is not None:
                    sandbox_cm = sandbox_factory(
                        preset=preset,
                        restricted_consumer=restricted_consumer,
                        include=include,
                    )
                else:
                    sandbox_cm = Sandbox._preset_offline_handle_cm(
                        engine_cls,
                        preset=preset,
                        restricted_consumer=restricted_consumer,
                        include=include,
                    )
                with sandbox_cm as sb:
                    if apply_overrides:
                        sb.apply_bundled_schema_overrides()
                    session_cm = sb.engine.session(mode=mode) if mode else sb.engine.session()
                    with session_cm as session:
                        step = session.accept_until_done(question)
                captured_logs = list(capture.get("logs", []))
            if not Sandbox.question_ok(step, question, slot_id=slot_id, profile=preset, tier=tier):
                faith_detail = Sandbox.check_sandbox_faithfulness(
                    step,
                    question,
                    slot_id=slot_id,
                    profile=preset,
                    tier=tier,
                )
                row = {
                    "kind": "faithfulness" if faith_detail else "question",
                    "tier": tier,
                    "name": question,
                    "detail": faith_detail or "expectation not met",
                }
                Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs)
                return row
        except Exception as exc:
            row = {"kind": "question", "tier": tier, "name": question, "detail": str(exc)}
            Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs, error=str(exc))
            return row
        return None

    @staticmethod
    def _validate_federation_slot(
        engine_cls: type[Any],
        question: str,
        *,
        slot_id: str | None = None,
        mode: str | None = None,
        sandbox_factory: Callable[..., Any] | None = None,
    ) -> dict[str, str] | None:
        """Run one federation offline slot and return a failure row when expectations are not met."""
        step: object | None = None
        captured_logs: list[str] = []
        try:
            with pipeline_capture(auto_responses=["y"]) as capture:
                if sandbox_factory is not None:
                    sandbox_cm = sandbox_factory()
                else:
                    sandbox_cm = Sandbox._preset_offline_handle_cm(engine_cls, preset="federation")
                with sandbox_cm as sb:
                    with Sandbox.federation_scenario_session(sb.engine, question, mode=mode) as session:
                        step = session.accept_until_done(question)
                captured_logs = list(capture.get("logs", []))
            if not Sandbox.question_ok(step, question, slot_id=slot_id, profile="owner_writer", tier="federation"):
                faith_detail = Sandbox.check_sandbox_faithfulness(
                    step,
                    question,
                    slot_id=slot_id,
                    profile="owner_writer",
                    tier="federation",
                )
                row = {
                    "kind": "faithfulness" if faith_detail else "question",
                    "tier": "federation",
                    "name": question,
                    "detail": faith_detail or "expectation not met",
                }
                Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs)
                return row
        except Exception as exc:
            row = {"kind": "question", "tier": "federation", "name": question, "detail": str(exc)}
            Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs, error=str(exc))
            return row
        return None

    @staticmethod
    def _sandbox_federation_questions() -> list[str]:
        """Return federation scenario questions from the bundled sandbox scenarios file."""
        questions: list[str] = []
        for row in Sandbox._load_sandbox_scenarios_by_question().values():
            if str(row.get("recipe", "")) != "federation":
                continue
            question = str(row.get("question", "")).strip()
            if question:
                questions.append(question)
        return questions

    @staticmethod
    def _validate_validation_failure_slot(engine_cls: type[Any], question: str) -> dict[str, str] | None:
        scenario = Sandbox._load_sandbox_scenarios_by_question().get(Sandbox._normalize_sandbox_question(question), {})
        mechanism = str(scenario.get("mechanism", ""))
        if mechanism == "bundled_overrides_hide_staff_ssn":
            return Sandbox._validate_question_slot(
                engine_cls,
                question,
                tier="validation_failures",
                apply_overrides=True,
            )
        if mechanism == "schema_validation_failure":
            return Sandbox._validate_question_slot(
                engine_cls,
                question,
                tier="validation_failures",
                preset="consumer_reader",
                mode="reader",
                restricted_consumer=True,
            )
        return Sandbox._validate_question_slot(engine_cls, question, tier="validation_failures")

    @staticmethod
    def _validate_direct_reuse_pair(engine_cls: type[Any]) -> dict[str, str] | None:
        canonical = "How many rentals happened in 2025?"
        paraphrase = "How many rentals happened in 2026?"
        step: SessionStep | None = None
        captured_logs: list[str] = []
        try:
            with pipeline_capture(auto_responses=["y"]) as capture:
                with Sandbox.create_offline_sandbox(engine_cls) as sb:
                    with sb.engine.session() as session:
                        session.accept_until_done(canonical)
                        step = session.ask(paraphrase)
                        if step.kind != SESSION_KIND_AWAITING_SQL_CONFIRM:
                            captured_logs = list(capture.get("logs", []))
                            row = {
                                "kind": "question",
                                "tier": "reuse",
                                "name": paraphrase,
                                "detail": f"expected kind {SESSION_KIND_AWAITING_SQL_CONFIRM!r}, got {step.kind!r}",
                            }
                            Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs)
                            return row
                        while not step.done:
                            if step.reply_shape == "yes_no":
                                step = session.step("y")
                            elif step.reply_shape == "free_text":
                                step = session.step("ok")
                            else:
                                break
                captured_logs = list(capture.get("logs", []))
            if not step.done or not Sandbox.question_ok(step, paraphrase, tier="reuse"):
                row = {
                    "kind": "question",
                    "tier": "reuse",
                    "name": paraphrase,
                    "detail": "direct reuse follow-through did not complete",
                }
                Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs)
                return row
        except Exception as exc:
            row = {"kind": "question", "tier": "reuse", "name": paraphrase, "detail": str(exc)}
            Sandbox._append_validate_trace_row(row, step, captured_logs=captured_logs, error=str(exc))
            return row
        return None

    @staticmethod
    def validate_sandbox_corpus(engine_cls: type[Any], *, smoke: bool = False) -> list[dict[str, str]]:
        """Run offline sandbox validation and return failure rows."""
        Sandbox._reset_validate_trace_file()
        failures: list[dict[str, str]] = []
        for question in Sandbox.sandbox_questions():
            row = Sandbox._validate_question_slot(engine_cls, question, tier="questions")
            if row is not None:
                failures.append(row)

        for question in Sandbox._sandbox_build_section("validation_failures"):
            row = Sandbox._validate_validation_failure_slot(engine_cls, question)
            if row is not None:
                failures.append(row)

        for question in Sandbox.sandbox_questions():
            row = Sandbox._validate_question_slot(
                engine_cls,
                question,
                tier="consumer_reader",
                preset="consumer_reader",
                mode="reader",
            )
            if row is not None:
                failures.append(row)

        for question in Sandbox._sandbox_federation_questions():
            row = Sandbox._validate_federation_slot(engine_cls, question)
            if row is not None:
                failures.append(row)

        if smoke:
            feedback_demo = Sandbox.sandbox_feedback_demo()
            anchor = str(feedback_demo.get("anchor_question", "")).strip()
            rejection = str(feedback_demo.get("allowed_rejection_text", "")).strip()
            if anchor and rejection:
                try:
                    with Sandbox.create_offline_sandbox(engine_cls) as sb:
                        with sb.engine.session() as session:
                            step = session.ask(anchor)
                            if not step.done and step.reply_shape == "yes_no":
                                step = session.step("n")
                            if not step.done and step.reply_shape == "free_text":
                                step = session.step(rejection)
                            while not step.done and step.reply_shape == "yes_no":
                                step = session.step("y")
                    if not step.done:
                        failures.append(
                            {"kind": "feedback", "tier": "", "name": rejection, "detail": "incomplete"},
                        )
                except Exception as exc:
                    failures.append(
                        {"kind": "feedback", "tier": "", "name": rejection, "detail": str(exc)},
                    )
            return failures

        reuse_row = Sandbox._validate_direct_reuse_pair(engine_cls)
        if reuse_row is not None:
            failures.append(reuse_row)

        for recipe in SANDBOX_RECIPES:
            try:
                Sandbox._execute_sandbox_recipe(recipe, engine_cls)
            except Exception as exc:
                failures.append({"kind": "recipe", "tier": "", "name": recipe, "detail": str(exc)})

        feedback_demo = Sandbox.sandbox_feedback_demo()
        anchor = str(feedback_demo.get("anchor_question", "")).strip()
        rejection = str(feedback_demo.get("allowed_rejection_text", "")).strip()
        if anchor and rejection:
            try:
                with Sandbox.create_offline_sandbox(engine_cls) as sb:
                    with sb.engine.session() as session:
                        step = session.ask(anchor)
                        if not step.done and step.reply_shape == "yes_no":
                            step = session.step("n")
                        if not step.done and step.reply_shape == "free_text":
                            step = session.step(rejection)
                        while not step.done and step.reply_shape == "yes_no":
                            step = session.step("y")
                if not step.done:
                    failures.append(
                        {"kind": "feedback", "tier": "", "name": rejection, "detail": "incomplete"},
                    )
            except Exception as exc:
                failures.append(
                    {"kind": "feedback", "tier": "", "name": rejection, "detail": str(exc)},
                )
        return failures

    @staticmethod
    def _dir_has_member_path(root: Path, leaf: str) -> bool:
        path = root / leaf
        if path.is_file() or path.is_dir():
            return True
        return any(path.is_file() and path.as_posix().endswith(leaf) for path in root.rglob("*"))

    @staticmethod
    def _zip_has_member_path(names: set[str], leaf: str) -> bool:
        if Sandbox._zip_contains_member(names, leaf):
            return True
        prefix = leaf.rstrip("/") + "/"
        return any(name == leaf or name.startswith(prefix) for name in names)

    @staticmethod
    def _check_optional_baseline_members(
        *,
        contains_member: Callable[[str], bool],
        bundle_label: str,
    ) -> list[str]:
        """When a views baseline directory is present, require its schema graph artifact."""
        issues: list[str] = []
        for baseline_dir in SANDBOX_DOCTOR_OPTIONAL_BASELINE_DIRS:
            if not contains_member(baseline_dir):
                continue
            for member in SANDBOX_DOCTOR_OPTIONAL_BASELINE_MEMBERS:
                if member.startswith(baseline_dir) and not contains_member(member):
                    issues.append(f"Missing {member} under {bundle_label}")
        return issues

    @staticmethod
    def sandbox_doctor() -> list[str]:
        """Return human-readable problems; empty list means the sandbox bundle looks healthy."""
        issues: list[str] = []
        bundle_path = Sandbox.data_zip_path()
        if bundle_path.is_dir():
            for required in SANDBOX_DOCTOR_REQUIRED_MEMBERS:
                if not Sandbox._dir_contains_member(bundle_path, required):
                    issues.append(f"Missing {required} under {bundle_path}")
            issues.extend(
                Sandbox._check_optional_baseline_members(
                    contains_member=lambda leaf: Sandbox._dir_has_member_path(bundle_path, leaf),
                    bundle_label=str(bundle_path),
                ),
            )
        elif bundle_path.is_file():
            with zipfile.ZipFile(bundle_path) as zf:
                names = set(zf.namelist())
                for required in SANDBOX_DOCTOR_REQUIRED_MEMBERS:
                    if not Sandbox._zip_contains_member(names, required):
                        issues.append(f"Missing {required} inside {bundle_path}")
                issues.extend(
                    Sandbox._check_optional_baseline_members(
                        contains_member=lambda leaf: Sandbox._zip_has_member_path(names, leaf),
                        bundle_label=str(bundle_path),
                    ),
                )
        else:
            issues.append(f"Missing data bundle: {bundle_path}")
        return issues

    @staticmethod
    def _parse_questions_file(path: str) -> dict[str, list[str]]:
        tiers: dict[str, list[str]] = {
            "questions": [],
            "validation_failures": [],
            "feedback_samples": [],
        }
        current = "questions"
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header = stripped.lstrip("#").strip().lower().replace(" ", "_")
                if header in tiers:
                    current = header
                continue
            if current:
                tiers[current].append(stripped)
        return tiers

    @staticmethod
    def _load_sandbox_questions_sections() -> dict[str, list[str]]:
        bundle_access = Sandbox._open_data_bundle()
        try:
            return Sandbox._parse_questions_file(str(bundle_access.path / "questions.txt"))
        except OSError as exc:
            raise ConfigError(f"sandbox questions bundle unavailable: {exc}") from exc
        finally:
            if bundle_access.owns_cleanup:
                shutil.rmtree(bundle_access.path, ignore_errors=True)

    @staticmethod
    def sandbox_questions() -> list[str]:
        """Return curated natural-language sandbox practice questions."""
        return list(Sandbox._load_sandbox_questions_sections()["questions"])

    @staticmethod
    def _sandbox_build_section(section: SandboxBuildSection | str) -> list[str]:
        """Return build-only question file sections (not part of ``sandbox_questions()``)."""
        return list(Sandbox._load_sandbox_questions_sections()[section])

    @staticmethod
    def _load_sandbox_catalog() -> dict[str, object]:
        bundle_access = Sandbox._open_data_bundle()
        try:
            catalog_path = bundle_access.path / "sandbox_catalog.json"
            if not catalog_path.is_file():
                raise ConfigError(f"missing sandbox_catalog.json under {bundle_access.path}")
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ConfigError("sandbox_catalog.json must be a JSON object")
            return payload
        except OSError as exc:
            raise ConfigError(f"sandbox catalog bundle unavailable: {exc}") from exc
        finally:
            if bundle_access.owns_cleanup:
                shutil.rmtree(bundle_access.path, ignore_errors=True)

    @staticmethod
    def sandbox_catalog() -> dict[str, object]:
        """Return the bundled user-facing sandbox discovery catalog."""
        return dict(Sandbox._load_sandbox_catalog())

    @staticmethod
    def sandbox_paraphrase_pairs() -> list[dict[str, object]]:
        """Return canonical→paraphrase wordings from the bundled sandbox catalog."""
        rows = Sandbox._load_sandbox_catalog().get("paraphrase_pairs")
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]

    @staticmethod
    def sandbox_validation_failure_demo() -> list[dict[str, str]]:
        """Return example validation-failure questions and short descriptions."""
        rows = Sandbox._load_sandbox_catalog().get("validation_failure_demo")
        if not isinstance(rows, list):
            return []
        out: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            out.append(
                {
                    "question": question,
                    "description": str(row.get("description", "")).strip(),
                },
            )
        return out

    @staticmethod
    def sandbox_feedback_demo() -> dict[str, str]:
        """Return the scripted reject/retry feedback demo (anchor + allowed rejection text)."""
        row = Sandbox._load_sandbox_catalog().get("feedback_demo")
        if not isinstance(row, dict):
            return {}
        anchor = str(row.get("anchor_question", "")).strip()
        rejection = str(row.get("allowed_rejection_text", "")).strip()
        if not anchor or not rejection:
            return {}
        return {
            "anchor_question": anchor,
            "allowed_rejection_text": rejection,
            "description": str(row.get("description", "")).strip(),
        }

    @staticmethod
    def _sandbox_doctor_verbose() -> list[str]:
        """Maintainer-only verbose corpus health check."""
        issues = Sandbox.sandbox_doctor()
        bundle_path = Sandbox.data_zip_path()
        if bundle_path.is_dir():
            if not Sandbox._dir_contains_member(bundle_path, "migration_demo/artifacts_v1/schema_graph.json.gz"):
                if not any(
                    path.is_file() and "migration_demo/artifacts_v1" in path.as_posix()
                    for path in bundle_path.rglob("*")
                ):
                    issues.append(f"Missing migration_demo/artifacts_v1 under {bundle_path}")
            fixture_path = bundle_path / "fixtures" / "rental_shop_mock.json"
            if fixture_path.is_file():
                try:
                    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
                    entries = raw.get("fixtures", raw) if isinstance(raw, dict) else raw
                    if isinstance(entries, list):
                        if len(entries) < SANDBOX_MIN_FIXTURE_COUNT:
                            issues.append(f"fixtures/rental_shop_mock.json too few fixtures ({len(entries)})")
                        intent_count = sum(
                            1 for item in entries if isinstance(item, dict) and str(item.get("task", "")) == "intent"
                        )
                        if intent_count < SANDBOX_MIN_INTENT_FIXTURE_COUNT:
                            issues.append(f"fixtures/rental_shop_mock.json too few intent fixtures ({intent_count})")
                except (AttributeError, TypeError, ValueError, OSError):
                    pass
        elif bundle_path.is_file():
            with zipfile.ZipFile(bundle_path) as zf:
                names = set(zf.namelist())
                if not any("migration_demo/artifacts_v1" in name for name in names):
                    issues.append(f"Missing migration_demo/artifacts_v1 inside {bundle_path}")
        return issues

    @staticmethod
    def assert_sandbox_complete(engine_cls: type[Any]) -> None:
        """Validate the shipped sandbox corpus and raise when any slot fails."""
        doctor = Sandbox.sandbox_doctor()
        if doctor:
            raise RuntimeError("sandbox_doctor failed: " + "; ".join(doctor))
        failures = Sandbox.validate_sandbox_corpus(engine_cls)
        if failures:
            lines = [f"[{row['kind']}] {row.get('tier', '')} {row['name'][:70]}: {row['detail']}" for row in failures]
            raise RuntimeError(
                f"{len(failures)} sandbox validation failures:\n" + "\n".join(lines),
            )

    @staticmethod
    def _write_queue_path(artifacts_dir: str | Any) -> Path:
        """Resolve ``write_queue.jsonl`` under the engine storage directory."""
        if hasattr(artifacts_dir, "_write_queue_path"):
            return cast(Path, artifacts_dir._write_queue_path)
        if hasattr(artifacts_dir, "write_queue_path"):
            return cast(Path, artifacts_dir.write_queue_path)
        if hasattr(artifacts_dir, "_artifacts_dir"):
            root = str(artifacts_dir._artifacts_dir)
        else:
            root = str(artifacts_dir)
        return Path(root) / WRITE_QUEUE_FILENAME

    @staticmethod
    def _count_utf8_lines(path: Path) -> int:
        """Return the number of lines in a UTF-8 text file."""
        with path.open(encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    @staticmethod
    def _apply_bundled_schema_overrides(t2s: Any, handle: SandboxHandle | None = None) -> None:
        """Copy bundled override JSON to artifacts_dir and call ``apply_schema_overrides`` on the engine."""
        bundle_access = Sandbox._open_data_bundle()
        try:
            source = bundle_access.path / "schema_overrides_demo.json"
            target = Path(t2s._artifacts_dir) / SCHEMA_OVERRIDES_DEFAULT_FILENAME
            shutil.copyfile(source, target)
            if handle is not None:
                handle.register_cwd_sidecar(target)
            t2s.apply_schema_overrides()
        finally:
            if bundle_access.owns_cleanup:
                shutil.rmtree(bundle_access.path, ignore_errors=True)

    @staticmethod
    def _stage_migration_corpus_variant(
        extract_path: Path,
        *,
        variant: str,
        artifacts_dir: str,
    ) -> _MigrationVariantState:
        demo_root = extract_path / variant
        artifacts_src = demo_root / "artifacts_v1"
        map_path = demo_root / "schema_migration_map.json"
        seed_path = extract_path / "rental_shop_seed.sql"
        if not artifacts_src.is_dir() or not map_path.is_file() or not seed_path.is_file():
            raise FileNotFoundError(f"Migration corpus variant assets incomplete under {demo_root}.")

        work = Path(tempfile.mkdtemp(prefix="aetherdialect_migration_variant_"))
        try:
            post_sql = work / "rental_shop_post_migration.sql"
            post_sql.write_text(Sandbox._post_migration_seed_sql(seed_path, map_path), encoding="utf-8")
            shutil.copytree(artifacts_src, Path(artifacts_dir), dirs_exist_ok=True)
            engine_dir = Sandbox._sandbox_memory_engine_dir(artifacts_dir)
            Sandbox._copy_baseline_cache_files(artifacts_src, engine_dir)
            connection = Sandbox._load_memory_connection(str(post_sql))
        finally:
            shutil.rmtree(work, ignore_errors=True)

        notes_file = extract_path / "rental_shop_notes.txt"
        sql_file = extract_path / "rental_shop.sql"
        schema_context = Sandbox._owner_writer_schema_context(
            notes_file=str(notes_file) if notes_file.is_file() else None,
            sql_file=str(sql_file) if sql_file.is_file() else None,
        )
        return _MigrationVariantState(
            connection=connection,
            schema_context=schema_context,
            map_path=map_path,
        )

    @staticmethod
    def _prepare_migration_corpus_variant(
        extract_path: Path,
        *,
        variant: str,
        artifacts_dir: str,
        config_path: str | None,
        engine_cls: type[Any],
    ) -> tuple[Any, Any]:
        """Return an owner engine and connection for a bundled migration corpus variant."""
        state = Sandbox._stage_migration_corpus_variant(extract_path, variant=variant, artifacts_dir=artifacts_dir)
        execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(state.connection)
        migrated = engine_cls.apply_migration_map(
            str(state.map_path),
            engine_context=state.schema_context,
            artifacts_dir=artifacts_dir,
            config_file=config_path,
            execution_engine=execution_engine,
            native_connection=state.connection,
            role=SchemaRole.OWNER,
        )
        migrated._sandbox_mode = True
        return migrated, state.connection

    @staticmethod
    def _run_sandbox_migration_demo(engine_cls: type[Any], *, verbose: bool = True) -> None:
        """Demonstrate migration map application on a toy column rename."""
        bundle_access = Sandbox._open_data_bundle()
        extract = bundle_access.path
        work = Path(tempfile.mkdtemp(prefix="aetherdialect_migration_demo_"))
        try:
            demo_root = extract / "migration_demo"
            artifacts_src = demo_root / "artifacts_v1"
            map_path = demo_root / "schema_migration_map.json"
            seed_path = extract / "rental_shop_seed.sql"
            if not artifacts_src.is_dir() or not map_path.is_file() or not seed_path.is_file():
                raise FileNotFoundError(f"Migration demo assets incomplete under {demo_root}.")

            post_sql = work / "rental_shop_post_migration.sql"
            post_sql.write_text(Sandbox._post_migration_seed_sql(seed_path, map_path), encoding="utf-8")

            artifacts_dir = str(work / "artifacts")
            shutil.copytree(artifacts_src, artifacts_dir)

            connection = Sandbox._load_memory_connection(str(post_sql))
            execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
            notes_file = extract / "rental_shop_notes.txt"
            notes_arg = str(notes_file) if notes_file.is_file() else None
            sql_file = extract / "rental_shop.sql"
            sql_arg = str(sql_file) if sql_file.is_file() else None
            schema_context = Sandbox._owner_writer_schema_context(notes_file=notes_arg, sql_file=sql_arg)
            config_file = Sandbox._write_sandbox_toml(fixtures_file=Sandbox._fixtures_path(extract))

            try:
                engine_cls(
                    schema_context,
                    artifacts_dir=artifacts_dir,
                    config_file=config_file,
                    execution_engine=execution_engine,
                    native_connection=connection,
                    role=SchemaRole.OWNER,
                )
                if verbose:
                    print("  Init completed without MigrationPendingError (soft refresh path).")
            except MigrationPendingError as exc:
                if verbose:
                    print(f"  MigrationPendingError (expected): {exc}")

            t2s = engine_cls.apply_migration_map(
                str(map_path),
                engine_context=schema_context,
                artifacts_dir=artifacts_dir,
                config_file=config_file,
                execution_engine=execution_engine,
                native_connection=connection,
                role=SchemaRole.OWNER,
            )
            t2s._sandbox_mode = True
            if verbose:
                print("  Migration map applied; asking post-migration question.")
            practice_q = Sandbox.sandbox_questions()
            post_q = practice_q[0] if practice_q else "How many films are in the Rental Shop catalog?"
            with t2s.session() as session:
                step = Sandbox._accept_until_done(session, post_q)
            if verbose:
                print(f"  Post-migration step.done={step.done} status={step.status!r}")
            try:
                Path(config_file).unlink(missing_ok=True)
            except OSError:
                pass
        finally:
            shutil.rmtree(work, ignore_errors=True)
            if bundle_access.owns_cleanup:
                shutil.rmtree(extract, ignore_errors=True)

    @staticmethod
    def _print_goal(msg: str) -> None:
        print(f"[sandbox] {msg}")

    @staticmethod
    def _practice_questions() -> list[str]:
        return Sandbox.sandbox_questions()

    @staticmethod
    def _recipe_chat_basics(handle: SandboxHandle) -> None:
        Sandbox._print_goal("chat_basics — first tour question accept path")
        qs = Sandbox._practice_questions()
        if not qs:
            return
        with handle.engine.session() as session:
            step = Sandbox._accept_until_done(session, qs[0])
        print(f"  done={step.done} kind={step.kind!r} ok={step.done and step.kind == SESSION_KIND_RESULT}")

    @staticmethod
    def _recipe_rejections(handle: SandboxHandle) -> None:
        Sandbox._print_goal("rejections — intent reject with feedback")
        qs = Sandbox._practice_questions()
        if len(qs) < 2:
            return
        with handle.engine.session() as session:
            step = session.ask(qs[1])
            if not step.done and step.reply_shape == "yes_no":
                step = session.step("n")
            if not step.done and step.reply_shape == "free_text":
                rejection = Sandbox.sandbox_feedback_demo().get("allowed_rejection_text", "")
                step = session.step(str(rejection) if rejection else "wrong intent")
            while not step.done and step.reply_shape == "yes_no":
                step = session.step("y")

    @staticmethod
    def _recipe_reader_writer(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("reader_writer — consumer enqueues, owner drains")
        shared = tempfile.mkdtemp(prefix="aetherdialect_rw_")
        try:
            print(f"  shared artifacts_dir={shared}")
            tour = Sandbox._practice_questions()
            q = tour[0] if tour else "How many films are in the Rental Shop catalog?"
            with Sandbox._preset_offline_handle_cm(
                engine_cls, preset="consumer_reader", artifacts_dir=shared
            ) as reader:
                reader.apply_bundled_schema_overrides()
                queue = Sandbox._write_queue_path(reader.engine)
                assert queue.is_file(), "reader should enqueue learning events"
                print(f"  write_queue lines={Sandbox._count_utf8_lines(queue)}")
            with Sandbox._preset_offline_handle_cm(engine_cls, preset="owner_writer", artifacts_dir=shared) as writer:
                with writer.engine.session(mode="writer") as session:
                    session.ask(q)
                queue = Sandbox._write_queue_path(writer.engine)
            remaining = Sandbox._count_utf8_lines(queue) if queue.is_file() else 0
            print(f"  write_queue remaining after drain={remaining}")
        finally:
            shutil.rmtree(shared, ignore_errors=True)

    @staticmethod
    def _recipe_overrides(handle: SandboxHandle) -> None:
        Sandbox._print_goal("overrides — owner applies bundled schema_overrides_demo.json")
        handle.apply_bundled_schema_overrides()

    @staticmethod
    def _recipe_migration(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("migration — predetermined v1→v2 rename demo")
        Sandbox._run_sandbox_migration_demo(engine_cls)

    @staticmethod
    def _recipe_validation_failures(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("validation_failures — schema_invalid and consumer permission_denied")
        fails = Sandbox._sandbox_build_section("validation_failures")
        with Sandbox.create_offline_sandbox(engine_cls) as owner:
            if fails:
                with owner.engine.session() as session:
                    step = Sandbox._accept_until_done(session, fails[0])
                print(f"  schema_invalid done={step.done} kind={step.kind!r}")
        with Sandbox._preset_offline_handle_cm(
            engine_cls,
            preset="consumer_reader",
            restricted_consumer=True,
        ) as consumer:
            with consumer.engine.session(mode="reader") as session:
                step = Sandbox._accept_until_done(session, "How many items are there?")
            print(f"  permission_denied status={step.status!r}")

    @staticmethod
    def _recipe_maintenance(handle: SandboxHandle) -> None:
        Sandbox._print_goal("maintenance — show_config and table inventory")
        snap = handle.engine.show_config()
        print(f"  config lines={len(snap.text.splitlines())}")
        meta = handle.engine.export_metadata()
        table_count = int(meta.get("table_count") or 0)
        print(f"  table_count={table_count}")
        assert table_count == 34, f"Rental Shop sandbox expects 34 tables, got {table_count}"

    @staticmethod
    def _recipe_errors(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("errors — MockFixtureMissing, OwnerOnly, SessionActive")
        with Sandbox.create_offline_sandbox(engine_cls) as sb:
            with sb.engine.session() as session:
                step = session.ask("This question is not in the offline corpus at all.")
                if step.error and "No mock fixture" in step.error:
                    print(f"  mock fixture missing: {step.error[:80]}...")
            with sb.engine.session(mode="reader") as session:
                session.ask(Sandbox.sandbox_questions()[0])
                try:
                    session.ask("second question while suspended")
                except SessionActiveError:
                    print("  SessionActiveError raised")
        with Sandbox._preset_offline_handle_cm(engine_cls, preset="consumer_reader") as sb:
            try:
                with sb.engine.session(mode="writer"):
                    pass
            except OwnerOnlyOperationError:
                print("  OwnerOnlyOperationError on consumer writer")

    @staticmethod
    def _recipe_column_security(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("column_security — deny_columns on customer.email")
        deny = frozenset({"customer.email"})
        with Sandbox.create_offline_sandbox(engine_cls, deny_columns=deny) as sb:
            with sb.engine.session() as session:
                step = Sandbox._accept_until_done(session, "Who are our top 5 customers by total payment?")
            codes = [d.code for d in step.diagnostics]
            print(f"  diagnostics sample={codes[:5]}")

    @staticmethod
    def _recipe_partition_pruning(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("partition_pruning — synthetic rental.rental_date partition_columns")
        with Sandbox.create_offline_sandbox(engine_cls) as sb:
            sg = sb.engine._schema_graph
            rental_tbl = sg.tables.get("rental")
            if rental_tbl and "rental_date" in rental_tbl.columns and not rental_tbl.partition_columns:
                rental_tbl.partition_columns = ["rental_date"]
            rental = sg.tables.get("rental")
            partition_cols = list(rental.partition_columns) if rental else []
            print(f"  rental.partition_columns={partition_cols}")
            intent = RuntimeIntent(
                tables=["rental"],
                grain="scalar",
                select_cols=[],
                group_by_cols=[],
                order_by_cols=[],
                where=PredicateGroup(
                    op="and",
                    predicates=(
                        WhereParam(
                            left_expr=NormalizedExpr.from_column("rental.rental_date"),
                            op="=",
                            param_key="p1",
                            raw_value=None,
                        ),
                    ),
                ),
                param_values={"p1": "2023-07-15"},
            )
            sql = "SELECT COUNT(*) FROM rental WHERE rental.rental_date = '2023-07-15'"
            dialect = DuckDBDialect.__new__(DuckDBDialect)
            out = dialect.inject_pruning_predicates(sql, schema=sg, intent=intent)
            print(f"  injected={out != sql}")
            if out != sql:
                print(f"  finalized_sql={out}")

    @staticmethod
    def _recipe_views(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("views — local view relations (include='views' scope)")
        with Sandbox.create_offline_sandbox(engine_cls) as sb:
            extract_path = Path(sb._extract_dir) if sb._extract_dir else Sandbox.data_zip_path()
            Sandbox._apply_rental_shop_views(sb.connection, extract_path=extract_path)
            rows = sb.connection.execute(
                "SELECT store_id, total_revenue FROM store_revenue_v ORDER BY total_revenue DESC LIMIT 3"
            ).fetchall()
            print(f"  store_revenue_v sample={rows}")
            view_names = [
                row[0]
                for row in sb.connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'VIEW'"
                ).fetchall()
            ]
            print(f"  views={sorted(view_names)}")
            print(
                "  Bundled rental_shop_views.sql is in data.zip; use EngineContext(include='views') to reflect views only."
            )

    @staticmethod
    def _recipe_aetherspace(engine_cls: type[Any]) -> None:
        Sandbox._print_goal("aetherspace — catalog space scoped to item/film/category")
        bundle_access = Sandbox._open_data_bundle()
        try:
            notes_path = bundle_access.path / "sandbox_space_catalog_notes.txt"
            notes_arg = str(notes_path) if notes_path.is_file() else None
            with Sandbox.create_offline_sandbox(engine_cls) as sb:
                catalog = SpaceContext(
                    tables=frozenset({"item", "film", "category", "item_category"}),
                    columns=frozenset(),
                )
                sb.engine.aetherspace("catalog", space_context=catalog, notes_file=notes_arg)
                with sb.engine.session(space="catalog") as session:
                    step = session.accept_until_done("How many films are in the catalog?")
                    print(f"  in_scope ok={step.done and step.kind == SESSION_KIND_RESULT and not step.error}")
                with sb.engine.session(space="catalog") as session:
                    step = session.accept_until_done("What is total revenue by store?")
                    blocked = step.sql is None or bool(step.error)
                    print(f"  out_of_scope blocked={blocked}")
        finally:
            if bundle_access.owns_cleanup:
                shutil.rmtree(bundle_access.path, ignore_errors=True)

    @staticmethod
    def _recipe_full_session(handle: SandboxHandle) -> None:
        Sandbox._print_goal("full_session — suspend loop on tour Q2")
        qs = Sandbox._practice_questions()
        if len(qs) < 2:
            return
        with handle.engine.session() as session:
            step = session.ask(qs[1])
            while not step.done:
                if step.reply_shape == "yes_no":
                    step = session.step("y")
                elif step.reply_shape == "free_text":
                    step = session.step("ok")
                else:
                    break
            print(f"  terminal status={step.status!r} intent_summary={bool(step.intent_summary)}")

    @staticmethod
    def _with_handle(fn: Any, engine_cls: type[Any]) -> None:
        with Sandbox.create_offline_sandbox(engine_cls) as handle:
            fn(handle)

    @staticmethod
    def _recipe_dispatch(engine_cls: type[Any]) -> dict[str, Any]:
        return {
            "chat_basics": lambda: Sandbox._with_handle(Sandbox._recipe_chat_basics, engine_cls),
            "rejections": lambda: Sandbox._with_handle(Sandbox._recipe_rejections, engine_cls),
            "reader_writer": lambda: Sandbox._recipe_reader_writer(engine_cls),
            "overrides": lambda: Sandbox._with_handle(Sandbox._recipe_overrides, engine_cls),
            "migration": lambda: Sandbox._recipe_migration(engine_cls),
            "validation_failures": lambda: Sandbox._recipe_validation_failures(engine_cls),
            "maintenance": lambda: Sandbox._with_handle(Sandbox._recipe_maintenance, engine_cls),
            "errors": lambda: Sandbox._recipe_errors(engine_cls),
            "column_security": lambda: Sandbox._recipe_column_security(engine_cls),
            "full_session": lambda: Sandbox._with_handle(Sandbox._recipe_full_session, engine_cls),
            "partition_pruning": lambda: Sandbox._recipe_partition_pruning(engine_cls),
            "views": lambda: Sandbox._recipe_views(engine_cls),
            "aetherspace": lambda: Sandbox._recipe_aetherspace(engine_cls),
        }

    @staticmethod
    def _execute_sandbox_recipe(name: str, engine_cls: type[Any]) -> None:
        """Run a single internal sandbox validation recipe."""
        if name not in SANDBOX_RECIPES:
            raise ValueError(f"Unknown sandbox recipe {name!r}; expected one of {SANDBOX_RECIPES}")
        dispatch = Sandbox._recipe_dispatch(engine_cls).get(name)
        if dispatch is None:
            raise ValueError(f"Unknown recipe {name!r}")
        dispatch()

    @staticmethod
    def _execute_sandbox_tour(engine_cls: type[Any], *, interactive: bool = False) -> None:
        """Run every internal sandbox validation recipe in curriculum order."""
        del interactive
        for name in SANDBOX_RECIPES:
            Sandbox._execute_sandbox_recipe(name, engine_cls)


class Sandbox(_SandboxPendingMethods, _SandboxNormalizeHelpers, _SandboxCorpusMethods):
    """Authoring environment: seeded in-memory databases, shared artifacts, and production-shaped engines."""

    __slots__ = (
        "_artifacts_dir",
        "_bundle_access",
        "_closed",
        "_config_path",
        "_datasets",
        "_extract_dir",
        "_extract_path",
        "_llm_config",
        "_llm_mode",
        "_maintainer_access",
        "_owned_artifacts",
        "_runtime",
        "_runtime_token",
        "_saved_embedded_runtime_state",
    )

    @staticmethod
    def _require_maintainer_access(*, enabled: bool, hook: str) -> None:
        if not enabled:
            raise ConfigError(
                f"{hook} leaves the closed sandbox world; pass maintainer_access=True to Sandbox or "
                "create_offline_sandbox when you intend to use maintainer hooks.",
            )

    @staticmethod
    def _sandbox_llm_mode_from_config(config_path: str | None) -> SandboxLlmMode:
        if config_path is None:
            return SandboxLlmMode.MOCK
        payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        llm_block = payload.get("llm")
        provider = "mock"
        if isinstance(llm_block, dict):
            provider = str(llm_block.get("provider", "mock")).strip().lower() or "mock"
        if provider == "mock":
            return SandboxLlmMode.MOCK
        return SandboxLlmMode.NETWORK

    @staticmethod
    def _iter_rental_shop_view_statements(views_sql_path: Path) -> list[str]:
        text = views_sql_path.read_text(encoding="utf-8")
        return [stmt.strip() for stmt in text.split(";") if stmt.strip()]

    @staticmethod
    def _apply_rental_shop_views(connection: Any, *, extract_path: Path) -> None:
        views_path = extract_path / "rental_shop_views.sql"
        if not views_path.is_file():
            raise FileNotFoundError(f"Missing bundled views DDL: {views_path}")
        for name in RENTAL_SHOP_VIEW_NAMES:
            connection.execute(f'DROP VIEW IF EXISTS "{name}"')
        for stmt in Sandbox._iter_rental_shop_view_statements(views_path):
            connection.execute(stmt)

    @staticmethod
    def _paraphrase_registry_from_catalog_path(extract_path: Path) -> dict[str, list[str]]:
        """Build canonical→paraphrase lists from the bundled sandbox catalog."""
        catalog_path = extract_path / "sandbox_catalog.json"
        if not catalog_path.is_file():
            return {}
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        pairs = payload.get("paraphrase_pairs")
        if not isinstance(pairs, list):
            return {}
        out: dict[str, list[str]] = {}
        for row in pairs:
            if not isinstance(row, dict):
                continue
            canonical = str(row.get("canonical", "")).strip()
            if not canonical:
                continue
            raw = row.get("paraphrases")
            if not isinstance(raw, list):
                continue
            out[canonical] = [str(item).strip() for item in raw if str(item).strip()]
        return out

    @staticmethod
    def _sandbox_root() -> Path:
        return Path(str(resources.files("aetherdialect") / "sandbox"))

    @staticmethod
    def data_zip_path(*, maintainer_access: bool = False) -> Path:
        override = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP", "").strip()
        if override:
            Sandbox._require_maintainer_access(
                enabled=maintainer_access,
                hook="AETHERDIALECT_SANDBOX_DATA_ZIP",
            )
            return Path(override)
        return Sandbox._sandbox_root() / "data.zip"

    @staticmethod
    def _fixtures_path(extract_dir: Path) -> str:
        return str(extract_dir / "fixtures" / "rental_shop_mock.json")

    @staticmethod
    def _zip_contains_member(names: set[str], leaf: str) -> bool:
        if leaf in names:
            return True
        return any(n.endswith(leaf) for n in names)

    @staticmethod
    def _dir_contains_member(root: Path, leaf: str) -> bool:
        if (root / leaf).is_file():
            return True
        return any(path.is_file() and path.as_posix().endswith(leaf) for path in root.rglob("*"))

    @staticmethod
    def _require_sandbox_bundle(bundle: Path) -> None:
        if not SandboxBundlePolicy.REQUIRE_BUNDLE:
            return
        if bundle.exists():
            return
        raise ConfigError(
            "The offline sandbox corpus is not bundled in this build. "
            "Connect a real engine with AetherEngine/EngineContext instead."
        )

    @staticmethod
    def fixtures_corpus_text() -> str:
        """Return the shipped mock fixture JSON from the active sandbox bundle."""
        bundle = Sandbox.data_zip_path()
        Sandbox._require_sandbox_bundle(bundle)
        if bundle.is_dir():
            fixture_path = bundle / "fixtures" / "rental_shop_mock.json"
            if fixture_path.is_file():
                return fixture_path.read_text(encoding="utf-8")
            return '{"version": 1, "fixtures": []}'
        with zipfile.ZipFile(bundle) as zf:
            for name in zf.namelist():
                if name.endswith("fixtures/rental_shop_mock.json"):
                    return zf.read(name).decode("utf-8")
        return '{"version": 1, "fixtures": []}'

    @staticmethod
    def _open_data_bundle(*, dest: Path | None = None, maintainer_access: bool = False) -> _DataBundleAccess:
        """Return sandbox bundle files; extract zip to a temp dir unless env points at a directory."""
        bundle = Sandbox.data_zip_path(maintainer_access=maintainer_access)
        Sandbox._require_sandbox_bundle(bundle)
        if bundle.is_dir():
            return _DataBundleAccess(path=bundle, owns_cleanup=False)
        owns_cleanup = dest is None
        target = dest if dest is not None else Path(tempfile.mkdtemp(prefix="aetherdialect_sandbox_extract_"))
        try:
            if dest is not None:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle) as zf:
                zf.extractall(target)
        except Exception:
            if owns_cleanup:
                shutil.rmtree(target, ignore_errors=True)
            raise
        return _DataBundleAccess(path=target, owns_cleanup=owns_cleanup)

    @staticmethod
    def _extract_data_bundle(*, dest: Path | None = None) -> _DataBundleAccess:
        """Extract shipped ``data.zip`` to a fresh temp directory (or wipe *dest* first)."""
        return Sandbox._open_data_bundle(dest=dest)

    @staticmethod
    def _bundle_path(extract_dir: Path, *parts: str) -> str:
        return str(extract_dir.joinpath(*parts))

    @staticmethod
    def _post_migration_seed_sql(seed_path: Path, migration_map_path: Path) -> str:
        """Apply column renames from the bundled migration map to seed SQL text."""
        text = seed_path.read_text(encoding="utf-8")
        payload = json.loads(migration_map_path.read_text(encoding="utf-8"))
        for rename in payload.get("column_renames", []):
            if not isinstance(rename, dict):
                continue
            from_col = str(rename.get("from", "")).strip()
            to_col = str(rename.get("to", "")).strip()
            if not from_col or not to_col or from_col == to_col:
                continue
            text = text.replace(f" {from_col} ", f" {to_col} ")
            text = text.replace(f"({from_col},", f"({to_col},")
            text = text.replace(f" {from_col},", f" {to_col},")
            text = text.replace(f" {from_col})", f" {to_col})")
        return text

    @staticmethod
    def _load_bundled_schema_literals(extract_path: Path) -> dict[str, str] | None:
        """Return owner/consumer schema literals shipped beside the sandbox bundle."""
        path = extract_path / SANDBOX_SCHEMA_LITERALS_FILENAME
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        owner = str(payload.get("owner", "")).strip()
        consumer = str(payload.get("consumer", "")).strip()
        if not owner or not consumer:
            return None
        return {"owner": owner, "consumer": consumer}

    @staticmethod
    def _load_bundled_interpret_domain(extract_path: Path) -> dict[str, Any] | None:
        path = extract_path / SANDBOX_INTERPRET_DOMAIN_FILENAME
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _pin_bundled_schema_literals(extract_path: Path) -> None:
        """Pin mock-fixture lookup keys from the active sandbox bundle."""
        MockProvider.pin_mock_fixture_keys_from_bundle(extract_path)

    @staticmethod
    def _split_sql_statements(sql: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        for line in sql.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            buf.append(line)
            if stripped.endswith(";"):
                parts.append("\n".join(buf))
                buf = []
        if buf:
            parts.append("\n".join(buf))
        return parts

    @staticmethod
    def _load_memory_connection(seed_sql: str) -> Any:
        """Create ``:memory:`` DuckDB, execute seed SQL, return native connection."""
        require_driver("duckdb")
        duckdb = importlib.import_module("duckdb")
        sql = Path(seed_sql).read_text(encoding="utf-8")
        connection = duckdb.connect(":memory:")
        for statement in Sandbox._split_sql_statements(sql):
            if statement.strip():
                connection.execute(statement)
        return connection

    @staticmethod
    def _accept_until_done(session: Any, question: str) -> Any:
        """Auto-confirm yes/no and free-text suspends until the turn ends."""
        return session.accept_until_done(question)

    @staticmethod
    def _owner_writer_schema_context(
        *,
        notes_file: str | None = None,
        sql_file: str | None = None,
    ) -> EngineContext:
        return EngineContext(
            notes_file=notes_file,
            sql_file=sql_file,
        )

    @staticmethod
    def _consumer_execution_allow_objects() -> frozenset[str]:
        return CONSUMER_ALLOW_OBJECTS

    @staticmethod
    def _consumer_reader_schema_context(
        *,
        notes_file: str | None = None,
        sql_file: str | None = None,
    ) -> EngineContext:
        return Sandbox._owner_writer_schema_context(notes_file=notes_file, sql_file=sql_file)

    @staticmethod
    def _effective_consumer_allow_objects(master: EngineContext) -> frozenset[str]:
        owner_allow = Sandbox._consumer_execution_allow_objects()
        if master.allow_objects:
            user_allow = frozenset(master.allow_objects)
            extra = user_allow - owner_allow
            if extra:
                raise ConfigError(
                    f"consumer allow_objects {sorted(extra)!r} exceed sandbox owner scope",
                )
            return user_allow
        return owner_allow

    @staticmethod
    def _consumer_schema_context_for_construction(master: EngineContext) -> EngineContext:
        """Narrow *master* to the consumer execution scope before engine construction."""
        eff_allow = Sandbox._effective_consumer_allow_objects(master)
        return replace(master, allow_objects=eff_allow)

    @staticmethod
    def _apply_sandbox_consumer_execution_scope(engine: Any) -> None:
        master = engine._runtime_config.engine_context
        eff_allow = Sandbox._effective_consumer_allow_objects(master)
        execution_ctx = replace(
            master,
            allow_objects=eff_allow,
        )
        engine._runtime_config = replace(
            engine._runtime_config,
            execution_context=execution_ctx,
            engine_context=execution_ctx,
        )
        engine._consumer_visible_objects = frozenset(eff_allow)

    @staticmethod
    def _read_notes_file_text(notes_file: str) -> str:
        path = os.path.expanduser(str(notes_file).strip())
        if not os.path.isfile(path):
            raise ConfigError(f"notes_file not found: {notes_file!r}")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _bundled_catalog_space_notes_text(extract_path: Path) -> str | None:
        path = extract_path / "sandbox_space_catalog_notes.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    @staticmethod
    def validate_sandbox_aetherspace_notes_pairing(
        space_context: SpaceContext,
        notes_file: str,
        *,
        extract_path: Path,
    ) -> None:
        """Require bundled catalog notes content and the documented table pairing."""
        notes_content = Sandbox._read_notes_file_text(notes_file)
        bundled = Sandbox._bundled_catalog_space_notes_text(extract_path)
        if bundled is None or notes_content != bundled:
            raise ConfigError(
                "custom notes_file content is not accepted on arbitrary sandbox spaces; "
                "create the space without notes, or use the documented catalog demo pairing "
                "(sandbox_space_catalog_notes.txt with tables item, film, category, item_category)",
            )
        if frozenset(space_context.tables) != SANDBOX_CATALOG_SPACE_TABLES:
            raise ConfigError(
                "custom notes_file content is not accepted on arbitrary sandbox spaces; "
                "create the space without notes, or use the documented catalog demo pairing "
                f"(sandbox_space_catalog_notes.txt with tables {sorted(SANDBOX_CATALOG_SPACE_TABLES)!r})",
            )

    @staticmethod
    def _release_engine_runtime_handles(engine: Any) -> None:
        engine._execution_engine = None
        engine._store = None
        engine._templates = {}
        engine._rejected = {}

    @staticmethod
    def _unlink_artifact_lock_files(artifacts_dir: str) -> None:
        root = Path(artifacts_dir)
        if not root.is_dir():
            return
        for pattern in ("*.lock", "*.__write.lock"):
            for path in root.rglob(pattern):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _remove_tree(path: str) -> None:
        def _on_rm_error(func: Any, p: str, _exc_info: Any) -> None:
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except OSError:
                pass

        shutil.rmtree(path, onerror=_on_rm_error)

    @staticmethod
    def _snapshot_embedded_runtime_state() -> tuple[Any | None, Any | None, str]:
        """Capture embedded-engine globals before sandbox init mutates runtime ClassVars."""
        return (
            DuckDBRuntimeConfig.NATIVE_CONNECTION,
            SQLiteRuntimeConfig.NATIVE_CONNECTION,
            str(DuckDBRuntimeConfig.DATABASE_PATH),
        )

    @staticmethod
    def _restore_embedded_runtime_state(saved: tuple[Any | None, Any | None, str]) -> None:
        """Restore embedded-engine globals saved by :meth:`_snapshot_embedded_runtime_state`."""
        duckdb_saved, sqlite_saved, duckdb_path_saved = saved
        if duckdb_saved is not None:
            DuckDBRuntimeConfig.attach_connection(duckdb_saved)
        else:
            DuckDBRuntimeConfig.clear_attached_connection()
        if sqlite_saved is not None:
            SQLiteRuntimeConfig.attach_connection(sqlite_saved)
        else:
            SQLiteRuntimeConfig.clear_attached_connection()
        DuckDBRuntimeConfig.DATABASE_PATH = duckdb_path_saved

    @staticmethod
    def _aether_engine_cls() -> type[Any]:
        mod = importlib.import_module("aetherdialect.aetherdialect")
        return cast(type[Any], mod.AetherEngine)

    @staticmethod
    def _mark_sandbox_managed_connection(connection: Any, sandbox: Sandbox) -> None:
        try:
            setattr(connection, SANDBOX_CONNECTION_HOST_ATTR, sandbox)
        except AttributeError:
            _CONNECTION_SANDBOX_HOSTS[id(connection)] = sandbox

    @staticmethod
    def sandbox_host_for_connection(connection: Any) -> Sandbox | None:
        host = getattr(connection, SANDBOX_CONNECTION_HOST_ATTR, None)
        if isinstance(host, Sandbox):
            return host
        return _CONNECTION_SANDBOX_HOSTS.get(id(connection))

    @staticmethod
    def require_sandbox_adoption(engine: Any) -> None:
        """Raise when *engine* uses a sandbox connection but has not been adopted."""
        connection = getattr(engine, "_native_connection", None)
        if connection is None:
            return
        if Sandbox.sandbox_host_for_connection(connection) is not None and not getattr(engine, "_sandbox_mode", False):
            raise ConfigError(
                "This engine uses a Sandbox connection but has not been adopted; "
                "call sandbox.adopt(engine) before session().",
            )

    @staticmethod
    def _sandbox_questions_from_path(extract_path: Path) -> tuple[str, ...]:
        return tuple(Sandbox._parse_questions_file(str(extract_path / "questions.txt"))["questions"])

    @classmethod
    def bundled_dataset_seed(cls, name: str) -> str:
        """Return the bundled seed filename for a sandbox dataset *name*."""
        if name == SANDBOX_DEFAULT_DATASET_NAME:
            return "rental_shop_seed.sql"
        for member_name, seed_name in SANDBOX_BUNDLED_MEMBER_SEEDS:
            if member_name == name:
                return seed_name
        raise KeyError(name)

    def __init__(
        self,
        *,
        llm_config: str | os.PathLike[str] | None = None,
        artifacts_dir: str | None = None,
        bundle_dir: str | None = None,
        cleanup: bool = True,
        auto_seed: bool = True,
        maintainer_access: bool = False,
    ) -> None:
        self._closed = False
        self._maintainer_access = bool(maintainer_access)
        self._runtime = SandboxRuntimeState()
        self._runtime_token = SandboxRuntimeState.bind_sandbox_runtime(self._runtime)
        self._llm_config = str(llm_config) if llm_config is not None else None
        if bundle_dir is not None:
            Sandbox._require_maintainer_access(enabled=self._maintainer_access, hook="bundle_dir")
            self._extract_path = Path(bundle_dir)
            self._extract_dir = ""
            self._bundle_access = _DataBundleAccess(path=self._extract_path, owns_cleanup=False)
        else:
            self._bundle_access = Sandbox._open_data_bundle(maintainer_access=self._maintainer_access)
            self._extract_path = self._bundle_access.path
            self._extract_dir = str(self._extract_path) if self._bundle_access.owns_cleanup else ""
        self._saved_embedded_runtime_state = Sandbox._snapshot_embedded_runtime_state()
        self_created_artifacts = artifacts_dir is None
        if self_created_artifacts:
            artifacts_dir = tempfile.mkdtemp(prefix="aetherdialect_sandbox_artifacts_")
        assert artifacts_dir is not None
        self._artifacts_dir = artifacts_dir
        self._owned_artifacts = cleanup and self_created_artifacts
        self._datasets: dict[str, _SandboxDataset] = {}
        self._config_path: str | None
        if llm_config is not None:
            config_path = Path(llm_config).expanduser().resolve()
            if not config_path.is_file():
                raise ConfigError(f"llm_config not found: {config_path}")
            self._config_path = str(config_path)
            SandboxRuntimeState.set_sandbox_recorded_corpus_question_count(None)
        else:
            self._config_path = Sandbox._write_sandbox_toml(fixtures_file=Sandbox._fixtures_path(self._extract_path))
            SandboxRuntimeState.set_sandbox_recorded_corpus_question_count(len(self.questions()))
        self._llm_mode = Sandbox._sandbox_llm_mode_from_config(self._config_path)
        Sandbox._pin_bundled_schema_literals(self._extract_path)
        MockProvider.reset_mock_provider(clear_literals=True)
        TemplateOps.clear_sandbox_paraphrase_source()
        MockProvider.clear_canonical_schema_literals_cache()
        if auto_seed:
            self._seed_default_datasets()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Sandbox is closed; create a new Sandbox() instance.")

    def _seed_default_datasets(self) -> None:
        if SANDBOX_DEFAULT_DATASET_NAME not in self._datasets:
            self.load_dataset(SANDBOX_DEFAULT_DATASET_NAME)
        for member_name, _seed_name in SANDBOX_BUNDLED_MEMBER_SEEDS:
            if member_name in self._datasets:
                continue
            seed_path = self._extract_path / _seed_name
            if seed_path.is_file():
                self.load_dataset(member_name)

    def _bundled_notes_and_sql(self) -> tuple[str | None, str | None]:
        notes_file = Sandbox._bundle_path(self._extract_path, "rental_shop_notes.txt")
        notes = notes_file if Path(notes_file).is_file() else None
        sql_file = Sandbox._bundle_path(self._extract_path, "rental_shop.sql")
        sql_arg = sql_file if Path(sql_file).is_file() else None
        return notes, sql_arg

    def _resolve_engine_context(
        self,
        engine_context: EngineContext | None,
        *,
        include: SchemaInclude,
    ) -> EngineContext:
        notes, sql_arg = self._bundled_notes_and_sql()
        if engine_context is None:
            ctx = Sandbox._owner_writer_schema_context(notes_file=notes, sql_file=sql_arg)
        else:
            ctx = engine_context
            if notes and not ctx.notes_file:
                ctx = replace(ctx, notes_file=notes)
            if sql_arg and not ctx.sql_file:
                ctx = replace(ctx, sql_file=sql_arg)
        return replace(ctx, include=include)

    def _seed_role_baseline(self, *, role: SchemaRole, include: SchemaInclude) -> None:
        preset: SandboxPreset = (
            SandboxPreset.CONSUMER_READER if role == SchemaRole.CONSUMER else SandboxPreset.OWNER_WRITER
        )
        baseline = Sandbox._baseline_dir_for_preset(self._extract_path, preset, include=include)
        if baseline is None:
            return
        engine_dir = Sandbox._sandbox_memory_engine_dir(self._artifacts_dir)
        graph_path = engine_dir / "schema_graph.json.gz"
        if graph_path.is_file() and role == "owner":
            return
        if graph_path.is_file() and role == "consumer":
            for name in SANDBOX_BASELINE_CACHE_FILES:
                target = engine_dir / name
                if target.is_file():
                    target.unlink()
        engine_dir.mkdir(parents=True, exist_ok=True)
        Sandbox._copy_baseline_cache_files(baseline, engine_dir)
        Sandbox._seed_bundled_aetherspaces(self._extract_path, engine_dir)

    def load_dataset(
        self,
        name: str,
        *,
        sql_file: str | None = None,
        seed_sql: str | None = None,
    ) -> str:
        """Seed an in-memory DuckDB database under a bundled dataset *name*."""
        self._ensure_open()
        dataset_name = str(name).strip()
        if not dataset_name:
            raise ConfigError("dataset name must be non-empty")
        if dataset_name in self._datasets:
            raise ConfigError(f"dataset {dataset_name!r} is already loaded")
        custom_seed = seed_sql or sql_file
        if custom_seed:
            Sandbox._require_maintainer_access(enabled=self._maintainer_access, hook="load_dataset seed paths")
            seed_path = custom_seed
        else:
            try:
                bundled_seed = Sandbox.bundled_dataset_seed(dataset_name)
            except KeyError as exc:
                raise ConfigError(
                    f"unknown sandbox dataset {dataset_name!r}; bundled datasets: "
                    f"{sorted(SANDBOX_BUNDLED_DATASET_NAMES)}",
                ) from exc
            seed_path = str(self._extract_path / bundled_seed)
        if not Path(seed_path).is_file():
            raise ConfigError(f"seed SQL not found: {seed_path}")
        connection = Sandbox._load_memory_connection(seed_path)
        self._datasets[dataset_name] = _SandboxDataset(connection=connection, owns_connection=True)
        Sandbox._mark_sandbox_managed_connection(connection, self)
        return dataset_name

    @property
    def datasets(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(self._datasets.keys())

    @property
    def artifacts_dir(self) -> str:
        self._ensure_open()
        return self._artifacts_dir

    @property
    def config_file(self) -> str | None:
        """Path to the temporary mock-provider TOML written for this environment."""
        self._ensure_open()
        return self._config_path

    @property
    def llm_mode(self) -> SandboxLlmMode:
        """Return whether this sandbox replays bundled fixtures or calls a live LLM provider."""
        self._ensure_open()
        return self._llm_mode

    @property
    def uses_network(self) -> bool:
        """Return True when this sandbox is configured to make live LLM network calls."""
        self._ensure_open()
        return self._llm_mode == "network"

    @property
    def maintainer_access(self) -> bool:
        """Return whether maintainer-only closed-world escape hooks are enabled."""
        self._ensure_open()
        return self._maintainer_access

    def connection(self, name: str = SANDBOX_DEFAULT_DATASET_NAME) -> Any:
        self._ensure_open()
        dataset = self._datasets.get(name)
        if dataset is None:
            raise ConfigError(f"unknown sandbox dataset {name!r}; loaded datasets: {sorted(self._datasets)}")
        return dataset.connection

    def engine(
        self,
        engine_context: EngineContext | None = None,
        *,
        role: SchemaRole = SchemaRole.OWNER,
        include: SchemaInclude = SchemaInclude.TABLES,
    ) -> Any:
        """Build an :class:`~aetherdialect.AetherEngine` on the default dataset using *engine_context*."""
        self._ensure_open()
        if SANDBOX_DEFAULT_DATASET_NAME not in self._datasets:
            raise ConfigError(f"default dataset {SANDBOX_DEFAULT_DATASET_NAME!r} is not loaded")
        connection = self.connection(SANDBOX_DEFAULT_DATASET_NAME)
        if include == "views":
            Sandbox._apply_rental_shop_views(connection, extract_path=self._extract_path)
        self._seed_role_baseline(role=role, include=include)
        notes, sql_arg = self._bundled_notes_and_sql()
        resolved_notes, resolved_sql, init_notices = Sandbox._resolve_sandbox_notes_and_sql(
            engine_context=engine_context,
            notes_file=engine_context.notes_file if engine_context is not None else None,
            sql_file=engine_context.sql_file if engine_context is not None else None,
            bundled_notes=notes,
            bundled_sql=sql_arg,
            extract_path=self._extract_path,
        )
        role_preset: SandboxPreset = (
            SandboxPreset.CONSUMER_READER if role == SchemaRole.CONSUMER else SandboxPreset.OWNER_WRITER
        )
        if engine_context is None:
            schema_context = Sandbox._schema_context_for_preset(
                role_preset,
                notes_file=resolved_notes,
                sql_file=resolved_sql,
                deny_columns=None,
                restricted_consumer=False,
                include=include,
            )
        else:
            schema_context = replace(
                engine_context,
                include=include,
                notes_file=resolved_notes,
                sql_file=resolved_sql,
            )
        if role == "consumer":
            schema_context = Sandbox._consumer_schema_context_for_construction(schema_context)
        trust_baseline = Sandbox._sandbox_trusts_bundled_baseline(
            preset=role_preset,
            schema_context=schema_context,
            bundled_notes=notes,
            bundled_sql=sql_arg,
            deny_columns=None,
            restricted_consumer=False,
            include=include,
            engine_context=engine_context,
            notes_file=engine_context.notes_file if engine_context is not None else None,
            sql_file=engine_context.sql_file if engine_context is not None else None,
        )
        execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
        engine = Sandbox._aether_engine_cls()(
            schema_context,
            artifacts_dir=self._artifacts_dir,
            config_file=self._config_path,
            execution_engine=execution_engine,
            native_connection=connection,
            role=role,
            trust_bundled_baseline=trust_baseline,
            init_notices=init_notices,
        )
        if role == "consumer":
            Sandbox._apply_sandbox_consumer_execution_scope(engine)
        self._attach_sandbox_runtime_to_engine(engine)
        self.adopt(engine)
        return engine

    def federation(
        self,
        federation_id: str,
        *,
        declaration_file: str | None = None,
        members: Mapping[str, str] | None = None,
        context: FederationContext | None = None,
    ) -> Any:
        """Build an :class:`~aetherdialect.AetherFederation` over named sandbox datasets."""
        self._ensure_open()
        config_path = self._config_path
        if config_path is None:
            raise ConfigError("sandbox configuration is unavailable")
        decl_path = Path(declaration_file) if declaration_file else self._extract_path / FEDERATION_DECLARATION_FILENAME
        if not decl_path.is_file():
            raise ConfigError(f"federation declaration not found: {decl_path}")
        authored_manifest, _ = parse_federation_declaration(json.loads(decl_path.read_text(encoding="utf-8")))
        if str(federation_id).strip() != authored_manifest.federation_id:
            raise ConfigError(
                f"federation_id {federation_id!r} does not match declaration {authored_manifest.federation_id!r}",
            )
        baseline = Sandbox._baseline_dir_for_preset(self._extract_path, "federation")
        if baseline is not None:
            Sandbox._seed_federation_composite_baseline(
                baseline=baseline,
                artifacts_dir=self._artifacts_dir,
                federation_id=authored_manifest.federation_id,
            )
        notes, sql_arg = self._bundled_notes_and_sql()
        member_specs: tuple[tuple[str, str], ...]
        if members is not None:
            if self._maintainer_access:
                member_specs = tuple((str(name), str(seed)) for name, seed in members.items())
            else:
                resolved: list[tuple[str, str]] = []
                for member_name, dataset_name in members.items():
                    try:
                        bundled_seed = Sandbox.bundled_dataset_seed(str(dataset_name))
                    except KeyError as exc:
                        raise ConfigError(
                            f"federation member {member_name!r} must name a bundled dataset; got {dataset_name!r}",
                        ) from exc
                    resolved.append((str(member_name), str(self._extract_path / bundled_seed)))
                member_specs = tuple(resolved)
        else:
            member_specs = Sandbox._default_federation_member_specs(self._extract_path)
        engines: dict[str, Any] = {}
        provisional_bindings: dict[str, FederationSourceBinding] = {}
        for member_name, _member_seed in member_specs:
            provisional_bindings[member_name] = FederationSourceBinding(
                source_id=member_name,
                engine="duckdb",
                connection=member_name,
                context="master",
                role=SchemaRole.OWNER,
            )
        for member_name, member_seed in member_specs:
            if not Path(member_seed).is_file():
                raise ConfigError(f"member seed SQL not found for {member_name!r}: {member_seed}")
            allow_tables = Sandbox._federation_partition_tables_from_bundle(self._extract_path, member_name)
            member_engine, _member_conn = Sandbox._create_federation_member_engine(
                Sandbox._aether_engine_cls(),
                member_name=member_name,
                seed_path=member_seed,
                extract_path=self._extract_path,
                artifacts_dir=self._artifacts_dir,
                config_path=config_path,
                notes_file=notes,
                sql_file=sql_arg,
                allow_tables=allow_tables,
                baseline=baseline,
                binding=provisional_bindings[member_name],
            )
            engines[member_name] = member_engine
        for member_name, member_engine in engines.items():
            binding_from_member_engine(member_name, member_engine)
        federation = Sandbox._aether_federation_cls()(
            authored_manifest.federation_id,
            members=engines,
            declaration_file=str(decl_path),
            context=context,
            artifacts_dir=self._artifacts_dir,
        )
        federation._sandbox_mode = True
        for member_engine in engines.values():
            self._attach_sandbox_runtime_to_engine(member_engine)
        TemplateOps.set_sandbox_paraphrase_source(Sandbox._paraphrase_registry_from_catalog_path(self._extract_path))
        return federation

    def _attach_sandbox_runtime_to_engine(self, engine: Any) -> None:
        engine._sandbox_runtime = self._runtime

    def adopt(self, engine: Any) -> None:
        """Apply sandbox mock configuration and warmup suppression to a caller-built engine."""
        self._ensure_open()
        self._attach_sandbox_runtime_to_engine(engine)
        engine._sandbox_mode = True
        TemplateOps.set_sandbox_paraphrase_source(Sandbox._paraphrase_registry_from_catalog_path(self._extract_path))
        literals = Sandbox._load_bundled_schema_literals(self._extract_path)
        if literals is None and hasattr(engine, "_schema_graph"):
            slot = "consumer" if getattr(engine, "_schema_role", "owner") == "consumer" else "owner"
            MockProvider.pin_schema_literal_slot(slot, engine._schema_graph.schema_literal_json)

    def questions(self) -> tuple[str, ...]:
        """Return recorded corpus questions when the active bundle ships them."""
        self._ensure_open()
        questions_path = self._extract_path / "questions.txt"
        if not questions_path.is_file():
            return ()
        return Sandbox._sandbox_questions_from_path(self._extract_path)

    def apply_bundled_schema_overrides(self, engine: Any) -> None:
        """Copy bundled override JSON to the engine artifacts dir and apply schema overrides."""
        self._ensure_open()
        source = self._extract_path / "schema_overrides_demo.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing bundled schema overrides: {source}")
        target = Path(engine._artifacts_dir) / SCHEMA_OVERRIDES_DEFAULT_FILENAME
        shutil.copyfile(source, target)
        engine.apply_schema_overrides()

    def load_migration_corpus_variant(
        self,
        variant: str = "migration_demo",
        *,
        handle: SandboxHandle | None = None,
    ) -> Any:
        """Load a bundled post-migration seed and stale pre-migration artifacts for migration exercises."""
        self._ensure_open()
        state = Sandbox._stage_migration_corpus_variant(
            self._extract_path,
            variant=variant,
            artifacts_dir=self._artifacts_dir,
        )
        self._migration_variant_state = state
        execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(state.connection)
        try:
            engine = Sandbox._aether_engine_cls()(
                state.schema_context,
                artifacts_dir=self._artifacts_dir,
                config_file=self._config_path,
                execution_engine=execution_engine,
                native_connection=state.connection,
                role=SchemaRole.OWNER,
                trust_bundled_baseline=False,
            )
        except MigrationPendingError:
            engine = None
        if engine is not None:
            engine._sandbox_mode = True
            self._attach_sandbox_runtime_to_engine(engine)
            self.adopt(engine)
            if handle is not None:
                handle.engine = engine
                handle._connection = state.connection
        elif handle is not None:
            handle._connection = state.connection
        return engine

    def preview_migration_corpus_variant(self, variant: str = "migration_demo") -> MigrationPreview:
        """Build against a bundled migration corpus variant and return a migration preview."""
        self._ensure_open()
        state = getattr(self, "_migration_variant_state", None)
        if state is None or state.map_path.parent.name != variant:
            state = Sandbox._stage_migration_corpus_variant(
                self._extract_path,
                variant=variant,
                artifacts_dir=self._artifacts_dir,
            )
            self._migration_variant_state = state
        dialect = DuckDBDialect(DuckDBRuntimeConfig())
        live_graph = dialect.reflect_only(state.schema_context)
        return MainExecutionOps.preview_schema_migration(artifacts_dir=self._artifacts_dir, schema_graph=live_graph)

    def apply_migration_corpus_variant(
        self,
        variant: str = "migration_demo",
        *,
        map_path: str | None = None,
        handle: SandboxHandle | None = None,
    ) -> Any:
        """Apply the bundled migration map for *variant* and return the migrated engine."""
        self._ensure_open()
        demo_root = self._extract_path / variant
        resolved_map = Path(map_path) if map_path is not None else demo_root / "schema_migration_map.json"
        if not resolved_map.is_file():
            raise FileNotFoundError(f"Migration map not found for variant {variant!r}: {resolved_map}")
        engine, connection = Sandbox._prepare_migration_corpus_variant(
            self._extract_path,
            variant=variant,
            artifacts_dir=self._artifacts_dir,
            config_path=self._config_path,
            engine_cls=Sandbox._aether_engine_cls(),
        )
        notes, sql_arg = self._bundled_notes_and_sql()
        schema_context = Sandbox._owner_writer_schema_context(notes_file=notes, sql_file=sql_arg)
        execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
        migrated = Sandbox._aether_engine_cls().apply_migration_map(
            str(resolved_map),
            engine_context=schema_context,
            artifacts_dir=self._artifacts_dir,
            config_file=self._config_path,
            execution_engine=execution_engine,
            native_connection=connection,
            role=SchemaRole.OWNER,
        )
        self._attach_sandbox_runtime_to_engine(migrated)
        self.adopt(migrated)
        if handle is not None:
            handle.engine = migrated
            handle._connection = connection
        return migrated

    def create_preset_engine(
        self,
        engine_cls: type[Any],
        *,
        preset: SandboxPreset | str,
        connection: Any,
        deny_columns: frozenset[str] | None = None,
        restricted_consumer: bool = False,
        include: SchemaInclude = SchemaInclude.TABLES,
        reset_shared_engine_cache: bool = False,
        engine_context: EngineContext | None = None,
        notes_file: str | None = None,
        sql_file: str | None = None,
    ) -> Any:
        """Build a preset offline engine on *connection* using bundled baselines and fixtures."""
        self._ensure_open()
        extract_path = self._extract_path
        notes, sql_arg = self._bundled_notes_and_sql()
        resolved_notes, resolved_sql, init_notices = Sandbox._resolve_sandbox_notes_and_sql(
            engine_context=engine_context,
            notes_file=notes_file,
            sql_file=sql_file,
            bundled_notes=notes,
            bundled_sql=sql_arg,
            extract_path=extract_path,
        )
        schema_context = Sandbox._schema_context_for_preset(
            preset,
            notes_file=resolved_notes,
            sql_file=resolved_sql,
            deny_columns=deny_columns,
            restricted_consumer=restricted_consumer,
            include=include,
            engine_context=engine_context,
        )
        if preset == "consumer_reader":
            schema_context = Sandbox._consumer_schema_context_for_construction(schema_context)
        trust_baseline = Sandbox._sandbox_trusts_bundled_baseline(
            preset=preset,
            schema_context=schema_context,
            bundled_notes=notes,
            bundled_sql=sql_arg,
            deny_columns=deny_columns,
            restricted_consumer=restricted_consumer,
            include=include,
            engine_context=engine_context,
            notes_file=notes_file,
            sql_file=sql_file,
        )
        baseline = Sandbox._baseline_dir_for_preset(extract_path, preset, include=include) if trust_baseline else None
        engine_dir = Sandbox._sandbox_memory_engine_dir(self._artifacts_dir)

        if reset_shared_engine_cache and engine_dir.is_dir():
            Sandbox._unlink_artifact_lock_files(self._artifacts_dir)
            shutil.rmtree(engine_dir, ignore_errors=True)

        graph_path = engine_dir / "schema_graph.json.gz"
        if baseline is None:
            if trust_baseline:
                debug(f"create_offline_sandbox: no bundled schema baseline under {extract_path / 'artifacts_baseline'}")
        elif graph_path.is_file():
            debug(f"create_offline_sandbox: schema cache already present at {engine_dir}")
        else:
            engine_dir.mkdir(parents=True, exist_ok=True)
            Sandbox._copy_baseline_cache_files(baseline, engine_dir)
            Sandbox._seed_bundled_aetherspaces(extract_path, engine_dir)
            copied = [name for name in SANDBOX_BASELINE_CACHE_FILES if (engine_dir / name).is_file()]
            debug(
                f"create_offline_sandbox: seeded baseline {baseline} -> {engine_dir} ({', '.join(copied) or 'no files'})",
            )
        if include == "views":
            Sandbox._apply_rental_shop_views(connection, extract_path=extract_path)

        execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
        role = Sandbox._role_for_preset(preset)
        engine = engine_cls(
            schema_context,
            artifacts_dir=self._artifacts_dir,
            config_file=self._config_path,
            execution_engine=execution_engine,
            native_connection=connection,
            role=role,
            trust_bundled_baseline=trust_baseline,
            init_notices=init_notices,
        )
        if preset == "consumer_reader":
            Sandbox._apply_sandbox_consumer_execution_scope(engine)
        if Sandbox._load_bundled_schema_literals(extract_path) is None:
            slot = "consumer" if preset == "consumer_reader" else "owner"
            MockProvider.pin_schema_literal_slot(slot, engine._schema_graph.schema_literal_json)
        self._attach_sandbox_runtime_to_engine(engine)
        self.adopt(engine)
        return engine

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime.faithfulness_by_question.clear()
        self._runtime.faithfulness_loaded = False
        self._runtime.paraphrase_source = None
        self._runtime.recorded_corpus_question_count = None
        self._runtime.mock_provider = None
        self._runtime.mock_fixtures_path = None
        TemplateOps.clear_sandbox_paraphrase_source()
        for dataset in self._datasets.values():
            _CONNECTION_SANDBOX_HOSTS.pop(id(dataset.connection), None)
            if dataset.owns_connection:
                try:
                    dataset.connection.close()
                except (OSError, AttributeError, TypeError):
                    pass
        self._datasets.clear()
        Sandbox._restore_embedded_runtime_state(self._saved_embedded_runtime_state)
        SandboxRuntimeState.set_sandbox_recorded_corpus_question_count(None)
        if self._config_path and self._llm_config is None:
            try:
                Path(self._config_path).unlink(missing_ok=True)
            except OSError:
                pass
            self._config_path = None
        if self._owned_artifacts:
            Sandbox._unlink_artifact_lock_files(self._artifacts_dir)
            for attempt in range(3):
                gc.collect()
                try:
                    Sandbox._remove_tree(self._artifacts_dir)
                    break
                except OSError:
                    if attempt == 2:
                        shutil.rmtree(self._artifacts_dir, ignore_errors=True)
                    else:
                        time.sleep(0.05 * (attempt + 1))
        if self._bundle_access.owns_cleanup and self._extract_dir:
            shutil.rmtree(self._extract_dir, ignore_errors=True)
            self._extract_dir = ""
        SandboxRuntimeState.reset_sandbox_runtime(self._runtime_token)

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()


class SandboxHandle:
    """Owns ephemeral sandbox resources; call :meth:`close` or use as context manager."""

    __slots__ = (
        "engine",
        "_sandbox",
        "_connection",
        "_member_connections",
        "_artifacts_dir",
        "_owned_artifacts",
        "_owns_connection",
        "_config_path",
        "_extract_dir",
        "_cwd_sidecars",
        "_closed",
        "_saved_embedded_runtime_state",
    )

    def __init__(
        self,
        engine: Any,
        *,
        connection: Any,
        artifacts_dir: str,
        owned_artifacts: bool,
        owns_connection: bool,
        config_path: str | None,
        extract_dir: str,
        saved_embedded_runtime_state: tuple[Any | None, Any | None, str],
        member_connections: tuple[Any, ...] | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.engine = engine
        self._sandbox = sandbox
        self._connection = connection
        self._member_connections = member_connections
        self._artifacts_dir = artifacts_dir
        self._owned_artifacts = owned_artifacts
        self._owns_connection = owns_connection
        self._config_path = config_path
        self._extract_dir = extract_dir
        self._cwd_sidecars: list[Path] = []
        self._closed = False
        self._saved_embedded_runtime_state = saved_embedded_runtime_state
        if owned_artifacts:
            _ARTIFACT_REFCOUNT[artifacts_dir] = _ARTIFACT_REFCOUNT.get(artifacts_dir, 0) + 1

    @property
    def artifacts_dir(self) -> str:
        return self._artifacts_dir

    @property
    def connection(self) -> Any:
        self._ensure_open()
        return self._connection

    @property
    def member_connections(self) -> tuple[Any, ...] | None:
        self._ensure_open()
        return self._member_connections

    def adopt(self, engine: Any) -> None:
        """Apply sandbox mock configuration to a caller-built engine on this handle."""
        self._ensure_open()
        if self._sandbox is None:
            raise ConfigError("adopt requires a Sandbox-backed offline handle")
        self._sandbox.adopt(engine)

    def register_cwd_sidecar(self, path: Path) -> None:
        self._cwd_sidecars.append(path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sandbox = self._sandbox
        if sandbox is None:
            TemplateOps.clear_sandbox_paraphrase_source()
        self.engine._sandbox_closed = True
        is_federation = getattr(self.engine, "_is_aether_federation", False)
        if is_federation:
            try:
                self.engine.close()
            except (OSError, AttributeError, TypeError):
                pass
        member_connections = self._member_connections
        if member_connections:
            for conn in member_connections:
                try:
                    conn.close()
                except (OSError, AttributeError, TypeError):
                    pass
        elif self._owns_connection and self._connection is not None:
            try:
                self._connection.close()
            except (OSError, AttributeError, TypeError):
                pass
        if not is_federation:
            engine = getattr(self.engine, "_execution_engine", None)
            if engine is not None:
                try:
                    engine.dispose()
                except (OSError, AttributeError, TypeError):
                    pass
            Sandbox._release_engine_runtime_handles(self.engine)
        if sandbox is not None:
            sandbox.close()
        else:
            Sandbox._restore_embedded_runtime_state(self._saved_embedded_runtime_state)
            if self._config_path:
                try:
                    Path(self._config_path).unlink(missing_ok=True)
                except OSError:
                    pass
            if self._extract_dir:
                shutil.rmtree(self._extract_dir, ignore_errors=True)
        for sidecar in self._cwd_sidecars:
            try:
                if sidecar.is_file():
                    sidecar.unlink()
                elif sidecar.exists():
                    shutil.rmtree(sidecar, ignore_errors=True)
            except OSError:
                pass
        if self._owned_artifacts:
            _ARTIFACT_REFCOUNT[self._artifacts_dir] = _ARTIFACT_REFCOUNT.get(self._artifacts_dir, 1) - 1
            if _ARTIFACT_REFCOUNT.get(self._artifacts_dir, 0) <= 0:
                _ARTIFACT_REFCOUNT.pop(self._artifacts_dir, None)
                Sandbox._unlink_artifact_lock_files(self._artifacts_dir)
                for attempt in range(3):
                    gc.collect()
                    try:
                        Sandbox._remove_tree(self._artifacts_dir)
                        break
                    except OSError:
                        if attempt == 2:
                            shutil.rmtree(self._artifacts_dir, ignore_errors=True)
                        else:
                            time.sleep(0.05 * (attempt + 1))

    def __enter__(self) -> SandboxHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Sandbox handle is closed; create a new create_offline_sandbox() instance.")

    def session(self, *args: Any, **kwargs: Any) -> PipelineSession:
        self._ensure_open()
        return cast(PipelineSession, self.engine.session(*args, **kwargs))

    def asession(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_open()
        return self.engine.asession(*args, **kwargs)

    def apply_bundled_schema_overrides(self) -> None:
        """Copy bundled override JSON to artifacts_dir and apply schema overrides."""
        self._ensure_open()
        Sandbox._apply_bundled_schema_overrides(self.engine, self)

    def load_migration_corpus_variant(self, variant: str = "migration_demo") -> None:
        """Switch this handle to a bundled post-migration seed with stale pre-migration artifacts."""
        self._ensure_open()
        if self._sandbox is None:
            raise ConfigError("migration corpus variants require a Sandbox-backed offline handle")
        self.engine = self._sandbox.load_migration_corpus_variant(variant, handle=self)

    def preview_migration_corpus_variant(self, variant: str = "migration_demo") -> MigrationPreview:
        """Preview migration impact for a bundled corpus variant."""
        self._ensure_open()
        if self._sandbox is None:
            raise ConfigError("migration corpus variants require a Sandbox-backed offline handle")
        return self._sandbox.preview_migration_corpus_variant(variant)

    def apply_migration_corpus_variant(
        self,
        variant: str = "migration_demo",
        *,
        map_path: str | None = None,
    ) -> Any:
        """Apply the bundled migration map for *variant* and replace this handle's engine."""
        self._ensure_open()
        if self._sandbox is None:
            raise ConfigError("migration corpus variants require a Sandbox-backed offline handle")
        self.engine = self._sandbox.apply_migration_corpus_variant(
            variant,
            map_path=map_path,
            handle=self,
        )
        return self.engine


@dataclass(frozen=True)
class SandboxFaithfulnessExpectation:
    """Deterministic logical checks for a sandbox question beyond status/SQL presence."""

    status: str | None = None
    required_tables: frozenset[str] = frozenset()
    forbidden_tables: frozenset[str] = frozenset()
    sql_contains: tuple[str, ...] = ()
    sql_excludes: tuple[str, ...] = ()
    contains_join: bool | None = None


@dataclass(frozen=True, slots=True)
class SandboxExpectationsCatalog:
    """Slot-keyed and profile/tier-keyed expectation rows from sandbox_expectations.json."""

    by_slot_id: dict[str, dict[str, object]]
    by_context: dict[tuple[str, str, str], dict[str, object]]


@dataclass(frozen=True, slots=True)
class _MigrationVariantState:
    connection: Any
    schema_context: EngineContext
    map_path: Path


SANDBOX_EXPECTATIONS_CATALOG: SandboxExpectationsCatalog | None = None
