"""Offline mock sandbox: zip-backed rental_shop data, fixture replay, tours."""

from __future__ import annotations

import gc
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

from aetherdialect._config import (
    DuckDBRuntimeConfig,
    PolicyConfig,
    SQLiteRuntimeConfig,
)
from aetherdialect._constants import (
    AETHERSPACES_SEGMENT,
    CONSUMER_ALLOW_OBJECTS,
    CONSUMER_RESTRICTED_ALLOW_OBJECTS,
    RENTAL_SHOP_VIEW_NAMES,
    SANDBOX_DOCTOR_REQUIRED_MEMBERS,
    SANDBOX_INTERPRET_DOMAIN_FILENAME,
    SANDBOX_MIN_FIXTURE_COUNT,
    SANDBOX_MIN_INTENT_FIXTURE_COUNT,
    SANDBOX_RECIPES,
    SANDBOX_SCHEMA_LITERALS_FILENAME,
    SANDBOX_TOUR_EXPECT_NO_SQL,
    SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL,
    SANDBOX_VALIDATION_FAILURE_QUESTIONS,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    WRITE_QUEUE_FILENAME,
)
from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    MigrationPendingError,
    OwnerOnlyOperationError,
    SchemaRole,
    SessionActiveError,
    SessionStep,
    SpaceContext,
)
from aetherdialect._contracts_core import FilterParam, NormalizedExpr, RuntimeIntent
from aetherdialect._core_utils import (
    append_failure_trace,
    build_session_step_trace,
    debug,
    pipeline_capture,
)
from aetherdialect._dialect import active_sqlglot_dialect, sql_tables_referenced
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect, create_duckdb_sqlalchemy_engine
from aetherdialect._llm_provider import (
    clear_canonical_schema_literals_cache,
    pin_mock_fixture_keys_from_bundle,
    pin_schema_literal_slot,
    reset_mock_provider,
)
from aetherdialect._main_execution import PipelineSession, compute_engine_storage_dir
from aetherdialect._templates import clear_sandbox_paraphrase_source, set_sandbox_paraphrase_source

SandboxPreset = Literal["owner_writer", "consumer_reader"]
SandboxBuildSection = Literal["validation_failures", "feedback_samples"]


def _iter_rental_shop_view_statements(views_sql_path: Path) -> list[str]:
    text = views_sql_path.read_text(encoding="utf-8")
    return [stmt.strip() for stmt in text.split(";") if stmt.strip()]


def _apply_rental_shop_views(connection: Any, *, extract_path: Path) -> None:
    views_path = extract_path / "rental_shop_views.sql"
    if not views_path.is_file():
        raise FileNotFoundError(f"Missing bundled views DDL: {views_path}")
    for name in RENTAL_SHOP_VIEW_NAMES:
        connection.execute(f'DROP VIEW IF EXISTS "{name}"')
    for stmt in _iter_rental_shop_view_statements(views_path):
        connection.execute(stmt)


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


_ARTIFACT_REFCOUNT: dict[str, int] = {}


def _sandbox_root() -> Path:
    return Path(str(resources.files("aetherdialect") / "sandbox"))


def data_zip_path() -> Path:
    override = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP", "").strip()
    if override:
        return Path(override)
    return _sandbox_root() / "data.zip"


@dataclass(frozen=True)
class _DataBundleAccess:
    path: Path
    owns_cleanup: bool


def _fixtures_path(extract_dir: Path) -> str:
    return str(extract_dir / "fixtures" / "rental_shop_mock.json")


def _zip_contains_member(names: set[str], leaf: str) -> bool:
    if leaf in names:
        return True
    return any(n.endswith(leaf) for n in names)


def _dir_contains_member(root: Path, leaf: str) -> bool:
    if (root / leaf).is_file():
        return True
    return any(path.is_file() and path.as_posix().endswith(leaf) for path in root.rglob("*"))


def fixtures_corpus_text() -> str:
    """Return the shipped mock fixture JSON from the active sandbox bundle."""
    bundle = data_zip_path()
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


def _open_data_bundle(*, dest: Path | None = None) -> _DataBundleAccess:
    """Return sandbox bundle files; extract zip to a temp dir unless env points at a directory."""
    bundle = data_zip_path()
    if bundle.is_dir():
        return _DataBundleAccess(path=bundle, owns_cleanup=False)
    target = dest if dest is not None else Path(tempfile.mkdtemp(prefix="aetherdialect_sandbox_extract_"))
    if dest is not None:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(target)
    return _DataBundleAccess(path=target, owns_cleanup=dest is None)


def _extract_data_bundle(*, dest: Path | None = None) -> Path:
    """Extract shipped ``data.zip`` to a fresh temp directory (or wipe *dest* first)."""
    return _open_data_bundle(dest=dest).path


def _bundle_path(extract_dir: Path, *parts: str) -> str:
    return str(extract_dir.joinpath(*parts))


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


def _load_bundled_interpret_domain(extract_path: Path) -> dict[str, Any] | None:
    path = extract_path / SANDBOX_INTERPRET_DOMAIN_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _pin_bundled_schema_literals(extract_path: Path) -> None:
    """Pin mock-fixture lookup keys from the active sandbox bundle."""
    pin_mock_fixture_keys_from_bundle(extract_path)


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


def _load_memory_connection(seed_sql: str) -> Any:
    """Create ``:memory:`` DuckDB, execute seed SQL, return native connection."""
    duckdb = __import__("duckdb")
    sql = Path(seed_sql).read_text(encoding="utf-8")
    connection = duckdb.connect(":memory:")
    for statement in _split_sql_statements(sql):
        if statement.strip():
            connection.execute(statement)
    return connection


def _accept_until_done(session: Any, question: str) -> Any:
    """Auto-confirm yes/no and free-text suspends until the turn ends."""
    return session.accept_until_done(question)


def _owner_writer_schema_context(
    *,
    notes_file: str | None = None,
    sql_file: str | None = None,
) -> EngineContext:
    return EngineContext(
        notes_file=notes_file,
        sql_file=sql_file,
    )


def _consumer_execution_allow_objects(*, restricted: bool) -> frozenset[str]:
    return CONSUMER_RESTRICTED_ALLOW_OBJECTS if restricted else CONSUMER_ALLOW_OBJECTS


def _consumer_reader_schema_context(
    *,
    notes_file: str | None = None,
    sql_file: str | None = None,
    restricted: bool = False,
) -> EngineContext:
    del restricted
    return _owner_writer_schema_context(notes_file=notes_file, sql_file=sql_file)


def _apply_sandbox_consumer_execution_scope(engine: Any, *, restricted: bool) -> None:
    allow = _consumer_execution_allow_objects(restricted=restricted)
    master = engine._runtime_config.engine_context
    execution_ctx = EngineContext(
        name=master.name,
        allow_objects=allow,
        include=master.include,
        deny_objects=master.deny_objects,
        deny_columns=master.deny_columns,
        allow_columns=master.allow_columns,
        notes_file=master.notes_file,
        sql_file=master.sql_file,
    )
    engine._runtime_config = replace(engine._runtime_config, execution_context=execution_ctx)
    engine._consumer_visible_objects = frozenset(allow)


def _release_engine_runtime_handles(engine: Any) -> None:
    engine._execution_engine = None
    engine._store = None
    engine._templates = {}
    engine._rejected = {}


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


def _remove_tree(path: str) -> None:
    def _on_rm_error(func: Any, p: str, _exc_info: Any) -> None:
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_on_rm_error)


def _snapshot_embedded_runtime_state() -> tuple[Any | None, Any | None, str]:
    """Capture embedded-engine globals before sandbox init mutates runtime ClassVars."""
    return (
        DuckDBRuntimeConfig.NATIVE_CONNECTION,
        SQLiteRuntimeConfig.NATIVE_CONNECTION,
        str(DuckDBRuntimeConfig.DATABASE_PATH),
    )


def _restore_embedded_runtime_state(saved: tuple[Any | None, Any | None, str]) -> None:
    """Restore embedded-engine globals saved by :func:`_snapshot_embedded_runtime_state`."""
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


class SandboxHandle:
    """Owns ephemeral sandbox resources; call :meth:`close` or use as context manager."""

    __slots__ = (
        "engine",
        "_connection",
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
    ) -> None:
        self.engine = engine
        self._connection = connection
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

    def register_cwd_sidecar(self, path: Path) -> None:
        self._cwd_sidecars.append(path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        clear_sandbox_paraphrase_source()
        self.engine._sandbox_closed = True
        if self._owns_connection:
            try:
                self._connection.close()
            except Exception:
                pass
        engine = getattr(self.engine, "_execution_engine", None)
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass
        _release_engine_runtime_handles(self.engine)
        _restore_embedded_runtime_state(self._saved_embedded_runtime_state)
        if self._config_path:
            try:
                Path(self._config_path).unlink(missing_ok=True)
            except OSError:
                pass
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
                _unlink_artifact_lock_files(self._artifacts_dir)
                for attempt in range(3):
                    gc.collect()
                    try:
                        _remove_tree(self._artifacts_dir)
                        break
                    except OSError:
                        if attempt == 2:
                            shutil.rmtree(self._artifacts_dir, ignore_errors=True)
                        else:
                            time.sleep(0.05 * (attempt + 1))
        if self._extract_dir:
            shutil.rmtree(self._extract_dir, ignore_errors=True)
            self._extract_dir = ""

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
        """Copy bundled override JSON to cwd and apply schema overrides."""
        self._ensure_open()
        _apply_bundled_schema_overrides(self.engine, self)


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


def _schema_context_for_preset(
    preset: SandboxPreset,
    *,
    notes_file: str | None,
    sql_file: str | None,
    deny_columns: frozenset[str] | None,
    restricted_consumer: bool,
) -> EngineContext:
    if preset == "owner_writer":
        ctx = _owner_writer_schema_context(notes_file=notes_file, sql_file=sql_file)
    else:
        ctx = _consumer_reader_schema_context(
            notes_file=notes_file,
            sql_file=sql_file,
            restricted=restricted_consumer,
        )
    if deny_columns:
        return EngineContext(
            notes_file=ctx.notes_file,
            sql_file=ctx.sql_file,
            allow_objects=ctx.allow_objects,
            include=ctx.include,
            deny_columns=deny_columns,
            allow_columns=ctx.allow_columns,
        )
    return ctx


def _role_for_preset(preset: SandboxPreset) -> SchemaRole:
    return "consumer" if preset == "consumer_reader" else "owner"


def _baseline_dir_for_preset(extract_path: Path, preset: SandboxPreset) -> Path | None:
    """Return the bundled schema baseline directory for *preset*, or ``None`` when absent."""
    del preset
    root = extract_path / "artifacts_baseline"
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


_BASELINE_CACHE_FILES = (
    "schema_graph.json.gz",
    "artifact_manifest.json",
    "schema_context.json",
)


def _reset_sandbox_duckdb_runtime() -> None:
    DuckDBRuntimeConfig.DATABASE_PATH = ":memory:"
    DuckDBRuntimeConfig.SCHEMA = "main"


def _sandbox_memory_engine_dir(artifacts_dir: str) -> Path:
    saved_path = DuckDBRuntimeConfig.DATABASE_PATH
    saved_schema = DuckDBRuntimeConfig.SCHEMA
    try:
        _reset_sandbox_duckdb_runtime()
        return Path(compute_engine_storage_dir(artifacts_dir, "duckdb"))
    finally:
        DuckDBRuntimeConfig.DATABASE_PATH = saved_path
        DuckDBRuntimeConfig.SCHEMA = saved_schema


def _copy_baseline_cache_files(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in _BASELINE_CACHE_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for sidecar in source.glob("schema_context*.json"):
        if sidecar.name == "schema_context.json":
            continue
        dst = dest / sidecar.name
        if not dst.is_file():
            shutil.copy2(sidecar, dst)


def _seed_bundled_aetherspaces(extract_path: Path, engine_dir: Path) -> None:
    src = extract_path / "artifacts_baseline" / AETHERSPACES_SEGMENT
    if not src.is_dir():
        return
    dest = engine_dir / AETHERSPACES_SEGMENT
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.json"):
        shutil.copy2(path, dest / path.name)


def create_offline_sandbox(
    engine_cls: type[Any],
    *,
    preset: SandboxPreset = "owner_writer",
    artifacts_dir: str | None = None,
    cleanup_artifacts: bool = True,
    deny_columns: frozenset[str] | None = None,
    restricted_consumer: bool = False,
    seed_sql: str | None = None,
    bundle_dir: str | None = None,
    connection: Any | None = None,
    owns_connection: bool | None = None,
) -> SandboxHandle:
    """Enter the offline sandbox: in-memory DuckDB rental_shop + mock LLM fixtures. When ``bundle_dir`` is set, read bundle files from that directory instead of extracting ``data.zip``. When ``connection`` is supplied, reuse an existing DuckDB connection (``owns_connection`` defaults to ``False`` in that case)."""
    pytest = __import__("sys").modules.get("pytest")
    if pytest is not None:
        duckdb = pytest.importorskip("duckdb")
    else:
        duckdb = __import__("duckdb")

    del duckdb
    if bundle_dir is not None or seed_sql is not None:
        reset_mock_provider(clear_literals=True)
        clear_sandbox_paraphrase_source()
        clear_canonical_schema_literals_cache()
    if bundle_dir is not None:
        extract_path = Path(bundle_dir)
        extract_dir = ""
    else:
        bundle_access = _open_data_bundle()
        extract_path = bundle_access.path
        extract_dir = str(extract_path) if bundle_access.owns_cleanup else ""
    seed_path = seed_sql or _bundle_path(extract_path, "rental_shop_seed.sql")
    saved_embedded_runtime_state = _snapshot_embedded_runtime_state()
    if connection is None:
        connection = _load_memory_connection(seed_path)
        resolved_owns_connection = True if owns_connection is None else owns_connection
    else:
        resolved_owns_connection = False if owns_connection is None else owns_connection
    execution_engine = create_duckdb_sqlalchemy_engine(connection)

    self_created_artifacts = artifacts_dir is None
    if self_created_artifacts:
        artifacts_dir = tempfile.mkdtemp(prefix="aetherdialect_sandbox_artifacts_")
    assert artifacts_dir is not None
    owned_artifacts = cleanup_artifacts and self_created_artifacts

    baseline = _baseline_dir_for_preset(extract_path, preset)
    reset_shared_engine_cache = not self_created_artifacts and bundle_dir is None and seed_sql is None
    engine_dir = _sandbox_memory_engine_dir(str(artifacts_dir))

    if reset_shared_engine_cache and engine_dir.is_dir():
        _unlink_artifact_lock_files(str(engine_dir))
        shutil.rmtree(engine_dir, ignore_errors=True)

    graph_path = engine_dir / "schema_graph.json.gz"
    if baseline is None:
        debug(f"create_offline_sandbox: no bundled schema baseline under {extract_path / 'artifacts_baseline'}")
    elif graph_path.is_file():
        debug(f"create_offline_sandbox: schema cache already present at {engine_dir}")
    else:
        engine_dir.mkdir(parents=True, exist_ok=True)
        _copy_baseline_cache_files(baseline, engine_dir)
        _seed_bundled_aetherspaces(extract_path, engine_dir)
        copied = [name for name in _BASELINE_CACHE_FILES if (engine_dir / name).is_file()]
        debug(
            f"create_offline_sandbox: seeded baseline {baseline} -> {engine_dir} ({', '.join(copied) or 'no files'})",
        )

    notes_file = _bundle_path(extract_path, "rental_shop_notes.txt")
    notes = notes_file if Path(notes_file).is_file() else None
    sql_file = _bundle_path(extract_path, "rental_shop.sql")
    sql_arg = sql_file if Path(sql_file).is_file() else None
    schema_context = _schema_context_for_preset(
        preset,
        notes_file=notes,
        sql_file=sql_arg,
        deny_columns=deny_columns,
        restricted_consumer=restricted_consumer,
    )
    if schema_context.include in ("views", "both"):
        _apply_rental_shop_views(connection, extract_path=extract_path)

    config_path = _write_sandbox_toml(fixtures_file=_fixtures_path(extract_path))
    _pin_bundled_schema_literals(extract_path)
    reset_mock_provider()
    role = _role_for_preset(preset)
    prev_sandbox_baseline = PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE
    PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE = True
    try:
        engine = engine_cls(
            schema_context,
            artifacts_dir=artifacts_dir,
            config_file=config_path,
            execution_engine=execution_engine,
            native_connection=connection,
            role=role,
        )
    finally:
        PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE = prev_sandbox_baseline
    if preset == "consumer_reader":
        _apply_sandbox_consumer_execution_scope(engine, restricted=restricted_consumer)
    if _load_bundled_schema_literals(extract_path) is None:
        slot = "consumer" if preset == "consumer_reader" else "owner"
        pin_schema_literal_slot(slot, engine._schema_graph.schema_literal_json)
    engine._sandbox_mode = True
    set_sandbox_paraphrase_source(_paraphrase_registry_from_catalog_path(extract_path))
    return SandboxHandle(
        engine,
        connection=connection,
        artifacts_dir=artifacts_dir,
        owned_artifacts=owned_artifacts,
        owns_connection=resolved_owns_connection,
        config_path=config_path,
        extract_dir=extract_dir,
        saved_embedded_runtime_state=saved_embedded_runtime_state,
    )


@dataclass(frozen=True)
class SandboxFaithfulnessExpectation:
    """Deterministic logical checks for a sandbox question beyond status/SQL presence."""

    status: str | None = None
    required_tables: frozenset[str] = frozenset()
    forbidden_tables: frozenset[str] = frozenset()
    sql_contains: tuple[str, ...] = ()
    sql_excludes: tuple[str, ...] = ()
    contains_join: bool | None = None


def _normalize_sandbox_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


SANDBOX_FAITHFULNESS_BY_QUESTION: dict[str, SandboxFaithfulnessExpectation] = {}


@dataclass(frozen=True, slots=True)
class SandboxExpectationsCatalog:
    """Slot-keyed and profile/tier-keyed expectation rows from sandbox_expectations.json."""

    by_slot_id: dict[str, dict[str, object]]
    by_context: dict[tuple[str, str, str], dict[str, object]]


SANDBOX_EXPECTATIONS_CATALOG: SandboxExpectationsCatalog | None = None


def _load_sandbox_expectations_catalog(extract_path: Path | None = None) -> SandboxExpectationsCatalog:
    """Load expectation rows keyed by slot_id and by (profile, tier, question)."""
    for path in _expectations_json_candidates(extract_path):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        slots = payload.get("slots") if isinstance(payload, dict) else None
        if not isinstance(slots, list):
            continue
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
            by_context[(profile, tier, _normalize_sandbox_question(question))] = expect
        if by_slot_id or by_context:
            return SandboxExpectationsCatalog(by_slot_id=by_slot_id, by_context=by_context)
    return SandboxExpectationsCatalog(by_slot_id={}, by_context={})


def _ensure_expectations_catalog(extract_path: Path | None = None) -> SandboxExpectationsCatalog:
    global SANDBOX_EXPECTATIONS_CATALOG
    if SANDBOX_EXPECTATIONS_CATALOG is not None:
        return SANDBOX_EXPECTATIONS_CATALOG
    SANDBOX_EXPECTATIONS_CATALOG = _load_sandbox_expectations_catalog(extract_path)
    return SANDBOX_EXPECTATIONS_CATALOG


def _expectation_payload_for_context(
    question: str,
    *,
    slot_id: str | None = None,
    profile: str | None = None,
    tier: str | None = None,
    extract_path: Path | None = None,
) -> dict[str, object] | None:
    """Resolve an expectation row by slot_id, then (profile, tier, question)."""
    catalog = _ensure_expectations_catalog(extract_path)
    if slot_id:
        expect = catalog.by_slot_id.get(slot_id)
        if isinstance(expect, dict):
            return expect
    norm = _normalize_sandbox_question(question)
    if profile is not None and tier is not None:
        expect = catalog.by_context.get((profile, tier, norm))
        if isinstance(expect, dict):
            return expect
    return None


def _expectations_json_candidates(extract_path: Path | None = None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if extract_path is not None:
        candidates.append(extract_path / "sandbox_expectations.json")
    repo_data = Path(__file__).resolve().parents[2] / "scripts" / "data" / "sandbox_expectations.json"
    candidates.append(repo_data)
    return tuple(candidates)


def _load_sandbox_expectations_index(extract_path: Path | None = None) -> dict[str, dict[str, object]]:
    """Legacy question-only index (owner_writer + questions rows only)."""
    catalog = _load_sandbox_expectations_catalog(extract_path)
    index: dict[str, dict[str, object]] = {}
    for (profile, tier, question_norm), expect in catalog.by_context.items():
        if profile == "owner_writer" and tier == "questions":
            index[question_norm] = expect
    return index


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


def _ensure_faithfulness_index(extract_path: Path | None = None) -> None:
    if SANDBOX_FAITHFULNESS_BY_QUESTION:
        return
    loaded = _load_sandbox_expectations_index(extract_path)
    if loaded:
        for question_norm, expect in loaded.items():
            SANDBOX_FAITHFULNESS_BY_QUESTION[question_norm] = _faithfulness_from_expect(expect)
    else:
        SANDBOX_FAITHFULNESS_BY_QUESTION.update(_LEGACY_FAITHFULNESS)


def _scenarios_json_candidates(extract_path: Path | None = None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if extract_path is not None:
        candidates.append(extract_path / "sandbox_scenarios.json")
    repo_data = Path(__file__).resolve().parents[2] / "scripts" / "data" / "sandbox_scenarios.json"
    candidates.append(repo_data)
    return tuple(candidates)


def _load_sandbox_scenarios_by_question(extract_path: Path | None = None) -> dict[str, dict[str, object]]:
    for path in _scenarios_json_candidates(extract_path):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("scenarios") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue
        index: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            if question:
                index[_normalize_sandbox_question(question)] = row
        if index:
            return index
    return {}


def _validation_failure_expect(question: str) -> dict[str, object] | None:
    """Expectation rows for validation_failures tier when not in the catalog."""
    norm = _normalize_sandbox_question(question)
    known = {_normalize_sandbox_question(q) for q in SANDBOX_VALIDATION_FAILURE_QUESTIONS}
    if norm not in known:
        return None
    return {
        "terminal_status": "error",
        "sql_required": False,
        "grain": "none",
        "must_tables": [],
        "must_filter": [],
        "sql_contains": [],
        "forbidden_sql_tokens": [],
        "validation_failure": True,
    }


def _expectation_payload_for_question(
    question: str,
    *,
    slot_id: str | None = None,
    profile: str | None = None,
    tier: str | None = None,
) -> dict[str, object] | None:
    resolved = _expectation_payload_for_context(
        question,
        slot_id=slot_id,
        profile=profile if profile is not None else "owner_writer",
        tier=tier if tier is not None else "questions",
    )
    if resolved is not None:
        return resolved
    if tier == "validation_failures":
        vf_expect = _validation_failure_expect(question)
        if vf_expect is not None:
            return vf_expect
    return _load_sandbox_expectations_index().get(_normalize_sandbox_question(question))


_LEGACY_FAITHFULNESS: dict[str, SandboxFaithfulnessExpectation] = {
    "which games support english?": SandboxFaithfulnessExpectation(
        required_tables=frozenset(),
        sql_contains=("game_supported_language",),
        contains_join=True,
    ),
    "which city has the most customers?": SandboxFaithfulnessExpectation(
        required_tables=frozenset({"city", "customer"}),
        contains_join=True,
    ),
    "how many customers are in each country?": SandboxFaithfulnessExpectation(
        required_tables=frozenset({"country", "customer"}),
        contains_join=True,
    ),
    "film title and replacement cost minus rental rate as profit margin": SandboxFaithfulnessExpectation(
        sql_excludes=("interval",),
    ),
    "what is the best pizza topping?": SandboxFaithfulnessExpectation(status="invalid_question"),
    "what's the weather today?": SandboxFaithfulnessExpectation(status="invalid_question"),
}


def _faithfulness_table_names(step: object) -> set[str]:
    """Return lowercased base schema table names from intent and SQL for faithfulness checks."""
    sql = str(getattr(step, "sql", "") or "")
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
                for t in sql_tables_referenced(sql, sqlglot_dialect=active_sqlglot_dialect())
                if t and t not in cte_aliases
            )
        return {str(t).lower() for t in names if t}
    summary = getattr(step, "intent_summary", None)
    tables = list(getattr(intent, "tables", None) or getattr(summary, "tables", None) or [])
    names = {str(t).lower() for t in tables if t}
    if sql:
        names.update(str(t).lower() for t in sql_tables_referenced(sql, sqlglot_dialect=active_sqlglot_dialect()) if t)
    return names


def _faithfulness_mismatch(step: object, expectation: SandboxFaithfulnessExpectation) -> str | None:
    status = getattr(step, "status", None)
    if expectation.status is not None and status != expectation.status:
        return f"status expected {expectation.status!r}, got {status!r}"
    sql = str(getattr(step, "sql", "") or "")
    sql_lower = sql.lower()
    used_tables = _faithfulness_table_names(step)
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


def faithfulness_expectation_for_question(
    question: str,
    *,
    slot_id: str | None = None,
    profile: str | None = None,
    tier: str | None = None,
) -> SandboxFaithfulnessExpectation | None:
    """Return the deterministic faithfulness descriptor for *question*, if any."""
    expect = _expectation_payload_for_question(
        question,
        slot_id=slot_id,
        profile=profile,
        tier=tier,
    )
    if expect is not None:
        return _faithfulness_from_expect(expect)
    _ensure_faithfulness_index()
    return SANDBOX_FAITHFULNESS_BY_QUESTION.get(_normalize_sandbox_question(question))


def check_sandbox_faithfulness(
    step: object,
    question: str,
    *,
    slot_id: str | None = None,
    profile: str | None = None,
    tier: str | None = None,
) -> str | None:
    """Return a mismatch detail when *step* fails the faithfulness descriptor for *question*."""
    expectation = faithfulness_expectation_for_question(
        question,
        slot_id=slot_id,
        profile=profile,
        tier=tier,
    )
    if expectation is None:
        return None
    return _faithfulness_mismatch(step, expectation)


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
    expect = _expectation_payload_for_question(
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
    expectation = faithfulness_expectation_for_question(
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
    if not bool(getattr(step, "sql", None)):
        return False
    if (
        check_sandbox_faithfulness(
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


def _validate_trace_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "logs" / "validate_trace.txt"


def _reset_validate_trace_file() -> None:
    path = _validate_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


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
    append_failure_trace(trace, _validate_trace_path())


def _validate_question_slot(
    engine_cls: type[Any],
    question: str,
    *,
    tier: str,
    preset: SandboxPreset = "owner_writer",
    mode: str | None = None,
    apply_overrides: bool = False,
    restricted_consumer: bool = False,
) -> dict[str, str] | None:
    """Run one offline slot and return a failure row when expectations are not met."""
    step: object | None = None
    captured_logs: list[str] = []
    try:
        with pipeline_capture(auto_responses=["y"]) as capture:
            with create_offline_sandbox(
                engine_cls,
                preset=preset,
                restricted_consumer=restricted_consumer,
            ) as sb:
                if apply_overrides:
                    sb.apply_bundled_schema_overrides()
                session_cm = sb.engine.session(mode=mode) if mode else sb.engine.session()
                with session_cm as session:
                    step = session.accept_until_done(question)
            captured_logs = list(capture.get("logs", []))
        if not question_ok(step, question, profile=preset, tier=tier):
            faith_detail = check_sandbox_faithfulness(step, question, profile=preset, tier=tier)
            row = {
                "kind": "faithfulness" if faith_detail else "question",
                "tier": tier,
                "name": question,
                "detail": faith_detail or "expectation not met",
            }
            _append_validate_trace_row(row, step, captured_logs=captured_logs)
            return row
    except Exception as exc:
        row = {"kind": "question", "tier": tier, "name": question, "detail": str(exc)}
        _append_validate_trace_row(row, step, captured_logs=captured_logs, error=str(exc))
        return row
    return None


def _validate_validation_failure_slot(engine_cls: type[Any], question: str) -> dict[str, str] | None:
    scenario = _load_sandbox_scenarios_by_question().get(_normalize_sandbox_question(question), {})
    mechanism = str(scenario.get("mechanism", ""))
    if mechanism == "bundled_overrides_hide_staff_ssn":
        return _validate_question_slot(
            engine_cls,
            question,
            tier="validation_failures",
            apply_overrides=True,
        )
    if mechanism == "schema_validation_failure":
        return _validate_question_slot(
            engine_cls,
            question,
            tier="validation_failures",
            preset="consumer_reader",
            mode="reader",
            restricted_consumer=True,
        )
    return _validate_question_slot(engine_cls, question, tier="validation_failures")


def _validate_direct_reuse_pair(engine_cls: type[Any]) -> dict[str, str] | None:
    canonical = "How many rentals happened in 2025?"
    paraphrase = "How many rentals happened in 2026?"
    step: SessionStep | None = None
    captured_logs: list[str] = []
    try:
        with pipeline_capture(auto_responses=["y"]) as capture:
            with create_offline_sandbox(engine_cls) as sb:
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
                        _append_validate_trace_row(row, step, captured_logs=captured_logs)
                        return row
                    while not step.done:
                        if step.reply_shape == "yes_no":
                            step = session.step("y")
                        elif step.reply_shape == "free_text":
                            step = session.step("ok")
                        else:
                            break
            captured_logs = list(capture.get("logs", []))
        if not step.done or not question_ok(step, paraphrase, tier="reuse"):
            row = {
                "kind": "question",
                "tier": "reuse",
                "name": paraphrase,
                "detail": "direct reuse follow-through did not complete",
            }
            _append_validate_trace_row(row, step, captured_logs=captured_logs)
            return row
    except Exception as exc:
        row = {"kind": "question", "tier": "reuse", "name": paraphrase, "detail": str(exc)}
        _append_validate_trace_row(row, step, captured_logs=captured_logs, error=str(exc))
        return row
    return None


def validate_sandbox_corpus(engine_cls: type[Any], *, smoke: bool = False) -> list[dict[str, str]]:
    """Run offline sandbox validation and return failure rows."""
    _reset_validate_trace_file()
    failures: list[dict[str, str]] = []
    for question in sandbox_questions():
        row = _validate_question_slot(engine_cls, question, tier="questions")
        if row is not None:
            failures.append(row)

    for question in _sandbox_build_section("validation_failures"):
        row = _validate_validation_failure_slot(engine_cls, question)
        if row is not None:
            failures.append(row)

    for question in sandbox_questions():
        row = _validate_question_slot(
            engine_cls,
            question,
            tier="consumer_reader",
            preset="consumer_reader",
            mode="reader",
        )
        if row is not None:
            failures.append(row)

    if smoke:
        feedback_demo = sandbox_feedback_demo()
        anchor = str(feedback_demo.get("anchor_question", "")).strip()
        rejection = str(feedback_demo.get("allowed_rejection_text", "")).strip()
        if anchor and rejection:
            try:
                with create_offline_sandbox(engine_cls) as sb:
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

    reuse_row = _validate_direct_reuse_pair(engine_cls)
    if reuse_row is not None:
        failures.append(reuse_row)

    for recipe in SANDBOX_RECIPES:
        try:
            _execute_sandbox_recipe(recipe, engine_cls)
        except Exception as exc:
            failures.append({"kind": "recipe", "tier": "", "name": recipe, "detail": str(exc)})

    feedback_demo = sandbox_feedback_demo()
    anchor = str(feedback_demo.get("anchor_question", "")).strip()
    rejection = str(feedback_demo.get("allowed_rejection_text", "")).strip()
    if anchor and rejection:
        try:
            with create_offline_sandbox(engine_cls) as sb:
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


def sandbox_doctor() -> list[str]:
    """Return human-readable problems; empty list means the sandbox bundle looks healthy."""
    issues: list[str] = []
    bundle_path = data_zip_path()
    if bundle_path.is_dir():
        for required in SANDBOX_DOCTOR_REQUIRED_MEMBERS:
            if not _dir_contains_member(bundle_path, required):
                issues.append(f"Missing {required} under {bundle_path}")
    elif bundle_path.is_file():
        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            for required in SANDBOX_DOCTOR_REQUIRED_MEMBERS:
                if not _zip_contains_member(names, required):
                    issues.append(f"Missing {required} inside {bundle_path}")
    else:
        issues.append(f"Missing data bundle: {bundle_path}")
    try:
        __import__("duckdb")
    except ImportError:
        issues.append("duckdb is not installed; pip install aetherdialect[duckdb]")
    return issues


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


def _load_sandbox_questions_sections() -> dict[str, list[str]]:
    bundle_access = _open_data_bundle()
    try:
        return _parse_questions_file(str(bundle_access.path / "questions.txt"))
    except OSError as exc:
        raise ConfigError(f"sandbox questions bundle unavailable: {exc}") from exc
    finally:
        if bundle_access.owns_cleanup:
            shutil.rmtree(bundle_access.path, ignore_errors=True)


def sandbox_questions() -> list[str]:
    """Return curated natural-language sandbox practice questions."""
    return list(_load_sandbox_questions_sections()["questions"])


def _sandbox_build_section(section: SandboxBuildSection) -> list[str]:
    """Return build-only question file sections (not part of ``sandbox_questions()``)."""
    return list(_load_sandbox_questions_sections()[section])


def _load_sandbox_catalog() -> dict[str, object]:
    bundle_access = _open_data_bundle()
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


def sandbox_catalog() -> dict[str, object]:
    """Return the bundled user-facing sandbox discovery catalog."""
    return dict(_load_sandbox_catalog())


def sandbox_paraphrase_pairs() -> list[dict[str, object]]:
    """Return canonical→paraphrase wordings from the bundled sandbox catalog."""
    rows = _load_sandbox_catalog().get("paraphrase_pairs")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def sandbox_validation_failure_demo() -> list[dict[str, str]]:
    """Return example validation-failure questions and short descriptions."""
    rows = _load_sandbox_catalog().get("validation_failure_demo")
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


def sandbox_feedback_demo() -> dict[str, str]:
    """Return the scripted reject/retry feedback demo (anchor + allowed rejection text)."""
    row = _load_sandbox_catalog().get("feedback_demo")
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


def _sandbox_doctor_verbose() -> list[str]:
    """Maintainer-only verbose corpus health check."""
    issues = sandbox_doctor()
    bundle_path = data_zip_path()
    if bundle_path.is_dir():
        if not _dir_contains_member(bundle_path, "migration_demo/artifacts_v1/schema_graph.json.gz"):
            if not any(
                path.is_file() and "migration_demo/artifacts_v1" in path.as_posix() for path in bundle_path.rglob("*")
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
            except Exception:
                pass
    elif bundle_path.is_file():
        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            if not any("migration_demo/artifacts_v1" in name for name in names):
                issues.append(f"Missing migration_demo/artifacts_v1 inside {bundle_path}")
    return issues


def assert_sandbox_complete(engine_cls: type[Any]) -> None:
    """Validate the shipped sandbox corpus and raise when any slot fails."""
    doctor = sandbox_doctor()
    if doctor:
        raise RuntimeError("sandbox_doctor failed: " + "; ".join(doctor))
    failures = validate_sandbox_corpus(engine_cls)
    if failures:
        lines = [f"[{row['kind']}] {row.get('tier', '')} {row['name'][:70]}: {row['detail']}" for row in failures]
        raise RuntimeError(
            f"{len(failures)} sandbox validation failures:\n" + "\n".join(lines),
        )


def _write_queue_path(artifacts_dir: str | Any) -> Path:
    """Resolve ``write_queue.jsonl`` under the engine storage directory."""
    if hasattr(artifacts_dir, "write_queue_path"):
        return cast(Path, artifacts_dir.write_queue_path)
    if hasattr(artifacts_dir, "_artifacts_dir"):
        root = str(artifacts_dir._artifacts_dir)
    else:
        root = str(artifacts_dir)
    return Path(root) / WRITE_QUEUE_FILENAME


def _count_utf8_lines(path: Path) -> int:
    """Return the number of lines in a UTF-8 text file."""
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _apply_bundled_schema_overrides(t2s: Any, handle: SandboxHandle | None = None) -> None:
    """Copy bundled override JSON to cwd and call ``apply_schema_overrides`` on the engine."""
    bundle_access = _open_data_bundle()
    try:
        source = bundle_access.path / "schema_overrides_demo.json"
        target = Path.cwd() / "schema_overrides.json"
        shutil.copyfile(source, target)
        if handle is not None:
            handle.register_cwd_sidecar(target)
        t2s.apply_schema_overrides()
    finally:
        if bundle_access.owns_cleanup:
            shutil.rmtree(bundle_access.path, ignore_errors=True)


def _run_sandbox_migration_demo(engine_cls: type[Any], *, verbose: bool = True) -> None:
    """Demonstrate migration map application on a toy column rename."""
    bundle_access = _open_data_bundle()
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
        post_sql.write_text(_post_migration_seed_sql(seed_path, map_path), encoding="utf-8")

        artifacts_dir = str(work / "artifacts")
        shutil.copytree(artifacts_src, artifacts_dir)

        connection = _load_memory_connection(str(post_sql))
        execution_engine = create_duckdb_sqlalchemy_engine(connection)
        notes_file = extract / "rental_shop_notes.txt"
        notes_arg = str(notes_file) if notes_file.is_file() else None
        sql_file = extract / "rental_shop.sql"
        sql_arg = str(sql_file) if sql_file.is_file() else None
        schema_context = _owner_writer_schema_context(notes_file=notes_arg, sql_file=sql_arg)
        config_file = _write_sandbox_toml(fixtures_file=_fixtures_path(extract))

        try:
            engine_cls(
                schema_context,
                artifacts_dir=artifacts_dir,
                config_file=config_file,
                execution_engine=execution_engine,
                native_connection=connection,
                role="owner",
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
            role="owner",
        )
        t2s._sandbox_mode = True
        if verbose:
            print("  Migration map applied; asking post-migration question.")
        practice_q = sandbox_questions()
        post_q = practice_q[0] if practice_q else "How many films are in the Rental Shop catalog?"
        with t2s.session() as session:
            step = _accept_until_done(session, post_q)
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


def _print_goal(msg: str) -> None:
    print(f"[sandbox] {msg}")


def _practice_questions() -> list[str]:
    return sandbox_questions()


def _recipe_chat_basics(handle: SandboxHandle) -> None:
    _print_goal("chat_basics — first tour question accept path")
    qs = _practice_questions()
    if not qs:
        return
    with handle.engine.session() as session:
        step = _accept_until_done(session, qs[0])
    print(f"  done={step.done} kind={step.kind!r} sql={bool(step.sql)}")


def _recipe_rejections(handle: SandboxHandle) -> None:
    _print_goal("rejections — intent reject with feedback")
    qs = _practice_questions()
    if len(qs) < 2:
        return
    with handle.engine.session() as session:
        step = session.ask(qs[1])
        if not step.done and step.reply_shape == "yes_no":
            step = session.step("n")
        if not step.done and step.reply_shape == "free_text":
            rejection = sandbox_feedback_demo().get("allowed_rejection_text", "")
            step = session.step(str(rejection) if rejection else "wrong intent")
        while not step.done and step.reply_shape == "yes_no":
            step = session.step("y")


def _recipe_reader_writer(engine_cls: type[Any]) -> None:
    _print_goal("reader_writer — consumer enqueues, owner drains")
    shared = tempfile.mkdtemp(prefix="aetherdialect_rw_")
    print(f"  shared artifacts_dir={shared}")
    tour = _practice_questions()
    q = tour[0] if tour else "How many films are in the Rental Shop catalog?"
    with create_offline_sandbox(engine_cls, preset="consumer_reader", artifacts_dir=shared) as reader:
        reader.apply_bundled_schema_overrides()
        queue = _write_queue_path(reader.engine)
        assert queue.is_file(), "reader should enqueue learning events"
        print(f"  write_queue lines={_count_utf8_lines(queue)}")
    with create_offline_sandbox(engine_cls, preset="owner_writer", artifacts_dir=shared) as writer:
        with writer.engine.session(mode="writer") as session:
            session.ask(q)
        queue = _write_queue_path(writer.engine)
    remaining = _count_utf8_lines(queue) if queue.is_file() else 0
    print(f"  write_queue remaining after drain={remaining}")


def _recipe_overrides(handle: SandboxHandle) -> None:
    _print_goal("overrides — owner applies bundled schema_overrides_demo.json")
    handle.apply_bundled_schema_overrides()


def _recipe_migration(engine_cls: type[Any]) -> None:
    _print_goal("migration — predetermined v1→v2 rename demo")
    _run_sandbox_migration_demo(engine_cls)


def _recipe_validation_failures(engine_cls: type[Any]) -> None:
    _print_goal("validation_failures — schema_invalid and consumer permission_denied")
    fails = _sandbox_build_section("validation_failures")
    with create_offline_sandbox(engine_cls) as owner:
        if fails:
            with owner.engine.session() as session:
                step = _accept_until_done(session, fails[0])
            print(f"  schema_invalid done={step.done} kind={step.kind!r}")
    with create_offline_sandbox(engine_cls, preset="consumer_reader", restricted_consumer=True) as consumer:
        with consumer.engine.session(mode="reader") as session:
            step = _accept_until_done(session, "How many items are there?")
        print(f"  permission_denied status={step.status!r}")


def _recipe_maintenance(handle: SandboxHandle) -> None:
    _print_goal("maintenance — show_config and get_schema_stats")
    snap = handle.engine.show_config()
    print(f"  config lines={len(snap.text.splitlines())}")
    stats = handle.engine.get_schema_stats()
    table_count = int(stats.stats.get("table_count") or 0)
    print(f"  table_count={table_count}")
    assert table_count == 34, f"Rental Shop sandbox expects 34 tables, got {table_count}"


def _recipe_errors(engine_cls: type[Any]) -> None:
    _print_goal("errors — MockFixtureMissing, OwnerOnly, SessionActive")
    with create_offline_sandbox(engine_cls) as sb:
        with sb.engine.session() as session:
            step = session.ask("This question is not in the offline corpus at all.")
            if step.error and "No mock fixture" in step.error:
                print(f"  mock fixture missing: {step.error[:80]}...")
        with sb.engine.session(mode="reader") as session:
            session.ask(sandbox_questions()[0])
            try:
                session.ask("second question while suspended")
            except SessionActiveError:
                print("  SessionActiveError raised")
    with create_offline_sandbox(engine_cls, preset="consumer_reader") as sb:
        try:
            with sb.engine.session(mode="writer"):
                pass
        except OwnerOnlyOperationError:
            print("  OwnerOnlyOperationError on consumer writer")


def _recipe_column_security(engine_cls: type[Any]) -> None:
    _print_goal("column_security — deny_columns on customer.email")
    deny = frozenset({"customer.email"})
    with create_offline_sandbox(engine_cls, deny_columns=deny) as sb:
        with sb.engine.session() as session:
            step = _accept_until_done(session, "Who are our top 5 customers by total payment?")
        codes = [d.code for d in step.diagnostics]
        print(f"  diagnostics sample={codes[:5]}")


def _recipe_partition_pruning(engine_cls: type[Any]) -> None:
    _print_goal("partition_pruning — synthetic rental.rental_date partition_columns")
    with create_offline_sandbox(engine_cls) as sb:
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
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("rental.rental_date"),
                    op="=",
                    param_key="p1",
                    raw_value=None,
                )
            ],
            param_values={"p1": "2023-07-15"},
        )
        sql = "SELECT COUNT(*) FROM rental WHERE rental.rental_date = '2023-07-15'"
        dialect = DuckDBDialect.__new__(DuckDBDialect)
        out = dialect.inject_pruning_predicates(sql, schema=sg, intent=intent)
        print(f"  injected={out != sql}")
        if out != sql:
            print(f"  finalized_sql={out}")


def _recipe_views(engine_cls: type[Any]) -> None:
    _print_goal("views — local view relations (include='views' scope)")
    with create_offline_sandbox(engine_cls) as sb:
        extract_path = Path(sb._extract_dir) if sb._extract_dir else data_zip_path()
        _apply_rental_shop_views(sb._connection, extract_path=extract_path)
        rows = sb._connection.execute(
            "SELECT store_id, total_revenue FROM store_revenue_v ORDER BY total_revenue DESC LIMIT 3"
        ).fetchall()
        print(f"  store_revenue_v sample={rows}")
        view_names = [
            row[0]
            for row in sb._connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'VIEW'"
            ).fetchall()
        ]
        print(f"  views={sorted(view_names)}")
        print(
            "  Bundled rental_shop_views.sql is in data.zip; use EngineContext(include='views') to reflect views only."
        )


def _recipe_aetherspace(engine_cls: type[Any]) -> None:
    _print_goal("aetherspace — catalog space scoped to item/film/category")
    bundle_access = _open_data_bundle()
    try:
        notes_path = bundle_access.path / "sandbox_space_catalog_notes.txt"
        notes_arg = str(notes_path) if notes_path.is_file() else None
        with create_offline_sandbox(engine_cls) as sb:
            catalog = SpaceContext(
                tables=frozenset({"item", "film", "category", "item_category"}),
                columns=frozenset(),
            )
            sb.engine.aetherspace("catalog", space_context=catalog, notes_file=notes_arg)
            with sb.engine.session(space="catalog") as session:
                step = session.accept_until_done("How many films are in the catalog?")
                print(f"  in_scope ok={bool(step.sql)}")
            with sb.engine.session(space="catalog") as session:
                step = session.accept_until_done("What is total revenue by store?")
                blocked = step.sql is None or bool(step.error)
                print(f"  out_of_scope blocked={blocked}")
    finally:
        if bundle_access.owns_cleanup:
            shutil.rmtree(bundle_access.path, ignore_errors=True)


def _recipe_full_session(handle: SandboxHandle) -> None:
    _print_goal("full_session — suspend loop on tour Q2")
    qs = _practice_questions()
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


def _with_handle(fn: Any, engine_cls: type[Any]) -> None:
    with create_offline_sandbox(engine_cls) as handle:
        fn(handle)


def _recipe_dispatch(engine_cls: type[Any]) -> dict[str, Any]:
    return {
        "chat_basics": lambda: _with_handle(_recipe_chat_basics, engine_cls),
        "rejections": lambda: _with_handle(_recipe_rejections, engine_cls),
        "reader_writer": lambda: _recipe_reader_writer(engine_cls),
        "overrides": lambda: _with_handle(_recipe_overrides, engine_cls),
        "migration": lambda: _recipe_migration(engine_cls),
        "validation_failures": lambda: _recipe_validation_failures(engine_cls),
        "maintenance": lambda: _with_handle(_recipe_maintenance, engine_cls),
        "errors": lambda: _recipe_errors(engine_cls),
        "column_security": lambda: _recipe_column_security(engine_cls),
        "full_session": lambda: _with_handle(_recipe_full_session, engine_cls),
        "partition_pruning": lambda: _recipe_partition_pruning(engine_cls),
        "views": lambda: _recipe_views(engine_cls),
        "aetherspace": lambda: _recipe_aetherspace(engine_cls),
    }


def _execute_sandbox_recipe(name: str, engine_cls: type[Any]) -> None:
    """Run a single internal sandbox validation recipe."""
    if name not in SANDBOX_RECIPES:
        raise ValueError(f"Unknown sandbox recipe {name!r}; expected one of {SANDBOX_RECIPES}")
    dispatch = _recipe_dispatch(engine_cls).get(name)
    if dispatch is None:
        raise ValueError(f"Unknown recipe {name!r}")
    dispatch()


def _execute_sandbox_tour(engine_cls: type[Any], *, interactive: bool = False) -> None:
    """Run every internal sandbox validation recipe in curriculum order."""
    del interactive
    for name in SANDBOX_RECIPES:
        _execute_sandbox_recipe(name, engine_cls)
