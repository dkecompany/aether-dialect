"""Pytest fixtures for live pipeline tests against a real database. Bootstraps a ``AetherEngine`` instance using a temporary TOML file built from the live ``KEY=value`` env file, redirects artifact storage to a ``livetest_`` prefixed directory (separate from interactive artifacts), wipes only the template store at the start of every session so each run starts clean, and builds a ``LiveTestRunner``."""

from __future__ import annotations

import copy
import glob
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import (
    EngineConfig,
    PolicyConfig,
    PostgresRuntimeConfig,
    QSimConfig,
)
from aetherdialect._contracts_base import EngineContext, SensitivityClassification
from aetherdialect._contracts_core import LiveTestRunner
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_finalize import (
    apply_structure_to_graph,
    load_structure_document_file,
)
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    StepResult,
    llm_usage_build_scope,
    llm_usage_question_scope,
    llm_usage_session_scope,
)
from sandbox_recording import (
    allocate_run_artifact_path,
    append_live_failure_trace,
    append_results_summary_line,
    append_run_total_invoice,
    flush_invoice_file,
    init_invoice_file,
    results_file,
    set_invoice_path,
    set_results_file,
    write_live_env_file_to_temp_config_toml,
)
from sandbox_recording import (
    init_results_file as _init_results_file,
)
from sandbox_recording import (
    parse_live_env_file as _parse_live_env_file,
)

from .mydb_profile import (
    PROFILE_CONSUMER_ALLOW_OBJECTS,
    PROFILE_DATABASE_NAME_DEFAULT,
    PROFILE_NOTES_DEFAULT,
    PROFILE_OVERRIDES_PATH,
    PROFILE_SQL_DEFAULT,
)

_RESULTS_BASE = Path(__file__).parent / "results.txt"
_INVOICE_BASE = Path(__file__).parent / "invoice.txt"
_LIVE_ARTIFACTS_ROOT = Path(__file__).parent / "_run_artifacts"
set_results_file(_RESULTS_BASE)
set_invoice_path(_INVOICE_BASE)

_step_results: dict[str, StepResult] = {}
_NODEID_SCENARIO_IDS: dict[str, list[str]] = {}
_CURRENT_TEST_NODEID: str | None = None
_results_trace_pending_sep = False

_append_failure_trace = append_live_failure_trace


class _ConftestModule(types.ModuleType):
    """Keep ``_RESULTS_FILE`` assignments synced with ``sandbox_recording``."""

    def __getattr__(self, name: str) -> Any:
        if name == "_RESULTS_FILE":
            return results_file()
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_RESULTS_FILE":
            set_results_file(Path(value))
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ConftestModule


def _session_includes_live_tests(session: pytest.Session) -> bool:
    """True when at least one selected item lives under ``live_tests/``."""
    for item in getattr(session, "items", []) or []:
        path = str(getattr(item, "path", "") or getattr(item, "fspath", "") or "")
        if "live_tests" in path.replace("\\", "/"):
            return True
    return False


_LIVE_RUN_ARTIFACTS_READY = False


def pytest_collection_finish(session: pytest.Session) -> None:
    """Allocate live invoice/results only when selected items are under ``live_tests/``."""
    global _results_trace_pending_sep, _LIVE_RUN_ARTIFACTS_READY
    if not _session_includes_live_tests(session):
        _LIVE_RUN_ARTIFACTS_READY = False
        return
    _results_trace_pending_sep = False
    chosen = allocate_run_artifact_path(_RESULTS_BASE)
    set_results_file(chosen)
    _init_results_file()
    invoice = allocate_run_artifact_path(_INVOICE_BASE)
    set_invoice_path(invoice)
    init_invoice_file()
    _LIVE_RUN_ARTIFACTS_READY = True
    print(f"Live tests results: {chosen.resolve()}", flush=True)
    print(f"Live tests invoice: {invoice.resolve()}", flush=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = session, exitstatus
    global _LIVE_RUN_ARTIFACTS_READY
    if not _LIVE_RUN_ARTIFACTS_READY:
        return
    append_run_total_invoice()
    _LIVE_RUN_ARTIFACTS_READY = False


def _is_databricks_live_nodeid(nodeid: str) -> bool:
    """Return True when the test lives under Databricks-only live modules."""
    path = nodeid.replace("\\", "/")
    return "test_databricks.py::" in path or "test_databricks_dialect.py::" in path


def _is_engine_specific_nodeid(nodeid: str) -> bool:
    """Return True when the test lives under a non-PostgreSQL engine- specific live module."""
    from .live_support import ENGINE_MODULE_FRAGMENTS

    path = nodeid.replace("\\", "/")
    return any(f"{fragment}" in path for fragment in ENGINE_MODULE_FRAGMENTS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    """Order engine-specific live items last (live tests are never tagged ``fast``)."""
    _ = config
    engine_items = [it for it in items if _is_engine_specific_nodeid(it.nodeid)]
    rest = [it for it in items if it not in engine_items]
    items[:] = rest + engine_items


def pytest_runtest_setup(item: Any) -> None:
    _ = item


@pytest.fixture(autouse=True)
def _bind_live_step_nodeid(request: pytest.FixtureRequest) -> Any:
    """Bind the active pytest nodeid so captured StepResults map to ``results.txt`` for non-parametrised tests."""
    global _CURRENT_TEST_NODEID
    previous = _CURRENT_TEST_NODEID
    _CURRENT_TEST_NODEID = request.node.nodeid
    yield
    _CURRENT_TEST_NODEID = previous


def _relax_rental_shop_selectability(schema: Any, database_name: str) -> None:
    """Mark every non-selectable column as selectable and clear restricted/hidden sensitivity for rental_shop-shaped live databases."""
    if "rental_shop" not in (database_name or "").lower():
        return
    for table in schema.tables.values():
        for column in table.columns.values():
            if getattr(column, "sensitivity", None) not in (
                None,
                SensitivityClassification.NONE,
            ):
                column.sensitivity = SensitivityClassification.NONE
            column.distinct_count = max(column.distinct_count or 0, 2)
            column.null_ratio = 0.0
            column.mode_frequency_ratio = 0.0


def _env_file() -> str:
    raw = os.environ.get("LIVE_ENV_FILE", "env.env")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / raw
    return str(p)


def _domain_notes_path() -> Path | None:
    """Schema-graph domain notes (profile notes path by default)."""
    raw = os.environ.get("LIVE_DOMAIN_NOTES_FILE")
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / raw
        return p if p.is_file() else None
    return PROFILE_NOTES_DEFAULT if PROFILE_NOTES_DEFAULT.is_file() else None


def _pg_param(name: str, default: str) -> str:
    return os.environ.get(name, default)


_CONSUMER_CREDENTIALS_SKIP_REASON = "Set PGUSER2 and PGPASSWORD2 in the live env file for permission live tests."
_POSTGRES_SKIP_REASON = "Set PGHOST, PGUSER, PGPASSWORD, and PGDATABASE in the live env file for PostgreSQL live tests."


def _consumer_credentials_from_env() -> tuple[str, str]:
    """Read restricted consumer Postgres credentials from the live env file or process environment."""
    flat = _parse_live_env_file(_env_file())
    user = (flat.get("PGUSER2") or os.environ.get("PGUSER2") or "").strip()
    password = (flat.get("PGPASSWORD2") or os.environ.get("PGPASSWORD2") or "").strip()
    return user, password


def _consumer_credentials_configured() -> bool:
    """Return True when both PGUSER2 and PGPASSWORD2 are available for consumer initialization."""
    user, password = _consumer_credentials_from_env()
    return bool(user and password)


def _postgres_credentials_configured() -> bool:
    """Return True when primary PostgreSQL credentials are present in the live env file."""
    flat = _parse_live_env_file(_env_file())
    for key in ("PGUSER", "PGPASSWORD", "PGHOST", "PGDATABASE"):
        if not (flat.get(key) or os.environ.get(key) or "").strip():
            return False
    return True


def _rbac_consumer_configured() -> bool:
    """Return True when both owner and consumer PostgreSQL credentials are configured."""
    return _postgres_credentials_configured() and _consumer_credentials_configured()


def _owner_engine_context(
    *, allow_objects: frozenset[str] | None = None, deny_columns: frozenset[str] | None = None
) -> EngineContext:
    """Build a master EngineContext for rental_shop owner live fixtures."""
    notes = _domain_notes_path()
    kwargs: dict[str, Any] = {
        "notes_file": str(notes) if notes else None,
        "sql_file": _pg_param("SQL_FILE", str(PROFILE_SQL_DEFAULT)),
    }
    if allow_objects is not None:
        kwargs["allow_objects"] = allow_objects
    if deny_columns is not None:
        kwargs["deny_columns"] = deny_columns
    return EngineContext(**kwargs)


def _consumer_engine_context() -> EngineContext:
    """Build the restricted EngineContext aligned with pguser2 database grants."""
    return _owner_engine_context(allow_objects=PROFILE_CONSUMER_ALLOW_OBJECTS)


def _copy_session_schema_cache(source_artifacts_dir: str, dest_artifacts_dir: str) -> None:
    """Copy a built schema graph cache into a fresh RBAC artifacts dir (skip full reflect)."""
    os.makedirs(dest_artifacts_dir, exist_ok=True)
    for name in ("schema_graph.json.gz", "artifact_manifest.json", "schema_context.json"):
        src = os.path.join(source_artifacts_dir, name)
        dst = os.path.join(dest_artifacts_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    src_dir = source_artifacts_dir
    for sidecar in glob.glob(os.path.join(src_dir, "schema_context*.json")):
        base = os.path.basename(sidecar)
        if base == "schema_context.json":
            continue
        dst = os.path.join(dest_artifacts_dir, base)
        if not os.path.isfile(dst):
            shutil.copy2(sidecar, dst)


def _build_rbac_owner_engine(
    *,
    engine_context: EngineContext | None = None,
    relax_sensitivity: bool = True,
    config_file: str | None = None,
    schema_cache_source: str | None = None,
) -> AetherEngine:
    """Construct an owner ``AetherEngine`` for RBAC live tests."""
    ctx = engine_context or _owner_engine_context()
    owns_config = config_file is None
    if config_file is None:
        config_file = write_live_env_file_to_temp_config_toml(
            _env_file(),
            {"AETHERDIALECT_ENGINE": "postgresql"},
        )
    seed_cache = schema_cache_source is not None
    try:
        artifacts_dir = tempfile.mkdtemp(prefix="live_rbac_owner_", dir=str(_LIVE_ARTIFACTS_ROOT))
        if seed_cache:
            _copy_session_schema_cache(schema_cache_source, artifacts_dir)
        prev_regen_graph = PolicyConfig.REGENERATE_SCHEMA_GRAPH
        if seed_cache:
            PolicyConfig.REGENERATE_SCHEMA_GRAPH = False
        try:
            instance = AetherEngine(
                ctx,
                artifacts_dir=artifacts_dir,
                config_file=config_file,
                role="owner",
            )
        finally:
            PolicyConfig.REGENERATE_SCHEMA_GRAPH = prev_regen_graph
        _redirect_to_livetest_dir(instance)
        master_ctx = instance._runtime_config.engine_context
        if MainExecutionOps.load_schema_context_cache(str(instance._artifacts_dir)) is None:
            MainExecutionOps.write_schema_context_cache(str(instance._artifacts_dir), master_ctx)
        if relax_sensitivity:
            _relax_rental_shop_selectability(
                instance._schema_graph,
                _pg_param("PGDATABASE", PROFILE_DATABASE_NAME_DEFAULT),
            )
        return instance
    finally:
        if owns_config:
            Path(config_file).unlink(missing_ok=True)


def _build_rbac_consumer_engine(
    owner: AetherEngine,
    *,
    engine_context: EngineContext | str | None = None,
    config_file: str | None = None,
) -> AetherEngine:
    """Construct a consumer ``AetherEngine`` sharing the owner artifact directory."""
    pguser2, pgpassword2 = _consumer_credentials_from_env()
    owns_config = config_file is None
    if config_file is None:
        config_file = write_live_env_file_to_temp_config_toml(
            _env_file(),
            {
                "AETHERDIALECT_ENGINE": "postgresql",
                "PGUSER": pguser2,
                "PGPASSWORD": pgpassword2,
            },
        )
    # Consumers must not pass an EngineContext object (owner-only definition).
    # Default None loads the owner's cached master and scopes via DB GRANTs.
    ctx: EngineContext | str | None
    if engine_context is None:
        ctx = None
    else:
        ctx = engine_context
    try:
        instance = AetherEngine(
            ctx,
            artifacts_dir=str(owner._artifacts_dir),
            config_file=config_file,
            role="consumer",
        )
        _relax_rental_shop_selectability(
            instance._schema_graph,
            _pg_param("PGDATABASE", PROFILE_DATABASE_NAME_DEFAULT),
        )
        return instance
    finally:
        if owns_config:
            Path(config_file).unlink(missing_ok=True)


def restore_default_engine_config_classvars() -> None:
    """Restore class-level engine storage paths and connection config after live sessions."""
    from aetherdialect._constants import ENGINE_STORAGE_PLACEHOLDER_DIR, TEMPLATE_STORE_SEGMENT

    EngineConfig.TYPE = "postgresql"
    EngineConfig.RUNTIME = PostgresRuntimeConfig
    EngineConfig.SCHEMA_JSON_PATH = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "schema_graph.json.gz")
    EngineConfig.TEMPLATE_STORE_DIR = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, TEMPLATE_STORE_SEGMENT)
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "qsim_skeletons.json.gz")
    PostgresRuntimeConfig.HOST = "localhost"
    PostgresRuntimeConfig.PORT = 5432
    PostgresRuntimeConfig.USER = "postgres"
    PostgresRuntimeConfig.PASSWORD = None
    PostgresRuntimeConfig.DATABASE = None
    PostgresRuntimeConfig.SCHEMA = "public"


def _redirect_to_livetest_dir(t2s: AetherEngine) -> str:
    """Swap artifact paths to the livetest directory and return its absolute path."""
    original = t2s._artifacts_dir
    parent = os.path.dirname(original)
    folder = os.path.basename(original)
    live_folder = folder.replace("artifacts_", "livetest_", 1)
    if live_folder == folder:
        live_folder = f"livetest_{folder}"
    live_dir = os.path.join(parent, live_folder)

    if os.path.isdir(original):
        if os.path.isdir(live_dir):
            shutil.rmtree(live_dir, ignore_errors=True)
        shutil.copytree(original, live_dir, dirs_exist_ok=True)
    else:
        os.makedirs(live_dir, exist_ok=True)

    schema_dst = os.path.join(live_dir, "schema_graph.json.gz")

    template_store_dir = os.path.join(live_dir, "intent_templates")
    if os.path.isdir(template_store_dir):
        shutil.rmtree(template_store_dir, ignore_errors=True)

    t2s._artifacts_dir = live_dir
    EngineConfig.SCHEMA_JSON_PATH = schema_dst
    EngineConfig.TEMPLATE_STORE_DIR = template_store_dir
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(live_dir, "qsim_skeletons.json.gz")

    return live_dir


@pytest.fixture(scope="session", autouse=True)
def _llm_usage_invoice_session() -> Any:
    """Keep one session-scoped LLM usage accumulator for ``invoice.txt``."""
    with llm_usage_session_scope():
        yield


def _build_live_aether_engine(*, relax_sensitivity: bool) -> AetherEngine:
    """Construct a session ``AetherEngine`` instance for rental_shop live tests."""
    notes = _domain_notes_path()
    cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": "postgresql"})
    try:
        _LIVE_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
        with llm_usage_build_scope():
            instance = AetherEngine(
                EngineContext(
                    notes_file=str(notes) if notes else None,
                    sql_file=_pg_param("SQL_FILE", str(PROFILE_SQL_DEFAULT)),
                ),
                artifacts_dir=tempfile.mkdtemp(prefix="live_pg_artifacts_", dir=str(_LIVE_ARTIFACTS_ROOT)),
                config_file=cfg_path,
            )

            _redirect_to_livetest_dir(instance)

            prev_regen_graph = PolicyConfig.REGENERATE_SCHEMA_GRAPH
            prev_regen_skeleton = PolicyConfig.REGENERATE_SKELETON_CACHE
            PolicyConfig.REGENERATE_SCHEMA_GRAPH = True
            PolicyConfig.REGENERATE_SKELETON_CACHE = True
            try:
                print("Live tests: building schema graph from PostgreSQL...", flush=True)
                schema_graph = instance._schema_graph
                table_count = len(schema_graph.tables)
                column_count = sum(len(table.columns) for table in schema_graph.tables.values())
                print(
                    f"Live tests: schema graph ready ({table_count} tables, {column_count} columns)",
                    flush=True,
                )
            finally:
                PolicyConfig.REGENERATE_SCHEMA_GRAPH = prev_regen_graph
                PolicyConfig.REGENERATE_SKELETON_CACHE = prev_regen_skeleton

            fresh_store = TemplateOps.load_template_store(
                instance._schema_graph.effective_structural_hash,
                instance._schema_graph,
            )
            instance._store = fresh_store
            instance._templates = TemplateOps.store_to_templates(fresh_store)
            instance._rejected = {}

            if relax_sensitivity:
                _relax_rental_shop_selectability(
                    instance._schema_graph, _pg_param("PGDATABASE", PROFILE_DATABASE_NAME_DEFAULT)
                )
            elif PROFILE_OVERRIDES_PATH.is_file():
                apply_structure_to_graph(
                    instance._schema_graph,
                    load_structure_document_file(PROFILE_OVERRIDES_PATH),
                )

            flush_invoice_file()
            return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def t2s() -> AetherEngine:
    """Session-scoped ``AetherEngine`` instance with a clean livetest artifact dir."""
    return _build_live_aether_engine(relax_sensitivity=True)


def _derive_enforce_sensitivity_engine(t2s: AetherEngine) -> AetherEngine:
    """Derive a session engine with bundled sensitivity overrides from the primary ``t2s`` build."""
    enforced_graph = copy.deepcopy(t2s._schema_graph)
    if PROFILE_OVERRIDES_PATH.is_file():
        apply_structure_to_graph(
            enforced_graph,
            load_structure_document_file(PROFILE_OVERRIDES_PATH),
        )
    derived = object.__new__(AetherEngine)
    for slot in AetherEngine.__slots__:
        setattr(derived, slot, getattr(t2s, slot))
    derived._schema_graph = enforced_graph
    derived._store = TemplateOps.load_template_store(enforced_graph.effective_structural_hash, enforced_graph)
    derived._templates = TemplateOps.store_to_templates(derived._store)
    derived._rejected = {}
    return derived


@pytest.fixture(scope="session")
def t2s_enforce_sensitivity(t2s: AetherEngine) -> AetherEngine:
    """Session ``AetherEngine`` with bundled sensitivity overrides and no selectability relax."""
    return _derive_enforce_sensitivity_engine(t2s)


@pytest.fixture(autouse=True)
def _enforce_postgresql_dialect(request: pytest.FixtureRequest) -> None:
    """Restore PostgreSQL engine config and owner credentials before each non-engine-module test."""
    from .live_support import ENGINE_MODULE_FRAGMENTS

    if any(fragment in request.node.nodeid for fragment in ENGINE_MODULE_FRAGMENTS):
        return
    EngineConfig.TYPE = "postgresql"
    EngineConfig.RUNTIME = PostgresRuntimeConfig
    flat = _parse_live_env_file(_env_file())
    pg_env = {
        k: flat[k]
        for k in ("PGUSER", "PGPASSWORD", "PGHOST", "PGPORT", "PGDATABASE")
        if k in flat and str(flat[k]).strip()
    }
    if pg_env:
        PostgresRuntimeConfig.apply_environment(pg_env)


@pytest.fixture(scope="session", autouse=True)
def _restore_engine_config_after_live_session() -> Any:
    """Restore EngineConfig ClassVars mutated by live session fixtures."""
    yield
    restore_default_engine_config_classvars()


@pytest.fixture(scope="session")
def schema(t2s: AetherEngine) -> Any:
    """Profiled ``SchemaGraph`` from the session ``AetherEngine`` instance."""
    return t2s._schema_graph


@pytest.fixture(scope="session")
def store(t2s: AetherEngine) -> dict[str, Any]:
    """Template store dict from the session ``AetherEngine`` instance."""
    return t2s._store


@pytest.fixture(scope="session")
def templates(t2s: AetherEngine) -> dict:
    """Accepted templates dict from the session ``AetherEngine`` instance."""
    return t2s._templates


@pytest.fixture(scope="session")
def rejected(t2s: AetherEngine) -> dict:
    """Rejected templates dict from the session ``AetherEngine`` instance."""
    return t2s._rejected


@pytest.fixture(scope="session")
def schema_terms(t2s: AetherEngine) -> set[str]:
    """Schema term tokens from the session ``AetherEngine`` instance."""
    return t2s._schema_terms


@pytest.fixture(scope="session")
def runner(schema, store, templates, rejected, schema_terms, t2s) -> LiveTestRunner:
    """Session-scoped ``LiveTestRunner`` wired to the test database resources."""
    r = LiveTestRunner(
        schema=schema,
        store=store,
        templates=templates,
        rejected=rejected,
        schema_terms=schema_terms,
        csv_dir=t2s._artifacts_dir,
        dialect=t2s._dialect,
    )
    _instrument_runner(r)
    return r


@pytest.fixture(scope="session")
def schema_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> Any:
    """Profiled schema graph with bundled sensitivity overrides applied."""
    return t2s_enforce_sensitivity._schema_graph


@pytest.fixture(scope="session")
def store_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> dict[str, Any]:
    return t2s_enforce_sensitivity._store


@pytest.fixture(scope="session")
def templates_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> dict:
    return t2s_enforce_sensitivity._templates


@pytest.fixture(scope="session")
def rejected_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> dict:
    return t2s_enforce_sensitivity._rejected


@pytest.fixture(scope="session")
def schema_terms_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> set[str]:
    return t2s_enforce_sensitivity._schema_terms


@pytest.fixture(scope="session")
def runner_enforce_sensitivity(
    schema_enforce_sensitivity,
    store_enforce_sensitivity,
    templates_enforce_sensitivity,
    rejected_enforce_sensitivity,
    schema_terms_enforce_sensitivity,
    t2s_enforce_sensitivity,
) -> LiveTestRunner:
    """Live runner that keeps rental_shop sensitivity tiers enforced."""
    r = LiveTestRunner(
        schema=schema_enforce_sensitivity,
        store=store_enforce_sensitivity,
        templates=templates_enforce_sensitivity,
        rejected=rejected_enforce_sensitivity,
        schema_terms=schema_terms_enforce_sensitivity,
        csv_dir=t2s_enforce_sensitivity._artifacts_dir,
        dialect=t2s_enforce_sensitivity._dialect,
    )
    _instrument_runner(r)
    return r


@pytest.fixture(scope="session")
def rbac_postgres_config_path() -> str:
    """Session-persistent TOML config for RBAC PostgreSQL live tests."""
    if not _postgres_credentials_configured():
        pytest.skip(_POSTGRES_SKIP_REASON)
    path = write_live_env_file_to_temp_config_toml(
        _env_file(),
        {"AETHERDIALECT_ENGINE": "postgresql"},
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def rbac_consumer_config_path() -> str:
    """Session-persistent TOML config for pguser2 consumer live tests."""
    if not _consumer_credentials_configured():
        pytest.skip(_CONSUMER_CREDENTIALS_SKIP_REASON)
    pguser2, pgpassword2 = _consumer_credentials_from_env()
    path = write_live_env_file_to_temp_config_toml(
        _env_file(),
        {
            "AETHERDIALECT_ENGINE": "postgresql",
            "PGUSER": pguser2,
            "PGPASSWORD": pgpassword2,
        },
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def t2s_rbac_owner(t2s: AetherEngine, rbac_postgres_config_path: str) -> AetherEngine:
    """Session-scoped owner ``AetherEngine`` for RBAC live tests (full master context)."""
    return _build_rbac_owner_engine(
        config_file=rbac_postgres_config_path,
        schema_cache_source=str(t2s._artifacts_dir),
    )


@pytest.fixture(scope="session")
def t2s_deny_columns_owner(
    t2s: AetherEngine,
    rbac_postgres_config_path: str,
) -> AetherEngine:
    """Session-scoped owner ``AetherEngine`` with ``deny_columns`` on hidden staff columns."""
    return _build_rbac_owner_engine(
        engine_context=_owner_engine_context(
            deny_columns=frozenset({"staff.ssn", "staff.password"}),
        ),
        relax_sensitivity=False,
        config_file=rbac_postgres_config_path,
        schema_cache_source=str(t2s._artifacts_dir),
    )


@pytest.fixture(scope="session")
def t2s_consumer_pguser2(
    t2s_rbac_owner: AetherEngine,
    rbac_consumer_config_path: str,
) -> AetherEngine:
    """Session-scoped consumer ``AetherEngine`` using PGUSER2/PGPASSWORD2 database grants."""
    return _build_rbac_consumer_engine(t2s_rbac_owner, config_file=rbac_consumer_config_path)


@pytest.fixture(scope="session")
def runner_consumer_pguser2(t2s_consumer_pguser2: AetherEngine) -> LiveTestRunner:
    """``LiveTestRunner`` wired to the restricted pguser2 consumer engine."""
    r = LiveTestRunner(
        schema=t2s_consumer_pguser2._schema_graph,
        store=t2s_consumer_pguser2._store,
        templates=t2s_consumer_pguser2._templates,
        rejected=t2s_consumer_pguser2._rejected,
        schema_terms=t2s_consumer_pguser2._schema_terms,
        csv_dir=t2s_consumer_pguser2._artifacts_dir,
        dialect=t2s_consumer_pguser2._dialect,
    )
    _instrument_runner(r)
    return r


def _capture_result(result: StepResult, scenario: Any) -> None:
    """Store a step result so it appears in results.txt diagnostics."""
    _step_results[scenario.id] = result
    nodeid = _CURRENT_TEST_NODEID
    if nodeid:
        bucket = _NODEID_SCENARIO_IDS.setdefault(nodeid, [])
        bucket.append(scenario.id)
    seq_id = getattr(scenario, "sequence_id", None)
    if seq_id:
        _step_results.setdefault(seq_id, [])
        bucket = _step_results[seq_id]
        if isinstance(bucket, list):
            bucket.append(result)


def _instrument_runner(target: LiveTestRunner) -> None:
    """Patch *run*, *run_deferred*, and *clone* to capture every step result."""
    bound_run = LiveTestRunner.run.__get__(target, LiveTestRunner)
    bound_deferred = LiveTestRunner.run_deferred.__get__(target, LiveTestRunner)
    bound_clone = LiveTestRunner.clone.__get__(target, LiveTestRunner)

    def _capturing_run(scenario: Any, retries: int = 0) -> StepResult:
        with llm_usage_question_scope():
            result = bound_run(scenario, retries=retries)
        _capture_result(result, scenario)
        return result

    def _capturing_run_deferred(scenario: Any, retries: int = 0) -> StepResult:
        with llm_usage_question_scope():
            result = bound_deferred(scenario, retries=retries)
        _capture_result(result, scenario)
        return result

    def _capturing_clone() -> LiveTestRunner:
        cloned = bound_clone()
        _instrument_runner(cloned)
        return cloned

    target.run = _capturing_run
    target.run_deferred = _capturing_run_deferred
    target.clone = _capturing_clone


_SCENARIO_ID_RE = re.compile(r"\[([A-Z0-9_-]+(?:-[A-Z0-9]+)*)\]$")


def _resolve_scenario_ids(nodeid: str) -> list[str]:
    """Return scenario ids captured for *nodeid*, else extract from parametrized ``[ID]`` suffix."""
    registered = _NODEID_SCENARIO_IDS.get(nodeid)
    if registered:
        return list(registered)
    match = _SCENARIO_ID_RE.search(nodeid)
    return [match.group(1)] if match else []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        flush_invoice_file()
        status = "OK" if report.outcome == "passed" else "FAIL"
        append_results_summary_line(f"{status} {report.nodeid}")
    if report.when != "call" or report.outcome != "failed":
        return
    scenario_ids = _resolve_scenario_ids(report.nodeid)
    step: StepResult | list[StepResult] | None = None
    if scenario_ids:
        ordered_steps: list[StepResult] = []
        seen_ids: set[int] = set()
        for sid in scenario_ids:
            candidate = _step_results.get(sid)
            if candidate is None:
                continue
            cid = id(candidate)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            ordered_steps.append(candidate)
        if len(ordered_steps) == 1:
            step = ordered_steps[0]
        elif len(ordered_steps) > 1:
            step = ordered_steps
    append_live_failure_trace(step)
