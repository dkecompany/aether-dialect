"""
Pytest fixtures for live pipeline tests against a real database.

Bootstraps a ``Text2SQL`` instance using a temporary TOML file built from the live ``KEY=value`` env file, redirects artifact storage to a ``livetest_`` prefixed directory (separate from interactive artifacts), wipes only the template store at the start of every session so each run starts clean, and builds a ``LiveTestRunner``.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aetherdialect._config import (
    EngineConfig,
    GenerationPath,
    PolicyConfig,
    PostgresRuntimeConfig,
    QSimConfig,
)
from aetherdialect._contracts_base import SchemaContext, SensitivityClassification
from aetherdialect._core_utils import write_gzip_json_atomic
from aetherdialect._live_testing import LiveTestRunner, StepResult
from aetherdialect._templates import (
    load_template_store,
    store_to_templates,
)
from aetherdialect.text2sql import Text2SQL

_RESULTS_FILE = Path(__file__).parent / "results.txt"

_collected_results: list[dict[str, Any]] = []
_step_results: dict[str, StepResult] = {}
_NODEID_SCENARIO_IDS: dict[str, list[str]] = {}
_CURRENT_TEST_NODEID: str | None = None
_SESSION_SCHEMA_ARTIFACTS: list[dict[str, Any]] = []
_saved_policy_diagnostics: tuple[bool, bool, bool, bool] | None = None


def pytest_sessionstart(session: pytest.Session) -> None:
    """Save diagnostic ClassVars, then enable ``LIVE_DEEP_TRACE`` for live-test ``results.txt`` sections."""

    global _saved_policy_diagnostics
    _saved_policy_diagnostics = (
        PolicyConfig.DEBUG,
        PolicyConfig.VERBOSE,
        PolicyConfig.PIPELINE_TRACE_FULL,
        PolicyConfig.LIVE_DEEP_TRACE,
    )
    PolicyConfig.LIVE_DEEP_TRACE = True


def _restore_saved_policy_diagnostics() -> None:
    """Restore ``PolicyConfig`` diagnostic flags saved at session start."""

    global _saved_policy_diagnostics
    if _saved_policy_diagnostics is None:
        return
    (
        PolicyConfig.DEBUG,
        PolicyConfig.VERBOSE,
        PolicyConfig.PIPELINE_TRACE_FULL,
        PolicyConfig.LIVE_DEEP_TRACE,
    ) = _saved_policy_diagnostics
    _saved_policy_diagnostics = None


def _is_databricks_live_nodeid(nodeid: str) -> bool:
    """Return True when the test lives under Databricks-only live modules."""
    path = nodeid.replace("\\", "/")
    return "test_databricks.py::" in path or "test_databricks_dialect.py::" in path


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    """Order collected items so non-Databricks live modules run before Databricks."""
    databricks_items = [it for it in items if _is_databricks_live_nodeid(it.nodeid)]
    rest = [it for it in items if it not in databricks_items]
    items[:] = rest + databricks_items


def pytest_runtest_setup(item: Any) -> None:
    """Log dialect and runtime for each live-marked test."""
    if item.get_closest_marker("live") is None and item.get_closest_marker("live_no_llm") is None:
        return
    from aetherdialect._core_utils import log

    runtime_name = getattr(EngineConfig.RUNTIME, "__name__", str(EngineConfig.RUNTIME))
    log(f"[live_tests] dialect={EngineConfig.TYPE!r} runtime={runtime_name} nodeid={item.nodeid}")


@pytest.fixture(autouse=True)
def _bind_live_step_nodeid(request: pytest.FixtureRequest) -> Any:
    """Bind the active pytest nodeid so captured StepResults map to ``results.txt`` for non-parametrised tests."""
    global _CURRENT_TEST_NODEID
    previous = _CURRENT_TEST_NODEID
    _CURRENT_TEST_NODEID = request.node.nodeid
    yield
    _CURRENT_TEST_NODEID = previous


def _relax_dvdrental_selectability(schema: Any, database_name: str) -> None:
    """Mark every non-selectable column as selectable and clear PII/restricted sensitivity for dvdrental-shaped live databases."""
    if "dvdrental" not in (database_name or "").lower():
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


def append_live_session_schema_artifact(engine: str, schema: Any) -> None:
    """Record a full schema graph JSON for inclusion in ``results.txt`` (PostgreSQL / Databricks first init)."""
    _SESSION_SCHEMA_ARTIFACTS.append({"engine": engine, "schema_graph": schema.to_dict()})


def _parse_live_env_file(path: str) -> dict[str, str]:
    """Parse a UTF-8 ``KEY=value`` environment file into a flat mapping of configuration keys."""

    raw = Path(path).read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    out: dict[str, str] = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        out[key] = value
    return out


def _toml_emit_section(full_name: str, table: Mapping[str, Any], lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in table.items() if isinstance(v, dict)}
    if scalars:
        lines.append(f"[{full_name}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {json.dumps(str(v))}")
    for child_name, child_table in nested.items():
        _toml_emit_section(f"{full_name}.{child_name}", child_table, lines)


def _flat_live_env_to_nested_document(flat: dict[str, str]) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    openai: dict[str, str] = {}
    if v := flat.get("OPENAI_API_KEY"):
        openai["api_key"] = v
    if v := flat.get("OPENAI_BASE_URL"):
        openai["base_url"] = v
    if openai:
        doc["openai"] = openai
    azure: dict[str, Any] = {}
    if v := flat.get("AZURE_OPENAI_ENDPOINT"):
        azure["endpoint"] = v
    if v := flat.get("AZURE_OPENAI_API_KEY"):
        azure["api_key"] = v
    if v := flat.get("AZURE_OPENAI_API_VERSION"):
        azure["api_version"] = v
    if v := flat.get("AZURE_OPENAI_BASE_URL"):
        azure["base_url"] = v
    deployments: dict[str, str] = {}
    if v := flat.get("AZURE_OPENAI_DEPLOYMENT_LIGHT"):
        deployments["light"] = v
    if v := flat.get("AZURE_OPENAI_DEPLOYMENT_MEDIUM"):
        deployments["medium"] = v
    if v := flat.get("AZURE_OPENAI_DEPLOYMENT_HEAVY"):
        deployments["heavy"] = v
    if deployments:
        azure["deployments"] = deployments
    if azure:
        doc["azure_openai"] = azure

    def _first_nonempty(*keys: str) -> str:
        for k in keys:
            raw = flat.get(k)
            if raw is None:
                continue
            t = str(raw).strip()
            if t:
                return t
        return ""

    pg: dict[str, str] = {}
    if v := _first_nonempty("POSTGRES_HOST", "PGHOST", "PGHOSTADDR"):
        pg["host"] = v
    if v := _first_nonempty("POSTGRES_PORT", "PGPORT"):
        pg["port"] = v
    if v := _first_nonempty("POSTGRES_DB", "PGDATABASE"):
        pg["database"] = v
    if v := _first_nonempty("POSTGRES_SCHEMA", "PGSCHEMA"):
        pg["schema"] = v
    if v := _first_nonempty("POSTGRES_USER", "PGUSER"):
        pg["user"] = v
    if v := _first_nonempty("POSTGRES_PASSWORD", "PGPASSWORD"):
        pg["password"] = v
    if pg:
        doc["postgresql"] = pg
    dbx: dict[str, str] = {}
    if v := flat.get("DATABRICKS_HOST"):
        dbx["host"] = v
    if v := flat.get("DATABRICKS_HTTP_PATH"):
        dbx["http_path"] = v
    if v := _first_nonempty("DATABRICKS_ACCESS_TOKEN", "DATABRICKS_TOKEN"):
        dbx["access_token"] = v
    if v := flat.get("DATABRICKS_CATALOG"):
        dbx["catalog"] = v
    if v := flat.get("DATABRICKS_SCHEMA"):
        dbx["schema"] = v
    if v := flat.get("DATABRICKS_CLUSTER_ID"):
        dbx["cluster_id"] = v
    if dbx:
        doc["databricks"] = dbx
    engine: dict[str, str] = {}
    if v := flat.get("AETHERDIALECT_ENGINE"):
        engine["selected"] = v
    if engine:
        doc["engine"] = engine
    execution: dict[str, str] = {}
    if v := flat.get("AETHERDIALECT_MAX_QUERY_COST_ROWS"):
        execution["max_query_cost_rows"] = v
    if v := flat.get("AETHERDIALECT_MAX_QUERY_COST_BYTES"):
        execution["max_query_cost_bytes"] = v
    if v := flat.get("AETHERDIALECT_STATEMENT_TIMEOUT_MS"):
        execution["statement_timeout_ms"] = v
    if v := flat.get("AETHERDIALECT_LLM_TIMEOUT_MS"):
        execution["llm_timeout_ms"] = v
    if v := flat.get("AETHERDIALECT_PROFILE_TIMEOUT_MS"):
        execution["profile_timeout_ms"] = v
    if v := flat.get("AETHERDIALECT_EXPLAIN_TIMEOUT_MS"):
        execution["explain_timeout_ms"] = v
    llm_flat: dict[str, str] = {}
    if v := flat.get("AETHERDIALECT_LLM_PROVIDER"):
        llm_flat["provider"] = v
    if llm_flat:
        doc["llm"] = llm_flat
    return doc


def _nested_document_to_toml_str(doc: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for name in (
        "openai",
        "azure_openai",
        "postgresql",
        "databricks",
        "engine",
        "llm",
        "execution",
    ):
        if name not in doc:
            continue
        _toml_emit_section(name, doc[name], lines)
    return "\n".join(lines) + ("\n" if lines else "")


def write_live_env_file_to_temp_config_toml(env_path: str, extra_flat: dict[str, str] | None = None) -> str:
    """
    Materialise a ``KEY=value`` live env file as a temporary TOML file understood by :func:`_load_config_file`.

    Callers must delete the returned path when finished.
    """

    flat = _parse_live_env_file(env_path)
    if extra_flat:
        flat = {**flat, **extra_flat}
    doc = _flat_live_env_to_nested_document(flat)
    fd, path = tempfile.mkstemp(prefix="live_aetherdialect_", suffix=".toml")
    os.close(fd)
    Path(path).write_text(_nested_document_to_toml_str(doc), encoding="utf-8")
    return path


def _env_file() -> str:
    raw = os.environ.get("LIVE_ENV_FILE", os.path.join("dev_workspace", "env.env"))
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / raw
    return str(p)


def _domain_notes_path() -> Path | None:
    """Schema-graph domain notes next to ``env.env`` (``dev_workspace/dvdrental_notes.txt`` by default)."""
    raw = os.environ.get("LIVE_DOMAIN_NOTES_FILE", os.path.join("dev_workspace", "dvdrental_notes.txt"))
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / raw
    return p if p.is_file() else None


def _pg_param(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _redirect_to_livetest_dir(t2s: Text2SQL) -> str:
    """
    Swap artifact paths from ``artifacts_...`` to ``livetest_...``.

    Creates the livetest directory if it doesn't exist.  Copies a fresh ``schema_graph.json.gz`` from the real artifacts dir when present, else builds it from legacy ``schema_graph.json``.  Wipes only the template store so every session starts with a clean slate for templates while keeping schema.

    Returns: The absolute path to the livetest artifacts directory.
    """
    original = t2s._artifacts_dir
    parent = os.path.dirname(original)
    folder = os.path.basename(original)
    live_folder = folder.replace("artifacts_", "livetest_", 1)
    if live_folder == folder:
        live_folder = f"livetest_{folder}"
    live_dir = os.path.join(parent, live_folder)

    os.makedirs(live_dir, exist_ok=True)

    schema_dst = os.path.join(live_dir, "schema_graph.json.gz")
    schema_src_gz = os.path.join(original, "schema_graph.json.gz")
    schema_src_json = os.path.join(original, "schema_graph.json")
    if os.path.exists(schema_src_gz) and os.path.normpath(schema_src_gz) != os.path.normpath(schema_dst):
        shutil.copy2(schema_src_gz, schema_dst)
    elif os.path.exists(schema_src_json):
        with open(schema_src_json, encoding="utf-8") as sf:
            schema_payload = json.load(sf)
        write_gzip_json_atomic(schema_dst, schema_payload, sort_keys=True)

    template_store_dir = os.path.join(live_dir, "intent_templates")
    if os.path.isdir(template_store_dir):
        shutil.rmtree(template_store_dir, ignore_errors=True)

    t2s._artifacts_dir = live_dir
    EngineConfig.SCHEMA_JSON_PATH = schema_dst
    EngineConfig.TEMPLATE_STORE_DIR = template_store_dir
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(live_dir, "qsim_skeletons.json.gz")

    return live_dir


@pytest.fixture(scope="session")
def t2s() -> Text2SQL:
    """Session-scoped ``Text2SQL`` instance with a clean livetest artifact dir."""
    _notes = _domain_notes_path()

    cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": "postgresql"})
    try:
        instance = Text2SQL(
            SchemaContext(
                notes_file=str(_notes) if _notes else None,
                sql_file=_pg_param("SQL_FILE", os.path.join("dev_workspace", "dvdrental.sql")),
            ),
            artifacts_dir=tempfile.mkdtemp(prefix="live_pg_artifacts_"),
            config_file=cfg_path,
        )

        _redirect_to_livetest_dir(instance)

        fresh_store = load_template_store(instance._schema_graph.effective_structural_hash, instance._schema_graph)
        instance._store = fresh_store
        instance._templates = store_to_templates(fresh_store)
        instance._rejected = {}

        append_live_session_schema_artifact("postgresql", instance._schema_graph)

        _relax_dvdrental_selectability(instance._schema_graph, _pg_param("PGDATABASE", "dvdrental_new"))

        return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _enforce_postgresql_dialect(request: pytest.FixtureRequest) -> None:
    """Restore ``EngineConfig.TYPE`` and ``EngineConfig.RUNTIME`` to PostgreSQL before each test so that module-scoped Databricks fixtures cannot leak into subsequent PostgreSQL test modules."""
    if "test_databricks" not in request.node.nodeid:
        EngineConfig.TYPE = "postgresql"
        EngineConfig.RUNTIME = PostgresRuntimeConfig


@pytest.fixture(scope="session")
def schema(t2s: Text2SQL) -> Any:
    """Profiled ``SchemaGraph`` from the session ``Text2SQL`` instance."""
    return t2s._schema_graph


@pytest.fixture(scope="session")
def store(t2s: Text2SQL) -> dict[str, Any]:
    """Template store dict from the session ``Text2SQL`` instance."""
    return t2s._store


@pytest.fixture(scope="session")
def templates(t2s: Text2SQL) -> dict:
    """Accepted templates dict from the session ``Text2SQL`` instance."""
    return t2s._templates


@pytest.fixture(scope="session")
def rejected(t2s: Text2SQL) -> dict:
    """Rejected templates dict from the session ``Text2SQL`` instance."""
    return t2s._rejected


@pytest.fixture(scope="session")
def schema_terms(t2s: Text2SQL) -> set[str]:
    """Schema term tokens from the session ``Text2SQL`` instance."""
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
        result = bound_run(scenario, retries=retries)
        _capture_result(result, scenario)
        return result

    def _capturing_run_deferred(scenario: Any, retries: int = 0) -> StepResult:
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


_LOG_PATTERNS = re.compile(
    r"intent_parse|semantic.?repair|group.?by|grain|join|strip_spurious|expand_fk|enforce_grain|repair_grouped|raw_llm|validate_sql|over_join|bridge.?table|validation.?iteration|error.?count|repeated.?identical",
    re.IGNORECASE,
)


def _format_single_step_diagnostic(step: StepResult) -> list[str]:
    """Build diagnostic lines for one StepResult."""
    diag: list[str] = []
    diag.append(f"    scenario_id: {step.scenario_id}")
    diag.append(f"    duration_seconds: {step.duration_seconds:.2f}")
    if step.template_id is not None:
        diag.append(f"    template_id: {step.template_id}")
    diag.append(f"    status: {step.status}")
    if step.generation_path is not None:
        try:
            path = GenerationPath.parse(step.generation_path)
            diag.append(f"    generation_path: {path.code} ({path.label})")
        except ValueError:
            diag.append(f"    generation_path: {step.generation_path}")
    if getattr(step, "reject_reason_actual", None) is not None:
        diag.append(f"    reject_reason_actual: {step.reject_reason_actual}")
    if getattr(step, "classified_category", None) is not None:
        diag.append(f"    classified_category: {step.classified_category}")
    if getattr(step, "classified_reason", None) is not None:
        diag.append(f"    classified_reason: {step.classified_reason}")
    if step.error:
        err_lines = step.error.strip().splitlines()
        diag.append(f"    error: {err_lines[-1] if err_lines else step.error[:200]}")
        for ln in err_lines[-5:]:
            diag.append(f"      {ln}")
    if step.intent:
        it = step.intent
        diag.append(f"    tables: {it.tables}")
        diag.append(f"    grain:  {it.grain}")
        gb_terms = [g.primary_term for g in (it.group_by_cols or [])]
        if gb_terms:
            diag.append(f"    group_by: {gb_terms}")
        agg_cols = [sc.expr.primary_term for sc in (it.select_cols or []) if sc.is_aggregated]
        if agg_cols:
            diag.append(f"    agg_select: {agg_cols}")
    if step.sql:
        if PolicyConfig.LIVE_DEEP_TRACE:
            diag.append("    sql (full):")
            for sql_ln in step.sql.splitlines():
                diag.append(f"      {sql_ln}")
        else:
            sql_display = step.sql.replace("\n", " ")
            if len(sql_display) > 200:
                sql_display = sql_display[:200] + " ..."
            diag.append(f"    sql: {sql_display}")
    if step.confidence is not None:
        diag.append(f"    confidence: {step.confidence:.3f}")
    diag.append(f"    llm_calls: {step.llm_calls}")
    if PolicyConfig.LIVE_DEEP_TRACE and step.captured_logs:
        diag.append(f"    captured_logs: {len(step.captured_logs)} lines (see PER-TEST FULL CAPTURED LOGS section)")
    elif step.llm_calls > 2:
        diag.append("    *** >2 LLM calls — full captured logs:")
        for ln in step.captured_logs:
            diag.append(f"      {ln}")
    if step.semantic_warnings:
        diag.append(f"    semantic_warnings: {step.semantic_warnings}")
    if not PolicyConfig.LIVE_DEEP_TRACE:
        relevant_logs = [ln for ln in step.captured_logs if _LOG_PATTERNS.search(ln)]
        if relevant_logs:
            diag.append("    relevant logs:")
            for ln in relevant_logs[-15:]:
                diag.append(f"      {ln}")
    return diag


def _format_step_diagnostic(step: StepResult | list[StepResult] | None) -> list[str]:
    """
    Build concise diagnostic lines for inclusion in results.txt.

    When step is a list (sequence run), formats each step with a header.
    """
    if step is None:
        return []
    if isinstance(step, list):
        lines: list[str] = []
        for i, s in enumerate(step):
            if i > 0:
                lines.append("")
            lines.append(f"    --- step {i + 1}: {s.scenario_id} ---")
            lines.extend(_format_single_step_diagnostic(s))
        return lines
    return _format_single_step_diagnostic(step)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    entry: dict[str, Any] = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration": round(report.duration, 2),
    }
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
    entry["step_diagnostic"] = _format_step_diagnostic(step) if step else []
    entry["full_logs"] = []
    if PolicyConfig.LIVE_DEEP_TRACE and step:
        if isinstance(step, list):
            entry["full_logs"] = [ln for s in step for ln in (s.captured_logs or [])]
        else:
            entry["full_logs"] = list(step.captured_logs or [])
    elif report.failed and step:
        if isinstance(step, list):
            entry["full_logs"] = [ln for s in step for ln in (s.captured_logs or [])]
        else:
            entry["full_logs"] = list(step.captured_logs) if step.captured_logs else []

    if report.failed:
        assertion_lines: list[str] = []
        for raw_line in str(report.longrepr).splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("E ") or stripped.startswith("["):
                assertion_lines.append(f"    {stripped}")
        entry["assertion_lines"] = assertion_lines
    _collected_results.append(entry)


def pytest_sessionfinish(session, exitstatus):
    if not _collected_results:
        _restore_saved_policy_diagnostics()
        return

    passed = [r for r in _collected_results if r["outcome"] == "passed"]
    failed = [r for r in _collected_results if r["outcome"] == "failed"]
    skipped = [r for r in _collected_results if r["outcome"] == "skipped"]
    total = len(_collected_results)
    total_dur = sum(r["duration"] for r in _collected_results)

    lines: list[str] = []
    lines.append(f"Live Test Results  —  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 72)
    lines.append(
        f"Total: {total}  |  Passed: {len(passed)}  |  "
        f"Failed: {len(failed)}  |  Skipped: {len(skipped)}  |  "
        f"Duration: {total_dur:.1f}s"
    )
    lines.append("")

    if _SESSION_SCHEMA_ARTIFACTS:
        lines.append("-" * 72)
        lines.append("SESSION SCHEMA GRAPH SNAPSHOTS (PostgreSQL session + Databricks module init)")
        lines.append("-" * 72)
        for i, art in enumerate(_SESSION_SCHEMA_ARTIFACTS):
            lines.append("")
            lines.append(f"  [{i + 1}] engine={art['engine']}")
            dumped = json.dumps(art["schema_graph"], indent=2, ensure_ascii=False, default=str)
            for jl in dumped.splitlines():
                lines.append(f"  {jl}")
        lines.append("")

    if failed:
        lines.append("-" * 72)
        lines.append("FAILURES")
        lines.append("-" * 72)
        for r in failed:
            lines.append("")
            lines.append(f">>> {r['nodeid']}  ({r['duration']}s)")
            for al in r.get("assertion_lines", []):
                lines.append(al)
            diag = r.get("step_diagnostic", [])
            if diag:
                lines.append("")
                for dl in diag:
                    lines.append(dl)
            full_logs = r.get("full_logs", [])
            if full_logs and not PolicyConfig.LIVE_DEEP_TRACE:
                lines.append("")
                lines.append("    Full logs:")
                for ln in full_logs:
                    lines.append(f"      {ln}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("ALL RESULTS")
    lines.append("-" * 72)
    for r in _collected_results:
        tag = r["outcome"].upper()
        diag = r.get("step_diagnostic", [])
        llm_line = next((d for d in diag if "llm_calls:" in d), None)
        llm_suffix = f"  llm_calls={llm_line.split(':')[1].strip()}" if llm_line else ""
        lines.append(f"  [{tag:>7}]  {r['nodeid']}  ({r['duration']}s){llm_suffix}")

    if PolicyConfig.LIVE_DEEP_TRACE:
        lines.append("")
        lines.append("-" * 72)
        lines.append("PER-TEST FULL CAPTURED LOGS (LIVE_DEEP_TRACE)")
        lines.append("-" * 72)
        for r in _collected_results:
            fl = r.get("full_logs") or []
            lines.append("")
            lines.append(f">>> {r['nodeid']}  ({r['outcome']})  ({r['duration']}s)")
            for ln in fl:
                lines.append(f"  {ln}")

    lines.append("")
    _RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    _restore_saved_policy_diagnostics()
