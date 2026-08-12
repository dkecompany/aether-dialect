"""Shared sandbox/live recording helpers: env→TOML, results traces, and LLM invoices.

Maintainer scripts and live-test harnesses both consume this module. ``scripts/`` must not
import the live-test package; the allowed direction is live-test code → scripts.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_core import LlmUsageRecord, StepResult
from aetherdialect._utils import llm_call_cost_usd, llm_price_table_as_of, snapshot_llm_usage_records
from aetherdialect._utils_artifacts import append_failure_trace

_RESULTS_FILE: Path = Path.cwd() / "results.txt"
_results_trace_pending_sep = False

_INVOICE_PATH: Path = Path.cwd() / "invoice.txt"
_INVOICE_WRITTEN_RECORD_COUNT = 0
_INVOICE_SCOPE_COUNTERS: dict[str, int] = {}


def results_file() -> Path:
    """Return the active results-trace path for this process."""
    return _RESULTS_FILE


def set_results_file(path: Path) -> None:
    """Point subsequent results-trace writes at *path* for this process."""
    global _RESULTS_FILE
    _RESULTS_FILE = Path(path)


def init_results_file() -> None:
    """Create an empty results file at the active path."""
    _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_FILE.write_text("", encoding="utf-8")


def clear_results_file() -> None:
    """Create an empty results file for the newly allocated run path."""
    init_results_file()


def append_results_summary_line(line: str) -> None:
    """Append one OK/FAIL summary line for the active run."""
    _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _RESULTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def append_live_failure_trace(step: StepResult | list[StepResult] | object | None) -> None:
    """Append a formatted failure trace to the active results file."""
    append_failure_trace(step, _RESULTS_FILE)


_append_failure_trace = append_live_failure_trace


def allocate_run_artifact_path(path: Path) -> Path:
    """Return *path* when unused; otherwise ``stem1.suffix``, ``stem2.suffix``, ..."""
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def invoice_path() -> Path:
    """Return the active invoice output path."""
    return _INVOICE_PATH


def set_invoice_path(path: Path) -> None:
    """Point subsequent invoice writes at *path* for this process."""
    global _INVOICE_PATH
    _INVOICE_PATH = Path(path)


def clear_invoice_file() -> None:
    """Create a new invoice file with a one-time header (alias for :func:`init_invoice_file`)."""
    init_invoice_file()


def init_invoice_file() -> None:
    """Create the active invoice path and write the run header once."""
    global _INVOICE_WRITTEN_RECORD_COUNT, _INVOICE_SCOPE_COUNTERS
    _INVOICE_WRITTEN_RECORD_COUNT = 0
    _INVOICE_SCOPE_COUNTERS = {}
    lines: list[str] = []
    provider = EngineConfig.LLM_PROVIDER
    if provider == "openai":
        lines.append(f"price_table_as_of={llm_price_table_as_of()}")
    lines.append(f"provider={provider}")
    lines.append("note=append-only; each question appends a usage block")
    lines.append("")
    _INVOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INVOICE_PATH.write_text("\n".join(lines), encoding="utf-8")


def _format_call_line(record: LlmUsageRecord) -> str:
    cost = llm_call_cost_usd(record)
    cost_part = f" cost=${cost:.6f}" if cost is not None else ""
    if record.provider == "openai" and cost is None:
        cost_part = f" unpriced={record.logical_model}"
    return (
        f"  {record.logical_model} task={record.task} "
        f"in={record.input_tokens} cached={record.cached_input_tokens} "
        f"out={record.output_tokens}{cost_part}"
    )


def _block_total_line(records: Sequence[LlmUsageRecord]) -> str:
    in_tok = sum(r.input_tokens for r in records)
    cached = sum(r.cached_input_tokens for r in records)
    out_tok = sum(r.output_tokens for r in records)
    costs = [c for r in records if (c := llm_call_cost_usd(r)) is not None]
    cost_part = f" cost=${sum(costs):.6f}" if costs else ""
    return f"  total in={in_tok} cached={cached} out={out_tok}{cost_part}"


def _group_usage_blocks(rows: Sequence[LlmUsageRecord]) -> list[tuple[str, int, list[LlmUsageRecord]]]:
    blocks: list[tuple[str, int, list[LlmUsageRecord]]] = []
    for record in rows:
        if blocks and blocks[-1][0] == record.scope and blocks[-1][1] == record.block_id:
            blocks[-1][2].append(record)
        else:
            blocks.append((record.scope, record.block_id, [record]))
    return blocks


def _format_usage_block_lines(block: Sequence[LlmUsageRecord], *, title: str) -> list[str]:
    lines = [f"[{title}]"]
    for record in block:
        lines.append(_format_call_line(record))
    lines.append(_block_total_line(block))
    lines.append("")
    return lines


def _append_invoice_text(text: str) -> None:
    if not text:
        return
    _INVOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _INVOICE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def _append_usage_records_to_invoice(records: Sequence[LlmUsageRecord]) -> None:
    global _INVOICE_SCOPE_COUNTERS
    if not records:
        return
    lines: list[str] = []
    for scope, _block_id, block in _group_usage_blocks(records):
        counter = _INVOICE_SCOPE_COUNTERS.get(scope, 0) + 1
        _INVOICE_SCOPE_COUNTERS[scope] = counter
        lines.extend(_format_usage_block_lines(block, title=f"{scope}_{counter}"))
    _append_invoice_text("\n".join(lines))


def _run_total_lines(rows: Sequence[LlmUsageRecord]) -> list[str]:
    blocks = _group_usage_blocks(rows)
    build_blocks = [b for b in blocks if b[0] == "build"]
    question_blocks = [b for b in blocks if b[0] == "question"]
    run_blocks = [b for b in blocks if b[0] == "run"]
    build_records = [r for _s, _b, block in build_blocks for r in block]
    question_records = [r for _s, _b, block in question_blocks for r in block]
    run_records = [r for _s, _b, block in run_blocks for r in block]
    build_cost = sum(c for r in build_records if (c := llm_call_cost_usd(r)) is not None)
    question_cost = sum(c for r in question_records if (c := llm_call_cost_usd(r)) is not None)
    run_cost = sum(c for r in run_records if (c := llm_call_cost_usd(r)) is not None)
    total_cost = build_cost + question_cost + run_cost
    provider = EngineConfig.LLM_PROVIDER
    unpriced = sorted({r.logical_model for r in rows if r.provider == "openai" and llm_call_cost_usd(r) is None})
    lines = ["[run_total]"]
    lines.append(f"  build_cost=${build_cost:.6f}")
    lines.append(f"  question_cost=${question_cost:.6f} questions={len(question_blocks)}")
    if run_records:
        lines.append(f"  other_cost=${run_cost:.6f}")
    if provider == "openai":
        lines.append(f"  total_cost=${total_cost:.6f}")
    if unpriced:
        lines.append(f"  unpriced_models={','.join(unpriced)}")
    lines.append("  note=reported totals are a floor; failed retries and batch calls carry no usage")
    return lines


def write_invoice_file(records: Sequence[LlmUsageRecord] | None = None) -> None:
    """Write a complete invoice from *records*, or append the final run total for the active run."""
    global _INVOICE_WRITTEN_RECORD_COUNT
    if records is not None:
        init_invoice_file()
        _append_usage_records_to_invoice(records)
        _append_invoice_text("\n".join(_run_total_lines(records)))
        _INVOICE_WRITTEN_RECORD_COUNT = len(records)
        return
    append_run_total_invoice()


def flush_invoice_file() -> None:
    """Append usage blocks recorded since the previous flush."""
    global _INVOICE_WRITTEN_RECORD_COUNT
    rows = snapshot_llm_usage_records()
    if len(rows) <= _INVOICE_WRITTEN_RECORD_COUNT:
        return
    new_records = rows[_INVOICE_WRITTEN_RECORD_COUNT:]
    _append_usage_records_to_invoice(new_records)
    _INVOICE_WRITTEN_RECORD_COUNT = len(rows)


def append_run_total_invoice() -> None:
    """Append the final ``[run_total]`` section for the active run after flushing pending usage."""
    flush_invoice_file()
    rows = snapshot_llm_usage_records()
    if not rows:
        return
    _append_invoice_text("\n".join(_run_total_lines(rows)))


def begin_eval_results(path: Path, *, invoice_path: Path | None = None) -> Path:
    """Allocate rotated results (and optional invoice) paths and reset write state."""
    global _results_trace_pending_sep
    results_path = allocate_run_artifact_path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    set_results_file(results_path)
    _results_trace_pending_sep = False
    init_results_file()
    print(f"Sandbox results: {results_path.resolve()}", flush=True)
    if invoice_path is not None:
        chosen_invoice = allocate_run_artifact_path(invoice_path)
        chosen_invoice.parent.mkdir(parents=True, exist_ok=True)
        set_invoice_path(chosen_invoice)
        init_invoice_file()
        print(f"Sandbox invoice: {chosen_invoice.resolve()}", flush=True)
    return results_path


def parse_live_env_file(path: str) -> dict[str, str]:
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


_parse_live_env_file = parse_live_env_file


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
    if dbx:
        doc["databricks"] = dbx
    mysql: dict[str, str] = {}
    if v := _first_nonempty("MYSQL_HOST", "MYSQL_SERVER", "MYSQL_HOSTNAME"):
        mysql["host"] = v
    if v := _first_nonempty("MYSQL_PORT"):
        mysql["port"] = v
    if v := _first_nonempty("MYSQL_USER"):
        mysql["user"] = v
    if v := _first_nonempty("MYSQL_PASSWORD"):
        mysql["password"] = v
    if v := _first_nonempty("MYSQL_DATABASE"):
        mysql["database"] = v
    if mysql:
        doc["mysql"] = mysql
    mariadb: dict[str, str] = {}
    if v := _first_nonempty("MARIADB_HOST", "MARIADB_SERVER"):
        mariadb["host"] = v
    if v := _first_nonempty("MARIADB_PORT"):
        mariadb["port"] = v
    if v := _first_nonempty("MARIADB_USER", "MARIADB_USERNAME"):
        mariadb["user"] = v
    if v := _first_nonempty("MARIADB_PASSWORD", "MARIADB_PWD"):
        mariadb["password"] = v
    if v := _first_nonempty("MARIADB_DATABASE", "MARIADB_DB"):
        mariadb["database"] = v
    if mariadb:
        doc["mariadb"] = mariadb
    duckdb_doc: dict[str, str] = {}
    if v := _first_nonempty(
        "DUCKDB_PATH",
        "DUCKDB_DATABASE",
        "DUCKDB_DATABASE_PATH",
        "DUCKDB_FILE",
        "DUCKDB_DB",
        "DUCKDB_DSN",
    ):
        duckdb_doc["path"] = v
    if v := _first_nonempty("DUCKDB_SCHEMA", "DUCKDB_DEFAULT_SCHEMA"):
        duckdb_doc["schema"] = v
    if duckdb_doc:
        doc["duckdb"] = duckdb_doc
    sqlite_doc: dict[str, str] = {}
    if v := _first_nonempty(
        "SQLITE_PATH",
        "SQLITE_DATABASE",
        "SQLITE_DATABASE_PATH",
        "SQLITE_FILE",
        "SQLITE_DB",
        "SQLITE_DSN",
        "SQLITE3_DATABASE",
    ):
        sqlite_doc["path"] = v
    if sqlite_doc:
        doc["sqlite"] = sqlite_doc
    sqlserver: dict[str, str] = {}
    if v := _first_nonempty("SQLSERVER_HOST", "MSSQL_HOST"):
        sqlserver["host"] = v
    if v := _first_nonempty("SQLSERVER_PORT", "MSSQL_PORT"):
        sqlserver["port"] = v
    if v := _first_nonempty("SQLSERVER_USER", "MSSQL_USER"):
        sqlserver["user"] = v
    if v := _first_nonempty("SQLSERVER_PASSWORD", "MSSQL_PASSWORD"):
        sqlserver["password"] = v
    if v := _first_nonempty("SQLSERVER_DATABASE", "MSSQL_DATABASE"):
        sqlserver["database"] = v
    if v := _first_nonempty("SQLSERVER_SCHEMA", "MSSQL_SCHEMA"):
        sqlserver["schema"] = v
    if v := _first_nonempty("SQLSERVER_DRIVER", "MSSQL_DRIVER"):
        sqlserver["driver"] = v
    if v := _first_nonempty("SQLSERVER_AUTH_MODE", "MSSQL_AUTH_MODE"):
        sqlserver["auth_mode"] = v
    if v := flat.get("SQLSERVER_TENANT_ID"):
        sqlserver["tenant_id"] = v
    if v := flat.get("SQLSERVER_CLIENT_ID"):
        sqlserver["client_id"] = v
    if v := flat.get("SQLSERVER_CLIENT_SECRET"):
        sqlserver["client_secret"] = v
    if sqlserver:
        doc["sqlserver"] = sqlserver
    oracle: dict[str, str] = {}
    if v := _first_nonempty("ORACLE_HOST", "ORACLE_SERVER"):
        oracle["host"] = v
    if v := flat.get("ORACLE_PORT"):
        oracle["port"] = v
    if v := _first_nonempty("ORACLE_USER", "ORACLE_USERNAME"):
        oracle["user"] = v
    if v := _first_nonempty("ORACLE_PASSWORD", "ORACLE_PWD"):
        oracle["password"] = v
    if v := _first_nonempty("ORACLE_SERVICE_NAME", "ORACLE_SERVICE"):
        oracle["service_name"] = v
    if v := flat.get("ORACLE_SID"):
        oracle["sid"] = v
    if v := _first_nonempty("ORACLE_SCHEMA", "ORACLE_DEFAULT_SCHEMA"):
        oracle["schema"] = v
    if v := flat.get("ORACLE_AUTH_MODE"):
        oracle["auth_mode"] = v
    if v := _first_nonempty("ORACLE_WALLET_LOCATION", "ORACLE_WALLET"):
        oracle["wallet_location"] = v
    if v := flat.get("ORACLE_CONFIG_DIR"):
        oracle["config_dir"] = v
    if v := _first_nonempty("ORACLE_TOKEN", "ORACLE_ACCESS_TOKEN"):
        oracle["token"] = v
    if v := flat.get("ORACLE_THICK_MODE"):
        oracle["thick_mode"] = v
    if oracle:
        doc["oracle"] = oracle
    snowflake: dict[str, str] = {}
    if v := flat.get("SNOWFLAKE_ACCOUNT"):
        snowflake["account"] = v
    if v := flat.get("SNOWFLAKE_USER"):
        snowflake["user"] = v
    if v := flat.get("SNOWFLAKE_PASSWORD"):
        snowflake["password"] = v
    if v := flat.get("SNOWFLAKE_DATABASE"):
        snowflake["database"] = v
    if v := flat.get("SNOWFLAKE_SCHEMA"):
        snowflake["schema"] = v
    if v := flat.get("SNOWFLAKE_WAREHOUSE"):
        snowflake["warehouse"] = v
    if v := flat.get("SNOWFLAKE_ROLE"):
        snowflake["role"] = v
    if v := flat.get("SNOWFLAKE_PRIVATE_KEY_PATH"):
        snowflake["private_key_path"] = v
    if v := flat.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"):
        snowflake["private_key_passphrase"] = v
    if v := flat.get("SNOWFLAKE_AUTHENTICATOR"):
        snowflake["authenticator"] = v
    if v := flat.get("SNOWFLAKE_OAUTH_TOKEN"):
        snowflake["oauth_token"] = v
    if snowflake:
        doc["snowflake"] = snowflake
    bigquery: dict[str, str] = {}
    if v := _first_nonempty("BIGQUERY_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        bigquery["project"] = v
    if v := flat.get("BIGQUERY_DATASET"):
        bigquery["dataset"] = v
    if v := _first_nonempty("BIGQUERY_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        bigquery["credentials_path"] = v
    if v := flat.get("BIGQUERY_LOCATION"):
        bigquery["location"] = v
    if bigquery:
        doc["bigquery"] = bigquery
    redshift: dict[str, str] = {}
    if v := _first_nonempty("REDSHIFT_HOST", "REDSHIFT_SERVER"):
        redshift["host"] = v
    if v := _first_nonempty("REDSHIFT_PORT", "REDSHIFT_TCP_PORT"):
        redshift["port"] = v
    if v := _first_nonempty("REDSHIFT_USER", "REDSHIFT_USERNAME"):
        redshift["user"] = v
    if v := _first_nonempty("REDSHIFT_PASSWORD", "REDSHIFT_PWD"):
        redshift["password"] = v
    if v := _first_nonempty("REDSHIFT_DATABASE", "REDSHIFT_DB"):
        redshift["database"] = v
    if v := _first_nonempty("REDSHIFT_SCHEMA"):
        redshift["schema"] = v
    if v := _first_nonempty("REDSHIFT_USE_IAM", "REDSHIFT_IAM"):
        redshift["use_iam"] = v
    if v := _first_nonempty("REDSHIFT_CLUSTER_IDENTIFIER", "REDSHIFT_CLUSTER_ID"):
        redshift["cluster_identifier"] = v
    if v := _first_nonempty("REDSHIFT_WORKGROUP", "REDSHIFT_SERVERLESS_WORKGROUP"):
        redshift["workgroup"] = v
    if v := _first_nonempty("REDSHIFT_REGION", "REDSHIFT_AWS_REGION"):
        redshift["region"] = v
    if redshift:
        doc["redshift"] = redshift
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
        "mysql",
        "mariadb",
        "duckdb",
        "sqlite",
        "sqlserver",
        "oracle",
        "snowflake",
        "bigquery",
        "redshift",
        "engine",
        "llm",
        "execution",
    ):
        if name not in doc:
            continue
        _toml_emit_section(name, doc[name], lines)
    return "\n".join(lines) + ("\n" if lines else "")


def write_live_env_file_to_temp_config_toml(env_path: str, extra_flat: dict[str, str] | None = None) -> str:
    """Materialise a ``KEY=value`` live env file as a temporary TOML file understood by config loading.

    Callers must delete the returned path when finished.
    """
    flat = parse_live_env_file(env_path)
    if extra_flat:
        flat = {**flat, **extra_flat}
    doc = _flat_live_env_to_nested_document(flat)
    fd, path = tempfile.mkstemp(prefix="live_aetherdialect_", suffix=".toml")
    os.close(fd)
    Path(path).write_text(_nested_document_to_toml_str(doc), encoding="utf-8")
    return path


def write_sandbox_recording_toml(env_path: str) -> str:
    """Materialise sandbox corpus recording config: LLM creds from *env_path*, in-memory DuckDB."""
    return write_live_env_file_to_temp_config_toml(
        env_path,
        {
            "AETHERDIALECT_ENGINE": "duckdb",
            "AETHERDIALECT_LLM_PROVIDER": "openai",
            "DUCKDB_PATH": ":memory:",
            "DUCKDB_SCHEMA": "main",
        },
    )
