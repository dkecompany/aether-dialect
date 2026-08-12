"""rental_shop dev tooling: load engines, ping one engine, or extract PostgreSQL CSVs."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import os
import re
import sys
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "scripts" / "data"
_SCRIPTS = _REPO_ROOT / "scripts"
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_cfg = importlib.import_module("aetherdialect._config")
BigQueryRuntimeConfig = _cfg.BigQueryRuntimeConfig
DatabricksRuntimeConfig = _cfg.DatabricksRuntimeConfig
DuckDBRuntimeConfig = _cfg.DuckDBRuntimeConfig
MariaDBRuntimeConfig = _cfg.MariaDBRuntimeConfig
MySQLRuntimeConfig = _cfg.MySQLRuntimeConfig
PostgresRuntimeConfig = _cfg.PostgresRuntimeConfig
RedshiftRuntimeConfig = _cfg.RedshiftRuntimeConfig
SnowflakeRuntimeConfig = _cfg.SnowflakeRuntimeConfig
SQLiteRuntimeConfig = _cfg.SQLiteRuntimeConfig
SQLServerRuntimeConfig = _cfg.SQLServerRuntimeConfig
OracleRuntimeConfig = _cfg.OracleRuntimeConfig

DEFAULT_ENV_FILE = _REPO_ROOT / "env.env"


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path:
    """Parse ``KEY=VALUE`` lines from *path* into ``os.environ``."""

    env_path = Path(path) if path else Path(os.environ.get("LIVE_ENV_FILE", str(DEFAULT_ENV_FILE)))
    if not env_path.is_file():
        raise FileNotFoundError(f"env file not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def repo_root() -> Path:
    """Return the repository root directory."""

    return _REPO_ROOT


def default_csv_dir() -> Path:
    """Return the canonical rental_shop CSV export directory."""

    return _DATA / "rental_shop_csvs"


def default_ddl_path() -> Path:
    """Return the canonical Postgres-flavoured rental_shop DDL file."""

    return _DATA / "rental_shop.sql"


def _log_progress(message: str) -> None:
    print(message, flush=True)


def _log_load_table(engine: str, table: str, row_count: int) -> None:
    _log_progress(f"[load] {engine}: {table} ({row_count} rows)")


def _configure_loader_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="google.cloud.bigquery.client")
    warnings.filterwarnings(
        "ignore",
        message=r".*_user_agent_entry.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*user_agent_entry.*deprecated.*",
        category=DeprecationWarning,
    )


_VERIFY_VERBOSE = False


_configure_loader_warnings()


_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(\w+)\s*\(",
    re.IGNORECASE,
)
_ALTER_FK_RE = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+CONSTRAINT\s+\w+\s+(FOREIGN\s+KEY\s+\(.+?\)\s+REFERENCES\s+\w+\s*\(.+?\))",
    re.IGNORECASE | re.DOTALL,
)
_ALTER_PK_RE = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+CONSTRAINT\s+\w+\s+PRIMARY\s+KEY\s*\((.+?)\)",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_FK_RE = re.compile(
    r",\s*(FOREIGN\s+KEY\s+\(.+?\)\s+REFERENCES\s+\w+\s*\(.+?\))",
    re.IGNORECASE | re.DOTALL,
)


def iter_create_table_blocks(ddl: str) -> list[str]:
    """Return each ``CREATE TABLE ... );`` statement from *ddl*."""

    blocks: list[str] = []
    for match in re.finditer(
        r"CREATE\s+TABLE\s+\w+\s*\(.+?\)\s*;",
        ddl,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        blocks.append(match.group(0).strip())
    return blocks


def split_block(block: str) -> tuple[str, str, list[str]]:
    """Return table name, CREATE without inline FK clauses, and inline FK fragments."""

    name_match = _CREATE_TABLE_RE.search(block)
    if not name_match:
        raise ValueError(f"Could not parse CREATE TABLE name from block: {block[:80]!r}")
    name = name_match.group(1)
    fks = [m.group(1).strip() for m in _INLINE_FK_RE.finditer(block)]
    no_fk = _INLINE_FK_RE.sub("", block)
    no_fk = re.sub(r",\s*\)", ")", no_fk)
    return name, no_fk.strip(), fks


def qualify_table_name(create_sql: str, schema: str) -> str:
    """Prefix ``CREATE TABLE name`` with *schema* (Postgres / Redshift style)."""

    return re.sub(
        r"CREATE\s+TABLE\s+(\w+)\b",
        rf"CREATE TABLE {schema}.\1",
        create_sql,
        count=1,
        flags=re.IGNORECASE,
    )


def alter_fk_sql(table: str, schema: str, fk: str, constraint_name: str) -> str:
    """Build ``ALTER TABLE schema.table ADD CONSTRAINT ... FOREIGN KEY ...``."""

    fk_clean = fk.strip().lstrip(",").strip()
    return f"ALTER TABLE {schema}.{table} ADD CONSTRAINT {constraint_name} {fk_clean}"


def iter_alter_foreign_keys(ddl: str) -> list[tuple[str, str]]:
    """Return ``(table_name, foreign_key_clause)`` pairs from trailing ALTER statements."""

    out: list[tuple[str, str]] = []
    for match in _ALTER_FK_RE.finditer(ddl):
        out.append((match.group(1), match.group(2).strip()))
    return out


def iter_alter_primary_keys(ddl: str) -> list[tuple[str, list[str]]]:
    """Return ``(table_name, [pk_column, ...])`` pairs from trailing ALTER statements."""

    out: list[tuple[str, list[str]]] = []
    for match in _ALTER_PK_RE.finditer(ddl):
        cols = [c.strip() for c in match.group(2).split(",") if c.strip()]
        out.append((match.group(1), cols))
    return out


_INTEGER_COLUMN_LINE_RE = re.compile(
    r"^\s*(\w+)\s+(?:SMALLINT|INTEGER|BIGINT)\b",
    re.IGNORECASE,
)


def parse_integer_columns_from_ddl(ddl: str) -> dict[str, frozenset[str]]:
    """Return ``{table_name: {integer_column, ...}}`` from Postgres DDL blocks."""

    out: dict[str, frozenset[str]] = {}
    for block in iter_create_table_blocks(ddl):
        name_match = _CREATE_TABLE_RE.search(block)
        if not name_match:
            continue
        table = name_match.group(1)
        body_match = re.search(r"\((.*)\)\s*;", block, flags=re.DOTALL)
        if not body_match:
            continue
        cols: list[str] = []
        for line in body_match.group(1).splitlines():
            col_match = _INTEGER_COLUMN_LINE_RE.match(line)
            if col_match:
                cols.append(col_match.group(1))
        out[table] = frozenset(cols)
    return out


_DDL_INTEGER_COLUMNS: dict[str, frozenset[str]] | None = None


def integer_columns_for_table(table: str, *, ddl_text: str | None = None) -> frozenset[str]:
    """Return integer/SMALLINT/BIGINT columns for *table* from rental_shop DDL."""

    global _DDL_INTEGER_COLUMNS
    if ddl_text is not None:
        return parse_integer_columns_from_ddl(ddl_text).get(table, frozenset())
    if _DDL_INTEGER_COLUMNS is None:
        _DDL_INTEGER_COLUMNS = parse_integer_columns_from_ddl(default_ddl_path().read_text(encoding="utf-8"))
    return _DDL_INTEGER_COLUMNS.get(table, frozenset())


def _strip_defaults(sql: str) -> str:
    return re.sub(r"\s+DEFAULT\s+[^,\n)]+", "", sql, flags=re.IGNORECASE)


def translate_create(engine: str, create_sql: str, *, schema: str) -> str:
    """Return a CREATE TABLE statement for *engine* derived from a Postgres block."""

    name_match = re.search(r"CREATE\s+TABLE\s+(\w+)\b", create_sql, re.IGNORECASE)
    if not name_match:
        raise ValueError("CREATE TABLE name missing")
    table = name_match.group(1)
    body_match = re.search(r"\((.*)\)\s*;", create_sql, re.DOTALL | re.IGNORECASE)
    if not body_match:
        raise ValueError(f"CREATE TABLE body missing for {table}")
    body = body_match.group(1)

    if engine == "postgresql":
        return re.sub(
            r"CREATE\s+TABLE\s+\w+\b",
            f"CREATE TABLE {schema}.{table}",
            create_sql,
            count=1,
            flags=re.IGNORECASE,
        )

    if engine in _MYSQL_FAMILY:
        body = _translate_body_mysql(body)
        return f"CREATE TABLE `{table}` (\n{body}\n)"

    if engine == "sqlserver":
        body = _translate_body_sqlserver(body)
        return f"CREATE TABLE [{schema}].[{table}] (\n{body}\n)"

    if engine == "oracle":
        body = _translate_body_oracle(body)
        owner = schema.upper()
        return f'CREATE TABLE "{owner}"."{table.upper()}" (\n{body}\n)'

    if engine == "snowflake":
        body = _translate_body_snowflake(body)
        return f"CREATE TABLE {schema}.{table} (\n{body}\n)"

    if engine == "bigquery":
        body = _translate_body_bigquery(body)
        return f"CREATE TABLE `{schema}.{table}` (\n{body}\n)"

    if engine == "redshift":
        body = _translate_body_redshift(body)
        return f"CREATE TABLE {schema}.{table} (\n{body}\n)"

    if engine == "databricks":
        body = _translate_body_databricks(body)
        return f"CREATE TABLE {schema}.{table} (\n{body}\n)"

    if engine == "sqlite":
        body = _translate_body_sqlite(body)
        return f'CREATE TABLE "{table}" (\n{body}\n)'

    if engine == "duckdb":
        body = _translate_body_duckdb(body)
        return f'CREATE TABLE "{table}" (\n{body}\n)'

    raise ValueError(f"Unsupported engine for DDL translation: {engine}")


def _translate_body_duckdb(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "VARCHAR[]", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "JSON", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "SMALLINT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BOOLEAN", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "VARCHAR", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", "DECIMAL(\\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bREAL\b", "DOUBLE", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\((\d+)\)", r"VARCHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<![A-Za-z])CHAR\((\d+)\)", r"VARCHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "TIMESTAMP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATE\b", "DATE", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBLOB\b", "BLOB", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_sqlite(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", "REAL", s, flags=re.IGNORECASE)
    s = re.sub(r"\bREAL\b", "REAL", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\((\d+)\)", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"CHAR\((\d+)\)", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "TIMESTAMP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATE\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBLOB\b", "BLOB", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_mysql(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "JSON", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "JSON", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfulltext\s+TSVECTOR\b", "`fulltext` TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "INT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BOOLEAN", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"DECIMAL(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\((\d+)\)", r"VARCHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"CHAR\((\d+)\)", r"CHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "DATETIME", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATE\b", "DATE", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_sqlserver(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "NVARCHAR(MAX)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "NVARCHAR(MAX)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfulltext\s+TSVECTOR\b", "[fulltext] NVARCHAR(MAX)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "NVARCHAR(MAX)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "INT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BIT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "NVARCHAR(MAX)", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"DECIMAL(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bVARCHAR\((\d+)\)", r"NVARCHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<![A-Za-z])CHAR\((\d+)\)", r"NCHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "DATETIME2", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_oracle(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "CLOB", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "CLOB", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfulltext\s+TSVECTOR\b", '"fulltext" CLOB', s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "CLOB", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "NUMBER(10)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "NUMBER(10)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "NUMBER(1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "CLOB", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"NUMBER(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bVARCHAR\((\d+)\)", r"VARCHAR2(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<![A-Za-z])CHAR\((\d+)\)", r"CHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "TIMESTAMP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATE\b", "DATE", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBLOB\b", "BLOB", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_snowflake(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "ARRAY", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "VARIANT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "NUMBER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "NUMBER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BOOLEAN", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "TEXT", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"NUMBER(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\((\d+)\)", "VARCHAR", s, flags=re.IGNORECASE)
    s = re.sub(r"CHAR\((\d+)\)", "VARCHAR", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "TIMESTAMP_NTZ", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_bigquery(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "ARRAY<STRING>", s, flags=re.IGNORECASE)
    s = re.sub(r"ARRAY<STRING>\s+NOT\s+NULL", "ARRAY<STRING>", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "JSON", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "INT64", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INT64", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BOOL", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"NUMERIC", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\((\d+)\)", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"CHAR\((\d+)\)", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "TIMESTAMP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATE\b", "DATE", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_redshift(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "VARCHAR(65535)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "VARCHAR(65535)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "VARCHAR(65535)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BOOLEAN", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "VARCHAR(65535)", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"DECIMAL(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\((\d+)\)", r"VARCHAR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"CHAR\((\d+)\)", r"CHAR(\1)", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def _translate_body_databricks(body: str) -> str:
    s = body
    s = re.sub(r"\bTEXT\s*\[\s*\]", "ARRAY<STRING>", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJSONB\b", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTSVECTOR\b", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSMALLINT\b", "INT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bINTEGER\b", "INT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "BOOLEAN", s, flags=re.IGNORECASE)
    s = re.sub(r"VARCHAR\(\d+\)", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"CHAR\(\d+\)", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTEXT\b", "STRING", s, flags=re.IGNORECASE)
    s = re.sub(r"NUMERIC\((\d+,\d+)\)", r"DECIMAL(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTIMESTAMP\b", "TIMESTAMP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATE\b", "DATE", s, flags=re.IGNORECASE)
    s = _strip_defaults(s)
    return s


def translate_alter_pk(engine: str, table: str, schema: str, pk_cols: list[str]) -> str | None:
    """Return an ADD PRIMARY KEY statement, or ``None`` when the engine has no enforced PKs."""

    if engine == "bigquery":
        return None
    cols = ", ".join(pk_cols)
    if engine in _MYSQL_FAMILY:
        return f"ALTER TABLE `{table}` ADD PRIMARY KEY ({cols})"
    if engine == "sqlserver":
        return f"ALTER TABLE [{schema}].[{table}] ADD CONSTRAINT [{table}_pkey] PRIMARY KEY ({cols})"
    if engine == "oracle":
        owner = schema.upper()
        col_list = ", ".join(c.upper() for c in pk_cols)
        return f'ALTER TABLE "{owner}"."{table.upper()}" ADD CONSTRAINT "{table.upper()}_PKEY" PRIMARY KEY ({col_list})'
    if engine == "snowflake":
        return f"ALTER TABLE {schema}.{table} ADD PRIMARY KEY ({cols})"
    if engine == "databricks":
        return f"ALTER TABLE {schema}.{table} ADD CONSTRAINT {table}_pkey PRIMARY KEY ({cols})"
    if engine in ("postgresql", "redshift"):
        return f"ALTER TABLE {schema}.{table} ADD CONSTRAINT {table}_pkey PRIMARY KEY ({cols})"
    return None


def translate_alter_fk(engine: str, table: str, schema: str, fk: str, constraint_name: str) -> str | None:
    """Return an ADD CONSTRAINT statement, or ``None`` when the engine has no FK support."""

    if engine == "bigquery":
        return None
    fk_clean = fk.strip()
    if engine in _MYSQL_FAMILY:
        return f"ALTER TABLE `{table}` ADD CONSTRAINT `{constraint_name}` {fk_clean}"
    if engine == "sqlserver":
        return f"ALTER TABLE [{schema}].[{table}] ADD CONSTRAINT [{constraint_name}] {fk_clean}"
    if engine == "oracle":
        owner = schema.upper()
        fk_clean = re.sub(
            r"\bREFERENCES\s+(\w+)\s*\(",
            lambda m: f'REFERENCES "{owner}"."{m.group(1).upper()}" (',
            fk_clean,
            count=1,
            flags=re.IGNORECASE,
        )
        fk_clean = re.sub(
            r"FOREIGN\s+KEY\s*\(([^)]+)\)",
            lambda m: "FOREIGN KEY (" + ", ".join(p.strip().upper() for p in m.group(1).split(",")) + ")",
            fk_clean,
            count=1,
            flags=re.IGNORECASE,
        )
        return f'ALTER TABLE "{owner}"."{table.upper()}" ADD CONSTRAINT "{constraint_name.upper()}" {fk_clean}'
    if engine == "snowflake":
        fk_clean = re.sub(
            r"\bREFERENCES\s+(\w+)\s*\(",
            rf"REFERENCES {schema}.\1(",
            fk_clean,
            count=1,
            flags=re.IGNORECASE,
        )
        return f"ALTER TABLE {schema}.{table} ADD CONSTRAINT {constraint_name} {fk_clean}"
    if engine == "databricks":
        fk_clean = re.sub(
            r"\bREFERENCES\s+(\w+)\s*\(",
            rf"REFERENCES {schema}.\1(",
            fk_clean,
            count=1,
            flags=re.IGNORECASE,
        )
        return f"ALTER TABLE {schema}.{table} ADD CONSTRAINT {constraint_name} {fk_clean}"
    if engine in ("postgresql", "redshift"):
        fk_clean = re.sub(
            r"\bREFERENCES\s+(\w+)\s*\(",
            rf"REFERENCES {schema}.\1(",
            fk_clean,
            count=1,
            flags=re.IGNORECASE,
        )
        return f"ALTER TABLE {schema}.{table} ADD CONSTRAINT {constraint_name} {fk_clean}"
    return None


def _json_object_cell(cell: object) -> str | None:
    """Normalize a JSON object CSV cell to compact JSON text."""

    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    raw = str(cell).strip()
    if not raw:
        return "{}"
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.dumps(json.loads(raw), separators=(",", ":"), sort_keys=True)
        except json.JSONDecodeError:
            return raw
    return raw


_MYSQL_FAMILY = frozenset({"mysql", "mariadb"})

_SUPPORTED = {
    "mysql": MySQLRuntimeConfig,
    "mariadb": MariaDBRuntimeConfig,
    "sqlserver": SQLServerRuntimeConfig,
    "oracle": OracleRuntimeConfig,
    "snowflake": SnowflakeRuntimeConfig,
    "bigquery": BigQueryRuntimeConfig,
    "redshift": RedshiftRuntimeConfig,
    "duckdb": DuckDBRuntimeConfig,
    "sqlite": SQLiteRuntimeConfig,
}

_TABLE_ORDER = [
    "actor",
    "category",
    "country",
    "language",
    "city",
    "address",
    "author",
    "publisher",
    "item",
    "film",
    "book",
    "game",
    "game_supported_language",
    "item_category",
    "film_actor",
    "item_feature",
    "store",
    "staff",
    "inventory",
    "inventory_status_history",
    "customer",
    "courier",
    "supplier",
    "warehouse",
    "promotion",
    "rental",
    "reservation",
    "damage_report",
    "payment",
    "delivery",
    "purchase_order",
    "purchase_line",
    "stock_transfer",
    "promotion_redemption",
]

_STAGING_DROP_TABLES = (
    "film_stg",
    "game_stg",
)

_RENTAL_SHOP_VIEWS_SQL = _DATA / "rental_shop_views.sql"
_RENTAL_SHOP_VIEW_NAMES = ("active_customer_v", "store_revenue_v", "film_catalog_v")


def _iter_rental_shop_view_statements() -> list[str]:
    text = _RENTAL_SHOP_VIEWS_SQL.read_text(encoding="utf-8")
    stmts: list[str] = []
    for raw in text.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        lines = [line for line in stmt.splitlines() if line.strip() and not line.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            stmts.append(cleaned)
    return stmts


def _drop_rental_shop_views(execute: Any) -> None:
    for name in _RENTAL_SHOP_VIEW_NAMES:
        execute(f'DROP VIEW IF EXISTS "{name}"')


def _create_rental_shop_views(execute: Any) -> None:
    for stmt in _iter_rental_shop_view_statements():
        execute(stmt)


def _default_schema(engine: str) -> str:
    if engine == "redshift":
        return os.environ.get("REDSHIFT_SCHEMA", "dvdrental_new")
    if engine == "sqlserver":
        return os.environ.get("SQLSERVER_SCHEMA", "dbo")
    if engine == "oracle":
        return (
            os.environ.get("ORACLE_SCHEMA")
            or (OracleRuntimeConfig.SCHEMA if OracleRuntimeConfig.SCHEMA else None)
            or (str(OracleRuntimeConfig.USER).upper() if OracleRuntimeConfig.USER else "RENTAL_SHOP")
        )
    if engine == "snowflake":
        return os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    if engine == "bigquery":
        return (
            os.environ.get("BIGQUERY_DATASET")
            or BigQueryRuntimeConfig.DATASET
            or BigQueryRuntimeConfig.SCHEMA
            or "dvdrental_new"
        )
    if engine == "postgresql":
        return os.environ.get("PGSCHEMA", "public")
    return "public"


def _drop_table_sql(engine: str, schema: str, table: str) -> str:
    if engine in _MYSQL_FAMILY:
        return f"DROP TABLE IF EXISTS `{table}`"
    if engine == "duckdb":
        return f'DROP TABLE IF EXISTS "{table}"'
    if engine == "sqlite":
        return f'DROP TABLE IF EXISTS "{table}"'
    if engine == "sqlserver":
        return f"IF OBJECT_ID(N'[{schema}].[{table}]', N'U') IS NOT NULL DROP TABLE [{schema}].[{table}]"
    if engine == "oracle":
        owner = schema.upper()
        return f'DROP TABLE IF EXISTS "{owner}"."{table.upper()}" CASCADE CONSTRAINTS'
    if engine == "snowflake":
        return f"DROP TABLE IF EXISTS {schema}.{table}"
    if engine == "bigquery":
        return f"DROP TABLE IF EXISTS `{schema}.{table}`"
    if engine == "redshift":
        return f"DROP TABLE IF EXISTS {schema}.{table} CASCADE"
    if engine == "databricks":
        return f"DROP TABLE IF EXISTS {schema}.{table}"
    raise ValueError(engine)


def _drop_sqlserver_foreign_keys(conn, schema: str) -> None:
    """Drop all FK constraints in *schema* before table drops (store↔staff cycle)."""

    conn.execute(
        text(
            """
            DECLARE @sql NVARCHAR(MAX) = N'';
            SELECT @sql = @sql + N'ALTER TABLE '
                + QUOTENAME(s.name) + N'.' + QUOTENAME(t.name)
                + N' DROP CONSTRAINT ' + QUOTENAME(fk.name) + N';'
            FROM sys.foreign_keys fk
            INNER JOIN sys.tables t ON fk.parent_object_id = t.object_id
            INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
            WHERE s.name = :schema;
            IF @sql <> N'' EXEC sp_executesql @sql;
            """
        ),
        {"schema": schema},
    )


def _drop_oracle_foreign_keys(conn, schema: str) -> None:
    """Drop all FK constraints in *schema* before table drops (store↔staff cycle)."""

    owner = schema.upper()
    rows = conn.execute(
        text("SELECT table_name, constraint_name FROM all_constraints WHERE owner = :owner AND constraint_type = 'R'"),
        {"owner": owner},
    ).fetchall()
    for table_name, constraint_name in rows:
        conn.execute(text(f'ALTER TABLE "{owner}"."{table_name}" DROP CONSTRAINT "{constraint_name}"'))


def _drop_databricks_foreign_keys(conn, target_schema: str, fk_map: dict[str, list[str]]) -> None:
    rows = conn.execute(text(f"SHOW TABLES IN {target_schema}")).fetchall()
    existing = set()
    for row in rows:
        name = row[1] if len(row) > 1 else row[0]
        existing.add(str(name).lower())
    for table in reversed(_TABLE_ORDER):
        if table.lower() not in existing:
            continue
        for i in range(len(fk_map.get(table, []))):
            conn.execute(text(f"ALTER TABLE {target_schema}.{table} DROP CONSTRAINT IF EXISTS {table}_fk_{i}"))


def _databricks_sql_literal(value: object) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(int(value)) if isinstance(value, float) and value == int(value) else str(value)
    text_val = str(value).replace("'", "''")
    return f"'{text_val}'"


def _databricks_create_string_staging(conn, target: str, stg_table: str, columns: list[str]) -> None:
    conn.execute(text(f"DROP TABLE IF EXISTS {target}.{stg_table}"))
    col_defs = ", ".join(f"{col} STRING" for col in columns)
    conn.execute(text(f"CREATE TABLE {target}.{stg_table} ({col_defs})"))


def _databricks_insert_dataframe(
    conn,
    target: str,
    table: str,
    frame: pd.DataFrame,
    *,
    chunk_size: int = 200,
) -> None:
    if frame.empty:
        return
    cols = list(frame.columns)
    col_list = ", ".join(cols)
    for start in range(0, len(frame), chunk_size):
        batch = frame.iloc[start : start + chunk_size]
        values_sql = ", ".join(
            "(" + ", ".join(_databricks_sql_literal(row[c]) for c in cols) + ")" for _, row in batch.iterrows()
        )
        try:
            conn.exec_driver_sql(f"INSERT INTO {target}.{table} ({col_list}) VALUES {values_sql}")
        except Exception as exc:
            raise RuntimeError(
                f"[load] databricks: failed inserting {table} "
                f"(rows {start + 1}-{start + len(batch)}): {_short_load_error_message(exc)}"
            ) from exc


def _strip_pg_tz_offset(cell: object) -> object:
    """Strip PostgreSQL timestamptz offsets without treating bare dates as offsets."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return cell
    s = str(cell).strip()
    if not s:
        return cell
    if ":" in s:
        s = re.sub(r"[+-]\d{2}(:\d{2})?$", "", s).strip()
        if "." in s:
            base, _frac = s.split(".", 1)
            s = base
    elif re.search(r"\+\d{2}(:\d{2})?$", s):
        s = re.sub(r"\+\d{2}(:\d{2})?$", "", s).strip()
    if "T" in s:
        s = s.replace("T", " ", 1)
    return s


_NAIVE_DATETIME_ENGINES = frozenset({"mysql", "mariadb", "sqlserver", "oracle", "redshift", "bigquery"})


def _prepare_dataframe(engine: str, table: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    int_cols = integer_columns_for_table(table)
    json_cols: dict[str, str] = {}
    json_col = json_cols.get(table)
    if json_col and json_col in out.columns:
        if engine != "postgresql":
            out[json_col] = out[json_col].map(_json_object_cell)
    if engine in _NAIVE_DATETIME_ENGINES:
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].map(_strip_pg_tz_offset)
    bool_cols = [c for c in out.columns if c == "activebool"]
    if table == "staff" and "active" in out.columns:
        bool_cols.append("active")
    for bool_name in ("is_active",):
        if bool_name in out.columns:
            bool_cols.append(bool_name)
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].map(_coerce_bool)
    if table == "address":
        if "district" in out.columns:
            out["district"] = out["district"].fillna("").replace("", "unknown")
        if "phone" in out.columns:
            out["phone"] = out["phone"].fillna("").replace("", "0000000000")
    for col in out.columns:
        if col in int_cols:
            continue
        if out[col].dtype in ("float64", "float32"):
            out[col] = out[col].map(
                lambda v: (
                    None
                    if v is None or (isinstance(v, float) and pd.isna(v))
                    else int(v)
                    if isinstance(v, float) and v == int(v)
                    else v
                )
            )
    for col in out.columns:
        if col in int_cols:
            continue
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else v
            )
    for col in int_cols:
        if col in out.columns:
            out[col] = _coerce_integer_column(out[col])
    return out


def _coerce_integer_column(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce").astype("Int64")
    return pd.Series(
        [None if pd.isna(v) else int(v) for v in nums],
        index=series.index,
        dtype=object,
    )


def _short_load_error_message(exc: BaseException) -> str:
    msg = str(exc).strip()
    if "[SQL:" in msg:
        msg = msg.split("[SQL:", 1)[0].strip()
    if len(msg) > 400:
        msg = msg[:397] + "..."
    return msg


def _coerce_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes"):
        return True
    if text in ("0", "false", "f", "no"):
        return False
    return bool(value)


_BQ_TIMESTAMP_COLS = frozenset({"last_update", "rental_date", "return_date", "payment_date"})
_BQ_DATE_COLS = frozenset({"create_date", "start_date", "end_date", "ordered_date", "received_date"})


def _bq_coerce_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce rental_shop date/timestamp text columns to pandas datetimes for BigQuery load jobs."""

    out = frame.copy()
    for col in out.columns:
        if col in _BQ_TIMESTAMP_COLS:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                continue
            out[col] = pd.to_datetime(out[col], format="mixed", errors="coerce", utc=True)
        elif col in _BQ_DATE_COLS:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.date
                continue
            out[col] = pd.to_datetime(out[col], format="mixed", errors="coerce").dt.date
    return out


def _load_bigquery_table(client, project: str, dataset: str, table: str, frame: pd.DataFrame) -> None:
    """Bulk-load a dataframe into an existing BigQuery table via a load job (fast, no DML quota)."""

    from google.cloud import bigquery

    ref = f"{project}.{dataset}.{table}"
    schema = client.get_table(ref).schema
    df = _bq_coerce_dtypes(frame)
    for field in schema:
        if field.name not in df.columns or field.mode == "REPEATED":
            continue
        if field.field_type in ("TIMESTAMP", "DATETIME"):
            if not pd.api.types.is_datetime64_any_dtype(df[field.name]):
                df[field.name] = pd.to_datetime(df[field.name], format="mixed", errors="coerce", utc=True)
        elif field.field_type == "DATE":
            if pd.api.types.is_datetime64_any_dtype(df[field.name]):
                df[field.name] = df[field.name].dt.date
            else:
                df[field.name] = pd.to_datetime(df[field.name], format="mixed", errors="coerce").dt.date
        elif field.field_type == "STRING":
            df[field.name] = df[field.name].map(lambda v: None if (v is None or pd.isna(v)) else str(v))
        elif field.field_type in ("NUMERIC", "BIGNUMERIC"):
            df[field.name] = df[field.name].map(lambda v: None if (v is None or pd.isna(v)) else Decimal(str(v)))
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND, schema=schema)
    client.load_table_from_dataframe(df, ref, job_config=job_config).result()


def _ensure_mysql_database(runtime_cls, database: str) -> None:
    from urllib.parse import quote

    user_q = quote(str(runtime_cls.USER), safe="")
    pwd_q = quote(str(runtime_cls.PASSWORD or ""), safe="")
    admin_url = f"mysql+pymysql://{user_q}:{pwd_q}@{runtime_cls.HOST}:{runtime_cls.PORT}/mysql?charset=utf8mb4"
    engine = create_engine(admin_url, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        )


def _ensure_sqlserver_database(runtime_cls, database: str) -> None:
    master_url = make_url(runtime_cls.db_url()).set(database="master")
    engine = create_engine(master_url, future=True, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text(f"IF DB_ID(N'{database}') IS NULL CREATE DATABASE [{database}]"))


def _ensure_snowflake_schema(conn, schema: str) -> None:
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))


def _ensure_redshift_schema(conn, schema: str) -> None:
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))


def _resolve_embedded_db_path(raw: str, default_filename: str) -> Path:
    """Resolve DUCKDB_PATH/SQLITE_PATH to a concrete database file."""

    path = Path(raw)
    if raw == ":memory:":
        return path
    if path.is_dir() or (not path.suffix and not path.exists()):
        return path / default_filename
    return path


_EMBEDDED_BOOL_COLUMNS = frozenset({"active", "activebool", "is_active"})


def _coerce_embedded_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(float(str(value).strip()))


def _embedded_cell_sqlite(table: str, col: str, value: object) -> object:
    if col == "phone" and not value:
        return "0000000000"
    if col == "district" and not value:
        return "unknown"
    if col in integer_columns_for_table(table):
        return _coerce_embedded_int(value)
    return value if value != "" else None


def _embedded_cell_duckdb(table: str, col: str, value: object) -> object:
    if col in _EMBEDDED_BOOL_COLUMNS or col.startswith("is_"):
        if not value:
            return None
        return str(value).lower() in ("1", "true", "t", "yes")
    if col in integer_columns_for_table(table):
        return _coerce_embedded_int(value)
    return value if value != "" else None


def _load_duckdb(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    DuckDBRuntimeConfig.apply_environment(os.environ)
    db_path = _resolve_embedded_db_path(DuckDBRuntimeConfig.DATABASE_PATH or ":memory:", "rental_shop.duckdb")
    if str(db_path) == ":memory:":
        raise SystemExit("Set DUCKDB_PATH in env.env to a file path (not :memory:)")

    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.drop_first and db_path.exists():
        db_path.unlink()
    ddl_text = args.ddl.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for block in iter_create_table_blocks(ddl_text):
        name, _, _ = split_block(block)
        blocks[name] = block

    con = duckdb.connect(str(db_path))
    try:
        if args.drop_first:
            _drop_rental_shop_views(con.execute)
            for table in reversed(_STAGING_DROP_TABLES):
                con.execute(f'DROP TABLE IF EXISTS "{table}"')
            for table in reversed(_TABLE_ORDER):
                con.execute(f'DROP TABLE IF EXISTS "{table}"')

        for table in _TABLE_ORDER:
            block = blocks.get(table)
            if block is None:
                raise SystemExit(f"Missing CREATE TABLE block for {table}")
            create_sql = translate_create("duckdb", block, schema="main")
            con.execute(create_sql)

        for table in _TABLE_ORDER:
            csv_path = args.csv_dir / f"{table}.csv"
            if not csv_path.is_file():
                raise SystemExit(f"Missing CSV: {csv_path}")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if not rows:
                continue
            columns = list(rows[0].keys())
            col_sql = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" for _ in columns)
            insert_sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'
            for row in rows:
                values = [_embedded_cell_duckdb(table, col, row.get(col)) for col in columns]
                con.execute(insert_sql, values)
            _log_load_table("duckdb", table, len(rows))
        _drop_rental_shop_views(con.execute)
        _create_rental_shop_views(con.execute)
    finally:
        con.close()
    _log_progress(f"[load] duckdb: finished ({db_path})")


def _load_sqlite(args: argparse.Namespace) -> None:
    import sqlite3

    load_env_file(args.env_file)
    SQLiteRuntimeConfig.apply_environment(os.environ)
    db_path = _resolve_embedded_db_path(SQLiteRuntimeConfig.DATABASE_PATH or ":memory:", "rental_shop.sqlite")
    if str(db_path) == ":memory:":
        raise SystemExit("Set SQLITE_PATH in env.env to a file path (not :memory:)")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.drop_first and db_path.exists():
        db_path.unlink()
    ddl_text = args.ddl.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for block in iter_create_table_blocks(ddl_text):
        name, _, _ = split_block(block)
        blocks[name] = block

    conn = sqlite3.connect(str(db_path))
    try:
        if args.drop_first:
            _drop_rental_shop_views(conn.execute)
            for table in reversed(_STAGING_DROP_TABLES):
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            for table in reversed(_TABLE_ORDER):
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')

        for table in _TABLE_ORDER:
            block = blocks.get(table)
            if block is None:
                raise SystemExit(f"Missing CREATE TABLE block for {table}")
            create_sql = translate_create("sqlite", block, schema="main")
            conn.execute(create_sql)

        for table in _TABLE_ORDER:
            csv_path = args.csv_dir / f"{table}.csv"
            if not csv_path.is_file():
                raise SystemExit(f"Missing CSV: {csv_path}")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if not rows:
                continue
            columns = list(rows[0].keys())
            col_sql = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})',
                [tuple(_embedded_cell_sqlite(table, col, row.get(col)) for col in columns) for row in rows],
            )
            _log_load_table("sqlite", table, len(rows))
        _drop_rental_shop_views(conn.execute)
        _create_rental_shop_views(conn.execute)
        conn.commit()
    finally:
        conn.close()
    _log_progress(f"[load] sqlite: finished ({db_path})")


def _load_sqlalchemy_engine(engine_name: str, args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    runtime_cls = _SUPPORTED[engine_name]
    runtime_cls.apply_environment(__import__("os").environ)
    schema = args.schema or _default_schema(engine_name)

    if engine_name in _MYSQL_FAMILY:
        env_key = "MARIADB_DATABASE" if engine_name == "mariadb" else "MYSQL_DATABASE"
        database = runtime_cls.DATABASE or __import__("os").environ.get(env_key, "rental_shop")
        runtime_cls.DATABASE = database
        _ensure_mysql_database(runtime_cls, database)
    if engine_name == "sqlserver":
        database = runtime_cls.DATABASE or __import__("os").environ.get("SQLSERVER_DATABASE", "rental_shop")
        runtime_cls.DATABASE = database
        _ensure_sqlserver_database(runtime_cls, database)
    if engine_name == "oracle":
        runtime_cls.ensure_driver_mode()
        if not runtime_cls.SCHEMA:
            runtime_cls.SCHEMA = str(runtime_cls.USER).upper() if runtime_cls.USER else "RENTAL_SHOP"

    url = runtime_cls.db_url()
    connect_args = runtime_cls.connect_args() if hasattr(runtime_cls, "connect_args") else {}
    sa_engine = create_engine(url, connect_args=connect_args, future=True)

    bq_client = None
    if engine_name == "bigquery":
        from google.cloud import bigquery

        if runtime_cls.CREDENTIALS_PATH:
            bq_client = bigquery.Client.from_service_account_json(
                runtime_cls.CREDENTIALS_PATH,
                project=runtime_cls.PROJECT,
                location=runtime_cls.LOCATION,
            )
        else:
            bq_client = bigquery.Client(project=runtime_cls.PROJECT, location=runtime_cls.LOCATION)

    ddl_text = args.ddl.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for block in iter_create_table_blocks(ddl_text):
        name, _, _ = split_block(block)
        blocks[name] = block
    fk_map: dict[str, list[str]] = {}
    for table, fk in iter_alter_foreign_keys(ddl_text):
        fk_map.setdefault(table, []).append(fk)
    pk_map: dict[str, list[str]] = {}
    for table, pk_cols in iter_alter_primary_keys(ddl_text):
        pk_map[table] = pk_cols

    with sa_engine.begin() as conn:
        if engine_name == "snowflake":
            _ensure_snowflake_schema(conn, schema)
            conn.execute(text(f"USE SCHEMA {schema}"))
        if engine_name == "oracle":
            conn.execute(text("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"))
            conn.execute(text("ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS'"))
            conn.execute(text("ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = 'YYYY-MM-DD HH24:MI:SS'"))
        if engine_name == "redshift":
            if args.drop_first:
                conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            _ensure_redshift_schema(conn, schema)

        if args.drop_first:
            if engine_name in _MYSQL_FAMILY:
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            if engine_name == "sqlserver":
                _drop_sqlserver_foreign_keys(conn, schema)
            if engine_name == "oracle":
                _drop_oracle_foreign_keys(conn, schema)
            if engine_name != "redshift":
                for table in reversed(_STAGING_DROP_TABLES):
                    conn.execute(text(_drop_table_sql(engine_name, schema, table)))
                for table in reversed(_TABLE_ORDER):
                    conn.execute(text(_drop_table_sql(engine_name, schema, table)))
            if engine_name in _MYSQL_FAMILY:
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        for table in _TABLE_ORDER:
            block = blocks.get(table)
            if block is None:
                raise SystemExit(f"Missing CREATE TABLE block for {table}")
            create_sql = translate_create(engine_name, block, schema=schema)
            conn.execute(text(create_sql))

        for table in _TABLE_ORDER:
            pk_cols = pk_map.get(table)
            if pk_cols:
                pk_stmt = translate_alter_pk(engine_name, table, schema, pk_cols)
                if pk_stmt:
                    conn.execute(text(pk_stmt))

        for table in _TABLE_ORDER:
            csv_path = args.csv_dir / f"{table}.csv"
            if not csv_path.is_file():
                raise SystemExit(f"Missing CSV: {csv_path}")
            frame = pd.read_csv(csv_path)
            frame = _prepare_dataframe(engine_name, table, frame)
            try:
                if engine_name in _MYSQL_FAMILY:
                    frame.to_sql(
                        table,
                        conn,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=500,
                    )
                elif engine_name == "sqlserver":
                    frame.to_sql(
                        table,
                        conn,
                        schema=schema,
                        if_exists="append",
                        index=False,
                        chunksize=100,
                    )
                elif engine_name == "oracle":
                    upper_frame = frame.copy()
                    upper_frame.columns = [str(c).upper() for c in upper_frame.columns]
                    upper_frame.to_sql(
                        table.upper(),
                        conn,
                        schema=schema.upper(),
                        if_exists="append",
                        index=False,
                        chunksize=100,
                    )
                elif engine_name == "snowflake":
                    frame.to_sql(
                        table.lower(),
                        conn,
                        schema=schema.lower(),
                        if_exists="append",
                        index=False,
                        chunksize=500,
                    )
                elif engine_name == "bigquery":
                    _load_bigquery_table(bq_client, runtime_cls.PROJECT, schema, table, frame)
                elif engine_name == "redshift":
                    frame.to_sql(
                        table,
                        conn,
                        schema=schema,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=200,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"[load] {engine_name}: failed loading {table}: {_short_load_error_message(exc)}"
                ) from exc
            _log_load_table(engine_name, table, len(frame))

        if engine_name != "bigquery":
            for table in _TABLE_ORDER:
                for i, fk in enumerate(fk_map.get(table, [])):
                    stmt = translate_alter_fk(engine_name, table, schema, fk, f"{table}_fk_{i}")
                    if stmt:
                        conn.execute(text(stmt))

    _log_progress(f"[load] {engine_name}: finished (schema/dataset={schema})")


def _load_postgresql(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    if args.schema is None:
        args.schema = os.environ.get("POSTGRESQL_SCHEMA", "public")

    import psycopg
    from psycopg import sql

    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        dbname=args.database,
    )
    conn.autocommit = True
    cur = conn.cursor()

    ddl_text = args.ddl.read_text(encoding="utf-8")
    parsed: list[tuple[str, str, list[str]]] = []
    for block in iter_create_table_blocks(ddl_text):
        name, no_fk, fks = split_block(block)
        q = qualify_table_name(no_fk, args.schema)
        parsed.append((name, q, fks))
    pk_map: dict[str, list[str]] = {}
    for table, pk_cols in iter_alter_primary_keys(ddl_text):
        pk_map[table] = pk_cols
    fk_map: dict[str, list[str]] = {}
    for table, fk in iter_alter_foreign_keys(ddl_text):
        fk_map.setdefault(table, []).append(fk)

    if args.recreate_schema:
        if args.schema == "public" and not args.allow_public_schema_recreate:
            raise SystemExit(
                "Refusing to recreate schema 'public'. Use a dedicated schema or pass --allow-public-schema-recreate."
            )
        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(args.schema)))
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(args.schema)))

    if args.drop_first:
        for name in reversed(_STAGING_DROP_TABLES):
            cur.execute(f"DROP TABLE IF EXISTS {args.schema}.{name} CASCADE")
        for name, _, _ in reversed(parsed):
            cur.execute(f"DROP TABLE IF EXISTS {args.schema}.{name} CASCADE")

    for _, create_sql, _ in parsed:
        cur.execute(create_sql)

    for name, _, _ in parsed:
        pk_cols = pk_map.get(name)
        if pk_cols:
            cols = ", ".join(pk_cols)
            cur.execute(f"ALTER TABLE {args.schema}.{name} ADD CONSTRAINT {name}_pkey PRIMARY KEY ({cols})")

    for name, _, _ in parsed:
        csv_path = args.csv_dir / f"{name}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing csv {csv_path}")
        fq = f"{args.schema}.{name}"
        copy_sql = f"COPY {fq} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
        frame = pd.read_csv(csv_path)
        frame = _prepare_dataframe("postgresql", name, frame)
        buf = io.StringIO()
        frame.to_csv(buf, index=False, header=True, na_rep="")
        buf.seek(0)
        with cur.copy(copy_sql) as copy:
            copy.write(buf.getvalue())
        _log_load_table("postgresql", name, len(frame))

    for name in _TABLE_ORDER:
        for i, fk in enumerate(fk_map.get(name, [])):
            cname = f"{name}_fk_{i}"
            cur.execute(alter_fk_sql(name, args.schema, fk, cname))

    _create_rental_shop_views(cur.execute)

    cur.close()
    conn.close()
    _log_progress(f"[load] postgresql: finished (schema={args.schema}, database={args.database})")


_ENGINE_CONFIG = {
    "postgresql": PostgresRuntimeConfig,
    "mysql": MySQLRuntimeConfig,
    "mariadb": MariaDBRuntimeConfig,
    "sqlserver": SQLServerRuntimeConfig,
    "oracle": OracleRuntimeConfig,
    "snowflake": SnowflakeRuntimeConfig,
    "bigquery": BigQueryRuntimeConfig,
    "databricks": DatabricksRuntimeConfig,
    "redshift": RedshiftRuntimeConfig,
    "duckdb": DuckDBRuntimeConfig,
    "sqlite": SQLiteRuntimeConfig,
}


def _runtime(engine_name: str):
    runtime_cls = _ENGINE_CONFIG.get(engine_name)
    if runtime_cls is None:
        raise SystemExit(f"Unsupported engine {engine_name!r}. Choose one of: {', '.join(sorted(_ENGINE_CONFIG))}")
    return runtime_cls


def _cmd_ping(args: argparse.Namespace) -> None:
    env_path = load_env_file(args.env_file)
    runtime_cls = _runtime(args.engine)
    runtime_cls.apply_environment(__import__("os").environ)

    if args.engine == "databricks":
        url = runtime_cls.sqlalchemy_url()
        if not url:
            raise SystemExit("Databricks ping requires DATABRICKS_HOST, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN")
    else:
        url = runtime_cls.db_url()
    connect_args = runtime_cls.connect_args() if hasattr(runtime_cls, "connect_args") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    try:
        with engine.connect() as conn:
            ping_sql = "SELECT 1 FROM DUAL" if args.engine == "oracle" else "SELECT 1"
            row = conn.execute(text(ping_sql)).scalar_one()
    finally:
        engine.dispose()
    print(f"OK {args.engine} via {env_path} ({ping_sql} -> {row})")


_EXPECTED_FILM_ROWS = 1000
_EXPECTED_TRAILERS = 535


def _verify_schema_name(engine: str, schema_arg: str | None) -> str:
    if schema_arg:
        return schema_arg
    if engine == "redshift":
        return os.environ.get("REDSHIFT_SCHEMA", "dvdrental_new")
    if engine == "sqlserver":
        return os.environ.get("SQLSERVER_SCHEMA", "dbo")
    if engine == "oracle":
        return os.environ.get("ORACLE_SCHEMA") or (
            str(OracleRuntimeConfig.USER).upper() if OracleRuntimeConfig.USER else "RENTAL_SHOP"
        )
    if engine == "snowflake":
        return os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    if engine == "bigquery":
        return os.environ.get("BIGQUERY_DATASET", "dvdrental_new")
    if engine == "postgresql":
        return os.environ.get("PGSCHEMA", "public")
    return _default_schema(engine)


def _qualified_table(engine: str, schema: str, table: str) -> str:
    if engine in _MYSQL_FAMILY:
        return f"`{table}`"
    if engine == "duckdb":
        return f'"{table}"'
    if engine == "sqlite":
        return f'"{table}"'
    if engine == "sqlserver":
        return f"[{schema}].[{table}]"
    if engine == "oracle":
        return f'"{schema.upper()}"."{table.upper()}"'
    if engine == "snowflake":
        return f"{schema}.{table.lower()}"
    if engine == "bigquery":
        project = BigQueryRuntimeConfig.PROJECT or os.environ.get("BIGQUERY_PROJECT", "")
        return f"`{project}.{schema}.{table}`"
    if engine == "redshift":
        return f'"{schema}"."{table}"'
    if engine == "postgresql":
        return f'"{schema}"."{table}"'
    return table


def _trailers_sql(engine: str, schema: str) -> str:
    item_feature = _qualified_table(engine, schema, "item_feature")
    return f"SELECT COUNT(*) FROM {item_feature} WHERE LOWER(feature_name) = 'trailers'"


def _validate_verify_metrics(engine_name: str, counts: dict[str, int], trailers: int) -> None:
    missing = [table for table in _TABLE_ORDER if counts.get(table, 0) <= 0]
    if missing:
        raise SystemExit(f"{engine_name} verify failed: missing or empty tables: {', '.join(missing)}")
    film_count = counts.get("film", 0)
    if film_count != _EXPECTED_FILM_ROWS:
        raise SystemExit(f"{engine_name} verify failed: film rows expected {_EXPECTED_FILM_ROWS}, got {film_count}")
    if trailers < _EXPECTED_TRAILERS:
        raise SystemExit(
            f"{engine_name} verify failed: trailers expected >= {_EXPECTED_TRAILERS}, "
            f"got {trailers}. Re-run load --drop-first from scripts/data/rental_shop_csvs/"
        )


def _log_verify_summary(engine_name: str, counts: dict[str, int], trailers: int) -> None:
    if _VERIFY_VERBOSE:
        parts = ", ".join(f"{table}={counts[table]}" for table in _TABLE_ORDER)
        _log_progress(f"[verify] {engine_name}: {parts}; item_feature.trailers={trailers}")
    else:
        _log_progress(f"[verify] {engine_name}: OK")


def _verify_sqlalchemy(engine_name: str, args: argparse.Namespace) -> None:
    runtime_cls = _runtime(engine_name)
    runtime_cls.apply_environment(os.environ)
    schema = _verify_schema_name(engine_name, args.schema)
    url = runtime_cls.db_url()
    connect_args = runtime_cls.connect_args() if hasattr(runtime_cls, "connect_args") else {}
    sa_engine = create_engine(url, connect_args=connect_args, future=True)
    counts: dict[str, int] = {}
    trailers = 0
    with sa_engine.connect() as conn:
        conn.execute(text("SELECT 1")).scalar_one()
        for table in _TABLE_ORDER:
            fq = _qualified_table(engine_name, schema, table)
            try:
                counts[table] = int(conn.execute(text(f"SELECT COUNT(*) FROM {fq}")).scalar_one())
            except Exception:
                counts[table] = 0
        trailers = int(conn.execute(text(_trailers_sql(engine_name, schema))).scalar_one())
    _validate_verify_metrics(engine_name, counts, trailers)
    _log_verify_summary(engine_name, counts, trailers)


def _verify_duckdb(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    DuckDBRuntimeConfig.apply_environment(os.environ)
    db_path = _resolve_embedded_db_path(DuckDBRuntimeConfig.DATABASE_PATH or ":memory:", "rental_shop.duckdb")
    if str(db_path) == ":memory:":
        raise SystemExit("Set DUCKDB_PATH in env.env to a file path (not :memory:)")
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        counts: dict[str, int] = {}
        for table in _TABLE_ORDER:
            counts[table] = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        trailers = int(
            con.execute("SELECT COUNT(*) FROM item_feature WHERE LOWER(feature_name) = 'trailers'").fetchone()[0]
        )
    finally:
        con.close()
    _validate_verify_metrics("duckdb", counts, trailers)
    _log_verify_summary("duckdb", counts, trailers)


def _verify_sqlite(args: argparse.Namespace) -> None:
    import sqlite3

    load_env_file(args.env_file)
    SQLiteRuntimeConfig.apply_environment(os.environ)
    db_path = _resolve_embedded_db_path(SQLiteRuntimeConfig.DATABASE_PATH or ":memory:", "rental_shop.sqlite")
    if str(db_path) == ":memory:":
        raise SystemExit("Set SQLITE_PATH in env.env to a file path (not :memory:)")
    conn = sqlite3.connect(str(db_path))
    try:
        counts: dict[str, int] = {}
        for table in _TABLE_ORDER:
            row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            counts[table] = int(row[0]) if row is not None else 0
        trailers = int(
            conn.execute("SELECT COUNT(*) FROM item_feature WHERE LOWER(feature_name) = 'trailers'").fetchone()[0]
        )
    finally:
        conn.close()
    _validate_verify_metrics("sqlite", counts, trailers)
    _log_verify_summary("sqlite", counts, trailers)


def _verify_postgresql(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    from aetherdialect._config import PostgresRuntimeConfig

    PostgresRuntimeConfig.apply_environment(os.environ)
    schema = _verify_schema_name("postgresql", args.schema)
    url = PostgresRuntimeConfig.db_url()
    connect_args = PostgresRuntimeConfig.connect_args() if hasattr(PostgresRuntimeConfig, "connect_args") else {}
    sa_engine = create_engine(url, connect_args=connect_args, future=True)
    counts: dict[str, int] = {}
    trailers = 0
    with sa_engine.connect() as conn:
        for table in _TABLE_ORDER:
            fq = _qualified_table("postgresql", schema, table)
            counts[table] = int(conn.execute(text(f"SELECT COUNT(*) FROM {fq}")).scalar_one())
        trailers = int(conn.execute(text(_trailers_sql("postgresql", schema))).scalar_one())
    _validate_verify_metrics("postgresql", counts, trailers)
    _log_verify_summary("postgresql", counts, trailers)


def _verify_databricks(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    from aetherdialect._config import DatabricksRuntimeConfig

    DatabricksRuntimeConfig.apply_environment(os.environ)
    schema = args.schema or os.environ.get("DATABRICKS_SCHEMA", "dvdrental_new")
    catalog = os.environ.get("DATABRICKS_CATALOG", "dev")
    url = DatabricksRuntimeConfig.sqlalchemy_url()
    if not url:
        raise SystemExit("Databricks verify requires DATABRICKS_HOST, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN")
    sa_engine = create_engine(url, future=True)
    counts: dict[str, int] = {}
    try:
        import logging

        logging.getLogger("databricks.sql").setLevel(logging.ERROR)
        with sa_engine.connect() as conn:
            conn.execute(text("SELECT 1")).scalar_one()
            for table in _TABLE_ORDER:
                fq = f"{catalog}.{schema}.{table}"
                try:
                    counts[table] = int(conn.execute(text(f"SELECT COUNT(*) FROM {fq}")).scalar_one())
                except Exception:
                    counts[table] = 0
            trailers = int(
                conn.execute(
                    text(f"SELECT COUNT(*) FROM {catalog}.{schema}.item_feature WHERE LOWER(feature_name) = 'trailers'")
                ).scalar_one()
            )
    finally:
        sa_engine.dispose()
    _validate_verify_metrics("databricks", counts, trailers)
    _log_verify_summary("databricks", counts, trailers)


def _cmd_verify(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    if args.engine == "duckdb":
        _verify_duckdb(args)
    elif args.engine == "sqlite":
        _verify_sqlite(args)
    elif args.engine == "postgresql":
        _verify_postgresql(args)
    elif args.engine == "databricks":
        _verify_databricks(args)
    elif args.engine in _SQLALCHEMY_ENGINES:
        _verify_sqlalchemy(args.engine, args)
    else:
        raise SystemExit(f"Unsupported engine for verify: {args.engine}")


def _cmd_extract_csv(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    from aetherdialect._config import PostgresRuntimeConfig

    PostgresRuntimeConfig.apply_environment(os.environ)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    import psycopg

    url = PostgresRuntimeConfig.db_url().replace("postgresql+psycopg://", "postgresql://")
    conn = psycopg.connect(url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = %s
        ORDER BY tablename;
        """,
        (args.schema,),
    )
    for (table,) in cur.fetchall():
        csv_path = out_dir / f"{table}.csv"
        with csv_path.open("wb") as f:
            with cur.copy(f"COPY {args.schema}.{table} TO STDOUT WITH CSV HEADER") as copy:
                for data in copy:
                    f.write(data)
    cur.close()
    conn.close()
    print(f"Extracted tables to {out_dir}")


def _load_databricks(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    from aetherdialect._config import DatabricksRuntimeConfig

    DatabricksRuntimeConfig.apply_environment(os.environ)
    url = DatabricksRuntimeConfig.sqlalchemy_url()
    if not url:
        raise SystemExit(
            "Databricks load requires DATABRICKS_HOST, DATABRICKS_HTTP_PATH, "
            "DATABRICKS_TOKEN, DATABRICKS_CATALOG, and DATABRICKS_SCHEMA "
            f"(from {args.env_file or DEFAULT_ENV_FILE})"
        )
    catalog = DatabricksRuntimeConfig.CATALOG or os.environ.get("DATABRICKS_CATALOG", "dev")
    schema_name = DatabricksRuntimeConfig.SCHEMA or os.environ.get("DATABRICKS_SCHEMA", "dvdrental_new")
    target = args.schema or f"{catalog}.{schema_name}"

    ddl_text = args.ddl.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for block in iter_create_table_blocks(ddl_text):
        name, _, _ = split_block(block)
        blocks[name] = block
    fk_map: dict[str, list[str]] = {}
    for table, fk in iter_alter_foreign_keys(ddl_text):
        fk_map.setdefault(table, []).append(fk)
    pk_map: dict[str, list[str]] = {}
    for table, pk_cols in iter_alter_primary_keys(ddl_text):
        pk_map[table] = pk_cols

    sa_engine = create_engine(url, future=True)
    try:
        import logging

        logging.getLogger("databricks.sql").setLevel(logging.ERROR)
        with sa_engine.begin() as conn:
            if args.drop_first:
                try:
                    conn.execute(text(f"DROP SCHEMA IF EXISTS {target} CASCADE"))
                except Exception as exc:
                    message = str(exc).lower()
                    if "schema_not_found" not in message and "42704" not in message:
                        raise
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {target}"))

            for table in _TABLE_ORDER:
                block = blocks.get(table)
                if block is None:
                    raise SystemExit(f"Missing CREATE TABLE block for {table}")
                create_sql = translate_create("databricks", block, schema=target)
                conn.execute(text(create_sql))

            for table in _TABLE_ORDER:
                pk_cols = pk_map.get(table)
                if pk_cols:
                    pk_stmt = translate_alter_pk("databricks", table, target, pk_cols)
                    if pk_stmt:
                        conn.execute(text(pk_stmt))

            for table in _TABLE_ORDER:
                csv_path = args.csv_dir / f"{table}.csv"
                if not csv_path.is_file():
                    raise SystemExit(f"Missing CSV: {csv_path}")
                frame = pd.read_csv(csv_path)
                frame = _prepare_dataframe("databricks", table, frame)
                _databricks_insert_dataframe(conn, target, table, frame)
                _log_load_table("databricks", table, len(frame))

            for table in _TABLE_ORDER:
                for i, fk in enumerate(fk_map.get(table, [])):
                    stmt = translate_alter_fk("databricks", table, target, fk, f"{table}_fk_{i}")
                    if stmt:
                        conn.execute(text(stmt))
    finally:
        sa_engine.dispose()
    _log_progress(f"[load] databricks: finished (schema={target})")


_ALL_ENGINES = sorted(
    {
        "postgresql",
        "mysql",
        "mariadb",
        "sqlserver",
        "oracle",
        "snowflake",
        "bigquery",
        "redshift",
        "databricks",
        "duckdb",
        "sqlite",
    }
)
_SQLALCHEMY_ENGINES = sorted(_SUPPORTED.keys())


def _partition_table_order(partition: frozenset[str]) -> list[str]:
    return [table for table in _TABLE_ORDER if table in partition]


def _apply_federation_column_projection(
    frame: pd.DataFrame,
    table: str,
    projections: dict[str, frozenset[str]],
) -> pd.DataFrame:
    """Drop CSV columns outside the federation declaration projection for *table*."""
    column_projection = projections.get(table)
    if not column_projection:
        return frame
    keep = [col for col in frame.columns if col in column_projection]
    return frame.loc[:, keep]


def _project_create_table_sql(create_sql: str, table: str, projections: dict[str, frozenset[str]]) -> str:
    """Drop non-projected columns from a CREATE TABLE body when a projection exists."""
    column_projection = projections.get(table)
    if not column_projection:
        return create_sql
    start = create_sql.find("(")
    end = create_sql.rfind(")")
    if start < 0 or end <= start:
        return create_sql
    body = create_sql[start + 1 : end]
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    kept: list[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        upper_part = token.upper()
        if upper_part.startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CONSTRAINT", "CHECK", "KEY ", "INDEX ")):
            continue
        name = token.split()[0].strip('`"[]')
        if name in column_projection:
            kept.append(token)
    if not kept:
        return create_sql
    return create_sql[: start + 1] + ", ".join(kept) + create_sql[end:]


def _filter_payment_frame(frame: pd.DataFrame, *, source_id: str, csv_dir: Path) -> pd.DataFrame:
    from source_rental_shop import PAYMENT_UNION_SPLIT_STORE_THRESHOLD, payment_store_id_by_rental_id

    if "rental_id" not in frame.columns:
        return frame
    rental_store = payment_store_id_by_rental_id(csv_dir)
    store_ids = frame["rental_id"].map(lambda rid: rental_store.get(int(rid), 0))
    if source_id == "storefront":
        return frame.loc[store_ids <= PAYMENT_UNION_SPLIT_STORE_THRESHOLD].copy()
    if source_id == "catalog":
        return frame.loc[store_ids > PAYMENT_UNION_SPLIT_STORE_THRESHOLD].copy()
    return frame


def _validate_federation_env(targets: tuple[str, ...]) -> None:
    """Fail fast when env.env lacks credentials for the requested federation targets."""
    blockers: list[str] = []
    if any(target in targets for target in ("storefront", "logistics")):
        blockers.extend(PostgresRuntimeConfig.selection_blockers(os.environ))
    if "catalog" in targets:
        blockers.extend(MySQLRuntimeConfig.selection_blockers(os.environ))
    if "crm" in targets:
        blockers.extend(MariaDBRuntimeConfig.selection_blockers(os.environ))
    if blockers:
        raise SystemExit(
            "Federation load/verify requires complete database credentials in env.env:\n"
            + "\n".join(f"  - {item}" for item in blockers)
        )


def _load_federation_postgresql_partition(args: argparse.Namespace, *, source_id: str, schema: str) -> None:
    from sandbox_corpus import federation_foreign_key_allowed, federation_partition_tables

    partition = federation_partition_tables(source_id)
    load_env_file(args.env_file)
    import psycopg
    from psycopg import sql as pg_sql

    PostgresRuntimeConfig.apply_environment(os.environ)
    database = args.database or PostgresRuntimeConfig.DATABASE
    if not database:
        raise SystemExit("PostgreSQL database required (set PGDATABASE in env.env)")
    conn = psycopg.connect(
        host=PostgresRuntimeConfig.HOST,
        port=int(PostgresRuntimeConfig.PORT),
        user=PostgresRuntimeConfig.USER,
        password=PostgresRuntimeConfig.PASSWORD or "",
        dbname=database,
    )
    conn.autocommit = True
    cur = conn.cursor()
    ddl_text = args.ddl.read_text(encoding="utf-8")
    parsed: list[tuple[str, str, list[str]]] = []
    for block in iter_create_table_blocks(ddl_text):
        name, no_fk, fks = split_block(block)
        if name not in partition:
            continue
        parsed.append((name, qualify_table_name(no_fk, schema), fks))
    pk_map: dict[str, list[str]] = {}
    for table, pk_cols in iter_alter_primary_keys(ddl_text):
        if table in partition:
            pk_map[table] = pk_cols
    fk_map: dict[str, list[str]] = {}
    for table, fk in iter_alter_foreign_keys(ddl_text):
        if table not in partition:
            continue
        if federation_foreign_key_allowed(table, fk, partition):
            fk_map.setdefault(table, []).append(fk)
    if args.drop_first:
        cur.execute(pg_sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(pg_sql.Identifier(schema)))
    cur.execute(pg_sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(pg_sql.Identifier(schema)))
    if args.drop_first:
        for name, _, _ in reversed(parsed):
            cur.execute(f"DROP TABLE IF EXISTS {schema}.{name} CASCADE")
    from sandbox_corpus import federation_member_column_projections

    column_projections = federation_member_column_projections(source_id)
    for name, create_sql, _ in parsed:
        create_sql = _project_create_table_sql(create_sql, name, column_projections)
        cur.execute(create_sql)
    for name, _, _ in parsed:
        pk_cols = pk_map.get(name)
        if pk_cols:
            cols = ", ".join(pk_cols)
            cur.execute(f"ALTER TABLE {schema}.{name} ADD CONSTRAINT {name}_pkey PRIMARY KEY ({cols})")
    for name, _, _ in parsed:
        csv_path = args.csv_dir / f"{name}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing csv {csv_path}")
        frame = pd.read_csv(csv_path)
        if name == "payment":
            frame = _filter_payment_frame(frame, source_id=source_id, csv_dir=args.csv_dir)
        frame = _apply_federation_column_projection(frame, name, column_projections)
        frame = _prepare_dataframe("postgresql", name, frame)
        fq = f"{schema}.{name}"
        copy_sql = f"COPY {fq} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
        buf = io.StringIO()
        frame.to_csv(buf, index=False, header=True, na_rep="")
        buf.seek(0)
        with cur.copy(copy_sql) as copy:
            copy.write(buf.getvalue())
        _log_load_table("postgresql", name, len(frame))
    for name in _partition_table_order(partition):
        for i, fk in enumerate(fk_map.get(name, [])):
            cur.execute(alter_fk_sql(name, schema, fk, f"{name}_fk_{i}"))
    loaded_tables = {name for name, _, _ in parsed}
    _load_federation_postgres_corpus_extras(
        cur,
        schema=schema,
        source_id=source_id,
        missing_tables=sorted(partition - loaded_tables),
    )
    cur.close()
    conn.close()
    _log_progress(f"[federation] postgresql {source_id}: finished (schema={schema}, database={database})")


def _load_federation_postgres_corpus_extras(
    cur: Any,
    *,
    schema: str,
    source_id: str,
    missing_tables: list[str],
) -> None:
    """Create/load partition tables present only in federation schema+CSV artifacts."""
    if not missing_tables:
        return
    schema_path = _DATA / f"federation_{source_id}_schema.sql"
    data_dir = _DATA / f"federation_{source_id}_data"
    if not schema_path.is_file():
        raise SystemExit(
            f"federation {source_id}: missing corpus schema for extras {missing_tables}: {schema_path}",
        )
    creates: dict[str, str] = {}
    for block in iter_create_table_blocks(schema_path.read_text(encoding="utf-8")):
        name, no_fk, _fks = split_block(block)
        creates[name] = qualify_table_name(no_fk, schema)
    for table in missing_tables:
        create_sql = creates.get(table)
        if create_sql is None:
            raise SystemExit(f"federation {source_id}: no CREATE for corpus-only table {table!r} in {schema_path}")
        csv_path = data_dir / f"{table}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"federation {source_id}: missing corpus CSV for {table}: {csv_path}")
        cur.execute(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE")
        cur.execute(create_sql)
        frame = pd.read_csv(csv_path)
        frame = _prepare_dataframe("postgresql", table, frame)
        fq = f"{schema}.{table}"
        copy_sql = f"COPY {fq} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
        buf = io.StringIO()
        frame.to_csv(buf, index=False, header=True, na_rep="")
        buf.seek(0)
        with cur.copy(copy_sql) as copy:
            copy.write(buf.getvalue())
        _log_load_table("postgresql", table, len(frame))


def _load_federation_mysql_partition(
    args: argparse.Namespace,
    *,
    source_id: str,
    database: str,
    runtime_cls: type = MySQLRuntimeConfig,
) -> None:
    from sandbox_corpus import federation_foreign_key_allowed, federation_partition_tables

    partition = federation_partition_tables(source_id)
    load_env_file(args.env_file)
    runtime_cls.apply_environment(os.environ)
    _ensure_mysql_database(runtime_cls, database)
    runtime_cls.DATABASE = database
    url = runtime_cls.db_url()
    sa_engine = create_engine(url, connect_args=runtime_cls.connect_args(), future=True)
    engine_name = str(getattr(runtime_cls, "ENGINE_NAME", "mysql"))
    schema = _default_schema(engine_name)
    ddl_text = args.ddl.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for block in iter_create_table_blocks(ddl_text):
        name, _, _ = split_block(block)
        if name in partition:
            blocks[name] = block
    fk_map: dict[str, list[str]] = {}
    for table, fk in iter_alter_foreign_keys(ddl_text):
        if table not in partition:
            continue
        if federation_foreign_key_allowed(table, fk, partition):
            fk_map.setdefault(table, []).append(fk)
    pk_map: dict[str, list[str]] = {}
    for table, pk_cols in iter_alter_primary_keys(ddl_text):
        if table in partition:
            pk_map[table] = pk_cols
    from sandbox_corpus import federation_member_column_projections

    column_projections = federation_member_column_projections(source_id)
    with sa_engine.begin() as conn:
        if args.drop_first:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            for table in reversed(_partition_table_order(partition)):
                conn.execute(text(_drop_table_sql(engine_name, schema, table)))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        for table in _partition_table_order(partition):
            block = blocks.get(table)
            if block is None:
                raise SystemExit(f"Missing CREATE TABLE block for {table}")
            create_sql = translate_create(engine_name, block, schema=schema)
            create_sql = _project_create_table_sql(create_sql, table, column_projections)
            conn.execute(text(create_sql))
        for table in _partition_table_order(partition):
            pk_cols = pk_map.get(table)
            if pk_cols:
                pk_stmt = translate_alter_pk(engine_name, table, schema, pk_cols)
                if pk_stmt:
                    conn.execute(text(pk_stmt))
        for table in _partition_table_order(partition):
            csv_path = args.csv_dir / f"{table}.csv"
            if not csv_path.is_file():
                raise SystemExit(f"Missing CSV: {csv_path}")
            frame = pd.read_csv(csv_path)
            if table == "payment":
                frame = _filter_payment_frame(frame, source_id=source_id, csv_dir=args.csv_dir)
            frame = _apply_federation_column_projection(frame, table, column_projections)
            frame = _prepare_dataframe(engine_name, table, frame)
            frame.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=500)
            _log_load_table(engine_name, table, len(frame))
        for table in _partition_table_order(partition):
            for i, fk in enumerate(fk_map.get(table, [])):
                stmt = translate_alter_fk(engine_name, table, schema, fk, f"{table}_fk_{i}")
                if stmt:
                    conn.execute(text(stmt))
    _log_progress(f"[federation] {engine_name} {source_id}: finished (database={database})")


def _validate_federation_partition_metrics(source_id: str, engine_label: str, counts: dict[str, int]) -> None:
    from sandbox_corpus import federation_partition_tables

    partition = federation_partition_tables(source_id)
    missing = [table for table in sorted(partition) if counts.get(table, 0) <= 0]
    if missing:
        raise SystemExit(f"{engine_label} federation verify failed: missing or empty tables: {', '.join(missing)}")


def _count_federation_table(conn: Any, engine_name: str, schema: str, table: str) -> int:
    """Return row count for a federation partition table, isolating query failures."""
    fq = _qualified_table(engine_name, schema, table)
    try:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {fq}")).scalar_one())
    except Exception:
        conn.rollback()
        return 0


def _log_federation_verify_summary(source_id: str, counts: dict[str, int]) -> None:
    if _VERIFY_VERBOSE:
        parts = ", ".join(f"{table}={counts[table]}" for table in sorted(counts))
        _log_progress(f"[federation-verify] {source_id}: {parts}")
    else:
        _log_progress(f"[federation-verify] {source_id}: OK")


def _verify_federation_postgresql_partition(args: argparse.Namespace, *, source_id: str, schema: str) -> None:
    from sandbox_corpus import federation_partition_tables

    partition = federation_partition_tables(source_id)
    load_env_file(args.env_file)
    from aetherdialect._config import PostgresRuntimeConfig

    PostgresRuntimeConfig.apply_environment(os.environ)
    url = PostgresRuntimeConfig.db_url()
    connect_args = PostgresRuntimeConfig.connect_args() if hasattr(PostgresRuntimeConfig, "connect_args") else {}
    sa_engine = create_engine(url, connect_args=connect_args, future=True)
    counts: dict[str, int] = {}
    with sa_engine.connect() as conn:
        for table in sorted(partition):
            counts[table] = _count_federation_table(conn, "postgresql", schema, table)
    _validate_federation_partition_metrics(source_id, f"postgresql {source_id}", counts)
    _log_federation_verify_summary(source_id, counts)


def _verify_federation_mysql_partition(
    args: argparse.Namespace,
    *,
    source_id: str,
    database: str,
    runtime_cls: type,
) -> None:
    from sandbox_corpus import federation_partition_tables

    partition = federation_partition_tables(source_id)
    load_env_file(args.env_file)
    runtime_cls.apply_environment(os.environ)
    runtime_cls.DATABASE = database
    engine_name = str(getattr(runtime_cls, "ENGINE_NAME", "mysql"))
    schema = _default_schema(engine_name)
    url = runtime_cls.db_url()
    connect_args = runtime_cls.connect_args() if hasattr(runtime_cls, "connect_args") else {}
    sa_engine = create_engine(url, connect_args=connect_args, future=True)
    counts: dict[str, int] = {}
    with sa_engine.connect() as conn:
        for table in sorted(partition):
            counts[table] = _count_federation_table(conn, engine_name, schema, table)
    _validate_federation_partition_metrics(source_id, f"{engine_name} {source_id}", counts)
    _log_federation_verify_summary(source_id, counts)


def _federation_target_ids(args: argparse.Namespace) -> tuple[str, ...]:
    all_targets = ("storefront", "catalog", "logistics", "crm")
    flag = args.federation_verify or args.federation_load
    if flag == "all":
        return all_targets
    return (flag,)


def _cmd_federation_verify(args: argparse.Namespace) -> None:
    from sandbox_corpus import (
        FEDERATION_CATALOG_MYSQL_DATABASE,
        FEDERATION_CRM_MARIADB_DATABASE,
        FEDERATION_LOGISTICS_PG_SCHEMA,
        FEDERATION_STOREFRONT_PG_SCHEMA,
    )

    load_env_file(args.env_file, override=True)
    targets = _federation_target_ids(args)
    _validate_federation_env(targets)
    if "storefront" in targets:
        _verify_federation_postgresql_partition(
            args,
            source_id="storefront",
            schema=FEDERATION_STOREFRONT_PG_SCHEMA,
        )
    if "catalog" in targets:
        _verify_federation_mysql_partition(
            args,
            source_id="catalog",
            database=FEDERATION_CATALOG_MYSQL_DATABASE,
            runtime_cls=MySQLRuntimeConfig,
        )
    if "logistics" in targets:
        _verify_federation_postgresql_partition(
            args,
            source_id="logistics",
            schema=FEDERATION_LOGISTICS_PG_SCHEMA,
        )
    if "crm" in targets:
        _verify_federation_mysql_partition(
            args,
            source_id="crm",
            database=FEDERATION_CRM_MARIADB_DATABASE,
            runtime_cls=MariaDBRuntimeConfig,
        )


def _cmd_federation_load(args: argparse.Namespace) -> None:
    from sandbox_corpus import (
        FEDERATION_CATALOG_MYSQL_DATABASE,
        FEDERATION_CRM_MARIADB_DATABASE,
        FEDERATION_LOGISTICS_PG_SCHEMA,
        FEDERATION_STOREFRONT_PG_SCHEMA,
    )

    load_env_file(args.env_file, override=True)
    if args.drop_first:
        _log_progress(
            "[federation] --drop-first: recreating federation partition targets only "
            f"(postgres schemas {FEDERATION_STOREFRONT_PG_SCHEMA} and {FEDERATION_LOGISTICS_PG_SCHEMA}, "
            f"mysql database {FEDERATION_CATALOG_MYSQL_DATABASE}, "
            f"mariadb database {FEDERATION_CRM_MARIADB_DATABASE}); "
            "full rental_shop databases are not modified",
        )
    all_targets = ("storefront", "catalog", "logistics", "crm")
    targets = all_targets if args.federation_load == "all" else (args.federation_load,)
    _validate_federation_env(targets)
    if "storefront" in targets:
        _load_federation_postgresql_partition(
            args,
            source_id="storefront",
            schema=FEDERATION_STOREFRONT_PG_SCHEMA,
        )
    if "catalog" in targets:
        _load_federation_mysql_partition(
            args,
            source_id="catalog",
            database=FEDERATION_CATALOG_MYSQL_DATABASE,
            runtime_cls=MySQLRuntimeConfig,
        )
    if "logistics" in targets:
        _load_federation_postgresql_partition(
            args,
            source_id="logistics",
            schema=FEDERATION_LOGISTICS_PG_SCHEMA,
        )
    if "crm" in targets:
        _load_federation_mysql_partition(
            args,
            source_id="crm",
            database=FEDERATION_CRM_MARIADB_DATABASE,
            runtime_cls=MariaDBRuntimeConfig,
        )
    _cmd_federation_verify(args)


def _cli_engine_includes(args: argparse.Namespace) -> list[str]:
    return [e for e in _ALL_ENGINES if getattr(args, f"engine_flag_{e}", False)]


def _cli_engine_excludes(args: argparse.Namespace) -> list[str]:
    return [e for e in _ALL_ENGINES if getattr(args, f"engine_exclude_{e}", False)]


def _cli_exclusive_mode_flags(args: argparse.Namespace) -> list[str]:
    """Return one representative flag per active exclusive CLI mode group."""
    modes: list[str] = []
    if args.ping:
        modes.append("--ping")
    if args.extract_csv is not None:
        modes.append("--extract-csv")
    if args.federation_load is not None:
        modes.append("--federation-load")
    if args.federation_verify is not None:
        modes.append("--federation-verify")
    includes = _cli_engine_includes(args)
    excludes = _cli_engine_excludes(args)
    if args.all:
        modes.append("--all")
    elif includes:
        modes.append(f"--{includes[0].replace('_', '-')}")
    elif excludes:
        modes.append(f"--exclude-{excludes[0].replace('_', '-')}")
    return modes


def _validate_cli_modes(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    mode_flags = _cli_exclusive_mode_flags(args)
    if len(mode_flags) > 1:
        parser.error(f"Cannot combine {mode_flags[0]} with {mode_flags[1]}")

    if args.federation_load is None and args.federation_verify is None:
        return

    fed_flag = "--federation-load" if args.federation_load is not None else "--federation-verify"
    if args.schema is not None:
        parser.error(f"Cannot combine {fed_flag} with --schema")
    if args.recreate_schema:
        parser.error(f"Cannot combine {fed_flag} with --recreate-schema")
    if args.allow_public_schema_recreate:
        parser.error(f"Cannot combine {fed_flag} with --allow-public-schema-recreate")


def main() -> None:
    global _VERIFY_VERBOSE
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples: load all engines; load mysql and mariadb only; "
            "load all except snowflake and bigquery (--all --exclude-snowflake --exclude-bigquery); "
            "load all except mysql (--exclude-mysql)."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Start from all supported engines (excludes may subtract)")
    for engine in _ALL_ENGINES:
        flag = engine.replace("_", "-")
        parser.add_argument(
            f"--{flag}",
            action="store_true",
            dest=f"engine_flag_{engine}",
            help=f"Include {engine}",
        )
        parser.add_argument(
            f"--exclude-{flag}",
            action="store_true",
            dest=f"engine_exclude_{engine}",
            help=f"Exclude {engine}",
        )
    parser.add_argument("--csv-dir", type=Path, default=default_csv_dir())
    parser.add_argument("--ddl", type=Path, default=default_ddl_path())
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--drop-first", action="store_true")
    parser.add_argument(
        "--recreate-schema",
        action="store_true",
        help="postgresql only: DROP SCHEMA CASCADE then CREATE SCHEMA",
    )
    parser.add_argument(
        "--allow-public-schema-recreate",
        action="store_true",
        help="postgresql only: allow --recreate-schema on public",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="postgresql only: target database name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-table row counts during post-load verify",
    )
    parser.add_argument(
        "--federation-load",
        choices=["storefront", "catalog", "logistics", "crm", "all"],
        default=None,
        help=(
            "Load federation partition targets only "
            "(postgres schemas rental_shop_fed_storefront and rental_shop_fed_logistics, "
            "mysql db rental_shop_fed_catalog, mariadb db rental_shop_fed_crm); "
            "does not modify full rental_shop databases"
        ),
    )
    parser.add_argument(
        "--federation-verify",
        choices=["storefront", "catalog", "logistics", "crm", "all"],
        default=None,
        help=(
            "Row-count verify federation partition targets without reloading "
            "(postgres schemas rental_shop_fed_storefront and rental_shop_fed_logistics, "
            "mysql db rental_shop_fed_catalog, mariadb db rental_shop_fed_crm)"
        ),
    )
    parser.add_argument(
        "--ping",
        metavar="ENGINE",
        choices=_ALL_ENGINES,
        default=None,
        help="Connectivity check for one engine (SELECT 1)",
    )
    parser.add_argument(
        "--extract-csv",
        type=Path,
        metavar="OUT_DIR",
        default=None,
        help="Extract PostgreSQL rental_shop schema tables to OUT_DIR as CSV",
    )
    args = parser.parse_args()
    _validate_cli_modes(args, parser)

    if args.ping:
        _cmd_ping(argparse.Namespace(engine=args.ping, env_file=args.env_file))
        return

    if args.extract_csv:
        schema = args.schema or _verify_schema_name("postgresql", None)
        _cmd_extract_csv(
            argparse.Namespace(
                env_file=args.env_file,
                out=args.extract_csv,
                schema=schema,
            ),
        )
        return

    if args.federation_verify:
        _VERIFY_VERBOSE = bool(args.verbose)
        _cmd_federation_verify(args)
        return

    if args.federation_load:
        _cmd_federation_load(args)
        return

    _VERIFY_VERBOSE = bool(args.verbose)

    includes = _cli_engine_includes(args)
    excludes = _cli_engine_excludes(args)
    if includes and excludes and not args.all:
        parser.error(
            "Cannot combine engine include flags with exclude flags (use --all with excludes, or includes only)"
        )

    if args.all or includes:
        base = list(_ALL_ENGINES) if args.all else includes
    elif excludes:
        base = list(_ALL_ENGINES)
    else:
        base = list(_ALL_ENGINES)

    exclude_set = set(excludes)
    selected = [e for e in base if e not in exclude_set]
    if not selected:
        parser.error("No engines selected after applying include/exclude flags")

    load_env_file(args.env_file, override=True)
    for engine in selected:
        _log_progress(f"[load] starting {engine}")
        load_args = argparse.Namespace(
            engine=engine,
            csv_dir=args.csv_dir,
            ddl=args.ddl,
            env_file=args.env_file,
            schema=args.schema,
            drop_first=args.drop_first,
            recreate_schema=args.recreate_schema,
            allow_public_schema_recreate=args.allow_public_schema_recreate,
            database=args.database,
        )
        if engine == "postgresql":
            if load_args.database is None:
                load_args.database = os.environ.get("PGDATABASE") or os.environ.get("DB_NAME", "rental_shop")
            if not load_args.csv_dir.is_dir():
                raise SystemExit(f"Missing CSV directory: {load_args.csv_dir}")
            _load_postgresql(load_args)
        elif engine == "databricks":
            _load_databricks(load_args)
        elif engine == "duckdb":
            if not load_args.csv_dir.is_dir():
                raise SystemExit(f"Missing CSV directory: {load_args.csv_dir}")
            _load_duckdb(load_args)
        elif engine == "sqlite":
            if not load_args.csv_dir.is_dir():
                raise SystemExit(f"Missing CSV directory: {load_args.csv_dir}")
            _load_sqlite(load_args)
        elif engine in _SQLALCHEMY_ENGINES:
            _load_sqlalchemy_engine(engine, load_args)
        else:
            raise SystemExit(f"Unsupported engine: {engine}")

        verify_args = argparse.Namespace(
            engine=engine,
            env_file=args.env_file,
            schema=args.schema,
        )
        _cmd_verify(verify_args)


if __name__ == "__main__":
    main()
