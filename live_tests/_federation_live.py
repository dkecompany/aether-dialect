"""Helpers for live federation tests across four rental_shop members."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from aetherdialect import AetherEngine, AetherFederation
from aetherdialect._config import MariaDBRuntimeConfig, MySQLRuntimeConfig, PostgresRuntimeConfig
from aetherdialect._contracts_base import EngineContext
from aetherdialect._core_utils import llm_usage_build_scope
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation import FederationConfigError, parse_federation_declaration, parse_federation_manifest
from aetherdialect._schema_graph import SchemaGraph, recompute_join_paths_multi

_REPO = Path(__file__).resolve().parents[1]
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_ENV_FILE = _REPO / "env.env"
_DECLARATION_PATH = _FIXTURES / "federation_live_declaration.json"


def _scripts_on_path() -> None:
    scripts = str(_REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def missing_federation_partition_engines() -> list[str]:
    """Return engine families that do not respond for federation partition loading."""
    _scripts_on_path()
    from load_rental_shop_engines import load_env_file

    missing: list[str] = []
    if not _ENV_FILE.is_file():
        return ["postgresql", "mysql", "mariadb"]
    load_env_file(_ENV_FILE, override=True)
    try:
        pg_engine = create_engine(PostgresRuntimeConfig.db_url(), future=True)
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        pg_engine.dispose()
    except Exception:
        missing.append("postgresql")
    try:
        MySQLRuntimeConfig.apply_environment(os.environ)
        mysql_engine = create_engine(
            MySQLRuntimeConfig.db_url(),
            connect_args=MySQLRuntimeConfig.connect_args(),
            future=True,
        )
        with mysql_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        mysql_engine.dispose()
    except Exception:
        missing.append("mysql")
    try:
        MariaDBRuntimeConfig.apply_environment(os.environ)
        mariadb_engine = create_engine(
            MariaDBRuntimeConfig.db_url(),
            connect_args=MariaDBRuntimeConfig.connect_args(),
            future=True,
        )
        with mariadb_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        mariadb_engine.dispose()
    except Exception:
        missing.append("mariadb")
    return missing


def federation_partitions_available() -> bool:
    """Return True when postgres, mysql and mariadb federation partition targets respond."""
    return not missing_federation_partition_engines()


def ensure_federation_partitions_loaded() -> None:
    """Load federation partition databases without touching full rental_shop targets."""
    loader = _REPO / "scripts" / "load_rental_shop_engines.py"
    subprocess.run(
        [
            sys.executable,
            str(loader),
            "--federation-load",
            "all",
            "--drop-first",
            "--env-file",
            str(_ENV_FILE),
        ],
        check=True,
        cwd=str(_REPO),
    )


def _stamp_source_id(graph: SchemaGraph, source_id: str) -> SchemaGraph:
    tables = {name: replace(table, source_id=source_id) for name, table in graph.tables.items()}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_stats=graph.schema_stats,
    )


def _reflect_member_graph(engine_type: str, sa_engine: Any, source_id: str) -> SchemaGraph:
    _scripts_on_path()
    from sandbox_corpus import federation_partition_tables

    tables = federation_partition_tables(source_id)
    if engine_type == "postgresql":
        PostgresRuntimeConfig.apply_environment(os.environ)
        runtime_cls = PostgresRuntimeConfig
    elif engine_type == "mariadb":
        MariaDBRuntimeConfig.apply_environment(os.environ)
        runtime_cls = MariaDBRuntimeConfig
    else:
        MySQLRuntimeConfig.apply_environment(os.environ)
        runtime_cls = MySQLRuntimeConfig
    dialect = DialectRegistry.get(engine_type, runtime_cls, sqlalchemy_engine=sa_engine)
    ctx = EngineContext(include="tables", allow_objects=frozenset(tables))
    graph = dialect.reflect_schema_graph(ctx)
    return _stamp_source_id(graph, source_id)


def _postgres_engine_for_schema(schema: str):
    env = dict(os.environ)
    env["PGSCHEMA"] = schema
    env["POSTGRESQL_SCHEMA"] = schema
    PostgresRuntimeConfig.apply_environment(env)
    return create_engine(PostgresRuntimeConfig.db_url(), future=True)


def build_federation_live_engine() -> AetherFederation:
    """Construct a federated owner scope over four live partition members."""
    _scripts_on_path()
    from load_rental_shop_engines import load_env_file
    from sandbox_corpus import (
        FEDERATION_CATALOG_MYSQL_DATABASE,
        FEDERATION_CRM_MARIADB_DATABASE,
        FEDERATION_LOGISTICS_PG_SCHEMA,
        FEDERATION_STOREFRONT_PG_SCHEMA,
    )

    load_env_file(_ENV_FILE, override=True)
    os.environ["MYSQL_DATABASE"] = FEDERATION_CATALOG_MYSQL_DATABASE
    os.environ["MARIADB_DATABASE"] = FEDERATION_CRM_MARIADB_DATABASE
    pg_storefront = _postgres_engine_for_schema(FEDERATION_STOREFRONT_PG_SCHEMA)
    pg_logistics = _postgres_engine_for_schema(FEDERATION_LOGISTICS_PG_SCHEMA)
    MySQLRuntimeConfig.apply_environment(os.environ)
    mysql_engine = create_engine(
        MySQLRuntimeConfig.db_url(),
        connect_args=MySQLRuntimeConfig.connect_args(),
        future=True,
    )
    MariaDBRuntimeConfig.apply_environment(os.environ)
    mariadb_engine = create_engine(
        MariaDBRuntimeConfig.db_url(),
        connect_args=MariaDBRuntimeConfig.connect_args(),
        future=True,
    )
    manifest, _ = parse_federation_declaration(json.loads(_DECLARATION_PATH.read_text(encoding="utf-8")))
    notes = _REPO / "scripts" / "data" / "rental_shop_notes.txt"
    sql_file = _REPO / "scripts" / "data" / "rental_shop.sql"
    cfg_path = _write_federation_toml()
    artifacts_root = tempfile.mkdtemp(prefix="live_fed_artifacts_")
    master_ctx = EngineContext(
        notes_file=str(notes) if notes.is_file() else None,
        sql_file=str(sql_file) if sql_file.is_file() else None,
    )
    try:
        with llm_usage_build_scope():
            members = {
                "storefront": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=pg_storefront,
                ),
                "catalog": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=mysql_engine,
                ),
                "logistics": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=pg_logistics,
                ),
                "crm": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=mariadb_engine,
                ),
            }
            return AetherFederation(
                manifest.federation_id,
                members=members,
                declaration_file=str(_DECLARATION_PATH),
                artifacts_dir=artifacts_root,
            )
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def _write_federation_toml() -> str:
    from live_tests.conftest import write_live_env_file_to_temp_config_toml

    return write_live_env_file_to_temp_config_toml(
        _ENV_FILE,
        {
            "AETHERDIALECT_ENGINE": "postgresql",
            "PGSCHEMA": os.environ.get("PGSCHEMA", "rental_shop_fed_storefront"),
            "MYSQL_DATABASE": os.environ.get("MYSQL_DATABASE", "rental_shop_fed_catalog"),
            "MARIADB_DATABASE": os.environ.get("MARIADB_DATABASE", "rental_shop_fed_crm"),
        },
    )


def sensitive_manifest_rejects_ssn_key() -> bool:
    """Return True when a sensitive cross-source join is rejected at manifest parse."""
    manifest_payload = json.loads(_DECLARATION_PATH.read_text(encoding="utf-8"))
    bad_manifest = dict(manifest_payload)
    bad_manifest["cross_source_joins"] = [
        {
            "left": "staff.ssn",
            "right": "customer.email",
            "kind": "inner",
            "logical_key": "ssn",
        }
    ]
    try:
        parse_federation_manifest(bad_manifest)
    except FederationConfigError:
        return True
    return False
