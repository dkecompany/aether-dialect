"""
Databricks dialect-specific live tests.

Exercises translate_to_spark behavior: concatenation (|| -> concat), date functions (strftime -> date_format, CURRENT_DATE -> current_date), and table qualification (catalog.schema.table).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from aetherdialect._config import EngineConfig, QSimConfig
from aetherdialect._contracts_base import SchemaContext
from aetherdialect._core_utils import write_gzip_json_atomic
from aetherdialect._live_testing import (
    Expected,
    LiveTestRunner,
    Scenario,
    run_and_assert,
)
from aetherdialect._templates import (
    load_template_store,
    store_to_templates,
)
from aetherdialect.text2sql import Text2SQL

from .conftest import (
    _domain_notes_path,
    _env_file,
    _instrument_runner,
    _relax_dvdrental_selectability,
    write_live_env_file_to_temp_config_toml,
)
def _dbr_param(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _redirect_to_livetest_dir(t2s: Text2SQL) -> str:
    original = t2s._artifacts_dir
    parent = os.path.dirname(original)
    folder = os.path.basename(original)
    live_folder = folder.replace("artifacts_", "livetest_", 1)
    if live_folder == folder:
        live_folder = f"livetest_{folder}"
    live_dir = os.path.join(parent, live_folder)

    os.makedirs(live_dir, exist_ok=True)

    schema_dst = os.path.join(live_dir, "schema_graph.json.gz")
    if not os.path.exists(schema_dst):
        schema_src_gz = os.path.join(original, "schema_graph.json.gz")
        schema_src_json = os.path.join(original, "schema_graph.json")
        if os.path.exists(schema_src_gz):
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


def _dialect_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="DBR-DIALECT-002",
            question="show the year from the last rental date of each customer",
            expected=Expected(
                min_rows=1,
                contains_group_by=True,
            ),
            category="databricks_dialect",
        ),
        Scenario(
            id="DBR-DIALECT-003",
            question="list film titles and their length in hours",
            expected=Expected(
                tables=["film"],
                min_rows=1,
            ),
            category="databricks_dialect",
        ),
    ]


@pytest.fixture(scope="module")
def t2s():
    schema = _dbr_param("DATABRICKS_SCHEMA", "dvdrental_new")
    sql_file = _dbr_param("SQL_FILE", os.path.join("dev_workspace", "dvdrental.sql"))

    _notes = _domain_notes_path()

    cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": "databricks"})
    try:
        instance = Text2SQL(
            SchemaContext(
                notes_file=str(_notes) if _notes else None,
                sql_file=sql_file,
            ),
            artifacts_dir=tempfile.mkdtemp(prefix="live_dbx_dialect_"),
            config_file=cfg_path,
        )

        _redirect_to_livetest_dir(instance)

        _relax_dvdrental_selectability(instance._schema_graph, schema)

        fresh_store = load_template_store(instance._schema_graph.effective_structural_hash, instance._schema_graph)
        instance._store = fresh_store
        instance._templates = store_to_templates(fresh_store)
        instance._rejected = {}

        return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def schema(t2s):
    return t2s._schema_graph


@pytest.fixture(scope="module")
def store(t2s):
    return t2s._store


@pytest.fixture(scope="module")
def templates(t2s):
    return t2s._templates


@pytest.fixture(scope="module")
def rejected(t2s):
    return t2s._rejected


@pytest.fixture(scope="module")
def schema_terms(t2s):
    return t2s._schema_terms


@pytest.fixture(scope="module")
def runner(schema, store, templates, rejected, schema_terms, t2s):
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


_dialect_scenarios_list = _dialect_scenarios()


@pytest.mark.live
@pytest.mark.parametrize(
    "scenario",
    _dialect_scenarios_list,
    ids=[s.id for s in _dialect_scenarios_list],
)
def test_databricks_dialect(runner, scenario):
    """Run Databricks dialect translation scenarios."""
    run_and_assert(runner, scenario, header=f"[databricks:{scenario.id}] {scenario.question}")
