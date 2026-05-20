"""
Databricks (Spark) live pipeline tests.

Overrides the session-scoped ``t2s`` and ``runner`` fixtures from conftest.py to use a Databricks engine against the ``dvdrental_new`` schema replicated on the Databricks dev catalog.
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
    LiveTestRunner,
    run_and_assert,
    run_sequence_and_assert,
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
    append_live_session_schema_artifact,
    write_live_env_file_to_temp_config_toml,
)
from .mydb_scenarios import (
    bundled_dvdrental_live_scenarios,
    stateful_scenarios,
    template_reuse_sequence_scenarios,
    trust_cycle_scenarios,
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


@pytest.fixture(scope="module")
def t2s():
    """Module-scoped ``Text2SQL`` instance configured for Databricks."""

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
            artifacts_dir=tempfile.mkdtemp(prefix="live_dbx_artifacts_"),
            config_file=cfg_path,
        )

        _redirect_to_livetest_dir(instance)

        fresh_store = load_template_store(instance._schema_graph.effective_structural_hash, instance._schema_graph)
        instance._store = fresh_store
        instance._templates = store_to_templates(fresh_store)
        instance._rejected = {}

        append_live_session_schema_artifact("databricks", instance._schema_graph)

        _relax_dvdrental_selectability(instance._schema_graph, schema)

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
    """Module-scoped ``LiveTestRunner`` wired to the Databricks test resources."""

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


def _all_scenarios():
    """Mirror all PostgreSQL ``Scenario``-based live tests (see ``bundled_dvdrental_live_scenarios``)."""

    return bundled_dvdrental_live_scenarios()


_scenarios = _all_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_databricks(runner, scenario):
    """Run a dvdrental scenario against the Databricks/Spark dialect."""

    run_and_assert(runner, scenario, header=f"[databricks:{scenario.id}] {scenario.question}")


_stateful_sequences = stateful_scenarios()
_template_reuse_sequences = template_reuse_sequence_scenarios()
_trust_cycle_sequences = trust_cycle_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("seq", _stateful_sequences, ids=[s.id for s in _stateful_sequences])
def test_databricks_stateful(runner, seq):
    """Run stateful sequence scenarios against Databricks."""

    run_sequence_and_assert(runner, seq)


@pytest.mark.live
@pytest.mark.parametrize("seq", _template_reuse_sequences, ids=[s.id for s in _template_reuse_sequences])
def test_databricks_template_reuse_sequences(runner, seq):
    """Run template reuse sequence scenarios against Databricks."""

    run_sequence_and_assert(runner, seq)


@pytest.mark.live
@pytest.mark.parametrize("seq", _trust_cycle_sequences, ids=[s.id for s in _trust_cycle_sequences])
def test_databricks_trust_cycle_sequences(runner, seq):
    """Run trust-cycle sequence scenarios against Databricks."""

    run_sequence_and_assert(runner, seq)
