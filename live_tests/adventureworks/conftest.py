"""
Pytest fixtures for AdventureWorks live tests.

Overrides the session-level ``t2s`` fixture from the parent conftest to
connect to the local ``adventureworks`` PostgreSQL database instead of dvdrental.
Reads connection config from ``dev_workspace/env.env`` (same file as the
parent conftest) but forces ``PGDATABASE=adventureworks``.

Set OPENAI_API_KEY in dev_workspace/env.env before running.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from aetherdialect._config import EngineConfig, QSimConfig
from aetherdialect._contracts_base import SchemaContext
from aetherdialect._live_testing import LiveTestRunner
from aetherdialect._templates import load_template_store, store_to_templates
from aetherdialect.text2sql import Text2SQL

from live_tests.conftest import (
    _parse_live_env_file,
    _flat_live_env_to_nested_document,
    _nested_document_to_toml_str,
    _redirect_to_livetest_dir,
    append_live_session_schema_artifact,
    _instrument_runner,
)


def _env_file() -> str:
    raw = os.environ.get("LIVE_ENV_FILE", os.path.join("dev_workspace", "env.env"))
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent.parent / raw
    return str(p)


def _write_aw_config() -> str:
    flat = _parse_live_env_file(_env_file())
    flat["PGDATABASE"] = "adventureworks"
    flat["AETHERDIALECT_ENGINE"] = "postgresql"
    flat.pop("POSTGRES_DB", None)
    doc = _flat_live_env_to_nested_document(flat)
    fd, path = tempfile.mkstemp(prefix="live_aw_", suffix=".toml")
    os.close(fd)
    Path(path).write_text(_nested_document_to_toml_str(doc), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def t2s() -> Text2SQL:
    """Session-scoped Text2SQL instance pointed at the adventureworks database."""
    cfg_path = _write_aw_config()
    try:
        instance = Text2SQL(
            SchemaContext(),
            artifacts_dir=tempfile.mkdtemp(prefix="live_aw_artifacts_"),
            config_file=cfg_path,
        )
        _redirect_to_livetest_dir(instance)

        fresh_store = load_template_store(
            instance._schema_graph.effective_structural_hash,
            instance._schema_graph,
        )
        instance._store = fresh_store
        instance._templates = store_to_templates(fresh_store)
        instance._rejected = {}

        append_live_session_schema_artifact("adventureworks", instance._schema_graph)
        return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def schema(t2s: Text2SQL) -> Any:
    return t2s._schema_graph


@pytest.fixture(scope="session")
def store(t2s: Text2SQL) -> dict[str, Any]:
    return t2s._store


@pytest.fixture(scope="session")
def templates(t2s: Text2SQL) -> dict:
    return t2s._templates


@pytest.fixture(scope="session")
def rejected(t2s: Text2SQL) -> dict:
    return t2s._rejected


@pytest.fixture(scope="session")
def schema_terms(t2s: Text2SQL) -> set[str]:
    return t2s._schema_terms


@pytest.fixture(scope="session")
def runner(schema, store, templates, rejected, schema_terms, t2s) -> LiveTestRunner:
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
