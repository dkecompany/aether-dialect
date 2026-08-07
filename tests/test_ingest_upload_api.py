"""Tests for validated upload ingestion into an existing embedded member."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import duckdb
from aetherdialect import AetherEngine, UploadIngestResult
from aetherdialect._config import DuckDBRuntimeConfig, EngineConfig
from aetherdialect._contracts_base import ConfigError, EngineContext, EngineIdentity, LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._core_utils import load_runtime_config, pop_engine_identity, push_engine_identity
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._templates import TemplateOps
from tests.test_aetherdialect import _make_aether_stub


def _mock_llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
    if task == "upload_summary":
        return {"summary": "Upload inspection completed."}
    if task == "upload_interpret":
        return {}
    raise AssertionError(f"unexpected llm_json task={task!r}")


@pytest.fixture(autouse=True)
def _duckdb_upload_engine_identity() -> None:
    runtime = DuckDBRuntimeConfig()
    orig_runtime = EngineConfig.RUNTIME
    token = push_engine_identity(EngineIdentity("duckdb", runtime))
    EngineConfig.TYPE = "duckdb"
    EngineConfig.RUNTIME = runtime
    yield
    pop_engine_identity(token)
    EngineConfig.RUNTIME = orig_runtime


@pytest.fixture(autouse=True)
def _patch_upload_llm() -> None:
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_mock_llm_json):
        yield


@pytest.fixture(autouse=True)
def _patch_ingest_schema_profiling() -> None:
    with patch("aetherdialect._schema_overrides._profile_subset"):
        yield


def _duckdb_engine(tmp_path: Path) -> AetherEngine:
    connection = duckdb.connect(":memory:")
    dialect = DuckDBDialect(DuckDBRuntimeConfig(), native_connection=connection)
    llm_exec = load_runtime_config(merged_env=dict(os.environ))
    return _make_aether_stub(
        _runtime_config=RuntimeConfig(
            engine="duckdb",
            artifacts_dir=str(tmp_path),
            engine_context=EngineContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=SchemaGraph(tables={}, join_paths_multi={}),
        _dialect=dialect,
        _artifacts_dir=tmp_path,
        _store=TemplateOps.empty_template_store("ingest-test"),
        _native_connection=connection,
    )


@pytest.mark.fast
def test_ingest_rejects_failing_validation_report(tmp_path: Path) -> None:
    engine = _duckdb_engine(tmp_path)
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="blocking data-quality issue") as exc_info:
        engine.ingest_upload_sources([path])
    report = getattr(exc_info.value, "data_quality_report", None)
    assert report is not None
    assert report.ok is False


@pytest.mark.fast
def test_ingest_materialises_into_existing_engine(tmp_path: Path) -> None:
    engine = _duckdb_engine(tmp_path)
    path = tmp_path / "customers.csv"
    path.write_text("id,name\n1,Alice\n", encoding="utf-8")
    result = engine.ingest_upload_sources([path])
    assert isinstance(result, UploadIngestResult)
    assert result.report.ok is True
    assert result.relation_names == ("customers",)
    assert "customers" in engine._schema_graph.tables
    rows = engine._native_connection.execute("SELECT id, name FROM customers").fetchall()
    assert rows == [(1, "Alice")]


@pytest.mark.fast
def test_ingest_returns_data_quality_report_shape(tmp_path: Path) -> None:
    engine = _duckdb_engine(tmp_path)
    path = tmp_path / "orders.csv"
    path.write_text("id,amount\n10,25.5\n", encoding="utf-8")
    result = engine.ingest_upload_sources([path])
    payload = result.report.to_json_dict()
    assert payload["ok"] is True
    assert isinstance(payload["issues"], list)
    assert isinstance(payload["narrative"], str)
    assert isinstance(payload["suggested_selections"], dict)
    assert isinstance(payload["confirmed_selections"], dict)
    assert "requires_review" in payload
    assert result.schema_diff is not None
    assert "orders" in result.schema_diff.added_tables
