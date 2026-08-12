"""Per-engine configuration: schema path and engine identity context."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._utils import active_engine_identity, bound_engine_runtime_config
from tests.test_aetherdialect import _make_aether_stub


@pytest.mark.fast
def test_active_engine_identity_raises_without_pushed_context(unbound_engine_identity: None) -> None:
    with pytest.raises(RuntimeError, match="no active engine identity"):
        active_engine_identity()


@pytest.mark.fast
def test_bound_engine_runtime_config_raises_without_active_identity(unbound_engine_identity: None) -> None:
    with pytest.raises(RuntimeError, match="no active engine identity"):
        bound_engine_runtime_config()


@pytest.mark.fast
def test_apply_structure_uses_engine_artifacts_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """apply_structure must persist to the owning engine's artifacts_dir."""
    engine_dir = tmp_path / "engine_a"
    engine_dir.mkdir()
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(global_dir / "schema_graph.json.gz"))

    engine = _make_aether_stub(_artifacts_dir=engine_dir)
    engine._schema_graph = MagicMock()
    engine._schema_graph.schema_graph_id = "sg_test000000000001__abcd1234"
    engine._schema_graph.effective_structural_hash = "eff"

    captured: dict[str, str] = {}

    def _capture_apply(*args: object, **kwargs: object) -> MagicMock:
        captured["schema_json_path"] = str(kwargs.get("schema_json_path", ""))
        return MagicMock(
            table_edits=0,
            column_edits=0,
            fks_added=0,
            fks_removed=0,
            pks_added=0,
            pks_endorsed=0,
            pks_blocked=0,
            coerced_columns=0,
            collapsed_inferences=0,
            domain_knowledge_entries=None,
        )

    with patch("aetherdialect.aetherdialect.apply_structure_document", side_effect=_capture_apply):
        engine.apply_structure(
            {
                "tables": {},
                "foreign_keys_add": [],
                "foreign_keys_remove": [],
                "primary_keys_add": [],
                "primary_keys_remove": [],
                "relationships": [],
                "table_count": 0,
            }
        )

    expected = str(engine_dir / "schema_graph.json.gz")
    assert captured["schema_json_path"] == expected
    assert captured["schema_json_path"] != str(global_dir / "schema_graph.json.gz")


@pytest.mark.fast
def test_two_constructions_keep_distinct_per_engine_schema_paths(tmp_path: Path) -> None:
    """Sequential construction must register distinct schema paths without last-writer global mutation."""
    from aetherdialect._config import DuckDBRuntimeConfig, EngineConfig, PostgresRuntimeConfig
    from aetherdialect._main_execution import MainExecutionOps

    original_global = EngineConfig.SCHEMA_JSON_PATH
    adir_a = str(tmp_path / "tenant_a" / "conn_duckdb_x")
    adir_b = str(tmp_path / "tenant_b" / "conn_postgresql_y")
    path_a = os.path.join(adir_a, "schema_graph.json.gz")
    path_b = os.path.join(adir_b, "schema_graph.json.gz")
    store_a = os.path.join(adir_a, "intent_templates", "spaces", "master")
    store_b = os.path.join(adir_b, "intent_templates", "spaces", "master")

    MainExecutionOps.register_engine_artifact_state(adir_a, schema_json_path=path_a, template_store_dir=store_a)
    MainExecutionOps.register_engine_artifact_state(adir_b, schema_json_path=path_b, template_store_dir=store_b)

    assert MainExecutionOps.engine_schema_json_path(adir_a) == path_a
    assert MainExecutionOps.engine_schema_json_path(adir_b) == path_b
    assert path_a != path_b
    assert EngineConfig.SCHEMA_JSON_PATH == original_global
    assert DuckDBRuntimeConfig is not PostgresRuntimeConfig
