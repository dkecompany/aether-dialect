"""Aetherspace export/apply/delete round-trip and version-guard behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import AETHERSPACE_ARTIFACT_VERSION
from aetherdialect._contracts_base import ConfigError, EngineContext, SpaceContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import (
    MainExecutionOps,
)


def _column(name: str, *, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _table(name: str, *, columns: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    cols = columns or {"id": _column("id")}
    return TableMetadata(name=name, columns=cols, primary_key=["id"], foreign_keys=[])


def _sample_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "film": _table(
                "film", columns={"film_id": _column("film_id"), "title": _column("title", data_type="text")}
            ),
            "customer": _table("customer"),
        },
        join_paths_multi={},
        effective_structural_hash="eff_t32",
    )


@pytest.mark.fast
def test_engine_exposes_aetherspace_and_delete_aetherspace() -> None:
    assert hasattr(AetherEngine, "aetherspace")
    assert hasattr(AetherEngine, "delete_aetherspace")


@pytest.mark.fast
def test_export_apply_round_trip_persists_named_space(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    schema = _sample_schema()
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainExecutionOps.save_aetherspace_snapshot(engine_dir, "films_only", snap)
    export_path = MainExecutionOps.write_space_snapshot(engine_dir, "films_only", schema)
    MainExecutionOps.delete_aetherspace_snapshot(engine_dir, "films_only")
    assert MainExecutionOps.load_aetherspace_snapshot(engine_dir, "films_only") is None

    desc = MainExecutionOps.read_space_snapshot(engine_dir, "films_only", schema)
    assert desc.name == "films_only"
    assert export_path.is_file()
    loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, "films_only")
    assert loaded is not None
    assert "film" in loaded["tables"]


@pytest.mark.fast
def test_apply_from_explicit_source_path(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    schema = _sample_schema()
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainExecutionOps.save_aetherspace_snapshot(engine_dir, "films_only", snap)
    export_path = MainExecutionOps.write_space_snapshot(engine_dir, "films_only", schema)
    MainExecutionOps.delete_aetherspace_snapshot(engine_dir, "films_only")

    desc = MainExecutionOps.read_space_snapshot(engine_dir, "imported", schema, source=export_path)
    assert desc.name == "imported"
    loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, "imported")
    assert loaded is not None
    assert "film" in loaded["tables"]


@pytest.mark.fast
def test_apply_rejects_export_version_mismatch(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    schema = _sample_schema()
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    export_path = Path(MainExecutionOps._aetherspace_export_path(engine_dir, "stale"))
    export_path.parent.mkdir(parents=True, exist_ok=True)
    stale = dict(snap)
    stale["name"] = "stale"
    stale["version"] = "0.0.0"
    export_path.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"version .*" + str(AETHERSPACE_ARTIFACT_VERSION)) as exc_info:
        MainExecutionOps.read_space_snapshot(engine_dir, "stale", schema)
    msg = str(exc_info.value)
    assert "0.0.0" in msg
    assert "Delete" in msg


@pytest.mark.fast
def test_delete_unknown_aetherspace_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown aetherspace"):
        MainExecutionOps.delete_aetherspace_snapshot(str(tmp_path), "missing")


@pytest.mark.fast
def test_delete_master_aetherspace_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot be deleted"):
        MainExecutionOps.delete_aetherspace_snapshot(str(tmp_path), "master")


@pytest.mark.fast
def test_apply_master_aetherspace_raises(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    schema = _sample_schema()
    export_path = MainExecutionOps.write_space_snapshot(engine_dir, "master", schema)
    with pytest.raises(ConfigError, match="cannot be created or overwritten"):
        MainExecutionOps.read_space_snapshot(engine_dir, "master", schema, source=export_path)


@pytest.mark.fast
def test_apply_missing_export_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="export file not found"):
        MainExecutionOps.read_space_snapshot(str(tmp_path), "films_only", _sample_schema())


@pytest.mark.fast
def test_delete_removes_snapshot_file(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    schema = _sample_schema()
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainExecutionOps.save_aetherspace_snapshot(engine_dir, "films_only", snap)
    path = MainExecutionOps._aetherspace_path(engine_dir, "films_only")
    assert Path(path).is_file()
    assert MainExecutionOps.delete_aetherspace_snapshot(engine_dir, "films_only") is True
    assert not Path(path).is_file()


@pytest.mark.fast
def test_engine_delete_aetherspace_delegates_to_helper(tmp_path: Path) -> None:
    schema = _sample_schema()
    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "films_only", snap)
    with patch("aetherdialect.aetherdialect.delete_aetherspace", return_value=True) as delete_mock:
        engine = AetherEngine.__new__(AetherEngine)
        engine._artifacts_dir = tmp_path
        engine._schema_graph = schema
        engine._schema_role = "owner"
        engine._context_name = "master"
        engine._runtime_config = SimpleNamespace(execution_context=None, engine_context=EngineContext())
        engine._consumer_visible_objects = None
        engine._pipeline_writer_lock = __import__("threading").Lock()
        assert engine.delete_aetherspace("films_only") is True
        delete_mock.assert_called_once()
