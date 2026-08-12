"""Operational and editor artifact paths resolve under artifacts_dir, not process cwd."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import MIGRATION_MAP_FILENAME
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h-cwd",
        schema_graph_id="sg-cwd",
    )


def _make_aether_stub(*, artifacts_dir: Path, **overrides: object) -> AetherEngine:
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir=str(artifacts_dir),
            engine_context=EngineContext(),
            llm_execution=load_runtime_config(merged_env=dict(os.environ)),
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=_schema(),
        _dialect=MagicMock(),
        _artifacts_dir=artifacts_dir,
        _store=TemplateOps.empty_template_store("unit_test_eff"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_stats={"table_count": 1, "total_filterable": 1},
        _schema_role="owner",
        _consumer_visible_objects=None,
        _context_name="master",
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, key, value)
    return obj


def test_export_structure_is_dict_not_cwd_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    engine = _make_aether_stub(artifacts_dir=artifacts_dir)
    out = engine.export_structure()
    assert isinstance(out, dict)
    assert out["table_count"] == 1
    assert not list(cwd_dir.iterdir())


def test_apply_migration_map_targets_artifacts_dir_not_cwd() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        artifacts_dir = root / "artifacts"
        artifacts_dir.mkdir()
        cwd_dir = root / "cwd"
        cwd_dir.mkdir()
        engine = _make_aether_stub(artifacts_dir=artifacts_dir)

        old = os.getcwd()
        os.chdir(cwd_dir)
        try:
            with (
                patch(
                    "aetherdialect.aetherdialect.TemplateOps.parse_schema_migration_map_payload",
                    return_value=MagicMock(),
                ),
                patch("aetherdialect.aetherdialect.TemplateOps.validate_schema_migration_map"),
                patch("aetherdialect.aetherdialect.load_schema_graph_snapshot", return_value=engine._schema_graph),
                patch("aetherdialect.aetherdialect.AetherEngine.refresh", return_value=MagicMock()),
            ):
                engine.apply_migration_map({"version": 1, "tables": []})
            dst = artifacts_dir / MIGRATION_MAP_FILENAME
            assert dst.is_file()
            assert not (cwd_dir / MIGRATION_MAP_FILENAME).exists()
        finally:
            os.chdir(old)
