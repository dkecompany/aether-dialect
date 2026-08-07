"""Operational and editor artifact paths resolve under artifacts_dir, not process cwd."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import MIGRATION_MAP_FILENAME, SCHEMA_OVERRIDES_DEFAULT_FILENAME
from aetherdialect._contracts_base import EngineContext, LLMConfig, RuntimeConfig
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._templates import TemplateOps


def _make_aether_stub(*, artifacts_dir: Path, **overrides: object) -> AetherEngine:
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir=str(artifacts_dir),
            engine_context=EngineContext(),
            llm_execution=load_runtime_config(merged_env=dict(os.environ)),
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=MagicMock(),
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
        _schema_stats={"table_count": 10, "total_filterable": 40},
        _schema_role="owner",
        _consumer_visible_objects=None,
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, key, value)
    return obj


def test_export_overrides_under_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    engine = _make_aether_stub(artifacts_dir=artifacts_dir)

    def _write_stub(_schema_graph: object, target: Path, **_kwargs: object) -> Path:
        target.write_text("{}", encoding="utf-8")
        return target

    with patch("aetherdialect.aetherdialect.dump_schema_overrides_to_path", side_effect=_write_stub):
        out = engine.export_overrides()

    expected = artifacts_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME
    assert out == expected
    assert expected.is_file()
    assert not (cwd_dir / SCHEMA_OVERRIDES_DEFAULT_FILENAME).exists()


def test_apply_migration_map_targets_artifacts_dir_not_cwd() -> None:
    captured: dict[str, object] = {}

    def _capture_init(self: AetherEngine, *args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        artifacts_dir = root / "artifacts"
        artifacts_dir.mkdir()
        cwd_dir = root / "cwd"
        cwd_dir.mkdir()
        src = root / "incoming_map.json"
        src.write_text('{"tables": []}', encoding="utf-8")

        old = os.getcwd()
        os.chdir(cwd_dir)
        try:
            with patch.object(AetherEngine, "__init__", _capture_init):
                out = AetherEngine.apply_migration_map(
                    str(src),
                    engine_context=EngineContext(),
                    artifacts_dir=str(artifacts_dir),
                )
            assert isinstance(out, AetherEngine)
            dst = artifacts_dir / MIGRATION_MAP_FILENAME
            assert dst.is_file()
            assert dst.read_text(encoding="utf-8") == '{"tables": []}'
            assert not (cwd_dir / MIGRATION_MAP_FILENAME).exists()
            kwargs = captured.get("kwargs", {})
            assert kwargs.get("artifacts_dir") == str(artifacts_dir)
        finally:
            os.chdir(old)
