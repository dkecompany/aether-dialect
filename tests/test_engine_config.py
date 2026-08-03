"""Per-engine configuration: schema path and engine identity context."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._core_utils import active_engine_identity
from tests.test_aetherdialect import _make_aether_stub


@pytest.mark.fast
@pytest.mark.no_default_engine_identity
def test_active_engine_identity_raises_without_pushed_context() -> None:
    with pytest.raises(RuntimeError, match="no active engine identity"):
        active_engine_identity()


@pytest.mark.fast
def test_apply_schema_overrides_uses_engine_artifacts_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """apply_schema_overrides must persist to the owning engine's artifacts_dir."""
    engine_dir = tmp_path / "engine_a"
    engine_dir.mkdir()
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    (engine_dir / "schema_overrides.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
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
        )

    with patch("aetherdialect.aetherdialect.apply_overrides_and_persist", side_effect=_capture_apply):
        engine.apply_schema_overrides()

    expected = str(engine_dir / "schema_graph.json.gz")
    assert captured["schema_json_path"] == expected
    assert captured["schema_json_path"] != str(global_dir / "schema_graph.json.gz")
