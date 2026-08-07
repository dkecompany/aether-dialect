"""Named schema-context sidecar version mismatch handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._constants import NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION
from aetherdialect._contracts_base import ConfigError, EngineContext
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_version_mismatch_raises(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    ctx = EngineContext(allow_objects=frozenset({"orders"}))
    path = MainExecutionOps.save_named_schema_context(engine_dir, "team_a", ctx)
    assert MainExecutionOps.load_named_schema_context(engine_dir, "team_a") == ctx

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["version"] = 999
    Path(path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"version .*999") as exc_info:
        MainExecutionOps.load_named_schema_context(engine_dir, "team_a")
    msg = str(exc_info.value)
    assert str(NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION) in msg
    assert "Delete" in msg
