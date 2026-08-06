"""Programmatic connection mapping must not mutate process environment."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import DuckDBRuntimeConfig
from aetherdialect._main_execution import MainExecutionOps


def _engine_bundle() -> MagicMock:
    bundle = MagicMock()
    bundle.dialect = MagicMock()
    bundle.data_quality_report = None
    return bundle


@pytest.mark.fast
def test_two_engines_mapping_no_env_mutation() -> None:
    before = dict(os.environ)
    left = {
        "AETHERDIALECT_ENGINE": "duckdb",
        "DUCKDB_DATABASE_PATH": "/tmp/aether_left.duckdb",
    }
    right = {
        "AETHERDIALECT_ENGINE": "duckdb",
        "DUCKDB_DATABASE_PATH": "/tmp/aether_right.duckdb",
    }

    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()) as init,
        patch.object(AetherEngine, "_audit_emit"),
    ):
        engine_left = AetherEngine(MagicMock(), connection=left, artifacts_dir="artifacts-left")
        engine_right = AetherEngine(MagicMock(), connection=right, artifacts_dir="artifacts-right")

    assert init.call_args_list[0].kwargs["connection"] == left
    assert init.call_args_list[1].kwargs["connection"] == right
    assert dict(os.environ) == before

    merged_left: dict[str, str] = {}
    merged_right: dict[str, str] = {}
    with patch.object(os.environ, "__setitem__") as setitem:
        MainExecutionOps.overlay_programmatic_connection(merged_left, left)
        MainExecutionOps.overlay_programmatic_connection(merged_right, right)
        setitem.assert_not_called()

    assert merged_left["DUCKDB_DATABASE_PATH"] == left["DUCKDB_DATABASE_PATH"]
    assert merged_right["DUCKDB_DATABASE_PATH"] == right["DUCKDB_DATABASE_PATH"]
    assert DuckDBRuntimeConfig.from_environment(merged_left).DATABASE_PATH == left["DUCKDB_DATABASE_PATH"]
    assert DuckDBRuntimeConfig.from_environment(merged_right).DATABASE_PATH == right["DUCKDB_DATABASE_PATH"]
    assert dict(os.environ) == before
    assert engine_left is not engine_right
    _: Any = engine_left, engine_right
