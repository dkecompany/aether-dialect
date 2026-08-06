"""Unit tests ensuring live session ClassVar mutations do not leak into fast tests."""

from __future__ import annotations

import os

import pytest

from aetherdialect._config import DuckDBRuntimeConfig, EngineConfig, PostgresRuntimeConfig, QSimConfig
from aetherdialect._constants import ENGINE_STORAGE_PLACEHOLDER_DIR, TEMPLATE_STORE_SEGMENT


@pytest.mark.fast
def test_unit_sees_default_paths_after_live_session() -> None:
    """Simulate live-session ClassVar mutations, then restore defaults without a real DB."""
    from live_tests.conftest import restore_default_engine_config_classvars

    expected_schema = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "schema_graph.json.gz")
    expected_templates = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, TEMPLATE_STORE_SEGMENT)
    expected_skeletons = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "qsim_skeletons.json.gz")

    EngineConfig.SCHEMA_JSON_PATH = "/tmp/livetest_artifacts/schema_graph.json.gz"
    EngineConfig.TEMPLATE_STORE_DIR = "/tmp/livetest_artifacts/intent_templates"
    QSimConfig.SKELETONS_JSON_PATH = "/tmp/livetest_artifacts/qsim_skeletons.json.gz"
    EngineConfig.TYPE = "duckdb"
    EngineConfig.RUNTIME = DuckDBRuntimeConfig
    PostgresRuntimeConfig.HOST = "live-db.example"
    PostgresRuntimeConfig.DATABASE = "rental_shop"

    restore_default_engine_config_classvars()

    assert EngineConfig.TYPE == "postgresql"
    assert EngineConfig.RUNTIME is PostgresRuntimeConfig
    assert EngineConfig.SCHEMA_JSON_PATH == expected_schema
    assert EngineConfig.TEMPLATE_STORE_DIR == expected_templates
    assert QSimConfig.SKELETONS_JSON_PATH == expected_skeletons
    assert PostgresRuntimeConfig.HOST == "localhost"
    assert PostgresRuntimeConfig.DATABASE is None
