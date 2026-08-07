"""Engine and federation lifecycle: close, context managers, resource release."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, AetherFederation
from aetherdialect._contracts_base import AetherFederationInitResult, LLMConfig, RuntimeConfig
from aetherdialect._core_utils import (
    open_resource_inventory,
    register_live_connection,
    track_close_temp_directory,
)


def _engine_bundle() -> MagicMock:
    bundle = MagicMock()
    bundle.dialect = MagicMock()
    bundle.data_quality_report = None
    return bundle


@pytest.mark.fast
def test_close_releases_pool() -> None:
    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()),
        patch("aetherdialect.aetherdialect.drain_write_queue"),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect") as dispose_mock,
        patch("aetherdialect.aetherdialect.LLMProvider.clear_llm_clients"),
        patch.object(AetherEngine, "_audit_emit"),
    ):
        engine = AetherEngine(MagicMock(), artifacts_dir="x")
        engine.close()
    dispose_mock.assert_called_once()
    assert engine._closed is True


@pytest.mark.fast
def test_close_is_idempotent() -> None:
    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()),
        patch("aetherdialect.aetherdialect.drain_write_queue"),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect"),
        patch("aetherdialect.aetherdialect.LLMProvider.clear_llm_clients"),
        patch.object(AetherEngine, "_audit_emit"),
    ):
        engine = AetherEngine(MagicMock(), artifacts_dir="x")
        engine.close()


@pytest.mark.fast
def test_method_after_close_raises() -> None:
    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()),
        patch("aetherdialect.aetherdialect.drain_write_queue"),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect"),
        patch("aetherdialect.aetherdialect.LLMProvider.clear_llm_clients"),
        patch.object(AetherEngine, "_audit_emit"),
    ):
        engine = AetherEngine(MagicMock(), artifacts_dir="x")
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.session()


@pytest.mark.fast
def test_context_manager_closes_on_exception() -> None:
    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()),
        patch("aetherdialect.aetherdialect.drain_write_queue"),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect"),
        patch("aetherdialect.aetherdialect.LLMProvider.clear_llm_clients"),
        patch.object(AetherEngine, "_audit_emit"),
    ):
        with pytest.raises(ValueError):
            with AetherEngine(MagicMock(), artifacts_dir="x") as engine:
                raise ValueError("boom")
        assert engine._closed is True


@pytest.mark.fast
def test_federation_close_disposes_coordinator() -> None:
    coordinator = MagicMock()
    bundle = AetherFederationInitResult(
        runtime_config=RuntimeConfig(
            engine="federation",
            artifacts_dir="/tmp/fed",
            engine_context=MagicMock(),
            llm_execution=MagicMock(),
        ),
        llm_config=LLMConfig(provider="openai"),
        schema_graph=MagicMock(),
        dialect=coordinator,
        artifacts_dir="/tmp/fed",
        store={},
        templates={},
        rejected={},
        schema_terms=set(),
        schema_stats={},
        schema_role="owner",
        consumer_visible_objects=None,
    )
    member = MagicMock()
    member._execution_engine = None
    member._native_connection = None
    with (
        patch(
            "aetherdialect.aetherdialect.initialize_aether_federation",
            return_value=bundle,
        ),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect") as dispose_mock,
        patch("aetherdialect.aetherdialect.dispose_federation_source_runtimes"),
        patch.object(AetherFederation, "_audit_emit"),
    ):
        fed = AetherFederation("fed", members={"a": member}, declaration_file="/tmp/decl.json")
        fed.close()
    assert any(call.args[0] is coordinator for call in dispose_mock.call_args_list)


@pytest.mark.fast
def test_resource_inventory_returns_to_baseline(tmp_path: Path) -> None:
    baseline = open_resource_inventory()
    temp_dir = tempfile.mkdtemp(prefix="aetherdialect_test_owned_")
    connection = MagicMock(name="live_connection")
    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle()),
        patch("aetherdialect.aetherdialect.drain_write_queue"),
        patch("aetherdialect.aetherdialect.dispose_engine_dialect"),
        patch("aetherdialect.aetherdialect.LLMProvider.clear_llm_clients"),
        patch.object(AetherEngine, "_audit_emit"),
    ):
        engine = AetherEngine(MagicMock(), artifacts_dir=str(tmp_path / "artifacts"))
        track_close_temp_directory(engine, temp_dir)
        register_live_connection(connection, owner=engine)
        assert open_resource_inventory() != baseline
        engine.close()
    assert open_resource_inventory() == baseline
