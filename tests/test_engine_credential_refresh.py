"""In-place database credential refresh without rebuilding schema artifacts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import DatabricksRuntimeConfig, PostgresRuntimeConfig
from aetherdialect._contracts_base import ConfigError, EngineContext
from tests.test_aetherdialect import _make_aether_stub


@pytest.mark.fast
def test_apply_connection_credentials_string_sets_primary_field() -> None:
    PostgresRuntimeConfig.apply_connection_credentials("rotated-secret")
    assert PostgresRuntimeConfig.PASSWORD == "rotated-secret"


@pytest.mark.fast
def test_apply_connection_credentials_dict_accepts_redacted_names() -> None:
    DatabricksRuntimeConfig.apply_connection_credentials({"access_token": "broker-minted"})
    assert DatabricksRuntimeConfig.ACCESS_TOKEN == "broker-minted"


@pytest.mark.fast
def test_refresh_connection_preserves_schema_graph_and_store() -> None:
    engine = _make_aether_stub()
    graph_before = engine._schema_graph
    store_before = engine._store
    templates_before = engine._templates
    old_dialect = MagicMock()
    old_dialect.dispose_native_connection = MagicMock()
    engine._dialect = old_dialect
    new_dialect = MagicMock(name="new_dialect")

    with patch(
        "aetherdialect.aetherdialect.refresh_engine_connection",
        return_value=new_dialect,
    ) as refresh_mock:
        engine.refresh_connection(credentials={"PASSWORD": "rotated"})

    refresh_mock.assert_called_once()
    assert engine._schema_graph is graph_before
    assert engine._store is store_before
    assert engine._templates is templates_before
    assert engine._dialect is new_dialect


@pytest.mark.fast
def test_refresh_connection_passes_existing_dialect_for_disposal() -> None:
    engine = _make_aether_stub()
    old_dialect = MagicMock()
    engine._dialect = old_dialect

    with patch(
        "aetherdialect.aetherdialect.refresh_engine_connection",
        return_value=MagicMock(),
    ) as refresh_mock:
        engine.refresh_connection(credentials="next-token")

    assert refresh_mock.call_args.kwargs["dialect"] is old_dialect


@pytest.mark.fast
def test_refresh_connection_uses_token_provider_when_credentials_omitted() -> None:
    engine = _make_aether_stub()
    provider = MagicMock(return_value={"PASSWORD": "from-provider"})
    engine._token_provider = provider
    new_dialect = MagicMock()

    with patch(
        "aetherdialect.aetherdialect.refresh_engine_connection",
        return_value=new_dialect,
    ) as refresh_mock:
        engine.refresh_connection()

    refresh_mock.assert_called_once()
    assert refresh_mock.call_args.kwargs["credentials"] is None
    assert refresh_mock.call_args.kwargs["token_provider"] is provider
    assert engine._dialect is new_dialect


@pytest.mark.fast
def test_resolve_connection_credentials_consults_token_provider() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    provider = MagicMock(return_value="broker-token")
    assert MainExecutionOps.resolve_connection_credentials(None, provider) == "broker-token"
    provider.assert_called_once_with()


@pytest.mark.fast
def test_refresh_connection_requires_credentials_or_provider() -> None:
    engine = _make_aether_stub()
    engine._token_provider = None

    with pytest.raises(ConfigError, match="credentials or a token_provider"):
        engine.refresh_connection()


@pytest.mark.fast
def test_initialize_consults_token_provider_before_dialect() -> None:
    provider = MagicMock(return_value="init-token")
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init_mock:
        init_mock.return_value = MagicMock(
            runtime_config=MagicMock(engine="postgresql"),
            llm_config=MagicMock(provider="openai"),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/x",
            store=MagicMock(),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={},
            data_quality_report=None,
        )
        AetherEngine(EngineContext(), artifacts_dir="x", token_provider=provider, audit_sink=None)

    assert init_mock.call_args.kwargs.get("token_provider") is provider
