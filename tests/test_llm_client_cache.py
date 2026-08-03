"""LLM client cache must be keyed by credential identity."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import LlmExecutionConfig
from aetherdialect._core_utils import llm_execution_scope
from aetherdialect._llm_provider import _build_client, _clients, clear_llm_clients


def _azure_llm_cfg(api_key: str) -> LlmExecutionConfig:
    return LlmExecutionConfig(
        azure_endpoint="https://tenant.openai.azure.com",
        azure_api_key=api_key,
        azure_api_version="2024-02-01",
        deployment_light="light",
        deployment_heavy="heavy",
        max_query_cost_rows=1,
        max_query_cost_bytes=1,
        statement_timeout_ms=1,
        llm_timeout_ms=30_000,
        profile_timeout_ms=1,
        explain_timeout_ms=None,
    )


@pytest.fixture
def _restore_llm_client_cache():
    prev = dict(_clients)
    _clients.clear()
    yield
    _clients.clear()
    _clients.update(prev)


@pytest.fixture
def _openai_provider():
    prev_provider = EngineConfig.LLM_PROVIDER
    prev_token = EngineConfig.API_TOKEN
    prev_base = EngineConfig.OPENAI_BASE_URL
    EngineConfig.LLM_PROVIDER = "openai"
    EngineConfig.OPENAI_BASE_URL = "https://api.openai.com/v1"
    yield
    EngineConfig.LLM_PROVIDER = prev_provider
    EngineConfig.API_TOKEN = prev_token
    EngineConfig.OPENAI_BASE_URL = prev_base


@pytest.mark.fast
@pytest.mark.usefixtures("_restore_llm_client_cache", "_openai_provider")
def test_distinct_api_tokens_get_distinct_cached_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    created_keys: list[str] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created_keys.append(str(kwargs.get("api_key")))

    monkeypatch.setattr("aetherdialect._llm_provider.OpenAI", FakeOpenAI)

    EngineConfig.API_TOKEN = "tenant-a-token"
    client_a = _build_client("openai")
    EngineConfig.API_TOKEN = "tenant-b-token"
    client_b = _build_client("openai")

    assert client_a is not client_b
    assert created_keys == ["tenant-a-token", "tenant-b-token"]
    assert len(_clients) == 2


@pytest.mark.fast
@pytest.mark.usefixtures("_restore_llm_client_cache", "_openai_provider")
def test_clear_llm_clients_preserves_other_credential_identities() -> None:
    EngineConfig.API_TOKEN = "tenant-a-token"
    client_a = _build_client("openai")
    EngineConfig.API_TOKEN = "tenant-b-token"
    client_b = _build_client("openai")
    assert len(_clients) == 2

    clear_llm_clients()

    assert client_a in _clients.values()
    assert client_b not in _clients.values()
    EngineConfig.API_TOKEN = "tenant-b-token"
    client_b_again = _build_client("openai")
    assert client_b_again is not client_b


@pytest.mark.fast
@pytest.mark.usefixtures("_restore_llm_client_cache")
def test_azure_clients_on_same_endpoint_differ_by_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created_keys: list[str] = []

    class FakeAzureOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created_keys.append(str(kwargs.get("api_key")))

    monkeypatch.setattr("aetherdialect._llm_provider.AzureOpenAI", FakeAzureOpenAI)
    EngineConfig.LLM_PROVIDER = "azure"

    with llm_execution_scope(_azure_llm_cfg("tenant-a-key")):
        client_a = _build_client("azure")
    with llm_execution_scope(_azure_llm_cfg("tenant-b-key")):
        client_b = _build_client("azure")

    assert client_a is not client_b
    assert created_keys == ["tenant-a-key", "tenant-b-key"]
    assert len(_clients) == 2
