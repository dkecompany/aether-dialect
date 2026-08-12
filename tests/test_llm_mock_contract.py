"""Mock vs live LLM provider contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import LlmTransientFailure
from aetherdialect._llm_provider import LLMProvider, MockProvider


@pytest.fixture(autouse=True)
def _reset_llm_env() -> None:
    orig_provider = EngineConfig.LLM_PROVIDER
    orig_mock = EngineConfig.MOCK_FIXTURES_FILE
    orig_token = EngineConfig.API_TOKEN
    try:
        MockProvider.reset_mock_provider()
        yield
    finally:
        EngineConfig.LLM_PROVIDER = orig_provider
        EngineConfig.MOCK_FIXTURES_FILE = orig_mock
        EngineConfig.API_TOKEN = orig_token
        MockProvider.reset_mock_provider()


@pytest.fixture
def _non_mock_llm_provider() -> None:
    prev = EngineConfig.LLM_PROVIDER
    EngineConfig.LLM_PROVIDER = "openai"
    EngineConfig.API_TOKEN = "test-token"
    yield
    EngineConfig.LLM_PROVIDER = prev


def _write_mock_fixtures(tmp_path: Path) -> Path:
    fixtures = {
        "fixtures": [
            {
                "task": "default",
                "system": "sys",
                "user": "hello",
                "output_text": '{"ok": true}',
            }
        ],
    }
    path = tmp_path / "mock.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    return path


@pytest.mark.fast
def test_mock_ignores_retries_and_timeout(tmp_path: Path) -> None:
    """Mock replay ignores max_retries/timeout: max_retries=0 still succeeds; no sleep."""
    path = _write_mock_fixtures(tmp_path)
    provider = MockProvider(str(path))

    with patch("aetherdialect._llm_provider.time.sleep") as sleep_mock:
        out = provider.chat_text(
            "sys",
            "hello",
            task="default",
            max_retries=0,
            timeout=0.0,
        )

    assert out == '{"ok": true}'
    sleep_mock.assert_not_called()

    EngineConfig.LLM_PROVIDER = "sandbox"
    EngineConfig.MOCK_FIXTURES_FILE = str(path)
    MockProvider.reset_mock_provider()

    with patch("aetherdialect._llm_provider.time.sleep") as sleep_mock:
        dispatch_out = LLMProvider.chat(
            "sys",
            "hello",
            task="default",
            max_retries=0,
            timeout=0.0,
        )

    assert dispatch_out == '{"ok": true}'
    sleep_mock.assert_not_called()


@pytest.mark.fast
def test_mock_provider_docstring_states_fixture_replay_only() -> None:
    doc = MockProvider.__doc__ or ""
    lowered = doc.lower()
    assert "fixture" in lowered
    assert "retry" in lowered or "retries" in lowered
    assert "timeout" in lowered


@pytest.mark.fast
@pytest.mark.usefixtures("_non_mock_llm_provider")
def test_live_transient_transport_error_raises_llm_transient_failure() -> None:
    client = MagicMock()
    client.responses.create.side_effect = RuntimeError("429 rate limit exceeded")

    with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
        with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
            with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=client):
                with patch("aetherdialect._utils.debug"):
                    with patch("aetherdialect._utils.pipeline_trace"):
                        with patch("aetherdialect._llm_provider.time.sleep"):
                            with pytest.raises(LlmTransientFailure, match="LLM call failed"):
                                LLMProvider.chat("s", "u", max_retries=1, task="join")

    assert client.responses.create.call_count == 1


@pytest.mark.fast
@pytest.mark.usefixtures("_non_mock_llm_provider")
def test_live_non_transient_transport_error_raises_runtime_error() -> None:
    client = MagicMock()
    client.responses.create.side_effect = ValueError("invalid request payload")

    with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
        with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
            with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=client):
                with patch("aetherdialect._utils.debug"):
                    with patch("aetherdialect._utils.pipeline_trace"):
                        with patch("aetherdialect._llm_provider.time.sleep"):
                            with pytest.raises(RuntimeError, match="LLM call failed"):
                                LLMProvider.chat("s", "u", max_retries=1, task="join")

    assert client.responses.create.call_count == 1


@pytest.mark.fast
def test_llm_error_likely_transient_classification() -> None:
    assert LLMProvider._llm_error_likely_transient(RuntimeError("429 Too Many Requests"))
    assert LLMProvider._llm_error_likely_transient(ConnectionError("connection reset by peer"))
    assert LLMProvider._llm_error_likely_transient(TimeoutError("request timed out"))
    assert not LLMProvider._llm_error_likely_transient(ValueError("invalid json in response"))
