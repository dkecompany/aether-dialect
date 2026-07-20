"""Tests for mock LLM provider dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._config import EngineConfig, llm_credentials_configured
from aetherdialect._llm_provider import MockFixtureMissingError, MockProvider, llm_chat, llm_json, reset_mock_provider


@pytest.fixture(autouse=True)
def _reset_llm_env() -> None:
    orig_provider = EngineConfig.LLM_PROVIDER
    orig_mock = EngineConfig.MOCK_FIXTURES_FILE
    orig_token = EngineConfig.API_TOKEN
    try:
        reset_mock_provider()
        yield
    finally:
        EngineConfig.LLM_PROVIDER = orig_provider
        EngineConfig.MOCK_FIXTURES_FILE = orig_mock
        EngineConfig.API_TOKEN = orig_token
        reset_mock_provider()


def test_mock_provider_hit(tmp_path: Path) -> None:
    fixtures = {
        "version": 1,
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
    provider = MockProvider(str(path))
    out = provider.chat_text("sys", "hello", task="default", max_retries=1, timeout=1.0)
    assert out == '{"ok": true}'


def test_mock_provider_miss(tmp_path: Path) -> None:
    path = tmp_path / "mock.json"
    path.write_text(json.dumps({"fixtures": []}), encoding="utf-8")
    provider = MockProvider(str(path))
    with pytest.raises(MockFixtureMissingError):
        provider.chat_text("sys", "missing", task="default", max_retries=1, timeout=1.0)


def test_llm_chat_mock_dispatch(tmp_path: Path) -> None:
    fixtures = {
        "fixtures": [
            {
                "task": "intent",
                "system": "S",
                "user": "U",
                "output_text": '{"intent": "x"}',
            }
        ],
    }
    path = tmp_path / "mock.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    EngineConfig.LLM_PROVIDER = "mock"
    EngineConfig.MOCK_FIXTURES_FILE = str(path)
    reset_mock_provider()
    assert llm_credentials_configured()
    out = llm_chat("S", "U", task="intent", max_retries=1, timeout=1.0)
    assert "intent" in out


def test_llm_json_parse_through_mock(tmp_path: Path) -> None:
    fixtures = {
        "fixtures": [
            {
                "task": "default",
                "system": "S",
                "user": "U",
                "output_text": '{"value": 42}',
            }
        ],
    }
    path = tmp_path / "mock.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    EngineConfig.LLM_PROVIDER = "mock"
    EngineConfig.MOCK_FIXTURES_FILE = str(path)
    reset_mock_provider()
    parsed = llm_json("S", "U", task="default")
    assert parsed == {"value": 42}
