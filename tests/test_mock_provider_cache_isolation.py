"""Mock provider fixture caches must isolate by fixtures path and sandbox runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._llm_provider import MockProvider, SandboxRuntimeState


def _write_fixtures(path: Path, marker: str) -> Path:
    payload = {
        "fixtures": [
            {
                "task": "default",
                "system": "sys",
                "user": "hello",
                "output_text": json.dumps({"marker": marker}),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.fast
def test_parallel_fixture_files_isolated(tmp_path: Path) -> None:
    path_a = _write_fixtures(tmp_path / "a.json", "A")
    path_b = _write_fixtures(tmp_path / "b.json", "B")
    MockProvider.reset_mock_provider()
    orig_provider = EngineConfig.LLM_PROVIDER
    orig_mock = EngineConfig.MOCK_FIXTURES_FILE
    try:
        EngineConfig.LLM_PROVIDER = "sandbox"
        runtime_a = SandboxRuntimeState()
        runtime_b = SandboxRuntimeState()
        token_a = SandboxRuntimeState.bind_sandbox_runtime(runtime_a)
        try:
            EngineConfig.MOCK_FIXTURES_FILE = str(path_a)
            provider_a = MockProvider.get()
            out_a = provider_a.chat_text("sys", "hello", task="default", max_retries=0, timeout=1.0)
        finally:
            SandboxRuntimeState.reset_sandbox_runtime(token_a)

        token_b = SandboxRuntimeState.bind_sandbox_runtime(runtime_b)
        try:
            EngineConfig.MOCK_FIXTURES_FILE = str(path_b)
            provider_b = MockProvider.get()
            out_b = provider_b.chat_text("sys", "hello", task="default", max_retries=0, timeout=1.0)
        finally:
            SandboxRuntimeState.reset_sandbox_runtime(token_b)

        assert provider_a is not provider_b
        assert json.loads(out_a)["marker"] == "A"
        assert json.loads(out_b)["marker"] == "B"
        assert runtime_a.mock_fixtures_path == str(path_a)
        assert runtime_b.mock_fixtures_path == str(path_b)
    finally:
        EngineConfig.LLM_PROVIDER = orig_provider
        EngineConfig.MOCK_FIXTURES_FILE = orig_mock
        MockProvider.reset_mock_provider()
