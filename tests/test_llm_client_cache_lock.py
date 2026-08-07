"""LLM client cache get-or-create must be lock-guarded."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._llm_provider import LLMProvider, _clients


@pytest.mark.fast
def test_threaded_first_build() -> None:
    LLMProvider.clear_all_clients_after_fork()
    orig_provider = EngineConfig.LLM_PROVIDER
    orig_token = EngineConfig.API_TOKEN
    EngineConfig.LLM_PROVIDER = "openai"
    EngineConfig.API_TOKEN = "test-token-lock"
    created: list[object] = []
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []
    create_gate = threading.Event()

    def fake_openai(**kwargs: object) -> MagicMock:
        create_gate.wait(timeout=5)
        time.sleep(0.02)
        client = MagicMock(name=f"client-{len(created)}")
        created.append(client)
        return client

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            LLMProvider._build_client("openai")
        except BaseException as exc:
            errors.append(exc)

    try:
        with patch("aetherdialect._llm_provider.OpenAI", side_effect=fake_openai):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            time.sleep(0.05)
            create_gate.set()
            for thread in threads:
                thread.join(timeout=10)
        assert not errors
        assert len(created) == 1
        assert len([k for k in _clients if k[0] == "openai"]) == 1
    finally:
        EngineConfig.LLM_PROVIDER = orig_provider
        EngineConfig.API_TOKEN = orig_token
        LLMProvider.clear_all_clients_after_fork()
