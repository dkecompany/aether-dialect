"""Logical model ClassVars can be overridden from the environment."""

from __future__ import annotations

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_env_overrides_logical_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_INTENT", "gpt-test-intent")
    MainExecutionOps._configure_openai_from_environment(
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL_INTENT": "gpt-test-intent",
        }
    )
    assert EngineConfig.OPENAI_MODEL_INTENT == "gpt-test-intent"
