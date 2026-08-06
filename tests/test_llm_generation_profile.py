"""Generation profile application for OpenAI and Azure providers."""

from __future__ import annotations

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import TASK_PROFILES
from aetherdialect._llm_provider import LLMProvider


@pytest.mark.fast
def test_azure_temperature_profile_uses_reasoning_not_temperature() -> None:
    """Azure always attaches reasoning and never sets temperature."""
    prev = EngineConfig.LLM_PROVIDER
    try:
        EngineConfig.LLM_PROVIDER = "azure"
        kwargs: dict = {}
        profile = TASK_PROFILES["ddl"]
        logical_model = LLMProvider._task_model_for_profile("ddl")
        assert not LLMProvider._logical_model_uses_reasoning(logical_model)

        LLMProvider._apply_generation_profile(kwargs, profile, logical_model)

        assert "temperature" not in kwargs
        assert kwargs["reasoning"]["effort"] == "low"
    finally:
        EngineConfig.LLM_PROVIDER = prev
