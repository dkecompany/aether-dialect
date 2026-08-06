"""Azure generation profile: effort remap and no temperature."""

from __future__ import annotations

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._llm_provider import LLMProvider


@pytest.mark.fast
def test_minimal_remaps_on_gpt_54(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "azure")
    kwargs: dict = {}
    profile = {"reasoning": {"effort": "minimal", "summary": "concise"}}
    LLMProvider._apply_generation_profile(kwargs, profile, "gpt-5.4-mini")
    assert kwargs["reasoning"]["effort"] == "none"
    assert "temperature" not in kwargs


@pytest.mark.fast
def test_azure_never_sets_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "azure")
    kwargs: dict = {}
    profile = {"temperature": 0.2}
    LLMProvider._apply_generation_profile(kwargs, profile, "gpt-4.1-nano")
    assert "temperature" not in kwargs
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "low"
