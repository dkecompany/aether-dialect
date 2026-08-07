"""Azure LIGHT/HEAVY deployment configuration is required."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import ConfigError
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_missing_light_or_heavy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "azure")
    llm_exec = SimpleNamespace(
        azure_endpoint="https://example.openai.azure.com/",
        azure_api_key="key",
        azure_api_version="2024-01-01",
        deployment_light="",
        deployment_heavy="heavy-dep",
    )
    with pytest.raises(ConfigError, match="deployment_light"):
        MainExecutionOps.validate_azure_llm_execution(llm_exec)
