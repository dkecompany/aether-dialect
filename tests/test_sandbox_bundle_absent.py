"""Sandbox entry points refuse cleanly when the bundled corpus is absent."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine, ConfigError, Sandbox

data_zip_path = Sandbox.data_zip_path


@pytest.mark.fast
def test_sandbox_constructor_raises_when_bundle_absent() -> None:
    assert not Sandbox.data_zip_path().exists()
    with pytest.raises(ConfigError, match="offline sandbox corpus is not bundled"):
        Sandbox()


@pytest.mark.fast
def test_offline_sandbox_raises_when_bundle_absent() -> None:
    assert not Sandbox.data_zip_path().exists()
    with pytest.raises(ConfigError, match="offline sandbox corpus is not bundled"):
        with AetherEngine.offline_sandbox():
            pass
