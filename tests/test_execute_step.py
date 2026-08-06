"""execute_sql removed from public API; execute_template is the agent bridge."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine


@pytest.mark.fast
def test_execute_sql_not_on_public_engine() -> None:
    assert not hasattr(AetherEngine, "execute_sql")
