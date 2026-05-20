"""Live tests for window function generation."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import window_function_scenarios

_scenarios = window_function_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_window_function(runner, scenario):
    """Window scenarios should emit SQL with an OVER clause."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
