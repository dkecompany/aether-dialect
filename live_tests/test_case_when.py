"""Live tests for CASE expressions in SELECT."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import case_when_scenarios

_scenarios = case_when_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_case_when(runner, scenario):
    """CASE scenarios should emit SQL with CASE."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
