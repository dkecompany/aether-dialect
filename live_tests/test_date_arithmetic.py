"""Date arithmetic live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import date_arithmetic_scenarios

_scenarios = date_arithmetic_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_date_arithmetic(runner, scenario):
    """Run a date arithmetic scenario and assert expectations."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
