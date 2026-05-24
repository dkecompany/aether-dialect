"""IN / NOT IN list filter live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import in_list_scenarios

_scenarios = in_list_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_in_list(runner, scenario):
    """Run an IN list filter scenario and assert expectations."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
