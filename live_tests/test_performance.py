"""Performance and cost-awareness live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import _assert_scenario

from .mydb_scenarios import performance_scenarios

_scenarios = performance_scenarios()

MAX_DURATION_SECONDS = 120.0


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_performance(runner, scenario):
    """Run a performance scenario and assert row-count bounds and duration."""
    result = runner.run(scenario, retries=0)
    soft = _assert_scenario(result, scenario.expected)
    soft.check(
        result.duration_seconds <= MAX_DURATION_SECONDS,
        "duration",
        f"<= {MAX_DURATION_SECONDS}s",
        f"{result.duration_seconds:.1f}s",
        message=f"run took {result.duration_seconds:.1f}s, limit is {MAX_DURATION_SECONDS}s",
    )
    soft.report(header=f"[{scenario.id}] {scenario.question}")
