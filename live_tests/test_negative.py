"""Negative and forbidden-SQL live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import negative_scenarios

_scenarios = negative_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_negative(runner, scenario):
    """Run a negative scenario and assert the pipeline rejects it."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question!r}", retries=0)
