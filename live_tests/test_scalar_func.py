"""Scalar function live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import scalar_func_scenarios

_scenarios = scalar_func_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_scalar_func(runner, scenario):
    """Run a scalar function scenario and assert expectations."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
