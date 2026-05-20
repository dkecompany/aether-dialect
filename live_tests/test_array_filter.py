"""Live pipeline tests for array ``contains`` on ``film.special_features`` (``dvdrental_new``)."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import array_filter_scenarios

_scenarios = array_filter_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_array_filter(runner, scenario):
    """Run array membership scenarios against PostgreSQL ``dvdrental_new``."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
