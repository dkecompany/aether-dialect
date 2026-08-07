"""High-traffic rental_shop scenarios with exact row oracles on isolated runners."""

from __future__ import annotations

import pytest

from ._seed_helpers import run_isolated_scenario
from .mydb_scenarios import core_isolated_live_scenarios

_CORE_ISOLATED = core_isolated_live_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _CORE_ISOLATED, ids=[scenario.id for scenario in _CORE_ISOLATED])
def test_core_isolated_oracle(scenario, schema, schema_terms, t2s) -> None:
    """Run one core scenario on a fresh template store with exact row oracles."""
    run_isolated_scenario(
        schema,
        schema_terms,
        t2s,
        scenario,
        label=f"core_{scenario.id.lower()}",
    )
