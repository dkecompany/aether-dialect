"""Compose parse binds advertised distinct_on, preserve_tables, and CTE emission."""

from __future__ import annotations

import json

import pytest

from aetherdialect._contracts_base import CteEmissionKind
from aetherdialect._intent_expr import parse_intent_response


@pytest.mark.fast
def test_distinct_on_preserve_tables_emission_round_trip_from_llm_json() -> None:
    payload = {
        "tables": ["orders", "customers"],
        "select_cols": ["orders.id"],
        "group_by_cols": [],
        "order_by_cols": [],
        "where": None,
        "having": None,
        "limit": None,
        "natural_language": "show ids",
        "distinct_on": ["orders.id"],
        "preserve_tables": ["customers"],
        "cte_steps": [
            {
                "cte_name": "probe",
                "tables": ["orders"],
                "select_cols": ["orders.id"],
                "group_by_cols": [],
                "order_by_cols": [],
                "where": None,
                "having": None,
                "output_columns": ["id"],
                "emission": "semi_join",
                "preserve_tables": ["orders"],
                "distinct_on": ["orders.id"],
            }
        ],
    }
    intent = parse_intent_response(json.dumps(payload), question="show ids")
    assert intent is not None
    assert [e.primary_column for e in intent.distinct_on] == ["orders.id"]
    assert intent.preserve_tables == ["customers"]
    assert len(intent.cte_steps) == 1
    cte = intent.cte_steps[0]
    assert cte.emission == CteEmissionKind.SEMI_JOIN
    assert cte.preserve_tables == ["orders"]
    assert [e.primary_column for e in cte.distinct_on] == ["orders.id"]
