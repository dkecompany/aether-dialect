"""INTENT_SCHEMA accepts null for optional where/having predicates."""

from __future__ import annotations

import json

import jsonschema
import pytest

from aetherdialect._constants_runtime import INTENT_SCHEMA
from aetherdialect._intent_expr import parse_intent_response


def _minimal_intent(**overrides: object) -> dict:
    base: dict = {
        "tables": ["customer"],
        "select_cols": [{"expr": "customer.customer_id"}],
        "group_by_cols": [],
        "order_by_cols": [],
        "where": None,
        "having": None,
        "limit": None,
        "natural_language": "list customers",
        "cte_steps": [],
        "window_registry": [],
        "case_registry": [],
    }
    base.update(overrides)
    return base


@pytest.mark.fast
def test_where_and_having_null_validate() -> None:
    jsonschema.validate(_minimal_intent(), INTENT_SCHEMA)
    jsonschema.validate(
        _minimal_intent(
            cte_steps=[
                {
                    "cte_name": "cte1",
                    "select_cols": [{"expr": "customer.customer_id"}],
                    "output_columns": ["customer_id"],
                    "where": None,
                    "having": None,
                }
            ]
        ),
        INTENT_SCHEMA,
    )


@pytest.mark.fast
def test_intent_parse_accepts_null_where_having() -> None:
    payload = json.dumps(_minimal_intent())
    parsed = parse_intent_response(payload, "list customers")
    assert parsed is not None
    assert parsed.where is None
    assert parsed.having is None
