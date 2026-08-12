"""INTENT_SCHEMA root includes grain for compose."""

from __future__ import annotations

import jsonschema
import pytest

from aetherdialect._constants_runtime import INTENT_SCHEMA
from aetherdialect._contracts_core import RuntimeIntent


@pytest.mark.fast
def test_grain_property_validates() -> None:
    grain_schema = INTENT_SCHEMA["properties"]["grain"]
    assert grain_schema["type"] == "string"
    assert set(grain_schema["enum"]) == {"row_level", "grouped", "scalar"}
    assert "grain" in RuntimeIntent.PROMPT_FIELD_SPEC
    assert "grain" in RuntimeIntent.prompt_structural_index()["RuntimeIntent"]
    payload = {
        "tables": ["customer"],
        "select_cols": [{"expr": "customer.customer_id"}],
        "group_by_cols": [],
        "order_by_cols": [],
        "where": None,
        "having": None,
        "limit": None,
        "natural_language": "list",
        "cte_steps": [],
        "window_registry": [],
        "case_registry": [],
        "grain": "row_level",
    }
    jsonschema.validate(payload, INTENT_SCHEMA)
