"""LLM-facing INTENT_SCHEMA emission enum is probe-only."""

from __future__ import annotations

import jsonschema
import pytest
from jsonschema.exceptions import ValidationError

from aetherdialect._constants import INTENT_SCHEMA
from aetherdialect._contracts_base import CteEmissionKind
from aetherdialect._contracts_core import RuntimeCteStep


def _cte_payload(emission: str) -> dict:
    return {
        "tables": ["customer"],
        "select_cols": [{"expr": "customer.customer_id"}],
        "group_by_cols": [],
        "order_by_cols": [],
        "where": [],
        "having": [],
        "limit": None,
        "natural_language": "probe",
        "cte_steps": [
            {
                "cte_name": "cte1",
                "select_cols": [{"expr": "customer.customer_id"}],
                "output_columns": ["customer_id"],
                "emission": emission,
            }
        ],
        "window_registry": [],
        "case_registry": [],
    }


@pytest.mark.fast
def test_llm_schema_rejects_join_table() -> None:
    with pytest.raises(ValidationError):
        jsonschema.validate(_cte_payload("join_table"), INTENT_SCHEMA)
    with pytest.raises(ValidationError):
        jsonschema.validate(_cte_payload("scalar_subquery"), INTENT_SCHEMA)


@pytest.mark.fast
def test_internal_cte_accepts_scalar_subquery() -> None:
    step = RuntimeCteStep(
        cte_name="cte1",
        tables=["customer"],
        select_cols=[],
        output_columns=["customer_id"],
        emission=CteEmissionKind.SCALAR_SUBQUERY,
    )
    assert step.emission == CteEmissionKind.SCALAR_SUBQUERY
    enum = INTENT_SCHEMA["properties"]["cte_steps"]["items"]["properties"]["emission"]["enum"]
    assert enum == ["semi_join", "anti_join"]
