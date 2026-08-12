"""LLM-facing INTENT_SCHEMA accepts all CTE emission kinds."""

from __future__ import annotations

import jsonschema
import pytest

from aetherdialect._constants_runtime import INTENT_SCHEMA
from aetherdialect._contracts_base import CteEmissionKind
from aetherdialect._contracts_core import RuntimeCteStep


def _cte_payload(emission: str) -> dict:
    return {
        "tables": ["customer"],
        "select_cols": [{"expr": "customer.customer_id"}],
        "group_by_cols": [],
        "order_by_cols": [],
        "where": None,
        "having": None,
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
@pytest.mark.parametrize("emission", ["semi_join", "anti_join", "join_table", "scalar_subquery"])
def test_llm_schema_accepts_all_emission_kinds(emission: str) -> None:
    jsonschema.validate(_cte_payload(emission), INTENT_SCHEMA)


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
    assert enum == ["semi_join", "anti_join", "join_table", "scalar_subquery"]
