"""
Seeded-intent live test for the execute-phase SQL validation failure path.

Drives ``deterministic_generate_validate_execute`` with an intent whose aggregation/grain combination is rejected by SQL validation so ``gen_out.success`` is ``False`` and ``sql_validation_error`` is populated.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from aetherdialect._contracts_core import (
    NormalizedExpr,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._live_testing import deterministic_generate_validate_execute
from aetherdialect._templates import TemplateStoreView


def _llm_forbidden(*_args, **_kwargs) -> None:
    raise AssertionError("LLM must not run in this deterministic validation-failure test")


@pytest.mark.live
@pytest.mark.live_no_llm
def test_deterministic_validation_fails_on_row_level_with_aggregate(schema, t2s) -> None:
    """Row-level grain mixed with a bare aggregate select is rejected by SQL validation."""
    intent = RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "customer.customer_id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = TemplateStoreView.empty(tmp, schema.effective_structural_hash)
        with patch("aetherdialect._pipeline.get_join_choice_from_llm", side_effect=_llm_forbidden):
            gen_out, rows = deterministic_generate_validate_execute(
                q_norm="invalid mixed grain and aggregate",
                intent=intent,
                schema=schema,
                dialect=t2s.dialect,
                store=store,
            )
    assert gen_out.success is False
    assert rows is None
    assert gen_out.sql_validation_error
