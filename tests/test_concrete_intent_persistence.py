"""Concrete template IR persists distinct_on and preserve_tables."""

from __future__ import annotations

import pytest

from aetherdialect._constants import TEMPLATE_STORE_FORMAT_VERSION
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol


@pytest.mark.fast
def test_distinct_on_and_preserve_tables_survive_store_round_trip() -> None:
    assert TEMPLATE_STORE_FORMAT_VERSION == "0.2.1"
    intent = RuntimeIntent(
        tables=["orders", "customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        preserve_tables=["customers"],
        distinct_on=[NormalizedExpr.from_column("orders.id")],
        cte_steps=[
            RuntimeCteStep(
                cte_name="probe",
                tables=["orders"],
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
                output_columns=["id"],
                preserve_tables=["orders"],
                distinct_on=[NormalizedExpr.from_column("orders.id")],
            )
        ],
    )
    concrete = intent.to_concrete("sig1")
    assert concrete.preserve_tables == ["customers"]
    assert [e.primary_column for e in concrete.distinct_on] == ["orders.id"]
    assert concrete.cte_steps[0].preserve_tables == ["orders"]
    assert [e.primary_column for e in concrete.cte_steps[0].distinct_on] == ["orders.id"]
    restored = type(concrete).from_dict(concrete.to_dict())
    assert restored.preserve_tables == ["customers"]
    assert [e.primary_column for e in restored.distinct_on] == ["orders.id"]
    assert restored.cte_steps[0].preserve_tables == ["orders"]
    assert [e.primary_column for e in restored.cte_steps[0].distinct_on] == ["orders.id"]
