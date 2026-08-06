"""Repair prompt dicts retain preserve_tables and CTE emission."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import CteEmissionKind
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol


@pytest.mark.fast
def test_repair_dict_keeps_preserve_tables_and_emission() -> None:
    select = SelectCol.from_dict({"expr": "customer.customer_id"})
    cte = RuntimeCteStep(
        cte_name="cte1",
        tables=["customer"],
        select_cols=[select],
        output_columns=["customer_id"],
        preserve_tables=["customer"],
        emission=CteEmissionKind.SEMI_JOIN,
    )
    intent = RuntimeIntent(
        tables=["customer"],
        select_cols=[select],
        preserve_tables=["customer"],
        cte_steps=[cte],
    )
    intent_dict = intent.to_prompt_dict()
    assert intent_dict["preserve_tables"] == ["customer"]
    cte_dict = intent_dict["cte_steps"][0]
    assert cte_dict["preserve_tables"] == ["customer"]
    assert cte_dict["emission"] == "semi_join"
    assert cte.to_prompt_dict()["emission"] == "semi_join"
    assert cte.to_prompt_dict()["preserve_tables"] == ["customer"]
