"""Join-choice LLM payloads use deterministic scope ordering."""

from __future__ import annotations

from aetherdialect._constants import JOIN_CHOICE_SCOPE_MAIN
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._sql_gen import join_scope_pass2_llm_scopes


def test_na_scope_keys_emitted_sorted(simple_schema) -> None:
    """Pass-2 NA scopes are emitted in sorted order for stable LLM payloads."""
    intent = RuntimeIntent(
        tables=["customers", "orders"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customers.id")),
            SelectCol(expr=NormalizedExpr.from_column("orders.id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[],
    )
    na_keys = frozenset({JOIN_CHOICE_SCOPE_MAIN, "cte:zebra", "cte:alpha"})
    scopes = join_scope_pass2_llm_scopes(
        na_keys,
        {"candidates": []},
        {
            "alpha": {"candidates": []},
            "zebra": {"candidates": []},
        },
        intent,
        simple_schema,
        {},
    )
    assert [block["scope"] for block in scopes] == sorted(na_keys)
