"""Warmup template insert must not merge unrelated single-table intents via empty join fp."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import SeedWarmupIntent, SeedWarmupResult, SelectCol
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._seed_warmup import SeedWarmupCacheSession


def _warmup_intent(*, intent_id: str, table: str, column: str) -> SeedWarmupIntent:
    return SeedWarmupIntent(
        intent_id=intent_id,
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{column}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        param_values={},
        expansion_metadata=None,
        limit=None,
        natural_language=f"show {table}",
    )


@pytest.mark.fast
def test_two_single_table_intents_do_not_merge_on_empty_join_fp(schema_graph: SchemaGraph) -> None:
    """Distinct single-table warmup bodies must not merge because join fingerprints are both empty."""
    orders_intent = _warmup_intent(intent_id="wu_orders", table="orders", column="order_id")
    orders_result = SeedWarmupResult(
        intent=orders_intent,
        sql="SELECT order_id FROM orders",
        success=True,
        question="show orders",
    )
    store: dict = {"next_id": 1}
    templates: dict = {}
    first = SeedWarmupCacheSession._create_template_from_result(
        orders_result,
        schema_graph,
        next_id=1,
        store=store,
        templates=templates,
    )
    assert first is not None
    templates[first.id] = first

    customers_intent = _warmup_intent(intent_id="wu_customers", table="customers", column="customer_id")
    customers_result = SeedWarmupResult(
        intent=customers_intent,
        sql="SELECT customer_id FROM customers",
        success=True,
        question="show customers",
    )
    second = SeedWarmupCacheSession._create_template_from_result(
        customers_result,
        schema_graph,
        next_id=int(store["next_id"]),
        store=store,
        templates=templates,
    )
    assert second is not None
    assert second.id != first.id
    assert len(templates) == 2
