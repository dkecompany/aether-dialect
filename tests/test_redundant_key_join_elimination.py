"""Redundant key-join elimination removes far tables referenced only by primary key."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import (
    InferenceTag,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._intent_repair import eliminate_redundant_key_joins, reconcile_tables


def _col(name: str, *, nullable: bool = False, is_pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        sensitivity="none",
        is_nullable=nullable,
        is_primary_key=is_pk,
    )


def _orders_customers_schema(
    *,
    fk_inference: InferenceTag | None = None,
    customer_pk: list[str] | None = None,
    fk_nullable: bool = False,
) -> SchemaGraph:
    pk = customer_pk or ["id"]
    fk = FKEdge(
        src_table="orders",
        src_cols=["customer_id"] if len(pk) == 1 else ["customer_id_a", "customer_id_b"],
        dst_table="customers",
        dst_cols=pk,
        inference_tag=fk_inference,
    )
    if len(pk) == 1:
        order_cols = {
            "order_id": _col("order_id", is_pk=True),
            "customer_id": _col("customer_id", nullable=fk_nullable),
        }
    else:
        order_cols = {
            "order_id": _col("order_id", is_pk=True),
            "customer_id_a": _col("customer_id_a", nullable=fk_nullable),
            "customer_id_b": _col("customer_id_b", nullable=fk_nullable),
        }
    customer_cols = {name: _col(name, is_pk=(name in pk)) for name in pk}
    edge = {
        "src_table": "orders",
        "src_cols": list(fk.src_cols),
        "dst_table": "customers",
        "dst_cols": list(fk.dst_cols),
    }
    path = [edge]
    return SchemaGraph(
        tables={
            "customers": TableMetadata(
                name="customers",
                columns=customer_cols,
                primary_key=pk,
                foreign_keys=[],
            ),
            "orders": TableMetadata(
                name="orders",
                columns=order_cols,
                primary_key=["order_id"],
                foreign_keys=[fk],
            ),
        },
        join_paths_multi={"orders": {"customers": [path]}},
        effective_structural_hash="h",
    )


def _orders_customers_intent(
    *,
    select_customer_name: bool = False,
    nullable_fk_null_predicate: bool = False,
    composite_partial_pk: bool = False,
    reference_customer_pk: bool = True,
) -> RuntimeIntent:
    select_cols = [SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))]
    if select_customer_name:
        select_cols.append(SelectCol(expr=NormalizedExpr.from_column("customers.name")))
    elif composite_partial_pk:
        select_cols.append(SelectCol(expr=NormalizedExpr.from_column("customers.id_a")))
    elif reference_customer_pk:
        select_cols.append(SelectCol(expr=NormalizedExpr.from_column("customers.id")))
    where = None
    if nullable_fk_null_predicate:
        where = PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.customer_id"),
                    op="is null",
                    value_type="integer",
                )
            ]
        )
    return RuntimeIntent(
        tables=["orders", "customers"],
        grain="row_level",
        select_cols=select_cols,
        group_by_cols=[],
        order_by_cols=[],
        where=where,
    )


@pytest.fixture
def enable_elimination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyConfig, "ELIMINATE_REDUNDANT_KEY_JOINS", True)


@pytest.mark.fast
def test_elimination_disabled_by_default() -> None:
    schema = _orders_customers_schema()
    intent = _orders_customers_intent()
    result = eliminate_redundant_key_joins(intent, schema)
    assert result.tables == ["orders", "customers"]


@pytest.mark.fast
def test_elimination_removes_far_table_referenced_only_by_primary_key(enable_elimination: None) -> None:
    schema = _orders_customers_schema()
    intent = reconcile_tables(_orders_customers_intent())
    result = eliminate_redundant_key_joins(intent, schema)
    assert result.tables == ["orders"]
    assert any(sc.expr.primary_term == "orders.customer_id" for sc in result.select_cols or [])


@pytest.mark.fast
def test_non_primary_key_select_blocks_elimination(enable_elimination: None) -> None:
    schema = _orders_customers_schema()
    customers = schema.tables["customers"]
    customers.columns["name"] = _col("name")
    intent = reconcile_tables(_orders_customers_intent(select_customer_name=True))
    result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in result.tables


@pytest.mark.fast
def test_inferred_foreign_key_blocks_elimination(enable_elimination: None) -> None:
    schema = _orders_customers_schema(fk_inference=InferenceTag.SUFFIX)
    intent = reconcile_tables(_orders_customers_intent())
    result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in result.tables


@pytest.mark.fast
def test_null_sensitive_predicate_on_nullable_foreign_key_blocks_elimination(enable_elimination: None) -> None:
    schema = _orders_customers_schema(fk_nullable=True)
    intent = reconcile_tables(_orders_customers_intent(nullable_fk_null_predicate=True))
    result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in result.tables


@pytest.mark.fast
def test_composite_primary_key_requires_all_components_referenced(enable_elimination: None) -> None:
    schema = _orders_customers_schema(customer_pk=["id_a", "id_b"])
    intent = reconcile_tables(_orders_customers_intent(composite_partial_pk=True))
    result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in result.tables


@pytest.mark.fast
def test_pinned_join_path_blocks_elimination(enable_elimination: None) -> None:
    schema = _orders_customers_schema()
    intent = reconcile_tables(_orders_customers_intent())
    intent = replace(
        intent,
        chosen_join_candidate_id="J01",
        chosen_join_path_signature=["orders.customer_id->customers.id"],
    )
    result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in result.tables


@pytest.mark.fast
def test_cte_key_passthrough_keeps_far_table_when_non_pk_column_is_retained(enable_elimination: None) -> None:
    schema = _orders_customers_schema()
    customers = schema.tables["customers"]
    customers.columns["segment"] = _col("segment")
    cte = RuntimeCteStep(
        cte_name="cust_keys",
        tables=["customers"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customers.id")),
            SelectCol(expr=NormalizedExpr.from_column("customers.segment")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent = RuntimeIntent(
        tables=["orders", "cust_keys"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    intent = reconcile_tables(intent)
    result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in (result.cte_steps or [])[0].tables


@pytest.mark.fast
def test_stored_template_shape_replays_unchanged_when_flag_is_off() -> None:
    schema = _orders_customers_schema()
    intent = reconcile_tables(_orders_customers_intent())
    before_tables = list(intent.tables)
    before_select = [sc.expr.primary_term for sc in intent.select_cols or []]
    result = eliminate_redundant_key_joins(intent, schema)
    assert result.tables == before_tables
    assert [sc.expr.primary_term for sc in result.select_cols or []] == before_select
