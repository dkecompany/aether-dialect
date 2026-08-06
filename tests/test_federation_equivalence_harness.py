"""Fast checks for federation equivalence question generation."""

from __future__ import annotations

import pytest
from live_tests._federation_equivalence_questions import generate_federation_equivalence_questions

from aetherdialect._contracts_base import ColumnRole
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi


def _synthetic_equivalence_schema() -> SchemaGraph:
    orders = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "ordered_at": ColumnMetadata(
                name="ordered_at",
                data_type="timestamp",
                value_type="temporal",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(src_table="orders", src_cols=["customer_id"], dst_table="customers", dst_cols=["customer_id"]),
        ],
    )
    customers = TableMetadata(
        name="customers",
        columns={
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "region": ColumnMetadata(
                name="region",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "lifetime_value": ColumnMetadata(
                name="lifetime_value",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "joined_at": ColumnMetadata(
                name="joined_at",
                data_type="date",
                value_type="temporal",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
    )
    products = TableMetadata(
        name="products",
        columns={
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "category": ColumnMetadata(
                name="category",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "unit_price": ColumnMetadata(
                name="unit_price",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "listed_on": ColumnMetadata(
                name="listed_on",
                data_type="date",
                value_type="temporal",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        primary_key=["product_id"],
        foreign_keys=[],
    )
    order_items = TableMetadata(
        name="order_items",
        columns={
            "order_item_id": ColumnMetadata(
                name="order_item_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "quantity": ColumnMetadata(
                name="quantity",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "shipped_at": ColumnMetadata(
                name="shipped_at",
                data_type="timestamp",
                value_type="temporal",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        primary_key=["order_item_id"],
        foreign_keys=[
            FKEdge(src_table="order_items", src_cols=["order_id"], dst_table="orders", dst_cols=["order_id"]),
            FKEdge(src_table="order_items", src_cols=["product_id"], dst_table="products", dst_cols=["product_id"]),
        ],
    )
    shipments = TableMetadata(
        name="shipments",
        columns={
            "shipment_id": ColumnMetadata(
                name="shipment_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "shipping_cost": ColumnMetadata(
                name="shipping_cost",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "carrier": ColumnMetadata(
                name="carrier",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "shipped_on": ColumnMetadata(
                name="shipped_on",
                data_type="date",
                value_type="temporal",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        primary_key=["shipment_id"],
        foreign_keys=[
            FKEdge(src_table="shipments", src_cols=["order_id"], dst_table="orders", dst_cols=["order_id"]),
        ],
    )
    payments = TableMetadata(
        name="payments",
        columns={
            "payment_id": ColumnMetadata(
                name="payment_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                role=ColumnRole.IDENTIFIER.value,
            ),
            "amount_paid": ColumnMetadata(
                name="amount_paid",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "method": ColumnMetadata(
                name="method",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "paid_at": ColumnMetadata(
                name="paid_at",
                data_type="timestamp",
                value_type="temporal",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        primary_key=["payment_id"],
        foreign_keys=[
            FKEdge(src_table="payments", src_cols=["order_id"], dst_table="orders", dst_cols=["order_id"]),
        ],
    )
    tables = {
        "customers": customers,
        "orders": orders,
        "products": products,
        "order_items": order_items,
        "payments": payments,
        "shipments": shipments,
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


@pytest.mark.fast
def test_question_generator_is_deterministic() -> None:
    schema = _synthetic_equivalence_schema()
    first = generate_federation_equivalence_questions(schema)
    second = generate_federation_equivalence_questions(schema)
    assert first == second
    assert len(first) >= 100
    categories = {row.category for row in first}
    assert {"join_pair", "aggregate", "grouping", "date_window"}.issubset(categories)
    assert first == sorted(first, key=lambda row: (row.category, row.question_id, row.question))
