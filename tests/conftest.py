"""Shared fixtures for aetherdialect test suite."""

from typing import Any

import pytest

from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    FKEdge,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TableRole,
    TemplateStats,
)
from aetherdialect._contracts_core import (
    FilterParam,
    NormalizedExpr,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
    runtime_intent_to_concrete,
)
from aetherdialect._schema_profiling import assign_column_ops


def _term_str(term: Any) -> str:
    """Render a multiply/divide leaf as the legacy display string ('*', 'COUNT(*)', 'tbl.col', etc.)."""
    if isinstance(term, str):
        return term
    if not isinstance(term, NormalizedExpr):
        return str(term)
    if term.column_ref:
        return term.column_ref
    if term.star:
        return "*"
    if term.keyword:
        return term.keyword.upper()
    if term.raw_sql:
        return term.raw_sql
    from aetherdialect._sql_gen import render_expr_sql

    return render_expr_sql(term)


def term_strs(terms: Any) -> list[str]:
    """Map a sequence of multiply/divide leaves to legacy display strings."""
    return [_term_str(t) for t in (terms or [])]


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "integration: mocked multi-module pipeline slices")


@pytest.fixture
def schema_graph() -> SchemaGraph:
    """Three-table schema: customers, orders, products with FK edges."""
    customers = TableMetadata(
        name="customers",
        columns={
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
                distinct_ratio=1.0,
                row_count=100,
            ),
            "name": ColumnMetadata(
                name="name",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=95,
                distinct_ratio=0.95,
                row_count=100,
            ),
            "email": ColumnMetadata(
                name="email",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
                distinct_count=100,
                distinct_ratio=1.0,
                row_count=100,
            ),
            "active": ColumnMetadata(
                name="active",
                data_type="boolean",
                value_type="boolean",
                role=ColumnRole.BOOLEAN.value,
                distinct_count=2,
                distinct_ratio=0.02,
                row_count=100,
                top_k_values=["true", "false"],
            ),
            "created_at": ColumnMetadata(
                name="created_at",
                data_type="timestamp",
                value_type="date",
                role=ColumnRole.AUDIT.value,
                distinct_count=100,
                distinct_ratio=1.0,
                row_count=100,
            ),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=100,
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=500,
                distinct_ratio=1.0,
                row_count=500,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                fk_target=("customers", "customer_id"),
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
                distinct_ratio=0.2,
                row_count=500,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=300,
                distinct_ratio=0.6,
                row_count=500,
            ),
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=5,
                distinct_ratio=0.01,
                row_count=500,
                top_k_values=[
                    "pending",
                    "shipped",
                    "delivered",
                    "cancelled",
                    "returned",
                ],
            ),
            "order_date": ColumnMetadata(
                name="order_date",
                data_type="date",
                value_type="date",
                role=ColumnRole.TEMPORAL.value,
                distinct_count=365,
                distinct_ratio=0.73,
                row_count=500,
            ),
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                fk_target=("products", "product_id"),
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=50,
                distinct_ratio=0.1,
                row_count=500,
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["customer_id"],
            ),
            FKEdge(
                src_table="orders",
                src_cols=["product_id"],
                dst_table="products",
                dst_cols=["product_id"],
            ),
        ],
        role=TableRole.FACT.value,
        row_count=500,
    )
    products = TableMetadata(
        name="products",
        columns={
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=50,
                distinct_ratio=1.0,
                row_count=50,
            ),
            "title": ColumnMetadata(
                name="title",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=50,
                distinct_ratio=1.0,
                row_count=50,
            ),
            "price": ColumnMetadata(
                name="price",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=40,
                distinct_ratio=0.8,
                row_count=50,
            ),
            "category": ColumnMetadata(
                name="category",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=10,
                distinct_ratio=0.2,
                row_count=50,
                top_k_values=["electronics", "books", "clothing", "food", "toys"],
            ),
        },
        primary_key=["product_id"],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=50,
    )
    sg = SchemaGraph(
        tables={"customers": customers, "orders": orders, "products": products},
        join_paths_multi={
            "customers": {
                "orders": [[{"src": "orders.customer_id", "dst": "customers.customer_id"}]],
            },
            "orders": {
                "customers": [[{"src": "orders.customer_id", "dst": "customers.customer_id"}]],
                "products": [[{"src": "orders.product_id", "dst": "products.product_id"}]],
            },
            "products": {
                "orders": [[{"src": "orders.product_id", "dst": "products.product_id"}]],
            },
        },
        effective_structural_hash="test_hash_abc123",
    )
    assign_column_ops(sg)
    return sg


@pytest.fixture
def minimal_intent() -> RuntimeIntent:
    """Minimal RuntimeIntent for orders table."""
    return RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )


@pytest.fixture
def grouped_intent() -> RuntimeIntent:
    """Grouped RuntimeIntent with aggregation."""
    return RuntimeIntent(
        tables=["orders", "customers"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customers.name")),
            SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
        ],
        group_by_cols=[NormalizedExpr.from_column("customers.name")],
        order_by_cols=[],
        filters_param=[
            FilterParam(
                left_expr=NormalizedExpr.from_column("orders.status"),
                op="=",
                value_type="string",
                param_key="p1",
                raw_value="shipped",
            ),
        ],
        having_param=[],
        param_values={"p1": "shipped"},
        column_map={"name": "customers", "amount": "orders", "status": "orders"},
    )


@pytest.fixture
def sample_template(grouped_intent) -> Template:
    """Sample Template wrapping grouped_intent."""
    return Template(
        id="T0001",
        effective_structural_hash="test_hash_abc123",
        intent_signature=runtime_intent_to_concrete(grouped_intent, "test_id"),
        intent_key="test_intent_key_hash",
        tables_used=["customers", "orders"],
        sql_param="SELECT customers.name, SUM(orders.amount) FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE orders.status = :p1 GROUP BY customers.name",
        sql_fp="test_sql_fp",
        shape=SQLShape(num_joins=1, has_group_by=True, has_agg=True),
        colmap_sig="test_colmap_sig",
        value_history=ValueHistory(
            param_values=[{"p1": "shipped"}],
            questions=["total amount by customer for shipped orders"],
            natural_language=["total amount by customer for shipped orders"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


@pytest.fixture
def simple_schema() -> SchemaGraph:
    """Schema with customers and orders tables for validation tests."""
    customers = TableMetadata(
        name="customers",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
            ),
            "name": ColumnMetadata(
                name="name",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=80,
            ),
            "email": ColumnMetadata(
                name="email",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
                distinct_count=100,
            ),
            "balance": ColumnMetadata(
                name="balance",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=50,
            ),
        },
        foreign_keys=[],
        primary_key="",
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=200,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=150,
            ),
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=5,
            ),
        },
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
            )
        ],
        primary_key="",
    )
    sg = SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"customers": customers, "orders": orders},
    )
    assign_column_ops(sg)
    return sg


@pytest.fixture
def typed_schema() -> SchemaGraph:
    """Schema with typed columns for validation tests."""
    customers = TableMetadata(
        name="customers",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
            ),
            "name": ColumnMetadata(
                name="name",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "balance": ColumnMetadata(
                name="balance",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "created_at": ColumnMetadata(
                name="created_at",
                data_type="timestamp",
                value_type="date",
                role=ColumnRole.TEMPORAL.value,
            ),
            "description": ColumnMetadata(
                name="description",
                data_type="text",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
            ),
        },
        foreign_keys=[],
        primary_key="",
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                role=ColumnRole.IDENTIFIER.value,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "order_date": ColumnMetadata(
                name="order_date",
                data_type="date",
                value_type="date",
                role=ColumnRole.TEMPORAL.value,
            ),
        },
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
            )
        ],
        primary_key="",
    )
    sg = SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"customers": customers, "orders": orders},
    )
    assign_column_ops(sg)
    return sg
