"""Nullable foreign keys must never be eliminated from redundant key- join removal."""

from __future__ import annotations

from typing import Any

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._core_utils import telemetry_capture
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


def _orders_customers_schema(*, fk_nullable: bool = False) -> SchemaGraph:
    fk = FKEdge(
        src_table="orders",
        src_cols=["customer_id"],
        dst_table="customers",
        dst_cols=["id"],
    )
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
                columns={"id": _col("id", is_pk=True)},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "orders": TableMetadata(
                name="orders",
                columns={
                    "order_id": _col("order_id", is_pk=True),
                    "customer_id": _col("customer_id", nullable=fk_nullable),
                },
                primary_key=["order_id"],
                foreign_keys=[fk],
            ),
        },
        join_paths_multi={"orders": {"customers": [path]}},
        effective_structural_hash="h",
    )


def _intent_with_fk_predicate(op: str | None, *, raw_value: Any = 1) -> RuntimeIntent:
    select_cols = [
        SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
        SelectCol(expr=NormalizedExpr.from_column("customers.id")),
    ]
    where = None
    if op is not None:
        where = PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.customer_id"),
                    op=op,
                    value_type="integer",
                    raw_value=raw_value,
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
@pytest.mark.parametrize(
    ("op", "raw_value"),
    [
        (None, None),
        ("=", 1),
        ("!=", 1),
        ("in", [1, 2]),
        ("not in", [1, 2]),
    ],
    ids=["no_predicate", "equals", "not_equals", "in", "not_in"],
)
def test_nullable_fk_never_eliminated(
    enable_elimination: None,
    op: str | None,
    raw_value: Any,
) -> None:
    schema = _orders_customers_schema(fk_nullable=True)
    intent = reconcile_tables(_intent_with_fk_predicate(op, raw_value=raw_value))
    with telemetry_capture(suppress_console=True) as trace:
        result = eliminate_redundant_key_joins(intent, schema)
    assert "customers" in result.tables
    if op is None:
        declined = [
            line for line in trace if "redundant_key_join_elimination" in line and "nullable_foreign_key" in line
        ]
        assert declined, "expected pipeline_trace decline for nullable foreign key"


@pytest.mark.fast
def test_non_nullable_fk_still_eliminated(enable_elimination: None) -> None:
    schema = _orders_customers_schema(fk_nullable=False)
    intent = reconcile_tables(_intent_with_fk_predicate(None))
    with telemetry_capture(suppress_console=True) as trace:
        result = eliminate_redundant_key_joins(intent, schema)
    assert result.tables == ["orders"]
    eliminated = [line for line in trace if "redundant_key_join_elimination" in line and "eliminated" in line]
    assert eliminated, "expected pipeline_trace for successful elimination"
