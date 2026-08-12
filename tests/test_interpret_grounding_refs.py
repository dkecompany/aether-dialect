"""Interpret grounding refs must be table or table.column only."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_core import InterpretPlan
from aetherdialect._intent_expr import interpret_plan_references_absent_entities


def _plan(
    *,
    tables: tuple[str, ...] = ("orders",),
    grounding: tuple[tuple[str, str], ...] = (),
) -> InterpretPlan:
    return InterpretPlan(approach="list rows", tables=tables, grounding=grounding)


@pytest.mark.fast
def test_grounding_accepts_table_and_qualified_column() -> None:
    allowed = frozenset({"orders", "customers"})
    columns = {
        "orders": frozenset({"order_id", "status", "customer_id"}),
        "customers": frozenset({"customer_id", "region"}),
    }
    plan = _plan(
        tables=("orders", "customers"),
        grounding=(
            ("orders", "fact rows"),
            ("orders.status", "status filter"),
            ("customers.region", "region filter"),
        ),
    )
    assert interpret_plan_references_absent_entities(plan, allowed, columns_by_table=columns) is False


@pytest.mark.fast
def test_grounding_refuses_bare_column_token() -> None:
    allowed = frozenset({"orders", "customers"})
    columns = {
        "orders": frozenset({"order_id", "status"}),
        "customers": frozenset({"region"}),
    }
    plan = _plan(grounding=(("status", "status filter"),))
    assert interpret_plan_references_absent_entities(plan, allowed, columns_by_table=columns) is True


@pytest.mark.fast
def test_grounding_refuses_unknown_qualified_column() -> None:
    allowed = frozenset({"orders"})
    columns = {"orders": frozenset({"order_id", "customer_id"})}
    plan = _plan(grounding=(("orders.status", "status filter"),))
    assert interpret_plan_references_absent_entities(plan, allowed, columns_by_table=columns) is True


@pytest.mark.fast
def test_grounding_refuses_multi_segment_ref() -> None:
    allowed = frozenset({"orders"})
    plan = _plan(grounding=(("src.orders.status", "status"),))
    assert interpret_plan_references_absent_entities(plan, allowed) is True


@pytest.mark.fast
def test_grounding_refuses_unknown_table() -> None:
    allowed = frozenset({"orders"})
    plan = _plan(tables=("ghost",), grounding=(("ghost", "rows"),))
    assert interpret_plan_references_absent_entities(plan, allowed) is True
