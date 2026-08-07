"""Repair must not clear join-unreachable errors by dropping tables."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FailureCategory
from aetherdialect._contracts_core import NormalizedExpr, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import IntentIssue
from aetherdialect._intent_process import resolve_repair_instruction
from aetherdialect._intent_repair import refusal_for_join_unreachable_table_removal


def _join_unreachable_issue(*, root: str = "orders", target: str = "products") -> IntentIssue:
    return IntentIssue.make(
        issue_id=f"join_unreachable_main_query_{root}_{target}",
        category=FailureCategory.WRONG_JOIN,
        severity="error",
        message=(
            f"main query: no schema join path between '{root}' and '{target}' "
            "(disconnected FK groups; add a bridging foreign_keys_add)."
        ),
        context={
            "root": root,
            "target": target,
            "tables": sorted([root, target]),
            "scope_label": "main query",
        },
    )


def _intent_with_tables(*tables: str) -> RuntimeIntent:
    return RuntimeIntent(
        tables=list(tables),
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{tables[0]}.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_join_unreachable_repair_instruction_forbids_table_removal() -> None:
    issue = _join_unreachable_issue()
    instruction = resolve_repair_instruction(issue)
    assert "orders" in instruction and "products" in instruction
    assert "Do not resolve this by removing either table" in instruction


@pytest.mark.fast
def test_repair_that_removes_table_while_join_unreachable_open_refuses() -> None:
    before = _intent_with_tables("orders", "products")
    after = _intent_with_tables("orders")
    refusal = refusal_for_join_unreachable_table_removal(before, after, [_join_unreachable_issue()])
    assert refusal is not None
    assert "orders" in refusal and "products" in refusal
    assert "no foreign key or semantic edge" in refusal


@pytest.mark.fast
def test_repair_that_keeps_tables_while_join_unreachable_open_is_allowed() -> None:
    before = _intent_with_tables("orders", "products")
    after = _intent_with_tables("orders", "products")
    refusal = refusal_for_join_unreachable_table_removal(before, after, [_join_unreachable_issue()])
    assert refusal is None
