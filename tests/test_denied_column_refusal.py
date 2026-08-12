"""Denied-column refusals stay neutral in user text and auditable in diagnostics."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
from aetherdialect._constants_runtime import REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE
from aetherdialect._contracts_base import FailureCategory, NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._utils import (
    failure_kind_is_permission_denied,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._validation_rules import (
    validate_denied_references,
    validate_deny_bare_select,
)


def _schema_with_denied_amount() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", value_type="integer"),
                    "amount": ColumnMetadata(name="amount", data_type="numeric", value_type="numeric"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
        deny_columns={"orders": {"amount"}},
    )


def _bare_select_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )


def _filter_denied_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op=">",
                    value_type="numeric",
                    raw_value=0,
                ),
            ]
        ),
        having=None,
    )


@pytest.mark.fast
def test_refusal_does_not_name_the_column() -> None:
    schema = _schema_with_denied_amount()
    bare_issues = validate_deny_bare_select(_bare_select_intent(), schema)
    denied_issues = validate_denied_references(_filter_denied_intent(), schema)
    assert bare_issues and denied_issues
    for issue in bare_issues + denied_issues:
        assert issue.message == REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE
        assert "orders" not in issue.message
        assert "amount" not in issue.message


@pytest.mark.fast
def test_denied_reference_is_a_permission_outcome() -> None:
    assert failure_kind_is_permission_denied(FailureCategory.DENIED_REFERENCE.value)
    assert failure_kind_is_permission_denied(FailureCategory.DENY_BARE_SELECT.value)
    assert not failure_kind_is_permission_denied(FailureCategory.WHERE_VALIDITY.value)


@pytest.mark.fast
def test_audit_field_carries_the_column() -> None:
    schema = _schema_with_denied_amount()
    buf: list = []
    tok = set_diagnostic_collector(buf)
    try:
        validate_deny_bare_select(_bare_select_intent(), schema)
        validate_denied_references(_filter_denied_intent(), schema)
    finally:
        reset_diagnostic_collector(tok)
    assert buf
    for diag in buf:
        assert diag.code == DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
        assert diag.subject == "orders.amount"
        detail_map = dict(diag.details)
        assert detail_map.get("table") == "orders"
        assert detail_map.get("column") == "amount"
