"""Null semantics for negated filters."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST
from aetherdialect._constants_runtime import REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE
from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._intent_normalize import decompose_in_not_in_where
from aetherdialect._sql_gen import build_deterministic_sql, render_predicate_clause
from aetherdialect._utils import refusal_diagnostic_code_for_intent_issue
from aetherdialect._validation_sql import validate_semantics


def _schema(*, nullable: bool = True) -> SchemaGraph:
    return SchemaGraph(
        tables={
            "t": TableMetadata(
                name="t",
                columns={
                    "status": ColumnMetadata(
                        name="status",
                        data_type="varchar",
                        value_type="string",
                        is_nullable=nullable,
                    ),
                },
                primary_key=[],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )


def _neq_intent(*, param_key: str = "p1", value: str = "active") -> RuntimeIntent:
    return RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.status"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.status"),
                    op="!=",
                    param_key=param_key,
                    value_type="string",
                    raw_value=value,
                )
            ]
        ),
        param_values={param_key: value},
    )


def _not_in_intent(values: list[object], *, param_key: str = "p1") -> RuntimeIntent:
    return RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.status"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.status"),
                    op="not in",
                    param_key=param_key,
                    value_type="string",
                    raw_value=values,
                )
            ]
        ),
        param_values={param_key: values},
    )


@pytest.mark.fast
def test_negation_includes_unknown_on_nullable_column() -> None:
    intent = _neq_intent()
    pred = (intent.where.leaves() if intent.where else [])[0]
    sql = render_predicate_clause(
        pred,
        DialectRegistry.get_dialect("duckdb"),
        schema=_schema(nullable=True),
        param_values=intent.param_values,
    )
    assert "<>" in sql
    assert "IS NULL" in sql
    assert " OR " in sql


@pytest.mark.fast
def test_negation_plain_on_non_nullable_column() -> None:
    intent = _neq_intent()
    pred = (intent.where.leaves() if intent.where else [])[0]
    sql = render_predicate_clause(
        pred,
        DialectRegistry.get_dialect("duckdb"),
        schema=_schema(nullable=False),
        param_values=intent.param_values,
    )
    assert "<>" in sql
    assert "IS NULL" not in sql


@pytest.mark.fast
def test_not_in_with_null_refuses() -> None:
    intent = _not_in_intent(["active", None, "pending"])
    result = validate_semantics(intent, _schema())
    errors = [issue for issue in result.issues if issue.severity == "error"]
    assert errors
    refusal_issues = [issue for issue in errors if issue.issue_id == "null_in_negated_list"]
    assert refusal_issues
    assert refusal_diagnostic_code_for_intent_issue(refusal_issues[0]) == DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST
    assert REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE.split(".")[0] in refusal_issues[0].message


@pytest.mark.fast
def test_not_in_nullable_column_includes_null_rows() -> None:
    intent = _not_in_intent(["active", "pending"])
    pred = (intent.where.leaves() if intent.where else [])[0]
    sql = render_predicate_clause(
        pred,
        DialectRegistry.get_dialect("duckdb"),
        schema=_schema(nullable=True),
        param_values=intent.param_values,
    )
    assert "NOT IN" in sql
    assert "IS NULL" in sql
    assert " OR " in sql

    non_nullable_sql = render_predicate_clause(
        pred,
        DialectRegistry.get_dialect("duckdb"),
        schema=_schema(nullable=False),
        param_values=intent.param_values,
    )
    assert "NOT IN" in non_nullable_sql
    assert "IS NULL" not in non_nullable_sql


@pytest.mark.fast
def test_not_in_rendering_independent_of_list_length() -> None:
    schema = _schema()
    dialect = DialectRegistry.get_dialect("duckdb")

    for values in (list(range(3)), list(range(11))):
        intent = _not_in_intent(values, param_key="p1")
        decomposed = decompose_in_not_in_where(intent)
        leaves = decomposed.where.leaves() if decomposed.where else []
        assert len(leaves) == 1
        assert leaves[0].op == "not in"
        assert leaves[0].raw_value == values

        sql = build_deterministic_sql(decomposed, schema=schema, dialect=dialect)
        assert "NOT IN" in sql
        assert "!= " not in sql.split("WHERE", 1)[-1]
