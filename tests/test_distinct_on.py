"""Tests for DISTINCT ON semantics via ``distinct_on`` and self-join CTE lift."""

from __future__ import annotations

import re

import pytest

from aetherdialect._constants import DISTINCT_ON_CTE_NAME_PREFIX, DISTINCT_ON_RANK_COLUMN, SELF_JOIN_CTE_NAME_PREFIX
from aetherdialect._contracts_base import NormalizedExpr, OrderByCol, SchemaInvariantError
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._intent_resolve import encode_inline_self_join_as_cte
from aetherdialect._sql_gen import build_deterministic_sql
from aetherdialect._validation_schema import validate_distinct_on_schema


def _pg_render() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def _distinct_on_intent(*, with_order: bool = True) -> RuntimeIntent:
    order_by = (
        [
            OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC"),
            OrderByCol(expr=NormalizedExpr.from_column("customers.balance"), direction="DESC"),
        ]
        if with_order
        else []
    )
    return RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customers.id")),
            SelectCol(expr=NormalizedExpr.from_column("customers.name")),
        ],
        group_by_cols=[],
        order_by_cols=order_by,
        where=None,
        distinct_on=[NormalizedExpr.from_column("customers.id")],
    )


class TestDistinctOnRenderer:
    def test_renders_row_number_wrapper_without_distinct_on_token(self, simple_schema: SchemaGraph) -> None:
        sql = build_deterministic_sql(_distinct_on_intent(), schema=simple_schema, dialect=_pg_render())
        compact = _norm(sql)
        assert "row_number()" in compact.replace(" ", "")
        assert DISTINCT_ON_RANK_COLUMN.lower() in compact
        assert f"{DISTINCT_ON_CTE_NAME_PREFIX}1" in compact
        assert "distinct on" not in compact

    def test_cte_scope_distinct_on_wraps_body(self, simple_schema: SchemaGraph) -> None:
        cte = RuntimeCteStep(
            cte_name="ranked",
            tables=["customers"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
            output_columns=["name"],
            group_by_cols=[],
            order_by_cols=[
                OrderByCol(expr=NormalizedExpr.from_column("customers.balance"), direction="DESC"),
            ],
            where=None,
            having=None,
            distinct_on=[NormalizedExpr.from_column("customers.id")],
        )
        intent = RuntimeIntent(
            tables=["ranked"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("ranked.name"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        sql = build_deterministic_sql(intent, schema=simple_schema, dialect=_pg_render())
        compact = _norm(sql)
        assert "row_number()" in compact.replace(" ", "")
        assert "partition by" in compact
        assert "customers" in compact and "id" in compact
        assert "distinct on" not in compact

    def test_main_distinct_on_multiline_select_appends_rank_in_select_list(
        self, simple_schema: SchemaGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ROW_NUMBER must be appended to the SELECT list, not spliced onto the first line."""
        from aetherdialect._sql_gen import _build_deterministic_select_block

        intent = _distinct_on_intent()
        multiline_core = "SELECT customers.id,\n       customers.name\nFROM customers"

        original_build = _build_deterministic_select_block

        def _fake_build(*args: object, **kwargs: object) -> str:
            if not kwargs.get("for_cte"):
                return multiline_core
            return original_build(*args, **kwargs)

        monkeypatch.setattr("aetherdialect._sql_gen._build_deterministic_select_block", _fake_build)
        sql = build_deterministic_sql(intent, schema=simple_schema, dialect=_pg_render())
        don_match = re.search(
            rf"{DISTINCT_ON_CTE_NAME_PREFIX}\d+\s+AS\s+\((.*?)\)\s*SELECT",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert don_match is not None, sql
        body = don_match.group(1)
        assert re.search(r"customers\.name\s*,\s*row_number\s*\(", body, flags=re.IGNORECASE | re.DOTALL), body
        assert "row_number()" in _norm(body).replace(" ", "")

    def test_allocates_non_colliding_don_name(self, simple_schema: SchemaGraph) -> None:
        intent = _distinct_on_intent()
        intent = RuntimeIntent(
            tables=intent.tables,
            grain=intent.grain,
            select_cols=intent.select_cols,
            group_by_cols=intent.group_by_cols,
            order_by_cols=intent.order_by_cols,
            where=intent.where,
            distinct_on=intent.distinct_on,
            planner_cte_names=[f"{DISTINCT_ON_CTE_NAME_PREFIX}1"],
        )
        sql = build_deterministic_sql(intent, schema=simple_schema, dialect=_pg_render())
        assert f"{DISTINCT_ON_CTE_NAME_PREFIX}2" in _norm(sql)


class TestDistinctOnValidation:
    def test_requires_order_by_when_distinct_on_set(self, simple_schema: SchemaGraph) -> None:
        issues = validate_distinct_on_schema(
            [NormalizedExpr.from_column("customers.id")],
            [],
            simple_schema,
            {"customers"},
            None,
            "main query",
        )
        assert issues
        assert any("order_by" in iss.message.lower() for iss in issues)

    def test_accepts_distinct_on_with_order_by(self, simple_schema: SchemaGraph) -> None:
        issues = validate_distinct_on_schema(
            [NormalizedExpr.from_column("customers.id")],
            [
                OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC"),
                OrderByCol(expr=NormalizedExpr.from_column("customers.balance"), direction="DESC"),
            ],
            simple_schema,
            {"customers"},
            None,
            "main query",
        )
        assert not issues


class TestSelfJoinComposerLift:
    def test_lifts_duplicate_physical_table_to_sj_cte(self, simple_schema: SchemaGraph) -> None:
        intent = RuntimeIntent(
            tables=["customers", "customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        repaired = encode_inline_self_join_as_cte(intent, simple_schema)
        cte_name = f"{SELF_JOIN_CTE_NAME_PREFIX}customers"
        assert cte_name in repaired.tables
        assert "customers" in repaired.tables
        assert repaired.tables.count("customers") == 1
        assert any(step.cte_name == cte_name for step in repaired.cte_steps)

    def test_refuses_triple_physical_table_reference(self, simple_schema: SchemaGraph) -> None:
        intent = RuntimeIntent(
            tables=["customers", "customers", "customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        with pytest.raises(SchemaInvariantError, match="referenced twice"):
            encode_inline_self_join_as_cte(intent, simple_schema)

    def test_refuses_multiple_duplicated_physical_tables(self, simple_schema: SchemaGraph) -> None:
        intent = RuntimeIntent(
            tables=["customers", "customers", "orders", "orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        with pytest.raises(SchemaInvariantError, match="referenced twice"):
            encode_inline_self_join_as_cte(intent, simple_schema)

    def test_materializes_sj_table_reference_without_duplicate(self, simple_schema: SchemaGraph) -> None:
        cte_name = f"{SELF_JOIN_CTE_NAME_PREFIX}customers"
        intent = RuntimeIntent(
            tables=["customers", cte_name],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        repaired = encode_inline_self_join_as_cte(intent, simple_schema)
        assert any(step.cte_name == cte_name for step in repaired.cte_steps)
        assert repaired.tables == ["customers", cte_name]
