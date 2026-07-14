"""Seeded-intent live tests for the schema + semantic repair loop and execute-phase SQL validation failures."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from aetherdialect._contracts_base import (
    FilterParam,
    NormalizedExpr,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._live_testing import (
    deterministic_generate_validate_execute,
    run_seeded_schema_semantic_repair,
)
from aetherdialect._templates import TemplateStoreView


def _llm_forbidden(*_args, **_kwargs) -> None:
    raise AssertionError("LLM must not run in this deterministic validation-failure test")


def test_seeded_repair_fixes_mixed_grain(schema) -> None:
    """Row-level grain plus an aggregated select must be normalised by repair."""
    seeded = RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "customer.customer_id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="list first name and count per customer",
    )
    repaired, _warnings, llm_calls = run_seeded_schema_semantic_repair(
        question="list first name and count per customer",
        seeded_intent=seeded,
        schema_graph=schema,
    )
    assert repaired is not None
    aggregated = [sc for sc in (repaired.select_cols or []) if sc.is_aggregated]
    scalar = [sc for sc in (repaired.select_cols or []) if not sc.is_aggregated]
    assert aggregated or repaired.grain == "row_level"
    if aggregated and scalar:
        assert repaired.grain == "grouped"
    assert llm_calls >= 0


def test_seeded_repair_backfills_group_by(schema) -> None:
    """Grouped grain with a scalar column missing from ``group_by_cols`` must be backfilled."""
    first_name = SelectCol(expr=NormalizedExpr.from_column("customer.first_name"))
    seeded = RuntimeIntent(
        tables=["customer"],
        grain="grouped",
        select_cols=[
            first_name,
            SelectCol(expr=NormalizedExpr.from_agg("count", "customer.customer_id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="count customers per first name",
    )
    repaired, _warnings, llm_calls = run_seeded_schema_semantic_repair(
        question="count customers per first name",
        seeded_intent=seeded,
        schema_graph=schema,
    )
    assert repaired is not None
    assert repaired.grain == "grouped"
    group_terms = {g.primary_term for g in (repaired.group_by_cols or [])}
    assert "customer.first_name" in group_terms
    assert llm_calls >= 0


def test_seeded_repair_realigns_filter_on_unrelated_table(schema) -> None:
    """A filter on a table outside ``tables`` must either be dropped or its table added."""
    bad_filter = FilterParam(
        left_expr=NormalizedExpr.from_column("store.last_update"),
        op="=",
        value_type="datetime",
        bool_op="AND",
        raw_value="2023-02-15 09:46:27",
    )
    seeded = RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.first_name"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[bad_filter],
        having_param=[],
        natural_language="first names of customers at a store last updated on 2023-02-15",
    )
    repaired, _warnings, llm_calls = run_seeded_schema_semantic_repair(
        question="first names of customers at a store last updated on 2023-02-15",
        seeded_intent=seeded,
        schema_graph=schema,
    )
    assert repaired is not None
    filter_tables = {
        fp.left_expr.primary_column.split(".", 1)[0]
        for fp in (repaired.filters_param or [])
        if fp.left_expr and "." in fp.left_expr.primary_column
    }
    if filter_tables:
        assert filter_tables.issubset(set(repaired.tables))
    assert llm_calls >= 0


def test_deterministic_validation_fails_on_row_level_with_aggregate(schema, t2s) -> None:
    """Row-level grain mixed with a bare aggregate select is rejected by SQL validation."""
    intent = RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "customer.customer_id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = TemplateStoreView.empty(tmp, schema.effective_structural_hash)
        with patch(
            "aetherdialect._pipeline.get_join_choice_from_llm",
            side_effect=_llm_forbidden,
        ):
            gen_out, rows = deterministic_generate_validate_execute(
                q_norm="invalid mixed grain and aggregate",
                intent=intent,
                schema=schema,
                dialect=t2s.dialect,
                store=store,
            )
    assert gen_out.success is False
    assert rows is None
    assert gen_out.sql_validation_error
