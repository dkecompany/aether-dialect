"""Registry token rendering refuses missing definitions before SQL build."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RegistryRenderError, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import WindowRegistryStep, WindowSpec
from aetherdialect._dialect import DialectRegistry
from aetherdialect._sql_gen import build_deterministic_sql, render_expr_sql
from aetherdialect._validation_shape import runtime_scope_registry_error_messages, validate_scope_registries


def test_missing_window_or_case_registry_refuses() -> None:
    """Bare registry refs without an active render scope raise instead of emitting ``0``."""
    with pytest.raises(RegistryRenderError, match="w01"):
        render_expr_sql(NormalizedExpr(column_ref="w01"))

    with pytest.raises(RegistryRenderError, match="c01"):
        render_expr_sql(NormalizedExpr(column_ref="c01"))


@pytest.mark.fast
def test_dangling_window_registry_is_intent_issue_before_render() -> None:
    """Undefined ``w01`` is a registry IntentIssue and blocks ``build_deterministic_sql``."""
    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[],
    )
    issues = validate_scope_registries(
        context="main query",
        window_registry=[],
        case_registry=[],
        select_cols=intent.select_cols or [],
        group_by_cols=[],
        order_by_cols=[],
    )
    assert any("undefined window registry_ref 'w01'" in (iss.message or "") for iss in issues)
    assert runtime_scope_registry_error_messages(intent)
    with pytest.raises(RegistryRenderError, match="undefined window registry_ref"):
        build_deterministic_sql(intent)


@pytest.mark.fast
def test_defined_window_registry_allows_deterministic_sql_build() -> None:
    """A matching window registry row lets ``build_deterministic_sql`` proceed past registry gate."""
    win = WindowRegistryStep(
        registry_id="w01",
        window_spec=WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("orders.id")],
            order_by=[],
        ),
    )
    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[win],
    )
    assert not runtime_scope_registry_error_messages(intent)
    dialect = DialectRegistry.get("duckdb")
    sql = build_deterministic_sql(intent, dialect=dialect)
    assert isinstance(sql, str)
    assert sql
