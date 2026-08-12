"""Result headers derive from intent projection on single-engine and federated paths."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import GenerationPath, RuntimeIntent, SelectCol
from aetherdialect._pipeline_execute import (
    build_result_dataframe,
    intent_result_column_headers,
    result_columns_for_session,
)


def _intent_with_column_map() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["customer"],
        grain="many",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.name")),
            SelectCol(expr=NormalizedExpr.from_column("customer.id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={"customer.name": "customer_name", "customer.id": "id"},
    )


@pytest.mark.fast
def test_intent_projection_headers_use_column_map_names() -> None:
    intent = _intent_with_column_map()
    headers = intent_result_column_headers(intent, row_width=2)
    assert headers == ("customer_name", "id")


@pytest.mark.fast
def test_result_columns_for_session_uses_intent_headers_without_parsing_sql() -> None:
    intent = _intent_with_column_map()
    rows = [("Alice", 1), ("Bob", 2)]
    cols = result_columns_for_session(
        "SELECT bogus FROM nowhere",
        rows,
        intent=intent,
    )
    assert cols == ("customer_name", "id")


@pytest.mark.fast
def test_build_result_dataframe_honours_supplied_column_names_width() -> None:
    intent = _intent_with_column_map()
    rows = [("Alice", 1)]
    df = build_result_dataframe(
        rows,
        intent,
        "SELECT 1",
        column_names=["custom_a", "custom_b"],
    )
    assert list(df.columns) == ["custom_a", "custom_b"]


@pytest.mark.fast
def test_result_columns_for_session_honours_supplied_column_names_width() -> None:
    intent = _intent_with_column_map()
    rows = [("Alice", 1)]
    cols = result_columns_for_session(
        "SELECT 1",
        rows,
        intent=intent,
        column_names=["custom_a", "custom_b"],
    )
    assert cols == ("custom_a", "custom_b")


@pytest.mark.fast
def test_federated_path_still_prefers_bundle_column_names() -> None:
    from aetherdialect._contracts_core import FederatedSqlBundle

    intent = _intent_with_column_map()
    rows = [(1,)]
    bundle = FederatedSqlBundle(statements=(), display_sql="", column_names=("fed_col",))
    cols = result_columns_for_session(
        "SELECT 1",
        rows,
        intent=intent,
        generation_path=GenerationPath.FEDERATION_PLAN,
        federated_bundle=bundle,
    )
    assert cols == ("fed_col",)


@pytest.mark.fast
def test_federated_path_derives_headers_from_intent_when_bundle_lacks_names() -> None:
    from aetherdialect._contracts_core import FederatedSqlBundle

    intent = _intent_with_column_map()
    rows = [("Alice", 1)]
    bundle = FederatedSqlBundle(statements=(), display_sql="", column_names=())
    cols = result_columns_for_session(
        None,
        rows,
        intent=intent,
        generation_path=GenerationPath.FEDERATION_PLAN,
        federated_bundle=bundle,
    )
    assert cols == ("customer_name", "id")
