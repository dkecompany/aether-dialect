"""Tests for exact numeric handling in federated averages."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import MulGroup, NormalizedExpr
from aetherdialect._contracts_core import ResidualSpec, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_plan import render_federation_residual_sql


def _graph_with_amount_columns() -> SchemaGraph:
    exact_col = ColumnMetadata(
        name="exact_amt",
        data_type="DECIMAL(19,4)",
        is_primary_key=False,
        numeric_precision=19,
        numeric_scale=4,
        is_exact_numeric=True,
    )
    approx_col = ColumnMetadata(
        name="approx_amt",
        data_type="DOUBLE PRECISION",
        is_primary_key=False,
        is_exact_numeric=False,
    )
    tables = {
        "store": TableMetadata(
            name="store",
            columns={"exact_amt": exact_col, "approx_amt": approx_col},
            primary_key=[],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi={})


@pytest.mark.fast
def test_exact_avg_residual_uses_decimal_cast() -> None:
    schema = _graph_with_amount_columns()
    residual = ResidualSpec(
        select_cols=(
            SelectCol(
                expr=NormalizedExpr(
                    add_groups=[MulGroup(coefficient=1.0, multiply=["store.exact_amt"], agg_func="avg")],
                ),
            ),
        ),
    )
    sql = render_federation_residual_sql("SELECT * FROM joined", residual, schema=schema)
    assert "DECIMAL(38," in sql.upper()
    assert "CAST" in sql.upper()
    assert "DOUBLE" not in sql.upper()


@pytest.mark.fast
def test_float_avg_residual_uses_double_cast() -> None:
    schema = _graph_with_amount_columns()
    residual = ResidualSpec(
        select_cols=(
            SelectCol(
                expr=NormalizedExpr(
                    add_groups=[MulGroup(coefficient=1.0, multiply=["store.approx_amt"], agg_func="avg")],
                ),
            ),
        ),
    )
    sql = render_federation_residual_sql("SELECT * FROM joined", residual, schema=schema)
    assert "CAST" in sql.upper()
    assert "DOUBLE" in sql.upper()
