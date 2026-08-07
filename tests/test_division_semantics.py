"""Integer division renders as true (fractional) division in every dialect."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import MulGroup, NormalizedExpr
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._sql_gen import _render_group_sql, sql_gen_type_scope


def _uninit_dialect(engine: str):
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _integer_division_group() -> MulGroup:
    return MulGroup(
        multiply=[NormalizedExpr(column_ref="t.a")],
        divide=[NormalizedExpr(column_ref="t.b")],
    )


def _integer_division_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "t": TableMetadata(
                name="t",
                columns={
                    "a": ColumnMetadata(name="a", data_type="integer", value_type="integer"),
                    "b": ColumnMetadata(name="b", data_type="integer", value_type="integer"),
                },
                primary_key=[],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_integer_division_is_fractional_in_every_dialect(engine: str) -> None:
    dialect = _uninit_dialect(engine)
    with sql_gen_type_scope(_integer_division_schema(), {}):
        sql = _render_group_sql(_integer_division_group(), dialect)
    if dialect.integer_division_truncates:
        assert "CAST(" in sql.upper(), f"{engine} should cast the numerator for true division"
    assert "/" in sql
