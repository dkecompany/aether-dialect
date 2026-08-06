"""Rename inference must refuse on ties, not pick first."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import (
    diff_schemas,
    resolve_column_renames,
    resolve_table_renames,
)

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _ProfileStubDialect(Dialect):
    name = "stub"

    def __init__(
        self,
        reflected_sg: SchemaGraph,
        topk_by_table_col: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        super().__init__(MagicMock())
        self._reflected = reflected_sg
        self._topk = topk_by_table_col or {}

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return "probe"

    def reflect_only(self, engine_context: EngineContext) -> SchemaGraph:
        return copy.deepcopy(self._reflected)

    def profile_schema(self, sg: SchemaGraph) -> None:
        for tname, t in sg.tables.items():
            t.row_count = max(t.row_count, 1)
            for cname, c in t.columns.items():
                vals = self._topk.get((tname, cname))
                if vals is not None:
                    c.frequent_values = list(vals)
                    c.distinct_count = max(c.distinct_count, len(vals))

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


def _col(name: str, data_type: str = "varchar", **kw: Any) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, **kw)


def _table(name: str, cols: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key="")


def _identical_pair_schema() -> tuple[SchemaGraph, SchemaGraph]:
    """Two dropped and two added tables with the same column multiset — swap is ambiguous."""
    shape = {"id": _col("id", "integer"), "label": _col("label", "varchar")}
    old = SchemaGraph(
        tables={"t_a": _table("t_a", shape), "t_b": _table("t_b", shape)},
        join_paths_multi={},
    )
    new = SchemaGraph(
        tables={"n_a": _table("n_a", shape), "n_b": _table("n_b", shape)},
        join_paths_multi={},
    )
    return old, new


@pytest.mark.fast
def test_diff_schemas_refuses_ambiguous_table_multiset_match() -> None:
    old, new = _identical_pair_schema()
    diff = diff_schemas(old, new)
    assert diff.table_renames == ()
    assert set(diff.dropped_tables) == {"t_a", "t_b"}
    assert set(diff.added_tables) == {"n_a", "n_b"}


@pytest.mark.fast
def test_diff_schemas_keeps_unique_multiset_match() -> None:
    """Distinct signatures still rename one-to-one."""
    old = SchemaGraph(
        tables={
            "t_a": _table("t_a", {"id": _col("id", "integer")}),
            "t_b": _table("t_b", {"code": _col("code", "varchar")}),
        },
        join_paths_multi={},
    )
    new = SchemaGraph(
        tables={
            "n_a": _table("n_a", {"id": _col("id", "integer")}),
            "n_b": _table("n_b", {"code": _col("code", "varchar")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(old, new)
    assert set(diff.table_renames) == {("t_a", "n_a"), ("t_b", "n_b")}


@pytest.mark.fast
def test_resolve_table_renames_refuses_tied_candidate_scores() -> None:
    """Two added tables with equal profile overlap must not pick the first."""
    shared_topk = ["alpha", "beta", "gamma"]
    cached = SchemaGraph(
        tables={
            "old_t": _table(
                "old_t",
                {
                    "a": _col("a", "varchar", frequent_values=shared_topk, distinct_count=3),
                    "b": _col("b", "integer", frequent_values=["1", "2", "3"], distinct_count=3),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "new_x": _table("new_x", {"a": _col("a", "varchar"), "b": _col("b", "integer")}),
            "new_y": _table("new_y", {"a": _col("a", "varchar"), "b": _col("b", "integer")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    assert diff.table_renames == ()
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={
            ("new_x", "a"): shared_topk,
            ("new_x", "b"): ["1", "2", "3"],
            ("new_y", "a"): shared_topk,
            ("new_y", "b"): ["1", "2", "3"],
        },
    )
    resolved = resolve_table_renames(diff, cached, new_struct, dialect)
    assert resolved.table_renames == ()
    assert set(resolved.dropped_tables) == {"old_t"}
    assert set(resolved.added_tables) == {"new_x", "new_y"}


@pytest.mark.fast
def test_resolve_column_renames_refuses_tied_jaccard_scores() -> None:
    """Two added columns with identical overlap to one dropped column must not rename."""
    shared_topk = ["alpha", "beta", "gamma"]
    cached = SchemaGraph(
        tables={
            "t": _table(
                "t",
                {
                    "old_col": _col(
                        "old_col",
                        "varchar",
                        frequent_values=shared_topk,
                        distinct_count=3,
                    ),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "t": _table(
                "t",
                {
                    "new_x": _col("new_x", "varchar"),
                    "new_y": _col("new_y", "varchar"),
                },
            ),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={
            ("t", "new_x"): shared_topk,
            ("t", "new_y"): shared_topk,
        },
    )
    resolved = resolve_column_renames(diff, cached, new_struct, dialect)
    td = resolved.per_table["t"]
    assert td.renamed_columns == ()
    assert set(td.added_columns) == {"new_x", "new_y"}
    assert td.dropped_columns == ("old_col",)


@pytest.mark.fast
def test_resolve_column_renames_keeps_unique_jaccard_match() -> None:
    """Unambiguous overlap still confirms a column rename."""
    cached = SchemaGraph(
        tables={
            "t": _table(
                "t",
                {
                    "old_col": _col(
                        "old_col",
                        "varchar",
                        frequent_values=["alpha", "beta", "gamma"],
                        distinct_count=3,
                    ),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"t": _table("t", {"new_col": _col("new_col", "varchar")})},
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={("t", "new_col"): ["alpha", "beta", "gamma"]},
    )
    resolved = resolve_column_renames(diff, cached, new_struct, dialect)
    td = resolved.per_table["t"]
    assert td.renamed_columns == (("old_col", "new_col"),)
    assert td.added_columns == ()
    assert td.dropped_columns == ()
