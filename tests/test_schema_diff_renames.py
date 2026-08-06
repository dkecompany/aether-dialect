"""Tests for column rename / value-type and table rename detection."""

from __future__ import annotations

import copy
import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import (
    EngineContext,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import (
    SchemaDiff,
    TableDiff,
    assign_schema_graph_hashes,
    diff_schemas,
    resolve_column_renames,
    resolve_table_renames,
)
from aetherdialect._schema_overrides import (
    apply_diff,
    build_schema_graph,
    save_schema_to_cache,
)

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _ProfileStubDialect(Dialect):
    name = "stub"

    def __init__(
        self,
        reflected_sg: SchemaGraph,
        probe_value: str = "probe-NEW",
        topk_by_table_col: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        super().__init__(MagicMock())
        self._reflected = reflected_sg
        self._probe_value = probe_value
        self._topk = topk_by_table_col or {}
        self.profile_schema_calls: list[list[str]] = []
        self.reflect_only_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return self._probe_value

    def reflect_only(self, engine_context: EngineContext) -> SchemaGraph:
        self.reflect_only_calls += 1
        return copy.deepcopy(self._reflected)

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        raise AssertionError("reflect_schema_graph should not run on partial-rebuild path")

    def profile_schema(self, sg: SchemaGraph) -> None:
        self.profile_schema_calls.append(sorted(sg.tables.keys()))
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


def _mk_col(name: str, data_type: str = "integer", **kw: Any) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, **kw)


def _mk_table(
    name: str,
    cols: dict[str, ColumnMetadata],
    pk: list[str] | None = None,
    fks: list[FKEdge] | None = None,
) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns=cols,
        primary_key=pk or [],
        foreign_keys=fks or [],
    )


def test_diff_classifies_value_type_changed() -> None:
    """Integer → varchar bumps both data_type AND value_type."""
    cached = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "integer")})},
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "varchar")})},
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    td = diff.per_table["a"]
    assert td.retyped_columns == (("x", "integer", "varchar"),)
    assert td.value_type_changed_columns == (("x", "integer", "string"),)


def test_diff_value_type_unchanged_when_only_raw_type_changes() -> None:
    """Integer → bigint changes data_type but value_type stays 'integer' → redeclared, not retyped."""
    cached = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "integer")})},
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "bigint")})},
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    td = diff.per_table["a"]
    assert td.redeclared_columns == (("x", "integer", "bigint"),)
    assert td.retyped_columns == ()
    assert td.value_type_changed_columns == ()


def test_resolve_column_renames_matches_by_topk_overlap() -> None:
    """A column dropped + a column added with overlapping Top-K → confirmed rename."""
    cached = SchemaGraph(
        tables={
            "a": _mk_table(
                "a",
                {
                    "x": _mk_col(
                        "x",
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
        tables={
            "a": _mk_table("a", {"y": _mk_col("y", "varchar")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    assert diff.per_table["a"].added_columns == ("y",)
    assert diff.per_table["a"].dropped_columns == ("x",)

    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={("a", "y"): ["alpha", "beta", "gamma"]},
    )
    resolved = resolve_column_renames(diff, cached, new_struct, dialect)
    td = resolved.per_table["a"]
    assert td.renamed_columns == (("x", "y"),)
    assert td.added_columns == ()
    assert td.dropped_columns == ()


def test_resolve_column_renames_skips_when_overlap_low() -> None:
    """Disjoint Top-K → no rename, drop+add stays."""
    cached = SchemaGraph(
        tables={
            "a": _mk_table(
                "a",
                {
                    "x": _mk_col(
                        "x",
                        "varchar",
                        frequent_values=["alpha", "beta"],
                        distinct_count=2,
                    )
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"a": _mk_table("a", {"y": _mk_col("y", "varchar")})},
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={("a", "y"): ["zulu", "yankee"]},
    )
    resolved = resolve_column_renames(diff, cached, new_struct, dialect)
    td = resolved.per_table["a"]
    assert td.renamed_columns == ()
    assert td.added_columns == ("y",)
    assert td.dropped_columns == ("x",)


def test_apply_diff_renamed_columns_preserve_cached_profile() -> None:
    """Renaming a column keeps cached top_k/role; only key + .name (and optionally type) change."""
    cached = SchemaGraph(
        tables={
            "a": _mk_table(
                "a",
                {
                    "x": _mk_col(
                        "x",
                        "varchar",
                        role="categorical",
                        frequent_values=["alpha", "beta"],
                        distinct_count=2,
                    ),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"a": _mk_table("a", {"y": _mk_col("y", "varchar")})},
        join_paths_multi={},
    )
    diff = SchemaDiff(
        per_table={
            "a": TableDiff(renamed_columns=(("x", "y"),)),
        },
    )
    dialect = _ProfileStubDialect(new_struct)
    out = apply_diff(cached, new_struct, diff, dialect)

    assert "x" not in out.tables["a"].columns
    assert "y" in out.tables["a"].columns
    new_col = out.tables["a"].columns["y"]
    assert new_col.name == "y"
    assert new_col.role == "categorical"
    assert new_col.frequent_values == ["alpha", "beta"]
    assert dialect.profile_schema_calls == []


def test_apply_diff_renamed_columns_with_retype_uses_new_data_type() -> None:
    """Rename + type change → cached profile preserved, data_type/value_type updated."""
    cached = SchemaGraph(
        tables={
            "a": _mk_table(
                "a",
                {"x": _mk_col("x", "integer", frequent_values=["1", "2"], distinct_count=2)},
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"a": _mk_table("a", {"y": _mk_col("y", "varchar")})},
        join_paths_multi={},
    )
    diff = SchemaDiff(
        per_table={"a": TableDiff(renamed_columns=(("x", "y"),))},
    )
    dialect = _ProfileStubDialect(new_struct)
    out = apply_diff(cached, new_struct, diff, dialect)
    col = out.tables["a"].columns["y"]
    assert col.data_type == "varchar"
    assert col.value_type == "string"

    assert col.frequent_values == ["1", "2"]


def test_resolve_table_renames_matches_by_column_overlap() -> None:
    """Dropped table + added table with overlapping profiles but different column names rename via profile overlap."""
    cached = SchemaGraph(
        tables={
            "old_t": _mk_table(
                "old_t",
                {
                    "a": _mk_col(
                        "a",
                        "varchar",
                        frequent_values=["x", "y", "z"],
                        distinct_count=3,
                    ),
                    "b": _mk_col(
                        "b",
                        "integer",
                        frequent_values=["1", "2", "3"],
                        distinct_count=3,
                    ),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "new_t": _mk_table(
                "new_t",
                {
                    "alpha": _mk_col("alpha", "varchar"),
                    "beta": _mk_col("beta", "integer"),
                },
            ),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    assert diff.added_tables == ("new_t",)
    assert diff.dropped_tables == ("old_t",)
    assert diff.table_renames == ()

    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={
            ("new_t", "alpha"): ["x", "y", "z"],
            ("new_t", "beta"): ["1", "2", "3"],
        },
    )
    resolved = resolve_table_renames(diff, cached, new_struct, dialect)
    assert resolved.table_renames == (("old_t", "new_t"),)
    assert resolved.added_tables == ()
    assert resolved.dropped_tables == ()
    td = resolved.per_table.get("new_t")
    assert td is not None
    assert set(td.renamed_columns) == {("a", "alpha"), ("b", "beta")}


def test_resolve_table_renames_with_partial_column_renames() -> None:
    """Table rename where some columns keep their names and one is also renamed."""
    cached = SchemaGraph(
        tables={
            "old_t": _mk_table(
                "old_t",
                {
                    "a": _mk_col(
                        "a",
                        "varchar",
                        frequent_values=["x", "y", "z"],
                        distinct_count=3,
                    ),
                    "b_old": _mk_col(
                        "b_old",
                        "integer",
                        frequent_values=["1", "2", "3"],
                        distinct_count=3,
                    ),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "new_t": _mk_table(
                "new_t",
                {
                    "a": _mk_col("a", "varchar"),
                    "b_new": _mk_col("b_new", "integer"),
                },
            ),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    assert diff.table_renames == ()
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={
            ("new_t", "a"): ["x", "y", "z"],
            ("new_t", "b_new"): ["1", "2", "3"],
        },
    )
    resolved = resolve_table_renames(diff, cached, new_struct, dialect)
    assert resolved.table_renames == (("old_t", "new_t"),)
    td = resolved.per_table.get("new_t")
    assert td is not None
    assert ("b_old", "b_new") in td.renamed_columns
    assert ("a", "a") not in td.renamed_columns


def test_resolve_table_renames_skips_when_disjoint_profiles() -> None:
    """No column overlap yields drop+add rather than rename when column- set equality also misses."""
    cached = SchemaGraph(
        tables={
            "old_t": _mk_table(
                "old_t",
                {
                    "a": _mk_col("a", "varchar", frequent_values=["x", "y"], distinct_count=2),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "new_t": _mk_table("new_t", {"alpha": _mk_col("alpha", "varchar")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={("new_t", "alpha"): ["totally", "different"]},
    )
    resolved = resolve_table_renames(diff, cached, new_struct, dialect)
    assert resolved.table_renames == ()
    assert resolved.added_tables == ("new_t",)
    assert resolved.dropped_tables == ("old_t",)


def test_resolve_table_renames_skips_when_column_count_differs() -> None:
    """Different column counts → not a rename candidate."""
    cached = SchemaGraph(
        tables={
            "old_t": _mk_table(
                "old_t",
                {
                    "a": _mk_col("a", "varchar", frequent_values=["x"], distinct_count=1),
                    "b": _mk_col("b", "integer", frequent_values=["1"], distinct_count=1),
                },
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"new_t": _mk_table("new_t", {"a": _mk_col("a", "varchar")})},
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _ProfileStubDialect(
        new_struct,
        topk_by_table_col={("new_t", "a"): ["x"]},
    )
    resolved = resolve_table_renames(diff, cached, new_struct, dialect)
    assert resolved.table_renames == ()


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _save_with_probe(sg: SchemaGraph, ctx: EngineContext, notes: str, probe: str, path: str) -> None:
    sg.notes_sha256 = hashlib.sha256(notes.encode("utf-8")).hexdigest()
    assign_schema_graph_hashes(sg, ctx, sg.notes_sha256)
    sg.ddl_probe_hash = probe
    save_schema_to_cache(sg, path)


def test_build_schema_graph_detects_column_rename(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    """Probe-mismatch + column rename inside customers → cached profile preserved."""
    ctx = EngineContext()
    schema_graph.tables["customers"].columns["email"].frequent_values = [
        "a@x.com",
        "b@x.com",
        "c@x.com",
    ]
    schema_graph.tables["customers"].columns["email"].distinct_count = 3
    _save_with_probe(schema_graph, ctx, "n", probe="probe-OLD", path=cache_path)

    new_struct = copy.deepcopy(schema_graph)
    customers = new_struct.tables["customers"]
    old_col = customers.columns.pop("email")
    customers.columns["email_addr"] = ColumnMetadata(
        name="email_addr",
        data_type=old_col.data_type,
        is_nullable=old_col.is_nullable,
    )

    dialect = _ProfileStubDialect(
        new_struct,
        probe_value="probe-NEW",
        topk_by_table_col={("customers", "email_addr"): ["a@x.com", "b@x.com", "c@x.com"]},
    )
    out = build_schema_graph(dialect, ctx, notes_content="n")

    assert "email" not in out.tables["customers"].columns
    assert "email_addr" in out.tables["customers"].columns


def test_build_schema_graph_detects_table_rename(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    """Table and column rename together resolves through profile overlap when column-set equality fails."""
    ctx = EngineContext()
    products = schema_graph.tables["products"]
    products.columns["title"].frequent_values = ["alpha", "beta", "gamma"]
    products.columns["title"].distinct_count = 3
    _save_with_probe(schema_graph, ctx, "n", probe="probe-OLD", path=cache_path)

    new_struct = copy.deepcopy(schema_graph)
    new_products = new_struct.tables.pop("products")
    new_products.name = "items"
    title_col = new_products.columns.pop("title")
    new_products.columns["label"] = ColumnMetadata(
        name="label",
        data_type=title_col.data_type,
        is_nullable=title_col.is_nullable,
    )
    for c in new_products.columns.values():
        c.frequent_values = []
        c.distinct_count = 0
    new_struct.tables["items"] = new_products
    if "orders" in new_struct.tables:
        for fk in new_struct.tables["orders"].foreign_keys:
            if fk.dst_table == "products":
                fk.dst_table = "items"

    dialect = _ProfileStubDialect(
        new_struct,
        probe_value="probe-NEW",
        topk_by_table_col={
            ("items", "label"): ["alpha", "beta", "gamma"],
        },
    )
    out = build_schema_graph(dialect, ctx, notes_content="n")
    assert "items" in out.tables
    assert "products" not in out.tables
    if "orders" in out.tables:
        for fk in out.tables["orders"].foreign_keys:
            if fk.src_cols == ["product_id"]:
                assert fk.dst_table == "items"
    assert "label" in out.tables["items"].columns
    assert "title" not in out.tables["items"].columns
