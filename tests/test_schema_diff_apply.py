"""Tests for SchemaDiff + apply_diff partial-rebuild path."""

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
from aetherdialect._schema_finalize import apply_diff, build_schema_graph
from aetherdialect._schema_graph import (
    assign_schema_graph_hashes,
    diff_schemas,
)
from aetherdialect._schema_reflect import save_schema_to_cache
from aetherdialect._utils_artifacts import read_gzip_json

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _PartialRebuildStubDialect(Dialect):
    name = "stub"

    def __init__(self, reflected_sg: SchemaGraph, probe_value: str) -> None:
        super().__init__(MagicMock())
        self._reflected = reflected_sg
        self._probe_value = probe_value
        self.reflect_only_calls = 0
        self.profile_schema_calls: list[list[str]] = []

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
        for t in sg.tables.values():
            t.row_count = max(t.row_count, 1)
            for c in t.columns.values():
                if c.distinct_count == 0:
                    c.distinct_count = 1

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


def test_diff_detects_added_table() -> None:
    old = SchemaGraph(tables={"a": _mk_table("a", {"x": _mk_col("x")})}, join_paths_multi={})
    new = SchemaGraph(
        tables={
            "a": _mk_table("a", {"x": _mk_col("x")}),
            "b": _mk_table("b", {"y": _mk_col("y")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(old, new)
    assert diff.added_tables == ("b",)
    assert diff.dropped_tables == ()
    assert diff.table_renames == ()
    assert diff.per_table == {}


def test_diff_detects_dropped_table() -> None:
    old = SchemaGraph(
        tables={
            "a": _mk_table("a", {"x": _mk_col("x")}),
            "b": _mk_table("b", {"y": _mk_col("y")}),
        },
        join_paths_multi={},
    )
    new = SchemaGraph(tables={"a": _mk_table("a", {"x": _mk_col("x")})}, join_paths_multi={})
    diff = diff_schemas(old, new)
    assert diff.dropped_tables == ("b",)
    assert diff.added_tables == ()


def test_diff_detects_added_and_dropped_columns() -> None:
    old = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x"), "y": _mk_col("y")})},
        join_paths_multi={},
    )
    new = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x"), "z": _mk_col("z")})},
        join_paths_multi={},
    )
    diff = diff_schemas(old, new)
    assert diff.per_table["a"].added_columns == ("z",)
    assert diff.per_table["a"].dropped_columns == ("y",)
    assert diff.per_table["a"].retyped_columns == ()


def test_diff_detects_redeclared_column_same_value_type() -> None:
    """Integer → bigint changes catalog type but normalized value_type stays integer."""
    old = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "integer")})},
        join_paths_multi={},
    )
    new = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "bigint")})},
        join_paths_multi={},
    )
    diff = diff_schemas(old, new)
    td = diff.per_table["a"]
    assert td.redeclared_columns == (("x", "integer", "bigint"),)
    assert td.retyped_columns == ()
    assert td.value_type_changed_columns == ()


def test_diff_detects_retyped_column_when_value_type_changes() -> None:
    old = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "integer")})},
        join_paths_multi={},
    )
    new = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "varchar")})},
        join_paths_multi={},
    )
    diff = diff_schemas(old, new)
    td = diff.per_table["a"]
    assert td.retyped_columns == (("x", "integer", "varchar"),)
    assert td.redeclared_columns == ()
    assert td.value_type_changed_columns == (("x", "integer", "string"),)


def test_diff_detects_table_rename_with_same_columns() -> None:
    cols = {"id": _mk_col("id"), "label": _mk_col("label", "varchar")}
    old = SchemaGraph(tables={"old_t": _mk_table("old_t", cols)}, join_paths_multi={})
    new = SchemaGraph(tables={"new_t": _mk_table("new_t", copy.deepcopy(cols))}, join_paths_multi={})
    diff = diff_schemas(old, new)
    assert diff.table_renames == (("old_t", "new_t"),)
    assert diff.added_tables == ()
    assert diff.dropped_tables == ()


def test_diff_detects_fk_change() -> None:
    fk_old = FKEdge(src_table="a", src_cols=["b_id"], dst_table="b", dst_cols=["id"])
    fk_new = FKEdge(src_table="a", src_cols=["b_id"], dst_table="c", dst_cols=["id"])
    cols = {"b_id": _mk_col("b_id")}
    old = SchemaGraph(tables={"a": _mk_table("a", cols, fks=[fk_old])}, join_paths_multi={})
    new = SchemaGraph(
        tables={"a": _mk_table("a", copy.deepcopy(cols), fks=[fk_new])},
        join_paths_multi={},
    )
    diff = diff_schemas(old, new)
    assert diff.per_table["a"].fk_changed is True


def test_diff_empty_when_identical() -> None:
    cols = {"x": _mk_col("x")}
    old = SchemaGraph(tables={"a": _mk_table("a", cols)}, join_paths_multi={})
    new = SchemaGraph(tables={"a": _mk_table("a", copy.deepcopy(cols))}, join_paths_multi={})
    assert diff_schemas(old, new).is_empty


def test_apply_diff_adds_table_and_profiles_only_it() -> None:
    cached = SchemaGraph(tables={"a": _mk_table("a", {"x": _mk_col("x")})}, join_paths_multi={})
    new_struct = SchemaGraph(
        tables={
            "a": _mk_table("a", {"x": _mk_col("x")}),
            "b": _mk_table("b", {"y": _mk_col("y")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-2")
    out = apply_diff(cached, new_struct, diff, dialect)
    assert "b" in out.tables
    assert dialect.profile_schema_calls == [["b"]]


def test_apply_diff_drops_table() -> None:
    cached = SchemaGraph(
        tables={
            "a": _mk_table("a", {"x": _mk_col("x")}),
            "b": _mk_table("b", {"y": _mk_col("y")}),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(tables={"a": _mk_table("a", {"x": _mk_col("x")})}, join_paths_multi={})
    diff = diff_schemas(cached, new_struct)
    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-2")
    out = apply_diff(cached, new_struct, diff, dialect)
    assert set(out.tables) == {"a"}
    assert dialect.profile_schema_calls == []


def test_apply_diff_adds_column_profiles_only_that_table() -> None:
    cached = SchemaGraph(
        tables={
            "a": _mk_table("a", {"x": _mk_col("x")}),
            "other": _mk_table("other", {"k": _mk_col("k")}),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "a": _mk_table("a", {"x": _mk_col("x"), "z": _mk_col("z", "varchar")}),
            "other": _mk_table("other", {"k": _mk_col("k")}),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-2")
    out = apply_diff(cached, new_struct, diff, dialect)
    assert "z" in out.tables["a"].columns
    assert dialect.profile_schema_calls == [["a"]]


def test_apply_diff_renames_table_and_updates_fks() -> None:
    cached = SchemaGraph(
        tables={
            "old_t": _mk_table("old_t", {"id": _mk_col("id")}, pk=["id"]),
            "child": _mk_table(
                "child",
                {"old_t_id": _mk_col("old_t_id")},
                fks=[
                    FKEdge(
                        src_table="child",
                        src_cols=["old_t_id"],
                        dst_table="old_t",
                        dst_cols=["id"],
                    )
                ],
            ),
        },
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={
            "new_t": _mk_table("new_t", {"id": _mk_col("id")}, pk=["id"]),
            "child": _mk_table(
                "child",
                {"old_t_id": _mk_col("old_t_id")},
                fks=[
                    FKEdge(
                        src_table="child",
                        src_cols=["old_t_id"],
                        dst_table="new_t",
                        dst_cols=["id"],
                    )
                ],
            ),
        },
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    assert diff.table_renames == (("old_t", "new_t"),)
    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-2")
    out = apply_diff(cached, new_struct, diff, dialect)
    assert set(out.tables) == {"new_t", "child"}
    assert out.tables["new_t"].name == "new_t"
    assert out.tables["child"].foreign_keys[0].dst_table == "new_t"


def test_apply_diff_updates_redeclared_column_without_full_reprofile_flag() -> None:
    cached = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "integer")})},
        join_paths_multi={},
    )
    new_struct = SchemaGraph(
        tables={"a": _mk_table("a", {"x": _mk_col("x", "bigint")})},
        join_paths_multi={},
    )
    diff = diff_schemas(cached, new_struct)
    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-2")
    out = apply_diff(cached, new_struct, diff, dialect)
    assert out.tables["a"].columns["x"].data_type == "bigint"
    assert out.tables["a"].columns["x"].value_type != ""


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


def test_build_schema_graph_partial_rebuild_on_probe_mismatch(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe-mismatch → reflect_only + apply_diff path; full reflect/profile NOT called."""
    ctx = EngineContext()
    _save_with_probe(schema_graph, ctx, "notes", probe="probe-OLD", path=cache_path)

    new_struct = copy.deepcopy(schema_graph)
    new_struct.tables["customers"].columns["new_col"] = _mk_col("new_col", "varchar")

    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-NEW")

    out = build_schema_graph(dialect, ctx, notes_content="notes")

    assert dialect.reflect_only_calls == 1
    assert dialect.profile_schema_calls == [["customers"]]
    assert "new_col" in out.tables["customers"].columns
    raw = read_gzip_json(cache_path)
    assert raw["ddl_probe_hash"] != "probe-OLD"
    assert raw["ddl_probe_hash"] != ""
    assert "new_col" in raw["tables"]["customers"]["columns"]


def test_build_schema_graph_partial_rebuild_dropped_table(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    ctx = EngineContext()
    _save_with_probe(schema_graph, ctx, "n", probe="probe-OLD", path=cache_path)

    new_struct = copy.deepcopy(schema_graph)
    new_struct.tables.pop("products", None)
    if "orders" in new_struct.tables:
        new_struct.tables["orders"].foreign_keys = [
            fk for fk in new_struct.tables["orders"].foreign_keys if fk.dst_table != "products"
        ]

    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-NEW")
    out = build_schema_graph(dialect, ctx, notes_content="n")
    assert "products" not in out.tables
    profiled = {t for call in dialect.profile_schema_calls for t in call}
    assert "products" not in profiled
