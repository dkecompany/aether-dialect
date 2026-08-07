"""Schema cache must invalidate stale join_paths_multi when enumeration policy changes."""

from __future__ import annotations

import copy
import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._core_utils import read_gzip_json
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import assign_schema_graph_hashes, compute_dialect_probe, recompute_join_paths_multi
from aetherdialect._schema_overrides import build_schema_graph, save_schema_to_cache


class _ProbeStubDialect(Dialect):
    name = "stub"

    def __init__(self, probe_value: str = "DIALECT_DIGEST") -> None:
        super().__init__(MagicMock())
        self._probe_value = probe_value
        self.reflect_calls = 0
        self.profile_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return self._probe_value

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        self.reflect_calls += 1
        raise AssertionError("reflect_schema_graph should not be called")

    def profile_schema(self, sg: SchemaGraph) -> None:
        self.profile_calls += 1

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _parallel_mid_tables(mid_count: int) -> dict[str, TableMetadata]:
    tables: dict[str, TableMetadata] = {
        "src": TableMetadata(name="src", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        "dst": TableMetadata(name="dst", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
    }
    for index in range(mid_count):
        mid_name = f"mid{index}"
        tables[mid_name] = TableMetadata(
            name=mid_name,
            columns={"id": _col("id"), "src_id": _col("src_id"), "dst_id": _col("dst_id")},
            primary_key=["id"],
            foreign_keys=[
                FKEdge(src_table=mid_name, src_cols=["src_id"], dst_table="src", dst_cols=["id"]),
                FKEdge(src_table=mid_name, src_cols=["dst_id"], dst_table="dst", dst_cols=["id"]),
            ],
        )
    return tables


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


@pytest.mark.fast
def test_cached_join_paths_multi_stale_enumeration_version_recomputes(cache_path: str) -> None:
    tables = _parallel_mid_tables(5)
    full_paths = recompute_join_paths_multi(tables)["src"]["dst"]
    assert len(full_paths) == 5
    assert len(full_paths) > PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP

    stale_paths = copy.deepcopy(recompute_join_paths_multi(tables))
    stale_paths["src"]["dst"] = stale_paths["src"]["dst"][: PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP]

    sg = SchemaGraph(
        tables=tables,
        join_paths_multi=stale_paths,
        effective_structural_hash="enum_cache",
    )
    ctx = EngineContext()
    notes_content = "notes"
    sg.notes_sha256 = hashlib.sha256(notes_content.encode("utf-8")).hexdigest()
    assign_schema_graph_hashes(sg, ctx, sg.notes_sha256)
    dialect = _ProbeStubDialect()
    sg.ddl_probe_hash = compute_dialect_probe(dialect, ctx)
    save_schema_to_cache(sg, cache_path)

    raw = read_gzip_json(cache_path)
    assert len(raw["join_paths_multi"]["src"]["dst"]) == PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP
    raw.pop("join_path_enumeration_version", None)
    from aetherdialect._core_utils import write_gzip_json_atomic

    write_gzip_json_atomic(cache_path, raw, sort_keys=True)

    out = build_schema_graph(dialect, ctx, notes_content=notes_content)

    assert dialect.reflect_calls == 0
    assert dialect.profile_calls == 0
    loaded_paths = out.join_paths_multi["src"]["dst"]
    assert len(loaded_paths) == 5
    assert len(loaded_paths) > len(stale_paths["src"]["dst"])
