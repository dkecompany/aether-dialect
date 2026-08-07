"""Join-path DFS stops at refusal ceiling during collection."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, TableMetadata
from aetherdialect._schema_graph import join_path_pair_tie_count, recompute_join_paths_multi


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


@pytest.mark.fast
def test_dfs_stops_at_ceiling() -> None:
    tie_cap = 3
    mid_count = 20
    tables = _parallel_mid_tables(mid_count)
    join_paths_multi = recompute_join_paths_multi(tables, tie_cap=tie_cap)
    assert join_path_pair_tie_count(join_paths_multi, "src", "dst") == tie_cap + 1
    raw = join_paths_multi["src"]["dst"]
    assert len(raw) == 1
    assert len(raw[0]) == 1
    assert isinstance(raw[0][0], dict)
