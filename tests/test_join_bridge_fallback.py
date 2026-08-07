"""Bridge-table join enumeration falls back only when the strict pass finds nothing."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import SchemaGraph, TableMetadata
from aetherdialect._sql_gen import _candidate_join_paths_for_tables, _join_path_signature_for_path


def _edge(src: str, dst: str, src_col: str = "id", dst_col: str = "id") -> dict[str, object]:
    return {
        "src_table": src,
        "src_cols": [src_col],
        "dst_table": dst,
        "dst_cols": [dst_col],
    }


@pytest.mark.fast
def test_two_endpoint_scope_resolves_through_unreferenced_junction() -> None:
    bridge_path = [_edge("a", "bridge", "id", "aid"), _edge("bridge", "c", "cid", "id")]
    schema = SchemaGraph(
        tables={
            "a": TableMetadata(name="a", columns={}, primary_key=[], foreign_keys=[]),
            "c": TableMetadata(name="c", columns={}, primary_key=[], foreign_keys=[]),
            "bridge": TableMetadata(name="bridge", columns={}, primary_key=[], foreign_keys=[]),
        },
        join_paths_multi={"a": {"c": [bridge_path]}},
        effective_structural_hash="bridge-fallback",
    )
    result = _candidate_join_paths_for_tables(schema, ["a", "c"])
    assert result
    assert any("bridge" in ".".join(_join_path_signature_for_path(path)) for path in result)


@pytest.mark.fast
def test_strict_pass_prefers_bridge_free_path_when_available() -> None:
    direct = [_edge("a", "c", "id", "aid")]
    via_bridge = [_edge("a", "bridge", "id", "aid"), _edge("bridge", "c", "cid", "id")]
    schema = SchemaGraph(
        tables={
            "a": TableMetadata(name="a", columns={}, primary_key=[], foreign_keys=[]),
            "c": TableMetadata(name="c", columns={}, primary_key=[], foreign_keys=[]),
            "bridge": TableMetadata(name="bridge", columns={}, primary_key=[], foreign_keys=[]),
        },
        join_paths_multi={"a": {"c": [via_bridge, direct]}},
        effective_structural_hash="bridge-fallback",
    )
    result = _candidate_join_paths_for_tables(schema, ["a", "c"])
    assert result
    for path in result:
        tables_in_path = {edge["src_table"] for edge in path} | {edge["dst_table"] for edge in path}
        assert "bridge" not in tables_in_path
        assert len(path) == 1
