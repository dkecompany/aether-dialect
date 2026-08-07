"""Federation composite statistics must not invent cardinality from summed or concatenated members."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _cross_source_column_suggestion_score,
    _merge_column_metadata_union_statistics,
    compose_composite_graph,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _member_graph(
    table: str,
    source_id: str,
    *,
    row_count: int,
    distinct_count: int,
    overlap_sample: tuple[str, ...],
    distinct_ratio: float | None = None,
    join_key: str = "id",
) -> SchemaGraph:
    ratio = distinct_ratio if distinct_ratio is not None else (distinct_count / row_count if row_count > 0 else 0.0)
    col = ColumnMetadata(
        name=join_key,
        data_type="integer",
        value_type="integer",
        sensitivity="none",
        is_primary_key=join_key == "id",
        row_count=row_count,
        distinct_count=distinct_count,
        distinct_ratio=ratio,
        value_overlap_sample=list(overlap_sample),
    )
    table_meta = TableMetadata(
        name=table,
        columns={join_key: col},
        primary_key=[join_key],
        foreign_keys=[],
        source_id=source_id,
        row_count=row_count,
    )
    tables = {table: table_meta}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
        profiling_hash=f"profile_{source_id}",
    )


def _replica_members_and_mappings(
    *,
    auth_overlap: tuple[str, ...] = ("1", "2", "3"),
    other_overlap: tuple[str, ...] = ("1", "2", "3"),
    auth_distinct: int = 100,
    other_distinct: int = 100,
    row_count: int = 100,
) -> tuple[dict[str, SchemaGraph], object, object]:
    members = {
        "a": _member_graph(
            "entity_a",
            "a",
            row_count=row_count,
            distinct_count=auth_distinct,
            overlap_sample=auth_overlap,
        ),
        "b": _member_graph(
            "entity_b",
            "b",
            row_count=row_count,
            distinct_count=other_distinct,
            overlap_sample=other_overlap,
        ),
    }
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_replica_stats",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "a", "entity_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "entity",
                    "semantics": "replica",
                    "authoritative_source": "a",
                    "members": [
                        {"source": "a", "table": "entity_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "entity_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        }
    )
    return members, manifest, mappings


def _union_members_and_mappings(
    *,
    a_overlap: tuple[str, ...] = ("1", "2"),
    b_overlap: tuple[str, ...] = ("3", "4"),
    a_distinct: int = 2,
    b_distinct: int = 2,
) -> tuple[dict[str, SchemaGraph], object, object]:
    members = {
        "a": _member_graph(
            "payment_a",
            "a",
            row_count=a_distinct,
            distinct_count=a_distinct,
            overlap_sample=a_overlap,
        ),
        "b": _member_graph(
            "payment_b",
            "b",
            row_count=b_distinct,
            distinct_count=b_distinct,
            overlap_sample=b_overlap,
        ),
    }
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_union_stats",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"payment_a": "a", "payment_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        }
    )
    return members, manifest, mappings


@pytest.mark.fast
def test_replica_composite_preserves_authoritative_distinct_count_and_ratio() -> None:
    members, manifest, mappings = _replica_members_and_mappings()
    composite = compose_composite_graph(members, manifest, mappings)
    col = composite.tables["entity"].columns["id"]
    assert col.distinct_count == 100
    assert col.distinct_ratio == 1.0


@pytest.mark.fast
def test_replica_composite_uses_authoritative_overlap_sample_not_member_union() -> None:
    members, manifest, mappings = _replica_members_and_mappings(
        auth_overlap=("1", "2", "3"),
        other_overlap=("3", "4", "5"),
    )
    composite = compose_composite_graph(members, manifest, mappings)
    col = composite.tables["entity"].columns["id"]
    assert col.value_overlap_sample == ["1", "2", "3"]


@pytest.mark.fast
def test_replica_composite_concatenated_overlap_inflates_cross_source_score() -> None:
    members, manifest, mappings = _replica_members_and_mappings(
        auth_overlap=("1", "2", "3"),
        other_overlap=("3", "4", "5"),
    )
    composite = compose_composite_graph(members, manifest, mappings)
    composite_col = composite.tables["entity"].columns["id"]
    other_col = ColumnMetadata(
        name="ref_id",
        data_type="integer",
        value_type="integer",
        sensitivity="none",
        value_overlap_sample=["4", "5", "6"],
    )
    inflated_score = _cross_source_column_suggestion_score(composite_col, other_col, "id", "ref_id")
    authoritative_col = members["a"].tables["entity_a"].columns["id"]
    expected_score = _cross_source_column_suggestion_score(authoritative_col, other_col, "id", "ref_id")
    assert inflated_score == expected_score
    assert inflated_score < 0.7583333333333333


@pytest.mark.fast
def test_union_composite_marks_multi_member_distinct_unknown_not_summed() -> None:
    members, manifest, mappings = _union_members_and_mappings(a_distinct=10, b_distinct=7)
    composite = compose_composite_graph(members, manifest, mappings)
    col = composite.tables["payment"].columns["id"]
    assert col.distinct_count == 0
    assert col.distinct_count != 17


@pytest.mark.fast
def test_union_composite_does_not_concatenate_disjoint_member_overlap_samples() -> None:
    members, manifest, mappings = _union_members_and_mappings()
    composite = compose_composite_graph(members, manifest, mappings)
    col = composite.tables["payment"].columns["id"]
    assert col.value_overlap_sample == []


@pytest.mark.fast
def test_merge_union_statistics_does_not_concatenate_member_overlap_samples() -> None:
    left = ColumnMetadata(
        name="id",
        data_type="integer",
        distinct_count=50,
        distinct_ratio=0.5,
        value_overlap_sample=["1", "2"],
    )
    right = ColumnMetadata(
        name="id",
        data_type="integer",
        distinct_count=50,
        distinct_ratio=0.5,
        value_overlap_sample=["3", "4"],
    )
    merged = _merge_column_metadata_union_statistics([left, right], composite_semantics="union")
    assert merged.distinct_count == 0
    assert merged.distinct_ratio == 0.0
    assert merged.value_overlap_sample == []
