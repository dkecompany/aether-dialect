"""Tests for collation-aware value-overlap comparison."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import parse_federation_mappings, rescore_declared_mapping_drift
from aetherdialect._schema_catalog import compute_semantic_profile_join_neighbors, value_overlap_stats_for_columns
from aetherdialect._schema_graph import _column_profiling_dict, fk_overlap_validates


def _col(
    name: str,
    sample: list[str],
    *,
    pk: bool = False,
    is_case_insensitive_collation: bool = False,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="varchar",
        value_type="string",
        is_primary_key=pk,
        value_overlap_sample=list(sample),
        is_case_insensitive_collation=is_case_insensitive_collation,
    )


_CASE_SAMPLES_LOWER = ["alpha", "bravo", "charlie", "delta", "echo"]
_CASE_SAMPLES_UPPER = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]


@pytest.mark.fast
def test_case_insensitive_member_matches_case_sensitive_member() -> None:
    ci_col = _col("code", _CASE_SAMPLES_LOWER, is_case_insensitive_collation=True)
    cs_col = _col("code", _CASE_SAMPLES_UPPER, pk=True)

    assert fk_overlap_validates(ci_col, cs_col) is True
    assert ci_col.overlap_comparison == "case_folded"
    assert cs_col.overlap_comparison == "case_folded"

    stats = value_overlap_stats_for_columns(ci_col, cs_col)
    assert stats is not None
    inter, ratio = stats
    assert inter == 5
    assert ratio == 1.0

    ci_table = TableMetadata(
        name="entity_ci",
        columns={"code": _col("code", _CASE_SAMPLES_LOWER, is_case_insensitive_collation=True)},
        primary_key=[],
        foreign_keys=[],
        source_id="mysql_member",
    )
    cs_table = TableMetadata(
        name="entity_cs",
        columns={"code": _col("code", _CASE_SAMPLES_UPPER, pk=True)},
        primary_key=["code"],
        foreign_keys=[],
        source_id="pg_member",
    )
    members = {
        "mysql_member": SchemaGraph(
            tables={"entity_ci": ci_table},
            join_paths_multi={},
            schema_graph_id="sg_mysql",
            effective_structural_hash="eff_mysql",
            profiling_hash="pr_mysql",
        ),
        "pg_member": SchemaGraph(
            tables={"entity_cs": cs_table},
            join_paths_multi={},
            schema_graph_id="sg_pg",
            effective_structural_hash="eff_pg",
            profiling_hash="pr_pg",
        ),
    }
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_columns": [
                {
                    "logical": "shared_code",
                    "unify_in_graph": True,
                    "members": ["entity_ci.code", "entity_cs.code"],
                },
            ],
            "logical_tables": [
                {
                    "logical": "entity",
                    "semantics": "union",
                    "members": [
                        {"source": "mysql_member", "table": "entity_ci", "columns": {"code": "code"}},
                        {"source": "pg_member", "table": "entity_cs", "columns": {"code": "code"}},
                    ],
                },
            ],
        },
    )
    drift = rescore_declared_mapping_drift(mappings, members)
    assert not any("value overlap rescoring drift" in msg for msg in drift)

    semantic_sg = SchemaGraph(
        tables={
            "statuses": TableMetadata(
                name="statuses",
                columns={
                    "code": _col("code", _CASE_SAMPLES_UPPER, pk=True),
                },
                primary_key=["code"],
                foreign_keys=[],
            ),
            "orders": TableMetadata(
                name="orders",
                columns={
                    "status_code": _col(
                        "status_code",
                        _CASE_SAMPLES_LOWER,
                        is_case_insensitive_collation=True,
                    ),
                },
                primary_key=[],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        created_at="",
    )
    semantic_sg.tables["statuses"].columns["code"].distinct_count = 10
    semantic_sg.tables["orders"].columns["status_code"].distinct_count = 10
    compute_semantic_profile_join_neighbors(semantic_sg)
    assert semantic_sg.tables["orders"].columns["status_code"].semantic_join_neighbors == [("statuses", "code")]
    assert semantic_sg.tables["orders"].columns["status_code"].overlap_comparison == "case_folded"

    profiling = _column_profiling_dict(ci_col)
    assert profiling["overlap_comparison"] == "case_folded"
