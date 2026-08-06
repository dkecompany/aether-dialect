"""Validate logical CTE tables lists and implied DAG edges."""

from __future__ import annotations

from aetherdialect._contracts_base import CteIntent, LogicalIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata


def _graph_with_tbl_a() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "tbl_a": TableMetadata(
                name="tbl_a",
                columns={"pk": ColumnMetadata(name="pk", data_type="integer")},
                primary_key=["pk"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )


def test_prior_cte_name_in_tables_is_valid() -> None:
    logical = LogicalIntent(
        tables=("tbl_a",),
        select="x from step_b",
        cte_steps=(
            CteIntent(name="step_a", tables=("tbl_a",), select="pk"),
            CteIntent(name="step_b", tables=("tbl_a", "step_a"), select="metric from step_a"),
        ),
    )
    issues = _graph_with_tbl_a().validate_cte_tables_and_dag(logical)
    assert not issues


def test_forward_cte_reference_rejected() -> None:
    logical = LogicalIntent(
        tables=("tbl_a",),
        select="x",
        cte_steps=(
            CteIntent(name="step_a", tables=("tbl_a", "step_b"), select="pk"),
            CteIntent(name="step_b", tables=("tbl_a",), select="metric"),
        ),
    )
    issues = _graph_with_tbl_a().validate_cte_tables_and_dag(logical)
    assert any(i.issue_id.startswith("cte_unknown_table") and "step_b" in i.issue_id for i in issues)


def test_unknown_table_token_rejected() -> None:
    logical = LogicalIntent(
        tables=("tbl_a",),
        select="x",
        cte_steps=(CteIntent(name="step_a", tables=("tbl_a", "missing"), select="pk"),),
    )
    issues = _graph_with_tbl_a().validate_cte_tables_and_dag(logical)
    assert any(i.issue_id.startswith("cte_unknown_table") and "missing" in i.issue_id for i in issues)
