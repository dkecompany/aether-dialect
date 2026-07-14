"""Validate logical CTE tables lists and implied DAG edges."""

from __future__ import annotations

from aetherdialect._contracts_base import CteIntent, LogicalIntent
from aetherdialect._contracts_schema import SchemaGraph, validate_cte_tables_and_dag


def _empty_graph() -> SchemaGraph:
    return SchemaGraph(
        tables={},
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
    issues = validate_cte_tables_and_dag(logical, _empty_graph())
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
    issues = validate_cte_tables_and_dag(logical, _empty_graph())
    assert any(i.issue_id.startswith("cte_table_forward_ref") for i in issues)


def test_unknown_table_token_rejected() -> None:
    logical = LogicalIntent(
        tables=("tbl_a",),
        select="x",
        cte_steps=(CteIntent(name="step_a", tables=("missing_tbl",), select="pk"),),
    )
    issues = validate_cte_tables_and_dag(logical, _empty_graph())
    assert any(i.issue_id.startswith("cte_table_unknown") for i in issues)
