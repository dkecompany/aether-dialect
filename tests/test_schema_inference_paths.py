"""Tests for pair-targeted FK inference, override collapse, and join- path reachability validation."""

from __future__ import annotations

from aetherdialect._contracts_base import OverrideSkip
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    FKEdge,
    InferenceTag,
    SchemaGraph,
    TableMetadata,
    TableRole,
)
from aetherdialect._schema_graph import (
    collapse_redundant_inferences,
    infer_missing_fks,
    pair_targeted_fk_inference,
)
from aetherdialect._validation_shape import validate_join_path_reachability_for_tables


def _col(**overrides) -> ColumnMetadata:
    defaults = dict(
        name="col",
        data_type="varchar",
        value_type="string",
        is_primary_key=False,
        is_foreign_key=False,
        fk_target=None,
        role=ColumnRole.IDENTIFIER.value,
        distinct_count=10,
        distinct_ratio=0.5,
        row_count=20,
        value_overlap_sample=["1", "2", "3", "4", "5"],
    )
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _table(name: str, columns: dict[str, ColumnMetadata], **kwargs) -> TableMetadata:
    base = dict(
        name=name,
        columns=columns,
        primary_key=[],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=100,
    )
    base.update(kwargs)
    return TableMetadata(**base)


class TestPairTargetedInference:
    def test_infer_missing_fks_single_table_restrict_allows_self_candidates(
        self,
    ) -> None:
        """``restrict_tables`` may contain one table for self-FK naming patterns."""
        t = _table(
            "node",
            {
                "node_id": _col(name="node_id", is_primary_key=True),
                "parent_node_id": _col(name="parent_node_id"),
            },
            primary_key=["node_id"],
        )
        edges = infer_missing_fks({"node": t}, restrict_tables=frozenset({"node"}))
        assert len(edges) == 1
        assert edges[0].dst_table == "node"
        assert edges[0].inference_tag == InferenceTag.SELF

    def test_pair_targeted_adds_cross_table_suffix_fk(self) -> None:
        ta = _table(
            "orders",
            {"customer_id": _col(name="customer_id", value_type="string")},
        )
        tb = _table(
            "customer",
            {"customer_id": _col(name="customer_id", is_primary_key=True)},
            primary_key=["customer_id"],
        )
        sg = SchemaGraph(tables={"orders": ta, "customer": tb}, join_paths_multi={})
        n = pair_targeted_fk_inference(sg, blocked=frozenset())
        assert n >= 1
        fk_targets = [e.dst_table for e in ta.foreign_keys]
        assert "customer" in fk_targets


class TestCollapseRedundantInferences:
    def test_collapse_drops_inferred_fk_when_user_truth_connects(self) -> None:
        ta = _table(
            "a",
            {
                "id": _col(name="id", is_primary_key=True),
                "b_id": _col(name="b_id", is_foreign_key=True, value_type="string"),
            },
            primary_key=["id"],
        )
        tb = _table(
            "b",
            {"id": _col(name="id", is_primary_key=True)},
            primary_key=["id"],
        )
        inferred = FKEdge(
            src_table="a",
            src_cols=["b_id"],
            dst_table="b",
            dst_cols=["id"],
            inference_tag=InferenceTag.SUFFIX,
        )
        user_edge = FKEdge(
            src_table="a",
            src_cols=["b_id"],
            dst_table="b",
            dst_cols=["id"],
            inference_tag=InferenceTag.USER_STRUCTURAL,
        )
        ta.foreign_keys = [inferred, user_edge]
        sg = SchemaGraph(tables={"a": ta, "b": tb}, join_paths_multi={})
        skipped: list[OverrideSkip] = []
        removed = collapse_redundant_inferences(sg, skipped)
        assert removed >= 1
        tags = [e.inference_tag for e in ta.foreign_keys]
        assert InferenceTag.SUFFIX not in tags
        assert any(s.reason == "superseded_by_user_fk" for s in skipped)


class TestJoinPathReachability:
    def test_reports_issue_when_no_join_path(self) -> None:
        o = _table("orders", {"id": _col(name="id", is_primary_key=True)}, primary_key=["id"])
        c = _table(
            "customers",
            {"id": _col(name="id", is_primary_key=True)},
            primary_key=["id"],
        )
        sg = SchemaGraph(
            tables={"orders": o, "customers": c},
            join_paths_multi={"orders": {"customers": []}, "customers": {"orders": []}},
        )
        issues = validate_join_path_reachability_for_tables(["orders", "customers"], sg, "main query")
        assert issues
