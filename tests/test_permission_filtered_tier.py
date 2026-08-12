"""Tests for permission-filtered migration tier and no-op migration policy."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_base import MigrationTier
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._schema_graph import consumer_graph_is_permission_subset
from aetherdialect._templates_ops import TemplateOps


def _table(name: str) -> TableMetadata:
    col = ColumnMetadata(name="id", data_type="integer")
    return TableMetadata(
        name=name,
        columns={"id": col},
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )


def _owner_consumer_pair() -> tuple[SchemaGraph, SchemaGraph]:
    owner = SchemaGraph(
        join_paths_multi={},
        tables={
            "a": _table("a"),
            "b": _table("b"),
        },
        schema_graph_id="sg_perm000000000001__abcd1234",
        effective_structural_hash="owner_eff",
        structural_hash="owner_struct",
    )
    consumer = SchemaGraph(
        join_paths_multi={},
        tables={"a": _table("a")},
        schema_graph_id="sg_perm000000000001__abcd1234",
        effective_structural_hash="consumer_eff",
        structural_hash="consumer_struct",
    )
    return owner, consumer


class TestPermissionFilteredTier:
    def test_consumer_subset_detection(self) -> None:
        owner, consumer = _owner_consumer_pair()
        assert consumer_graph_is_permission_subset(owner, consumer) is True
        consumer.tables["c"] = _table("c")
        assert consumer_graph_is_permission_subset(owner, consumer) is False

    def test_apply_migration_policy_no_op_on_permission_filtered(self, tmp_path) -> None:
        owner, consumer = _owner_consumer_pair()
        with patch(
            "aetherdialect._templates.classify_migration_tier",
            return_value=MigrationTier.PERMISSION_FILTERED,
        ):
            report = TemplateOps.apply_migration_policy(str(tmp_path), consumer)
        assert report.tier == MigrationTier.PERMISSION_FILTERED
        assert report.destroyed_templates == 0
