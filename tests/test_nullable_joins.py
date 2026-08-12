"""Diagnostics and invariant checks for nullable foreign-key joins."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_JOIN_NULLABLE_KEY
from aetherdialect._contracts_base import SchemaInvariantError
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._sql_gen import _join_edges_from_signature
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


def _col(
    name: str,
    *,
    nullable: bool = False,
    fk_target: tuple[str, str] | None = None,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_nullable=nullable,
        is_foreign_key=fk_target is not None,
        fk_target=fk_target,
    )


def _nullable_child_schema() -> SchemaGraph:
    parent = TableMetadata(
        name="parent",
        columns={"id": _col("id", nullable=False)},
        primary_key=["id"],
        foreign_keys=[],
    )
    child = TableMetadata(
        name="child",
        columns={
            "id": _col("id", nullable=False),
            "parent_id": _col("parent_id", nullable=True, fk_target=("parent", "id")),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"parent": parent, "child": child},
        join_paths_multi={},
        effective_structural_hash="h",
    )


def test_nullable_key_emits_diagnostic() -> None:
    schema = _nullable_child_schema()
    sig = ["parent.id->child.parent_id"]
    token = set_diagnostic_collector([])
    try:
        resolved = _join_edges_from_signature(sig, ["catalog_fk"], "parent", schema)
        assert resolved is not None
        join_edges, _where, _extra, _anti = resolved
        assert join_edges[0].kind == "LEFT"
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert any(d.code == DIAGNOSTIC_CODE_JOIN_NULLABLE_KEY for d in diags)
    diag = next(d for d in diags if d.code == DIAGNOSTIC_CODE_JOIN_NULLABLE_KEY)
    assert "parent" in diag.message and "child" in diag.message
    assert "preserved" in diag.message.lower()


def test_inner_join_on_nullable_key_is_invariant_error() -> None:
    schema = _nullable_child_schema()
    sig = ["parent.id->child.parent_id"]
    with pytest.raises(SchemaInvariantError, match="nullable foreign key"):
        _join_edges_from_signature(
            sig,
            ["catalog_fk"],
            "parent",
            schema,
            cte_emissions={"child": "semi_join"},
        )
