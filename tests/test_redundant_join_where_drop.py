"""Resolved INNER join edges drop duplicate column-equality filters from WHERE."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED
from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_normalize import drop_redundant_resolved_join_where_predicates
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from tests.join_test_helpers import catalog_edge_kinds_for_signatures
from tests.test_join_kind_preservation import _nullable_child_schema, _parent_child_schema


def _fk_join_intent(
    *,
    preserve_tables: list[str] | None = None,
) -> RuntimeIntent:
    fp = WhereParam(
        left_expr=NormalizedExpr.from_column("child.parent_id"),
        op="=",
        right_expr=NormalizedExpr.from_column("parent.id"),
        value_type="integer",
    )
    return RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([fp]),
        preserve_tables=list(preserve_tables or []),
    )


def _drop_main(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    signature: list[str],
    edge_kinds: list[str],
) -> RuntimeIntent:
    updated, _changed = drop_redundant_resolved_join_where_predicates(
        intent,
        schema,
        join_sigs_ordered=[[], signature],
        edge_kinds_ordered=[[], edge_kinds],
    )
    return updated


@pytest.mark.fast
def test_inner_join_duplicate_filter_is_dropped() -> None:
    schema = _parent_child_schema()
    signature = ["child.parent_id->parent.id"]
    edge_kinds = catalog_edge_kinds_for_signatures([signature])[0]
    intent = _fk_join_intent()
    token = set_diagnostic_collector([])
    try:
        updated, changed = drop_redundant_resolved_join_where_predicates(
            intent,
            schema,
            join_sigs_ordered=[[], signature],
            edge_kinds_ordered=[[], edge_kinds],
        )
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert changed is True
    assert PredicateGroup.where_leaves(updated.where) == []
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED


@pytest.mark.fast
def test_nullable_foreign_key_left_edge_keeps_duplicate_filter() -> None:
    schema = _nullable_child_schema()
    signature = ["child.parent_id->parent.id"]
    edge_kinds = catalog_edge_kinds_for_signatures([signature])[0]
    intent = _fk_join_intent()
    updated = _drop_main(intent, schema, signature, edge_kinds)
    assert len(PredicateGroup.where_leaves(updated.where) or []) == 1


@pytest.mark.fast
def test_preserve_tables_keeps_duplicate_filter() -> None:
    schema = _parent_child_schema()
    signature = ["child.parent_id->parent.id"]
    edge_kinds = catalog_edge_kinds_for_signatures([signature])[0]
    intent = _fk_join_intent(preserve_tables=["parent"])
    updated = _drop_main(intent, schema, signature, edge_kinds)
    assert len(PredicateGroup.where_leaves(updated.where) or []) == 1


@pytest.mark.fast
def test_semantic_profile_edge_keeps_duplicate_filter() -> None:
    orders = TableMetadata(
        name="orders",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
            "status_code": ColumnMetadata(
                name="status_code",
                data_type="varchar",
                sensitivity="none",
                value_overlap_sample=["open", "closed"],
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    statuses = TableMetadata(
        name="statuses",
        columns={
            "code": ColumnMetadata(
                name="code",
                data_type="varchar",
                sensitivity="none",
                is_primary_key=True,
                value_overlap_sample=["open", "closed", "archived"],
            ),
        },
        primary_key=["code"],
        foreign_keys=[],
    )
    schema = SchemaGraph(
        tables={"orders": orders, "statuses": statuses},
        join_paths_multi={},
        effective_structural_hash="h",
    )
    signature = ["orders.status_code->statuses.code"]
    edge_kinds = ["semantic_profile"]
    fp = WhereParam(
        left_expr=NormalizedExpr.from_column("orders.status_code"),
        op="=",
        right_expr=NormalizedExpr.from_column("statuses.code"),
        value_type="string",
    )
    intent = RuntimeIntent(
        tables=["orders", "statuses"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([fp]),
    )
    updated = _drop_main(intent, schema, signature, edge_kinds)
    assert len(PredicateGroup.where_leaves(updated.where) or []) == 1
