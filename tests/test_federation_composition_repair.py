"""Tests for federation composition repair after member graph drift."""

from __future__ import annotations

import copy

import pytest

from aetherdialect._contracts_base import EngineContext, SensitivityClassification
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationConfigError,
    compose_composite_graph,
    manifest_hash,
    mappings_hash,
    parse_federation_manifest,
    parse_federation_mappings,
    plan_federated_intent,
    reconcile_composite_classifications,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import assign_schema_graph_hashes, recompute_join_paths_multi
from tests.federation_helpers import stamp_union_disjointness_profiling


def _table(
    name: str,
    *,
    source_id: str = "",
    columns: dict[str, ColumnMetadata] | None = None,
    foreign_keys: list[FKEdge] | None = None,
    row_count: int = 0,
    description: str = "",
    partition_columns: list[str] | None = None,
    deny_columns: dict[str, set[str]] | None = None,
    created_at: str = "",
) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns=columns
        or {
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=foreign_keys or [],
        source_id=source_id,
        row_count=row_count,
        description=description,
        partition_columns=partition_columns or [],
    )


def _graph(
    table: str,
    *,
    source_id: str = "",
    created_at: str = "2024-01-01T00:00:00",
    deny_columns: dict[str, set[str]] | None = None,
    tables: dict[str, TableMetadata] | None = None,
) -> SchemaGraph:
    table_map = tables or {_table(table, source_id=source_id).name: _table(table, source_id=source_id)}
    return SchemaGraph(
        tables=table_map,
        join_paths_multi=recompute_join_paths_multi(table_map),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
        created_at=created_at,
        deny_columns=deny_columns or {},
    )


_MANIFEST = {
    "federation_id": "fed_repair",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {
        "customers": "a",
        "orders": "b",
        "payment_a": "a",
        "payment_b": "b",
    },
    "cross_source_joins": [
        {"left": "orders.customer_id", "right": "customers.id", "kind": "inner", "logical_key": "id"},
    ],
}

_PAYMENT_ONLY_MANIFEST = {
    "federation_id": "fed_payment_only",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {
        "payment_a": "a",
        "payment_b": "b",
    },
    "cross_source_joins": [],
}


def _payment_members(
    *,
    a_table: TableMetadata | None = None,
    b_table: TableMetadata | None = None,
) -> dict[str, SchemaGraph]:
    a_tbl = a_table or _table("payment", source_id="a")
    b_tbl = b_table or _table("payment", source_id="b")
    stamp_union_disjointness_profiling(a_tbl, overlap_sample=("a1", "a2"))
    stamp_union_disjointness_profiling(b_tbl, overlap_sample=("b1", "b2"))
    return {
        "a": _graph("payment", source_id="a", tables={"payment": a_tbl}),
        "b": _graph("payment", source_id="b", tables={"payment": b_tbl}),
    }


def _orders_table() -> TableMetadata:
    return _table(
        "orders",
        source_id="b",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
        },
    )


def test_collapse_remaps_foreign_keys() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph(
            "customers",
            source_id="a",
            tables={
                "customers": _table(
                    "customers",
                    source_id="a",
                    foreign_keys=[],
                ),
                "payment": _table(
                    "payment",
                    source_id="a",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                        "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
                    },
                    foreign_keys=[
                        FKEdge(
                            src_table="payment",
                            src_cols=["customer_id"],
                            dst_table="customers",
                            dst_cols=["id"],
                        ),
                    ],
                ),
            },
        ),
        "b": _graph(
            "orders",
            source_id="b",
            tables={
                "orders": _orders_table(),
                "payment": _table("payment", source_id="b"),
            },
        ),
    }
    stamp_union_disjointness_profiling(members["a"].tables["payment"], overlap_sample=("a1", "a2"))
    stamp_union_disjointness_profiling(members["b"].tables["payment"], overlap_sample=("b1", "b2"))
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id", "customer_id": "customer_id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    payment = composite.tables["payment"]
    assert payment.foreign_keys
    assert payment.foreign_keys[0].dst_table == "customers"
    assert "payment_a" not in composite.tables
    assert "payment_b" not in composite.tables
    assert payment.member_source_ids == ["a", "b"]


def test_collapse_sums_row_count_for_union() -> None:
    manifest = parse_federation_manifest(_PAYMENT_ONLY_MANIFEST, include_derived_roster=True)
    members = _payment_members(
        a_table=_table("payment", source_id="a", row_count=10),
        b_table=_table("payment", source_id="b", row_count=7),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    assert composite.tables["payment"].row_count == 17


def test_collapse_rejects_partition_mismatch() -> None:
    manifest = parse_federation_manifest(_PAYMENT_ONLY_MANIFEST, include_derived_roster=True)
    members = _payment_members(
        a_table=_table("payment", source_id="a", partition_columns=["dt"]),
        b_table=_table("payment", source_id="b", partition_columns=["month"]),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    with pytest.raises(FederationConfigError, match="partition"):
        compose_composite_graph(members, manifest, mappings)


def test_compose_merges_member_deny_columns() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a", deny_columns={"customers": {"ssn"}}),
        "b": _graph("orders", source_id="b", tables={"orders": _orders_table()}),
    }
    composite = compose_composite_graph(members, manifest)
    assert "ssn" in composite.deny_columns.get("customers", set())


def test_collapse_uses_strictest_sensitivity() -> None:
    manifest = parse_federation_manifest(_PAYMENT_ONLY_MANIFEST, include_derived_roster=True)
    members = _payment_members(
        a_table=_table(
            "payment",
            source_id="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "note": ColumnMetadata(name="note", data_type="text", sensitivity="none"),
            },
        ),
        b_table=_table(
            "payment",
            source_id="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "note": ColumnMetadata(name="note", data_type="text", sensitivity="hidden"),
            },
        ),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {
                            "source": "a",
                            "table": "payment",
                            "columns": {"id": "id", "note": "note"},
                        },
                        {
                            "source": "b",
                            "table": "payment",
                            "columns": {"id": "id", "note": "note"},
                        },
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    assert composite.tables["payment"].columns["note"].sensitivity == SensitivityClassification.HIDDEN


def test_reconcile_composite_classifications_agrees_deterministically() -> None:
    manifest = parse_federation_manifest(_PAYMENT_ONLY_MANIFEST, include_derived_roster=True)
    members = _payment_members(
        a_table=_table("payment", source_id="a", description="Payments"),
        b_table=_table("payment", source_id="b", description="Payments"),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    assert composite.tables["payment"].description == "Payments"
    changed = reconcile_composite_classifications(composite, members, mappings)
    assert changed is False


def test_classification_conflict_reconciles_without_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = parse_federation_manifest(_PAYMENT_ONLY_MANIFEST, include_derived_roster=True)
    members = _payment_members(
        a_table=_table("payment", source_id="a", description="Member A ledger"),
        b_table=_table("payment", source_id="b", description="Member B ledger"),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    calls: list[str | None] = []

    def _fake_classify(_schema: SchemaGraph, notes: str | None) -> dict:
        calls.append(notes)
        return {"payment": (None, "Unified payments", {})}

    monkeypatch.setattr("aetherdialect._federation.llm_classify_schema", _fake_classify)
    composite = compose_composite_graph(members, manifest, mappings)
    assert calls
    assert composite.tables["payment"].description == "Unified payments"


def test_replica_requires_authoritative_source() -> None:
    with pytest.raises(FederationConfigError, match="authoritative_source"):
        parse_federation_mappings(
            {
                "version": "0.2.1",
                "logical_tables": [
                    {
                        "logical": "payment",
                        "semantics": "replica",
                        "members": [
                            {"source": "a", "table": "payment", "columns": {"id": "id"}},
                            {"source": "b", "table": "payment", "columns": {"id": "id"}},
                        ],
                    },
                ],
            },
        )


def test_replica_plan_uses_authoritative_member_only() -> None:
    from aetherdialect._federation import _union_specs_for_intent

    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "replica",
                    "authoritative_source": "b",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    specs = _union_specs_for_intent({"payment"}, mappings, {})
    assert len(specs) == 1
    assert specs[0].member_source_ids == ("b",)


def test_unresolved_mapping_member_raises() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("orders", source_id="b"),
    }
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "missing", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    with pytest.raises(FederationConfigError, match="unresolved"):
        compose_composite_graph(members, manifest, mappings)


def test_composition_created_at_is_deterministic() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a", created_at="2024-01-01T00:00:00"),
        "b": _graph("orders", source_id="b", created_at="2024-02-01T00:00:00", tables={"orders": _orders_table()}),
    }
    first = compose_composite_graph(copy.deepcopy(members), manifest)
    second = compose_composite_graph(copy.deepcopy(members), manifest)
    assert first.created_at == "2024-02-01T00:00:00"
    assert first.created_at == second.created_at
    assert first.schema_graph_id == second.schema_graph_id


def test_mappings_hash_is_order_independent() -> None:
    mapping_a = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    mapping_b = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    assert mappings_hash(mapping_a) == mappings_hash(mapping_b)


def test_composite_scope_hash_survives_assign_schema_graph_hashes() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("orders", source_id="b", tables={"orders": _orders_table()}),
    }
    composite = compose_composite_graph(members, manifest)
    expected_scope = manifest_hash(manifest)
    assert composite.scope_hash == expected_scope
    assign_schema_graph_hashes(
        composite,
        EngineContext(deny_columns=frozenset({"customers.ssn"})),
        "",
        federation_scope_hash=composite.scope_hash,
    )
    assert composite.scope_hash == expected_scope


def test_different_federation_ids_mint_different_composite_ids() -> None:
    members = {
        "a": _graph("customers", source_id="a"),
        "b": _graph("orders", source_id="b", tables={"orders": _orders_table()}),
    }
    manifest_one = parse_federation_manifest(
        {**_MANIFEST, "federation_id": "fed_one"},
        include_derived_roster=True,
    )
    manifest_two = parse_federation_manifest(
        {**_MANIFEST, "federation_id": "fed_two"},
        include_derived_roster=True,
    )
    one = compose_composite_graph(members, manifest_one)
    two = compose_composite_graph(members, manifest_two)
    assert one.schema_graph_id != two.schema_graph_id


def test_column_coverage_marks_ineligible_plan() -> None:
    manifest = parse_federation_manifest(_PAYMENT_ONLY_MANIFEST, include_derived_roster=True)
    members = _payment_members(
        a_table=_table(
            "payment",
            source_id="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "only_a": ColumnMetadata(name="only_a", data_type="text", sensitivity="none"),
            },
        ),
        b_table=_table(
            "payment",
            source_id="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "only_b": ColumnMetadata(name="only_b", data_type="text", sensitivity="none"),
            },
        ),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {
                            "source": "a",
                            "table": "payment",
                            "columns": {"id": "id", "only_a": "only_a"},
                        },
                        {
                            "source": "b",
                            "table": "payment",
                            "columns": {"id": "id", "only_b": "only_b"},
                        },
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    intent = RuntimeIntent(
        tables=["payment"],
        grain="row",
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("payment.only_a")),
            SelectCol(expr=NormalizedExpr.from_column("payment.only_b")),
        ],
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason == ("union logical column 'only_a' on 'payment' is not present on members: b")
