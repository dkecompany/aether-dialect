"""Federation plan guards for coordinator fan-out, key uniqueness, union overlap, and PK agreement."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationDeclarationError, FederationJoinFanOutError
from aetherdialect._contracts_core import (
    FederatedPlan,
    JoinSpec,
    NormalizedExpr,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationConfigError,
    compose_composite_graph,
    parse_federation_manifest,
    parse_federation_mappings,
    plan_federated_intent,
    validate_coordinator_join_fan_out,
    validate_cross_source_keys_on_graph,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(
    table: str,
    *,
    source_id: str,
    row_count: int = 2,
    id_unique: bool = True,
    id_distinct: int | None = None,
    overlap_sample: tuple[str, ...] = (),
    primary_key: list[str] | None = None,
) -> SchemaGraph:
    distinct = id_distinct if id_distinct is not None else (row_count if id_unique else max(1, row_count // 2))
    pk = primary_key if primary_key is not None else (["id"] if id_unique else [])
    tables = {
        table: TableMetadata(
            name=table,
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    sensitivity="none",
                    is_primary_key="id" in pk,
                    is_unique=id_unique,
                    row_count=row_count,
                    distinct_count=distinct,
                    value_overlap_sample=list(overlap_sample),
                ),
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="integer",
                    value_type="integer",
                    sensitivity="none",
                ),
            },
            primary_key=pk,
            foreign_keys=[],
            source_id=source_id,
            row_count=row_count,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


def _two_member_join_manifest(*, kind: str = "inner") -> tuple[object, object]:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_join",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [
                {"left": "t_a.id", "right": "t_b.id", "kind": kind, "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings({"version": "0.2.1", "logical_columns": []})
    return manifest, mappings


def _three_member_star_manifest() -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_star",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
                {"source_id": "c", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_c": "c"},
            "cross_source_joins": [
                {"left": "t_b.id", "right": "t_a.id", "kind": "inner", "logical_key": "id"},
                {"left": "t_c.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )


# --- Coordinator combine fan-out ---


@pytest.mark.fast
def test_coordinator_left_join_fan_out_refuses_on_non_preserved_side() -> None:
    plan = FederatedPlan(
        steps=(),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="left",
            ),
        ),
    )
    with pytest.raises(FederationJoinFanOutError) as exc_info:
        validate_coordinator_join_fan_out(plan, {"a": 2, "b": 4}, 4)
    assert exc_info.value.source_id == "b"
    assert exc_info.value.phase == "coordinator"


@pytest.mark.fast
def test_coordinator_star_join_fan_out_checks_beyond_first_combine_edge() -> None:
    manifest = _three_member_star_manifest()
    intent = RuntimeIntent(
        tables=["t_a", "t_b", "t_c"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    composite = compose_composite_graph(
        {
            "a": _graph("t_a", source_id="a", row_count=2),
            "b": _graph("t_b", source_id="b", row_count=2),
            "c": _graph("t_c", source_id="c", row_count=4, id_unique=True, id_distinct=4),
        },
        manifest,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.combine is not None
    assert len(plan.combine) == 2
    with pytest.raises(FederationJoinFanOutError) as exc_info:
        validate_coordinator_join_fan_out(plan, {"a": 2, "b": 2, "c": 4}, 2, combine_row_count=8)
    assert exc_info.value.source_id in {"a", "b", "c"}


@pytest.mark.fast
def test_coordinator_join_fan_out_not_masked_by_residual_limit() -> None:
    manifest, _ = _two_member_join_manifest()
    composite = compose_composite_graph(
        {"a": _graph("t_a", source_id="a"), "b": _graph("t_b", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        limit=1,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.residual is not None
    assert plan.residual.limit == 1
    with pytest.raises(FederationJoinFanOutError):
        validate_coordinator_join_fan_out(plan, {"a": 2, "b": 2}, 1, combine_row_count=4)


@pytest.mark.fast
def test_residual_aggregate_sum_join_fan_out_refuses() -> None:
    manifest, mappings = _two_member_join_manifest()
    schema = SchemaGraph(
        tables={
            "t_a": TableMetadata(
                name="t_a",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                        is_primary_key=True,
                    ),
                    "amount": ColumnMetadata(
                        name="amount",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                    ),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
                row_count=2,
            ),
            "t_b": TableMetadata(
                name="t_b",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                        row_count=4,
                        distinct_count=2,
                    ),
                    "amount": ColumnMetadata(
                        name="amount",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                    ),
                },
                primary_key=[],
                foreign_keys=[],
                source_id="b",
                row_count=4,
            ),
        },
        join_paths_multi=recompute_join_paths_multi(
            {
                "t_a": TableMetadata(
                    name="t_a",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", sensitivity="none")
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="a",
                ),
                "t_b": TableMetadata(
                    name="t_b",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", sensitivity="none")
                    },
                    primary_key=[],
                    foreign_keys=[],
                    source_id="b",
                ),
            }
        ),
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "t_a.amount"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(FederationJoinFanOutError) as exc_info:
        plan_federated_intent(intent, schema, manifest, mappings)
    assert "sum" in str(exc_info.value).lower() or "aggregate" in str(exc_info.value).lower()
    assert exc_info.value.phase == "coordinator"


# --- Cross-source key uniqueness ---


@pytest.mark.fast
def test_cross_source_join_non_unique_left_key_refuses_at_declaration() -> None:
    manifest, mappings = _two_member_join_manifest()
    schema = SchemaGraph(
        tables={
            "t_a": TableMetadata(
                name="t_a",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                        row_count=10,
                        distinct_count=4,
                    ),
                },
                primary_key=[],
                foreign_keys=[],
                source_id="a",
                row_count=10,
            ),
            "t_b": TableMetadata(
                name="t_b",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                        is_primary_key=True,
                    ),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
                row_count=10,
            ),
        },
        join_paths_multi=recompute_join_paths_multi(
            {
                "t_a": TableMetadata(
                    name="t_a",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", sensitivity="none")
                    },
                    primary_key=[],
                    foreign_keys=[],
                    source_id="a",
                ),
                "t_b": TableMetadata(
                    name="t_b",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", sensitivity="none")
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="b",
                ),
            }
        ),
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        validate_cross_source_keys_on_graph(schema, manifest, mappings)
    message = str(exc_info.value)
    assert "t_a.id" in message
    assert "not unique" in message


@pytest.mark.fast
def test_join_key_clique_non_unique_endpoint_refuses_at_declaration() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_clique",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_columns": [
                {
                    "logical": "join_id",
                    "role": "join_key",
                    "unify_in_graph": True,
                    "members": ["t_a.join_id", "t_b.join_id"],
                }
            ],
        }
    )
    schema = SchemaGraph(
        tables={
            "t_a": TableMetadata(
                name="t_a",
                columns={
                    "join_id": ColumnMetadata(
                        name="join_id",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                        row_count=10,
                        distinct_count=4,
                    ),
                },
                primary_key=[],
                foreign_keys=[],
                source_id="a",
                row_count=10,
            ),
            "t_b": TableMetadata(
                name="t_b",
                columns={
                    "join_id": ColumnMetadata(
                        name="join_id",
                        data_type="integer",
                        value_type="integer",
                        sensitivity="none",
                        is_primary_key=True,
                    ),
                },
                primary_key=["join_id"],
                foreign_keys=[],
                source_id="b",
                row_count=10,
            ),
        },
        join_paths_multi=recompute_join_paths_multi(
            {
                "t_a": TableMetadata(
                    name="t_a",
                    columns={
                        "join_id": ColumnMetadata(
                            name="join_id", data_type="integer", value_type="integer", sensitivity="none"
                        )
                    },
                    primary_key=[],
                    foreign_keys=[],
                    source_id="a",
                ),
                "t_b": TableMetadata(
                    name="t_b",
                    columns={
                        "join_id": ColumnMetadata(
                            name="join_id", data_type="integer", value_type="integer", sensitivity="none"
                        )
                    },
                    primary_key=["join_id"],
                    foreign_keys=[],
                    source_id="b",
                ),
            }
        ),
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        validate_cross_source_keys_on_graph(schema, manifest, mappings)
    message = str(exc_info.value)
    assert "t_a.join_id" in message
    assert "not unique" in message


# --- Union member disjointness ---


def _union_members_and_mappings(
    *,
    a_sample: tuple[str, ...] = ("1", "2"),
    b_sample: tuple[str, ...] = ("3", "4"),
    a_row_count: int = 2,
    b_row_count: int = 2,
) -> tuple[dict[str, SchemaGraph], object, object]:
    members = {
        "a": _graph(
            "payment_a",
            source_id="a",
            row_count=a_row_count,
            overlap_sample=a_sample,
            id_distinct=a_row_count,
        ),
        "b": _graph(
            "payment_b",
            source_id="b",
            row_count=b_row_count,
            overlap_sample=b_sample,
            id_distinct=b_row_count,
        ),
    }
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_union_overlap",
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
def test_union_members_overlapping_keys_refuses_at_declaration() -> None:
    members, manifest, mappings = _union_members_and_mappings(
        a_sample=("1", "2", "3"),
        b_sample=("3", "4"),
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    message = str(exc_info.value)
    assert "payment" in message
    assert "id" in message
    assert "a" in message
    assert "b" in message


@pytest.mark.fast
def test_union_row_count_sums_disjoint_members() -> None:
    members, manifest, mappings = _union_members_and_mappings(
        a_sample=("1", "2"),
        b_sample=("3", "4"),
    )
    composite = compose_composite_graph(members, manifest, mappings)
    payment = composite.tables["payment"]
    assert payment.row_count == 4


# --- Disagreeing primary keys ---


def _entity_graph(table: str, *, source_id: str, primary_key: list[str]) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    sensitivity="none",
                    is_primary_key="id" in primary_key,
                ),
                "code": ColumnMetadata(
                    name="code",
                    data_type="varchar",
                    value_type="varchar",
                    sensitivity="none",
                    is_primary_key="code" in primary_key,
                ),
            },
            primary_key=primary_key,
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


@pytest.mark.fast
def test_disagreeing_primary_keys_raise_at_collapse() -> None:
    members = {
        "a": _entity_graph("entity_a", source_id="a", primary_key=["id"]),
        "b": _entity_graph("entity_b", source_id="b", primary_key=["code"]),
    }
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_pk",
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
                        {"source": "a", "table": "entity_a", "columns": {"id": "id", "code": "code"}},
                        {"source": "b", "table": "entity_b", "columns": {"id": "id", "code": "code"}},
                    ],
                },
            ],
            "logical_columns": [],
        }
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    assert "primary" in str(exc_info.value).lower()


@pytest.mark.fast
def test_union_unprofiled_members_refuse_disjointness_check() -> None:
    members = {
        "a": _graph("payment_a", source_id="a", row_count=0, overlap_sample=(), id_distinct=0),
        "b": _graph("payment_b", source_id="b", row_count=0, overlap_sample=(), id_distinct=0),
    }
    _, manifest, mappings = _union_members_and_mappings()
    with pytest.raises(FederationDeclarationError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    message = str(exc_info.value)
    assert "payment" in message
    assert "disjoint" in message.lower() or "establish" in message.lower()


@pytest.mark.fast
def test_disagreeing_member_grains_raise_at_collapse() -> None:
    members = {
        "a": _graph(
            "entity_a",
            source_id="a",
            row_count=10,
            id_unique=True,
            id_distinct=10,
        ),
        "b": _graph(
            "entity_b",
            source_id="b",
            row_count=10,
            id_unique=False,
            id_distinct=5,
        ),
    }
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_grain",
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
    with pytest.raises(FederationDeclarationError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    message = str(exc_info.value)
    assert "grain" in message.lower()
    assert "a" in message
    assert "b" in message


@pytest.mark.fast
def test_replica_merge_raises_on_conflicting_value_type() -> None:
    members = {
        "a": _entity_graph("entity_a", source_id="a", primary_key=["id"]),
        "b": _entity_graph("entity_b", source_id="b", primary_key=["id"]),
    }
    members["a"].tables["entity_a"].columns["id"].value_type = "integer"
    members["b"].tables["entity_b"].columns["id"].value_type = "string"
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_value_type",
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
    with pytest.raises(FederationConfigError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    assert "value_type" in str(exc_info.value)


@pytest.mark.fast
def test_union_merge_raises_on_conflicting_value_type() -> None:
    members, manifest, mappings = _union_members_and_mappings()
    members["a"].tables["payment_a"].columns["id"].value_type = "integer"
    members["b"].tables["payment_b"].columns["id"].value_type = "string"
    with pytest.raises(FederationConfigError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    assert "value_type" in str(exc_info.value)
