"""QSim skeleton cache must not collide across schema graphs with identical table names."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ColumnRole
from aetherdialect._contracts_schema import ColumnMetadata, QSimSkeleton, SchemaGraph, TableMetadata
from aetherdialect._qsim import _SKELETON_CACHE, generate_all_skeletons


def _column_roles(schema: SchemaGraph) -> dict[str, str]:
    roles: dict[str, str] = {}
    for tname, tmeta in schema.tables.items():
        for cname, cmeta in tmeta.columns.items():
            roles[f"{tname}.{cname}"] = cmeta.role
    return roles


def _tenant_orders_schema(
    *,
    schema_graph_id: str,
    structural_hash: str,
    column_names: tuple[str, ...],
) -> SchemaGraph:
    """Build a minimal orders-only schema for tenant isolation tests."""
    columns = {
        name: ColumnMetadata(
            name=name,
            data_type="varchar",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=10,
        )
        for name in column_names
    }
    orders = TableMetadata(name="orders", columns=columns, foreign_keys=[], primary_key="")
    return SchemaGraph(
        join_paths_multi={},
        structural_hash=structural_hash,
        effective_structural_hash=structural_hash,
        schema_graph_id=schema_graph_id,
        tables={"orders": orders},
    )


@pytest.mark.fast
def test_skeleton_cache_isolated_by_schema_graph_identity() -> None:
    """Two tenants with table name orders must not share skeleton cache entries."""
    tenant_a = _tenant_orders_schema(
        schema_graph_id="sgid_tenant_a",
        structural_hash="struct_hash_tenant_a",
        column_names=("status", "category", "region"),
    )
    tenant_b = _tenant_orders_schema(
        schema_graph_id="sgid_tenant_b",
        structural_hash="struct_hash_tenant_b",
        column_names=("status",),
    )
    roles_a = _column_roles(tenant_a)
    roles_b = _column_roles(tenant_b)

    _SKELETON_CACHE.clear()
    result_a = generate_all_skeletons(["orders"], tenant_a, roles_a)
    result_b = generate_all_skeletons(["orders"], tenant_b, roles_b)

    assert isinstance(result_a[0], QSimSkeleton)
    assert isinstance(result_b[0], QSimSkeleton)
    max_where_a = max(s.num_where for s in result_a)
    max_where_b = max(s.num_where for s in result_b)
    assert max_where_a == 4
    assert max_where_b == 2
    assert result_a is not result_b
