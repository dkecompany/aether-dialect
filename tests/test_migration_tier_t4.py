"""T4: classify_migration_tier and apply_migration_policy must agree on tier."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION
from aetherdialect._contracts_base import ColumnRole, MigrationTier
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._core_utils import ArtifactManifest, write_artifact_manifest
from aetherdialect._schema_graph import classify_migration_tier, diff_schemas
from aetherdialect._templates import apply_migration_policy


def _col(
    name: str,
    *,
    dt: str = "integer",
    pk: bool = False,
    nullable: bool = True,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type=dt,
        value_type=dt,
        is_primary_key=pk,
        is_nullable=nullable,
        is_unique=pk,
        role=ColumnRole.IDENTIFIER.value if pk else ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str, cols: dict[str, ColumnMetadata], *, fks: list[FKEdge] | None = None) -> TableMetadata:
    pk = next((c.name for c in cols.values() if c.is_primary_key), "")
    return TableMetadata(name=name, columns=cols, foreign_keys=fks or [], primary_key=pk)


def _schema(
    tables: dict[str, TableMetadata],
    *,
    structural: str = "s_new",
    scope: str = "sc",
    effective: str = "eff_new",
) -> SchemaGraph:
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        structural_hash=structural,
        profiling_hash="p_new",
        scope_hash=scope,
        effective_structural_hash=effective,
    )


def _manifest(*, scope: str = "sc") -> ArtifactManifest:
    return ArtifactManifest(
        artifact_format_version=ARTIFACT_FORMAT_VERSION,
        effective_structural_hash="eff_old",
        structural_hash="s_old",
        profiling_hash="p_old",
        scope_hash=scope,
        notes_hash="n",
        semantic_edges_hash="s",
    )


def _assert_tier_agreement(
    tmp_path,
    old: SchemaGraph,
    new: SchemaGraph,
    *,
    expected: MigrationTier,
) -> None:
    diff = diff_schemas(old, new)
    manifest = _manifest(scope=old.scope_hash)
    tier = classify_migration_tier(manifest, new, previous_schema=old, schema_diff=diff)
    assert tier == expected
    write_artifact_manifest(
        str(tmp_path),
        structural_hash="s_old",
        profiling_hash="p_old",
        scope_hash=old.scope_hash,
        effective_structural_hash="eff_old",
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )
    report = apply_migration_policy(str(tmp_path), new, previous_schema=old, schema_diff=diff)
    assert report.tier == tier


@pytest.mark.fast
def test_additive_column_tier_matches_policy(tmp_path) -> None:
    old = _schema(
        {"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})},
        structural="s_old",
        effective="eff_old",
    )
    new = _schema(
        {
            "orders": _table(
                "orders",
                {
                    "order_id": _col("order_id", pk=True),
                    "amount": _col("amount"),
                    "note": _col("note", dt="varchar"),
                },
            ),
        },
    )
    _assert_tier_agreement(tmp_path, old, new, expected=MigrationTier.ADDITIVE)


@pytest.mark.fast
def test_additive_table_tier_matches_policy(tmp_path) -> None:
    old = _schema({"orders": _table("orders", {"order_id": _col("order_id", pk=True)})}, structural="s_old", effective="eff_old")
    new = _schema(
        {
            "orders": _table("orders", {"order_id": _col("order_id", pk=True)}),
            "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
        },
    )
    _assert_tier_agreement(tmp_path, old, new, expected=MigrationTier.ADDITIVE)


@pytest.mark.fast
def test_nullability_change_tier_matches_policy(tmp_path) -> None:
    old = _schema({"t": _table("t", {"a": _col("a", nullable=True)})}, structural="s_old", effective="eff_old")
    new = _schema({"t": _table("t", {"a": replace(_col("a", nullable=True), is_nullable=False)})})
    _assert_tier_agreement(tmp_path, old, new, expected=MigrationTier.SOFT_REFRESH)


@pytest.mark.fast
def test_redeclared_column_tier_matches_policy(tmp_path) -> None:
    old = _schema({"t": _table("t", {"a": _col("a", dt="integer")})}, structural="s_old", effective="eff_old")
    new = _schema({"t": _table("t", {"a": _col("a", dt="bigint")})})
    _assert_tier_agreement(tmp_path, old, new, expected=MigrationTier.SOFT_REFRESH)


@pytest.mark.fast
def test_fk_change_tier_matches_policy(tmp_path) -> None:
    fk = FKEdge(src_table="orders", src_cols=["customer_id"], dst_table="customers", dst_cols=["customer_id"])
    old = _schema(
        {
            "orders": _table(
                "orders",
                {"order_id": _col("order_id", pk=True), "customer_id": _col("customer_id")},
                fks=[fk],
            ),
            "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
        },
        structural="s_old",
        effective="eff_old",
    )
    new = _schema(
        {
            "orders": _table("orders", {"order_id": _col("order_id", pk=True), "customer_id": _col("customer_id")}),
            "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
        },
    )
    _assert_tier_agreement(tmp_path, old, new, expected=MigrationTier.SOFT_REFRESH)
