"""T3: rename migration must not abort on anonymous-signature bucket size alone."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ColumnRole, MigrationTier
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import ArtifactManifest, assess_rename_migration_plan, try_rename_migration_plan
from aetherdialect._schema_graph import classify_migration_tier


def _col(
    name: str,
    *,
    pk: bool = False,
    overlap: tuple[str, ...] = (),
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_primary_key=pk,
        is_nullable=not pk,
        is_unique=pk,
        role=ColumnRole.IDENTIFIER.value if pk else ColumnRole.NUMERIC_MEASURE.value,
        value_overlap_sample=list(overlap),
    )


def _table(name: str, *, tag: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={
            "id": _col("id", pk=True),
            "code_a": _col("code_a", overlap=(f"{tag}_a",)),
            "code_b": _col("code_b", overlap=(f"{tag}_b",)),
        },
        foreign_keys=[],
        primary_key="id",
    )


def _symmetric_table(name: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={
            "id": _col("id", pk=True),
            "code_a": _col("code_a", overlap=("shared_a",)),
            "code_b": _col("code_b", overlap=("shared_b",)),
        },
        foreign_keys=[],
        primary_key="id",
    )


def _schema(tables: dict[str, TableMetadata]) -> SchemaGraph:
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
    )


def _seven_table_rename_pair() -> tuple[SchemaGraph, SchemaGraph]:
    old_names = [f"t{i}" for i in range(1, 8)]
    new_names = [f"n{i}" for i in range(1, 8)]
    old = _schema({name: _table(name, tag=f"s{i}") for i, name in enumerate(old_names, start=1)})
    new = _schema({name: _table(name, tag=f"s{i}") for i, name in enumerate(new_names, start=1)})
    return old, new


@pytest.mark.fast
def test_seven_table_confident_rename_returns_plan() -> None:
    old, new = _seven_table_rename_pair()
    plan = try_rename_migration_plan(old, new)
    assert plan is not None
    table_renames = dict(plan[0])
    assert len(table_renames) == 7
    assert table_renames == {f"t{i}": f"n{i}" for i in range(1, 8)}
    assessment = assess_rename_migration_plan(old, new)
    assert assessment is not None
    assert assessment.confidence == 1.0


@pytest.mark.fast
def test_seven_table_symmetric_rename_refuses_ambiguity() -> None:
    old_names = [f"t{i}" for i in range(1, 8)]
    new_names = [f"n{i}" for i in range(1, 8)]
    old = _schema({name: _symmetric_table(name) for name in old_names})
    new = _schema({name: _symmetric_table(name) for name in new_names})
    assert try_rename_migration_plan(old, new) is None
    assert assess_rename_migration_plan(old, new) is None


@pytest.mark.fast
def test_seven_table_rename_only_classifies_as_remap() -> None:
    old, new = _seven_table_rename_pair()
    manifest = ArtifactManifest(
        effective_structural_hash="eff_old",
        structural_hash="s_old",
        profiling_hash="p_old",
        scope_hash="sc",
    )
    tier = classify_migration_tier(manifest, new, previous_schema=old)
    assert tier == MigrationTier.REMAP
