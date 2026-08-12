"""Consumer init must not run structural migration when tier is masked or destructive."""

from __future__ import annotations

import os

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import MigrationTier, NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import consumer_graph_is_permission_subset, diff_schemas
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import write_artifact_manifest


def _col(name: str, *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_primary_key=pk,
        is_nullable=not pk,
        is_unique=pk,
        role=ColumnRole.IDENTIFIER.value if pk else ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": _col("id", pk=True)},
        foreign_keys=[],
        primary_key="id",
        row_count=1,
    )


def _owner_consumer_pair() -> tuple[SchemaGraph, SchemaGraph]:
    owner = SchemaGraph(
        join_paths_multi={},
        tables={"a": _table("a"), "b": _table("b")},
        schema_graph_id="sg_perm000000000001__abcd1234",
        effective_structural_hash="owner_eff",
        structural_hash="owner_struct",
        scope_hash="scope_shared",
    )
    consumer = SchemaGraph(
        join_paths_multi={},
        tables={"a": _table("a")},
        schema_graph_id="sg_perm000000000001__abcd1234",
        effective_structural_hash="consumer_eff",
        structural_hash="consumer_struct",
        scope_hash="scope_shared",
    )
    return owner, consumer


def _make_template(tid: str, table: str) -> Template:
    intent = ConcreteIntent(
        intent_id=f"intent_{tid}",
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={"id": table},
    )
    return Template(
        id=tid,
        effective_structural_hash="eff_old",
        intent_signature=intent,
        intent_key=f"key_{tid}",
        tables_used=[table],
        sql_param=f"SELECT id FROM {table}",
        sql_fp=f"fp_{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig=f"sig_{tid}",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


def _seed_store(artifacts_dir: str, schema: SchemaGraph, templates: dict[str, Template]) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = TemplateOps.empty_template_store("eff_old")
        TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=schema.structural_hash,
        profiling_hash="p_old",
        scope_hash=schema.scope_hash,
        effective_structural_hash="eff_old",
        schema_graph_id=schema.schema_graph_id,
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )


def _reload_template_ids(artifacts_dir: str) -> set[str]:
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        raw = TemplateOps._load_partitioned_view_unlocked(store_dir)
        if raw is None:
            return set()
        return set(raw.partition_map.keys())
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev


@pytest.mark.fast
def test_permission_filtered_consumer_subset_detected() -> None:
    owner, consumer = _owner_consumer_pair()
    assert consumer_graph_is_permission_subset(owner, consumer) is True


@pytest.mark.fast
def test_consumer_init_migration_report_is_no_change_when_permission_filtered(tmp_path) -> None:
    owner, consumer = _owner_consumer_pair()
    artifacts_dir = str(tmp_path)
    _seed_store(artifacts_dir, owner, {"T_a": _make_template("T_a", "a"), "T_b": _make_template("T_b", "b")})
    schema_diff = diff_schemas(owner, consumer)
    assert schema_diff.dropped_tables == ("b",)

    report = MainExecutionOps.migration_report_for_init(
        artifacts_dir,
        owner,
        schema_role="consumer",
        previous_schema=owner,
        schema_diff=schema_diff,
    )

    assert report.tier == MigrationTier.NO_CHANGE
    assert report.destroyed_templates == 0
    assert report.surgically_invalidated == 0
    assert _reload_template_ids(artifacts_dir) == {"T_a", "T_b"}


@pytest.mark.fast
def test_allow_destructive_false_skips_diff_driven_mutation(tmp_path) -> None:
    owner, consumer = _owner_consumer_pair()
    artifacts_dir = str(tmp_path)
    _seed_store(artifacts_dir, owner, {"T_a": _make_template("T_a", "a"), "T_b": _make_template("T_b", "b")})
    schema_diff = diff_schemas(owner, consumer)

    report = TemplateOps.apply_migration_policy(
        artifacts_dir,
        owner,
        allow_destructive=False,
        previous_schema=owner,
        schema_diff=schema_diff,
    )

    assert report.tier == MigrationTier.NO_CHANGE
    assert report.destroyed_templates == 0
    assert report.surgically_invalidated == 0
    assert _reload_template_ids(artifacts_dir) == {"T_a", "T_b"}
