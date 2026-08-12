"""Lazy template-store mapping: no eager partition loads at construction."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import TEMPLATE_STORE_PARTITION_LRU_MAX
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateStoreView
from aetherdialect._templates_ops import TemplateOps


def _tiny_schema(*, schema_graph_id: str = "sg_test000000000001__abcd1234") -> SchemaGraph:
    tables = {
        "t": TableMetadata(
            name="t",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=schema_graph_id,
        effective_structural_hash=schema_graph_id,
    )


def _typed_template(*, tid: str = "T0001", schema_graph_id: str = "sg_test000000000001__abcd1234") -> Template:
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return Template(
        id=tid,
        effective_structural_hash="h",
        schema_graph_id=schema_graph_id,
        intent_signature=concrete,
        intent_key="ik_t",
        tables_used=["t"],
        sql_param="SELECT t.id FROM t",
        sql_fp="fp_t",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
        schema_column_types={"t.id": "integer"},
    )


def _store_dir(tmp_path) -> str:
    return str(tmp_path / "intent_templates" / "spaces" / "master")


def _distinct_partition_template_ids(n: int) -> list[str]:
    ids: list[str] = []
    seen: set[int] = set()
    for i in range(500_000):
        tid = f"T{i:06d}"
        part = TemplateStoreView.template_partition_number(tid)
        if part in seen:
            continue
        seen.add(part)
        ids.append(tid)
        if len(ids) >= n:
            break
    assert len(ids) == n
    return ids


def _loaded_partition_count(view: TemplateStoreView) -> int:
    return len(view._partition_cache)


def _build_multi_partition_store(tmp_path, monkeypatch, *, n_templates: int) -> TemplateStoreView:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    os.makedirs(_store_dir(tmp_path), exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    graph_id = "sg_test000000000001__abcd1234"
    view = TemplateOps.empty_template_store(graph_id)
    tids = _distinct_partition_template_ids(n_templates)
    templates = {tid: replace(_typed_template(tid=tid), trust_level=1) for tid in tids}
    TemplateOps.templates_to_store(view, templates)
    TemplateOps.save_template_store(view)
    loaded = TemplateOps.load_template_store(graph_id, schema=None)
    assert len(loaded.partition_map) == n_templates
    with loaded._partition_cache_lock:
        loaded._partition_cache.clear()
    return loaded


@pytest.mark.fast
def test_engine_construction_loads_no_partitions(tmp_path, monkeypatch) -> None:
    view = _build_multi_partition_store(tmp_path, monkeypatch, n_templates=40)
    assert _loaded_partition_count(view) == 0

    templates = TemplateOps.store_to_templates(view)
    assert len(templates) == 40
    assert _loaded_partition_count(view) == 0


@pytest.mark.fast
def test_lookup_loads_only_its_partition(tmp_path, monkeypatch) -> None:
    view = _build_multi_partition_store(tmp_path, monkeypatch, n_templates=2)
    tids = sorted(view.partition_map)
    part_a = view.partition_map[tids[0]]
    part_b = view.partition_map[tids[1]]
    assert part_a != part_b

    templates = TemplateOps.store_to_templates(view)
    assert _loaded_partition_count(view) == 0

    _ = templates[tids[0]]
    assert _loaded_partition_count(view) == 1
    assert part_a in view._partition_cache
    assert part_b not in view._partition_cache


@pytest.mark.fast
def test_full_scan_releases_partitions(tmp_path, monkeypatch) -> None:
    n_templates = TEMPLATE_STORE_PARTITION_LRU_MAX + 8
    view = _build_multi_partition_store(tmp_path, monkeypatch, n_templates=n_templates)
    peak = 0
    total = 0
    for batch in view.iter_templates_by_partition():
        total += len(batch)
        peak = max(peak, _loaded_partition_count(view))
    assert total == n_templates
    assert peak <= TEMPLATE_STORE_PARTITION_LRU_MAX
    assert _loaded_partition_count(view) == 0
