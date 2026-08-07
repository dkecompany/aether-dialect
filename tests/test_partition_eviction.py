"""Partition cache eviction must not flush dirty shards outside save_template_store."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._contracts_core import (
    ConcreteIntent,
    SelectCol,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps, TemplateStoreView


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
def test_dirty_eviction_does_not_write_without_header(tmp_path, monkeypatch) -> None:
    lru_max = 4
    n_templates = lru_max + 1
    view = _build_multi_partition_store(tmp_path, monkeypatch, n_templates=n_templates)
    view._lru_max = lru_max

    tids = sorted(view.partition_map)
    loaded_parts: list[int] = []
    for tid in tids[:lru_max]:
        view.get_template_raw(tid)
        loaded_parts.append(int(view.partition_map[tid]))
    assert len(view._partition_cache) == lru_max

    victim_part = next(iter(view._partition_cache))
    victim_tid = next(tid for tid, part in view.partition_map.items() if int(part) == victim_part)
    with view._partition_cache_lock:
        payload = view._partition_cache[victim_part]
        raw = payload[victim_tid]
        payload[victim_tid] = {**raw, "trust_level": 99}
        view._dirty_partitions.add(victim_part)
    assert victim_part in view.dirty_partitions()

    victim_shard = view._partition_file_path(victim_part)
    victim_mtime_before = os.path.getmtime(victim_shard) if os.path.isfile(victim_shard) else None

    extra_tid = tids[lru_max]
    extra_part = int(view.partition_map[extra_tid])

    flushed_parts: list[int] = []
    real_flush = TemplateStoreView._flush_partition_to_disk

    def _track_flush(self, part: int, payload: dict) -> None:
        flushed_parts.append(part)
        return real_flush(self, part, payload)

    monkeypatch.setattr(TemplateStoreView, "_flush_partition_to_disk", _track_flush)
    view.get_template_raw(extra_tid)

    assert flushed_parts == []

    if victim_mtime_before is not None:
        assert os.path.getmtime(victim_shard) == victim_mtime_before

    assert victim_part in view.dirty_partitions()
    assert victim_part in view._partition_cache
    assert extra_part in view._partition_cache
    assert len(view._partition_cache) == lru_max

    evicted_clean = [p for p in loaded_parts if p != victim_part and p not in view._partition_cache]
    assert len(evicted_clean) == 1
