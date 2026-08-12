"""Template store growth bounds and shard/header write integrity."""

from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import TEMPLATE_STORE_HEADER_FILENAME, TEMPLATE_STORE_PARTITION_PREFIX
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateStoreView
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import read_gzip_json, write_gzip_json_atomic


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


@pytest.mark.fast
def test_load_reconciles_live_templates_on_schema_graph_id_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    store_dir = _store_dir(tmp_path)
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    tmpl = _typed_template()
    old_id = tmpl.schema_graph_id
    view = TemplateOps.empty_template_store(old_id)
    TemplateOps.templates_to_store(view, {tmpl.id: tmpl})
    TemplateOps.save_template_store(view)

    new_schema = _tiny_schema(schema_graph_id="sg_new000000000002__efgh5678")
    loaded = TemplateOps.load_template_store(new_schema.schema_graph_id, new_schema)
    assert tmpl.id in loaded.partition_map
    assert loaded.schema_graph_id == new_schema.schema_graph_id
    reloaded = TemplateOps.load_template_store(new_schema.schema_graph_id, schema=None)
    assert tmpl.id in reloaded.partition_map
    assert reloaded.schema_graph_id == new_schema.schema_graph_id


@pytest.mark.fast
def test_prune_enforces_template_count_and_disk_size_on_save(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    from aetherdialect._config import EngineLimits

    monkeypatch.setattr(
        "aetherdialect._templates.TemplateStoreLifecycleOps._resolve_engine_limits",
        lambda: EngineLimits(template_store_max_count=2, template_store_max_disk_bytes=512),
    )
    store_dir = str(tmp_path / "intent_templates")
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", store_dir)
    graph_id = "sg_test000000000001__abcd1234"
    view = TemplateOps.empty_template_store(graph_id)
    templates = {f"T{i:04d}": replace(_typed_template(tid=f"T{i:04d}"), trust_level=i) for i in range(5)}
    TemplateOps.templates_to_store(view, templates)
    TemplateOps.save_template_store(view)
    loaded = TemplateOps.load_template_store(graph_id, schema=None)
    assert len(loaded.partition_map) <= 2
    assert "T0000" not in loaded.partition_map
    assert "T0004" in loaded.partition_map


@pytest.mark.fast
def test_load_repairs_uncommitted_shard_bodies_not_in_header(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    store_dir = _store_dir(tmp_path)
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    tmpl = _typed_template()
    view = TemplateOps.empty_template_store(tmpl.schema_graph_id)
    TemplateOps.templates_to_store(view, {tmpl.id: tmpl})
    TemplateOps.save_template_store(view)

    ghost = replace(_typed_template(tid="TGHOST"), trust_level=9)
    part = TemplateStoreView.template_partition_number(tmpl.id)
    part_path = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}{part:02x}.json.gz")
    shard = read_gzip_json(part_path)
    shard[ghost.id] = ghost.to_dict()
    write_gzip_json_atomic(part_path, shard, sort_keys=True)

    loaded = TemplateOps.load_template_store(tmpl.schema_graph_id, schema=None)
    assert ghost.id not in loaded.partition_map
    assert tmpl.id in loaded.partition_map
    shard_after = read_gzip_json(part_path)
    assert ghost.id not in shard_after


@pytest.mark.fast
def test_partial_save_crash_leaves_consistent_state_on_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    store_dir = _store_dir(tmp_path)
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    graph_id = "sg_test000000000001__abcd1234"
    view = TemplateOps.empty_template_store(graph_id)
    first = _typed_template(tid="T0001")
    TemplateOps.templates_to_store(view, {first.id: first})
    TemplateOps.save_template_store(view)

    second = replace(_typed_template(tid="T0002"), trust_level=2)
    TemplateOps.templates_to_store(view, {first.id: first, second.id: second})
    original_replace = os.replace
    calls: list[tuple[str, str]] = []

    def _replace_after_shard(src: str, dst: str) -> None:
        calls.append((src, dst))
        if len(calls) == 1 and TEMPLATE_STORE_HEADER_FILENAME not in dst:
            raise OSError("simulated crash before header commit")
        return original_replace(src, dst)

    with patch("aetherdialect._templates.os.replace", side_effect=_replace_after_shard):
        with pytest.raises(OSError, match="simulated crash"):
            TemplateOps.save_template_store(view)

    loaded = TemplateOps.load_template_store(graph_id, schema=None)
    assert loaded.partition_map.keys() <= {"T0001"}
    assert "T0002" not in loaded.partition_map
    assert loaded.get_template("T0001") is not None
