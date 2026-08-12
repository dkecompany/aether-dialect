"""Template store liveness depth, pruning policy, and atomic multi-shard saves."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import (
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_PARTITION_PREFIX,
)
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    ConcreteCteStep,
    ConcreteIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._templates import TemplateRefs, TemplateStoreView
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import (
    read_artifact_manifest,
    read_gzip_json,
    write_gzip_json_atomic,
)


def _edge(src: str, sc: list[str], dst: str, dc: list[str]) -> dict:
    return {"src_table": src, "src_cols": sc, "dst_table": dst, "dst_cols": dc}


def _seg(src: str, sc: str, dst: str, dc: str) -> str:
    return f"{src}.{sc}->{dst}.{dc}"


def _three_hop_schema() -> SchemaGraph:
    fk_ab = FKEdge(src_table="a", src_cols=["b_id"], dst_table="b", dst_cols=["id"])
    fk_bc = FKEdge(src_table="b", src_cols=["c_id"], dst_table="c", dst_cols=["id"])
    fk_ad = FKEdge(src_table="a", src_cols=["d_id"], dst_table="d", dst_cols=["id"])
    fk_dc = FKEdge(src_table="d", src_cols=["c_id"], dst_table="c", dst_cols=["id"])
    path_ab = [_edge("a", ["b_id"], "b", ["id"])]
    path_bc = [_edge("b", ["c_id"], "c", ["id"])]
    path_abc = path_ab + path_bc
    path_ad = [_edge("a", ["d_id"], "d", ["id"])]
    path_dc = [_edge("d", ["c_id"], "c", ["id"])]
    path_adc = path_ad + path_dc
    tables = {
        "a": TableMetadata(
            name="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "b_id": ColumnMetadata(name="b_id", data_type="integer", sensitivity="none"),
                "d_id": ColumnMetadata(name="d_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[fk_ab, fk_ad],
        ),
        "b": TableMetadata(
            name="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "c_id": ColumnMetadata(name="c_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[fk_bc],
        ),
        "c": TableMetadata(
            name="c",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "d": TableMetadata(
            name="d",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "c_id": ColumnMetadata(name="c_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[fk_dc],
        ),
    }
    join_paths = {
        "a": {"b": [path_ab], "c": [path_abc, path_adc], "d": [path_ad]},
        "b": {"c": [path_bc]},
        "d": {"c": [path_dc]},
    }
    return SchemaGraph(tables=tables, join_paths_multi=join_paths, effective_structural_hash="h3")


def _cte_join_template() -> Template:
    main_sig = [_seg("a", "b_id", "b", "id")]
    cte_sig = [_seg("b", "c_id", "c", "id")]
    cte = ConcreteCteStep(
        cte_name="rollup",
        tables=["b", "c"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("c.id"))],
        chosen_join_path_signature=cte_sig,
        chosen_join_candidate_id="J02",
    )
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["a", "b"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=main_sig,
        chosen_join_candidate_id="J01",
        cte_steps=[cte],
    )
    return Template(
        id="TCTE",
        effective_structural_hash="h3",
        schema_graph_id="sg_test000000000001__abcd1234",
        intent_signature=concrete,
        intent_key="ik_cte",
        tables_used=["a", "b", "c"],
        sql_param="WITH rollup AS (SELECT c.id FROM b JOIN c ON b.c_id = c.id) SELECT a.id FROM a JOIN b ON a.b_id = b.id",
        sql_fp="fp_cte",
        shape=SQLShape(num_joins=2, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
        schema_column_types={
            "a.id": "integer",
            "b.id": "integer",
            "c.id": "integer",
        },
    )


def _typed_template(*, col_type: str = "integer") -> Template:
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
        id="TTYP",
        effective_structural_hash="h",
        schema_graph_id="sg_test000000000001__abcd1234",
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
        schema_column_types={"t.id": col_type},
    )


def _tiny_schema(*, id_type: str = "integer") -> SchemaGraph:
    tbl = TableMetadata(
        name="t",
        columns={"id": ColumnMetadata(name="id", data_type=id_type, sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="sh1", tables={"t": tbl})


@pytest.mark.fast
def test_template_schema_refs_collects_cte_join_segments() -> None:
    refs = TemplateRefs.template_schema_refs(_cte_join_template())
    assert _seg("a", "b_id", "b", "id") in refs.fk_edges
    assert _seg("b", "c_id", "c", "id") in refs.fk_edges


@pytest.mark.fast
def test_template_is_live_rejects_stale_cte_join_segment() -> None:
    tmpl = _cte_join_template()
    schema = _three_hop_schema()
    live = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(tmpl), schema)
    assert live[0] is True

    stale_tables = {
        "a": schema.tables["a"],
        "b": TableMetadata(
            name="b",
            columns={"id": schema.tables["b"].columns["id"]},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "c": schema.tables["c"],
        "d": schema.tables["d"],
    }
    stale_paths = {
        "a": schema.join_paths_multi["a"],
        "b": {},
        "d": schema.join_paths_multi["d"],
    }
    stale_schema = replace(schema, tables=stale_tables, join_paths_multi=stale_paths)
    dead = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(tmpl), stale_schema)
    assert dead[0] is False
    assert any(r.startswith("stale_join_path:") or r.startswith("missing_join_segment:") for r in dead[1])


@pytest.mark.fast
def test_template_is_live_rejects_column_type_mismatch() -> None:
    tmpl = _typed_template(col_type="integer")
    ok, reasons = TemplateRefs.template_is_live(
        TemplateRefs.template_schema_refs(tmpl), _tiny_schema(id_type="integer")
    )
    assert ok is True
    bad, reasons = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(tmpl), _tiny_schema(id_type="text"))
    assert bad is False
    assert any(r.startswith("column_type_mismatch:") for r in reasons)


@pytest.mark.fast
def test_template_is_live_rejects_join_path_not_current_in_join_paths_multi() -> None:
    schema = _three_hop_schema()
    stale_sig = (
        _seg("a", "b_id", "b", "id"),
        _seg("b", "c_id", "c", "id"),
    )
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=list(stale_sig),
        chosen_join_candidate_id="J01",
    )
    tmpl = replace(
        _cte_join_template(),
        intent_signature=concrete,
        tables_used=["a", "b", "c"],
    )
    pairwise_only = replace(
        schema,
        join_paths_multi={
            "a": {"b": schema.join_paths_multi["a"]["b"]},
            "b": {"c": schema.join_paths_multi["b"]["c"]},
            "d": schema.join_paths_multi["d"],
        },
    )
    ok, reasons = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(tmpl), pairwise_only)
    assert ok is False
    assert any(r.startswith("stale_join_path:") for r in reasons)


@pytest.mark.fast
def test_remove_template_compacts_empty_shard_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    store_dir = str(tmp_path / "intent_templates" / "spaces" / "master")
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    tmpl = _typed_template()
    view = TemplateOps.empty_template_store(tmpl.schema_graph_id)
    TemplateOps.templates_to_store(view, {tmpl.id: tmpl})
    TemplateOps.save_template_store(view)
    part = TemplateStoreView.template_partition_number(tmpl.id)
    part_path = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}{part:02x}.json.gz")
    assert os.path.isfile(part_path)
    view.remove_template_id(tmpl.id)
    TemplateOps.save_template_store(view)
    assert not os.path.isfile(part_path)


@pytest.mark.fast
def test_prune_caps_template_count(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    from aetherdialect._config import EngineLimits

    monkeypatch.setattr(
        "aetherdialect._templates.TemplateStoreLifecycleOps._resolve_engine_limits",
        lambda: EngineLimits(template_store_max_count=3),
    )
    store_dir = str(tmp_path / "intent_templates")
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", store_dir)
    cap = 3
    view = TemplateOps.empty_template_store("sg_test000000000001__abcd1234")
    templates: dict[str, Template] = {}
    for i in range(cap + 2):
        tid = f"T{i:04d}"
        templates[tid] = replace(_typed_template(), id=tid, trust_level=1 if i < cap else 3)
    TemplateOps.templates_to_store(view, templates)
    TemplateOps.save_template_store(view)
    loaded = TemplateOps.load_template_store("sg_test000000000001__abcd1234", schema=None)
    assert len(loaded.partition_map) <= cap
    assert "T0000" not in loaded.partition_map
    assert f"T{cap + 1:04d}" in loaded.partition_map


@pytest.mark.fast
def test_prune_value_history_rows_caps_per_template(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    from aetherdialect._config import EngineLimits

    monkeypatch.setattr(
        "aetherdialect._templates.TemplateStoreLifecycleOps._resolve_engine_limits",
        lambda: EngineLimits(template_value_history_depth=4),
    )
    store_dir = str(tmp_path / "intent_templates")
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", store_dir)
    cap = 4
    questions = [f"q{i}" for i in range(cap + 3)]
    vh = ValueHistory(
        param_values=[{} for _ in questions],
        questions=questions,
        natural_language=["nl"] * len(questions),
    )
    tmpl = replace(_typed_template(), value_history=vh)
    view = TemplateOps.empty_template_store(tmpl.schema_graph_id)
    TemplateOps.templates_to_store(view, {tmpl.id: tmpl})
    TemplateOps.save_template_store(view)
    loaded = TemplateOps.load_template_store(tmpl.schema_graph_id, schema=None)
    stored = loaded.get_template(tmpl.id)
    assert stored is not None
    assert len(stored.value_history.questions) == cap
    assert stored.value_history.questions[-1] == questions[-1]


@pytest.mark.fast
def test_load_detects_cross_shard_partition_map_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    store_dir = str(tmp_path / "intent_templates" / "spaces" / "master")
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    tmpl = _typed_template()
    view = TemplateOps.empty_template_store(tmpl.schema_graph_id)
    TemplateOps.templates_to_store(view, {tmpl.id: tmpl})
    TemplateOps.save_template_store(view)
    hdr_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    header = read_gzip_json(hdr_path)
    header["partition_map"]["TGHOST"] = header["partition_map"][tmpl.id]
    write_gzip_json_atomic(hdr_path, header, sort_keys=True)
    loaded = TemplateOps.load_template_store(tmpl.schema_graph_id, schema=None)
    assert "TGHOST" not in loaded.partition_map
    assert tmpl.id in loaded.partition_map


@pytest.mark.fast
def test_save_refreshes_manifest_structural_and_profiling_hashes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    artifacts = str(tmp_path / "artifacts")
    store_dir = os.path.join(artifacts, "intent_templates", "spaces", "master")
    os.makedirs(store_dir, exist_ok=True)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", os.path.join(artifacts, "intent_templates"))
    schema = _tiny_schema()
    schema = replace(
        schema,
        schema_graph_id="sg_test000000000001__abcd1234",
        structural_hash="struct_abc",
        profiling_hash="prof_def",
        effective_structural_hash="eff_ghi",
    )
    schema_path = os.path.join(artifacts, "schema_graph.json.gz")
    write_gzip_json_atomic(schema_path, schema.to_dict(), sort_keys=True)
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", schema_path)
    tmpl = replace(_typed_template(), schema_graph_id=schema.schema_graph_id)
    view = TemplateOps.empty_template_store(schema.schema_graph_id)
    TemplateOps.templates_to_store(view, {tmpl.id: tmpl})
    TemplateOps.save_template_store(view)
    manifest = read_artifact_manifest(artifacts)
    assert manifest is not None
    assert manifest.structural_hash == "struct_abc"
    assert manifest.profiling_hash == "prof_def"
    assert manifest.effective_structural_hash == "eff_ghi"
