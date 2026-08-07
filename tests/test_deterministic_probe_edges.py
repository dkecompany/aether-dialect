"""Probe CTE join edges are resolved deterministically and withheld from join-choice prompts."""

from __future__ import annotations

from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    CteOutputColumnMeta,
    FKEdge,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._intent_expr import build_virtual_table_specs
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import (
    JOIN_CHOICE_SCOPE_MAIN,
    ScopeClass,
    apply_probe_edge_lineage_resolution,
    build_deterministic_sql,
    classify_scope_candidates,
    collapse_probe_edge_candidate_variation,
    inject_join_into_deterministic_sql,
    join_hints_multi,
    join_scope_pass1_plan,
    probe_cte_names,
    resolve_probe_edge_segments_from_lineage,
)
from tests.test_sql_gen import _pg_render


def _parent_child_schema() -> SchemaGraph:
    parent_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
    }
    child_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "parent_id": ColumnMetadata(
            name="parent_id",
            data_type="integer",
            sensitivity="none",
            is_foreign_key=True,
            fk_target=("parent", "id"),
        ),
    }
    fk = FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"])
    tables = {
        "parent": TableMetadata(name="parent", columns=parent_cols, primary_key=["id"], foreign_keys=[]),
        "child": TableMetadata(
            name="child",
            columns=child_cols,
            primary_key=["id"],
            foreign_keys=[fk],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="probe_deterministic",
    )


def _semi_probe_intent(*, probe_name: str = "active_parents") -> RuntimeIntent:
    ocm = {
        "parent_id": CteOutputColumnMeta(
            source="passthrough",
            lineage_phys_table="child",
            lineage_phys_column="parent_id",
            lineage_fk_to_table="parent",
            lineage_fk_to_column="id",
        )
    }
    semi = RuntimeCteStep(
        cte_name=probe_name,
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        output_column_metadata=ocm,
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return RuntimeIntent(
        tables=["parent", probe_name],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
    )


def test_probe_only_candidate_variation_skips_join_choice_llm() -> None:
    schema = _parent_child_schema()
    intent = _semi_probe_intent()
    virtual_specs = build_virtual_table_specs(intent, schema)
    hints = join_hints_multi(
        schema,
        ["parent", "active_parents"],
        intent,
        virtual_specs=virtual_specs,
        include_semantic=False,
    )
    assert classify_scope_candidates(hints["candidates"]) == ScopeClass.single_fk
    preset, llm_scopes, _, _ = join_scope_pass1_plan(
        main_multi_table=True,
        main_tables=["parent", "active_parents"],
        main_candidates=hints["candidates"],
        cte_scopes=[],
        forbid_na=False,
    )
    assert JOIN_CHOICE_SCOPE_MAIN in preset
    assert llm_scopes == []


def test_collapsed_probe_variation_yields_single_distinct_candidate() -> None:
    probe = probe_cte_names(
        [
            RuntimeCteStep(
                cte_name="probe",
                emission="semi_join",
                tables=["child"],
                output_columns=["parent_id"],
            )
        ]
    )
    candidates = [
        {
            "candidate_id": "J01",
            "join_path_signature": ["parent.id->probe.parent_id", "parent.id->child.parent_id"],
            "candidate_tier": "base",
        },
        {
            "candidate_id": "J02",
            "join_path_signature": ["parent.id->probe.other_key", "parent.id->child.parent_id"],
            "candidate_tier": "base",
        },
    ]
    collapsed = collapse_probe_edge_candidate_variation(candidates, probe)
    assert len(collapsed) == 1
    assert classify_scope_candidates(collapsed) == ScopeClass.single_fk


def test_resolved_probe_edge_matches_output_column_lineage() -> None:
    schema = _parent_child_schema()
    intent = _semi_probe_intent(probe_name="probe")
    virtual_specs = build_virtual_table_specs(intent, schema)
    hints = join_hints_multi(
        schema,
        ["parent", "probe"],
        intent,
        virtual_specs=virtual_specs,
        include_semantic=False,
    )
    fk_candidates = [c for c in hints["candidates"] if c.get("candidate_id") != "J00"]
    assert len(fk_candidates) == 1
    signature = fk_candidates[0]["join_path_signature"]
    assert signature == ["parent.id->probe.parent_id"]
    resolved_segments, resolved_kinds = resolve_probe_edge_segments_from_lineage(
        intent, probe_cte_names(intent.cte_steps), schema, virtual_specs
    )
    assert resolved_segments == ["parent.id->probe.parent_id"]
    assert resolved_kinds == ["virtual_fk_bridge"]
    det = build_deterministic_sql(intent, None, schema, _pg_render())
    assert "probe" in det.lower()
    joined = inject_join_into_deterministic_sql(
        det,
        [[], signature],
        schema=schema,
        edge_kinds_ordered=[[], fk_candidates[0].get("edge_kinds") or ["catalog_fk"]],
        dialect=_pg_render(),
        cte_emissions={"probe": "semi_join"},
    )
    assert "inner join" in joined.lower()
    assert "probe.parent_id" in joined.lower() or "parent.id" in joined.lower()


def test_wrong_probe_lineage_resolves_to_declared_edge_not_enumeration_guess() -> None:
    schema = _parent_child_schema()
    wrong_ocm = {
        "parent_id": CteOutputColumnMeta(
            source="passthrough",
            lineage_phys_table="child",
            lineage_phys_column="parent_id",
            lineage_fk_to_table="child",
            lineage_fk_to_column="id",
        )
    }
    wrong_cte = RuntimeCteStep(
        cte_name="probe",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        output_column_metadata=wrong_ocm,
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    wrong_intent = RuntimeIntent(
        tables=["parent", "probe"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[wrong_cte],
    )
    virtual_specs = build_virtual_table_specs(wrong_intent, schema)
    hints = join_hints_multi(
        schema,
        ["parent", "probe"],
        wrong_intent,
        virtual_specs=virtual_specs,
        include_semantic=False,
    )
    fk_candidates = [c for c in hints["candidates"] if c.get("candidate_id") != "J00"]
    assert len(fk_candidates) == 1
    signature = fk_candidates[0]["join_path_signature"]
    resolved_segments, resolved_kinds = resolve_probe_edge_segments_from_lineage(
        wrong_intent, probe_cte_names(wrong_intent.cte_steps), schema, virtual_specs
    )
    assert resolved_segments == ["child.id->probe.parent_id"]
    assert "child.id->probe.parent_id" in signature
    assert "parent.id->probe.parent_id" not in signature
    candidates_with_wrong_probe_edge = [
        {
            "candidate_id": "J01",
            "join_path_signature": ["parent.id->probe.parent_id", "parent.id->child.parent_id"],
            "edge_kinds": ["virtual_fk_bridge", "catalog_fk"],
            "candidate_tier": "base",
            "edge_count": 2,
        }
    ]
    normalized = apply_probe_edge_lineage_resolution(
        candidates_with_wrong_probe_edge,
        probe_cte_names(wrong_intent.cte_steps),
        resolved_segments,
        resolved_kinds,
    )
    assert "child.id->probe.parent_id" in normalized[0]["join_path_signature"]
    assert "parent.id->probe.parent_id" not in normalized[0]["join_path_signature"]

    correct_intent = _semi_probe_intent(probe_name="probe")
    correct_virtual_specs = build_virtual_table_specs(correct_intent, schema)
    correct_resolved, _ = resolve_probe_edge_segments_from_lineage(
        correct_intent, probe_cte_names(correct_intent.cte_steps), schema, correct_virtual_specs
    )
    assert correct_resolved == ["parent.id->probe.parent_id"]
    assert correct_resolved != resolved_segments


def test_semantic_edge_variation_still_requests_join_choice() -> None:
    left_cols = {
        "right_tbl_id": ColumnMetadata(
            name="right_tbl_id",
            data_type="varchar",
            value_type="string",
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=100,
            row_count=200,
            null_ratio=0.0,
            semantic_join_neighbors=[("right_tbl", "other")],
        ),
    }
    right_cols = {
        "id": ColumnMetadata(
            name="id",
            data_type="varchar",
            value_type="string",
            is_primary_key=True,
            distinct_count=100,
            row_count=100,
            null_ratio=0.0,
            semantic_join_neighbors=[],
        ),
        "other": ColumnMetadata(
            name="other",
            data_type="varchar",
            value_type="string",
            distinct_count=50,
            row_count=100,
            null_ratio=0.0,
            semantic_join_neighbors=[("left_tbl", "right_tbl_id")],
        ),
    }
    tables = {
        "left_tbl": TableMetadata(name="left_tbl", columns=left_cols, primary_key=[], foreign_keys=[]),
        "right_tbl": TableMetadata(
            name="right_tbl",
            columns=right_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="semantic_probe",
    )
    hints = join_hints_multi(
        schema,
        ["left_tbl", "right_tbl"],
        None,
        virtual_specs={},
        include_semantic=True,
    )
    semantic_candidates = [c for c in hints["candidates"] if c.get("candidate_tier") in ("extended", "semantic")]
    assert semantic_candidates
    assert classify_scope_candidates(hints["candidates"]) == ScopeClass.semantic_only
    preset, llm_scopes, _, _ = join_scope_pass1_plan(
        main_multi_table=True,
        main_tables=["left_tbl", "right_tbl"],
        main_candidates=hints["candidates"],
        cte_scopes=[],
        forbid_na=False,
    )
    assert len(semantic_candidates) == 1
    assert JOIN_CHOICE_SCOPE_MAIN in preset
    assert llm_scopes == []


def test_probe_cte_body_present_in_deterministic_sql_with_and_without_semantic_pass() -> None:
    schema = _parent_child_schema()
    intent = _semi_probe_intent()
    virtual_specs = build_virtual_table_specs(intent, schema)
    for include_semantic in (False, True):
        hints = join_hints_multi(
            schema,
            ["parent", "active_parents"],
            intent,
            virtual_specs=virtual_specs,
            include_semantic=include_semantic,
        )
        assert hints["candidates"]
        det = build_deterministic_sql(intent, None, schema, _pg_render())
        assert "active_parents" in det
        assert "select distinct" in det.lower()
