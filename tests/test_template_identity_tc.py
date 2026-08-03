"""Template and reuse identity regression coverage."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import FederationPlanTemplate, OrderByCol, WhereParam
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedPlan,
    GenerationPath,
    ResidualSpec,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SqlGenerationOutcome,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, SQLShape, TableMetadata
from aetherdialect._core_utils import colmap_signature, stable_json
from aetherdialect._federation import (
    federation_plan_residual_hash,
)
from aetherdialect._intent_process import (
    NormalizedExpr,
    collect_structural_match_templates,
    join_path_key_concrete,
    list_union_match_candidates,
    predicate_group_from_list,
)
from aetherdialect._pipeline import (
    _join_preset_scope_from_concrete,
    generate_and_validate_sql,
    match_question_level_template_reuse,
    replay_federated_prepare_from_plan_template,
)
from aetherdialect._templates import _remap_concrete_intent, template_schema_refs
from aetherdialect._utils import intent_key, template_instance_key_from_parts


def _parent_child_schema(*, with_fk: bool = True) -> SchemaGraph:
    fk = FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"])
    edge = {"src_table": "child", "src_cols": ["parent_id"], "dst_table": "parent", "dst_cols": ["id"]}
    path = [edge]
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "parent_id": ColumnMetadata(name="parent_id", data_type="integer", sensitivity="none"),
                "score": ColumnMetadata(name="score", data_type="integer", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[fk] if with_fk else [],
        ),
    }
    join_paths = {"child": {"parent": [path]}} if with_fk else {}
    return SchemaGraph(tables=tables, join_paths_multi=join_paths, effective_structural_hash="h")


def _join_template(*, select_col: str = "child.id", question: str = "norm_q") -> Template:
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(select_col))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["child.parent_id->parent.id"],
        chosen_join_candidate_id="J01",
    )
    return Template(
        id="T1",
        effective_structural_hash="h",
        intent_signature=concrete,
        intent_key="ik",
        tables_used=["child", "parent"],
        sql_param="SELECT child.id FROM child JOIN parent ON child.parent_id = parent.id",
        sql_fp="fp",
        shape=SQLShape(num_joins=1, has_group_by=False, has_agg=False),
        colmap_sig="c",
        value_history=ValueHistory(param_values=[{}], questions=[question], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
    )


# --- remap concrete intent clauses ---


@pytest.mark.fast
def test_remap_concrete_intent_rewrites_select_and_where_columns() -> None:
    where = predicate_group_from_list(
        [WhereParam(left_expr=NormalizedExpr.from_column("child.score"), op="=", right_expr=None, param_key="p1")]
    )
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["child"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.score"))],
        group_by_cols=[NormalizedExpr.from_column("child.score")],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("child.score"))],
        where=where,
        having=None,
        column_map={"score": "child"},
    )
    remapped = _remap_concrete_intent(concrete, {"child": "kid"}, {"child": {"score": "points"}})
    assert remapped.select_cols[0].expr.column_ref == "kid.points"
    assert remapped.group_by_cols[0].column_ref == "kid.points"
    assert remapped.order_by_cols[0].expr.column_ref == "kid.points"
    assert (remapped.where.leaves() if remapped.where else [])[0].left_expr.column_ref == "kid.points"
    refs = template_schema_refs(
        replace(
            _join_template(),
            intent_signature=remapped,
            tables_used=["kid"],
            colmap_sig=colmap_signature(remapped.column_map),
        )
    )
    assert ("kid", "points") in refs.columns


# --- structural match liveness ---


@pytest.mark.fast
def test_collect_structural_match_templates_skips_stale_template() -> None:
    intent = RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    tmpl = _join_template()
    live_schema = _parent_child_schema(with_fk=True)
    stale_schema = _parent_child_schema(with_fk=False)
    tmpl = replace(tmpl, intent_key=intent_key(intent))
    assert collect_structural_match_templates(intent, {"T1": tmpl}, schema=live_schema)
    assert collect_structural_match_templates(intent, {"T1": tmpl}, schema=stale_schema) == []


@pytest.mark.fast
def test_list_union_match_candidates_skips_stale_template() -> None:
    intent = RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    tmpl = replace(_join_template(), intent_key=intent_key(intent))
    assert list_union_match_candidates(intent, {"T1": tmpl}, schema=_parent_child_schema(with_fk=True))
    assert list_union_match_candidates(intent, {"T1": tmpl}, schema=_parent_child_schema(with_fk=False)) == []


@pytest.mark.fast
def test_match_question_level_template_reuse_skips_stale_template() -> None:
    tmpl = _join_template()
    live = match_question_level_template_reuse("norm_q", {"T1": tmpl}, schema=_parent_child_schema(with_fk=True))
    stale = match_question_level_template_reuse("norm_q", {"T1": tmpl}, schema=_parent_child_schema(with_fk=False))
    assert live.best_template is not None
    assert stale.best_template is None


# --- join preset replay ---


@pytest.mark.fast
def test_join_preset_scope_from_concrete_pins_main_and_cte() -> None:
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_candidate_id="J02",
        chosen_join_path_signature=["child.parent_id->parent.id"],
        cte_steps=[],
    )
    cte = replace(
        RuntimeCteStep(cte_name="rollup", tables=["child", "parent"], select_cols=[]),
        chosen_join_candidate_id="J03",
        chosen_join_path_signature=["child.parent_id->parent.id"],
    )
    from aetherdialect._contracts_core import _runtime_cte_to_concrete

    concrete = replace(concrete, cte_steps=[_runtime_cte_to_concrete(cte)])
    preset = _join_preset_scope_from_concrete(concrete)
    assert preset["main"] == "J02"
    assert preset["cte:rollup"] == "J03"


@patch("aetherdialect._pipeline.llm_chat", return_value='{"aliases":{}}')
@patch("aetherdialect._pipeline.save_template_store")
@patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
@patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
@patch("aetherdialect._pipeline._resolve_joins_fresh", return_value=("SELECT 1", {}))
@patch("aetherdialect._pipeline.build_deterministic_sql", return_value="SELECT 1")
@patch("aetherdialect._pipeline.generate_join_candidates")
@pytest.mark.fast
def test_path3_replay_passes_stored_join_preset(
    mock_generate_join_candidates,
    _mock_det,
    mock_resolve_joins_fresh,
    _mock_val,
    _mock_tts,
    _mock_save,
    _mock_llm,
) -> None:
    tmpl = _join_template()
    intent = RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    schema = _parent_child_schema(with_fk=True)
    jc = {
        "candidates": [{"candidate_id": "J01", "join_path_signature": ["child.parent_id->parent.id"], "edge_kinds": []}]
    }
    cmap = {"J01": ["child.parent_id->parent.id"]}
    mock_generate_join_candidates.return_value = (jc, cmap, {})
    dialect = MagicMock()
    dialect.finalize_render.return_value = "EXEC"
    generate_and_validate_sql(
        "norm_q",
        intent,
        schema,
        jc,
        cmap,
        dialect,
        {},
        matched_template=tmpl,
        union_sql_path=GenerationPath.INTENT_DIRECT_MATCH,
        persist_template_learning=False,
    )
    _kwargs = mock_resolve_joins_fresh.call_args.kwargs
    assert _kwargs.get("join_preset_scope") == {"main": "J01"}


# --- join path key layers ---


@pytest.mark.fast
def test_join_path_key_differs_for_candidate_id_and_cte_emission() -> None:
    base = ConcreteIntent(
        intent_id="id",
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["child.parent_id->parent.id"],
        chosen_join_candidate_id="J01",
    )
    alt_candidate = replace(base, chosen_join_candidate_id="J02")
    assert join_path_key_concrete(base) != join_path_key_concrete(alt_candidate)

    cte_join = replace(
        RuntimeCteStep(
            cte_name="probe",
            tables=["child", "parent"],
            select_cols=[],
            emission="join_table",
            chosen_join_path_signature=["child.parent_id->parent.id"],
            chosen_join_candidate_id="J01",
        )
    )
    cte_semi = replace(cte_join, emission="semi_join")
    from aetherdialect._contracts_core import _runtime_cte_to_concrete

    with_join = replace(base, cte_steps=[_runtime_cte_to_concrete(cte_join)])
    with_semi = replace(base, cte_steps=[_runtime_cte_to_concrete(cte_semi)])
    assert join_path_key_concrete(with_join) != join_path_key_concrete(with_semi)


@pytest.mark.fast
def test_intent_key_cte_skeleton_includes_emission() -> None:
    cte_a = RuntimeCteStep(cte_name="x", tables=["child"], select_cols=[], emission="join_table")
    cte_b = replace(cte_a, emission="semi_join")
    intent_a = RuntimeIntent(
        tables=["child"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte_a],
    )
    intent_b = replace(intent_a, cte_steps=[cte_b])
    assert intent_key(intent_a) != intent_key(intent_b)


# --- federation residual hash ---


@pytest.mark.fast
def test_federation_residual_hash_distinguishes_predicate_content() -> None:
    residual_a = ResidualSpec(
        where=predicate_group_from_list(
            [WhereParam(left_expr=NormalizedExpr.from_column("a.id"), op="=", right_expr=None, param_key="p1")]
        )
    )
    residual_b = ResidualSpec(
        where=predicate_group_from_list(
            [WhereParam(left_expr=NormalizedExpr.from_column("b.id"), op="=", right_expr=None, param_key="p1")]
        )
    )
    plan_a = FederatedPlan(steps=(), residual=residual_a)
    plan_b = FederatedPlan(steps=(), residual=residual_b)
    assert federation_plan_residual_hash(plan_a) != federation_plan_residual_hash(plan_b)


@pytest.mark.fast
def test_replay_federated_prepare_pins_member_schema_graph_ids() -> None:
    from aetherdialect._contracts_core import SourceStep

    composite = _parent_child_schema(with_fk=True)
    composite = replace(composite, schema_graph_id="sg_composite")
    for name in list(composite.tables.keys()):
        composite.tables[name] = replace(composite.tables[name], source_id="a")
    sub_intent = RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["child.parent_id->parent.id"],
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=sub_intent, projected_keys=("child.id",)),),
    )
    tmpl = _join_template()
    member_graph = replace(_parent_child_schema(with_fk=True), schema_graph_id="sg_a")
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="sg_composite",
        intent_key="ik",
        step_fingerprints=(),
        combine_hash="hash1",
        member_template_ids=(("a", tmpl.id),),
    )
    dialect = MagicMock()
    gen_out = SqlGenerationOutcome(tmpl.sql_param, True, None, tmpl)
    with patch("aetherdialect._pipeline.generate_and_validate_sql", return_value=gen_out):
        outcome = replay_federated_prepare_from_plan_template(
            plan,
            cached,
            composite,
            stores_by_source={"a": {"templates": {tmpl.id: tmpl}}},
            default_dialect=dialect,
            member_graphs={"a": member_graph},
        )
    assert outcome.member_schema_graph_ids == (("a", "sg_a"),)


@pytest.mark.fast
def test_federation_plan_question_reuse_requires_template_match() -> None:
    from aetherdialect._pipeline import _try_federation_plan_question_reuse

    composite = replace(_parent_child_schema(with_fk=True), schema_graph_id="sg_composite")
    tmpl = _join_template()
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="sg_composite",
        intent_key="ik",
        step_fingerprints=(("a", "other_fp"),),
        combine_hash="hash1",
        member_template_ids=(("a", tmpl.id),),
    )
    with (
        patch(
            "aetherdialect._pipeline._resolve_federation_plan_template_for_reuse",
            return_value=cached,
        ),
        patch(
            "aetherdialect._pipeline.plan_federated_intent",
        ) as mock_plan,
    ):
        from aetherdialect._contracts_core import JoinSpec

        mock_plan.return_value = FederatedPlan(
            steps=(),
            combine=(
                JoinSpec(
                    left_source="a",
                    right_source="b",
                    left_key="id",
                    right_key="id",
                    logical_key="id",
                    kind="inner",
                ),
            ),
        )
        out = _try_federation_plan_question_reuse(
            "norm_q",
            tmpl,
            {},
            composite,
            MagicMock(),
            federation_dir="/fed",
            federation_manifest=MagicMock(),
            stores_by_source={"a": {"templates": {tmpl.id: tmpl}}},
            member_graphs={"a": composite},
        )
        assert out is None


# --- template instance key ---


@pytest.mark.fast
def test_template_instance_key_includes_colmap_grain_limit_and_params() -> None:
    body = "body"
    join_fp = "join"
    sql_fp = "sql"
    base = template_instance_key_from_parts(body, join_fp, sql_fp)
    with_colmap = template_instance_key_from_parts(body, join_fp, sql_fp, colmap_sig="cm1")
    with_grain = template_instance_key_from_parts(body, join_fp, sql_fp, grain="scalar")
    with_limit = template_instance_key_from_parts(body, join_fp, sql_fp, limit=5)
    with_params = template_instance_key_from_parts(body, join_fp, sql_fp, params_fp=stable_json({"p1": 1}))
    assert len({base, with_colmap, with_grain, with_limit, with_params}) == 5
