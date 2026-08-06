"""Reuse paths must re-check stored join segments before replaying SQL."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import FederationConfigError
from aetherdialect._contracts_base import FederationPlanTemplate
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedPlan,
    GenerationPath,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    SqlGenerationOutcome,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, SQLShape, TableMetadata
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import execute_reuse_with_params, replay_federated_prepare_from_plan_template

_JOIN_SEG = "child.parent_id->parent.id"


def _parent_child_schema(*, with_fk: bool = True) -> SchemaGraph:
    fk = FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"])
    edge = {
        "src_table": "child",
        "src_cols": ["parent_id"],
        "dst_table": "parent",
        "dst_cols": ["id"],
    }
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
            },
            primary_key=["id"],
            foreign_keys=[fk] if with_fk else [],
        ),
    }
    join_paths = {"child": {"parent": [path]}} if with_fk else {}
    return SchemaGraph(tables=tables, join_paths_multi=join_paths, effective_structural_hash="h")


def _join_template() -> Template:
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=[_JOIN_SEG],
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
        value_history=ValueHistory(
            param_values=[{}],
            questions=["norm_q"],
            natural_language=["nl"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
    )


def _fed_member_schema(*, with_fk: bool = True) -> SchemaGraph:
    fk = FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"])
    edge = {
        "src_table": "child",
        "src_cols": ["parent_id"],
        "dst_table": "parent",
        "dst_cols": ["id"],
    }
    path = [edge]
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
        "child": TableMetadata(
            name="child",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "parent_id": ColumnMetadata(name="parent_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[fk] if with_fk else [],
            source_id="a",
        ),
    }
    join_paths = {"child": {"parent": [path]}} if with_fk else {}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=join_paths,
        schema_graph_id="sg_a",
        effective_structural_hash="eff_a",
    )


@patch("aetherdialect._pipeline.LLMProvider.chat", return_value='{"aliases":{}}')
@patch("aetherdialect._templates.TemplateOps.save_template_store")
@patch("aetherdialect._templates.TemplateOps.templates_to_store", side_effect=lambda s, t: s)
@patch("aetherdialect._templates.TemplateOps.delete_rejected_templates_matching_question")
@patch("aetherdialect._pipeline.save_result_csv_for_store")
@patch("aetherdialect._pipeline.print_query_result")
@patch("aetherdialect._templates.TemplateOps.promote_trust")
@patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
def test_stale_join_path_skips_direct_reuse(
    _mock_val,
    _mock_promote,
    _mock_print,
    _mock_csv,
    _mock_del,
    _mock_tts,
    _mock_save,
    _mock_llm,
) -> None:
    tmpl = _join_template()
    dialect = MagicMock()
    dialect.finalize_render.return_value = "EXEC"
    dialect.explain_validation_sql = lambda sql, _pv: sql
    dialect.execute.return_value = [("row",)]
    store: dict = {"templates": {"T1": tmpl}}

    live = execute_reuse_with_params(
        "norm_q",
        tmpl,
        {},
        dialect,
        store,
        {"T1": tmpl},
        {},
        _parent_child_schema(with_fk=True),
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        prompt=False,
    )
    assert live is not None and live.success is True
    dialect.execute.assert_called_once()

    dialect.execute.reset_mock()
    stale = execute_reuse_with_params(
        "norm_q",
        tmpl,
        {},
        dialect,
        store,
        {"T1": tmpl},
        {},
        _parent_child_schema(with_fk=False),
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        prompt=False,
    )
    assert stale is None
    dialect.execute.assert_not_called()


@patch("aetherdialect._pipeline.LLMProvider.chat", return_value='{"aliases":{}}')
@patch("aetherdialect._templates.TemplateOps.save_template_store")
@patch("aetherdialect._templates.TemplateOps.templates_to_store", side_effect=lambda s, t: s)
@patch("aetherdialect._templates.TemplateOps.delete_rejected_templates_matching_question")
@patch("aetherdialect._pipeline.save_result_csv_for_store")
@patch("aetherdialect._pipeline.print_query_result")
@patch("aetherdialect._templates.TemplateOps.promote_trust")
@patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
def test_restored_join_path_allows_direct_reuse_again(
    _mock_val,
    _mock_promote,
    _mock_print,
    _mock_csv,
    _mock_del,
    _mock_tts,
    _mock_save,
    _mock_llm,
) -> None:
    tmpl = _join_template()
    dialect = MagicMock()
    dialect.finalize_render.return_value = "EXEC"
    dialect.explain_validation_sql = lambda sql, _pv: sql
    dialect.execute.return_value = [("row",)]
    store: dict = {"templates": {"T1": tmpl}}

    assert (
        execute_reuse_with_params(
            "norm_q",
            tmpl,
            {},
            dialect,
            store,
            {"T1": tmpl},
            {},
            _parent_child_schema(with_fk=False),
            reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
            prompt=False,
        )
        is None
    )

    restored = execute_reuse_with_params(
        "norm_q",
        tmpl,
        {},
        dialect,
        store,
        {"T1": tmpl},
        {},
        _parent_child_schema(with_fk=True),
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        prompt=False,
    )
    assert restored is not None and restored.success is True
    dialect.execute.assert_called_once()


def test_federated_member_schema_change_refuses_plan_replay() -> None:
    composite = _fed_member_schema(with_fk=True)
    sub_intent = RuntimeIntent(
        tables=["child", "parent"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        sql_param="SELECT child.id FROM child JOIN parent ON child.parent_id = parent.id",
        chosen_join_path_signature=[_JOIN_SEG],
    )
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=sub_intent,
                projected_keys=("child.id",),
            ),
        ),
    )
    tmpl = _join_template()
    store_a = {"templates": {tmpl.id: tmpl}}
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=(("a", "fp1"),),
        combine_hash="hash1",
        member_template_ids=(("a", tmpl.id),),
    )
    dialect = MagicMock()
    sql = "SELECT child.id FROM child JOIN parent ON child.parent_id = parent.id"
    gen_out = SqlGenerationOutcome(sql, True, None, tmpl)
    with patch("aetherdialect._pipeline.generate_and_validate_sql", return_value=gen_out):
        replay = replay_federated_prepare_from_plan_template(
            plan,
            cached,
            composite,
            stores_by_source={"a": store_a},
            default_dialect=dialect,
        )
    assert replay.success

    stale_composite = _fed_member_schema(with_fk=False)
    with pytest.raises(FederationConfigError, match="stale"):
        replay_federated_prepare_from_plan_template(
            plan,
            cached,
            stale_composite,
            stores_by_source={"a": store_a},
            default_dialect=dialect,
        )
