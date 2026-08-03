"""Federation plan replay must re-validate member SQL and enforce topology pins."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import FederationPlanTemplate
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedPlan,
    ResidualSpec,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    SqlGenerationOutcome,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, SQLShape, TableMetadata
from aetherdialect._federation import (
    federation_plan_combine_hash,
    federation_plan_matches_template,
    federation_plan_step_fingerprints,
    lookup_federation_plan_template,
    parse_federation_manifest,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import replay_federated_prepare_from_plan_template
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import intent_key


def _graph(table: str, *, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


def _member_template(*, sql_param: str = "SELECT id FROM left_t") -> Template:
    return Template(
        id="T0001",
        effective_structural_hash="eff_a_left_t",
        intent_signature=ConcreteIntent(
            intent_id="t1",
            tables=["left_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="ik1",
        tables_used=["left_t"],
        sql_param=sql_param,
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(param_values=[{}], questions=[], natural_language=[]),
        stats=TemplateStats(accept=0, reject=0),
        trust_level=1,
    )


def _simple_plan(*, sub_intent: RuntimeIntent | None = None) -> FederatedPlan:
    intent = sub_intent or RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        sql_param="SELECT id FROM left_t",
    )
    return FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=intent, projected_keys=("left_t.id",)),),
    )


@pytest.mark.fast
def test_replay_calls_generate_and_validate_sql_match_gate() -> None:
    composite = _graph("left_t", source_id="a")
    plan = _simple_plan()
    tmpl = _member_template()
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=(("a", "fp1"),),
        combine_hash=federation_plan_combine_hash(plan),
        member_template_ids=(("a", tmpl.id),),
    )
    store_a = {"templates": {tmpl.id: tmpl}}
    dialect = MagicMock()
    dialect.finalize_render.return_value = "SELECT id FROM left_t"
    gen_out = SqlGenerationOutcome(
        "SELECT id FROM left_t",
        True,
        None,
        tmpl,
    )
    with patch(
        "aetherdialect._pipeline.generate_and_validate_sql",
        return_value=gen_out,
    ) as mock_gen:
        outcome = replay_federated_prepare_from_plan_template(
            plan,
            cached,
            composite,
            stores_by_source={"a": store_a},
            default_dialect=dialect,
            q_norm="show orders",
        )
    assert outcome.success
    mock_gen.assert_called_once()
    dialect.finalize_render.assert_not_called()


@pytest.mark.fast
def test_replay_pins_member_schema_graph_ids() -> None:
    composite = _graph("left_t", source_id="a")
    member_graph = replace(_graph("left_t", source_id="a"), schema_graph_id="sg_member_a")
    plan = _simple_plan()
    tmpl = _member_template()
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=(("a", "fp1"),),
        combine_hash=federation_plan_combine_hash(plan),
        member_template_ids=(("a", tmpl.id),),
    )
    dialect = MagicMock()
    gen_out = SqlGenerationOutcome("SELECT id FROM left_t", True, None, tmpl)
    with patch("aetherdialect._pipeline.generate_and_validate_sql", return_value=gen_out):
        outcome = replay_federated_prepare_from_plan_template(
            plan,
            cached,
            composite,
            stores_by_source={"a": {"templates": {tmpl.id: tmpl}}},
            default_dialect=dialect,
            member_graphs={"a": member_graph},
            q_norm="show orders",
        )
    assert outcome.member_schema_graph_ids == (("a", "sg_member_a"),)


@pytest.mark.fast
def test_replay_rejects_invalid_sub_intent() -> None:
    composite = _graph("left_t", source_id="a")
    bad_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.missing_col"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=bad_intent, projected_keys=("left_t.id",)),),
        residual=ResidualSpec(limit=10),
    )
    tmpl = _member_template()
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=(("a", "fp1"),),
        combine_hash=federation_plan_combine_hash(plan),
        member_template_ids=(("a", tmpl.id),),
    )
    outcome = replay_federated_prepare_from_plan_template(
        plan,
        cached,
        composite,
        stores_by_source={"a": {"templates": {tmpl.id: tmpl}}},
        default_dialect=MagicMock(),
        q_norm="show orders",
    )
    assert not outcome.success
    assert "missing_col" in (outcome.sql_validation_error or "")


@pytest.mark.fast
def test_topology_hash_missing_on_current_side_is_miss() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_topo",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"left_t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    composite = _graph("left_t", source_id="a")
    plan = _simple_plan()
    fingerprints = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key, manifest=manifest)
    template = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=fingerprints,
        combine_hash=federation_plan_combine_hash(plan),
        manifest_hash="stored_manifest_hash",
        member_tuple_hash="stored_member_hash",
    )
    assert not federation_plan_matches_template(
        plan,
        template,
        step_fingerprints=fingerprints,
        manifest_hash_value="",
        member_tuple_hash_value="",
    )


@pytest.mark.fast
def test_topology_hash_missing_on_template_side_is_miss() -> None:
    composite = _graph("left_t", source_id="a")
    plan = _simple_plan()
    fingerprints = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key)
    template = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=fingerprints,
        combine_hash=federation_plan_combine_hash(plan),
        manifest_hash="",
        member_tuple_hash="",
    )
    assert not federation_plan_matches_template(
        plan,
        template,
        step_fingerprints=fingerprints,
        manifest_hash_value="live_manifest_hash",
        member_tuple_hash_value="live_member_hash",
    )


@pytest.mark.fast
def test_lookup_rejects_template_when_current_topology_hash_missing() -> None:
    import tempfile

    from aetherdialect._federation import save_federation_plan_template

    composite = _graph("left_t", source_id="a")
    plan = _simple_plan()
    fingerprints = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key)
    template = FederationPlanTemplate(
        plan_id="ik1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=fingerprints,
        combine_hash=federation_plan_combine_hash(plan),
        manifest_hash="stored_manifest_hash",
        member_tuple_hash="stored_member_hash",
    )
    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(fed_dir, template)
        loaded = lookup_federation_plan_template(
            fed_dir,
            str(composite.schema_graph_id),
            "ik1",
            manifest_hash_value="",
            member_tuple_hash_value="",
        )
    assert loaded is None


@pytest.mark.fast
def test_question_reuse_passes_topology_hashes_to_match_gate() -> None:
    from aetherdialect._pipeline import _try_federation_plan_question_reuse

    composite = _graph("left_t", source_id="a")
    tmpl = _member_template()
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_topo",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"left_t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=(("a", "fp1"),),
        combine_hash="hash1",
        member_template_ids=(("a", tmpl.id),),
        manifest_hash="mh",
        member_tuple_hash="mth",
    )
    with (
        patch(
            "aetherdialect._pipeline._resolve_federation_plan_template_for_reuse",
            return_value=cached,
        ),
        patch("aetherdialect._pipeline.plan_federated_intent") as mock_plan,
        patch(
            "aetherdialect._pipeline.federation_plan_topology_identity",
            return_value=("live_mh", "live_mth"),
        ),
        patch(
            "aetherdialect._pipeline.federation_plan_matches_template",
            return_value=False,
        ) as mock_match,
        patch(
            "aetherdialect._pipeline.replay_federated_prepare_from_plan_template",
        ),
    ):
        mock_plan.return_value = _simple_plan()
        out = _try_federation_plan_question_reuse(
            "norm_q",
            tmpl,
            {},
            composite,
            MagicMock(),
            federation_dir="/fed",
            federation_manifest=manifest,
            stores_by_source={"a": {"templates": {tmpl.id: tmpl}}},
            member_graphs={"a": composite},
        )
    assert out is None
    mock_match.assert_called_once()
    _, kwargs = mock_match.call_args
    assert kwargs.get("manifest_hash_value") == "live_mh"
    assert kwargs.get("member_tuple_hash_value") == "live_mth"
