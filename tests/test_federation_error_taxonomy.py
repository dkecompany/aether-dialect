"""Fast regressions for federation error taxonomy and learning isolation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import (
    FederationConfigError,
    FederationDeclarationError,
    FederationInvariantError,
)
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPrepareOutcome,
    RuntimeIntent,
    SourceStep,
    SqlGenerationOutcome,
    Template,
    UserFeedbackRejectSuspendContext,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    parse_federation_manifest,
    stamp_federation_member_template,
    template_is_federation_plan_fragment,
)
from aetherdialect._intent_loop import find_trusted_template_match
from aetherdialect._pipeline_execute import execute_federated_prepare
from aetherdialect._pipeline_generate import complete_user_feedback_reject
from aetherdialect._schema_graph import recompute_join_paths_multi


def _table(name: str, *, source_id: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _graph(table: str, *, source_id: str) -> SchemaGraph:
    tables = {table: _table(table, source_id=source_id)}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


def _template(*, plan_only: bool = False, plan_id: str = "") -> Template:
    intent = RuntimeIntent(
        tables=["t"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return Template(
        id="tmpl_fed",
        intent_signature=intent.to_concrete(""),
        intent_key="ik",
        tables_used=["t"],
        sql_param="SELECT 1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="",
        value_history=ValueHistory(
            param_values=[{}],
            questions=["how many rows"],
            natural_language=["how many rows"],
        ),
        stats=TemplateStats(),
        federation_plan_id=plan_id,
        federation_plan_only=plan_only,
    )


@pytest.mark.fast
def test_template_is_federation_plan_fragment() -> None:
    assert not template_is_federation_plan_fragment(_template())
    assert template_is_federation_plan_fragment(_template(plan_only=True, plan_id="p1"))


@pytest.mark.fast
def test_stamp_federation_member_template_sets_provenance() -> None:
    tmpl = _template()
    stamp_federation_member_template(tmpl, plan_id="plan_x", source_id="storefront")
    assert tmpl.federation_plan_only is True
    assert tmpl.federation_plan_id == "plan_x"
    assert tmpl.member_source_id == "storefront"


@pytest.mark.fast
def test_find_trusted_template_match_skips_federation_plan_only() -> None:
    standalone = _template()
    fragment = _template(plan_only=True, plan_id="p1")
    hit = find_trusted_template_match(
        "how many rows",
        [fragment, standalone],
    )
    assert hit is not None
    assert hit.template.id == "tmpl_fed"
    assert hit.template.federation_plan_only is False


@pytest.mark.fast
def test_execute_prepared_federation_plan_preserves_config_error() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cfg",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("ta", source_id="a"), "b": _graph("tb", source_id="b")},
        manifest,
    )
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
                    grain="scalar",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
    )
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        steps=(),
        display_sql="",
        glue_sql="",
        combine_hash="different",
    )
    with pytest.raises(FederationInvariantError):
        execute_federated_prepare(
            prepared,
            composite,
            dialect=MagicMock(),
            dialects_by_source={},
            manifest=manifest,
        )


@pytest.mark.fast
def test_federation_declaration_error_is_config_subclass() -> None:
    with pytest.raises(FederationConfigError):
        raise FederationDeclarationError("bad manifest join")


@pytest.mark.fast
def test_federated_reject_routes_to_plan_not_member_store() -> None:
    store: dict = {"templates": {}, "question_feedback": {}}
    templates: dict = {}
    rejected: dict = {}
    intent = RuntimeIntent(
        tables=["t"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    schema = _graph("t", source_id="a")
    tmpl = _template()
    ctx = UserFeedbackRejectSuspendContext(
        intent=intent,
        sql="SELECT 1",
        schema=schema,
        store=store,
        templates=templates,
        rejected=rejected,
        q_norm="how many rows",
        generation_path="federation_plan",
        matched_template=tmpl,
        matched_rejected_template=None,
        dialect=None,
        structural_match_templates=[],
    )
    with patch("aetherdialect._pipeline_generate.record_federation_join_feedback") as fed_fb:
        with patch("aetherdialect._templates_ops.TemplateOps.record_question_feedback") as member_fb:
            with patch("aetherdialect._templates_ops.TemplateOps.save_template_store"):
                complete_user_feedback_reject(
                    ctx,
                    needs_reason=False,
                    reject_reason="",
                    federation_dir="/tmp/fed",
                    federation_plan_id="plan1",
                    cross_source_join_feedback=True,
                )
            fed_fb.assert_called_once()
            member_fb.assert_not_called()


@pytest.mark.fast
def test_sub_intent_post_compose_repair_uses_member_schema() -> None:
    """Post-compose repair on a sub-intent is judged by the member graph, not the composite."""
    from aetherdialect._contracts_base import MulGroup, NormalizedExpr
    from aetherdialect._contracts_core import SelectCol
    from aetherdialect._contracts_schema import FederationMappings
    from aetherdialect._federation_plan import _build_source_sub_intent
    from aetherdialect._intent_loop import apply_runtime_post_processing_lite

    member_left = TableMetadata(
        name="left_t",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                sensitivity="none",
                is_primary_key=True,
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="a",
    )
    composite_left = TableMetadata(
        name="left_t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
        },
        primary_key=[],
        foreign_keys=[],
        source_id="a",
    )
    right_t = TableMetadata(
        name="right_t",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                sensitivity="none",
                is_primary_key=True,
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="b",
    )
    member_graph = SchemaGraph(
        tables={"left_t": member_left},
        join_paths_multi=recompute_join_paths_multi({"left_t": member_left}),
    )
    composite_tables = {"left_t": composite_left, "right_t": right_t}
    composite = SchemaGraph(
        tables=composite_tables,
        join_paths_multi=recompute_join_paths_multi(composite_tables),
    )
    count_expr = NormalizedExpr(
        add_groups=[
            MulGroup(
                multiply=[NormalizedExpr(column_ref="left_t.id")],
                agg_func="count",
                distinct=True,
            ),
        ],
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="scalar",
        select_cols=[SelectCol(expr=count_expr)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    composite_out, _ = apply_runtime_post_processing_lite(intent, composite)
    assert composite_out is not None
    assert composite_out.select_cols[0].expr.add_groups[0].distinct is True

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_repair_schema",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    step = _build_source_sub_intent(
        intent,
        "a",
        {"left_t"},
        {"left_t": "a", "right_t": "b"},
        FederationMappings(version="0.2.3"),
        composite,
        manifest,
        multi_source=False,
        member_schema=member_graph,
    )
    assert step is not None
    assert step.sub_intent.select_cols[0].expr.add_groups[0].distinct is False


@pytest.mark.fast
def test_shared_key_expansion_after_decomposition_does_not_widen_sources() -> None:
    """Composite-level shared-key expansion can pull a foreign member; per-member expansion must not."""
    from dataclasses import replace

    from aetherdialect._contracts_base import NormalizedExpr
    from aetherdialect._contracts_core import SelectCol
    from aetherdialect._contracts_schema import FederationMappings, FKEdge, InferenceTag
    from aetherdialect._federation_compose import compose_composite_graph
    from aetherdialect._federation_plan import _build_source_sub_intent, plan_federated_intent
    from aetherdialect._intent_normalize import expand_shared_pk_tables_for_refs

    left_t = TableMetadata(
        name="left_t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
            "parent_id": ColumnMetadata(
                name="parent_id",
                data_type="integer",
                sensitivity="none",
                is_foreign_key=True,
                fk_target=("parent_t", "id"),
            ),
            "right_id": ColumnMetadata(
                name="right_id",
                data_type="integer",
                sensitivity="none",
                is_foreign_key=True,
                fk_target=("right_t", "id"),
            ),
        },
        primary_key=["id"],
        foreign_keys=[
            FKEdge(
                src_table="left_t",
                src_cols=["parent_id"],
                dst_table="parent_t",
                dst_cols=["id"],
            ),
            FKEdge(
                src_table="left_t",
                src_cols=["right_id"],
                dst_table="right_t",
                dst_cols=["id"],
            ),
        ],
    )
    parent_t = TableMetadata(
        name="parent_t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
            "name": ColumnMetadata(name="name", data_type="text", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    right_t = TableMetadata(
        name="right_t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    hazard_tables = {"left_t": left_t, "parent_t": parent_t, "right_t": right_t}
    hazard_composite = SchemaGraph(
        tables=hazard_tables,
        join_paths_multi=recompute_join_paths_multi(hazard_tables),
    )
    hazard_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent_t.name")),
            SelectCol(expr=NormalizedExpr.from_column("right_t.id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    widened = expand_shared_pk_tables_for_refs(hazard_intent, hazard_composite)
    assert "parent_t" in widened.tables
    assert "right_t" in widened.tables
    assert set(widened.tables) > {"left_t"}

    stamped_left = replace(
        left_t,
        source_id="a",
        foreign_keys=[
            FKEdge(
                src_table="left_t",
                src_cols=["parent_id"],
                dst_table="parent_t",
                dst_cols=["id"],
            ),
            FKEdge(
                src_table="left_t",
                src_cols=["right_id"],
                dst_table="right_t",
                dst_cols=["id"],
                inference_tag=InferenceTag.CROSS_SOURCE,
            ),
        ],
    )
    stamped_parent = replace(parent_t, source_id="a")
    stamped_right = replace(right_t, source_id="b")
    member_a_tables = {"left_t": stamped_left, "parent_t": stamped_parent}
    member_a = SchemaGraph(
        tables=member_a_tables,
        join_paths_multi=recompute_join_paths_multi(member_a_tables),
    )
    member_b = SchemaGraph(
        tables={"right_t": stamped_right},
        join_paths_multi=recompute_join_paths_multi({"right_t": stamped_right}),
    )
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_expand_span",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "parent_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph({"a": member_a, "b": member_b}, manifest)
    single_source_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent_t.name"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    original_sources = {"a"}
    step = _build_source_sub_intent(
        single_source_intent,
        "a",
        {"left_t", "parent_t"},
        {"left_t": "a", "parent_t": "a", "right_t": "b"},
        FederationMappings(version="0.2.3"),
        composite,
        manifest,
        multi_source=False,
        member_schema=member_a,
    )
    assert step is not None
    assert "right_t" not in (step.sub_intent.tables or [])
    assert set(step.sub_intent.tables or []) <= {"left_t", "parent_t"}

    plan = plan_federated_intent(
        single_source_intent,
        composite,
        manifest,
        member_graphs={"a": member_a, "b": member_b},
    )
    assert not plan.ineligible_reason
    assert {s.source_id for s in plan.steps} == original_sources
    for plan_step in plan.steps:
        assert "right_t" not in (plan_step.sub_intent.tables or [])


@pytest.mark.fast
def test_partial_failure_interactive_turn_is_structured() -> None:
    """Partial federation failures surface source_id, phase, and succeeded — not error=str(exc)."""
    from aetherdialect._contracts_base import FederationPartialFailureError
    from aetherdialect._contracts_schema import FederationPlanTemplate
    from aetherdialect._main_execution import MainExecutionOps

    owner = MagicMock()
    port = MagicMock()
    port._pending_federation_plan_template = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="cg",
        intent_key="ik",
        step_fingerprints=(),
        combine_hash="h",
    )
    exc = FederationPartialFailureError(
        "member b failed",
        source_id="b",
        phase="member",
        succeeded=(("a", 2, "2026-01-01T00:00:00+00:00"),),
    )
    with patch("aetherdialect._pipeline_execute.save_federation_plan_template") as save_plan:
        MainExecutionOps._handle_federation_partial_failure_interactive(port, owner, exc)
    save_plan.assert_not_called()
    assert port._pending_federation_plan_template is None
    port.note_turn_outcome.assert_called_once()
    kwargs = port.note_turn_outcome.call_args.kwargs
    assert kwargs["outcome"] == "federation_partial_failure"
    assert kwargs["error"] is None
    assert kwargs["federation_source_id"] == "b"
    assert kwargs["federation_phase"] == "member"
    assert kwargs["federation_succeeded"] == (("a", 2, "2026-01-01T00:00:00+00:00"),)


@pytest.mark.fast
def test_persist_pending_plan_template_runs_after_successful_execute() -> None:
    """Deferred plan templates must survive execute without persisting until user accept."""
    from aetherdialect._contracts_core import FederatedPlan, FederatedPrepareOutcome, GenerationPath, SourceStep
    from aetherdialect._contracts_schema import FederationPlanTemplate
    from aetherdialect._main_execution import MainExecutionOps

    sub_intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="a", sub_intent=sub_intent),))
    fed_prep = FederatedPrepareOutcome(success=True, plan=plan, display_sql="display")
    gen_out = SqlGenerationOutcome(
        "display",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        federation_plan_id="plan1",
    )
    pending = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="cg",
        intent_key="ik",
        step_fingerprints=(),
        combine_hash="h",
    )
    owner = MagicMock()
    owner._federation_manifest = parse_federation_manifest(
        {
            "federation_id": "fed_persist",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    owner._federation_dialects = {}
    owner._federation_source_runtimes = {}
    owner._federation_storage_dir = "/tmp/fed"
    owner._schema_role = "owner"
    session = MagicMock()
    session._owner = owner
    session._pending_federation_plan_template = pending
    with (
        patch("aetherdialect._main_interactive.execute_federated_prepare") as mock_exec,
        patch("aetherdialect._pipeline_execute.save_federation_plan_template") as save_plan,
    ):
        mock_exec.return_value = MagicMock(rows=[(1,)], bundle=MagicMock())
        MainExecutionOps._run_sql_execution_for_gen_out(
            intent=sub_intent,
            exec_schema=_graph("t", source_id="a"),
            exec_dialect=MagicMock(),
            tmpl_sd=None,
            gen_out=gen_out,
            owner=owner,
            choice_port=session,
            federated_prepare=fed_prep,
        )
    save_plan.assert_not_called()
    assert session._pending_federation_plan_template is pending


@pytest.mark.fast
def test_partial_failure_does_not_persist_member_learning() -> None:
    from aetherdialect._contracts_base import FederationPartialFailureError
    from aetherdialect._contracts_core import FederatedPlan, FederatedPrepareOutcome, GenerationPath, SourceStep
    from aetherdialect._main_execution import MainExecutionOps

    sub_intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="a", sub_intent=sub_intent),))
    fed_prep = FederatedPrepareOutcome(success=True, plan=plan, display_sql="display")
    gen_out = SqlGenerationOutcome(
        "display",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        federation_plan_id="plan1",
    )
    owner = MagicMock()
    owner._federation_manifest = parse_federation_manifest(
        {
            "federation_id": "fed_partial",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    owner._federation_dialects = {}
    owner._federation_source_runtimes = {}
    owner._federation_member_graphs = {"a": _graph("t", source_id="a")}
    owner._federation_storage_dir = "/tmp/fed"
    session = MagicMock()
    session._owner = owner
    with (
        patch(
            "aetherdialect._main_interactive.execute_federated_prepare",
            side_effect=FederationPartialFailureError(
                "member failed",
                source_id="a",
                phase="member",
                succeeded=(),
            ),
        ),
        patch("aetherdialect._main_interactive.persist_federated_member_stores") as persist_members,
        patch(
            "aetherdialect._main_interactive.MainInteractiveOps.persist_template_learning_for_pipeline_session",
            return_value=True,
        ),
        patch(
            "aetherdialect._main_spaces.MainSpaceOps.federation_stores_by_source",
            return_value={"a": MagicMock()},
        ),
    ):
        with pytest.raises(FederationPartialFailureError):
            MainExecutionOps._run_sql_execution_for_gen_out(
                intent=sub_intent,
                exec_schema=_graph("t", source_id="a"),
                exec_dialect=MagicMock(),
                tmpl_sd=None,
                gen_out=gen_out,
                owner=owner,
                choice_port=session,
                federated_prepare=fed_prep,
            )
    persist_members.assert_not_called()


@pytest.mark.fast
def test_non_federated_generate_does_not_reexpand_shared_pk() -> None:
    """Shared-key expansion runs during intent processing, not SQL generation."""
    from aetherdialect._contracts_base import NormalizedExpr
    from aetherdialect._contracts_core import SelectCol
    from aetherdialect._pipeline_generate import generate_and_validate_sql
    from aetherdialect._templates_ops import TemplateOps

    schema = _graph("t", source_id="solo")
    intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    store = TemplateOps.empty_template_store(schema.schema_graph_id)
    with patch("aetherdialect._intent_normalize.expand_shared_pk_tables_for_refs") as expand_mock:
        expand_mock.side_effect = lambda value, _schema: value
        with patch("aetherdialect._sql_gen.build_deterministic_sql", return_value="SELECT id FROM t"):
            with patch(
                "aetherdialect._pipeline_generate.run_sql_validation_cascade",
                return_value=(True, "", None, []),
            ):
                generate_and_validate_sql(
                    "q",
                    intent,
                    schema,
                    {},
                    {},
                    None,
                    store,
                    member_source_id=None,
                )
    expand_mock.assert_not_called()
