"""Tests for federation decomposition repair: execute paths, suspend pins, templates, errors."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import (
    FederationDeclarationError,
    FederationInvariantError,
    FederationRuntimeError,
)
from aetherdialect._contracts_base import (
    FederationMappings,
    FederationPlanTemplate,
    HavingParam,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedPlan,
    FederatedPrepareOutcome,
    GenerationPath,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    SqlExecuteSuspendContext,
    SqlGenerationOutcome,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata
from aetherdialect._federation import (
    credit_federation_plan_accept,
    load_federation_plan_templates,
    parse_federation_manifest,
    save_federation_plan_template,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import replay_federated_prepare_from_plan_template
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps


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


_MANIFEST = {
    "federation_id": "fed_repair_exec",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def test_parse_federation_manifest_rejects_unknown_keys() -> None:
    bad = {
        "federation_id": "fed_repair_exec",
        "cross_source_joins": [],
        "unexpected_field": True,
    }
    with pytest.raises(FederationDeclarationError, match="unknown keys"):
        parse_federation_manifest(bad)


def test_credit_federation_plan_accept_updates_member_template_ids() -> None:
    with tempfile.TemporaryDirectory() as fed_dir:
        template = FederationPlanTemplate(
            plan_id="plan1",
            composite_schema_graph_id="cg1",
            intent_key="ik1",
            step_fingerprints=(("a", "sk1"),),
            combine_hash="hash1",
            question="q",
            member_template_ids=(("a", "T0001"),),
        )
        save_federation_plan_template(fed_dir, template)
        credit_federation_plan_accept(
            fed_dir,
            "plan1",
            "show orders",
            member_template_ids=(("a", "T0002"), ("b", "T0003")),
        )
        loaded = load_federation_plan_templates(fed_dir)["plan1"]
        assert "show orders" in loaded.accepted_questions
        assert loaded.member_template_ids == (("a", "T0002"), ("b", "T0003"))


def test_replay_federated_prepare_from_plan_template() -> None:
    composite = _graph("left_t", source_id="a")
    sub_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        sql_param="SELECT id FROM left_t",
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=sub_intent, projected_keys=("left_t.id",)),),
    )
    store_a = {"templates": {}, "question_feedback": {}, "next_id": 2}
    tmpl = Template(
        id="T0001",
        effective_structural_hash=composite.effective_structural_hash,
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
        sql_param="SELECT id FROM left_t",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(param_values=[{}], questions=[], natural_language=[]),
        stats=TemplateStats(accept=0, reject=0),
        trust_level=1,
    )
    store_a["templates"]["T0001"] = tmpl
    cached = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik1",
        step_fingerprints=(("a", "fp1"),),
        combine_hash="hash1",
        member_template_ids=(("a", tmpl.id),),
    )
    dialect = MagicMock()
    gen_out = SqlGenerationOutcome("SELECT id FROM left_t", True, None, tmpl)
    with patch("aetherdialect._pipeline.generate_and_validate_sql", return_value=gen_out):
        replay = replay_federated_prepare_from_plan_template(
            plan,
            cached,
            composite,
            stores_by_source={"a": store_a},
            default_dialect=dialect,
        )
    assert replay is not None
    assert replay.success
    assert replay.steps[0].sql == "SELECT id FROM left_t"
    assert replay.steps[0].matched_template is tmpl


def test_verify_federation_execute_resume_rejects_plan_id_mismatch() -> None:
    snap = MagicMock()
    snap.q_norm = "q"
    snap.store = {}
    snap.templates = {}
    snap.rejected = {}
    snap.schema_terms = set()
    snap.dialect = MagicMock()
    snap.schema = _graph("left_t", source_id="a")
    gen_out = SqlGenerationOutcome(
        "sql",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        federation_plan_id="expected",
    )
    ctx = SqlExecuteSuspendContext(
        tail=snap,
        execution_intent=RuntimeIntent(
            tables=["left_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        sql="sql",
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=False,
        tmpl_sd=None,
        federation_plan_id="other",
    )
    with pytest.raises(FederationInvariantError, match="plan id mismatch"):
        MainExecutionOps._verify_federation_execute_resume(ctx)


def test_run_sql_execution_for_gen_out_uses_federated_coordinator() -> None:
    composite = _graph("left_t", source_id="a")
    sub_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="a", sub_intent=sub_intent),))
    fed_prep = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="display",
    )
    gen_out = SqlGenerationOutcome(
        "display",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        federation_plan_id="plan1",
    )
    owner = MagicMock()
    owner._federation_manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    owner._federation_dialects = {}
    owner._federation_source_runtimes = {}
    owner._federation_storage_dir = None
    with patch("aetherdialect._main_execution.execute_federated_prepare") as mock_exec:
        mock_exec.return_value = MagicMock(rows=[(1,)], bundle=MagicMock())
        rows, _bundle = MainExecutionOps._run_sql_execution_for_gen_out(
            intent=sub_intent,
            exec_schema=composite,
            exec_dialect=MagicMock(),
            tmpl_sd=None,
            gen_out=gen_out,
            owner=owner,
            choice_port=None,
            federated_prepare=fed_prep,
        )
    assert rows == [(1,)]
    mock_exec.assert_called_once()


def test_complete_interactive_execute_uses_suspend_federated_prepare() -> None:
    composite = _graph("left_t", source_id="a")
    sub_intent = RuntimeIntent(
        tables=["left_t"],
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
    snap = MagicMock()
    snap.q_norm = "q"
    snap.store = {}
    snap.templates = {}
    snap.rejected = {}
    snap.schema_terms = set()
    snap.dialect = MagicMock()
    snap.schema = composite
    ctx = MainExecutionOps._sql_execute_suspend_context(
        snap,
        "display",
        None,
        gen_out,
        None,
        False,
        sub_intent,
        federated_prepare=fed_prep,
        federation_plan_id="plan1",
        federation_exec_context={"q_norm": "q"},
    )
    owner = MagicMock()
    owner._federation_manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    owner._federation_dialects = {}
    owner._federation_source_runtimes = {}
    owner._federation_storage_dir = None
    session = MagicMock()
    session._owner = owner
    with patch(
        "aetherdialect._main_execution.MainExecutionOps._run_sql_execution_for_gen_out", return_value=([(1,)], None)
    ) as mock_run:
        with patch("aetherdialect._main_execution.MainExecutionOps._offer_sql_feedback_after_execute"):
            MainExecutionOps._complete_interactive_execute(ctx, "y", choice_port=session)
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["federated_prepare"] is fed_prep


def test_union_missing_frames_raises_runtime_error() -> None:
    from aetherdialect._contracts_core import UnionSpec
    from aetherdialect._federation import _render_union_relation_sql

    spec = UnionSpec(logical_table="payment", member_source_ids=("a", "b"), semantics="union")
    with pytest.raises(FederationRuntimeError, match="missing member frames"):
        _render_union_relation_sql(spec, {})


def test_intersect_member_where_ops_uses_member_intersection() -> None:
    from aetherdialect._federation import intersect_member_where_ops

    duck = MagicMock()
    duck.supports_ilike = True
    duck.supports_case_insensitive_wrap = True
    duck.extra_where_ops.return_value = frozenset({"ilike", "not ilike"})
    bq = MagicMock()
    bq.supports_ilike = False
    bq.supports_case_insensitive_wrap = True
    bq.extra_where_ops.return_value = frozenset()
    ops = intersect_member_where_ops({"a": duck, "b": bq})
    assert "ilike" in ops
    assert "=" in ops


def test_generate_and_validate_sql_rejects_unsupported_federation_filter_op() -> None:
    from aetherdialect._contracts_core import WhereParam
    from aetherdialect._intent_process import NormalizedExpr
    from aetherdialect._pipeline import generate_and_validate_sql

    composite = _graph("left_t", source_id="a")
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op="ilike",
                    value_type="string",
                    raw_value="x",
                ),
            ]
        ),
    )
    store = {"templates": {}, "question_feedback": {}, "next_id": 1}
    out = generate_and_validate_sql(
        "q",
        intent,
        composite,
        {},
        {},
        MagicMock(),
        store,
        persist_template_learning=False,
        allowed_where_ops=frozenset({"=", "!=", "like"}),
    )
    assert not out.success
    assert "not supported by federation members" in (out.sql_validation_error or "")


def test_cross_source_having_routes_to_residual() -> None:
    from aetherdialect._federation import parse_federation_manifest, plan_federated_intent

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        sensitivity="none",
                        is_primary_key=True,
                        is_unique=True,
                        row_count=10,
                        distinct_count=10,
                    )
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
                row_count=10,
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        sensitivity="none",
                        is_primary_key=True,
                        is_unique=True,
                        row_count=10,
                        distinct_count=10,
                    )
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
                row_count=10,
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="grouped",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("max", "left_t.id"))],
        group_by_cols=[NormalizedExpr.from_column("right_t.id")],
        order_by_cols=[],
        where=None,
        having=PredicateGroup.from_list(
            [
                HavingParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op="=",
                    right_expr=NormalizedExpr.from_column("right_t.id"),
                    value_type="column",
                ),
            ]
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert plan.residual is not None
    assert plan.residual.having


def test_cross_source_or_filter_marks_ineligible() -> None:
    from aetherdialect._federation import parse_federation_manifest, plan_federated_intent

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="or",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op="=",
                    right_expr=NormalizedExpr.from_column("right_t.id"),
                    value_type="column",
                ),
            ),
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source OR filter is not supported: left_t.id = right_t.id"


def test_federation_table_set_ignores_unreferenced_window_registry() -> None:
    from aetherdialect._contracts_schema import WindowRegistryStep, WindowSpec
    from aetherdialect._federation import federation_table_set, parse_federation_manifest

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "unused_t": TableMetadata(
                name="unused_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=[],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[
            WindowRegistryStep(
                registry_id="w01",
                window_spec=WindowSpec(
                    function="row_number",
                    partition_by=[NormalizedExpr.from_column("unused_t.id")],
                ),
            ),
        ],
    )
    table_set = federation_table_set(intent, composite, manifest)
    assert "left_t" in table_set.tables
    assert "unused_t" not in table_set.tables


def test_expand_fk_select_skips_cross_source_fk() -> None:
    from aetherdialect._contracts_base import InferenceTag
    from aetherdialect._contracts_schema import FKEdge
    from aetherdialect._intent_repair import expand_fk_select_to_descriptive

    schema = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        sensitivity="none",
                        is_foreign_key=True,
                        fk_target=("right_t", "id"),
                    ),
                    "name": ColumnMetadata(name="name", data_type="text", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[
                    FKEdge(
                        src_table="left_t",
                        src_cols=["id"],
                        dst_table="right_t",
                        dst_cols=["id"],
                        inference_tag=InferenceTag.CROSS_SOURCE,
                    ),
                ],
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "name": ColumnMetadata(name="name", data_type="text", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    out = expand_fk_select_to_descriptive(intent, schema)
    assert out.select_cols[0].expr.column_ref == "left_t.id"


def test_federation_residual_column_headers_from_select_cols() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._federation import federation_residual_column_headers
    from aetherdialect._intent_process import NormalizedExpr

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.customer_id"))],
        ),
    )
    assert federation_residual_column_headers(plan) == ("customer_id",)


def test_insert_template_stamps_member_source_id() -> None:
    from aetherdialect._templates import TemplateOps

    composite = _graph("left_t", source_id="a")
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    store = {"templates": {}, "question_feedback": {}, "next_id": 1}
    templates: dict = {}
    tmpl = TemplateOps.insert_template(
        store,
        templates,
        composite,
        "member q",
        intent,
        "SELECT id FROM left_t",
        member_source_id="a",
        record_accept=True,
    )
    assert tmpl.member_source_id == "a"


def test_cross_source_join_feedback_routes_to_plan_store() -> None:
    from aetherdialect._contracts_core import GenerationPath, RejectionBucket
    from aetherdialect._federation import (
        load_federation_plan_templates,
        lookup_federation_join_feedback,
        save_federation_plan_template,
    )
    from aetherdialect._pipeline import complete_user_feedback_reject

    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(
            fed_dir,
            FederationPlanTemplate(
                plan_id="plan1",
                composite_schema_graph_id="cg1",
                intent_key="ik1",
                step_fingerprints=(("a", "fp1"),),
                combine_hash="hash1",
            ),
        )
        ctx = MagicMock()
        ctx.intent = RuntimeIntent(
            tables=["left_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        ctx.sql = "sql"
        ctx.schema = _graph("left_t", source_id="a")
        ctx.store = TemplateOps.empty_template_store("cg1")
        ctx.templates = dict(ctx.store["templates"])
        ctx.rejected = {}
        ctx.q_norm = "q"
        ctx.generation_path = GenerationPath.FEDERATION_PLAN
        ctx.matched_template = None
        ctx.matched_rejected_template = None
        ctx.dialect = MagicMock()
        ctx.structural_match_templates = None
        with patch("aetherdialect._templates.TemplateOps.summarize_failure_for_memory") as mock_summary:
            mock_summary.return_value = MagicMock(
                summary="wrong declared join",
                buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
            )
            complete_user_feedback_reject(
                ctx,
                needs_reason=True,
                reject_reason="wrong declared join",
                persist_template_learning=True,
                federation_dir=fed_dir,
                federation_plan_id="plan1",
                cross_source_join_feedback=True,
            )
        loaded = load_federation_plan_templates(fed_dir)["plan1"]
        assert loaded.join_feedback == ("wrong declared join",)
        assert lookup_federation_join_feedback(fed_dir, "plan1") == ["wrong declared join"]
        assert not ctx.store.question_feedback.get("q")


def test_apply_federation_migration_remap_clears_plan_templates_and_writes_sidecar() -> None:
    from aetherdialect._federation import (
        apply_federation_migration_map,
        federation_artifact_paths,
        load_federation_plan_templates,
        parse_federation_migration_map,
    )

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.1")
    migration = parse_federation_migration_map(
        {
            "version": "1",
            "action": "remap",
            "qualified_column_renames": [
                {"from": "left_t.id", "to": "left_t.identifier"},
            ],
        },
    )
    with tempfile.TemporaryDirectory() as fed_dir:
        save_federation_plan_template(
            fed_dir,
            FederationPlanTemplate(
                plan_id="plan1",
                composite_schema_graph_id="cg1",
                intent_key="ik1",
                step_fingerprints=(("a", "fp1"),),
                combine_hash="hash1",
            ),
        )
        updated_manifest, updated_mappings = apply_federation_migration_map(
            migration,
            manifest,
            mappings,
            fed_dir,
        )
        assert updated_manifest.cross_source_joins[0].left == "left_t.identifier"
        assert load_federation_plan_templates(fed_dir) == {}
        sidecar = federation_artifact_paths(fed_dir)["mappings_applied"]
        assert os.path.isfile(sidecar)


def test_federation_plan_template_topology_identity() -> None:
    from aetherdialect._federation import (
        federation_member_tuple_hash,
        federation_plan_topology_identity,
    )

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("left_t", source_id="a"),
        "b": _graph("right_t", source_id="b"),
    }
    mh, mth = federation_plan_topology_identity(members, manifest)
    assert mh
    assert mth == federation_member_tuple_hash(members, manifest)


def test_validate_federated_sub_intent_rejects_unknown_column() -> None:
    from aetherdialect._federation import validate_federated_sub_intent

    schema = _graph("left_t", source_id="a")
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.missing"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    err = validate_federated_sub_intent(intent, schema)
    assert err is not None


def test_federation_partial_failure_error_carries_attribution() -> None:
    from aetherdialect._contracts_base import FederationPartialFailureError

    exc = FederationPartialFailureError(
        "member a failed",
        source_id="a",
        phase="member",
        succeeded=(("b", 3, "2026-01-01T00:00:00+00:00"),),
        retryable=True,
    )
    assert exc.source_id == "a"
    assert exc.phase == "member"
    assert exc.succeeded[0][0] == "b"
    assert exc.retryable is True


def test_single_source_federated_intent_has_one_plan_step() -> None:
    from aetherdialect._federation import federation_plan_is_degenerate, plan_federated_intent

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "a"
    assert federation_plan_is_degenerate(plan)
    assert plan.ineligible_reason is None
    assert plan.combine is None or plan.combine == ()


@pytest.mark.fast
def test_single_source_federated_plan_byte_identical_sql() -> None:
    """A one-node federated plan renders the same SQL as the standalone member engine."""
    from aetherdialect._dialect import DialectRegistry
    from aetherdialect._federation import plan_federated_intent
    from aetherdialect._sql_gen import build_deterministic_sql

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    member_schema = _graph("left_t", source_id="a")
    composite = SchemaGraph(
        tables={
            "left_t": member_schema.tables["left_t"],
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op=">",
                    value_type="integer",
                    param_key="p1",
                    raw_value=0,
                ),
            ]
        ),
        param_values={"p1": 0},
    )
    plan = plan_federated_intent(intent, composite, manifest, member_graphs={"a": member_schema})
    assert len(plan.steps) == 1
    dialect = DialectRegistry.get("duckdb")
    standalone_sql = build_deterministic_sql(intent, schema=member_schema, dialect=dialect)
    federated_sql = build_deterministic_sql(plan.steps[0].sub_intent, schema=member_schema, dialect=dialect)
    assert federated_sql == standalone_sql


def test_narrow_bind_map_for_sub_intent_keeps_referenced_keys_only() -> None:
    from aetherdialect._contracts_core import WhereParam
    from aetherdialect._intent_expr import narrow_bind_map_for_sub_intent
    from aetherdialect._intent_process import NormalizedExpr

    sub_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op="=",
                    value_type="integer",
                    param_key="p1",
                ),
            ]
        ),
        param_values={"p1": 1, "p2": 99},
    )
    parent = dict(sub_intent.param_values)
    parent["p3"] = 100
    narrowed = narrow_bind_map_for_sub_intent(sub_intent, parent)
    assert narrowed == {"p1": 1}
    assert "p2" not in narrowed
    assert "p3" not in narrowed


def test_collect_param_slot_meta_includes_structural_keys() -> None:
    from aetherdialect._contracts_core import ConcreteIntent
    from aetherdialect._intent_expr import extract_structural_params
    from aetherdialect._templates import TemplateOps

    runtime = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        limit=5,
    )
    tagged = extract_structural_params(runtime)
    intent_sig = ConcreteIntent(
        intent_id="t1",
        tables=["left_t"],
        grain="many",
        select_cols=tagged.select_cols,
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        limit=tagged.limit,
        limit_param_key=tagged.limit_param_key,
        param_values=tagged.param_values,
    )
    slots = TemplateOps._collect_param_slot_meta(intent_sig)
    assert tagged.limit_param_key in slots


def test_render_federation_residual_sql_binds_filter_values() -> None:
    from aetherdialect._contracts_core import ResidualSpec, WhereParam
    from aetherdialect._federation import render_federation_residual_sql

    fp = WhereParam(
        left_expr=NormalizedExpr.from_column("left_t.name"),
        op="=",
        value_type="string",
        param_key="p1",
    )
    residual = ResidualSpec(
        select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.id")),),
        where=PredicateGroup.from_list([fp]),
    )
    sql = render_federation_residual_sql(
        "SELECT * FROM src_a",
        residual,
        param_values={"p1": "Horror'; DROP TABLE users;--"},
    )
    assert ":p1" in sql
    assert "Horror" not in sql
    assert "DROP TABLE" not in sql


def test_render_combine_select_keyword_requires_explicit_projection() -> None:
    from aetherdialect._federation import _render_combine_select_keyword, _render_join_select_keyword

    with pytest.raises(FederationRuntimeError, match="explicit column projection"):
        _render_combine_select_keyword(None)
    with pytest.raises(FederationRuntimeError, match="explicit column projection"):
        _render_join_select_keyword(None, left_alias="l", right_alias="r", left_cols=set(), right_cols=set())


def test_render_federation_residual_sql_adds_deterministic_order_by() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._federation import render_federation_residual_sql

    residual = ResidualSpec(
        select_cols=(
            SelectCol(expr=NormalizedExpr.from_agg("count", "left_t.id")),
            SelectCol(expr=NormalizedExpr.from_column("right_t.id")),
        ),
        group_by_cols=(NormalizedExpr.from_column("right_t.id"),),
    )
    sql = render_federation_residual_sql("SELECT * FROM joined", residual)
    assert "ORDER BY" in sql.upper()


def test_execute_federation_coordinator_empty_aggregate_returns_identity_row() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._federation import execute_federation_coordinator

    plan = FederatedPlan(
        steps=(),
        grain="scalar",
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_agg("count", "left_t.id")),),
        ),
    )
    result = execute_federation_coordinator({}, plan)
    assert len(result) == 1
    assert int(result.iloc[0, 0]) == 0


def test_execute_federation_coordinator_enforces_scalar_grain() -> None:
    from aetherdialect._federation import execute_federation_coordinator

    sub_intent = RuntimeIntent(
        tables=["a"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=sub_intent, projected_keys=("id",)),),
        grain="scalar",
    )
    frames = {"a": __import__("pandas").DataFrame({"id": [1, 2]})}
    with pytest.raises(FederationRuntimeError, match="scalar result has 2 rows"):
        execute_federation_coordinator(frames, plan)


def test_build_result_dataframe_prefers_driver_column_names() -> None:
    from aetherdialect._pipeline import build_result_dataframe

    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    df = build_result_dataframe(
        [(1,), (2,)],
        intent,
        "SELECT wrong_alias FROM left_t",
        column_names=("driver_id",),
    )
    assert df is not None
    assert list(df.columns) == ["driver_id"]


def test_unattributable_raw_sql_marks_plan_ineligible() -> None:
    from aetherdialect._contracts_base import NormalizedExpr
    from aetherdialect._federation import (
        _semijoin_reduction_stage_dependencies,
        federation_ineligible_answerable_hint,
        plan_federated_intent,
    )

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.1")
    schema_a = _graph("left_t", source_id="a")
    schema_b = _graph("right_t", source_id="b")
    schema = SchemaGraph(
        tables={**schema_a.tables, **schema_b.tables},
        join_paths_multi=recompute_join_paths_multi({**schema_a.tables, **schema_b.tables}),
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr(raw_sql="CUSTOM_UNPARSEABLE()"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, schema, manifest, mappings)
    assert plan.ineligible_reason == "expression contains unattributable raw_sql fragment"
    hint = federation_ineligible_answerable_hint(plan.ineligible_reason)
    assert hint is not None
    deps = _semijoin_reduction_stage_dependencies(manifest, {"a", "b"})
    assert deps.get("b") == ("member_a",)


def test_probe_federation_member_connections_executes_select_one() -> None:
    from aetherdialect._federation import probe_federation_member_connections

    mock_engine = MagicMock()
    mock_engine.dialect = "postgresql"
    mock_conn = MagicMock()
    mock_engine._execution_engine.connect.return_value.__enter__.return_value = mock_conn

    def fake_execute(stmt: object) -> MagicMock:
        sql = str(stmt)
        result = MagicMock()
        if "current_setting" in sql.lower():
            result.fetchone.return_value = ("UTC",)
        return result

    mock_conn.execute.side_effect = fake_execute
    probe_federation_member_connections({"storefront": mock_engine})
    assert mock_conn.execute.call_count == 2
    assert mock_engine._session_timezone == "UTC"


def test_rewrite_logical_references_updates_cte_column_maps() -> None:
    from aetherdialect._contracts_base import FederationMappings, LogicalTableMapping, LogicalTableMember
    from aetherdialect._contracts_core import RuntimeCteStep
    from aetherdialect._federation import _rewrite_logical_references

    schema = _graph("left_t", source_id="a")
    mappings = FederationMappings(
        version="0.2.1",
        logical_tables=[
            LogicalTableMapping(
                logical="left_t",
                members=(LogicalTableMember(source="a", table="left_t", columns={"logical_id": "id"}),),
                semantics="union",
            ),
        ],
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[
            RuntimeCteStep(
                cte_name="cte1",
                tables=["left_t"],
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
                column_map={"logical_id": "logical_id"},
            ),
        ],
    )
    rewritten = _rewrite_logical_references(intent, "a", mappings, schema)
    assert rewritten.cte_steps[0].column_map["logical_id"] == "id"


def test_partition_cte_steps_for_source_deep_copies_steps() -> None:
    from aetherdialect._contracts_core import RuntimeCteStep
    from aetherdialect._federation import _partition_cte_steps_for_source

    cte = RuntimeCteStep(
        cte_name="cte1",
        tables=["left_t"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
    )
    source_by_table = {"left_t": "a"}
    kept = _partition_cte_steps_for_source([cte], "a", source_by_table)
    assert kept
    assert kept[0] is not cte
    kept[0].column_map["mutated"] = "x"
    assert "mutated" not in cte.column_map


def test_result_columns_for_session_uses_federated_residual_headers() -> None:
    from aetherdialect._contracts_core import FederatedPlan, GenerationPath, ResidualSpec
    from aetherdialect._pipeline import result_columns_for_session

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.id")),),
        ),
    )
    rows = [(1,), (2,)]
    cols = result_columns_for_session(
        "-- source: a\nSELECT bad",
        rows,
        generation_path=GenerationPath.FEDERATION_PLAN,
        federated_plan=plan,
    )
    assert cols == ("id",)


def test_build_result_dataframe_uses_federated_residual_headers() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._pipeline import build_result_dataframe

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.id")),),
        ),
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    df = build_result_dataframe(
        [(1,), (2,)],
        intent,
        "-- source: a\nSELECT bad",
        generation_path=GenerationPath.FEDERATION_PLAN,
        federated_plan=plan,
    )
    assert df is not None
    assert list(df.columns) == ["id"]


def test_display_final_results_to_stdout_uses_federated_residual_headers() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._pipeline import display_final_results_to_stdout

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.id")),),
        ),
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with patch("aetherdialect._pipeline._final_display_sql_for_results", return_value="SELECT id"):
        with patch("aetherdialect._pipeline.print_query_result") as mock_print:
            display_final_results_to_stdout(
                "show ids",
                intent,
                "-- source: a\nSELECT bad",
                [(1,), (2,)],
                generation_path=GenerationPath.FEDERATION_PLAN,
                federated_plan=plan,
            )
    mock_print.assert_called_once()
    assert mock_print.call_args.kwargs["headers"] == ["id"]


def test_result_columns_for_session_skips_display_sql_on_federation_path() -> None:
    from aetherdialect._pipeline import result_columns_for_session

    rows = [(1,), (2,)]
    cols = result_columns_for_session(
        "-- source: a\nSELECT good_alias FROM left_t",
        rows,
        generation_path=GenerationPath.FEDERATION_PLAN,
    )
    assert cols == ("c0",)


def test_stamp_sql_shape_uses_federated_plan() -> None:
    from aetherdialect._contracts_core import FederatedPlan, GenerationPath, JoinSpec, ResidualSpec
    from aetherdialect._pipeline import stamp_sql_shape

    plan = FederatedPlan(
        steps=(),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="left_t.id",
                right_key="right_t.id",
                logical_key="id",
                kind="inner",
            ),
        ),
        residual=ResidualSpec(select_cols=(SelectCol(expr=NormalizedExpr.from_agg("count", "*")),)),
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "*"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    stamp_sql_shape(
        "-- display blob",
        intent,
        generation_path=GenerationPath.FEDERATION_PLAN,
        federated_plan=plan,
    )
    assert intent.sql_shape is not None
    assert intent.sql_shape.num_joins == 1
    assert intent.sql_shape.has_agg is True


def test_runtime_intent_planner_cte_names_round_trip() -> None:
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        planner_cte_names=["ranked", "totals"],
    )
    rebuilt = RuntimeIntent.from_dict(intent.to_dict())
    assert rebuilt.planner_cte_names == ["ranked", "totals"]


def test_extract_fuzzy_reuse_params_uses_intent_slots_not_sql() -> None:
    from aetherdialect._contracts_core import ConcreteIntent, WhereParam
    from aetherdialect._intent_process import NormalizedExpr
    from aetherdialect._pipeline import extract_fuzzy_reuse_params

    intent_sig = ConcreteIntent(
        intent_id="t1",
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.name"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                ),
            ]
        ),
    )
    template = Template(
        id="T0001",
        effective_structural_hash="eff1",
        intent_signature=intent_sig,
        intent_key="ik1",
        tables_used=["left_t"],
        sql_param="SELECT 1",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(
            param_values=[{"p1": "Horror"}],
            questions=["horror films"],
            natural_language=[],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )
    captured: dict[str, Any] = {}

    def fake_llm_chat(_system: str, user: str, *, task: str = "default") -> str:
        captured["user"] = user
        return '{"param_values": {"p1": "Comedy"}}'

    with patch("aetherdialect._pipeline.LLMProvider.chat", side_effect=fake_llm_chat):
        result = extract_fuzzy_reuse_params(
            "comedy films",
            template,
            history_index=0,
            literal_structural_only=True,
        )
    assert result == {"p1": "Comedy"}
    assert "parameterized_sql" not in captured["user"]
    assert "param_slots" in captured["user"]
    assert "left_t.name" in captured["user"]


def test_build_result_dataframe_uses_federated_plan_without_generation_path() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._pipeline import build_result_dataframe

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.customer_id")),),
        ),
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    df = build_result_dataframe(
        [(1,), (2,)],
        intent,
        "-- source: a\nSELECT misleading_alias FROM left_t",
        federated_plan=plan,
    )
    assert df is not None
    assert list(df.columns) == ["customer_id"]


def test_execute_reuse_with_params_uses_federated_residual_headers() -> None:
    from aetherdialect._contracts_core import ResidualSpec
    from aetherdialect._pipeline import execute_reuse_with_params

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.customer_id")),),
        ),
    )
    intent_sig = ConcreteIntent(
        intent_id="t1",
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    template = Template(
        id="T0001",
        effective_structural_hash="eff1",
        intent_signature=intent_sig,
        intent_key="ik1",
        tables_used=["left_t"],
        sql_param="SELECT customer_id FROM left_t",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(param_values=[{}], questions=["show customers"], natural_language=[]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )
    dialect = MagicMock()
    dialect.finalize_render.return_value = "SELECT customer_id FROM left_t"
    dialect.execute.return_value = [(1,), (2,)]
    left_tbl = TableMetadata(
        name="left_t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="a",
    )
    tables = {"left_t": left_tbl}
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_a_left_t",
        effective_structural_hash="eff_a_left_t",
    )
    store = TemplateOps.empty_template_store(schema.schema_graph_id)
    with patch("aetherdialect._pipeline._run_sql_validation_cascade", return_value=(True, None, None, ())):
        with patch("aetherdialect._pipeline._should_prompt_direct_reuse_user", return_value=False):
            with patch("aetherdialect._pipeline.print_query_result") as mock_print:
                outcome = execute_reuse_with_params(
                    "show customers",
                    template,
                    {},
                    dialect,
                    store,
                    {},
                    {},
                    schema,
                    reuse_path=GenerationPath.FEDERATION_PLAN,
                    prompt=False,
                    federated_plan=plan,
                )
    assert outcome is not None
    mock_print.assert_called_once()
    assert mock_print.call_args.kwargs["headers"] == ["customer_id"]


@pytest.mark.fast
def test_ask_phase_vocabulary_includes_decompose_and_combine() -> None:
    from aetherdialect._constants import (
        ASK_PHASE_H,
        ASK_PHASE_I,
        ASK_PHASE_J,
        ASK_PHASE_K,
        ASK_PHASE_L,
        ASK_PHASE_M,
        ASK_PHASE_N,
        FEDERATION_COMPOSITION_PHASE_A,
        FEDERATION_COMPOSITION_PHASE_H,
    )

    assert ASK_PHASE_H == "H:intent.finalize"
    assert ASK_PHASE_I == "I:plan.decompose"
    assert ASK_PHASE_J == "J:sql.build_joins"
    assert ASK_PHASE_K == "K:sql.validate_scope"
    assert ASK_PHASE_L == "L:sql.execute"
    assert ASK_PHASE_M == "M:plan.combine"
    assert ASK_PHASE_N == "N:feedback"
    assert FEDERATION_COMPOSITION_PHASE_A == "A:roster"
    assert FEDERATION_COMPOSITION_PHASE_H == "H:persist"


@pytest.mark.fast
def test_two_member_federation_fixture_declares_cross_source_join(two_member_federation) -> None:
    fed = two_member_federation
    assert fed.manifest.federation_id == "fed_two_member"
    assert len(fed.member_graphs) == 2
    assert fed.left_source in fed.member_graphs
    assert fed.right_source in fed.member_graphs
    assert fed.manifest.cross_source_joins
    assert fed.manifest.cross_source_joins[0].logical_key == "id"
    assert fed.composite.tables[fed.left_table].source_id == fed.left_source
    assert fed.composite.tables[fed.right_table].source_id == fed.right_source


@pytest.mark.fast
def test_display_final_results_skips_extract_column_headers_with_column_names() -> None:
    from unittest.mock import patch

    from aetherdialect._pipeline import display_final_results_to_stdout

    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with (
        patch("aetherdialect._pipeline.extract_column_headers") as mock_extract,
        patch("aetherdialect._pipeline.print_query_result") as mock_print,
    ):
        display_final_results_to_stdout(
            "show ids",
            intent,
            "SELECT bad_alias FROM left_t",
            [(1,), (2,)],
            column_names=("id",),
        )
    mock_extract.assert_not_called()
    assert mock_print.call_args.kwargs["headers"] == ["id"]
