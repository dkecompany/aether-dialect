"""Regression tests for federation composition, coordinator, session, security, artifacts, and planning."""

from __future__ import annotations

import hashlib
import json
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_base import (
    ConfigError,
    FederationMappings,
    FederationPlanTemplate,
    OwnerOnlyOperationError,
    PredicateGroup,
    SpaceContext,
)
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    GenerationPath,
    JoinSpec,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    SqlExecuteSuspendContext,
    SqlGenerationOutcome,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationConfigError,
    FederationInvariantError,
    FederationRuntimeError,
    _apply_coordinator_probe_joins,
    _assert_combine_join_plan_structure,
    _dataframe_memory_bytes,
    _looks_aggregated,
    _normalize_stored_member_hash_row,
    _qualified_ref_source_id,
    _schema_column_duckdb_type,
    assert_composite_invariants,
    clear_federated_turn_state,
    compose_composite_graph,
    execute_federation_coordinator,
    federation_plan_combine_hash,
    federation_plan_is_degenerate,
    federation_plan_matches_template,
    federation_plan_sql_shape,
    federation_plan_step_fingerprints,
    federation_table_set,
    mappings_replay_matches,
    parse_federation_manifest,
    parse_federation_mappings,
    plan_federated_intent,
    prune_federation_mappings,
    reconcile_composite_classifications,
    render_federation_glue,
    revalidate_prepared_federation_plan,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import execute_federated_prepare
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import anti_join_presence_column
from aetherdialect._utils import intent_key
from tests.federation_helpers import union_member_graph_pair


def _table(name: str, *, source_id: str, id_type: str = "integer") -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": ColumnMetadata(name="id", data_type=id_type, sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _graph(
    table: str,
    *,
    source_id: str,
    id_type: str = "integer",
    ddl_probe_hash: str = "",
) -> SchemaGraph:
    tables = {table: _table(table, source_id=source_id, id_type=id_type)}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
        ddl_probe_hash=ddl_probe_hash,
    )


_MANIFEST = {
    "federation_id": "fed_regression",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}

_UNION_MANIFEST = {
    "federation_id": "fed_union_regression",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"payment_a": "a", "payment_b": "b"},
    "cross_source_joins": [],
}


def _union_mappings() -> FederationMappings:
    return parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )


@pytest.mark.fast
def test_plan_federated_intent_denied_column_returns_space_ineligible_reason() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {
            "a": _graph("left_t", source_id="a", id_type="integer"),
            "b": _graph("right_t", source_id="b", id_type="integer"),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    space = SpaceContext(deny_columns=frozenset({"left_t.id"}))
    plan = plan_federated_intent(intent, composite, manifest, space=space)
    assert plan.ineligible_reason is not None
    assert "space" in plan.ineligible_reason.lower()


@pytest.mark.fast
def test_member_ddl_probe_hash_changes_composite_identity() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members_a = {
        "a": _graph("left_t", source_id="a", ddl_probe_hash="probe_a"),
        "b": _graph("right_t", source_id="b", ddl_probe_hash="probe_b"),
    }
    members_b = {
        "a": _graph("left_t", source_id="a", ddl_probe_hash="probe_a_changed"),
        "b": _graph("right_t", source_id="b", ddl_probe_hash="probe_b"),
    }
    composite_a = compose_composite_graph(members_a, manifest)
    composite_b = compose_composite_graph(members_b, manifest)
    assert composite_a.ddl_probe_hash
    assert composite_b.ddl_probe_hash
    assert composite_a.ddl_probe_hash != composite_b.ddl_probe_hash
    assert composite_a.schema_graph_id != composite_b.schema_graph_id


@pytest.mark.fast
def test_compose_sets_ddl_probe_hash_from_member_probes() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "a": _graph("left_t", source_id="a", ddl_probe_hash="alpha"),
        "b": _graph("right_t", source_id="b", ddl_probe_hash="beta"),
    }
    composite = compose_composite_graph(members, manifest)
    probe_blob = json.dumps(
        sorted((sid, str(graph.ddl_probe_hash or "")) for sid, graph in members.items()),
        separators=(",", ":"),
    )
    expected = hashlib.sha256(probe_blob.encode()).hexdigest()[:32]
    assert composite.ddl_probe_hash == expected
    assert_composite_invariants(composite, members, manifest, FederationMappings(version="0.2.1"))
    assert composite.structural_hash
    assert composite.scope_hash
    assert composite.effective_structural_hash


@pytest.mark.fast
def test_render_join_coerces_typed_join_keys() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "CAST" in glue.upper()
    assert "BIGINT" in glue.upper()


def _union_payment_graphs() -> dict[str, SchemaGraph]:
    return union_member_graph_pair("payment_a", "payment_b")


@pytest.mark.fast
def test_union_scalar_glue_aggregates_member_counts() -> None:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    composite = compose_composite_graph(_union_payment_graphs(), manifest, mappings)
    intent = RuntimeIntent(
        tables=["payment"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "UNION ALL" in glue.upper()
    assert "COUNT" in glue.upper()
    assert plan.stages[-1].stage_id == "coordinator_scalar"


@pytest.mark.fast
def test_scalar_cross_source_plan_emits_coordinator_scalar_stage() -> None:
    from aetherdialect._federation import plan_federated_stages

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    stages = plan_federated_stages(
        {"a", "b"},
        plan.steps,
        intent=intent,
        source_by_table={"left_t": "a", "right_t": "b"},
        manifest=manifest,
        residual=plan.residual,
    )
    assert stages[-1].stage_id == "coordinator_scalar"
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "NULLIF" in glue.upper()
    assert "SUM" in glue.upper()
    assert "COUNT" in glue.upper()


@pytest.mark.fast
def test_clear_federated_turn_state_clears_session_slots() -> None:
    from aetherdialect._main_execution import PipelineSession

    owner = MagicMock()
    session = PipelineSession(owner)
    session._pending_federation_plan_template = FederationPlanTemplate(
        plan_id="p1",
        composite_schema_graph_id="cg",
        intent_key="ik",
        step_fingerprints=(),
        combine_hash="h",
    )
    clear_federated_turn_state(session)
    assert session._pending_federation_plan_template is None


@pytest.mark.fast
def test_sql_execute_suspend_context_freezes_turn_policy_and_exec_context() -> None:
    snap = MagicMock()
    gen_out = SqlGenerationOutcome("sql", True, GenerationPath.FEDERATION_PLAN, None)
    fed_prep = FederatedPrepareOutcome(success=True, plan=FederatedPlan(steps=()), display_sql="sql")
    policy = MainExecutionOps.snapshot_turn_policy()
    ctx = MainExecutionOps._sql_execute_suspend_context(
        snap,
        "sql",
        None,
        gen_out,
        None,
        False,
        RuntimeIntent(
            tables=["left_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        federated_prepare=fed_prep,
        federation_plan_id="plan_x",
        federation_exec_context={"q_norm": "how many", "join_candidates": {}},
        turn_policy=policy,
    )
    assert ctx.federated_prepare is fed_prep
    assert ctx.federation_plan_id == "plan_x"
    assert ctx.turn_policy is policy
    assert dict(ctx.federation_exec_context) == {"q_norm": "how many", "join_candidates": {}}


@pytest.mark.fast
def test_verify_federation_execute_resume_accepts_matching_plan_id() -> None:
    snap = MagicMock()
    gen_out = SqlGenerationOutcome(
        "sql",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        federation_plan_id="plan_match",
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
        federation_plan_id="plan_match",
    )
    MainExecutionOps._verify_federation_execute_resume(ctx)


@pytest.mark.fast
def test_federation_apply_migration_map_requires_owner() -> None:
    with patch("aetherdialect.aetherdialect.initialize_aether_federation"):
        fed = AetherFederation(
            "fed_gate",
            members={"conn_a": MagicMock(), "conn_b": MagicMock()},
            declaration_file="/tmp/aether_fed_gate_declaration.json",
        )
    fed._schema_role = "consumer"
    with pytest.raises(OwnerOnlyOperationError, match="apply_migration_map"):
        fed.apply_migration_map(path="/tmp/federation_migration_map.json")


@pytest.mark.fast
def test_revalidate_prepared_plan_rejects_combine_hash_drift() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="",
        composite_schema_graph_id=str(composite.schema_graph_id),
        combine_hash="stale_combine_hash",
        step_fingerprints=federation_plan_step_fingerprints(plan, intent_key_fn=intent_key),
    )
    with pytest.raises(FederationInvariantError, match="combine specification changed"):
        revalidate_prepared_federation_plan(prepared, composite, manifest=manifest)


@pytest.mark.fast
def test_revalidate_prepared_plan_rejects_step_fingerprint_drift() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest)
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    fingerprints = federation_plan_step_fingerprints(
        plan,
        intent_key_fn=intent_key,
        manifest=manifest,
        member_graphs=members,
    )
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="",
        composite_schema_graph_id=str(composite.schema_graph_id),
        combine_hash=federation_plan_combine_hash(plan),
        step_fingerprints=fingerprints,
    )
    drifted_members = {
        "a": _graph("left_t", source_id="a"),
        "b": _graph("right_t", source_id="b"),
    }
    drifted_members["a"].schema_graph_id = "sg_drifted_member"
    with pytest.raises(FederationInvariantError, match="step fingerprints changed"):
        revalidate_prepared_federation_plan(
            prepared,
            composite,
            manifest=manifest,
            member_graphs=drifted_members,
            intent_key_fn=intent_key,
        )


@pytest.mark.fast
def test_federation_plan_matches_template_rejects_stale_fingerprints() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    fingerprints = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key)
    template = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik",
        step_fingerprints=(("a", "stale_fp"),),
        combine_hash=federation_plan_combine_hash(plan),
    )
    assert (
        federation_plan_matches_template(
            plan,
            template,
            step_fingerprints=fingerprints,
        )
        is False
    )
    matching_template = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id=str(composite.schema_graph_id),
        intent_key="ik",
        step_fingerprints=fingerprints,
        combine_hash=federation_plan_combine_hash(plan),
    )
    assert (
        federation_plan_matches_template(
            plan,
            matching_template,
            step_fingerprints=fingerprints,
        )
        is True
    )


@pytest.mark.fast
def test_execute_federated_prepare_degenerate_skips_coordinator() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_degen",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"left_t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph({"a": _graph("left_t", source_id="a")}, manifest)
    sub_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="a", sub_intent=sub_intent),))
    assert federation_plan_is_degenerate(plan)
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="SELECT id FROM left_t",
        steps=(
            FederatedPreparedStep(
                source_id="a",
                sub_intent=sub_intent,
                sql="SELECT id FROM left_t",
                structural_defaults={},
            ),
        ),
        composite_schema_graph_id=str(composite.schema_graph_id),
        combine_hash=federation_plan_combine_hash(plan),
    )
    dialect = MagicMock()
    dialect.finalize_render.return_value = "SELECT id FROM left_t"
    with patch("aetherdialect._pipeline.execute_guarded_sql", return_value=[(1,), (2,)]) as exec_mock:
        outcome = execute_federated_prepare(
            prepared,
            composite,
            dialect=dialect,
            dialects_by_source={"a": dialect},
        )
    exec_mock.assert_called_once()
    assert outcome.bundle is not None
    assert "-- coordinator" not in outcome.bundle.display_sql.lower()


@pytest.mark.fast
def test_federation_table_set_widens_join_path_endpoints() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["right_t"],
    )
    table_set = federation_table_set(intent, composite, manifest)
    assert "left_t" in table_set.tables
    assert "right_t" in table_set.tables
    assert table_set.sources == frozenset({"a", "b"})


@pytest.mark.fast
def test_union_logical_table_plans_member_steps_and_union_specs() -> None:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    composite = compose_composite_graph(_union_payment_graphs(), manifest, mappings)
    intent = RuntimeIntent(
        tables=["payment"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert {step.source_id for step in plan.steps} == {"a", "b"}
    assert plan.union_specs
    assert plan.union_specs[0].logical_table == "payment"
    assert federation_plan_is_degenerate(plan) is False
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "UNION ALL" in glue.upper()


@pytest.mark.fast
def test_mappings_replay_fingerprint_detects_member_drift() -> None:
    from aetherdialect._federation import (
        check_federation_member_drift_at_turn_start,
        mappings_replay_matches,
        persist_federation_tree,
    )

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.1")
    members = {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest, mappings)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        assert mappings_replay_matches(tmp, members, manifest, mappings)
        drifted = dict(members)
        drifted["a"] = _graph("left_t", source_id="a")
        drifted["a"].schema_graph_id = "sg_drifted_replay"
        assert not mappings_replay_matches(tmp, drifted, manifest, mappings)
        owner = MagicMock()
        owner._is_aether_federation = True
        owner._federation_manifest = manifest
        owner._federation_member_graphs = drifted
        owner._federation_storage_dir = tmp
        owner._federation_mappings = mappings
        with pytest.raises(FederationInvariantError, match="member graphs changed"):
            check_federation_member_drift_at_turn_start(owner)


@pytest.mark.fast
def test_federation_result_contract_kwargs_prefers_bundle_then_residual() -> None:
    from aetherdialect._contracts_core import FederatedSqlBundle, ResidualSpec
    from aetherdialect._intent_process import NormalizedExpr
    from aetherdialect._main_execution import MainExecutionOps

    plan = FederatedPlan(
        steps=(),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.customer_id")),),
        ),
    )
    gen_out = SqlGenerationOutcome(
        "",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        (),
        None,
        0,
        (),
    )
    prep = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="-- display",
    )
    bundle = FederatedSqlBundle(statements=(), display_sql="-- display", column_names=("driver_id",))
    kwargs = MainExecutionOps._federation_result_contract_kwargs(
        gen_out,
        federated_prepare=prep,
        federated_bundle=bundle,
    )
    assert kwargs["column_names"] == ("driver_id",)
    empty_bundle = FederatedSqlBundle(statements=(), display_sql="-- display", column_names=())
    kwargs_residual = MainExecutionOps._federation_result_contract_kwargs(
        gen_out,
        federated_prepare=prep,
        federated_bundle=empty_bundle,
    )
    assert kwargs_residual["column_names"] == ("customer_id",)


@pytest.mark.fast
def test_completed_step_uses_federated_bundle_column_names() -> None:
    from aetherdialect._contracts_core import FederatedSqlBundle
    from aetherdialect._main_execution import PipelineSession

    owner = MagicMock()
    owner._schema_graph = None
    owner._audit_emit = MagicMock()
    sess = PipelineSession(owner)
    sess._turn_question = "show customers"
    bundle = FederatedSqlBundle(
        statements=(),
        display_sql="-- source: a\nSELECT bad_alias FROM left_t",
        column_names=("customer_id",),
    )
    sess._last_turn_outcome = {
        "outcome": "success",
        "error": None,
        "sql": bundle.display_sql,
        "rows": [(1,), (2,)],
        "columns": None,
        "federated_bundle": bundle,
        "generation_path": GenerationPath.FEDERATION_PLAN,
    }
    step = sess._completed_step()
    assert step.data is not None
    assert list(step.data.columns) == ["customer_id"]
    assert step.sql is None
    assert step.federated_bundle is bundle
    assert bundle.display_sql
    audit_calls = [call.args[0] for call in owner._audit_emit.call_args_list]
    assert "ask_done" in audit_calls
    done_call = next(call for call in owner._audit_emit.call_args_list if call.args[0] == "ask_done")
    detail_keys = {key for key, _value in done_call.kwargs["details"]}
    assert "result_columns" in detail_keys


@pytest.mark.fast
def test_flat_or_chain_across_sources_is_ineligible() -> None:
    from aetherdialect._contracts_core import WhereParam

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
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
                    value_type="integer",
                    raw_value=1,
                ),
                WhereParam(
                    left_expr=NormalizedExpr.from_column("right_t.id"),
                    op="=",
                    value_type="integer",
                    raw_value=2,
                ),
            ),
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is not None
    assert plan.ineligible_reason.startswith("cross-source predicate disjunction is not supported:")
    assert "left_t.id" in plan.ineligible_reason
    assert "right_t.id" in plan.ineligible_reason


@pytest.mark.fast
def test_same_source_filter_group_disjunction_remains_eligible() -> None:
    from aetherdialect._contracts_core import WhereParam

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op="=",
                    value_type="integer",
                    raw_value=1,
                ),
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op="=",
                    value_type="integer",
                    raw_value=2,
                ),
            ]
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert "filter_group disjunction" not in (plan.ineligible_reason or "")
    assert "cross-source OR filter" not in (plan.ineligible_reason or "")


@pytest.mark.fast
def test_logical_table_union_decomposes_without_ir_set_op() -> None:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    composite = compose_composite_graph(_union_payment_graphs(), manifest, mappings)
    intent = RuntimeIntent(
        tables=["payment"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert plan.union_specs
    assert plan.union_specs[0].semantics == "union"
    assert set(plan.union_specs[0].member_source_ids) == {"a", "b"}


@pytest.mark.fast
def test_combine_join_validation_uses_plan_structure_not_glue_text() -> None:
    from aetherdialect._contracts_core import JoinSpec

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert isinstance(plan.combine, tuple) and plan.combine
    _assert_combine_join_plan_structure(plan)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"})
    glue_no_join_token = glue.replace(" JOIN ", " ").replace(" join ", " ")
    assert " JOIN " not in glue_no_join_token.upper()
    frames = {
        "a": pd.DataFrame({"id": [1, 2]}),
        "b": pd.DataFrame({"id": [2, 3]}),
    }
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    assert len(result) == 1
    broken = FederatedPlan(
        steps=plan.steps,
        combine=(
            JoinSpec(
                left_source="a",
                right_source="orphan",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="inner",
            ),
        ),
    )
    with pytest.raises(FederationRuntimeError, match="missing declared edges|missing join edges"):
        _assert_combine_join_plan_structure(broken)


@pytest.mark.fast
def test_parenthesized_non_aggregate_select_is_not_treated_as_aggregate() -> None:
    sc = SelectCol(expr=NormalizedExpr.from_column("coalesce(left_t.id, 0)"))
    assert "(" in (sc.expr.primary_term or "")
    assert sc.is_aggregated is False
    assert _looks_aggregated(sc) is False
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["left_t"],
                    grain="many",
                    select_cols=[sc],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
    )
    shape = federation_plan_sql_shape(plan)
    assert shape.has_agg is False


@pytest.mark.fast
def test_interval_schema_type_does_not_map_to_integer() -> None:
    assert _schema_column_duckdb_type("interval") == "INTERVAL"
    assert _schema_column_duckdb_type("integer") == "INTEGER"
    assert _schema_column_duckdb_type("int") == "INTEGER"


@pytest.mark.fast
def test_dataframe_memory_measurement_failure_does_not_undercount() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3]})
    with patch.object(pd.DataFrame, "memory_usage", side_effect=RuntimeError("mem probe failed")):
        with pytest.raises(RuntimeError, match="mem probe failed"):
            _dataframe_memory_bytes(frame)
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    frames = {
        "a": pd.DataFrame({"id": [1, 2]}),
        "b": pd.DataFrame({"id": [2]}),
    }
    with patch("aetherdialect._federation._dataframe_memory_bytes", side_effect=RuntimeError("cap probe")):
        with pytest.raises(RuntimeError, match="cap probe"):
            execute_federation_coordinator(frames, plan, row_cap=100, total_input_byte_cap=1)


@pytest.mark.fast
def test_reconcile_classification_failure_surfaces() -> None:
    from aetherdialect._contracts_base import LogicalTableMapping, LogicalTableMember

    a_table = TableMetadata(
        name="payment",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="a",
        description="Payments A",
    )
    b_table = TableMetadata(
        name="payment",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="b",
        description="Payments B",
    )
    members = {
        "a": SchemaGraph(
            tables={"payment": a_table}, join_paths_multi=recompute_join_paths_multi({"payment": a_table})
        ),
        "b": SchemaGraph(
            tables={"payment": b_table}, join_paths_multi=recompute_join_paths_multi({"payment": b_table})
        ),
    }
    composite_table = TableMetadata(
        name="payment",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        column_member_sources={"id": ("a", "b")},
    )
    composite = SchemaGraph(
        tables={"payment": composite_table},
        join_paths_multi=recompute_join_paths_multi({"payment": composite_table}),
    )
    mappings = FederationMappings(
        version="0.2.1",
        logical_tables=(
            LogicalTableMapping(
                logical="payment",
                semantics="union",
                members=(
                    LogicalTableMember(source="a", table="payment", columns={"id": "id"}),
                    LogicalTableMember(source="b", table="payment", columns={"id": "id"}),
                ),
            ),
        ),
    )

    def _boom(_schema: SchemaGraph, _notes: str | None = None) -> dict:
        raise RuntimeError("classifier down")

    with pytest.raises(FederationRuntimeError, match="classification reconciliation failed"):
        reconcile_composite_classifications(
            composite,
            members,
            mappings,
            notes_content="notes",
            llm_classify=_boom,
        )


@pytest.mark.fast
def test_unresolvable_qualified_ref_does_not_yield_empty_source_id() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    with pytest.raises(ConfigError):
        _qualified_ref_source_id("missing_table.id", manifest)
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_columns": [
                {
                    "logical": "shared_id",
                    "members": ["left_t.id", "ghost_t.id"],
                    "unify_in_graph": True,
                },
            ],
        },
    )
    with pytest.raises(ConfigError):
        prune_federation_mappings(mappings, manifest, active_source_ids={"a", "b"})


@pytest.mark.fast
def test_corrupt_member_hash_row_does_not_false_match() -> None:
    with pytest.raises(FederationConfigError, match="corrupt federation member hash row"):
        _normalize_stored_member_hash_row(["only_one_field"])
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")}
    mappings = FederationMappings(version="0.2.1")
    composite = compose_composite_graph(members, manifest)
    with tempfile.TemporaryDirectory() as tmp:
        from aetherdialect._federation import federation_artifact_paths, persist_federation_tree

        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        paths = federation_artifact_paths(tmp)
        with open(paths["artifact_manifest"], encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["federation_members"].append(["corrupt"])
        with open(paths["artifact_manifest"], "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with pytest.raises(FederationConfigError, match="corrupt federation member hash row"):
            mappings_replay_matches(tmp, members, manifest, mappings)


@pytest.mark.fast
def test_cross_source_anti_join_lift_coordinator_presence_null_and_row_count(two_member_federation) -> None:
    fed = two_member_federation
    probe_name = "absent_right"
    anti_cte = RuntimeCteStep(
        cte_name=probe_name,
        emission="anti_join",
        tables=[fed.right_table],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.right_table}.id"))],
        output_columns=["id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    sub_left = RuntimeIntent(
        tables=[fed.left_table],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.left_table}.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    sub_right = RuntimeIntent(
        tables=[fed.right_table],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.right_table}.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(
            SourceStep(source_id=fed.left_source, sub_intent=sub_left),
            SourceStep(source_id=fed.right_source, sub_intent=sub_right),
        ),
        combine=(
            JoinSpec(
                left_source=fed.left_source,
                right_source=fed.right_source,
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="left",
            ),
        ),
        lifted_probe_ctes=(anti_cte,),
        grain="many",
        scope_sources=frozenset({fed.left_source, fed.right_source}),
    )

    step_ids = {fed.left_source: "src_a", fed.right_source: "src_b"}
    glue = render_federation_glue(plan, step_ids, schema=fed.composite)
    marker = anti_join_presence_column(probe_name)
    assert "IS NULL" in glue.upper()
    assert marker in glue

    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.register("src_a", pd.DataFrame({"id": [1, 2, 3]}))
    conn.register("src_b", pd.DataFrame({"id": [2]}))
    source_by_table = {fed.left_table: fed.left_source, fed.right_table: fed.right_source}
    lifted_sql = _apply_coordinator_probe_joins(
        "SELECT id FROM src_a",
        (anti_cte,),
        step_ids,
        source_by_table,
    )
    assert marker in lifted_sql
    rows = conn.execute(lifted_sql).fetchall()
    assert len(rows) == 2
    assert sorted(row[0] for row in rows) == [1, 3]
