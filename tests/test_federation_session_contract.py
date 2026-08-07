"""Regression tests for federated reuse, repair scoping, telemetry, attribution, and turn ownership."""

from __future__ import annotations

import json
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY,
)
from aetherdialect._contracts_base import InferenceTag
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPrepareOutcome,
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
from aetherdialect._federation import parse_federation_manifest
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._intent_repair import expand_shared_pk_tables_for_refs
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import handle_direct_sql_reuse
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps


def _member_table(name: str, source_id: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def test_expand_shared_pk_skips_cross_source_parent() -> None:
    left_t = TableMetadata(
        name="left_t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
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
                inference_tag=InferenceTag.CROSS_SOURCE,
            ),
        ],
        source_id="a",
    )
    parent_t = _member_table("parent_t", "a")
    right_t = _member_table("right_t", "b")
    tables = {"left_t": left_t, "parent_t": parent_t, "right_t": right_t}
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent_t.name"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    out = expand_shared_pk_tables_for_refs(intent, schema)
    assert "parent_t" in out.tables
    assert "right_t" not in out.tables


def test_handle_direct_sql_reuse_blocked_when_federation_manifest_active() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_reuse",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    tables = {"t": _member_table("t", "a")}
    schema = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    tmpl = Template(
        id="tmpl1",
        effective_structural_hash=schema.effective_structural_hash,
        intent_signature=RuntimeIntent(
            tables=["t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="ik",
        tables_used=["t"],
        sql_param="SELECT id FROM t WHERE id = :p1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="c",
        value_history=ValueHistory(
            param_values=[{"p1": 1}],
            questions=["how many t"],
            natural_language=["how many t"],
        ),
        stats=TemplateStats(accept=1, reject=0),
    )
    store = TemplateOps.empty_template_store(schema.schema_graph_id)
    templates = {tmpl.id: tmpl}
    with patch("aetherdialect._pipeline._try_federation_plan_question_reuse", return_value=None):
        with patch("aetherdialect._pipeline.execute_reuse_with_params") as exec_mock:
            result = handle_direct_sql_reuse(
                "how many t",
                tmpl,
                MagicMock(),
                store,
                templates,
                {},
                schema,
                federation_manifest=manifest,
            )
    exec_mock.assert_not_called()
    assert result is None


def test_sql_execute_suspend_context_carries_federated_prepare() -> None:
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
    )
    assert ctx.federated_prepare is fed_prep
    assert ctx.federation_plan_id == "plan1"


def test_prepare_member_failure_emits_federation_diagnostic() -> None:
    from aetherdialect._pipeline import prepare_federated_sql_plan

    sub_intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="west", sub_intent=sub_intent),))
    tables = {"t": _member_table("t", "west")}
    schema = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    store = TemplateOps.empty_template_store(schema.schema_graph_id)
    member_graphs = {"west": schema}
    with patch("aetherdialect._pipeline.generate_and_validate_sql") as gen_mock:
        gen_mock.return_value = SqlGenerationOutcome(
            "",
            False,
            GenerationPath.INTENT_DIRECT_MATCH,
            None,
            sql_validation_error="bad sql",
            error_kind="validation_failed",
        )
        with patch("aetherdialect._pipeline.notify") as notify_mock:
            outcome = prepare_federated_sql_plan(
                "q",
                plan,
                schema,
                dialect=MagicMock(),
                dialects_by_source={"west": MagicMock()},
                join_candidates={},
                cmap={},
                store=store,
                stores_by_source={"west": store},
                member_graphs=member_graphs,
            )
    assert outcome.success is False
    assert outcome.source_id == "west"
    assert outcome.phase == "prepare"
    notify_mock.assert_called()
    assert notify_mock.call_args.kwargs.get("code") == DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED
    assert notify_mock.call_args.kwargs.get("source_id") == "west"


def test_prepared_federated_outcome_reads_suspend_not_engine_slot() -> None:
    from aetherdialect.aetherdialect import AetherFederation

    with patch("aetherdialect.aetherdialect.initialize_aether_federation"):
        fed = AetherFederation(
            "fed_surface",
            members={"conn_a": MagicMock(), "conn_b": MagicMock()},
            declaration_file="/tmp/aether_fed_surface_declaration.json",
        )
    assert fed.prepared_federated_outcome() is None


@pytest.mark.fast
def test_concurrent_federation_sessions_keep_distinct_prepared_plans() -> None:
    from concurrent.futures import ThreadPoolExecutor

    def _suspend_ctx(source_id: str, q_norm: str) -> tuple[MagicMock, object]:
        sub_intent = RuntimeIntent(
            tables=[f"{source_id}_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        plan = FederatedPlan(steps=(SourceStep(source_id=source_id, sub_intent=sub_intent),))
        fed_prep = FederatedPrepareOutcome(success=True, plan=plan, display_sql=f"display-{source_id}")
        gen_out = SqlGenerationOutcome(
            f"display-{source_id}",
            True,
            GenerationPath.FEDERATION_PLAN,
            None,
            federation_plan_id=f"plan-{source_id}",
        )
        snap = MagicMock()
        snap.q_norm = q_norm
        snap.store = {}
        snap.templates = {}
        snap.rejected = {}
        snap.schema_terms = set()
        snap.dialect = MagicMock()
        snap.schema = MagicMock()
        session = MagicMock()
        owner = MagicMock()
        owner._federation_manifest = MagicMock()
        owner._federation_dialects = {}
        owner._federation_source_runtimes = {}
        owner._federation_storage_dir = None
        session._owner = owner
        ctx = MainExecutionOps._sql_execute_suspend_context(
            snap,
            f"display-{source_id}",
            None,
            gen_out,
            None,
            False,
            sub_intent,
            federated_prepare=fed_prep,
            federation_plan_id=f"plan-{source_id}",
            federation_exec_context={"q_norm": q_norm},
        )
        return session, ctx

    session_a, ctx_a = _suspend_ctx("west", "how many west orders")
    session_b, ctx_b = _suspend_ctx("east", "how many east orders")
    captured: dict[str, FederatedPrepareOutcome] = {}

    def _resume(session: MagicMock, ctx: object, label: str) -> None:
        with patch(
            "aetherdialect._main_execution.MainExecutionOps._run_sql_execution_for_gen_out",
            side_effect=lambda **kwargs: captured.__setitem__(label, kwargs["federated_prepare"]) or ([(1,)], None),
        ):
            with patch("aetherdialect._main_execution.MainExecutionOps._offer_sql_feedback_after_execute"):
                MainExecutionOps._complete_interactive_execute(ctx, "y", choice_port=session)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_resume, session_a, ctx_a, "west"),
            pool.submit(_resume, session_b, ctx_b, "east"),
        ]
        for fut in futures:
            fut.result()

    assert captured["west"].plan.steps[0].source_id == "west"
    assert captured["east"].plan.steps[0].source_id == "east"
    assert captured["west"] is not captured["east"]


def test_federation_plan_replay_diagnostic_code_exists() -> None:
    assert DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY == "FEDERATION_PLAN_REPLAY"


@pytest.mark.fast
def test_member_feedback_q_norm_scopes_member_store_keys() -> None:
    from aetherdialect._federation import member_feedback_q_norm

    assert member_feedback_q_norm("west", "how many orders") == "west::how many orders"
    assert member_feedback_q_norm("west", "west::how many orders") == "west::how many orders"


@pytest.mark.fast
def test_credit_federation_accept_raises_when_member_store_missing() -> None:
    from aetherdialect._contracts_core import FederatedPreparedStep
    from aetherdialect._federation import FederationConfigError
    from aetherdialect._pipeline import credit_federation_accept

    sub_intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    tmpl = Template(
        id="tmpl1",
        effective_structural_hash="h1",
        intent_signature=sub_intent,
        intent_key="ik",
        tables_used=["t"],
        sql_param="SELECT 1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="c",
        value_history=ValueHistory(param_values=[], questions=[], natural_language=[]),
        stats=TemplateStats(accept=0, reject=0),
    )
    step = FederatedPreparedStep(
        source_id="west",
        sub_intent=sub_intent,
        sql="SELECT 1",
        matched_template=tmpl,
    )
    with pytest.raises(FederationConfigError, match="member store missing for source_id 'west'"):
        credit_federation_accept(
            q_norm="how many t",
            federation_dir="",
            plan_id="plan1",
            steps=(step,),
            stores_by_source={},
        )


@pytest.mark.fast
def test_drain_write_queue_applies_member_tree_events(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from datetime import datetime

    from aetherdialect._config import EngineConfig
    from aetherdialect._contracts_base import WriteQueueEvent
    from aetherdialect._contracts_core import FeedbackKind, QuestionFeedbackEntry, RejectionBucket
    from aetherdialect._core_utils import emit_write_queue_event
    from aetherdialect._main_execution import MainExecutionOps
    from aetherdialect._templates import TemplateOps

    composite_dir = tmp_path / "fed"
    member_dir = tmp_path / "member_west"
    composite_dir.mkdir()
    member_dir.mkdir()
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(composite_dir / "intent_templates"))

    member_graph = SchemaGraph(
        tables={"t": _member_table("t", "west")},
        join_paths_multi=recompute_join_paths_multi({"t": _member_table("t", "west")}),
        schema_graph_id="member_west",
        effective_structural_hash="member_west",
    )
    member_store = TemplateOps.empty_template_store("member_west")
    monkeypatch.setattr(
        "aetherdialect._templates.TemplateOps.load_template_store",
        lambda *_a, **_k: member_store,
    )
    saves: list[int] = []
    monkeypatch.setattr(
        "aetherdialect._templates.TemplateOps.save_template_store",
        lambda _s: saves.append(1),
    )

    owner = MagicMock()
    owner._is_aether_federation = True
    owner._schema_graph = MagicMock(schema_graph_id="composite")
    owner._store = TemplateOps.empty_template_store("composite")
    owner._templates = {}
    owner._rejected = {}
    owner._dialect = None
    owner._artifacts_dir = str(composite_dir)
    owner._audit_emit = MagicMock()
    owner._federation_source_runtimes = {"west": MagicMock(artifacts_dir=str(member_dir), dialect=None)}
    owner._federation_member_graphs = {"west": member_graph}

    ts = datetime.now(UTC).isoformat()
    entry = QuestionFeedbackEntry(
        summary="s",
        buckets=(RejectionBucket.OTHER,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash="member_west",
        intent_structural_hash="ik",
        intent_payload="{}",
        created_at=ts,
        updated_at=ts,
    )
    ev = WriteQueueEvent(
        kind="feedback_record",
        schema_graph_id="member_west",
        schema_hash="member_west",
        produced_at=ts,
        payload=(("q_norm", "west::how many t"), ("entry_json", json.dumps(entry.to_dict()))),
    )
    emit_write_queue_event(str(member_dir), ev)

    applied = MainExecutionOps.drain_write_queue(owner, str(composite_dir))
    assert applied == 1
    assert "west::how many t" in member_store.question_feedback
    assert saves == [1]


@pytest.mark.fast
def test_residual_group_by_excludes_keys_not_needed_post_join() -> None:
    """Grouped cross-source residual must not copy unattributable group_by keys."""
    from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
    from aetherdialect._federation import compose_composite_graph, plan_federated_intent
    from aetherdialect._schema_graph import recompute_join_paths_multi

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_plan_items_gb",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [
                {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    tables_a = {
        "t_a": TableMetadata(
            name="t_a",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        )
    }
    tables_b = {
        "t_b": TableMetadata(
            name="t_b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "name": ColumnMetadata(name="name", data_type="text", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id="b",
        )
    }
    composite = compose_composite_graph(
        {
            "a": SchemaGraph(tables=tables_a, join_paths_multi=recompute_join_paths_multi(tables_a)),
            "b": SchemaGraph(tables=tables_b, join_paths_multi=recompute_join_paths_multi(tables_b)),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="grouped",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t_a.id"))],
        group_by_cols=[NormalizedExpr.from_column("t_b.name"), NormalizedExpr()],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert plan.residual is not None
    assert [g.column_ref for g in plan.residual.group_by_cols] == ["t_b.name"]
    assert plan.residual.select_cols
