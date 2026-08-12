"""Federation pipeline coordinator fan-out, cancellation, wave stages, and gate context."""

from __future__ import annotations

import threading
from contextvars import copy_context
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._constants import MASTER_AETHERSPACE_NAME
from aetherdialect._contracts_base import EngineContext, FederationJoinFanOutError, FederationRuntimeError
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    FederatedStage,
    FederationExecutionContext,
    JoinSpec,
    RuntimeIntent,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    federation_plan_combine_hash,
    federation_stage_execution_waves,
    order_federation_execution_steps,
    validate_coordinator_join_fan_out,
)
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline_execute import (
    _execute_federation_steps_parallel,
    execute_federated_prepare,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import (
    federation_turn_cancelled,
    pop_federation_execution_context,
    push_federation_execution_context,
)


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


def _parallel_manifest() -> Any:
    return parse_federation_manifest(
        {
            "federation_id": "fed_t58",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "connection": "conn_a", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "connection": "conn_b", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
            "coordinator": {"max_parallel_members": 2},
        },
        include_derived_roster=True,
    )


def _parallel_prepared(manifest: Any) -> tuple[FederatedPrepareOutcome, SchemaGraph]:
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent_a = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent_b = RuntimeIntent(
        tables=["right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    step_a = SourceStep(source_id="a", sub_intent=intent_a)
    step_b = SourceStep(source_id="b", sub_intent=intent_b)
    plan = FederatedPlan(
        steps=(step_a, step_b),
        combine=JoinSpec(
            left_source="a",
            right_source="b",
            left_key="id",
            right_key="id",
            logical_key="id",
            kind="inner",
        ),
        scope_sources=frozenset({"a", "b"}),
    )
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="SELECT a.id FROM left_t a JOIN right_t b ON a.id = b.id",
        glue_sql="SELECT * FROM src_a INNER JOIN src_b USING (id)",
        steps=(
            FederatedPreparedStep(source_id="a", sub_intent=intent_a, sql="SELECT id FROM left_t"),
            FederatedPreparedStep(source_id="b", sub_intent=intent_b, sql="SELECT id FROM right_t"),
        ),
        composite_schema_graph_id=str(composite.schema_graph_id),
        combine_hash=federation_plan_combine_hash(plan),
    )
    return prepared, composite


def _staged_plan_with_cte_and_coordinator() -> FederatedPlan:
    return FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["left_t"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["right_t"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
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
        stages=(
            FederatedStage(stage_id="member_a", kind="member", source_ids=("a",)),
            FederatedStage(stage_id="member_b", kind="member", source_ids=("b",)),
            FederatedStage(
                stage_id="coordinator_cte",
                kind="cte",
                source_ids=("a", "b"),
                depends_on=("member_a", "member_b"),
                spanning_cte_names=("span_cte",),
            ),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=("a", "b"),
                depends_on=("coordinator_cte",),
            ),
        ),
        scope_sources=frozenset({"a", "b"}),
    )


# --- Multi-edge combine fan-out ---


@pytest.mark.fast
def test_coordinator_join_fan_out_checks_second_combine_edge() -> None:
    """Fan-out on combine[1] must raise even when combine[0] passes."""
    plan = FederatedPlan(
        steps=(),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="left",
            ),
            JoinSpec(
                left_source="c",
                right_source="b",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="inner",
            ),
        ),
    )
    with pytest.raises(FederationJoinFanOutError) as exc_info:
        validate_coordinator_join_fan_out(plan, {"a": 2, "b": 4, "c": 4}, 2, combine_row_count=8)
    assert exc_info.value.phase == "coordinator"
    assert exc_info.value.source_id in {"b", "c"}


# --- Parallel cancellation via contextvars ---


@pytest.mark.fast
def test_parallel_member_submit_captures_contextvars() -> None:
    manifest = _parallel_manifest()
    prepared, composite = _parallel_prepared(manifest)
    prepared_by_source = {step.source_id: step for step in prepared.steps}
    copy_calls = 0
    original_copy = copy_context

    def _counting_copy() -> Any:
        nonlocal copy_calls
        copy_calls += 1
        return original_copy()

    def _member(_step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"id": [1]})

    with (
        patch("aetherdialect._pipeline_execute.copy_context", side_effect=_counting_copy),
        patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_member),
    ):
        _execute_federation_steps_parallel(
            prepared.plan.steps,
            prepared_by_source=prepared_by_source,
            composite_schema=composite,
            dialect_map={"a": MagicMock(), "b": MagicMock()},
            dialect=MagicMock(),
            manifest=manifest,
            q_norm="show joined rows",
            join_candidates={},
            cmap={},
            store=MagicMock(),
            gate_kwargs={},
            plan=prepared.plan,
        )

    assert copy_calls == len(prepared.plan.steps)


@pytest.mark.fast
def test_parallel_member_worker_observes_turn_cancellation() -> None:
    manifest = _parallel_manifest()
    prepared, composite = _parallel_prepared(manifest)
    prepared_by_source = {step.source_id: step for step in prepared.steps}
    cancelled_in_worker: list[bool] = []
    worker_threads: list[threading.Thread] = []
    ready_count = 0
    ready_lock = threading.Lock()
    all_workers_blocked = threading.Event()
    release_workers = threading.Event()
    main_thread = threading.current_thread()

    def _member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        nonlocal ready_count
        worker = threading.current_thread()
        assert worker is not main_thread
        worker_threads.append(worker)
        with ready_lock:
            ready_count += 1
            if ready_count == 2:
                all_workers_blocked.set()
        assert release_workers.wait(timeout=5.0)
        cancelled_in_worker.append(federation_turn_cancelled())
        return pd.DataFrame({"id": [1]})

    ctx = FederationExecutionContext(plan_id="parallel-cancel-t58")
    token = push_federation_execution_context(ctx)
    try:

        def _cancel_when_ready() -> None:
            assert all_workers_blocked.wait(timeout=5.0)
            ctx.cancel()
            release_workers.set()

        cancel_thread = threading.Thread(target=_cancel_when_ready)
        with patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_member):
            cancel_thread.start()
            with pytest.raises(FederationRuntimeError, match="cancelled"):
                _execute_federation_steps_parallel(
                    prepared.plan.steps,
                    prepared_by_source=prepared_by_source,
                    composite_schema=composite,
                    dialect_map={"a": MagicMock(), "b": MagicMock()},
                    dialect=MagicMock(),
                    manifest=manifest,
                    q_norm="show joined rows",
                    join_candidates={},
                    cmap={},
                    store=MagicMock(),
                    gate_kwargs={},
                    plan=prepared.plan,
                )
            cancel_thread.join(timeout=5.0)
    finally:
        pop_federation_execution_context(token)

    assert len(worker_threads) == 2
    assert cancelled_in_worker == [True, True]


# --- Derived cte/coordinator waves vs executor ---


@pytest.mark.fast
def test_federation_executor_honors_every_derived_wave_stage_kind() -> None:
    """Derived cte/coordinator waves use empty member_steps; executor must still emit phases for them."""
    plan = _staged_plan_with_cte_and_coordinator()
    ordered = order_federation_execution_steps(plan)
    waves = federation_stage_execution_waves(plan, ordered)
    derived_kinds = [wave.stage.kind for wave in waves if wave.stage.kind != "member"]
    assert derived_kinds == ["cte", "coordinator"]
    for wave in waves:
        if wave.stage.kind != "member":
            assert not wave.member_steps


@pytest.mark.fast
def test_federation_executor_emits_phase_for_derived_wave_stages() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_t59",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "connection": "conn_a", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "connection": "conn_b", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    plan = _staged_plan_with_cte_and_coordinator()
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="SELECT left_t.id FROM left_t JOIN right_t ON left_t.id = right_t.id",
        glue_sql="SELECT * FROM src_a INNER JOIN src_b USING (id)",
        steps=(
            FederatedPreparedStep(
                source_id="a",
                sub_intent=plan.steps[0].sub_intent,
                sql="SELECT id FROM left_t",
            ),
            FederatedPreparedStep(
                source_id="b",
                sub_intent=plan.steps[1].sub_intent,
                sql="SELECT id FROM right_t",
            ),
        ),
        composite_schema_graph_id=str(composite.schema_graph_id),
        combine_hash=federation_plan_combine_hash(plan),
    )
    emitted_stage_kinds: list[str] = []

    def _capture_emit(_phase: str, **kwargs: Any) -> None:
        stage = kwargs.get("stage")
        if stage is not None:
            emitted_stage_kinds.append(stage.kind)

    def _member(_step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"id": [1]})

    dialect = MagicMock()
    dialect.finalize_render.return_value = prepared.glue_sql
    with (
        patch("aetherdialect._pipeline_execute.emit_ask_phase", side_effect=_capture_emit),
        patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_member),
        patch(
            "aetherdialect._pipeline_execute.execute_federation_coordinator",
            return_value=pd.DataFrame({"id": [1]}),
        ),
    ):
        execute_federated_prepare(
            prepared,
            composite,
            dialect=dialect,
            dialects_by_source={"a": dialect, "b": dialect},
            manifest=manifest,
        )

    waves = federation_stage_execution_waves(plan, order_federation_execution_steps(plan))
    expected_derived = {wave.stage.kind for wave in waves if wave.stage.kind != "member"}
    assert expected_derived <= set(emitted_stage_kinds), (
        f"executor never emitted phases for derived wave stages: {sorted(expected_derived - set(emitted_stage_kinds))}"
    )


# --- Composite context must not bleed into master-bound members ---


@pytest.mark.fast
def test_federation_gate_kwargs_master_member_uses_member_context() -> None:
    composite_ctx = EngineContext(deny_objects=frozenset({"secret_table"}))
    owner = MagicMock()
    owner._schema_role = "owner"
    owner._context_name = MASTER_AETHERSPACE_NAME
    owner._runtime_config = MagicMock(engine_context=composite_ctx, execution_context=None)
    owner._federation_source_runtimes = {}
    owner._artifacts_root = None
    port = MagicMock(
        _owner=owner,
        execution_visible_objects=None,
        space_tables=None,
        space_columns=None,
    )
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_t60",
            "sources": [
                {"source_id": "alpha", "engine": "duckdb", "context": "master", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "alpha"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    gates = MainExecutionOps.federation_gate_kwargs_by_source(owner, port, manifest)
    member_gate = gates["alpha"]
    assert member_gate["context_name"] == MASTER_AETHERSPACE_NAME
    assert member_gate["schema_context"] == EngineContext()
    assert member_gate["schema_context"] is not composite_ctx


@pytest.mark.fast
def test_consumer_sql_gate_kwargs_does_not_supply_composite_to_master_members() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    composite_ctx = EngineContext(deny_objects=frozenset({"orders"}))
    owner = MagicMock()
    owner._schema_role = "owner"
    owner._context_name = MASTER_AETHERSPACE_NAME
    owner._runtime_config = MagicMock(engine_context=composite_ctx, execution_context=None)
    port = MagicMock(
        _owner=owner,
        execution_visible_objects=None,
        space_tables=None,
        space_columns=None,
    )
    kwargs = MainExecutionOps.consumer_sql_gate_kwargs(port)
    assert kwargs["schema_context"] is composite_ctx
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_t60b",
            "sources": [{"source_id": "alpha", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"entity_a": "alpha"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    gates = MainExecutionOps.federation_gate_kwargs_by_source(owner, port, manifest)
    assert gates["alpha"]["schema_context"] == EngineContext()
    assert gates["alpha"]["schema_context"] is not composite_ctx
