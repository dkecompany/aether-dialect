"""Cross-thread and async federation cancellation contract tests."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import aetherdialect
from aetherdialect import AsyncPipelineSession
from aetherdialect._contracts_base import FederationPartialFailureError, FederationRuntimeError
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    FederatedStage,
    FederationExecutionContext,
    JoinSpec,
    LlmExecutionConfig,
    RuntimeIntent,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import federation_plan_combine_hash
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._main_session import PipelineSession
from aetherdialect._pipeline_execute import (
    _execute_federation_steps_parallel,
    _raise_partial_member_failure,
    execute_federated_prepare,
    prepare_federated_sql_plan,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import (
    LLM_EXECUTION_CONTEXT,
    federation_turn_cancelled,
    llm_execution_scope,
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
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST = {
    "federation_id": "fed_cancel",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def _staged_prepared() -> tuple[FederatedPrepareOutcome, SchemaGraph, Any]:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
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
        stages=(
            FederatedStage(stage_id="member_a", kind="member", source_ids=("a",)),
            FederatedStage(stage_id="member_b", kind="member", source_ids=("b",), depends_on=("member_a",)),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=("a", "b"),
                depends_on=("member_a", "member_b"),
            ),
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
    return prepared, composite, manifest


@pytest.mark.fast
def test_cancel_with_no_active_turn_returns_false() -> None:
    session = PipelineSession(MagicMock())
    assert session.cancel() is False


@pytest.mark.fast
def test_session_cross_thread_cancel_stops_at_stage_boundary() -> None:
    prepared, composite, manifest = _staged_prepared()
    session = PipelineSession(MagicMock())
    first_entered = threading.Event()
    release_first = threading.Event()
    started_sources: list[str] = []

    def _slow_member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        started_sources.append(step.source_id)
        if step.source_id == "a":
            first_entered.set()
            assert release_first.wait(timeout=5.0)
            return pd.DataFrame({"id": [1]})
        return pd.DataFrame({"id": [1]})

    error_box: list[BaseException] = []

    def _run_turn() -> None:
        try:
            with (
                patch("aetherdialect._federation_execute.revalidate_prepared_federation_plan"),
                patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_slow_member),
                patch(
                    "aetherdialect._federation_execute.execute_federation_coordinator",
                    return_value=pd.DataFrame({"id": [1]}),
                ),
            ):
                execute_federated_prepare(
                    prepared,
                    composite,
                    dialect=MagicMock(),
                    dialects_by_source={"a": MagicMock(), "b": MagicMock()},
                    manifest=manifest,
                    turn_session=session,
                )
        except BaseException as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_run_turn)
    worker.start()
    assert first_entered.wait(timeout=5.0)
    assert session.cancel() is True
    release_first.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert len(error_box) == 1
    assert isinstance(error_box[0], FederationRuntimeError)
    assert "cancelled" in str(error_box[0]).lower()
    assert started_sources == ["a"]
    assert session.active_federation_execution_context is None


@pytest.mark.fast
def test_async_cancel_during_in_flight_ask() -> None:
    prepared, composite, manifest = _staged_prepared()
    session = PipelineSession(MagicMock())
    async_sess = AsyncPipelineSession(session)
    first_entered = threading.Event()
    release_first = threading.Event()

    def _slow_member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        if step.source_id == "a":
            first_entered.set()
            assert release_first.wait(timeout=5.0)
            return pd.DataFrame({"id": [1]})
        return pd.DataFrame({"id": [1]})

    def _blocking_ask(_question: str) -> Any:
        with (
            patch("aetherdialect._federation_execute.revalidate_prepared_federation_plan"),
            patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_slow_member),
            patch(
                "aetherdialect._federation_execute.execute_federation_coordinator",
                return_value=pd.DataFrame({"id": [1]}),
            ),
        ):
            return execute_federated_prepare(
                prepared,
                composite,
                dialect=MagicMock(),
                dialects_by_source={"a": MagicMock(), "b": MagicMock()},
                manifest=manifest,
                turn_session=session,
            )

    async def _run() -> None:
        ask_task = asyncio.create_task(asyncio.to_thread(_blocking_ask, "q"))
        assert await asyncio.to_thread(first_entered.wait, 5.0)
        assert await async_sess.cancel() is True
        release_first.set()
        with pytest.raises(FederationRuntimeError, match="cancelled"):
            await ask_task

    asyncio.run(_run())
    assert session.active_federation_execution_context is None


@pytest.mark.fast
def test_partial_failure_internal_cancel_path() -> None:
    ctx = FederationExecutionContext(plan_id="partial")
    token = push_federation_execution_context(ctx)
    try:
        assert not federation_turn_cancelled()
        with pytest.raises(FederationPartialFailureError):
            _raise_partial_member_failure(
                RuntimeError("member blew up"),
                source_id="a",
                phase="member",
                succeeded=(),
            )
        assert federation_turn_cancelled()
    finally:
        pop_federation_execution_context(token)


@pytest.mark.fast
def test_cancel_is_cooperative_between_stages_not_mid_statement() -> None:
    prepared, composite, manifest = _staged_prepared()
    session = PipelineSession(MagicMock())
    first_entered = threading.Event()
    release_first = threading.Event()
    cancel_during_first = threading.Event()
    completed_sources: list[str] = []

    def _slow_member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        if step.source_id == "a":
            first_entered.set()
            assert cancel_during_first.wait(timeout=5.0)
            assert session.active_federation_execution_context is not None
            assert session.active_federation_execution_context.cancelled is True
            assert release_first.wait(timeout=5.0)
            completed_sources.append("a")
            return pd.DataFrame({"id": [1]})
        completed_sources.append(step.source_id)
        return pd.DataFrame({"id": [1]})

    error_box: list[BaseException] = []

    def _run_turn() -> None:
        try:
            with (
                patch("aetherdialect._federation_execute.revalidate_prepared_federation_plan"),
                patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_slow_member),
                patch(
                    "aetherdialect._federation_execute.execute_federation_coordinator",
                    return_value=pd.DataFrame({"id": [1]}),
                ),
            ):
                execute_federated_prepare(
                    prepared,
                    composite,
                    dialect=MagicMock(),
                    dialects_by_source={"a": MagicMock(), "b": MagicMock()},
                    manifest=manifest,
                    turn_session=session,
                )
        except BaseException as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_run_turn)
    worker.start()
    assert first_entered.wait(timeout=5.0)
    assert session.cancel() is True
    cancel_during_first.set()
    release_first.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert completed_sources == ["a"]
    assert len(error_box) == 1
    assert isinstance(error_box[0], FederationRuntimeError)
    assert "cancelled" in str(error_box[0]).lower()


@pytest.mark.fast
def test_prepare_federated_sql_plan_raises_when_turn_cancelled() -> None:
    prepared, composite, manifest = _staged_prepared()
    ctx = FederationExecutionContext(plan_id="prepare-cancel")
    token = push_federation_execution_context(ctx)
    try:
        ctx.cancel()
        with pytest.raises(FederationRuntimeError, match="cancelled"):
            prepare_federated_sql_plan(
                "show joined rows",
                prepared.plan,
                composite,
                dialect=MagicMock(),
                dialects_by_source={"a": MagicMock(), "b": MagicMock()},
                join_candidates={},
                cmap={},
                store=MagicMock(),
                manifest=manifest,
                member_graphs={"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
            )
    finally:
        pop_federation_execution_context(token)


def _parallel_manifest() -> Any:
    return parse_federation_manifest(
        {
            "federation_id": "fed_parallel_ctx",
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


@pytest.mark.fast
def test_parallel_member_worker_observes_cancellation_context() -> None:
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

    ctx = FederationExecutionContext(plan_id="parallel-cancel")
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


@pytest.mark.fast
def test_parallel_member_worker_inherits_llm_execution_scope() -> None:
    manifest = _parallel_manifest()
    prepared, composite = _parallel_prepared(manifest)
    prepared_by_source = {step.source_id: step for step in prepared.steps}
    cfg = LlmExecutionConfig(
        azure_endpoint="https://parallel.example",
        azure_api_key="k",
        azure_api_version="v",
        deployment_light="D4O",
        deployment_heavy="D54",
        max_query_cost_rows=1,
        max_query_cost_bytes=1,
        statement_timeout_ms=1,
        llm_timeout_ms=4242,
        profile_timeout_ms=1,
        explain_timeout_ms=None,
    )
    llm_in_worker: list[LlmExecutionConfig | None] = []
    worker_threads: list[threading.Thread] = []
    main_thread = threading.current_thread()

    def _member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        worker = threading.current_thread()
        assert worker is not main_thread
        worker_threads.append(worker)
        llm_in_worker.append(LLM_EXECUTION_CONTEXT.get())
        return pd.DataFrame({"id": [1]})

    with llm_execution_scope(cfg):
        with patch("aetherdialect._pipeline_execute._execute_federation_source_step", side_effect=_member):
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

    assert len(worker_threads) == 2
    assert llm_in_worker == [cfg, cfg]


@pytest.mark.fast
def test_parallel_member_pool_shutdown_cancels_pending_futures() -> None:
    from concurrent.futures import ThreadPoolExecutor

    shutdown_calls: list[dict[str, Any]] = []
    original_shutdown = ThreadPoolExecutor.shutdown

    def _record_shutdown(self: ThreadPoolExecutor, *args: Any, **kwargs: Any) -> None:
        shutdown_calls.append(dict(kwargs))
        return original_shutdown(self, *args, **kwargs)

    prepared, composite, manifest = _staged_prepared()
    prepared_by_source = {step.source_id: step for step in prepared.steps}

    def _member(_step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"id": [1]})

    with (
        patch.object(ThreadPoolExecutor, "shutdown", _record_shutdown),
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
    assert shutdown_calls
    assert shutdown_calls[-1].get("cancel_futures") is True


@pytest.mark.fast
def test_public_surface_drops_module_helpers_keeps_exceptions() -> None:
    assert "cancel" not in aetherdialect.__all__
    with pytest.raises(ImportError):
        exec(
            "from aetherdialect import cancel",
            {"__builtins__": __builtins__},
        )
    assert aetherdialect.FederationMemberExecutionError is not None
    assert aetherdialect.FederationCapExceededError is not None
