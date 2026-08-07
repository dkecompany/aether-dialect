"""Cancellation must stop in-flight federation member and coordinator statements with a structured outcome."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationTurnCancelledError
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
from aetherdialect._core_utils import pop_federation_execution_context, push_federation_execution_context
from aetherdialect._federation import compose_composite_graph, federation_plan_combine_hash, parse_federation_manifest
from aetherdialect._main_execution import PipelineSession
from aetherdialect._pipeline import _execute_federation_steps_parallel, execute_federated_prepare
from aetherdialect._schema_graph import recompute_join_paths_multi


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
    "federation_id": "fed_cancel_l38",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
    "coordinator": {"max_parallel_members": 2},
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
def test_cancellation_raises_structured_turn_cancelled_error() -> None:
    prepared, composite, manifest = _staged_prepared()
    session = PipelineSession(MagicMock())
    first_entered = threading.Event()
    release_first = threading.Event()

    def _slow_member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        if step.source_id == "a":
            first_entered.set()
            release_first.wait(timeout=5.0)
        return pd.DataFrame({"id": [1]})

    with (
        patch("aetherdialect._pipeline.revalidate_prepared_federation_plan"),
        patch("aetherdialect._pipeline._execute_federation_source_step", side_effect=_slow_member),
        patch(
            "aetherdialect._pipeline.execute_federation_coordinator",
            return_value=pd.DataFrame({"id": [1]}),
        ),
    ):
        worker_error: list[BaseException] = []

        def _run() -> None:
            try:
                execute_federated_prepare(
                    prepared,
                    composite,
                    dialect=MagicMock(),
                    dialects_by_source={"a": MagicMock(), "b": MagicMock()},
                    manifest=manifest,
                    turn_session=session,
                )
            except BaseException as exc:
                worker_error.append(exc)

        worker = threading.Thread(target=_run)
        worker.start()
        assert first_entered.wait(timeout=5.0)
        assert session.cancel_active_federation_turn() is True
        worker.join(timeout=5.0)

    assert len(worker_error) == 1
    exc = worker_error[0]
    assert type(exc) is FederationTurnCancelledError
    assert exc.source_id
    assert exc.phase in {"member", "coordinator"}


@pytest.mark.fast
def test_parallel_cancel_calls_cancel_statement_on_in_flight_members() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    prepared, composite, _manifest = _staged_prepared()
    prepared_by_source = {step.source_id: step for step in prepared.steps}
    dialect_a = MagicMock()
    dialect_b = MagicMock()
    entered = threading.Event()
    hang = threading.Event()

    def _member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        entered.set()
        hang.wait(timeout=5.0)
        return pd.DataFrame({"id": [1]})

    ctx = FederationExecutionContext(plan_id="parallel-cancel-l38")
    token = push_federation_execution_context(ctx)
    try:

        def _cancel_when_ready() -> None:
            entered.wait(timeout=5.0)
            ctx.cancel()
            hang.set()

        cancel_thread = threading.Thread(target=_cancel_when_ready)
        with patch("aetherdialect._pipeline._execute_federation_source_step", side_effect=_member):
            cancel_thread.start()
            with pytest.raises(FederationTurnCancelledError):
                _execute_federation_steps_parallel(
                    prepared.plan.steps,
                    prepared_by_source=prepared_by_source,
                    composite_schema=composite,
                    dialect_map={"a": dialect_a, "b": dialect_b},
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

    assert dialect_a.cancel_statement.called or dialect_b.cancel_statement.called


@pytest.mark.fast
def test_sequential_staged_cancel_calls_cancel_statement() -> None:
    prepared, composite, manifest = _staged_prepared()
    session = PipelineSession(MagicMock())
    member_dialect = MagicMock()
    first_entered = threading.Event()
    release_first = threading.Event()

    def _slow_member(step: SourceStep, **_kwargs: Any) -> pd.DataFrame:
        if step.source_id == "a":
            first_entered.set()
            release_first.wait(timeout=5.0)
        return pd.DataFrame({"id": [1]})

    with (
        patch("aetherdialect._pipeline.revalidate_prepared_federation_plan"),
        patch("aetherdialect._pipeline._execute_federation_source_step", side_effect=_slow_member),
        patch(
            "aetherdialect._pipeline.execute_federation_coordinator",
            return_value=pd.DataFrame({"id": [1]}),
        ),
    ):
        worker_error: list[BaseException] = []

        def _run() -> None:
            try:
                execute_federated_prepare(
                    prepared,
                    composite,
                    dialect=MagicMock(),
                    dialects_by_source={"a": member_dialect, "b": MagicMock()},
                    manifest=manifest,
                    turn_session=session,
                )
            except BaseException as exc:
                worker_error.append(exc)

        worker = threading.Thread(target=_run)
        worker.start()
        assert first_entered.wait(timeout=5.0)
        assert session.cancel_active_federation_turn() is True
        worker.join(timeout=5.0)

    assert len(worker_error) == 1
    assert type(worker_error[0]) is FederationTurnCancelledError
    member_dialect.cancel_statement.assert_called()


@pytest.mark.fast
def test_sqlalchemy_backend_cancel_invokes_connection_cancel() -> None:
    from aetherdialect._dialect_sqlglot_helper import SqlAlchemyResultBackend

    raw_conn = MagicMock()
    sa_conn = MagicMock()
    sa_conn.connection = raw_conn
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = sa_conn
    engine.connect.return_value.__exit__.return_value = None
    backend = SqlAlchemyResultBackend(engine, dialect_name="postgresql")

    hang = threading.Event()
    started = threading.Event()

    def _slow_execute(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        hang.wait(timeout=2.0)
        result = MagicMock()
        result.fetchmany.side_effect = [[(1,)], []]
        return result

    sa_conn.execute.side_effect = _slow_execute

    def _run() -> None:
        backend.fetch_rows("SELECT pg_sleep(30)")

    worker = threading.Thread(target=_run)
    worker.start()
    assert started.wait(timeout=2.0)
    backend.cancel_statement()
    hang.set()
    worker.join(timeout=2.0)
    raw_conn.cancel.assert_called_once()
