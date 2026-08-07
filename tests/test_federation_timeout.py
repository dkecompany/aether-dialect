"""Coordinator glue execution and whole-plan wall-clock timeouts."""

from __future__ import annotations

import inspect
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationCapExceededError, FederationPartialFailureError
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    FederatedStage,
    JoinSpec,
    RuntimeIntent,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    compose_composite_graph,
    execute_federation_coordinator,
    federation_plan_combine_hash,
    parse_federation_manifest,
    plan_federated_intent,
)
from aetherdialect._pipeline import execute_federated_prepare
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
    "federation_id": "fed_timeout_l30",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
    "coordinator": {
        "coordinator_timeout_ms": 12_345,
        "plan_timeout_ms": 67_890,
    },
}


def _join_plan() -> tuple[FederatedPlan, SchemaGraph, object]:
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
    return plan, composite, manifest


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
def test_execute_federation_coordinator_accepts_coordinator_timeout_ms() -> None:
    params = inspect.signature(execute_federation_coordinator).parameters
    assert "coordinator_timeout_ms" in params


@pytest.mark.fast
def test_coordinator_glue_execution_uses_timeout_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import aetherdialect._federation

    assert hasattr(aetherdialect._federation, "_execute_coordinator_sql_with_timeout")
    plan, composite, _manifest = _join_plan()
    frames = {
        "a": pd.DataFrame({"id": [1, 2]}),
        "b": pd.DataFrame({"id": [2, 3]}),
    }
    captured: dict[str, int | None] = {}

    def _recording(conn: object, sql: str, bind_map: dict[str, object] | None, *, timeout_ms: int | None) -> object:
        captured["timeout_ms"] = timeout_ms
        return conn.execute(sql, bind_map or {})

    monkeypatch.setattr(aetherdialect._federation, "_execute_coordinator_sql_with_timeout", _recording)
    result = execute_federation_coordinator(
        frames,
        plan,
        row_cap=100,
        coordinator_timeout_ms=12_345,
    )
    assert len(result) == 1
    assert captured["timeout_ms"] == 12_345


@pytest.mark.fast
def test_coordinator_glue_timeout_raises_cap_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    import aetherdialect._federation

    plan, composite, _manifest = _join_plan()
    frames = {
        "a": pd.DataFrame({"id": [1]}),
        "b": pd.DataFrame({"id": [1]}),
    }

    def _timeout(conn: object, sql: str, bind_map: dict[str, object] | None, *, timeout_ms: int | None) -> object:
        raise FederationCapExceededError(
            "federation coordinator glue timeout exceeded after 1ms",
            limit_key="coordinator_timeout_ms",
            source_id="coordinator",
        )

    monkeypatch.setattr(aetherdialect._federation, "_execute_coordinator_sql_with_timeout", _timeout)
    with pytest.raises(FederationCapExceededError, match="coordinator glue timeout exceeded") as exc_info:
        execute_federation_coordinator(
            frames,
            plan,
            row_cap=100,
            coordinator_timeout_ms=1,
        )
    assert exc_info.value.limit_key == "coordinator_timeout_ms"
    assert exc_info.value.source_id == "coordinator"


@pytest.mark.fast
def test_execute_federated_prepare_passes_manifest_coordinator_timeout() -> None:
    prepared, composite, manifest = _staged_prepared()
    captured: dict[str, int | None] = {}

    def _coordinator(
        frames: dict[str, object],
        plan: FederatedPlan,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["coordinator_timeout_ms"] = kwargs.get("coordinator_timeout_ms")
        return pd.DataFrame({"id": [1]})

    with (
        patch("aetherdialect._pipeline.revalidate_prepared_federation_plan"),
        patch(
            "aetherdialect._pipeline._execute_federation_source_step",
            return_value=pd.DataFrame({"id": [1]}),
        ),
        patch("aetherdialect._pipeline.execute_federation_coordinator", side_effect=_coordinator),
    ):
        execute_federated_prepare(
            prepared,
            composite,
            dialect=MagicMock(),
            dialects_by_source={"a": MagicMock(), "b": MagicMock()},
            manifest=manifest,
            turn_session=MagicMock(),
        )
    assert captured["coordinator_timeout_ms"] == 12_345


@pytest.mark.fast
def test_execute_federated_prepare_enforces_plan_timeout() -> None:
    prepared, composite, manifest = _staged_prepared()
    perf_calls = {"n": 0}

    def _fake_perf() -> float:
        perf_calls["n"] += 1
        return 0.0 if perf_calls["n"] <= 6 else 70.0

    with (
        patch("aetherdialect._pipeline.revalidate_prepared_federation_plan"),
        patch("aetherdialect._pipeline.time.perf_counter", side_effect=_fake_perf),
        patch(
            "aetherdialect._pipeline._execute_federation_source_step",
            return_value=pd.DataFrame({"id": [1]}),
        ),
    ):
        with pytest.raises(FederationPartialFailureError, match="plan timeout exceeded") as exc_info:
            execute_federated_prepare(
                prepared,
                composite,
                dialect=MagicMock(),
                dialects_by_source={"a": MagicMock(), "b": MagicMock()},
                manifest=manifest,
                turn_session=MagicMock(),
            )
    assert exc_info.value.phase == "member"
    assert exc_info.value.__cause__ is not None
    assert "plan timeout exceeded" in str(exc_info.value.__cause__)


@pytest.mark.fast
def test_execute_coordinator_sql_with_timeout_interrupts_hanging_query() -> None:
    from aetherdialect._federation import _execute_coordinator_sql_with_timeout

    hang = threading.Event()
    release = threading.Event()
    interrupted = threading.Event()

    class _FakeResult:
        def fetchdf(self) -> pd.DataFrame:
            return pd.DataFrame({"id": [1]})

    class _FakeConn:
        def execute(self, sql: str, params: dict[str, object] | None = None) -> _FakeResult:
            hang.set()
            release.wait(timeout=5.0)
            return _FakeResult()

        def interrupt(self) -> None:
            interrupted.set()
            release.set()

    worker_error: list[BaseException] = []

    def _run() -> None:
        try:
            _execute_coordinator_sql_with_timeout(_FakeConn(), "SELECT 1", {}, timeout_ms=50)
        except BaseException as exc:
            worker_error.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    assert hang.wait(timeout=5.0)
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert worker_error
    assert isinstance(worker_error[0], FederationCapExceededError)
    assert "coordinator glue timeout exceeded" in str(worker_error[0])
    assert interrupted.is_set()
