"""Construction- and ask-phase progress callbacks for integrator UIs."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants_runtime import ASK_PHASE_B, SCHEMA_BUILD_PHASE_C, SCHEMA_BUILD_PHASE_E
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_core import PhaseProgressEvent
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect import Dialect
from aetherdialect._main_session import PipelineSession
from aetherdialect._schema_finalize import build_schema_graph_with_diff
from aetherdialect._utils import emit_ask_phase, emit_construction_phase
from tests.test_aetherdialect import _make_aether_stub


class _FullBuildDialect(Dialect):
    """Dialect stub that forces a full reflect + profile build."""

    name = "stub"

    def __init__(self, reflect_result: SchemaGraph) -> None:
        super().__init__(MagicMock())
        self._reflect_result = reflect_result
        self.reflect_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return ""

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        self.reflect_calls += 1
        return self._reflect_result

    def profile_schema(self, sg: SchemaGraph) -> None:
        return None

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.mark.fast
def test_phase_progress_event_exposes_source_not_source_id() -> None:
    """Progress events carry ``source`` and never ``source_id``."""
    names = {f.name for f in fields(PhaseProgressEvent)}
    assert "source" in names
    assert "source_id" not in names
    ev = PhaseProgressEvent(phase="A:test", timestamp_iso="2026-01-01T00:00:00+00:00", source="storefront")
    assert ev.source == "storefront"


@pytest.mark.fast
def test_emit_construction_phase_invokes_callback() -> None:
    events: list[PhaseProgressEvent] = []

    def capture(ev: PhaseProgressEvent) -> None:
        events.append(ev)

    from aetherdialect._utils import pop_construction_phase_callback, push_construction_phase_callback

    token = push_construction_phase_callback(capture)
    try:
        emit_construction_phase(SCHEMA_BUILD_PHASE_C)
    finally:
        pop_construction_phase_callback(token)

    assert len(events) == 1
    assert events[0].phase == SCHEMA_BUILD_PHASE_C
    assert events[0].timestamp_iso


@pytest.mark.fast
def test_emit_ask_phase_reports_source_and_stage() -> None:
    events: list[PhaseProgressEvent] = []

    def capture(ev: PhaseProgressEvent) -> None:
        events.append(ev)

    from aetherdialect._utils import pop_ask_phase_callback, push_ask_phase_callback

    token = push_ask_phase_callback(capture)
    try:
        emit_ask_phase(ASK_PHASE_B, source="catalog", stage=2)
    finally:
        pop_ask_phase_callback(token)

    assert len(events) == 1
    assert events[0].phase == ASK_PHASE_B
    assert events[0].source == "catalog"
    assert events[0].stage == 2
    assert not hasattr(events[0], "source_id")


@pytest.mark.fast
def test_emit_ask_phase_sets_elapsed_ms() -> None:
    events: list[PhaseProgressEvent] = []

    def capture(ev: PhaseProgressEvent) -> None:
        events.append(ev)

    from aetherdialect._utils import pop_ask_phase_callback, pop_turn_timing, push_ask_phase_callback, push_turn_timing

    callback_token = push_ask_phase_callback(capture)
    timing_tokens = push_turn_timing()
    try:
        emit_ask_phase(ASK_PHASE_B)
        emit_ask_phase(ASK_PHASE_B)
    finally:
        pop_turn_timing(*timing_tokens)
        pop_ask_phase_callback(callback_token)

    assert len(events) == 2
    assert events[0].elapsed_ms is not None and events[0].elapsed_ms >= 0
    assert events[1].elapsed_ms is not None and events[1].elapsed_ms >= 0


@pytest.mark.fast
def test_build_schema_graph_emits_reflect_and_profile_phases(
    schema_graph: SchemaGraph,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full schema construction reports slow reflect and profile phases."""
    from aetherdialect._config import EngineConfig

    cache_path = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", cache_path)

    events: list[PhaseProgressEvent] = []

    def capture(ev: PhaseProgressEvent) -> None:
        events.append(ev)

    dialect = _FullBuildDialect(schema_graph)
    from aetherdialect._utils import pop_construction_phase_callback, push_construction_phase_callback

    token = push_construction_phase_callback(capture)
    try:
        with patch("aetherdialect._schema_finalize.apply_column_roles_llm"):
            build_schema_graph_with_diff(dialect, EngineContext(), notes_content="notes")
    finally:
        pop_construction_phase_callback(token)

    phases = [ev.phase for ev in events]
    assert SCHEMA_BUILD_PHASE_C in phases
    assert SCHEMA_BUILD_PHASE_E in phases
    for ev in events:
        assert not hasattr(ev, "source_id")


@pytest.mark.fast
def test_pipeline_session_wires_phase_callback() -> None:
    """An ask turn invokes the owner's phase callback."""
    events: list[PhaseProgressEvent] = []
    engine = _make_aether_stub(_phase_callback=lambda ev: events.append(ev))
    session = PipelineSession(engine)

    def ask_side_effect(*_args: object, **_kwargs: object) -> None:
        emit_ask_phase(ASK_PHASE_B)

    with patch("aetherdialect._main_init.MainInitOps.interactive_run_once", side_effect=ask_side_effect):
        session.ask("how many rentals")

    assert any(ev.phase == ASK_PHASE_B for ev in events)
