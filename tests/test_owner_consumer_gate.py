"""Tests for owner versus consumer role gates and override proposal queueing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import (
    EngineContext,
    NormalizedExpr,
    OwnerOnlyOperationError,
    WriteQueueEvent,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._main_execution import PipelineSession, drain_write_queue
from aetherdialect._schema_graph import assert_consumer_intent_in_scope
from aetherdialect._templates import empty_template_store


def _make_engine(*, role: str = "owner", graph_id: str = "sg_test000000000001__abcd1234") -> AetherEngine:
    obj = AetherEngine.__new__(AetherEngine)
    obj._schema_role = role
    obj._consumer_visible_objects = frozenset({"a"}) if role == "consumer" else None
    obj._schema_graph = MagicMock()
    obj._schema_graph.schema_graph_id = graph_id
    obj._schema_graph.effective_structural_hash = "eff"
    obj._artifacts_dir = Path("/tmp/artifacts")
    obj._store = empty_template_store(graph_id)
    obj._templates = {}
    obj._rejected = {}
    obj._dialect = None
    obj._pipeline_writer_lock = __import__("threading").Lock()
    obj._runtime_config = MagicMock()
    obj._runtime_config.llm_execution = None
    obj._audit_sink = None
    return obj


class TestOwnerConsumerGate:
    def test_consumer_writer_session_raises(self) -> None:
        t = _make_engine(role="consumer")
        with pytest.raises(OwnerOnlyOperationError, match="writer"):
            with t.session(mode="writer"):
                pass

    def test_consumer_reader_session_allowed(self) -> None:
        t = _make_engine(role="consumer")
        with t.session(mode="reader") as sess:
            assert isinstance(sess, PipelineSession)
            assert sess.visible_objects is None
            assert sess.execution_visible_objects == frozenset({"a"})

    def test_apply_migration_map_requires_owner(self) -> None:
        with pytest.raises(OwnerOnlyOperationError):
            AetherEngine.apply_migration_map(
                engine_context=EngineContext(),
                artifacts_dir="/tmp/x",
                role="consumer",
            )

    def test_clear_template_store_requires_owner(self) -> None:
        t = _make_engine(role="consumer")
        with pytest.raises(OwnerOnlyOperationError):
            t.clear_template_store()

    def test_export_schema_overrides_requires_owner(self) -> None:
        t = _make_engine(role="consumer")
        with pytest.raises(OwnerOnlyOperationError):
            t.export_schema_overrides()

    def test_consumer_apply_schema_overrides_enqueues_proposal(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        doc = {"tables": {}}
        (tmp_path / "schema_overrides.json").write_text(json.dumps(doc), encoding="utf-8")
        t = _make_engine(role="consumer")
        t._artifacts_dir = tmp_path
        emitted: list[WriteQueueEvent] = []

        def _capture(_adir: str, ev: WriteQueueEvent) -> None:
            emitted.append(ev)

        monkeypatch.setattr("aetherdialect.aetherdialect.emit_write_queue_event", _capture)
        t.apply_schema_overrides()
        assert len(emitted) == 1
        assert emitted[0].kind == "override_proposal"
        assert emitted[0].schema_graph_id == "sg_test000000000001__abcd1234"

    def test_owner_drain_applies_override_proposal(self, tmp_path, monkeypatch) -> None:
        from datetime import datetime, timezone

        from aetherdialect._core_utils import emit_write_queue_event

        graph_id = "sg_drain000000000001__abcd1234"
        owner = _make_engine(role="owner", graph_id=graph_id)
        owner._artifacts_dir = tmp_path
        owner._schema_graph = MagicMock()
        owner._schema_graph.schema_graph_id = graph_id
        doc = {"tables": {}}
        ev = WriteQueueEvent(
            kind="override_proposal",
            schema_graph_id=graph_id,
            schema_hash="eff",
            produced_at=datetime.now(timezone.utc).isoformat(),
            payload=(("document_json", json.dumps(doc)),),
        )
        emit_write_queue_event(str(tmp_path), ev)
        with patch(
            "aetherdialect._main_execution.apply_overrides_and_persist",
            return_value=MagicMock(),
        ) as apply_mock:
            n = drain_write_queue(owner, str(tmp_path))
        assert n == 1
        apply_mock.assert_called_once()


class TestConsumerIntentScopeGate:
    def test_denied_column_blocks_sql_scope(self) -> None:
        graph = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={"email": ColumnMetadata(name="email", data_type="text")},
                    primary_key=[],
                    foreign_keys=[],
                )
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        ctx = EngineContext(deny_columns=frozenset({"customer.email"}))
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert assert_consumer_intent_in_scope(intent, ctx, graph, frozenset({"customer"})) is False

    def test_visible_table_outside_set_blocks(self) -> None:
        graph = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={"email": ColumnMetadata(name="email", data_type="text")},
                    primary_key=[],
                    foreign_keys=[],
                )
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert assert_consumer_intent_in_scope(intent, EngineContext(), graph, frozenset()) is False
