"""Tests for permission-filtered consumer intent schema visibility."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._pipeline import parse_intent_via_llm


def _table(name: str) -> TableMetadata:
    col = ColumnMetadata(name="id", data_type="integer")
    return TableMetadata(
        name=name,
        columns={"id": col},
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )


def _owner_graph() -> SchemaGraph:
    return SchemaGraph(
        join_paths_multi={},
        tables={
            "a": _table("a"),
            "b": _table("b"),
        },
        effective_structural_hash="owner_eff",
    )


class TestResolveIntentVisibleObjects:
    def test_execution_scope_used_when_space_absent(self) -> None:
        from aetherdialect._contracts_schema import resolve_intent_visible_objects

        assert resolve_intent_visible_objects(
            visible_objects=None,
            execution_visible_objects=frozenset({"a"}),
        ) == frozenset({"a"})

    def test_space_scope_intersects_execution_scope(self) -> None:
        from aetherdialect._contracts_schema import resolve_intent_visible_objects

        assert resolve_intent_visible_objects(
            visible_objects=frozenset({"a", "c"}),
            execution_visible_objects=frozenset({"a", "b"}),
        ) == frozenset({"a"})

    def test_space_scope_preserved_for_owner(self) -> None:
        from aetherdialect._contracts_schema import resolve_intent_visible_objects

        assert resolve_intent_visible_objects(
            visible_objects=frozenset({"a"}),
            execution_visible_objects=None,
        ) == frozenset({"a"})


class TestPermissionFilteredIntentScope:
    def test_owner_graph_payload_excludes_denied_tables(self) -> None:
        graph = _owner_graph()
        payload = json.loads(
            graph.schema_payload_interpret(visible_objects=frozenset({"a"})),
        )
        assert "a" in payload
        assert "b" not in payload

    def test_parse_intent_via_llm_uses_execution_visible_objects(self) -> None:
        port = MagicMock()
        port.visible_objects = None
        port.execution_visible_objects = frozenset({"a"})
        port.space_columns = None
        port.space_deny_objects = None
        port.space_deny_columns = None
        port.space_description_overlay = None
        with patch(
            "aetherdialect._pipeline.invoke_intent_parse_with_hints",
            return_value=(None, [], 0, None),
        ) as mock_parse:
            parse_intent_via_llm("question", _owner_graph(), {}, {}, choice_port=port)
        assert mock_parse.call_args.kwargs["visible_objects"] == frozenset({"a"})

    def test_parse_intent_via_llm_intersects_space_with_execution(self) -> None:
        port = MagicMock()
        port.visible_objects = frozenset({"a"})
        port.execution_visible_objects = frozenset({"a", "b"})
        port.space_columns = None
        port.space_deny_objects = None
        port.space_deny_columns = None
        port.space_description_overlay = None
        with patch(
            "aetherdialect._pipeline.invoke_intent_parse_with_hints",
            return_value=(None, [], 0, None),
        ) as mock_parse:
            parse_intent_via_llm("question", _owner_graph(), {}, {}, choice_port=port)
        assert mock_parse.call_args.kwargs["visible_objects"] == frozenset({"a"})
