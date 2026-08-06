"""Tests that consumer preview hides unknown tables behind the same denial as scope."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import AccessError, ConfigError, EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.test_aetherdialect import _make_aether_stub


def _schema_with_table(table_name: str = "tbl_a") -> SchemaGraph:
    table = TableMetadata(
        name=table_name,
        columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={table_name: table},
        join_paths_multi=recompute_join_paths_multi({table_name: table}),
    )


@pytest.mark.fast
def test_consumer_unknown_table_matches_denied_message() -> None:
    schema = _schema_with_table("tbl_a")
    consumer = _make_aether_stub(
        _schema_graph=schema,
        _schema_role="consumer",
        _consumer_visible_objects=frozenset({"tbl_a"}),
        _runtime_config=MagicMock(
            engine_context=EngineContext(allow_objects=frozenset({"tbl_a"})),
            execution_context=EngineContext(allow_objects=frozenset({"tbl_a"})),
        ),
    )
    with pytest.raises(AccessError, match=PERMISSION_DENIED_USER_MESSAGE):
        consumer.preview_table("missing_table", limit=2)

    denied_exc: AccessError | None = None
    with patch("aetherdialect._main_execution.execute_guarded_sql", return_value=[(1,)]):
        try:
            consumer.preview_table("tbl_a", limit=2)
        except AccessError:
            pass
        try:
            consumer.preview_table("missing_table", limit=2)
        except AccessError as exc:
            denied_exc = exc
    assert denied_exc is not None
    assert str(denied_exc) == PERMISSION_DENIED_USER_MESSAGE

    owner = _make_aether_stub(_schema_graph=schema, _schema_role="owner")
    with pytest.raises(ConfigError, match="unknown table"):
        owner.preview_table("missing_table", limit=2)
