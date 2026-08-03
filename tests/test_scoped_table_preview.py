"""Tests for scoped table preview on member engines and federations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE, TABLE_PREVIEW_MAX_LIMIT
from aetherdialect._contracts_base import (
    AccessError,
    ConfigError,
    EngineContext,
    FederationContext,
    SensitivityClassification,
    TablePreviewResult,
    set_sensitivity,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import compose_composite_graph, parse_federation_manifest
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.test_aether_federation_public_surface import _fed, _init_bundle, _minimal_member
from tests.test_aetherdialect import _make_aether_stub


def _preview_table(
    table_name: str,
    *,
    columns: dict[str, ColumnMetadata],
    rows: list[tuple[object, ...]],
    allow_objects: frozenset[str] | None = None,
) -> TablePreviewResult:
    table = TableMetadata(
        name=table_name,
        columns=columns,
        primary_key=["id"],
        foreign_keys=[],
    )
    schema = SchemaGraph(
        tables={table_name: table},
        join_paths_multi=recompute_join_paths_multi({table_name: table}),
    )
    dialect = MagicMock()
    dialect.quote_identifier.side_effect = lambda ident: f'"{ident}"'
    dialect.execute.return_value = rows
    engine = _make_aether_stub(
        _schema_graph=schema,
        _dialect=dialect,
        _runtime_config=MagicMock(
            engine_context=EngineContext(allow_objects=allow_objects or frozenset()),
            execution_context=EngineContext(allow_objects=allow_objects or frozenset()),
        ),
    )
    with patch("aetherdialect._main_execution.execute_guarded_sql", return_value=rows):
        return engine.preview_table(table_name, limit=5)


@pytest.mark.fast
def test_preview_table_returns_bounded_rows_on_engine() -> None:
    visible_col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
    name_col = ColumnMetadata(name="title", data_type="varchar", sensitivity="none")
    preview = _preview_table(
        "tbl_a",
        columns={"id": visible_col, "title": name_col},
        rows=[(idx, f"row-{idx}") for idx in range(1, 8)],
        allow_objects=frozenset({"tbl_a"}),
    )
    assert isinstance(preview, TablePreviewResult)
    assert preview.columns == ("id", "title")
    assert len(preview.rows) <= 5
    assert all(len(row) == len(preview.columns) for row in preview.rows)


@pytest.mark.fast
def test_preview_omits_hidden_columns() -> None:
    hidden_col = ColumnMetadata(name="secret_col", data_type="varchar")
    set_sensitivity(hidden_col, SensitivityClassification.HIDDEN)
    visible_col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
    preview = _preview_table(
        "tbl_a",
        columns={"id": visible_col, "secret_col": hidden_col},
        rows=[(1, "hidden-value")],
    )
    assert preview.columns == ("id",)
    assert preview.rows == ((1,),)


@pytest.mark.fast
def test_federation_preview_respects_scope() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_preview",
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    left_graph = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            )
        },
        join_paths_multi=recompute_join_paths_multi(
            {
                "left_t": TableMetadata(
                    name="left_t",
                    columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="a",
                )
            }
        ),
    )
    right_graph = SchemaGraph(
        tables={
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            )
        },
        join_paths_multi=recompute_join_paths_multi(
            {
                "right_t": TableMetadata(
                    name="right_t",
                    columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="b",
                )
            }
        ),
    )
    composite = compose_composite_graph({"a": left_graph, "b": right_graph}, manifest)
    bundle = _init_bundle(manifest, composite)
    scoped_context = FederationContext(allow_objects=frozenset({"left_t"}))
    left_member = _minimal_member(connection="a")
    left_member._schema_graph = left_graph
    left_member._dialect = MagicMock()
    left_member._dialect.quote_identifier.side_effect = lambda ident: f'"{ident}"'
    left_member._dialect.execute.return_value = [(1,), (2,)]
    right_member = _minimal_member(connection="b")
    right_member._schema_graph = right_graph
    fed = _fed()
    fed._apply_init_bundle(bundle)
    fed._members = {"a": left_member, "b": right_member}
    fed._federation_member_graphs = {"a": left_graph, "b": right_graph}
    with patch("aetherdialect._main_execution._resolve_preview_scope_context", return_value=scoped_context):
        with patch("aetherdialect._main_execution.validate_sql", return_value=(True, None, None, [])):
            allowed = fed.preview_table("left_t", limit=2)
    assert allowed.columns == ("id",)
    assert len(allowed.rows) == 2
    with patch("aetherdialect._main_execution._resolve_preview_scope_context", return_value=scoped_context):
        with pytest.raises(AccessError, match=PERMISSION_DENIED_USER_MESSAGE):
            fed.preview_table("right_t", limit=2)


@pytest.mark.fast
def test_federation_preview_rejects_unknown_table() -> None:
    fed = _fed()
    with pytest.raises(ConfigError, match="unknown table"):
        fed.preview_table("missing_table", limit=2)


@pytest.mark.fast
def test_preview_limit_is_capped() -> None:
    visible_col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
    many_rows = [(idx,) for idx in range(TABLE_PREVIEW_MAX_LIMIT + 5)]
    preview = _preview_table("tbl_a", columns={"id": visible_col}, rows=many_rows)
    assert len(preview.rows) <= TABLE_PREVIEW_MAX_LIMIT
