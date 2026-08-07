"""Deny-only aetherspace scopes must reach assert_intent_in_scope in SQL generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import SpaceContext
from aetherdialect._contracts_core import NormalizedExpr, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import generate_and_validate_sql
from aetherdialect._templates import TemplateOps


def _column(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _table(name: str) -> TableMetadata:
    return TableMetadata(name=name, columns={"id": _column("id")}, primary_key=["id"], foreign_keys=[])


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={"allowed": _table("allowed"), "secret": _table("secret")},
        join_paths_multi={},
        effective_structural_hash="eff_deny_gate",
    )


@pytest.mark.fast
def test_deny_only_space_blocks_denied_table(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _schema()
    store_dir = tmp_path / "intent_templates" / "spaces" / "master"
    store_dir.mkdir(parents=True)
    store = TemplateOps.empty_template_store(schema.effective_structural_hash)
    store._store_dir = str(store_dir)

    snap = MainExecutionOps.subset_graph_for_space(schema, SpaceContext(deny_objects=frozenset({"secret"})))
    space_tables, space_columns = MainExecutionOps.space_allowed_sets_from_snapshot(snap)
    deny_objects, deny_columns = MainExecutionOps.space_deny_sets_from_snapshot(snap)

    intent = RuntimeIntent(
        tables=["secret"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("secret.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )

    dialect = MagicMock()
    dialect.sqlglot_dialect = "duckdb"
    dialect.finalize_render = lambda sql, *_a, **_k: sql
    dialect.execute = MagicMock(return_value=[(1,)])

    out = generate_and_validate_sql(
        "secret count",
        intent,
        schema,
        {"candidates": []},
        {"J00": []},
        dialect,
        store,
        space_allowed_tables=space_tables,
        space_allowed_columns=space_columns,
        space_deny_tables=deny_objects,
        space_deny_columns=deny_columns,
    )

    assert out.success is False
    assert "aetherspace" in (out.sql_validation_error or "").lower()
