"""Member template fingerprints stay stable across coordinator dialect changes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import compute_sql_fp, sqlglot_dialect_for_engine
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import _template_from_store_dict, empty_template_store, insert_template


def _member_schema(source_id: str) -> SchemaGraph:
    tables = {
        "customers": TableMetadata(
            name="customers",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}",
    )


@pytest.mark.fast
def test_member_template_sql_fp_unchanged_after_postgresql_coordinator_reload() -> None:
    schema = _member_schema("west")
    intent = RuntimeIntent(
        tables=["customers"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    mysql_dialect = SimpleNamespace(engine="mysql")
    store = empty_template_store(schema.schema_graph_id)
    templates: dict = {}
    tmpl = insert_template(
        store,
        templates,
        schema,
        "west::how many customers",
        intent,
        "SELECT `customers`.`id` FROM `customers`",
        dialect=mysql_dialect,
        member_source_id="west",
        federation_plan_id="plan1",
        federation_plan_only=True,
    )
    expected_fp = compute_sql_fp(tmpl.sql_param, sqlglot_dialect=sqlglot_dialect_for_engine("mysql"))
    assert tmpl.sql_fp == expected_fp
    assert tmpl.member_engine == "mysql"

    raw = tmpl.to_dict()
    with patch(
        "aetherdialect._templates.active_sqlglot_dialect", return_value=sqlglot_dialect_for_engine("postgresql")
    ):
        reloaded = _template_from_store_dict(tmpl.id, raw)
    assert reloaded is not None
    assert reloaded.sql_fp == expected_fp
    assert reloaded.member_engine == "mysql"
