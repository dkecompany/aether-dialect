"""Fast tests for Databricks deny_objects on the structural build path."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import (
    CatalogStructuralConstraintsIndex,
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_overrides import load_or_create_schema_databricks


def _tables_meta(table_names: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "column_names_original": ["id"],
            "column_types": ["integer"],
            "primary_key": ["id"],
            "foreign_keys": [],
        }
        for name in table_names
    }


@pytest.mark.fast
def test_databricks_load_or_create_schema_applies_deny_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Denied tables are removed from Databricks-built graphs before return."""
    monkeypatch.setattr(
        "aetherdialect._schema_overrides.extract_tables_from_catalog_sql_connector",
        lambda *args, **kwargs: _tables_meta(("orders", "secret")),
    )

    sg = load_or_create_schema_databricks(
        connection=object(),
        include="tables",
        deny_objects=frozenset({"secret"}),
        table_kinds_map={"orders": "table", "secret": "table"},
        structural_constraints_index=CatalogStructuralConstraintsIndex(),
    )

    assert "orders" in sg.tables
    assert "secret" not in sg.tables


@pytest.mark.fast
def test_databricks_reflect_schema_graph_passes_deny_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """DatabricksDialect.reflect_schema_graph forwards deny_objects to the builder."""
    from aetherdialect._dialect_sqlglot_engines import DatabricksDialect

    captured: dict[str, object] = {}

    def _fake_load(**kwargs: object) -> SchemaGraph:
        captured.update(kwargs)
        tables = {
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            )
        }
        return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))

    monkeypatch.setattr(
        "aetherdialect._dialect_sqlglot_engines.load_or_create_schema_databricks",
        _fake_load,
    )

    dialect = DatabricksDialect.__new__(DatabricksDialect)
    dialect.spark = object()
    dialect.connection = None
    dialect.table_kinds_map = lambda: {"orders": "table"}
    dialect.structural_constraints_index = lambda: CatalogStructuralConstraintsIndex()

    dialect.reflect_schema_graph(deny_objects=frozenset({"secret"}))

    assert captured.get("deny_objects") == frozenset({"secret"})
