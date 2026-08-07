"""Unit tests for dialect parity completion (reflection, query debug, pruning helpers)."""

from __future__ import annotations

from unittest.mock import MagicMock

from aetherdialect._config import SQLServerRuntimeConfig
from aetherdialect._contracts_base import SqlDiagnosticCode
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._dialect_sqlglot_engines import SQLServerQueryLogSource
from aetherdialect._dialect_sqlglot_helper import SqlglotEngineDialect
from aetherdialect._schema_build import (
    parse_mysql_enum_or_set_labels,
    parse_redshift_sortkey_columns,
)

ExplainDiagnostics = SqlglotEngineDialect
InformationSchemaSupport = SqlglotEngineDialect


def test_parse_mysql_enum_and_set_labels() -> None:
    """MySQL enum/set parser returns kind and label lists."""
    enum_kind, enum_labels = parse_mysql_enum_or_set_labels("enum('a','b')")
    set_kind, set_labels = parse_mysql_enum_or_set_labels("set('x','y')")
    assert enum_kind == "enum"
    assert enum_labels == ["a", "b"]
    assert set_kind == "set"
    assert set_labels == ["x", "y"]


def test_parse_redshift_compound_and_interleaved_sortkeys() -> None:
    """Redshift sortkey parser expands compound and interleaved keys."""
    compound = parse_redshift_sortkey_columns("COMPOUND: order_date, customer_id")
    interleaved = parse_redshift_sortkey_columns("INTERLEAVED: sku, region")
    assert compound == ["order_date", "customer_id"]
    assert interleaved == ["sku", "region"]
    assert parse_redshift_sortkey_columns("order_date") == ["order_date"]


def test_mysql_index_awareness_diagnostic_with_schema() -> None:
    """MySQL EXPLAIN JSON flags filtered indexed columns on full scans."""
    table = TableMetadata(
        name="orders",
        columns={"status": ColumnMetadata(name="status", data_type="varchar")},
        primary_key=[],
        foreign_keys=[],
        indexed_columns=["status"],
    )
    schema = SchemaGraph(tables={"orders": table}, join_paths_multi={}, created_at="")
    payload = (
        '{"query_block": {"table": {"access_type": "ALL", "table_name": "orders", '
        '"attached_condition": "(`orders`.`status` = \'open\')"}}}'
    )
    diags = ExplainDiagnostics.mysql_diagnostics_from_explain_json(payload, schema=schema)
    indexed_msgs = [d for d in diags if d.code == SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED]
    assert len(indexed_msgs) >= 2


def test_sqlserver_aad_db_url_modes() -> None:
    """SQL Server db_url emits Azure AD authentication query parameters."""
    SQLServerRuntimeConfig.AUTH_MODE = "aad_password"
    SQLServerRuntimeConfig.USER = "user@example.com"
    SQLServerRuntimeConfig.PASSWORD = "secret"
    SQLServerRuntimeConfig.DATABASE = "db"
    SQLServerRuntimeConfig.HOST = "localhost"
    SQLServerRuntimeConfig.PORT = 1433
    pwd_url = SQLServerRuntimeConfig.db_url()
    assert "ActiveDirectoryPassword" in pwd_url

    SQLServerRuntimeConfig.AUTH_MODE = "aad_sp"
    SQLServerRuntimeConfig.CLIENT_ID = "client-id"
    SQLServerRuntimeConfig.CLIENT_SECRET = "client-secret"
    sp_url = SQLServerRuntimeConfig.db_url()
    assert "ActiveDirectoryServicePrincipal" in sp_url


def test_sqlserver_query_store_preferred_over_dmv() -> None:
    """SQL Server query log uses Query Store when availability probe succeeds."""
    source = SQLServerQueryLogSource()
    avail_cursor = MagicMock()
    avail_cursor.fetchall.return_value = [(1,)]
    fetch_cursor = MagicMock()
    fetch_cursor.fetchall.return_value = [("SELECT 1",)]
    conn = MagicMock()
    conn.cursor.side_effect = [avail_cursor, fetch_cursor]

    texts = source.fetch(conn, lookback_days=7, max_queries=5, min_runs=1, user_filter=None)
    assert texts == ["SELECT <num>"]
    fetch_sql = str(fetch_cursor.execute.call_args.args[0])
    assert "query_store" in fetch_sql.lower()


def test_reflect_redshift_unique_columns_from_meta() -> None:
    """Redshift reflection populates unique_columns on table metadata."""
    from aetherdialect._schema_build import tables_meta_to_schema_graph

    meta = {
        "items": {
            "column_names_original": ["id", "email"],
            "column_types": ["integer", "varchar"],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "unique_columns": ["email"],
            "column_is_nullable": [False, True],
        }
    }
    sg = tables_meta_to_schema_graph(meta)
    assert sg.tables["items"].columns["email"].is_unique is True


def test_reflect_sqlserver_identity_flag() -> None:
    """SQL Server reflection stores is_identity without promoting to PK."""
    from aetherdialect._schema_build import tables_meta_to_schema_graph

    meta = {
        "orders": {
            "column_names_original": ["order_id", "note"],
            "column_types": ["int", "varchar"],
            "primary_keys": [],
            "foreign_keys": [],
            "unique_columns": [],
            "column_is_nullable": [False, True],
            "column_is_identity": [True, False],
        }
    }
    sg = tables_meta_to_schema_graph(meta)
    assert sg.tables["orders"].columns["order_id"].is_identity is True
    assert sg.tables["orders"].primary_key == []


def test_databricks_nullability_from_structural_index() -> None:
    """CatalogStructuralConstraintsIndex carries column nullability maps."""
    from aetherdialect._contracts_schema import CatalogStructuralConstraintsIndex

    rows = [{"table_name": "t1", "column_name": "id", "is_nullable": "NO"}]
    idx = CatalogStructuralConstraintsIndex(
        column_nullability=SqlglotEngineDialect.column_nullability_from_information_schema_rows(rows),
    )
    assert idx.column_nullability["t1"]["id"] is False
