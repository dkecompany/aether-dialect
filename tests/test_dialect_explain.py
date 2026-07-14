"""Unit tests for shared EXPLAIN plan parsers in ``_dialect_sqlglot_helper``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aetherdialect._contracts_base import SqlDiagnosticCode
from aetherdialect._dialect_sqlglot_helper import (
    bigquery_diagnostics_from_dry_run,
    mysql_diagnostics_from_explain_json,
    mysql_root_plan_estimates,
    redshift_diagnostics_from_explain_text,
    redshift_root_plan_estimates,
    snowflake_diagnostics_from_explain_json,
    snowflake_root_plan_estimates,
    sqlserver_diagnostics_from_showplan_rows,
    sqlserver_diagnostics_from_showplan_xml,
    sqlserver_root_plan_estimates,
)

MYSQL_EXPLAIN_JSON = (
    '{"query_block": {"table": {"rows_examined_per_scan": 42, "access_type": "ALL", '
    '"table_name": "line_items", "using_filesort": true, "using_temporary_table": true}, '
    '"nested_loop": [{"table": {"table_name": "orders", "access_type": "ALL"}}], '
    '"cost_info": {"query_cost": 10.5}}}'
)

REDSHIFT_EXPLAIN_TEXT = "XN Seq Scan on orders\n  rows=100 width=8\nXN Nested Loop\n  rows=0 width=4\nDS_DIST_BOTH\n"

SNOWFLAKE_EXPLAIN_JSON = (
    '[{"operation": "CartesianJoin", "rows": 50, "bytesAssigned": 4096}, '
    '{"operation": "TableScan", "partitionsAssigned": 0}]'
)

SQLSERVER_SHOWPLAN_ROWS = [
    ("Clustered Index Scan", "EstimateRows=120", "StmtText=SELECT * FROM t"),
    ("Nested Loops", "EstimateRows=0", "StmtText=SELECT 1"),
]


def test_mysql_explain_json_estimates_and_diagnostics() -> None:
    """MySQL JSON parser returns row estimate and full-scan diagnostic."""
    rows, _bytes = mysql_root_plan_estimates(MYSQL_EXPLAIN_JSON)
    assert rows == 42.0
    diags = mysql_diagnostics_from_explain_json(MYSQL_EXPLAIN_JSON)
    assert any(d.code == SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED for d in diags)
    messages = {d.message for d in diags}
    assert any("filesort" in m for m in messages)
    assert any("temporary table" in m for m in messages)


def test_redshift_explain_text_estimates_and_diagnostics() -> None:
    """Redshift text parser extracts rows/width and nested-loop findings."""
    rows, byte_est = redshift_root_plan_estimates(REDSHIFT_EXPLAIN_TEXT)
    assert rows == 100.0
    assert byte_est == 800.0
    diags = redshift_diagnostics_from_explain_text(REDSHIFT_EXPLAIN_TEXT)
    codes = {d.code for d in diags}
    assert SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN in codes
    assert SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE in codes
    assert any("DS_DIST_BOTH" in d.message for d in diags)


def test_snowflake_explain_json_estimates_and_diagnostics() -> None:
    """Snowflake JSON parser surfaces cartesian join and zero-partition diagnostics."""
    rows, bytes_est = snowflake_root_plan_estimates(SNOWFLAKE_EXPLAIN_JSON)
    assert rows == 50.0
    assert bytes_est == 4096.0
    diags = snowflake_diagnostics_from_explain_json(SNOWFLAKE_EXPLAIN_JSON)
    codes = {d.code for d in diags}
    assert SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN in codes
    assert SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE in codes


def test_sqlserver_showplan_rows_estimates_and_diagnostics() -> None:
    """SQL Server SHOWPLAN_ALL row parser finds estimate rows and nested-loop risk."""
    rows, _bytes = sqlserver_root_plan_estimates(SQLSERVER_SHOWPLAN_ROWS)
    assert rows == 120.0
    diags = sqlserver_diagnostics_from_showplan_rows(SQLSERVER_SHOWPLAN_ROWS)
    codes = {d.code for d in diags}
    assert SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN in codes
    assert SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE in codes


SQLSERVER_SHOWPLAN_XML = (
    '<ShowPlanXML><MissingIndexes><MissingIndexGroup/></MissingIndexes><RelOp PhysicalOp="Table Scan"/></ShowPlanXML>'
)


def test_sqlserver_showplan_xml_diagnostics() -> None:
    """SQL Server SHOWPLAN_XML parser surfaces missing index and table scan findings."""
    diags = sqlserver_diagnostics_from_showplan_xml(SQLSERVER_SHOWPLAN_XML)
    messages = {d.message for d in diags}
    assert any("missing index" in m.lower() for m in messages)
    assert any(d.code == SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED for d in diags)


def test_redshift_explain_bcast_and_dist_inner_diagnostics() -> None:
    """Redshift text parser flags broadcast and dist-inner distribution risks."""
    bcast = redshift_diagnostics_from_explain_text("XN Hash Join\nDS_BCAST_INNER\n")
    inner = redshift_diagnostics_from_explain_text("XN Hash Join\nDS_DIST_INNER\n")
    assert any("DS_BCAST_INNER" in d.message for d in bcast)
    assert any("DS_DIST_INNER" in d.message for d in inner)


def test_bigquery_diagnostics_from_dry_run_partition_and_bytes() -> None:
    """BigQuery dry-run diagnostics flag missing partition filters and large scans."""
    with patch("aetherdialect._dialect_sqlglot_helper.PolicyConfig.MAX_QUERY_COST_BYTES", 1000):
        diags = bigquery_diagnostics_from_dry_run(
            800.0,
            partition_filter_present=False,
            require_partition_filter_tables=["events"],
        )
    codes = {d.code for d in diags}
    assert SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED in codes
    assert any("partition" in d.message.lower() for d in diags)


def test_snowflake_explain_diagnose_appends_cost_exceeded_diagnostic() -> None:
    """Snowflake explain_diagnose surfaces EXPLAIN_COST_EXCEEDED on cap violations."""
    from aetherdialect._contracts_base import SqlDiagnosticCode
    from aetherdialect._dialect_sqlglot_engines import SnowflakeDialect

    dialect = object.__new__(SnowflakeDialect)
    dialect._explain_disabled = False
    dialect.engine = MagicMock()
    backend = MagicMock()
    backend.kind = "snowflake_arrow"
    backend.fetch_rows.return_value = [('{"operation": "TableScan", "rows": 999999}"',)]
    dialect._backend = backend
    with (
        patch.object(SnowflakeDialect, "finalize_render", return_value="SELECT 1"),
        patch.object(
            SnowflakeDialect,
            "parse_explain_plan",
            return_value=(999999.0, None, [], "{}"),
        ),
        patch("aetherdialect._dialect_sqlglot_engines.cost_cap_active", return_value=True),
        patch("aetherdialect._dialect_sqlglot_engines.PolicyConfig.MAX_QUERY_COST_ROWS", 10),
        patch("aetherdialect._dialect_sqlglot_engines.effective_explain_timeout_ms", return_value=None),
    ):
        ok, diags, _msg = SnowflakeDialect.explain_diagnose(dialect, "SELECT 1")
    assert ok is False
    assert any(d.code == SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED for d in diags)


def test_bigquery_explain_diagnose_wires_diagnostics() -> None:
    """BigQueryDialect.explain_diagnose returns dry-run soft diagnostics."""
    from aetherdialect._dialect_sqlglot_engines import BigQueryDialect

    job = MagicMock()
    job.total_bytes_processed = 500
    client = MagicMock()
    client.query.return_value = job
    dialect = object.__new__(BigQueryDialect)
    dialect._bq_client = client
    dialect.config = MagicMock()
    with (
        patch.object(BigQueryDialect, "finalize_render", return_value="SELECT 1"),
        patch.object(
            BigQueryDialect,
            "_bq_job_limits",
            return_value=(None, None),
        ),
        patch("google.cloud.bigquery", create=True) as mock_bq_pkg,
        patch("aetherdialect._dialect_sqlglot_engines.explain_cost_gate_violation", return_value=(False, "")),
        patch("aetherdialect._dialect_sqlglot_engines.PolicyConfig.MAX_QUERY_COST_BYTES", 1000000),
        patch("aetherdialect._dialect_sqlglot_engines.PolicyConfig.MAX_QUERY_COST_ROWS", 1000000),
    ):
        mock_bq_pkg.QueryJobConfig.return_value = MagicMock()
        ok, diags, _msg = BigQueryDialect.explain_diagnose(dialect, "SELECT 1")
    if not ok:
        print(f"DEBUG: _msg={_msg}")
        print(f"DEBUG: diags={diags}")
    assert ok is True
    assert isinstance(diags, list)
