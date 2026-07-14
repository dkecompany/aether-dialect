"""Tests for dialect module: SQL dialect abstraction, AST validation, and date rendering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._dialect import Dialect, get_dialect, resolve_dialect
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import (
    BigQueryDialect,
    DatabricksDialect,
    DuckDBDialect,
    MariaDBDialect,
    MySQLDialect,
    RedshiftDialect,
    SnowflakeDialect,
    SQLiteDialect,
    SQLServerDialect,
    databricks_normalize_datetrunc_sql,
)
from aetherdialect._dialect_sqlglot_helper import array_storage_kind, ast_structural_valid_sqlglot


def _pg_uninit() -> PostgresDialect:
    """Return a ``PostgresDialect`` instance without running ``__init__`` (render-only tests)."""
    return PostgresDialect.__new__(PostgresDialect)


def _dbr_uninit() -> DatabricksDialect:
    """Return a ``DatabricksDialect`` instance without running ``__init__`` (render-only tests)."""
    return DatabricksDialect.__new__(DatabricksDialect)


def _mysql_uninit() -> MySQLDialect:
    """Return a ``MySQLDialect`` instance without running ``__init__`` (render-only tests)."""
    return MySQLDialect.__new__(MySQLDialect)


def _snowflake_uninit() -> SnowflakeDialect:
    """Return a ``SnowflakeDialect`` instance without running ``__init__`` (render-only tests)."""
    return SnowflakeDialect.__new__(SnowflakeDialect)


def _redshift_uninit() -> RedshiftDialect:
    """Return a ``RedshiftDialect`` instance without running ``__init__`` (render-only tests)."""
    return RedshiftDialect.__new__(RedshiftDialect)


def test_resolve_dialect_accepts_name_or_instance() -> None:
    """``resolve_dialect`` returns the same dialect object or delegates to ``get_dialect`` for names."""
    d = _pg_uninit()
    assert resolve_dialect(d) is d
    with patch("aetherdialect._dialect.get_dialect", return_value=d) as m:
        assert resolve_dialect("postgresql") is d
        m.assert_called_once_with("postgresql")


class TestPostgresPglastWalk:
    """``_pg_walk_nodes`` must descend into pglast ``Node`` children (slot-based AST)."""

    def test_finds_rangevars_under_cross_join(self) -> None:
        from pglast.parser import parse_sql

        from aetherdialect._dialect_postgres import pg_node_kind, pg_walk_nodes

        sql = "WITH cte1 AS (SELECT 1 AS x), cte2 AS (SELECT 1 AS y FROM cte1) SELECT * FROM cte1 CROSS JOIN cte2"
        root = parse_sql(sql)[0].stmt
        rels = {n.relname for n in pg_walk_nodes(root) if pg_node_kind(n) == "RangeVar" and getattr(n, "relname", None)}
        assert "cte1" in rels
        assert "cte2" in rels


def test_databricks_quote_table_column_backticks() -> None:
    """Databricks dialect uses backticks for reserved-word identifiers."""
    d = _dbr_uninit()
    assert d.quote_table_column("orders", "order") == "`orders`.`order`"


class TestRenderArrayContainsExpr:
    """Tests for ``PostgresDialect.render_array_contains`` / ``DatabricksDialect.render_array_contains``."""

    def test_postgresql_uses_any_with_lowered_array(self) -> None:
        sql = _pg_uninit().render_array_contains("film.special_features", "p1")
        assert "= ANY(STRING_TO_ARRAY(LOWER(ARRAY_TO_STRING(film.special_features, CHR(31))), CHR(31)))" in sql
        assert "LOWER(BTRIM(CAST(:p1 AS TEXT)" in sql
        assert "EXISTS" not in sql
        assert "ARRAY[" not in sql
        assert "unnest" not in sql.lower()

    def test_databricks_uses_transform_and_trim(self) -> None:
        sql = _dbr_uninit().render_array_contains("`film`.`special_features`", "p2")
        assert "ARRAY_CONTAINS" in sql
        assert "LOWER(TRIM(CAST(_ac_x AS STRING)" in sql
        assert "LOWER(TRIM(CAST(:p2 AS STRING)" in sql
        assert "NOT `film`.`special_features` IS NULL" in sql

    def test_mysql_lowercases_json_elements(self) -> None:
        sql = _mysql_uninit().render_array_contains("film.special_features", "p1")
        assert "LOCATE" in sql or "INSTR" in sql
        assert "LOWER(CAST(film.special_features AS CHAR))" in sql
        assert "LOWER(TRIM(BOTH '%' FROM CAST(:p1 AS CHAR)))" in sql
        assert "JSON_SEARCH" not in sql

    def test_snowflake_uses_transform_and_trim(self) -> None:
        sql = _snowflake_uninit().render_array_contains("film.special_features", "p1")
        assert "ARRAY_CONTAINS" in sql
        assert "TRANSFORM(film.special_features" in sql
        assert "LOWER(TRIM(CAST(:p1 AS VARCHAR)" in sql
        assert "AS VARIANT" in sql
        assert sql.index("ARRAY_CONTAINS") < sql.index("TRANSFORM")

    def test_redshift_matches_json_elements_case_insensitively(self) -> None:
        sql = _redshift_uninit().render_array_contains("film.special_features", "p1")
        assert "STRPOS" in sql or "POSITION" in sql
        assert "LOWER(CAST(film.special_features AS VARCHAR))" in sql
        assert "REGEXP_INSTR" not in sql
        assert "CONCAT('\"'" in sql or "CONCAT('\"" in sql or "\"' ||" in sql

    def test_array_storage_kind_classifies_native_and_json_text(self) -> None:
        native = ColumnMetadata(name="special_features", data_type="VARCHAR[]", element_type="string")
        json_text = ColumnMetadata(
            name="special_features",
            data_type="VARCHAR",
            element_type="string",
            frequent_values=['["Trailers", "Commentaries"]'],
        )
        json_col = ColumnMetadata(name="special_features", data_type="JSON", element_type="string")
        assert array_storage_kind(native) == "native_array"
        assert array_storage_kind(json_text) == "json_text_array"
        assert array_storage_kind(json_col) == "json_text_array"

    def test_postgres_parse_select_does_not_raise(self) -> None:
        d = _pg_uninit()
        assert d.parse_select("SELECT film_id FROM film") is not None

    def test_mysql_routes_by_column_meta(self) -> None:
        json_meta = ColumnMetadata(name="special_features", data_type="JSON", element_type="string")
        text_meta = ColumnMetadata(name="special_features", data_type="VARCHAR", element_type="string")
        json_sql = _mysql_uninit().render_array_contains("film.special_features", "p1", column_meta=json_meta)
        text_sql = _mysql_uninit().render_array_contains("film.special_features", "p1", column_meta=text_meta)
        assert array_storage_kind(json_meta) == "json_text_array"
        assert "LOCATE" in json_sql or "INSTR" in json_sql
        assert "LOCATE" in text_sql or "INSTR" in text_sql
        assert "JSON_CONTAINS" not in json_sql


class TestRenderDateDiffExpr:
    """Tests for dialect ``render_date_diff``."""

    def test_postgresql_day_interval(self):
        """PostgreSQL produces INTERVAL for day unit."""
        result = _pg_uninit().render_date_diff("rental.return_date - rental.rental_date", ">", "day", 7)
        assert "INTERVAL '7 DAYS'" in result
        assert ">" in result

    def test_postgresql_singular_day(self):
        """PostgreSQL uses singular 'day' for amount 1."""
        result = _pg_uninit().render_date_diff("col1 - col2", ">=", "day", 1)
        assert "INTERVAL '1 DAY'" in result

    def test_databricks_day_numeric(self):
        """Databricks produces numeric comparison for day unit."""
        result = _dbr_uninit().render_date_diff("col1 - col2", ">", "day", 7)
        assert result == "(col1 - col2) > 7"

    def test_databricks_week_multiplied(self):
        """Databricks week unit multiplies by 7."""
        result = _dbr_uninit().render_date_diff("a - b", ">=", "week", 2)
        assert result == "(a - b) >= 14"

    def test_snowflake_two_column_datediff(self):
        """Snowflake uses DATEDIFF for two date columns."""
        result = _snowflake_uninit().render_date_diff(
            "rental.return_date - rental.rental_date",
            ">",
            "day",
            7,
            minuend_sql="rental.return_date",
            subtrahend_sql="rental.rental_date",
        )
        assert "DATEDIFF(DAY, rental.rental_date, rental.return_date) > 7" in result


class TestRenderDateWindowExprPostgresql:
    """Tests for render_date_window_expr with postgresql dialect."""

    def test_offset_zero_truncates_current_date(self):
        """Offset 0 should produce DATE_TRUNC for the unit."""
        result = _pg_uninit().render_date_window("o.order_date", ">=", "month", 0)
        assert result == "o.order_date >= DATE_TRUNC('MONTH', CURRENT_DATE)"

    def test_offset_zero_day(self):
        """Offset 0 with day unit."""
        result = _pg_uninit().render_date_window("t.created", ">=", "day", 0)
        assert result == "t.created >= DATE_TRUNC('DAY', CURRENT_DATE)"

    def test_offset_zero_year(self):
        """Offset 0 with year unit."""
        result = _pg_uninit().render_date_window("col", "<", "year", 0)
        assert result == "col < DATE_TRUNC('YEAR', CURRENT_DATE)"

    def test_positive_offset_interval(self):
        """Positive offset should produce INTERVAL subtraction."""
        result = _pg_uninit().render_date_window("d.ts", ">=", "day", 7)
        assert result == "d.ts >= CURRENT_DATE - INTERVAL '7 DAYS'"

    def test_week_offset(self):
        """Week offset should use week unit in INTERVAL."""
        result = _pg_uninit().render_date_window("col", ">=", "week", 2)
        assert result == "col >= CURRENT_DATE - INTERVAL '2 WEEKS'"

    def test_month_offset(self):
        """Month offset should use month unit in INTERVAL."""
        result = _pg_uninit().render_date_window("col", "<", "month", 3)
        assert result == "col < CURRENT_DATE - INTERVAL '3 MONTHS'"

    def test_year_offset(self):
        """Year offset should use year unit in INTERVAL."""
        result = _pg_uninit().render_date_window("col", ">=", "year", 1)
        assert result == "col >= CURRENT_DATE - INTERVAL '1 YEAR'"

    def test_offset_zero_week(self):
        """Offset 0 with week unit uses DATE_TRUNC week."""
        result = _pg_uninit().render_date_window("col", ">=", "week", 0)
        assert result == "col >= DATE_TRUNC('WEEK', CURRENT_DATE)"

    def test_positive_offset_day_interval(self):
        """Positive day offset produces INTERVAL 'N days'."""
        result = _pg_uninit().render_date_window("d.col", "<", "day", 1)
        assert result == "d.col < CURRENT_DATE - INTERVAL '1 DAY'"


class TestRenderDateWindowExprDatabricks:
    """Tests for render_date_window_expr with databricks dialect."""

    def test_offset_zero_truncates_current_date(self):
        """Offset 0 should produce date_trunc for the unit."""
        result = _dbr_uninit().render_date_window("o.order_date", ">=", "month", 0)
        assert result == "o.order_date >= DATE_TRUNC('MONTH', CURRENT_DATE)"

    def test_day_offset(self):
        """Day offset should produce date_sub."""
        result = _dbr_uninit().render_date_window("col", ">=", "day", 7)
        assert result == "col >= DATE_ADD(CURRENT_DATE, 7 * -1)"

    def test_week_offset(self):
        """Week offset should multiply by 7 and use date_sub."""
        result = _dbr_uninit().render_date_window("col", ">=", "week", 2)
        assert result == "col >= DATE_ADD(CURRENT_DATE, 14 * -1)"

    def test_month_offset(self):
        """Month offset should use add_months with negative value."""
        result = _dbr_uninit().render_date_window("col", ">=", "month", 3)
        assert result == "col >= ADD_MONTHS(CURRENT_DATE, -3)"

    def test_year_offset(self):
        """Year offset should use add_months with months = -offset*12."""
        result = _dbr_uninit().render_date_window("col", ">=", "year", 1)
        assert result == "col >= ADD_MONTHS(CURRENT_DATE, -12)"

    def test_year_offset_multiple(self):
        """Multiple year offset converts to months correctly."""
        result = _dbr_uninit().render_date_window("col", "<", "year", 3)
        assert result == "col < ADD_MONTHS(CURRENT_DATE, -36)"

    def test_quarter_offset(self):
        """Quarter offset converts to months and uses add_months."""
        result = _dbr_uninit().render_date_window("col", ">=", "quarter", 2)
        assert result == "col >= ADD_MONTHS(CURRENT_DATE, -6)"

    def test_half_year_offset(self):
        """Half-year offset converts to months and uses add_months."""
        result = _dbr_uninit().render_date_window("col", ">=", "half_year", 1)
        assert result == "col >= ADD_MONTHS(CURRENT_DATE, -6)"

    def test_offset_zero_week(self):
        """Offset 0 with week unit uses date_trunc."""
        result = _dbr_uninit().render_date_window("col", ">=", "week", 0)
        assert result == "col >= DATE_TRUNC('WEEK', CURRENT_DATE)"

    def test_offset_zero_year(self):
        """Offset 0 with year unit uses date_trunc."""
        result = _dbr_uninit().render_date_window("col", "<", "year", 0)
        assert result == "col < DATE_TRUNC('YEAR', CURRENT_DATE)"


class TestGetDialect:
    """Tests for get_dialect factory function."""

    @patch("aetherdialect._dialect.EngineConfig")
    def test_postgresql_returns_postgres_dialect(self, mock_config):
        """Requesting postgresql should return PostgresDialect instance."""
        mock_runtime = SimpleNamespace(
            HOST="localhost",
            PORT=5432,
            USER="test",
            PASSWORD="test",
            DATABASE="testdb",
            SCHEMA="public",
        )
        mock_runtime.db_url = lambda: "postgresql://test:test@localhost:5432/testdb"
        mock_config.TYPE = "postgresql"
        mock_config.RUNTIME = mock_runtime

        with patch("sqlalchemy.create_engine"):
            d = get_dialect("postgresql", mock_runtime)
        assert isinstance(d, PostgresDialect)
        assert d.name == "postgresql"

    def test_unsupported_raises_value_error(self):
        """Unsupported dialect name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dialect"):
            get_dialect("nonexistent_engine", MagicMock())

    @patch("aetherdialect._dialect.EngineConfig")
    def test_none_engine_type_uses_config_type(self, mock_config):
        """get_dialect with engine_type None uses EngineConfig.TYPE."""
        mock_runtime = SimpleNamespace(
            HOST="localhost",
            PORT=5432,
            USER="u",
            PASSWORD="p",
            DATABASE="d",
            SCHEMA="public",
        )
        mock_runtime.db_url = lambda: "postgresql://u:p@localhost:5432/d"
        mock_config.TYPE = "postgresql"
        mock_config.RUNTIME = mock_runtime
        with patch("sqlalchemy.create_engine"):
            d = get_dialect(engine_type=None, config=mock_runtime)
        assert isinstance(d, PostgresDialect)


class TestDialectBase:
    """Tests for abstract Dialect base class."""

    def test_ast_validate_raises_not_implemented(self):
        """Base ast_validate should raise NotImplementedError."""
        d = Dialect(config=MagicMock())
        with pytest.raises(NotImplementedError):
            d.ast_validate("SELECT 1")

    def test_explain_sql_raises_not_implemented(self):
        """Base explain_sql should raise NotImplementedError."""
        d = Dialect(config=MagicMock())
        with pytest.raises(NotImplementedError):
            d.explain_sql("SELECT 1")

    def test_name_attribute(self):
        """Base dialect name attribute should be 'base'."""
        d = Dialect(config=MagicMock())
        assert d.name == "base"

    def test_can_explain_false_without_engine(self):
        """Base ``can_explain`` is False when no engine attribute is present."""
        d = Dialect(config=MagicMock())
        assert d.can_explain() is False

    def test_disable_explain_on_permission_denied_matches_known_patterns(self):
        """Permission-denied classification sets ``_explain_disabled`` and returns True."""
        d = Dialect(config=MagicMock())
        d.engine = MagicMock()
        assert d.can_explain() is True
        assert d._disable_explain_on_permission_denied("ERROR: permission denied for relation foo") is True
        assert d._explain_disabled is True
        assert d.can_explain() is False

    def test_disable_explain_ignores_unrelated_errors(self):
        """Non-credential errors should not disable EXPLAIN."""
        d = Dialect(config=MagicMock())
        d.engine = MagicMock()
        assert d._disable_explain_on_permission_denied("syntax error near WHERE") is False
        assert d._explain_disabled is False
        assert d.can_explain() is True

    def test_quote_table_column_base_raises_not_implemented(self):
        """Base dialect requires subclasses to implement quote_table_column."""
        d = Dialect(config=MagicMock())
        with pytest.raises(NotImplementedError):
            d.quote_table_column("orders", "id")


class TestPostgresExplainPermissionHandling:
    """Tests for ``PostgresDialect.explain_sql`` permission-denied self- disable behaviour."""

    def _make_dialect(self):
        """Create a PostgresDialect bypassing ``create_engine`` for engine stubbing."""
        with patch("sqlalchemy.create_engine"):
            mock_runtime = SimpleNamespace(
                HOST="localhost",
                PORT=5432,
                USER="u",
                PASSWORD="p",
                DATABASE="db",
                SCHEMA="public",
            )
            mock_runtime.db_url = lambda: "postgresql://u:p@localhost:5432/db"
            d = PostgresDialect(mock_runtime)
        return d

    def test_permission_denied_returns_success_and_disables(self):
        """Permission-denied responses are masked as success and disable EXPLAIN."""
        d = self._make_dialect()
        bad_engine = MagicMock()
        cm = MagicMock()
        cm.__enter__.side_effect = RuntimeError("ERROR: permission denied for relation film")
        bad_engine.begin.return_value = cm
        d.engine = bad_engine

        ok, err = d.explain_sql("SELECT 1")

        assert ok is True
        assert err == ""
        assert d._explain_disabled is True
        assert d.can_explain() is False

    def test_non_permission_error_still_fails(self):
        """A non-permission failure preserves the error and keeps EXPLAIN enabled."""
        d = self._make_dialect()
        bad_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError('relation "missing" does not exist')
        cm = MagicMock()
        cm.__enter__.return_value = mock_conn
        bad_engine.begin.return_value = cm
        d.engine = bad_engine

        ok, err = d.explain_sql("SELECT * FROM missing")

        assert ok is False
        assert "missing" in err
        assert d._explain_disabled is False
        assert d.can_explain() is True

    def test_quote_table_column_postgres(self):
        """Postgres dialect double-quotes identifiers."""
        d = self._make_dialect()
        assert d.quote_table_column("orders", "order") == '"orders"."order"'


class TestAstStructuralValid:
    """Tests for ``PostgresDialect._ast_structural_valid`` (PostgreSQL pglast-based)."""

    @pytest.fixture
    def pg(self):
        """Uninitialized Postgres dialect for AST helpers."""
        return _pg_uninit()

    def test_simple_select_passes(self, pg):
        """Simple SELECT should pass validation."""
        ok, err = pg._ast_structural_valid("SELECT customer_id, name FROM customers")
        assert ok is True
        assert err == ""

    def test_select_with_join_passes(self, pg):
        """SELECT with JOIN should pass validation."""
        sql = "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True

    def test_multiple_statements_rejected(self, pg):
        """Multiple statements should be rejected."""
        ok, err = pg._ast_structural_valid("SELECT 1; SELECT 2")
        assert ok is False
        assert err == "multiple_statements"

    def test_non_select_rejected(self, pg):
        """Non-SELECT statements should be rejected."""
        ok, err = pg._ast_structural_valid("INSERT INTO t VALUES (1)")
        assert ok is False
        assert err == "not_select"

    def test_cte_select_passes(self, pg):
        """CTE with simple SELECT should pass."""
        sql = "WITH active AS (SELECT customer_id FROM customers WHERE active = true) SELECT * FROM active"
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True

    def test_parse_error_returns_ast_parse_failed(self, pg):
        """Malformed SQL should return ast_parse_failed."""
        ok, err = pg._ast_structural_valid("NOT VALID SQL AT ALL %%%")
        assert ok is False
        assert err == "ast_parse_failed"

    def test_rank_over_allowed(self, pg):
        """Window functions (e.g. RANK OVER) should pass structural validation."""
        sql = "SELECT customer_id, RANK() OVER (ORDER BY customer_id) AS rnk FROM customers"
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True, err
        assert err == ""

    def test_row_number_partition_by_allowed(self, pg):
        """ROW_NUMBER with PARTITION BY / ORDER BY should pass."""
        sql = "SELECT store_id, ROW_NUMBER() OVER (PARTITION BY store_id ORDER BY staff_id) AS n FROM staff"
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True, err

    def test_case_expression_in_select_allowed(self, pg):
        """Simple CASE in the SELECT list should pass."""
        sql = "SELECT customer_id, CASE WHEN active THEN 'yes' ELSE 'no' END AS flag FROM customers"
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True, err

    def test_case_without_from_allowed(self, pg):
        """CASE-only SELECT (no FROM) should pass when structurally valid."""
        sql = "SELECT CASE WHEN 1 = 1 THEN 'a' WHEN 2 = 2 THEN 'b' ELSE 'c' END AS lbl"
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True, err

    def test_nested_case_allowed(self, pg):
        """Nested CASE should pass."""
        sql = (
            "SELECT CASE WHEN customer_id < 10 "
            "THEN CASE WHEN active THEN 1 ELSE 0 END "
            "ELSE 2 END AS bucket "
            "FROM customers"
        )
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True, err

    def test_cte_body_window_and_case_allowed(self, pg):
        """CTE bodies may use window functions and CASE."""
        sql = (
            "WITH ranked AS ("
            "SELECT customer_id, active, "
            "RANK() OVER (ORDER BY customer_id) AS rnk, "
            "CASE WHEN active THEN 1 ELSE 0 END AS flag "
            "FROM customers"
            ") SELECT customer_id, rnk FROM ranked WHERE flag = 1"
        )
        ok, err = pg._ast_structural_valid(sql)
        assert ok is True, err

    def test_scalar_cte_cross_join_allowed_when_whitelisted(self, pg):
        sql = "WITH c AS (SELECT 1 AS x) SELECT * FROM customers CROSS JOIN c"
        ok, err = pg._ast_structural_valid(sql, scalar_cte_names=frozenset({"c"}))
        assert ok is True, err


class TestAstSparkStructuralValid:
    """Tests for shared sqlglot structural validation (Spark dialect string)."""

    @pytest.fixture
    def dbr(self):
        """Uninitialized Databricks dialect for validation-only tests."""
        return _dbr_uninit()

    def test_simple_select_passes(self):
        ok, err = ast_structural_valid_sqlglot(
            "SELECT customer_id, name FROM customers",
            sqlglot_dialect="spark",
        )
        assert ok is True
        assert err == ""

    def test_join_on_passes(self):
        sql = "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
        ok, err = ast_structural_valid_sqlglot(sql, sqlglot_dialect="spark")
        assert ok is True, err

    def test_recursive_cte_rejected(self):
        sql = "WITH RECURSIVE r AS (SELECT 1 AS n) SELECT * FROM r"
        ok, err = ast_structural_valid_sqlglot(sql, sqlglot_dialect="spark")
        assert ok is False
        assert err == "cte_recursive"

    def test_cte_union_rejected(self):
        sql = "WITH a AS (SELECT 1 UNION SELECT 2) SELECT * FROM a"
        ok, err = ast_structural_valid_sqlglot(sql, sqlglot_dialect="spark")
        assert ok is False
        assert err == "cte_contains_set_op"

    def test_exists_rejected(self):
        sql = "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)"
        ok, err = ast_structural_valid_sqlglot(sql, sqlglot_dialect="spark")
        assert ok is False
        assert err == "exists_not_allowed"

    def test_subquery_in_from_rejected(self):
        ok, err = ast_structural_valid_sqlglot(
            "SELECT * FROM (SELECT 1 AS x) t",
            sqlglot_dialect="spark",
        )
        assert ok is False
        assert err == "subquery_not_allowed"

    def test_cross_join_rejected(self):
        ok, err = ast_structural_valid_sqlglot(
            "SELECT * FROM a CROSS JOIN b",
            sqlglot_dialect="spark",
        )
        assert ok is False
        assert err == "cross_join_not_allowed"

    def test_cross_join_scalar_cte_allowed_when_whitelisted(self):
        sql = "WITH c AS (SELECT 1 AS x) SELECT * FROM customers CROSS JOIN c"
        ok, err = ast_structural_valid_sqlglot(
            sql,
            sqlglot_dialect="spark",
            scalar_cte_names=frozenset({"c"}),
        )
        assert ok is True, err

    def test_using_join_rejected(self):
        ok, err = ast_structural_valid_sqlglot(
            "SELECT * FROM a JOIN b USING (id)",
            sqlglot_dialect="spark",
        )
        assert ok is False
        assert err == "using_not_allowed"

    def test_ast_validate_does_not_call_explain(self, dbr):
        """``ast_validate`` must not hit the warehouse; ``explain_sql`` owns ``EXPLAIN``."""
        dbr.config = SimpleNamespace(CATALOG="c", SCHEMA="s", DEBUG=False)
        mock_cursor = MagicMock()
        dbr.connection = MagicMock()
        dbr.connection.cursor.return_value = mock_cursor
        dbr.spark = None
        dbr.engine = None
        ok, err = dbr.ast_validate("SELECT 1 FROM t")
        assert ok is True
        assert err == ""
        dbr.connection.cursor.assert_not_called()


class TestQualifyTablesForExecution:
    """Tests for DatabricksDialect table qualification in finalize_render."""

    def _make_dialect(self):
        """Create a DatabricksDialect with mocked internals."""
        mock_config = SimpleNamespace(CATALOG="dev", SCHEMA="rental_shop", DEBUG=False)
        d = DatabricksDialect.__new__(DatabricksDialect)
        d.config = mock_config
        d.sqlglot_dialect = "databricks"
        return d

    def test_qualifies_table_references(self):
        """_qualify_tables_for_execution qualifies bare table references."""
        d = self._make_dialect()
        result = d._qualify_tables_for_execution("SELECT * FROM customers")
        assert "dev" in result
        assert "rental_shop" in result
        assert "customers" in result


class TestDatabricksNormalizeDatetruncSql:
    """``databricks_normalize_datetrunc_sql`` normalizes Anonymous DATETRUNC call sites."""

    def test_rewrites_to_date_trunc_with_unit_first(self):
        sql = "SELECT DATETRUNC('MONTH', rental.created_at) AS m FROM rental"
        out = databricks_normalize_datetrunc_sql(sql)
        assert "DATETRUNC(" not in out
        assert "DATE_TRUNC('MONTH'" in out


class TestPostgresNormalizeDatetruncSql:
    """``PostgresDialect.post_render_normalize`` rewrites DATETRUNC to DATE_TRUNC."""

    def test_rewrites_datetrunc_to_date_trunc(self) -> None:
        sql = "SELECT DATETRUNC(rental.created_at, 'MONTH') AS m FROM rental"
        out = _pg_uninit().post_render_normalize(sql, stage="post_substitute")
        assert "DATETRUNC(" not in out.upper()
        assert "date_trunc('month'" in out.lower()


class TestRenderDateWindowExprEdgeCases:
    """Edge case tests for render_date_window_expr."""

    def test_different_operators(self):
        """Different comparison operators should be preserved."""
        for op in [">=", "<=", ">", "<"]:
            result = _pg_uninit().render_date_window("col", op, "day", 1)
            assert f"col {op} " in result

    def test_zero_offset_with_different_ops(self):
        """Zero offset with various operators should all produce DATE_TRUNC."""
        for op in [">=", "<"]:
            result = _pg_uninit().render_date_window("col", op, "week", 0)
            assert "DATE_TRUNC" in result


class TestDatabricksDialectInitFallback:
    """Tests for DatabricksDialect.__init__ connector-to-PySpark fallback."""

    def _config(self, *, native: bool = True) -> SimpleNamespace:
        """Return a minimal config namespace."""
        if native:
            return SimpleNamespace(
                CATALOG="cat",
                SCHEMA="sch",
                SERVER_HOSTNAME="host",
                HTTP_PATH="/sql",
                ACCESS_TOKEN="tok",
                DEBUG=False,
                has_native_connection=lambda: True,
                sqlalchemy_url=lambda: None,
            )
        return SimpleNamespace(
            CATALOG="cat",
            SCHEMA="sch",
            SERVER_HOSTNAME=None,
            HTTP_PATH=None,
            ACCESS_TOKEN=None,
            DEBUG=False,
            has_native_connection=lambda: False,
            sqlalchemy_url=lambda: None,
        )

    @staticmethod
    def _dbr_sql_module(*, connect_rv=None, connect_side_effect=None) -> MagicMock:
        """Build a fake ``databricks.sql`` module with a controlled ``connect``."""
        mod = MagicMock()
        if connect_side_effect is not None:
            mod.connect.side_effect = connect_side_effect
        else:
            mod.connect.return_value = connect_rv
        parent = MagicMock()
        parent.sql = mod
        return parent, mod

    @staticmethod
    def _pyspark_modules(*, spark_rv=None, side_effect=None) -> tuple[MagicMock, MagicMock]:
        """Build fake ``pyspark`` / ``pyspark.sql`` modules with a controlled ``SparkSession``."""
        mock_session_cls = MagicMock()
        if side_effect is not None:
            mock_session_cls.builder.getOrCreate.side_effect = side_effect
        else:
            mock_session_cls.builder.getOrCreate.return_value = spark_rv
        mod = MagicMock()
        mod.SparkSession = mock_session_cls
        parent = MagicMock()
        parent.sql = mod
        return parent, mod

    @patch("aetherdialect._dialect.Dialect.__init__", return_value=None)
    def test_connector_succeeds(self, _super_init: MagicMock) -> None:
        """When databricks-sql-connector connects, PySpark is not imported."""
        mock_conn = MagicMock()
        dbr_parent, dbr_mod = self._dbr_sql_module(connect_rv=mock_conn)
        with patch.dict("sys.modules", {"databricks": dbr_parent, "databricks.sql": dbr_mod}):
            d = DatabricksDialect(self._config(native=True))
        assert d.connection is mock_conn
        assert d.spark is None

    @patch("aetherdialect._dialect.Dialect.__init__", return_value=None)
    def test_connector_fails_falls_back_to_spark(self, _super_init: MagicMock) -> None:
        """When warehouse connector fails, SQLAlchemy is attempted before PySpark."""
        mock_spark = MagicMock()
        dbr_parent, dbr_mod = self._dbr_sql_module(connect_side_effect=RuntimeError("bad token"))
        ps_parent, ps_mod = self._pyspark_modules(spark_rv=mock_spark)
        with patch.dict(
            "sys.modules",
            {
                "databricks": dbr_parent,
                "databricks.sql": dbr_mod,
                "pyspark": ps_parent,
                "pyspark.sql": ps_mod,
            },
        ):
            d = DatabricksDialect(self._config(native=True))

        assert d.connection is None
        assert d.spark is mock_spark

    @patch("aetherdialect._dialect.Dialect.__init__", return_value=None)
    def test_no_credentials_uses_spark(self, _super_init: MagicMock) -> None:
        """When no warehouse credentials exist, PySpark is used directly."""
        mock_spark = MagicMock()
        ps_parent, ps_mod = self._pyspark_modules(spark_rv=mock_spark)
        with patch.dict("sys.modules", {"pyspark": ps_parent, "pyspark.sql": ps_mod}):
            d = DatabricksDialect(self._config(native=False))
        assert d.connection is None
        assert d.spark is mock_spark

    @patch("aetherdialect._dialect.Dialect.__init__", return_value=None)
    def test_both_fail_raises(self, _super_init: MagicMock) -> None:
        """When connector, SQLAlchemy, and PySpark all fail, ConfigError is raised."""
        from aetherdialect._config import ConfigError

        dbr_parent, dbr_mod = self._dbr_sql_module(connect_side_effect=RuntimeError("conn fail"))
        with patch.dict(
            "sys.modules",
            {
                "databricks": dbr_parent,
                "databricks.sql": dbr_mod,
                "pyspark": None,
                "pyspark.sql": None,
            },
        ):
            with pytest.raises(ConfigError, match="Databricks requires"):
                DatabricksDialect(self._config(native=True))


class TestAttachExtraFromAndWhere:
    """Tier-B semantic edge injection (FROM extension + WHERE AND- conjuncts)."""

    def _edge(self, left_tok: str, lc: str, right_tok: str, rc: str):
        from aetherdialect._dialect import JoinEdge

        return JoinEdge(
            table=right_tok,
            alias=None,
            kind="INNER",
            on_terms=((left_tok, lc, right_tok, rc),),
        )

    def test_postgres_empty_lists_noop_returns_true(self):
        pg = _pg_uninit()
        parsed = pg.parse_select("SELECT 1 FROM a WHERE TRUE")
        carriers = pg.ordered_join_carrier_froms(parsed)
        assert carriers
        assert pg.attach_extra_from_and_where(parsed, carriers[0], [], []) is True

    def test_postgres_appends_from_and_ands_where(self):
        pg = _pg_uninit()
        parsed = pg.parse_select("SELECT 1 FROM a WHERE a.flag = TRUE")
        carriers = pg.ordered_join_carrier_froms(parsed)
        assert carriers
        edges = [self._edge("a", "x", "b", "x")]
        assert pg.attach_extra_from_and_where(parsed, carriers[0], ["b"], edges) is True
        out = pg.emit_sql(parsed).lower()
        assert " from a, b" in out or "from a , b" in out or ", b " in out
        assert "a.x = b.x" in out
        assert "a.flag" in out

    def test_postgres_sets_where_when_absent(self):
        pg = _pg_uninit()
        parsed = pg.parse_select("SELECT 1 FROM a")
        carriers = pg.ordered_join_carrier_froms(parsed)
        assert carriers
        edges = [self._edge("a", "x", "b", "x")]
        assert pg.attach_extra_from_and_where(parsed, carriers[0], ["b"], edges) is True
        out = pg.emit_sql(parsed).lower()
        assert "where" in out
        assert "a.x = b.x" in out

    def test_databricks_empty_lists_noop_returns_true(self):
        dx = _dbr_uninit()
        parsed = dx.parse_select("SELECT 1 FROM a WHERE TRUE")
        carriers = dx.ordered_join_carrier_froms(parsed)
        assert carriers
        assert dx.attach_extra_from_and_where(parsed, carriers[0], [], []) is True

    def test_databricks_appends_from_and_ands_where(self):
        dx = _dbr_uninit()
        parsed = dx.parse_select("SELECT 1 FROM a WHERE a.flag = TRUE")
        carriers = dx.ordered_join_carrier_froms(parsed)
        assert carriers
        edges = [self._edge("a", "x", "b", "x")]
        assert dx.attach_extra_from_and_where(parsed, carriers[0], ["b"], edges) is True
        out = dx.emit_sql(parsed).lower()
        assert "b" in out
        assert "`a`.`x`" in out and "`b`.`x`" in out
        assert "a.flag" in out

    def test_databricks_sets_where_when_absent(self):
        dx = _dbr_uninit()
        parsed = dx.parse_select("SELECT 1 FROM a")
        carriers = dx.ordered_join_carrier_froms(parsed)
        assert carriers
        edges = [self._edge("a", "x", "b", "x")]
        assert dx.attach_extra_from_and_where(parsed, carriers[0], ["b"], edges) is True
        out = dx.emit_sql(parsed).lower()
        assert "where" in out
        assert "`a`.`x`" in out and "`b`.`x`" in out


def _mysql_uninit() -> MySQLDialect:
    return MySQLDialect.__new__(MySQLDialect)


def _redshift_uninit() -> RedshiftDialect:
    return RedshiftDialect.__new__(RedshiftDialect)


def _sqlserver_uninit() -> SQLServerDialect:
    return SQLServerDialect.__new__(SQLServerDialect)


def _snowflake_uninit() -> SnowflakeDialect:
    return SnowflakeDialect.__new__(SnowflakeDialect)


def _bigquery_uninit() -> BigQueryDialect:
    return BigQueryDialect.__new__(BigQueryDialect)


def _duckdb_uninit() -> DuckDBDialect:
    return DuckDBDialect.__new__(DuckDBDialect)


def _sqlite_uninit() -> SQLiteDialect:
    return SQLiteDialect.__new__(SQLiteDialect)


def _mariadb_uninit() -> MariaDBDialect:
    return MariaDBDialect.__new__(MariaDBDialect)


class TestMySQLDialect:
    """Basic render and quote tests for MySQL dialect."""

    def test_quote_table_column_backticks(self) -> None:
        d = _mysql_uninit()
        assert d.quote_table_column("orders", "order") == "`orders`.`order`"

    def test_render_date_window_day_offset(self) -> None:
        sql = _mysql_uninit().render_date_window("t.created", ">=", "day", 3)
        assert "DATE_SUB" in sql
        assert "t.created" in sql

    def test_render_case_insensitive_wrap(self) -> None:
        assert _mysql_uninit().render_case_insensitive_wrap("col") == "LOWER(col)"

    def test_date_window_upper_bound_sql(self) -> None:
        d = _mysql_uninit()
        assert d.date_window_upper_bound_sql("day") == "CURRENT_DATE"
        assert d.date_window_upper_bound_sql("hour") == "CURRENT_TIMESTAMP"


class TestDuckDBDialect:
    """Basic render and quote tests for DuckDB dialect."""

    def test_in_memory_shared_native_and_sqlalchemy_handles(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from aetherdialect._config import DuckDBRuntimeConfig, EngineConfig
        from aetherdialect._dialect import get_dialect
        from aetherdialect._dialect_sqlglot_engines import extract_static_pool_connection

        orig_path = EngineConfig.SCHEMA_JSON_PATH
        orig_type = EngineConfig.TYPE
        orig_runtime = EngineConfig.RUNTIME
        try:
            EngineConfig.SCHEMA_JSON_PATH = ""
            EngineConfig.TYPE = "duckdb"
            EngineConfig.RUNTIME = DuckDBRuntimeConfig
            connection = duckdb.connect(":memory:")
            connection.execute("CREATE TABLE shared_probe (id INTEGER)")
            connection.execute("INSERT INTO shared_probe VALUES (7)")
            dialect = get_dialect("duckdb", DuckDBRuntimeConfig, native_connection=connection)
            pooled = extract_static_pool_connection(dialect.engine)
            assert pooled is connection
            graph = dialect.reflect_schema_graph(include="tables")
            assert "shared_probe" in graph.tables
            assert dialect.execute("SELECT id FROM shared_probe") == [(7,)]
        finally:
            EngineConfig.SCHEMA_JSON_PATH = orig_path
            EngineConfig.TYPE = orig_type
            EngineConfig.RUNTIME = orig_runtime

    def test_quote_table_column_double_quotes(self) -> None:
        d = _duckdb_uninit()
        assert d.quote_table_column("orders", "order") == '"orders"."order"'

    def test_supports_ilike_and_unnest(self) -> None:
        d = _duckdb_uninit()
        assert d.supports_ilike is True
        assert d.supports_unnest_select_item is True

    def test_render_array_contains(self) -> None:
        sql = _duckdb_uninit().render_array_contains("film.special_features", "p0")
        assert "array_contains" in sql.lower() or "list_contains" in sql.lower()

    def test_render_date_diff(self) -> None:
        sql = _duckdb_uninit().render_date_diff("t.d", ">", "day", 7)
        assert "date_diff" in sql.lower()

    def test_date_window_upper_bound_sql(self) -> None:
        d = _duckdb_uninit()
        assert d.date_window_upper_bound_sql("day") == "current_date"
        assert d.date_window_upper_bound_sql("hour") == "current_timestamp"


class TestSQLiteDialect:
    """Basic render and quote tests for SQLite dialect."""

    def test_quote_table_column_double_quotes(self) -> None:
        d = _sqlite_uninit()
        assert d.quote_table_column("orders", "order") == '"orders"."order"'

    def test_supports_ilike_and_unnest_false(self) -> None:
        d = _sqlite_uninit()
        assert d.supports_ilike is False
        assert d.supports_unnest_select_item is False

    def test_render_array_contains(self) -> None:
        sql = _sqlite_uninit().render_array_contains("film.special_features", "p0")
        assert "instr" in sql.lower()
        assert "EXISTS(" not in sql.upper()

    def test_render_date_diff(self) -> None:
        sql = _sqlite_uninit().render_date_diff("t.d", ">", "day", 7)
        assert "julianday" in sql.lower()

    def test_render_date_window(self) -> None:
        sql = _sqlite_uninit().render_date_window("t.d", ">=", "day", 30)
        assert "date('now'" in sql.lower()

    def test_explain_row_estimate_none(self) -> None:
        assert _sqlite_uninit().explain_row_estimate("SELECT 1") is None


class TestMariaDBDialect:
    """Basic render and quote tests for MariaDB dialect."""

    def test_sqlglot_dialect_is_mysql(self) -> None:
        assert _mariadb_uninit().sqlglot_dialect == "mysql"

    def test_quote_table_column_backticks(self) -> None:
        d = _mariadb_uninit()
        assert d.quote_table_column("orders", "order") == "`orders`.`order`"

    def test_render_array_contains(self) -> None:
        sql = _mariadb_uninit().render_array_contains("film.special_features", "p0")
        assert "LOCATE" in sql or "INSTR" in sql
        assert "JSON_CONTAINS" not in sql.upper()

    def test_render_date_diff(self) -> None:
        sql = _mariadb_uninit().render_date_diff("t.end_date", ">", "day", 7)
        assert "TIMESTAMPDIFF" in sql.upper()


class TestRedshiftDialect:
    """Basic render and quote tests for Redshift dialect."""

    def test_quote_table_column_double_quotes(self) -> None:
        d = _redshift_uninit()
        assert d.quote_table_column("orders", "order") == '"orders"."order"'

    def test_render_case_insensitive_wrap(self) -> None:
        assert _redshift_uninit().render_case_insensitive_wrap("col") == "LOWER(col)"

    def test_date_window_upper_bound_sql(self) -> None:
        d = _redshift_uninit()
        assert d.date_window_upper_bound_sql("day") == "CURRENT_DATE"
        assert d.date_window_upper_bound_sql("hour") == "CURRENT_TIMESTAMP"


class TestSQLServerDialect:
    """Basic render and quote tests for SQL Server dialect."""

    def test_quote_table_column_brackets(self) -> None:
        d = _sqlserver_uninit()
        assert d.quote_table_column("orders", "order") == "[orders].[order]"

    def test_parse_select_emits_tsql(self) -> None:
        parsed = _sqlserver_uninit().parse_select("SELECT 1 AS x FROM dbo.t")
        assert parsed is not None
        assert "SELECT" in _sqlserver_uninit().emit_sql(parsed).upper()

    def test_render_case_insensitive_wrap(self) -> None:
        assert _sqlserver_uninit().render_case_insensitive_wrap("col") == "LOWER(col)"

    def test_date_window_upper_bound_sql(self) -> None:
        d = _sqlserver_uninit()
        assert "GETDATE()" in d.date_window_upper_bound_sql("hour")
        assert "DATE" in d.date_window_upper_bound_sql("day")


class TestSnowflakeDialect:
    """Basic render and quote tests for Snowflake dialect."""

    def test_quote_table_column_uppercase(self) -> None:
        d = _snowflake_uninit()
        assert d.quote_table_column("orders", "status") == "ORDERS.STATUS"

    def test_render_case_insensitive_wrap(self) -> None:
        assert _snowflake_uninit().render_case_insensitive_wrap("col") == "LOWER(col)"

    def test_date_window_upper_bound_sql(self) -> None:
        d = _snowflake_uninit()
        assert d.date_window_upper_bound_sql("day") == "CURRENT_DATE()"
        assert d.date_window_upper_bound_sql("hour") == "CURRENT_TIMESTAMP()"


class TestBigQueryDialect:
    """Basic render and quote tests for BigQuery dialect."""

    def test_quote_table_column_backticks(self) -> None:
        d = _bigquery_uninit()
        assert d.quote_table_column("proj.dataset.table", "col") == "`proj.dataset.table`.`col`"

    def test_pre_execute_rewrite_rewrites_placeholders(self) -> None:
        d = _bigquery_uninit()
        out = d.pre_execute_rewrite("SELECT :p1")
        assert "@@p1" not in out
        assert "@p1" in out

    def test_render_date_window_inclusive_upper_wraps_date(self) -> None:
        d = _bigquery_uninit()
        upper = d.render_date_window_inclusive_upper("rental.rental_date", "day")
        assert "DATE(rental.rental_date)" in upper
        assert "CURRENT_DATE()" in upper

    def test_date_window_upper_bound_sql(self) -> None:
        d = _bigquery_uninit()
        assert d.date_window_upper_bound_sql("day") == "CURRENT_DATE()"
        assert d.date_window_upper_bound_sql("hour") == "CURRENT_TIMESTAMP()"


class TestResultBackendKinds:
    """Mocked result-backend kind selection for native engine paths."""

    def test_databricks_connector_backend_fetch_rows(self) -> None:
        from aetherdialect._dialect_sqlglot_engines import DatabricksConnectorBackend

        class _Cursor:
            def execute(self, _sql: str) -> None:
                pass

            def fetchall(self):
                return [(1, "a")]

            def close(self) -> None:
                pass

        class _Conn:
            def cursor(self):
                return _Cursor()

        rows = DatabricksConnectorBackend(_Conn()).fetch_rows("SELECT 1")
        assert rows == [(1, "a")]
        assert DatabricksConnectorBackend.kind == "connector"

    def test_databricks_result_reader_kind_from_backend(self) -> None:
        from aetherdialect._dialect_sqlglot_engines import DatabricksConnectorBackend

        d = _dbr_uninit()
        d._backend = DatabricksConnectorBackend(object())
        assert d.result_reader_kind == "connector"

    def test_bigquery_storage_backend_kind(self) -> None:
        from aetherdialect._dialect_sqlglot_engines import BigQueryStorageBackend

        d = _bigquery_uninit()
        d._backend = BigQueryStorageBackend(object(), object())
        assert d.result_reader_kind == "bq_storage"
        assert BigQueryStorageBackend.kind == "bq_storage"

    def test_snowflake_arrow_backend_kind(self) -> None:
        from aetherdialect._dialect_sqlglot_engines import SnowflakeArrowBackend

        d = _snowflake_uninit()
        d._backend = SnowflakeArrowBackend(snowpark=object())
        assert d.result_reader_kind == "snowflake_arrow"

    def test_bigquery_client_backend_kind_and_fetch_rows(self, monkeypatch) -> None:
        from aetherdialect._dialect_sqlglot_engines import BigQueryClientBackend

        class _Row:
            def values(self):
                return [1, "a"]

        class _Job:
            def result(self):
                return [_Row()]

        class _Client:
            def query(self, _sql, job_config=None):
                _ = job_config
                return _Job()

        monkeypatch.setattr(BigQueryClientBackend, "_job_config", lambda self, **kwargs: object())
        rows = BigQueryClientBackend(_Client()).fetch_rows("SELECT 1")
        assert rows == [(1, "a")]
        assert BigQueryClientBackend.kind == "bq_client"

    def test_bigquery_storage_backend_fetch_rows(self, monkeypatch) -> None:
        from aetherdialect._dialect_sqlglot_engines import BigQueryStorageBackend

        class _Row:
            def values(self):
                return [7]

        class _Job:
            def result(self, bqstorage_client=None):
                _ = bqstorage_client
                return [_Row()]

        class _Client:
            def query(self, _sql, job_config=None):
                _ = job_config
                return _Job()

        from aetherdialect._dialect_sqlglot_engines import BigQueryClientBackend

        monkeypatch.setattr(BigQueryClientBackend, "_job_config", lambda self, **kwargs: object())
        backend = BigQueryStorageBackend(_Client(), object())
        assert backend.fetch_rows("SELECT 7") == [(7,)]

    def test_snowflake_arrow_backend_connector_fetch_rows(self) -> None:
        from aetherdialect._dialect_sqlglot_engines import SnowflakeArrowBackend

        class _Cursor:
            def execute(self, _sql: str) -> None:
                pass

            def fetchall(self):
                return [(3, "x")]

            def close(self) -> None:
                pass

        class _Conn:
            def cursor(self):
                return _Cursor()

        rows = SnowflakeArrowBackend(connection=_Conn()).fetch_rows("SELECT 3")
        assert rows == [(3, "x")]


class TestSnowflakeQualifyTablesForExecution:
    """Snowflake qualifies bare tables with uppercase unquoted three- part names."""

    def test_from_film_uppercases_table(self) -> None:
        d = _snowflake_uninit()
        d.config = SimpleNamespace(DATABASE="DVDRENTAL_NEW", SCHEMA="PUBLIC", DEBUG=False)
        d.sqlglot_dialect = "snowflake"
        result = d._qualify_tables_for_execution("SELECT title FROM film")
        assert 'PUBLIC."film"' not in result
        assert "DVDRENTAL_NEW.PUBLIC.FILM" in result.replace(" ", "")


class TestPerEngineQualifyTablesForExecution:
    """Each sqlglot engine dialect qualifies bare FROM tables for execution."""

    @pytest.mark.parametrize(
        ("factory", "dialect_name", "catalog_attr", "schema_attr"),
        [
            (lambda: PostgresDialect.__new__(PostgresDialect), "postgres", None, "public"),
            (lambda: MySQLDialect.__new__(MySQLDialect), "mysql", None, None),
            (lambda: _bigquery_uninit(), "bigquery", "PROJECT", "dataset"),
            (lambda: _dbr_uninit(), "databricks", "CATALOG", "SCHEMA"),
            (lambda: _sqlite_uninit(), "sqlite", None, "main"),
            (lambda: _duckdb_uninit(), "duckdb", None, "main"),
            (lambda: RedshiftDialect.__new__(RedshiftDialect), "redshift", None, "public"),
            (lambda: _sqlserver_uninit(), "tsql", None, "dbo"),
            (lambda: _snowflake_uninit(), "snowflake", "DATABASE", "SCHEMA"),
        ],
    )
    def test_qualifies_bare_table(self, factory, dialect_name, catalog_attr, schema_attr) -> None:
        d = factory()
        cfg = {"DEBUG": False}
        if catalog_attr:
            cfg[catalog_attr] = "CAT"
        if schema_attr:
            cfg["SCHEMA"] = schema_attr
        d.config = SimpleNamespace(**cfg)
        d.sqlglot_dialect = dialect_name
        if hasattr(d, "schema_name"):
            pass
        result = d._qualify_tables_for_execution("SELECT 1 FROM orders")
        assert "orders" in result.lower()
