"""Tests for dialect module: SQL dialect abstraction, AST validation, and date rendering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aetherdialect._dialect import (
    DatabricksDialect,
    Dialect,
    PostgresDialect,
    get_dialect,
    resolve_dialect,
)


def _pg_uninit() -> PostgresDialect:
    """Return a ``PostgresDialect`` instance without running ``__init__`` (render-only tests)."""

    return PostgresDialect.__new__(PostgresDialect)


def _dbr_uninit() -> DatabricksDialect:
    """Return a ``DatabricksDialect`` instance without running ``__init__`` (render-only tests)."""

    return DatabricksDialect.__new__(DatabricksDialect)


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

        from aetherdialect._dialect import _pg_node_kind, _pg_walk_nodes

        sql = "WITH cte1 AS (SELECT 1 AS x), cte2 AS (SELECT 1 AS y FROM cte1) SELECT * FROM cte1 CROSS JOIN cte2"
        root = parse_sql(sql)[0].stmt
        rels = {
            n.relname for n in _pg_walk_nodes(root) if _pg_node_kind(n) == "RangeVar" and getattr(n, "relname", None)
        }
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
            get_dialect("mysql", MagicMock())

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

    def test_quote_table_column_base_is_unquoted(self):
        """Base dialect leaves identifiers unquoted."""

        d = Dialect(config=MagicMock())
        assert d.quote_table_column("orders", "id") == "orders.id"


class TestPostgresExplainPermissionHandling:
    """Tests for ``PostgresDialect.explain_sql`` permission-denied self-disable behaviour."""

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
    """Tests for ``DatabricksDialect._ast_spark_structural_valid`` (sqlglot, parity with Postgres intent)."""

    @pytest.fixture
    def dbr(self):
        """Uninitialized Databricks dialect for AST helpers."""

        return _dbr_uninit()

    def test_simple_select_passes(self, dbr):
        ok, err = dbr._ast_spark_structural_valid("SELECT customer_id, name FROM customers")
        assert ok is True
        assert err == ""

    def test_join_on_passes(self, dbr):
        sql = "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
        ok, err = dbr._ast_spark_structural_valid(sql)
        assert ok is True, err

    def test_recursive_cte_rejected(self, dbr):
        sql = "WITH RECURSIVE r AS (SELECT 1 AS n) SELECT * FROM r"
        ok, err = dbr._ast_spark_structural_valid(sql)
        assert ok is False
        assert err == "cte_recursive"

    def test_cte_union_rejected(self, dbr):
        sql = "WITH a AS (SELECT 1 UNION SELECT 2) SELECT * FROM a"
        ok, err = dbr._ast_spark_structural_valid(sql)
        assert ok is False
        assert err == "cte_contains_set_op"

    def test_exists_rejected(self, dbr):
        sql = "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)"
        ok, err = dbr._ast_spark_structural_valid(sql)
        assert ok is False
        assert err == "exists_not_allowed"

    def test_subquery_in_from_rejected(self, dbr):
        ok, err = dbr._ast_spark_structural_valid("SELECT * FROM (SELECT 1 AS x) t")
        assert ok is False
        assert err == "subquery_not_allowed"

    def test_cross_join_rejected(self, dbr):
        ok, err = dbr._ast_spark_structural_valid("SELECT * FROM a CROSS JOIN b")
        assert ok is False
        assert err == "cross_join_not_allowed"

    def test_cross_join_scalar_cte_allowed_when_whitelisted(self, dbr):
        sql = "WITH c AS (SELECT 1 AS x) SELECT * FROM customers CROSS JOIN c"
        ok, err = dbr._ast_spark_structural_valid(sql, scalar_cte_names=frozenset({"c"}))
        assert ok is True, err

    def test_using_join_rejected(self, dbr):
        ok, err = dbr._ast_spark_structural_valid("SELECT * FROM a JOIN b USING (id)")
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


class TestPrepareForExecution:
    """Tests for DatabricksDialect.prepare_for_execution."""

    def _make_dialect(self):
        """Create a DatabricksDialect with mocked internals."""
        mock_config = SimpleNamespace(CATALOG="dev", SCHEMA="dvdrental", DEBUG=False)
        d = DatabricksDialect.__new__(DatabricksDialect)
        d.config = mock_config
        return d

    def test_qualifies_table_references(self):
        """prepare_for_execution qualifies table references."""
        d = self._make_dialect()
        result = d.prepare_for_execution("SELECT * FROM customers")
        assert "dev" in result
        assert "dvdrental" in result
        assert "customers" in result


class TestQualifyTableReferences:
    """Tests for DatabricksDialect._qualify_table_references."""

    def _make_dialect(self):
        """Create a DatabricksDialect with mocked internals."""
        mock_config = SimpleNamespace(CATALOG="dev", SCHEMA="dvdrental", DEBUG=False)
        d = DatabricksDialect.__new__(DatabricksDialect)
        d.config = mock_config
        return d

    def test_from_clause_qualified(self):
        """Table in FROM should get catalog.schema prefix."""
        d = self._make_dialect()
        result = d._qualify_table_references("SELECT * FROM customers")
        assert "dev" in result
        assert "dvdrental" in result
        assert "customers" in result

    def test_join_clause_qualified(self):
        """Table in JOIN should get catalog.schema prefix."""
        d = self._make_dialect()
        result = d._qualify_table_references("SELECT * FROM orders JOIN customers ON orders.id = customers.id")
        assert result.count("dev") >= 2

    def test_extract_from_max_not_catalog_qualified(self):
        """Do not qualify MAX after EXTRACT(... FROM ...)."""
        d = self._make_dialect()
        sql = "SELECT EXTRACT(YEAR FROM MAX(rental.rental_date)) AS y FROM rental GROUP BY rental.customer_id"
        result = d._qualify_table_references(sql)
        assert "`dev`.`dvdrental`.`MAX`" not in result
        assert "`dev`.`dvdrental`.`rental`" in result

    def test_three_part_dotted_reference_not_double_wrapped(self):
        """Leave ``catalog.schema.table`` dotted references qualified without double-wrapping."""
        d = self._make_dialect()
        sql = "SELECT title FROM dev.dvdrental.film"
        result = d._qualify_table_references(sql)
        assert "`dev`.`dvdrental`.`dev`" not in result
        assert "dev.dvdrental.dev" not in result
        assert "film" in result

    def test_three_part_backticked_reference_not_double_wrapped(self):
        """Leave fully backtick-qualified references qualified without double-wrapping."""
        d = self._make_dialect()
        sql = "SELECT title FROM `dev`.`dvdrental`.`film`"
        result = d._qualify_table_references(sql)
        assert "`dev`.`dvdrental`.`dev`" not in result
        assert "dev.dvdrental.dev" not in result
        assert "film" in result


class TestDatabricksNormalizeDatetruncSql:
    """``_databricks_normalize_datetrunc_sql`` normalizes Anonymous DATETRUNC call sites."""

    def test_rewrites_to_date_trunc_with_unit_first(self):
        from aetherdialect._dialect import _databricks_normalize_datetrunc_sql

        sql = "SELECT DATETRUNC('MONTH', rental.created_at) AS m FROM rental"
        out = _databricks_normalize_datetrunc_sql(sql)
        assert "DATETRUNC(" not in out
        assert "DATE_TRUNC('MONTH'" in out


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
    def test_connector_fails_with_warehouse_creds_raises(self, _super_init: MagicMock) -> None:
        """
        When warehouse credentials are configured and the connector raises, the dialect
        surfaces the connector error instead of silently falling back to PySpark (which
        could bind to an unrelated SPARK_REMOTE / Databricks Connect endpoint).
        """
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
            with pytest.raises(RuntimeError, match="databricks-sql-connector failed"):
                DatabricksDialect(self._config(native=True))

        assert ps_mod.SparkSession.builder.getOrCreate.call_count == 0

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
        """When connector and PySpark both fail, RuntimeError is raised."""
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
            with pytest.raises(RuntimeError, match="databricks-sql-connector failed"):
                DatabricksDialect(self._config(native=True))


class TestAttachExtraFromAndWhere:
    """Tier-B semantic edge injection (FROM extension + WHERE AND-conjuncts)."""

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
