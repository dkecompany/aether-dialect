"""Tests for seven-engine TOML parsing, env aliases, and accelerator auto-detect."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aetherdialect._config import (
    BigQueryRuntimeConfig,
    MySQLRuntimeConfig,
    RedshiftRuntimeConfig,
    SnowflakeRuntimeConfig,
    SQLServerRuntimeConfig,
)


def _restore_mysql(**kwargs: object) -> dict[str, object]:
    saved = {
        "HOST": MySQLRuntimeConfig.HOST,
        "PORT": MySQLRuntimeConfig.PORT,
        "USER": MySQLRuntimeConfig.USER,
        "PASSWORD": MySQLRuntimeConfig.PASSWORD,
        "DATABASE": MySQLRuntimeConfig.DATABASE,
        "SCHEMA": MySQLRuntimeConfig.SCHEMA,
    }
    for key, value in kwargs.items():
        setattr(MySQLRuntimeConfig, key, value)
    return saved


def _restore_all_mysql(saved: dict[str, object]) -> None:
    for key, value in saved.items():
        setattr(MySQLRuntimeConfig, key, value)


class TestMySQLRuntimeConfigParity:
    """MySQL URL and env alias parity."""

    def test_db_url_hardcodes_utf8mb4_charset(self) -> None:
        saved = _restore_mysql(PASSWORD="secret", DATABASE="app")
        try:
            url = MySQLRuntimeConfig.db_url()
            assert url.endswith("?charset=utf8mb4")
            assert "CHARSET" not in url.upper().replace("CHARSET=UTF8MB4", "")
        finally:
            _restore_all_mysql(saved)

    def test_apply_environment_resolves_standard_aliases(self) -> None:
        saved = _restore_mysql()
        try:
            MySQLRuntimeConfig.apply_environment(
                {
                    "MYSQL_TCP_PORT": "3307",
                    "MYSQL_PWD": "pw",
                    "MYSQL_DB": "shop",
                    "MYSQL_USER": "app",
                    "MYSQL_HOST": "db.local",
                },
            )
            assert MySQLRuntimeConfig.PORT == 3307
            assert MySQLRuntimeConfig.PASSWORD == "pw"
            assert MySQLRuntimeConfig.DATABASE == "shop"
            assert MySQLRuntimeConfig.SCHEMA == "shop"
            assert MySQLRuntimeConfig.USER == "app"
            assert MySQLRuntimeConfig.HOST == "db.local"
        finally:
            _restore_all_mysql(saved)

    def test_apply_environment_clears_password_when_toml_empty(self) -> None:
        saved = _restore_mysql(PASSWORD="old")
        try:
            MySQLRuntimeConfig.apply_environment({"MYSQL_PASSWORD": ""})
            assert MySQLRuntimeConfig.PASSWORD is None
        finally:
            _restore_all_mysql(saved)


class TestSQLServerRuntimeConfigParity:
    """SQL Server env alias parity."""

    def test_apply_environment_resolves_mssql_aliases(self) -> None:
        orig = {
            "HOST": SQLServerRuntimeConfig.HOST,
            "PORT": SQLServerRuntimeConfig.PORT,
            "USER": SQLServerRuntimeConfig.USER,
            "PASSWORD": SQLServerRuntimeConfig.PASSWORD,
            "DATABASE": SQLServerRuntimeConfig.DATABASE,
        }
        try:
            SQLServerRuntimeConfig.apply_environment(
                {
                    "MSSQL_SERVER": "sql.internal",
                    "MSSQL_PORT": "1444",
                    "MSSQL_USER": "sa",
                    "MSSQL_SA_PASSWORD": "pw",
                    "MSSQL_DATABASE": "app",
                },
            )
            assert SQLServerRuntimeConfig.HOST == "sql.internal"
            assert SQLServerRuntimeConfig.PORT == 1444
            assert SQLServerRuntimeConfig.USER == "sa"
            assert SQLServerRuntimeConfig.PASSWORD == "pw"
            assert SQLServerRuntimeConfig.DATABASE == "app"
        finally:
            for key, value in orig.items():
                setattr(SQLServerRuntimeConfig, key, value)


class TestSnowflakeRuntimeConfigParity:
    """Snowflake env alias parity."""

    def test_apply_environment_resolves_snowsql_aliases(self) -> None:
        orig = {
            "ACCOUNT": SnowflakeRuntimeConfig.ACCOUNT,
            "USER": SnowflakeRuntimeConfig.USER,
            "PASSWORD": SnowflakeRuntimeConfig.PASSWORD,
            "DATABASE": SnowflakeRuntimeConfig.DATABASE,
            "WAREHOUSE": SnowflakeRuntimeConfig.WAREHOUSE,
        }
        try:
            SnowflakeRuntimeConfig.apply_environment(
                {
                    "SNOWSQL_ACCOUNT": "xy123",
                    "SNOWSQL_USER": "u",
                    "SNOWSQL_PWD": "pw",
                    "SNOWSQL_DATABASE": "db",
                    "SNOWSQL_WAREHOUSE": "wh",
                },
            )
            assert SnowflakeRuntimeConfig.ACCOUNT == "xy123"
            assert SnowflakeRuntimeConfig.USER == "u"
            assert SnowflakeRuntimeConfig.PASSWORD == "pw"
            assert SnowflakeRuntimeConfig.DATABASE == "db"
            assert SnowflakeRuntimeConfig.WAREHOUSE == "wh"
        finally:
            for key, value in orig.items():
                setattr(SnowflakeRuntimeConfig, key, value)

    def test_snowpark_session_reachable_false_without_package(self) -> None:
        real_import = __import__("builtins").__import__

        def _import(name, *args, **kwargs):
            if name == "snowflake.snowpark.context":
                raise ImportError("no snowpark")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", _import):
            assert SnowflakeRuntimeConfig.snowpark_session_reachable() is False


class TestBigQueryRuntimeConfigParity:
    """BigQuery env alias parity."""

    def test_apply_environment_resolves_gcp_aliases(self) -> None:
        orig = {
            "PROJECT": BigQueryRuntimeConfig.PROJECT,
            "DATASET": BigQueryRuntimeConfig.DATASET,
            "SCHEMA": BigQueryRuntimeConfig.SCHEMA,
            "CREDENTIALS_PATH": BigQueryRuntimeConfig.CREDENTIALS_PATH,
        }
        try:
            BigQueryRuntimeConfig.apply_environment(
                {
                    "GCP_PROJECT": "proj",
                    "BIGQUERY_DATASET": "ds",
                    "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/sa.json",
                },
            )
            assert BigQueryRuntimeConfig.PROJECT == "proj"
            assert BigQueryRuntimeConfig.DATASET == "ds"
            assert BigQueryRuntimeConfig.SCHEMA == "ds"
            assert BigQueryRuntimeConfig.CREDENTIALS_PATH == "/tmp/sa.json"
        finally:
            for key, value in orig.items():
                setattr(BigQueryRuntimeConfig, key, value)


class TestRedshiftRuntimeConfigParity:
    """Redshift env alias parity."""

    def test_apply_environment_resolves_redshift_native_aliases(self) -> None:
        orig = {
            "HOST": RedshiftRuntimeConfig.HOST,
            "PORT": RedshiftRuntimeConfig.PORT,
            "USER": RedshiftRuntimeConfig.USER,
            "PASSWORD": RedshiftRuntimeConfig.PASSWORD,
            "DATABASE": RedshiftRuntimeConfig.DATABASE,
            "SCHEMA": RedshiftRuntimeConfig.SCHEMA,
        }
        try:
            RedshiftRuntimeConfig.apply_environment(
                {
                    "REDSHIFT_SERVER": "rs.local",
                    "REDSHIFT_TCP_PORT": "5440",
                    "REDSHIFT_USERNAME": "u",
                    "REDSHIFT_PWD": "pw",
                    "REDSHIFT_DB": "dev",
                    "REDSHIFT_SCHEMA": "analytics",
                },
            )
            assert RedshiftRuntimeConfig.HOST == "rs.local"
            assert RedshiftRuntimeConfig.PORT == 5440
            assert RedshiftRuntimeConfig.USER == "u"
            assert RedshiftRuntimeConfig.PASSWORD == "pw"
            assert RedshiftRuntimeConfig.DATABASE == "dev"
            assert RedshiftRuntimeConfig.SCHEMA == "analytics"
        finally:
            for key, value in orig.items():
                setattr(RedshiftRuntimeConfig, key, value)

    def test_db_url_uses_supported_sslmode(self) -> None:
        orig = {
            "HOST": RedshiftRuntimeConfig.HOST,
            "PORT": RedshiftRuntimeConfig.PORT,
            "USER": RedshiftRuntimeConfig.USER,
            "PASSWORD": RedshiftRuntimeConfig.PASSWORD,
            "DATABASE": RedshiftRuntimeConfig.DATABASE,
            "USE_IAM": RedshiftRuntimeConfig.USE_IAM,
            "CLUSTER_IDENTIFIER": RedshiftRuntimeConfig.CLUSTER_IDENTIFIER,
            "WORKGROUP": RedshiftRuntimeConfig.WORKGROUP,
        }
        try:
            RedshiftRuntimeConfig.HOST = "rs.local"
            RedshiftRuntimeConfig.PORT = 5439
            RedshiftRuntimeConfig.USER = "admin"
            RedshiftRuntimeConfig.PASSWORD = "pw"
            RedshiftRuntimeConfig.DATABASE = "dev"
            RedshiftRuntimeConfig.USE_IAM = False
            RedshiftRuntimeConfig.CLUSTER_IDENTIFIER = None
            RedshiftRuntimeConfig.WORKGROUP = None
            url = RedshiftRuntimeConfig.db_url()
            assert "sslmode=verify-full" in url
            assert "sslmode=require" not in url
            assert RedshiftRuntimeConfig.connect_args() == {
                "ssl": True,
                "sslmode": "verify-full",
            }
        finally:
            for key, value in orig.items():
                setattr(RedshiftRuntimeConfig, key, value)

    def test_env_complete_does_not_treat_libpq_vars_as_redshift_selection(self) -> None:
        env = {
            "PGDATABASE": "rental_shop",
            "PGUSER": "postgres",
            "PGPASSWORD": "secret",
        }
        assert RedshiftRuntimeConfig.env_complete(env) is False

    def test_apply_environment_ignores_removed_libpq_host_alias(self) -> None:
        orig = {
            "HOST": RedshiftRuntimeConfig.HOST,
            "PORT": RedshiftRuntimeConfig.PORT,
        }
        try:
            RedshiftRuntimeConfig.apply_environment(
                {
                    "PGHOST": "should-not-apply",
                    "REDSHIFT_HOST": "rs.local",
                },
            )
            assert RedshiftRuntimeConfig.HOST == "rs.local"
        finally:
            for key, value in orig.items():
                setattr(RedshiftRuntimeConfig, key, value)


class TestSevenEngineTomlParsing:
    """All five sqlglot-engine TOML blocks flatten to env keys."""

    def test_load_config_file_parses_five_sqlglot_engines(self, tmp_path) -> None:
        from aetherdialect._main_execution import MainExecutionOps

        path = tmp_path / "engines.toml"
        path.write_text(
            "\n".join(
                (
                    "[mysql]",
                    'host = "mh"',
                    "port = 3307",
                    'user = "mu"',
                    'password = "mp"',
                    'database = "mdb"',
                    "",
                    "[sqlserver]",
                    'host = "sh"',
                    "port = 1434",
                    'user = "su"',
                    'password = "sp"',
                    'database = "sdb"',
                    'schema = "custom"',
                    'auth_mode = "sql"',
                    "",
                    "[snowflake]",
                    'account = "acct"',
                    'user = "sfu"',
                    'password = "sfp"',
                    'database = "sfdb"',
                    'warehouse = "wh"',
                    "",
                    "[bigquery]",
                    'project = "bp"',
                    'dataset = "bd"',
                    'credentials_path = "/tmp/key.json"',
                    'location = "EU"',
                    "",
                    "[redshift]",
                    'host = "rh"',
                    "port = 5440",
                    'user = "ru"',
                    'password = "rp"',
                    'database = "rdb"',
                    'use_iam = "true"',
                    'cluster_identifier = "cid"',
                    'region = "us-east-1"',
                ),
            ),
            encoding="utf-8",
        )
        got, claimed, _named = MainExecutionOps._load_config_file(str(path))
        expected = {
            "MYSQL_HOST": "mh",
            "MYSQL_PORT": "3307",
            "MYSQL_USER": "mu",
            "MYSQL_PASSWORD": "mp",
            "MYSQL_DATABASE": "mdb",
            "SQLSERVER_HOST": "sh",
            "SQLSERVER_PORT": "1434",
            "SQLSERVER_USER": "su",
            "SQLSERVER_PASSWORD": "sp",
            "SQLSERVER_DATABASE": "sdb",
            "SQLSERVER_SCHEMA": "custom",
            "SQLSERVER_AUTH_MODE": "sql",
            "SNOWFLAKE_ACCOUNT": "acct",
            "SNOWFLAKE_USER": "sfu",
            "SNOWFLAKE_PASSWORD": "sfp",
            "SNOWFLAKE_DATABASE": "sfdb",
            "SNOWFLAKE_WAREHOUSE": "wh",
            "BIGQUERY_PROJECT": "bp",
            "BIGQUERY_DATASET": "bd",
            "BIGQUERY_CREDENTIALS_PATH": "/tmp/key.json",
            "BIGQUERY_LOCATION": "EU",
            "REDSHIFT_HOST": "rh",
            "REDSHIFT_PORT": "5440",
            "REDSHIFT_USER": "ru",
            "REDSHIFT_PASSWORD": "rp",
            "REDSHIFT_DATABASE": "rdb",
            "REDSHIFT_USE_IAM": "true",
            "REDSHIFT_CLUSTER_IDENTIFIER": "cid",
            "REDSHIFT_REGION": "us-east-1",
        }
        assert got == expected
        assert frozenset(expected.keys()) <= claimed


class TestAcceleratorAutoDetect:
    """Snowpark and BigQuery storage backends auto-detect without toggles."""

    def test_snowflake_uses_connector_when_no_ambient_snowpark(self, monkeypatch) -> None:
        from aetherdialect._config import SnowflakeRuntimeConfig
        from aetherdialect._dialect_sqlglot_engines import SnowflakeDialect

        monkeypatch.setattr(
            SnowflakeRuntimeConfig,
            "snowpark_session_reachable",
            classmethod(lambda cls: False),
        )
        monkeypatch.setattr(SnowflakeRuntimeConfig, "ACCOUNT", "acct", raising=False)
        monkeypatch.setattr(SnowflakeRuntimeConfig, "USER", "u", raising=False)
        monkeypatch.setattr(SnowflakeRuntimeConfig, "PASSWORD", "pw", raising=False)
        connector_mod = MagicMock()
        connector_mod.connect.return_value = object()
        snowflake_pkg = MagicMock()
        snowflake_pkg.connector = connector_mod
        monkeypatch.setitem(__import__("sys").modules, "snowflake", snowflake_pkg)
        monkeypatch.setitem(__import__("sys").modules, "snowflake.connector", connector_mod)
        dialect = SnowflakeDialect(SnowflakeRuntimeConfig, sqlalchemy_engine=MagicMock())
        assert dialect._snowpark_session is None
        connector_mod.connect.assert_called_once()
        assert dialect._snowflake_connection is not None

    def test_bigquery_skips_storage_client_on_import_error(self, monkeypatch) -> None:
        import types

        from aetherdialect._config import BigQueryRuntimeConfig
        from aetherdialect._dialect_sqlglot_engines import BigQueryDialect

        bigquery_mod = MagicMock()
        bigquery_mod.Client.return_value = object()
        google_cloud_mod = types.ModuleType("google.cloud")
        google_cloud_mod.bigquery = bigquery_mod
        monkeypatch.setitem(__import__("sys").modules, "google.cloud", google_cloud_mod)
        monkeypatch.setitem(__import__("sys").modules, "google.cloud.bigquery", bigquery_mod)

        real_import = __import__("builtins").__import__

        def _import(name, *args, **kwargs):
            if name == "google.cloud.bigquery_storage":
                raise ImportError("no storage")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _import)
        with patch.object(BigQueryDialect, "_select_result_backend"):
            dialect = BigQueryDialect(BigQueryRuntimeConfig, sqlalchemy_engine=MagicMock())
        assert dialect._bq_client is not None
        assert dialect._bq_storage_client is None


class TestMariaDBRuntimeConfigParity:
    """MariaDB env alias parity."""

    def test_apply_environment_resolves_mariadb_aliases(self) -> None:
        from aetherdialect._config import MariaDBRuntimeConfig

        orig = {
            "HOST": MariaDBRuntimeConfig.HOST,
            "PORT": MariaDBRuntimeConfig.PORT,
            "USER": MariaDBRuntimeConfig.USER,
            "PASSWORD": MariaDBRuntimeConfig.PASSWORD,
            "DATABASE": MariaDBRuntimeConfig.DATABASE,
        }
        try:
            MariaDBRuntimeConfig.apply_environment(
                {
                    "MARIADB_SERVER": "mdb.local",
                    "MARIADB_TCP_PORT": "3308",
                    "MARIADB_USERNAME": "app",
                    "MARIADB_PWD": "pw",
                    "MARIADB_DB": "shop",
                },
            )
            assert MariaDBRuntimeConfig.HOST == "mdb.local"
            assert MariaDBRuntimeConfig.PORT == 3308
            assert MariaDBRuntimeConfig.USER == "app"
            assert MariaDBRuntimeConfig.PASSWORD == "pw"
            assert MariaDBRuntimeConfig.DATABASE == "shop"
        finally:
            for key, value in orig.items():
                setattr(MariaDBRuntimeConfig, key, value)


class TestDuckDBRuntimeConfigParity:
    """DuckDB env alias parity."""

    def test_apply_environment_resolves_path_aliases(self) -> None:
        from aetherdialect._config import DuckDBRuntimeConfig

        orig_path = DuckDBRuntimeConfig.DATABASE_PATH
        try:
            DuckDBRuntimeConfig.apply_environment({"DUCKDB_FILE": "/data/test.duckdb"})
            assert DuckDBRuntimeConfig.DATABASE_PATH == "/data/test.duckdb"
        finally:
            DuckDBRuntimeConfig.DATABASE_PATH = orig_path

    def test_attach_connection_survives_apply_environment(self) -> None:
        from aetherdialect._config import DuckDBRuntimeConfig

        sentinel = object()
        orig_path = DuckDBRuntimeConfig.DATABASE_PATH
        orig_connection = DuckDBRuntimeConfig.NATIVE_CONNECTION
        try:
            DuckDBRuntimeConfig.attach_connection(sentinel)
            DuckDBRuntimeConfig.apply_environment({"DUCKDB_FILE": "/data/other.duckdb"})
            assert DuckDBRuntimeConfig.NATIVE_CONNECTION is sentinel
            assert DuckDBRuntimeConfig.DATABASE_PATH == "/data/other.duckdb"
        finally:
            DuckDBRuntimeConfig.DATABASE_PATH = orig_path
            DuckDBRuntimeConfig.NATIVE_CONNECTION = orig_connection

    def test_clear_attached_connection_resets_slot(self) -> None:
        from aetherdialect._config import DuckDBRuntimeConfig

        orig_connection = DuckDBRuntimeConfig.NATIVE_CONNECTION
        try:
            DuckDBRuntimeConfig.attach_connection(object())
            DuckDBRuntimeConfig.clear_attached_connection()
            assert DuckDBRuntimeConfig.NATIVE_CONNECTION is None
        finally:
            DuckDBRuntimeConfig.NATIVE_CONNECTION = orig_connection


class TestSQLiteRuntimeConfigParity:
    """SQLite env alias parity."""

    def test_apply_environment_resolves_path_aliases(self) -> None:
        from aetherdialect._config import SQLiteRuntimeConfig

        orig_path = SQLiteRuntimeConfig.DATABASE_PATH
        try:
            SQLiteRuntimeConfig.apply_environment({"SQLITE3_DATABASE": "/data/test.sqlite"})
            assert SQLiteRuntimeConfig.DATABASE_PATH == "/data/test.sqlite"
        finally:
            SQLiteRuntimeConfig.DATABASE_PATH = orig_path

    def test_attach_connection_survives_apply_environment(self) -> None:
        from aetherdialect._config import SQLiteRuntimeConfig

        sentinel = object()
        orig_path = SQLiteRuntimeConfig.DATABASE_PATH
        orig_connection = SQLiteRuntimeConfig.NATIVE_CONNECTION
        try:
            SQLiteRuntimeConfig.attach_connection(sentinel)
            SQLiteRuntimeConfig.apply_environment({"SQLITE3_DATABASE": "/data/other.sqlite"})
            assert SQLiteRuntimeConfig.NATIVE_CONNECTION is sentinel
            assert SQLiteRuntimeConfig.DATABASE_PATH == "/data/other.sqlite"
        finally:
            SQLiteRuntimeConfig.DATABASE_PATH = orig_path
            SQLiteRuntimeConfig.NATIVE_CONNECTION = orig_connection

    def test_clear_attached_connection_resets_slot(self) -> None:
        from aetherdialect._config import SQLiteRuntimeConfig

        orig_connection = SQLiteRuntimeConfig.NATIVE_CONNECTION
        try:
            SQLiteRuntimeConfig.attach_connection(object())
            SQLiteRuntimeConfig.clear_attached_connection()
            assert SQLiteRuntimeConfig.NATIVE_CONNECTION is None
        finally:
            SQLiteRuntimeConfig.NATIVE_CONNECTION = orig_connection
