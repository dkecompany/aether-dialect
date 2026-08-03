"""Engine and policy settings, tunable thresholds, and shared validation constants. `BOOLEAN_TRUTH_PATTERN_MAP` maps lowercased two-valued top-K sets to the canonical affirmative literal (lowercase) used when recording ``ColumnMetadata.boolean_truth_value``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

from ._constants import (
    BIGQUERY_ENV_CREDENTIALS_PATH,
    BIGQUERY_ENV_DATASET,
    BIGQUERY_ENV_LOCATION,
    BIGQUERY_ENV_PROJECT,
    CSV_ENV_DIRECTORY,
    CSV_ENV_FILES,
    DATABRICKS_ENV_CATALOG,
    DATABRICKS_ENV_HTTP_PATH,
    DATABRICKS_ENV_SCHEMA,
    DATABRICKS_ENV_SERVER_HOSTNAME,
    DATABRICKS_ENV_TOKEN,
    DEFAULT_RANDOM_SEED,
    DUCKDB_ENV_PATH,
    DUCKDB_ENV_SCHEMA,
    ENGINE_STORAGE_PLACEHOLDER_DIR,
    EXCLUDED_WHERE_PATTERNS,
    MARIADB_ENV_DATABASE,
    MARIADB_ENV_HOST,
    MARIADB_ENV_PASSWORD,
    MARIADB_ENV_PORT,
    MARIADB_ENV_USER,
    MYSQL_ENV_DATABASE,
    MYSQL_ENV_HOST,
    MYSQL_ENV_PASSWORD,
    MYSQL_ENV_PORT,
    MYSQL_ENV_USER,
    POSTGRES_ENV_DATABASE,
    POSTGRES_ENV_HOST,
    POSTGRES_ENV_PASSWORD,
    POSTGRES_ENV_PORT,
    POSTGRES_ENV_SCHEMA,
    POSTGRES_ENV_USER,
    REDSHIFT_ENV_CLUSTER_IDENTIFIER,
    REDSHIFT_ENV_DATABASE,
    REDSHIFT_ENV_HOST,
    REDSHIFT_ENV_PASSWORD,
    REDSHIFT_ENV_PORT,
    REDSHIFT_ENV_REGION,
    REDSHIFT_ENV_SCHEMA,
    REDSHIFT_ENV_USE_IAM,
    REDSHIFT_ENV_USER,
    REDSHIFT_ENV_WORKGROUP,
    SEED_WARMUP_CACHE_ZIP,
    SNOWFLAKE_ENV_ACCOUNT,
    SNOWFLAKE_ENV_AUTHENTICATOR,
    SNOWFLAKE_ENV_DATABASE,
    SNOWFLAKE_ENV_OAUTH_TOKEN,
    SNOWFLAKE_ENV_PASSWORD,
    SNOWFLAKE_ENV_PRIVATE_KEY_PASSPHRASE,
    SNOWFLAKE_ENV_PRIVATE_KEY_PATH,
    SNOWFLAKE_ENV_ROLE,
    SNOWFLAKE_ENV_SCHEMA,
    SNOWFLAKE_ENV_USER,
    SNOWFLAKE_ENV_WAREHOUSE,
    SQLITE_ENV_PATH,
    SQLSERVER_ENV_AUTH_MODE,
    SQLSERVER_ENV_CLIENT_ID,
    SQLSERVER_ENV_CLIENT_SECRET,
    SQLSERVER_ENV_DATABASE,
    SQLSERVER_ENV_DRIVER,
    SQLSERVER_ENV_HOST,
    SQLSERVER_ENV_PASSWORD,
    SQLSERVER_ENV_PORT,
    SQLSERVER_ENV_SCHEMA,
    SQLSERVER_ENV_TENANT_ID,
    SQLSERVER_ENV_USER,
    STOPWORDS_GRAMMATICAL_PARTICLES,
    WARMUP_ANCHOR_LATTICE_SUBDIR,
    env_any_nonempty,
    env_first_nonempty,
    env_role_hint,
    package_importable,
)
from ._contracts_base import ConfigError


def _read_optional_positive_float_env(name: str, *, default: float | None = None) -> float | None:
    """Read a non-negative float from *os.environ[name]*, returning *default* if missing or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
        return val if val > 0.0 else None
    except ValueError:
        return default


def _read_optional_positive_int_env(name: str, *, default: int | None = None) -> int | None:
    """Read a non-negative integer from *os.environ[name]*, returning *default* if missing or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw, 10)
        return val if val > 0 else None
    except ValueError:
        return default


class PolicyConfig:
    """ClassVar thresholds, penalties, stopwords, and SQL rejection patterns. Developer tracing: set ClassVar ``DEBUG``. ``telemetry_capture(..., force_diagnostic_flags=True)`` bumps an internal depth counter so diagnostics emit into the capture buffer without mutating ClassVars. Live tests opt in per session via ``live_tests/conftest.py`` (``PolicyConfig.DEBUG``) so failures and optional full logs can be written to ``live_tests/results.txt``. Cache rebuild shortcuts: set ``REGENERATE_TEMPLATE_STORE``, ``REGENERATE_SCHEMA_GRAPH``, or ``REGENERATE_SKELETON_CACHE`` to skip loading the corresponding on-disk artifact when present. Semantic join hints (non-FK overlap): profiling stores ``frequent_values`` and a separate ascending distinct ``value_overlap_sample`` (``VALUE_OVERLAP_SAMPLE_LIMIT``) for overlap; ``compute_semantic_profile_join_neighbors`` stores symmetric edges on ``ColumnMetadata.semantic_join_neighbors``. ``SEMANTIC_JOIN_MIN_OVERLAP_RATIO`` is the minimum ``|intersection| / min(|A|,|B|)`` on those two samples before an edge is recorded."""

    SCHEMA_CACHE_HASH_DEBUG_CLIP_CHARS: ClassVar[int] = 800

    JOIN_SHORTEST_PATH_TIE_CAP: ClassVar[int] = 4

    JOIN_COMPARISON_SCOPE_MAX_HOPS: ClassVar[int] = 2
    TABLE_REFERENCE_MAX_PER_SCOPE: ClassVar[int] = 2

    MAX_CTE_STEPS: ClassVar[int] = 16
    MAX_CTE_REFERENCE_DEPTH: ClassVar[int] = 8

    JOIN_CANDIDATE_CROSS_PRODUCT_CAP: ClassVar[int] = 16

    ELIMINATE_REDUNDANT_KEY_JOINS: ClassVar[bool] = False

    DEBUG: ClassVar[bool] = False

    REGENERATE_TEMPLATE_STORE: ClassVar[bool] = False
    REGENERATE_SCHEMA_GRAPH: ClassVar[bool] = False
    REGENERATE_SKELETON_CACHE: ClassVar[bool] = False
    SANDBOX_TRUST_SCHEMA_BASELINE: ClassVar[bool] = False

    LLM_BATCH_ENABLED: ClassVar[bool] = str(
        os.environ.get("AETHERDIALECT_LLM_BATCH_ENABLED", "") or ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    TABULAR_LLM_ASSIST: ClassVar[bool] = str(
        os.environ.get("AETHERDIALECT_TABULAR_LLM_ASSIST", "1") or "1"
    ).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    MAX_ASK_INTERPRET_GROUND_RETRIES: ClassVar[int] = 2
    MAX_ASK_COMPOSE_REPAIRS: ClassVar[int] = 2
    NATURAL_LANGUAGE_REFUSAL_SUBSTRINGS: ClassVar[tuple[str, ...]] = (
        "cannot",
        "can't",
        "unable to",
        "not available",
        "do not have",
        "don't have",
        "permission",
        "denied",
        "refuse",
        "unavailable",
        "no access",
        "not permitted",
        "not allowed",
    )
    MAX_FRESH_RESTARTS: ClassVar[int] = 1
    SANDBOX_RECORDING_MAX_ATTEMPTS: ClassVar[int] = 8
    SEMANTIC_RESTART_REASONS: ClassVar[frozenset[str]] = frozenset({"semantic_oscillation", "semantic_max_rounds"})

    MAX_USER_REFINEMENTS: ClassVar[int] = 1

    CATEGORICAL_MAX_RATIO = 0.05
    CATEGORICAL_MAX_CARDINALITY: ClassVar[int] = 50
    FREE_TEXT_CATEGORICAL_MAX_CARDINALITY = 200
    IDENTIFIER_MIN_UNIQUENESS = 0.98
    CATEGORICAL_SAMPLE_SIZE = 20

    LOW_CARDINALITY_FULL_VALUES_LIMIT: ClassVar[int] = 200
    LOW_CARDINALITY_DISTINCT_RATIO = 0.05
    ZERO_ROW_WHERE_AUTO_FIX_ENABLED = True
    ZERO_ROW_WHERE_FUZZY_MAX_DISTANCE: ClassVar[int] = 5
    GOLD_INTENT_PARSE_ATTEMPTS: ClassVar[int] = 3

    UNUSABLE_NULL_RATIO_THRESHOLD = 0.99
    SENTINEL_MODE_FREQUENCY_THRESHOLD = 0.99
    INFERRED_PK_MIN_ROW_COUNT = 50
    INFERRED_PK_COMPOSITE_MAX_COLUMNS = 4
    FK_INFER_OVERLAP_MIN_RATIO = 0.10
    FK_INFER_OVERLAP_MIN_SAMPLE = 5
    FK_INFER_CONTAINMENT_MIN_RATIO = 0.6

    VALUE_OVERLAP_SAMPLE_LIMIT = 100
    SEMANTIC_JOIN_MIN_OVERLAP_RATIO = 0.15
    SEMANTIC_JOIN_MIN_DISTINCT = 4
    SEMANTIC_JOIN_MIN_INTERSECTION = 3

    RESULT_ROW_COUNT_SOFT_WARNING: ClassVar[int] = 5_000
    FUZZY_MATCH_MAX_DISTANCE = 2
    FEDERATION_MAPPING_SUGGESTION_CROSS_SOURCE_CUTOFF: ClassVar[float] = 0.35
    FEDERATION_MAPPING_SUGGESTION_WITHIN_SOURCE_CUTOFF: ClassVar[float] = 0.50
    QUESTION_TOKEN_INDEX_NEIGHBOR_CAP = 2048

    PENALTY_CAP = 0.30

    TRUST_PROMOTE_MAX_REJECT_RATIO = 0.25
    TRUST_PROMOTE_PER_QUESTION_ACCEPTS = 2
    PER_QUESTION_REJECT_OUT_THRESHOLD = 2

    PEN_BY_THREE_SOURCE_UNIT = 0.05

    MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION: ClassVar[int] = 8

    MAX_SUMMARY_BULLETS: ClassVar[int] = 6

    MAX_QUERY_COST_ROWS: ClassVar[float | None] = _read_optional_positive_float_env(
        "AETHERDIALECT_MAX_QUERY_COST_ROWS", default=50_000_000.0
    )
    MAX_QUERY_COST_BYTES: ClassVar[float | None] = _read_optional_positive_float_env(
        "AETHERDIALECT_MAX_QUERY_COST_BYTES", default=50_000_000_000.0
    )
    STATEMENT_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_STATEMENT_TIMEOUT_MS", default=30_000
    )
    LLM_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_LLM_TIMEOUT_MS", default=60_000
    )
    PROFILE_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_PROFILE_TIMEOUT_MS", default=120_000
    )
    PROFILING_SAMPLE_THRESHOLD: ClassVar[int] = 100_000
    PROFILING_SAMPLE_SIZE: ClassVar[int] = 10_000
    PROFILING_SCHEMA_DEEP_QUERY_BUDGET: ClassVar[int | None] = 25_000
    MAX_ROLE_CLASSIFICATION_RETRIES: ClassVar[int] = 2
    EXPLAIN_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_EXPLAIN_TIMEOUT_MS", default=None
    )
    STOPWORDS = STOPWORDS_GRAMMATICAL_PARTICLES

    FORBIDDEN_SQL = [
        r"\bupdate\b",
        r"\bdelete\b",
        r"\binsert\b",
        r"\bmerge\b",
        r"\balter\b",
        r"\bdrop\b",
        r"\btruncate\b",
        r"\bgrant\b",
        r"\brevoke\b",
        r"\bcreate\b",
        r"\bcomment\b",
        r"\brename\b",
        r"\bcall\b",
        r"\bexecute\b",
        r"\bdo\b",
        r"\bcopy\b",
        r";\s*\S",
        r"\bUNION\b",
        r"\bINTERSECT\b",
        r"\bEXCEPT\b",
        r"\bLATERAL\b",
        r"\bOFFSET\b",
        r"\bFETCH\s+FIRST\b",
        r"\bDISTINCT\s+ON\b",
        r"\bARRAY\s*\[",
        r"\bARRAY_AGG\b",
        r"::json\b",
        r"\bEXISTS\s*\(",
    ]

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Refresh execution-limit ClassVars from *env* without mutating ``os.environ``."""
        float_keys = (
            ("MAX_QUERY_COST_ROWS", "AETHERDIALECT_MAX_QUERY_COST_ROWS", 50_000_000.0),
            ("MAX_QUERY_COST_BYTES", "AETHERDIALECT_MAX_QUERY_COST_BYTES", 50_000_000_000.0),
        )
        int_keys = (
            ("STATEMENT_TIMEOUT_MS", "AETHERDIALECT_STATEMENT_TIMEOUT_MS", 30_000),
            ("LLM_TIMEOUT_MS", "AETHERDIALECT_LLM_TIMEOUT_MS", 60_000),
            ("PROFILE_TIMEOUT_MS", "AETHERDIALECT_PROFILE_TIMEOUT_MS", 120_000),
            ("EXPLAIN_TIMEOUT_MS", "AETHERDIALECT_EXPLAIN_TIMEOUT_MS", None),
        )
        for attr, env_name, _ in float_keys:
            raw = str(env.get(env_name, "") or "").strip()
            if not raw:
                continue
            try:
                val = float(raw)
            except ValueError as exc:
                raise ConfigError(f"invalid {env_name}: {raw!r}") from exc
            setattr(cls, attr, val if val > 0 else None)
        for attr, env_name, _ in int_keys:
            raw = str(env.get(env_name, "") or "").strip()
            if not raw:
                continue
            try:
                val = int(raw, 10)
            except ValueError as exc:
                raise ConfigError(f"invalid {env_name}: {raw!r}") from exc
            setattr(cls, attr, val if val > 0 else None)


def _env_first_nonempty(env: Mapping[str, str], *keys: str) -> str:
    """Return the first non-blank value among *keys*, else an empty string."""
    for key in keys:
        value = str(env.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    """Return True when at least one key maps to a non-blank string."""
    return any(str(env.get(key, "") or "").strip() for key in keys)


def _env_role_hint(label: str, keys: tuple[str, ...]) -> str:
    """Return a human-readable hint listing acceptable environment variable names."""
    return f"{label}: {' or '.join(keys)}"


class EngineRuntimeConfig:
    """Shared runtime-configuration contract for registered database engines."""

    ENGINE_NAME: ClassVar[str] = ""

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Load connection settings from *env* into this runtime config class."""
        raise NotImplementedError

    @classmethod
    def should_apply_environment(cls, env: Mapping[str, str]) -> bool:
        """Return True when enough of *env* is present to call :meth:`apply_environment`."""
        return cls.env_complete(env)

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when *env* satisfies the minimum credentials for engine selection."""
        return False

    @classmethod
    def selection_blockers(cls, env: Mapping[str, str]) -> list[str]:
        """Return human-readable reasons this engine cannot be selected from *env*."""
        if cls.env_complete(env):
            return []
        return [f"{cls.ENGINE_NAME} environment is incomplete"]

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return connection field values used for artifact storage slug computation."""
        return {}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return ordered keys from :meth:`connection_slug_fields` that participate in storage slugs."""
        return ()

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return connection field names that must be redacted in runtime introspection."""
        return frozenset()

    @classmethod
    def rotatable_credential_fields(cls) -> tuple[str, ...]:
        """Return ClassVar names that may be replaced without rebuilding schema artifacts."""
        fields: list[str] = []
        for name in sorted(cls.redacted_fields()):
            upper = str(name).upper()
            if hasattr(cls, upper):
                fields.append(upper)
            elif hasattr(cls, name):
                fields.append(str(name))
        return tuple(fields)

    @classmethod
    def apply_connection_credentials(cls, credentials: str | Mapping[str, str]) -> None:
        """Apply in-place credential replacement before (re)opening a database connection."""
        if isinstance(credentials, str):
            fields = cls.rotatable_credential_fields()
            if not fields:
                raise ValueError(f"{cls.ENGINE_NAME or cls.__name__} has no rotatable credential fields")
            setattr(cls, fields[0], credentials)
            return
        for raw_key, value in credentials.items():
            key = str(raw_key)
            if hasattr(cls, key):
                field = key
            else:
                alias = key.lower()
                field = alias.upper() if hasattr(cls, alias.upper()) else alias
                if not hasattr(cls, field):
                    raise ValueError(f"unknown credential field {raw_key!r} for {cls.ENGINE_NAME or cls.__name__}")
            setattr(cls, field, value)


class PostgresRuntimeConfig(EngineRuntimeConfig):
    """PostgreSQL connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "postgresql"

    HOST: ClassVar[str] = "localhost"
    PORT: ClassVar[int] = 5432
    USER: ClassVar[str] = "postgres"
    PASSWORD: ClassVar[str | None] = None
    DATABASE: ClassVar[str | None] = None
    SCHEMA: ClassVar[str] = "public"

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Copy PostgreSQL connection variables from *env* into ClassVars."""
        host = _env_first_nonempty(env, *POSTGRES_ENV_HOST)
        cls.HOST = host or "localhost"
        port_raw = _env_first_nonempty(env, *POSTGRES_ENV_PORT)
        cls.PORT = int(port_raw) if port_raw else 5432
        cls.USER = _env_first_nonempty(env, *POSTGRES_ENV_USER) or "postgres"
        cls.PASSWORD = _env_first_nonempty(env, *POSTGRES_ENV_PASSWORD)
        cls.DATABASE = _env_first_nonempty(env, *POSTGRES_ENV_DATABASE)
        cls.SCHEMA = _env_first_nonempty(env, *POSTGRES_ENV_SCHEMA) or "public"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when database, user, and password are configured."""
        return (
            _env_any_nonempty(env, POSTGRES_ENV_DATABASE)
            and _env_any_nonempty(env, POSTGRES_ENV_USER)
            and _env_any_nonempty(env, POSTGRES_ENV_PASSWORD)
        )

    @classmethod
    def selection_blockers(cls, env: Mapping[str, str]) -> list[str]:
        """Return driver or credential gaps preventing PostgreSQL selection."""
        driver_ok = package_importable("psycopg2") or package_importable("psycopg")
        if cls.env_complete(env) and driver_ok:
            return []
        blockers: list[str] = []
        if not driver_ok:
            blockers.append("PostgreSQL driver (psycopg or psycopg2)")
        if driver_ok and not cls.env_complete(env):
            blockers.append(
                "PostgreSQL env (set one name from each required group): "
                + _env_role_hint("database", POSTGRES_ENV_DATABASE)
                + "; "
                + _env_role_hint("user", POSTGRES_ENV_USER)
                + "; "
                + _env_role_hint("password", POSTGRES_ENV_PASSWORD)
                + "; optional "
                + _env_role_hint("host", POSTGRES_ENV_HOST)
                + "; "
                + _env_role_hint("port", POSTGRES_ENV_PORT)
                + "; "
                + _env_role_hint("schema", POSTGRES_ENV_SCHEMA)
            )
        return blockers

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return PostgreSQL connection values for slug and introspection."""
        return {
            "host": cls.HOST or "localhost",
            "port": str(int(cls.PORT)),
            "database": cls.DATABASE or "db",
            "schema": cls.SCHEMA or "public",
            "user": cls.USER or "",
            "password": cls.PASSWORD or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for PostgreSQL storage paths."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return PostgreSQL secret field names."""
        return frozenset({"password"})

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy PostgreSQL URL from ClassVars, preferring. ``psycopg`` when installed."""
        if not cls.PASSWORD:
            raise ValueError("PostgreSQL password required")
        if not cls.DATABASE:
            raise ValueError("PostgreSQL database required")
        user_q = quote(str(cls.USER), safe="")
        pwd_q = quote(str(cls.PASSWORD), safe="")
        db_q = quote(str(cls.DATABASE), safe="")
        driver = "postgresql+psycopg2"
        if find_spec("psycopg") is not None:
            driver = "postgresql+psycopg"
        return f"{driver}://{user_q}:{pwd_q}@{cls.HOST}:{cls.PORT}/{db_q}"


class DatabricksRuntimeConfig(EngineRuntimeConfig):
    """Unity Catalog `CATALOG`/`SCHEMA` and optional ODBC connector settings (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "databricks"

    CATALOG: ClassVar[str | None] = None
    SCHEMA: ClassVar[str | None] = None

    SERVER_HOSTNAME: ClassVar[str | None] = None
    HTTP_PATH: ClassVar[str | None] = None
    ACCESS_TOKEN: ClassVar[str | None] = None

    @classmethod
    def uc_scope_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when catalog and schema are configured."""
        return env_any_nonempty(env, DATABRICKS_ENV_CATALOG) and env_any_nonempty(env, DATABRICKS_ENV_SCHEMA)

    @classmethod
    def sql_warehouse_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when SQL warehouse hostname, HTTP path, and token are configured."""
        return (
            env_any_nonempty(env, DATABRICKS_ENV_SERVER_HOSTNAME)
            and env_any_nonempty(env, DATABRICKS_ENV_HTTP_PATH)
            and env_any_nonempty(env, DATABRICKS_ENV_TOKEN)
        )

    @classmethod
    def pyspark_session_reachable(cls) -> bool:
        """Return True when an active PySpark session can be created."""
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            return False
        try:
            SparkSession.builder.getOrCreate()
        except Exception:
            return False
        return True

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Copy Databricks connection variables from *env* into ClassVars."""
        cls.SERVER_HOSTNAME = env_first_nonempty(env, *DATABRICKS_ENV_SERVER_HOSTNAME)
        cls.HTTP_PATH = env_first_nonempty(env, *DATABRICKS_ENV_HTTP_PATH)
        cls.ACCESS_TOKEN = env_first_nonempty(env, *DATABRICKS_ENV_TOKEN)
        cls.CATALOG = env_first_nonempty(env, *DATABRICKS_ENV_CATALOG)
        cls.SCHEMA = env_first_nonempty(env, *DATABRICKS_ENV_SCHEMA)
        cls.validate()

    @classmethod
    def should_apply_environment(cls, env: Mapping[str, str]) -> bool:
        """Return True when Unity Catalog scope is present."""
        return cls.uc_scope_complete(env)

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when UC scope and a warehouse or PySpark path is available."""
        if not cls.uc_scope_complete(env):
            return False
        if cls.sql_warehouse_complete(env):
            return package_importable("databricks.sql")
        return cls.pyspark_session_reachable()

    @classmethod
    def selection_blockers(cls, env: Mapping[str, str]) -> list[str]:
        """Return credential or driver gaps preventing Databricks selection."""
        if cls.env_complete(env):
            return []
        blockers: list[str] = []
        if (
            not cls.env_complete(env)
            and cls.uc_scope_complete(env)
            and cls.sql_warehouse_complete(env)
            and not package_importable("databricks.sql")
        ):
            blockers.append(
                "Databricks SQL warehouse variables are set but the databricks-sql-connector package is not installed."
            )
        elif not cls.env_complete(env):
            blockers.append(
                "Databricks env: "
                + env_role_hint("catalog", DATABRICKS_ENV_CATALOG)
                + "; "
                + env_role_hint("schema", DATABRICKS_ENV_SCHEMA)
                + "; then either all of "
                + env_role_hint("server hostname", DATABRICKS_ENV_SERVER_HOSTNAME)
                + ", "
                + env_role_hint("SQL warehouse HTTP path", DATABRICKS_ENV_HTTP_PATH)
                + ", "
                + env_role_hint("access token", DATABRICKS_ENV_TOKEN)
                + " (with databricks-sql-connector installed), or an active PySpark session."
            )
        return blockers

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return Databricks connection values for slug and introspection."""
        host_raw = (cls.SERVER_HOSTNAME or "").strip() or "pyspark"
        return {
            "server_hostname": cls.SERVER_HOSTNAME or "",
            "http_path": cls.HTTP_PATH or "",
            "catalog": cls.CATALOG or "catalog",
            "schema": cls.SCHEMA or "schema",
            "host": host_raw.split(".")[0],
            "access_token": cls.ACCESS_TOKEN or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Databricks storage paths."""
        return ("host", "catalog", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Databricks secret field names."""
        return frozenset({"access_token"})

    @classmethod
    def has_native_connection(cls) -> bool:
        """True when hostname, HTTP path, and access token are all non- empty."""
        return bool(cls.SERVER_HOSTNAME and cls.HTTP_PATH and cls.ACCESS_TOKEN)

    @classmethod
    def validate(cls) -> None:
        """Require `CATALOG` and `SCHEMA`."""
        if not cls.CATALOG:
            raise ValueError("Databricks catalog required")
        if not cls.SCHEMA:
            raise ValueError("Databricks schema required")

    @classmethod
    def sqlalchemy_url(cls) -> str | None:
        """Build a SQLAlchemy URL for the Databricks SQL connector when. PAT credentials exist."""
        if not cls.has_native_connection():
            return None

        token = quote(cls.ACCESS_TOKEN or "", safe="")
        host = cls.SERVER_HOSTNAME or ""
        http_path = quote(cls.HTTP_PATH or "", safe="")
        catalog = quote(cls.CATALOG or "", safe="")
        schema = quote(cls.SCHEMA or "", safe="")
        return f"databricks://token:{token}@{host}?http_path={http_path}&catalog={catalog}&schema={schema}"


class MySQLRuntimeConfig(EngineRuntimeConfig):
    """MySQL connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "mysql"

    HOST: ClassVar[str] = "localhost"
    PORT: ClassVar[int] = 3306
    USER: ClassVar[str] = "root"
    PASSWORD: ClassVar[str | None] = None
    DATABASE: ClassVar[str | None] = None
    SCHEMA: ClassVar[str | None] = None

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy MySQL URL from ClassVars."""
        if not cls.PASSWORD:
            raise ValueError("MySQL password required")
        if not cls.DATABASE:
            raise ValueError("MySQL database required")
        user_q = quote(str(cls.USER), safe="")
        pwd_q = quote(str(cls.PASSWORD), safe="")
        db_q = quote(str(cls.DATABASE), safe="")
        return f"mysql+pymysql://{user_q}:{pwd_q}@{cls.HOST}:{cls.PORT}/{db_q}?charset=utf8mb4"

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return driver connect arguments for MySQL."""
        return {}

    @classmethod
    def has_password_auth(cls) -> bool:
        """Return True when password authentication is configured."""
        return bool(cls.PASSWORD)

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Apply MySQL environment variables to ClassVars."""
        cls.HOST = env_first_nonempty(env, *MYSQL_ENV_HOST) or "localhost"
        port_raw = env_first_nonempty(env, *MYSQL_ENV_PORT)
        cls.PORT = int(port_raw) if port_raw else 3306
        cls.USER = env_first_nonempty(env, *MYSQL_ENV_USER) or "root"
        cls.PASSWORD = env_first_nonempty(env, *MYSQL_ENV_PASSWORD) or None
        database = env_first_nonempty(env, *MYSQL_ENV_DATABASE)
        cls.DATABASE = database or None
        cls.SCHEMA = database or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required MySQL env vars are present."""
        return (
            env_any_nonempty(env, MYSQL_ENV_PASSWORD)
            and env_any_nonempty(env, MYSQL_ENV_DATABASE)
            and env_any_nonempty(env, MYSQL_ENV_USER)
        )

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return MySQL connection values for slug and introspection."""
        return {
            "host": cls.HOST or "localhost",
            "port": str(int(cls.PORT)),
            "database": cls.DATABASE or "db",
            "schema": cls.SCHEMA or cls.DATABASE or "db",
            "user": cls.USER or "",
            "password": cls.PASSWORD or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for MySQL storage paths."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return MySQL secret field names."""
        return frozenset({"password"})


class MariaDBRuntimeConfig(MySQLRuntimeConfig):
    """MariaDB connection defaults that reuse the MySQL backend via the pymysql driver."""

    ENGINE_NAME: ClassVar[str] = "mariadb"

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Populate MariaDB ClassVars from MARIADB_* env keys only."""
        cls.HOST = env_first_nonempty(env, *MARIADB_ENV_HOST) or "localhost"
        port_raw = env_first_nonempty(env, *MARIADB_ENV_PORT)
        cls.PORT = int(port_raw) if port_raw else 3306
        cls.USER = env_first_nonempty(env, *MARIADB_ENV_USER) or "root"
        cls.PASSWORD = env_first_nonempty(env, *MARIADB_ENV_PASSWORD) or None
        database = env_first_nonempty(env, *MARIADB_ENV_DATABASE)
        cls.DATABASE = database or None
        cls.SCHEMA = database or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when MariaDB user, password, and database env keys are all present."""
        return (
            env_any_nonempty(env, MARIADB_ENV_PASSWORD)
            and env_any_nonempty(env, MARIADB_ENV_DATABASE)
            and env_any_nonempty(env, MARIADB_ENV_USER)
        )


class DuckDBRuntimeConfig(EngineRuntimeConfig):
    """DuckDB embedded-database connection defaults sourced from a local file path or :memory:."""

    ENGINE_NAME: ClassVar[str] = "duckdb"

    DATABASE_PATH: ClassVar[str] = ":memory:"
    SCHEMA: ClassVar[str] = "main"
    NATIVE_CONNECTION: ClassVar[Any | None] = None

    @classmethod
    def attach_connection(cls, connection: Any) -> None:
        """Store a caller-owned DuckDB connection for dialect construction."""
        cls.NATIVE_CONNECTION = connection

    @classmethod
    def clear_attached_connection(cls) -> None:
        """Clear the in-process native connection slot."""
        cls.NATIVE_CONNECTION = None

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy DuckDB URL from the configured file path or :memory:."""
        path = str(cls.DATABASE_PATH or ":memory:")
        return f"duckdb:///{path}"

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return driver connect arguments for DuckDB."""
        return {}

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Populate DuckDB ClassVars from DUCKDB_* env keys."""
        cls.DATABASE_PATH = env_first_nonempty(env, *DUCKDB_ENV_PATH) or ":memory:"
        cls.SCHEMA = env_first_nonempty(env, *DUCKDB_ENV_SCHEMA) or "main"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when a DuckDB database path env key is present."""
        return env_any_nonempty(env, DUCKDB_ENV_PATH)

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return DuckDB connection values for slug and introspection."""
        raw = str(cls.DATABASE_PATH or ":memory:")
        base = "memory" if raw == ":memory:" else os.path.splitext(os.path.basename(raw))[0] or "duckdb"
        return {"database": base, "schema": cls.SCHEMA or "main"}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for DuckDB storage paths."""
        return ("database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return DuckDB secret field names (none for an embedded database)."""
        return frozenset()


class CsvRuntimeConfig(EngineRuntimeConfig):
    """CSV/Excel file-source connection defaults for the in-memory DuckDB backend."""

    ENGINE_NAME: ClassVar[str] = "csv"

    DIRECTORY: ClassVar[str | None] = None
    FILES: ClassVar[tuple[str, ...]] = ()
    SOURCE_SELECTIONS: ClassVar[dict[str, dict[str, Any]]] = {}
    SCHEMA: ClassVar[str] = "main"
    NATIVE_CONNECTION: ClassVar[Any | None] = None

    @classmethod
    def attach_connection(cls, connection: Any) -> None:
        """Store a caller-owned DuckDB connection for dialect construction."""
        cls.NATIVE_CONNECTION = connection

    @classmethod
    def clear_attached_connection(cls) -> None:
        """Clear the in-process native connection slot."""
        cls.NATIVE_CONNECTION = None

    @classmethod
    def db_url(cls) -> str:
        """Return the in-memory DuckDB SQLAlchemy URL used by the CSV backend."""
        return "duckdb:///:memory:"

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return driver connect arguments for the CSV DuckDB backend."""
        return {}

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Populate CSV ClassVars from CSV_DIRECTORY / CSV_FILES env keys."""
        directory = env_first_nonempty(env, *CSV_ENV_DIRECTORY)
        files_raw = env_first_nonempty(env, *CSV_ENV_FILES)
        cls.DIRECTORY = directory or None
        if files_raw:
            cls.FILES = tuple(part.strip() for part in files_raw.split(",") if part.strip())
        else:
            cls.FILES = ()
        cls.SCHEMA = "main"
        if cls.DIRECTORY and cls.FILES:
            raise ConfigError("csv: set either CSV_DIRECTORY or CSV_FILES, not both")

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when a CSV directory or explicit file list is configured."""
        return env_any_nonempty(env, CSV_ENV_DIRECTORY) or env_any_nonempty(env, CSV_ENV_FILES)

    @classmethod
    def selection_blockers(cls, env: Mapping[str, str]) -> list[str]:
        """Return driver or configuration gaps preventing CSV engine selection."""
        driver_ok = package_importable("duckdb")
        if cls.env_complete(env) and driver_ok:
            try:
                cls.resolve_source_files()
                return []
            except ConfigError as exc:
                return [str(exc)]
        blockers: list[str] = []
        if not driver_ok:
            blockers.append("CSV backend driver (duckdb)")
        if driver_ok and not cls.env_complete(env):
            blockers.append("CSV env (set CSV_DIRECTORY or CSV_FILES; mutually exclusive)")
        return blockers

    @classmethod
    def _allowed_source_suffixes(cls) -> frozenset[str]:
        """Return upload suffixes permitted for the CSV file engine."""
        return frozenset({".csv", ".xlsx"})

    @classmethod
    def resolve_source_files(cls) -> tuple[Path, ...]:
        """Resolve configured CSV/Excel inputs to absolute file paths."""
        engine = cls.ENGINE_NAME
        allowed = cls._allowed_source_suffixes()
        directory = str(cls.DIRECTORY or "").strip()
        files = tuple(str(item).strip() for item in cls.FILES if str(item).strip())
        if directory and files:
            raise ConfigError(f"{engine}: set either CSV_DIRECTORY or CSV_FILES, not both")
        paths: list[Path]
        if directory:
            dir_path = Path(os.path.expanduser(directory))
            if not dir_path.is_dir():
                raise ConfigError(f"{engine} directory not found: {directory}")
            paths = sorted(path for path in dir_path.iterdir() if path.is_file() and path.suffix.lower() in allowed)
        elif files:
            paths = []
            for raw in files:
                path = Path(os.path.expanduser(raw))
                if not path.is_file():
                    raise ConfigError(f"{engine} file not found: {raw}")
                suffix = path.suffix.lower()
                if suffix not in allowed:
                    raise ConfigError(f"{engine} unsupported file type: {raw}")
                paths.append(path)
            paths.sort(key=lambda item: item.name.lower())
        else:
            raise ConfigError(f"{engine}: set CSV_DIRECTORY or CSV_FILES")
        if not paths:
            allowed_label = ", ".join(sorted(allowed))
            raise ConfigError(f"{engine}: no matching files with suffix ({allowed_label}) found")
        stems = [path.stem.lower() for path in paths]
        if len(stems) != len(set(stems)):
            raise ConfigError(f"{engine}: duplicate relation names (same file stem)")
        return tuple(path.resolve() for path in paths)

    @classmethod
    def set_source_selections(cls, raw: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Store per-file upload interpretation choices for the CSV file engine."""
        if not raw:
            cls.SOURCE_SELECTIONS = {}
            return
        cls.SOURCE_SELECTIONS = {str(key): dict(value) for key, value in raw.items()}

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return CSV source values for slug and introspection."""
        try:
            paths = cls.resolve_source_files()
            source_key = hashlib.sha256("|".join(str(path) for path in paths).encode()).hexdigest()
        except Exception:
            source_key = hashlib.sha256(f"{cls.DIRECTORY or ''}|{','.join(cls.FILES)}".encode()).hexdigest()
        return {"source": source_key[:32], "schema": cls.SCHEMA or "main"}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for CSV storage paths."""
        return ("source", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return CSV secret field names (none for file sources)."""
        return frozenset()


class SQLiteRuntimeConfig(EngineRuntimeConfig):
    """SQLite embedded-database connection defaults sourced from a local file path or :memory:."""

    ENGINE_NAME: ClassVar[str] = "sqlite"

    DATABASE_PATH: ClassVar[str] = ":memory:"
    SCHEMA: ClassVar[str] = "main"
    NATIVE_CONNECTION: ClassVar[Any | None] = None

    @classmethod
    def attach_connection(cls, connection: Any) -> None:
        """Store a caller-owned sqlite3 connection for dialect construction."""
        cls.NATIVE_CONNECTION = connection

    @classmethod
    def clear_attached_connection(cls) -> None:
        """Clear the in-process native connection slot."""
        cls.NATIVE_CONNECTION = None

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy SQLite URL from the configured file path or :memory:."""
        path = str(cls.DATABASE_PATH or ":memory:")
        return f"sqlite:///{path}"

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return driver connect arguments for SQLite."""
        return {}

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Populate SQLite ClassVars from SQLITE_* env keys."""
        cls.DATABASE_PATH = env_first_nonempty(env, *SQLITE_ENV_PATH) or ":memory:"
        cls.SCHEMA = "main"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when a SQLite database path env key is present."""
        return env_any_nonempty(env, SQLITE_ENV_PATH)

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return SQLite connection values for slug and introspection."""
        raw = str(cls.DATABASE_PATH or ":memory:")
        base = "memory" if raw == ":memory:" else os.path.splitext(os.path.basename(raw))[0] or "sqlite"
        return {"database": base, "schema": "main"}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for SQLite storage paths."""
        return ("database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return SQLite secret field names (none for an embedded database)."""
        return frozenset()


class RedshiftRuntimeConfig(EngineRuntimeConfig):
    """Amazon Redshift connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "redshift"

    HOST: ClassVar[str] = "localhost"
    PORT: ClassVar[int] = 5439
    USER: ClassVar[str] = "awsuser"
    PASSWORD: ClassVar[str | None] = None
    DATABASE: ClassVar[str] = "dev"
    SCHEMA: ClassVar[str] = "public"
    USE_IAM: ClassVar[bool] = False
    CLUSTER_IDENTIFIER: ClassVar[str | None] = None
    WORKGROUP: ClassVar[str | None] = None
    REGION: ClassVar[str | None] = None

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy Redshift URL from ClassVars."""
        if cls.has_iam_credentials():
            user_q = quote(str(cls.USER), safe="")
            db_q = quote(str(cls.DATABASE), safe="")
            host_q = quote(str(cls.HOST), safe="")
            params = ["sslmode=verify-full"]
            if cls.CLUSTER_IDENTIFIER:
                params.append(f"cluster_identifier={quote(str(cls.CLUSTER_IDENTIFIER), safe='')}")
            if cls.WORKGROUP:
                params.append(f"workgroup={quote(str(cls.WORKGROUP), safe='')}")
            if cls.REGION:
                params.append(f"region={quote(str(cls.REGION), safe='')}")
            query = "&".join(params)
            return f"redshift+redshift_connector://{user_q}@{host_q}:{cls.PORT}/{db_q}?{query}"
        if not cls.PASSWORD:
            raise ValueError("Redshift password required when IAM is disabled")
        user_q = quote(str(cls.USER), safe="")
        pwd_q = quote(str(cls.PASSWORD), safe="")
        db_q = quote(str(cls.DATABASE), safe="")
        return f"redshift+redshift_connector://{user_q}:{pwd_q}@{cls.HOST}:{cls.PORT}/{db_q}?sslmode=verify-full"

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return driver connect arguments for Redshift."""
        if cls.has_iam_credentials():
            return {"iam": True, "ssl": True, "sslmode": "verify-full"}
        return {"ssl": True, "sslmode": "verify-full"}

    @classmethod
    def has_iam_credentials(cls) -> bool:
        """Return True when IAM authentication is configured."""
        return bool(cls.USE_IAM and (cls.CLUSTER_IDENTIFIER or cls.WORKGROUP))

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Apply Redshift environment variables to ClassVars."""
        cls.HOST = env_first_nonempty(env, *REDSHIFT_ENV_HOST) or "localhost"
        port_raw = env_first_nonempty(env, *REDSHIFT_ENV_PORT)
        cls.PORT = int(port_raw) if port_raw else 5439
        cls.USER = env_first_nonempty(env, *REDSHIFT_ENV_USER) or "awsuser"
        cls.PASSWORD = env_first_nonempty(env, *REDSHIFT_ENV_PASSWORD) or None
        cls.DATABASE = env_first_nonempty(env, *REDSHIFT_ENV_DATABASE) or "dev"
        cls.SCHEMA = env_first_nonempty(env, *REDSHIFT_ENV_SCHEMA) or "public"
        use_iam_raw = env_first_nonempty(env, *REDSHIFT_ENV_USE_IAM).lower()
        cls.USE_IAM = use_iam_raw in ("1", "true", "yes", "on") if use_iam_raw else False
        cluster = env_first_nonempty(env, *REDSHIFT_ENV_CLUSTER_IDENTIFIER)
        cls.CLUSTER_IDENTIFIER = cluster or None
        workgroup = env_first_nonempty(env, *REDSHIFT_ENV_WORKGROUP)
        cls.WORKGROUP = workgroup or None
        region = env_first_nonempty(env, *REDSHIFT_ENV_REGION)
        cls.REGION = region or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required Redshift env vars are present."""
        use_iam_raw = env_first_nonempty(env, *REDSHIFT_ENV_USE_IAM).lower()
        use_iam = use_iam_raw in ("1", "true", "yes", "on") if use_iam_raw else cls.USE_IAM
        if use_iam:
            user = env_first_nonempty(env, *REDSHIFT_ENV_USER) or (cls.USER or "")
            cluster = env_first_nonempty(env, *REDSHIFT_ENV_CLUSTER_IDENTIFIER) or (cls.CLUSTER_IDENTIFIER or "")
            workgroup = env_first_nonempty(env, *REDSHIFT_ENV_WORKGROUP) or (cls.WORKGROUP or "")
            return bool(str(user).strip() and (str(cluster).strip() or str(workgroup).strip()))
        return (
            env_any_nonempty(env, REDSHIFT_ENV_PASSWORD)
            and env_any_nonempty(env, REDSHIFT_ENV_USER)
            and env_any_nonempty(env, REDSHIFT_ENV_DATABASE)
        )

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return Redshift connection values for slug and introspection."""
        return {
            "host": cls.HOST or "localhost",
            "port": str(int(cls.PORT)),
            "database": cls.DATABASE or "dev",
            "schema": cls.SCHEMA or "public",
            "user": cls.USER or "",
            "password": cls.PASSWORD or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Redshift storage paths (excludes cluster/workgroup/region)."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Redshift secret field names."""
        return frozenset({"password"})


class SQLServerRuntimeConfig(EngineRuntimeConfig):
    """SQL Server connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "sqlserver"

    HOST: ClassVar[str] = "localhost"
    PORT: ClassVar[int] = 1433
    USER: ClassVar[str | None] = None
    PASSWORD: ClassVar[str | None] = None
    DATABASE: ClassVar[str | None] = None
    SCHEMA: ClassVar[str] = "dbo"
    DRIVER: ClassVar[str] = "ODBC Driver 18 for SQL Server"
    AUTH_MODE: ClassVar[str] = "sql"
    TENANT_ID: ClassVar[str | None] = None
    CLIENT_ID: ClassVar[str | None] = None
    CLIENT_SECRET: ClassVar[str | None] = None

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy SQL Server URL from ClassVars."""
        if not cls.DATABASE:
            raise ValueError("SQL Server database required")
        driver_q = quote(str(cls.DRIVER), safe="")
        if cls.AUTH_MODE == "windows":
            return (
                f"mssql+pyodbc://@{cls.HOST}:{cls.PORT}/{quote(str(cls.DATABASE), safe='')}"
                f"?driver={driver_q}&Trusted_Connection=yes"
            )
        if cls.AUTH_MODE == "aad_password":
            if not cls.USER or not cls.PASSWORD:
                raise ValueError("SQL Server user and password required for Azure AD password authentication")
            user_q = quote(str(cls.USER), safe="")
            pwd_q = quote(str(cls.PASSWORD), safe="")
            db_q = quote(str(cls.DATABASE), safe="")
            return (
                f"mssql+pyodbc://{user_q}:{pwd_q}@{cls.HOST}:{cls.PORT}/{db_q}"
                f"?driver={driver_q}&Authentication=ActiveDirectoryPassword"
                f"&Encrypt=yes&TrustServerCertificate=yes"
            )
        if cls.AUTH_MODE == "aad_sp":
            if not cls.CLIENT_ID or not cls.CLIENT_SECRET:
                raise ValueError("SQL Server client id and secret required for Azure AD service principal")
            client_q = quote(str(cls.CLIENT_ID), safe="")
            secret_q = quote(str(cls.CLIENT_SECRET), safe="")
            db_q = quote(str(cls.DATABASE), safe="")
            return (
                f"mssql+pyodbc://@{cls.HOST}:{cls.PORT}/{db_q}"
                f"?driver={driver_q}&Authentication=ActiveDirectoryServicePrincipal"
                f"&UID={client_q}&PWD={secret_q}&Encrypt=yes&TrustServerCertificate=yes"
            )
        if not cls.USER or not cls.PASSWORD:
            raise ValueError("SQL Server user and password required for SQL authentication")
        user_q = quote(str(cls.USER), safe="")
        pwd_q = quote(str(cls.PASSWORD), safe="")
        db_q = quote(str(cls.DATABASE), safe="")
        return (
            f"mssql+pyodbc://{user_q}:{pwd_q}@{cls.HOST}:{cls.PORT}/{db_q}?driver={driver_q}&TrustServerCertificate=yes"
        )

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return driver connect arguments for SQL Server."""
        return {}

    @classmethod
    def has_sql_auth(cls) -> bool:
        """Return True when SQL authentication is configured."""
        return cls.AUTH_MODE == "sql" and bool(cls.USER and cls.PASSWORD)

    @classmethod
    def has_windows_auth(cls) -> bool:
        """Return True when Windows integrated authentication is selected."""
        return cls.AUTH_MODE == "windows"

    @classmethod
    def has_aad_password_auth(cls) -> bool:
        """Return True when Azure AD password authentication is selected."""
        return cls.AUTH_MODE == "aad_password" and bool(cls.USER and cls.PASSWORD and cls.TENANT_ID)

    @classmethod
    def has_aad_service_principal_auth(cls) -> bool:
        """Return True when Azure AD service principal authentication is selected."""
        return cls.AUTH_MODE == "aad_sp" and bool(cls.CLIENT_ID and cls.CLIENT_SECRET and cls.TENANT_ID)

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Apply SQL Server environment variables to ClassVars."""
        cls.HOST = env_first_nonempty(env, *SQLSERVER_ENV_HOST) or "localhost"
        port_raw = env_first_nonempty(env, *SQLSERVER_ENV_PORT)
        cls.PORT = int(port_raw) if port_raw else 1433
        user = env_first_nonempty(env, *SQLSERVER_ENV_USER)
        cls.USER = user or None
        cls.PASSWORD = env_first_nonempty(env, *SQLSERVER_ENV_PASSWORD) or None
        database = env_first_nonempty(env, *SQLSERVER_ENV_DATABASE)
        cls.DATABASE = database or None
        cls.SCHEMA = env_first_nonempty(env, *SQLSERVER_ENV_SCHEMA) or "dbo"
        cls.DRIVER = env_first_nonempty(env, *SQLSERVER_ENV_DRIVER) or "ODBC Driver 18 for SQL Server"
        auth_mode = env_first_nonempty(env, *SQLSERVER_ENV_AUTH_MODE)
        cls.AUTH_MODE = auth_mode.lower() if auth_mode else "sql"
        tenant = env_first_nonempty(env, *SQLSERVER_ENV_TENANT_ID)
        cls.TENANT_ID = tenant or None
        client_id = env_first_nonempty(env, *SQLSERVER_ENV_CLIENT_ID)
        cls.CLIENT_ID = client_id or None
        client_secret = env_first_nonempty(env, *SQLSERVER_ENV_CLIENT_SECRET)
        cls.CLIENT_SECRET = client_secret or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required SQL Server env vars are present."""
        auth = env_first_nonempty(env, *SQLSERVER_ENV_AUTH_MODE) or cls.AUTH_MODE or "sql"
        auth = auth.strip().lower()
        if not env_any_nonempty(env, SQLSERVER_ENV_DATABASE) and not cls.DATABASE:
            return False
        if auth == "windows":
            return True
        if auth == "aad_password":
            return (
                env_any_nonempty(env, SQLSERVER_ENV_USER)
                and env_any_nonempty(env, SQLSERVER_ENV_PASSWORD)
                and env_any_nonempty(env, SQLSERVER_ENV_TENANT_ID)
            )
        if auth == "aad_sp":
            return (
                env_any_nonempty(env, SQLSERVER_ENV_TENANT_ID)
                and env_any_nonempty(env, SQLSERVER_ENV_CLIENT_ID)
                and env_any_nonempty(env, SQLSERVER_ENV_CLIENT_SECRET)
            )
        return env_any_nonempty(env, SQLSERVER_ENV_USER) and env_any_nonempty(env, SQLSERVER_ENV_PASSWORD)

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return SQL Server connection values for slug and introspection."""
        return {
            "host": cls.HOST or "localhost",
            "port": str(int(cls.PORT)),
            "database": cls.DATABASE or "db",
            "schema": cls.SCHEMA or "dbo",
            "user": cls.USER or "",
            "password": cls.PASSWORD or "",
            "auth_mode": cls.AUTH_MODE or "sql",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for SQL Server storage paths (excludes auth_mode/driver)."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return SQL Server secret field names."""
        return frozenset({"password"})


class SnowflakeRuntimeConfig(EngineRuntimeConfig):
    """Snowflake connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "snowflake"

    ACCOUNT: ClassVar[str | None] = None
    USER: ClassVar[str | None] = None
    PASSWORD: ClassVar[str | None] = None
    DATABASE: ClassVar[str | None] = None
    SCHEMA: ClassVar[str] = "PUBLIC"
    WAREHOUSE: ClassVar[str | None] = None
    ROLE: ClassVar[str | None] = None
    PRIVATE_KEY_PATH: ClassVar[str | None] = None
    PRIVATE_KEY_PASSPHRASE: ClassVar[str | None] = None
    AUTHENTICATOR: ClassVar[str | None] = None
    OAUTH_TOKEN: ClassVar[str | None] = None

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy Snowflake URL from ClassVars."""
        if not cls.ACCOUNT or not cls.USER:
            raise ValueError("Snowflake account and user required")
        account_q = quote(str(cls.ACCOUNT), safe="")
        user_q = quote(str(cls.USER), safe="")
        if cls.has_password_auth():
            pwd_q = quote(str(cls.PASSWORD or ""), safe="")
            auth = f"{user_q}:{pwd_q}"
        else:
            auth = user_q
        params: list[str] = []
        if cls.DATABASE:
            params.append(f"database={quote(str(cls.DATABASE), safe='')}")
        if cls.SCHEMA:
            params.append(f"schema={quote(str(cls.SCHEMA), safe='')}")
        if cls.WAREHOUSE:
            params.append(f"warehouse={quote(str(cls.WAREHOUSE), safe='')}")
        if cls.ROLE:
            params.append(f"role={quote(str(cls.ROLE), safe='')}")
        if cls.AUTHENTICATOR:
            params.append(f"authenticator={quote(str(cls.AUTHENTICATOR), safe='')}")
        query = "&".join(params)
        base = f"snowflake://{auth}@{account_q}"
        return f"{base}/?{query}" if query else base

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return Snowflake driver connect arguments."""
        out: dict[str, Any] = {}
        if cls.has_oauth_auth() and cls.OAUTH_TOKEN:
            out["token"] = cls.OAUTH_TOKEN
        if cls.has_keypair_auth() and cls.PRIVATE_KEY_PATH:
            out["private_key_file"] = cls.PRIVATE_KEY_PATH
            if cls.PRIVATE_KEY_PASSPHRASE:
                out["private_key_file_pwd"] = cls.PRIVATE_KEY_PASSPHRASE
        return out

    @classmethod
    def has_password_auth(cls) -> bool:
        """Return True when password authentication is configured."""
        return bool(cls.PASSWORD) and not cls.has_keypair_auth() and not cls.has_oauth_auth()

    @classmethod
    def has_keypair_auth(cls) -> bool:
        """Return True when key-pair authentication is configured."""
        return bool(cls.PRIVATE_KEY_PATH)

    @classmethod
    def has_oauth_auth(cls) -> bool:
        """Return True when OAuth authentication is configured."""
        return bool(cls.OAUTH_TOKEN or (cls.AUTHENTICATOR and "oauth" in str(cls.AUTHENTICATOR).lower()))

    @classmethod
    def snowpark_session_reachable(cls) -> bool:
        """Return True when an ambient Snowpark session is available."""
        try:
            from snowflake.snowpark.context import get_active_session
        except ImportError:
            return False
        try:
            get_active_session()
        except Exception:
            return False
        return True

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Apply Snowflake environment variables to ClassVars."""
        account = env_first_nonempty(env, *SNOWFLAKE_ENV_ACCOUNT)
        cls.ACCOUNT = account or None
        user = env_first_nonempty(env, *SNOWFLAKE_ENV_USER)
        cls.USER = user or None
        cls.PASSWORD = env_first_nonempty(env, *SNOWFLAKE_ENV_PASSWORD) or None
        database = env_first_nonempty(env, *SNOWFLAKE_ENV_DATABASE)
        cls.DATABASE = database or None
        cls.SCHEMA = env_first_nonempty(env, *SNOWFLAKE_ENV_SCHEMA) or "PUBLIC"
        warehouse = env_first_nonempty(env, *SNOWFLAKE_ENV_WAREHOUSE)
        cls.WAREHOUSE = warehouse or None
        role = env_first_nonempty(env, *SNOWFLAKE_ENV_ROLE)
        cls.ROLE = role or None
        private_key = env_first_nonempty(env, *SNOWFLAKE_ENV_PRIVATE_KEY_PATH)
        cls.PRIVATE_KEY_PATH = private_key or None
        cls.PRIVATE_KEY_PASSPHRASE = env_first_nonempty(env, *SNOWFLAKE_ENV_PRIVATE_KEY_PASSPHRASE) or None
        authenticator = env_first_nonempty(env, *SNOWFLAKE_ENV_AUTHENTICATOR)
        cls.AUTHENTICATOR = authenticator or None
        oauth_token = env_first_nonempty(env, *SNOWFLAKE_ENV_OAUTH_TOKEN)
        cls.OAUTH_TOKEN = oauth_token or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required Snowflake env vars are present."""
        if not (env_any_nonempty(env, SNOWFLAKE_ENV_ACCOUNT) and env_any_nonempty(env, SNOWFLAKE_ENV_USER)):
            return False
        if env_any_nonempty(env, SNOWFLAKE_ENV_PRIVATE_KEY_PATH):
            return True
        if env_any_nonempty(env, SNOWFLAKE_ENV_OAUTH_TOKEN):
            return True
        auth = env_first_nonempty(env, *SNOWFLAKE_ENV_AUTHENTICATOR)
        if auth.strip().lower() in ("externalbrowser", "oauth", "sso"):
            return True
        return env_any_nonempty(env, SNOWFLAKE_ENV_PASSWORD)

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return Snowflake connection values for slug and introspection."""
        return {
            "account": cls.ACCOUNT or "",
            "user": cls.USER or "",
            "password": cls.PASSWORD or "",
            "database": cls.DATABASE or "db",
            "schema": cls.SCHEMA or "PUBLIC",
            "warehouse": cls.WAREHOUSE or "",
            "role": cls.ROLE or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Snowflake storage paths (excludes warehouse/role)."""
        return ("account", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Snowflake secret field names."""
        return frozenset({"password"})


class BigQueryRuntimeConfig(EngineRuntimeConfig):
    """BigQuery connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "bigquery"

    PROJECT: ClassVar[str | None] = None
    DATASET: ClassVar[str | None] = None
    SCHEMA: ClassVar[str | None] = None
    CREDENTIALS_PATH: ClassVar[str | None] = None
    LOCATION: ClassVar[str] = "US"

    @classmethod
    def db_url(cls) -> str:
        """Build a SQLAlchemy BigQuery URL for inspection."""
        if not cls.PROJECT:
            raise ValueError("BigQuery project required")
        dataset = cls.DATASET or cls.SCHEMA
        if not dataset:
            raise ValueError("BigQuery dataset required")
        project_q = quote(str(cls.PROJECT), safe="")
        dataset_q = quote(str(dataset), safe="")
        location_q = quote(str(cls.LOCATION or "US"), safe="")
        return f"bigquery://{project_q}/{dataset_q}?location={location_q}"

    @classmethod
    def connect_args(cls) -> dict[str, Any]:
        """Return BigQuery driver connect arguments."""
        out: dict[str, Any] = {}
        if cls.has_service_account() and cls.CREDENTIALS_PATH:
            out["credentials_path"] = cls.CREDENTIALS_PATH
        return out

    @classmethod
    def has_service_account(cls) -> bool:
        """Return True when a service account JSON path is configured."""
        return bool(cls.CREDENTIALS_PATH)

    @classmethod
    def apply_environment(cls, env: Mapping[str, str]) -> None:
        """Apply BigQuery environment variables to ClassVars."""
        project = env_first_nonempty(env, *BIGQUERY_ENV_PROJECT)
        cls.PROJECT = project or None
        dataset = env_first_nonempty(env, *BIGQUERY_ENV_DATASET)
        cls.DATASET = dataset or None
        cls.SCHEMA = dataset or None
        credentials_path = env_first_nonempty(env, *BIGQUERY_ENV_CREDENTIALS_PATH)
        cls.CREDENTIALS_PATH = credentials_path or None
        cls.LOCATION = env_first_nonempty(env, *BIGQUERY_ENV_LOCATION) or "US"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required BigQuery env vars are present."""
        project = env_first_nonempty(env, *BIGQUERY_ENV_PROJECT)
        dataset = env_first_nonempty(env, *BIGQUERY_ENV_DATASET)
        return bool(project.strip() and dataset.strip())

    @classmethod
    def connection_slug_fields(cls) -> dict[str, str]:
        """Return BigQuery connection values for slug and introspection."""
        return {
            "project": cls.PROJECT or "",
            "dataset": cls.DATASET or cls.SCHEMA or "",
            "schema": cls.SCHEMA or cls.DATASET or "",
            "location": cls.LOCATION or "US",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for BigQuery storage paths (excludes location)."""
        return ("project", "dataset")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return BigQuery secret field names (none; credentials live in a file path)."""
        return frozenset()


class SandboxBundlePolicy:
    """Internal: refuse sandbox entry when the bundled corpus is absent."""

    REQUIRE_BUNDLE: ClassVar[bool] = True


class EngineConfig:
    """Internal process-wide defaults for backend selection (`TYPE`/`RUNTIME`), LLM credentials/models, and JSON artifact paths. This class is not part of the public API and is not exported from the ``aetherdialect`` package root. The only supported user-facing configuration paths are the documented environment variables (for example ``AZURE_OPENAI_DEPLOYMENT_LIGHT`` and ``AZURE_OPENAI_DEPLOYMENT_HEAVY``) and the ``EngineContext`` object passed to public entry points."""

    TYPE: ClassVar[str] = "postgresql"

    RUNTIME: ClassVar[type] = PostgresRuntimeConfig

    API_TOKEN: ClassVar[str | None] = os.environ.get("OPENAI_API_KEY")
    AZURE_API_TOKEN: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_API_KEY")
    LLM_PROVIDER: ClassVar[str] = "openai"
    MOCK_FIXTURES_FILE: ClassVar[str] = ""
    OPENAI_MODEL: ClassVar[str] = "gpt-4.1-nano"
    OPENAI_MODEL_INTENT: ClassVar[str] = "gpt-5.4-mini"
    OPENAI_MODEL_JOIN: ClassVar[str] = "gpt-5.4-nano"
    OPENAI_MODEL_SCHEMA_BASE: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_DDL: ClassVar[str] = "gpt-4.1-nano"
    OPENAI_MODEL_SCHEMA: ClassVar[str] = "gpt-5-mini"
    OPENAI_MODEL_SYNTH: ClassVar[str] = "gpt-5-mini"
    OPENAI_MODEL_SYNTH_VARIETY: ClassVar[str] = "gpt-5-nano"
    OPENAI_MODEL_INTENT_FORMAT: ClassVar[str] = "gpt-4.1-nano"
    OPENAI_MODEL_INTENT_SCHEMA_REPAIR: ClassVar[str] = "gpt-5.4-nano"
    OPENAI_MODEL_UPLOAD_SUMMARY: ClassVar[str] = "gpt-5.4-nano"
    OPENAI_MODEL_UPLOAD_INTERPRET: ClassVar[str] = "gpt-5-mini"
    OPENAI_BASE_URL: ClassVar[str | None] = "https://api.openai.com/v1"
    AZURE_OPENAI_BASE_URL: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_BASE_URL")
    AZURE_OPENAI_ENDPOINT: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_API_VERSION")

    SCHEMA_JSON_PATH: ClassVar[str] = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "schema_graph.json.gz")
    TEMPLATE_STORE_DIR: ClassVar[str] = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "intent_templates")

    @classmethod
    def azure_base_url(cls) -> str | None:
        """Return Azure OpenAI base URL in v1 form when configured."""
        if cls.AZURE_OPENAI_BASE_URL:
            return cls.AZURE_OPENAI_BASE_URL.rstrip("/")
        if cls.AZURE_OPENAI_ENDPOINT:
            return f"{cls.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1"
        return None


class QSimConfig:
    """QSim generation limits, ratios, sampling, output paths, and skeleton reference."""

    INTENT_TYPES = 20
    QUESTIONS_COUNT = 100
    MAX_TABLES_PER_INTENT = 3
    MAX_WHERE_PREDICATES_PER_INTENT = 4
    MAX_WHERE_COLUMNS = 2
    MAX_GROUP_BY_COLUMNS = 2

    MIN_AVG_VARIANTS_PER_INTENT = 1
    MAX_AVG_VARIANTS_PER_INTENT = 10

    MAX_NO_VARIANCE_RATIO = 0.25
    SINGLE_TABLE_RATIO = 0.40
    TWO_TABLE_RATIO = 0.40
    THREE_TABLE_RATIO = 0.20

    MAX_CONSECUTIVE_DUPLICATES = 5
    MAX_CONSECUTIVE_FAILURES = 5

    MIN_FILTER_RATIO = 0.70
    MIN_HAVING_RATIO = 0.15
    MIN_THREE_TABLE_RATIO = 0.10

    RANDOM_SEED = DEFAULT_RANDOM_SEED

    SELECT_COL_GEOMETRIC_P: float = 0.6

    COMPLEXITY_TARGET_PROPORTIONS: dict[str, float] = {
        "simple": 0.20,
        "moderate": 0.40,
        "complex": 0.30,
        "highly_complex": 0.10,
    }

    EXCLUDED_WHERE_PATTERNS = EXCLUDED_WHERE_PATTERNS

    SKELETONS_JSON_PATH = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "qsim_skeletons.json.gz")

    MIN_ADVANCED_FEATURE_RATIO = 0.15
    MIN_PER_FEATURE_FLOOR = 2


class SeedWarmupConfig:
    """Seed warmup expansion depth, artifact paths, sampling caps, and date/limit presets."""

    MAX_SEED_QUESTIONS: int = 500

    MAX_WHERE_PREDICATES = 3
    MAX_TABLES = 3
    MAX_GROUPBY = 2

    MAX_EXPR_COMPARISONS = 2
    MAX_HAVING_CONDITIONS = 2
    MAX_EXPANSION_DEPTH = 2

    ALLOW_HAVING_EXPR_EXPANSION: bool = False
    ALLOW_EMI_MUTATE_EXPANSION: bool = False

    SEED_WARMUP_BUNDLE_PATTERN = "seed_warmup_v{version}.zip"
    SEED_WARMUP_REPORT_PATTERN = "seed_warmup_report_v{version}.json"
    SEED_WARMUP_CACHE_ZIP = SEED_WARMUP_CACHE_ZIP
    WARMUP_CACHE_MANIFEST = "cache_manifest.json"
    WARMUP_CACHE_WORK_PREFIX = "work_units/"
    WARMUP_CACHE_GOLD_INTENTS_JSON = "gold/gold_intents.json"

    WARMUP_TARGET_CAP: int = 2000
    WARMUP_KEEP_ALL_BELOW: int = 2000
    WARMUP_MIN_GOLD_FRACTION: float = 0.15
    WARMUP_MIN_GOLD_FRACTION_BENCHMARK: float = 0.25
    WARMUP_SAMPLING_PROFILE: str = "default"
    WARMUP_MAX_FILLBACK_ROUNDS: int = 2
    WARMUP_STRATUM_MIN: int = 2
    WARMUP_SAMPLING_POLICY_VERSION: str = "5"
    MAX_WARMUP_EXECUTE_UNITS: int = 500_000
    SEED_WARMUP_CODE_VERSION: str = "3"

    WARMUP_ANCHOR_LATTICE_SUBDIR: str = WARMUP_ANCHOR_LATTICE_SUBDIR
    WARMUP_ANCHOR_LATTICE_CODE_VERSION: str = "3"

    WARMUP_QUESTION_STYLES: tuple[str, ...] = (
        "formal",
        "colloquial",
        "imperative",
        "interrogative",
        "descriptive",
        "concise",
        "keyword",
        "business_jargon",
        "beginner",
        "verbose",
    )

    WARMUP_QUESTION_STYLE_GUIDANCE: dict[str, str] = {
        "formal": "Polished professional analyst tone; complete sentences; no slang.",
        "colloquial": "Casual everyday wording as a colleague would speak.",
        "imperative": "Lead with a verb; compact command-style request.",
        "interrogative": "Clear question form using wh-words or how as appropriate.",
        "descriptive": "Neutral narrative statement of the insight or figures requested.",
        "concise": "Minimal words; one short sentence or tight fragment only.",
        "keyword": "Search-bar style; short keyword phrases without full grammar.",
        "business_jargon": "Domain analyst jargon and KPI language where natural.",
        "beginner": "Plain language for a newcomer; avoid insider abbreviations.",
        "verbose": "Fully spelled-out, slightly longer wording with explicit context.",
    }

    WARMUP_PARAPHRASES_PER_STYLE_MAX: int = 5

    COMPLEXITY_TARGET_PROPORTIONS: dict[str, float] = {
        "simple": 0.20,
        "moderate": 0.40,
        "complex": 0.30,
        "highly_complex": 0.10,
    }

    RULE_NLG_ANCHOR_COUNT: int = 12

    WARMUP_LLM_DIVERSITY_SUBSAMPLE_DIVISOR: int = 4

    WARMUP_MMR_LAMBDA: float = 0.7

    WARMUP_DIAGNOSTIC_REPAIR_MAX_ROUNDS: int = 1

    RANDOM_SEED = DEFAULT_RANDOM_SEED

    EXTRACT_EXPANSION_UNITS: list[str] = ["year", "month", "day", "quarter", "dow"]
    DATE_TRUNC_EXPANSION_UNITS: list[str] = ["month", "quarter", "year"]
    LIMIT_EXPANSION_VALUES: list[int] = [10, 50, 100]

    DATE_WINDOW_EXPANSION_PRESETS: list[dict[str, int | str]] = [
        {"unit": "day", "amount": 7},
        {"unit": "day", "amount": 30},
        {"unit": "day", "amount": 90},
        {"unit": "month", "amount": 1},
        {"unit": "month", "amount": 3},
        {"unit": "month", "amount": 6},
        {"unit": "month", "amount": 12},
        {"unit": "year", "amount": 1},
    ]

    DATE_DIFF_EXPANSION_PRESETS: list[dict[str, int | str]] = [
        {"unit": "day", "amount": 7},
        {"unit": "day", "amount": 30},
        {"unit": "day", "amount": 90},
    ]


def llm_credentials_configured() -> bool:
    """Return True when at least one LLM provider has required credentials on ``EngineConfig``."""

    def _non_empty_str(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    if EngineConfig.LLM_PROVIDER == "mock":
        return _non_empty_str(EngineConfig.MOCK_FIXTURES_FILE)
    openai_ok = _non_empty_str(EngineConfig.API_TOKEN)
    azure_ok = (
        _non_empty_str(EngineConfig.AZURE_API_TOKEN)
        and _non_empty_str(EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL)
        and _non_empty_str(EngineConfig.AZURE_OPENAI_API_VERSION)
    )
    return openai_ok or azure_ok
