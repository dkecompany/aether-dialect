"""Engine and policy settings, tunable thresholds, and shared validation constants. `BOOLEAN_TRUTH_PATTERN_MAP` maps lowercased two-valued top-K sets to the canonical affirmative literal (lowercase) used when recording ``ColumnMetadata.boolean_truth_value``. Connection identity keys stay on the config-file / environment path."""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from pathlib import Path
from types import MethodType
from typing import Any, ClassVar, Self
from urllib.parse import quote

from ._constants import (
    BIGQUERY_ENV_CREDENTIALS_PATH,
    BIGQUERY_ENV_DATASET,
    BIGQUERY_ENV_LOCATION,
    BIGQUERY_ENV_PROJECT,
    CLASS_DELEGATED_METHODS,
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
    ORACLE_ENV_AUTH_MODE,
    ORACLE_ENV_CONFIG_DIR,
    ORACLE_ENV_HOST,
    ORACLE_ENV_PASSWORD,
    ORACLE_ENV_PORT,
    ORACLE_ENV_SCHEMA,
    ORACLE_ENV_SERVICE_NAME,
    ORACLE_ENV_SID,
    ORACLE_ENV_THICK_MODE,
    ORACLE_ENV_TOKEN,
    ORACLE_ENV_USER,
    ORACLE_ENV_WALLET_LOCATION,
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
    SENTINEL_MODE_FREQUENCY_THRESHOLD,
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
    UNUSABLE_NULL_RATIO_THRESHOLD,
    WARMUP_ANCHOR_LATTICE_SUBDIR,
)
from ._constants_runtime import STOPWORDS_GRAMMATICAL_PARTICLES
from ._contracts_base import ConfigError


@dataclass(frozen=True, slots=True)
class EngineLimits:
    """Per-engine behavioural limits; ``None`` optional fields mean unlimited."""

    pool_size: int = 1
    pool_max_overflow: int = 4
    pool_recycle_seconds: int = 1800
    pool_pre_ping: bool = True
    pool_timeout_seconds: int = 30
    statement_timeout_ms: int | None = 30_000
    profile_timeout_ms: int | None = 120_000
    profiling_total_budget_seconds: int | None = None
    max_result_rows: int | None = 100_000
    max_result_bytes: int | None = 268_435_456
    max_upload_bytes: int = 268_435_456
    result_fetch_batch_rows: int | None = 10_000
    prompt_payload_max_bytes: int | None = 262_144
    write_queue_max_record_bytes: int | None = 1_048_576
    write_queue_max_file_bytes: int | None = None
    template_store_max_count: int | None = None
    template_store_max_disk_bytes: int | None = None
    template_value_history_depth: int = 64
    feedback_rows_per_question: int = 8
    template_partition_cache_size: int = 32
    artifact_lock_timeout_seconds: int = 30
    applied_map_archive_count: int = 3
    suspended_session_ttl_seconds: int | None = None

    @staticmethod
    def kwargs_from_table(table: Mapping[str, Any], limits_cls: type[Any]) -> dict[str, Any]:
        known = {field.name for field in fields(limits_cls)}
        kwargs: dict[str, Any] = {}
        for raw_key, raw_value in table.items():
            key = str(raw_key)
            if key not in known:
                raise ConfigError(f"unknown {limits_cls.__name__} field {key!r}")
            kwargs[key] = raw_value
        return kwargs

    def __post_init__(self) -> None:
        if self.pool_size < 1:
            raise ConfigError("pool_size must be at least 1")
        if self.pool_max_overflow < 0:
            raise ConfigError("pool_max_overflow must be non-negative")
        if self.pool_recycle_seconds < 0:
            raise ConfigError("pool_recycle_seconds must be non-negative")
        if self.pool_timeout_seconds < 0:
            raise ConfigError("pool_timeout_seconds must be non-negative")
        for name in (
            "statement_timeout_ms",
            "profile_timeout_ms",
            "profiling_total_budget_seconds",
            "max_result_rows",
            "max_result_bytes",
            "max_upload_bytes",
            "result_fetch_batch_rows",
            "prompt_payload_max_bytes",
            "write_queue_max_record_bytes",
            "write_queue_max_file_bytes",
            "template_store_max_count",
            "template_store_max_disk_bytes",
            "template_value_history_depth",
            "feedback_rows_per_question",
            "template_partition_cache_size",
            "artifact_lock_timeout_seconds",
            "applied_map_archive_count",
            "suspended_session_ttl_seconds",
        ):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ConfigError(f"{name} must be non-negative when set")
        if (
            self.max_result_rows is not None
            and self.result_fetch_batch_rows is not None
            and self.result_fetch_batch_rows > self.max_result_rows
        ):
            raise ConfigError("result_fetch_batch_rows cannot exceed max_result_rows")

    @classmethod
    def unlimited_optional_fields(cls) -> tuple[str, ...]:
        return (
            "profiling_total_budget_seconds",
            "write_queue_max_file_bytes",
            "template_store_max_count",
            "template_store_max_disk_bytes",
            "suspended_session_ttl_seconds",
        )

    @classmethod
    def from_config_file(cls, path: str | os.PathLike[str]) -> EngineLimits:
        """Load behavioural limits from the ``[limits]`` table in a TOML config file."""
        table = EngineConfig.load_toml_document(path).get("limits")
        if table is None:
            return cls()
        if not isinstance(table, dict):
            raise ConfigError("[limits] must be a table")
        return cls(**EngineLimits.kwargs_from_table(table, cls))


@dataclass(frozen=True, slots=True)
class FederationLimits:
    """Federation composition and coordination limits."""

    member_defaults: EngineLimits | None = None
    max_members: int = 8
    max_parallel_members: int = 4
    member_row_cap: int | None = 100_000
    member_bytes_cap: int | None = 268_435_456
    member_statement_timeout_ms: int | None = None
    member_probe_timeout_seconds: int = 10
    transfer_max_bytes: int | None = 536_870_912
    reduction_key_max_count: int | None = 10_000
    plan_step_count_max: int | None = 32
    coordinator_memory_limit_bytes: int | None = 2_147_483_648
    coordinator_threads: int = 4
    coordinator_temp_dir: str | None = None
    coordinator_spill_max_bytes: int | None = None
    federation_plan_template_count: int | None = None

    def __post_init__(self) -> None:
        if self.max_members < 1:
            raise ConfigError("max_members must be at least 1")
        if self.max_parallel_members < 1:
            raise ConfigError("max_parallel_members must be at least 1")
        if self.max_parallel_members > self.max_members:
            raise ConfigError("max_parallel_members cannot exceed max_members")
        for name in (
            "member_row_cap",
            "member_bytes_cap",
            "member_statement_timeout_ms",
            "member_probe_timeout_seconds",
            "transfer_max_bytes",
            "reduction_key_max_count",
            "plan_step_count_max",
            "coordinator_memory_limit_bytes",
            "coordinator_threads",
            "coordinator_spill_max_bytes",
            "federation_plan_template_count",
        ):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ConfigError(f"{name} must be non-negative when set")

    @classmethod
    def unlimited_optional_fields(cls) -> tuple[str, ...]:
        return (
            "member_defaults",
            "member_statement_timeout_ms",
            "coordinator_temp_dir",
            "coordinator_spill_max_bytes",
            "federation_plan_template_count",
        )

    @classmethod
    def from_config_file(cls, path: str | os.PathLike[str]) -> FederationLimits:
        """Load federation limits from the ``[federation_limits]`` table in a TOML config file."""
        table = EngineConfig.load_toml_document(path).get("federation_limits")
        if table is None:
            return cls()
        if not isinstance(table, dict):
            raise ConfigError("[federation_limits] must be a table")
        kwargs = EngineLimits.kwargs_from_table(table, cls)
        member_defaults = kwargs.get("member_defaults")
        if isinstance(member_defaults, dict):
            kwargs["member_defaults"] = EngineLimits(**EngineLimits.kwargs_from_table(member_defaults, EngineLimits))
        return cls(**kwargs)


class PolicyConfig:
    """ClassVar thresholds, penalties, stopwords, and SQL rejection patterns. Developer tracing: set ClassVar ``DEBUG``. ``telemetry_capture(..., force_diagnostic_flags=True)`` bumps an internal depth counter so diagnostics emit into the capture buffer without mutating ClassVars. Integrators can set ``PolicyConfig.DEBUG`` so failures and optional full diagnostic logs are emitted for capture. Cache rebuild shortcuts: set ``REGENERATE_TEMPLATE_STORE``, ``REGENERATE_SCHEMA_GRAPH``, or ``REGENERATE_SKELETON_CACHE`` to skip loading the corresponding on-disk artifact when present. Semantic join hints (non-FK overlap): profiling stores ``frequent_values`` and a separate ascending distinct ``value_overlap_sample`` (``VALUE_OVERLAP_SAMPLE_LIMIT``) for overlap; ``compute_semantic_profile_join_neighbors`` stores symmetric edges on ``ColumnMetadata.semantic_join_neighbors``. ``SEMANTIC_JOIN_MIN_OVERLAP_RATIO`` is the minimum ``|intersection| / min(|A|,|B|)`` on those two samples before an edge is recorded."""

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

    LLM_BATCH_ENABLED: ClassVar[bool] = False
    TABULAR_LLM_ASSIST: ClassVar[bool] = True

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

    UNUSABLE_NULL_RATIO_THRESHOLD = UNUSABLE_NULL_RATIO_THRESHOLD
    SENTINEL_MODE_FREQUENCY_THRESHOLD = SENTINEL_MODE_FREQUENCY_THRESHOLD
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

    MAX_QUERY_COST_ROWS: ClassVar[float | None] = 50_000_000.0
    MAX_QUERY_COST_BYTES: ClassVar[float | None] = 50_000_000_000.0
    STATEMENT_TIMEOUT_MS: ClassVar[int | None] = 30_000
    LLM_TIMEOUT_MS: ClassVar[int | None] = 60_000
    PROFILE_TIMEOUT_MS: ClassVar[int | None] = 120_000
    PROFILING_SAMPLE_THRESHOLD: ClassVar[int] = 100_000
    PROFILING_SAMPLE_SIZE: ClassVar[int] = 10_000
    PROFILING_SCHEMA_DEEP_QUERY_BUDGET: ClassVar[int | None] = 25_000
    MAX_ROLE_CLASSIFICATION_RETRIES: ClassVar[int] = 2
    EXPLAIN_TIMEOUT_MS: ClassVar[int | None] = None
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
        """Apply policy flags from *env* during engine construction."""
        batch_raw = EngineConfig.env_first_nonempty(env, "AETHERDIALECT_LLM_BATCH_ENABLED")
        if batch_raw:
            cls.LLM_BATCH_ENABLED = batch_raw.strip().lower() in ("1", "true", "yes", "on")
        assist_raw = EngineConfig.env_first_nonempty(env, "AETHERDIALECT_TABULAR_LLM_ASSIST")
        if assist_raw:
            cls.TABULAR_LLM_ASSIST = assist_raw.strip().lower() not in ("0", "false", "no", "off")


class _RuntimeConfigMeta(type):
    def __getattribute__(cls, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            return type.__getattribute__(cls, name)
        runtime_config_cls = globals().get("EngineRuntimeConfig")
        if runtime_config_cls is None:
            return type.__getattribute__(cls, name)
        if name == "apply_environment":

            def _deprecated_class_apply(_cls: type[Any], env: Mapping[str, str]) -> None:
                _cls.load_process_default_from_environment(env)

            return MethodType(_deprecated_class_apply, cls)
        instance_field_names = type.__getattribute__(runtime_config_cls, "instance_field_names")
        field_names = instance_field_names(cls)
        if name in field_names:
            process_default = type.__getattribute__(runtime_config_cls, "process_default_for_class")
            return getattr(process_default(cls), name)
        if name == "NATIVE_CONNECTION":
            attached = type.__getattribute__(runtime_config_cls, "_attached_natives")
            return attached.get(cls)
        if name in CLASS_DELEGATED_METHODS:
            process_default = type.__getattribute__(runtime_config_cls, "process_default_for_class")
            default = process_default(cls)
            attr = object.__getattribute__(default, name)
            if callable(attr):
                func = getattr(attr, "__func__", None)
                if func is not None:
                    return MethodType(func, default)
                return attr
            return attr
        return type.__getattribute__(cls, name)

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(f"{cls.__name__!r} has no attribute {name!r}")

    def __setattr__(cls, name: str, value: Any) -> None:
        if name.startswith("__"):
            return type.__setattr__(cls, name, value)
        runtime_config_cls = globals().get("EngineRuntimeConfig")
        if runtime_config_cls is None:
            return type.__setattr__(cls, name, value)
        instance_field_names = type.__getattribute__(runtime_config_cls, "instance_field_names")
        if name in instance_field_names(cls):
            process_default = type.__getattribute__(runtime_config_cls, "process_default_for_class")
            setattr(process_default(cls), name, value)
            return
        if name == "NATIVE_CONNECTION":
            attached = type.__getattribute__(runtime_config_cls, "_attached_natives")
            attached[cls] = value
            return
        type.__setattr__(cls, name, value)


@dataclass
class EngineRuntimeConfig(metaclass=_RuntimeConfigMeta):
    """Shared runtime-configuration contract for registered database engines."""

    ENGINE_NAME: ClassVar[str] = ""
    _attached_natives: ClassVar[dict[type[EngineRuntimeConfig], Any]] = {}
    _PROCESS_DEFAULT_RUNTIME_CONFIG: ClassVar[EngineRuntimeConfig | None] = None
    _PROCESS_DEFAULTS_BY_CLASS: ClassVar[dict[type[EngineRuntimeConfig], EngineRuntimeConfig]] = {}

    @classmethod
    def attach_connection(cls, connection: Any) -> None:
        """Attach a process-scoped native connection for engines that borrow an open handle."""
        EngineRuntimeConfig._attached_natives[cls] = connection

    @classmethod
    def clear_attached_connection(cls) -> None:
        """Clear any process-scoped native connection attached to this runtime class."""
        EngineRuntimeConfig._attached_natives.pop(cls, None)

    @classmethod
    def process_default_for_class(cls, target: type[EngineRuntimeConfig] | EngineRuntimeConfig) -> EngineRuntimeConfig:
        if not isinstance(target, type):
            return target
        default = EngineRuntimeConfig._PROCESS_DEFAULTS_BY_CLASS.get(target)
        if default is None:
            default = target()
            EngineRuntimeConfig._PROCESS_DEFAULTS_BY_CLASS[target] = default
        return default

    @classmethod
    def instance_field_names(cls, target: type[Any]) -> frozenset[str]:
        try:
            raw_fields = type.__getattribute__(target, "__dataclass_fields__")
        except AttributeError:
            return frozenset()
        classvar_marker = getattr(__import__("dataclasses"), "_FIELD_CLASSVAR", object())

        return frozenset(
            name for name, field in raw_fields.items() if getattr(field, "_field_type", None) is not classvar_marker
        )

    @classmethod
    def set_process_default_runtime_config(cls, config: EngineRuntimeConfig) -> None:
        EngineRuntimeConfig._PROCESS_DEFAULT_RUNTIME_CONFIG = config

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> Self:
        """Build a runtime config instance from *env*."""
        instance = cls()
        instance.apply_environment(env)
        return instance

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Load connection settings from *env* into this runtime config instance."""
        raise NotImplementedError

    @classmethod
    def load_process_default_from_environment(cls, env: Mapping[str, str]) -> None:
        """Populate the process-default instance used only during engine selection."""
        instance = cls.from_environment(env)
        EngineRuntimeConfig.set_process_default_runtime_config(instance)
        EngineRuntimeConfig._PROCESS_DEFAULTS_BY_CLASS[cls] = instance

    @classmethod
    def should_apply_environment(cls, env: Mapping[str, str]) -> bool:
        """Return True when enough of *env* is present to call :meth:`from_environment`."""
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

    def connection_slug_fields(self) -> dict[str, str]:
        """Return connection field values used for artifact storage slug computation."""
        return {}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return ordered keys from :meth:`connection_slug_fields` that participate in storage slugs."""
        return ()

    @classmethod
    def accepted_connection_keys(cls) -> frozenset[str]:
        """Return the environment variable names this engine's :meth:`apply_environment` recognizes, derived from its ``*_ENV_*`` alias tuples. Walks the MRO directly (skipping the metaclass ``apply_environment`` wrapper) to find the real implementation."""
        func = None
        for klass in cls.__mro__:
            candidate = klass.__dict__.get("apply_environment")
            if candidate is not None:
                func = candidate
                break
        if func is None:
            return frozenset()
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return frozenset()
        tuple_names = set(re.findall(r"\b([A-Z][A-Z0-9]*_ENV_[A-Z0-9_]+)\b", source))
        keys: set[str] = set()
        for name in tuple_names:
            value = globals().get(name)
            if isinstance(value, tuple):
                keys.update(str(v) for v in value)
        return frozenset(keys)

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return connection field names that must be redacted in runtime introspection."""
        return frozenset()

    def rotatable_credential_fields(self) -> tuple[str, ...]:
        """Return field names that may be replaced without rebuilding schema artifacts."""
        names: list[str] = []
        for name in sorted(self.redacted_fields()):
            upper = str(name).upper()
            if hasattr(self, upper):
                names.append(upper)
            elif hasattr(self, name):
                names.append(str(name))
        return tuple(names)

    def apply_connection_credentials(self, credentials: str | Mapping[str, str]) -> None:
        """Apply in-place credential replacement before (re)opening a database connection."""
        if isinstance(credentials, str):
            fields = self.rotatable_credential_fields()
            if not fields:
                raise ValueError(f"{self.ENGINE_NAME or type(self).__name__} has no rotatable credential fields")
            setattr(self, fields[0], credentials)
            return
        for raw_key, value in credentials.items():
            key = str(raw_key)
            if hasattr(self, key):
                field = key
            else:
                alias = key.lower()
                field = alias.upper() if hasattr(self, alias.upper()) else alias
                if not hasattr(self, field):
                    raise ValueError(
                        f"unknown credential field {raw_key!r} for {self.ENGINE_NAME or type(self).__name__}"
                    )
            setattr(self, field, value)


@dataclass
class PostgresRuntimeConfig(EngineRuntimeConfig):
    """PostgreSQL connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "postgresql"

    HOST: str = "localhost"
    PORT: int = 5432
    USER: str = "postgres"
    PASSWORD: str | None = None
    DATABASE: str | None = None
    SCHEMA: str = "public"

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Copy PostgreSQL connection variables from *env* into ClassVars."""
        host = EngineConfig.env_first_nonempty(env, *POSTGRES_ENV_HOST)
        self.HOST = host or "localhost"
        port_raw = EngineConfig.env_first_nonempty(env, *POSTGRES_ENV_PORT)
        self.PORT = int(port_raw) if port_raw else 5432
        self.USER = EngineConfig.env_first_nonempty(env, *POSTGRES_ENV_USER) or "postgres"
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *POSTGRES_ENV_PASSWORD)
        self.DATABASE = EngineConfig.env_first_nonempty(env, *POSTGRES_ENV_DATABASE)
        self.SCHEMA = EngineConfig.env_first_nonempty(env, *POSTGRES_ENV_SCHEMA) or "public"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when database, user, and password are configured."""
        return (
            EngineConfig.env_any_nonempty(env, POSTGRES_ENV_DATABASE)
            and EngineConfig.env_any_nonempty(env, POSTGRES_ENV_USER)
            and EngineConfig.env_any_nonempty(env, POSTGRES_ENV_PASSWORD)
        )

    @classmethod
    def selection_blockers(cls, env: Mapping[str, str]) -> list[str]:
        """Return driver or credential gaps preventing PostgreSQL selection."""
        driver_ok = EngineConfig.package_importable("psycopg")
        if cls.env_complete(env) and driver_ok:
            return []
        blockers: list[str] = []
        if not driver_ok:
            blockers.append("PostgreSQL driver (psycopg)")
        if driver_ok and not cls.env_complete(env):
            blockers.append(
                "PostgreSQL env (set one name from each required group): "
                + EngineConfig.env_role_hint("database", POSTGRES_ENV_DATABASE)
                + "; "
                + EngineConfig.env_role_hint("user", POSTGRES_ENV_USER)
                + "; "
                + EngineConfig.env_role_hint("password", POSTGRES_ENV_PASSWORD)
                + "; optional "
                + EngineConfig.env_role_hint("host", POSTGRES_ENV_HOST)
                + "; "
                + EngineConfig.env_role_hint("port", POSTGRES_ENV_PORT)
                + "; "
                + EngineConfig.env_role_hint("schema", POSTGRES_ENV_SCHEMA)
            )
        return blockers

    def connection_slug_fields(self) -> dict[str, str]:
        """Return PostgreSQL connection values for slug and introspection."""
        return {
            "host": self.HOST or "localhost",
            "port": str(int(self.PORT)),
            "database": self.DATABASE or "db",
            "schema": self.SCHEMA or "public",
            "user": self.USER or "",
            "password": self.PASSWORD or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for PostgreSQL storage paths."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return PostgreSQL secret field names."""
        return frozenset({"password"})

    def db_url(self) -> str:
        """Build a SQLAlchemy PostgreSQL URL from ClassVars (``postgresql+psycopg``)."""
        if not self.PASSWORD:
            raise ValueError("PostgreSQL password required")
        if not self.DATABASE:
            raise ValueError("PostgreSQL database required")
        user_q = quote(str(self.USER), safe="")
        pwd_q = quote(str(self.PASSWORD), safe="")
        db_q = quote(str(self.DATABASE), safe="")
        return f"postgresql+psycopg://{user_q}:{pwd_q}@{self.HOST}:{self.PORT}/{db_q}"


@dataclass
class DatabricksRuntimeConfig(EngineRuntimeConfig):
    """Unity Catalog `CATALOG`/`SCHEMA` and optional ODBC connector settings (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "databricks"

    CATALOG: str | None = None
    SCHEMA: str | None = None

    SERVER_HOSTNAME: str | None = None
    HTTP_PATH: str | None = None
    ACCESS_TOKEN: str | None = None

    @classmethod
    def uc_scope_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when catalog and schema are configured."""
        return EngineConfig.env_any_nonempty(env, DATABRICKS_ENV_CATALOG) and EngineConfig.env_any_nonempty(
            env, DATABRICKS_ENV_SCHEMA
        )

    @classmethod
    def sql_warehouse_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when SQL warehouse hostname, HTTP path, and token are configured."""
        return (
            EngineConfig.env_any_nonempty(env, DATABRICKS_ENV_SERVER_HOSTNAME)
            and EngineConfig.env_any_nonempty(env, DATABRICKS_ENV_HTTP_PATH)
            and EngineConfig.env_any_nonempty(env, DATABRICKS_ENV_TOKEN)
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

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Copy Databricks connection variables from *env* into ClassVars."""
        self.SERVER_HOSTNAME = EngineConfig.env_first_nonempty(env, *DATABRICKS_ENV_SERVER_HOSTNAME)
        self.HTTP_PATH = EngineConfig.env_first_nonempty(env, *DATABRICKS_ENV_HTTP_PATH)
        self.ACCESS_TOKEN = EngineConfig.env_first_nonempty(env, *DATABRICKS_ENV_TOKEN)
        self.CATALOG = EngineConfig.env_first_nonempty(env, *DATABRICKS_ENV_CATALOG)
        self.SCHEMA = EngineConfig.env_first_nonempty(env, *DATABRICKS_ENV_SCHEMA)
        self.validate()

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
            return EngineConfig.package_importable("databricks.sql")
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
            and not EngineConfig.package_importable("databricks.sql")
        ):
            blockers.append(
                "Databricks SQL warehouse variables are set but the databricks-sql-connector package is not installed."
            )
        elif not cls.env_complete(env):
            blockers.append(
                "Databricks env: "
                + EngineConfig.env_role_hint("catalog", DATABRICKS_ENV_CATALOG)
                + "; "
                + EngineConfig.env_role_hint("schema", DATABRICKS_ENV_SCHEMA)
                + "; then either all of "
                + EngineConfig.env_role_hint("server hostname", DATABRICKS_ENV_SERVER_HOSTNAME)
                + ", "
                + EngineConfig.env_role_hint("SQL warehouse HTTP path", DATABRICKS_ENV_HTTP_PATH)
                + ", "
                + EngineConfig.env_role_hint("access token", DATABRICKS_ENV_TOKEN)
                + " (with databricks-sql-connector installed), or an active PySpark session."
            )
        return blockers

    def connection_slug_fields(self) -> dict[str, str]:
        """Return Databricks connection values for slug and introspection."""
        host_raw = (self.SERVER_HOSTNAME or "").strip() or "pyspark"
        return {
            "server_hostname": self.SERVER_HOSTNAME or "",
            "http_path": self.HTTP_PATH or "",
            "catalog": self.CATALOG or "catalog",
            "schema": self.SCHEMA or "schema",
            "host": host_raw.split(".")[0],
            "access_token": self.ACCESS_TOKEN or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Databricks storage paths."""
        return ("host", "catalog", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Databricks secret field names."""
        return frozenset({"access_token"})

    def has_native_connection(self) -> bool:
        """True when hostname, HTTP path, and access token are all non- empty."""
        return bool(self.SERVER_HOSTNAME and self.HTTP_PATH and self.ACCESS_TOKEN)

    def validate(self) -> None:
        """Require `CATALOG` and `SCHEMA`."""
        if not self.CATALOG:
            raise ValueError("Databricks catalog required")
        if not self.SCHEMA:
            raise ValueError("Databricks schema required")

    def sqlalchemy_url(self) -> str | None:
        """Build a SQLAlchemy URL for the Databricks SQL connector when. PAT credentials exist."""
        if not self.has_native_connection():
            return None

        token = quote(self.ACCESS_TOKEN or "", safe="")
        host = self.SERVER_HOSTNAME or ""
        http_path = quote(self.HTTP_PATH or "", safe="")
        catalog = quote(self.CATALOG or "", safe="")
        schema = quote(self.SCHEMA or "", safe="")
        return f"databricks://token:{token}@{host}?http_path={http_path}&catalog={catalog}&schema={schema}"


@dataclass
class MySQLRuntimeConfig(EngineRuntimeConfig):
    """MySQL connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "mysql"

    HOST: str = "localhost"
    PORT: int = 3306
    USER: str = "root"
    PASSWORD: str | None = None
    DATABASE: str | None = None
    SCHEMA: str | None = None

    def db_url(self) -> str:
        """Build a SQLAlchemy MySQL URL from ClassVars."""
        if not self.PASSWORD:
            raise ValueError("MySQL password required")
        if not self.DATABASE:
            raise ValueError("MySQL database required")
        user_q = quote(str(self.USER), safe="")
        pwd_q = quote(str(self.PASSWORD), safe="")
        db_q = quote(str(self.DATABASE), safe="")
        return f"mysql+pymysql://{user_q}:{pwd_q}@{self.HOST}:{self.PORT}/{db_q}?charset=utf8mb4"

    def connect_args(self) -> dict[str, Any]:
        """Return driver connect arguments for MySQL."""
        return {}

    def has_password_auth(self) -> bool:
        """Return True when password authentication is configured."""
        return bool(self.PASSWORD)

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Apply MySQL environment variables to ClassVars."""
        self.HOST = EngineConfig.env_first_nonempty(env, *MYSQL_ENV_HOST) or "localhost"
        port_raw = EngineConfig.env_first_nonempty(env, *MYSQL_ENV_PORT)
        self.PORT = int(port_raw) if port_raw else 3306
        self.USER = EngineConfig.env_first_nonempty(env, *MYSQL_ENV_USER) or "root"
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *MYSQL_ENV_PASSWORD) or None
        database = EngineConfig.env_first_nonempty(env, *MYSQL_ENV_DATABASE)
        self.DATABASE = database or None
        self.SCHEMA = database or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required MySQL env vars are present."""
        return (
            EngineConfig.env_any_nonempty(env, MYSQL_ENV_PASSWORD)
            and EngineConfig.env_any_nonempty(env, MYSQL_ENV_DATABASE)
            and EngineConfig.env_any_nonempty(env, MYSQL_ENV_USER)
        )

    def connection_slug_fields(self) -> dict[str, str]:
        """Return MySQL connection values for slug and introspection."""
        return {
            "host": self.HOST or "localhost",
            "port": str(int(self.PORT)),
            "database": self.DATABASE or "db",
            "schema": self.SCHEMA or self.DATABASE or "db",
            "user": self.USER or "",
            "password": self.PASSWORD or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for MySQL storage paths."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return MySQL secret field names."""
        return frozenset({"password"})


@dataclass
class MariaDBRuntimeConfig(MySQLRuntimeConfig):
    """MariaDB connection defaults that reuse the MySQL backend via the pymysql driver."""

    ENGINE_NAME: ClassVar[str] = "mariadb"

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Populate MariaDB ClassVars from MARIADB_* env keys only."""
        self.HOST = EngineConfig.env_first_nonempty(env, *MARIADB_ENV_HOST) or "localhost"
        port_raw = EngineConfig.env_first_nonempty(env, *MARIADB_ENV_PORT)
        self.PORT = int(port_raw) if port_raw else 3306
        self.USER = EngineConfig.env_first_nonempty(env, *MARIADB_ENV_USER) or "root"
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *MARIADB_ENV_PASSWORD) or None
        database = EngineConfig.env_first_nonempty(env, *MARIADB_ENV_DATABASE)
        self.DATABASE = database or None
        self.SCHEMA = database or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when MariaDB user, password, and database env keys are all present."""
        return (
            EngineConfig.env_any_nonempty(env, MARIADB_ENV_PASSWORD)
            and EngineConfig.env_any_nonempty(env, MARIADB_ENV_DATABASE)
            and EngineConfig.env_any_nonempty(env, MARIADB_ENV_USER)
        )


@dataclass
class DuckDBRuntimeConfig(EngineRuntimeConfig):
    """DuckDB embedded-database connection defaults sourced from a local file path or :memory:."""

    ENGINE_NAME: ClassVar[str] = "duckdb"

    DATABASE_PATH: str = ":memory:"
    SCHEMA: str = "main"

    def db_url(self) -> str:
        """Build a SQLAlchemy DuckDB URL from the configured file path or :memory:."""
        path = str(self.DATABASE_PATH or ":memory:")
        return f"duckdb:///{path}"

    def connect_args(self) -> dict[str, Any]:
        """Return driver connect arguments for DuckDB."""
        return {}

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Populate DuckDB ClassVars from DUCKDB_* env keys."""
        self.DATABASE_PATH = EngineConfig.env_first_nonempty(env, *DUCKDB_ENV_PATH) or ":memory:"
        self.SCHEMA = EngineConfig.env_first_nonempty(env, *DUCKDB_ENV_SCHEMA) or "main"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when a DuckDB database path env key is present."""
        return EngineConfig.env_any_nonempty(env, DUCKDB_ENV_PATH)

    def connection_slug_fields(self) -> dict[str, str]:
        """Return DuckDB connection values for slug and introspection."""
        raw = str(self.DATABASE_PATH or ":memory:")
        base = "memory" if raw == ":memory:" else os.path.splitext(os.path.basename(raw))[0] or "duckdb"
        return {"database": base, "schema": self.SCHEMA or "main"}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for DuckDB storage paths."""
        return ("database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return DuckDB secret field names (none for an embedded database)."""
        return frozenset()


@dataclass
class CsvRuntimeConfig(EngineRuntimeConfig):
    """CSV/Excel file-source connection defaults for the DuckDB upload store."""

    ENGINE_NAME: ClassVar[str] = "csv"

    DIRECTORY: str | None = None
    FILES: tuple[str, ...] = ()
    SOURCE_SELECTIONS: dict[str, dict[str, Any]] = field(default_factory=dict)
    SCHEMA: str = "main"

    def db_url(self) -> str:
        """Return the DuckDB SQLAlchemy URL used by the CSV backend (memory when no artifacts)."""
        return "duckdb:///:memory:"

    def connect_args(self) -> dict[str, Any]:
        """Return driver connect arguments for the CSV DuckDB backend."""
        return {}

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Populate CSV ClassVars from CSV_DIRECTORY / CSV_FILES env keys."""
        directory = EngineConfig.env_first_nonempty(env, *CSV_ENV_DIRECTORY)
        files_raw = EngineConfig.env_first_nonempty(env, *CSV_ENV_FILES)
        self.DIRECTORY = directory or None
        if files_raw:
            self.FILES = tuple(part.strip() for part in files_raw.split(",") if part.strip())
        else:
            self.FILES = ()
        self.SCHEMA = "main"
        if self.DIRECTORY and self.FILES:
            raise ConfigError("csv: set either CSV_DIRECTORY or CSV_FILES, not both")

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when a CSV directory or explicit file list is configured."""
        return EngineConfig.env_any_nonempty(env, CSV_ENV_DIRECTORY) or EngineConfig.env_any_nonempty(
            env, CSV_ENV_FILES
        )

    @classmethod
    def selection_blockers(cls, env: Mapping[str, str]) -> list[str]:
        """Return driver or configuration gaps preventing CSV engine selection."""
        driver_ok = EngineConfig.package_importable("duckdb")
        if cls.env_complete(env) and driver_ok:
            try:
                cls.from_environment(env).resolve_source_files()
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

    def resolve_source_files(self) -> tuple[Path, ...]:
        """Resolve configured CSV/Excel inputs to absolute file paths."""
        engine = self.ENGINE_NAME
        allowed = self._allowed_source_suffixes()
        directory = str(self.DIRECTORY or "").strip()
        files = tuple(str(item).strip() for item in self.FILES if str(item).strip())
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

    def set_source_selections(self, raw: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Store per-file upload interpretation choices for the CSV file engine."""
        if not raw:
            self.SOURCE_SELECTIONS = {}
            return
        self.SOURCE_SELECTIONS = {str(key): dict(value) for key, value in raw.items()}

    def connection_slug_fields(self) -> dict[str, str]:
        """Return CSV source values for slug and introspection."""
        try:
            paths = self.resolve_source_files()
            source_key = hashlib.sha256("|".join(path.as_posix() for path in paths).encode()).hexdigest()
        except Exception:
            files_key = ",".join(Path(str(item)).as_posix() for item in self.FILES)
            source_key = hashlib.sha256(f"{self.DIRECTORY or ''}|{files_key}".encode()).hexdigest()
        return {"source": source_key[:32], "schema": self.SCHEMA or "main"}

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for CSV storage paths."""
        return ("source", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return CSV secret field names (none for file sources)."""
        return frozenset()


@dataclass
class SQLiteRuntimeConfig(EngineRuntimeConfig):
    """SQLite embedded-database connection defaults sourced from a local file path or :memory:."""

    ENGINE_NAME: ClassVar[str] = "sqlite"

    DATABASE_PATH: str = ":memory:"
    SCHEMA: str = "main"

    def db_url(self) -> str:
        """Build a SQLAlchemy SQLite URL from the configured file path or :memory:."""
        path = str(self.DATABASE_PATH or ":memory:")
        return f"sqlite:///{path}"

    def connect_args(self) -> dict[str, Any]:
        """Return driver connect arguments for SQLite."""
        return {}

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Populate SQLite ClassVars from SQLITE_* env keys."""
        self.DATABASE_PATH = EngineConfig.env_first_nonempty(env, *SQLITE_ENV_PATH) or ":memory:"
        self.SCHEMA = "main"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when a SQLite database path env key is present."""
        return EngineConfig.env_any_nonempty(env, SQLITE_ENV_PATH)

    def connection_slug_fields(self) -> dict[str, str]:
        """Return SQLite connection values for slug and introspection."""
        raw = str(self.DATABASE_PATH or ":memory:")
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


@dataclass
class RedshiftRuntimeConfig(EngineRuntimeConfig):
    """Amazon Redshift connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "redshift"

    HOST: str = "localhost"
    PORT: int = 5439
    USER: str = "awsuser"
    PASSWORD: str | None = None
    DATABASE: str = "dev"
    SCHEMA: str = "public"
    USE_IAM: bool = False
    CLUSTER_IDENTIFIER: str | None = None
    WORKGROUP: str | None = None
    REGION: str | None = None

    def db_url(self) -> str:
        """Build a SQLAlchemy Redshift URL from ClassVars."""
        if self.has_iam_credentials():
            user_q = quote(str(self.USER), safe="")
            db_q = quote(str(self.DATABASE), safe="")
            host_q = quote(str(self.HOST), safe="")
            params = ["sslmode=verify-full"]
            if self.CLUSTER_IDENTIFIER:
                params.append(f"cluster_identifier={quote(str(self.CLUSTER_IDENTIFIER), safe='')}")
            if self.WORKGROUP:
                params.append(f"workgroup={quote(str(self.WORKGROUP), safe='')}")
            if self.REGION:
                params.append(f"region={quote(str(self.REGION), safe='')}")
            query = "&".join(params)
            return f"redshift+redshift_connector://{user_q}@{host_q}:{self.PORT}/{db_q}?{query}"
        if not self.PASSWORD:
            raise ValueError("Redshift password required when IAM is disabled")
        user_q = quote(str(self.USER), safe="")
        pwd_q = quote(str(self.PASSWORD), safe="")
        db_q = quote(str(self.DATABASE), safe="")
        return f"redshift+redshift_connector://{user_q}:{pwd_q}@{self.HOST}:{self.PORT}/{db_q}?sslmode=verify-full"

    def connect_args(self) -> dict[str, Any]:
        """Return driver connect arguments for Redshift."""
        if self.has_iam_credentials():
            return {"iam": True, "ssl": True, "sslmode": "verify-full"}
        return {"ssl": True, "sslmode": "verify-full"}

    def has_iam_credentials(self) -> bool:
        """Return True when IAM authentication is configured."""
        return bool(self.USE_IAM and (self.CLUSTER_IDENTIFIER or self.WORKGROUP))

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Apply Redshift environment variables to ClassVars."""
        self.HOST = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_HOST) or "localhost"
        port_raw = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_PORT)
        self.PORT = int(port_raw) if port_raw else 5439
        self.USER = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_USER) or "awsuser"
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_PASSWORD) or None
        self.DATABASE = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_DATABASE) or "dev"
        self.SCHEMA = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_SCHEMA) or "public"
        use_iam_raw = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_USE_IAM).lower()
        self.USE_IAM = use_iam_raw in ("1", "true", "yes", "on") if use_iam_raw else False
        cluster = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_CLUSTER_IDENTIFIER)
        self.CLUSTER_IDENTIFIER = cluster or None
        workgroup = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_WORKGROUP)
        self.WORKGROUP = workgroup or None
        region = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_REGION)
        self.REGION = region or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required Redshift env vars are present."""
        use_iam_raw = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_USE_IAM).lower()
        use_iam = use_iam_raw in ("1", "true", "yes", "on") if use_iam_raw else False
        if use_iam:
            user = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_USER) or "awsuser"
            cluster = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_CLUSTER_IDENTIFIER) or ""
            workgroup = EngineConfig.env_first_nonempty(env, *REDSHIFT_ENV_WORKGROUP) or ""
            return bool(str(user).strip() and (str(cluster).strip() or str(workgroup).strip()))
        return (
            EngineConfig.env_any_nonempty(env, REDSHIFT_ENV_PASSWORD)
            and EngineConfig.env_any_nonempty(env, REDSHIFT_ENV_USER)
            and EngineConfig.env_any_nonempty(env, REDSHIFT_ENV_DATABASE)
        )

    def connection_slug_fields(self) -> dict[str, str]:
        """Return Redshift connection values for slug and introspection."""
        return {
            "host": self.HOST or "localhost",
            "port": str(int(self.PORT)),
            "database": self.DATABASE or "dev",
            "schema": self.SCHEMA or "public",
            "user": self.USER or "",
            "password": self.PASSWORD or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Redshift storage paths (excludes cluster/workgroup/region)."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Redshift secret field names."""
        return frozenset({"password"})


@dataclass
class SQLServerRuntimeConfig(EngineRuntimeConfig):
    """SQL Server connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "sqlserver"

    HOST: str = "localhost"
    PORT: int = 1433
    USER: str | None = None
    PASSWORD: str | None = None
    DATABASE: str | None = None
    SCHEMA: str = "dbo"
    DRIVER: str = "ODBC Driver 18 for SQL Server"
    AUTH_MODE: str = "sql"
    TENANT_ID: str | None = None
    CLIENT_ID: str | None = None
    CLIENT_SECRET: str | None = None

    def db_url(self) -> str:
        """Build a SQLAlchemy SQL Server URL from ClassVars."""
        if not self.DATABASE:
            raise ValueError("SQL Server database required")
        driver_q = quote(str(self.DRIVER), safe="")
        if self.AUTH_MODE == "windows":
            return (
                f"mssql+pyodbc://@{self.HOST}:{self.PORT}/{quote(str(self.DATABASE), safe='')}"
                f"?driver={driver_q}&Trusted_Connection=yes"
            )
        if self.AUTH_MODE == "aad_password":
            if not self.USER or not self.PASSWORD:
                raise ValueError("SQL Server user and password required for Azure AD password authentication")
            user_q = quote(str(self.USER), safe="")
            pwd_q = quote(str(self.PASSWORD), safe="")
            db_q = quote(str(self.DATABASE), safe="")
            return (
                f"mssql+pyodbc://{user_q}:{pwd_q}@{self.HOST}:{self.PORT}/{db_q}"
                f"?driver={driver_q}&Authentication=ActiveDirectoryPassword"
                f"&Encrypt=yes&TrustServerCertificate=yes"
            )
        if self.AUTH_MODE == "aad_sp":
            if not self.CLIENT_ID or not self.CLIENT_SECRET:
                raise ValueError("SQL Server client id and secret required for Azure AD service principal")
            client_q = quote(str(self.CLIENT_ID), safe="")
            secret_q = quote(str(self.CLIENT_SECRET), safe="")
            db_q = quote(str(self.DATABASE), safe="")
            return (
                f"mssql+pyodbc://@{self.HOST}:{self.PORT}/{db_q}"
                f"?driver={driver_q}&Authentication=ActiveDirectoryServicePrincipal"
                f"&UID={client_q}&PWD={secret_q}&Encrypt=yes&TrustServerCertificate=yes"
            )
        if not self.USER or not self.PASSWORD:
            raise ValueError("SQL Server user and password required for SQL authentication")
        user_q = quote(str(self.USER), safe="")
        pwd_q = quote(str(self.PASSWORD), safe="")
        db_q = quote(str(self.DATABASE), safe="")
        return f"mssql+pyodbc://{user_q}:{pwd_q}@{self.HOST}:{self.PORT}/{db_q}?driver={driver_q}&TrustServerCertificate=yes"

    def connect_args(self) -> dict[str, Any]:
        """Return driver connect arguments for SQL Server."""
        return {}

    def has_sql_auth(self) -> bool:
        """Return True when SQL authentication is configured."""
        return self.AUTH_MODE == "sql" and bool(self.USER and self.PASSWORD)

    def has_windows_auth(self) -> bool:
        """Return True when Windows integrated authentication is selected."""
        return self.AUTH_MODE == "windows"

    def has_aad_password_auth(self) -> bool:
        """Return True when Azure AD password authentication is selected."""
        return self.AUTH_MODE == "aad_password" and bool(self.USER and self.PASSWORD and self.TENANT_ID)

    def has_aad_service_principal_auth(self) -> bool:
        """Return True when Azure AD service principal authentication is selected."""
        return self.AUTH_MODE == "aad_sp" and bool(self.CLIENT_ID and self.CLIENT_SECRET and self.TENANT_ID)

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Apply SQL Server environment variables to ClassVars."""
        self.HOST = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_HOST) or "localhost"
        port_raw = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_PORT)
        self.PORT = int(port_raw) if port_raw else 1433
        user = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_USER)
        self.USER = user or None
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_PASSWORD) or None
        database = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_DATABASE)
        self.DATABASE = database or None
        self.SCHEMA = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_SCHEMA) or "dbo"
        self.DRIVER = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_DRIVER) or "ODBC Driver 18 for SQL Server"
        auth_mode = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_AUTH_MODE)
        self.AUTH_MODE = auth_mode.lower() if auth_mode else "sql"
        tenant = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_TENANT_ID)
        self.TENANT_ID = tenant or None
        client_id = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_CLIENT_ID)
        self.CLIENT_ID = client_id or None
        client_secret = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_CLIENT_SECRET)
        self.CLIENT_SECRET = client_secret or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required SQL Server env vars are present."""
        auth = EngineConfig.env_first_nonempty(env, *SQLSERVER_ENV_AUTH_MODE) or "sql"
        auth = auth.strip().lower()
        if not EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_DATABASE):
            return False
        if auth == "windows":
            return True
        if auth == "aad_password":
            return (
                EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_USER)
                and EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_PASSWORD)
                and EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_TENANT_ID)
            )
        if auth == "aad_sp":
            return (
                EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_TENANT_ID)
                and EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_CLIENT_ID)
                and EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_CLIENT_SECRET)
            )
        return EngineConfig.env_any_nonempty(env, SQLSERVER_ENV_USER) and EngineConfig.env_any_nonempty(
            env, SQLSERVER_ENV_PASSWORD
        )

    def connection_slug_fields(self) -> dict[str, str]:
        """Return SQL Server connection values for slug and introspection."""
        return {
            "host": self.HOST or "localhost",
            "port": str(int(self.PORT)),
            "database": self.DATABASE or "db",
            "schema": self.SCHEMA or "dbo",
            "user": self.USER or "",
            "password": self.PASSWORD or "",
            "auth_mode": self.AUTH_MODE or "sql",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for SQL Server storage paths (excludes auth_mode/driver)."""
        return ("host", "port", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return SQL Server secret field names."""
        return frozenset({"password"})


@dataclass
class OracleRuntimeConfig(EngineRuntimeConfig):
    """Oracle connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "oracle"

    HOST: str = "localhost"
    PORT: int = 1521
    USER: str | None = None
    PASSWORD: str | None = None
    SERVICE_NAME: str | None = None
    SID: str | None = None
    SCHEMA: str | None = None
    AUTH_MODE: str = "password"
    WALLET_LOCATION: str | None = None
    CONFIG_DIR: str | None = None
    TOKEN: str | None = None
    THICK_MODE: bool = False

    def db_url(self) -> str:
        """Build a SQLAlchemy ``oracle+oracledb`` URL from ClassVars."""
        service = (self.SERVICE_NAME or "").strip()
        sid = (self.SID or "").strip()
        if not service and not sid:
            raise ValueError("Oracle service_name or sid required")
        if service and sid:
            raise ValueError("Oracle service_name and sid are mutually exclusive")
        host = self.HOST or "localhost"
        port = int(self.PORT)
        mode = (self.AUTH_MODE or "password").strip().lower()
        if mode == "token":
            if not self.TOKEN:
                raise ValueError("Oracle token required for token authentication")
            user_q = quote(str(self.USER or ""), safe="") if self.USER else ""
            auth = f"{user_q}@" if user_q else ""
        elif mode == "wallet":
            if not self.WALLET_LOCATION and not self.CONFIG_DIR:
                raise ValueError("Oracle wallet_location or config_dir required for wallet authentication")
            user_q = quote(str(self.USER or ""), safe="") if self.USER else ""
            if self.USER and self.PASSWORD:
                pwd_q = quote(str(self.PASSWORD), safe="")
                auth = f"{user_q}:{pwd_q}@"
            elif user_q:
                auth = f"{user_q}@"
            else:
                auth = ""
        else:
            if not self.USER or not self.PASSWORD:
                raise ValueError("Oracle user and password required for password authentication")
            user_q = quote(str(self.USER), safe="")
            pwd_q = quote(str(self.PASSWORD), safe="")
            auth = f"{user_q}:{pwd_q}@"
        if service:
            query = f"service_name={quote(service, safe='')}"
        else:
            query = f"sid={quote(sid, safe='')}"
        return f"oracle+oracledb://{auth}{host}:{port}/?{query}"

    def connect_args(self) -> dict[str, Any]:
        """Return python-oracledb connect arguments for wallet or token auth."""
        out: dict[str, Any] = {}
        mode = (self.AUTH_MODE or "password").strip().lower()
        if self.CONFIG_DIR:
            out["config_dir"] = str(self.CONFIG_DIR)
        if self.WALLET_LOCATION:
            out["wallet_location"] = str(self.WALLET_LOCATION)
        if mode == "token" and self.TOKEN:
            out["access_token"] = str(self.TOKEN)
        return out

    def ensure_driver_mode(self) -> None:
        """Initialize thick-mode Oracle client libraries when configured."""
        if not self.THICK_MODE:
            return
        import oracledb

        if getattr(oracledb, "is_thin_mode", lambda: True)():
            oracledb.init_oracle_client()

    def has_password_auth(self) -> bool:
        """Return True when password authentication is configured."""
        return (self.AUTH_MODE or "password").strip().lower() == "password" and bool(self.USER and self.PASSWORD)

    def has_wallet_auth(self) -> bool:
        """Return True when wallet authentication is selected with a wallet path."""
        return (self.AUTH_MODE or "").strip().lower() == "wallet" and bool(self.WALLET_LOCATION or self.CONFIG_DIR)

    def has_token_auth(self) -> bool:
        """Return True when token authentication is selected with a token."""
        return (self.AUTH_MODE or "").strip().lower() == "token" and bool(self.TOKEN)

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Apply Oracle environment variables to ClassVars."""
        self.HOST = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_HOST) or "localhost"
        port_raw = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_PORT)
        self.PORT = int(port_raw) if port_raw else 1521
        user = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_USER)
        self.USER = user or None
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_PASSWORD) or None
        service = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_SERVICE_NAME)
        self.SERVICE_NAME = service or None
        sid = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_SID)
        self.SID = sid or None
        schema = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_SCHEMA)
        if schema:
            self.SCHEMA = schema
        elif self.USER:
            self.SCHEMA = str(self.USER).upper()
        else:
            self.SCHEMA = None
        auth_mode = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_AUTH_MODE)
        self.AUTH_MODE = auth_mode.lower() if auth_mode else "password"
        self.WALLET_LOCATION = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_WALLET_LOCATION) or None
        self.CONFIG_DIR = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_CONFIG_DIR) or None
        self.TOKEN = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_TOKEN) or None
        thick_raw = EngineConfig.env_first_nonempty(env, *ORACLE_ENV_THICK_MODE)
        self.THICK_MODE = bool(thick_raw) and thick_raw.strip().lower() not in {"0", "false", "no", "off"}

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required Oracle env vars are present."""
        auth = (EngineConfig.env_first_nonempty(env, *ORACLE_ENV_AUTH_MODE) or "password").strip().lower()
        has_target = EngineConfig.env_any_nonempty(env, ORACLE_ENV_SERVICE_NAME) or EngineConfig.env_any_nonempty(
            env, ORACLE_ENV_SID
        )
        if not has_target:
            return False
        if auth == "wallet":
            return EngineConfig.env_any_nonempty(env, ORACLE_ENV_WALLET_LOCATION) or EngineConfig.env_any_nonempty(
                env, ORACLE_ENV_CONFIG_DIR
            )
        if auth == "token":
            return EngineConfig.env_any_nonempty(env, ORACLE_ENV_TOKEN)
        return EngineConfig.env_any_nonempty(env, ORACLE_ENV_USER) and EngineConfig.env_any_nonempty(
            env, ORACLE_ENV_PASSWORD
        )

    def connection_slug_fields(self) -> dict[str, str]:
        """Return Oracle connection values for slug and introspection."""
        return {
            "host": self.HOST or "localhost",
            "port": str(int(self.PORT)),
            "service_name": self.SERVICE_NAME or "",
            "sid": self.SID or "",
            "schema": self.SCHEMA or (str(self.USER).upper() if self.USER else ""),
            "user": self.USER or "",
            "password": self.PASSWORD or "",
            "auth_mode": self.AUTH_MODE or "password",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Oracle storage paths."""
        return ("host", "port", "service_name", "sid", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Oracle secret field names."""
        return frozenset({"password", "token"})


@dataclass
class SnowflakeRuntimeConfig(EngineRuntimeConfig):
    """Snowflake connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "snowflake"

    ACCOUNT: str | None = None
    USER: str | None = None
    PASSWORD: str | None = None
    DATABASE: str | None = None
    SCHEMA: str = "PUBLIC"
    WAREHOUSE: str | None = None
    ROLE: str | None = None
    PRIVATE_KEY_PATH: str | None = None
    PRIVATE_KEY_PASSPHRASE: str | None = None
    AUTHENTICATOR: str | None = None
    OAUTH_TOKEN: str | None = None

    def db_url(self) -> str:
        """Build a SQLAlchemy Snowflake URL from ClassVars."""
        if not self.ACCOUNT or not self.USER:
            raise ValueError("Snowflake account and user required")
        account_q = quote(str(self.ACCOUNT), safe="")
        user_q = quote(str(self.USER), safe="")
        if self.has_password_auth():
            pwd_q = quote(str(self.PASSWORD or ""), safe="")
            auth = f"{user_q}:{pwd_q}"
        else:
            auth = user_q
        params: list[str] = []
        if self.DATABASE:
            params.append(f"database={quote(str(self.DATABASE), safe='')}")
        if self.SCHEMA:
            params.append(f"schema={quote(str(self.SCHEMA), safe='')}")
        if self.WAREHOUSE:
            params.append(f"warehouse={quote(str(self.WAREHOUSE), safe='')}")
        if self.ROLE:
            params.append(f"role={quote(str(self.ROLE), safe='')}")
        if self.AUTHENTICATOR:
            params.append(f"authenticator={quote(str(self.AUTHENTICATOR), safe='')}")
        query = "&".join(params)
        base = f"snowflake://{auth}@{account_q}"
        return f"{base}/?{query}" if query else base

    def connect_args(self) -> dict[str, Any]:
        """Return Snowflake driver connect arguments."""
        out: dict[str, Any] = {}
        if self.has_oauth_auth() and self.OAUTH_TOKEN:
            out["token"] = self.OAUTH_TOKEN
        if self.has_keypair_auth() and self.PRIVATE_KEY_PATH:
            out["private_key_file"] = self.PRIVATE_KEY_PATH
            if self.PRIVATE_KEY_PASSPHRASE:
                out["private_key_file_pwd"] = self.PRIVATE_KEY_PASSPHRASE
        return out

    def has_password_auth(self) -> bool:
        """Return True when password authentication is configured."""
        return bool(self.PASSWORD) and not self.has_keypair_auth() and not self.has_oauth_auth()

    def has_keypair_auth(self) -> bool:
        """Return True when key-pair authentication is configured."""
        return bool(self.PRIVATE_KEY_PATH)

    def has_oauth_auth(self) -> bool:
        """Return True when OAuth authentication is configured."""
        return bool(self.OAUTH_TOKEN or (self.AUTHENTICATOR and "oauth" in str(self.AUTHENTICATOR).lower()))

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

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Apply Snowflake environment variables to ClassVars."""
        account = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_ACCOUNT)
        self.ACCOUNT = account or None
        user = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_USER)
        self.USER = user or None
        self.PASSWORD = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_PASSWORD) or None
        database = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_DATABASE)
        self.DATABASE = database or None
        self.SCHEMA = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_SCHEMA) or "PUBLIC"
        warehouse = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_WAREHOUSE)
        self.WAREHOUSE = warehouse or None
        role = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_ROLE)
        self.ROLE = role or None
        private_key = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_PRIVATE_KEY_PATH)
        self.PRIVATE_KEY_PATH = private_key or None
        self.PRIVATE_KEY_PASSPHRASE = (
            EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_PRIVATE_KEY_PASSPHRASE) or None
        )
        authenticator = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_AUTHENTICATOR)
        self.AUTHENTICATOR = authenticator or None
        oauth_token = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_OAUTH_TOKEN)
        self.OAUTH_TOKEN = oauth_token or None

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required Snowflake env vars are present."""
        if not (
            EngineConfig.env_any_nonempty(env, SNOWFLAKE_ENV_ACCOUNT)
            and EngineConfig.env_any_nonempty(env, SNOWFLAKE_ENV_USER)
        ):
            return False
        if EngineConfig.env_any_nonempty(env, SNOWFLAKE_ENV_PRIVATE_KEY_PATH):
            return True
        if EngineConfig.env_any_nonempty(env, SNOWFLAKE_ENV_OAUTH_TOKEN):
            return True
        auth = EngineConfig.env_first_nonempty(env, *SNOWFLAKE_ENV_AUTHENTICATOR)
        if auth.strip().lower() in ("externalbrowser", "oauth", "sso"):
            return True
        return EngineConfig.env_any_nonempty(env, SNOWFLAKE_ENV_PASSWORD)

    def connection_slug_fields(self) -> dict[str, str]:
        """Return Snowflake connection values for slug and introspection."""
        return {
            "account": self.ACCOUNT or "",
            "user": self.USER or "",
            "password": self.PASSWORD or "",
            "database": self.DATABASE or "db",
            "schema": self.SCHEMA or "PUBLIC",
            "warehouse": self.WAREHOUSE or "",
            "role": self.ROLE or "",
        }

    @classmethod
    def connection_slug_keys(cls) -> tuple[str, ...]:
        """Return slug field order for Snowflake storage paths (excludes warehouse/role)."""
        return ("account", "database", "schema")

    @classmethod
    def redacted_fields(cls) -> frozenset[str]:
        """Return Snowflake secret field names."""
        return frozenset({"password"})


@dataclass
class BigQueryRuntimeConfig(EngineRuntimeConfig):
    """BigQuery connection defaults (ClassVars)."""

    ENGINE_NAME: ClassVar[str] = "bigquery"

    PROJECT: str | None = None
    DATASET: str | None = None
    SCHEMA: str | None = None
    CREDENTIALS_PATH: str | None = None
    LOCATION: str = "US"

    def db_url(self) -> str:
        """Build a SQLAlchemy BigQuery URL for inspection."""
        if not self.PROJECT:
            raise ValueError("BigQuery project required")
        dataset = self.DATASET or self.SCHEMA
        if not dataset:
            raise ValueError("BigQuery dataset required")
        project_q = quote(str(self.PROJECT), safe="")
        dataset_q = quote(str(dataset), safe="")
        location_q = quote(str(self.LOCATION or "US"), safe="")
        return f"bigquery://{project_q}/{dataset_q}?location={location_q}"

    def connect_args(self) -> dict[str, Any]:
        """Return BigQuery driver connect arguments."""
        out: dict[str, Any] = {}
        if self.has_service_account() and self.CREDENTIALS_PATH:
            out["credentials_path"] = self.CREDENTIALS_PATH
        return out

    def has_service_account(self) -> bool:
        """Return True when a service account JSON path is configured."""
        return bool(self.CREDENTIALS_PATH)

    def apply_environment(self, env: Mapping[str, str]) -> None:
        """Apply BigQuery environment variables to ClassVars."""
        project = EngineConfig.env_first_nonempty(env, *BIGQUERY_ENV_PROJECT)
        self.PROJECT = project or None
        dataset = EngineConfig.env_first_nonempty(env, *BIGQUERY_ENV_DATASET)
        self.DATASET = dataset or None
        self.SCHEMA = dataset or None
        credentials_path = EngineConfig.env_first_nonempty(env, *BIGQUERY_ENV_CREDENTIALS_PATH)
        self.CREDENTIALS_PATH = credentials_path or None
        self.LOCATION = EngineConfig.env_first_nonempty(env, *BIGQUERY_ENV_LOCATION) or "US"

    @classmethod
    def env_complete(cls, env: Mapping[str, str]) -> bool:
        """Return True when required BigQuery env vars are present."""
        project = EngineConfig.env_first_nonempty(env, *BIGQUERY_ENV_PROJECT)
        dataset = EngineConfig.env_first_nonempty(env, *BIGQUERY_ENV_DATASET)
        return bool(project.strip() and dataset.strip())

    def connection_slug_fields(self) -> dict[str, str]:
        """Return BigQuery connection values for slug and introspection."""
        return {
            "project": self.PROJECT or "",
            "dataset": self.DATASET or self.SCHEMA or "",
            "schema": self.SCHEMA or self.DATASET or "",
            "location": self.LOCATION or "US",
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

    RUNTIME: ClassVar[type[EngineRuntimeConfig] | EngineRuntimeConfig] = PostgresRuntimeConfig

    API_TOKEN: ClassVar[str | None] = None
    AZURE_API_TOKEN: ClassVar[str | None] = None
    LLM_PROVIDER: ClassVar[str] = "openai"
    MOCK_FIXTURES_FILE: ClassVar[str] = ""
    OPENAI_MODEL: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_INTENT: ClassVar[str] = "gpt-5.4-mini"
    OPENAI_MODEL_JOIN: ClassVar[str] = "gpt-5.4-nano"
    OPENAI_MODEL_SCHEMA_BASE: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_DDL: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_SCHEMA: ClassVar[str] = "gpt-5-mini"
    OPENAI_MODEL_DOMAIN_KNOWLEDGE: ClassVar[str] = "gpt-5.4-mini"
    OPENAI_MODEL_SYNTH: ClassVar[str] = "gpt-5-mini"
    OPENAI_MODEL_SYNTH_VARIETY: ClassVar[str] = "gpt-5-nano"
    OPENAI_MODEL_INTENT_FORMAT: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_INTENT_SCHEMA_REPAIR: ClassVar[str] = "gpt-5.4-nano"
    OPENAI_MODEL_UPLOAD_SUMMARY: ClassVar[str] = "gpt-5.4-nano"
    OPENAI_MODEL_UPLOAD_INTERPRET: ClassVar[str] = "gpt-5-mini"
    OPENAI_BASE_URL: ClassVar[str | None] = "https://api.openai.com/v1"
    AZURE_OPENAI_BASE_URL: ClassVar[str | None] = None
    AZURE_OPENAI_ENDPOINT: ClassVar[str | None] = None
    AZURE_OPENAI_API_VERSION: ClassVar[str | None] = None

    SCHEMA_JSON_PATH: ClassVar[str] = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "schema_graph.json.gz")
    TEMPLATE_STORE_DIR: ClassVar[str] = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "intent_templates")

    @staticmethod
    def default_artifacts_root() -> Path:
        """Return the default on-disk parent directory for engine artifacts."""
        from platformdirs import user_data_dir

        return Path(user_data_dir(appname="aetherdialect", appauthor=False))

    @staticmethod
    def load_toml_document(path: str | os.PathLike[str]) -> dict[str, Any]:
        path_str = str(path).strip()
        if not path_str:
            raise ConfigError("config path is empty")
        expanded = os.path.expanduser(path_str)
        try:
            with open(expanded, "rb") as file_handle:
                document = tomllib.load(file_handle)
        except OSError as exc:
            raise ConfigError(f"config_file cannot be opened: {expanded}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"config_file TOML parse error in {expanded}: {exc}") from exc
        if not isinstance(document, dict):
            raise ConfigError(f"config_file root must be a table: {expanded}")
        return document

    @staticmethod
    def normalize_llm_provider(raw: str) -> str:
        """Return the canonical LLM provider label (``mock`` maps to ``sandbox``)."""
        value = str(raw or "").strip().lower()
        if value == "mock":
            return "sandbox"
        return value

    @staticmethod
    def is_sandbox_llm_provider(provider: str | None) -> bool:
        """Return True when *provider* selects offline sandbox fixture replay."""
        return EngineConfig.normalize_llm_provider(str(provider or "")) == "sandbox"

    @staticmethod
    def llm_credentials_configured() -> bool:
        """Return True when at least one LLM provider has required credentials on ``EngineConfig``."""

        def _non_empty_str(value: object) -> bool:
            return isinstance(value, str) and bool(value.strip())

        if EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
            return _non_empty_str(EngineConfig.MOCK_FIXTURES_FILE)
        openai_ok = _non_empty_str(EngineConfig.API_TOKEN)
        azure_ok = (
            _non_empty_str(EngineConfig.AZURE_API_TOKEN)
            and _non_empty_str(EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL)
            and _non_empty_str(EngineConfig.AZURE_OPENAI_API_VERSION)
        )
        return openai_ok or azure_ok

    @classmethod
    def azure_base_url(cls) -> str | None:
        """Return Azure OpenAI base URL in v1 form when configured."""
        if cls.AZURE_OPENAI_BASE_URL:
            return cls.AZURE_OPENAI_BASE_URL.rstrip("/")
        if cls.AZURE_OPENAI_ENDPOINT:
            return f"{cls.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1"
        return None

    @staticmethod
    def env_first_nonempty(env: Mapping[str, str], *keys: str) -> str:
        """Return the first non-blank value among *keys*, else an empty string."""
        for key in keys:
            value = str(env.get(key, "") or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
        """Return True when at least one key maps to a non-blank string."""
        return any(str(env.get(key, "") or "").strip() for key in keys)

    @staticmethod
    def env_role_hint(label: str, keys: tuple[str, ...]) -> str:
        """Return a human-readable hint listing acceptable environment variable names."""
        return f"{label}: {' or '.join(keys)}"

    @staticmethod
    def package_importable(name: str) -> bool:
        """Return True when *name* can be imported as a top-level module."""
        return find_spec(name) is not None


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
    WARMUP_ANCHOR_LATTICE_CODE_VERSION: str = "4"

    WARMUP_QUESTION_STYLES: tuple[str, ...] = (
        "formal",
        "colloquial",
        "imperative",
        "interrogative",
        "descriptive",
        "concise",
        "keyword",
        "domain_jargon",
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
        "domain_jargon": "Domain-specific jargon and measure language where natural.",
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

    EXTRACT_EXPANSION_UNITS: list[str] = ["year", "month", "day", "quarter", "week", "dow"]
    DATE_TRUNC_EXPANSION_UNITS: list[str] = ["month", "quarter", "year", "week"]
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
