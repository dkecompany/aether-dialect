"""Tests for config module constants and normalization functions."""

import os
import re

import pytest

from aetherdialect._config import (
    AGG_PATTERN,
    AGG_QUANTITY_RE,
    AGGREGATION_ALLOWED_COLUMN_TYPES,
    BOOLEAN_FILTER_OPS,
    CATEGORICAL_FILTER_OPS,
    COLUMN_TYPE_TO_VALUE_TYPE,
    ENGINE_STORAGE_PLACEHOLDER_DIR,
    INTENT_SCHEMA,
    NUMERIC_CATEGORICAL_FILTER_OPS,
    NUMERIC_FILTER_OPS,
    NUMERIC_ONLY_AGGREGATIONS,
    QSIM_QUESTIONS_PATTERN,
    REGISTRY_CASE_ID_RE,
    REGISTRY_WINDOW_ID_RE,
    ROLE_ALLOWED_AGGREGATIONS,
    ROLE_VALUE_TYPE_COMPAT,
    SCALAR_FUNCTIONS_LEADING_ARG,
    SCALAR_FUNCTIONS_NUMERIC,
    SCALAR_FUNCTIONS_STRING,
    SCALAR_FUNCTIONS_TEMPORAL,
    SEED_FAILURE_CODE_REALISM_DROPPED,
    SEED_WARMUP_FAILURE_CODES,
    TABLE_COL_PATTERN,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_EXPECTED_ROWS,
    VALID_FILTER_OPS,
    VALID_FILTER_VALUE_TYPES,
    VALID_GRAINS,
    VALID_HAVING_OPS,
    VALID_HAVING_VALUE_TYPES,
    VALID_RELATIVE_DATE_UNITS,
    VALID_SCALAR_FUNCTIONS,
    VALID_VALUE_TYPES,
    VALID_WINDOW_FUNCTIONS,
    VALUE_TYPE_NORMALIZATION,
    WINDOW_AGG_FUNCTIONS,
    WINDOW_OFFSET_FUNCTIONS,
    WINDOW_RANKING_FUNCTIONS,
    WINDOW_VALUE_FUNCTIONS,
    DatabricksRuntimeConfig,
    EngineConfig,
    PolicyConfig,
    PostgresRuntimeConfig,
    QSimConfig,
    SeedWarmupConfig,
    diagnostic_debug_enabled,
    diagnostic_force_enter,
    diagnostic_force_exit,
    effective_explain_timeout_ms,
    effective_llm_timeout_ms,
    normalize_column_type,
    normalize_value_type,
    seed_warmup_failure_code_from_validate_sql_error,
)


class TestNormalizeColumnType:
    """Tests for normalize_column_type."""

    def test_strips_parenthesized_params(self):
        """normalize_column_type strips (N) from types."""
        assert normalize_column_type("VARCHAR(255)") == "varchar"
        assert normalize_column_type("NUMERIC(10,2)") == "numeric"

    def test_lowercases(self):
        """normalize_column_type lowercases."""
        assert normalize_column_type("INTEGER") == "integer"

    def test_strips_whitespace(self):
        """normalize_column_type strips whitespace."""
        assert normalize_column_type("  text  ") == "text"

    def test_no_params(self):
        """normalize_column_type with no params."""
        assert normalize_column_type("bigint") == "bigint"

    def test_empty_string_returns_empty(self):
        """normalize_column_type with empty string returns empty string."""
        assert normalize_column_type("") == ""

    def test_single_digit_in_parens(self):
        """normalize_column_type strips single numeric param."""
        assert normalize_column_type("char(1)") == "char"

    def test_complex_whitespace_after_strip(self):
        """normalize_column_type normalizes type with internal spaces in parens."""
        assert normalize_column_type("numeric(10, 2)") == "numeric"

    def test_parens_only_strips_to_empty(self):
        """normalize_column_type strips to empty when only parens and digits."""
        assert normalize_column_type("(10)") == ""

    def test_multiple_paren_groups_stripped(self):
        """normalize_column_type strips first parenthesized group only."""
        result = normalize_column_type("decimal(10,2)")
        assert result == "decimal"


class TestRoleValueTypeCompat:
    """Tests for ``ROLE_VALUE_TYPE_COMPAT``."""

    def test_boolean_role_accepts_boolean_integer_string_value_types(self):
        """Boolean column roles remain compatible with boolean-shaped physical storage kinds."""

        assert ROLE_VALUE_TYPE_COMPAT["boolean"] == frozenset({"boolean", "integer", "string"})


class TestNormalizeValueType:
    """Tests for normalize_value_type."""

    def test_timestamp_to_date(self):
        """normalize_value_type maps timestamp -> date."""
        assert normalize_value_type("timestamp") == "date"

    def test_numeric_to_number(self):
        """normalize_value_type maps numeric -> number."""
        assert normalize_value_type("numeric") == "number"

    def test_varchar_to_string(self):
        """normalize_value_type maps varchar -> string."""
        assert normalize_value_type("varchar") == "string"

    def test_bool_to_boolean(self):
        """normalize_value_type maps bool -> boolean."""
        assert normalize_value_type("bool") == "boolean"

    def test_already_valid(self):
        """normalize_value_type passes through already valid types."""
        assert normalize_value_type("integer") == "integer"
        assert normalize_value_type("string") == "string"
        assert normalize_value_type("date") == "date"

    def test_empty_to_string(self):
        """normalize_value_type maps empty to string."""
        assert normalize_value_type("") == "string"

    def test_unknown_to_string(self):
        """normalize_value_type maps unknown to string."""
        assert normalize_value_type("xml_blob") == "string"

    def test_case_insensitive(self):
        """normalize_value_type is case-insensitive."""
        assert normalize_value_type("TIMESTAMP") == "date"
        assert normalize_value_type("Integer") == "integer"

    def test_whitespace_only_returns_string(self):
        """normalize_value_type returns string when input is only whitespace."""
        assert normalize_value_type("   ") == "string"
        assert normalize_value_type("\t\n") == "string"

    def test_stripped_whitespace_around_valid_type(self):
        """normalize_value_type strips leading and trailing whitespace."""
        assert normalize_value_type("  integer  ") == "integer"

    def test_all_value_type_normalization_keys(self):
        """normalize_value_type maps every VALUE_TYPE_NORMALIZATION key."""
        for raw, expected in VALUE_TYPE_NORMALIZATION.items():
            assert normalize_value_type(raw) == expected, f"{raw!r} -> {expected!r}"

    def test_timestamptz_maps_to_date(self):
        """normalize_value_type maps timestamptz to date."""
        assert normalize_value_type("timestamptz") == "date"

    def test_decimal_maps_to_number(self):
        """normalize_value_type maps decimal to number."""
        assert normalize_value_type("decimal") == "number"

    def test_uuid_maps_to_string(self):
        """normalize_value_type maps uuid to string."""
        assert normalize_value_type("uuid") == "string"

    def test_enum_maps_to_string(self):
        """normalize_value_type maps enum to string."""
        assert normalize_value_type("enum") == "string"

    def test_null_passthrough(self):
        """normalize_value_type passes through null as valid type."""
        assert normalize_value_type("null") == "null"


class TestAggQuantityRe:
    """Tests for AGG_QUANTITY_RE regex."""

    def test_more_than_matches(self):
        """AGG_QUANTITY_RE matches 'more than N'."""
        assert AGG_QUANTITY_RE.search("Orders with more than 5 items") is not None
        assert AGG_QUANTITY_RE.search("more than 10") is not None

    def test_at_least_matches(self):
        """AGG_QUANTITY_RE matches 'at least N'."""
        assert AGG_QUANTITY_RE.search("at least 3") is not None

    def test_over_matches(self):
        """AGG_QUANTITY_RE matches 'over N'."""
        assert AGG_QUANTITY_RE.search("over 100") is not None

    def test_less_than_matches(self):
        """AGG_QUANTITY_RE matches 'less than N'."""
        assert AGG_QUANTITY_RE.search("less than 2") is not None

    def test_no_more_than_matches(self):
        """AGG_QUANTITY_RE matches 'no more than N'."""
        assert AGG_QUANTITY_RE.search("no more than 1") is not None

    def test_a_minimum_of_matches(self):
        """AGG_QUANTITY_RE matches 'a minimum of N'."""
        assert AGG_QUANTITY_RE.search("a minimum of 5") is not None

    def test_a_maximum_of_matches(self):
        """AGG_QUANTITY_RE matches 'a maximum of N'."""
        assert AGG_QUANTITY_RE.search("a maximum of 10") is not None

    def test_no_match_without_number(self):
        """AGG_QUANTITY_RE does not match when no number follows phrase."""
        assert AGG_QUANTITY_RE.search("more than many") is None

    def test_case_insensitive(self):
        """AGG_QUANTITY_RE is case-insensitive."""
        assert AGG_QUANTITY_RE.search("MORE THAN 5") is not None

    def test_above_matches(self):
        """AGG_QUANTITY_RE matches 'above N'."""
        assert AGG_QUANTITY_RE.search("above 3") is not None
        assert AGG_QUANTITY_RE.search("customers with above 5 rentals") is not None

    def test_below_matches(self):
        """AGG_QUANTITY_RE matches 'below N'."""
        assert AGG_QUANTITY_RE.search("below 10") is not None
        assert AGG_QUANTITY_RE.search("stores with below 2 employees") is not None


class TestAggPattern:
    """Tests for AGG_PATTERN regex."""

    def test_matches_count(self):
        """AGG_PATTERN matches COUNT(expr)."""
        m = AGG_PATTERN.match("COUNT(table.id)")
        assert m is not None
        assert m.group(1).lower() == "count"
        assert m.group(2) == "table.id"

    def test_matches_sum_with_spaces(self):
        """AGG_PATTERN matches SUM( expr ) with spaces."""
        m = AGG_PATTERN.match("SUM( t.amount )")
        assert m is not None
        assert m.group(1).lower() == "sum"

    def test_no_match_bare_column(self):
        """AGG_PATTERN does not match bare column."""
        assert AGG_PATTERN.match("table.col") is None

    def test_matches_avg_min_max(self):
        """AGG_PATTERN matches AVG, MIN, MAX."""
        for agg in ("AVG", "MIN", "MAX"):
            m = AGG_PATTERN.match(f"{agg}(x)")
            assert m is not None, agg
            assert m.group(1).lower() == agg.lower()


class TestValidAggregationFunctions:
    """Tests for VALID_AGGREGATION_FUNCTIONS constant."""

    def test_exactly_five_functions(self):
        """VALID_AGGREGATION_FUNCTIONS has exactly 5 entries."""
        assert len(VALID_AGGREGATION_FUNCTIONS) == 5

    def test_expected_members(self):
        """VALID_AGGREGATION_FUNCTIONS contains expected members."""
        assert VALID_AGGREGATION_FUNCTIONS == {"count", "sum", "avg", "min", "max"}

    def test_no_count_distinct(self):
        """count_distinct is not in VALID_AGGREGATION_FUNCTIONS."""
        assert "count_distinct" not in VALID_AGGREGATION_FUNCTIONS


class TestRoleAllowedAggregations:
    """Tests for ROLE_ALLOWED_AGGREGATIONS constant."""

    def test_identifier_only_count(self):
        """IDENTIFIER role only allows count."""
        assert ROLE_ALLOWED_AGGREGATIONS["IDENTIFIER"] == {"count"}

    def test_numeric_measure_all(self):
        """NUMERIC_MEASURE role allows all agg functions."""
        assert ROLE_ALLOWED_AGGREGATIONS["NUMERIC_MEASURE"] == {
            "count",
            "sum",
            "avg",
            "min",
            "max",
        }

    def test_audit_empty(self):
        """AUDIT role allows no aggregations."""
        assert ROLE_ALLOWED_AGGREGATIONS["AUDIT"] == set()

    def test_boolean_count_and_sum(self):
        """BOOLEAN role allows only count (sum is rejected by AGGREGATION_ALLOWED_COLUMN_TYPES['sum'])."""
        assert ROLE_ALLOWED_AGGREGATIONS["BOOLEAN"] == {"count"}

    def test_all_roles_are_subset_of_valid(self):
        """Every role's aggregations are subset of VALID_AGGREGATION_FUNCTIONS."""
        for role, aggs in ROLE_ALLOWED_AGGREGATIONS.items():
            assert aggs.issubset(VALID_AGGREGATION_FUNCTIONS), (
                f"{role} has invalid aggs: {aggs - VALID_AGGREGATION_FUNCTIONS}"
            )


class TestNumericOnlyAggregations:
    """Tests for NUMERIC_ONLY_AGGREGATIONS constant."""

    def test_expected_members(self):
        """NUMERIC_ONLY_AGGREGATIONS is sum and avg."""
        assert NUMERIC_ONLY_AGGREGATIONS == {"sum", "avg"}


class TestValidConstants:
    """Tests for constant sets."""

    def test_valid_grains(self):
        """VALID_GRAINS has expected values."""
        assert VALID_GRAINS == {"scalar", "grouped", "row_level"}

    def test_valid_filter_ops_includes_like(self):
        """VALID_FILTER_OPS includes like operators."""
        assert "like" in VALID_FILTER_OPS
        assert "ilike" in VALID_FILTER_OPS
        assert "contains" in VALID_FILTER_OPS

    def test_valid_having_ops_no_like(self):
        """VALID_HAVING_OPS does not include like."""
        assert "like" not in VALID_HAVING_OPS

    def test_valid_value_types(self):
        """VALID_VALUE_TYPES has expected members."""
        for vt in (
            "integer",
            "string",
            "date",
            "number",
            "null",
            "boolean",
            "date_window",
        ):
            assert vt in VALID_VALUE_TYPES

    def test_valid_scalar_functions(self):
        """VALID_SCALAR_FUNCTIONS includes core functions."""
        for f in (
            "upper",
            "lower",
            "trim",
            "abs",
            "round",
            "year",
            "month",
            "day",
            "coalesce",
        ):
            assert f in VALID_SCALAR_FUNCTIONS


class TestPolicyConfig:
    """Tests for PolicyConfig constants."""

    def test_penalty_cap(self):
        """PolicyConfig.PENALTY_CAP is a float."""
        assert isinstance(PolicyConfig.PENALTY_CAP, float)
        assert PolicyConfig.PENALTY_CAP > 0

    def test_stopwords_excludes_data_terms(self):
        """PolicyConfig.STOPWORDS does not contain data-relevant words."""
        assert "count" not in PolicyConfig.STOPWORDS
        assert "total" not in PolicyConfig.STOPWORDS
        assert "sum" not in PolicyConfig.STOPWORDS

    def test_max_repair_loops_positive(self):
        """PolicyConfig.MAX_STAGE_B_REPAIRS and MAX_FRESH_RESTARTS are non-negative integers with valid bounds."""
        assert isinstance(PolicyConfig.MAX_STAGE_B_REPAIRS, int)
        assert PolicyConfig.MAX_STAGE_B_REPAIRS >= 1
        assert isinstance(PolicyConfig.MAX_FRESH_RESTARTS, int)
        assert PolicyConfig.MAX_FRESH_RESTARTS >= 0

    def test_forbidden_sql_is_list_of_patterns(self):
        """PolicyConfig.FORBIDDEN_SQL is a non-empty list of regex strings."""
        assert isinstance(PolicyConfig.FORBIDDEN_SQL, list)
        assert len(PolicyConfig.FORBIDDEN_SQL) > 0

    def test_categorical_thresholds_positive(self):
        """Categorical thresholds are positive."""
        assert PolicyConfig.CATEGORICAL_MAX_CARDINALITY > 0
        assert 0 < PolicyConfig.CATEGORICAL_MAX_RATIO < 1

    def test_final_sql_auto_accept_threshold_in_unit_interval(self):
        """FINAL_SQL_AUTO_ACCEPT_THRESHOLD is in (0, 1]."""
        assert 0 < PolicyConfig.FINAL_SQL_AUTO_ACCEPT_THRESHOLD <= 1

    def test_debug_and_verbose_are_bool(self):
        """PolicyConfig.DEBUG and VERBOSE are booleans."""
        assert isinstance(PolicyConfig.DEBUG, bool)
        assert isinstance(PolicyConfig.VERBOSE, bool)

    def test_pipeline_trace_full_is_bool(self):
        """PIPELINE_TRACE_FULL gates verbose pipeline tracing with DEBUG."""
        assert isinstance(PolicyConfig.PIPELINE_TRACE_FULL, bool)

    def test_live_deep_trace_is_bool(self):
        """LIVE_DEEP_TRACE toggles live test deep tracing."""
        assert isinstance(PolicyConfig.LIVE_DEEP_TRACE, bool)

    def test_regenerate_flags_are_bool(self):
        """REGENERATE flags are booleans."""
        assert isinstance(PolicyConfig.REGENERATE_TEMPLATE_STORE, bool)
        assert isinstance(PolicyConfig.REGENERATE_SCHEMA_GRAPH, bool)
        assert isinstance(PolicyConfig.REGENERATE_SKELETON_CACHE, bool)

    def test_registry_id_patterns(self):
        """Window and case registry ids use strict two-digit id patterns."""
        assert REGISTRY_WINDOW_ID_RE.match("w01")
        assert REGISTRY_WINDOW_ID_RE.match("w99")
        assert REGISTRY_WINDOW_ID_RE.match("w1") is None
        assert REGISTRY_CASE_ID_RE.match("c01")
        assert REGISTRY_CASE_ID_RE.match("c9") is None

    def test_diagnostic_debug_policy_flag(self, monkeypatch):
        """PolicyConfig.DEBUG toggles diagnostic_debug_enabled when force depth is zero."""
        import aetherdialect._config as cfg_mod

        monkeypatch.setattr(cfg_mod, "_DIAGNOSTIC_FORCE_DEPTH", 0, raising=False)
        prev_debug = PolicyConfig.DEBUG
        PolicyConfig.DEBUG = False
        try:
            assert not diagnostic_debug_enabled()
            PolicyConfig.DEBUG = True
            assert diagnostic_debug_enabled()
        finally:
            PolicyConfig.DEBUG = prev_debug

    def test_diagnostic_force_enter_exit(self, monkeypatch):
        """diagnostic_force_enter enables debug until diagnostic_force_exit."""
        import aetherdialect._config as cfg_mod

        monkeypatch.setattr(cfg_mod, "_DIAGNOSTIC_FORCE_DEPTH", 0, raising=False)
        prev_debug = PolicyConfig.DEBUG
        PolicyConfig.DEBUG = False
        try:
            diagnostic_force_enter()
            try:
                assert diagnostic_debug_enabled()
            finally:
                diagnostic_force_exit()
            assert not diagnostic_debug_enabled()
        finally:
            PolicyConfig.DEBUG = prev_debug


class TestEffectiveTimeouts:
    """effective_llm_timeout_ms / effective_explain_timeout_ms resolution."""

    def test_effective_llm_timeout_respects_policy(self, monkeypatch):
        monkeypatch.setattr(PolicyConfig, "LLM_TIMEOUT_MS", 8000)
        assert effective_llm_timeout_ms() == 8000

    def test_effective_llm_timeout_fallback_sixty_seconds(self, monkeypatch):
        monkeypatch.setattr(PolicyConfig, "LLM_TIMEOUT_MS", None)
        assert effective_llm_timeout_ms() == 60_000

    def test_effective_explain_prefers_explicit(self, monkeypatch):
        monkeypatch.setattr(PolicyConfig, "EXPLAIN_TIMEOUT_MS", 5000)
        monkeypatch.setattr(PolicyConfig, "STATEMENT_TIMEOUT_MS", 30_000)
        assert effective_explain_timeout_ms() == 5000

    def test_effective_explain_falls_back_to_statement(self, monkeypatch):
        monkeypatch.setattr(PolicyConfig, "EXPLAIN_TIMEOUT_MS", None)
        monkeypatch.setattr(PolicyConfig, "STATEMENT_TIMEOUT_MS", 42_000)
        assert effective_explain_timeout_ms() == 42_000

    def test_effective_explain_disabled_when_both_off(self, monkeypatch):
        monkeypatch.setattr(PolicyConfig, "EXPLAIN_TIMEOUT_MS", None)
        monkeypatch.setattr(PolicyConfig, "STATEMENT_TIMEOUT_MS", None)
        assert effective_explain_timeout_ms() is None


class TestValidRelativeDateUnits:
    """Tests for VALID_RELATIVE_DATE_UNITS."""

    def test_contains_standard_units(self):
        """Covers day through year granularity."""
        assert "day" in VALID_RELATIVE_DATE_UNITS
        assert "week" in VALID_RELATIVE_DATE_UNITS
        assert "month" in VALID_RELATIVE_DATE_UNITS
        assert "year" in VALID_RELATIVE_DATE_UNITS

    def test_full_membership(self):
        """Matches the closed set used for date_window and date_diff."""
        assert VALID_RELATIVE_DATE_UNITS == frozenset(
            {
                "day",
                "week",
                "month",
                "quarter",
                "half_year",
                "year",
                "hour",
                "minute",
                "second",
            }
        )


class TestValidWindowFunctionsComposed:
    """Tests that VALID_WINDOW_FUNCTIONS is the union of all window families."""

    def test_union_equals_valid_window_functions(self):
        """Ranking, agg, offset, and value window sets compose the allowed window set."""
        assert (
            WINDOW_RANKING_FUNCTIONS | WINDOW_AGG_FUNCTIONS | WINDOW_OFFSET_FUNCTIONS | WINDOW_VALUE_FUNCTIONS
            == VALID_WINDOW_FUNCTIONS
        )


class TestPostgresRuntimeConfig:
    """Tests for PostgresRuntimeConfig."""

    def test_default_host(self):
        """Default host is localhost."""
        assert PostgresRuntimeConfig.HOST == "localhost"

    def test_default_port(self):
        """Default port is 5432."""
        assert PostgresRuntimeConfig.PORT == 5432

    def test_default_schema(self):
        """Default schema is public."""
        assert PostgresRuntimeConfig.SCHEMA == "public"

    def test_db_url_raises_without_password(self):
        """db_url raises ValueError when PASSWORD is not set."""
        orig_pw = PostgresRuntimeConfig.PASSWORD
        orig_db = PostgresRuntimeConfig.DATABASE
        try:
            PostgresRuntimeConfig.PASSWORD = None
            PostgresRuntimeConfig.DATABASE = "testdb"
            with pytest.raises(ValueError, match="password"):
                PostgresRuntimeConfig.db_url()
        finally:
            PostgresRuntimeConfig.PASSWORD = orig_pw
            PostgresRuntimeConfig.DATABASE = orig_db

    def test_db_url_raises_without_database(self):
        """db_url raises ValueError when DATABASE is not set."""
        orig_pw = PostgresRuntimeConfig.PASSWORD
        orig_db = PostgresRuntimeConfig.DATABASE
        try:
            PostgresRuntimeConfig.PASSWORD = "pw"
            PostgresRuntimeConfig.DATABASE = None
            with pytest.raises(ValueError, match="database"):
                PostgresRuntimeConfig.db_url()
        finally:
            PostgresRuntimeConfig.PASSWORD = orig_pw
            PostgresRuntimeConfig.DATABASE = orig_db

    def test_db_url_success(self):
        """db_url returns proper connection string when configured."""
        orig_pw = PostgresRuntimeConfig.PASSWORD
        orig_db = PostgresRuntimeConfig.DATABASE
        try:
            PostgresRuntimeConfig.PASSWORD = "secret"
            PostgresRuntimeConfig.DATABASE = "mydb"
            url = PostgresRuntimeConfig.db_url()
            assert "postgresql+psycopg2://" in url
            assert "secret" in url
            assert "mydb" in url
        finally:
            PostgresRuntimeConfig.PASSWORD = orig_pw
            PostgresRuntimeConfig.DATABASE = orig_db


class TestDatabricksRuntimeConfig:
    """Tests for DatabricksRuntimeConfig."""

    def test_validate_raises_without_catalog(self):
        """Validate raises ValueError when CATALOG is not set."""
        orig_cat = DatabricksRuntimeConfig.CATALOG
        orig_sch = DatabricksRuntimeConfig.SCHEMA
        try:
            DatabricksRuntimeConfig.CATALOG = None
            DatabricksRuntimeConfig.SCHEMA = "myschema"
            with pytest.raises(ValueError, match="catalog"):
                DatabricksRuntimeConfig.validate()
        finally:
            DatabricksRuntimeConfig.CATALOG = orig_cat
            DatabricksRuntimeConfig.SCHEMA = orig_sch

    def test_validate_raises_without_schema(self):
        """Validate raises ValueError when SCHEMA is not set."""
        orig_cat = DatabricksRuntimeConfig.CATALOG
        orig_sch = DatabricksRuntimeConfig.SCHEMA
        try:
            DatabricksRuntimeConfig.CATALOG = "mycat"
            DatabricksRuntimeConfig.SCHEMA = None
            with pytest.raises(ValueError, match="schema"):
                DatabricksRuntimeConfig.validate()
        finally:
            DatabricksRuntimeConfig.CATALOG = orig_cat
            DatabricksRuntimeConfig.SCHEMA = orig_sch

    def test_validate_success(self):
        """Validate succeeds when both fields are set."""
        orig_cat = DatabricksRuntimeConfig.CATALOG
        orig_sch = DatabricksRuntimeConfig.SCHEMA
        try:
            DatabricksRuntimeConfig.CATALOG = "mycat"
            DatabricksRuntimeConfig.SCHEMA = "myschema"
            DatabricksRuntimeConfig.validate()
        finally:
            DatabricksRuntimeConfig.CATALOG = orig_cat
            DatabricksRuntimeConfig.SCHEMA = orig_sch


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_type_postgresql(self):
        """Default engine type is postgresql."""
        assert EngineConfig.TYPE == "postgresql"

    def test_runtime_is_postgres_by_default(self):
        """Default runtime class is PostgresRuntimeConfig."""
        assert EngineConfig.RUNTIME is PostgresRuntimeConfig

    def test_schema_json_path(self):
        """SCHEMA_JSON_PATH is an absolute path under ENGINE_STORAGE_PLACEHOLDER_DIR."""

        assert isinstance(EngineConfig.SCHEMA_JSON_PATH, str)
        assert os.path.isabs(EngineConfig.SCHEMA_JSON_PATH)
        assert os.path.dirname(EngineConfig.SCHEMA_JSON_PATH) == ENGINE_STORAGE_PLACEHOLDER_DIR
        assert os.path.basename(EngineConfig.SCHEMA_JSON_PATH) == "schema_graph.json.gz"

    def test_template_store_dir(self):
        """TEMPLATE_STORE_DIR is an absolute path under ENGINE_STORAGE_PLACEHOLDER_DIR."""

        assert isinstance(EngineConfig.TEMPLATE_STORE_DIR, str)
        assert os.path.isabs(EngineConfig.TEMPLATE_STORE_DIR)
        assert os.path.dirname(EngineConfig.TEMPLATE_STORE_DIR) == ENGINE_STORAGE_PLACEHOLDER_DIR
        assert os.path.basename(EngineConfig.TEMPLATE_STORE_DIR) == "intent_templates"

    def test_skeletons_json_path(self):
        """QSimConfig.SKELETONS_JSON_PATH is an absolute path under ENGINE_STORAGE_PLACEHOLDER_DIR."""

        assert isinstance(QSimConfig.SKELETONS_JSON_PATH, str)
        assert os.path.isabs(QSimConfig.SKELETONS_JSON_PATH)
        assert os.path.dirname(QSimConfig.SKELETONS_JSON_PATH) == ENGINE_STORAGE_PLACEHOLDER_DIR
        assert os.path.basename(QSimConfig.SKELETONS_JSON_PATH) == "qsim_skeletons.json.gz"

    def test_engine_config_not_importable_from_package_root(self) -> None:
        """EngineConfig is internal and must not be importable from the public ``aetherdialect`` namespace."""
        import importlib

        import pytest

        ad_pkg = importlib.import_module("aetherdialect")
        assert "EngineConfig" not in getattr(ad_pkg, "__all__", ())
        with pytest.raises(ImportError):
            from aetherdialect import EngineConfig  # noqa: F401 — intentionally invalid public import


class TestQSimConfig:
    """Tests for QSimConfig."""

    def test_table_ratios_sum_to_one(self):
        """Single/two/three table ratios sum to 1.0."""
        total = QSimConfig.SINGLE_TABLE_RATIO + QSimConfig.TWO_TABLE_RATIO + QSimConfig.THREE_TABLE_RATIO
        assert abs(total - 1.0) < 1e-9

    def test_max_tables_per_intent_positive(self):
        """MAX_TABLES_PER_INTENT is positive."""
        assert QSimConfig.MAX_TABLES_PER_INTENT > 0

    def test_max_filters_per_intent_positive(self):
        """MAX_FILTERS_PER_INTENT is positive."""
        assert QSimConfig.MAX_FILTERS_PER_INTENT > 0

    def test_qsim_questions_pattern_has_version_placeholder(self):
        """QSIM_QUESTIONS_PATTERN contains {version}."""
        assert "{version}" in QSIM_QUESTIONS_PATTERN

    def test_excluded_filter_patterns_are_valid_regex(self):
        """EXCLUDED_FILTER_PATTERNS are valid regex strings."""
        for pat in QSimConfig.EXCLUDED_FILTER_PATTERNS:
            re.compile(pat)


class TestSeedWarmupConfig:
    """Tests for SeedWarmupConfig."""

    def test_max_filters_positive(self):
        """MAX_FILTERS is positive."""
        assert SeedWarmupConfig.MAX_FILTERS > 0

    def test_max_tables_positive(self):
        """MAX_TABLES is positive."""
        assert SeedWarmupConfig.MAX_TABLES > 0

    def test_seed_warmup_bundle_pattern_has_version_placeholder(self):
        """SEED_WARMUP_BUNDLE_PATTERN contains {version}."""
        assert "{version}" in SeedWarmupConfig.SEED_WARMUP_BUNDLE_PATTERN

    def test_report_pattern_has_version_placeholder(self):
        """SEED_WARMUP_REPORT_PATTERN contains {version}."""
        assert "{version}" in SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN

    def test_realism_dropped_registered_as_failure_code(self) -> None:
        """Question realism gate uses a single canonical string in ``SEED_WARMUP_FAILURE_CODES``."""

        assert SEED_FAILURE_CODE_REALISM_DROPPED in SEED_WARMUP_FAILURE_CODES

    def test_max_expr_comparisons_positive(self):
        """MAX_EXPR_COMPARISONS is positive."""
        assert SeedWarmupConfig.MAX_EXPR_COMPARISONS > 0

    def test_max_having_conditions_positive(self):
        """MAX_HAVING_CONDITIONS is positive."""
        assert SeedWarmupConfig.MAX_HAVING_CONDITIONS > 0

    def test_max_seed_questions_fixed_cap(self):
        """Seed warmup seed loading uses a fixed internal cap (documented in API_REFERENCE)."""
        assert SeedWarmupConfig.MAX_SEED_QUESTIONS == 500


class TestSeedWarmupFailureCodeFromValidateSqlError:
    """``seed_warmup_failure_code_from_validate_sql_error`` classification."""

    def test_not_select(self):
        assert seed_warmup_failure_code_from_validate_sql_error("not_select") == "ast_validate_unsupported_construct"

    def test_sql_structure_cte(self):
        msg = "SQL structure error: syntax error near WITH clause"
        assert seed_warmup_failure_code_from_validate_sql_error(msg) == "ast_validate_cte_error"

    def test_classified_explain_prefixes(self):
        assert (
            seed_warmup_failure_code_from_validate_sql_error("[explain_schema] relation x does not exist")
            == "explain_schema"
        )
        assert (
            seed_warmup_failure_code_from_validate_sql_error("[explain_transient] statement timeout")
            == "explain_transient"
        )

    def test_failure_category_execution_buckets(self):
        assert (
            seed_warmup_failure_code_from_validate_sql_error(
                "x",
                failure_category="execution_schema_error",
            )
            == "explain_schema"
        )
        assert (
            seed_warmup_failure_code_from_validate_sql_error(
                "x",
                failure_category="execution_timeout",
            )
            == "explain_transient"
        )


class TestIntentSchema:
    """Tests for INTENT_SCHEMA."""

    def test_is_dict(self):
        """INTENT_SCHEMA is a dict."""
        assert isinstance(INTENT_SCHEMA, dict)

    def test_required_includes_tables(self):
        """INTENT_SCHEMA required includes tables."""
        assert "tables" in INTENT_SCHEMA["required"]

    def test_properties_include_filters_param_and_having_param(self):
        """INTENT_SCHEMA properties include filters_param and having_param."""
        props = INTENT_SCHEMA["properties"]
        assert "filters_param" in props
        assert "having_param" in props

    def test_properties_include_cte_steps(self):
        """INTENT_SCHEMA properties include cte_steps."""
        assert "cte_steps" in INTENT_SCHEMA["properties"]

    def test_cte_steps_items_require_cte_name(self):
        """INTENT_SCHEMA cte_steps items require cte_name, select_cols, output_columns."""
        cte = INTENT_SCHEMA["properties"]["cte_steps"]["items"]
        assert "cte_name" in cte["required"]
        assert "select_cols" in cte["required"]
        assert "output_columns" in cte["required"]


class TestTableColPattern:
    """Tests for TABLE_COL_PATTERN regex."""

    def test_matches_table_dot_column(self):
        """TABLE_COL_PATTERN matches table.column."""
        m = TABLE_COL_PATTERN.match("orders.id")
        assert m is not None
        assert m.group(1) == "orders"
        assert m.group(2) == "id"

    def test_matches_schema_table_dot_column(self):
        """TABLE_COL_PATTERN matches first two word parts (schema.table)."""
        m = TABLE_COL_PATTERN.search("public.orders.id")
        assert m is not None
        assert m.group(1) == "public"
        assert m.group(2) == "orders"

    def test_no_match_without_dot(self):
        """TABLE_COL_PATTERN does not match identifier without dot."""
        assert TABLE_COL_PATTERN.match("tablename") is None


class TestScalarFunctionSets:
    """Tests for SCALAR_FUNCTIONS_* constants."""

    def test_string_subset_of_valid(self):
        """SCALAR_FUNCTIONS_STRING is subset of VALID_SCALAR_FUNCTIONS."""
        assert SCALAR_FUNCTIONS_STRING.issubset(VALID_SCALAR_FUNCTIONS)

    def test_numeric_subset_of_valid(self):
        """
        SCALAR_FUNCTIONS_NUMERIC is subset of VALID_SCALAR_FUNCTIONS.

        The same set is what validation allows wrapping an aggregate (for example ``ROUND(SUM(...))``).
        """
        assert SCALAR_FUNCTIONS_NUMERIC.issubset(VALID_SCALAR_FUNCTIONS)

    def test_temporal_subset_of_valid(self):
        """SCALAR_FUNCTIONS_TEMPORAL is subset of VALID_SCALAR_FUNCTIONS."""
        assert SCALAR_FUNCTIONS_TEMPORAL.issubset(VALID_SCALAR_FUNCTIONS)

    def test_leading_arg_subset_of_valid(self):
        """SCALAR_FUNCTIONS_LEADING_ARG is subset of VALID_SCALAR_FUNCTIONS."""
        assert SCALAR_FUNCTIONS_LEADING_ARG.issubset(VALID_SCALAR_FUNCTIONS)


class TestFilterOpSets:
    """Tests for per-role filter operator sets."""

    def test_boolean_ops_subset_of_valid(self):
        """BOOLEAN_FILTER_OPS is subset of VALID_FILTER_OPS."""
        assert BOOLEAN_FILTER_OPS.issubset(VALID_FILTER_OPS)

    def test_categorical_ops_subset_of_valid(self):
        """CATEGORICAL_FILTER_OPS is subset of VALID_FILTER_OPS."""
        assert CATEGORICAL_FILTER_OPS.issubset(VALID_FILTER_OPS)

    def test_numeric_categorical_ops_subset_of_valid(self):
        """NUMERIC_CATEGORICAL_FILTER_OPS is subset of VALID_FILTER_OPS."""
        assert NUMERIC_CATEGORICAL_FILTER_OPS.issubset(VALID_FILTER_OPS)

    def test_numeric_ops_subset_of_valid(self):
        """NUMERIC_FILTER_OPS is subset of VALID_FILTER_OPS."""
        assert NUMERIC_FILTER_OPS.issubset(VALID_FILTER_OPS)

    def test_boolean_no_like(self):
        """BOOLEAN_FILTER_OPS does not include like."""
        assert "like" not in BOOLEAN_FILTER_OPS


class TestMiscConstants:
    """Tests for remaining module-level constant sets."""

    def test_valid_expected_rows(self):
        """VALID_EXPECTED_ROWS has three values."""
        assert VALID_EXPECTED_ROWS == {"one", "few", "many"}

    def test_valid_filter_value_types(self):
        """VALID_FILTER_VALUE_TYPES has expected members."""
        for vt in ("categorical", "numeric", "temporal", "boolean"):
            assert vt in VALID_FILTER_VALUE_TYPES

    def test_valid_having_value_types(self):
        """VALID_HAVING_VALUE_TYPES has expected members."""
        assert VALID_HAVING_VALUE_TYPES == {"number", "integer"}

    def test_value_type_normalization_keys_map_to_valid(self):
        """All VALUE_TYPE_NORMALIZATION values are in VALID_VALUE_TYPES."""
        for v in VALUE_TYPE_NORMALIZATION.values():
            assert v in VALID_VALUE_TYPES

    def test_column_type_to_value_type_values_valid(self):
        """All COLUMN_TYPE_TO_VALUE_TYPE values are in VALID_VALUE_TYPES."""
        for v in COLUMN_TYPE_TO_VALUE_TYPE.values():
            assert v in VALID_VALUE_TYPES

    def test_aggregation_allowed_keys_match_valid(self):
        """AGGREGATION_ALLOWED_COLUMN_TYPES keys match VALID_AGGREGATION_FUNCTIONS."""
        assert set(AGGREGATION_ALLOWED_COLUMN_TYPES.keys()) == VALID_AGGREGATION_FUNCTIONS

    def test_valid_having_ops_subset_of_filter_ops(self):
        """VALID_HAVING_OPS is subset of VALID_FILTER_OPS."""
        assert VALID_HAVING_OPS.issubset(VALID_FILTER_OPS)


class TestPromptNeutrality:
    """Regression checks for neutral vocabulary constants."""

    def test_vocab_guidance_has_no_polarity_example_tokens(self):
        """QUESTION_NORMALIZE_VOCABULARY_GUIDANCE avoids enumerated polarity pairs."""

        from aetherdialect._utils import QUESTION_NORMALIZE_VOCABULARY_GUIDANCE

        low = QUESTION_NORMALIZE_VOCABULARY_GUIDANCE.lower()
        assert "positive:" not in low
        assert "negative:" not in low

    def test_irregular_plurals_exclude_demographic_tokens(self):
        """IRREGULAR_PLURALS_MAP lists morphological irregulars only."""

        from aetherdialect._config import IRREGULAR_PLURALS_MAP

        assert "women" not in IRREGULAR_PLURALS_MAP
        assert "men" not in IRREGULAR_PLURALS_MAP

    def test_negation_tokens_not_in_stopwords(self):
        """Negators must remain available to the question normaliser."""

        from aetherdialect._config import PolicyConfig, STOPWORDS_GRAMMATICAL_PARTICLES

        negators = ("not", "without", "except", "no", "never", "neither", "nor")
        for t in negators:
            assert t not in STOPWORDS_GRAMMATICAL_PARTICLES
            assert t not in PolicyConfig.STOPWORDS


class TestPolicyRestartReasons:
    """Tests for semantic fresh-restart reason gating."""

    def test_semantic_restart_reasons_frozenset(self):
        """PolicyConfig.SEMANTIC_RESTART_REASONS is the expected frozenset."""
        reasons = PolicyConfig.SEMANTIC_RESTART_REASONS
        assert isinstance(reasons, frozenset)
        assert reasons == frozenset({"semantic_oscillation", "semantic_max_rounds"})


class TestPlannerMandatoryConventions:
    """Planner NL conventions include anti-spurious clause guidance."""

    def test_groupby_orderby_limit_rule_present(self) -> None:
        from aetherdialect._intent_process import PLANNER_NL_CONVENTIONS

        mandatory = list(PLANNER_NL_CONVENTIONS["mandatory"])
        assert any(
            "group_by" in m and "order_by" in m and "limit" in m and "empty" in m.lower() for m in mandatory
        )


class TestConfigFileFullCoverage:
    """TOML ``config_file`` flattening covers every documented key."""

    def test_config_file_full_coverage(self, tmp_path) -> None:
        from aetherdialect._main_execution import _load_config_file

        path = tmp_path / "full.toml"
        path.write_text(
            "\n".join(
                (
                    '[openai]',
                    'api_key = "oak"',
                    'base_url = "https://example-openai/v1"',
                    "",
                    "[azure_openai]",
                    'endpoint = "https://ex.azure.com"',
                    'api_key = "aak"',
                    'api_version = "2024-01-01"',
                    'base_url = "https://ex.azure.com/base"',
                    "",
                    "[azure_openai.deployments]",
                    'light = "al"',
                    'medium = "am"',
                    'heavy = "ah"',
                    "",
                    "[postgresql]",
                    'host = "h"',
                    "port = 5433",
                    'database = "d"',
                    'schema = "sch"',
                    'user = "u"',
                    'password = "pw"',
                    "",
                    "[databricks]",
                    'host = "dh"',
                    'http_path = "/sql"',
                    'access_token = "tok"',
                    'catalog = "cat"',
                    'schema = "ds"',
                    'cluster_id = "cid"',
                    "",
                    "[engine]",
                    'selected = "postgresql"',
                    "",
                    "[llm]",
                    'provider = "openai"',
                    "",
                    "[execution]",
                    "max_query_cost_rows = 100",
                    "max_query_cost_bytes = 200",
                    "statement_timeout_ms = 300",
                    "llm_timeout_ms = 400",
                    "profile_timeout_ms = 500",
                    "explain_timeout_ms = 600",
                ),
            ),
            encoding="utf-8",
        )
        got, _claimed = _load_config_file(str(path))
        expected = {
            "OPENAI_API_KEY": "oak",
            "OPENAI_BASE_URL": "https://example-openai/v1",
            "AZURE_OPENAI_ENDPOINT": "https://ex.azure.com",
            "AZURE_OPENAI_API_KEY": "aak",
            "AZURE_OPENAI_API_VERSION": "2024-01-01",
            "AZURE_OPENAI_BASE_URL": "https://ex.azure.com/base",
            "AZURE_OPENAI_DEPLOYMENT_LIGHT": "al",
            "AZURE_OPENAI_DEPLOYMENT_MEDIUM": "am",
            "AZURE_OPENAI_DEPLOYMENT_HEAVY": "ah",
            "POSTGRES_HOST": "h",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "d",
            "POSTGRES_SCHEMA": "sch",
            "POSTGRES_USER": "u",
            "POSTGRES_PASSWORD": "pw",
            "DATABRICKS_HOST": "dh",
            "DATABRICKS_HTTP_PATH": "/sql",
            "DATABRICKS_ACCESS_TOKEN": "tok",
            "DATABRICKS_CATALOG": "cat",
            "DATABRICKS_SCHEMA": "ds",
            "DATABRICKS_CLUSTER_ID": "cid",
            "AETHERDIALECT_ENGINE": "postgresql",
            "AETHERDIALECT_LLM_PROVIDER": "openai",
            "AETHERDIALECT_MAX_QUERY_COST_ROWS": "100",
            "AETHERDIALECT_MAX_QUERY_COST_BYTES": "200",
            "AETHERDIALECT_STATEMENT_TIMEOUT_MS": "300",
            "AETHERDIALECT_LLM_TIMEOUT_MS": "400",
            "AETHERDIALECT_PROFILE_TIMEOUT_MS": "500",
            "AETHERDIALECT_EXPLAIN_TIMEOUT_MS": "600",
        }
        assert got == expected
