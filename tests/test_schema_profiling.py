"""Tests for schema_profiling module."""

import json
from typing import Any

from aetherdialect._config import (
    BOOLEAN_TRUTH_PATTERN_MAP,
    EngineConfig,
    PolicyConfig,
)
from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    FKEdge,
    RoleOwner,
    SchemaGraph,
    TableMetadata,
    TableRole,
)
from aetherdialect._contracts_core import FilterParam, NormalizedExpr, RuntimeIntent
from aetherdialect._core_utils import stable_json
from aetherdialect._dialect import (
    DatabricksDialect,
    unity_structural_constraints_index_from_information_schema_rows,
)
from aetherdialect._schema_profiling import (
    SCHEMA_CONSISTENCY_REFINE_SYSTEM,
    SCHEMA_NOTES_REFINE_SYSTEM,
    _array_element_type_from_data_type,
    _build_column_profile_for_llm,
    _coerce_antonym_pair_column,
    _coerce_zero_one_column,
    _cursor_rows_as_dicts,
    _enrich_fk_column_descriptions,
    _extract_column_block_from_create,
    _extract_fk_definition,
    _extract_pk_columns,
    _has_boolean_like_values,
    _infer_column_role,
    _llm_classify_schema,
    _parse_column_name_and_sql_type,
    _parse_columns_and_constraints,
    _parse_partition_columns_from_create_stmt,
    _parse_sql_file_fallback,
    _parse_sql_file_via_llm,
    _parse_unity_catalog_constraints,
    _partition_column_names_from_create_ddl,
    _split_by_top_level_comma,
    _strip_leading_articles,
    _table_metadata_dict_from_ddl_parts,
    _validate_column_classification,
    apply_boolean_coercion_pass,
    apply_column_roles_llm,
    assign_column_ops,
    collect_profiling_topk_values,
    parse_sql_file,
)


class TestBooleanTruthPatternMap:
    """Tests for ``BOOLEAN_TRUTH_PATTERN_MAP``."""

    def test_contains_zero_one(self):
        """Verify 0/1 pair maps to affirmative ``1``."""
        k = frozenset({"0", "1"})
        assert k in BOOLEAN_TRUTH_PATTERN_MAP
        assert BOOLEAN_TRUTH_PATTERN_MAP[k] == "1"

    def test_contains_true_false(self):
        """Verify true/false pair maps to ``true``."""
        k = frozenset({"true", "false"})
        assert k in BOOLEAN_TRUTH_PATTERN_MAP
        assert BOOLEAN_TRUTH_PATTERN_MAP[k] == "true"

    def test_contains_yes_no(self):
        """Verify yes/no pair maps to ``yes``."""
        k = frozenset({"yes", "no"})
        assert k in BOOLEAN_TRUTH_PATTERN_MAP
        assert BOOLEAN_TRUTH_PATTERN_MAP[k] == "yes"

    def test_contains_active_inactive(self):
        k = frozenset({"active", "inactive"})
        assert k in BOOLEAN_TRUTH_PATTERN_MAP
        assert BOOLEAN_TRUTH_PATTERN_MAP[k] == "active"

    def test_contains_enabled_disabled(self):
        k = frozenset({"enabled", "disabled"})
        assert k in BOOLEAN_TRUTH_PATTERN_MAP
        assert BOOLEAN_TRUTH_PATTERN_MAP[k] == "enabled"

    def test_contains_pass_fail_family(self):
        """Verify pass/fail literal families are present."""
        assert frozenset({"pass", "fail"}) in BOOLEAN_TRUTH_PATTERN_MAP
        assert frozenset({"present", "absent"}) in BOOLEAN_TRUTH_PATTERN_MAP


class TestHasBooleanLikeValues:
    """Tests for _has_boolean_like_values."""

    def test_zero_one_integer(self):
        """Detect integer 0/1 as boolean-like."""
        col = ColumnMetadata(name="active", data_type="integer", distinct_count=2, top_k_values=[0, 1])
        assert _has_boolean_like_values(col)[0] is True

    def test_true_false_strings(self):
        """Detect true/false strings as boolean-like."""
        col = ColumnMetadata(
            name="flag",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["True", "False"],
        )
        assert _has_boolean_like_values(col)[0] is True

    def test_yes_no_mixed_case(self):
        """Detect YES/NO as boolean-like regardless of case."""
        col = ColumnMetadata(
            name="approved",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["YES", "NO"],
        )
        assert _has_boolean_like_values(col)[0] is True

    def test_pass_fail_mixed_case(self):
        """Detect Pass/Fail as boolean-like regardless of case."""
        col = ColumnMetadata(
            name="result",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["Pass", "Fail"],
        )
        assert _has_boolean_like_values(col)[0] is True

    def test_three_distinct_not_boolean(self):
        """Reject column with distinct_count != 2."""
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            distinct_count=3,
            top_k_values=["yes", "no", "maybe"],
        )
        assert _has_boolean_like_values(col)[0] is False

    def test_none_top_k(self):
        """Reject column with no top_k_values."""
        col = ColumnMetadata(name="flag", data_type="integer", distinct_count=2, top_k_values=None)
        assert _has_boolean_like_values(col)[0] is False

    def test_one_top_k_value(self):
        """Reject column with only one top_k value."""
        col = ColumnMetadata(name="flag", data_type="integer", distinct_count=2, top_k_values=["1"])
        assert _has_boolean_like_values(col)[0] is False

    def test_non_pattern_pair(self):
        """Reject column with values not matching any known pattern."""
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["pending", "done"],
        )
        assert _has_boolean_like_values(col)[0] is False


class TestInferColumnRole:
    """Tests for infer_column_role."""

    def test_timestamp_column(self):
        """Infer TEMPORAL role for timestamp columns."""
        col = ColumnMetadata(name="created_at", data_type="timestamp", value_type="date")
        assert _infer_column_role(col) == ColumnRole.TEMPORAL

    def test_boolean_value_type(self):
        """Infer BOOLEAN role from value_type=boolean."""
        col = ColumnMetadata(name="is_active", data_type="boolean", value_type="boolean")
        assert _infer_column_role(col) == ColumnRole.BOOLEAN

    def test_primary_key(self):
        """Infer IDENTIFIER role for primary key columns."""
        col = ColumnMetadata(name="id", data_type="integer", value_type="integer", is_primary_key=True)
        assert _infer_column_role(col) == ColumnRole.IDENTIFIER

    def test_foreign_key(self):
        """Infer IDENTIFIER role for foreign key columns."""
        col = ColumnMetadata(
            name="customer_id",
            data_type="integer",
            value_type="integer",
            is_foreign_key=True,
        )
        assert _infer_column_role(col) == ColumnRole.IDENTIFIER

    def test_boolean_like_values(self):
        """Infer BOOLEAN role from boolean-like top_k_values."""
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            value_type="string",
            distinct_count=2,
            top_k_values=["0", "1"],
        )
        assert _infer_column_role(col) == ColumnRole.BOOLEAN

    def test_temporal(self):
        """Infer TEMPORAL role for date value_type."""
        col = ColumnMetadata(name="order_date", data_type="date", value_type="date")
        assert _infer_column_role(col) == ColumnRole.TEMPORAL

    def test_free_text_high_uniqueness(self):
        """Infer FREE_TEXT role for high distinct_ratio."""
        col = ColumnMetadata(
            name="description",
            data_type="varchar",
            value_type="string",
            distinct_ratio=PolicyConfig.IDENTIFIER_MIN_UNIQUENESS,
        )
        assert _infer_column_role(col) == ColumnRole.FREE_TEXT

    def test_categorical_string(self):
        """Infer CATEGORICAL role for low-cardinality string column."""
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            value_type="string",
            distinct_count=5,
            distinct_ratio=0.01,
        )
        assert _infer_column_role(col) == ColumnRole.CATEGORICAL

    def test_numeric_categorical(self):
        """Infer NUMERIC_CATEGORICAL role for low-cardinality integer column."""
        col = ColumnMetadata(
            name="rating",
            data_type="integer",
            value_type="integer",
            distinct_count=5,
            distinct_ratio=0.01,
        )
        assert _infer_column_role(col) == ColumnRole.NUMERIC_CATEGORICAL

    def test_numeric_measure(self):
        """Infer NUMERIC_MEASURE role for high-cardinality numeric column."""
        col = ColumnMetadata(
            name="amount",
            data_type="numeric",
            value_type="number",
            distinct_count=500,
            distinct_ratio=0.5,
        )
        assert _infer_column_role(col) == ColumnRole.NUMERIC_MEASURE


class TestParseColumnNameAndSqlType:
    """Tests for _parse_column_name_and_sql_type."""

    def test_simple_name_type(self):
        assert _parse_column_name_and_sql_type("id INTEGER") == ("id", "INTEGER")

    def test_stops_at_not_null(self):
        assert _parse_column_name_and_sql_type("name VARCHAR(100) NOT NULL") == (
            "name",
            "VARCHAR(100)",
        )

    def test_returns_none_for_short_line(self):
        assert _parse_column_name_and_sql_type("id") is None

    def test_strips_quotes(self):
        assert _parse_column_name_and_sql_type('"user_id" BIGINT') == (
            "user_id",
            "BIGINT",
        )


class TestParseColumnsAndConstraints:
    """Tests for _parse_columns_and_constraints."""

    def test_simple_columns(self):
        """Parse simple column definitions."""
        block = "id INTEGER PRIMARY KEY, name VARCHAR, age INTEGER"
        columns, types, pks, fks, _unique, nulls = _parse_columns_and_constraints(block)
        assert columns == ["id", "name", "age"]
        assert types == ["INTEGER", "VARCHAR", "INTEGER"]
        assert "id" in pks
        assert nulls == [False, True, True]

    def test_separate_pk_constraint(self):
        """Parse PRIMARY KEY as separate constraint."""
        block = "id INTEGER, name VARCHAR, PRIMARY KEY (id)"
        columns, types, pks, fks, _unique, nulls = _parse_columns_and_constraints(block)
        assert columns == ["id", "name"]
        assert "id" in pks
        assert nulls == [False, True]

    def test_foreign_key_constraint(self):
        """Parse FOREIGN KEY constraint."""
        block = "id INTEGER, customer_id INTEGER, FOREIGN KEY (customer_id) REFERENCES customers(id)"
        columns, types, pks, fks, _unique, nulls = _parse_columns_and_constraints(block)
        assert columns == ["id", "customer_id"]
        assert len(fks) == 1
        assert fks[0]["src_cols"] == ["customer_id"]
        assert fks[0]["dst_table"] == "customers"
        assert fks[0]["dst_cols"] == ["id"]
        assert nulls == [True, True]

    def test_empty_block(self):
        """Parse empty column block."""
        columns, types, pks, fks, _unique, nulls = _parse_columns_and_constraints("")
        assert columns == []
        assert fks == []
        assert nulls == []

    def test_quoted_column_names(self):
        """Parse quoted column names."""
        block = '"order_id" INTEGER PRIMARY KEY, "total" NUMERIC'
        columns, types, pks, fks, _unique, nulls = _parse_columns_and_constraints(block)
        assert "order_id" in columns
        assert "order_id" in pks
        assert nulls == [False, True]

    def test_numeric_precision_scale_across_tokens(self):
        """NUMERIC(4, 2) with a split after the comma still forms one type."""
        block = "rental_rate NUMERIC(4, 2) NOT NULL"
        columns, types, pks, fks, _, nulls = _parse_columns_and_constraints(block)
        assert columns == ["rental_rate"]
        assert "4" in types[0] and "2" in types[0]
        assert nulls == [False]


class TestTableMetadataDictFromDdlParts:
    """``column_is_nullable`` is persisted on DDL-derived table metadata dicts."""

    def test_not_null_flags_parallel_columns(self):
        block = "user_id INTEGER NOT NULL, email VARCHAR, PRIMARY KEY (user_id)"
        full = "CREATE TABLE users ( user_id INTEGER NOT NULL, email VARCHAR, PRIMARY KEY (user_id) )"
        tmeta = _table_metadata_dict_from_ddl_parts("users", block, full)
        assert tmeta["column_names_original"] == ["user_id", "email"]
        assert tmeta["column_is_nullable"] == [False, True]


class TestExtractPkColumns:
    """Tests for _extract_pk_columns."""

    def test_single_pk(self):
        """Extract single primary key column."""
        result = _extract_pk_columns("PRIMARY KEY (id)")
        assert result == ["id"]

    def test_composite_pk(self):
        """Extract composite primary key columns."""
        result = _extract_pk_columns("PRIMARY KEY (order_id, product_id)")
        assert result == ["order_id", "product_id"]

    def test_no_match(self):
        """Return empty list for non-PK line."""
        result = _extract_pk_columns("id INTEGER NOT NULL")
        assert result == []


class TestExtractFkDefinition:
    """Tests for _extract_fk_definition."""

    def test_simple_fk(self):
        """Extract simple foreign key definition."""
        result = _extract_fk_definition("FOREIGN KEY (customer_id) REFERENCES customers(id)")
        assert result is not None
        assert result["src_cols"] == ["customer_id"]
        assert result["dst_table"] == "customers"
        assert result["dst_cols"] == ["id"]

    def test_no_match(self):
        """Return None for non-FK line."""
        result = _extract_fk_definition("id INTEGER PRIMARY KEY")
        assert result is None


class TestValidateColumnClassification:
    """Tests for _validate_column_classification."""

    def test_numeric_measure_on_string(self):
        """Hard error for NUMERIC_MEASURE on non-numeric column."""
        col = ColumnMetadata(name="name", data_type="varchar", value_type="string")
        hard, soft = _validate_column_classification(col, ColumnRole.NUMERIC_MEASURE.value)
        assert len(hard) > 0
        assert "NUMERIC_MEASURE" in hard[0]

    def test_temporal_on_integer(self):
        """Hard error for TEMPORAL on non-date column."""
        col = ColumnMetadata(name="count", data_type="integer", value_type="integer")
        hard, soft = _validate_column_classification(col, ColumnRole.TEMPORAL.value)
        assert len(hard) > 0

    def test_boolean_high_cardinality(self):
        """Hard error for BOOLEAN with distinct_count > 2."""
        col = ColumnMetadata(name="status", data_type="varchar", value_type="string", distinct_count=5)
        hard, soft = _validate_column_classification(col, ColumnRole.BOOLEAN.value)
        assert len(hard) > 0
        assert "distinct_count" in hard[0]

    def test_categorical_high_cardinality_warning(self):
        """Soft warning for CATEGORICAL with very high cardinality."""
        col = ColumnMetadata(name="code", data_type="varchar", value_type="string", distinct_count=2000)
        hard, soft = _validate_column_classification(col, ColumnRole.CATEGORICAL.value)
        assert len(soft) > 0
        assert "high cardinality" in soft[0]

    def test_identifier_non_pk_fk_warning(self):
        """Soft warning for IDENTIFIER on non-PK/FK column."""
        col = ColumnMetadata(
            name="code",
            data_type="varchar",
            value_type="string",
            is_primary_key=False,
            is_foreign_key=False,
        )
        hard, soft = _validate_column_classification(col, ColumnRole.IDENTIFIER.value)
        assert len(soft) > 0
        assert "non-PK/FK" in soft[0]

    def test_valid_classification_no_errors(self):
        """No errors for valid NUMERIC_MEASURE on numeric column."""
        col = ColumnMetadata(name="amount", data_type="numeric", value_type="number", distinct_count=100)
        hard, soft = _validate_column_classification(col, ColumnRole.NUMERIC_MEASURE.value)
        assert len(hard) == 0

    def test_numeric_categorical_on_string_hard_error(self):
        """Hard error for NUMERIC_CATEGORICAL on non-numeric column."""
        col = ColumnMetadata(name="code", data_type="varchar", value_type="string")
        hard, soft = _validate_column_classification(col, ColumnRole.NUMERIC_CATEGORICAL.value)
        assert len(hard) > 0
        assert "NUMERIC_CATEGORICAL" in hard[0]


class TestBuildColumnProfileForLlm:
    """Tests for _build_column_profile_for_llm."""

    def test_basic_fields(self):
        """Profile contains name, data_type, is_primary_key, is_foreign_key."""
        col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
        result = _build_column_profile_for_llm(col)
        assert result["name"] == "id"
        assert result["data_type"] == "integer"
        assert result["is_primary_key"] is True
        assert result["is_foreign_key"] is False

    def test_hints_with_stats(self):
        """Profile includes profile_hints when stats present."""
        col = ColumnMetadata(
            name="x",
            data_type="int",
            distinct_count=10,
            distinct_ratio=0.5,
            null_ratio=0.1,
        )
        result = _build_column_profile_for_llm(col)
        assert "profile_hints" in result
        assert result["profile_hints"]["distinct_count"] == 10
        assert result["profile_hints"]["distinct_ratio"] == 0.5
        assert result["profile_hints"]["null_ratio"] == 0.1

    def test_default_stats_included(self):
        """Profile includes default stats in hints."""
        col = ColumnMetadata(name="x", data_type="int")
        result = _build_column_profile_for_llm(col)
        assert result["name"] == "x"
        assert result["data_type"] == "int"

    def test_hints_include_distinct_count(self):
        """Profile hints include distinct_count when set."""
        col = ColumnMetadata(name="x", data_type="int", distinct_count=5)
        result = _build_column_profile_for_llm(col)
        assert "profile_hints" in result
        assert result["profile_hints"]["distinct_count"] == 5

    def test_distinct_ratio_rounded(self):
        """Distinct ratio rounded to 3 decimal places."""
        col = ColumnMetadata(name="x", data_type="int", distinct_ratio=0.123456789)
        result = _build_column_profile_for_llm(col)
        assert result["profile_hints"]["distinct_ratio"] == 0.123

    def test_fk_target_in_profile(self):
        """Profile includes is_foreign_key and reflects fk_target."""
        col = ColumnMetadata(
            name="customer_id",
            data_type="integer",
            is_foreign_key=True,
            fk_target=("customers", "customer_id"),
        )
        result = _build_column_profile_for_llm(col)
        assert result["is_foreign_key"] is True
        assert result["is_primary_key"] is False


class TestColumnProfileTopValues:
    """Policy-gated top_values in profile_hints for schema classification prompts."""

    def test_default_zero_omits_top_values(self) -> None:
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            value_type="string",
            distinct_ratio=0.02,
            top_k_values=["a", "b", "c"],
        )
        result = _build_column_profile_for_llm(col)
        hints = result.get("profile_hints") or {}
        assert "top_values" not in hints

    def test_positive_sample_size_includes_top_values_when_qualified(self) -> None:
        PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 5
        try:
            col = ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                distinct_ratio=0.02,
                top_k_values=["a", "b", "c", "d", "e", "f"],
            )
            result = _build_column_profile_for_llm(col)
            tv = result["profile_hints"]["top_values"]
            assert len(tv) <= 5
            assert tv[0] == "a"
        finally:
            PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 0

    def test_high_distinct_ratio_omits_top_values(self) -> None:
        PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 5
        try:
            col = ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                distinct_ratio=0.5,
                top_k_values=["a", "b"],
            )
            result = _build_column_profile_for_llm(col)
            assert "top_values" not in result.get("profile_hints", {})
        finally:
            PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 0

    def test_pii_sensitivity_omits_top_values(self) -> None:
        PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 5
        try:
            col = ColumnMetadata(
                name="email",
                data_type="varchar",
                value_type="string",
                distinct_ratio=0.02,
                sensitivity="pii",
                top_k_values=["a@b.com"],
            )
            result = _build_column_profile_for_llm(col)
            assert "top_values" not in result.get("profile_hints", {})
        finally:
            PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 0

    def test_restricted_sensitivity_omits_top_values(self) -> None:
        PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 5
        try:
            col = ColumnMetadata(
                name="secret",
                data_type="varchar",
                value_type="string",
                distinct_ratio=0.02,
                sensitivity="restricted",
                top_k_values=["x"],
            )
            result = _build_column_profile_for_llm(col)
            assert "top_values" not in result.get("profile_hints", {})
        finally:
            PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 0

    def test_numeric_column_omits_top_values(self) -> None:
        PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 5
        try:
            col = ColumnMetadata(
                name="amt",
                data_type="numeric",
                value_type="number",
                distinct_ratio=0.02,
                top_k_values=["1", "2"],
            )
            result = _build_column_profile_for_llm(col)
            assert "top_values" not in result.get("profile_hints", {})
        finally:
            PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES = 0


class TestArrayElementTypeFromDataType:
    """Tests for _array_element_type_from_data_type."""

    def test_empty_returns_false(self):
        assert _array_element_type_from_data_type("") == (False, None)
        assert _array_element_type_from_data_type("   ") == (False, None)

    def test_bare_array(self):
        is_arr, elt = _array_element_type_from_data_type("ARRAY")
        assert is_arr is True
        assert elt == "string"

    def test_array_angle_brackets(self):
        is_arr, elt = _array_element_type_from_data_type("ARRAY<INT>")
        assert is_arr is True
        assert elt == "int"

    def test_postgres_array_suffix(self):
        is_arr, elt = _array_element_type_from_data_type("TEXT[]")
        assert is_arr is True
        assert elt == "text"

    def test_array_paren_form(self):
        is_arr, elt = _array_element_type_from_data_type("ARRAY( STRING )")
        assert is_arr is True
        assert elt == "string"

    def test_array_bracket_heuristic(self):
        is_arr, elt = _array_element_type_from_data_type("some_array[1]")
        assert is_arr is True
        assert elt == "string"

    def test_plain_varchar_not_array(self):
        assert _array_element_type_from_data_type("VARCHAR(50)") == (False, None)


class TestSplitByTopLevelComma:
    """Tests for _split_by_top_level_comma."""

    def test_nested_parens(self):
        s = "a INT, b NUMERIC(4, 2), c STRING"
        parts = _split_by_top_level_comma(s)
        assert len(parts) == 3
        assert "NUMERIC(4, 2)" in parts[1]

    def test_empty_and_trailing_comma(self):
        assert _split_by_top_level_comma("") == []
        assert _split_by_top_level_comma("x INT") == ["x INT"]


class TestStripLeadingArticles:
    """Tests for _strip_leading_articles."""

    def test_strips_a_an(self):
        assert _strip_leading_articles("a yes") == "yes"
        assert _strip_leading_articles("an active") == "active"


class TestCoerceZeroOneColumn:
    """Tests for _coerce_zero_one_column."""

    def test_coerces_float_and_int_strings(self):
        col = ColumnMetadata(
            name="flag",
            data_type="numeric",
            distinct_count=2,
            top_k_values=["0", "1.0"],
            role=ColumnRole.NUMERIC_CATEGORICAL.value,
        )
        assert _coerce_zero_one_column(col) is True
        assert col.role == ColumnRole.BOOLEAN.value

    def test_rejects_three_values(self):
        col = ColumnMetadata(name="x", data_type="int", distinct_count=2, top_k_values=["0", "2"])
        assert _coerce_zero_one_column(col) is False


class TestCoerceAntonymPairColumn:
    """Tests for _coerce_antonym_pair_column."""

    def test_prefix_inactive_active(self):
        col = ColumnMetadata(
            name="state",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["active", "inactive"],
            role=ColumnRole.CATEGORICAL.value,
        )
        assert _coerce_antonym_pair_column(col) is True
        assert col.role == ColumnRole.BOOLEAN.value

    def test_prefix_dis_honest(self):
        col = ColumnMetadata(
            name="trait",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["honest", "dishonest"],
            role=ColumnRole.CATEGORICAL.value,
        )
        assert _coerce_antonym_pair_column(col) is True

    def test_prefix_de_activate(self):
        col = ColumnMetadata(
            name="state",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["activate", "deactivate"],
            role=ColumnRole.CATEGORICAL.value,
        )
        assert _coerce_antonym_pair_column(col) is True

    def test_suffix_less_care(self):
        col = ColumnMetadata(
            name="mode",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["care", "careless"],
            role=ColumnRole.CATEGORICAL.value,
        )
        assert _coerce_antonym_pair_column(col) is True

    def test_rejects_short_stems_before_prefix_rules(self):
        col = ColumnMetadata(
            name="x",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["ab", "disab"],
            role=ColumnRole.CATEGORICAL.value,
        )
        assert _coerce_antonym_pair_column(col) is False


class TestLlmClassifySchemaRefinePasses:
    """Second LLM classification pass is always invoked (notes vs consistency)."""

    def test_runs_consistency_refine_when_no_notes(self, monkeypatch):
        col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
        table = TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": table})
        raw = json.dumps(
            {
                "t": {
                    "table_role": "dimension",
                    "description": "d",
                    "columns": {"id": {"role": "identifier", "hint": "h", "sensitivity": None}},
                },
            }
        )
        calls: list[tuple[str, str]] = []

        def fake_llm(
            system: str,
            user: str,
            max_retries: int = 3,
            timeout: Any = None,
            task: str = "default",
        ) -> str:
            calls.append((system, task))
            return raw

        monkeypatch.setattr("aetherdialect._schema_profiling.llm_chat", fake_llm)
        out = _llm_classify_schema(sg, None)
        assert len(calls) == 2
        assert calls[0][1] == "schema_base"
        assert calls[1][0] == SCHEMA_CONSISTENCY_REFINE_SYSTEM
        assert calls[1][1] == "schema"
        assert "t" in out

    def test_runs_notes_refine_when_notes_present(self, monkeypatch):
        col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
        table = TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": table})
        raw = json.dumps(
            {
                "t": {
                    "table_role": "dimension",
                    "description": "d",
                    "columns": {"id": {"role": "identifier", "hint": "h", "sensitivity": None}},
                },
            }
        )
        calls: list[str] = []

        def fake_llm(
            system: str,
            user: str,
            max_retries: int = 3,
            timeout: Any = None,
            task: str = "default",
        ) -> str:
            calls.append(system)
            return raw

        monkeypatch.setattr("aetherdialect._schema_profiling.llm_chat", fake_llm)
        _llm_classify_schema(sg, "domain notes here")
        assert len(calls) == 2
        assert calls[1] == SCHEMA_NOTES_REFINE_SYSTEM


class TestApplyColumnRolesLlmPostOverrides:
    """Tests for post-LLM role overrides in apply_column_roles_llm."""

    def test_date_type_overrides_to_temporal(self, monkeypatch):
        col = ColumnMetadata(
            name="created_at",
            data_type="timestamptz",
            value_type="date",
            role=ColumnRole.CATEGORICAL.value,
        )
        table = TableMetadata(name="events", columns={"created_at": col}, primary_key=[], foreign_keys=[])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="h", tables={"events": table})

        def fake_classify(schema: SchemaGraph, notes_content: str | None = None):
            return {
                "events": (
                    TableRole.DIMENSION.value,
                    "t",
                    {"created_at": (ColumnRole.CATEGORICAL.value, "when", None)},
                )
            }

        monkeypatch.setattr("aetherdialect._schema_profiling._llm_classify_schema", fake_classify)
        apply_column_roles_llm(sg)
        assert col.role == ColumnRole.TEMPORAL.value

    def test_audit_role_not_overridden_to_temporal(self, monkeypatch):
        col = ColumnMetadata(
            name="updated_at",
            data_type="timestamptz",
            value_type="date",
            role=ColumnRole.CATEGORICAL.value,
        )
        table = TableMetadata(name="events", columns={"updated_at": col}, primary_key=[], foreign_keys=[])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="h", tables={"events": table})

        def fake_classify(schema: SchemaGraph, notes_content: str | None = None):
            return {
                "events": (
                    TableRole.DIMENSION.value,
                    "t",
                    {"updated_at": (ColumnRole.AUDIT.value, "audit stamp", None)},
                )
            }

        monkeypatch.setattr("aetherdialect._schema_profiling._llm_classify_schema", fake_classify)
        apply_column_roles_llm(sg)
        assert col.role == ColumnRole.AUDIT.value

    def test_free_text_low_cardinality_not_forced_to_categorical(self, monkeypatch):
        col = ColumnMetadata(
            name="notes",
            data_type="text",
            value_type="string",
            distinct_count=3,
            role=ColumnRole.CATEGORICAL.value,
        )
        table = TableMetadata(name="t", columns={"notes": col}, primary_key=[], foreign_keys=[])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="h", tables={"t": table})

        def fake_classify(schema: SchemaGraph, notes_content: str | None = None):
            return {
                "t": (
                    TableRole.FACT.value,
                    "t",
                    {"notes": (ColumnRole.FREE_TEXT.value, "n", None)},
                )
            }

        monkeypatch.setattr("aetherdialect._schema_profiling._llm_classify_schema", fake_classify)
        apply_column_roles_llm(sg)
        assert col.role == ColumnRole.FREE_TEXT.value


class TestApplyBooleanCoercionPass:
    """Tests for apply_boolean_coercion_pass."""

    def test_skips_pk_columns(self):
        col = ColumnMetadata(
            name="id",
            data_type="integer",
            is_primary_key=True,
            distinct_count=2,
            top_k_values=["0", "1"],
            role=ColumnRole.IDENTIFIER.value,
        )
        t = TableMetadata(name="t", columns={"id": col}, foreign_keys=[], primary_key=["id"])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        apply_boolean_coercion_pass(sg)
        assert col.role == ColumnRole.IDENTIFIER.value

    def test_promotes_boolean_like_non_fk(self):
        col = ColumnMetadata(
            name="active",
            data_type="varchar",
            value_type="string",
            distinct_count=2,
            top_k_values=["true", "false"],
            role=ColumnRole.CATEGORICAL.value,
        )
        t = TableMetadata(name="t", columns={"active": col}, foreign_keys=[], primary_key=[])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        apply_boolean_coercion_pass(sg)
        assert col.role == ColumnRole.BOOLEAN.value


class TestCursorRowsAsDicts:
    """Tests for _cursor_rows_as_dicts."""

    def test_no_description_returns_empty(self):
        class C:
            description = None

            def fetchall(self):
                return [("a",)]

        assert _cursor_rows_as_dicts(C()) == []

    def test_maps_rows(self):
        class C:
            description = [("cnt",), ("x",)]

            def fetchall(self):
                return [(1, "y")]

        assert _cursor_rows_as_dicts(C()) == [{"cnt": 1, "x": "y"}]


class TestEnrichFkColumnDescriptions:
    """Tests for _enrich_fk_column_descriptions."""

    def test_appends_join_hint(self):
        user = TableMetadata(
            name="users",
            columns={
                "id": ColumnMetadata(name="id", data_type="int", is_primary_key=True),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role="categorical",
                ),
            },
            foreign_keys=[],
            primary_key=["id"],
        )
        oid = ColumnMetadata(
            name="user_id",
            data_type="int",
            is_foreign_key=True,
            fk_target=("users", "id"),
            description="",
        )
        orders = TableMetadata(
            name="orders",
            columns={"user_id": oid},
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["user_id"],
                    dst_table="users",
                    dst_cols=["id"],
                )
            ],
            primary_key=[],
        )
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"orders": orders, "users": user},
        )
        _enrich_fk_column_descriptions(sg)
        assert "join users" in (oid.description or "").lower()
        assert "name" in (oid.description or "").lower()


class TestParseSqlFile:
    """Tests for parse_sql_file (file IO + deterministic parser path)."""

    def test_matches_fallback_for_same_content(self, tmp_path):
        sql = "CREATE TABLE pf_file (id INTEGER PRIMARY KEY);"
        p = tmp_path / "schema.sql"
        p.write_text(sql, encoding="utf-8-sig")
        assert parse_sql_file(p) == _parse_sql_file_fallback(sql)

    def test_utf8_sig_bom_stripped(self, tmp_path):
        sql = "CREATE TABLE bom_t (id INT PRIMARY KEY);"
        p = tmp_path / "b.sql"
        p.write_bytes(b"\xef\xbb\xbf" + sql.encode("utf-8"))
        assert parse_sql_file(p) == _parse_sql_file_fallback(sql)


class TestParseSqlFileConditionalLlm:
    """Conditional DDL LLM fallback when deterministic parsers return nothing."""

    def test_skips_llm_when_reflected_schema_has_foreign_keys(self, monkeypatch, tmp_path):
        """Reflection already carries FK edges; empty deterministic parse must not call LLM."""

        path = tmp_path / "odd.sql"
        path.write_text("-- unparsed\n", encoding="utf-8")

        bid = ColumnMetadata(name="bid", data_type="integer")
        pid = ColumnMetadata(name="id", data_type="integer")
        edge = FKEdge(src_table="a", src_cols=["bid"], dst_table="b", dst_cols=["id"])
        ta = TableMetadata(name="a", columns={"bid": bid}, foreign_keys=[edge], primary_key=[])
        tb = TableMetadata(name="b", columns={"id": pid}, foreign_keys=[], primary_key=["id"])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"a": ta, "b": tb})

        monkeypatch.setattr("aetherdialect._schema_profiling._parse_sql_file_fallback", lambda _c: {})
        monkeypatch.setattr(
            "aetherdialect._schema_profiling._parse_sql_file_regex_reflect",
            lambda _c: {},
        )

        calls: list[tuple[Any, ...]] = []

        def _capture_llm(*args: Any, **kwargs: Any) -> str:
            calls.append(args)
            return '{"tables": {}}'

        monkeypatch.setattr("aetherdialect._schema_profiling.llm_chat", _capture_llm)

        out = parse_sql_file(path, reflected_schema=sg)
        assert out == {}
        assert calls == []

    def test_calls_llm_when_reflected_schema_has_no_foreign_keys(self, monkeypatch, tmp_path):
        """Empty deterministic parse and no reflected FKs still invokes DDL LLM."""

        path = tmp_path / "odd.sql"
        path.write_text("-- unparsed\n", encoding="utf-8")

        cid = ColumnMetadata(name="id", data_type="integer")
        t = TableMetadata(name="t", columns={"id": cid}, foreign_keys=[], primary_key=["id"])
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})

        monkeypatch.setattr("aetherdialect._schema_profiling._parse_sql_file_fallback", lambda _c: {})
        monkeypatch.setattr(
            "aetherdialect._schema_profiling._parse_sql_file_regex_reflect",
            lambda _c: {},
        )

        calls: list[Any] = []

        def _capture_llm(*args: Any, **kwargs: Any) -> str:
            calls.append(1)
            return '{"tables": {}}'

        monkeypatch.setattr("aetherdialect._schema_profiling.llm_chat", _capture_llm)

        out = parse_sql_file(path, reflected_schema=sg)
        assert out == {}
        assert len(calls) == 1


class TestParseSqlFileViaLlmNullability:
    """DDL LLM path maps ``column_not_null`` / ``column_unique`` into canonical metadata."""

    def test_parse_sql_file_via_llm_extracts_not_null_and_unique(self, monkeypatch):
        ddl = "CREATE TABLE u (a INT PRIMARY KEY, b VARCHAR(20) UNIQUE NOT NULL, c INT);"
        payload = {
            "tables": {
                "u": {
                    "table_name_original": "u",
                    "column_names_original": ["a", "b", "c"],
                    "column_types": ["INT", "VARCHAR(20)", "INT"],
                    "column_not_null": [True, True, False],
                    "column_unique": [True, True, False],
                    "primary_keys": ["a"],
                    "foreign_keys": [],
                }
            }
        }
        monkeypatch.setattr(
            "aetherdialect._schema_profiling.llm_chat",
            lambda *a, **k: stable_json(payload),
        )
        out = _parse_sql_file_via_llm(ddl)
        u = out["u"]
        assert u["column_is_nullable"] == [False, False, True]
        assert u["unique_columns"] == ["a", "b"]
        assert "column_not_null" not in u
        assert "column_unique" not in u
        assert u["partition_columns"] == []

    def test_parse_sql_file_via_llm_handles_missing_nullability_keys(self, monkeypatch):
        ddl = "CREATE TABLE v (x INT);"
        payload = {
            "tables": {
                "v": {
                    "table_name_original": "v",
                    "column_names_original": ["x"],
                    "column_types": ["INT"],
                    "primary_keys": [],
                    "foreign_keys": [],
                }
            }
        }
        monkeypatch.setattr(
            "aetherdialect._schema_profiling.llm_chat",
            lambda *a, **k: stable_json(payload),
        )
        out = _parse_sql_file_via_llm(ddl)
        v = out["v"]
        assert v["column_is_nullable"] == [True]
        assert v["unique_columns"] == []


class TestAssignColumnOps:
    """Tests for assign_column_ops."""

    def test_audit_gets_typed_ops(self):
        """AUDIT temporal columns get conservative filter ops, count-only aggregations, and numeric HAVING ops."""
        col = ColumnMetadata(
            name="created_at",
            data_type="timestamp",
            value_type="date",
            role=ColumnRole.AUDIT.value,
            distinct_count=10,
        )
        t = TableMetadata(name="t", columns={"created_at": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "between" in col.valid_filter_ops
        assert "is null" in col.valid_filter_ops
        assert col.valid_aggregations == ["count"]
        assert col.valid_having_ops == ["=", "!=", "<", "<=", ">", ">="]

    def test_identifier_pk(self):
        """PK columns get identifier-level ops."""
        col = ColumnMetadata(
            name="id",
            data_type="integer",
            value_type="integer",
            is_primary_key=True,
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=100,
        )
        t = TableMetadata(name="t", columns={"id": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "=" in col.valid_filter_ops
        assert "between" in col.valid_filter_ops
        assert "count" in col.valid_aggregations
        assert "sum" not in col.valid_aggregations

    def test_categorical_string(self):
        """CATEGORICAL string columns include LIKE operators."""
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=5,
        )
        t = TableMetadata(name="t", columns={"status": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "like" in col.valid_filter_ops
        assert "ilike" in col.valid_filter_ops

    def test_numeric_measure(self):
        """NUMERIC_MEASURE gets sum/avg aggregations."""
        col = ColumnMetadata(
            name="amount",
            data_type="numeric",
            value_type="number",
            role=ColumnRole.NUMERIC_MEASURE.value,
            distinct_count=100,
        )
        t = TableMetadata(name="t", columns={"amount": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "sum" in col.valid_aggregations
        assert "avg" in col.valid_aggregations
        assert "between" in col.valid_filter_ops

    def test_temporal(self):
        """TEMPORAL columns get date-appropriate ops."""
        col = ColumnMetadata(
            name="order_date",
            data_type="date",
            value_type="date",
            role=ColumnRole.TEMPORAL.value,
            distinct_count=100,
        )
        t = TableMetadata(name="t", columns={"order_date": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "between" in col.valid_filter_ops
        assert "min" in col.valid_aggregations
        assert "sum" not in col.valid_aggregations

    def test_boolean(self):
        """BOOLEAN columns get equality-only filter ops."""
        col = ColumnMetadata(
            name="is_active",
            data_type="boolean",
            value_type="boolean",
            role=ColumnRole.BOOLEAN.value,
            distinct_count=2,
        )
        t = TableMetadata(name="t", columns={"is_active": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "=" in col.valid_filter_ops
        assert "between" not in col.valid_filter_ops

    def test_free_text(self):
        """FREE_TEXT columns retain pattern ops despite is_filterable=False."""
        col = ColumnMetadata(
            name="description",
            data_type="text",
            value_type="string",
            role=ColumnRole.FREE_TEXT.value,
            distinct_count=1000,
        )
        t = TableMetadata(name="t", columns={"description": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "like" in col.valid_filter_ops
        assert "ilike" in col.valid_filter_ops
        assert "not like" in col.valid_filter_ops
        assert "not ilike" in col.valid_filter_ops
        assert "is null" in col.valid_filter_ops
        assert "is not null" in col.valid_filter_ops
        assert "=" not in col.valid_filter_ops
        assert col.valid_aggregations == ["count"]

    def test_numeric_categorical(self):
        """NUMERIC_CATEGORICAL gets appropriate ops."""
        col = ColumnMetadata(
            name="rating",
            data_type="integer",
            value_type="integer",
            role=ColumnRole.NUMERIC_CATEGORICAL.value,
            distinct_count=5,
        )
        t = TableMetadata(name="t", columns={"rating": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "between" in col.valid_filter_ops
        assert "like" not in col.valid_filter_ops

    def test_non_filterable_empty_ops(self):
        """Non-filterable columns get empty filter ops regardless of role."""
        col = ColumnMetadata(
            name="user_password",
            data_type="varchar",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=100,
        )
        t = TableMetadata(name="t", columns={"x": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert col.valid_filter_ops == []

    def test_string_only_ops_removed_from_numeric(self):
        """String-only ops removed from non-string columns."""
        col = ColumnMetadata(
            name="amount",
            data_type="numeric",
            value_type="number",
            role=ColumnRole.NUMERIC_MEASURE.value,
            distinct_count=100,
        )
        t = TableMetadata(name="t", columns={"amount": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "like" not in col.valid_filter_ops
        assert "ilike" not in col.valid_filter_ops

    def test_numeric_aggs_removed_from_string(self):
        """Numeric-only aggregations removed from string columns."""
        col = ColumnMetadata(
            name="status",
            data_type="varchar",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=5,
        )
        t = TableMetadata(name="t", columns={"status": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "sum" not in col.valid_aggregations
        assert "avg" not in col.valid_aggregations

    def test_bare_array_data_type_sets_element_type_and_contains(self):
        """Catalog-style ``ARRAY`` without brackets still enables ``contains`` and element_type."""

        col = ColumnMetadata(
            name="tags",
            data_type="ARRAY",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=10,
        )
        t = TableMetadata(name="t", columns={"tags": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert col.element_type == "string"
        assert "contains" in col.valid_filter_ops

    def test_array_bracket_type_puts_contains_first(self):
        """Typed array columns prepend ``contains`` ahead of role-based operators."""
        col = ColumnMetadata(
            name="ids",
            data_type="ARRAY<INT>",
            value_type="integer",
            role=ColumnRole.NUMERIC_CATEGORICAL.value,
            distinct_count=5,
        )
        t = TableMetadata(name="t", columns={"ids": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert col.valid_filter_ops[0] == "contains"

    def test_fallback_role_branch_when_filterable(self):
        """Non-enum role string hits the default branch and keeps basic equality ops when filterable."""
        col = ColumnMetadata(
            name="x",
            data_type="integer",
            value_type="integer",
            role="unclassified",
            distinct_count=10,
            is_filterable_override=True,
        )
        t = TableMetadata(name="t", columns={"x": col}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        assign_column_ops(schema)
        assert "=" in col.valid_filter_ops
        assert "is null" in col.valid_filter_ops
        assert col.valid_aggregations == ["count"]


class TestParseUnityCatalogConstraints:
    """Tests for _parse_unity_catalog_constraints."""

    def test_pk_constraint(self):
        """Extract PK from CONSTRAINT clause."""
        stmt = "CREATE TABLE t (id INT, CONSTRAINT pk_t PRIMARY KEY (id))"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert pks == ["id"]
        assert fks == []

    def test_fk_constraint(self):
        """Extract FK from CONSTRAINT clause."""
        stmt = "CREATE TABLE t (id INT, user_id INT, CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id))"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert len(fks) == 1
        assert fks[0]["src_cols"] == ["user_id"]
        assert fks[0]["dst_table"] == "users"
        assert fks[0]["dst_cols"] == ["id"]

    def test_composite_pk(self):
        """Extract composite PK columns."""
        stmt = "CONSTRAINT pk_t PRIMARY KEY (a, b)"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert pks == ["a", "b"]

    def test_no_constraints(self):
        """Return empty lists when no constraints present."""
        stmt = "CREATE TABLE t (id INT, name VARCHAR)"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert pks == []
        assert fks == []

    def test_multiple_fks(self):
        """Extract multiple FK constraints."""
        stmt = """
        CONSTRAINT fk1 FOREIGN KEY (a_id) REFERENCES a(id),
        CONSTRAINT fk2 FOREIGN KEY (b_id) REFERENCES b(id)
        """
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert len(fks) == 2

    def test_anonymous_fk_without_constraint_name(self):
        """Parse FOREIGN KEY without a leading CONSTRAINT name clause."""

        stmt = "CREATE TABLE t (user_id INT, FOREIGN KEY (user_id) REFERENCES users(id))"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert len(fks) == 1
        assert fks[0]["src_cols"] == ["user_id"]
        assert fks[0]["dst_table"] == "users"
        assert fks[0]["dst_cols"] == ["id"]

    def test_qualified_reference_table_trailing_name(self):
        """Reduce qualified REFERENCES targets to the trailing table identifier."""

        stmt = "CREATE TABLE t (a_id INT, FOREIGN KEY (a_id) REFERENCES my_cat.my_sch.parent(id))"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert len(fks) == 1
        assert fks[0]["dst_table"] == "parent"

    def test_backticked_qualified_reference_table(self):
        """Parse REFERENCES targets that use backticked dotted identifiers."""

        stmt = "CREATE TABLE t (x INT, FOREIGN KEY (x) REFERENCES `c`.`s`.`p`(`id`))"
        pks, fks, _unique = _parse_unity_catalog_constraints(stmt)
        assert len(fks) == 1
        assert fks[0]["dst_table"] == "p"
        assert fks[0]["dst_cols"] == ["id"]


class TestUnityStructuralConstraintsIndexFromRows:
    """Tests for ``unity_structural_constraints_index_from_information_schema_rows``."""

    def test_pk_fk_and_single_column_unique(self):
        """Join table_constraints, key_column_usage, and referential_constraints into FKEdge and PK lists."""

        sch = "clinical"
        tc_rows = [
            {
                "constraint_schema": sch,
                "constraint_name": "pk_users",
                "table_name": "users",
                "constraint_type": "PRIMARY KEY",
            },
            {
                "constraint_schema": sch,
                "constraint_name": "uq_users_email",
                "table_name": "users",
                "constraint_type": "UNIQUE",
            },
            {
                "constraint_schema": sch,
                "constraint_name": "fk_orders_user",
                "table_name": "orders",
                "constraint_type": "FOREIGN KEY",
            },
        ]
        kcu_rows = [
            {
                "constraint_schema": sch,
                "constraint_name": "pk_users",
                "table_name": "users",
                "column_name": "id",
                "ordinal_position": 1,
            },
            {
                "constraint_schema": sch,
                "constraint_name": "uq_users_email",
                "table_name": "users",
                "column_name": "email",
                "ordinal_position": 1,
            },
            {
                "constraint_schema": sch,
                "constraint_name": "fk_orders_user",
                "table_name": "orders",
                "column_name": "user_id",
                "ordinal_position": 1,
            },
        ]
        rc_rows = [
            {
                "constraint_schema": sch,
                "constraint_name": "fk_orders_user",
                "unique_constraint_schema": sch,
                "unique_constraint_name": "pk_users",
            },
        ]
        idx = unity_structural_constraints_index_from_information_schema_rows(tc_rows, kcu_rows, rc_rows)
        ub = idx.tables["users"]
        assert ub.primary_keys == ["id"]
        assert "email" in ub.unique_columns
        ob = idx.tables["orders"]
        assert len(ob.foreign_keys) == 1
        e = ob.foreign_keys[0]
        assert e.src_table == "orders"
        assert e.src_cols == ["user_id"]
        assert e.dst_table == "users"
        assert e.dst_cols == ["id"]


class TestApplyColumnRolesLlmBooleanStringValueType:
    """Regression tests for boolean role with string ``value_type``."""

    def test_no_llm_mismatch_notify_for_boolean_on_two_valued_string(self, monkeypatch):
        """Widened compat prevents user-visible mismatch lines for yes/no string columns."""

        captured: list[str] = []

        def capture_notify(msg: str) -> None:
            captured.append(msg)

        monkeypatch.setattr("aetherdialect._schema_profiling._core_utils.notify", capture_notify)
        col = ColumnMetadata(
            name="blood_culture_before_antibiotics",
            data_type="string",
            value_type="string",
            distinct_count=2,
            top_k_values=["yes", "no"],
            role=ColumnRole.BOOLEAN.value,
            role_owner=RoleOwner.LLM,
        )
        table = TableMetadata(
            name="clinical_data",
            columns={"blood_culture_before_antibiotics": col},
            primary_key=[],
            foreign_keys=[],
        )
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="h",
            tables={"clinical_data": table},
        )

        def fake_classify(schema: SchemaGraph, notes_content: str | None = None):
            return {
                "clinical_data": (
                    TableRole.FACT.value,
                    "clinical",
                    {
                        "blood_culture_before_antibiotics": (
                            ColumnRole.BOOLEAN.value,
                            "flag",
                            None,
                        ),
                    },
                ),
            }

        monkeypatch.setattr("aetherdialect._schema_profiling._llm_classify_schema", fake_classify)
        apply_column_roles_llm(sg)
        assert col.role == ColumnRole.BOOLEAN.value
        assert not any("LLM role/value_type mismatch" in m for m in captured)


class TestParseSqlFileFallback:
    """Tests for _parse_sql_file_fallback."""

    def test_simple_create_table(self):
        """Parse simple CREATE TABLE statement."""
        sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR);"
        result = _parse_sql_file_fallback(sql)
        assert "users" in result
        assert "id" in result["users"]["column_names_original"]
        assert "name" in result["users"]["column_names_original"]
        assert "id" in result["users"]["primary_keys"]

    def test_multiple_tables(self):
        """Parse multiple CREATE TABLE statements."""
        sql = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER);
        """
        result = _parse_sql_file_fallback(sql)
        assert len(result) == 2
        assert "users" in result
        assert "orders" in result

    def test_empty_sql(self):
        """Empty SQL returns empty dict."""
        result = _parse_sql_file_fallback("")
        assert result == {}

    def test_non_create_statements(self):
        """Non-CREATE statements ignored."""
        sql = "SELECT 1; INSERT INTO t VALUES (1); DROP TABLE t;"
        result = _parse_sql_file_fallback(sql)
        assert result == {}

    def test_columns_preserved(self):
        """Parsed columns preserved in result."""
        sql = "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER);"
        result = _parse_sql_file_fallback(sql)
        assert "orders" in result
        assert "id" in result["orders"]["column_names_original"]
        assert "user_id" in result["orders"]["column_names_original"]

    def test_numeric_with_comma_in_type(self):
        """NUMERIC(4,2) and NUMERIC(5,2) parsed without splitting on internal comma."""
        sql = """
        CREATE TABLE film (
            film_id INTEGER PRIMARY KEY,
            rental_rate NUMERIC(4,2) DEFAULT 4.99 NOT NULL,
            replacement_cost NUMERIC(5,2) DEFAULT 19.99 NOT NULL
        );
        """
        result = _parse_sql_file_fallback(sql)
        assert "film" in result
        cols = result["film"]["column_names_original"]
        types = result["film"]["column_types"]
        assert "rental_rate" in cols
        assert "replacement_cost" in cols
        assert "2)" not in cols
        assert cols.count("rental_rate") == 1
        assert cols.count("replacement_cost") == 1
        has_42 = any("4,2" in t.replace(" ", "") for t in types)
        has_52 = any("5,2" in t.replace(" ", "") for t in types)
        assert has_42 and has_52
        assert any("NUMERIC" in t.upper() or "DECIMAL" in t.upper() for t in types)

    def test_partitioned_by_single_column(self):
        """Parse PARTITIONED BY with single column."""
        sql = "CREATE TABLE events (id BIGINT, dt DATE) PARTITIONED BY (dt)"
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback(sql)
        finally:
            EngineConfig.TYPE = orig
        table_key = next((k for k in result if result[k].get("partition_columns") == ["dt"]), None)
        assert table_key is not None
        assert result[table_key]["partition_columns"] == ["dt"]

    def test_partitioned_by_multiple_columns(self):
        """Parse PARTITIONED BY with multiple columns."""
        sql = "CREATE TABLE logs (id BIGINT, region STRING, dt DATE) PARTITIONED BY (region, dt)"
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback(sql)
        finally:
            EngineConfig.TYPE = orig
        assert "logs" in result
        assert result["logs"].get("partition_columns") == ["region", "dt"]

    def test_parses_language_table_reserved_keyword(self):
        """Parse CREATE TABLE language when language is a reserved keyword."""
        sql = """
        CREATE TABLE language (
            language_id INTEGER NOT NULL PRIMARY KEY,
            name CHAR(20) NOT NULL
        );
        """
        result = _parse_sql_file_fallback(sql)
        assert "language" in result
        assert "language_id" in result["language"]["column_names_original"]
        assert "name" in result["language"]["column_names_original"]

    def test_parses_create_table_spark(self):
        """Parse CREATE TABLE with Spark dialect."""
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback("CREATE TABLE users (id INT, name STRING);")
            assert "users" in result
            assert result["users"]["column_names_original"] == ["id", "name"]
        finally:
            EngineConfig.TYPE = orig

    def test_parses_reserved_keyword_table_databricks(self):
        """Parse CREATE TABLE when table name is reserved keyword (Databricks path)."""
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback("CREATE TABLE language (language_id INT PRIMARY KEY);")
            assert "language" in result
        finally:
            EngineConfig.TYPE = orig

    def test_returns_empty_for_select(self):
        """Return empty dict for SELECT statement."""
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback("SELECT 1;")
            assert result == {}
        finally:
            EngineConfig.TYPE = orig

    def test_postgres_alter_add_foreign_key(self):
        """Merge ALTER TABLE ADD CONSTRAINT FOREIGN KEY after CREATE (pglast path)."""
        sql = """
        CREATE TABLE parent (id integer);
        CREATE TABLE child (id integer, parent_id integer);
        ALTER TABLE child ADD CONSTRAINT child_fk FOREIGN KEY (parent_id) REFERENCES parent(id);
        """
        result = _parse_sql_file_fallback(sql)
        assert "parent" in result and "child" in result
        fks = result["child"]["foreign_keys"]
        assert len(fks) == 1
        assert fks[0] == {
            "src_cols": ["parent_id"],
            "dst_table": "parent",
            "dst_cols": ["id"],
        }

    def test_postgres_alter_add_primary_key(self):
        """Merge ALTER TABLE ADD CONSTRAINT PRIMARY KEY (pglast path)."""
        sql = """
        CREATE TABLE t (a integer, b integer);
        ALTER TABLE t ADD CONSTRAINT t_pk PRIMARY KEY (a, b);
        """
        result = _parse_sql_file_fallback(sql)
        assert result["t"]["primary_keys"] == ["a", "b"]

    def test_postgres_alter_add_column(self):
        """Merge ALTER TABLE ADD COLUMN (pglast path)."""
        sql = """
        CREATE TABLE t (a integer);
        ALTER TABLE t ADD COLUMN b text;
        """
        result = _parse_sql_file_fallback(sql)
        assert result["t"]["column_names_original"] == ["a", "b"]
        assert "text" in result["t"]["column_types"][1].lower()

    def test_postgres_alter_add_two_columns_one_statement(self):
        """Multiple ADD COLUMN in one ALTER each get distinct types (pglast RawStream)."""
        sql = """
        CREATE TABLE t (a integer);
        ALTER TABLE t ADD COLUMN b integer, ADD COLUMN c text;
        """
        result = _parse_sql_file_fallback(sql)
        assert result["t"]["column_names_original"] == ["a", "b", "c"]
        assert "int" in result["t"]["column_types"][1].lower()
        assert "text" in result["t"]["column_types"][2].lower()

    def test_postgres_alter_unknown_table_skipped(self):
        """ALTER on a table not created earlier does not add entries."""
        sql = """
        CREATE TABLE t (a integer);
        ALTER TABLE missing ADD COLUMN x integer;
        """
        result = _parse_sql_file_fallback(sql)
        assert set(result.keys()) == {"t"}
        assert "x" not in result["t"]["column_names_original"]

    def test_databricks_alter_add_foreign_key(self):
        """Merge ALTER ADD CONSTRAINT FOREIGN KEY (sqlglot Spark path)."""
        sql = """
        CREATE TABLE parent (id INT);
        CREATE TABLE child (id INT, parent_id INT);
        ALTER TABLE child ADD CONSTRAINT child_fk FOREIGN KEY (parent_id) REFERENCES parent(id);
        """
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback(sql)
        finally:
            EngineConfig.TYPE = orig
        fks = result["child"]["foreign_keys"]
        assert len(fks) == 1
        assert fks[0]["dst_table"] == "parent"
        assert fks[0]["src_cols"] == ["parent_id"]
        assert fks[0]["dst_cols"] == ["id"]

    def test_databricks_alter_add_columns_plural(self):
        """Merge ALTER TABLE ADD COLUMNS with multiple definitions (sqlglot)."""
        sql = """
        CREATE TABLE t (a INT);
        ALTER TABLE t ADD COLUMNS (b STRING, c INT);
        """
        orig = EngineConfig.TYPE
        try:
            EngineConfig.TYPE = "databricks"
            result = _parse_sql_file_fallback(sql)
        finally:
            EngineConfig.TYPE = orig
        assert result["t"]["column_names_original"] == ["a", "b", "c"]


class TestPartitionColumnNamesFromCreateDdl:
    """Tests for _partition_column_names_from_create_ddl."""

    def test_postgres_partition_by_range(self):
        """PostgreSQL PARTITION BY RANGE lists partition key columns."""
        stmt = "CREATE TABLE m (id INT, d DATE) PARTITION BY RANGE (d);"
        assert _partition_column_names_from_create_ddl(stmt) == ["d"]

    def test_postgres_partition_by_list(self):
        """PostgreSQL PARTITION BY LIST is parsed when Spark-style PARTITIONED BY is absent."""
        stmt = "CREATE TABLE t (id INT, region VARCHAR) PARTITION BY LIST (region);"
        assert _partition_column_names_from_create_ddl(stmt) == ["region"]


class TestParsePartitionColumnsFromCreateStmt:
    """Tests for _parse_partition_columns_from_create_stmt."""

    def test_single_partition_column(self):
        """Extract single partition column."""
        stmt = "CREATE TABLE t (id INT, dt DATE) PARTITIONED BY (dt)"
        assert _parse_partition_columns_from_create_stmt(stmt) == ["dt"]

    def test_multiple_partition_columns(self):
        """Extract multiple partition columns."""
        stmt = "CREATE TABLE t (id INT, r STRING, dt DATE) PARTITIONED BY (r, dt)"
        assert _parse_partition_columns_from_create_stmt(stmt) == ["r", "dt"]

    def test_backtick_quoted_columns(self):
        """Strip backticks from partition column names."""
        stmt = "CREATE TABLE t (id INT) PARTITIONED BY (`dt`)"
        assert _parse_partition_columns_from_create_stmt(stmt) == ["dt"]

    def test_case_insensitive(self):
        """PARTITIONED BY matches case-insensitively."""
        stmt = "CREATE TABLE t (id INT) partitioned by (dt)"
        assert _parse_partition_columns_from_create_stmt(stmt) == ["dt"]

    def test_no_partitioned_by(self):
        """Return empty list when no PARTITIONED BY."""
        stmt = "CREATE TABLE t (id INT, name VARCHAR)"
        assert _parse_partition_columns_from_create_stmt(stmt) == []

    def test_empty_statement(self):
        """Return empty list for empty string."""
        assert _parse_partition_columns_from_create_stmt("") == []


class TestExtractColumnBlockFromCreate:
    """Tests for _extract_column_block_from_create."""

    def test_extracts_block(self):
        """Extract column definition block from sqlglot Create."""
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one("CREATE TABLE t (id INT, name VARCHAR);", dialect="spark")
        assert isinstance(parsed, exp.Create)
        block = _extract_column_block_from_create(parsed)
        assert "id" in block
        assert "name" in block

    def test_handles_simple_schema(self):
        """Extract block from CREATE TABLE with single column."""
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one("CREATE TABLE t (id INT PRIMARY KEY);", dialect="spark")
        assert isinstance(parsed, exp.Create)
        block = _extract_column_block_from_create(parsed)
        assert "id" in block


class TestHasBooleanLikeValuesEdgeCases:
    """Edge case tests for _has_boolean_like_values."""

    def test_empty_top_k(self):
        """Empty top_k_values list returns False."""
        col = ColumnMetadata(name="flag", data_type="integer", distinct_count=2, top_k_values=[])
        assert _has_boolean_like_values(col)[0] is False

    def test_on_off_values(self):
        """Detect on/off as boolean-like."""
        col = ColumnMetadata(
            name="switch",
            data_type="varchar",
            distinct_count=2,
            top_k_values=["on", "off"],
        )
        assert _has_boolean_like_values(col)[0] is True

    def test_y_n_values(self):
        """Detect y/n as boolean-like."""
        col = ColumnMetadata(name="active", data_type="char", distinct_count=2, top_k_values=["Y", "N"])
        assert _has_boolean_like_values(col)[0] is True


class TestInferColumnRoleEdgeCases:
    """Edge case tests for infer_column_role."""

    def test_identifier_high_uniqueness_numeric(self):
        """High uniqueness numeric column with _id suffix gets IDENTIFIER."""
        col = ColumnMetadata(
            name="record_id",
            data_type="integer",
            value_type="integer",
            distinct_ratio=0.99,
        )
        result = _infer_column_role(col)
        assert result in (
            ColumnRole.IDENTIFIER,
            ColumnRole.FREE_TEXT,
            ColumnRole.NUMERIC_MEASURE,
        )

    def test_no_stats_string(self):
        """String column without stats defaults to CATEGORICAL."""
        col = ColumnMetadata(name="label", data_type="varchar", value_type="string")
        result = _infer_column_role(col)
        assert result == ColumnRole.CATEGORICAL


class TestValidateColumnClassificationEdgeCases:
    """Edge case tests for _validate_column_classification."""

    def test_numeric_measure_low_cardinality_warning(self):
        """Soft warning for NUMERIC_MEASURE with distinct_count <= 5."""
        col = ColumnMetadata(name="score", data_type="integer", value_type="integer", distinct_count=3)
        hard, soft = _validate_column_classification(col, ColumnRole.NUMERIC_MEASURE.value)
        assert len(soft) > 0

    def test_temporal_on_year_column(self):
        """TEMPORAL on year column gives soft warning not hard error."""
        col = ColumnMetadata(name="birth_year", data_type="integer", value_type="integer")
        hard, soft = _validate_column_classification(col, ColumnRole.TEMPORAL.value)
        assert len(hard) > 0 or len(soft) > 0

    def test_temporal_on_domain_dtype_soft_warning_only(self):
        """TEMPORAL on a DOMAIN-typed column yields a soft warning, not a hard error."""
        col = ColumnMetadata(name="code", data_type="INTEGER DOMAIN year_t", value_type="integer")
        hard, soft = _validate_column_classification(col, ColumnRole.TEMPORAL.value)
        assert len(hard) == 0
        assert len(soft) > 0

    def test_boolean_distinct_none(self):
        """BOOLEAN with no distinct_count passes validation."""
        col = ColumnMetadata(name="flag", data_type="boolean", value_type="boolean")
        hard, soft = _validate_column_classification(col, ColumnRole.BOOLEAN.value)
        assert len(hard) == 0

    def test_categorical_low_cardinality(self):
        """CATEGORICAL with low cardinality produces no warnings."""
        col = ColumnMetadata(name="status", data_type="varchar", value_type="string", distinct_count=3)
        hard, soft = _validate_column_classification(col, ColumnRole.CATEGORICAL.value)
        assert len(hard) == 0
        assert len(soft) == 0


class TestExtractPkColumnsEdgeCases:
    """Edge case tests for _extract_pk_columns."""

    def test_quoted_columns(self):
        """Extract PK columns with quotes."""
        result = _extract_pk_columns('PRIMARY KEY ("id", "code")')
        assert "id" in result
        assert "code" in result

    def test_extra_whitespace(self):
        """Handle extra whitespace in PK definition."""
        result = _extract_pk_columns("PRIMARY  KEY  ( id ,  name )")
        assert "id" in result
        assert "name" in result


class TestExtractFkDefinitionEdgeCases:
    """Edge case tests for _extract_fk_definition."""

    def test_composite_fk(self):
        """Extract composite foreign key."""
        result = _extract_fk_definition("FOREIGN KEY (a, b) REFERENCES t2(x, y)")
        assert result is not None
        assert result["src_cols"] == ["a", "b"]
        assert result["dst_cols"] == ["x", "y"]

    def test_case_insensitive(self):
        """Handle lowercase foreign key keywords."""
        result = _extract_fk_definition("foreign key (col) references other(id)")
        assert result is not None
        assert result["dst_table"] == "other"


def _schema_with_partition(table: str, partition_cols: list[str]) -> SchemaGraph:
    """Build a SchemaGraph with a single table having the given partition columns."""
    cols = {"id": ColumnMetadata(name="id", data_type="integer")}
    for c in partition_cols:
        cols[c] = ColumnMetadata(name=c, data_type="string" if c != "dt" else "date")
    meta = TableMetadata(
        name=table,
        columns=cols,
        foreign_keys=[],
        primary_key="id",
        partition_columns=partition_cols,
    )
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={table: meta})


def _filter_param(col: str, op: str, param_key: str | None = None, raw_value=None) -> FilterParam:
    """Build a FilterParam for partition injection tests."""
    expr = NormalizedExpr.from_column(col)
    return FilterParam(left_expr=expr, op=op, param_key=param_key or "", raw_value=raw_value)


def _databricks_partition_dialect() -> DatabricksDialect:
    """Return a ``DatabricksDialect`` shell for ``inject_partition_filters`` tests."""

    return DatabricksDialect.__new__(DatabricksDialect)


class TestInjectPartitionFilters:
    """Tests for ``DatabricksDialect.inject_partition_filters``."""

    def test_inject_equality_partition_predicate(self):
        """Inject single equality partition predicate into WHERE."""
        schema = _schema_with_partition("events", ["dt"])
        intent = RuntimeIntent(
            tables=["events"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[_filter_param("events.dt", "=", "p1", None)],
            param_values={"p1": "2024-01-15"},
        )
        sql = "SELECT * FROM events"
        result = _databricks_partition_dialect().inject_partition_filters(sql, schema, intent)
        assert "`events`.`dt` = '2024-01-15'" in result
        assert "WHERE" in result

    def test_inject_in_partition_predicate(self):
        """Inject IN partition predicate from multiple equality filters."""
        schema = _schema_with_partition("logs", ["region"])
        intent = RuntimeIntent(
            tables=["logs"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                _filter_param("logs.region", "=", "p1", None),
                _filter_param("logs.region", "=", "p2", None),
            ],
            param_values={"p1": "us", "p2": "eu"},
        )
        sql = "SELECT * FROM logs"
        result = _databricks_partition_dialect().inject_partition_filters(sql, schema, intent)
        assert "`logs`.`region` IN ('us', 'eu')" in result

    def test_inject_between_partition_predicate(self):
        """Inject BETWEEN as >= and <= partition predicates."""
        schema = _schema_with_partition("sales", ["dt"])
        intent = RuntimeIntent(
            tables=["sales"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                _filter_param("sales.dt", ">=", "p1", None),
                _filter_param("sales.dt", "<=", "p2", None),
            ],
            param_values={"p1": "2024-01-01", "p2": "2024-01-31"},
        )
        sql = "SELECT * FROM sales"
        result = _databricks_partition_dialect().inject_partition_filters(sql, schema, intent)
        assert ">=" in result
        assert "<=" in result
        assert "2024-01-01" in result
        assert "2024-01-31" in result

    def test_no_partition_columns_unchanged(self):
        """SQL unchanged when table has no partition columns."""
        meta = TableMetadata(
            name="plain",
            columns={"id": ColumnMetadata(name="id", data_type="integer")},
            foreign_keys=[],
            primary_key="id",
            partition_columns=[],
        )
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"plain": meta})
        intent = RuntimeIntent(
            tables=["plain"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[_filter_param("plain.id", "=", "p1", None)],
            param_values={"p1": 1},
        )
        sql = "SELECT * FROM plain WHERE id = 1"
        result = _databricks_partition_dialect().inject_partition_filters(sql, schema, intent)
        assert result == sql

    def test_predicate_already_present_unchanged(self):
        """SQL unchanged when partition predicate already present."""
        schema = _schema_with_partition("events", ["dt"])
        intent = RuntimeIntent(
            tables=["events"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[_filter_param("events.dt", "=", "p1", None)],
            param_values={"p1": "2024-01-15"},
        )
        sql = "SELECT * FROM events WHERE `events`.`dt` = '2024-01-15'"
        result = _databricks_partition_dialect().inject_partition_filters(sql, schema, intent)
        assert result == sql

    def test_append_to_existing_where(self):
        """Append partition predicate to existing WHERE clause."""
        schema = _schema_with_partition("events", ["dt"])
        intent = RuntimeIntent(
            tables=["events"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[_filter_param("events.dt", "=", "p1", None)],
            param_values={"p1": "2024-01-15"},
        )
        sql = "SELECT * FROM events WHERE status = 'active'"
        result = _databricks_partition_dialect().inject_partition_filters(sql, schema, intent)
        assert "status = 'active'" in result
        assert "`events`.`dt` = '2024-01-15'" in result
        assert " AND " in result


class TestCollectProfilingTopKValues:
    """``collect_profiling_topk_values`` preserves profiling sample order."""

    def test_preserves_first_seen_order(self) -> None:
        assert collect_profiling_topk_values(["30", "10", "20", "10"]) == [
            "30",
            "10",
            "20",
        ]
