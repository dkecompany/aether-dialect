"""Tests for core_utils module: hashing, SQL normalization, question normalization."""

import json
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_MANIFEST_FILENAME,
    EngineConfig,
    PolicyConfig,
    diagnostic_force_enter,
    diagnostic_force_exit,
)
from aetherdialect._contracts_base import (
    LlmJsonExhausted,
    MigrationReport,
    MigrationTier,
)
from aetherdialect._core_utils import (
    ArtifactManifest,
    RephraseHint,
    _build_client,
    _extract_first_json_object,
    _format_cell,
    _llm_user_text_without_sensitivity_classification,
    _provider_order,
    _strip_fences,
    ask_user_choice,
    canonicalize_sql,
    classify_migration_tier,
    colmap_signature,
    dataframe_to_row_tuples,
    debug,
    detect_legacy_artifacts,
    diagnostic_print_listener,
    error,
    intent_id,
    interactive_yes_no,
    invalid_input,
    join_sig_string,
    llm_chat,
    llm_json,
    log,
    manifest_path,
    normalize_array_contains_param_value,
    normalize_op,
    normalize_question,
    normalize_sql,
    normalize_sql_operator_spaces,
    notify,
    pipeline_trace,
    pipeline_trace_lazy,
    print_info,
    print_query_result,
    print_rephrase_hint,
    progress,
    progress_enabled,
    prompt,
    read_artifact_manifest,
    read_gzip_json,
    result,
    safe_json_loads,
    schema_hash_fp,
    scope_hash_fp,
    sha256,
    stable_json,
    substitute_params,
    telemetry_capture,
    terminated,
    warn,
    wipe_versioned_artifacts,
    write_artifact_manifest,
    write_gzip_json_atomic,
)
from aetherdialect._dialect import (
    compute_sql_fp,
    parameter_abstract,
    sql_simplify_executable,
)
from aetherdialect._templates import apply_migration_policy


class TestSha256:
    """Tests for sha256."""

    def test_deterministic(self):
        """Sha256 is deterministic."""
        assert sha256("hello") == sha256("hello")

    def test_different_input_different_hash(self):
        """Sha256 differs for different input."""
        assert sha256("hello") != sha256("world")

    def test_returns_64_hex_chars(self):
        """Sha256 returns 64-char hex string."""
        h = sha256("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestStableJson:
    """Tests for stable_json."""

    def test_sorted_keys(self):
        """stable_json sorts keys."""
        result = stable_json({"b": 1, "a": 2})
        assert result.index('"a"') < result.index('"b"')

    def test_no_spaces(self):
        """stable_json uses compact separators."""
        result = stable_json({"a": 1})
        assert " " not in result

    def test_deterministic(self):
        """stable_json is deterministic."""
        d = {"z": 3, "a": 1, "m": 2}
        assert stable_json(d) == stable_json(d)


class TestStripFences:
    """Tests for strip_fences."""

    def test_removes_code_fences(self):
        """strip_fences removes markdown code fences."""
        assert _strip_fences("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_no_fences(self):
        """strip_fences leaves plain text unchanged."""
        assert _strip_fences("SELECT 1") == "SELECT 1"

    def test_strips_whitespace(self):
        """strip_fences strips surrounding whitespace."""
        assert _strip_fences("  SELECT 1  ") == "SELECT 1"


class TestNormalizeOp:
    """Tests for normalize_op."""

    def test_double_equals(self):
        """normalize_op maps == to =."""
        assert normalize_op("==") == "="

    def test_diamond(self):
        """normalize_op maps <> to !=."""
        assert normalize_op("<>") == "!="

    def test_ne(self):
        """normalize_op maps ne to !=."""
        assert normalize_op("ne") == "!="

    def test_gte(self):
        """normalize_op maps gte to >=."""
        assert normalize_op("gte") == ">="

    def test_lte(self):
        """normalize_op maps lte to <=."""
        assert normalize_op("lte") == "<="

    def test_eq(self):
        """normalize_op maps eq to =."""
        assert normalize_op("eq") == "="

    def test_gt(self):
        """normalize_op maps gt to >."""
        assert normalize_op("gt") == ">"

    def test_lt(self):
        """normalize_op maps lt to <."""
        assert normalize_op("lt") == "<"

    def test_ge(self):
        """normalize_op maps ge to >=."""
        assert normalize_op("ge") == ">="

    def test_le(self):
        """normalize_op maps le to <=."""
        assert normalize_op("le") == "<="

    def test_passthrough(self):
        """normalize_op passes through valid ops unchanged."""
        assert normalize_op("like") == "like"
        assert normalize_op("in") == "in"
        assert normalize_op("between") == "between"

    def test_case_insensitive(self):
        """normalize_op is case-insensitive."""
        assert normalize_op("NE") == "!="
        assert normalize_op("GTE") == ">="

    def test_whitespace_stripped(self):
        """normalize_op strips surrounding whitespace before lookup."""
        assert normalize_op("  ==  ") == "="

    def test_empty_string_returns_empty(self):
        """normalize_op returns empty string for empty input."""
        assert normalize_op("") == ""


class TestCanonicalizeSql:
    """Tests for canonicalize_sql."""

    def test_normalizes_whitespace(self):
        """canonicalize_sql collapses whitespace."""
        assert "  " not in canonicalize_sql("SELECT  *   FROM   t")

    def test_removes_trailing_semicolon(self):
        """canonicalize_sql removes trailing semicolon."""
        assert canonicalize_sql("SELECT 1;").endswith("1")

    def test_normalizes_equality_order(self):
        """canonicalize_sql normalizes equality operands alphabetically."""
        result = canonicalize_sql("SELECT * FROM t WHERE z.col = a.col")
        assert result.index("a.col") < result.index("z.col")

    def test_equality_order_preserved_when_left_less_than_right(self):
        """canonicalize_sql leaves operand order when left < right."""
        result = canonicalize_sql("SELECT * FROM t WHERE a.col = z.col")
        assert "a.col = z.col" in result

    def test_strips_code_fences(self):
        """canonicalize_sql handles code-fenced SQL."""
        result = canonicalize_sql("```sql\nSELECT 1\n```")
        assert result == "SELECT 1"


class TestNormalizeQuestion:
    """Tests for normalize_question."""

    def test_lowercases(self):
        """normalize_question lowercases unquoted text."""
        result = normalize_question("SHOW ALL Orders")
        assert "show" in result
        assert "orders" in result

    def test_preserves_quoted_case(self):
        """normalize_question preserves case inside quotes."""
        result = normalize_question("find orders with status 'Shipped'")
        assert "'Shipped'" in result

    def test_normalizes_smart_quotes(self):
        """normalize_question converts smart quotes to plain quotes."""
        result = normalize_question("status is \u2018Active\u2019")
        assert "'Active'" in result

    def test_removes_special_chars(self):
        """normalize_question removes non-alphanum special chars."""
        result = normalize_question("total $$ amount!!")
        assert "$" not in result
        assert "!" not in result


class TestColmapSignature:
    """Tests for colmap_signature."""

    def test_deterministic(self):
        """colmap_signature is deterministic."""
        cm = {"name": "customers", "amount": "orders"}
        assert colmap_signature(cm) == colmap_signature(cm)

    def test_different_maps_different_sig(self):
        """colmap_signature differs for different maps."""
        assert colmap_signature({"a": "t1"}) != colmap_signature({"b": "t2"})


class TestIntentId:
    """Tests for intent_id."""

    def test_returns_16_chars(self):
        """intent_id returns 16-character string."""
        result = intent_id({"tables": ["t"], "grain": "row_level"})
        assert len(result) == 16

    def test_deterministic(self):
        """intent_id is deterministic."""
        d = {"a": 1, "b": 2}
        assert intent_id(d) == intent_id(d)


class TestSchemaHashFp:
    """Tests for schema_hash_fp."""

    def test_deterministic(self):
        """schema_hash_fp is deterministic."""
        tables = {"t1": {"columns": ["a", "b"]}}
        assert schema_hash_fp(tables) == schema_hash_fp(tables)

    def test_different_tables_different_hash(self):
        """schema_hash_fp differs for different tables."""
        assert schema_hash_fp({"t1": {}}) != schema_hash_fp({"t2": {}})


class TestScopeHashFp:
    """scope_hash_fp incorporates allow_columns alongside deny_columns."""

    def test_allow_columns_changes_scope_hash(self):
        from aetherdialect._contracts_base import SchemaContext

        a = SchemaContext()
        b = SchemaContext(allow_columns=frozenset({"users.id"}))
        assert scope_hash_fp(a) != scope_hash_fp(b)

    def test_allow_columns_order_independent(self):
        from aetherdialect._contracts_base import SchemaContext

        a = SchemaContext(allow_columns=frozenset({"users.id", "orders.amount"}))
        b = SchemaContext(allow_columns=frozenset({"orders.amount", "users.id"}))
        assert scope_hash_fp(a) == scope_hash_fp(b)

    def test_allow_columns_distinct_from_deny_columns(self):
        from aetherdialect._contracts_base import SchemaContext

        a = SchemaContext(allow_columns=frozenset({"users.id"}))
        b = SchemaContext(deny_columns=frozenset({"users.id"}))
        assert scope_hash_fp(a) != scope_hash_fp(b)


class TestExtractFirstJsonObject:
    """Tests for extract_first_json_object."""

    def test_extracts_embedded_json(self):
        """extract_first_json_object extracts JSON from surrounding text."""
        result = _extract_first_json_object('Here is the answer: {"key": "value"} done.')
        assert result == '{"key": "value"}'

    def test_returns_none_for_no_json(self):
        """extract_first_json_object returns None when no braces found."""
        assert _extract_first_json_object("no json here") is None

    def test_handles_nested_braces(self):
        """extract_first_json_object handles nested objects."""
        result = _extract_first_json_object('{"a": {"b": 1}}')
        assert result == '{"a": {"b": 1}}'

    def test_handles_code_fences(self):
        """extract_first_json_object strips code fences first."""
        result = _extract_first_json_object('```json\n{"x": 1}\n```')
        assert result == '{"x": 1}'

    def test_returns_none_for_unclosed_brace(self):
        """extract_first_json_object returns None for unclosed brace."""
        assert _extract_first_json_object('{"key": "value"') is None


class TestSafeJsonLoads:
    """Tests for safe_json_loads."""

    def test_parses_valid_json(self):
        """safe_json_loads parses valid JSON directly."""
        result = safe_json_loads('{"a": 1}')
        assert result == {"a": 1}

    def test_parses_embedded_json(self):
        """safe_json_loads falls back to fragment extraction."""
        result = safe_json_loads('Some text {"a": 1} more text')
        assert result == {"a": 1}

    def test_returns_none_for_garbage(self):
        """safe_json_loads returns None for unparseable input."""
        assert safe_json_loads("not json at all") is None

    def test_handles_whitespace(self):
        """safe_json_loads strips whitespace."""
        result = safe_json_loads('  {"b": 2}  ')
        assert result == {"b": 2}


class TestParameterAbstract:
    """Tests for parameter_abstract."""

    def test_replaces_string_literal(self):
        """parameter_abstract replaces quoted strings."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE name = 'Alice'", sqlglot_dialect="postgres")
        assert ":p" in sql
        assert "'Alice'" in params.values()

    def test_replaces_numeric_literal(self):
        """parameter_abstract replaces numeric values."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE id = 42", sqlglot_dialect="postgres")
        assert ":p" in sql
        assert 42 in params.values()

    def test_replaces_date_literal(self):
        """parameter_abstract replaces ISO date string literals."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE dt = '2024-01-15'", sqlglot_dialect="postgres")
        assert "'2024-01-15'" in params.values()

    def test_returns_dict_of_params(self):
        """parameter_abstract returns dict with p1, p2, etc. keys."""
        _, params = parameter_abstract("WHERE a = 1 AND b = 'x'", sqlglot_dialect="postgres")
        assert all(k.startswith("p") for k in params)


class TestSubstituteParams:
    """Tests for substitute_params."""

    def test_substitutes_string(self):
        """substitute_params replaces string placeholders."""
        result = substitute_params("WHERE name = :p1", {"p1": "Alice"})
        assert "'Alice'" in result

    def test_substitutes_number(self):
        """substitute_params replaces numeric placeholders."""
        result = substitute_params("WHERE id = :p1", {"p1": 42})
        assert "42" in result

    def test_substitutes_list(self):
        """substitute_params replaces list placeholders."""
        result = substitute_params("WHERE id IN (:p1)", {"p1": [1, 2, 3]})
        assert "1, 2, 3" in result

    def test_substitutes_boolean(self):
        """substitute_params replaces boolean placeholders."""
        result = substitute_params("WHERE active = :p1", {"p1": True})
        assert "TRUE" in result

    def test_strips_trivial_coefficient(self):
        """sql_simplify_executable removes 1 * prefix."""
        result = sql_simplify_executable("SELECT 1 * col FROM t", sqlglot_dialect="postgres")
        assert "1 *" not in result

    def test_strips_trivial_offset(self):
        """sql_simplify_executable removes + 0 suffix."""
        result = sql_simplify_executable("SELECT col + 0 FROM t", sqlglot_dialect="postgres")
        assert "+ 0" not in result

    def test_strips_limit_none(self):
        """sql_simplify_executable removes LIMIT NULL."""
        result = sql_simplify_executable("SELECT * FROM t LIMIT NULL", sqlglot_dialect="postgres")
        assert "LIMIT" not in result.upper()

    def test_substitutes_quoted_string_in_list(self):
        """substitute_params directly substitutes pre-formatted quoted IN-lists."""
        result = substitute_params("WHERE x IN (:p1)", {"p1": "'R','PG-13'"})
        assert "'R','PG-13'" in result

    def test_substitutes_numeric_in_list_string(self):
        """substitute_params directly substitutes numeric comma- separated strings."""
        result = substitute_params("WHERE id IN (:p1)", {"p1": "1, 2, 3"})
        assert "1, 2, 3" in result
        assert "'" not in result

    def test_substitutes_float_in_list_string(self):
        """substitute_params directly substitutes float comma-separated strings."""
        result = substitute_params("WHERE price IN (:p1)", {"p1": "9.99, 19.99"})
        assert "9.99, 19.99" in result
        assert "'" not in result

    def test_non_numeric_string_gets_quoted(self):
        """substitute_params quotes a plain string that is not a numeric list."""
        result = substitute_params("WHERE name = :p1", {"p1": "Alice"})
        assert "'Alice'" in result

    def test_escapes_apostrophe_in_string_scalar(self):
        """Single quotes inside bound strings are doubled for SQL literals."""
        result = substitute_params("WHERE name = :p1", {"p1": "O'Brien"})
        assert "O''Brien" in result

    def test_escapes_apostrophe_in_string_list(self):
        """List string elements are escaped before quoting."""
        result = substitute_params("WHERE x IN (:p1)", {"p1": ["a'b", "c"]})
        assert "a''b" in result


class TestNormalizeArrayContainsParamValue:
    """Tests for normalize_array_contains_param_value."""

    def test_strips_quotes(self):
        """Quoted string values are unwrapped for array contains binding."""
        assert normalize_array_contains_param_value('"Trailers"') == "Trailers"
        assert normalize_array_contains_param_value("'Behind the Scenes'") == "Behind the Scenes"
        assert normalize_array_contains_param_value('  "Deleted Scenes"  ') == "Deleted Scenes"
        assert normalize_array_contains_param_value('"""x"""') == "x"
        assert normalize_array_contains_param_value(7) == 7


class TestComputeSqlFp:
    """Tests for compute_sql_fp."""

    def test_deterministic(self):
        assert compute_sql_fp("SELECT 1", sqlglot_dialect="postgres") == compute_sql_fp(
            "SELECT 1", sqlglot_dialect="postgres"
        )

    def test_case_insensitive(self):
        assert compute_sql_fp("SELECT 1", sqlglot_dialect="postgres") == compute_sql_fp(
            "select 1", sqlglot_dialect="postgres"
        )

    def test_returns_64_hex(self):
        result = compute_sql_fp("SELECT 1", sqlglot_dialect="postgres")
        assert len(result) == 64

    def test_whitespace_and_literal_forms_converge(self):
        a = compute_sql_fp("SELECT 1 FROM t WHERE x = 'abc'", sqlglot_dialect="postgres")
        b = compute_sql_fp("select   1   from   t   where   x   =   'abc'", sqlglot_dialect="postgres")
        assert a == b

    def test_different_literals_produce_same_fp(self):
        a = compute_sql_fp("SELECT * FROM t WHERE x = 1", sqlglot_dialect="postgres")
        b = compute_sql_fp("SELECT * FROM t WHERE x = 2", sqlglot_dialect="postgres")
        assert a == b

    def test_empty_input(self):
        assert compute_sql_fp("", sqlglot_dialect="postgres") == compute_sql_fp("", sqlglot_dialect="postgres")


class TestJoinSigString:
    """Tests for join_sig_string."""

    def test_joins_with_pipe(self):
        """join_sig_string joins elements with pipe."""
        assert join_sig_string(["a->b", "b->c"]) == "a->b|b->c"

    def test_empty_list(self):
        """join_sig_string returns empty string for empty list."""
        assert join_sig_string([]) == ""

    def test_single_element(self):
        """join_sig_string returns element for single-item list."""
        assert join_sig_string(["a->b"]) == "a->b"


class TestFormatCell:
    """Tests for _format_cell."""

    def test_none_returns_null(self):
        """_format_cell returns NULL for None."""
        assert _format_cell(None) == "NULL"

    def test_string_passthrough(self):
        """_format_cell returns string unchanged."""
        assert _format_cell("hello") == "hello"

    def test_integer_to_string(self):
        """_format_cell converts integer to string."""
        assert _format_cell(42) == "42"

    def test_decimal_integral(self):
        """_format_cell formats integral Decimal."""
        assert _format_cell(Decimal("5")) == "5"

    def test_decimal_fractional(self):
        """_format_cell formats fractional Decimal."""
        result = _format_cell(Decimal("3.14"))
        assert "3.14" in result


class TestLog:
    """Tests for log."""

    def test_prints_when_verbose(self, capsys):
        """Log prints when verbose diagnostics are enabled."""
        with patch("aetherdialect._core_utils.diagnostic_verbose_enabled", return_value=True):
            log("hello")
        out = capsys.readouterr().out
        assert "[LOG] hello" in out

    def test_silent_when_not_verbose(self, capsys):
        """Log is silent when verbose diagnostics are off."""
        with patch("aetherdialect._core_utils.diagnostic_verbose_enabled", return_value=False):
            log("hello")
        out = capsys.readouterr().out
        assert out == ""


class TestDebug:
    """Tests for debug."""

    def test_prints_when_debug(self, capsys):
        """Debug prints when debug diagnostics are enabled."""
        with patch("aetherdialect._core_utils.diagnostic_debug_enabled", return_value=True):
            debug("msg")
        out = capsys.readouterr().out
        assert "[DEBUG] msg" in out

    def test_silent_when_not_debug(self, capsys):
        """Debug is silent when debug diagnostics are off."""
        with patch("aetherdialect._core_utils.diagnostic_debug_enabled", return_value=False):
            debug("msg")
        out = capsys.readouterr().out
        assert out == ""


class TestNormalizeSql:
    """Tests for normalize_sql."""

    def test_adds_asc_to_bare_order_by(self):
        """normalize_sql appends ASC to bare ORDER BY columns."""
        result = normalize_sql("SELECT * FROM t ORDER BY name")
        assert "name ASC" in result

    def test_preserves_explicit_desc(self):
        """normalize_sql keeps explicit DESC."""
        result = normalize_sql("SELECT * FROM t ORDER BY name DESC")
        assert "DESC" in result

    def test_preserves_explicit_asc(self):
        """normalize_sql keeps explicit ASC."""
        result = normalize_sql("SELECT * FROM t ORDER BY name ASC")
        assert "ASC" in result

    def test_multiple_order_columns(self):
        """normalize_sql handles multiple ORDER BY columns."""
        result = normalize_sql("SELECT * FROM t ORDER BY a, b DESC, c")
        assert "a ASC" in result
        assert "b DESC" in result
        assert "c ASC" in result

    def test_no_order_by(self):
        """normalize_sql returns canonicalized SQL when no ORDER BY present."""
        result = normalize_sql("SELECT * FROM t WHERE id = 1")
        assert "ORDER" not in result

    def test_order_by_with_limit(self):
        """normalize_sql handles ORDER BY followed by LIMIT."""
        result = normalize_sql("SELECT * FROM t ORDER BY id LIMIT 10")
        assert "id ASC" in result
        assert "LIMIT" in result

    def test_empty_string(self):
        """normalize_sql returns empty string for empty input."""
        assert normalize_sql("") == ""

    def test_order_by_skips_empty_segments(self):
        """normalize_sql skips empty segments between commas in ORDER BY."""
        result = normalize_sql("SELECT 1 FROM t ORDER BY a, , b")
        assert "a ASC" in result
        assert "b ASC" in result

    def test_order_by_single_column_no_direction(self):
        """normalize_sql handles single column without ASC/DESC."""
        result = normalize_sql("SELECT * FROM t ORDER BY x")
        assert result.strip().endswith("x ASC") or "x ASC" in result


class TestPrintQueryResult:
    """Tests for print_query_result."""

    def test_single_scalar(self, capsys):
        """print_query_result prints scalar answer for single-row single-col."""
        with diagnostic_print_listener(print):
            print_query_result([(42,)], "SELECT COUNT(*)")
        out = capsys.readouterr().out
        assert "Answer: 42" in out

    def test_multi_row(self, capsys):
        """print_query_result prints tabular output for multiple rows."""
        rows = [(1, "a"), (2, "b")]
        with diagnostic_print_listener(print):
            print_query_result(rows, "SELECT * FROM t")
        out = capsys.readouterr().out
        assert "col1" in out
        assert "col2" in out

    def test_custom_headers(self, capsys):
        """print_query_result uses provided column headers."""
        rows = [(1, "a")]
        with diagnostic_print_listener(print):
            print_query_result(rows, "SELECT id, name FROM t", headers=["id", "name"])
        out = capsys.readouterr().out
        assert "id" in out
        assert "name" in out

    def test_truncates_beyond_five_rows(self, capsys):
        """print_query_result shows ellipsis when more than 5 rows."""
        rows = [(i,) for i in range(10)]
        with diagnostic_print_listener(print):
            print_query_result(rows, "SELECT * FROM t")
        out = capsys.readouterr().out
        assert "10 total rows" in out

    def test_results_heading(self, capsys):
        """print_query_result prints the configured results heading."""

        from aetherdialect._core_utils import QUERY_RESULTS_HEADER

        with diagnostic_print_listener(print):
            print_query_result([(1,)], "SELECT 1")
        out = capsys.readouterr().out
        assert QUERY_RESULTS_HEADER in out


class TestPrintInfo:
    """Tests for print_info."""

    def test_title_only(self, capsys):
        """print_info prints title without items or footer."""
        with diagnostic_print_listener(print):
            print_info("My Title")
        out = capsys.readouterr().out
        assert "My Title" in out

    def test_with_items(self, capsys):
        """print_info prints key-value pairs."""
        with diagnostic_print_listener(print):
            print_info("Header", items={"key1": "val1", "key2": "val2"})
        out = capsys.readouterr().out
        assert "key1: val1" in out
        assert "key2: val2" in out

    def test_with_footer(self, capsys):
        """print_info prints footer text."""
        with diagnostic_print_listener(print):
            print_info("Title", footer="Done!")
        out = capsys.readouterr().out
        assert "Done!" in out

    def test_list_items_joined(self, capsys):
        """print_info joins list values with commas."""
        with diagnostic_print_listener(print):
            print_info("Title", items={"tags": ["a", "b", "c"]})
        out = capsys.readouterr().out
        assert "a, b, c" in out

    def test_set_items_displayed(self, capsys):
        """print_info handles set values."""
        with diagnostic_print_listener(print):
            print_info("Title", items={"s": {1}})
        out = capsys.readouterr().out
        assert "1" in out


class TestSha256EdgeCases:
    """Edge-case tests for sha256."""

    def test_empty_string(self):
        """Sha256 handles empty string."""
        h = sha256("")
        assert len(h) == 64

    def test_unicode(self):
        """Sha256 handles unicode characters."""
        h = sha256("日本語")
        assert len(h) == 64

    def test_long_string(self):
        """Sha256 handles long input."""
        h = sha256("x" * 100000)
        assert len(h) == 64


class TestStableJsonEdgeCases:
    """Edge-case tests for stable_json."""

    def test_nested_dict(self):
        """stable_json handles nested dicts with sorted keys."""
        result = stable_json({"z": {"b": 1, "a": 2}})
        assert '"a":2' in result or '"a": 2' in result

    def test_list_values(self):
        """stable_json handles list values."""
        result = stable_json({"items": [3, 1, 2]})
        assert "[3,1,2]" in result

    def test_empty_dict(self):
        """stable_json handles empty dict."""
        assert stable_json({}) == "{}"

    def test_boolean_values(self):
        """stable_json handles boolean values."""
        result = stable_json({"flag": True})
        assert "true" in result

    def test_null_value(self):
        """stable_json handles None values."""
        result = stable_json({"key": None})
        assert "null" in result


class TestStripFencesEdgeCases:
    """Edge-case tests for strip_fences."""

    def test_json_fence(self):
        """strip_fences removes json code fences."""
        assert _strip_fences("```json\n{}\n```") == "{}"

    def test_empty_fences(self):
        """strip_fences handles empty content in fences."""
        assert _strip_fences("```\n\n```") == ""

    def test_no_language_tag(self):
        """strip_fences removes fences without language tag."""
        assert _strip_fences("```\ncode\n```") == "code"

    def test_multiple_fences_keeps_inner(self):
        """strip_fences only removes outer fences."""
        result = _strip_fences("```sql\nSELECT '```'\n```")
        assert "SELECT" in result


class TestCanonicalizeSqlEdgeCases:
    """Edge-case tests for canonicalize_sql."""

    def test_empty_string(self):
        """canonicalize_sql handles empty string."""
        assert canonicalize_sql("") == ""

    def test_normalizes_comma_spacing(self):
        """canonicalize_sql normalizes spaces around commas."""
        result = canonicalize_sql("SELECT a,b,c FROM t")
        assert ", " in result

    def test_normalizes_paren_spacing(self):
        """canonicalize_sql removes spaces inside parentheses."""
        result = canonicalize_sql("SELECT COUNT( * ) FROM t")
        assert "( " not in result
        assert " )" not in result

    def test_equal_sign_spacing(self):
        """canonicalize_sql normalizes spaces around equals."""
        result = canonicalize_sql("WHERE a=1")
        assert " = " in result

    def test_strips_explain_prefix(self):
        """canonicalize_sql strips EXPLAIN prefix."""
        assert canonicalize_sql("EXPLAIN SELECT 1") == "SELECT 1"

    def test_strips_explain_analyze_prefix(self):
        """canonicalize_sql strips EXPLAIN ANALYZE prefix."""
        result = canonicalize_sql("EXPLAIN ANALYZE SELECT * FROM t")
        assert result.startswith("SELECT")
        assert "EXPLAIN" not in result

    def test_strips_explain_case_insensitive(self):
        """canonicalize_sql strips explain in any case."""
        assert canonicalize_sql("explain select 1") == "select 1"

    def test_plain_select_unchanged_by_explain_strip(self):
        """canonicalize_sql leaves plain SELECT unchanged."""
        result = canonicalize_sql("SELECT 1")
        assert result == "SELECT 1"


class TestNormalizeQuestionEdgeCases:
    """Edge-case tests for normalize_question."""

    def test_empty_string(self):
        """normalize_question handles empty string."""
        assert normalize_question("") == ""

    def test_multiple_quoted_values(self):
        """normalize_question preserves multiple quoted values."""
        result = normalize_question("find 'Alpha' and 'Beta'")
        assert "'Alpha'" in result
        assert "'Beta'" in result

    def test_double_smart_quotes(self):
        """normalize_question converts double smart quotes."""
        result = normalize_question("status is \u201cActive\u201d")
        assert "'Active'" in result

    def test_collapses_whitespace(self):
        """normalize_question collapses multiple spaces."""
        result = normalize_question("show   all   orders")
        assert "  " not in result


class TestExtractFirstJsonObjectEdgeCases:
    """Edge-case tests for extract_first_json_object."""

    def test_empty_string(self):
        """extract_first_json_object returns None for empty string."""
        assert _extract_first_json_object("") is None

    def test_multiple_json_objects(self):
        """extract_first_json_object returns the first object."""
        result = _extract_first_json_object('{"a":1} {"b":2}')
        assert result == '{"a":1}'

    def test_deeply_nested(self):
        """extract_first_json_object handles deep nesting."""
        result = _extract_first_json_object('{"a":{"b":{"c":1}}}')
        assert result == '{"a":{"b":{"c":1}}}'


class TestSafeJsonLoadsEdgeCases:
    """Edge-case tests for safe_json_loads."""

    def test_empty_string(self):
        """safe_json_loads returns None for empty string."""
        assert safe_json_loads("") is None

    def test_json_array(self):
        """safe_json_loads parses JSON arrays."""
        result = safe_json_loads("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_invalid_json_fragment(self):
        """safe_json_loads returns None when fragment is invalid."""
        assert safe_json_loads("{broken json") is None


class TestParameterAbstractEdgeCases:
    """Edge-case tests for parameter_abstract."""

    def test_no_literals(self):
        """parameter_abstract returns empty params for no literals."""
        sql, params = parameter_abstract("SELECT * FROM t", sqlglot_dialect="postgres")
        assert len(params) == 0

    def test_multiple_string_literals(self):
        """parameter_abstract replaces multiple string literals."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE a = 'foo' AND b = 'bar'", sqlglot_dialect="postgres")
        assert len([v for v in params.values() if isinstance(v, str) and v.startswith("'")]) == 2

    def test_date_format_slash(self):
        """parameter_abstract replaces slash-format date string literals."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE dt = '01/15/2024'", sqlglot_dialect="postgres")
        assert "'01/15/2024'" in params.values()

    def test_decimal_number(self):
        """parameter_abstract replaces decimal numbers."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE price > 9.99", sqlglot_dialect="postgres")
        assert 9.99 in params.values()


class TestSubstituteParamsEdgeCases:
    """Edge-case tests for substitute_params."""

    def test_none_value(self):
        """substitute_params handles None value."""
        result = substitute_params("WHERE x = :p1", {"p1": None})
        assert "None" in result

    def test_already_quoted_csv_string(self):
        """substitute_params handles pre-quoted comma-separated string."""
        result = substitute_params("WHERE x IN (:p1)", {"p1": "'a','b'"})
        assert "'a','b'" in result

    def test_empty_params(self):
        """substitute_params returns SQL unchanged for empty params."""
        result = substitute_params("SELECT 1", {})
        assert result == "SELECT 1"

    def test_multiple_params_sorted_by_length(self):
        """substitute_params replaces longer keys first to avoid partial matches."""
        result = substitute_params("WHERE a = :p10 AND b = :p1", {"p1": "x", "p10": "y"})
        assert "'y'" in result
        assert "'x'" in result

    def test_skips_empty_key_in_params(self):
        """substitute_params skips placeholder for empty string key."""
        result = substitute_params("WHERE x = :p1", {"": "ignored", "p1": "val"})
        assert "val" in result


class TestComputeSqlFpEdgeCases:
    """Edge-case tests for compute_sql_fp."""

    def test_empty_string(self):
        h = compute_sql_fp("", sqlglot_dialect="postgres")
        assert len(h) == 64

    def test_whitespace_only(self):
        h = compute_sql_fp("   ", sqlglot_dialect="postgres")
        assert len(h) == 64


class TestJoinSigStringEdgeCases:
    """Edge-case tests for join_sig_string."""

    def test_multiple_elements(self):
        """join_sig_string joins multiple elements."""
        result = join_sig_string(["a->b", "c->d", "e->f"])
        assert result == "a->b|c->d|e->f"

    def test_elements_with_special_chars(self):
        """join_sig_string handles elements with special characters."""
        result = join_sig_string(["t1.id=t2.fk"])
        assert result == "t1.id=t2.fk"


class TestFormatCellEdgeCases:
    """Edge-case tests for _format_cell."""

    def test_boolean(self):
        """_format_cell formats boolean."""
        assert _format_cell(True) == "True"

    def test_float(self):
        """_format_cell formats float."""
        assert _format_cell(3.14) == "3.14"

    def test_zero(self):
        """_format_cell formats zero."""
        assert _format_cell(0) == "0"

    def test_empty_string(self):
        """_format_cell formats empty string."""
        assert _format_cell("") == ""

    def test_decimal_zero(self):
        """_format_cell formats Decimal zero."""
        result = _format_cell(Decimal("0"))
        assert result == "0"

    def test_large_decimal(self):
        """_format_cell formats large Decimal."""
        result = _format_cell(Decimal("99999999"))
        assert "99999999" in result


class TestColmapSignatureEdgeCases:
    """Edge-case tests for colmap_signature."""

    def test_empty_map(self):
        """colmap_signature handles empty column map."""
        h = colmap_signature({})
        assert len(h) == 64

    def test_order_independent(self):
        """colmap_signature is order-independent."""
        assert colmap_signature({"a": "t1", "b": "t2"}) == colmap_signature({"b": "t2", "a": "t1"})


class TestIntentIdEdgeCases:
    """Edge-case tests for intent_id."""

    def test_empty_dict(self):
        """intent_id handles empty dict."""
        result = intent_id({})
        assert len(result) == 16

    def test_nested_structure(self):
        """intent_id handles nested structures."""
        result = intent_id({"a": {"b": [1, 2]}})
        assert len(result) == 16


class TestSchemaHashFpEdgeCases:
    """Edge-case tests for schema_hash_fp."""

    def test_empty_tables(self):
        """schema_hash_fp handles empty tables dict."""
        h = schema_hash_fp({})
        assert len(h) == 64

    def test_nested_column_info(self):
        """schema_hash_fp handles nested column info."""
        h = schema_hash_fp({"t1": {"columns": [{"name": "id", "type": "int"}]}})
        assert len(h) == 64


class TestTelemetryCapture:
    """Tests for telemetry_capture."""

    def test_collects_log_and_debug_lines(self):
        """Buffered lines include log/debug output while sink is active."""
        diagnostic_force_enter()
        try:
            with telemetry_capture(suppress_console=True) as buf:
                log("L1")
                debug("D1")
            assert any("[LOG] L1" in line for line in buf)
            assert any("[DEBUG] D1" in line for line in buf)
        finally:
            diagnostic_force_exit()

    def test_suppress_console_skips_print(self, capsys):
        """suppress_console=True avoids printing even when flags are on."""
        diagnostic_force_enter()
        try:
            with telemetry_capture(suppress_console=True):
                log("silent")
            assert capsys.readouterr().out == ""
        finally:
            diagnostic_force_exit()

    def test_force_diagnostic_flags_restored(self, monkeypatch):
        """force_diagnostic_flags enables diagnostics only inside the block."""
        import aetherdialect._config as cfg_mod
        from aetherdialect._config import (
            diagnostic_debug_enabled,
            diagnostic_pipeline_trace_full_enabled,
        )

        monkeypatch.setattr(cfg_mod, "_DIAGNOSTIC_FORCE_DEPTH", 0, raising=False)
        saved = (
            PolicyConfig.DEBUG,
            PolicyConfig.VERBOSE,
            PolicyConfig.PIPELINE_TRACE_FULL,
            PolicyConfig.LIVE_DEEP_TRACE,
        )
        PolicyConfig.DEBUG = False
        PolicyConfig.VERBOSE = False
        PolicyConfig.PIPELINE_TRACE_FULL = False
        PolicyConfig.LIVE_DEEP_TRACE = False
        try:
            assert not diagnostic_debug_enabled()
            with telemetry_capture(force_diagnostic_flags=True, suppress_console=True):
                assert diagnostic_debug_enabled()
                assert diagnostic_pipeline_trace_full_enabled()
            assert not diagnostic_debug_enabled()
            assert not diagnostic_pipeline_trace_full_enabled()
        finally:
            (
                PolicyConfig.DEBUG,
                PolicyConfig.VERBOSE,
                PolicyConfig.PIPELINE_TRACE_FULL,
                PolicyConfig.LIVE_DEEP_TRACE,
            ) = saved


class TestPipelineTrace:
    """Tests for pipeline_trace."""

    def test_prints_when_debug_and_full_trace(self, capsys):
        """pipeline_trace prints when debug and full pipeline trace diagnostics are on."""
        with (
            patch("aetherdialect._core_utils.diagnostic_debug_enabled", return_value=True),
            patch(
                "aetherdialect._core_utils.diagnostic_pipeline_trace_full_enabled",
                return_value=True,
            ),
        ):
            pipeline_trace("step", "payload")
        out = capsys.readouterr().out
        assert "[PIPELINE_TRACE] step" in out
        assert "payload" in out

    def test_silent_when_not_full_trace(self, capsys):
        """pipeline_trace does not print when full pipeline trace is off."""
        with (
            patch("aetherdialect._core_utils.diagnostic_debug_enabled", return_value=True),
            patch(
                "aetherdialect._core_utils.diagnostic_pipeline_trace_full_enabled",
                return_value=False,
            ),
        ):
            pipeline_trace("step", "x")
        assert capsys.readouterr().out == ""


class TestPipelineTraceLazy:
    """Tests for pipeline_trace_lazy."""

    def test_skips_body_factory_when_inactive_and_no_sink(self) -> None:
        """Body factory is not invoked when console trace is off and no telemetry sink."""
        from unittest.mock import patch

        called = {"n": 0}

        def factory() -> str:
            called["n"] += 1
            return "x"

        with (
            patch("aetherdialect._core_utils.diagnostic_debug_enabled", return_value=False),
            patch(
                "aetherdialect._core_utils.diagnostic_pipeline_trace_full_enabled",
                return_value=False,
            ),
            patch("aetherdialect._core_utils._telemetry_sink", None),
        ):
            pipeline_trace_lazy("h", factory)
        assert called["n"] == 0


class TestText2SqlInitSurface:
    """Public package surface excludes internal diagnostics helpers."""

    def test_init_does_not_export_diagnostics_enabled(self) -> None:
        """``diagnostics_enabled`` is not part of the stable import surface."""
        import aetherdialect as pkg

        assert "diagnostics_enabled" not in pkg.__all__


class TestNormalizeSqlOperatorSpaces:
    """Tests for normalize_sql_operator_spaces."""

    def test_empty_returns_empty(self):
        """Empty string is returned unchanged."""
        assert normalize_sql_operator_spaces("") == ""

    def test_whitespace_only_returns_original(self):
        """Whitespace-only input is returned without merging operators."""
        assert normalize_sql_operator_spaces("   ") == "   "

    def test_merges_split_operators(self):
        """Split comparison operators are merged."""
        s = "WHERE x > = 1 AND y < = 2 AND z ! = 3"
        out = normalize_sql_operator_spaces(s)
        assert ">=" in out
        assert "<=" in out
        assert "!=" in out
        assert "> =" not in out


class TestNormalizeSqlVsOperatorSpaces:
    """normalize_sql applies operator-space merging after canonicalization."""

    def test_split_gte_merged_under_normalize_sql(self):
        """Split comparison operators are merged before ORDER BY normalization."""
        result = normalize_sql("SELECT * FROM t WHERE id > = 1 ORDER BY x")
        assert ">=" in result
        assert "x ASC" in result


class TestCanonicalizeSqlNotEqualPreserved:
    """Equality canonicalization must not break != ."""

    def test_not_equal_unchanged(self):
        """!= remains a single token (not turned into spaced =)."""
        result = canonicalize_sql("SELECT * FROM t WHERE a != 'x'")
        assert "!=" in result
        assert "a = !" not in result


class TestParameterAbstractEscapedQuotes:
    """Edge cases for parameter_abstract."""

    def test_sql_escaped_single_quote(self):
        """Doubled single-quote is parsed as a single string literal under the Postgres AST parser."""
        sql, params = parameter_abstract("SELECT * FROM t WHERE note = 'it''s fine'", sqlglot_dialect="postgres")
        values = [v for v in params.values() if isinstance(v, str) and v.startswith("'")]
        assert values == ["'it's fine'"]


class TestSubstituteParamsMoreCases:
    """Additional substitute_params behaviour."""

    def test_false_boolean(self):
        """Boolean False becomes FALSE."""
        assert "FALSE" in substitute_params("WHERE x = :p1", {"p1": False})

    def test_trailing_star_one_removed(self):
        """Trailing * 1 is simplified."""
        result = sql_simplify_executable("SELECT col * 1 FROM t", sqlglot_dialect="postgres")
        assert "* 1" not in result

    def test_subtract_zero_removed(self):
        """Subtraction of zero is stripped."""
        result = sql_simplify_executable("SELECT col - 0 FROM t", sqlglot_dialect="postgres")
        assert "- 0" not in result


class TestNormalizeArrayContainsMore:
    """More normalize_array_contains_param_value cases."""

    def test_empty_string(self):
        assert normalize_array_contains_param_value("") == ""

    def test_unbalanced_quotes_unchanged(self):
        """Odd leading quote without closing pair is not stripped in a loop."""
        assert normalize_array_contains_param_value('"only') == '"only'

    def test_non_string_unchanged(self):
        assert normalize_array_contains_param_value([1]) == [1]


class TestPrintQueryResultEdgeCases:
    """Edge cases for print_query_result."""

    def test_empty_rows_no_crash(self, capsys):
        """Zero rows prints headers area without scalar branch."""
        with diagnostic_print_listener(print):
            print_query_result([], "SELECT 1 WHERE 1=0")
        out = capsys.readouterr().out
        assert "SQL:" in out


class TestPrintInfoEdgeCases:
    """Edge cases for print_info."""

    def test_explicit_none_items_skips_block(self, capsys):
        """items=None does not iterate."""
        with diagnostic_print_listener(print):
            print_info("T", items=None)
        assert "T" in capsys.readouterr().out

    def test_tuple_value_joined_like_list(self, capsys):
        """Tuple values are formatted like lists."""
        with diagnostic_print_listener(print):
            print_info("T", items={"k": (1, 2)})
        out = capsys.readouterr().out
        assert "1, 2" in out


class TestProgressChannel:
    """Tests for interactive-only :func:`progress` / :func:`progress_enabled`."""

    def test_progress_noop_when_disabled(self, capsys) -> None:
        progress("hello")
        assert capsys.readouterr().out == ""

    def test_progress_prints_when_enabled(self, capsys) -> None:
        with progress_enabled():
            with diagnostic_print_listener(print):
                progress("hello")
        out = capsys.readouterr().out
        assert "hello" in out


class TestAskUserChoice:
    """Tests for ask_user_choice."""

    def test_yes_short(self, capsys):
        with diagnostic_print_listener(print):
            with patch("aetherdialect._core_utils.input", return_value="y"):
                assert ask_user_choice("OK?", ["y", "n"]) == "y"
        out = capsys.readouterr().out
        assert "OK? (y/n): " in out
        assert "Yes" in out

    def test_no_short_silent_no(self, capsys):
        with diagnostic_print_listener(print):
            with patch("aetherdialect._core_utils.input", return_value="n"):
                assert ask_user_choice("OK?", ["y", "n"], silent_no=True) == "n"
        out = capsys.readouterr().out
        assert "No" in out
        assert "User terminated." not in out

    def test_eof_returns_none(self, capsys):
        with diagnostic_print_listener(print):
            with patch("aetherdialect._core_utils.input", side_effect=EOFError()):
                assert ask_user_choice("OK?", ["y", "n"]) is None
        assert "User terminated." in capsys.readouterr().out

    def test_keyboard_interrupt_returns_none(self, capsys):
        with patch("aetherdialect._core_utils.input", side_effect=KeyboardInterrupt()):
            assert ask_user_choice("OK?", ["y", "n"]) is None

    def test_empty_input_invalid(self, capsys):
        with diagnostic_print_listener(print):
            with patch("aetherdialect._core_utils.input", return_value=""):
                assert ask_user_choice("OK?", ["y", "n"]) is None
        assert "Invalid input." in capsys.readouterr().out

    def test_garbage_invalid(self, capsys):
        with patch("aetherdialect._core_utils.input", return_value="maybe"):
            assert ask_user_choice("OK?", ["y", "n"]) is None


class TestInteractiveYesNo:
    """Tests for interactive_yes_no."""

    def test_delegates_to_choice_port(self):
        class Port:
            def take_yes_no(self, stage, prompt, options, silent_no=False):
                assert stage == "s1"
                assert "hello" in prompt
                return "n"

        port = Port()
        assert interactive_yes_no("s1", "hello", ["y", "n"], choice_port=port) == "n"

    def test_falls_back_to_ask_user_choice(self, capsys):
        with patch("aetherdialect._core_utils.input", return_value="yes"):
            assert interactive_yes_no("s1", "Q", ["y", "n"]) == "y"


class TestGzipJsonRoundtrip:
    """read_gzip_json and write_gzip_json_atomic."""

    def test_roundtrip_sort_keys(self, tmp_path):
        path = tmp_path / "data.json.gz"
        obj = {"b": 2, "a": 1}
        write_gzip_json_atomic(str(path), obj, sort_keys=True)
        loaded = read_gzip_json(str(path))
        assert loaded == obj
        raw = path.read_bytes()
        assert raw[:2] == b"\x1f\x8b"

    def test_roundtrip_sort_keys_false(self, tmp_path):
        path = tmp_path / "ns.json.gz"
        write_gzip_json_atomic(str(path), {"x": [3, 1]}, sort_keys=False)
        assert read_gzip_json(str(path)) == {"x": [3, 1]}


def _manifest_schema_stub() -> MagicMock:
    """Minimal schema graph stand-in for migration policy tests."""

    s = MagicMock()
    s.effective_structural_hash = "live_e"
    s.profiling_hash = "live_p"
    s.notes_hash = "live_n"
    s.semantic_edges_hash = "live_s"
    s.structural_hash = "live_t"
    s.scope_hash = "live_c"
    return s


class TestArtifactManifest:
    """manifest_path, read/write manifest, apply_migration_policy."""

    def test_manifest_path_joins_filename(self, tmp_path):
        d = str(tmp_path)
        assert manifest_path(d) == os.path.join(d, ARTIFACT_MANIFEST_FILENAME)

    def test_read_missing_returns_none(self, tmp_path):
        assert read_artifact_manifest(str(tmp_path)) is None

    def test_read_invalid_json_returns_none(self, tmp_path):
        p = tmp_path / ARTIFACT_MANIFEST_FILENAME
        p.write_text("{", encoding="utf-8")
        assert read_artifact_manifest(str(tmp_path)) is None

    def test_read_non_dict_returns_none(self, tmp_path):
        p = tmp_path / ARTIFACT_MANIFEST_FILENAME
        p.write_text('["a"]', encoding="utf-8")
        assert read_artifact_manifest(str(tmp_path)) is None

    def test_write_artifact_manifest_creates_file(self, tmp_path, capsys):
        d = str(tmp_path)
        with patch("aetherdialect._core_utils.debug"):
            write_artifact_manifest(d)
        mp = manifest_path(d)
        assert os.path.isfile(mp)
        data = json.loads(open(mp, encoding="utf-8").read())
        assert "artifact_format_version" in data
        assert "created_with_package_version" in data

    def test_apply_migration_no_change_skips_manifest_write(self, tmp_path):
        d = str(tmp_path)
        sch = _manifest_schema_stub()
        with patch(
            "aetherdialect._templates.classify_migration_tier",
            return_value=MigrationTier.NO_CHANGE,
        ):
            with patch("aetherdialect._templates.write_artifact_manifest") as w:
                rep = apply_migration_policy(d, sch)
        w.assert_not_called()
        assert isinstance(rep, MigrationReport)
        assert rep.tier == MigrationTier.NO_CHANGE

    def test_apply_migration_soft_refresh_writes_manifest(self, tmp_path):
        d = str(tmp_path)
        sch = _manifest_schema_stub()
        with patch(
            "aetherdialect._templates.classify_migration_tier",
            return_value=MigrationTier.SOFT_REFRESH,
        ):
            with patch("aetherdialect._templates.write_artifact_manifest") as w:
                with patch("aetherdialect._templates.debug"):
                    rep = apply_migration_policy(d, sch)
        w.assert_called_once()
        assert w.call_args[0][0] == d
        assert rep.tier == MigrationTier.SOFT_REFRESH

    def test_apply_migration_destructive_removes_versioned_artifacts(self, tmp_path):
        d = str(tmp_path)
        os.makedirs(d, exist_ok=True)
        stale = os.path.join(d, "schema_graph.json.gz")
        open(stale, "wb").close()
        sch = _manifest_schema_stub()
        with patch(
            "aetherdialect._templates.classify_migration_tier",
            return_value=MigrationTier.DESTRUCTIVE,
        ):
            with patch("aetherdialect._templates.debug"):
                rep = apply_migration_policy(d, sch)
        assert not os.path.isfile(stale)
        assert rep.tier == MigrationTier.DESTRUCTIVE

    def test_classify_profiling_change_overlap_gate(self):
        m = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            effective_structural_hash="e",
            structural_hash="t",
            profiling_hash="p0",
            scope_hash="c",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        sch = MagicMock()
        sch.effective_structural_hash = "e"
        sch.structural_hash = "t"
        sch.profiling_hash = "p1"
        sch.scope_hash = "c"
        sch.notes_hash = "n"
        sch.semantic_edges_hash = "s"
        prev = MagicMock()
        with patch("aetherdialect._core_utils.profiling_value_overlap", return_value=0.5):
            assert classify_migration_tier(m, sch, previous_schema=prev) == MigrationTier.SOFT_REFRESH
        with patch("aetherdialect._core_utils.profiling_value_overlap", return_value=0.01):
            assert classify_migration_tier(m, sch, previous_schema=prev) == MigrationTier.DESTRUCTIVE

    def test_classify_stale_artifact_format_is_destructive(self):
        m = ArtifactManifest(
            artifact_format_version=1,
            effective_structural_hash="e",
            structural_hash="t",
            profiling_hash="p0",
            scope_hash="c",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        sch = MagicMock()
        sch.effective_structural_hash = "e2"
        sch.structural_hash = "t2"
        sch.profiling_hash = "p0"
        sch.scope_hash = "c"
        sch.notes_hash = "n"
        sch.semantic_edges_hash = "s"
        assert classify_migration_tier(m, sch) == MigrationTier.DESTRUCTIVE

    def test_classify_package_below_manifest_min_is_destructive(self):
        m = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            min_compatible_package_version="99.0.0",
            effective_structural_hash="e",
            structural_hash="t",
            profiling_hash="p0",
            scope_hash="c",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        sch = MagicMock()
        sch.effective_structural_hash = "e2"
        sch.structural_hash = "t"
        sch.profiling_hash = "p0"
        sch.scope_hash = "c"
        sch.notes_hash = "n"
        sch.semantic_edges_hash = "s"
        assert classify_migration_tier(m, sch) == MigrationTier.DESTRUCTIVE

    def test_classify_schema_diff_implies_remap_when_rename_plan_missing(self) -> None:
        """Non-empty diff with column renames yields REMAP when scope is stable even if try_rename is None."""

        from aetherdialect._schema import SchemaDiff, TableDiff

        m = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            effective_structural_hash="e1",
            structural_hash="t1",
            profiling_hash="p0",
            scope_hash="c",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        sch = MagicMock()
        sch.effective_structural_hash = "e2"
        sch.structural_hash = "t2"
        sch.profiling_hash = "p0"
        sch.scope_hash = "c"
        sch.notes_hash = "n"
        sch.semantic_edges_hash = "s"
        prev = MagicMock()
        diff = SchemaDiff(per_table={"t": TableDiff(renamed_columns=(("a", "b"),))})
        with patch("aetherdialect._core_utils.try_rename_migration_plan", return_value=None):
            assert classify_migration_tier(m, sch, previous_schema=prev, schema_diff=diff) == MigrationTier.REMAP


class TestWipeVersionedArtifacts:
    """Destructive migration wipe removes QSim versioned globs."""

    def test_removes_qsim_glob_files(self, tmp_path):
        d = str(tmp_path)
        for name in (
            "qsim_questions_v1.json.gz",
            "qsim_summary_v2.json.gz",
            "qsim_skeletons_v3.json.gz",
        ):
            open(os.path.join(d, name), "wb").close()
        wipe_versioned_artifacts(d)
        assert not any(os.path.isfile(os.path.join(d, n)) for n in os.listdir(d))


class TestDetectLegacyArtifacts:
    """``detect_legacy_artifacts`` only returns names when no manifest exists."""

    def test_missing_dir_returns_empty(self, tmp_path):
        assert detect_legacy_artifacts(str(tmp_path / "missing")) == []

    def test_empty_dir_returns_empty(self, tmp_path):
        assert detect_legacy_artifacts(str(tmp_path)) == []

    def test_unknown_file_only_returns_empty(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        assert detect_legacy_artifacts(str(tmp_path)) == []

    def test_legacy_files_no_manifest_returns_sorted(self, tmp_path):
        for name in ("schema_graph.json.gz", "intent_templates.json.gz"):
            (tmp_path / name).write_bytes(b"")
        result = detect_legacy_artifacts(str(tmp_path))
        assert result == sorted(["schema_graph.json.gz", "intent_templates.json.gz"])

    def test_glob_legacy_files_no_manifest(self, tmp_path):
        (tmp_path / "qsim_summary_v1.json.gz").write_bytes(b"")
        result = detect_legacy_artifacts(str(tmp_path))
        assert "qsim_summary_v1.json.gz" in result

    def test_manifest_present_returns_empty(self, tmp_path):
        from aetherdialect._core_utils import ARTIFACT_MANIFEST_FILENAME

        (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        (tmp_path / "schema_graph.json.gz").write_bytes(b"")
        assert detect_legacy_artifacts(str(tmp_path)) == []


class TestLlmChat:
    """llm_chat with mocked OpenAI client."""

    def test_success_returns_stripped_output(self):
        mock_resp = MagicMock()
        mock_resp.output_text = '  {"a": 1}  \n'
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch("aetherdialect._core_utils._provider_order", return_value=["openai"]):
            with patch("aetherdialect._core_utils._provider_is_configured", return_value=True):
                with patch("aetherdialect._core_utils._build_client", return_value=mock_client):
                    with patch("aetherdialect._core_utils.debug"):
                        with patch("aetherdialect._core_utils.pipeline_trace"):
                            out = llm_chat("sys", "usr", max_retries=1, task="join")
        assert out == '{"a": 1}'
        mock_client.responses.create.assert_called()

    def test_azure_deployment_env_rewrites_model(self, monkeypatch):
        from aetherdialect._config import EngineConfig

        snap_p = EngineConfig.LLM_PROVIDER
        try:
            EngineConfig.LLM_PROVIDER = "azure"
            monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_MEDIUM", "dep-sql")
            monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_LIGHT", "dep0")
            monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_HEAVY", "dep2")
            mock_resp = MagicMock()
            mock_resp.output_text = "{}"
            mock_client = MagicMock()
            mock_client.responses.create.return_value = mock_resp
            with patch("aetherdialect._core_utils._provider_order", return_value=["azure"]):
                with patch(
                    "aetherdialect._core_utils._provider_is_configured",
                    return_value=True,
                ):
                    with patch(
                        "aetherdialect._core_utils._build_client",
                        return_value=mock_client,
                    ):
                        with patch("aetherdialect._core_utils.debug"):
                            with patch("aetherdialect._core_utils.pipeline_trace"):
                                llm_chat("sys", "usr", max_retries=1, task="join")
            call_kw = mock_client.responses.create.call_args.kwargs
            assert call_kw["model"] == "dep2"
        finally:
            EngineConfig.LLM_PROVIDER = snap_p

    def test_openai_ignores_azure_deployment_env(self, monkeypatch):
        from aetherdialect._config import EngineConfig

        snap_p = EngineConfig.LLM_PROVIDER
        try:
            EngineConfig.LLM_PROVIDER = "openai"
            monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_MEDIUM", "y")
            mock_resp = MagicMock()
            mock_resp.output_text = "{}"
            mock_client = MagicMock()
            mock_client.responses.create.return_value = mock_resp
            with patch("aetherdialect._core_utils._provider_order", return_value=["openai"]):
                with patch(
                    "aetherdialect._core_utils._provider_is_configured",
                    return_value=True,
                ):
                    with patch(
                        "aetherdialect._core_utils._build_client",
                        return_value=mock_client,
                    ):
                        with patch("aetherdialect._core_utils.debug"):
                            with patch("aetherdialect._core_utils.pipeline_trace"):
                                llm_chat("sys", "usr", max_retries=1, task="join")
            call_kw = mock_client.responses.create.call_args.kwargs
            assert call_kw["model"] == "gpt-5.4-mini"
        finally:
            EngineConfig.LLM_PROVIDER = snap_p

    def test_no_provider_raises(self):
        with patch("aetherdialect._core_utils._provider_order", return_value=["openai"]):
            with patch("aetherdialect._core_utils._provider_is_configured", return_value=False):
                with pytest.raises(RuntimeError, match="No configured"):
                    llm_chat("s", "u", max_retries=1)


class TestLlmJson:
    """llm_json behaviour via mocked llm_chat."""

    def test_parses_json_object(self):
        with patch("aetherdialect._core_utils.llm_chat", return_value='{"k": 1}'):
            with patch("aetherdialect._core_utils.debug"):
                assert llm_json("s", "u") == {"k": 1}

    def test_wraps_raw_select(self):
        with patch("aetherdialect._core_utils.llm_chat", return_value="SELECT 1"):
            with patch("aetherdialect._core_utils.debug"):
                d = llm_json("s", "u", retries=0)
        assert d["sql"] == "SELECT 1"
        assert d["chosen_join_candidate_id"] == "J00"

    def test_retries_exhausted_raises(self):
        calls = {"n": 0}

        def side_effect(*_a, **_k):
            calls["n"] += 1
            return "not json"

        with patch("aetherdialect._core_utils.llm_chat", side_effect=side_effect):
            with patch("aetherdialect._core_utils.debug"):
                with pytest.raises(LlmJsonExhausted) as exc_info:
                    llm_json("s", "u", retries=0, task="intent")
        assert exc_info.value.task == "intent"
        assert exc_info.value.attempts == 1

    def test_retries_exhausted_after_multiple_retries(self):
        with patch("aetherdialect._core_utils.llm_chat", return_value="still not json"):
            with patch("aetherdialect._core_utils.debug"):
                with pytest.raises(LlmJsonExhausted) as exc_info:
                    llm_json("s", "u", retries=2)
        assert exc_info.value.attempts == 3


class TestNormalizeOpWhitespaceCollapse:
    """normalize_op collapses redundant surrounding whitespace."""

    def test_padded_known_token(self):
        """Surrounding whitespace does not prevent alias lookup."""
        assert normalize_op("  LTE  ") == "<="


class TestTelemetryCapturesPipelineTrace:
    """Trace helpers append to the telemetry buffer before the print gate."""

    def test_pipeline_trace_always_appends_to_sink(self):
        with telemetry_capture(suppress_console=True) as buf:
            with (
                patch(
                    "aetherdialect._core_utils.diagnostic_debug_enabled",
                    return_value=False,
                ),
                patch(
                    "aetherdialect._core_utils.diagnostic_pipeline_trace_full_enabled",
                    return_value=False,
                ),
            ):
                pipeline_trace("ev", "data")
        assert any("[PIPELINE_TRACE] ev" in line for line in buf)
        assert any("data" in line for line in buf)

    def test_pipeline_trace_lazy_always_appends_to_sink(self):
        with telemetry_capture(suppress_console=True) as buf:
            with (
                patch(
                    "aetherdialect._core_utils.diagnostic_debug_enabled",
                    return_value=False,
                ),
                patch(
                    "aetherdialect._core_utils.diagnostic_pipeline_trace_full_enabled",
                    return_value=False,
                ),
            ):
                pipeline_trace_lazy("ev", lambda: "payload")
        joined = "\n".join(buf)
        assert "[PIPELINE_TRACE]" in joined
        assert "payload" in joined


class TestProviderOrderAndBuildClient:
    """Private LLM routing helpers (pure config + cache)."""

    def test_provider_order_openai(self):
        with patch.object(EngineConfig, "LLM_PROVIDER", "openai"):
            assert _provider_order() == ["openai"]

    def test_provider_order_azure(self):
        with patch.object(EngineConfig, "LLM_PROVIDER", "azure"):
            assert _provider_order() == ["azure"]

    def test_provider_order_unknown_falls_back_to_openai(self):
        with patch.object(EngineConfig, "LLM_PROVIDER", "bogus"):
            assert _provider_order() == ["openai"]

    def test_build_client_unsupported_raises(self):
        import aetherdialect._core_utils as cu

        prev = dict(cu._clients)
        cu._clients.clear()
        try:
            with pytest.raises(RuntimeError, match="Unsupported LLM provider"):
                _build_client("not-a-provider")
        finally:
            cu._clients.clear()
            cu._clients.update(prev)


class TestLlmChatBranches:
    """More llm_chat branches without a real API."""

    def test_second_provider_succeeds_after_first_raises(self):
        good = MagicMock()
        good.responses.create.return_value = MagicMock(output_text="  ok  ")
        bad = MagicMock()
        bad.responses.create.side_effect = ConnectionError("down")

        def pick_client(name):
            return bad if name == "openai" else good

        with patch(
            "aetherdialect._core_utils._provider_order",
            return_value=["openai", "azure"],
        ):
            with patch("aetherdialect._core_utils._provider_is_configured", return_value=True):
                with patch("aetherdialect._core_utils._build_client", side_effect=pick_client):
                    with patch("aetherdialect._core_utils.debug"):
                        with patch("aetherdialect._core_utils.pipeline_trace"):
                            with patch("aetherdialect._core_utils.log"):
                                out = llm_chat("s", "u", max_retries=1, task="join")
        assert out == "ok"
        assert bad.responses.create.call_count == 1
        assert good.responses.create.call_count == 1

    def test_all_providers_fail_then_runtime_error(self):
        client = MagicMock()
        client.responses.create.side_effect = ValueError("nope")

        with patch("aetherdialect._core_utils._provider_order", return_value=["openai"]):
            with patch("aetherdialect._core_utils._provider_is_configured", return_value=True):
                with patch("aetherdialect._core_utils._build_client", return_value=client):
                    with patch("aetherdialect._core_utils.debug"):
                        with patch("aetherdialect._core_utils.pipeline_trace"):
                            with patch("aetherdialect._core_utils.log"):
                                with patch("aetherdialect._core_utils.time.sleep"):
                                    with pytest.raises(RuntimeError, match="LLM call failed"):
                                        llm_chat("s", "u", max_retries=2, task="join")
        assert client.responses.create.call_count == 2

    def test_intent_task_passes_reasoning_kwarg(self):
        mock_resp = MagicMock()
        mock_resp.output_text = "{}"
        client = MagicMock()
        client.responses.create.return_value = mock_resp

        with patch("aetherdialect._core_utils._provider_order", return_value=["openai"]):
            with patch("aetherdialect._core_utils._provider_is_configured", return_value=True):
                with patch("aetherdialect._core_utils._build_client", return_value=client):
                    with patch("aetherdialect._core_utils.debug"):
                        with patch("aetherdialect._core_utils.pipeline_trace"):
                            llm_chat("s", "u", max_retries=1, task="intent")

        kwargs = client.responses.create.call_args.kwargs
        assert "reasoning" in kwargs
        assert "temperature" not in kwargs

    def test_join_task_passes_reasoning_not_temperature(self):
        mock_resp = MagicMock()
        mock_resp.output_text = "{}"
        client = MagicMock()
        client.responses.create.return_value = mock_resp

        with patch("aetherdialect._core_utils._provider_order", return_value=["openai"]):
            with patch("aetherdialect._core_utils._provider_is_configured", return_value=True):
                with patch("aetherdialect._core_utils._build_client", return_value=client):
                    with patch("aetherdialect._core_utils.debug"):
                        with patch("aetherdialect._core_utils.pipeline_trace"):
                            llm_chat("s", "u", max_retries=1, task="join")

        kwargs = client.responses.create.call_args.kwargs
        assert "reasoning" in kwargs
        assert "temperature" not in kwargs


class TestLlmJsonBranches:
    """llm_json paths beyond the first successful dict parse."""

    def test_retry_returns_dict(self):
        replies = iter(["not json", '{"fixed": true}'])

        def fake_chat(*_a, **_k):
            return next(replies)

        with patch("aetherdialect._core_utils.llm_chat", side_effect=fake_chat):
            with patch("aetherdialect._core_utils.debug"):
                assert llm_json("s", "u", retries=1) == {"fixed": True}

    def test_retry_wraps_select(self):
        replies = iter(["x", "SELECT 2"])

        def fake_chat(*_a, **_k):
            return next(replies)

        with patch("aetherdialect._core_utils.llm_chat", side_effect=fake_chat):
            with patch("aetherdialect._core_utils.debug"):
                d = llm_json("s", "u", retries=1)
        assert d["sql"] == "SELECT 2"
        assert d["chosen_join_candidate_id"] == "J00"

    def test_json_array_triggers_retry_then_dict(self):
        replies = iter(["[1, 2]", '{"only": "dict"}'])

        def fake_chat(*_a, **_k):
            return next(replies)

        with patch("aetherdialect._core_utils.llm_chat", side_effect=fake_chat):
            with patch("aetherdialect._core_utils.debug"):
                assert llm_json("s", "u", retries=1) == {"only": "dict"}


class TestSubstituteParamsPlusZero:
    """sql_simplify_executable strips + 0 as well as - 0."""

    def test_plus_zero_removed(self):
        result = sql_simplify_executable("SELECT col + 0 FROM t", sqlglot_dialect="postgres")
        assert "+ 0" not in result


class TestPrintInfoFooterOnly:
    """print_info with only footer."""

    def test_footer_without_items(self, capsys):
        with diagnostic_print_listener(print):
            print_info("Title", items=None, footer="Foot note")
        out = capsys.readouterr().out
        assert "Title" in out
        assert "Foot note" in out


class TestWriteGzipJsonAtomicFailure:
    """write_gzip_json_atomic cleans up temp file on replace failure."""

    def test_unlinks_temp_when_replace_fails(self, tmp_path):
        path = tmp_path / "out.json.gz"
        with patch(
            "aetherdialect._core_utils.os.replace",
            side_effect=OSError("replace failed"),
        ):
            with pytest.raises(OSError, match="replace failed"):
                write_gzip_json_atomic(str(path), {"a": 1}, sort_keys=True)
        assert not path.is_file()
        assert not list(tmp_path.glob(".tmp_*.json.gz"))


class TestAskUserChoiceMore:
    """Extra stdin normalisation cases."""

    def test_yes_full_word_case_insensitive(self, capsys):
        with patch("aetherdialect._core_utils.input", return_value="YES"):
            assert ask_user_choice("?", ["y", "n"]) == "y"

    def test_no_prints_terminated_when_not_silent(self, capsys):
        with diagnostic_print_listener(print):
            with patch("aetherdialect._core_utils.input", return_value="no"):
                assert ask_user_choice("?", ["y", "n"], silent_no=False) == "n"
        assert "User terminated." in capsys.readouterr().out


class TestSafeJsonLoadsNonDictFragment:
    """safe_json_loads when extracted object is valid JSON but not useful to llm_json."""

    def test_fragment_array_not_used_by_extract_first_object(self):
        assert safe_json_loads("prefix [1, 2] suffix") is None


class TestPrintRephraseHint:
    """print_rephrase_hint emits a tailored, non-technical message per reason."""

    def test_intent_parse_failed_suggests_specific_tables(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.INTENT_PARSE_FAILED)
        out = capsys.readouterr().out
        assert "Please rephrase" in out
        assert "tables" in out.lower() or "columns" in out.lower()

    def test_schema_invalid_declined_suggests_existing_tables(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.SCHEMA_INVALID_DECLINED)
        out = capsys.readouterr().out
        assert "exist" in out.lower()

    def test_sql_validation_failed_suggests_simpler_query(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.SQL_VALIDATION_FAILED)
        out = capsys.readouterr().out
        assert "retry" in out.lower() or "simpl" in out.lower()

    def test_user_rejected_intent_mentions_specificity(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.USER_REJECTED_INTENT)
        out = capsys.readouterr().out
        assert "specific" in out.lower()

    def test_user_rejected_result_is_actionable(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.USER_REJECTED_RESULT)
        out = capsys.readouterr().out
        assert "retry" in out.lower() or "rephrase" in out.lower()

    def test_user_rejected_result_bucket_tip(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.USER_REJECTED_RESULT, rejection_bucket="MISSING_FILTER")
        out = capsys.readouterr().out
        assert "filter" in out.lower()

    def test_restricted_question_hint(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.RESTRICTED_QUESTION)
        out = capsys.readouterr().out.lower()
        assert "deny_columns" in out or "allow_columns" in out

    def test_vague_question_hint(self, capsys):
        with diagnostic_print_listener(print):
            print_rephrase_hint(RephraseHint.VAGUE_QUESTION)
        out = capsys.readouterr().out.lower()
        assert "metric" in out or "filter" in out


class TestNotify:
    """``notify`` writes a plain status line to stdout."""

    def test_writes_message(self, capsys):
        with diagnostic_print_listener(print):
            notify("hello world")
        assert capsys.readouterr().out == "hello world\n"

    def test_no_prefix(self, capsys):
        with diagnostic_print_listener(print):
            notify("plain")
        out = capsys.readouterr().out
        assert not out.startswith("Error")
        assert not out.startswith("!")


class TestResult:
    """``result`` writes a query-result line to stdout."""

    def test_writes_message(self, capsys):
        with diagnostic_print_listener(print):
            result("col1 | col2")
        assert capsys.readouterr().out == "col1 | col2\n"


class TestWarn:
    """``warn`` prefixes with ``! ``."""

    def test_warn_prefix(self, capsys):
        with diagnostic_print_listener(print):
            warn("careful")
        assert capsys.readouterr().out == "! careful\n"


class TestError:
    """``error`` prefixes with ``Error: ``."""

    def test_error_prefix(self, capsys):
        with diagnostic_print_listener(print):
            error("boom")
        assert capsys.readouterr().out == "Error: boom\n"


class TestDataframeToRowTuples:
    """``dataframe_to_row_tuples`` prepares rows for ``print_query_result``."""

    def test_none_returns_empty(self) -> None:
        assert dataframe_to_row_tuples(None) == []

    def test_dataframe_values(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert dataframe_to_row_tuples(df) == [(1, "x"), (2, "y")]


class TestTerminated:
    """``terminated`` writes the canonical termination line."""

    def test_writes_user_terminated(self, capsys):
        with diagnostic_print_listener(print):
            terminated()
        assert capsys.readouterr().out == "\nUser terminated.\n"


class TestInvalidInput:
    """``invalid_input`` writes the canonical invalid-input line."""

    def test_writes_invalid_input(self, capsys):
        with diagnostic_print_listener(print):
            invalid_input()
        assert capsys.readouterr().out == "\nInvalid input.\n"

    def test_writes_custom_detail(self, capsys):
        with diagnostic_print_listener(print):
            invalid_input("Please answer y or n.")
        assert capsys.readouterr().out.strip() == "Please answer y or n."


class TestPrompt:
    """``prompt`` reads from stdin and strips whitespace."""

    def test_strips_whitespace(self):
        with patch("aetherdialect._core_utils.input", return_value="  hi  "):
            assert prompt("Q? ") == "hi"

    def test_propagates_eof(self):
        with patch("aetherdialect._core_utils.input", side_effect=EOFError()):
            with pytest.raises(EOFError):
                prompt("Q? ")

    def test_propagates_keyboard_interrupt(self):
        with patch("aetherdialect._core_utils.input", side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                prompt("Q? ")


class TestLlmUserSensitivityStrip:
    """``_llm_user_text_without_sensitivity_classification`` removes tier keys from JSON user payloads."""

    def test_strips_nested_sensitivity_and_pii(self) -> None:
        payload = '{"tables":{"t":{"columns":{"c":{"role":"x","sensitivity":"strict","pii":null}}}}}'
        out = _llm_user_text_without_sensitivity_classification(payload)
        data = json.loads(out)
        col = data["tables"]["t"]["columns"]["c"]
        assert "sensitivity" not in col
        assert "pii" not in col
        assert col["role"] == "x"

    def test_plain_text_unchanged(self) -> None:
        s = "not json at all"
        assert _llm_user_text_without_sensitivity_classification(s) is s
