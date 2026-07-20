"""Generic contract tests for every registered dialect implementation."""

from __future__ import annotations

from typing import Any

import pytest

from aetherdialect._constants import (
    NATIVE_BACKEND_ENGINES,
    QUALIFIED_TABLE_REF_ENGINES,
    RESULT_READER_KINDS,
    SQLGLOT_DIALECT_HOOK_ENGINES,
    STRUCTURAL_INDEX_ENGINES,
)
from aetherdialect._dialect import Dialect, get_registered_engines, register_dialect

_REQUIRED_METHODS: tuple[str, ...] = (
    "ast_validate_full",
    "parse_select",
    "ordered_join_carrier_froms",
    "attach_joins",
    "attach_extra_from_and_where",
    "from_anchor_of",
    "replace_projection",
    "emit_sql",
    "explain_diagnose",
    "execute",
    "reflect_schema_graph",
    "profile_schema",
    "quote_table_column",
)

_SQLGLOT_HOOK_METHODS: tuple[str, ...] = (
    "qualified_table_ref",
    "structural_constraints_index",
    "post_render_normalize",
    "pre_execute_rewrite",
    "profile_schema_dispatch",
    "finalize_render",
    "apply_execute_cost_limits",
)

_QUOTE_FIXTURES: dict[str, str] = {
    "mysql": "`fixture`.`column`",
    "redshift": '"fixture"."column"',
    "sqlserver": "[fixture].[column]",
    "snowflake": "FIXTURE.COLUMN",
    "bigquery": "`fixture`.`column`",
    "postgresql": '"fixture"."column"',
    "databricks": "`fixture`.`column`",
}

_EXPLAIN_PREFIX: dict[str, str] = {
    "mysql": "EXPLAIN FORMAT=JSON",
    "redshift": "EXPLAIN ",
    "snowflake": "EXPLAIN USING JSON",
    "postgresql": "EXPLAIN (FORMAT JSON",
}

_QUERY_LOG_ENGINES: frozenset[str] = frozenset(
    {
        "mysql",
        "redshift",
        "sqlserver",
        "snowflake",
        "bigquery",
        "postgresql",
        "databricks",
    },
)

_PROFILE_TIMEOUT_ENGINES: dict[str, str] = {
    "mysql": "MAX_EXECUTION_TIME",
    "redshift": "statement_timeout",
    "snowflake": "STATEMENT_TIMEOUT_IN_SECONDS",
    "postgresql": "statement_timeout",
}

_SAMPLE_SUFFIX_ENGINES: frozenset[str] = frozenset(
    {"mysql", "redshift", "sqlserver", "snowflake", "bigquery", "databricks"},
)

_SAMPLE_SUFFIX_WHERE_ENGINES: frozenset[str] = frozenset({"mysql", "redshift"})


def _uninit_dialect(cls: type[Dialect]) -> Dialect:
    return cls.__new__(cls)


def _method_is_overridden(dialect: Dialect, name: str) -> bool:
    impl = getattr(type(dialect), name, None)
    base_impl = getattr(Dialect, name, None)
    return impl is not base_impl


def _assert_dialect_contract(dialect: Dialect, engine: str) -> None:
    for name in _REQUIRED_METHODS:
        assert _method_is_overridden(dialect, name), f"{engine}.{name} must override base Dialect"

    assert isinstance(dialect.sqlglot_dialect, str)
    assert dialect.sqlglot_dialect or engine == "postgresql"
    assert isinstance(dialect.dialect_label, str) and dialect.dialect_label
    assert isinstance(dialect.extra_filter_ops(), frozenset)
    assert dialect.result_reader_kind in RESULT_READER_KINDS
    assert isinstance(dialect.inject_pruning_predicates("SELECT 1", schema=None, intent=None), str)
    assert isinstance(dialect.date_window_upper_bound_sql("day"), str) and dialect.date_window_upper_bound_sql("day")
    assert isinstance(dialect.supports_ilike, bool)

    expected = _QUOTE_FIXTURES.get(engine)
    if expected is not None:
        assert dialect.quote_table_column("fixture", "column") == expected

    explain_prefix = _EXPLAIN_PREFIX.get(engine)
    if explain_prefix is not None and _method_is_overridden(dialect, "explain_statement_sql"):
        assert dialect.explain_statement_sql("SELECT 1").startswith(explain_prefix)

    if engine in _QUERY_LOG_ENGINES:
        src = dialect.query_log_source()
        assert src is not None

    timeout_fragment = _PROFILE_TIMEOUT_ENGINES.get(engine)
    if timeout_fragment is not None:
        sql = dialect.profile_statement_timeout_sql(30_000)
        assert sql is not None and timeout_fragment in sql

    if engine in _SAMPLE_SUFFIX_ENGINES:
        suffix = dialect.profiling_stats_sample_suffix(
            use_sample=True,
            row_count=1000,
            sample_size=100,
            random_seed=1,
        )
        assert suffix
        if engine in _SAMPLE_SUFFIX_WHERE_ENGINES:
            assert suffix.startswith("WHERE ")
        elif engine == "databricks":
            assert "TABLESAMPLE" in suffix
        else:
            assert "SAMPLE" in suffix.upper() or "TABLESAMPLE" in suffix.upper()

    if engine in SQLGLOT_DIALECT_HOOK_ENGINES:
        for name in _SQLGLOT_HOOK_METHODS:
            assert _method_is_overridden(dialect, name), f"{engine}.{name} must override base Dialect"
        assert isinstance(dialect.post_render_normalize("SELECT 1", stage="post_substitute"), str)
        assert isinstance(dialect.pre_execute_rewrite("SELECT 1"), str)
        assert dialect.structural_constraints_index() is not None

    if engine in QUALIFIED_TABLE_REF_ENGINES:
        ref = dialect.qualified_table_ref("fixture")
        assert "fixture" in ref.lower()

    if engine in STRUCTURAL_INDEX_ENGINES:
        idx = dialect.structural_constraints_index()
        assert hasattr(idx, "tables")

    if engine in NATIVE_BACKEND_ENGINES:
        assert dialect.result_reader_kind in RESULT_READER_KINDS


@pytest.mark.parametrize("engine", get_registered_engines())
def test_registered_dialect_satisfies_contract(engine: str) -> None:
    """Every registered dialect must implement the full override contract."""
    from aetherdialect._dialect import get_dialect_class

    cls = get_dialect_class(engine)
    dialect = _uninit_dialect(cls)
    _assert_dialect_contract(dialect, engine)


class _StubDialect(Dialect):
    """Deliberately incomplete dialect used to prove the contract harness detects gaps."""

    name = "_stub"
    sqlglot_dialect = "postgres"


@pytest.fixture
def stub_dialect_registration():
    register_dialect("_stub", _StubDialect)
    yield
    from aetherdialect._dialect import _DIALECT_REGISTRY, _RUNTIME_REGISTRY

    _DIALECT_REGISTRY.pop("_stub", None)
    _RUNTIME_REGISTRY.pop("_stub", None)


def test_stub_dialect_fails_contract(stub_dialect_registration: Any) -> None:
    """The contract harness must reject a dialect that only sets ``name``."""
    stub = _uninit_dialect(_StubDialect)
    missing = [name for name in _REQUIRED_METHODS if not _method_is_overridden(stub, name)]
    assert missing, "stub dialect should leave required methods on the base class"
    with pytest.raises(AssertionError):
        _assert_dialect_contract(stub, "_stub")


def test_sqlglot_hook_engine_sets_cover_config() -> None:
    """Registered sqlglot engines match the SQLGLOT_DIALECT_HOOK_ENGINES contract set."""
    registered = set(get_registered_engines())
    assert SQLGLOT_DIALECT_HOOK_ENGINES <= registered
    assert "postgresql" not in SQLGLOT_DIALECT_HOOK_ENGINES
