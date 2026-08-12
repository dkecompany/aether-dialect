"""Profiling sample SQL must be deterministic on every engine."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DEFAULT_RANDOM_SEED
from aetherdialect._dialect import Dialect, DialectRegistry
from aetherdialect._schema_profile import _build_profile_stats_sql

_SEEDED_RAND_ENGINES = frozenset({"mysql", "mariadb"})
_SEEDED_TABLESAMPLE_ENGINES = frozenset({"postgresql", "snowflake", "duckdb", "csv", "databricks"})
_HASH_PREDICATE_ENGINES = frozenset({"redshift", "sqlite"})
_ORDERED_LIMIT_ENGINES = frozenset({"bigquery", "sqlserver", "oracle"})


def _uninit(cls: type) -> object:
    return cls.__new__(cls)


def _suffix(engine: str, *, seed: int = DEFAULT_RANDOM_SEED) -> str:
    dialect = DialectRegistry.get_class(engine).__new__(DialectRegistry.get_class(engine))
    return dialect.profiling_stats_sample_suffix(
        use_sample=True,
        row_count=250_000,
        sample_size=10_000,
        random_seed=seed,
    )


def _assert_no_bare_random_tokens(suffix: str) -> None:
    upper = suffix.upper()
    assert "RAND()" not in upper
    assert "RANDOM()" not in upper
    assert "ABS(RANDOM())" not in upper


@pytest.mark.fast
@pytest.mark.parametrize("engine", sorted(_SEEDED_RAND_ENGINES))
def test_seeded_rand_engines_embed_random_seed(engine: str) -> None:
    seed = 42
    suffix = _suffix(engine, seed=seed)
    assert suffix.startswith("WHERE ")
    assert f"RAND({seed})" in suffix
    _assert_no_bare_random_tokens(suffix)


@pytest.mark.fast
@pytest.mark.parametrize("engine", sorted(_HASH_PREDICATE_ENGINES))
def test_hash_predicate_engines_embed_seed_and_column_placeholder(engine: str) -> None:
    seed = 42
    suffix = _suffix(engine, seed=seed)
    assert suffix.startswith("WHERE ")
    assert "{col}" in suffix or "HASH(" in suffix.upper() or "FNV_HASH(" in suffix.upper()
    assert str(seed) in suffix
    _assert_no_bare_random_tokens(suffix)


@pytest.mark.fast
@pytest.mark.parametrize("engine", sorted(_SEEDED_TABLESAMPLE_ENGINES))
def test_seeded_tablesample_engines_embed_random_seed(engine: str) -> None:
    seed = 42
    suffix = _suffix(engine, seed=seed)
    upper = suffix.upper()
    assert "REPEATABLE" in upper or "SEED" in upper or f", {seed})" in suffix
    _assert_no_bare_random_tokens(suffix)


@pytest.mark.fast
@pytest.mark.parametrize("engine", sorted(_ORDERED_LIMIT_ENGINES))
def test_unseedable_tablesample_engines_use_ordered_limit(engine: str) -> None:
    suffix = _suffix(engine)
    assert suffix.upper().startswith("ORDER BY")
    assert "LIMIT 10000" in suffix
    _assert_no_bare_random_tokens(suffix)


@pytest.mark.fast
def test_default_dialect_sample_suffix_uses_ordered_limit_not_bare_row_cap() -> None:
    dialect = _uninit(Dialect)
    suffix = dialect.profiling_stats_sample_suffix(
        use_sample=True,
        row_count=250_000,
        sample_size=10_000,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    assert suffix
    assert suffix != "LIMIT 10000"
    assert suffix.upper().startswith("ORDER BY")
    assert "LIMIT 10000" in suffix


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine",
    sorted(_SEEDED_RAND_ENGINES | _HASH_PREDICATE_ENGINES | _SEEDED_TABLESAMPLE_ENGINES),
)
def test_profiling_sample_suffix_varies_with_seed(engine: str) -> None:
    suffix_a = _suffix(engine, seed=11)
    suffix_b = _suffix(engine, seed=29)
    assert suffix_a != suffix_b


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_profiling_stats_sql_is_deterministic_for_large_tables(engine: str) -> None:
    dialect = DialectRegistry.get_class(engine).__new__(DialectRegistry.get_class(engine))
    suffix = dialect.profiling_stats_sample_suffix(
        use_sample=True,
        row_count=250_000,
        sample_size=10_000,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    assert suffix
    stats_sql = _build_profile_stats_sql(
        '"col"',
        '"tbl"',
        use_sample=True,
        sample_clause=suffix,
        use_subquery=dialect.profiling_stats_use_subquery_when_sampling(),
    )
    _assert_no_bare_random_tokens(stats_sql)
    if engine in _HASH_PREDICATE_ENGINES:
        assert '"col"' in stats_sql
        assert "{col}" not in stats_sql
    elif engine in _ORDERED_LIMIT_ENGINES:
        assert "ORDER BY" in stats_sql.upper()
        assert "LIMIT 10000" in stats_sql
    else:
        upper = stats_sql.upper()
        assert "LIMIT 10000" in stats_sql or "SAMPLE" in upper or "TABLESAMPLE" in upper or "WHERE" in upper
