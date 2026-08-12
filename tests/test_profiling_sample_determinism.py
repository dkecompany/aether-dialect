"""Profiling sample reproducibility and per-engine sampling behavior."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DEFAULT_RANDOM_SEED
from aetherdialect._dialect import Dialect, DialectRegistry
from aetherdialect._schema_profile import _build_profile_stats_sql


def _uninit(cls: type) -> object:
    return cls.__new__(cls)


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
    assert "ORDER BY" in suffix.upper()
    stats_sql = _build_profile_stats_sql(
        '"col"',
        '"tbl"',
        use_sample=True,
        sample_clause=suffix,
        use_subquery=True,
    )
    assert "ORDER BY" in stats_sql.upper()
    assert "LIMIT 10000" in stats_sql


@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_profiling_sample_engine_behavior_documented(engine: str) -> None:
    """Document which engines honor random_seed vs probabilistic sampling."""
    dialect = DialectRegistry.get_class(engine).__new__(DialectRegistry.get_class(engine))
    suffix = dialect.profiling_stats_sample_suffix(
        use_sample=True,
        row_count=250_000,
        sample_size=10_000,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    assert suffix
    honors_seed = (
        "REPEATABLE" in suffix.upper()
        or " SEED " in suffix.upper()
        or "FNV_HASH(" in suffix.upper()
        or "HASH(" in suffix.upper()
        or f"RAND({DEFAULT_RANDOM_SEED})" in suffix
        or suffix.upper().startswith("ORDER BY")
        or f", {DEFAULT_RANDOM_SEED})" in suffix
    )
    probabilistic = any(token in suffix.upper() for token in ("RAND", "BERNOULLI", "SAMPLE", "TABLESAMPLE"))
    if engine == "postgresql":
        assert honors_seed
    elif engine in {"mysql", "mariadb"}:
        assert suffix.startswith("WHERE ")
        assert probabilistic and honors_seed
    elif engine in {"redshift", "sqlite"}:
        assert suffix.startswith("WHERE ")
        assert honors_seed
    elif engine in {"duckdb", "snowflake", "databricks", "csv"}:
        assert honors_seed
    elif engine in {"bigquery", "sqlserver", "oracle"}:
        assert suffix.upper().startswith("ORDER BY")
    else:
        assert honors_seed or probabilistic
