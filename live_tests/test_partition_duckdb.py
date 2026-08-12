"""DuckDB live partition metadata checks for the rental_shop synthetic partition graph."""

from __future__ import annotations

import pytest

from .live_support import build_engine_t2s, skip_unless_configured

_ENGINE = "duckdb"
_SKIP_REASON = skip_unless_configured(_ENGINE)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


@pytest.fixture(scope="module")
def t2s():
    return build_engine_t2s(_ENGINE, "rental_shop")


def test_rental_partition_metadata_present(t2s) -> None:
    """Reflected rental_shop graph carries synthetic partition columns on rental."""
    rental = t2s._schema_graph.tables.get("rental")
    assert rental is not None
    assert "rental_date" in rental.partition_columns
