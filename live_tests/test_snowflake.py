"""Snowflake live connection and reflection tests. Dialect-syntax coverage runs in ``test_snowflake_dialect.py``."""

from __future__ import annotations

import pytest

from .live_support import (
    build_engine_t2s,
    engine_schema,
    skip_unless_configured,
)

_ENGINE = "snowflake"
_SKIP_REASON = skip_unless_configured(_ENGINE)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

_EXPECTED_CORE_TABLES = ("film", "customer", "rental")


@pytest.fixture(scope="module")
def t2s():
    return build_engine_t2s(_ENGINE, engine_schema("SNOWFLAKE_DATABASE", "rental_shop"))


def test_snowflake_connection_and_reflection(t2s):
    reflected = {name.lower() for name in t2s._schema_graph.tables}
    for table in _EXPECTED_CORE_TABLES:
        assert table in reflected
