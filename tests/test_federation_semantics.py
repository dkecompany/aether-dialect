"""Logical table semantics are limited to union and replica with explicit errors."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationConfigError
from aetherdialect._federation import parse_federation_mappings


@pytest.mark.fast
@pytest.mark.parametrize("semantics", ["partition", "shard"])
def test_unsupported_logical_table_semantics_names_supported_set(semantics: str) -> None:
    with pytest.raises(FederationConfigError) as exc_info:
        parse_federation_mappings(
            {
                "version": "0.2.1",
                "logical_tables": [
                    {
                        "logical": "orders",
                        "semantics": semantics,
                        "members": [{"source": "a", "table": "orders", "columns": {}}],
                    },
                ],
            },
        )
    message = str(exc_info.value)
    assert semantics in message
    assert "union" in message
    assert "replica" in message
