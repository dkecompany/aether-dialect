"""Orphan diagnostics are keyed to the engine that produced them."""

from __future__ import annotations

import pytest

from aetherdialect._config import DuckDBRuntimeConfig
from aetherdialect._contracts_base import EngineIdentity
from aetherdialect._core_utils import (
    notify,
    pop_engine_identity,
    push_engine_identity,
    reset_diagnostic_collector,
    set_diagnostic_collector,
    take_and_clear_orphan_diagnostics,
)


@pytest.mark.fast
def test_entries_reach_their_own_engine(unbound_engine_identity: None) -> None:
    runtime_a = DuckDBRuntimeConfig()
    runtime_b = DuckDBRuntimeConfig()
    identity_a = EngineIdentity("duckdb", runtime_a)
    identity_b = EngineIdentity("duckdb", runtime_b)

    collector_token = set_diagnostic_collector(None)
    try:
        token_a = push_engine_identity(identity_a)
        try:
            notify("construction orphan A", code="ORPHAN_TEST_A")
        finally:
            pop_engine_identity(token_a)

        token_b = push_engine_identity(identity_b)
        try:
            notify("construction orphan B", code="ORPHAN_TEST_B")
        finally:
            pop_engine_identity(token_b)

        drained_a = take_and_clear_orphan_diagnostics(identity_a)
        assert [d.code for d in drained_a] == ["ORPHAN_TEST_A"]

        drained_b = take_and_clear_orphan_diagnostics(identity_b)
        assert [d.code for d in drained_b] == ["ORPHAN_TEST_B"]
    finally:
        reset_diagnostic_collector(collector_token)
