"""Construction-time orphan diagnostics must not bleed across engines."""

from __future__ import annotations

import pytest

from aetherdialect._config import DuckDBRuntimeConfig
from aetherdialect._contracts_base import EngineIdentity
from aetherdialect._core_utils import (
    bind_construction_orphan_identity,
    notify,
    release_construction_orphan_identity,
    reset_diagnostic_collector,
    set_diagnostic_collector,
    take_and_clear_orphan_diagnostics,
)
from aetherdialect._main_execution import PipelineSession


@pytest.mark.fast
def test_second_engine_ask_does_not_ingest_first_engine_orphans(unbound_engine_identity: None) -> None:
    identity_a = EngineIdentity("duckdb", DuckDBRuntimeConfig())
    identity_b = EngineIdentity("duckdb", DuckDBRuntimeConfig())

    collector_token = set_diagnostic_collector(None)
    try:
        token_a = bind_construction_orphan_identity(identity_a)
        try:
            notify("engine A construction orphan", code="ORPHAN_ENGINE_A")
        finally:
            release_construction_orphan_identity(token_a)

        owner_b = type("Owner", (), {})()
        owner_b._engine_identity = identity_b
        owner_b._dialect = None
        owner_b._runtime_config = identity_b.runtime_config
        owner_b._sandbox_closed = False
        session = PipelineSession(owner_b, mode="writer")

        buf: list = []
        for diag in take_and_clear_orphan_diagnostics(session._owner_engine_identity()):
            buf.append(diag)

        assert [d.code for d in buf] == []
        assert [d.code for d in take_and_clear_orphan_diagnostics(identity_a)] == ["ORPHAN_ENGINE_A"]
    finally:
        reset_diagnostic_collector(collector_token)
