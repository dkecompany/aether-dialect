"""Schema cache hits must probe live database freshness, not only self- consistency."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

import aetherdialect._schema_overrides
from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._core_utils import stable_json
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import assign_schema_graph_hashes
from aetherdialect._schema_overrides import save_schema_to_cache

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _FreshnessStubDialect(Dialect):
    """Dialect with controllable DDL and row-count probes for cache- freshness tests."""

    name = "stub"

    def __init__(
        self,
        *,
        ddl_probe: str = "",
        live_row_counts: dict[str, int] | None = None,
        reflect_result: SchemaGraph | None = None,
    ) -> None:
        super().__init__(MagicMock())
        self._ddl_probe = ddl_probe
        self._live_row_counts = live_row_counts or {}
        self._reflect_result = reflect_result
        self.reflect_calls = 0
        self.profile_calls = 0
        self.row_count_probe_calls = 0
        self.ddl_probe_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        _ = engine_context
        self.ddl_probe_calls += 1
        return self._ddl_probe

    def compute_row_count_probe(self, sg: SchemaGraph) -> str:
        self.row_count_probe_calls += 1
        payload = {
            name: int(self._live_row_counts.get(name, sg.tables[name].row_count or 0)) for name in sorted(sg.tables)
        }
        return hashlib.sha256(stable_json(payload).encode()).hexdigest()

    def reflect_only(self, schema_context: EngineContext) -> SchemaGraph:
        self.reflect_calls += 1
        if self._reflect_result is None:
            raise AssertionError("reflect_only should not run in this test")
        return self._reflect_result

    def reflect_schema_graph(self, **kwargs: Any) -> SchemaGraph:
        self.reflect_calls += 1
        if self._reflect_result is None:
            raise AssertionError("reflect_schema_graph should not run in this test")
        return self._reflect_result

    def profile_schema(self, sg: SchemaGraph) -> None:
        self.profile_calls += 1
        for tname, cnt in self._live_row_counts.items():
            if tname in sg.tables:
                sg.tables[tname].row_count = cnt
                for col in sg.tables[tname].columns.values():
                    col.row_count = cnt

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _patch_build_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aetherdialect._schema_overrides, "finalize_with_overrides", lambda *a, **k: False)
    monkeypatch.setattr(aetherdialect._schema_overrides, "notify_schema_path_health", lambda sg: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "validate_scope_against_graph", lambda sg, c: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "raise_if_schema_unusable", lambda sg, c: None)


def _save_legacy_fingerprint_cache(
    schema_graph: SchemaGraph,
    cache_path: str,
    *,
    notes: str = "",
) -> None:
    """Persist a legacy cache with no ddl_probe_hash (fingerprint self- check path only)."""
    ctx = EngineContext()
    schema_graph.notes_sha256 = hashlib.sha256(notes.encode()).hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)
    schema_graph.ddl_probe_hash = ""
    save_schema_to_cache(schema_graph, cache_path)


@pytest.mark.fast
def test_fingerprint_cache_hit_probes_live_row_counts_on_drift(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fingerprint-only cache hit must compare live row counts, not profiling_hash alone."""
    notes = ""
    _save_legacy_fingerprint_cache(schema_graph, cache_path, notes=notes)

    dialect = _FreshnessStubDialect(ddl_probe="", live_row_counts={"customers": 9999})
    _patch_build_helpers(monkeypatch)

    sg, diff = aetherdialect._schema_overrides.build_schema_graph_with_diff(
        dialect,
        EngineContext(),
        notes_content=notes,
    )

    assert diff is None
    assert dialect.reflect_calls == 0
    assert dialect.row_count_probe_calls >= 1
    assert dialect.profile_calls == 1
    assert sg.tables["customers"].row_count == 9999


@pytest.mark.fast
def test_fingerprint_cache_hit_skips_reprofile_when_live_row_counts_match(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live row-count probe that matches cached statistics must not re- profile."""
    notes = ""
    _save_legacy_fingerprint_cache(schema_graph, cache_path, notes=notes)

    dialect = _FreshnessStubDialect(ddl_probe="")
    _patch_build_helpers(monkeypatch)

    sg, diff = aetherdialect._schema_overrides.build_schema_graph_with_diff(
        dialect,
        EngineContext(),
        notes_content=notes,
    )

    assert diff is None
    assert dialect.reflect_calls == 0
    assert dialect.row_count_probe_calls >= 1
    assert dialect.profile_calls == 0
    assert sg.tables["customers"].row_count == 100


@pytest.mark.fast
def test_fingerprint_cache_hit_refuses_stale_cache_when_ddl_probe_drift_detected(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-consistent fingerprints must not mask live DDL drift when a probe is available."""
    import copy

    from aetherdialect._contracts_schema import ColumnMetadata

    notes = ""
    _save_legacy_fingerprint_cache(schema_graph, cache_path, notes=notes)

    live_struct = copy.deepcopy(schema_graph)
    live_struct.tables["customers"].columns["loyalty_tier"] = ColumnMetadata(
        name="loyalty_tier",
        data_type="varchar",
        value_type="string",
        row_count=100,
    )

    dialect = _FreshnessStubDialect(
        ddl_probe="LIVE_DDL_DIGEST",
        reflect_result=live_struct,
    )
    _patch_build_helpers(monkeypatch)
    monkeypatch.setattr(
        aetherdialect._schema_overrides,
        "compute_dialect_probe",
        lambda _d, _c: "LIVE_DDL_DIGEST",
    )
    monkeypatch.setattr(aetherdialect._schema_overrides, "_add_profiling_data", lambda *a, **k: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "migrate_sidecar_for_diff", lambda *a, **k: None)

    sg, diff = aetherdialect._schema_overrides.build_schema_graph_with_diff(
        dialect,
        EngineContext(),
        notes_content=notes,
    )

    assert dialect.reflect_calls >= 1
    assert diff is not None
    assert "loyalty_tier" in sg.tables["customers"].columns
