"""Bundle I fixes: provenance round-trip, sidecar replay, hidden-column skip, profiling cache."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import aetherdialect._schema_overrides
from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import SCHEMA_OVERRIDES_SIDECAR_FILENAME, SCHEMA_OVERRIDES_VERSION
from aetherdialect._contracts_base import DescriptionOwner, RoleOwner
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SensitivityClassification
from aetherdialect._core_utils import stable_json
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import assign_schema_graph_hashes, compute_dialect_probe
from aetherdialect._schema_overrides import (
    apply_overrides_and_persist,
    apply_schema_overrides_to_graph,
    build_schema_graph,
    dump_schema_overrides_dict,
    finalize_with_overrides,
    save_overrides_sidecar,
    save_schema_to_cache,
)

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


def _ov_doc(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
    }
    base.update(kwargs)
    return base


class _ProfilingDriftStubDialect(Dialect):
    name = "stub"

    def __init__(self, probe_value: str, live_row_counts: dict[str, int] | None = None) -> None:
        super().__init__(MagicMock())
        self._probe_value = probe_value
        self._live_row_counts = live_row_counts or {}
        self.profile_calls = 0

    def compute_ddl_probe(self, engine_context: Any) -> str:
        return self._probe_value

    def compute_row_count_probe(self, sg: SchemaGraph) -> str:
        payload = {
            name: int(self._live_row_counts.get(name, sg.tables[name].row_count or 0)) for name in sorted(sg.tables)
        }
        return hashlib.sha256(stable_json(payload).encode()).hexdigest()

    def reflect_schema_graph(self, **kwargs: Any) -> SchemaGraph:
        raise AssertionError("reflect_schema_graph should not run on cache hit")

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


def test_export_apply_preserves_user_override_provenance(
    schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I1: USER_OVERRIDE description/role survives export and re- apply."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)

    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(
            tables={
                "orders": {
                    "description": "User-authored order text.",
                    "role": "dimension",
                },
            },
        ),
    )
    assert schema_graph.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE
    assert schema_graph.tables["orders"].role_owner == RoleOwner.USER_OVERRIDE

    exported = dump_schema_overrides_dict(schema_graph)
    assert exported["tables"]["orders"]["description"]["owner"] == "user"
    assert exported["tables"]["orders"]["role"]["owner"] == "user"

    fresh = copy.deepcopy(schema_graph)
    fresh.tables["orders"].role = "fact"
    for tbl in fresh.tables.values():
        tbl.description_owner = DescriptionOwner.CATALOG
        tbl.role_owner = RoleOwner.CATALOG
        for col in tbl.columns.values():
            col.description_owner = DescriptionOwner.CATALOG
            col.role_owner = RoleOwner.CATALOG

    apply_schema_overrides_to_graph(fresh, exported)
    assert fresh.tables["orders"].description == "User-authored order text."
    assert fresh.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE
    assert fresh.tables["orders"].role_owner == RoleOwner.USER_OVERRIDE


def test_export_round_trip_does_not_invoke_llm_refinement(
    schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I1: unchanged export/apply must not call the description refiner."""
    llm_calls: list[Any] = []

    def _fake_llm_chat(**kwargs: Any) -> str:
        llm_calls.append(kwargs)
        return json.dumps({"items": []})

    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(tables={"orders": {"description": "Stable user description."}}),
    )
    exported = dump_schema_overrides_dict(schema_graph)

    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: True)
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_chat", _fake_llm_chat)
    apply_schema_overrides_to_graph(schema_graph, exported)
    assert llm_calls == []


def test_finalize_with_overrides_strict_raises_on_unknown_table(
    schema_graph: SchemaGraph, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I2: sidecar replay must not silently skip unknown table references."""
    from aetherdialect._config import ConfigError

    from aetherdialect._schema_overrides import compute_metadata_hash

    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(tmp_path / "schema.json.gz"))
    cache_path = tmp_path / "schema.json.gz"
    cache_path.write_bytes(b"x")
    save_overrides_sidecar(
        cache_path,
        {"tables": {"orders": {"columns": {"missing_col": {"description": "ghost"}}}}},
        source_schema_hash="old",
        metadata_hash="old",
    )
    with pytest.raises(ConfigError, match="unknown column"):
        finalize_with_overrides(schema_graph, cache_path)


def test_finalize_with_overrides_prunes_stale_table_keys(
    schema_graph: SchemaGraph, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I2: replay prunes sidecar table keys that no longer exist on the graph."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(tmp_path / "schema.json.gz"))
    cache_path = tmp_path / "schema.json.gz"
    cache_path.write_bytes(b"x")

    from aetherdialect._schema_overrides import compute_metadata_hash, load_overrides_sidecar

    save_overrides_sidecar(
        cache_path,
        {
            "tables": {
                "orders": {"description": "still here"},
                "ghost": {"description": "stale"},
            },
        },
        source_schema_hash="stale",
        metadata_hash="stale",
    )
    finalize_with_overrides(schema_graph, cache_path)
    sidecar = load_overrides_sidecar(cache_path)
    assert sidecar is not None
    assert "ghost" not in (sidecar.get("tables") or {})
    assert "orders" in (sidecar.get("tables") or {})


def test_apply_overrides_and_persist_updates_last_overrides_skipped(
    schema_graph: SchemaGraph, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I2: direct owner apply records skipped entries on the graph."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(tmp_path / "schema.json.gz"))
    col = schema_graph.tables["customers"].columns["email"]
    col.sensitivity = SensitivityClassification.HIDDEN
    overrides_path = tmp_path / "schema_overrides.json"
    overrides_path.write_text(
        json.dumps(
            _ov_doc(
                tables={
                    "customers": {
                        "columns": {
                            "email": {"description": "blocked"},
                        },
                    },
                },
            ),
        ),
        encoding="utf-8",
    )
    apply_overrides_and_persist(schema_graph, overrides_path, schema_json_path=str(tmp_path / "schema.json.gz"))
    assert any(s.code == "hidden_column_override" for s in schema_graph._last_overrides_skipped)


def test_hidden_column_override_is_not_applied(schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch) -> None:
    """I4: hidden columns record a skip and must not keep the mutation."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    col = schema_graph.tables["customers"].columns["email"]
    prev_desc = col.description or ""
    prev_role = col.role
    col.sensitivity = SensitivityClassification.HIDDEN

    report = apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(
            tables={
                "customers": {
                    "columns": {
                        "email": {
                            "description": "Should not stick.",
                            "role": "free_text",
                        },
                    },
                },
            },
        ),
    )
    assert col.description == prev_desc
    assert col.role == prev_role
    assert any(s.code == "hidden_column_override" for s in report.skipped)


def test_cache_hit_refreshes_profiling_when_live_row_counts_drift(
    schema_graph: SchemaGraph, cache_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I7: cache hit must detect live row-count drift and re-profile."""
    from aetherdialect._contracts_base import EngineContext

    ctx = EngineContext()
    notes = ""
    schema_graph.notes_sha256 = hashlib.sha256(notes.encode()).hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)
    probe = compute_dialect_probe(_ProfilingDriftStubDialect("probe_a"), ctx)
    schema_graph.ddl_probe_hash = probe
    save_schema_to_cache(schema_graph, cache_path)

    dialect = _ProfilingDriftStubDialect("probe_a", live_row_counts={"customers": 9999})
    monkeypatch.setattr(
        aetherdialect._schema_overrides,
        "compute_dialect_probe",
        lambda _d, _c: probe,
    )
    monkeypatch.setattr(
        aetherdialect._schema_overrides, "finalize_with_overrides", lambda sg, path, dialect=None: False
    )
    monkeypatch.setattr(aetherdialect._schema_overrides, "notify_schema_path_health", lambda sg: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "validate_scope_against_graph", lambda sg, c: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "raise_if_schema_unusable", lambda sg, c: None)

    sg, _diff = aetherdialect._schema_overrides.build_schema_graph_with_diff(dialect, ctx, notes_content=notes)
    assert dialect.profile_calls == 1
    assert sg.tables["customers"].row_count == 9999


def test_sandbox_trust_does_not_skip_profiling_drift_refresh(
    schema_graph: SchemaGraph, cache_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I7: SANDBOX_TRUST_SCHEMA_BASELINE must not bypass live profiling drift refresh."""
    from aetherdialect._contracts_base import EngineContext

    ctx = EngineContext()
    notes = ""
    schema_graph.notes_sha256 = hashlib.sha256(notes.encode()).hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)
    probe = compute_dialect_probe(_ProfilingDriftStubDialect("probe_a"), ctx)
    schema_graph.ddl_probe_hash = probe
    save_schema_to_cache(schema_graph, cache_path)

    dialect = _ProfilingDriftStubDialect("probe_b", live_row_counts={"customers": 9999})
    monkeypatch.setattr(PolicyConfig, "SANDBOX_TRUST_SCHEMA_BASELINE", True)
    monkeypatch.setattr(
        aetherdialect._schema_overrides,
        "compute_dialect_probe",
        lambda _d, _c: "probe_b",
    )
    monkeypatch.setattr(aetherdialect._schema_overrides, "finalize_with_overrides", lambda *a, **k: False)
    monkeypatch.setattr(aetherdialect._schema_overrides, "notify_schema_path_health", lambda sg: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "validate_scope_against_graph", lambda sg, c: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "raise_if_schema_unusable", lambda sg, c: None)

    sg, _diff = aetherdialect._schema_overrides.build_schema_graph_with_diff(dialect, ctx, notes_content=notes)
    assert dialect.profile_calls == 1
