"""Schema override migration fixes: scope superset diff, sidecar prune, export round-trip."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import SCHEMA_OVERRIDES_VERSION
from aetherdialect._contracts_base import DescriptionOwner, EngineContext, MigrationTier, RoleOwner
from aetherdialect._core_utils import ArtifactManifest
from aetherdialect._contracts_schema import FKEdge, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import (
    SchemaDiff,
    assign_schema_graph_hashes,
    classify_migration_tier,
    compute_dialect_probe,
    diff_schemas,
)
from aetherdialect._schema_overrides import (
    apply_diff,
    apply_schema_overrides_to_graph,
    build_schema_graph_with_diff,
    dump_schema_overrides_dict,
    load_overrides_sidecar,
    migrate_sidecar_for_diff,
    reconcile_sidecar_against_graph,
    save_overrides_sidecar,
    save_schema_to_cache,
)

pytestmark = [pytest.mark.fast, pytest.mark.usefixtures("stub_schema_llm_classifier")]


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


class _ScopeReflectDialect(Dialect):
    """Reflects a fixed template graph; records reflect calls."""

    name = "stub"

    def __init__(self, template: SchemaGraph, probe_value: str = "probe_scope") -> None:
        super().__init__(MagicMock())
        self._template = template
        self._probe_value = probe_value
        self.reflect_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return self._probe_value

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        self.reflect_calls += 1
        sg = deepcopy(self._template)
        if allow_objects:
            sg.tables = {k: v for k, v in sg.tables.items() if k in allow_objects}
        if deny_objects:
            for name in deny_objects:
                sg.tables.pop(name, None)
        return sg

    def reflect_only(self, schema_context: EngineContext) -> SchemaGraph:
        return self.reflect_schema_graph(
            include=schema_context.include or "tables",
            allow_objects=schema_context.allow_objects or None,
            deny_objects=schema_context.deny_objects or None,
            sql_file=getattr(schema_context, "sql_file", None),
        )

    def profile_schema(self, sg: SchemaGraph) -> None:
        pass

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _stub_schema_build_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aetherdialect._schema_overrides.finalize_with_overrides", lambda *a, **k: False)
    monkeypatch.setattr("aetherdialect._schema_overrides.notify_schema_path_health", lambda sg: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.validate_scope_against_graph", lambda sg, c: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.raise_if_schema_unusable", lambda sg, c: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.apply_column_roles_llm", lambda *a, **k: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.apply_boolean_coercion_pass", lambda sg: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.assign_column_ops", lambda sg: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.infer_missing_pks_from_profile", lambda *a, **k: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.run_fk_inference_if_disconnected", lambda *a, **k: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.redact_hidden_sensitivity_profile_values", lambda sg: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.mark_canonical_duplicates", lambda sg: None)
    monkeypatch.setattr("aetherdialect._schema_overrides.coerce_pk_fk_columns_to_identifier", lambda sg: [])
    monkeypatch.setattr("aetherdialect._schema_overrides.collapse_redundant_inferences", lambda sg, skipped: 0)


def _save_cached(
    sg: SchemaGraph,
    ctx: EngineContext,
    notes: str,
    probe: str,
    cache_path: str,
) -> None:
    sg.notes_sha256 = hashlib.sha256(notes.encode()).hexdigest()
    assign_schema_graph_hashes(sg, ctx, sg.notes_sha256)
    sg.ddl_probe_hash = probe
    save_schema_to_cache(sg, cache_path)


@pytest.mark.fast
def test_scope_superset_rebuild_returns_schema_diff(
    schema_graph: SchemaGraph, cache_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening scope must return a SchemaDiff (not None) so migration tier is not destructive."""
    narrow_ctx = EngineContext(allow_objects=frozenset({"customers", "orders"}))
    wide_ctx = EngineContext()
    notes = "notes"
    dialect = _ScopeReflectDialect(schema_graph)
    probe = compute_dialect_probe(dialect, narrow_ctx)
    narrow_graph = deepcopy(schema_graph)
    narrow_graph.tables.pop("products", None)
    _save_cached(narrow_graph, narrow_ctx, notes, probe, cache_path)
    _stub_schema_build_helpers(monkeypatch)

    sg, diff = build_schema_graph_with_diff(dialect, wide_ctx, notes_content=notes)
    assert diff is not None, "scope-superset rebuild must not discard schema_diff"
    assert "products" in diff.added_tables
    assert "products" in sg.tables


@pytest.mark.fast
def test_scope_superset_diff_enables_additive_tier(
    schema_graph: SchemaGraph, cache_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scope-superset diff must classify as ADDITIVE, not DESTRUCTIVE."""
    narrow_ctx = EngineContext(allow_objects=frozenset({"customers", "orders"}))
    wide_ctx = EngineContext()
    notes = "notes"
    dialect = _ScopeReflectDialect(schema_graph)
    probe = compute_dialect_probe(dialect, narrow_ctx)
    narrow_graph = deepcopy(schema_graph)
    narrow_graph.tables.pop("products", None)
    _save_cached(narrow_graph, narrow_ctx, notes, probe, cache_path)
    _stub_schema_build_helpers(monkeypatch)

    sg, diff = build_schema_graph_with_diff(dialect, wide_ctx, notes_content=notes)
    manifest = ArtifactManifest(
        structural_hash="old",
        profiling_hash="old",
        scope_hash="old_scope",
        effective_structural_hash="old_eff",
        schema_graph_id="sg_old",
        notes_hash=sg.notes_sha256,
        semantic_edges_hash=sg.semantic_edges_hash,
    )
    tier = classify_migration_tier(manifest, sg, previous_schema=narrow_graph, schema_diff=diff)
    assert tier == MigrationTier.ADDITIVE


@pytest.mark.fast
def test_sidecar_pruned_after_partial_rebuild_drop(
    schema_graph: SchemaGraph, cache_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropped tables must be pruned from the overrides sidecar after apply_diff."""
    from aetherdialect._contracts_schema import TableMetadata

    ghost_tbl = TableMetadata(name="ghost", columns={}, foreign_keys=[], primary_key="")
    schema_graph.tables["ghost"] = ghost_tbl
    sidecar_doc = _ov_doc(
        tables={
            "ghost": {"description": "stale override"},
            "orders": {"description": {"value": "keep me", "owner": "user"}},
        },
    )
    save_overrides_sidecar(
        cache_path,
        sidecar_doc,
        source_schema_hash="h",
        metadata_hash="m",
    )
    new_sg = deepcopy(schema_graph)
    new_sg.tables.pop("ghost", None)
    diff = diff_schemas(schema_graph, new_sg)

    class _FakeDialect:
        name = "test"

        def profile_schema(self, *_a, **_k):
            pass

        def refresh_full_table_distinct_for_pk_inference(self, *_a, **_k):
            return None

    apply_diff(schema_graph, new_sg, diff, _FakeDialect(), schema_json_path=cache_path)
    loaded = load_overrides_sidecar(cache_path)
    assert loaded is not None
    assert "ghost" not in loaded.get("tables", {})
    assert "orders" in loaded.get("tables", {})


@pytest.mark.fast
def test_migration_map_reconcile_prunes_dropped_sidecar_entries(
    schema_graph: SchemaGraph, cache_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REMAP migration path must reconcile sidecar entries for dropped objects."""
    save_overrides_sidecar(
        cache_path,
        _ov_doc(tables={"ghost": {"description": "gone"}}),
        source_schema_hash="h",
        metadata_hash="m",
    )
    schema_graph.tables["ghost"] = TableMetadata(name="ghost", columns={}, foreign_keys=[], primary_key="")
    new_sg = deepcopy(schema_graph)
    new_sg.tables.pop("ghost", None)
    diff = SchemaDiff(dropped_tables=("ghost",))
    migrate_sidecar_for_diff(cache_path, diff)
    reconcile_sidecar_against_graph(new_sg, cache_path)
    loaded = load_overrides_sidecar(cache_path)
    assert loaded is not None
    assert "ghost" not in loaded.get("tables", {})


@pytest.mark.fast
def test_export_only_includes_overridden_tables(schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch) -> None:
    """Export must include only tables/columns with user overrides, not the full graph."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(tables={"orders": {"description": {"value": "User order text.", "owner": "user"}}}),
    )
    exported = dump_schema_overrides_dict(schema_graph)
    assert set(exported["tables"].keys()) == {"orders"}
    assert "customers" not in exported["tables"]
    assert "products" not in exported["tables"]


@pytest.mark.fast
def test_export_edit_apply_export_preserves_user_provenance(
    schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export → edit → apply → export must preserve user authorship."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(
            tables={
                "orders": {
                    "description": {"value": "Round trip.", "owner": "user"},
                    "role": {"value": "dimension", "owner": "user"},
                }
            }
        ),
    )
    first = dump_schema_overrides_dict(schema_graph)
    fresh = deepcopy(schema_graph)
    for tbl in fresh.tables.values():
        tbl.description_owner = DescriptionOwner.CATALOG
        tbl.role_owner = RoleOwner.CATALOG
    apply_schema_overrides_to_graph(fresh, first)
    second = dump_schema_overrides_dict(fresh)
    assert second["tables"]["orders"]["description"]["owner"] == "user"
    assert second["tables"]["orders"]["role"]["owner"] == "user"
    assert fresh.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE


@pytest.mark.fast
def test_export_persists_fk_pk_removal_lists(schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch) -> None:
    """Export must surface active FK/PK suppressions in removal lists."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    inferred = [e for e in schema_graph.tables["orders"].foreign_keys if e.inference_tag is not None]
    if not inferred:
        schema_graph.tables["orders"].foreign_keys.append(
            FKEdge(
                src_table="orders",
                src_cols=["amount"],
                dst_table="customers",
                dst_cols=["customer_id"],
                inference_tag="suffix",
            )
        )
        inferred = [schema_graph.tables["orders"].foreign_keys[-1]]
    fk = inferred[0]
    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(
            foreign_keys_remove=[
                {
                    "from": f"{fk.src_table}.{fk.src_cols[0]}",
                    "to": f"{fk.dst_table}.{fk.dst_cols[0]}",
                }
            ],
        ),
    )
    exported = dump_schema_overrides_dict(schema_graph)
    assert exported["foreign_keys_remove"]
    assert isinstance(exported["primary_keys_remove"], list)


@pytest.mark.fast
def test_apply_ignores_readonly_envelope_edits(schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutations under _readonly must not be applied."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    exported = dump_schema_overrides_dict(schema_graph)
    original_desc = schema_graph.tables["orders"].description or ""
    readonly = exported.setdefault("_readonly", {})
    tables_current = list(readonly.get("tables_current", []))
    for rec in tables_current:
        if rec.get("name") == "orders":
            rec["description"] = "READONLY INJECTION"
    readonly["tables_current"] = tables_current
    apply_schema_overrides_to_graph(schema_graph, exported)
    assert (schema_graph.tables["orders"].description or "") == original_desc


@pytest.mark.fast
def test_foreign_keys_add_refuses_resurrected_catalog_edge(
    schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisted foreign_keys_add must not resurrect a catalog FK removed upstream."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    catalog_edge = next(e for e in schema_graph.tables["orders"].foreign_keys if e.inference_tag is None)
    from_str = f"{catalog_edge.src_table}.{catalog_edge.src_cols[0]}"
    to_str = f"{catalog_edge.dst_table}.{catalog_edge.dst_cols[0]}"
    schema_graph.tables["orders"].foreign_keys = [
        e for e in schema_graph.tables["orders"].foreign_keys if e.inference_tag is not None
    ]
    report = apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(
            foreign_keys_add=[{"from": from_str, "to": to_str, "kind": "structural"}],
            _internal={"catalog_fk_revoked": [{"from": from_str, "to": to_str}]},
        ),
    )
    restored = [
        e
        for e in schema_graph.tables["orders"].foreign_keys
        if e.src_cols == catalog_edge.src_cols and e.dst_cols == catalog_edge.dst_cols
    ]
    assert report.fks_added == 0
    assert not restored


@pytest.mark.fast
def test_user_fk_addition_does_not_change_effective_structural_hash(
    schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User-added structural FK must not change effective_structural_hash."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    ctx = EngineContext()
    assign_schema_graph_hashes(schema_graph, ctx, "")
    before = schema_graph.effective_structural_hash
    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(
            foreign_keys_add=[
                {
                    "from": "orders.amount",
                    "to": "customers.customer_id",
                    "kind": "structural",
                }
            ],
        ),
    )
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256 or "")
    assert schema_graph.effective_structural_hash == before


@pytest.mark.fast
def test_user_pk_addition_does_not_change_effective_structural_hash(
    schema_graph: SchemaGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User-promoted PK must not change effective_structural_hash."""
    monkeypatch.setattr("aetherdialect._schema_overrides.llm_credentials_configured", lambda: False)
    ctx = EngineContext()
    schema_graph.tables["orders"].primary_key = ["order_id"]
    assign_schema_graph_hashes(schema_graph, ctx, "")
    before = schema_graph.effective_structural_hash
    apply_schema_overrides_to_graph(
        schema_graph,
        _ov_doc(primary_keys_add=[{"table": "orders", "column": "status"}]),
    )
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256 or "")
    assert schema_graph.effective_structural_hash == before
