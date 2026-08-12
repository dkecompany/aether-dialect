"""Live structure-document roundtrips against the rental_shop PostgreSQL database. Exercises export/apply of structure documents on a reflected schema graph: owned ``{value, owner}`` envelopes on description/role leaves, the ``_readonly`` envelope, drift-safe FK merge, removal/block behaviour, sidecar I/O, and ``finalize_with_structure`` replay. Each test deep-copies the session schema graph and writes the sidecar under a per-test ``tmp_path`` so the shared session ``AetherEngine`` is not mutated. Marked ``live_no_llm``; runs without LLM credentials."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aetherdialect._constants import STRUCTURE_DOCUMENT_VERSION
from aetherdialect._contracts_schema import DescriptionOwner, FKEdge
from aetherdialect._schema_finalize import (
    apply_structure_to_graph,
    compute_metadata_hash,
    delete_persisted_structure_artifacts,
    dump_structure_edits,
    finalize_with_structure,
    load_structure_sidecar,
    save_structure_sidecar,
)
from aetherdialect._schema_reflect import structure_sidecar_path


@pytest.fixture
def graph_copy(schema):
    """Per-test deep copy of the session schema graph so mutations don't leak."""
    return copy.deepcopy(schema)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    """Synthetic schema cache path used to derive the sidecar location."""
    p = tmp_path / "schema_graph.json.gz"
    p.write_bytes(b"")
    return p


def test_export_structure_shape_on_rental_shop(schema) -> None:
    """Exporting the rental_shop graph yields a structure document with the readonly envelope populated."""
    doc = dump_structure_edits(schema)
    assert doc["version"] == STRUCTURE_DOCUMENT_VERSION
    for top_key in (
        "tables",
        "foreign_keys_add",
        "foreign_keys_remove",
        "primary_keys_remove",
        "_readonly",
    ):
        assert top_key in doc, f"missing top-level key {top_key!r}"

    sample_table = next(iter(doc["tables"].values()))
    assert isinstance(sample_table["description"], dict)
    assert set(sample_table["description"].keys()) == {"value", "owner"}
    assert isinstance(sample_table["role"], dict)
    assert set(sample_table["role"].keys()) == {"value", "owner"}
    sample_col = next(iter(sample_table["columns"].values()))
    assert isinstance(sample_col["description"], dict)
    assert set(sample_col["description"].keys()) == {"value", "owner"}
    assert isinstance(sample_col["role"], dict)
    assert set(sample_col["role"].keys()) == {"value", "owner"}

    ro = doc["_readonly"]
    assert set(ro.keys()) == {
        "foreign_keys_current",
        "primary_keys_current",
        "tables_current",
        "columns_current",
    }

    table_names = {entry["name"] for entry in ro["tables_current"]}
    expected_subset = {"actor", "film", "film_actor", "rental", "customer", "category"}
    assert expected_subset.issubset(table_names), f"missing rental_shop tables: {expected_subset - table_names}"

    fks = ro["foreign_keys_current"]
    assert fks, "rental_shop must surface at least one FK"
    catalog_fks = [e for e in fks if e["inference_tag"] is None]
    assert catalog_fks, "rental_shop has many catalog FKs (e.g. film_actor.actor_id -> actor.actor_id)"
    for entry in catalog_fks:
        assert entry["removable"] is False
    inferred_or_user = [e for e in fks if e["inference_tag"] is not None]
    for entry in inferred_or_user:
        assert entry["removable"] is True

    pk_entries = ro["primary_keys_current"]
    assert pk_entries, "rental_shop tables have primary keys"
    for entry in pk_entries:
        assert entry["pk_inference_tag"] in (None, "profile")


def test_apply_user_fk_then_clear(graph_copy, cache_path: Path) -> None:
    """Adding a user FK is reflected in the graph, persists to the sidecar, and ``delete_persisted_structure_artifacts`` removes the file."""
    doc = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {},
        "foreign_keys_add": [{"from": "staff.store_id", "to": "store.store_id", "kind": "logical"}],
    }
    report = apply_structure_to_graph(graph_copy, doc)
    assert report.fks_added == 0
    assert report.fks_endorsed == 1
    assert report.skipped == ()

    user_edges = [
        e
        for e in graph_copy.tables["staff"].foreign_keys
        if (e.inference_tag or "") == "user_override_structural"
        and tuple(e.src_cols) == ("store_id",)
        and e.dst_table == "store"
        and tuple(e.dst_cols) == ("store_id",)
    ]
    assert len(user_edges) == 1, "user-added FK must land on the source table with the right tag"

    persisted_doc = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {},
        "foreign_keys_add": [{"from": "staff.store_id", "to": "store.store_id", "kind": "logical"}],
        "_internal": {"fk_block_inferred": [], "pk_block_inferred": []},
    }
    save_structure_sidecar(
        cache_path,
        persisted_doc,
        source_schema_hash=graph_copy.effective_structural_hash,
        metadata_hash=compute_metadata_hash(graph_copy),
    )
    sidecar = structure_sidecar_path(cache_path)
    assert sidecar.is_file()
    on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
    assert on_disk["version"] == STRUCTURE_DOCUMENT_VERSION
    assert on_disk["foreign_keys_add"][0]["from"] == "staff.store_id"
    assert "_readonly" not in on_disk
    assert "foreign_keys_remove" not in on_disk
    assert on_disk["source_schema_hash"] == graph_copy.effective_structural_hash
    assert "metadata_hash" in on_disk and len(on_disk["metadata_hash"]) == 64

    assert delete_persisted_structure_artifacts(cache_path) is True
    assert not sidecar.exists()
    assert delete_persisted_structure_artifacts(cache_path) is False


def test_finalize_replays_user_layer_after_simulated_drift(graph_copy, cache_path: Path) -> None:
    """A sidecar with a stale ``source_schema_hash`` is replayed onto a graph whose user-added FK is missing."""
    pristine_hash = graph_copy.effective_structural_hash
    apply_structure_to_graph(
        graph_copy,
        {
            "version": STRUCTURE_DOCUMENT_VERSION,
            "tables": {
                "actor": {
                    "description": "Catalogue of actors who can appear in films.",
                }
            },
            "foreign_keys_add": [{"from": "staff.store_id", "to": "store.store_id", "kind": "logical"}],
        },
    )
    sidecar_doc = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {
            "actor": {
                "description": "Catalogue of actors who can appear in films.",
            }
        },
        "foreign_keys_add": [{"from": "staff.store_id", "to": "store.store_id", "kind": "logical"}],
        "_internal": {"fk_block_inferred": [], "pk_block_inferred": []},
    }

    save_structure_sidecar(
        cache_path,
        sidecar_doc,
        source_schema_hash="stale_hash_xyz",
        metadata_hash=compute_metadata_hash(graph_copy),
    )

    fresh = copy.deepcopy(graph_copy)
    fresh.tables["staff"].foreign_keys = [
        e for e in fresh.tables["staff"].foreign_keys if (e.inference_tag or "") != "user_override_structural"
    ]
    fresh.tables["actor"].description = ""
    fresh.tables["actor"].description_owner = None
    assert not any((e.inference_tag or "") == "user_override_structural" for e in fresh.tables["staff"].foreign_keys)

    replayed = finalize_with_structure(fresh, cache_path)
    assert replayed is True

    user_edges = [
        e
        for e in fresh.tables["staff"].foreign_keys
        if (e.inference_tag or "") == "user_override_structural" and tuple(e.src_cols) == ("store_id",)
    ]
    assert len(user_edges) == 1
    actor_desc = fresh.tables["actor"].description or ""
    assert actor_desc.strip(), "[OVR-REPLAY] expected a non-empty actor description after replay"
    _actor_desc_msg = f"[OVR-REPLAY] expected refined description to retain user-supplied facts, got {actor_desc!r}"
    assert "actor" in actor_desc.lower() and "film" in actor_desc.lower(), _actor_desc_msg
    assert fresh.tables["actor"].description_owner == DescriptionOwner.USER_OVERRIDE

    refreshed = load_structure_sidecar(cache_path)
    assert refreshed is not None
    assert refreshed["source_schema_hash"] == fresh.effective_structural_hash
    assert refreshed["source_schema_hash"] != "stale_hash_xyz"

    skip_run = finalize_with_structure(fresh, cache_path)
    assert skip_run is False

    assert pristine_hash != fresh.effective_structural_hash


def test_remove_inferred_fk_persists_and_filters_replay(graph_copy, cache_path: Path) -> None:
    """An injected inferred FK is removed via ``foreign_keys_remove``, the system records the suppression internally, and a fresh graph re- acquires-then-loses the edge after replay."""
    edge = FKEdge(
        src_table="category",
        src_cols=["last_update"],
        dst_table="language",
        dst_cols=["last_update"],
        inference_tag="suffix",
    )
    graph_copy.tables["category"].foreign_keys.append(edge)

    doc = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {},
        "foreign_keys_remove": [{"from": "category.last_update", "to": "language.last_update"}],
    }
    report = apply_structure_to_graph(graph_copy, doc)
    assert report.fks_removed == 1
    surviving_inferred = [
        e
        for e in graph_copy.tables["category"].foreign_keys
        if (e.inference_tag or "") == "suffix" and tuple(e.src_cols) == ("last_update",) and e.dst_table == "language"
    ]
    assert surviving_inferred == [], "removed inferred FK must be deleted from the in-memory graph"

    save_structure_sidecar(
        cache_path,
        {
            "version": STRUCTURE_DOCUMENT_VERSION,
            "tables": {},
            "foreign_keys_add": [],
            "_internal": {
                "fk_block_inferred": [{"from": "category.last_update", "to": "language.last_update"}],
                "pk_block_inferred": [],
            },
        },
        source_schema_hash="drifted",
        metadata_hash=compute_metadata_hash(graph_copy),
    )
    fresh = copy.deepcopy(graph_copy)
    fresh.tables["category"].foreign_keys.append(
        FKEdge(
            src_table="category",
            src_cols=["last_update"],
            dst_table="language",
            dst_cols=["last_update"],
            inference_tag="suffix",
        )
    )
    replayed = finalize_with_structure(fresh, cache_path)
    assert replayed is True
    surviving = [
        e
        for e in fresh.tables["category"].foreign_keys
        if (e.inference_tag or "") == "suffix" and tuple(e.src_cols) == ("last_update",) and e.dst_table == "language"
    ]
    assert surviving == [], "internal block list must filter the re-injected inferred edge during replay"
