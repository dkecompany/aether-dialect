"""Sandbox AetherSpace notes: content-gated pairing with documented demo fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect import SpaceContext
from aetherdialect._constants import SANDBOX_CATALOG_SPACE_TABLES
from aetherdialect._contracts_base import ConfigError
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._sandbox import Sandbox
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect.aetherdialect import AetherEngine

_REPO = Path(__file__).resolve().parents[1]
_CATALOG_NOTES_SOURCE = _REPO / "scripts" / "data" / "sandbox_space_catalog_notes.txt"


def _schema_graph(table_names: tuple[str, ...]) -> SchemaGraph:
    tables = {
        name: TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
        for name in table_names
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="space_notes_sg",
        effective_structural_hash="space_notes_sg",
    )


def _write_space_notes_bundle(root: Path) -> None:
    (root / "sandbox_space_catalog_notes.txt").write_text(
        _CATALOG_NOTES_SOURCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _sandbox_engine(tmp_path: Path) -> tuple[Sandbox, object]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_space_notes_bundle(bundle)
    sandbox = Sandbox.__new__(Sandbox)
    sandbox._extract_path = bundle
    sandbox._closed = False
    connection = object()
    Sandbox._mark_sandbox_managed_connection(connection, sandbox)
    arts = str(tmp_path / "arts")
    Path(arts).mkdir(parents=True, exist_ok=True)

    class _EngineStub:
        def __init__(self) -> None:
            self._schema_graph = _schema_graph(tuple(SANDBOX_CATALOG_SPACE_TABLES | frozenset({"rental", "payment"})))
            self._artifacts_dir = arts
            self._native_connection = connection
            self._sandbox_mode = True

        def _require_owner(self, _operation: str) -> None:
            return None

        def _require_master_context(self, _operation: str) -> None:
            return None

        def aetherspace(self, *args: object, **kwargs: object) -> object:
            return AetherEngine.aetherspace(self, *args, **kwargs)

    return sandbox, _EngineStub()


def _close_sandbox(_sandbox: Sandbox) -> None:
    return None


@pytest.mark.fast
def test_space_without_notes_any_subset(tmp_path: Path) -> None:
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        desc = engine.aetherspace(
            "rental_only",
            SpaceContext(tables=frozenset({"rental", "payment"})),
        )
        assert frozenset(desc.list_scope()["tables"]) == frozenset({"rental", "payment"})
        assert desc.notes is None
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_catalog_notes_accepted_under_renamed_file(tmp_path: Path) -> None:
    bundled = _CATALOG_NOTES_SOURCE.read_text(encoding="utf-8")
    renamed = tmp_path / "my_catalog_notes_copy.txt"
    renamed.write_text(bundled, encoding="utf-8")
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        desc = engine.aetherspace(
            "catalog",
            SpaceContext(tables=SANDBOX_CATALOG_SPACE_TABLES, notes_file=str(renamed)),
        )
        assert "catalog inventory" in (desc.notes or "").lower()
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_catalog_notes_rejected_on_mismatched_tables(tmp_path: Path) -> None:
    bundled_path = tmp_path / "bundle" / "sandbox_space_catalog_notes.txt"
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        with pytest.raises(ConfigError, match="custom notes"):
            engine.aetherspace(
                "catalog_bad_tables",
                SpaceContext(tables=frozenset({"film"}), notes_file=str(bundled_path)),
            )
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_arbitrary_notes_content_rejected(tmp_path: Path) -> None:
    notes = tmp_path / "custom_notes.txt"
    notes.write_text("film.rating is MPAA classification.\n", encoding="utf-8")
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        with pytest.raises(ConfigError, match="custom notes"):
            engine.aetherspace(
                "films_notes",
                SpaceContext(tables=frozenset({"film"}), notes_file=str(notes)),
            )
    finally:
        _close_sandbox(sandbox)
