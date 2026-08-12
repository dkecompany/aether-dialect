"""Sandbox AetherSpace lock: master plus the four bundled member spaces only, exact table sets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aetherdialect import SpaceContext
from aetherdialect._constants_runtime import (
    SANDBOX_MEMBER_SPACE_NOTES_FILES,
    SANDBOX_MEMBER_SPACE_TABLES,
)
from aetherdialect._contracts_base import ConfigError, EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._sandbox import Sandbox
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect.aetherdialect import AetherEngine

_REPO = Path(__file__).resolve().parents[1]
_CATALOG_NOTES_SOURCE = _REPO / "scripts" / "data" / SANDBOX_MEMBER_SPACE_NOTES_FILES["catalog"]
_CATALOG_TABLES = SANDBOX_MEMBER_SPACE_TABLES["catalog"]


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
    for notes_name in SANDBOX_MEMBER_SPACE_NOTES_FILES.values():
        src = _REPO / "scripts" / "data" / notes_name
        (root / notes_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


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
            all_tables = set().union(*SANDBOX_MEMBER_SPACE_TABLES.values()) | {"rental"}
            self._schema_graph = _schema_graph(tuple(sorted(all_tables)))
            self._artifacts_dir = arts
            self._native_connection = connection
            self._sandbox_mode = True
            self._schema_role = "owner"
            self._pipeline_writer_lock = __import__("threading").Lock()
            self._runtime_config = SimpleNamespace(
                execution_context=None,
                engine_context=EngineContext(),
            )
            self._consumer_visible_objects = None
            self._domain_knowledge = None

        def _require_owner(self, _operation: str) -> None:
            return None

        def _caller_visibility(self) -> tuple[object, None]:
            return AetherEngine._caller_visibility(self)

        def _domain_knowledge_entries(self) -> object:
            return AetherEngine._domain_knowledge_entries(self)

        def _resolve_aetherspace_visible_by_name(self, name: str) -> object:
            return AetherEngine._resolve_aetherspace_visible_by_name(self, name)

        def _raise_if_duplicate_aetherspace_name(self, display: str, *, exclude_uid: str | None = None) -> None:
            return AetherEngine._raise_if_duplicate_aetherspace_name(self, display, exclude_uid=exclude_uid)

        def aetherspace(self, *args: object, **kwargs: object) -> object:
            return AetherEngine.aetherspace(self, *args, **kwargs)

    return sandbox, _EngineStub()


def _close_sandbox(_sandbox: Sandbox) -> None:
    return None


@pytest.mark.fast
def test_space_rejects_name_outside_locked_set(tmp_path: Path) -> None:
    """Any AetherSpace name other than master or a bundled member is rejected."""
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        with pytest.raises(ConfigError, match="limited to master and the four bundled member spaces"):
            engine.aetherspace(
                "rental_only",
                SpaceContext(tables=frozenset({"rental", "payment"})),
            )
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_space_rejects_member_name_with_mismatched_tables(tmp_path: Path) -> None:
    """A member-space name must carry its exact bundled table set, not an arbitrary subset."""
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        with pytest.raises(ConfigError, match="limited to master and the four bundled member spaces"):
            engine.aetherspace(
                "catalog",
                SpaceContext(tables=frozenset({"film"})),
            )
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_space_accepts_member_name_with_exact_tables_and_no_notes(tmp_path: Path) -> None:
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        desc = engine.aetherspace(
            "catalog",
            SpaceContext(tables=_CATALOG_TABLES),
        )
        assert frozenset(desc.tables) == frozenset(_CATALOG_TABLES)
        assert desc.notes is None
    finally:
        _close_sandbox(sandbox)


def test_catalog_notes_accepted_under_bundled_basename(tmp_path: Path) -> None:
    """A notes_file matching the bundled basename resolves even when the caller path differs."""
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        caller_path = tmp_path / SANDBOX_MEMBER_SPACE_NOTES_FILES["catalog"]
        caller_path.write_text(_CATALOG_NOTES_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
        desc = engine.aetherspace(
            "catalog",
            SpaceContext(tables=_CATALOG_TABLES, notes_file=str(caller_path)),
        )
        assert "shared spine" in (desc.notes or "").lower()
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_catalog_notes_rejected_under_renamed_file(tmp_path: Path) -> None:
    """A notes_file with a non-bundled basename is rejected even with identical content."""
    bundled = _CATALOG_NOTES_SOURCE.read_text(encoding="utf-8")
    renamed = tmp_path / "my_catalog_notes_copy.txt"
    renamed.write_text(bundled, encoding="utf-8")
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        with pytest.raises(ConfigError, match="custom notes_file is not accepted"):
            engine.aetherspace(
                "catalog",
                SpaceContext(tables=_CATALOG_TABLES, notes_file=str(renamed)),
            )
    finally:
        _close_sandbox(sandbox)


@pytest.mark.fast
def test_arbitrary_notes_content_rejected(tmp_path: Path) -> None:
    notes = tmp_path / "custom_notes.txt"
    notes.write_text("film.rating is MPAA classification.\n", encoding="utf-8")
    sandbox, engine = _sandbox_engine(tmp_path)
    try:
        with pytest.raises(ConfigError, match="limited to master and the four bundled member spaces"):
            engine.aetherspace(
                "films_notes",
                SpaceContext(tables=frozenset({"film"}), notes_file=str(notes)),
            )
    finally:
        _close_sandbox(sandbox)
