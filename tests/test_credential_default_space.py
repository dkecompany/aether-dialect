"""Credential-default aetherspace is always a distinct system space."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._constants import (
    CREDENTIAL_DEFAULT_AETHERSPACE_NAME,
    CREDENTIAL_DEFAULT_SNAPSHOT_FLAG,
    MASTER_AETHERSPACE_UID,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "customers": TableMetadata(
                name="customers",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h-cd",
        schema_graph_id="sg-cd__h",
    )


@pytest.mark.fast
def test_reclaim_stale_credential_default_when_grant_changes(tmp_path: Path) -> None:
    schema = _schema()
    narrow = frozenset({"orders"})
    wide = frozenset(schema.tables.keys())
    old_uid = MainExecutionOps.ensure_credential_default_aetherspace(str(tmp_path), schema, narrow)
    assert MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), old_uid) is not None
    new_uid = MainExecutionOps.ensure_credential_default_aetherspace(str(tmp_path), schema, wide)
    assert new_uid != old_uid
    assert MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), old_uid) is None
    assert MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), new_uid) is not None


@pytest.mark.fast
def test_full_visibility_consumer_gets_separate_credential_default(tmp_path: Path) -> None:
    """Even when RBAC visibility equals the full master table set, mint a system space — never master."""
    schema = _schema()
    full = frozenset(schema.tables.keys())
    uid = MainExecutionOps.ensure_credential_default_aetherspace(
        str(tmp_path),
        schema,
        full,
    )
    assert uid != MASTER_AETHERSPACE_UID
    assert uid.startswith("S")
    snap = MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), uid)
    assert snap is not None
    assert snap.get("name") == CREDENTIAL_DEFAULT_AETHERSPACE_NAME
    assert snap.get(CREDENTIAL_DEFAULT_SNAPSHOT_FLAG) is True
    assert frozenset(snap.get("tables") or ()) == full
    again = MainExecutionOps.ensure_credential_default_aetherspace(str(tmp_path), schema, full)
    assert again == uid
