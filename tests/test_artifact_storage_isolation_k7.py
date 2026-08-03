"""Artifact storage directory isolation across tenants and federation members."""

from __future__ import annotations

import os

import pytest

from aetherdialect._constants import ARTIFACT_DIRECTORY_SEGMENT, FEDERATION_SOURCE_STORAGE_PREFIX
from aetherdialect._contracts_base import FederationConfigError, FederationSourceBinding
from aetherdialect._federation import (
    federation_source_storage_slug,
    validate_federation_source_slug_uniqueness,
)
from aetherdialect._main_execution import compute_connection_storage_slug, compute_engine_storage_dir


@pytest.mark.fast
def test_different_tenant_slugs_get_distinct_storage_dirs(tmp_path) -> None:
    root = str(tmp_path)
    slug = compute_connection_storage_slug("postgresql")
    dir_a = compute_engine_storage_dir(root, "postgresql", tenant_slug="tenant-a")
    dir_b = compute_engine_storage_dir(root, "postgresql", tenant_slug="tenant-b")
    dir_default = compute_engine_storage_dir(root, "postgresql")

    assert dir_a != dir_b
    assert dir_a != dir_default
    assert dir_b != dir_default
    assert dir_a == os.path.join(os.path.abspath(root), ARTIFACT_DIRECTORY_SEGMENT, "tenant-a", slug)
    assert dir_b == os.path.join(os.path.abspath(root), ARTIFACT_DIRECTORY_SEGMENT, "tenant-b", slug)
    assert dir_default == os.path.join(os.path.abspath(root), ARTIFACT_DIRECTORY_SEGMENT, slug)


@pytest.mark.fast
def test_federation_member_slug_does_not_match_standalone_engine_slug(monkeypatch) -> None:
    from aetherdialect._config import PostgresRuntimeConfig

    monkeypatch.setattr(
        PostgresRuntimeConfig,
        "connection_slug_fields",
        classmethod(
            lambda cls: {
                "host": "db.internal",
                "port": "5432",
                "database": "analytics",
                "schema": "public",
            }
        ),
    )
    monkeypatch.setattr(
        PostgresRuntimeConfig,
        "connection_slug_keys",
        classmethod(lambda cls: ("host", "port", "database", "schema")),
    )
    binding = FederationSourceBinding(
        source_id="storefront",
        engine="postgresql",
        connection="storefront_pg",
    )
    standalone_slug = compute_connection_storage_slug("postgresql")
    federation_slug = federation_source_storage_slug(binding)
    assert standalone_slug.startswith("conn_")
    assert federation_slug.startswith(FEDERATION_SOURCE_STORAGE_PREFIX)
    assert federation_slug != standalone_slug


@pytest.mark.fast
def test_validate_federation_source_slug_uniqueness_still_detects_collisions() -> None:
    binding_a = FederationSourceBinding(source_id="a", engine="duckdb", connection="")
    binding_b = FederationSourceBinding(source_id="b", engine="duckdb", connection="")
    slug = federation_source_storage_slug(binding_a)
    assert slug == federation_source_storage_slug(binding_b)
    with pytest.raises(FederationConfigError):
        validate_federation_source_slug_uniqueness([binding_a, binding_b])
