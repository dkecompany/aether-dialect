"""Federation composite cache and manifest hash refresh share one artifact lock."""

from __future__ import annotations

import contextlib
import json
import tempfile
from unittest.mock import patch

import pytest

import aetherdialect._utils_artifacts
from aetherdialect._constants import ARTIFACT_MANIFEST_FILENAME, FEDERATION_ARTIFACT_FORMAT_VERSION
from aetherdialect._federation_execute import (
    _persist_federation_composite_schema_cache,
    _refresh_federation_artifact_manifest_hashes,
)
from tests.federation_helpers import federation_member_graph

_ORIGINAL_ARTIFACT_LOCK = aetherdialect._utils_artifacts.artifact_lock


@pytest.mark.fast
def test_composite_and_manifest_hashes_commit_together() -> None:
    """Composite write and manifest hash refresh must run inside one artifact_lock hold."""
    composite = federation_member_graph("orders", source_id="storefront")
    lock_state = {"holding": False}
    refresh_while_locked: list[bool] = []

    @contextlib.contextmanager
    def tracking_artifact_lock(federation_dir: str, *args: object, **kwargs: object):
        with _ORIGINAL_ARTIFACT_LOCK(federation_dir, *args, **kwargs):
            lock_state["holding"] = True
            try:
                yield
            finally:
                lock_state["holding"] = False

    def tracking_refresh(federation_dir: str, graph) -> None:
        refresh_while_locked.append(lock_state["holding"])
        return _refresh_federation_artifact_manifest_hashes(federation_dir, graph)

    with tempfile.TemporaryDirectory() as fed_dir:
        manifest_path = f"{fed_dir}/{ARTIFACT_MANIFEST_FILENAME}"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "artifact_format_version": FEDERATION_ARTIFACT_FORMAT_VERSION,
                    "schema_graph_id": str(composite.schema_graph_id or ""),
                    "structural_hash": "stale",
                    "effective_structural_hash": "stale",
                },
                handle,
            )

        with (
            patch.object(aetherdialect._federation_execute, "artifact_lock", tracking_artifact_lock),
            patch.object(
                aetherdialect._federation_execute,
                "_refresh_federation_artifact_manifest_hashes",
                side_effect=tracking_refresh,
            ),
        ):
            _persist_federation_composite_schema_cache(fed_dir, composite)

    assert refresh_while_locked == [True]
