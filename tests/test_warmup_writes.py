"""Warmup artifact writes must acquire the artifacts directory lock."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._seed_warmup import SeedWarmupCacheSession


@pytest.mark.fast
def test_warmup_writes_hold_the_artifact_lock(
    tmp_path: Path,
    schema_graph: SchemaGraph,
) -> None:
    """Cache zip and anchor-lattice writes must enter artifact_lock for the artifacts directory."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    lattice_path = SeedWarmupCacheSession._warmup_anchor_lattice_json_path(str(artifacts_dir), schema_graph)
    cells = {"k1": ["alpha"]}

    lock_calls: list[str] = []
    real_lock = __import__("aetherdialect._core_utils", fromlist=["artifact_lock"]).artifact_lock

    def tracking_lock(artifacts_dir_arg: str, *args: object, **kwargs: object) -> object:
        lock_calls.append(os.path.abspath(str(artifacts_dir_arg)))
        return real_lock(artifacts_dir_arg, *args, **kwargs)

    with patch("aetherdialect._seed_warmup.artifact_lock", side_effect=tracking_lock):
        SeedWarmupCacheSession.save_seed_warmup_cache_zip(str(artifacts_dir), {"version": 1}, {"wu": {"id": "wu"}})
        SeedWarmupCacheSession._save_warmup_anchor_lattice(lattice_path, schema_graph, cells)

    expected = os.path.abspath(str(artifacts_dir))
    assert lock_calls == [expected, expected]

    report_path = artifacts_dir / "seed_warmup_report_v1.json"
    lock_calls.clear()
    with patch("aetherdialect._seed_warmup.artifact_lock", side_effect=tracking_lock):
        SeedWarmupCacheSession.save_seed_warmup_report([], str(report_path))

    assert all(call == expected for call in lock_calls)
    assert len(lock_calls) >= 1
    assert report_path.is_file()
