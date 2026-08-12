"""Owned text exports must use LF line endings regardless of platform."""

from __future__ import annotations

import os
from pathlib import Path

import pandas
import pytest

from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._pipeline_execute import save_result_csv
from aetherdialect._seed_warmup import SeedWarmupCacheSession


def _raw_bytes(path: Path) -> bytes:
    return path.read_bytes()


@pytest.mark.fast
def test_written_files_use_fixed_line_endings(
    tmp_path: Path,
    schema_graph: SchemaGraph,
) -> None:
    """CSV and warmup text siblings must contain only LF bytes, never CRLF."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    csv_path = artifacts_dir / "results.csv"
    save_result_csv(pandas.DataFrame({"col": [1, 2]}), output_path=csv_path)
    csv_bytes = _raw_bytes(csv_path)
    assert b"\r\n" not in csv_bytes
    assert b"\n" in csv_bytes

    report_path = artifacts_dir / "seed_warmup_report_v1.json"
    SeedWarmupCacheSession.save_seed_warmup_report([], str(report_path))
    report_bytes = _raw_bytes(report_path)
    assert b"\r\n" not in report_bytes

    lattice_path = SeedWarmupCacheSession._warmup_anchor_lattice_json_path(str(artifacts_dir), schema_graph)
    SeedWarmupCacheSession._save_warmup_anchor_lattice(
        lattice_path,
        schema_graph,
        {"k1": ["alpha"]},
    )
    lattice_bytes = _raw_bytes(Path(lattice_path))
    assert b"\r\n" not in lattice_bytes

    SeedWarmupCacheSession.save_seed_warmup_cache_zip(str(artifacts_dir), {"version": 1}, {"wu": {"id": "wu"}})
    zip_path = artifacts_dir / "seed_warmup_cache.zip"
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                body = zf.read(name)
                assert b"\r\n" not in body

    if os.name == "nt":
        # On Windows, default text mode would emit CRLF; owned exports must still use LF.
        assert csv_bytes.count(b"\n") >= 1
