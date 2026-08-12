"""Atomic federation and artifact writes keep temporary files beside their targets."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aetherdialect._config import SeedWarmupConfig
from aetherdialect._constants import WRITE_QUEUE_FILENAME
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._federation_execute import _write_federation_json_atomic, archive_federation_editor_file
from aetherdialect._schema_finalize import (
    _write_overrides_sidecar_payload,
    dump_structure_to_path,
)
from aetherdialect._schema_graph import upgrade_artifacts_schema_graph_id
from aetherdialect._seed_warmup import SeedWarmupCacheSession
from aetherdialect._utils_artifacts import write_artifact_manifest

save_seed_warmup_cache_zip = SeedWarmupCacheSession.save_seed_warmup_cache_zip


@pytest.mark.fast
def test_temp_file_created_in_target_directory(tmp_path, monkeypatch) -> None:
    """A bare filename writes its temporary file in the resolved target directory."""
    os.chdir(tmp_path)
    captured_dirs: list[str | None] = []
    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(*args, **kwargs):
        captured_dirs.append(kwargs.get("dir"))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr("tempfile.mkstemp", tracking_mkstemp)
    _write_federation_json_atomic("bare.json", {"k": "v"})
    assert captured_dirs == [str(tmp_path)]
    assert json.loads((tmp_path / "bare.json").read_text(encoding="utf-8")) == {"k": "v"}


def _fail_replace_to(target: Path, module: str) -> Any:
    real_replace = os.replace

    def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if os.path.abspath(str(dst)) == os.path.abspath(str(target)):
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    return patch(f"{module}.os.replace", side_effect=flaky_replace)


def _seed_good_zip(path: Path) -> bytes:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"version": 1}))
    return path.read_bytes()


WRITE_CASES = (
    pytest.param("overrides_editor", id="overrides_editor"),
    pytest.param("overrides_sidecar", id="overrides_sidecar"),
    pytest.param("warmup_cache_zip", id="warmup_cache_zip"),
    pytest.param("schema_graph_queue", id="schema_graph_queue"),
    pytest.param("federation_archive", id="federation_archive"),
)


@pytest.mark.fast
@pytest.mark.parametrize("write_case", WRITE_CASES)
def test_no_partial_file_after_failed_write(
    tmp_path: Path,
    schema_graph: SchemaGraph,
    write_case: str,
) -> None:
    """A failed write must not corrupt or truncate an existing target file."""
    if write_case == "overrides_editor":
        target = tmp_path / "schema_structure.json"
        good = json.dumps({"tables": {}, "version": 4}, indent=2, sort_keys=True) + "\n"
        target.write_text(good, encoding="utf-8")
        module = "aetherdialect._utils"
        with _fail_replace_to(target, module), pytest.raises(OSError, match="simulated replace failure"):
            dump_structure_to_path(schema_graph, target)
        assert target.read_text(encoding="utf-8") == good

    elif write_case == "overrides_sidecar":
        target = tmp_path / "schema_overrides.sidecar.json"
        good = json.dumps({"version": 1, "tables": {}}, indent=2, sort_keys=True) + "\n"
        target.write_text(good, encoding="utf-8")
        module = "aetherdialect._utils"
        with _fail_replace_to(target, module), pytest.raises(OSError, match="simulated replace failure"):
            _write_overrides_sidecar_payload(
                target,
                {"tables": {}, "_internal": {}},
                source_schema_hash="src",
                metadata_hash="meta",
            )
        assert target.read_text(encoding="utf-8") == good

    elif write_case == "warmup_cache_zip":
        target = tmp_path / SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP
        good_bytes = _seed_good_zip(target)
        module = "aetherdialect._seed_warmup"
        with _fail_replace_to(target, module), pytest.raises(OSError, match="simulated replace failure"):
            SeedWarmupCacheSession.save_seed_warmup_cache_zip(
                str(tmp_path), {"version": 2}, {"wu1": {"work_unit_id": "wu1"}}
            )
        assert target.read_bytes() == good_bytes

    elif write_case == "schema_graph_queue":
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        write_artifact_manifest(
            str(artifacts_dir),
            effective_structural_hash="effhash01",
            structural_hash="effhash01",
        )
        target = artifacts_dir / WRITE_QUEUE_FILENAME
        good = '{"schema_graph_id": "legacy", "kind": "note"}\n'
        target.write_text(good, encoding="utf-8")
        module = "aetherdialect._utils"
        with _fail_replace_to(target, module), pytest.raises(OSError, match="simulated replace failure"):
            upgrade_artifacts_schema_graph_id(str(artifacts_dir))
        assert target.read_text(encoding="utf-8") == good

    elif write_case == "federation_archive":
        editor = tmp_path / "federation_manifest.json"
        editor.write_text('{"version": 1}\n', encoding="utf-8")
        target = tmp_path / "federation_manifest.applied.json"
        good = '{"version": 0, "archived": true}\n'
        target.write_text(good, encoding="utf-8")
        module = "aetherdialect._utils"
        with _fail_replace_to(target, module), pytest.raises(OSError, match="simulated replace failure"):
            archive_federation_editor_file(str(editor))
        assert target.read_text(encoding="utf-8") == good

    else:
        raise AssertionError(f"unknown write case: {write_case}")
