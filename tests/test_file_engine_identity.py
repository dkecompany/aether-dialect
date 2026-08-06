"""CSV file-engine connection identity behavior."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

pre_fix_failure: str | None = None


@dataclass
class _LinkedSourcePath:
    """Path-like stand-in that keeps a stable supplied name while retargeting I/O."""

    supplied: Path
    target: Path

    def __fspath__(self) -> str:
        return str(self.supplied)

    def resolve(self, strict: bool = False) -> Path:
        return self.target.resolve()

    def stat(self, *args: object, **kwargs: object) -> os.stat_result:
        return self.target.stat(*args, **kwargs)

    def read_bytes(self) -> bytes:
        return self.target.read_bytes()


def test_symlink_change_does_not_change_identity(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")

    content = "id,value\n1,alpha\n"
    real_a = tmp_path / "data_a.csv"
    real_b = tmp_path / "data_b.csv"
    real_a.write_text(content, encoding="utf-8")
    real_b.write_text(content, encoding="utf-8")
    stamp = time.time()
    os.utime(real_a, (stamp, stamp))
    os.utime(real_b, (stamp, stamp))

    supplied = tmp_path / "source.csv"
    linked = _LinkedSourcePath(supplied=supplied, target=real_a)

    from aetherdialect._dialect_sqlglot_engines import CsvDialect

    first = CsvDialect._csv_source_probe_payload([linked])

    linked.target = real_b
    second = CsvDialect._csv_source_probe_payload([linked])

    global pre_fix_failure
    if first != second:
        pre_fix_failure = (
            "connection identity changed when only the symlink target changed "
            f"for supplied path {supplied!s}: {first!r} != {second!r}"
        )
    assert first == second, pre_fix_failure
