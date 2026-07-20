"""Run the full docstring pipeline: prepare → docformatter → finalize.

Each file is round-tripped in memory first. We only write when the full
pipeline output differs from what is on disk (fixed-point check).
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from normalize_docstrings import (
    _REPO,
    iter_py_files,
    replace_docstrings,
    write_source_file,
)

_TARGETS = [
    _REPO / "src" / "aetherdialect",
    _REPO / "tests",
    _REPO / "live_tests",
]


def _prepare_source(raw: str) -> str:
    tree = ast.parse(raw)
    assert isinstance(tree, ast.Module)
    prepared = replace_docstrings(raw, tree, "prepare")
    ast.parse(prepared)
    return prepared


def _finalize_source(raw: str) -> str:
    tree = ast.parse(raw)
    assert isinstance(tree, ast.Module)
    final = replace_docstrings(raw, tree, "finalize")
    ast.parse(final)
    return final


def _docformat_python_source(source: str) -> str:
    fd, tmp_name = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(source, encoding="utf-8", newline="\n")
        result = subprocess.run(
            [sys.executable, "-m", "docformatter", "-i", str(tmp)],
            cwd=_REPO,
            check=False,
        )
        if result.returncode not in (0, 3):
            raise RuntimeError(f"docformatter failed with exit code {result.returncode}")
        return tmp.read_text(encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)


def _roundtrip_once(raw: str) -> str:
    """Single prepare → docformatter → finalize pass."""
    prepared = _prepare_source(raw)
    formatted = _docformat_python_source(prepared)
    return _finalize_source(formatted)


def roundtrip_source(raw: str, *, max_passes: int = 5) -> str:
    """Run the pipeline until a fixed point (idempotent output)."""
    current = raw
    for _ in range(max_passes):
        nxt = _roundtrip_once(current)
        if nxt == current:
            return current
        current = nxt
    raise RuntimeError("docstring round-trip did not converge")


def _collect_py_files() -> list[Path]:
    out: list[Path] = []
    for root in _TARGETS:
        out.extend(iter_py_files(root))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the round-trip pipeline would change any file",
    )
    args = parser.parse_args(argv)

    changed = 0
    errors = 0
    for path in _collect_py_files():
        initial = path.read_text(encoding="utf-8")
        try:
            final = roundtrip_source(initial)
        except Exception as exc:
            print(f"error ({exc}): {path.relative_to(_REPO)}", file=sys.stderr)
            errors += 1
            continue
        if final == initial:
            continue
        print(path.relative_to(_REPO))
        changed += 1
        if not args.check:
            write_source_file(path, final)

    if errors:
        print(f"failed on {errors} file(s)", file=sys.stderr)
        return 1

    if args.check and changed:
        print(f"would update {changed} file(s)", file=sys.stderr)
        return 1

    if not args.check:
        print(f"updated {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
