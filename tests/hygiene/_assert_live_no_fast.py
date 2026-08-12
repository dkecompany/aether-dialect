"""Collect-only helper: fail if any live_tests item carries the fast marker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(sys.argv[1])


class _AssertNoLiveFast:
    def __init__(self) -> None:
        self.bad: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            fspath = getattr(item, "path", None) or getattr(item, "fspath", None)
            if fspath is None:
                is_live = "live_tests/" in item.nodeid.replace("\\", "/")
            else:
                is_live = "live_tests" in Path(fspath).parts
            if is_live and item.get_closest_marker("fast") is not None:
                self.bad.append(item.nodeid)


plugin = _AssertNoLiveFast()
code = pytest.main(
    [
        str(ROOT / "tests"),
        str(ROOT / "live_tests"),
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
    ],
    plugins=[plugin],
)
if plugin.bad:
    print("live_tests marked fast:")
    print("\n".join(plugin.bad[:40]))
    raise SystemExit(1)
print("OK")
raise SystemExit(0 if code in (0, 5) else code)
