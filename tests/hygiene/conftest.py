"""Mark the static hygiene package for ``pytest -m hygiene`` selection."""

from __future__ import annotations

from pathlib import Path

import pytest

_HYGIENE_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    hygiene = pytest.mark.hygiene
    fast = pytest.mark.fast
    for item in items:
        path = Path(str(getattr(item, "path", getattr(item, "fspath", ""))))
        try:
            path.resolve().relative_to(_HYGIENE_ROOT)
        except ValueError:
            continue
        item.add_marker(hygiene)
        item.add_marker(fast)
