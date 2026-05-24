"""Ensure dev notebooks do not regress to removed export_schema_overrides kwargs."""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_dev_notebooks_have_no_export_overwrite_true() -> None:
    root = _repo_root()
    for rel in (
        Path("dev_workspace") / "main_pg.ipynb",
        Path("dev_workspace") / "main_db.ipynb",
        Path("dev_workspace") / "main_hubspot.ipynb",
    ):
        path = root / rel
        if not path.is_file():
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb.get("cells", [])):
            src = cell.get("source", [])
            text = "".join(src) if isinstance(src, list) else str(src)
            assert "export_schema_overrides(overwrite=" not in text, f"{rel} cell {i}: remove overwrite="
