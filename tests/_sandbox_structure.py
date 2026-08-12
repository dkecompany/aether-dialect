"""Helpers for applying bundled schema structure demos through ``apply_structure``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aetherdialect._sandbox import Sandbox


def apply_demo_schema_structure_from_bundle(
    engine: Any,
    *,
    handle: Any | None = None,
    bundle_root: Path | None = None,
) -> Path:
    """Load schema_structure_demo.json from the bundle and call ``apply_structure``."""
    del handle  # accepted for call-site signature parity; apply is in-memory dict I/O
    root = bundle_root
    if root is None:
        root = Sandbox._sandbox_extract_path_for_engine(engine)
    if root is None:
        raise FileNotFoundError("Missing sandbox bundle root for structure demo staging")
    source = root / "schema_structure_demo.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing bundled schema structure demo: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"schema structure demo must be a JSON object: {source}")
    engine.apply_structure(document)
    return source
