"""Diagnostic records carry phase, subject, and remediation as first- class fields."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aetherdialect._contracts_base import Diagnostic, DiagnosticSeverity

_SRC = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"


@pytest.mark.fast
def test_phase_populated_at_every_site() -> None:
    fields = set(Diagnostic.__dataclass_fields__)
    assert {"phase", "remediation", "subject", "count"} <= fields
    sample = Diagnostic(
        stage="validation",
        level=DiagnosticSeverity.INFO,
        code="ENGINE_INFO",
        message="ok",
        phase="validation",
        subject="sql",
        remediation=None,
        count=1,
    )
    assert sample.phase == "validation"
    assert sample.subject == "sql"

    missing_phase: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name != "Diagnostic":
                continue
            keys = {kw.arg for kw in node.keywords if kw.arg}
            if "phase" not in keys:
                missing_phase.append(f"{path.name}:{node.lineno}")
    assert missing_phase == [], f"Diagnostic() sites missing phase=: {missing_phase}"
