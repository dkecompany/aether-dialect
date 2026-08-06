"""Diagnostic severity is a closed enumerated set."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aetherdialect._contracts_base import Diagnostic, DiagnosticSeverity

_SRC = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"
_ALLOWED_LEVEL_STRINGS = frozenset({"info", "warning", "error"})


@pytest.mark.fast
def test_severity_is_enumerated_everywhere() -> None:
    assert {m.value for m in DiagnosticSeverity} == _ALLOWED_LEVEL_STRINGS
    with pytest.raises((TypeError, ValueError)):
        Diagnostic(stage="t", level="warn", code="ENGINE_INFO", message="x")
    with pytest.raises((TypeError, ValueError)):
        Diagnostic(stage="t", level="_error", code="ENGINE_INFO", message="x")
    ok = Diagnostic(stage="t", level=DiagnosticSeverity.ERROR, code="ENGINE_INFO", message="x")
    assert ok.level == DiagnosticSeverity.ERROR

    offenders: list[str] = []
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
            if name not in {"notify", "Diagnostic"}:
                continue
            for kw in node.keywords:
                if kw.arg != "level":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    if kw.value.value not in _ALLOWED_LEVEL_STRINGS:
                        offenders.append(f"{path.name}:{node.lineno}:{kw.value.value!r}")
    assert offenders == [], f"non-enumerated severity literals: {offenders}"
