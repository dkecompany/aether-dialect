"""Fast guard against skip-only IR capability live placeholders."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_IR_CAPABILITY_LIVE = _REPO / "live_tests" / "test_ir_capability_live.py"


def _skip_only_test_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        body: list[ast.stmt] = []
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue
            body.append(stmt)
        if len(body) != 1:
            continue
        stmt = body[0]
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "skip"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pytest"
        ):
            offenders.append(node.name)
    return offenders


@pytest.mark.fast
def test_no_skip_only_ir_capability_placeholders() -> None:
    if not _IR_CAPABILITY_LIVE.is_file():
        return
    offenders = _skip_only_test_functions(_IR_CAPABILITY_LIVE)
    assert offenders == [], f"skip-only placeholders in {_IR_CAPABILITY_LIVE.name}: {offenders}"
