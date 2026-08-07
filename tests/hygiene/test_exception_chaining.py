"""Exception chaining hygiene: every reraise inside except must use ``from``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "aetherdialect"


def _unchained_reraises(tree: ast.AST, *, path: Path) -> list[str]:
    """Return human-readable violations for raises inside except without an explicit cause."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Raise):
                continue
            if child.exc is None:
                continue  # bare ``raise`` preserves the active exception chain
            if child.cause is not None:
                continue
            violations.append(f"{path.name}:{child.lineno}: {ast.unparse(child)}")
    return violations


@pytest.mark.fast
def test_no_unchained_reraise() -> None:
    """Every ``raise`` inside an ``except`` block in ``src/`` must chain with ``from``."""
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_unchained_reraises(tree, path=path))
    if violations:
        pytest.fail("Unchained reraise(s) in src/:\n" + "\n".join(sorted(violations)))
