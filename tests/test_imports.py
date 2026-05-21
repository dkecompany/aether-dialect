"""Static import hygiene for ``src/aetherdialect`` (Part H8)."""

from __future__ import annotations

import ast
from pathlib import Path

_OPTIONAL_DEP_MODULE_ROOTS = frozenset(
    {
        "psycopg2",
        "databricks",
        "pyspark",
        "pglast",
        "msvcrt",
        "fcntl",
    },
)


def _import_roots(stmt: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(stmt, ast.Import):
        return {alias.name.split(".")[0] for alias in stmt.names}
    if isinstance(stmt, ast.ImportFrom):
        if stmt.module:
            return {stmt.module.split(".")[0]}
        return set()
    return set()


def _import_allowed_in_function(stmt: ast.Import | ast.ImportFrom) -> bool:
    roots = _import_roots(stmt)
    if not roots:
        return True
    return roots.issubset(_OPTIONAL_DEP_MODULE_ROOTS)


class _FunctionLocalImportVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self._rel = rel_path
        self._fn_depth = 0
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn_depth += 1
        self.generic_visit(node)
        self._fn_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._fn_depth += 1
        self.generic_visit(node)
        self._fn_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        if self._fn_depth and not _import_allowed_in_function(node):
            self.violations.append(f"{self._rel}:{node.lineno} import {ast.dump(node)}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._fn_depth and not _import_allowed_in_function(node):
            self.violations.append(f"{self._rel}:{node.lineno} import_from {ast.dump(node)}")
        self.generic_visit(node)


def test_no_function_local_imports() -> None:
    """Disallow function-scoped imports except optional-driver roots (pglast, databricks, pyspark, …)."""

    pkg = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"
    assert pkg.is_dir(), pkg
    all_violations: list[str] = []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg.parent.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        vis = _FunctionLocalImportVisitor(rel)
        vis.visit(tree)
        all_violations.extend(vis.violations)
    assert not all_violations, "function-local imports:\n" + "\n".join(all_violations)
