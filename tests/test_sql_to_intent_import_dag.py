"""Import DAG and cross-module hygiene for SQL import modules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from aetherdialect._contracts_schema import SchemaGraph


def test_sql_to_intent_imports_sqlglot_module_only() -> None:
    """``_sql_to_intent_sqlglot`` must not import ``_sql_to_intent``."""
    root = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"
    src = (root / "_sql_to_intent_sqlglot.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "_sql_to_intent" not in node.module.replace("_sql_to_intent_sqlglot", "")


def test_sql_to_intent_imports_sqlglot_extractor() -> None:
    """Orchestration module wires the public sqlglot extractor symbol."""
    s2i = importlib.import_module("aetherdialect._sql_to_intent")
    sg = importlib.import_module("aetherdialect._sql_to_intent_sqlglot")
    assert (
        "_sql_to_intent_sqlglot" in s2i.__dict__.get("convert_sql_via_sqlglot", sg.convert_sql_via_sqlglot).__module__
    )
    assert callable(sg.runtime_from_sqlglot_tree)
    assert callable(sg.convert_sql_via_sqlglot)


def test_no_function_scoped_imports_in_sql_import_modules() -> None:
    """SQL import modules keep imports at module top."""
    root = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"
    violations: list[str] = []
    for module_name in ("_sql_to_intent.py", "_sql_to_intent_sqlglot.py"):
        tree = ast.parse((root / module_name).read_text(encoding="utf-8"))

        class _Visitor(ast.NodeVisitor):
            def __init__(self, file_name: str) -> None:
                self.in_function = 0
                self.file_name = file_name

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.in_function += 1
                self.generic_visit(node)
                self.in_function -= 1

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.in_function += 1
                self.generic_visit(node)
                self.in_function -= 1

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if self.in_function and node.level > 0:
                    violations.append(f"{self.file_name}:{node.lineno}")
                self.generic_visit(node)

        _Visitor(module_name).visit(tree)
    assert violations == []


def test_dialect_import_hooks_exist(schema_graph: SchemaGraph) -> None:
    """Base ``Dialect`` exposes import hook protocol defaults."""
    from aetherdialect._contracts_core import RuntimeIntent
    from aetherdialect._dialect import Dialect
    from aetherdialect._dialect_postgres import PostgresDialect

    pg = PostgresDialect.__new__(PostgresDialect)
    assert pg.parse_backend == "pglast"
    assert pg.preparse_sql_for_import("SELECT 1") == "SELECT 1"
    assert pg.map_import_where_op("=") is None
    assert pg.map_import_scalar_func("count") == "count"
    assert pg.import_unnest_policy() == "select_item"
    intent = pg.postprocess_imported_intent(
        RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        ),
        schema_graph,
    )
    assert intent.tables == ["customers"]

    base = Dialect.__new__(Dialect)
    assert base.parse_backend == "sqlglot"
    assert base.import_unnest_policy() == "from_only"
