"""Handlers must not silently swallow broad exceptions."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sqlglot

from aetherdialect._contracts_base import ConfigError, NormalizedExpr, SpaceContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_profile import NotesExtractionLedger, NotesExtractionResult
from aetherdialect._validation_sql import _enforce_select_only, _pg_parsed_has_forbidden_sql

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "aetherdialect"


def _is_broad_exception_handler(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    if isinstance(t, ast.Name) and t.id == "Exception":
        return True
    if isinstance(t, ast.Tuple):
        return any(isinstance(elt, ast.Name) and elt.id == "Exception" for elt in t.elts)
    return False


def _is_silent_swallow_body(body: list[ast.stmt]) -> bool:
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Continue):
        return True
    if isinstance(stmt, ast.Return) and stmt.value is None:
        return True
    return False


def _bare_exception_pass_sites(tree: ast.AST, *, path: Path) -> list[tuple[str, int, str]]:
    sites: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_broad_exception_handler(node):
            continue
        if not _is_silent_swallow_body(node.body):
            continue
        kind = "pass"
        if isinstance(node.body[0], ast.Continue):
            kind = "continue"
        elif isinstance(node.body[0], ast.Return):
            kind = "bare return"
        sites.append((path.name, node.lineno, kind))
    return sites


def _graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )


@pytest.mark.fast
def test_named_handlers_report() -> None:
    """Named swallow sites must propagate or emit an explicit diagnostic/refusal."""
    # 1) Classification failure in enrich_space_snapshot_with_notes must propagate.
    notes = _ROOT / "tests" / "_tmp_notes_swallow.txt"
    notes.write_text("notes", encoding="utf-8")
    snapshot = {"tables": ["orders"], "table_descriptions": {}, "column_meta": {}}
    space = SpaceContext(tables=frozenset({"orders"}), notes_file=str(notes))
    sk_fact = __import__("aetherdialect._contracts_base", fromlist=["StructuralKnowledgeFact"]).StructuralKnowledgeFact(
        kind="relation",
        text="order rows",
        referenced_entities=frozenset({"orders"}),
    )
    with (
        patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True),
        patch(
            "aetherdialect._main_spaces.extract_knowledge_from_notes",
            return_value=NotesExtractionResult((), (sk_fact,), NotesExtractionLedger(())),
        ),
        patch(
            "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
            side_effect=RuntimeError("model unavailable"),
        ),
        pytest.raises(RuntimeError, match="model unavailable"),
    ):
        MainExecutionOps.enrich_space_snapshot_with_notes(snapshot, _graph(), space, str(notes))
    notes.unlink(missing_ok=True)

    # 2) SQL parse failure must not be classified as forbidden SQL.
    with patch("sqlglot.parse_one", side_effect=sqlglot.errors.ParseError("broken")):
        with pytest.raises(ConfigError, match="SQL parse failed"):
            _pg_parsed_has_forbidden_sql(object(), sql="SELECT broken", sqlglot_dialect="postgres")

    mock_dialect = MagicMock()
    mock_dialect.name = "postgresql"
    mock_dialect.sqlglot_dialect = "postgres"
    mock_dialect.parse_select.return_value = object()
    with patch("sqlglot.parse_one", side_effect=sqlglot.errors.ParseError("broken")):
        ok, reason = _enforce_select_only("SELECT broken", mock_dialect)
    assert ok is False
    assert reason.startswith("sql_parse_failed")

    # 3) Expression parse failure must raise, not return a raw expression fallback.
    previous_parse = NormalizedExpr._parse_expr_string_fn

    def _boom(_s: str) -> NormalizedExpr:
        raise RuntimeError("boom")

    NormalizedExpr.register_parse_expr_string(_boom)
    try:
        with pytest.raises(ConfigError, match="expression parse failed"):
            NormalizedExpr.coerce_mul_term("sum(broken)")
    finally:
        NormalizedExpr.register_parse_expr_string(previous_parse)
        if NormalizedExpr._parse_expr_string_fn is None:
            import aetherdialect._intent_expr

            assert aetherdialect._intent_expr is not None


@pytest.mark.fast
def test_no_bare_exception_pass_in_src() -> None:
    """No broad ``except Exception`` with pass/continue/bare-return anywhere in the package."""
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, lineno, kind in _bare_exception_pass_sites(tree, path=path):
            violations.append(f"{name}:{lineno} ({kind})")
    if violations:
        pytest.fail("Bare broad-exception swallow site(s):\n    " + "\n    ".join(violations))
