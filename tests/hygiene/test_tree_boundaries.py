"""Tree-boundary hygiene: keep src/scripts/docs/tests import and path walls intact."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
_DOCS = _ROOT / "docs"
_HYGIENE = Path(__file__).resolve().parent

_FORBIDDEN_PKG_ROOTS = frozenset(
    {
        "dev_workspace",
        "localdemo",
        "tests",
        "live_tests",
        "scripts",
    }
)
_SCRIPTS_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "dev_workspace",
        "localdemo",
        "tests",
        "live_tests",
    }
)

_SRC_PATH_LITERAL_RE = re.compile(
    r"(?:"
    r"\bdev_workspace\b"
    r"|\blocaldemo\b"
    r"|\blive_tests\b"
    r"|tests/"
    r"|scripts/"
    r"|`tests`"
    r"|`scripts`"
    r"|['\"]tests['\"]"
    r"|['\"]scripts['\"]"
    r")",
    re.IGNORECASE,
)

_DOCS_PATH_RE = re.compile(
    r"(?:"
    r"\bdev_workspace\b"
    r"|\blocaldemo\b"
    r"|live_tests/"
    r"|`live_tests`"
    r"|tests/"
    r"|scripts/"
    r"|`tests`"
    r"|`scripts`"
    r")",
    re.IGNORECASE,
)

_TEXT_SUFFIXES = frozenset({".py", ".md", ".rst", ".toml", ".yml", ".yaml", ".txt", ".json"})


def _should_skip_path(path: Path) -> bool:
    for part in path.parts:
        if part in {".egg-info", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}:
            return True
        if part.endswith(".egg-info"):
            return True
    return False


def _iter_py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file() and not _should_skip_path(path))


def _iter_text_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_path(path):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        out.append(path)
    return sorted(out)


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root:
                    roots.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue
            module = node.module or ""
            root = module.split(".", 1)[0] if module else ""
            if root:
                roots.add(root)
    return roots


def _is_hygiene_path(path: Path) -> bool:
    try:
        path.relative_to(_HYGIENE)
        return True
    except ValueError:
        return False


@pytest.mark.fast
def test_src_ast_forbids_cross_tree_imports() -> None:
    hits: list[str] = []
    for path in _iter_py_files(_SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = sorted(_import_roots(tree) & _FORBIDDEN_PKG_ROOTS)
        if bad:
            rel = path.relative_to(_ROOT).as_posix()
            hits.append(f"{rel}: forbidden import root(s) {bad}")
    if hits:
        pytest.fail("src/ imports outside the package boundary:\n" + "\n".join(hits))


@pytest.mark.fast
def test_src_text_forbids_cross_tree_path_literals() -> None:
    hits: list[str] = []
    for path in _iter_text_files(_SRC):
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            if _SRC_PATH_LITERAL_RE.search(line):
                rel = path.relative_to(_ROOT).as_posix()
                hits.append(f"{rel}:{idx}: {line.strip()[:160]}")
    if hits:
        pytest.fail("src/ path literal(s) mention forbidden trees:\n" + "\n".join(hits))


@pytest.mark.fast
def test_scripts_ast_forbids_live_tests_and_private_trees() -> None:
    hits: list[str] = []
    for path in _iter_py_files(_SCRIPTS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = sorted(_import_roots(tree) & _SCRIPTS_FORBIDDEN_IMPORT_ROOTS)
        if bad:
            rel = path.relative_to(_ROOT).as_posix()
            hits.append(f"{rel}: forbidden import root(s) {bad}")
    if hits:
        pytest.fail("scripts/ imports forbidden trees:\n" + "\n".join(hits))


@pytest.mark.fast
def test_docs_forbid_repo_tree_path_mentions() -> None:
    hits: list[str] = []
    for path in _iter_text_files(_DOCS):
        if _is_hygiene_path(path):
            continue
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            if _DOCS_PATH_RE.search(line):
                rel = path.relative_to(_ROOT).as_posix()
                hits.append(f"{rel}:{idx}: {line.strip()[:160]}")
    if hits:
        pytest.fail("docs/ mention maintainer/repo trees:\n" + "\n".join(hits))


@pytest.mark.fast
def test_shipped_trees_forbid_dev_workspace_and_localdemo_outside_hygiene() -> None:
    """``dev_workspace`` / ``localdemo`` may appear only under ``tests/hygiene/``."""
    pattern = re.compile(r"\b(?:dev_workspace|localdemo)\b")
    hits: list[str] = []
    for root_name in ("src", "scripts", "tests", "live_tests", "docs"):
        root = _ROOT / root_name
        for path in _iter_text_files(root):
            if _is_hygiene_path(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if pattern.search(text):
                hits.append(path.relative_to(_ROOT).as_posix())
    if hits:
        pytest.fail(
            "dev_workspace/localdemo mentions outside tests/hygiene/:\n" + "\n".join(hits),
        )
