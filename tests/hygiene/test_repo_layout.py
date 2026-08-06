"""Ensure shipped package code never walks into repository maintainer directories."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "aetherdialect"

_PARENTS_ESCAPE_RE = re.compile(
    r"\.parents\[(?:[2-9]|\d{2,})\]",
)
# Concatenated raw strings (not triple-quotes) keep docformatter from
# treating pattern bodies as docstrings and oscillating with ruff-format.
_REPO_DIR_LITERAL_RE = re.compile(
    (
        r"Scripts[\\/]data | scripts[\\/]logs | dev_workspace[\\/] |"
        r" [\\/]dev_workspace(?:[\\/]|$) | live_tests[\\/] |"
        r" [\\/]live_tests(?:[\\/]|$) | (?:^|[\\/])tests[\\/]"
    ),
    re.IGNORECASE | re.VERBOSE,
)
_ALWAYS_FORBIDDEN_SEGMENT_RE = re.compile(
    r'(?:/|\\)\s*["\'](?:scripts|dev_workspace)["\']',
)
_REPO_LAYOUT_SEGMENT_RE = re.compile(
    r'(?:/|\\)\s*["\'](?:live_tests|tests)["\']',
)
_FILE_PARENTS_ESCAPE_RE = re.compile(
    r"Path\s*\(\s*__file__\s*\)(?:\s*\.\s*\w+\([^)]*\))*"
    r"\.parents\[(?:[2-9]|\d{2,})\]",
)
_FILE_DOTDOT_SEGMENT_RE = re.compile(
    r'__file__[^\n]*?/\s*["\']\.\.["\']',
)
_PATH_TRAVERSAL_LITERAL_RE = re.compile(
    r"(?:/|\\)\s*\.\.(?:/|\\)",
)


def _docstring_nodes(tree: ast.Module) -> set[ast.AST]:
    documented: set[ast.AST] = set()
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        documented.add(tree.body[0].value)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                documented.add(node.body[0].value)
    return documented


def _parents_index(node: ast.Subscript) -> int | None:
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, int):
        return key.value
    return None


def _scan_source_tree_reference_violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    docstrings = _docstring_nodes(tree)
    violations: list[str] = []

    for match in _PARENTS_ESCAPE_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        violations.append(f"line {line}: package escape via {match.group(0)!r}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            attr = node.value
            if (
                isinstance(attr, ast.Attribute)
                and attr.attr == "parents"
                and (idx := _parents_index(node)) is not None
                and idx >= 2
            ):
                violations.append(f"line {node.lineno}: parents[{idx}] escapes package root")

        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings:
            value = node.value
            if _REPO_DIR_LITERAL_RE.search(value):
                violations.append(f"line {node.lineno}: repository directory literal {value!r}")

    for idx, line in enumerate(text.splitlines(), 1):
        segment_match = _ALWAYS_FORBIDDEN_SEGMENT_RE.search(line)
        if segment_match:
            violations.append(
                f"line {idx}: repository path segment {segment_match.group(0)!r}",
            )
            continue
        if "__file__" in line or ".parents[" in line:
            layout_match = _REPO_LAYOUT_SEGMENT_RE.search(line)
            if layout_match:
                violations.append(
                    f"line {idx}: repository path segment {layout_match.group(0)!r}",
                )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in violations:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _scan_package_root_escape_violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    docstrings = _docstring_nodes(tree)
    violations: list[str] = []

    for match in _PARENTS_ESCAPE_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        violations.append(f"line {line}: package escape via {match.group(0)!r}")

    for match in _FILE_PARENTS_ESCAPE_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        violations.append(f"line {line}: __file__ path escapes package via {match.group(0)!r}")

    for match in _FILE_DOTDOT_SEGMENT_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        violations.append(f"line {line}: __file__ path escapes package via {match.group(0)!r}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            attr = node.value
            if (
                isinstance(attr, ast.Attribute)
                and attr.attr == "parents"
                and (idx := _parents_index(node)) is not None
                and idx >= 2
            ):
                violations.append(f"line {node.lineno}: parents[{idx}] escapes package root")

        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings:
            value = node.value
            if _PATH_TRAVERSAL_LITERAL_RE.search(value):
                violations.append(f"line {node.lineno}: path traversal literal {value!r}")

    for idx, line in enumerate(text.splitlines(), 1):
        if _PATH_TRAVERSAL_LITERAL_RE.search(line):
            violations.append(f"line {idx}: path traversal segment {line.strip()!r}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in violations:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _iter_src_modules() -> list[Path]:
    return sorted(_SRC.rglob("*.py"))


@pytest.mark.fast
def test_src_never_references_repository_directories() -> None:
    hits: list[str] = []
    for path in _iter_src_modules():
        rel = path.relative_to(_SRC)
        violations = _scan_source_tree_reference_violations(path)
        for violation in violations:
            hits.append(f"{rel}: {violation}")
    if hits:
        pytest.fail("Repository directory reference(s) in src/aetherdialect:\n" + "\n".join(hits))


@pytest.mark.fast
def test_src_never_traverses_above_package_root() -> None:
    hits: list[str] = []
    for path in _iter_src_modules():
        rel = path.relative_to(_SRC)
        violations = _scan_package_root_escape_violations(path)
        for violation in violations:
            hits.append(f"{rel}: {violation}")
    if hits:
        pytest.fail("Package root escape(s) in src/aetherdialect:\n" + "\n".join(hits))


@pytest.mark.fast
def test_package_root_escape_scanner_catches_violations() -> None:
    """Guardrail: extended scanner must fail on representative escape patterns."""
    samples = [
        'ROOT = Path(__file__).resolve().parents[2] / "scripts"\n',
        "p = Path(__file__).resolve().parents[3]\n",
        'data = "../../scripts/data/foo.json"\n',
        'p = Path(__file__).parent / ".." / ".." / "outside"\n',
    ]
    for sample in samples:
        violations = _scan_package_root_escape_violations_from_text(sample)
        assert violations, f"scanner missed violation in sample: {sample!r}"


def _scan_package_root_escape_violations_from_text(text: str) -> list[str]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return _scan_package_root_escape_violations(path)
    finally:
        path.unlink(missing_ok=True)
