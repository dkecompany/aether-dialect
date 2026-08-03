"""Enforce static hygiene: forbidden suppressions, import rules, and constant locations."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "aetherdialect"
_SCRIPTS = _ROOT / "scripts"

# Files exempt from prefix sweeps (contract/config façades and public entrypoints).
_PREFIX_EXEMPT = {
    "__init__.py",
    "aetherdialect.py",
    "_config.py",
    "_constants.py",
    "_contracts_base.py",
    "_contracts_core.py",
    "_contracts_schema.py",
}

# Files exempt from the "no constants outside _constants.py" rule.
# Static package data lives in _constants.py; contract/config façades define their own shapes.
_CONST_EXEMPT = {
    "_constants.py",
    "_config.py",
    "_contracts_base.py",
    "_contracts_core.py",
    "_contracts_schema.py",
    "aetherdialect.py",
}

# Pre-existing module-level constants outside _constants.py. Frozen: do not add entries;
# remove a row when the constant is migrated into _constants.py.
_CONST_LOCATION_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset(
    {
        ("_core_utils.py", "_DIAGNOSTIC_FORCE_DEPTH"),
    }
)

# Files allowed to re-export via __all__
_REEXPORT_ALLOWED = {
    "__init__.py",
    "aetherdialect.py",
}

_FORBIDDEN_PATTERNS = [
    r"#\s*type:\s*ignore",
    r"#\s*noqa",
    r"#\s*pylint:\s*disable",
    r"#\s*pyright:\s*ignore",
    r"#\s*mypy:\s*ignore",
    r"#\s*ruff:\s*noqa",
    r"#\s*pragma:\s*no\s*cover",
    r"#\s*fmt:\s*off",
    r"#\s*fmt:\s*on",
]

_DIRS_TO_SCAN = ["src", "scripts", "tests", "live_tests"]
_DOCS = _ROOT / "docs"

# Low-level modules must not depend on orchestration layers.
_ORCHESTRATION_MODULES = frozenset({"_pipeline", "_main_execution"})
_LOW_LEVEL_MODULES = frozenset(
    {
        "_constants",
        "_config",
        "_contracts_base",
        "_contracts_core",
        "_contracts_schema",
        "_core_utils",
    }
)

# Directed edges that must never appear even if the overall graph stays acyclic.
_BANNED_IMPORT_EDGES = frozenset({("_sql_to_intent_sqlglot", "_sql_to_intent")})

_INTERNAL_DOC_IMPORT_RE = re.compile(
    r"(?:from\s+aetherdialect\s+import\s+_|import\s+aetherdialect\._|aetherdialect\._[A-Za-z]\w*)"
)


def _module_stem(path: Path) -> str:
    return "__init__" if path.name == "__init__.py" else path.stem


def _resolve_relative_target(importer: str, node: ast.ImportFrom) -> str | None:
    if node.level <= 0:
        if node.module and node.module.startswith("aetherdialect."):
            return node.module.split(".", 1)[1].split(".")[0]
        return None
    if not node.module:
        return None
    return node.module.split(".")[0]


def _build_internal_import_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in _get_src_files():
        importer = _module_stem(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deps: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve_relative_target(importer, node)
            if target and (target.startswith("_") or target == "aetherdialect"):
                if target != "aetherdialect":
                    deps.add(target)
        graph[importer] = deps
    return graph


def _find_import_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visited: set[str] = set()
    stack: list[str] = []
    on_stack: set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        on_stack.add(node)
        stack.append(node)
        for neighbor in sorted(graph.get(node, ())):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in on_stack:
                idx = stack.index(neighbor)
                cycles.append(stack[idx:] + [neighbor])
        stack.pop()
        on_stack.remove(node)

    for node in sorted(graph):
        if node not in visited:
            dfs(node)
    return cycles


def _iter_comment_violations(path: Path) -> list[str]:
    import io
    import tokenize

    source = path.read_text(encoding="utf-8")
    violations: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            text = tok.string.strip()
            if text.startswith("#!"):
                continue
            if "coding" in text and "utf-8" in text:
                continue
            violations.append(f"line {tok.start[0]}: {text}")
    except tokenize.TokenError as exc:
        violations.append(f"tokenize error: {exc}")
    return violations


def _module_level_imports_after_definitions(tree: ast.Module) -> list[int]:
    first_def_line: int | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def_line = node.lineno
            break
    if first_def_line is None:
        return []

    violations: list[int] = []
    for node in tree.body:
        if node.lineno <= first_def_line:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append(node.lineno)
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    violations.append(child.lineno)
        if isinstance(node, ast.If):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    violations.append(child.lineno)
    return violations


def _get_python_files():
    for dname in _DIRS_TO_SCAN:
        dpath = _ROOT / dname
        if not dpath.is_dir():
            continue
        for p in dpath.rglob("*.py"):
            # Skip this file itself to avoid false positives from the pattern list
            if p.name == "test_static_hygiene.py":
                continue
            yield p


def _is_re_compile_call(node: ast.AST) -> bool:
    """Return True when *node* is a ``re.compile(...)`` call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
        and func.attr == "compile"
    )


_MODULE_CONSTANT_NAME_RE = re.compile(r"^(?:_[A-Z][A-Z0-9_]*|[A-Z][A-Z0-9_]*)$")

_CONSTANT_FACTORY_NAMES = frozenset({"dict", "frozenset", "list", "set", "tuple"})


def _is_module_constant_name(name: str) -> bool:
    """Return True for module-level constant identifiers, including private ``_NAME`` forms."""
    if name.startswith("__") and name.endswith("__"):
        return False
    return _MODULE_CONSTANT_NAME_RE.match(name) is not None


def _is_mutable_runtime_global(node: ast.AST | None) -> bool:
    """Return True for module globals that are intentionally mutable runtime state."""
    if node is None:
        return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Dict) and len(node.keys) == 0:
        return True
    if isinstance(node, ast.List) and len(node.elts) == 0:
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "object":
        return True
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "threading"
        and func.attr == "local"
    ):
        return True
    return isinstance(func, ast.Name) and func.id == "set" and not node.args


def _is_constant_like_value(node: ast.AST | None) -> bool:
    """Return True when *node* looks like static package data rather than runtime state."""
    if node is None or _is_mutable_runtime_global(node):
        return False
    if isinstance(node, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return True
    if _is_re_compile_call(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_constant_like_value(node.left) and _is_constant_like_value(node.right)
    if isinstance(node, ast.JoinedStr):
        return all(
            isinstance(part, ast.Constant)
            or (isinstance(part, ast.FormattedValue) and _is_constant_like_value(part.value))
            for part in node.values
        )
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id in _CONSTANT_FACTORY_NAMES:
        return True
    if isinstance(func, ast.Attribute) and func.attr in _CONSTANT_FACTORY_NAMES:
        return True
    return False


def _module_constant_assignment_violations(tree: ast.Module, *, file_name: str) -> list[str]:
    """Return human-readable violations for bare module-level constants."""
    violations: list[str] = []
    for node in tree.body:
        targets: list[ast.Name] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets.append(node.target)
            value = node.value
        for target in targets:
            if (file_name, target.id) in _CONST_LOCATION_GRANDFATHERED:
                continue
            if not _is_module_constant_name(target.id):
                continue
            if not _is_constant_like_value(value):
                continue
            violations.append(f"line {node.lineno}: {target.id}")
    return violations


def _get_src_files():
    yield from _SRC.glob("*.py")


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_python_files()))
def test_no_forbidden_suppressions(file_path: Path) -> None:
    """Scan file for forbidden suppression comments."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    violations = []
    for i, line in enumerate(lines, 1):
        for pattern in _FORBIDDEN_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(f"Line {i}: {line.strip()}")
                break

    if violations:
        pytest.fail(f"Forbidden suppression(s) found in {file_path.relative_to(_ROOT)}:\n" + "\n".join(violations))


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_no_lazy_or_local_imports(file_path: Path) -> None:
    """Ensure all aetherdialect imports are at module top-level."""
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in ast.walk(tree):
        # We only care about imports inside functions or classes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    # Check if it imports from . or aetherdialect
                    is_package_import = False
                    if isinstance(subnode, ast.Import):
                        for alias in subnode.names:
                            if alias.name.startswith("aetherdialect"):
                                is_package_import = True
                    elif isinstance(subnode, ast.ImportFrom):
                        if subnode.level > 0 or (
                            subnode.module
                            and (subnode.module == "aetherdialect" or subnode.module.startswith("aetherdialect."))
                        ):
                            is_package_import = True

                    if is_package_import:
                        pytest.fail(
                            f"Local/lazy import of aetherdialect found in {file_path.name} "
                            f"at line {subnode.lineno}. All package imports must be at top-level."
                        )


@pytest.mark.fast
def test_module_constant_name_includes_private_re_suffix() -> None:
    """Private ``_NAME_RE`` identifiers must be treated as module constants."""
    assert _is_module_constant_name("_QUALIFIED_COLUMN_TOKEN_RE")
    assert not "qualified_column_token_re".isupper()


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_no_compiled_regex_outside_constants(file_path: Path) -> None:
    """Ensure ``re.compile`` patterns live only in ``_constants.py``."""
    if file_path.name in _CONST_EXEMPT:
        return

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in tree.body:
        value: ast.AST | None = None
        names: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            names.append(node.target.id)
        if value is None or not names or not _is_re_compile_call(value):
            continue
        label = ", ".join(names)
        violations.append(f"line {node.lineno}: {label}")

    if violations:
        pytest.fail(f"Compiled regex found in {file_path.name}; move to _constants.py:\n" + "\n".join(violations))


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_constant_locations(file_path: Path) -> None:
    """Ensure bare module-level constants (including ``_PRIVATE_NAME``) live only in ``_constants.py``."""
    if file_path.name in _CONST_EXEMPT:
        return

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    violations = _module_constant_assignment_violations(tree, file_name=file_path.name)
    if violations:
        pytest.fail(f"Bare constant(s) found in {file_path.name}; move to _constants.py:\n" + "\n".join(violations))


@pytest.mark.fast
def test_config_file_has_only_classes() -> None:
    """Ensure _config.py only contains configuration classes and no bare logic or constants."""
    path = _SRC / "_config.py"
    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    allowed_names = {
        "ConfigError",
        "llm_credentials_configured",
        "normalize_column_type",
        "env_any_nonempty",
        "env_first_nonempty",
        "env_role_hint",
        "package_importable",
    }

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    pytest.fail(
                        f"Bare assignment {target.id} found in _config.py. Only classes and ConfigError allowed."
                    )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_") and node.name not in allowed_names:
                pytest.fail(f"Public function {node.name} found in _config.py. Only classes and ConfigError allowed.")


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_python_files()))
def test_no_aliased_imports(file_path: Path) -> None:
    """Ensure no aliased imports of any symbols from aetherdialect."""
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            is_internal = False
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("aetherdialect"):
                        is_internal = True
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 or (
                    node.module and (node.module == "aetherdialect" or node.module.startswith("aetherdialect."))
                ):
                    is_internal = True

            if is_internal:
                for alias in node.names:
                    if alias.asname:
                        pytest.fail(
                            f"Aliased internal import '{alias.name} as {alias.asname}' found in {file_path.relative_to(_ROOT)} "
                            f"at line {node.lineno}. No aliasing of internal symbols."
                        )


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_no_internal_reexporting(file_path: Path) -> None:
    """Ensure internal modules do not re-export symbols via __all__."""
    if file_path.name in _REEXPORT_ALLOWED:
        return

    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    pytest.fail(
                        f"Found __all__ in internal module {file_path.name} at line {node.lineno}. "
                        "Re-exporting is only allowed in __init__.py and aetherdialect.py."
                    )


@pytest.mark.fast
def test_prefix_reachability() -> None:
    """Cross-module imports must not pull ``_``-prefixed symbols from sibling modules."""
    all_files = list(_get_src_files())

    defined_symbols: dict[str, set[str]] = {}
    imported_symbols: dict[str, set[str]] = {}

    for file_path in all_files:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        if file_path.name not in _PREFIX_EXEMPT:
            for node in tree.body:
                names = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names = [node.name]
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.append(target.id)

                for name in names:
                    defined_symbols.setdefault(name, set()).add(file_path.name)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                is_internal = node.level > 0 or (
                    node.module is not None
                    and (node.module == "aetherdialect" or node.module.startswith("aetherdialect."))
                )
                if is_internal:
                    for alias in node.names:
                        imported_symbols.setdefault(alias.name, set()).add(file_path.name)

    violations = []
    for sym, def_files in defined_symbols.items():
        if not sym.startswith("_") or sym.startswith("__"):
            continue
        importers = imported_symbols.get(sym, set())
        cross_module_importers = importers - def_files
        if cross_module_importers:
            violations.append(
                f"Symbol '{sym}' defined in {def_files} starts with '_' but is imported cross-module by {cross_module_importers}. Remove '_' prefix if cross-module."
            )

    if violations:
        pytest.fail("\n".join(violations))


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_no_namespace_imports(file_path: Path) -> None:
    """Disallow ``from . import _module`` namespace imports."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is None and node.level > 0:
            names = ", ".join(alias.name for alias in node.names)
            pytest.fail(f"Namespace import from . in {file_path.name} at line {node.lineno}: {names}")


@pytest.mark.fast
def test_package_import_dag_acyclic() -> None:
    """Internal package modules must form a directed acyclic import graph."""
    graph = _build_internal_import_graph()
    cycles = _find_import_cycles(graph)
    if cycles:
        rendered = [" -> ".join(cycle) for cycle in cycles]
        pytest.fail("Import cycle(s) detected:\n" + "\n".join(rendered))


@pytest.mark.fast
def test_banned_import_edges() -> None:
    """Enforce explicit import prohibitions beyond global acyclicity."""
    graph = _build_internal_import_graph()
    violations = [
        f"{src} must not import {dst}" for src, dst in sorted(_BANNED_IMPORT_EDGES) if dst in graph.get(src, set())
    ]
    if violations:
        pytest.fail("\n".join(violations))


@pytest.mark.fast
def test_low_level_avoids_orchestration_imports() -> None:
    """Foundation and utility modules must not import orchestration layers."""
    graph = _build_internal_import_graph()
    violations = []
    for module in sorted(_LOW_LEVEL_MODULES):
        bad = sorted(graph.get(module, set()) & _ORCHESTRATION_MODULES)
        if bad:
            violations.append(f"{module} imports orchestration module(s): {bad}")
    if violations:
        pytest.fail("\n".join(violations))


@pytest.mark.fast
@pytest.mark.parametrize(
    "file_path",
    sorted(p for d in (_SRC, _SCRIPTS) if d.is_dir() for p in d.rglob("*.py") if p.name != "test_static_hygiene.py"),
)
def test_no_inline_hash_comments(file_path: Path) -> None:
    """Library and maintainer scripts use docstrings instead of inline ``#`` comments."""
    violations = _iter_comment_violations(file_path)
    if violations:
        rel = file_path.relative_to(_ROOT)
        pytest.fail(f"Inline comment(s) in {rel}:\n" + "\n".join(violations))


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_no_wildcard_imports(file_path: Path) -> None:
    """Disallow star-imports inside the package."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            pytest.fail(f"Wildcard import in {file_path.name} at line {node.lineno}")


@pytest.mark.fast
@pytest.mark.parametrize("file_path", list(_get_src_files()))
def test_no_post_definition_module_imports(file_path: Path) -> None:
    """Module-level imports must appear before the first top-level definition."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    violations = _module_level_imports_after_definitions(tree)
    if violations:
        lines = ", ".join(str(line) for line in violations)
        pytest.fail(f"Post-definition module import(s) in {file_path.name} at line(s) {lines}")


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", sorted(_DOCS.glob("*.md")) if _DOCS.is_dir() else [])
def test_docs_avoid_internal_import_paths(doc_path: Path) -> None:
    """User docs must not teach ``aetherdialect._*`` internal import paths."""
    text = doc_path.read_text(encoding="utf-8")
    hits = [
        f"line {idx}: {line.strip()}"
        for idx, line in enumerate(text.splitlines(), 1)
        if _INTERNAL_DOC_IMPORT_RE.search(line)
    ]
    if hits:
        rel = doc_path.relative_to(_ROOT)
        pytest.fail(f"Internal import path(s) in {rel}:\n" + "\n".join(hits))


_DOC_META_HEADING_RE = re.compile(r"^##\s+(New|Updated|Changelog)\b", re.IGNORECASE)
_DOC_PLAN_TEST_FILE_RE = re.compile(r"test_(?:step\d+|phase_[a-z])", re.IGNORECASE)


def _doc_body_sections(lines: list[str]) -> list[int]:
    """Return line indices of ``##`` headings that are not the front- matter ``## Sections``."""
    return [idx for idx, line in enumerate(lines) if line.startswith("## ") and not line.startswith("## Sections")]


def _doc_front_matter_violations(doc_path: Path) -> list[str]:
    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    violations: list[str] = []
    if not lines or not lines[0].startswith("# "):
        violations.append("first line must be an H1 title")
        return violations
    try:
        reading_idx = next(idx for idx, line in enumerate(lines) if line.startswith("**Reading order:**"))
    except StopIteration:
        violations.append("missing **Reading order:** line")
        return violations
    try:
        sections_idx = next(idx for idx, line in enumerate(lines) if line.strip() == "## Sections")
    except StopIteration:
        violations.append("missing ## Sections heading")
        return violations
    about_lines = [line for line in lines[1:reading_idx] if line.strip()]
    if not about_lines:
        violations.append("missing About paragraph before **Reading order:**")
    if not (reading_idx < sections_idx):
        violations.append("**Reading order:** must appear before ## Sections")
    separator_indices = [idx for idx, line in enumerate(lines) if line.strip() == "---"]
    if not separator_indices:
        violations.append("missing --- separator before body")
        return violations
    body_headings = _doc_body_sections(lines)
    if not body_headings:
        violations.append("missing body section heading (## …)")
        return violations
    first_body = body_headings[0]
    if not any(idx < first_body for idx in separator_indices):
        violations.append("--- must appear before the first body section")
    return violations


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", sorted(_DOCS.glob("*.md")) if _DOCS.is_dir() else [])
def test_doc_guides_follow_front_matter_template(doc_path: Path) -> None:
    """Each docs/*.md guide uses About → Reading order → Sections → --- → body."""
    violations = _doc_front_matter_violations(doc_path)
    if violations:
        rel = doc_path.relative_to(_ROOT)
        pytest.fail(f"{rel} front-matter violations:\n" + "\n".join(violations))


@pytest.mark.fast
def test_join_helper_modules_are_not_standalone_files() -> None:
    assert not (_SRC / "_join_fan_out.py").is_file()
    assert not (_SRC / "_qsim_ops.py").is_file()
    assert not (_SRC / "_join_comparison_scope.py").is_file()


@pytest.mark.fast
def test_needs_corpus_marker_is_registered() -> None:
    import tomllib

    markers = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"][
        "markers"
    ]
    assert any(str(entry).startswith("needs_corpus:") for entry in markers)


@pytest.mark.fast
def test_corpus_absent_skip_reason_names_needs_corpus() -> None:
    text = (_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "needs_corpus: bundled sandbox data.zip absent" in text
    assert "sandbox data.zip not ready" not in text


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", sorted(_DOCS.glob("*.md")) if _DOCS.is_dir() else [])
def test_docs_avoid_plan_meta_headings(doc_path: Path) -> None:
    """User docs must not use changelog-style headings or reference plan-style test filenames."""
    text = doc_path.read_text(encoding="utf-8")
    hits: list[str] = []
    for idx, line in enumerate(text.splitlines(), 1):
        if _DOC_META_HEADING_RE.search(line):
            hits.append(f"line {idx}: {line.strip()}")
        if _DOC_PLAN_TEST_FILE_RE.search(line):
            hits.append(f"line {idx}: plan-style test filename reference: {line.strip()}")
    if hits:
        rel = doc_path.relative_to(_ROOT)
        pytest.fail(f"Plan meta heading(s) in {rel}:\n" + "\n".join(hits))
