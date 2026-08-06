"""Enforce static hygiene: forbidden suppressions, import rules, and constant locations."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "aetherdialect"
_SCRIPTS = _ROOT / "scripts"

# Public façades only: definitions in config/constants/contracts are scanned for cross-module ``_`` leaks.
_PREFIX_EXEMPT = {
    "__init__.py",
    "aetherdialect.py",
}

# Static package data lives in ``_constants.py``. Contract modules may host type-adjacent
# membership tables; ``_config.py`` is not exempt (no module-level SCREAMING registries).
_CONST_EXEMPT = {
    "_constants.py",
    "_contracts_base.py",
    "_contracts_core.py",
    "_contracts_schema.py",
    "aetherdialect.py",
}

# Files allowed to re-export via __all__
_REEXPORT_ALLOWED = {
    "__init__.py",
    "aetherdialect.py",
}

_FACADE_MODULES = frozenset({"__init__", "aetherdialect"})

# Cross-module imports of public template helpers are allowed without a private-prefix exemption.
_ALLOWED_PRIVATE_SYMBOL_IMPORTS = frozenset()

_DATA_MODULES = frozenset({"_constants"})

_CLASS_MODULES = frozenset(
    {
        "_contracts_base",
        "_contracts_core",
        "_contracts_schema",
        "_config",
        "_dialect",
        "_dialect_postgres",
        "_dialect_sqlglot_helper",
        "_dialect_sqlglot_engines",
        "_sandbox",
        "_templates",
        "_llm_provider",
        "_seed_warmup",
        "_main_execution",
    }
)

_EXPECTED_PACKAGE_MODULES = frozenset(
    {
        "__init__",
        "aetherdialect",
        "_constants",
        "_contracts_base",
        "_contracts_core",
        "_contracts_schema",
        "_config",
        "_dialect",
        "_dialect_postgres",
        "_dialect_sqlglot_helper",
        "_dialect_sqlglot_engines",
        "_sandbox",
        "_templates",
        "_llm_provider",
        "_seed_warmup",
        "_main_execution",
        "_data_quality",
        "_expansion_ops",
        "_intent_expr",
        "_intent_repair",
        "_intent_resolve",
        "_intent_process",
        "_qsim",
        "_schema_build",
        "_schema_overrides",
        "_schema_catalog",
        "_schema_graph",
        "_utils",
        "_validation_execute",
        "_validation_schema",
        "_validation_semantic",
        "_sql_gen",
        "_sql_to_intent",
        "_sql_to_intent_sqlglot",
        "_core_utils",
        "_federation",
        "_pipeline",
        "_live_testing",
    }
)

_CONTRACT_MODULES = frozenset({"_contracts_base", "_contracts_core", "_contracts_schema"})
_DIALECT_MODULES = frozenset(
    {
        "_dialect",
        "_dialect_postgres",
        "_dialect_sqlglot_helper",
        "_dialect_sqlglot_engines",
    }
)

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
_BANNED_IMPORT_EDGES = frozenset(
    {
        ("_sql_to_intent_sqlglot", "_sql_to_intent"),
        ("_dialect_sqlglot_engines", "_sql_to_intent"),
        ("_dialect_sqlglot_helper", "_sql_to_intent"),
        ("_dialect_postgres", "_sql_to_intent"),
        ("_dialect", "_sql_to_intent"),
        ("_dialect_sqlglot_engines", "_sql_to_intent_sqlglot"),
        ("_dialect_sqlglot_helper", "_sql_to_intent_sqlglot"),
        ("_dialect_postgres", "_sql_to_intent_sqlglot"),
        ("_dialect", "_sql_to_intent_sqlglot"),
    }
)

_CLASS_NAME_RE = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
_FUNC_NAME_RE = re.compile(r"^_?[a-z][a-z0-9_]*$")
_CONST_NAME_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
_DUNDER_NAME_RE = re.compile(r"^__[a-zA-Z0-9_]+__$")


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
            if p.name == "test_static_core.py":
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


def _is_mutable_runtime_global(node: ast.AST | None, *, allow_empty_containers: bool = True) -> bool:
    """Return True for module globals that are intentionally mutable runtime state."""
    if node is None:
        return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if allow_empty_containers:
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


def _is_constant_like_value(node: ast.AST | None, *, allow_empty_containers: bool = True) -> bool:
    """Return True when *node* looks like static package data rather than runtime state."""
    if node is None or _is_mutable_runtime_global(node, allow_empty_containers=allow_empty_containers):
        return False
    if isinstance(node, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return True
    if _is_re_compile_call(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_constant_like_value(
            node.left, allow_empty_containers=allow_empty_containers
        ) and _is_constant_like_value(node.right, allow_empty_containers=allow_empty_containers)
    if isinstance(node, ast.JoinedStr):
        return all(
            isinstance(part, ast.Constant)
            or (
                isinstance(part, ast.FormattedValue)
                and _is_constant_like_value(part.value, allow_empty_containers=allow_empty_containers)
            )
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


def _is_literal_subscript(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if isinstance(node.value, ast.Name) and node.value.id == "Literal":
        return True
    return isinstance(node.value, ast.Attribute) and node.value.attr == "Literal"


def _is_literal_type_alias_assignment(node: ast.AST) -> bool:
    """Return True for module-level ``Name = Literal[...]`` (optionally annotated)."""
    value: ast.AST | None = None
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) for t in node.targets):
        value = node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        value = node.value
    return _is_literal_subscript(value)


def _top_level_defs(tree: ast.Module) -> tuple[list[ast.FunctionDef | ast.AsyncFunctionDef], list[ast.ClassDef]]:
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    classes: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
    return functions, classes


def _module_constant_assignment_violations(tree: ast.Module, *, file_name: str) -> list[str]:
    """Return human-readable violations for bare module-level constants."""
    allow_empty = file_name != "_config.py"
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
            if not _is_module_constant_name(target.id):
                continue
            if file_name == "_config.py":
                violations.append(f"line {node.lineno}: {target.id}")
                continue
            if not _is_constant_like_value(value, allow_empty_containers=allow_empty):
                continue
            violations.append(f"line {node.lineno}: {target.id}")
    return violations


def _get_src_files():
    yield from sorted(_SRC.glob("*.py"), key=lambda p: p.name)


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
def test_no_module_level_constants_outside_constants(file_path: Path) -> None:
    """Ensure bare module-level constants (including ``_PRIVATE_NAME``) live only in ``_constants.py``."""
    if file_path.name in _CONST_EXEMPT:
        return

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    violations = _module_constant_assignment_violations(tree, file_name=file_path.name)
    if violations:
        pytest.fail(f"Bare constant(s) found in {file_path.name}; move to _constants.py:\n" + "\n".join(violations))


@pytest.mark.fast
def test_config_has_no_module_level_registries() -> None:
    """``_config.py`` must not host module-level SCREAMING registries or empty-dict globals."""
    path = _SRC / "_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
            if _is_module_constant_name(target.id):
                violations.append(f"line {node.lineno}: SCREAMING registry {target.id}")
                continue
            if isinstance(value, ast.Dict) and len(value.keys) == 0:
                violations.append(f"line {node.lineno}: empty-dict global {target.id}")
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "dict"
                and not value.args
                and not value.keywords
            ):
                violations.append(f"line {node.lineno}: empty dict() global {target.id}")
    if violations:
        pytest.fail("Module-level registries in _config.py:\n" + "\n".join(violations))


@pytest.mark.fast
def test_config_file_has_only_classes() -> None:
    """Ensure ``_config.py`` is class-only: no free functions and no bare public assignments."""
    path = _SRC / "_config.py"
    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    pytest.fail(f"Bare assignment {target.id} found in _config.py. Only classes allowed.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pytest.fail(f"Free function {node.name} found in _config.py. Only classes allowed.")


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
def test_no_internal_reexport_shims(file_path: Path) -> None:
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
def test_no_module_level_method_alias_shims() -> None:
    """Non-façade modules must not bind ``name = Class.attr`` or install ``globals()[name] = ...`` shims."""
    violations: list[str] = []
    for path in _get_src_files():
        stem = _module_stem(path)
        if stem in _FACADE_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute):
                    if isinstance(node.value.value, ast.Name):
                        violations.append(
                            f"{path.name}:{node.lineno}: {target.id} = {node.value.value.id}.{node.value.attr}"
                        )
            if isinstance(node, ast.For):
                segment = ast.get_source_segment(text, node) or ""
                if "globals()" in segment:
                    violations.append(f"{path.name}:{node.lineno}: globals() install loop")
    if violations:
        pytest.fail("Module-level method alias / globals shim violations:\n" + "\n".join(violations))


@pytest.mark.fast
def test_no_cross_module_private_imports() -> None:
    """Package modules must not import ``_``-prefixed symbols from sibling modules."""
    violations: list[str] = []
    for path in _get_src_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            is_internal = node.level > 0 or (
                node.module is not None and (node.module == "aetherdialect" or node.module.startswith("aetherdialect."))
            )
            if not is_internal:
                continue
            for alias in node.names:
                if alias.name.startswith("_") and not alias.name.startswith("__"):
                    mod_suffix = (node.module or "").split(".")[-1]
                    if (mod_suffix, alias.name) in _ALLOWED_PRIVATE_SYMBOL_IMPORTS:
                        continue
                    violations.append(f"{path.name}:{node.lineno} imports {alias.name} from {node.module}")
    if violations:
        pytest.fail("Cross-module private import(s):\n" + "\n".join(violations))


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
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.append(node.target.id)

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
        if sym in {name for _mod, name in _ALLOWED_PRIVATE_SYMBOL_IMPORTS}:
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
    sorted(p for d in (_SRC, _SCRIPTS) if d.is_dir() for p in d.rglob("*.py") if p.name != "test_static_core.py"),
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
    assert any(str(entry).startswith("hygiene:") for entry in markers)


@pytest.mark.fast
def test_corpus_absent_skip_reason_names_needs_corpus() -> None:
    text = (_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "needs_corpus: bundled sandbox data.zip absent" in text
    assert "sandbox data.zip not ready" not in text


@pytest.mark.fast
def test_module_kind_purity() -> None:
    """Each non-façade package module is exactly data-only, class-only, or ops-only."""
    data_violators: list[str] = []
    class_violators: list[str] = []
    ops_violators: list[str] = []
    unmapped: list[str] = []

    for path in _get_src_files():
        stem = _module_stem(path)
        if stem in _FACADE_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions, classes = _top_level_defs(tree)
        fn_names = [node.name for node in functions]
        class_names = [node.name for node in classes]

        if stem in _DATA_MODULES:
            if functions or classes:
                data_violators.append(
                    f"{path.name}: data module must have zero FunctionDef/ClassDef "
                    f"(functions={fn_names}, classes={class_names})"
                )
            continue
        if stem in _CLASS_MODULES:
            if functions:
                class_violators.append(
                    f"{path.name}: class module must have zero top-level FunctionDef (found {fn_names})"
                )
            continue
        if stem not in _EXPECTED_PACKAGE_MODULES:
            unmapped.append(path.name)
            continue
        if classes:
            ops_violators.append(f"{path.name}: ops module must have zero top-level ClassDef (found {class_names})")

    violations = (
        data_violators
        + class_violators
        + ops_violators
        + [f"{name}: not in locked module-kind map" for name in unmapped]
    )
    if violations:
        pytest.fail("Module-kind purity violations:\n" + "\n".join(violations))


@pytest.mark.fast
def test_no_new_package_modules() -> None:
    """Freeze the set of ``src/aetherdialect/*.py`` stems; new files fail this lock."""
    actual = frozenset(_module_stem(path) for path in _get_src_files())
    unexpected = sorted(actual - _EXPECTED_PACKAGE_MODULES)
    missing = sorted(_EXPECTED_PACKAGE_MODULES - actual)
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected package modules: {unexpected}")
    if missing:
        errors.append(f"missing package modules: {missing}")
    if errors:
        pytest.fail("\n".join(errors))


@pytest.mark.fast
def test_constants_define_no_functions() -> None:
    """``_constants.py`` must not define functions."""
    tree = ast.parse((_SRC / "_constants.py").read_text(encoding="utf-8"))
    violations = [
        f"line {node.lineno}: {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if violations:
        pytest.fail("FunctionDef in _constants.py:\n" + "\n".join(violations))


@pytest.mark.fast
def test_constants_define_no_classes_or_literal_types() -> None:
    """``_constants.py`` must not define classes or ``Name = Literal[...]`` aliases."""
    constants_path = _SRC / "_constants.py"
    tree = ast.parse(constants_path.read_text(encoding="utf-8"))
    _, classes = _top_level_defs(tree)
    violations: list[str] = [f"line {node.lineno}: class {node.name}" for node in classes]
    for node in tree.body:
        if _is_literal_type_alias_assignment(node):
            if isinstance(node, ast.Assign):
                names = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))
            else:
                assert isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                names = node.target.id
            violations.append(f"line {node.lineno}: Literal type alias {names}")

    for path in _get_src_files():
        if path.name == "_constants.py":
            continue
        other = ast.parse(path.read_text(encoding="utf-8"))
        for node in other.body:
            if _is_literal_type_alias_assignment(node):
                if isinstance(node, ast.Assign):
                    names = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))
                else:
                    assert isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    names = node.target.id
                violations.append(f"{path.name}:{node.lineno}: Literal type alias {names}")

    if violations:
        pytest.fail("ClassDef / Literal type alias violations:\n" + "\n".join(violations))


@pytest.mark.fast
def test_contracts_define_no_free_functions() -> None:
    """Contract modules are class-only."""
    violations: list[str] = []
    for stem in sorted(_CONTRACT_MODULES):
        path = _SRC / f"{stem}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions, _classes = _top_level_defs(tree)
        for node in functions:
            violations.append(f"{path.name}:{node.lineno}: {node.name}")
    if violations:
        pytest.fail("Free functions in contracts:\n" + "\n".join(violations))


@pytest.mark.fast
def test_config_define_no_free_functions() -> None:
    """``_config.py`` is class-only."""
    tree = ast.parse((_SRC / "_config.py").read_text(encoding="utf-8"))
    functions, _classes = _top_level_defs(tree)
    if functions:
        names = [f"line {node.lineno}: {node.name}" for node in functions]
        pytest.fail("Free functions in _config.py:\n" + "\n".join(names))


@pytest.mark.fast
def test_dialect_define_no_free_functions() -> None:
    """Dialect modules are class-only."""
    violations: list[str] = []
    for stem in sorted(_DIALECT_MODULES):
        path = _SRC / f"{stem}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions, _classes = _top_level_defs(tree)
        for node in functions:
            violations.append(f"{path.name}:{node.lineno}: {node.name}")
    if violations:
        pytest.fail("Free functions in dialect modules:\n" + "\n".join(violations))


@pytest.mark.fast
def test_closed_vocabularies_are_str_enums() -> None:
    """Closed typed vocabularies must be ``str, Enum`` types, not module-level ``Literal`` aliases."""
    violations: list[str] = []
    for path in _get_src_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not _is_literal_type_alias_assignment(node):
                continue
            if isinstance(node, ast.Assign):
                names = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))
            else:
                assert isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                names = node.target.id
            violations.append(f"{path.name}:{node.lineno}: {names} = Literal[...]")
    if violations:
        pytest.fail("Module-level Literal vocabulary aliases:\n" + "\n".join(violations))


@pytest.mark.fast
def test_identifier_naming_convention() -> None:
    """Classes are PascalCase, constants SCREAMING, functions snake_case (dunders exempt)."""
    violations: list[str] = []
    for path in _get_src_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if _DUNDER_NAME_RE.match(node.name):
                    continue
                if not _CLASS_NAME_RE.match(node.name):
                    violations.append(f"{path.name}:{node.lineno}: class {node.name} is not PascalCase")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _DUNDER_NAME_RE.match(node.name):
                    continue
                if not _FUNC_NAME_RE.match(node.name):
                    violations.append(f"{path.name}:{node.lineno}: function {node.name} is not snake_case")
        for node in tree.body:
            targets: list[ast.Name] = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append(target)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets.append(node.target)
            for target in targets:
                name = target.id
                if _DUNDER_NAME_RE.match(name) or name == "_":
                    continue
                if not _is_module_constant_name(name):
                    continue
                if not _CONST_NAME_RE.match(name):
                    violations.append(f"{path.name}:{node.lineno}: constant {name} is not SCREAMING_SNAKE")
    if violations:
        pytest.fail("Identifier naming violations:\n" + "\n".join(violations))
