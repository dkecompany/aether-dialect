"""Third-party import declarations and guarded optional imports."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "aetherdialect"

STDLIB_ROOT_NAMES = {
    "__future__",
    "abc",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "concurrent",
    "contextlib",
    "contextvars",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "functools",
    "hashlib",
    "importlib",
    "io",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "queue",
    "random",
    "re",
    "shutil",
    "string",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "tomllib",
    "traceback",
    "types",
    "typing",
    "unicodedata",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "glob",
    "gzip",
    "fcntl",
    "msvcrt",
    "ctypes",
    "unittest",
    "subprocess",
    "signal",
    "socket",
    "logging",
    "email",
    "html",
    "http",
    "inspect",
    "pickle",
    "platform",
    "errno",
    "mmap",
    "stat",
    "fnmatch",
    "codecs",
    "binascii",
    "calendar",
    "configparser",
    "gettext",
    "ipaddress",
    "keyword",
    "locale",
    "operator",
    "pprint",
    "tokenize",
    "xml",
    "sqlite3",
    "zipfile",
}

OPTIONAL_LAZY_FUNCTION_ROOTS = frozenset({"pyarrow", "google", "snowflake", "pyspark"})

KNOWN_OPTIONAL_TOP_LEVEL: dict[str, frozenset[str]] = {
    "_main_execution.py": frozenset({"pyspark"}),
    "_schema_overrides.py": frozenset({"pyspark"}),
}


def _requirement_package_name(requirement: str) -> str:
    token = requirement.split("[", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)", token)
    if match is None:
        return token.lower()
    return match.group(1).lower()


PACKAGE_IMPORT_ALIASES: dict[str, set[str]] = {
    "sqlalchemy": {"sqlalchemy"},
    "jsonschema": {"jsonschema"},
    "openai": {"openai"},
    "platformdirs": {"platformdirs"},
    "sqlglot": {"sqlglot"},
    "pandas": {"pandas"},
    "packaging": {"packaging"},
    "duckdb": {"duckdb"},
    "openpyxl": {"openpyxl"},
    "pyarrow": {"pyarrow"},
    "psycopg": {"psycopg"},
    "pglast": {"pglast"},
    "pymysql": {"pymysql"},
    "mysql-connector-python": {"mysql", "mysql.connector"},
    "pyodbc": {"pyodbc"},
    "snowflake-connector-python": {"snowflake"},
    "snowflake-sqlalchemy": {"snowflake"},
    "google-cloud-bigquery": {"google"},
    "databricks-sql-connector": {"databricks"},
    "redshift-connector": {"redshift_connector"},
    "sqlite": {"sqlite3"},
}


def _declared_third_party_roots() -> set[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    roots: set[str] = set()
    for requirement in data["project"]["dependencies"]:
        package = _requirement_package_name(requirement)
        roots.update(PACKAGE_IMPORT_ALIASES.get(package, {package.split(".")[0]}))
    for extra_requirements in data["project"].get("optional-dependencies", {}).values():
        for requirement in extra_requirements:
            package = _requirement_package_name(requirement)
            roots.update(PACKAGE_IMPORT_ALIASES.get(package, {package.split(".")[0]}))
    return roots


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def _imported_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {_module_root(alias.name) for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return set()
        if node.module is None:
            return set()
        root = _module_root(node.module)
        if root.startswith("_"):
            return set()
        return {root}
    return set()


def _contains(parent: ast.AST, child: ast.AST) -> bool:
    for node in ast.walk(parent):
        if node is child:
            return True
    return False


def _function_guarded(function: ast.FunctionDef | ast.AsyncFunctionDef, import_node: ast.AST) -> bool:
    import_line = getattr(import_node, "lineno", 0)
    for statement in function.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if statement.lineno >= import_line:
                continue
            if isinstance(call.func, ast.Name) and call.func.id == "require_driver":
                return True
            if isinstance(call.func, ast.Attribute) and call.func.attr == "require_driver":
                return True
        if isinstance(statement, ast.Try) and statement.lineno <= import_line:
            if not _contains(statement, import_node):
                continue
            for handler in statement.handlers:
                if handler.type is None:
                    return True
                for exc_type in ast.walk(handler.type):
                    if isinstance(exc_type, ast.Name) and exc_type.id == "ImportError":
                        return True
                    if isinstance(exc_type, ast.Attribute) and exc_type.attr == "ImportError":
                        return True
        if isinstance(statement, ast.If) and statement.lineno <= import_line and _contains(statement, import_node):
            test = statement.test
            if isinstance(test, ast.Call):
                func = test.func
                if isinstance(func, ast.Attribute) and func.attr == "find_spec":
                    return True
    return False


def _collect_violations(source_path: Path, declared_roots: set[str]) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    def _enclosing_function(child: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        current: ast.AST | None = child
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = parent_map.get(current)
        return None

    violations: list[str] = []
    for import_node in [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]:
        for root in _imported_roots(import_node):
            if root in STDLIB_ROOT_NAMES or root == "aetherdialect" or root in declared_roots:
                continue
            rel_path = source_path.relative_to(REPO_ROOT)
            enclosing = _enclosing_function(import_node)
            if enclosing is None and root in KNOWN_OPTIONAL_TOP_LEVEL.get(source_path.name, frozenset()):
                continue
            if enclosing is not None and root in OPTIONAL_LAZY_FUNCTION_ROOTS:
                continue
            if enclosing is not None and _function_guarded(enclosing, import_node):
                continue
            if enclosing is None:
                violations.append(f"{rel_path}: unguarded top-level import {root!r}")
            else:
                violations.append(f"{rel_path}: unguarded function import {root!r} in {enclosing.name}")
    return violations


@pytest.mark.fast
def test_every_third_party_import_is_declared_or_guarded() -> None:
    declared_roots = _declared_third_party_roots()
    violations: list[str] = []
    for source_path in sorted(SRC_ROOT.rglob("*.py")):
        if source_path.name == "_sandbox.py":
            continue
        violations.extend(_collect_violations(source_path, declared_roots))
    assert not violations, "Unguarded third-party imports:\n" + "\n".join(violations)
