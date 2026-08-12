"""Exception hierarchy, builtin shadowing, and raise/export reconciliation."""

from __future__ import annotations

import ast
import builtins
import inspect
from pathlib import Path

import pytest

import aetherdialect
from aetherdialect import _contracts_base, _contracts_core

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"
_CONTRACTS_PATH = _SRC_ROOT / "_contracts_base.py"
_CONTRACTS_MODULES = (_contracts_base, _contracts_core)

_ROOT_EXPORT_MARKERS = frozenset({"AetherError", "RetryableError"})
_INDIRECT_RAISE_FACTORIES = {
    "wrap_database_execution_error": frozenset({"DatabaseExecutionError", "RetryableDatabaseExecutionError"}),
}
_FACTORY_INSTANTIATED = frozenset({"RetryableFederationPartialFailureError"})


def _library_exception_classes() -> list[type[BaseException]]:
    classes: list[type[BaseException]] = []
    for module in _CONTRACTS_MODULES:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if not issubclass(obj, BaseException):
                continue
            if name.startswith("_"):
                continue
            classes.append(obj)
    return classes


def _library_exception_by_name() -> dict[str, type[BaseException]]:
    return {cls.__name__: cls for cls in _library_exception_classes()}


def _collect_raised_exception_names() -> set[str]:
    names: set[str] = set()
    for path in sorted(_SRC_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            if isinstance(exc, ast.Call):
                if isinstance(exc.func, ast.Name) and exc.func.id in _INDIRECT_RAISE_FACTORIES:
                    names |= _INDIRECT_RAISE_FACTORIES[exc.func.id]
                    continue
                exc = exc.func
            if isinstance(exc, ast.Name):
                names.add(exc.id)
            elif isinstance(exc, ast.Attribute):
                names.add(exc.attr)
    return names


def _expand_with_library_bases(raised: set[str], library_by_name: dict[str, type[BaseException]]) -> set[str]:
    expanded = set(raised)
    for name in list(raised):
        cls = library_by_name.get(name)
        if cls is None:
            continue
        for base in cls.__mro__:
            bname = base.__name__
            if bname in library_by_name:
                expanded.add(bname)
    if "FederationPartialFailureError" in expanded:
        expanded |= _FACTORY_INSTANTIATED
    return expanded | _ROOT_EXPORT_MARKERS


def _exported_exception_names() -> set[str]:
    exported: set[str] = set()
    for name in aetherdialect.__all__:
        obj = getattr(aetherdialect, name, None)
        if inspect.isclass(obj) and issubclass(obj, BaseException):
            exported.add(name)
    return exported


def test_every_exception_inherits_the_root() -> None:
    root = _contracts_base.AetherError
    offenders = [cls.__name__ for cls in _library_exception_classes() if not issubclass(cls, root)]
    assert offenders == [], f"exceptions not inheriting AetherError: {offenders}"


def test_existing_handlers_still_catch() -> None:
    assert issubclass(_contracts_base.ConfigError, ValueError)
    assert issubclass(_contracts_base.MigrationPendingError, ValueError)
    assert issubclass(_contracts_base.SchemaAccessError, ValueError)
    assert issubclass(_contracts_base.DatabaseConnectionError, OSError)
    assert issubclass(_contracts_base.MockFixtureMissingError, RuntimeError)
    assert issubclass(_contracts_base.SessionActiveError, RuntimeError)
    assert issubclass(_contracts_base.StatementTimeoutError, RuntimeError)
    assert issubclass(_contracts_base.ResultCapExceededError, RuntimeError)
    assert issubclass(_contracts_base.SchemaInvariantError, RuntimeError)
    assert issubclass(_contracts_core.AccessError, _contracts_base.SchemaAccessError)
    assert issubclass(_contracts_core.AccessError, RuntimeError)
    assert issubclass(_contracts_base.DatabasePingFailed, _contracts_base.RetryableError)
    assert issubclass(_contracts_base.LlmTransientFailure, _contracts_base.RetryableError)
    assert issubclass(_contracts_base.StatementTimeoutError, _contracts_base.RetryableError)
    assert issubclass(_contracts_base.ArtifactLockTimeoutError, _contracts_base.RetryableError)
    assert issubclass(_contracts_base.ConfigError, _contracts_base.AetherError)

    with pytest.raises(ValueError):
        raise _contracts_base.ConfigError("cfg")
    with pytest.raises(OSError):
        raise _contracts_base.DatabaseConnectionError("conn")
    with pytest.raises(RuntimeError):
        raise _contracts_base.SessionActiveError("busy")
    with pytest.raises(_contracts_base.AetherError):
        raise _contracts_core.NoJoinPathError("scope", ["a", "b"])


def test_no_exception_shadows_a_builtin() -> None:
    builtin_names = set(dir(builtins))
    shadowed = sorted(name for name in _exported_exception_names() if name in builtin_names)
    assert shadowed == [], f"exported exceptions shadow builtins: {shadowed}"
    assert "ConnectionError" not in _exported_exception_names()


def test_raised_and_exported_sets_match() -> None:
    library_by_name = _library_exception_by_name()
    raised = _expand_with_library_bases(_collect_raised_exception_names(), library_by_name)
    exported = _exported_exception_names()
    library_names = set(library_by_name)
    raised_library = raised & library_names
    missing_exports = sorted(raised_library - exported)
    unraised_exports = sorted(exported - raised_library)
    assert missing_exports == [], f"raised but not exported: {missing_exports}"
    assert unraised_exports == [], f"exported but never raised: {unraised_exports}"
