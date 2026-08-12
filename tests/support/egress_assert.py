"""Hermetic egress leak detection for observable pipeline outputs."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import Any

from aetherdialect._contracts_base import KnowledgeScope
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._schema_profile import out_of_scope_description_tokens


def forbidden_identifiers_for_scope(
    full_graph: SchemaGraph,
    caller_scope: KnowledgeScope,
    *,
    extra_forbidden: Iterable[str] = (),
) -> frozenset[str]:
    """Return table/column/source tokens that must not appear in caller- visible egress."""
    tokens = set(out_of_scope_description_tokens(full_graph, caller_scope))
    for raw in extra_forbidden:
        token = str(raw).strip()
        if token:
            tokens.add(token)
    return frozenset(tokens)


def _forbidden_pattern(forbidden: frozenset[str]) -> re.Pattern[str] | None:
    parts = [rf"\b{re.escape(token)}\b" for token in sorted(forbidden, key=len, reverse=True) if token]
    if not parts:
        return None
    return re.compile("|".join(parts), flags=re.IGNORECASE)


def _check_string(value: str, forbidden: frozenset[str], path: str) -> None:
    pattern = _forbidden_pattern(forbidden)
    if pattern is None:
        return
    match = pattern.search(value)
    if match is not None:
        raise AssertionError(f"forbidden identifier {match.group(0)!r} leaked at {path}")


def _is_dataframeish(obj: Any) -> bool:
    columns = getattr(obj, "columns", None)
    return columns is not None and hasattr(columns, "__iter__") and not isinstance(obj, (str, bytes, bytearray))


def _is_binaryish(obj: Any) -> bool:
    return isinstance(obj, (bytes, bytearray, memoryview))


def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_asdict")


def assert_no_forbidden_identifiers(
    obj: Any,
    forbidden: frozenset[str],
    *,
    path: str = "root",
) -> None:
    """Recursively scan *obj* for forbidden identifier substrings."""
    if not forbidden:
        return
    if obj is None:
        return

    if isinstance(obj, str):
        _check_string(obj, forbidden, path)
        return

    if _is_binaryish(obj):
        return

    if _is_dataframeish(obj):
        for col in obj.columns:
            assert_no_forbidden_identifiers(str(col), forbidden, path=f"{path}.columns[{col!r}]")
        return

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            assert_no_forbidden_identifiers(
                getattr(obj, field.name),
                forbidden,
                path=f"{path}.{field.name}",
            )
        return

    if _is_namedtuple(obj):
        for name in obj._fields:
            assert_no_forbidden_identifiers(getattr(obj, name), forbidden, path=f"{path}.{name}")
        return

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            assert_no_forbidden_identifiers(key, forbidden, path=f"{path}[key={key!r}]")
            assert_no_forbidden_identifiers(value, forbidden, path=f"{path}[{key!r}]")
        return

    if isinstance(obj, (list, tuple, set, frozenset)):
        for index, item in enumerate(obj):
            assert_no_forbidden_identifiers(item, forbidden, path=f"{path}[{index}]")
        return

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict) and obj_dict:
        for key, value in obj_dict.items():
            if str(key).startswith("_"):
                continue
            assert_no_forbidden_identifiers(value, forbidden, path=f"{path}.{key}")
        return
