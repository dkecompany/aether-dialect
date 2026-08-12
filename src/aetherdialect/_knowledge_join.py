"""Join fan-out metadata and negative FK override materialization from structural knowledge."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._contracts_base import StructuralKnowledgeFact, StructuralKnowledgeKind
from ._contracts_schema import ColumnMetadata, SchemaGraph
from ._utils import phys_table_key


def _parse_qualified_join_endpoint(ref: str) -> tuple[str, list[str]] | None:
    text = str(ref or "").strip()
    if not text:
        return None
    if "|" in text:
        table_name, cols_text = text.split("|", 1)
        cols = [c.strip() for c in cols_text.split(",") if c.strip()]
        return table_name.strip(), cols
    if "." in text:
        table_name, column_name = text.split(".", 1)
        if table_name.strip() and column_name.strip():
            return table_name.strip(), [column_name.strip()]
    return None


def _parent_child_from_qualified(
    schema: SchemaGraph,
    qualified: Sequence[str],
) -> tuple[tuple[str, list[str]], tuple[str, list[str]]] | None:
    """Resolve parent/child endpoint tuples from qualified column references."""
    parsed: list[tuple[str, list[str], ColumnMetadata]] = []
    for ref in qualified:
        ep = _parse_qualified_join_endpoint(ref)
        if ep is None:
            continue
        tbl, cols = ep
        table_meta = schema.tables.get(tbl)
        if table_meta is None or not cols or cols[0] not in table_meta.columns:
            continue
        parsed.append((tbl, cols, table_meta.columns[cols[0]]))
    if len(parsed) != 2:
        return None
    parent: tuple[str, list[str]] | None = None
    child: tuple[str, list[str]] | None = None
    for tbl, cols, col in parsed:
        if col.is_primary_key:
            parent = (tbl, cols)
        else:
            child = (tbl, cols)
    if parent is None or child is None:
        return None
    return parent, child


def _notes_one_to_many_edge_keys(
    facts: Sequence[StructuralKnowledgeFact], schema: SchemaGraph
) -> frozenset[tuple[str, ...]]:
    """Edges notes declare as one-to-many (child many side); ratchet- only fan-out input."""
    keys: set[tuple[str, ...]] = set()
    for fact in facts:
        kind = str(fact.kind or "").strip().lower()
        if kind == StructuralKnowledgeKind.CARDINALITY.value:
            payload = fact.payload or {}
            direction = str(payload.get("direction") or "").strip().lower()
            qualified = sorted(ref for ref in fact.referenced_entities if "." in ref)
            if len(qualified) != 2:
                continue
            resolved = _parent_child_from_qualified(schema, qualified)
            if resolved is None:
                continue
            parent, child = resolved
            if direction == "one_to_many":
                keys.add((parent[0], ",".join(parent[1]), child[0], ",".join(child[1])))
            elif direction == "many_to_one":
                keys.add((child[0], ",".join(child[1]), parent[0], ",".join(parent[1])))
            else:
                keys.add((parent[0], ",".join(parent[1]), child[0], ",".join(child[1])))
        elif kind == StructuralKnowledgeKind.JOIN.value:
            payload = fact.payload or {}
            if str(payload.get("cardinality") or "").strip().lower() != "one_to_many":
                continue
            parsed = _parent_child_from_qualified(schema, sorted(fact.referenced_entities))
            if parsed is None:
                continue
            parent, child = parsed
            keys.add((parent[0], ",".join(parent[1]), child[0], ",".join(child[1])))
    return frozenset(keys)


def attach_structural_fanout_metadata(schema: SchemaGraph, facts: Sequence[StructuralKnowledgeFact]) -> None:
    """Stamp notes-sourced one-to-many edges and preserve-table defaults on *schema*."""
    many_edges = _notes_one_to_many_edge_keys(facts, schema)
    if many_edges:
        object.__setattr__(schema, "_notes_one_to_many_edges", many_edges)
    preserve: set[str] = set()
    for fact in facts:
        if fact.kind != StructuralKnowledgeKind.CARDINALITY.value:
            continue
        qualified = sorted(ref for ref in fact.referenced_entities if "." in ref)
        if len(qualified) != 2:
            continue
        resolved = _parent_child_from_qualified(schema, qualified)
        if resolved is None:
            continue
        parent, _child = resolved
        preserve.add(parent[0])
    if preserve:
        object.__setattr__(schema, "_notes_preserve_tables", frozenset(preserve))


def notes_preserve_table_defaults(schema: SchemaGraph | None) -> frozenset[str]:
    """Return tables notes declare as preserve defaults via cardinality facts."""
    if schema is None:
        return frozenset()
    return frozenset(getattr(schema, "_notes_preserve_tables", frozenset()))


def merge_preserve_tables_with_notes_defaults(
    preserve_tables: Sequence[str] | None,
    schema: SchemaGraph | None,
    *,
    query_tables: Sequence[str] | None = None,
) -> list[str]:
    """Union intent/list preserve_tables with notes defaults, scoped to *query_tables* when given."""
    merged: list[str] = []
    seen: set[str] = set()
    for table in preserve_tables or []:
        name = str(table).strip()
        if not name:
            continue
        key = phys_table_key(name)
        if key in seen:
            continue
        seen.add(key)
        merged.append(name)
    notes_defaults = notes_preserve_table_defaults(schema)
    if query_tables is not None:
        query_keys = {phys_table_key(str(table).strip()) for table in query_tables if str(table).strip()}
        notes_defaults = frozenset(table for table in notes_defaults if phys_table_key(table) in query_keys)
    extras = sorted(table for table in notes_defaults if phys_table_key(table) not in seen)
    merged.extend(extras)
    return merged


def notes_declares_one_to_many_edge(
    schema: SchemaGraph | None,
    join_table: str,
    join_cols: list[str],
    paired_table: str,
    paired_cols: list[str],
) -> bool:
    """True when notes ratchet declares this edge one-to-many (more conservative fan-out only)."""
    if schema is None:
        return False
    keys: frozenset[tuple[str, str, str, str]] = getattr(schema, "_notes_one_to_many_edges", frozenset())
    if not keys:
        return False
    fwd = (join_table, ",".join(join_cols), paired_table, ",".join(paired_cols))
    rev = (paired_table, ",".join(paired_cols), join_table, ",".join(join_cols))
    return fwd in keys or rev in keys


def notes_ratchet_multiplies_table(
    schema: SchemaGraph | None,
    table: str,
    join_table: str,
    join_cols: list[str],
    paired_table: str,
    paired_cols: list[str],
) -> bool:
    """True when notes declare one-to-many and *table* is the parent (one) side of that edge."""
    if schema is None:
        return False
    keys: frozenset[tuple[str, str, str, str]] = getattr(schema, "_notes_one_to_many_edges", frozenset())
    if not keys:
        return False
    fwd = (join_table, ",".join(join_cols), paired_table, ",".join(paired_cols))
    rev = (paired_table, ",".join(paired_cols), join_table, ",".join(join_cols))
    for key in keys:
        edge_fwd = (key[0], key[1], key[2], key[3])
        edge_rev = (key[2], key[3], key[0], key[1])
        if fwd == edge_fwd or fwd == edge_rev or rev == edge_fwd or rev == edge_rev:
            return phys_table_key(table) == phys_table_key(key[0])
    return False


def materialize_fk_remove_to_overrides(
    overrides_path: str | Any,
    removals: Sequence[Mapping[str, str]],
) -> bool:
    """Merge negative join proposals into overrides ``foreign_keys_remove``."""
    path = Path(overrides_path)
    if not removals:
        return False
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            doc = {}
    else:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    existing = list(doc.get("foreign_keys_remove") or [])
    seen = {
        (str(item.get("from") or "").strip(), str(item.get("to") or "").strip())
        for item in existing
        if isinstance(item, dict)
    }
    changed = False
    for removal in removals:
        from_ref = str(removal.get("from") or "").strip()
        to_ref = str(removal.get("to") or "").strip()
        if not from_ref or not to_ref:
            continue
        key = (from_ref, to_ref)
        if key in seen:
            continue
        seen.add(key)
        existing.append({"from": from_ref, "to": to_ref})
        changed = True
    if not changed:
        return False
    doc["foreign_keys_remove"] = existing
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return True
