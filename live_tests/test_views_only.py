"""Live checks that schema scope metadata matches the configured reflection mode."""

from __future__ import annotations


def test_schema_object_kind_is_literal(schema) -> None:
    """Schema ``include`` mode matches reflected relations; each relation ``kind`` is table or view."""
    assert schema.include in ("tables", "views", "both")
    for rel in schema.tables.values():
        assert rel.kind in ("table", "view")
