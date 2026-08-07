"""BusinessKnowledgeKind validation on normalize."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import BusinessKnowledgeEntry, BusinessKnowledgeKind, ConfigError


@pytest.mark.fast
def test_unknown_kind_raises() -> None:
    with pytest.raises(ConfigError, match="kind"):
        BusinessKnowledgeEntry.normalize(BusinessKnowledgeEntry(key="k", text="t", kind="unknown_kind"))


@pytest.mark.fast
def test_default_glossary() -> None:
    entry = BusinessKnowledgeEntry.normalize(
        BusinessKnowledgeEntry(key="arr", text="annual recurring revenue", kind="")
    )
    assert entry.kind == BusinessKnowledgeKind.GLOSSARY.value
    assert entry.kind == "glossary"
    for kind in BusinessKnowledgeKind:
        ok = BusinessKnowledgeEntry.normalize(BusinessKnowledgeEntry(key=f"k_{kind.value}", text="ok", kind=kind.value))
        assert ok.kind == kind.value
