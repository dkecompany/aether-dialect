"""DomainKnowledgeKind validation on normalize."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError, DomainKnowledgeEntry, DomainKnowledgeKind


@pytest.mark.fast
def test_unknown_kind_raises() -> None:
    with pytest.raises(ConfigError, match="kind"):
        DomainKnowledgeEntry.normalize(DomainKnowledgeEntry(key="k", text="t", kind="unknown_kind"))


@pytest.mark.fast
def test_default_glossary() -> None:
    entry = DomainKnowledgeEntry.normalize(DomainKnowledgeEntry(key="arr", text="annual recurring revenue", kind=""))
    assert entry.kind == DomainKnowledgeKind.GLOSSARY.value
    assert entry.kind == "glossary"
    for kind in DomainKnowledgeKind:
        ok = DomainKnowledgeEntry.normalize(DomainKnowledgeEntry(key=f"k_{kind.value}", text="ok", kind=kind.value))
        assert ok.kind == kind.value
