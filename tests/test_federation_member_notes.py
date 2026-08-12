"""Member-authored notes must not name federation source identifiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_base import ConfigError
from aetherdialect._federation_manifest import raise_if_member_notes_name_federation_sources


@pytest.mark.fast
def test_member_notes_reject_source_id_tokens(tmp_path: Path) -> None:
    notes = tmp_path / "member_notes.txt"
    notes.write_text("Orders from storefront include tax.", encoding="utf-8")
    with pytest.raises(ConfigError, match="storefront"):
        raise_if_member_notes_name_federation_sources(str(notes), ["storefront", "catalog"])


@pytest.mark.fast
def test_member_notes_allow_unrelated_prose(tmp_path: Path) -> None:
    notes = tmp_path / "member_notes.txt"
    notes.write_text("Customer lifetime value uses rolling 90-day revenue.", encoding="utf-8")
    raise_if_member_notes_name_federation_sources(str(notes), ["storefront", "catalog"])
