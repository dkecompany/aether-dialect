"""Master-context gates for federation aetherspace export and listing."""

from __future__ import annotations

import pytest

from tests.test_aether_federation_public_surface import _fed


@pytest.mark.fast
def test_federation_export_structure_callable_outside_master_label() -> None:
    fed = _fed()
    fed._context_name = "team_scope"
    doc = fed.export_structure()
    assert isinstance(doc, dict)
    assert "tables" in doc


@pytest.mark.fast
def test_federation_list_aetherspaces_callable_outside_master_label() -> None:
    fed = _fed()
    fed._context_name = "team_scope"
    spaces = fed.list_aetherspaces()
    assert isinstance(spaces, (list, tuple))
