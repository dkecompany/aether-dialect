"""Master-context gates for federation aetherspace export and listing."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError
from tests.test_aether_federation_public_surface import _fed


@pytest.mark.fast
def test_federation_export_aetherspace_requires_master_context() -> None:
    fed = _fed()
    fed._context_name = "team_scope"
    with pytest.raises(ConfigError, match="requires the master engine context"):
        fed.export_aetherspace("master")


@pytest.mark.fast
def test_federation_list_aetherspaces_requires_master_context() -> None:
    fed = _fed()
    fed._context_name = "team_scope"
    with pytest.raises(ConfigError, match="requires the master engine context"):
        fed.list_aetherspaces()
