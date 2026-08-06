"""Hygiene package must carry both ``hygiene`` and ``fast`` markers."""

from __future__ import annotations

import pytest


@pytest.mark.hygiene
@pytest.mark.fast
def test_hygiene_item_carries_fast(request: pytest.FixtureRequest) -> None:
    """Every hygiene test (including this one) must also be selectable via ``-m fast``."""
    assert "hygiene" in request.node.keywords
    assert "fast" in request.node.keywords
