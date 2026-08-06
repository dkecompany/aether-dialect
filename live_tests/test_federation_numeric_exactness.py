"""Live federation numeric exactness checks (operator sequence)."""

from __future__ import annotations

import pytest


@pytest.mark.needs_corpus
def test_federated_average_matches_single_engine() -> None:
    pytest.skip("requires operator corpus sequence")
