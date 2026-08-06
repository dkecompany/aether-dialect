"""Permission denied pattern hygiene."""

from __future__ import annotations

import pytest

from aetherdialect._constants import EXPLAIN_PERMISSION_DENIED_PATTERNS
from aetherdialect._dialect import Dialect


@pytest.mark.fast
def test_undefined_table_not_permission_denied() -> None:
    assert "undefinedtable" not in EXPLAIN_PERMISSION_DENIED_PATTERNS
    assert "42p01" not in EXPLAIN_PERMISSION_DENIED_PATTERNS
    assert Dialect.is_permission_denied_error("relation does not exist / undefinedtable") is False


@pytest.mark.fast
def test_insufficient_privilege_still_matches() -> None:
    assert "insufficient privilege" in EXPLAIN_PERMISSION_DENIED_PATTERNS
    assert Dialect.is_permission_denied_error("insufficient privilege for relation") is True
