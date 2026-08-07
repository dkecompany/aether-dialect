"""Registry token rendering refuses missing definitions."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr, RegistryRenderError
from aetherdialect._sql_gen import render_expr_sql


def test_missing_window_or_case_registry_refuses() -> None:
    """Bare registry refs without an active render scope raise instead of emitting ``0``."""
    with pytest.raises(RegistryRenderError, match="w01"):
        render_expr_sql(NormalizedExpr(column_ref="w01"))

    with pytest.raises(RegistryRenderError, match="c01"):
        render_expr_sql(NormalizedExpr(column_ref="c01"))
