"""Runtime access denial and federation execution-scope composition."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import AccessError, FederationContext, SchemaAccessError
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_runtime_access_error_is_schema_access_error() -> None:
    """Integrators catching SchemaAccessError also catch runtime permission denial."""
    err = AccessError("execute", "permission denied for relation secret")
    assert isinstance(err, SchemaAccessError)


@pytest.mark.fast
def test_federation_execution_allow_objects_intersects_master_subset() -> None:
    master = FederationContext(allow_objects=frozenset({"customers"}))
    composite_tables = frozenset({"customers", "orders"})
    assert MainExecutionOps._federation_execution_allow_objects(master, composite_tables) == frozenset({"customers"})


@pytest.mark.fast
def test_federation_execution_allow_objects_uses_composite_when_master_unrestricted() -> None:
    master = FederationContext()
    composite_tables = frozenset({"customers", "orders"})
    assert MainExecutionOps._federation_execution_allow_objects(master, composite_tables) == composite_tables
