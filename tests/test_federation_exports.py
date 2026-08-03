"""Federation exception types must be exported from the public API."""

from __future__ import annotations

import pytest

import aetherdialect
from aetherdialect import (
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMemberUnprofilableError,
)


@pytest.mark.fast
def test_federation_exception_types_are_public_exports() -> None:
    for name in (
        "FederationMalformedMemberAnswerError",
        "FederationJoinFanOutError",
        "FederationMemberUnprofilableError",
    ):
        assert name in aetherdialect.__all__
        assert hasattr(aetherdialect, name)


@pytest.mark.fast
def test_exported_federation_exceptions_are_subclasses_of_config_error() -> None:
    from aetherdialect import ConfigError, FederationRuntimeError

    assert issubclass(FederationMalformedMemberAnswerError, FederationRuntimeError)
    assert issubclass(FederationJoinFanOutError, FederationRuntimeError)
    assert issubclass(FederationMemberUnprofilableError, ConfigError)
