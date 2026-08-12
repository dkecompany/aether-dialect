"""Cross-source join kinds are a closed set validated at manifest parse."""

from __future__ import annotations

import pytest

from aetherdialect._constants import FEDERATION_CROSS_SOURCE_JOIN_KINDS
from aetherdialect._contracts_base import FederationDeclarationError
from aetherdialect._federation_manifest import (
    parse_federation_manifest,
    validate_federation_cross_source_join_kind,
)


@pytest.mark.fast
@pytest.mark.parametrize("kind", sorted(FEDERATION_CROSS_SOURCE_JOIN_KINDS))
def test_accepted_cross_source_join_kinds_parse(kind: str) -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_kind",
            "cross_source_joins": [
                {
                    "left": "entity_a.email",
                    "right": "entity_b.email",
                    "kind": kind,
                    "logical_key": "email",
                }
            ],
        }
    )
    assert manifest.cross_source_joins[0].kind == kind


@pytest.mark.fast
@pytest.mark.parametrize("kind", ["right", "full", "full_outer", "cross", "lefft"])
def test_unrecognised_cross_source_join_kind_raises_with_accepted_set(kind: str) -> None:
    with pytest.raises(FederationDeclarationError, match="accepted values: inner, left") as exc:
        parse_federation_manifest(
            {
                "federation_id": "fed_kind",
                "cross_source_joins": [
                    {
                        "left": "entity_a.email",
                        "right": "entity_b.email",
                        "kind": kind,
                        "logical_key": "email",
                    }
                ],
            }
        )
    assert kind in str(exc.value)


@pytest.mark.fast
def test_validate_federation_cross_source_join_kind_normalizes_whitespace() -> None:
    assert validate_federation_cross_source_join_kind(" LEFT ") == "left"
