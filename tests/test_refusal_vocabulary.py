"""User-facing refusal text must not expose internal vocabulary."""

from __future__ import annotations

import pytest

from aetherdialect._constants import REPHRASE_HINT_REFUSAL_CODES
from aetherdialect._constants_runtime import REFUSAL_CATALOGUE, REPHRASE_HINT_MESSAGES

_FORBIDDEN_TERMS: tuple[str, ...] = (
    "foreign_keys_add",
    "semantic neighbour override",
    "semi-join",
    "semi_join",
    "common table expression",
    "intermediate representation",
    "deny lists",
    "deny_columns",
    "allow_columns",
    "deny columns",
    "allow columns",
)


def _collect_refusal_user_strings() -> list[str]:
    strings: list[str] = []
    for entry in REFUSAL_CATALOGUE.values():
        strings.append(entry["user_text"])
        strings.append(entry["reformulation_hint"])
    for key in REPHRASE_HINT_REFUSAL_CODES:
        strings.append(REPHRASE_HINT_MESSAGES[key])
    return strings


@pytest.mark.fast
def test_no_internal_terms_in_refusals() -> None:
    for text in _collect_refusal_user_strings():
        lowered = text.lower()
        for term in _FORBIDDEN_TERMS:
            assert term not in lowered, f"forbidden term {term!r} in refusal text: {text[:120]!r}"
        assert " space " not in lowered and not lowered.startswith("space ")
