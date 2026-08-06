"""Shared helpers for join-injection tests."""

from __future__ import annotations


def catalog_edge_kinds_for_signatures(join_sigs_ordered: list[list[str]]) -> list[list[str]]:
    """Build per-carrier edge-kind lists assigning ``catalog_fk`` to every signature segment."""
    return [["catalog_fk"] * len(signature) for signature in join_sigs_ordered]
