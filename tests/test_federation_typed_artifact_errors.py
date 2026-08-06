"""Federation artifact load failures must raise typed config errors, not silent fallbacks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aetherdialect._contracts_base import FederationConfigError, FederationMappings
from aetherdialect._federation import (
    compose_composite_graph,
    federation_artifact_paths,
    load_cached_federation_mapping_suggestions,
    load_federation_composite_graph,
    load_federation_plan_templates,
    mappings_replay_matches,
    parse_federation_manifest,
    persist_federation_tree,
)
from tests.test_federation_artifacts import _MANIFEST, _member_graphs


def _persisted_tree(tmp: str) -> tuple[object, FederationMappings, dict]:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.1")
    members = _member_graphs()
    composite = compose_composite_graph(members, manifest, mappings)
    persist_federation_tree(
        tmp,
        manifest=manifest,
        mappings=mappings,
        composite=composite,
        member_graphs=members,
    )
    return manifest, mappings, members


@pytest.mark.fast
def test_corrupt_artifact_manifest_raises_instead_of_replay_miss() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.1")
    members = _member_graphs()
    with tempfile.TemporaryDirectory() as tmp:
        _persisted_tree(tmp)
        manifest_path = Path(federation_artifact_paths(tmp)["artifact_manifest"])
        manifest_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(FederationConfigError, match="unreadable"):
            mappings_replay_matches(tmp, members, manifest, mappings)


@pytest.mark.fast
def test_non_object_plan_templates_file_raises_config_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _persisted_tree(tmp)
        templates_path = Path(federation_artifact_paths(tmp)["plan_templates"])
        templates_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        with pytest.raises(FederationConfigError, match="not a JSON object"):
            load_federation_plan_templates(tmp)


@pytest.mark.fast
def test_invalid_composite_schema_revision_raises_config_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _persisted_tree(tmp)
        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["schema_revision"] = "not-a-number"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with pytest.raises(FederationConfigError, match="schema revision"):
            load_federation_composite_graph(tmp)


@pytest.mark.fast
def test_corrupt_mapping_suggestions_cache_raises_config_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _persisted_tree(tmp)
        cache_path = Path(federation_artifact_paths(tmp)["mapping_suggestions_cache"])
        cache_path.write_text("{bad", encoding="utf-8")
        with pytest.raises(FederationConfigError, match="mapping suggestions cache"):
            load_cached_federation_mapping_suggestions(tmp, member_tuple_hash_value="any")
