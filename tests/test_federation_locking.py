"""Federation composition must hold the artifact lock for the full init sequence."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import FederationContext
from aetherdialect._federation import (
    load_federation_declaration_from_path,
    load_federation_member_graphs,
    persist_federation_tree,
)
from aetherdialect._main_execution import MainExecutionOps
from tests.federation_helpers import write_federation_declaration_file
from tests.test_migration_cache_drift import _MANIFEST, _member_graph, _mock_member


@pytest.mark.fast
def test_composition_holds_the_lock_throughout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    member_dict = {sid: _mock_member(graph, sid, tmp_path) for sid, graph in members.items()}
    declaration_path = write_federation_declaration_file(tmp_path, _MANIFEST)
    authored_manifest, fed_mappings = load_federation_declaration_from_path(str(declaration_path))

    fed_lock_depth = 0
    load_during_fed_lock = False
    persist_during_fed_lock = False
    member_refresh_during_lock = False

    @contextmanager
    def _combined_lock_cm(directory: str, timeout: object | None = None):
        nonlocal fed_lock_depth, member_refresh_during_lock
        _ = timeout
        path = str(directory)
        if "fed_fed_gate" in path:
            fed_lock_depth += 1
        if "fedsrc_" in path and fed_lock_depth > 0:
            member_refresh_during_lock = True
        try:
            yield
        finally:
            if "fed_fed_gate" in path:
                fed_lock_depth -= 1

    def _tracked_load(*args, **kwargs):
        nonlocal load_during_fed_lock
        load_during_fed_lock = fed_lock_depth > 0
        return load_federation_member_graphs(*args, **kwargs)

    def _tracked_persist(*args, **kwargs):
        nonlocal persist_during_fed_lock
        persist_during_fed_lock = fed_lock_depth > 0
        return persist_federation_tree(*args, **kwargs)

    with (
        patch("aetherdialect._main_execution.artifact_lock", side_effect=_combined_lock_cm),
        patch("aetherdialect._federation.artifact_lock", side_effect=_combined_lock_cm),
        patch("aetherdialect._main_execution.load_federation_member_graphs", side_effect=_tracked_load),
        patch("aetherdialect._main_execution.persist_federation_tree", side_effect=_tracked_persist),
    ):
        MainExecutionOps.initialize_aether_federation(
            "fed_gate",
            members=member_dict,
            declaration_file=str(declaration_path),
            declaration=(authored_manifest, fed_mappings),
            artifacts_dir=str(tmp_path),
            master_context=FederationContext(),
            log_sink=lambda _msg: None,
        )

    assert load_during_fed_lock
    assert persist_during_fed_lock
    assert member_refresh_during_lock
