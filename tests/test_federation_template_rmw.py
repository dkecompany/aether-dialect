"""Federation template save must read-merge-write entirely under artifact_lock."""

from __future__ import annotations

import contextlib
import tempfile
import threading
from unittest.mock import patch

import pytest

import aetherdialect._federation
from aetherdialect._contracts_base import FederationPlanTemplate
from aetherdialect._federation import load_federation_plan_templates, save_federation_plan_template

_ORIGINAL_ARTIFACT_LOCK = aetherdialect._federation.artifact_lock


def _stored_template(template_id: str) -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=template_id,
        composite_schema_graph_id="cg_full",
        intent_key=f"ik_{template_id}",
        step_fingerprints=(("alpha", "fp_a"), ("beta", "fp_b")),
        combine_hash="combine_hash",
        question="show joined entities",
        accepted_questions=("accepted q",),
        format_version="0.2.1",
        member_template_ids=(("alpha", "T0001"), ("beta", "T0002")),
        residual_hash="residual_hash",
        join_feedback=("join hint",),
        manifest_hash="manifest_hash",
        member_tuple_hash="member_tuple_hash",
    )


@pytest.mark.fast
def test_two_writers_both_plans_survive() -> None:
    """Concurrent saves must not clobber each other when reads race outside the lock."""
    lock_state = threading.local()
    pre_lock_read_barrier = threading.Barrier(2, timeout=5)
    real_open = open

    def coordinating_open(file: str, mode: str = "r", *args: object, **kwargs: object):
        handle = real_open(file, mode, *args, **kwargs)
        if "plan_templates" in str(file) and "r" in mode and not getattr(lock_state, "holding", False):
            pre_lock_read_barrier.wait(timeout=5)
        return handle

    @contextlib.contextmanager
    def tracking_artifact_lock(federation_dir: str, *args: object, **kwargs: object):
        with _ORIGINAL_ARTIFACT_LOCK(federation_dir, *args, **kwargs) as lock:
            lock_state.holding = True
            try:
                yield lock
            finally:
                lock_state.holding = False

    errors: list[BaseException] = []
    with tempfile.TemporaryDirectory() as fed_dir:
        template_a = _stored_template("plan_alpha")
        template_b = _stored_template("plan_beta")

        def _save(template: FederationPlanTemplate) -> None:
            try:
                save_federation_plan_template(fed_dir, template)
            except BaseException as exc:
                errors.append(exc)

        with (
            patch("builtins.open", side_effect=coordinating_open),
            patch.object(aetherdialect._federation, "artifact_lock", tracking_artifact_lock),
        ):
            threads = [
                threading.Thread(target=_save, args=(template_a,)),
                threading.Thread(target=_save, args=(template_b,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()

        assert not errors, f"save_federation_plan_template raised: {errors!r}"
        loaded = load_federation_plan_templates(fed_dir)
        assert "plan_alpha" in loaded, f"plan_alpha lost to concurrent write: {sorted(loaded)}"
        assert "plan_beta" in loaded, f"plan_beta lost to concurrent write: {sorted(loaded)}"
