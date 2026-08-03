"""Smoke-test runnable doc examples from SANDBOX.md (offline, no network)."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine
from aetherdialect._llm_provider import reset_mock_provider

pytest.importorskip("duckdb")

pytestmark = pytest.mark.needs_corpus


@pytest.fixture(autouse=True)
def _reset_mock() -> None:
    reset_mock_provider()
    yield
    reset_mock_provider()


def test_sandbox_manual_step_snippet() -> None:
    question = AetherEngine.sandbox_questions()[0]
    namespace: dict[str, object] = {"AetherEngine": AetherEngine, "question": question}
    exec(
        compile(
            """
with AetherEngine.offline_sandbox() as sb:
    with sb.engine.session() as session:
        step = session.ask(question)
        while not step.done:
            if step.reply_shape == "yes_no":
                step = session.step("y")
            elif step.reply_shape == "free_text":
                step = session.step("looks good")
        assert step.done
        assert not step.error
""",
            "<SANDBOX.md manual step>",
            "exec",
        ),
        namespace,
    )


def test_sandbox_accept_until_done_snippet() -> None:
    question = AetherEngine.sandbox_questions()[0]
    namespace: dict[str, object] = {"AetherEngine": AetherEngine, "question": question}
    exec(
        compile(
            """
with AetherEngine.offline_sandbox() as sb:
    with sb.engine.session() as session:
        step = session.accept_until_done(question)
        assert step.done
        assert not step.error
""",
            "<SANDBOX.md accept_until_done>",
            "exec",
        ),
        namespace,
    )
