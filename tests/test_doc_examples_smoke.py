"""Smoke-test runnable doc examples from SANDBOX.md (offline, no network)."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine
from aetherdialect._llm_provider import MockProvider
from aetherdialect._sandbox import Sandbox

pytest.importorskip("duckdb")

pytestmark = pytest.mark.needs_corpus


@pytest.fixture(autouse=True)
def _reset_mock() -> None:
    MockProvider.reset_mock_provider()
    yield
    MockProvider.reset_mock_provider()


def test_sandbox_manual_step_snippet() -> None:
    question = Sandbox._sandbox_questions()[0]
    namespace: dict[str, object] = {"AetherEngine": AetherEngine, "question": question}
    exec(
        compile(
            """
with Sandbox.create_offline_sandbox(AetherEngine) as sb:
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
    question = Sandbox._sandbox_questions()[0]
    namespace: dict[str, object] = {"AetherEngine": AetherEngine, "question": question}
    exec(
        compile(
            """
with Sandbox.create_offline_sandbox(AetherEngine) as sb:
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
