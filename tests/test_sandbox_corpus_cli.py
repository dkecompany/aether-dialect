"""CLI validation for scripts/sandbox_corpus.py flag combinations."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "sandbox_corpus.py"


def _load_sandbox_corpus_module():
    spec = importlib.util.spec_from_file_location("sandbox_corpus_cli_test", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_corpus_cli_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def sandbox_corpus():
    return _load_sandbox_corpus_module()


@pytest.mark.fast
@pytest.mark.parametrize(
    "extra_args,expected_fragment",
    [
        pytest.param(["--force"], "--force", id="force_without_repair"),
        pytest.param(["--repair", "--smoke"], "--smoke", id="repair_with_smoke"),
        pytest.param(
            ["--repair", "--record-reuse-pairs"],
            "--record-reuse-pairs",
            id="repair_with_record_reuse_pairs",
        ),
    ],
)
def test_invalid_flag_combos_exit_with_argparse_error(
    extra_args: list[str],
    expected_fragment: str,
) -> None:
    proc = _run_cli(*extra_args)
    assert proc.returncode == 2
    assert "error:" in proc.stderr.lower()
    assert expected_fragment in proc.stderr


@pytest.mark.fast
def test_help_exits_zero() -> None:
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "--repair" in proc.stdout
    assert "--force" in proc.stdout


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["sandbox_corpus.py"], id="bare_run"),
        pytest.param(["sandbox_corpus.py", "--repair"], id="repair"),
        pytest.param(["sandbox_corpus.py", "--repair", "--force"], id="repair_force"),
        pytest.param(["sandbox_corpus.py", "--smoke"], id="smoke"),
        pytest.param(["sandbox_corpus.py", "--record-reuse-pairs"], id="record_reuse_pairs"),
    ],
)
def test_valid_flag_combos_pass_argparse(sandbox_corpus, argv: list[str]) -> None:
    with (
        patch.object(sandbox_corpus, "_run_full_build") as build_mock,
        patch.object(sandbox_corpus, "finalize_validate") as pack_mock,
        patch.object(sandbox_corpus, "corpus_message"),
        patch("sys.argv", argv),
    ):
        sandbox_corpus.main()
    if "--repair" in argv:
        pack_mock.assert_called_once()
        assert pack_mock.call_args.kwargs.get("force") is ("--force" in argv)
        build_mock.assert_not_called()
    else:
        build_mock.assert_called_once()
        pack_mock.assert_not_called()
