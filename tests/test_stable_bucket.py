"""stable_bucket is independent of PYTHONHASHSEED."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_SUBPROCESS_SNIPPET = """
from aetherdialect._core_utils import stable_bucket
from aetherdialect._contracts_schema import ValueDomain
from aetherdialect._qsim import _sample_categorical

domain = ValueDomain(values=["alpha", "beta", "gamma"])
print(
    _sample_categorical(domain, 2),
    stable_bucket("intent-42", 11),
    stable_bucket(repr(("x", "y")), 5),
)
"""


@pytest.mark.fast
def test_same_inputs_across_hashseed() -> None:
    """Builtin hash() would vary; stable_bucket and qsim sampling stay stable."""
    outputs: list[str] = []
    for seed in ("0", "12345"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SNIPPET],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0]
    assert outputs[0] == outputs[1]
