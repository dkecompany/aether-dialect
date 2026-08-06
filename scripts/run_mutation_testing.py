"""Mutation-testing harness for high-risk validator and identity modules.

Targets ``_validation_execute.py``, ``_validation_schema.py``,
``_validation_semantic.py``, ``_intent_resolve.py`` (identity helpers),
``_intent_repair.py`` (elimination guards), and ``_core_utils.py`` (hashing).

Operator workflow (not run in CI or pre-commit):

1. ``pip install -e '.[dev]'``
2. ``python scripts/run_mutation_testing.py --init``  # seed baseline structure
3. ``python scripts/run_mutation_testing.py --update``  # run mutmut; may take hours
4. ``python scripts/run_mutation_testing.py --check``  # fail if survivors increased

The PR gate only ships the script, baseline schema, and a fast harness test.
Full mutmut green is an operator responsibility after ``--update``. Mutmut is
installed with the ``dev`` extra but is never invoked by CI or pre-commit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "dev_workspace" / "mutation_baseline.json"
PACKAGE_SRC = REPO_ROOT / "src" / "aetherdialect"

TARGET_MODULES: tuple[str, ...] = (
    "_validation_execute.py",
    "_validation_schema.py",
    "_validation_semantic.py",
    "_intent_resolve.py",
    "_intent_repair.py",
    "_core_utils.py",
)

BASELINE_VERSION = 1


@dataclass(frozen=True)
class ModuleTarget:
    """One mutation target module on disk."""

    module: str
    path: Path

    @property
    def key(self) -> str:
        return self.module


def discover_targets() -> list[ModuleTarget]:
    """Resolve configured target modules to existing source paths."""
    missing: list[str] = []
    found: list[ModuleTarget] = []
    for module in TARGET_MODULES:
        path = PACKAGE_SRC / module
        if path.is_file():
            found.append(ModuleTarget(module=module, path=path))
        else:
            missing.append(module)
    if missing:
        raise FileNotFoundError(f"mutation targets missing on disk: {', '.join(missing)}")
    return found


def baseline_template(*, status: str = "operator_run_required", decision: str | None = None) -> dict[str, Any]:
    """Return a fresh baseline document with per-module survivor slots."""
    targets = discover_targets()
    doc: dict[str, Any] = {
        "version": BASELINE_VERSION,
        "tool": "mutmut",
        "status": status,
        "targets": {t.key: None for t in targets},
    }
    if decision:
        doc["decision"] = decision
    return doc


def load_baseline(path: Path) -> dict[str, Any]:
    """Load baseline JSON from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(path: Path, doc: dict[str, Any]) -> None:
    """Persist baseline JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_baseline_schema(doc: dict[str, Any]) -> None:
    """Raise ``ValueError`` when *doc* does not match the expected baseline shape."""
    if doc.get("version") != BASELINE_VERSION:
        raise ValueError(f"unexpected baseline version: {doc.get('version')!r}")
    if doc.get("tool") != "mutmut":
        raise ValueError(f"unexpected tool: {doc.get('tool')!r}")
    if "status" not in doc:
        raise ValueError("baseline missing status")
    targets = doc.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("baseline targets must be an object")
    expected = {t.key for t in discover_targets()}
    if set(targets) != expected:
        raise ValueError(f"baseline targets keys mismatch: expected {sorted(expected)} got {sorted(targets)}")


def _mutmut_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mutmut", "--version"],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _parse_mutmut_results(stdout: str) -> dict[str, int | None]:
    """Best-effort parse of ``mutmut results`` output into per-file survivor counts."""
    counts: dict[str, int] = {}
    pattern = re.compile(r"^\s*(?P<survivors>\d+)\s+(?P<path>.+)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        survivors = int(match.group("survivors"))
        raw_path = match.group("path").replace("\\", "/")
        name = Path(raw_path).name
        if name in TARGET_MODULES:
            counts[name] = survivors
    return {module: counts.get(module) for module in TARGET_MODULES}


def run_mutmut_update() -> dict[str, int | None]:
    """Run mutmut for configured targets and return surviving-mutant counts per module."""
    if not _mutmut_available():
        raise RuntimeError("mutmut is not installed; pip install -e '.[dev]' first")
    paths = [str(PACKAGE_SRC / module) for module in TARGET_MODULES]
    run_cmd = [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        "--paths-to-mutate",
        ",".join(paths),
        "--no-progress",
        "--CI",
    ]
    proc = subprocess.run(run_cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if proc.returncode not in (0, 1, 2):
        raise RuntimeError(
            f"mutmut run failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    results = subprocess.run(
        [sys.executable, "-m", "mutmut", "results"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if results.returncode != 0:
        raise RuntimeError(f"mutmut results failed:\n{results.stdout}\n{results.stderr}")
    return _parse_mutmut_results(results.stdout)


def cmd_init(args: argparse.Namespace) -> int:
    """Write baseline structure without running a full mutation campaign."""
    decision = (
        "Baseline seeded by --init; operator must run --update after installing mutmut "
        "to record surviving-mutant counts."
    )
    write_baseline(args.baseline, baseline_template(status="operator_run_required", decision=decision))
    print(f"wrote baseline template to {args.baseline}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Run mutmut and refresh survivor counts in the baseline file."""
    counts = run_mutmut_update()
    doc = baseline_template(status="recorded")
    doc["targets"] = counts
    doc["decision"] = "Counts recorded by --update."
    write_baseline(args.baseline, doc)
    print(json.dumps(doc, indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Fail when any module survivor count exceeds the baseline."""
    if not args.baseline.is_file():
        raise SystemExit(f"baseline missing: {args.baseline}")
    doc = load_baseline(args.baseline)
    validate_baseline_schema(doc)
    if doc.get("status") == "operator_run_required":
        print("baseline status is operator_run_required; skipping survivor comparison")
        return 0
    targets = doc["targets"]
    unresolved = [k for k, v in targets.items() if v is None]
    if unresolved:
        raise SystemExit(f"baseline has unset survivor counts: {', '.join(unresolved)}")
    if args.compare:
        fresh = run_mutmut_update()
        regressions: list[str] = []
        for module, baseline_count in targets.items():
            current = fresh.get(module)
            if current is None:
                regressions.append(f"{module}: no mutmut result")
            elif int(current) > int(baseline_count):
                regressions.append(f"{module}: {current} survivors > baseline {baseline_count}")
        if regressions:
            raise SystemExit("mutation survivor regression:\n" + "\n".join(regressions))
        print("mutation survivor check passed")
        return 0
    print("baseline schema valid; use --check --compare to compare live mutmut counts")
    return 0


def cmd_dry_run(_args: argparse.Namespace) -> int:
    """List discovered modules without mutating or writing files."""
    for target in discover_targets():
        print(target.path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to mutation_baseline.json (default: dev_workspace/mutation_baseline.json)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init", action="store_true", help="Seed baseline structure with placeholder counts")
    group.add_argument("--update", action="store_true", help="Run mutmut and write survivor counts")
    group.add_argument("--check", action="store_true", help="Validate baseline schema and survivor bounds")
    group.add_argument("--dry-run", action="store_true", help="List target module paths only")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="With --check, run mutmut and fail when survivors increase vs baseline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.init:
        return cmd_init(args)
    if args.update and not args.check:
        return cmd_update(args)
    if args.check:
        return cmd_check(args)
    if args.dry_run:
        return cmd_dry_run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
