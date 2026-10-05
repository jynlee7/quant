"""The trace matrix: spec, architecture and plan ids against the tests that name them.

Tests name ids with `@pytest.mark.spec(...)`. A step can't merge while an id it claims to cover
(tests/steps.toml) has no green test.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from quant_check.findings import Finding
from quant_check.repo import Repo, gate_env

DOCS = ("SPEC.md", "ARCHITECTURE.md", "PLAN.md")
STEPS_FILE = "tests/steps.toml"

# An id is the first cell of a table row in the approved docs: D1, A1, I1, T-NL1, G1, C1, K1,
# AD1, P1. Build steps (B0) and checkpoints (CP1) are not ids.
_ID_ROW = re.compile(r"^\|\s*(AD\d+|T-NL\d+|[DAIGCKP]\d+)\b")
_ID_SORT = re.compile(r"^([A-Z-]+?)(\d+)$")


def ids_in_docs(root: Path) -> frozenset[str] | None:
    """Every id defined in a table row of the docs, or None when the docs aren't there."""
    docs = [root / "docs" / name for name in DOCS]
    if not all(path.exists() for path in docs):
        return None
    found = set()
    for path in docs:
        for line in path.read_text().splitlines():
            if match := _ID_ROW.match(line):
                found.add(match.group(1))
    return frozenset(found)


def known_ids(repo: Repo) -> frozenset[str]:
    """Every id defined in a table row of docs/SPEC.md, ARCHITECTURE.md or PLAN.md."""
    return ids_in_docs(repo.root) or frozenset()


def step_claims(repo: Repo) -> dict[str, list[str]]:
    """Step -> the ids it claims to cover, from tests/steps.toml."""
    path = repo.root / STEPS_FILE
    if not path.exists():
        return {}
    steps = tomllib.loads(path.read_text()).get("steps", {})
    return {step: list(entry.get("covers", [])) for step, entry in steps.items()}


def _id_key(cid: str) -> tuple[str, int]:
    match = _ID_SORT.match(cid)
    return (match.group(1), int(match.group(2))) if match else (cid, 0)


def collect(repo: Repo) -> tuple[list[dict[str, Any]], str | None]:
    """Each collected test's ids and red mark, across every tier. Returns (tests, error)."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "trace.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tier",
                "all",
                "--quant-trace",
                str(output),
            ],
            cwd=repo.root,
            capture_output=True,
            text=True,
            check=False,
            env=gate_env(repo.root),
        )
        if not output.exists():
            return [], (proc.stdout + proc.stderr).strip()[-800:]
        tests: list[dict[str, Any]] = json.loads(output.read_text())
    return tests, None


def check_trace(repo: Repo) -> tuple[list[Finding], str]:
    """Findings for claimed ids with no green test, and the rendered matrix."""
    tests, error = collect(repo)
    if error is not None:
        return [Finding("trace", "-", f"couldn't collect the tests: {error}")], ""

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for test in tests:
        for cid in test["ids"]:
            by_id[cid].append(test)

    claims = step_claims(repo)
    known = known_ids(repo)
    findings = []
    steps = sorted(repo.done_steps() | {s for s in [repo.current_step()] if s}, key=_id_key)
    for step in steps:
        if step not in claims:
            findings.append(Finding("trace", step, f"{step} has no entry in {STEPS_FILE}"))
            continue
        for cid in claims[step]:
            if cid not in known:
                findings.append(
                    Finding("trace", step, f"{step} claims {cid}, which no doc defines")
                )
            elif not any(test["red"] is None for test in by_id.get(cid, [])):
                findings.append(
                    Finding("trace", step, f"{step} claims {cid}, but no green test names it")
                )

    claimed_by: dict[str, list[str]] = defaultdict(list)
    for step, ids in claims.items():
        for cid in ids:
            claimed_by[cid].append(step)
    lines = ["Trace matrix: id, tests naming it (red ones marked), steps claiming it"]
    for cid in sorted(set(by_id) | set(claimed_by), key=_id_key):
        named = by_id.get(cid, [])
        steps_note = (
            f"  claimed by {', '.join(sorted(claimed_by[cid], key=_id_key))}"
            if cid in claimed_by
            else ""
        )
        count = f"{len(named)} test" + ("" if len(named) == 1 else "s")
        lines.append(f"{cid:<7} {count:>9}{steps_note}")
        for test in sorted(named, key=lambda t: str(t["nodeid"])):
            red = f"  [red:{test['red']}]" if test["red"] else ""
            lines.append(f"          {test['nodeid']}{red}")
    return findings, "\n".join(lines)
