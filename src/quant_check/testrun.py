"""The test tiers (P4): fast on every commit, full before each merge and nightly."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from quant_check.findings import Finding
from quant_check.repo import Repo, gate_env

FAST_TIER_BUDGET_S = 120.0  # PLAN P4: the fast tier stays under 2 minutes


def run_tests(repo: Repo, tier: str) -> tuple[list[Finding], str]:
    """Run one tier. Returns findings, and a one-line summary of the run."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--tier",
            tier,
            "-q",
            "-p",
            "no:cacheprovider",
            "--quant-report",
            str(report_path),
        ]
        started = time.perf_counter()
        proc = subprocess.run(
            command,
            cwd=repo.root,
            capture_output=True,
            text=True,
            check=False,
            env=gate_env(repo.root),
        )
        elapsed = time.perf_counter() - started
        report: dict[str, Any] | None = (
            json.loads(report_path.read_text()) if report_path.exists() else None
        )
    tail = (proc.stdout + proc.stderr).strip()[-800:]
    if report is None:
        return [
            Finding("tests", "-", f"pytest exited {proc.returncode} with no report: {tail}")
        ], ""

    findings = [Finding(p["rule"], p["nodeid"], p["message"]) for p in report["problems"]]
    counts: dict[str, int] = {}
    for result in report["results"]:
        outcome = result["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "xpass-strict":
            findings.append(
                Finding(
                    "P2",
                    result["nodeid"],
                    f"red test passes; remove its red:{result['red']} mark in the green commit",
                )
            )
        elif outcome in ("failed", "error"):
            findings.append(Finding("tests", result["nodeid"], result["message"]))
    if not findings and proc.returncode != 0:
        why = "no tests collected" if proc.returncode == 5 else f"pytest exited {proc.returncode}"
        findings.append(Finding("tests", "-", f"{why}: {tail}"))

    summary = ", ".join(f"{n} {outcome}" for outcome, n in sorted(counts.items()))
    over = tier == "fast" and elapsed > FAST_TIER_BUDGET_S
    budget = f", OVER the {FAST_TIER_BUDGET_S:.0f}s budget (P4)" if over else ""
    return findings, f"{summary or 'nothing ran'} in {elapsed:.0f}s{budget}"
