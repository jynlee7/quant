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
    findings += budget_findings(report["results"])
    if not findings and proc.returncode != 0:
        why = "no tests collected" if proc.returncode == 5 else f"pytest exited {proc.returncode}"
        findings.append(Finding("tests", "-", f"{why}: {tail}"))

    summary = ", ".join(f"{n} {outcome}" for outcome, n in sorted(counts.items()))
    return findings, f"{summary or 'nothing ran'} in {elapsed:.0f}s"


def budget_findings(results: list[dict[str, Any]]) -> list[Finding]:
    """P4: the fast tier's tests must take under two minutes in total. In the full tier the
    fast tests' share is measured too, so the budget is checked before every merge."""
    fast = sum(float(r.get("duration", 0.0)) for r in results if r.get("tier") == "fast")
    if fast <= FAST_TIER_BUDGET_S:
        return []
    return [
        Finding(
            "P4",
            "-",
            f"the fast tier's tests took {fast:.0f}s against a {FAST_TIER_BUDGET_S:.0f}s budget; "
            "mark slow tests full, or make them faster",
        )
    ]
