"""I6: no secret in any file that is or could be committed, or anywhere in git history.

gitleaks scans a snapshot of the candidate files (tracked, plus untracked files git doesn't
ignore), so the real keys in the gitignored `.env` files are never read. A second pass scans
every commit, which catches a key that was committed and deleted again.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from quant_check.findings import Finding
from quant_check.repo import Repo, scrubbed_env

CONFIG = ".gitleaks.toml"
_LEAKS_FOUND = 3  # gitleaks' exit code when it finds leaks; anything else non-zero is an error


def _gitleaks(binary: str, args: list[str], repo: Repo, cwd: Path) -> list[dict[str, Any]] | str:
    """Run one gitleaks scan. Returns its findings, or an error message."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        command = [
            binary,
            *args,
            "--report-format",
            "json",
            "--report-path",
            str(report),
            "--exit-code",
            str(_LEAKS_FOUND),
            "--redact",
            "--no-banner",
            "--log-level",
            "error",
            "--ignore-gitleaks-allow",
        ]
        if (repo.root / CONFIG).exists():
            command += ["--config", str(repo.root / CONFIG)]
        proc = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False, env=scrubbed_env()
        )
        if proc.returncode not in (0, _LEAKS_FOUND) or not report.exists():
            return (proc.stderr or proc.stdout).strip()[-500:] or f"exit {proc.returncode}"
        leaks: list[dict[str, Any]] = json.loads(report.read_text() or "[]")
        return leaks


def _order(finding: Finding) -> tuple[str, int]:
    """Sort by file, then by line number as a number."""
    where, _, line = finding.location.rpartition(":")
    return (where, int(line)) if line.isdigit() else (finding.location, 0)


def scan_secrets(repo: Repo) -> list[Finding]:
    binary = shutil.which("gitleaks")
    if binary is None:
        return [Finding("I6", "-", "gitleaks is not installed: brew install gitleaks")]
    if not repo.is_git():
        return [
            Finding("I6", "-", "not a git work tree; the scan covers tracked files and history")
        ]

    findings = []
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp)
        for path in repo.candidate_files():
            dest = snapshot / repo.relative(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        tree = _gitleaks(binary, ["dir", "."], repo, cwd=snapshot)
        if isinstance(tree, str):
            return [Finding("I6", "-", f"gitleaks failed on the working tree: {tree}")]
        for leak in tree:
            where = Path(leak["File"])
            rel = (
                where.relative_to(snapshot).as_posix() if where.is_absolute() else where.as_posix()
            )
            findings.append(
                Finding("I6", f"{rel.removeprefix('./')}:{leak['StartLine']}", leak["RuleID"])
            )

    if repo.has_commits():
        history = _gitleaks(binary, ["git", str(repo.root)], repo, cwd=repo.root)
        if isinstance(history, str):
            return [*findings, Finding("I6", "-", f"gitleaks failed on the history: {history}")]
        for leak in history:
            location = f"history:{leak['Commit'][:10]}:{leak['File']}:{leak['StartLine']}"
            findings.append(Finding("I6", location, leak["RuleID"]))
    return sorted(findings, key=_order)
