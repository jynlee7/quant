"""I6: no secret in any file that is or could be committed, or anywhere in git history."""

from __future__ import annotations

from quant_check.findings import Finding
from quant_check.repo import Repo


def scan_secrets(repo: Repo) -> list[Finding]:
    raise NotImplementedError("red:B0")
