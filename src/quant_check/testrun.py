"""The test tiers (P4): fast on every commit, full before each merge and nightly."""

from __future__ import annotations

from quant_check.findings import Finding
from quant_check.repo import Repo


def run_tests(repo: Repo, tier: str) -> list[Finding]:
    raise NotImplementedError("red:B0")
