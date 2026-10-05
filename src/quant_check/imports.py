"""Import rules K1–K4 and P11, checked with import-linter on the tree being checked."""

from __future__ import annotations

from quant_check.findings import Finding
from quant_check.repo import Repo


def check_imports(repo: Repo) -> list[Finding]:
    """Broken contracts, contracts that miss a subpackage, and alpaca-py's trading client
    imported anywhere but the adapter (K4)."""
    raise NotImplementedError("red:B0")
