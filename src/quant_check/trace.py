"""The trace matrix: spec, architecture and plan ids against the tests that name them.

A step can't merge while an id it claims to cover (tests/steps.toml) has no green test.
"""

from __future__ import annotations

from quant_check.findings import Finding
from quant_check.repo import Repo


def known_ids(repo: Repo) -> frozenset[str]:
    """Every id defined in a table row of docs/SPEC.md, ARCHITECTURE.md or PLAN.md."""
    raise NotImplementedError("red:B0")


def step_claims(repo: Repo) -> dict[str, list[str]]:
    """Step -> the ids it claims to cover, from tests/steps.toml."""
    raise NotImplementedError("red:B0")


def check_trace(repo: Repo) -> tuple[list[Finding], str]:
    """Findings for claimed ids with no green test, and the rendered matrix."""
    raise NotImplementedError("red:B0")
