"""P2's audit: red marks may exist only on their own step's branch, and never after it is done.

A red mark is `pytest.mark.xfail(strict=True, reason="red:BNN", raises=...)`. The commit that
adds a step's tests carries them, and the green commit removes them.
"""

from __future__ import annotations

from quant_check.findings import Finding
from quant_check.repo import Repo


def audit_red_marks(repo: Repo, *, merging: bool) -> list[Finding]:
    """With `merging`, any red mark at all is a finding: a step can't merge while red."""
    raise NotImplementedError("red:B0")
