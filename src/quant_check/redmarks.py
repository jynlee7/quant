"""P2's audit: red marks may exist only on their own step's branch, and never after it is done.

A red mark is `pytest.mark.xfail(strict=True, reason="red:BNN", raises=...)`. The commit that
adds a step's tests carries them, and the green commit removes them. The pytest plugin checks
each mark's shape; this audit checks where marks may be at all. It reads the AST, so a red mark
quoted inside a string (as the planted-violation tests do) isn't mistaken for a real one.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from quant_check.findings import Finding
from quant_check.repo import Repo

# Steps are written without zero padding (B3, not B03), as the plan numbers them.
RED_REASON = re.compile(r"^red:(B(?:0|[1-9][0-9]*))$")


def red_marks(path: Path) -> Iterator[tuple[int, str]]:
    """(line, step) for every `xfail(..., reason="red:BNN")` call in a test file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "xfail":
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg == "reason" and isinstance(value, ast.Constant):
                match = RED_REASON.match(str(value.value))
                if match:
                    yield node.lineno, match.group(1)


def _tag(step: str) -> str:
    return f"b{int(step[1:]):02d}-done"


def audit_red_marks(repo: Repo, *, merging: bool) -> list[Finding]:
    """With `merging`, any red mark at all is a finding: a step can't merge while red."""
    step, done, branch = repo.current_step(), repo.done_steps(), repo.branch()
    findings = []
    for path in sorted((repo.root / "tests").rglob("*.py")):
        try:
            marks = list(red_marks(path))
        except SyntaxError as exc:
            findings.append(Finding("P2", repo.relative(path), f"can't parse: {exc.msg}"))
            continue
        for line, mark in marks:
            where = f"{repo.relative(path)}:{line}"
            if merging:
                message = f"red:{mark} is still marked; a step merges only when it is green"
            elif mark in done:
                message = f"red:{mark} outlives its step: {_tag(mark)} is tagged"
            elif mark != step:
                on = f"branch {branch}" if branch else "no step branch"
                message = f"red:{mark} belongs to step {mark}, but {on} is checked out"
            else:
                continue
            findings.append(Finding("P2", where, message))
    return findings
