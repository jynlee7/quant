"""Planted-violation copies of the repo (P3).

A guard is proven by copying the repo, planting one violation and running the guard through
`quant-check`, which must fail and name the rule.
"""

from __future__ import annotations

import os
import random
import shutil
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUANT_CHECK = Path(sys.executable).with_name("quant-check")

# What a copy needs for every guard to run. tests/ is replaced by one sanity test, so a gate
# run inside a copy never recurses into these tests.
COPIED = (
    "src",
    "docs",
    ".githooks",
    "pyproject.toml",
    ".importlinter",
    ".gitleaks.toml",
    ".python-version",
    ".gitignore",
    "tests/__init__.py",
    "tests/steps.toml",
)

SANITY_TEST = """\
import pytest


@pytest.mark.spec("P3")
def test_the_copy_imports() -> None:
    import quant

    assert quant.__doc__
"""


@dataclass(frozen=True)
class GateResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def findings(self) -> list[tuple[str, str, str]]:
        """(rule, location, message) for every `FAIL` line."""
        found = []
        for line in self.stdout.splitlines():
            if line.startswith("FAIL "):
                _, rule, location, *message = line.split(" ", 3)
                found.append((rule, location, message[0] if message else ""))
        return found

    def caught(self, rule: str, location: str) -> bool:
        """The gate failed, and one finding names `rule` at a location starting with `location`."""
        return self.returncode != 0 and any(
            r == rule and loc.startswith(location) for r, loc, _ in self.findings
        )

    def explain(self) -> str:
        return f"exit {self.returncode}\n--- stdout\n{self.stdout}\n--- stderr\n{self.stderr}"


def _raise_if_unimplemented(*outputs: str) -> None:
    """In a red phase the gate is a stub; surface its NotImplementedError to the red mark."""
    for output in outputs:
        lines = output.strip().splitlines()
        if any(line.startswith("NotImplementedError") for line in lines[-3:]):
            raise NotImplementedError(lines[-1])


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST", "QUANT_"))}
    env.pop("PYTHONPATH", None)
    env.update(extra or {})
    return env


class PlantedRepo:
    """A throwaway copy of the repo to plant violations in."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def copy_of_repo(cls, dest: Path) -> PlantedRepo:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".mypy_cache", ".ruff_cache")
        for rel in COPIED:
            src = REPO / rel
            if src.is_dir():
                shutil.copytree(src, dest / rel, ignore=ignore)
            elif src.exists():
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
        planted = cls(dest)
        planted.write("tests/test_sanity.py", SANITY_TEST)
        return planted

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def git(self, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            env=_clean_env(env),
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
        return proc.stdout + proc.stderr

    def git_init(self, branch: str = "main") -> None:
        self.git("init", "-q", "-b", branch)
        self.git("config", "user.name", "planted")
        self.git("config", "user.email", "planted@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", ".githooks")

    def commit_all(self, message: str = "planted") -> None:
        """Commit everything, skipping the hook."""
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "-m", message)

    def commit_through_hook(
        self, message: str = "planted", *, stage_all: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Commit with the hook, which runs this venv's quant-check. Stages everything first
        unless `stage_all` is False."""
        if stage_all:
            self.git("add", "-A")
        proc = subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            env=_clean_env({"QUANT_CHECK": str(QUANT_CHECK)}),
            timeout=600,
        )
        _raise_if_unimplemented(proc.stdout, proc.stderr)
        return proc

    def run_gate(self, *args: str, timeout: float = 600) -> GateResult:
        proc = subprocess.run(
            [str(QUANT_CHECK), "--root", str(self.root), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            env=_clean_env(),
            timeout=timeout,
        )
        _raise_if_unimplemented(proc.stderr)
        return GateResult(proc.returncode, proc.stdout, proc.stderr)


def fake_secret_line(kind: str, seed: int = 20160104) -> str:
    """A line holding a fake key of the given kind, built at runtime so no key sits in a file."""
    rng = random.Random(seed)
    upper = string.ascii_uppercase + string.digits
    alnum = string.ascii_letters + string.digits

    def pick(alphabet: str, n: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(n))

    if kind == "alpaca":
        key_id = "P" + "K" + pick(upper, 18)
        return f'ALPACA_API_KEY = "{key_id}"\nALPACA_SECRET_KEY = "{pick(alnum, 40)}"\n'
    if kind == "anthropic":
        return f'api_key = "{"sk-" + "ant-" + "api03-" + pick(alnum, 93) + "AA"}"\n'
    if kind == "telegram":
        return f'TELEGRAM_BOT_TOKEN = "{pick(string.digits, 10)}:AA{pick(alnum, 33)}"\n'
    raise ValueError(kind)
