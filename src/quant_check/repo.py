"""The tree being checked: its files, git branch and finished steps."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_STEP_BRANCH = re.compile(r"^b(\d{2})-[a-z0-9][a-z0-9-]*$")
_DONE_TAG = re.compile(r"^b(\d{2})-done$")

# Variables git exports to hooks that pin commands to one repository. A gate run from inside a
# hook must not leak them into the git commands its tests run in other repositories.
GIT_LOCAL_ENV = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_INTERNAL_SUPER_PREFIX",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
)


def step_id(number: str) -> str:
    """'03' gives 'B3', the id the plan and the red marks use."""
    return f"B{int(number)}"


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """This process's environment without git's per-repository variables."""
    env = {k: v for k, v in os.environ.items() if k not in GIT_LOCAL_ENV}
    env.update(extra or {})
    return env


@dataclass(frozen=True)
class Repo:
    """A checkout of this project, or a planted copy of one."""

    root: Path

    @property
    def package_dir(self) -> Path:
        """`src/quant`, the code every structural rule applies to."""
        return self.root / "src" / "quant"

    def module_files(self) -> list[tuple[str, Path]]:
        """Every module under `src/quant` as (dotted name, path), sorted by name."""
        modules = []
        for path in self.package_dir.rglob("*.py"):
            parts = path.relative_to(self.root / "src").with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            modules.append((".".join(parts), path))
        return sorted(modules)

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            check=False,
            env=scrubbed_env(),
        )

    def is_git(self) -> bool:
        """True when the root is the top of a git work tree, not a folder inside another one."""
        proc = self.git("rev-parse", "--show-toplevel")
        return proc.returncode == 0 and Path(proc.stdout.strip()).resolve() == self.root.resolve()

    def has_commits(self) -> bool:
        return (
            self.is_git() and self.git("rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
        )

    def branch(self) -> str | None:
        """The current branch, or None when detached or not a git repo."""
        if not self.is_git():
            return None
        proc = self.git("symbolic-ref", "--quiet", "--short", "HEAD")
        name = proc.stdout.strip()
        return name if proc.returncode == 0 and name else None

    def current_step(self) -> str | None:
        """The build step whose branch is checked out: `b03-features` gives "B3"."""
        match = _STEP_BRANCH.match(self.branch() or "")
        return step_id(match.group(1)) if match else None

    def merging_steps(self) -> frozenset[str]:
        """Steps whose branch is being merged in (MERGE_HEAD), as the merge hook sees them."""
        if not self.is_git():
            return frozenset()
        head = self.git("rev-parse", "--verify", "--quiet", "MERGE_HEAD")
        if head.returncode != 0:
            return frozenset()
        refs = self.git("branch", "--format=%(refname:short)", "--points-at", head.stdout.strip())
        return frozenset(
            step_id(m.group(1)) for b in refs.stdout.split() if (m := _STEP_BRANCH.match(b))
        )

    def done_steps(self) -> frozenset[str]:
        """Steps tagged `bNN-done`, as "BN"."""
        if not self.is_git():
            return frozenset()
        tags = self.git("tag", "--list", "b*-done").stdout.split()
        return frozenset(step_id(m.group(1)) for t in tags if (m := _DONE_TAG.match(t)))

    def candidate_files(self) -> list[Path]:
        """Files that are or could be committed: tracked, plus untracked files git doesn't ignore."""
        proc = self.git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
        paths = {self.root / rel for rel in proc.stdout.split("\0") if rel}
        return sorted(p for p in paths if p.is_file() and not p.is_symlink())

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()


def gate_env(root: Path) -> dict[str, str]:
    """The environment for test and tool subprocesses: the checked tree's src/ first on the
    path, and the determinism settings of ARCHITECTURE §14 (fixed hash seed, one BLAS thread)."""
    return scrubbed_env(
        {
            "PYTHONPATH": str(root / "src"),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NO_COLOR": "1",
        }
    )
