"""The tree being checked: its files, git branch and finished steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
        raise NotImplementedError("red:B0")

    def is_git(self) -> bool:
        raise NotImplementedError("red:B0")

    def branch(self) -> str | None:
        """The current branch, or None when detached or not a git repo."""
        raise NotImplementedError("red:B0")

    def current_step(self) -> str | None:
        """The build step whose branch is checked out: `b03-features` gives "B3"."""
        raise NotImplementedError("red:B0")

    def done_steps(self) -> frozenset[str]:
        """Steps tagged `bNN-done`, as "BN"."""
        raise NotImplementedError("red:B0")

    def candidate_files(self) -> list[Path]:
        """Files that are or could be committed: tracked, plus untracked files git doesn't ignore."""
        raise NotImplementedError("red:B0")

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()
