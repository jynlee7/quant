"""`quant-check`: the build gate that runs before every commit (ARCHITECTURE §13, PLAN §2).

    quant-check            fast tier: lint, types, structural guards, secrets, red marks, fast tests
    quant-check --full     before a merge: full-tier tests, no red marks left, trace claims covered
    quant-check --trace    the matrix of spec ids against tests, and the claim check
    quant-check --only K5,K6   run named stages (as the planted-violation tests do)

Every finding prints as `FAIL <rule> <location> <message>`, and the exit code is 1 if any did.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from quant_check import imports, redmarks, scans, secrets, testrun, trace
from quant_check.findings import Finding
from quant_check.repo import Repo, gate_env, scrubbed_env

StageResult = tuple[list[Finding], str]
# pre-commit runs the fast gate; pre-merge-commit runs the full gate on `git merge --no-ff`.
HOOKS = ("pre-commit", "pre-merge-commit")


@dataclass(frozen=True)
class Context:
    repo: Repo
    tier: str
    merging: bool
    reports: list[str] = field(default_factory=list)  # printed after the stage table


def _tool(name: str) -> str:
    """A console script installed next to this Python, so the gate never depends on PATH."""
    return str(Path(sys.executable).with_name(name))


def _run(ctx: Context, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ctx.repo.root,
        capture_output=True,
        text=True,
        check=False,
        env=gate_env(ctx.repo.root),
    )


def stage_python(ctx: Context) -> StageResult:
    """ARCHITECTURE §14: the Python patch version is pinned."""
    pin_file = ctx.repo.root / ".python-version"
    running = platform.python_version()
    if not pin_file.exists():
        return [Finding("python", ".python-version", "the Python pin is missing")], running
    pinned = pin_file.read_text().strip()
    if running != pinned:
        return [
            Finding("python", ".python-version", f"running {running}, pinned {pinned}")
        ], running
    return [], running


def stage_ruff(ctx: Context) -> StageResult:
    proc = _run(ctx, [_tool("ruff"), "check", "--output-format", "concise", "--quiet", "."])
    findings = []
    for line in proc.stdout.splitlines():
        path, sep, rest = line.partition(": ")
        if sep and ":" in path:
            file, row = path.split(":")[:2]
            findings.append(Finding("ruff", f"{file}:{row}", rest.strip()))
    if proc.returncode != 0 and not findings:
        findings.append(Finding("ruff", "-", (proc.stdout + proc.stderr).strip()[-500:]))
    return findings, ""


def stage_format(ctx: Context) -> StageResult:
    proc = _run(ctx, [_tool("ruff"), "format", "--check", "."])
    findings = [
        Finding("format", line.removeprefix("Would reformat: ").strip(), "run ruff format")
        for line in proc.stdout.splitlines()
        if line.startswith("Would reformat: ")
    ]
    if proc.returncode != 0 and not findings:
        findings.append(Finding("format", "-", (proc.stdout + proc.stderr).strip()[-500:]))
    return findings, ""


def stage_mypy(ctx: Context) -> StageResult:
    proc = _run(
        ctx,
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            "pyproject.toml",
            "--no-error-summary",
            "--no-color-output",
            "--hide-error-context",
        ],
    )
    findings = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 3)
        if len(parts) == 4 and parts[2].strip() == "error":
            findings.append(Finding("mypy", f"{parts[0]}:{parts[1]}", parts[3].strip()))
    if proc.returncode != 0 and not findings:
        findings.append(Finding("mypy", "-", (proc.stdout + proc.stderr).strip()[-500:]))
    return findings, ""


def stage_hook(ctx: Context) -> StageResult:
    """The commit hook must be installed, or the gate silently stops gating commits."""
    repo = ctx.repo
    if not repo.is_git():
        return [], "not a git work tree"
    hooks_path = repo.git("config", "--get", "core.hooksPath").stdout.strip()
    findings = []
    for name in HOOKS:
        hook = repo.root / ".githooks" / name
        if hooks_path != ".githooks" or not hook.exists() or not os.access(hook, os.X_OK):
            findings.append(
                Finding(
                    "hook",
                    f".githooks/{name}",
                    "not installed: git config core.hooksPath .githooks",
                )
            )
    return findings, ""


def _findings_only(check: Callable[[Repo], list[Finding]]) -> Callable[[Context], StageResult]:
    def stage(ctx: Context) -> StageResult:
        return check(ctx.repo), ""

    return stage


def stage_red_marks(ctx: Context) -> StageResult:
    return redmarks.audit_red_marks(ctx.repo, merging=ctx.merging), ""


def stage_tests(ctx: Context) -> StageResult:
    return testrun.run_tests(ctx.repo, ctx.tier)


def stage_trace(ctx: Context) -> StageResult:
    findings, matrix = trace.check_trace(ctx.repo)
    ctx.reports.append(matrix)
    return findings, ""


STAGES: dict[str, Callable[[Context], StageResult]] = {
    "python": stage_python,
    "ruff": stage_ruff,
    "format": stage_format,
    "mypy": stage_mypy,
    "imports": _findings_only(imports.check_imports),
    "K5": _findings_only(scans.scan_k5),
    "K6": _findings_only(scans.scan_k6),
    "I2": _findings_only(scans.scan_i2),
    "I6": _findings_only(secrets.scan_secrets),
    "P2": stage_red_marks,
    "hook": stage_hook,
    "tests": stage_tests,
    "trace": stage_trace,
}
FAST = [name for name in STAGES if name != "trace"]
FULL = list(STAGES)


def _default_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        env=scrubbed_env(),
    )
    return Path(proc.stdout.strip()) if proc.returncode == 0 else Path.cwd()


def _stage_list(value: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage {', '.join(unknown)}; known: {', '.join(STAGES)}"
        )
    return names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-check", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, help="tree to check (default: this git work tree)")
    parser.add_argument("--full", action="store_true", help="full tier, as before a merge")
    parser.add_argument(
        "--trace", action="store_true", help="print the trace matrix and check claims"
    )
    parser.add_argument("--only", type=_stage_list, help="comma-separated stages to run")
    parser.add_argument(
        "--skip", type=_stage_list, default=[], help="comma-separated stages to skip"
    )
    args = parser.parse_args(argv)

    repo = Repo((args.root or _default_root()).resolve())
    ctx = Context(repo=repo, tier="full" if args.full else "fast", merging=args.full)
    if args.only:
        names = args.only
    elif args.trace and not args.full:
        names = ["trace"]
    else:
        names = FULL if args.full else FAST
    names = [n for n in names if n not in args.skip]

    print(f"quant-check · {repo.root} · {ctx.tier} tier")
    findings: list[Finding] = []
    started = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        try:
            found, note = STAGES[name](ctx)
        except Exception as exc:  # a crashed stage is a failed stage, and the rest still run
            found, note = [Finding(name, "-", f"stage crashed: {type(exc).__name__}: {exc}")], ""
        status = "ok  " if not found else "FAIL"
        detail = note
        if found:
            count = f"{len(found)} finding{'s' if len(found) != 1 else ''}"
            detail = f"{count}; {note}" if note else count
        print(f"  {status}  {name:<8} {detail:<48} {time.perf_counter() - t0:6.1f}s")
        findings.extend(found)

    for report in ctx.reports:
        print(report)
    for finding in findings:
        print(finding.render())
    elapsed = time.perf_counter() - started
    if findings:
        print(f"quant-check: FAILED, {len(findings)} finding(s) in {elapsed:.1f}s")
        return 1
    print(f"quant-check: passed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
