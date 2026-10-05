"""Import rules K1–K4 and P11, checked with import-linter on the tree being checked."""

from __future__ import annotations

import configparser
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from quant_check.findings import Finding
from quant_check.repo import Repo, scrubbed_env
from quant_check.scans import k4_trading_violations, scan

CONFIG = ".importlinter"

# Contracts written as "everything except": each subpackage of quant must appear in the listed
# field unless it is one of the exemptions, so a new subpackage can't slip past the rule.
EXHAUSTIVE: dict[str, tuple[str, frozenset[str]]] = {
    "K1": ("forbidden_modules", frozenset({"risk", "core", "store"})),
    "K2": ("forbidden_modules", frozenset({"llm", "core"})),
    "K4": ("source_modules", frozenset({"engine", "control", "cli", "execution"})),
}


def _module_location(repo: Repo, module: str, line: int | None) -> str:
    base = repo.root / "src" / Path(*module.split("."))
    path = base.with_suffix(".py") if base.with_suffix(".py").exists() else base / "__init__.py"
    where = repo.relative(path) if path.exists() else module
    return f"{where}:{line}" if line else where


def _broken_contracts(repo: Repo) -> list[Finding]:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "importlinter.json"
        proc = subprocess.run(
            [sys.executable, "-m", "quant_check.importlinter_runner", CONFIG, str(output)],
            cwd=repo.root,
            capture_output=True,
            text=True,
            check=False,
            env=scrubbed_env({"PYTHONPATH": str(repo.root / "src"), "NO_COLOR": "1"}),
        )
        if not output.exists():
            tail = (proc.stderr or proc.stdout).strip()[-500:]
            return [Finding("imports", CONFIG, f"import-linter didn't run: {tail}")]
        result: dict[str, Any] = json.loads(output.read_text())
    if "error" in result:
        return [Finding("imports", CONFIG, f"import-linter failed: {result['error']}")]
    findings = [
        Finding("imports", CONFIG, f"invalid contract {name}: {error}")
        for name, error in result["invalid"].items()
    ]
    for contract in result["contracts"]:
        for chain in contract["chains"]:
            first = chain[0]
            lines = first["line_numbers"]
            path = " -> ".join([first["importer"], *(link["imported"] for link in chain)])
            findings.append(
                Finding(
                    contract["id"],
                    _module_location(repo, first["importer"], lines[0] if lines else None),
                    f"{path} ({contract['name']})",
                )
            )
    return findings


def _subpackages(repo: Repo) -> set[str]:
    return {p.parent.name for p in repo.package_dir.glob("*/__init__.py")}


def _uncovered_subpackages(repo: Repo) -> list[Finding]:
    parser = configparser.ConfigParser()
    parser.read(repo.root / CONFIG)
    findings = []
    for rule, (field, exempt) in EXHAUSTIVE.items():
        section = f"importlinter:contract:{rule}"
        if not parser.has_section(section):
            findings.append(Finding(rule, CONFIG, f"contract {rule} is missing"))
            continue
        listed = {
            m.strip().removeprefix("quant.")
            for m in parser.get(section, field).split()
            if m.strip().startswith("quant.") and m.strip().count(".") == 1
        }
        for name in sorted(_subpackages(repo) - listed - exempt):
            findings.append(
                Finding(rule, CONFIG, f"contract {rule} doesn't place quant.{name} in {field}")
            )
    return findings


def check_imports(repo: Repo) -> list[Finding]:
    """Broken contracts, contracts that miss a subpackage, and alpaca-py's trading client
    imported anywhere but the adapter (K4)."""
    if not (repo.root / CONFIG).exists():
        return [Finding("imports", CONFIG, "the import contracts are missing")]
    return [
        *_broken_contracts(repo),
        *_uncovered_subpackages(repo),
        *scan(repo, "K4", k4_trading_violations),
    ]
