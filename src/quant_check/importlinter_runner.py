"""Runs import-linter and writes broken contracts as JSON.

quant-check starts this in a subprocess whose PYTHONPATH begins with the checked tree's src/,
so the graph is built from that tree (a planted copy, say) and not the installed package.
Usage: python -m quant_check.importlinter_runner <config> <output.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from importlinter import configuration
from importlinter.application import use_cases


def lint(config: str) -> dict[str, Any]:
    configuration.configure()  # type: ignore[no-untyped-call]
    options = use_cases.read_user_options(config_filename=config)
    use_cases._register_contract_types(options)
    report = use_cases.create_report(options, cache_dir=None)
    invalid = {name: str(exc) for name, exc in report.invalid_contract_options.items()}
    contracts = []
    for contract, check in report.get_contracts_and_checks():
        chains = [
            chain
            for data in check.metadata.get("invalid_chains", [])
            for chain in data.get("chains", [])
        ]
        contracts.append(
            {
                "id": contract.contract_options.get("id", contract.name),
                "name": contract.name,
                "kept": check.kept,
                "chains": chains,
            }
        )
    return {"contracts": contracts, "invalid": invalid}


def main() -> None:
    config, output = sys.argv[1], Path(sys.argv[2])
    try:
        result = lint(config)
    except Exception as exc:  # reported as a finding by quant-check, never swallowed
        result = {"error": f"{type(exc).__name__}: {exc}"}
    output.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
