"""pytest plugin loaded by every test run (pyproject addopts).

Red phase of B0: it only registers the project's markers.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "spec(*ids): spec, architecture or plan ids the test enforces"
    )
    config.addinivalue_line(
        "markers", "full: full tier only, run before each merge and nightly (P4)"
    )
    config.addinivalue_line("markers", "integration: needs the Alpaca test account (AD3)")
