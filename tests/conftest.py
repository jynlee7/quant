"""Shared fixtures. Markers, tiers and red-mark rules live in quant_check.pytest_plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.planted import PlantedRepo


@pytest.fixture
def planted(tmp_path: Path) -> PlantedRepo:
    """A fresh copy of the repo to plant one violation in (P3)."""
    return PlantedRepo.copy_of_repo(tmp_path / "repo")
