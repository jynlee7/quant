"""B0: the ids the trace matrix knows come from the approved documents."""

from __future__ import annotations

import pytest

from quant_check import trace
from quant_check.repo import Repo
from tests.support.planted import REPO


@pytest.mark.spec("P2")
def test_known_ids_are_the_ids_defined_in_the_docs() -> None:
    ids = trace.known_ids(Repo(REPO))
    for expected in ("D1", "D13", "A4", "I6", "T-NL3", "G6", "C1", "C17", "K6", "AD6", "P11"):
        assert expected in ids
    for not_an_id in ("B0", "CP1", "C18", "K7", "D14"):
        assert not_an_id not in ids


@pytest.mark.spec("P2")
def test_every_claimed_id_is_known() -> None:
    repo = Repo(REPO)
    known = trace.known_ids(repo)
    claims = trace.step_claims(repo)
    assert "B0" in claims
    for step, ids in claims.items():
        assert set(ids) <= known, (step, set(ids) - known)


@pytest.mark.spec("P2")
def test_section_references_are_ids_too() -> None:
    # Requirements written as prose (SPEC §2's data contract, ARCHITECTURE §5's schedule) have no
    # table id, so tests name their section: SPEC§2, ARCH§4.3, PLAN§5.
    ids = trace.known_ids(Repo(REPO))
    for expected in ("SPEC§2", "SPEC§0.1", "ARCH§4.3", "ARCH§5", "PLAN§2", "PLAN§5"):
        assert expected in ids
    for not_an_id in ("SPEC§99", "ARCH§4.9", "PLAN§0.9"):
        assert not_an_id not in ids
