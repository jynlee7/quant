"""B0: the red/green rules (P2), the test tiers (P4), the trace matrix and the commit hook."""

from __future__ import annotations

import pytest

from tests.support.planted import PlantedRepo, fake_secret_line

pytestmark = pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)

RED_TEST_THAT_PASSES = """\
import pytest


@pytest.mark.spec("P2")
@pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)
def test_already_green() -> None:
    assert True
"""

RED_TEST_WITH_A_TYPO = """\
import pytest


@pytest.mark.spec("P2")
@pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)
def test_with_a_typo() -> None:
    assert undefined_name == 1  # noqa: F821
"""

RED_TEST_STILL_RED = """\
import pytest


@pytest.mark.spec("P2")
@pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)
def test_still_red() -> None:
    raise NotImplementedError
"""


# --- P2: red marks -------------------------------------------------------------------------


@pytest.mark.spec("P2", "P3")
def test_a_red_test_that_passes_fails_the_gate(planted: PlantedRepo) -> None:
    planted.write("tests/test_planted.py", RED_TEST_THAT_PASSES)
    result = planted.run_gate("--only", "tests")
    assert result.caught("P2", "tests/test_planted.py::test_already_green"), result.explain()


@pytest.mark.spec("P2", "P3")
def test_a_red_test_failing_for_another_reason_fails_the_gate(planted: PlantedRepo) -> None:
    planted.write("tests/test_planted.py", RED_TEST_WITH_A_TYPO)
    result = planted.run_gate("--only", "tests")
    assert result.caught("tests", "tests/test_planted.py::test_with_a_typo"), result.explain()


@pytest.mark.spec("P2", "P3")
@pytest.mark.parametrize(
    "mark",
    [
        'xfail(reason="red:B0", raises=NotImplementedError)',
        'xfail(strict=True, reason="later", raises=NotImplementedError)',
        'xfail(strict=True, reason="red:B0")',
    ],
    ids=["not-strict", "no-step", "no-expected-failure"],
)
def test_a_malformed_red_mark_fails_the_gate(planted: PlantedRepo, mark: str) -> None:
    planted.write(
        "tests/test_planted.py",
        "import pytest\n\n\n"
        f'@pytest.mark.spec("P2")\n@pytest.mark.{mark}\n'
        "def test_marked() -> None:\n    raise NotImplementedError\n",
    )
    result = planted.run_gate("--only", "tests")
    assert result.caught("P2", "tests/test_planted.py::test_marked"), result.explain()


@pytest.mark.spec("P2", "P3")
def test_a_red_mark_left_over_from_a_finished_step_fails_the_gate(planted: PlantedRepo) -> None:
    planted.write("tests/test_planted.py", RED_TEST_STILL_RED)
    planted.git_init(branch="b01-time")
    planted.commit_all()
    planted.git("tag", "b00-done")
    result = planted.run_gate("--only", "P2")
    assert result.caught("P2", "tests/test_planted.py:5"), result.explain()


@pytest.mark.spec("P2", "P3")
def test_a_red_mark_for_another_step_fails_the_gate(planted: PlantedRepo) -> None:
    planted.write("tests/test_planted.py", RED_TEST_STILL_RED)
    planted.git_init(branch="b01-time")
    planted.commit_all()
    result = planted.run_gate("--only", "P2")
    assert result.caught("P2", "tests/test_planted.py:5"), result.explain()


@pytest.mark.spec("P2")
def test_a_red_mark_on_its_own_unfinished_step_passes_the_audit(planted: PlantedRepo) -> None:
    planted.write("tests/test_planted.py", RED_TEST_STILL_RED)
    planted.git_init(branch="b00-bootstrap")
    planted.commit_all()
    assert planted.run_gate("--only", "P2").returncode == 0
    # ...but a step can't merge while it is still red.
    merging = planted.run_gate("--full", "--only", "P2")
    assert merging.caught("P2", "tests/test_planted.py:5"), merging.explain()


# --- Trace: every test names its ids, and every claimed id has a green test -----------------


@pytest.mark.spec("P2")
def test_a_test_naming_no_spec_id_fails_the_gate(planted: PlantedRepo) -> None:
    planted.write("tests/test_planted.py", "def test_unnamed() -> None:\n    pass\n")
    result = planted.run_gate("--only", "tests")
    assert result.caught("trace", "tests/test_planted.py::test_unnamed"), result.explain()


@pytest.mark.spec("P2")
def test_an_unknown_spec_id_fails_the_gate(planted: PlantedRepo) -> None:
    planted.write(
        "tests/test_planted.py",
        'import pytest\n\n\n@pytest.mark.spec("K7")\ndef test_k7() -> None:\n    pass\n',
    )
    result = planted.run_gate("--only", "tests")
    assert result.caught("trace", "tests/test_planted.py::test_k7"), result.explain()


@pytest.mark.spec("P2")
def test_trace_fails_while_a_claimed_id_has_no_green_test(planted: PlantedRepo) -> None:
    planted.write("tests/steps.toml", '[steps.B0]\ncovers = ["P3", "K1"]\n')
    planted.git_init(branch="b00-bootstrap")
    planted.commit_all()
    result = planted.run_gate("--trace")
    assert result.caught("trace", "B0"), result.explain()
    assert "K1" in result.stdout


@pytest.mark.spec("P2")
def test_trace_passes_and_prints_the_matrix_when_claims_are_covered(planted: PlantedRepo) -> None:
    planted.write("tests/steps.toml", '[steps.B0]\ncovers = ["P3"]\n')
    planted.git_init(branch="b00-bootstrap")
    planted.commit_all()
    result = planted.run_gate("--trace")
    assert result.returncode == 0, result.explain()
    assert "tests/test_sanity.py::test_the_copy_imports" in result.stdout


# --- P4: tiers ----------------------------------------------------------------------------


@pytest.mark.spec("P4")
def test_the_fast_tier_is_offline(planted: PlantedRepo) -> None:
    planted.write(
        "tests/test_planted.py",
        "import socket\n\nimport pytest\n\n\n"
        '@pytest.mark.spec("P4")\ndef test_reaches_out() -> None:\n'
        '    socket.create_connection(("127.0.0.1", 9), timeout=1)\n',
    )
    result = planted.run_gate("--only", "tests")
    assert result.caught("tests", "tests/test_planted.py::test_reaches_out"), result.explain()
    assert "offline" in result.stdout


@pytest.mark.spec("P4")
def test_full_tier_tests_run_only_in_the_full_tier(planted: PlantedRepo) -> None:
    planted.write(
        "tests/test_planted.py",
        "import pytest\n\n\n"
        '@pytest.mark.full\n@pytest.mark.spec("P4")\ndef test_slow() -> None:\n'
        "    raise AssertionError('ran')\n",
    )
    assert planted.run_gate("--only", "tests").returncode == 0
    result = planted.run_gate("--full", "--only", "tests")
    assert result.caught("tests", "tests/test_planted.py::test_slow"), result.explain()


# --- The whole gate, and the commit hook ---------------------------------------------------


@pytest.mark.full
@pytest.mark.spec("P2", "P3")
def test_the_hook_passes_a_clean_commit_and_blocks_a_bad_one(planted: PlantedRepo) -> None:
    planted.git_init(branch="b00-bootstrap")
    clean = planted.commit_through_hook("clean skeleton")
    assert clean.returncode == 0, clean.stdout + clean.stderr

    head = planted.git("rev-parse", "HEAD")
    planted.write("src/quant/core/leak.py", fake_secret_line("alpaca"))
    planted.write(
        "src/quant/features/leak.py",
        '"""Planted."""\n\nfrom datetime import UTC, datetime\n\nSTAMP = datetime.now(UTC)\n',
    )
    blocked = planted.commit_through_hook("bad commit")
    output = blocked.stdout + blocked.stderr
    assert blocked.returncode != 0, output
    assert "FAIL K6 src/quant/features/leak.py:5" in output
    assert "FAIL I6 src/quant/core/leak.py:1" in output
    assert planted.git("rev-parse", "HEAD") == head


@pytest.mark.full
@pytest.mark.spec("P2")
def test_the_hook_refuses_unstaged_changes(planted: PlantedRepo) -> None:
    planted.git_init(branch="b00-bootstrap")
    planted.commit_all()
    planted.write("src/quant/core/new.py", '"""Staged."""\n')
    planted.git("add", "-A")
    planted.write("src/quant/core/new.py", '"""Edited after staging."""\n')
    proc = planted.commit_through_hook("partial", stage_all=False)
    assert proc.returncode != 0
    assert "unstaged changes" in proc.stderr
