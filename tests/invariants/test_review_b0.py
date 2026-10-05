"""B0 review: guards the first code review showed were missing (BUILD_LOG, B0).

Each test plants a violation that the first version of a guard let through.
"""

from __future__ import annotations

import pytest

from quant_check.testrun import budget_findings
from tests.support.planted import PlantedRepo

RED = pytest.mark.xfail(strict=True, reason="red:B0", raises=AssertionError)


@RED
@pytest.mark.spec("K4", "P3")
def test_k4_an_execution_sibling_importing_the_adapter_is_caught(planted: PlantedRepo) -> None:
    planted.write("src/quant/execution/router.py", "from quant.execution import alpaca_broker\n")
    planted.write("src/quant/execution/manager.py", "from . import alpaca_broker\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("K4", "src/quant/execution/router.py:1"), result.explain()
    assert result.caught("K4", "src/quant/execution/manager.py:1"), result.explain()


@RED
@pytest.mark.spec("K1", "P3")
def test_a_broken_contract_is_a_finding_even_without_import_chains(planted: PlantedRepo) -> None:
    contracts = (planted.root / ".importlinter").read_text()
    planted.write(
        ".importlinter",
        contracts
        + "\n[importlinter:contract:LAYERS]\nname = LAYERS decide above data\n"
        + "type = layers\nlayers =\n    quant.decide\n    quant.data\n",
    )
    planted.write("src/quant/data/leak.py", "import quant.decide\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("LAYERS", ".importlinter"), result.explain()


@RED
@pytest.mark.spec("P2", "P3")
def test_a_zero_padded_red_mark_is_malformed(planted: PlantedRepo) -> None:
    planted.write(
        "tests/test_planted.py",
        "import pytest\n\n\n"
        '@pytest.mark.spec("P2")\n'
        '@pytest.mark.xfail(strict=True, reason="red:B03", raises=NotImplementedError)\n'
        "def test_padded() -> None:\n    raise NotImplementedError\n",
    )
    result = planted.run_gate("--only", "tests")
    assert result.caught("P2", "tests/test_planted.py::test_padded"), result.explain()


@RED
@pytest.mark.spec("P4")
@pytest.mark.parametrize(
    "reach_out",
    [
        'socket.gethostbyname("example.invalid")',
        'socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("192.0.2.1", 9))',
    ],
    ids=["dns", "udp"],
)
def test_the_fast_tier_blocks_lookups_and_datagrams(planted: PlantedRepo, reach_out: str) -> None:
    planted.write(
        "tests/test_planted.py",
        "import socket\n\nimport pytest\n\n\n"
        f'@pytest.mark.spec("P4")\ndef test_reaches_out() -> None:\n    {reach_out}\n',
    )
    result = planted.run_gate("--only", "tests")
    assert result.caught("tests", "tests/test_planted.py::test_reaches_out"), result.explain()
    assert "offline" in result.stdout


@pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)
@pytest.mark.spec("P4")
def test_a_fast_tier_over_two_minutes_is_a_finding() -> None:
    results = [
        {"nodeid": "tests/a.py::t1", "tier": "fast", "duration": 70.0},
        {"nodeid": "tests/a.py::t2", "tier": "fast", "duration": 60.0},
        {"nodeid": "tests/a.py::t3", "tier": "full", "duration": 900.0},
    ]
    findings = budget_findings(results)
    assert [f.rule for f in findings] == ["P4"]
    assert "130" in findings[0].message
    assert budget_findings(results[1:]) == []


@RED
@pytest.mark.full
@pytest.mark.spec("P2", "P3")
def test_the_merge_hook_blocks_a_step_that_is_still_red(planted: PlantedRepo) -> None:
    planted.git_init(branch="main")
    planted.commit_all("main")
    planted.git("checkout", "-q", "-b", "b00-bootstrap")
    planted.write(
        "tests/test_planted.py",
        "import pytest\n\n\n"
        '@pytest.mark.spec("P2")\n'
        '@pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)\n'
        "def test_still_red() -> None:\n    raise NotImplementedError\n",
    )
    planted.commit_all("red")
    planted.git("checkout", "-q", "main")
    merge = planted.merge_through_hook("b00-bootstrap")
    output = merge.stdout + merge.stderr
    assert merge.returncode != 0, output
    assert "FAIL P2 tests/test_planted.py:5" in output
