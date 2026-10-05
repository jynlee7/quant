"""B0: each structural guard catches a planted violation and names its rule (P3).

Every test copies the repo, plants one violation and runs one stage of quant-check on the
copy. A guard that has never failed proves nothing.
"""

from __future__ import annotations

import pytest

from tests.support.planted import PlantedRepo, fake_secret_line

pytestmark = pytest.mark.xfail(strict=True, reason="red:B0", raises=NotImplementedError)


# --- K1–K4 and P11: import rules -----------------------------------------------------------


@pytest.mark.spec("K1", "I3", "P3")
def test_k1_risk_importing_llm_is_caught(planted: PlantedRepo) -> None:
    planted.write("src/quant/risk/leak.py", "import quant.llm\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("K1", "src/quant/risk/leak.py:1"), result.explain()


@pytest.mark.spec("K2", "P3")
def test_k2_llm_importing_execution_is_caught(planted: PlantedRepo) -> None:
    planted.write("src/quant/llm/leak.py", "from quant.execution import alpaca_broker\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("K2", "src/quant/llm/leak.py:1"), result.explain()


@pytest.mark.spec("K3", "I2", "P3")
@pytest.mark.parametrize("line", ["import hmmlearn.hmm\n", "from quant.eval import smoother\n"])
def test_k3_decide_importing_hmmlearn_or_eval_is_caught(planted: PlantedRepo, line: str) -> None:
    planted.write("src/quant/decide/leak.py", line)
    result = planted.run_gate("--only", "imports")
    assert result.caught("K3", "src/quant/decide/leak.py:1"), result.explain()


@pytest.mark.spec("K3", "I2", "P3")
def test_k3_catches_an_indirect_chain(planted: PlantedRepo) -> None:
    # decide -> research is allowed, but research -> eval puts eval on the decision path.
    planted.write("src/quant/research/helper.py", "import quant.eval.smoother\n")
    planted.write("src/quant/decide/leak.py", "import quant.research.helper\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("K3", "src/quant/decide/leak.py:1"), result.explain()


@pytest.mark.spec("K4", "P3")
def test_k4_dashboard_importing_the_alpaca_adapter_is_caught(planted: PlantedRepo) -> None:
    planted.write("src/quant/dashboard/leak.py", "import quant.execution.alpaca_broker\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("K4", "src/quant/dashboard/leak.py:1"), result.explain()


@pytest.mark.spec("K4", "P3")
def test_k4_alpaca_trading_client_outside_the_adapter_is_caught(planted: PlantedRepo) -> None:
    planted.write(
        "src/quant/execution/side_door.py",
        '"""Planted."""\n\nfrom alpaca.trading.client import TradingClient\n',
    )
    result = planted.run_gate("--only", "imports")
    assert result.caught("K4", "src/quant/execution/side_door.py:3"), result.explain()


@pytest.mark.spec("K1", "K2", "K4", "P3")
def test_a_subpackage_missing_from_the_contracts_is_caught(planted: PlantedRepo) -> None:
    planted.write("src/quant/newpkg/__init__.py", '"""Not yet placed in the import rules."""\n')
    result = planted.run_gate("--only", "imports")
    for rule in ("K1", "K2", "K4"):
        assert result.caught(rule, ".importlinter"), result.explain()


@pytest.mark.spec("P11", "P3")
def test_p11_src_importing_a_spike_is_caught(planted: PlantedRepo) -> None:
    planted.write("src/quant/data/leak.py", "import spikes.alpaca_probe\n")
    result = planted.run_gate("--only", "imports")
    assert result.caught("P11", "src/quant/data/leak.py:1"), result.explain()


# --- K5, K6 and I2: AST scans --------------------------------------------------------------


@pytest.mark.spec("K5", "I4", "P3")
def test_k5_risk_approved_order_built_outside_risk_engine_is_caught(planted: PlantedRepo) -> None:
    planted.write(
        "src/quant/decide/forge.py",
        "from quant.risk.engine import RiskApprovedOrder\n\norder = RiskApprovedOrder(intent=None)\n",
    )
    result = planted.run_gate("--only", "K5")
    assert result.caught("K5", "src/quant/decide/forge.py:3"), result.explain()


@pytest.mark.spec("K5", "P3")
def test_k5_trial_ticket_built_outside_the_ledger_is_caught(planted: PlantedRepo) -> None:
    planted.write(
        "src/quant/harness/forge.py",
        "from quant.research.ledger import TrialTicket\n\nticket = TrialTicket()\n",
    )
    result = planted.run_gate("--only", "K5")
    assert result.caught("K5", "src/quant/harness/forge.py:3"), result.explain()


@pytest.mark.spec("K6", "P3")
def test_k6_datetime_now_in_features_is_caught(planted: PlantedRepo) -> None:
    planted.write(
        "src/quant/features/leak.py",
        "from datetime import UTC, datetime\n\nstamp = datetime.now(UTC)\n",
    )
    result = planted.run_gate("--only", "K6")
    assert result.caught("K6", "src/quant/features/leak.py:3"), result.explain()


@pytest.mark.spec("I2", "P3")
def test_i2_predict_proba_in_hmm_filter_is_caught(planted: PlantedRepo) -> None:
    planted.write(
        "src/quant/hmm/filter.py",
        '"""Planted."""\n\n\ndef step(model, x):\n    return model.predict_proba(x)\n',
    )
    result = planted.run_gate("--only", "I2")
    assert result.caught("I2", "src/quant/hmm/filter.py:5"), result.explain()


# --- I6: secrets ---------------------------------------------------------------------------


@pytest.mark.spec("I6", "P3")
@pytest.mark.parametrize("kind", ["alpaca", "anthropic", "telegram"])
def test_i6_fake_key_in_a_tracked_file_is_caught(planted: PlantedRepo, kind: str) -> None:
    planted.git_init()
    planted.write("src/quant/core/leak.py", fake_secret_line(kind))
    planted.git("add", "-A")
    result = planted.run_gate("--only", "I6")
    assert result.caught("I6", "src/quant/core/leak.py:1"), result.explain()


@pytest.mark.spec("I6", "P3")
def test_i6_finds_a_key_that_was_committed_and_then_deleted(planted: PlantedRepo) -> None:
    planted.git_init()
    planted.write("src/quant/core/leak.py", fake_secret_line("alpaca"))
    planted.commit_all("add a key")
    (planted.root / "src/quant/core/leak.py").unlink()
    planted.commit_all("remove it again")
    result = planted.run_gate("--only", "I6")
    assert result.caught("I6", "history:"), result.explain()


@pytest.mark.spec("I6", "P3")
def test_i6_ignores_keys_in_gitignored_env_files(planted: PlantedRepo) -> None:
    planted.git_init()
    planted.write(".env", fake_secret_line("alpaca").replace(" = ", "="))
    planted.write(".env.test", fake_secret_line("alpaca", seed=7).replace(" = ", "="))
    planted.commit_all()
    result = planted.run_gate("--only", "I6")
    assert result.returncode == 0, result.explain()
