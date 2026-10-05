"""B0: the AST scans behind K5, K6 and I2, case by case.

Each scan must catch the ways a violation is usually written (aliases, attribute chains,
`from` imports) without flagging the legitimate code next to it.
"""

from __future__ import annotations

import pytest

from quant_check import scans

# --- K6: the wall clock is read only in quant.core.clock ------------------------------------

K6_CAUGHT = {
    "datetime.datetime.now": "import datetime\nx = datetime.datetime.now()\n",
    "from-import now": "from datetime import datetime\nx = datetime.now()\n",
    "aliased utcnow": "from datetime import datetime as dt\nx = dt.utcnow()\n",
    "date.today": "from datetime import date\nx = date.today()\n",
    "module alias": "import datetime as d\nx = d.date.today()\n",
    "time.time": "import time\nx = time.time()\n",
    "time_ns via alias": "import time as t\nx = t.time_ns()\n",
    "from time import time": "from time import time\nx = time()\n",
    "monotonic": "from time import monotonic\nx = monotonic()\n",
    "perf_counter alias": "from time import perf_counter as pc\nx = pc()\n",
    "pandas now": "import pandas as pd\nx = pd.Timestamp.now()\n",
    "pandas 'now'": "import pandas as pd\nx = pd.Timestamp('now')\n",
    "to_datetime 'today'": "import pandas as pd\nx = pd.to_datetime('today')\n",
    "numpy 'now'": "import numpy as np\nx = np.datetime64('now')\n",
}

K6_ALLOWED = {
    "the Clock port": "def f(clock):\n    return clock.now()\n",
    "sleep": "import time\ntime.sleep(0)\n",
    "parsing": "from datetime import datetime\nx = datetime.fromisoformat('2020-01-02T00:00:00+00:00')\n",
    "pandas literal": "import pandas as pd\nx = pd.Timestamp('2020-01-02')\n",
}


@pytest.mark.spec("K6")
@pytest.mark.parametrize("source", K6_CAUGHT.values(), ids=K6_CAUGHT.keys())
def test_k6_catches(source: str) -> None:
    assert [line for line, _ in scans.k6_violations(source, "quant.features.x")] == [2]


@pytest.mark.spec("K6")
@pytest.mark.parametrize("source", K6_ALLOWED.values(), ids=K6_ALLOWED.keys())
def test_k6_allows(source: str) -> None:
    assert scans.k6_violations(source, "quant.features.x") == []


@pytest.mark.spec("K6")
def test_k6_allows_the_clock_module_itself() -> None:
    assert scans.k6_violations("import time\nx = time.time()\n", "quant.core.clock") == []


# --- K5: approved orders and trial tickets have one constructor each -----------------------

K5_CAUGHT = {
    "direct": "from quant.risk.engine import RiskApprovedOrder\nRiskApprovedOrder(intent=1)\n",
    "aliased": "from quant.risk.engine import RiskApprovedOrder as R\nR()\n",
    "module alias": "import quant.risk.engine as e\ne.RiskApprovedFlatten()\n",
    "attribute": "from quant.risk import engine\nengine.RiskApprovedCancel()\n",
    "__new__": "from quant.risk.engine import RiskApprovedOrder\nobject.__new__(RiskApprovedOrder)\n",
    "model_construct": "from quant.risk.engine import RiskApprovedOrder\nRiskApprovedOrder.model_construct()\n",
    "ticket": "from quant.research.ledger import TrialTicket\nTrialTicket(trial_id=1)\n",
}

K5_ALLOWED = {
    "annotation": "from quant.risk.engine import RiskApprovedOrder\ndef submit(o: RiskApprovedOrder) -> None: ...\n",
    "isinstance": "from quant.risk.engine import RiskApprovedOrder\nok = isinstance(1, RiskApprovedOrder)\n",
}


@pytest.mark.spec("K5", "I4")
@pytest.mark.parametrize("source", K5_CAUGHT.values(), ids=K5_CAUGHT.keys())
def test_k5_catches(source: str) -> None:
    assert [line for line, _ in scans.k5_violations(source, "quant.decide.x")] == [2]


@pytest.mark.spec("K5", "I4")
@pytest.mark.parametrize("source", K5_ALLOWED.values(), ids=K5_ALLOWED.keys())
def test_k5_allows(source: str) -> None:
    assert scans.k5_violations(source, "quant.decide.x") == []


@pytest.mark.spec("K5", "I4")
def test_k5_allows_each_type_only_in_its_own_module() -> None:
    order = "from quant.risk.engine import RiskApprovedOrder\nRiskApprovedOrder()\n"
    ticket = "from quant.research.ledger import TrialTicket\nTrialTicket()\n"
    assert scans.k5_violations(order, "quant.risk.engine") == []
    assert scans.k5_violations(ticket, "quant.research.ledger") == []
    assert scans.k5_violations(order, "quant.research.ledger") != []
    assert scans.k5_violations(ticket, "quant.risk.engine") != []


# --- I2: no smoother or Viterbi outside eval, and hmmlearn only in hmm.train ---------------

I2_CAUGHT = {
    "predict_proba": ("quant.hmm.filter", "def f(m, x):\n    return m.predict_proba(x)\n"),
    "predict": ("quant.decide.x", "def f(m, x):\n    return m.predict(x)\n"),
    "score_samples": ("quant.engine.x", "def f(m, x):\n    return m.score_samples(x)\n"),
    "decode viterbi": (
        "quant.hmm.filter",
        "def f(m, x):\n    return m.decode(x, algorithm='viterbi')\n",
    ),
    "decode": ("quant.hmm.filter", "def f(m, x):\n    return m.decode(x)\n"),
    "getattr": ("quant.hmm.filter", "def f(m):\n    return getattr(m, 'predict_proba')\n"),
    "smoother module": ("quant.decide.x", "import os\nfrom quant.eval import smoother\n"),
    "smoother import": ("quant.decide.x", "import os\nimport quant.eval.smoother\n"),
    "smoother function": ("quant.decide.x", "import os\nfrom quant.eval.smoother import smooth\n"),
    "hmmlearn in filter": ("quant.hmm.filter", "import os\nimport hmmlearn\n"),
    "hmmlearn in research": (
        "quant.research.x",
        "import os\nfrom hmmlearn.hmm import GaussianHMM\n",
    ),
}

I2_ALLOWED = {
    "smoother in eval": ("quant.eval.smoother", "def f(m, x):\n    return m.predict_proba(x)\n"),
    "isotonic in eval": ("quant.eval.calibration", "def f(iso, x):\n    return iso.predict(x)\n"),
    "hmmlearn in train": ("quant.hmm.train", "from hmmlearn.hmm import GaussianHMM\n"),
    "bytes.decode()": ("quant.data.x", "def f(b: bytes) -> str:\n    return b.decode()\n"),
    "bytes.decode(enc)": (
        "quant.data.x",
        "def f(b: bytes) -> str:\n    return b.decode('utf-8')\n",
    ),
    "bytes.decode(kw)": (
        "quant.data.x",
        "def f(b: bytes) -> str:\n    return b.decode(encoding='ascii', errors='strict')\n",
    ),
}


@pytest.mark.spec("I2")
@pytest.mark.parametrize(("module", "source"), I2_CAUGHT.values(), ids=I2_CAUGHT.keys())
def test_i2_catches(module: str, source: str) -> None:
    assert [line for line, _ in scans.i2_violations(source, module)] == [2]


@pytest.mark.spec("I2")
@pytest.mark.parametrize(("module", "source"), I2_ALLOWED.values(), ids=I2_ALLOWED.keys())
def test_i2_allows(module: str, source: str) -> None:
    assert scans.i2_violations(source, module) == []


# --- Review of B0: cases the first version missed --------------------------------------------

K5_COPIES = {
    "model_copy": (
        "from quant.risk.engine import RiskApprovedOrder\n"
        "def f(o: RiskApprovedOrder):\n    return o.model_copy(update={'qty': 9})\n"
    ),
    "dataclasses.replace": (
        "import dataclasses\nfrom quant.risk.engine import RiskApprovedOrder\n"
        "def f(o: RiskApprovedOrder):\n    return dataclasses.replace(o, qty=9)\n"
    ),
    "copy.copy": (
        "import copy\nfrom quant.risk.engine import RiskApprovedOrder\n"
        "def f(o: RiskApprovedOrder):\n    return copy.copy(o)\n"
    ),
    "copy.replace": (
        "from copy import replace\nfrom quant.research.ledger import TrialTicket\n"
        "def f(t: TrialTicket):\n    return replace(t)\n"
    ),
}


@pytest.mark.xfail(strict=True, reason="red:B0", raises=AssertionError)
@pytest.mark.spec("K5", "I4")
@pytest.mark.parametrize("source", K5_COPIES.values(), ids=K5_COPIES.keys())
def test_k5_catches_copies_of_an_approved_instance(source: str) -> None:
    last_line = len(source.splitlines())
    assert [line for line, _ in scans.k5_violations(source, "quant.execution.x")] == [last_line]


K6_MORE = {
    "process_time": "import time\nx = time.process_time()\n",
    "thread_time": "import time\nx = time.thread_time_ns()\n",
    "clock_gettime": "import time\nx = time.clock_gettime(0)\n",
    "os.times": "import os\nx = os.times()\n",
    "uuid1": "import uuid\nx = uuid.uuid1()\n",
    "event loop": "import asyncio\nx = asyncio.get_event_loop().time()\n",
    "running loop": "import asyncio\nloop = asyncio.get_running_loop()\nx = loop.time()\n",
}


@pytest.mark.xfail(strict=True, reason="red:B0", raises=AssertionError)
@pytest.mark.spec("K6")
@pytest.mark.parametrize("source", K6_MORE.values(), ids=K6_MORE.keys())
def test_k6_catches_other_clock_reads(source: str) -> None:
    last_line = len(source.splitlines())
    assert [line for line, _ in scans.k6_violations(source, "quant.engine.x")] == [last_line]


@pytest.mark.spec("K6")
def test_k6_allows_the_time_of_day_of_a_datetime() -> None:
    source = "def f(tau):\n    return tau.time()\n"
    assert scans.k6_violations(source, "quant.engine.x") == []


@pytest.mark.xfail(strict=True, reason="red:B0", raises=AssertionError)
@pytest.mark.spec("I2")
def test_i2_hmmlearn_is_imported_only_by_train_even_in_eval() -> None:
    assert [
        line for line, _ in scans.i2_violations("import os\nimport hmmlearn\n", "quant.eval.x")
    ] == [2]
