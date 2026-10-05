"""AST scans for the rules import-linter can't express (ARCHITECTURE §3 and §13).

- K5: the approved-order types are constructed only in `quant.risk.engine`, and `TrialTicket`
  only in `quant.research.ledger`.
- K6: `datetime.now`, `time.time` and every other read of the wall clock appear only in
  `quant.core.clock`.
- I2: nothing outside `quant.eval` calls a smoother or Viterbi (`predict`, `predict_proba`,
  `decode`, `score_samples`, `quant.eval.smoother`), and only `quant.hmm.train` imports hmmlearn.

Each `*_violations` function takes one module's source and returns (line, message) pairs.
"""

from __future__ import annotations

from quant_check.findings import Finding
from quant_check.repo import Repo

Violation = tuple[int, str]


def k5_violations(source: str, module: str) -> list[Violation]:
    raise NotImplementedError("red:B0")


def k6_violations(source: str, module: str) -> list[Violation]:
    raise NotImplementedError("red:B0")


def i2_violations(source: str, module: str) -> list[Violation]:
    raise NotImplementedError("red:B0")


def scan_k5(repo: Repo) -> list[Finding]:
    raise NotImplementedError("red:B0")


def scan_k6(repo: Repo) -> list[Finding]:
    raise NotImplementedError("red:B0")


def scan_i2(repo: Repo) -> list[Finding]:
    raise NotImplementedError("red:B0")
