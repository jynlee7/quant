"""The DecisionRecord: what one decision saw, ruled and asked for (ARCHITECTURE §4.3, §13, spec §8
"Log", T-NL2).

A record holds:

- **when**: `decision_id`, `tau` and the view's `as_of` (ARCHITECTURE §6.2), and the view's
  `digest()` after every read decide made (the ids and hashes of the bars used);
- **what governed it**: the release id, the model's artifact hash and fit end, the
  calibration's artifact hash and as-of time (None without one), engine.yaml's `config_hash`,
  the caps' `caps_hash` and the code commit. T-NL2 needs the model's fit end and the
  calibration's as-of time strictly before tau; the record names both, whatever they are, and
  rule 6 makes a bar that breaks them unhealthy;
- **what it read**: the decision bar's raw feature vector (None when it is unavailable), and
  the filter's outputs for it (`FilterView`: state, label and next-bar probabilities, the
  entropy, the predictive log-likelihood and the top label), None on a bar whose features or
  filter failed;
- **what it ruled**: rule 6's verdict (`health`), the active label after switching, whether it
  switched or de-risked, whether its playbook is blocked, the rule trace (one entry per
  `quant.decide.switching.RuleId`, in trace order, with whether it fired and why), the live
  primitive's reason on the bar and the invalidation that hit, if any;
- **what it asked for**: the target and its factors (`TargetView`) and the intents;
- **the risk verdicts**: empty from decide; the driver adds them with `with_verdicts`, which
  returns a new record (design call 2);
- **the state hashes** before and after (`EngineState.state_hash`).

Every field is plain JSON data and the record is frozen. `record_hash()` is
`quant.core.canonical.sha256_hex(self)`: replays and the T-NL1 harness compare records hash for
hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import AwareDatetime

from quant.core.canonical import canonical_dumps, canonical_loads, sha256_hex
from quant.core.model import StrictModel
from quant.core.orders import OrderIntent
from quant.core.regime import Label


class HealthView(StrictModel):
    """Rule 6 on the bar: `cause` (a `quant.decide.health.Cause` value, None when healthy),
    the healthy `streak` and whether the bar is `operating`."""

    cause: str | None
    streak: int
    operating: bool


class FilterView(StrictModel):
    """The filter's outputs for the decision bar (`quant.hmm.filter.FilterOutput`), as data:
    the state and next-state probabilities (lists, in state order), the label and next-bar label
    probabilities (keyed by the model's labels; a label it lacks is absent), the normalized
    entropy, the predictive log-likelihood and the top label."""

    state_probs: list[float]
    next_state_probs: list[float]
    label_probs: dict[Label, float]
    next_label_probs: dict[Label, float]
    entropy: float
    loglik: float
    top_label: Label


class RuleView(StrictModel):
    """One `quant.decide.switching.RuleTrace`: its rule id, whether it fired, and why."""

    rule: str
    fired: bool
    reason: str


class TargetView(StrictModel):
    """The target and how it was reached (`quant.decide.sizing`):

    - `value`: the target (`Target.value`); `zero`: its ZeroReason value, or None;
    - `sizing`: the formula's factors as a mapping (`Sizing`'s fields, labels by value), None
      when the formula didn't run;
    - `current`: the exposure before the decision (shares x price / equity), None without a
      price;
    - `ceiling`, `desired`, `move`, `limited`, `forced`, `held`: `sizing.Rebalance`'s fields,
      None without a price;
    - `price`: the decision's last close the exposure and the intent use, None when the view
      holds no bar.
    """

    value: float
    zero: str | None
    sizing: dict[str, Any] | None
    current: float | None
    ceiling: float | None
    desired: float | None
    move: bool | None
    limited: bool | None
    forced: bool | None
    held: bool | None
    price: float | None


class DecisionRecord(StrictModel):
    """One decision's record (module docstring)."""

    decision_id: str
    tau: AwareDatetime
    as_of: AwareDatetime
    view_digest: str
    release_id: str
    model_hash: str
    model_fit_end: AwareDatetime
    calibration_hash: str | None
    calibration_as_of: AwareDatetime | None
    engine_hash: str
    caps_hash: str
    code_commit: str
    account: dict[str, float | int]
    flags: dict[str, bool]
    health: HealthView
    features: list[float] | None
    filter: FilterView | None
    active: Label | None
    switched: bool
    derisk: bool
    blocked: bool
    trace: tuple[RuleView, ...]
    trade_reason: str | None
    invalidation: str | None
    target: TargetView
    intents: tuple[OrderIntent, ...]
    verdicts: tuple[dict[str, Any], ...]
    state_before: str
    state_after: str

    def record_hash(self) -> str:
        """`sha256_hex(self)`, 64 lowercase hex characters."""
        return sha256_hex(self)

    def with_verdicts(self, verdicts: Sequence[Mapping[str, Any]]) -> DecisionRecord:
        """A new record, equal to this one but with `verdicts`: one plain-JSON mapping per
        intent, in the intents' order (the driver's rendering of each RiskVerdict). This record
        is unchanged. ValueError, its message saying "verdict", when the count isn't the intents', CanonicalJSONError
        (`quant.core.canonical`) for a value without a canonical JSON form."""
        if len(verdicts) != len(self.intents):
            raise ValueError("one verdict per intent")
        items = tuple(dict(canonical_loads(canonical_dumps(dict(v)))) for v in verdicts)
        return self.model_copy(update={"verdicts": items})

    def provenance_times(self) -> tuple[datetime, ...]:
        """The fit end and as-of times the decision depended on, for T-NL2: `model_fit_end`,
        and `calibration_as_of` when there is a calibration."""
        times = [self.model_fit_end]
        if self.calibration_as_of is not None:
            times.append(self.calibration_as_of)
        return tuple(times)
