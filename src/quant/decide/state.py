"""EngineState: everything `decide()` carries from one decision to the next (ARCHITECTURE §4.3).

The state is decide's own: the feature stream, the filter, each trading playbook's indicators,
the switching state, the live trade, each playbook's shadow and the edge accumulators. The risk
state (start-of-day and peak equity, the daily-loss lock, intents today) stays the RiskEngine's
(design call 2): decide receives its verdicts as flags, and B14 persists both states together.
ARCHITECTURE §4.3 lists the risk state inside EngineState; the ARCH note records the split.

**Fields** (every one plain JSON data, so the state has one canonical JSON form):

- `features`: the FeatureStream's snapshot (`quant.features.stream.FeatureState`), None before
  the first warm-up;
- `filter`: the ForwardFilter's snapshot (`quant.hmm.filter.FilterState`), whose `model_hash`
  names the model it runs; None before the first warm-up. A model other than the one named
  (a refit's activation, a rollback) makes decide warm a fresh stream and filter (SPEC §6,
  "Model swap");
- `indicators`: per trading playbook's label (trend_following and mean_reversion; flat
  playbooks have none), its IndicatorStream's snapshot. Empty before the first warm-up; a state
  whose config hash isn't the playbook's (a new release) is warmed afresh;
- `switching`: `quant.decide.switching.SwitchState` (active label, challenger, cooldown,
  health streak, invalidation block);
- `trade`: the live trade under the active playbook (`quant.decide.primitives.TradeState`:
  entry price, stop, take-profit, bars in trade, peak close); flat when no label is active;
- `shadows`: per trading playbook's label, its shadow (`quant.decide.edge.ShadowState`); a
  label without one starts cold;
- `edge`: the edge accumulators (`quant.decide.edge.EdgeBook`).

**Canonical form.** `to_json()` is `quant.core.canonical.canonical_dumps(self)`, `from_json`
reads it back with pydantic's strict JSON mode (`model_validate_json`: labels and tuples are
read from their JSON text, as everywhere else in the project), and `state_hash()` is
`quant.core.canonical.sha256_hex(self)`, SHA-256 of those bytes. Floats round-trip exactly, so
`from_json(s.to_json()) == s` and its hash is `s.state_hash()`.
"""

from __future__ import annotations

import hashlib
import json

from quant.core.canonical import canonical_dumps
from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.decide.config import SwitchingConfig
from quant.decide.edge import EdgeBook, ShadowState
from quant.decide.indicators import IndicatorState
from quant.decide.primitives import TradeState
from quant.decide.switching import SwitchState
from quant.features.stream import FeatureState
from quant.hmm.filter import FilterState


class EngineState(StrictModel):
    """decide's state (module docstring). Frozen: decide returns a new one."""

    features: FeatureState | None
    filter: FilterState | None
    indicators: dict[Label, IndicatorState]
    switching: SwitchState
    trade: TradeState
    shadows: dict[Label, ShadowState]
    edge: EdgeBook

    @classmethod
    def cold(cls, config: SwitchingConfig, *, edge: EdgeBook | None = None) -> EngineState:
        """A run's first state (C4): no stream, filter or indicators yet (the first decision
        warms them), `SwitchState.cold(config)` (no active label, a full health streak), a flat
        live trade (`TradeState.flat()`), no shadows, and `edge` (an empty EdgeBook when None;
        a backtest passes its fit-window seed, spec §9)."""
        return cls(
            features=None,
            filter=None,
            indicators={},
            switching=SwitchState.cold(config),
            trade=TradeState.flat(),
            shadows={},
            edge=EdgeBook.empty() if edge is None else edge,
        )

    def to_json(self) -> bytes:
        """`canonical_dumps(self)`."""
        return canonical_dumps(self)

    @classmethod
    def from_json(cls, data: bytes) -> EngineState:
        """The state `data` (canonical JSON, as `to_json` writes it) describes, through
        `model_validate_json`. pydantic's ValidationError for anything else."""
        return cls.model_validate_json(data)

    def state_hash(self) -> str:
        """`sha256_hex(self)`: 64 lowercase hex characters, the same for equal states and for a
        state read back with `from_json`. The DecisionRecord's `state_before` and
        `state_after`, and the T-NL1 harness's `Record.state`."""
        # These states contain only JSON scalars and containers (no datetimes).
        # Pydantic serializes them directly; avoid a second recursive normalization.
        # Recompute: model_copy and nested state changes must never reuse a digest.
        text = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
