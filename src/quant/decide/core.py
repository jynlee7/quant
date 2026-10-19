"""`decide()`: the pure decision core (ARCHITECTURE §4.1-§4.3, spec §6-§9, design calls 1-2).

    decide(state, view, inputs) -> DecisionOutcome(state, record, intents)

ARCHITECTURE §4.1 writes `decide(state, view, release)`; `inputs` widens `release` with what the
driver (the engine, the backtest, a replay) knows and decide may not look up itself (design call
1): the governing model and calibration, engine.yaml, the caps as data, the Decision, the
account and the flags, and the code commit. **Pure**: no I/O beyond the view's reads, no clock
(tau comes from the Decision, K6), no randomness, no logging. The same state, view contents and
inputs give the same outcome, byte for byte.

**One decision, in order** (ARCHITECTURE §4.2 steps 3-10), with tau = `inputs.decision.tau`,
symbol SYMBOL and every view read with purpose OPERATE:

1. **Preconditions** (ValueError, a driver bug, not rule 6, its message naming `tau`): `view.tau`
   must be tau (a view
   at another time is the driver's lookahead, I1). The Account's own fields bound the equity
   (> 0) and the position (>= 0: long or flat, D9).
2. **Warm-up** (SPEC §6 "Model swap", C4): when `state.filter` is None or names another model
   than `inputs.model` (its artifact hash), a fresh FeatureStream under the release's feature
   config and a fresh `ForwardFilter(inputs.model)` are warmed over the view: the bars opening
   at or after FIT's first New York midnight (`quant.data.policy.FIT_START`, where every
   model's feature stream starts, `quant.jobs.first_model.training_data`) and before the
   decision bar, complete ones only, feed the stream; each warm row with finite values whose
   `available_at` is at or after `inputs.model.fit_start` steps the filter. So the filter runs
   over the model's own fit window up to the swap, as `ForwardFilter.warmed` does, and only
   ever sees the past. The switching state, the live trade, the shadows and the edge are kept:
   labels are names, inherited through B7's matching, so the active label carries over. An
   indicator stream that is missing, or whose config hash isn't its playbook's
   (`IndicatorConfig.for_playbook`), is warmed over the same bars. A model whose
   `feature_config_hash` isn't the release's feature config's is a MODEL failure (step 4).
3. **The decision bar**: `features_for` on the stream and `ForwardFilter.step` on its row; each
   trading playbook's `indicators_for`; and the latest visible bar
   (`view.hourly(SYMBOL, purpose=OPERATE, last=1)`), whose close is the decision's price
   (stale when the decision bar itself is missing, but the best a forced exit can use). Streams
   that refuse a bar keep the bars they consumed (`features_for`'s contract).
4. **Health, rule 6** (`quant.decide.health`): the first cause that holds, in this order:
   MISSED (`inputs.flags.missed`, C6), MODEL_PROVENANCE or CALIBRATION_PROVENANCE
   (`provenance_cause`), then `cause_of` the step-3 error of the features, the filter or the
   indicators (FEATURES or MODEL), MODEL for a feature-config mismatch. A healthy bar's
   probabilities are the filter's (`BarProbs`); an unhealthy bar has none.
5. **Switching** (`quant.decide.switching.step`) with engine.yaml's switching section and the
   caps' label caps. The label active before this step is `active_before`.
6. **Shadows and the edge** (C12, `quant.decide.edge`): on a bar whose decision bar and
   indicators are present, each trading playbook's shadow steps (`shadow_step`), and its return
   is added to its label's accumulator when that label is `active_before` and the shadow was
   held across the bar. Health and the flags don't stop the shadows.
7. **The live trade** under the active playbook (`quant.decide.primitives`): after a switch,
   the trade starts flat (`TradeState.flat()`; the old position is sized by the new label).
   Entries are allowed when switching allows them (`SwitchOutcome.entries_allowed`) and none of
   halted, the drift freeze and the daily-loss lock holds. With the indicators present it steps
   with `label_prob` the filtered P(active) (so invalidation applies); an INVALIDATED step
   blocks the playbook (`switching.invalidate`). A bar that isn't operating, or halted, closes
   an open trade (`close_trade`): its target is 0.
8. **Sizing** (`quant.decide.sizing.target`) with f* from the edge after step 6
   (`EdgeBook.f_star(active, min_bars=engine.sizing.min_edge_bars)`), P(active), the entropy
   and the multipliers; then `rebalance` from the current exposure (shares x price / equity)
   with the ceiling `quant.decide.intents.ceiling(decision, caps, active)`, `rule_forced` when
   rule 4, 5 or 6 fired, the flags, and `reduce_only` the decision's. Without a price nothing
   is sized: the target is 0 and no intent is made.
9. **Intents** (`quant.decide.intents.order`): at most one, from the desired exposure, with the
   live trade's stop on a BUY. When that stop isn't strictly between 0 and the price, the
   desired exposure is held at the current one (no increase without protection).
10. **The record** (`quant.decide.record.DecisionRecord`): every field from the steps above,
    `verdicts` empty, `state_before` = `state.state_hash()` and `state_after` the new state's.

**Errors.** Rule 6 failures are records, never exceptions. decide raises only for a broken
precondition (step 1) and for programming errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from quant.core.calibration import Calibration
from quant.core.model import StrictModel
from quant.core.orders import OrderIntent
from quant.core.schedule import Decision
from quant.data.view import PointInTimeView
from quant.decide.config import EngineConfig
from quant.decide.playbook import Release
from quant.decide.record import DecisionRecord
from quant.decide.sizing import Caps
from quant.decide.state import EngineState
from quant.hmm.model import HmmModel


class Account(StrictModel):
    """The account as the driver reads it before the decision: `equity` in dollars (> 0) and
    `position` in SPY shares (>= 0: long or flat)."""

    equity: float = Field(gt=0.0)
    position: int = Field(ge=0)


class Flags(StrictModel):
    """What the driver knows that decide can't compute (design call 2):

    - `halted`: the RiskEngine is halted (the kill switch): target 0, entries blocked;
    - `daily_loss_lock`: the daily loss limit tripped today: no entries or increases;
    - `drift_frozen`: a refit failed a drift check (C5): no entries or increases, exits and
      stops keep working;
    - `missed`: the engine was down at tau and replays the decision (C6): unhealthy, MISSED.
    """

    halted: bool = False
    daily_loss_lock: bool = False
    drift_frozen: bool = False
    missed: bool = False


@dataclass(frozen=True, eq=False)
class DecideInputs:
    """Everything decide reads besides its state and the view (design call 1):

    - `release`: the loaded release (playbooks and manifest);
    - `model`: the governing HmmModel (the ops.db pointer at tau);
    - `calibration`: the governing calibration, or None (every label then fails, C13);
    - `engine`: config/engine.yaml;
    - `caps`: risk.yaml's limits as data (`quant.decide.sizing.Caps`);
    - `decision`: the scheduled Decision (`quant.core.schedule.decision_schedule`);
    - `account`, `flags`: as above;
    - `code_commit`: the commit the engine runs, for the record.
    """

    release: Release
    model: HmmModel
    calibration: Calibration | None
    engine: EngineConfig
    caps: Caps
    decision: Decision
    account: Account
    flags: Flags
    code_commit: str


@dataclass(frozen=True, eq=False)
class DecisionOutcome:
    """decide's result: the new `state`, the `record` and the `intents` (at most one, also in
    the record)."""

    state: EngineState
    record: DecisionRecord
    intents: tuple[OrderIntent, ...]


def decide(state: EngineState, view: PointInTimeView, inputs: DecideInputs) -> DecisionOutcome:
    """One decision (module docstring). Pure: `state` is unchanged, and only `view` is read."""
    raise NotImplementedError
