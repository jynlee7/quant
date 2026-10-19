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

import copy
import math
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import Field

from quant.core.calendar import NEW_YORK
from quant.core.calibration import Calibration
from quant.core.errors import FeatureError, ModelError
from quant.core.model import StrictModel
from quant.core.orders import SYMBOL, OrderIntent
from quant.core.regime import Label
from quant.core.schedule import Decision
from quant.data.policy import FIT_START, Purpose
from quant.data.view import PointInTimeView
from quant.decide import edge as edge_mod
from quant.decide import intents as intents_mod
from quant.decide import sizing as sizing_mod
from quant.decide import switching
from quant.decide.config import EngineConfig
from quant.decide.health import Cause, cause_of, provenance_cause
from quant.decide.indicators import IndicatorConfig, IndicatorState, IndicatorStream, indicators_for
from quant.decide.playbook import Release
from quant.decide.primitives import Reason, TradeState, close_trade
from quant.decide.primitives import step as primitive_step
from quant.decide.record import DecisionRecord, FilterView, HealthView, RuleView, TargetView
from quant.decide.sizing import Caps
from quant.decide.state import EngineState
from quant.features.menu import FeatureConfig
from quant.features.stream import FeatureState, FeatureStream, features_for
from quant.hmm.filter import FilterState, ForwardFilter
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
    d = inputs.decision
    if view.tau != d.tau:
        raise ValueError("view.tau must be the decision's tau")
    release, model, engine, caps = inputs.release, inputs.model, inputs.engine, inputs.caps
    fconfig = release.manifest.features
    model_hash = _model_hash(model)
    trading = [label for label in Label if release.playbook(label).strategy != "flat"]
    iconfigs = {label: IndicatorConfig.for_playbook(release.playbook(label)) for label in trading}

    warm_error: BaseException | None = None
    need_warm = state.filter is None or state.filter.model_hash != model_hash
    streams: dict[Label, IndicatorStream] = {}
    warm_ind = []
    for label in trading:
        st = state.indicators.get(label)
        if st is None or st.config_hash != iconfigs[label].config_hash():
            streams[label] = IndicatorStream(iconfigs[label])
            warm_ind.append(label)
        else:
            streams[label] = _restore_ind(iconfigs[label], st)
    if need_warm:
        stream = FeatureStream(fconfig)
    else:
        if state.features is None or state.filter is None:
            raise ValueError("a warm filter requires feature and filter state")
        stream = _restore_stream(fconfig, state.features)
    filt: ForwardFilter | None = None
    try:
        filt = (
            _fresh_filter(model)
            if need_warm or state.filter is None
            else _restore_filter(model, state.filter)
        )
    except ModelError as exc:
        warm_error = exc
    if need_warm or warm_ind:
        bars = view.hourly(SYMBOL, purpose=Purpose.OPERATE, since=FIT_MIDNIGHT)
        for b in bars:
            if b.t_open >= d.bar.t_open:
                break
            if not b.complete:
                continue
            if need_warm:
                try:
                    row = stream.update(b)
                except FeatureError:
                    row = None
                if (
                    row is not None
                    and warm_error is None
                    and filt is not None
                    and row.warm
                    and all(math.isfinite(v) for v in row.values)
                    and row.available_at >= model.fit_start
                ):
                    try:
                        filt.step(np.array(row.values, dtype=np.float64), columns=fconfig.columns)
                    except ModelError as exc:
                        warm_error = exc
            for label in warm_ind:
                with suppress(FeatureError):
                    streams[label].update(b)

    cause: Cause | None = None
    if inputs.flags.missed:
        cause = Cause.MISSED
    if cause is None:
        cause = provenance_cause(
            d.tau,
            model_fit_end=model.fit_end,
            calibration_as_of=None if inputs.calibration is None else inputs.calibration.as_of,
        )
    features = None
    out = None
    error: BaseException | None = warm_error
    try:
        vector = features_for(stream, view, SYMBOL, d.bar)
        features = [float(x) for x in vector]
        if error is None and filt is not None:
            out = filt.step(vector, columns=fconfig.columns)
    except (FeatureError, ModelError) as exc:
        error = error or exc
    values = {}
    for label in trading:
        try:
            values[label] = indicators_for(streams[label], view, SYMBOL, d.bar)
        except FeatureError as exc:
            error = error or exc
    if model.feature_config_hash != fconfig.config_hash() and error is None:
        error = ModelError("model feature config differs from the release's")
    latest = view.hourly(SYMBOL, purpose=Purpose.OPERATE, last=1)
    bar = latest[-1] if latest else None
    price = None if bar is None else float(bar.close)
    present = bar is not None and bar.t_open == d.bar.t_open and bar.complete
    if cause is None and error is not None:
        cause = cause_of(error)
    if cause is None and out is None:
        cause = Cause.MODEL
    probs = None
    if cause is None and out is not None:
        probs = switching.BarProbs(out.label_probs, out.next_label_probs, out.entropy)

    active_before = state.switching.active
    sw = switching.step(
        state.switching, probs, cause=cause, config=engine.switching, caps=caps.label_caps
    )
    sw_state = sw.state

    # Shadows learn independently of live sizing, under the label held across the bar.
    book = state.edge
    shadows = dict(state.shadows)
    if present and bar is not None:
        for label in trading:
            if label not in values:
                continue
            sh = shadows.get(label) or edge_mod.ShadowState.cold()
            s = edge_mod.shadow_step(
                release.playbook(label),
                sh,
                values[label],
                bar,
                cost_bp=engine.sizing.shadow_cost_bp,
            )
            shadows[label] = s.state
            if s.ret is not None and s.held and active_before == label:
                book = book.add(label, s.ret)

    # A takeover starts the new primitive flat; invalidation blocks its next entry.
    flags = inputs.flags
    active = sw.active
    trade = state.trade if not sw.switched else TradeState.flat()
    if active is None:
        trade = TradeState.flat()
    reason: Reason | None = None
    invalidation = None
    blocked = sw.blocked
    if active is not None and present and bar is not None:
        pb = release.playbook(active)
        if pb.strategy == "flat":
            trade = TradeState.flat()
            reason = Reason.FLAT
        elif active in values and probs is not None:
            allowed = (
                sw.entries_allowed
                and not flags.halted
                and not flags.drift_frozen
                and not flags.daily_loss_lock
            )
            stepped = primitive_step(
                pb,
                trade,
                values[active],
                bar,
                entries_allowed=allowed,
                label_prob=probs.label_probs.get(active, 0.0),
            )
            trade, reason = stepped.state, stepped.reason
            if stepped.invalidation is not None:
                invalidation = stepped.invalidation.reason
            if reason is Reason.INVALIDATED:
                sw_state = switching.invalidate(sw_state)
                blocked = True
    if (not sw.health.operating or flags.halted) and trade.in_position:
        trade = close_trade(trade)

    # Keep the formula and the executable move separately in the record.
    f_star = 0.0 if active is None else book.f_star(active, min_bars=engine.sizing.min_edge_bars)
    p_active = None if probs is None or active is None else probs.label_probs.get(active, 0.0)
    tgt = sizing_mod.target(
        active,
        playbook=None if active is None else release.playbook(active),
        in_position=trade.in_position,
        operating=sw.health.operating,
        blocked=blocked,
        halted=flags.halted,
        caps=caps,
        calibration=inputs.calibration,
        f_star=f_star,
        label_prob=p_active,
        entropy=None if probs is None else probs.entropy,
        m_rule4=sw.m_rule4,
        m_rule5=sw.m_rule5,
        config=engine.sizing,
    )
    intents: tuple[OrderIntent, ...] = ()
    tv: dict[str, Any] = {
        "value": tgt.value,
        "zero": None if tgt.zero is None else tgt.zero.value,
        "sizing": None
        if tgt.sizing is None
        else {k: (v.value if isinstance(v, Label) else v) for k, v in asdict(tgt.sizing).items()},
        "current": None,
        "ceiling": None,
        "desired": None,
        "move": None,
        "limited": None,
        "forced": None,
        "held": None,
        "price": price,
    }
    if price is not None:
        account = inputs.account
        current = account.position * price / account.equity
        ceil = intents_mod.ceiling(d, caps, active)
        rule_forced = (
            not sw.health.operating
            or sw.fired(switching.RuleId.R4)
            or sw.fired(switching.RuleId.R5)
        )
        stop = trade.stop if trade.in_position else None
        tg = tgt.value
        if not (stop is not None and 0 < stop < price):
            tg = min(tg, current)
        rb = sizing_mod.rebalance(
            tg,
            current,
            ceiling=ceil,
            deadband=engine.sizing.deadband,
            rule_forced=rule_forced,
            halted=flags.halted,
            frozen=flags.drift_frozen,
            locked=flags.daily_loss_lock,
            reduce_only=d.reduce_only,
        )
        tv.update(
            current=current,
            ceiling=rb.ceiling,
            desired=rb.desired,
            move=rb.move,
            limited=rb.limited,
            forced=rb.forced,
            held=rb.held,
        )
        if rb.move:
            o = intents_mod.order(
                d,
                desired=rb.desired,
                shares=account.position,
                equity=account.equity,
                price=price,
                caps=caps,
                label=active,
                uncertain=sw.fired(switching.RuleId.R5),
                forced=rb.forced,
                stop=stop,
            )
            if o is not None:
                intents = (o,)

    new_state = EngineState(
        # Restore detached the histories; these local streams are never used again.
        features=stream._state,
        filter=None if warm_error is not None or filt is None else filt.snapshot(),
        indicators={label: streams[label]._state for label in trading},
        switching=sw_state,
        trade=trade,
        shadows=shadows,
        edge=book,
    )
    fv = None
    if out is not None:
        fv = FilterView(
            state_probs=[float(x) for x in out.state_probs],
            next_state_probs=[float(x) for x in out.next_state_probs],
            label_probs=dict(out.label_probs),
            next_label_probs=dict(out.next_label_probs),
            entropy=float(out.entropy),
            loglik=float(out.loglik),
            top_label=out.top_label,
        )
    cal = inputs.calibration
    record = DecisionRecord(
        decision_id=d.id,
        tau=d.tau,
        as_of=view.as_of,
        view_digest=view.digest(),
        release_id=release.release_id,
        model_hash=model_hash,
        model_fit_end=model.fit_end,
        calibration_hash=None if cal is None else _hash_of(cal),
        calibration_as_of=None if cal is None else cal.as_of,
        engine_hash=engine.config_hash(),
        caps_hash=caps.caps_hash(),
        code_commit=inputs.code_commit,
        health=HealthView(
            cause=None if sw.health.cause is None else sw.health.cause.value,
            streak=sw.health.streak,
            operating=sw.health.operating,
        ),
        features=features,
        filter=fv,
        active=active,
        switched=sw.switched,
        derisk=sw.derisk,
        blocked=blocked,
        trace=tuple(RuleView(rule=t.rule.value, fired=t.fired, reason=t.reason) for t in sw.trace),
        trade_reason=None if reason is None else reason.value,
        invalidation=invalidation,
        target=TargetView(**tv),
        intents=intents,
        verdicts=(),
        state_before=state.state_hash(),
        state_after=new_state.state_hash(),
    )
    return DecisionOutcome(new_state, record, intents)


FIT_MIDNIGHT = datetime.combine(FIT_START, time(), tzinfo=NEW_YORK).astimezone(UTC)


@lru_cache(maxsize=64)
def _model(model: HmmModel) -> ForwardFilter:
    """Memoize immutable model constants by identity; never step the template.

    HmmModel has identity equality and owns read-only arrays. The bounded cache
    retains the key, so neither object-id reuse nor model swaps can alias it.
    """
    return ForwardFilter(model)


def _fresh_filter(model: HmmModel) -> ForwardFilter:
    """Share model constants, but give each decision its own filter posterior."""
    template = _model(model)
    f = copy.copy(template)
    f._log_probs = None
    f._bars = 0
    return f


def _restore_filter(model: HmmModel, state: FilterState) -> ForwardFilter:
    """Restore a trusted EngineState snapshot without rebuilding model constants."""
    f = _fresh_filter(model)
    if state.log_probs is not None:
        f._log_probs = np.array(
            [float("-inf") if p is None else p for p in state.log_probs], dtype=np.float64
        )
    f._bars = state.bars
    return f


def _copy[T: (FeatureState, IndicatorState)](state: T) -> T:
    """Detach every mutable history in a trusted stream snapshot.

    Both schemas have scalar fields, lists, and dictionaries of scalars/lists.
    Copy those containers instead of revalidating or recursively copying scalars.
    """
    fields = {}
    for k in type(state).model_fields:
        v = getattr(state, k)
        if isinstance(v, list):
            v = list(v)
        elif isinstance(v, dict):
            v = {a: list(b) if isinstance(b, list) else b for a, b in v.items()}
        fields[k] = v
    return type(state).model_construct(**fields)


def _restore_stream(config: FeatureConfig, state: FeatureState) -> FeatureStream:
    s = FeatureStream(config)
    s._state = _copy(state)
    return s


def _restore_ind(config: IndicatorConfig, state: IndicatorState) -> IndicatorStream:
    s = IndicatorStream(config)
    s._state = _copy(state)
    return s


@lru_cache(maxsize=64)
def _hash_of(calibration: Calibration) -> str:
    """Memoize the immutable calibration artifact's digest by identity."""
    return calibration.artifact_hash()


@lru_cache(maxsize=64)
def _model_hash(model: HmmModel) -> str:
    """Keep provenance available even when the model cannot build a filter."""
    return model.artifact_hash()
