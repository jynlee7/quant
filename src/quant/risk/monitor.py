"""The minute monitor's logic (C8, spec §10, ARCHITECTURE §5 and §16).

Daily loss and drawdown can be breached between bars, so every minute of regular hours the live
engine (B14) and the backtest call `MinuteMonitor.tick` with the freshest price: the IEX
real-time trade live, the SIP 1-minute close in the backtest. That price feeds only the order
collar and this monitor, never a feature (C8). Each tick:

- marks equity on that price through `RiskEngine.mark`, which sets the daily-loss lock and fires
  the drawdown trigger;
- reconciles the broker with the engine's book (quant.risk.reconcile);
- checks that a broker-side stop covers the position (ARCHITECTURE §16).

It returns what the caller must do: flatten for the daily-loss lock, re-place a missing stop, or
run the kill executor for the triggers in `MonitorReport.kill`. It sends nothing, cancels
nothing and alerts no one itself; scheduling and acting are B14's. "Now" is the monitor's Clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from math import isfinite

import structlog

from quant.core.clock import Clock, ensure_utc
from quant.core.orders import BrokerSnapshot, InternalBook, OrderKind, Side
from quant.risk.engine import MarkOutcome, RiskEngine
from quant.risk.limits import RiskConfig
from quant.risk.reconcile import ReconcileStatus, ReconciliationResult, reconcile
from quant.risk.triggers import KillTrigger

log = structlog.get_logger()


@dataclass(frozen=True)
class MinuteInput:
    """What one tick reads.

    - `price`: the freshest SPY price (C8), dollars;
    - `price_at`: that price's own time, UTC;
    - `cash`: the account's cash balance at the broker, dollars (B13's adapter reports it; the
      simulated broker in a backtest). It may be negative; one that isn't a finite number is a
      data fault (MonitorReport.data_fault);
    - `snapshot`: the broker's position and working orders, read for this tick;
    - `book`: the engine's position and working orders.
    """

    price: float
    price_at: datetime
    cash: float
    snapshot: BrokerSnapshot
    book: InternalBook


@dataclass(frozen=True)
class MonitorReport:
    """What one tick found and what the caller must do.

    - `at`: the clock's time at the tick;
    - `equity`: `cash + snapshot.position * price`, or None when nothing was marked;
    - `mark`: what `RiskEngine.mark(equity)` returned, or None when nothing was marked;
    - `price_stale`: the price was too old, from the future, or not finite and positive, so
      nothing was marked;
    - `reconciliation`: the comparison of `snapshot` with `book`;
    - `stop_covered`: a broker-side stop covers the position; None when the snapshot is stale;
    - `stop_missing_since`: when the position was first seen uncovered, in the run of ticks
      that ends with this one; None while covered;
    - `replace_stop`: the caller must re-place the protective stop now;
    - `flatten`: this tick's mark set the daily-loss lock (`MarkOutcome.flatten`): the caller
      must take the position flat now, with entries locked until the next session;
    - `kill`: the triggers that fire on this tick, in the order DRAWDOWN, RECONCILIATION,
      STOP_MISSING; the caller runs the kill executor when it is not empty, unless one has
      already run for the halt in force;
    - `data_fault`: why this tick's cash or price can't be used (a cash that isn't a finite
      number, a price that isn't finite and positive), or None. Nothing is marked and no
      DRAWDOWN fires on it; the caller treats the engine as unhealthy (rule 6) and alerts;
    - `exit_covered`: no stop covers the position, but risk-approved exit SELLs working at the
      broker do (C5, your call of 2026-10-14): the stop isn't reported missing and its clock
      pauses;
    - `exit_since`: when exits first covered the position in this uncovered spell, while
      `exit_covered`; else None. STOP_MISSING fires once it is `kill.stop_missing_s` old;
    - `stop_missing_s`: the stop-missing clock, the seconds the position has gone with neither
      a stop nor an exit over it in this spell; None when a stop covers it or the snapshot is
      stale.
    """

    at: datetime
    equity: float | None
    mark: MarkOutcome | None
    price_stale: bool
    reconciliation: ReconciliationResult
    stop_covered: bool | None
    stop_missing_since: datetime | None
    replace_stop: bool
    flatten: bool
    kill: tuple[KillTrigger, ...]
    data_fault: str | None = None
    exit_covered: bool = False
    exit_since: datetime | None = None
    stop_missing_s: float | None = None


class MinuteMonitor:
    """The per-minute checks for one engine (C8). It remembers only when the stop went
    missing, so a restarted monitor starts that count again."""

    def __init__(
        self,
        engine: RiskEngine,
        config: RiskConfig,
        *,
        clock: Clock,
        instance_prefix: str | None = None,
    ) -> None:
        """A monitor that marks `engine`, with the limits in `config` (the engine's own),
        judging time by `clock`. `instance_prefix` goes to `reconcile`, so a broker order from
        another instance is a mismatch."""
        self._engine = engine
        self._config = config
        self._clock = clock
        self._instance_prefix = instance_prefix
        self._stop_missing_since: datetime | None = None

    @property
    def stop_missing_since(self) -> datetime | None:
        """When the position was first seen without a covering stop, or None (MonitorReport)."""
        return self._stop_missing_since

    def tick(self, minute: MinuteInput) -> MonitorReport:
        """One minute's checks at `now = clock.now()`, in this order:

        1. **Reconcile** `minute.snapshot` with `minute.book`
           (`reconcile(..., now=now, max_age_s=config.snapshot_max_age_s, instance_prefix)`).
           A MISMATCH adds RECONCILIATION to `kill`; STALE adds nothing.
        2. **Mark.** The price is fresh when it is finite and positive and
           `0 <= now - minute.price_at <= monitor_price_max_age_s` seconds (exactly the maximum
           is fresh). A cash that isn't a finite number, or a price that isn't finite and
           positive, is a `data_fault`: nothing is marked and nothing fires. When the price is
           fresh, the cash is good, the snapshot isn't STALE and the engine has started a
           session, `equity = minute.cash + minute.snapshot.position * minute.price`, summed in
           the decimal values the floats print (as `RiskEngine.mark` compares them, so exactly
           2% or 10% lands on the inclusive side), and the engine is marked with it
           (`RiskEngine.mark`, which publishes through `on_change`). `flatten` is the mark's
           `flatten`, and the mark's `kill` adds DRAWDOWN. An equity at or below 0 is not
           marked and adds DRAWDOWN: the account has lost everything the drawdown limit
           protects. Otherwise nothing is marked and the engine is untouched; `price_stale`
           says whether the price was the reason.
        3. **Stop.** Skipped (`stop_covered=None`, `stop_missing_since` unchanged, no
           `replace_stop`) when the snapshot is STALE. Otherwise the position is covered when
           `snapshot.position <= 0`, or the snapshot's open STOP SELL orders hold at least
           `snapshot.position` shares together. Covered clears `stop_missing_since`. Uncovered
           sets it to `now` if it was None, sets `replace_stop`, and adds STOP_MISSING to `kill`
           once `now - stop_missing_since >= kill.stop_missing_s`.

        RiskError only as `RiskEngine.mark` raises it, or from `on_change`.
        """
        now = self._clock.now()
        config = self._config
        snapshot = minute.snapshot
        fired: set[KillTrigger] = set()

        reconciliation = reconcile(
            snapshot,
            minute.book,
            now=now,
            max_age_s=config.snapshot_max_age_s,
            instance_prefix=self._instance_prefix,
        )
        stale_snapshot = reconciliation.status is ReconcileStatus.STALE
        if reconciliation.mismatch:
            fired.add(KillTrigger.RECONCILIATION)

        data_fault = _data_fault(minute)
        if data_fault is not None:
            log.error("monitor_data_fault", fault=data_fault)
        price_fresh = _fresh(minute.price, minute.price_at, now, config.monitor_price_max_age_s)
        equity: float | None = None
        mark: MarkOutcome | None = None
        if (
            data_fault is None
            and price_fresh
            and not stale_snapshot
            and self._engine.state.session is not None
        ):
            value = _exact(minute.cash) + snapshot.position * _exact(minute.price)
            if value > 0:
                equity = float(value)
                mark = self._engine.mark(equity)
                if mark.kill:
                    fired.add(KillTrigger.DRAWDOWN)
            else:
                log.error("monitor_equity_wiped_out", equity=str(value), price=minute.price)
                fired.add(KillTrigger.DRAWDOWN)

        stop_covered: bool | None = None
        replace_stop = False
        if not stale_snapshot:
            stop_covered = _covered(snapshot)
            if stop_covered:
                self._stop_missing_since = None
            else:
                if self._stop_missing_since is None:
                    self._stop_missing_since = now
                replace_stop = True
                missing_s = (now - self._stop_missing_since).total_seconds()
                if missing_s >= config.kill.stop_missing_s:
                    fired.add(KillTrigger.STOP_MISSING)

        report = MonitorReport(
            at=now,
            equity=equity,
            mark=mark,
            price_stale=not price_fresh,
            reconciliation=reconciliation,
            stop_covered=stop_covered,
            stop_missing_since=self._stop_missing_since,
            replace_stop=replace_stop,
            flatten=mark is not None and mark.flatten,
            kill=tuple(t for t in _KILL_ORDER if t in fired),
            data_fault=data_fault,
        )
        if report.kill or report.flatten or report.replace_stop or reconciliation.mismatch:
            log.warning(
                "monitor_tick",
                equity=equity,
                kill=[t.value for t in report.kill],
                flatten=report.flatten,
                replace_stop=replace_stop,
                reconciliation=reconciliation.status.value,
            )
        return report


_KILL_ORDER: tuple[KillTrigger, ...] = (
    KillTrigger.DRAWDOWN,
    KillTrigger.RECONCILIATION,
    KillTrigger.STOP_MISSING,
)
"""The order of `MonitorReport.kill`."""


def _finite(value: object) -> bool:
    """True when `value` is a finite int or float (not a bool)."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def _data_fault(minute: MinuteInput) -> str | None:
    """Why `minute`'s cash or price can't be used, or None (MonitorReport.data_fault)."""
    if not _finite(minute.cash):
        return f"cash is not a finite number: {minute.cash!r}"
    if not _finite(minute.price) or minute.price <= 0:
        return f"price is not finite and positive: {minute.price!r}"
    return None


def _exact(value: float) -> Decimal:
    """The decimal value `value` prints as (as quant.risk.engine compares limits)."""
    return Decimal(repr(value))


def _fresh(price: float, price_at: datetime, now: datetime, max_age_s: float) -> bool:
    """True when `price` is finite and positive, and no older than `max_age_s` nor from the
    future."""
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return False
    if not isfinite(price) or price <= 0:
        return False
    age = (now - ensure_utc(price_at)).total_seconds()
    return 0 <= age <= max_age_s


def _covered(snapshot: BrokerSnapshot) -> bool:
    """True when the position is flat or short, or its open STOP SELLs hold every share."""
    if snapshot.position <= 0:
        return True
    stops = sum(
        order.quantity
        for order in snapshot.open_orders
        if order.kind is OrderKind.STOP and order.side is Side.SELL
    )
    return stops >= snapshot.position
