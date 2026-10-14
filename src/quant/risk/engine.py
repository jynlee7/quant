"""RiskEngine.check and the approved types; the only place an approval is built (K5, I4).

ARCHITECTURE §4.4 and §8.1, spec §10. Every order, cancel and kill-switch close reaches the
broker only as a RiskApprovedOrder, RiskApprovedCancel or RiskApprovedFlatten. Each constructor
demands this module's private token, so only the engine can build one. The engine registers every
approval it issues and `spend` honours each exactly once: the broker adapter (B13) calls it right
before it sends anything, and persists spent ids through a SpentApprovals store so a restart
can't reuse one.

`check` runs ARCHITECTURE §8.1's seven checks in order, cheapest first, after a precondition on
the intent itself (RiskCheck.INTENT):

1. HALTED: only risk-reducing intents pass.
2. RECONCILIATION: a fresh broker snapshot must match the engine's book; a mismatch halts.
3. LIMITS: long or flat only, gross exposure, the label cap, the single-order cap, and the
   overnight cap on the last decision that can fill (A3).
4. LOCKS: the daily-loss lock, a drift freeze and rule 6 veto entries.
5. RATE: more than `max_intents_per_day` intents in a session halts trading (C3). Every
   intent counts, vetoed or not, and the one past the limit halts whichever check decides it.
6. PRICE: the limit sits inside the collar of a fresh quote, and every entry carries a stop.
7. APPROVAL (D7): a live BUY above the notional threshold waits for you; paper flags it.

In a long-or-flat system a SELL is exactly a risk-reducing intent. It skips the caps in 3, the
locks in 4 and the approval in 7, and is clamped so it can never take the position short.

An approved order is still risk until it is sent, so the engine keeps every one it issued as
outstanding until it is spent, released, revoked or expired. Outstanding BUYs count as held in
check 3's caps, and outstanding SELLs reserve their shares, so approvals issued together can
never add up to a breach or a short. A spent order counts the same way until a broker snapshot
taken after the spend can show it. The engine revokes outstanding approvals when what it learns
makes them unsafe: a re-check of the same decision and leg replaces the earlier approval, a
block on entries (HALTED, the daily-loss lock, a drift freeze, rule 6) revokes the BUYs, a
snapshot holding fewer shares than the SELLs reserve revokes the SELLs, and a spent flatten
revokes everything. `spend` refuses a revoked approval.

Monetary limits are compared on the decimal values of the floats (`Decimal(repr(x))`), so an
inclusive boundary such as exactly 2%, 10%, a cap or 10 bp holds however the float arithmetic
would round.

The engine also keeps C3's counters, the daily-loss lock and the drawdown trigger (`mark`, fed
by the minute monitor, C8). Its state is a RiskState the live engine persists with EngineState;
every change reaches the engine's `on_change` before the call that made it returns, so B14 can
persist it first. Every halt is signalled where it happens: `Veto.halts`,
`RiskApprovedOrder.halts`, `MarkOutcome.kill`, SubmissionLimitError, and the published state.
A halt keeps its first reason, and only `resume` (B14's `quant resume`) clears it.
The kill executor (quant.risk.kill), reconciliation (quant.risk.reconcile), the minute monitor
(quant.risk.monitor), the kill triggers (quant.risk.triggers) and the broker port
(quant.risk.broker) are B9b's. "Now" comes from the engine's Clock. Every veto and halt is
logged with structlog.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Any, Final, Protocol, cast, final

import structlog
from pydantic import Field

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.canonical import sha256_hex
from quant.core.clock import Clock
from quant.core.epoch import to_us
from quant.core.errors import (
    ApprovalError,
    CalendarError,
    QuantError,
    RiskError,
    SubmissionLimitError,
)
from quant.core.model import StrictModel
from quant.core.orders import (
    SYMBOL,
    BrokerSnapshot,
    InternalBook,
    OrderIntent,
    OrderKind,
    Quote,
    Side,
    TradingMode,
)
from quant.core.schedule import Timing, decision_schedule
from quant.risk.limits import RiskConfig

log = structlog.get_logger()

_TOKEN: Final = object()
"""The private token every approved type's constructor demands. It never leaves this module."""

INTENT_LIMIT: Final = "daily intent limit exceeded"
SUBMISSION_LIMIT: Final = "raw submission limit reached"
DRAWDOWN: Final = "maximum drawdown reached"
"""The halt reasons the engine's own counters and trigger give (`RiskState.halt_reason`)."""


class RiskCheck(StrEnum):
    """Where a verdict was decided, in the order `check` runs them."""

    INTENT = "intent"  # the precondition: a scheduled decision, inside its window, this session
    HALTED = "halted"  # 1
    RECONCILIATION = "reconciliation"  # 2
    LIMITS = "limits"  # 3
    LOCKS = "locks"  # 4
    RATE = "rate"  # 5
    PRICE = "price"  # 6
    APPROVAL = "approval"  # 7


CHECK_ORDER: Final = tuple(RiskCheck)


# --- the approved types (K5, I4) ------------------------------------------------------------------


class _Approved:
    """What the three approved types share. Each instance is immutable: setting or deleting an
    attribute raises AttributeError, and `copy.copy` and `copy.deepcopy` raise TypeError (so does
    pickling, which the project never does). Equality and hashing are by identity. A repr never
    shows the token.

    - `approval_id`: unique among the approvals one engine issues (see `RiskEngine`);
    - `issued_at`: the engine's clock when it issued the approval.
    """

    __slots__ = ("_token", "approval_id", "issued_at")

    approval_id: str
    issued_at: datetime
    _token: object

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("approvals are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("approvals are immutable")

    def __copy__(self) -> _Approved:
        raise TypeError("approvals cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> _Approved:
        raise TypeError("approvals cannot be copied")

    def __reduce_ex__(self, protocol: object) -> tuple[object, ...]:
        raise TypeError("approvals cannot be pickled")


@final
class RiskApprovedOrder(_Approved):
    """An order the risk engine approved: the only order the broker adapter sends (I4).

    The constructor raises ApprovalError unless `token` is this module's private token.

    - `intent`: the intent as checked;
    - `quantity`: shares to send: the intent's, or for a SELL clamped to what may be sold;
    - `limit_price`: the collared limit that check 6 accepted;
    - `notional`: `quantity * limit_price`, dollars;
    - `expires_at`: the end of the bar in which the decision executes; `spend` refuses it from
      then on;
    - `flagged`: a BUY above D7's threshold, approved because the engine trades paper, or one
      you approved (`approve_pending`). The engine alerts on it;
    - `halts`: this check halted trading (a risk-reducing intent past C3's intent limit). The
      caller persists HALTED and alerts, as for a Veto with `halts`.
    """

    __slots__ = ("expires_at", "flagged", "halts", "intent", "limit_price", "notional", "quantity")

    intent: OrderIntent
    quantity: int
    limit_price: float
    notional: float
    expires_at: datetime
    flagged: bool
    halts: bool

    def __init__(
        self,
        *,
        token: object,
        approval_id: str,
        intent: OrderIntent,
        quantity: int,
        limit_price: float,
        notional: float,
        issued_at: datetime,
        expires_at: datetime,
        flagged: bool,
        halts: bool,
    ) -> None:
        _initialize(
            self,
            token,
            {
                "approval_id": approval_id,
                "intent": intent,
                "quantity": quantity,
                "limit_price": limit_price,
                "notional": notional,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "flagged": flagged,
                "halts": halts,
            },
        )

    def __repr__(self) -> str:
        """`RiskApprovedOrder(<approval_id>, <side> <quantity> @ <limit_price>, until <expires_at>)`."""
        return (
            f"RiskApprovedOrder({self.approval_id}, {self.intent.side} {self.quantity} "
            f"@ {self.limit_price}, until {self.expires_at})"
        )


@final
class RiskApprovedCancel(_Approved):
    """Leave to cancel one working order. Cancels only reduce risk, so the engine always issues
    them, halted or not. They don't expire; each is spendable once.

    The constructor raises ApprovalError unless `token` is this module's private token.
    """

    __slots__ = ("client_order_id", "reason")

    client_order_id: str
    reason: str

    def __init__(
        self,
        *,
        token: object,
        approval_id: str,
        client_order_id: str,
        reason: str,
        issued_at: datetime,
    ) -> None:
        _initialize(
            self,
            token,
            {
                "approval_id": approval_id,
                "client_order_id": client_order_id,
                "reason": reason,
                "issued_at": issued_at,
            },
        )

    def __repr__(self) -> str:
        """`RiskApprovedCancel(<approval_id>, <client_order_id>)`."""
        return f"RiskApprovedCancel({self.approval_id}, {self.client_order_id})"


@final
class RiskApprovedFlatten(_Approved):
    """Leave for the kill switch to close the whole position in `symbol` (ARCHITECTURE §8.3).
    It only reduces risk, so the engine always issues it, halted or not. It doesn't expire; it is
    spendable once.

    The constructor raises ApprovalError unless `token` is this module's private token.
    """

    __slots__ = ("reason", "symbol")

    symbol: str
    reason: str

    def __init__(
        self,
        *,
        token: object,
        approval_id: str,
        symbol: str,
        reason: str,
        issued_at: datetime,
    ) -> None:
        _initialize(
            self,
            token,
            {
                "approval_id": approval_id,
                "symbol": symbol,
                "reason": reason,
                "issued_at": issued_at,
            },
        )

    def __repr__(self) -> str:
        """`RiskApprovedFlatten(<approval_id>, <symbol>)`."""
        return f"RiskApprovedFlatten({self.approval_id}, {self.symbol})"


Approval = RiskApprovedOrder | RiskApprovedCancel | RiskApprovedFlatten


# --- verdicts and context -------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PendingApproval:
    """A live BUY above D7's threshold, waiting for you (ARCHITECTURE §8.4: CLI or dashboard).
    It is not an approval: the adapter refuses it. `RiskEngine.approve_pending` turns it into
    one, once, before `expires_at` (the end of the bar). Equality is by identity.

    `quantity`, `limit_price` and `notional` are as check 7 saw them; `pending_id` is unique
    among the ids the engine issues.
    """

    pending_id: str
    intent: OrderIntent
    quantity: int
    limit_price: float
    notional: float
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class Veto:
    """An intent the engine refused. `check` names the first check that failed and `reason`
    says why in plain words. `halts` is True when this verdict halted trading (a reconciliation
    mismatch, a short position, or the intent past C3's limit, whichever check refused it); the
    caller then persists HALTED and alerts."""

    intent: OrderIntent
    check: RiskCheck
    reason: str
    halts: bool = False


RiskVerdict = RiskApprovedOrder | PendingApproval | Veto


@dataclass(frozen=True)
class RiskContext:
    """What one check reads besides the intent (ARCHITECTURE §4.2 step 11: pure, given a fresh
    broker snapshot).

    - `broker`: the broker's answer, no older than `snapshot_max_age_s`;
    - `book`: what the engine believes it holds;
    - `quote`: the freshest quote, no older than `quote_max_age_s`;
    - `limit_price`: the limit execution would send (ARCHITECTURE §8.2: ask + 10 bp to buy,
      bid - 10 bp to sell); also the price every exposure is valued at;
    - `healthy`: False while rule 6 holds (spec §8);
    - `drift_frozen`: entries frozen after a failed drift check (C5);
    - `halted`: the HALTED flag in control.db, as the engine last read it.
    """

    broker: BrokerSnapshot
    book: InternalBook
    quote: Quote
    limit_price: float
    healthy: bool = True
    drift_frozen: bool = False
    halted: bool = False


class RiskState(StrictModel):
    """The risk engine's state, persisted with EngineState (ARCHITECTURE §4.3) and restored by
    passing it to RiskEngine.

    - `session`: the session started last (`start_session`), or None before the first;
    - `start_equity`: that session's start-of-day equity;
    - `peak_equity`: the highest equity seen across sessions since the last resume, which
      re-bases it to the equity it is given;
    - `intents`: every intent checked this session, vetoed or not, as `"<decision_id>#<leg>"`
      in the order first seen;
    - `submissions`: raw broker submissions this session: RiskApprovedOrders spent;
    - `daily_loss_locked`: entries locked until the next session (spec §10);
    - `halted`, `halt_reason`: trading halted by this engine, and the first reason it halted
      for (a later halt keeps it). Only `RiskEngine.resume` clears them;
    - `killed`: the drawdown trigger has fired. Only `RiskEngine.resume` clears it;
    - `issued`: approvals and pending approvals issued so far, which numbers the next id.
    """

    session: date | None = None
    start_equity: float | None = Field(default=None, gt=0.0)
    peak_equity: float | None = Field(default=None, gt=0.0)
    intents: tuple[str, ...] = ()
    submissions: int = Field(default=0, ge=0)
    daily_loss_locked: bool = False
    halted: bool = False
    halt_reason: str | None = None
    killed: bool = False
    issued: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class MarkOutcome:
    """What one equity mark found (`RiskEngine.mark`).

    - `day_loss`: `(start_equity - equity) / start_equity` (negative on a gain);
    - `drawdown`: `(peak_equity - equity) / peak_equity`, after the mark raised the peak;
    - `locked`: the daily-loss lock holds after this mark;
    - `flatten`: this mark set the lock, so the monitor must flatten now;
    - `kill`: this mark fired the drawdown trigger, so the kill switch must run now. It fires
      once; the engine is halted from then on, until a resume.
    """

    equity: float
    day_loss: float
    drawdown: float
    locked: bool
    flatten: bool
    kill: bool


@dataclass(frozen=True)
class ResumeOutcome:
    """What one resume did (`RiskEngine.resume`), so B14 can log and alert the re-base.

    - `equity`: the equity the resume was given;
    - `old_peak`: the peak the drawdown was measured from before the resume;
    - `new_peak`: the peak it is measured from now, which is `equity`.
    """

    equity: float
    old_peak: float
    new_peak: float


class SpentApprovals(Protocol):
    """Where spent approval ids are kept. The adapter (B13) uses one backed by ops.db, so a
    restart can't spend an id twice (ARCHITECTURE §4.4)."""

    def is_spent(self, approval_id: str) -> bool:
        """True once `approval_id` has been spent."""
        ...

    def spend(self, approval_id: str, at: datetime) -> bool:
        """Record `approval_id` as spent at `at`, durably, before returning. True if it was not
        spent before; False, changing nothing, if it was."""
        ...


class MemorySpentApprovals:
    """SpentApprovals in memory, for the backtest and tests."""

    def __init__(self) -> None:
        self._spent: dict[str, datetime] = {}

    def is_spent(self, approval_id: str) -> bool:
        return approval_id in self._spent

    def spend(self, approval_id: str, at: datetime) -> bool:
        if self.is_spent(approval_id):
            return False
        self._spent[approval_id] = at
        return True


# --- the engine -----------------------------------------------------------------------------------


class RiskEngine:
    """The single choke point (spec §10, I4).

    Approval ids are `f"{prefix}{n}-{to_us(issued_at)}"` (quant.core.epoch.to_us), where the
    prefix is O for an order, C a cancel, F a flatten and P a pending approval, and n is
    `state.issued` after counting this one. Ids are therefore deterministic in a replay, and
    unique across restarts as long as the engine restarts from its persisted RiskState.

    Outstanding approvals (see the module docstring) live in the engine object, not in
    RiskState: a restarted engine never honours an earlier object's approvals, so it has none.
    The engine lets go of an order approval once it expires, of a pending approval once it
    expires or is approved, and of a cancel or flatten once it is spent.

    **Persisting (C7).** Every public method that changes RiskState calls `on_change` with the
    new state once, before it returns or raises: `start_session`, `mark`, `resume`, `halt`,
    `check`, `approve_pending`, `approve_cancel`, `approve_flatten` and `spend`. The live engine (B14)
    persists the state there, so a crash after a call never restores a state from before it;
    `spend` persists the submission before the adapter sends. An exception from `on_change`
    propagates out of the method (`check` included), so nothing is sent on a state that wasn't
    saved; the change stays in the engine's memory.
    """

    def __init__(
        self,
        config: RiskConfig,
        *,
        clock: Clock,
        mode: TradingMode,
        calendar: TradingCalendar | None = None,
        timing: Timing | None = None,
        spent: SpentApprovals | None = None,
        state: RiskState | None = None,
        on_change: Callable[[RiskState], None] | None = None,
    ) -> None:
        """An engine with `config`'s limits, trading in `mode`.

        `calendar` defaults to `TradingCalendar.xnys()`, `timing` to `Timing()`, `spent` to a new
        MemorySpentApprovals, and `state` to `RiskState()`. `on_change` receives every new state
        (see the class docstring); None persists nothing. The engine starts with no approvals
        issued: approvals from an earlier engine object are never honoured, even with the same
        `spent` store and state."""
        self._config = config
        self._clock = clock
        self._mode = mode
        self._calendar = TradingCalendar.xnys() if calendar is None else calendar
        self._timing = Timing() if timing is None else timing
        self._spent = MemorySpentApprovals() if spent is None else spent
        self._state = RiskState() if state is None else state
        self._issued: dict[Approval, _Issued] = {}
        self._pending: dict[PendingApproval, tuple[object, ...]] = {}
        self._outstanding: dict[str, _Hold] = {}  # by "<decision_id>#<leg>"
        self._in_flight: list[_InFlight] = []
        self._revoked: dict[Approval, str] = {}
        self._on_change = on_change

    @property
    def state(self) -> RiskState:
        """The current state, as an immutable snapshot."""
        return self._state

    def start_session(self, day: date, *, equity: float) -> None:
        """Begin session `day` with start-of-day equity `equity`: the peak becomes the larger of
        the old peak and `equity`, the intent and submission counters go to zero, and the
        daily-loss lock clears. HALTED and the drawdown trigger stay as they are.

        RiskError unless `day` is a session of the calendar later than `state.session`, and
        `equity` is finite and positive."""
        _check_equity(equity)
        try:
            self._calendar.session(day)
        except CalendarError as exc:
            raise RiskError(str(exc)) from exc
        if self.state.session is not None and day <= self.state.session:
            raise RiskError("a new session must be later than the current session")
        with self._publishing():
            self._forget_expired(self._clock.now())
            self._update(
                session=day,
                start_equity=float(equity),
                peak_equity=max(self.state.peak_equity or equity, equity),
                intents=(),
                submissions=0,
                daily_loss_locked=False,
            )

    def resume(self, *, equity: float) -> ResumeOutcome:
        """Clear the engine's halt and the drawdown trigger, and re-base the peak (C8). Only
        B14's `quant resume` calls it, after your typed confirmation and a clean reconciliation
        (ARCHITECTURE §8.3); B14 logs and alerts the outcome.

        `halted`, `halt_reason` and `killed` clear, and the peak becomes `equity`, higher or
        lower than before (your call, 2026-10-13): after a drawdown kill the account is flat and
        equity can't recover, so the next 10% drawdown is measured from where you resumed.
        Nothing else changes: the session's intent and submission counts and the daily-loss lock
        (until the next session) stay, so a counter already past its limit halts again at the
        next new intent or submission. Approvals revoked by the halt stay revoked.

        RiskError, changing nothing, when no session has started, `equity` isn't finite and
        positive, or trading isn't halted (a resume with nothing to clear would only lower the
        peak)."""
        _check_equity(equity)
        old_peak = self.state.peak_equity
        if self.state.session is None or old_peak is None:
            raise RiskError("a resume needs a started session")
        if not self.state.halted:
            raise RiskError("nothing to resume: trading isn't halted")
        outcome = ResumeOutcome(float(equity), old_peak, float(equity))
        with self._publishing():
            self._update(peak_equity=outcome.new_peak, halted=False, halt_reason=None, killed=False)
        log.warning(
            "risk_resumed",
            equity=outcome.equity,
            old_peak=outcome.old_peak,
            new_peak=outcome.new_peak,
        )
        return outcome

    def halt(self, reason: str) -> bool:
        """Halt trading for `reason`: the kill executor's first step (ARCHITECTURE §8.3), for
        every trigger, manual or automatic (quant.risk.triggers.KillTrigger).

        Sets `halted`; `halt_reason` becomes `reason` unless trading is already halted, when the
        first reason stays (C10). Revokes every outstanding BUY approval, as any halt does, and
        leaves SELLs, cancels and flattens alone. `killed` (the drawdown trigger) is unchanged.
        Works with or without a started session. The new state reaches `on_change` before
        `halt` returns (C7), and the halt is logged like every other.

        Returns True when this call halted trading, False when it was already halted (the state
        is then unchanged and `on_change` isn't called). RiskError, changing nothing, when
        `reason` is not a non-empty string."""
        raise NotImplementedError("B9b: RiskEngine.halt")

    def mark(self, equity: float) -> MarkOutcome:
        """One equity mark from the minute monitor, on the freshest price (C8), or from a broker
        snapshot. The peak rises to `equity` if it is higher.

        - Daily loss: when `start_equity - equity >= daily_loss_fraction * start_equity`, the
          daily-loss lock is set until the next session (exactly 2% locks), and the outstanding
          BUY approvals are revoked.
        - Drawdown: when `peak_equity - equity >= max_drawdown_fraction * peak_equity`, the
          engine halts (`halt_reason` names the drawdown) and the trigger fires (exactly 10%
          fires).

        Both compare decimal values (module docstring), so exactly 2% or 10% of any equity in
        cents locks or fires.

        RiskError when no session has started, or `equity` isn't finite and positive."""
        _check_equity(equity)
        start = self.state.start_equity
        if self.state.session is None or start is None:
            raise RiskError("an equity mark needs a started session")
        peak = max(self.state.peak_equity or equity, equity)
        daily_breach = _fell_by(start, equity, self._config.daily_loss_fraction)
        drawdown_breach = _fell_by(peak, equity, self._config.max_drawdown_fraction)
        flatten = daily_breach and not self.state.daily_loss_locked
        kill = drawdown_breach and not self.state.killed
        with self._publishing():
            self._update(
                peak_equity=float(peak),
                daily_loss_locked=self.state.daily_loss_locked or daily_breach,
                killed=self.state.killed or kill,
            )
            if flatten:
                log.warning("risk_daily_loss_lock", equity=equity, start_equity=start)
                self._revoke_side(Side.BUY, "the daily-loss lock set")
            if kill:
                self._halt(DRAWDOWN)
        return MarkOutcome(
            float(equity),
            (start - equity) / start,
            (peak - equity) / peak,
            self.state.daily_loss_locked,
            flatten,
            kill,
        )

    def check(self, intent: OrderIntent, context: RiskContext) -> RiskVerdict:
        """The verdict on `intent`: RiskApprovedOrder, Veto or PendingApproval. The first check
        that fails decides; nothing after it runs. "now" is the engine's clock. Exposure is
        always `shares * context.limit_price / context.broker.equity`.

        **Intent** (RiskCheck.INTENT). `intent.tau` must be the tau of a decision in
        `decision_schedule(calendar, <tau's New York date>, timing)`; call it `d`. Its window
        runs from `d.executes_at` to `expires_at`, the `t_close` of the bar of `d.executes_at`'s
        session with `t_open <= d.executes_at < t_close`. So an intent at 10:46 ET expires at
        11:30, one at the last fillable decision at the close, and a closing-bar intent at the
        end of the next session's first bar. Vetoed unless `d.executes_at <= now < expires_at`
        (an intent is never sent late, ARCHITECTURE §8.2) and `state.session` is the session of
        `d.executes_at`.

        Before the precondition, outstanding approvals that have expired are dropped, a
        re-check revokes the earlier approval of the same `"<decision_id>#<leg>"` whatever its
        verdict, and a context with `halted`, `drift_frozen` or not `healthy` revokes every
        outstanding BUY. Then, once a session has started, the intent's key is counted toward
        C3's limit the first time the engine sees it, whatever the verdict turns out to be;
        re-checks of a counted key (re-quotes) don't count again. The key that makes
        `len(state.intents)` exceed `max_intents_per_day` while the engine isn't halted is
        *over the limit*: its verdict halts trading, whichever check decides it. A Veto then has
        `halts=True` (the first check that fails still names it), and check 5 below decides
        only when nothing before it fails.

        1. **HALTED.** When `state.halted` or `context.halted`, a BUY is vetoed.
        2. **Reconciliation.** Vetoed when the snapshot is older than `snapshot_max_age_s`
           (`now - broker.taken_at > max`; exactly the maximum passes) or from the future.
           Vetoed, halting trading, when `broker.position != book.position`, when the open
           orders differ (compared as multisets of OpenOrder), or when `broker.position < 0`.
           Nothing is ever corrected.
           In flight below means spent orders the snapshot can't show yet: those spent at or
           after `broker.taken_at`.
        3. **Limits.** Let `sellable = broker.position` minus the open SELL LIMIT orders'
           quantity and the in-flight SELLs' (stops don't count: execution cancels the stop
           before it sells, ARCHITECTURE §8.2). When the outstanding SELLs reserve more than
           `sellable`, they are all revoked.
           - SELL: clamped to `sellable` minus the outstanding SELLs' quantity. Vetoed when
             nothing is left to sell. No cap applies.
           - BUY: vetoed on a decision that can only reduce (`d.reduce_only`). Otherwise let
             `held = broker.position + open BUY LIMIT quantity + in-flight BUYs + outstanding
             BUYs + intent.quantity`; vetoed unless `held * limit_price <= cap * equity`, where
             cap is the smallest of
             `max_gross_exposure`, the label's cap (`uncertain_cap` when `intent.uncertain` or
             `intent.label is None`) and, on the last fillable decision (`d.last_fillable`),
             `overnight_cap`; and vetoed unless
             `intent.quantity * limit_price <= single_order_cap * equity`. Equality passes, in
             decimal values.
        4. **Locks.** A BUY is vetoed while `state.daily_loss_locked`, `context.drift_frozen`,
           or not `context.healthy` (rule 6).
        5. **Rate.** An intent over the limit halts trading: a BUY is vetoed (`halts=True`),
           and a SELL goes on as it would under HALTED, its approval carrying `halts=True`.
        6. **Price.** Vetoed when the quote is older than `quote_max_age_s` (exactly the maximum
           passes) or from the future, when it is crossed (`bid > ask`), or unless
           `abs(limit_price - ref) <= ref * collar_bp / 10_000` in decimal values, with `ref`
           the ask for a BUY and the bid for a SELL. The collar binds SELLs too: spec §10 gives
           discretionary exits the same ±10 bp, and only the kill switch's exits widen, through
           a RiskApprovedFlatten that no collar checks. A BUY is vetoed unless
           `intent.stop_price` is set and below `limit_price`.
        7. **Approval (D7).** A BUY whose notional `intent.quantity * limit_price` exceeds
           `manual_approval_notional` (exactly the threshold doesn't) returns a registered
           PendingApproval when `mode` is LIVE, and a RiskApprovedOrder with `flagged=True`
           when it is PAPER. Approving a PendingApproval vetoes a price moved more than the
           collar above the one you were asked about (`approve_pending`).

        A pass returns a registered RiskApprovedOrder, expiring at the window's end, and
        outstanding from then on. Checking never raises for a bad intent or context: every
        refusal is a Veto. Only `on_change` can make it raise (class docstring).
        """
        with self._publishing():
            return self._check(intent, context, pending=None)

    def approve_pending(self, pending: PendingApproval, context: RiskContext) -> RiskVerdict:
        """Your approval of `pending` (D7, ARCHITECTURE §8.4): checks 1–6 run again on its intent
        with `context`, and a pass returns a RiskApprovedOrder with `flagged=True`. A Veto is
        returned as from `check`. The intent is not counted again. A PendingApproval holds no
        exposure, so approving it counts every approval outstanding by then.

        You approved the order at `pending.limit_price`, so check 7 vetoes it when
        `context.limit_price` is more than `collar_bp` above that price, in decimal values
        (exactly the collar passes; a lower price always does). The pending approval stays
        approvable until it expires, at a price that passes.

        ApprovalError when this engine object didn't issue `pending` (judged by identity), it
        was already approved, or `now >= pending.expires_at`."""
        if type(pending) is not PendingApproval:
            raise ApprovalError("pending must be exactly a PendingApproval")
        issued = self._pending.get(pending)
        if issued is None or not _unchanged(_pending_fields(pending), issued):
            raise ApprovalError("pending approval was not issued here or has changed")
        if self._clock.now() >= pending.expires_at:
            raise ApprovalError("pending approval has expired")
        with self._publishing():
            verdict = self._check(pending.intent, context, pending=pending)
        if isinstance(verdict, RiskApprovedOrder):
            del self._pending[pending]
        return verdict

    def approve_cancel(self, client_order_id: str, *, reason: str) -> RiskApprovedCancel:
        """A registered approval to cancel `client_order_id`, issued whatever the state."""
        now = self._clock.now()
        with self._publishing():
            approval = RiskApprovedCancel(
                token=_TOKEN,
                approval_id=self._next_id("C", now),
                client_order_id=client_order_id,
                reason=reason,
                issued_at=now,
            )
            self._issued[approval] = _Issued(_approval_fields(approval))
        return approval

    def approve_flatten(self, *, reason: str) -> RiskApprovedFlatten:
        """A registered approval for the kill switch to close the whole SPY position, issued
        whatever the state."""
        now = self._clock.now()
        with self._publishing():
            approval = RiskApprovedFlatten(
                token=_TOKEN,
                approval_id=self._next_id("F", now),
                symbol=SYMBOL,
                reason=reason,
                issued_at=now,
            )
            self._issued[approval] = _Issued(_approval_fields(approval))
        return approval

    def release(self, approval: RiskApprovedOrder) -> None:
        """Give up `approval` unsent: execution calls this when it drops an approved order
        without sending it. The approval stops holding exposure or shares, and `spend` refuses
        it from then on. Releasing one already spent, released, revoked or expired changes
        nothing; an expired one isn't even looked up, since the engine lets go of it.

        ApprovalError, as in `spend` cases 1–3, when `approval` is not exactly a
        RiskApprovedOrder this engine object issued."""
        if type(approval) is not RiskApprovedOrder:
            raise ApprovalError("only an approved order can be released")
        if getattr(approval, "_token", None) is not _TOKEN:
            raise ApprovalError("approval is missing the risk engine's private token")
        if self._clock.now() >= approval.expires_at:
            return
        issued = self._registered(approval)
        hold = self._outstanding.get(issued.key)
        if hold is not None and hold.approval is approval:
            self._revoke(issued.key, "released")

    def spend(self, approval: object) -> Approval:
        """Spend `approval` once; the broker adapter calls this right before it sends anything,
        and sends nothing unless it returns.

        ApprovalError, in this order, for:
        1. anything whose type is not exactly RiskApprovedOrder, RiskApprovedCancel or
           RiskApprovedFlatten (plain objects, subclasses);
        2. a missing token, or one that isn't this module's;
        3. an approval this engine object never issued (judged by identity: copies, one built
           with the token by anyone else, another engine's), one any of whose fields is not
           the very object issued, or a RiskApprovedOrder whose intent's canonical hash differs
           from the one taken at issue (an intent changed in place);
        4. a RiskApprovedOrder at or after its `expires_at`;
        5. an id the SpentApprovals store already holds (reused);
        6. a RiskApprovedOrder that was released or revoked, so no longer outstanding;
        7. SubmissionLimitError, halting trading, for a BUY when the session already has
           `submission_halt_at` submissions (reachable only after a resume, since reaching the
           number halts); ApprovalError for a BUY while `state.halted`, `state.killed` or
           `state.daily_loss_locked`. A SELL is never refused here.

        Then the id is recorded in the store (`spend` returning False is case 5). A
        RiskApprovedOrder counts as one raw submission (C3) and is in flight from now on; the
        submission that brings the count to `submission_halt_at` or past it is sent and halts
        trading (C3 halts *at* 60, so the 60th goes out, BUY or SELL). A RiskApprovedFlatten
        revokes every outstanding order approval. Cancels and flattens never count as
        submissions, and the engine lets go of them once spent. Every change is published
        through `on_change` before `spend` returns or raises.
        """
        if type(approval) not in (RiskApprovedOrder, RiskApprovedCancel, RiskApprovedFlatten):
            raise ApprovalError("expected exactly an approved order, cancel or flatten")
        approved = cast(Approval, approval)
        issued = self._registered(approved)
        now = self._clock.now()
        if isinstance(approved, RiskApprovedOrder) and now >= approved.expires_at:
            raise ApprovalError("order approval has expired")
        if self._spent.is_spent(approved.approval_id):
            raise ApprovalError("approval has already been spent")
        with self._publishing():
            limit = self._config.submission_halt_at
            if isinstance(approved, RiskApprovedOrder):
                hold = self._outstanding.get(issued.key)
                if hold is None or hold.approval is not approved:
                    reason = self._revoked.get(approved, "it is no longer outstanding")
                    raise ApprovalError(f"order approval was revoked: {reason}")
                if hold.side is Side.BUY:
                    if self.state.submissions >= limit:
                        self._halt(SUBMISSION_LIMIT)
                        raise SubmissionLimitError(SUBMISSION_LIMIT)
                    if self.state.halted or self.state.killed or self.state.daily_loss_locked:
                        raise ApprovalError("entries are halted or locked")
            if not self._spent.spend(approved.approval_id, now):
                raise ApprovalError("approval has already been spent")
            if isinstance(approved, RiskApprovedOrder):
                hold = self._outstanding.pop(issued.key)
                self._in_flight.append(_InFlight(hold.side, hold.quantity, now))
                self._update(submissions=self.state.submissions + 1)
                if self.state.submissions >= limit:
                    self._halt(SUBMISSION_LIMIT)
            else:
                del self._issued[approved]
                if isinstance(approved, RiskApprovedFlatten):
                    self._revoke_side(None, "a flatten was sent")
        return approved

    def _registered(self, approved: Approval) -> _Issued:
        """The registration of `approved`, or ApprovalError for `spend` cases 2 and 3."""
        if getattr(approved, "_token", None) is not _TOKEN:
            raise ApprovalError("approval is missing the risk engine's private token")
        issued = self._issued.get(approved)
        if issued is None or not _unchanged(_approval_fields(approved), issued.fields):
            raise ApprovalError("approval was not issued here or has changed")
        if isinstance(approved, RiskApprovedOrder) and _intent_hash(approved.intent) != (
            issued.intent_hash
        ):
            raise ApprovalError("the approved intent has changed")
        return issued

    def _revoke(self, key: str, reason: str) -> None:
        hold = self._outstanding.pop(key, None)
        if hold is not None:
            self._revoked[hold.approval] = reason
            log.info("risk_approval_revoked", approval_id=hold.approval.approval_id, reason=reason)

    def _revoke_side(self, side: Side | None, reason: str) -> None:
        """Revoke every outstanding approval on `side`, or on both when it is None."""
        for key, hold in list(self._outstanding.items()):
            if side is None or hold.side is side:
                self._revoke(key, reason)

    def _forget_expired(self, now: datetime) -> None:
        """Let go of approvals that have expired (outstanding, issued, revoked and pending), and
        of in-flight orders every acceptable snapshot must already show."""
        for key, hold in list(self._outstanding.items()):
            if now >= hold.expires_at:
                del self._outstanding[key]
        for approval, issued in list(self._issued.items()):
            if issued.expires_at is not None and now >= issued.expires_at:
                del self._issued[approval]
                self._revoked.pop(approval, None)
        for pending in [p for p in self._pending if now >= p.expires_at]:
            del self._pending[pending]
        horizon = now - timedelta(seconds=self._config.snapshot_max_age_s)
        self._in_flight = [f for f in self._in_flight if f.spent_at >= horizon]

    @contextmanager
    def _publishing(self) -> Iterator[None]:
        """Run the body, then hand `on_change` the state if the body changed it, even when the
        body raises (class docstring)."""
        before = self._state
        try:
            yield
        finally:
            if self._on_change is not None and self._state != before:
                self._on_change(self._state)

    def _update(self, **changes: Any) -> None:
        self._state = self.state.model_copy(update=changes)

    def _halt(self, reason: str) -> None:
        """Halt trading. A halt already in force keeps its first reason (C10)."""
        first = self.state.halt_reason if self.state.halted else None
        self._update(halted=True, halt_reason=first or reason)
        log.error("risk_halted", reason=reason, halt_reason=self.state.halt_reason)
        self._revoke_side(Side.BUY, f"trading halted: {reason}")

    def _next_id(self, prefix: str, now: datetime) -> str:
        self._update(issued=self.state.issued + 1)
        return f"{prefix}{self.state.issued}-{to_us(now)}"

    def _count(self, key: str) -> bool:
        """Count `key` toward C3's intent limit if a session has started and it is new. True when
        this made the count exceed the limit while the engine wasn't halted (`check`)."""
        if self.state.session is None or key in self.state.intents:
            return False
        self._update(intents=(*self.state.intents, key))
        return len(self.state.intents) > self._config.max_intents_per_day and not (
            self.state.halted
        )

    def _check(
        self, intent: OrderIntent, context: RiskContext, *, pending: PendingApproval | None
    ) -> RiskVerdict:
        now = self._clock.now()
        config = self._config
        stage = RiskCheck.INTENT
        over = False  # this intent is over C3's limit, so its verdict halts trading

        def veto(reason: str, *, halts: bool = False) -> Veto:
            if halts:
                self._halt(reason)
            if over:
                self._halt(INTENT_LIMIT)
            log.warning("risk_veto", check=stage.value, reason=reason, halts=halts or over)
            return Veto(intent, stage, reason, halts or over)

        try:
            self._forget_expired(now)
            key = f"{intent.decision_id}#{intent.leg}"
            self._revoke(key, "replaced by a re-check of its intent")
            if context.halted or context.drift_frozen or not context.healthy:
                self._revoke_side(Side.BUY, "a check saw HALTED, a drift freeze or rule 6")
            if pending is None:
                over = self._count(key)
            # Revalidate even a model constructed or copied without pydantic validation.
            OrderIntent.model_validate(intent.model_dump())
            day = intent.tau.astimezone(NEW_YORK).date()
            decision = next(
                (
                    d
                    for d in decision_schedule(self._calendar, day, self._timing)
                    if d.tau == intent.tau
                ),
                None,
            )
            if decision is None:
                return veto("intent is not a scheduled decision")
            execution_day = decision.executes_at.astimezone(NEW_YORK).date()
            session = self._calendar.session(execution_day)
            expires = next(
                bar.t_close
                for bar in session.bars
                if bar.t_open <= decision.executes_at < bar.t_close
            )
            if self.state.session != execution_day:
                return veto("intent needs its execution session started")
            if not decision.executes_at <= now < expires:
                return veto("intent is outside its execution window")
            buy = intent.side is Side.BUY

            stage = RiskCheck.HALTED
            if buy and (self.state.halted or context.halted):
                return veto("trading is halted")

            stage = RiskCheck.RECONCILIATION
            broker, book = context.broker, context.book
            BrokerSnapshot.model_validate(broker.model_dump())
            InternalBook.model_validate(book.model_dump())
            age = (now - broker.taken_at).total_seconds()
            if not 0 <= age <= config.snapshot_max_age_s:
                return veto("broker snapshot is stale or from the future")
            if (
                broker.position < 0
                or broker.position != book.position
                or Counter(broker.open_orders) != Counter(book.open_orders)
            ):
                return veto(
                    "broker and internal positions or orders disagree, or the position is short",
                    halts=True,
                )

            stage = RiskCheck.LIMITS
            price = context.limit_price
            valid_price = isinstance(price, (int, float)) and not isinstance(price, bool)
            valid_price = valid_price and isfinite(price) and price > 0
            quantity = intent.quantity
            in_flight = [f for f in self._in_flight if f.spent_at >= broker.taken_at]
            sellable = (
                broker.position
                - sum(
                    order.quantity
                    for order in broker.open_orders
                    if order.side is Side.SELL and order.kind is OrderKind.LIMIT
                )
                - sum(f.quantity for f in in_flight if f.side is Side.SELL)
            )
            if self._outstanding_quantity(Side.SELL) > sellable:
                self._revoke_side(Side.SELL, "a snapshot holds fewer shares than they reserve")
            if buy:
                if decision.reduce_only:
                    return veto("this decision can only reduce exposure")
                held = (
                    broker.position
                    + quantity
                    + sum(
                        order.quantity
                        for order in broker.open_orders
                        if order.side is Side.BUY and order.kind is OrderKind.LIMIT
                    )
                    + sum(f.quantity for f in in_flight if f.side is Side.BUY)
                    + self._outstanding_quantity(Side.BUY)
                )
                cap = min(
                    config.max_gross_exposure,
                    config.uncertain_cap
                    if intent.uncertain or intent.label is None
                    else config.label_caps.cap(intent.label),
                )
                if decision.last_fillable:
                    cap = min(cap, config.overnight_cap)
                # Invalid prices are refused by PRICE; they cannot earn an approval.
                equity = _exact(broker.equity)
                if valid_price and _exact(held) * _exact(price) > _exact(cap) * equity:
                    return veto("resulting position exceeds its exposure cap")
                if valid_price and _exact(quantity) * _exact(price) > (
                    _exact(config.single_order_cap) * equity
                ):
                    return veto("order exceeds the single-order cap")
            else:
                quantity = min(quantity, sellable - self._outstanding_quantity(Side.SELL))
                if quantity <= 0:
                    return veto("no unreserved shares remain to sell")

            stage = RiskCheck.LOCKS
            if buy:
                if self.state.daily_loss_locked:
                    return veto("daily-loss lock blocks entries")
                if context.drift_frozen:
                    return veto("drift freeze blocks entries")
                if not context.healthy:
                    return veto("rule 6 blocks entries")

            stage = RiskCheck.RATE
            if over and buy:
                return veto(INTENT_LIMIT)

            stage = RiskCheck.PRICE
            quote = context.quote
            Quote.model_validate(quote.model_dump())
            age = (now - quote.at).total_seconds()
            if not 0 <= age <= config.quote_max_age_s:
                return veto("quote is stale or from the future")
            if quote.bid > quote.ask:
                return veto("quote is crossed")
            ref = _exact(quote.ask if buy else quote.bid)
            collar = _exact(config.collar_bp) / 10_000
            if not valid_price or abs(_exact(price) - ref) > ref * collar:
                return veto("limit price is invalid or outside the collar")
            if buy and (intent.stop_price is None or intent.stop_price >= price):
                return veto("entry needs a protective stop below its limit")
            notional = quantity * price
            if not isfinite(notional):
                return veto("order notional must be finite")

            stage = RiskCheck.APPROVAL
            if pending is not None and _exact(price) > _exact(pending.limit_price) * (1 + collar):
                return veto("re-priced more than the collar above the price you approved")
            large = buy and notional > config.manual_approval_notional
            if large and self._mode is TradingMode.LIVE and pending is None:
                waiting = PendingApproval(
                    self._next_id("P", now),
                    intent,
                    quantity,
                    float(price),
                    float(notional),
                    now,
                    expires,
                )
                self._pending[waiting] = _pending_fields(waiting)
                log.warning(
                    "risk_pending_approval", pending_id=waiting.pending_id, notional=notional
                )
                return waiting
            if over:
                self._halt(INTENT_LIMIT)  # a reduction past the limit passes, and halts
            flagged = large or pending is not None
            approval = RiskApprovedOrder(
                token=_TOKEN,
                approval_id=self._next_id("O", now),
                intent=intent,
                quantity=quantity,
                limit_price=float(price),
                notional=float(notional),
                issued_at=now,
                expires_at=expires,
                flagged=flagged,
                halts=over,
            )
            self._issued[approval] = _Issued(
                _approval_fields(approval), key, _intent_hash(intent), expires
            )
            self._outstanding[key] = _Hold(approval, intent.side, quantity, expires)
            if flagged:
                log.warning(
                    "risk_flagged_order", approval_id=approval.approval_id, notional=notional
                )
            return approval
        except (QuantError, ValueError, TypeError, AttributeError, ArithmeticError) as exc:
            return veto(f"invalid intent or context: {exc}")

    def _outstanding_quantity(self, side: Side) -> int:
        return sum(hold.quantity for hold in self._outstanding.values() if hold.side is side)


def verdict_record(verdict: RiskVerdict) -> dict[str, Any]:
    """The verdict as plain data for the DecisionRecord (ARCHITECTURE §4.3), ready for canonical
    JSON. Never the token.

    - approved: `{"verdict": "approved", "approval_id", "intent", "quantity", "limit_price",
      "notional", "issued_at", "expires_at", "flagged", "halts"}`;
    - pending: `{"verdict": "pending", "pending_id", "intent", "quantity", "limit_price",
      "notional", "issued_at", "expires_at"}`;
    - veto: `{"verdict": "veto", "intent", "check", "reason", "halts"}`, `check` as its value.

    `intent` is the OrderIntent itself (canonical JSON dumps a model); times stay datetimes.
    """
    if isinstance(verdict, Veto):
        return {
            "verdict": "veto",
            "intent": verdict.intent,
            "check": verdict.check.value,
            "reason": verdict.reason,
            "halts": verdict.halts,
        }
    record: dict[str, Any] = {
        "intent": verdict.intent,
        "quantity": verdict.quantity,
        "limit_price": verdict.limit_price,
        "notional": verdict.notional,
        "issued_at": verdict.issued_at,
        "expires_at": verdict.expires_at,
    }
    if isinstance(verdict, PendingApproval):
        record.update(verdict="pending", pending_id=verdict.pending_id)
    else:
        record.update(
            verdict="approved",
            approval_id=verdict.approval_id,
            flagged=verdict.flagged,
            halts=verdict.halts,
        )
    return record


@dataclass(frozen=True)
class _Issued:
    """How the engine registered an approval: its fields as issued and, for an order, the key
    of its intent, the intent's canonical hash and its expiry."""

    fields: tuple[object, ...]
    key: str = ""
    intent_hash: str | None = None
    expires_at: datetime | None = None  # an order's; the engine lets go of it from then on


@dataclass(frozen=True, eq=False)
class _Hold:
    """What an outstanding order approval holds until it is spent, released, revoked or
    expires. The side and quantity are the engine's own copies, never read back from the
    approval."""

    approval: RiskApprovedOrder
    side: Side
    quantity: int
    expires_at: datetime


@dataclass(frozen=True)
class _InFlight:
    """A spent order, counted until a snapshot taken after `spent_at` can show it."""

    side: Side
    quantity: int
    spent_at: datetime


def _exact(value: float) -> Decimal:
    """The decimal value `value` prints as, for comparisons that must hold at the boundary."""
    return Decimal(repr(value))


def _fell_by(base: float, equity: float, fraction: float) -> bool:
    """True when `equity` is at least `fraction` of `base` below it, in decimal values."""
    return _exact(base) - _exact(equity) >= _exact(fraction) * _exact(base)


def _intent_hash(intent: OrderIntent) -> str | None:
    try:
        return sha256_hex(intent)
    except (QuantError, ValueError, TypeError):
        return None


def _initialize(approval: _Approved, token: object, fields: dict[str, object]) -> None:
    if token is not _TOKEN:
        raise ApprovalError("an approval requires the risk engine's private token")
    object.__setattr__(approval, "_token", token)
    for name, value in fields.items():
        object.__setattr__(approval, name, value)


def _approval_fields(approval: Approval) -> tuple[object, ...]:
    return tuple(
        getattr(approval, name, None) for name in (*_Approved.__slots__, *type(approval).__slots__)
    )


def _pending_fields(pending: PendingApproval) -> tuple[object, ...]:
    return tuple(getattr(pending, name, None) for name in PendingApproval.__dataclass_fields__)


def _unchanged(current: tuple[object, ...], issued: tuple[object, ...]) -> bool:
    return all(now is then for now, then in zip(current, issued, strict=True))


def _check_equity(equity: float) -> None:
    if isinstance(equity, bool) or not isinstance(equity, (int, float)):
        raise RiskError("equity must be finite and positive")
    if not isfinite(equity) or equity <= 0:
        raise RiskError("equity must be finite and positive")
