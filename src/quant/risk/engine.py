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
5. RATE: more than `max_intents_per_day` intents in a session halts trading (C3).
6. PRICE: the limit sits inside the collar of a fresh quote, and every entry carries a stop.
7. APPROVAL (D7): a live BUY above the notional threshold waits for you; paper flags it.

In a long-or-flat system a SELL is exactly a risk-reducing intent. It skips the caps in 3, the
locks in 4 and the approval in 7, and is clamped so it can never take the position short.

The engine also keeps C3's counters, the daily-loss lock and the drawdown trigger (`mark`, fed
by the minute monitor, C8). Its state is a RiskState the live engine persists with EngineState.
The kill executor, reconciliation snapshots, the minute monitor itself and the broker port are
B9b's. "Now" comes from the engine's Clock. Every veto and halt is logged with structlog.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Final, Protocol, final

from pydantic import Field

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.core.model import StrictModel
from quant.core.orders import BrokerSnapshot, InternalBook, OrderIntent, Quote, TradingMode
from quant.core.schedule import Timing
from quant.risk.limits import RiskConfig

_TOKEN: Final = object()
"""The private token every approved type's constructor demands. It never leaves this module."""


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
        raise NotImplementedError

    def __delattr__(self, name: str) -> None:
        raise NotImplementedError

    def __copy__(self) -> _Approved:
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, object]) -> _Approved:
        raise NotImplementedError

    def __reduce_ex__(self, protocol: object) -> tuple[object, ...]:
        raise NotImplementedError


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
    - `flagged`: a BUY above D7's threshold, approved because the engine trades paper. The
      engine alerts on it.
    """

    __slots__ = ("expires_at", "flagged", "intent", "limit_price", "notional", "quantity")

    intent: OrderIntent
    quantity: int
    limit_price: float
    notional: float
    expires_at: datetime
    flagged: bool

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
    ) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        """`RiskApprovedOrder(<approval_id>, <side> <quantity> @ <limit_price>, until <expires_at>)`."""
        raise NotImplementedError


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
        raise NotImplementedError

    def __repr__(self) -> str:
        """`RiskApprovedCancel(<approval_id>, <client_order_id>)`."""
        raise NotImplementedError


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
        raise NotImplementedError

    def __repr__(self) -> str:
        """`RiskApprovedFlatten(<approval_id>, <symbol>)`."""
        raise NotImplementedError


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
    mismatch, a short position, or the intent past C3's limit); the engine then persists HALTED
    and alerts."""

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
    - `peak_equity`: the highest equity seen, across sessions;
    - `intents`: the intents of this session that reached check 5, as `"<decision_id>#<leg>"`
      in the order first seen;
    - `submissions`: raw broker submissions this session: RiskApprovedOrders spent;
    - `daily_loss_locked`: entries locked until the next session (spec §10);
    - `halted`, `halt_reason`: trading halted by this engine, and why. Only a resume (B14)
      clears it;
    - `killed`: the drawdown trigger has fired;
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
      once; the engine is halted from then on.
    """

    equity: float
    day_loss: float
    drawdown: float
    locked: bool
    flatten: bool
    kill: bool


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
        raise NotImplementedError

    def is_spent(self, approval_id: str) -> bool:
        raise NotImplementedError

    def spend(self, approval_id: str, at: datetime) -> bool:
        raise NotImplementedError


# --- the engine -----------------------------------------------------------------------------------


class RiskEngine:
    """The single choke point (spec §10, I4).

    Approval ids are `f"{prefix}{n}-{to_us(issued_at)}"` (quant.core.epoch.to_us), where the
    prefix is O for an order, C a cancel, F a flatten and P a pending approval, and n is
    `state.issued` after counting this one. Ids are therefore deterministic in a replay, and
    unique across restarts as long as the engine restarts from its persisted RiskState.
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
    ) -> None:
        """An engine with `config`'s limits, trading in `mode`.

        `calendar` defaults to `TradingCalendar.xnys()`, `timing` to `Timing()`, `spent` to a new
        MemorySpentApprovals, and `state` to `RiskState()`. The engine starts with no approvals
        issued: approvals from an earlier engine object are never honoured, even with the same
        `spent` store and state."""
        raise NotImplementedError

    @property
    def state(self) -> RiskState:
        """The current state, as an immutable snapshot."""
        raise NotImplementedError

    def start_session(self, day: date, *, equity: float) -> None:
        """Begin session `day` with start-of-day equity `equity`: the peak becomes the larger of
        the old peak and `equity`, the intent and submission counters go to zero, and the
        daily-loss lock clears. HALTED and the drawdown trigger stay as they are.

        RiskError unless `day` is a session of the calendar later than `state.session`, and
        `equity` is finite and positive."""
        raise NotImplementedError

    def mark(self, equity: float) -> MarkOutcome:
        """One equity mark from the minute monitor, on the freshest price (C8), or from a broker
        snapshot. The peak rises to `equity` if it is higher.

        - Daily loss: when `start_equity - equity >= daily_loss_fraction * start_equity`, the
          daily-loss lock is set until the next session (exactly 2% locks).
        - Drawdown: when `peak_equity - equity >= max_drawdown_fraction * peak_equity`, the
          engine halts (`halt_reason` names the drawdown) and the trigger fires (exactly 10%
          fires).

        RiskError when no session has started, or `equity` isn't finite and positive."""
        raise NotImplementedError

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

        1. **HALTED.** When `state.halted` or `context.halted`, a BUY is vetoed.
        2. **Reconciliation.** Vetoed when the snapshot is older than `snapshot_max_age_s`
           (`now - broker.taken_at > max`; exactly the maximum passes) or from the future.
           Vetoed, halting trading, when `broker.position != book.position`, when the open
           orders differ (compared as multisets of OpenOrder), or when `broker.position < 0`.
           Nothing is ever corrected.
        3. **Limits.**
           - SELL: clamped to `broker.position` minus the open SELL LIMIT orders' quantity
             (stops don't count: execution cancels the stop before it sells, ARCHITECTURE §8.2).
             Vetoed when nothing is left to sell. No cap applies.
           - BUY: vetoed on a decision that can only reduce (`d.reduce_only`). Otherwise let
             `held = broker.position + open BUY LIMIT quantity + intent.quantity`; vetoed unless
             `held * limit_price <= cap * equity`, where cap is the smallest of
             `max_gross_exposure`, the label's cap (`uncertain_cap` when `intent.uncertain` or
             `intent.label is None`) and, on the last fillable decision (`d.last_fillable`),
             `overnight_cap`; and vetoed unless
             `intent.quantity * limit_price <= single_order_cap * equity`. Equality passes.
        4. **Locks.** A BUY is vetoed while `state.daily_loss_locked`, `context.drift_frozen`,
           or not `context.healthy` (rule 6).
        5. **Rate.** The intent's key `"<decision_id>#<leg>"` is counted the first time it gets
           here; re-checks of a counted intent (re-quotes) don't count again. The key that would
           make `len(state.intents)` exceed `max_intents_per_day` is counted and halts trading:
           a BUY is vetoed (`halts=True`), and a SELL goes on as it would under HALTED.
        6. **Price.** Vetoed when the quote is older than `quote_max_age_s` (exactly the maximum
           passes) or from the future, when it is crossed (`bid > ask`), or unless
           `abs(limit_price - ref) <= ref * collar_bp / 10_000`, with `ref` the ask for a BUY
           and the bid for a SELL. A BUY is vetoed unless `intent.stop_price` is set and below
           `limit_price`.
        7. **Approval (D7).** A BUY whose notional `intent.quantity * limit_price` exceeds
           `manual_approval_notional` (exactly the threshold doesn't) returns a registered
           PendingApproval when `mode` is LIVE, and a RiskApprovedOrder with `flagged=True`
           when it is PAPER.

        A pass returns a registered RiskApprovedOrder, expiring at the window's end. Checking
        never raises for a bad intent or context: every refusal is a Veto.
        """
        raise NotImplementedError

    def approve_pending(self, pending: PendingApproval, context: RiskContext) -> RiskVerdict:
        """Your approval of `pending` (D7, ARCHITECTURE §8.4): checks 1–6 run again on its intent
        with `context`, and a pass returns a RiskApprovedOrder with `flagged=True`. A Veto is
        returned as from `check`. The intent is not counted again.

        ApprovalError when this engine object didn't issue `pending` (judged by identity), it
        was already approved, or `now >= pending.expires_at`."""
        raise NotImplementedError

    def approve_cancel(self, client_order_id: str, *, reason: str) -> RiskApprovedCancel:
        """A registered approval to cancel `client_order_id`, issued whatever the state."""
        raise NotImplementedError

    def approve_flatten(self, *, reason: str) -> RiskApprovedFlatten:
        """A registered approval for the kill switch to close the whole SPY position, issued
        whatever the state."""
        raise NotImplementedError

    def spend(self, approval: object) -> Approval:
        """Spend `approval` once; the broker adapter calls this right before it sends anything,
        and sends nothing unless it returns.

        ApprovalError, in this order, for:
        1. anything whose type is not exactly RiskApprovedOrder, RiskApprovedCancel or
           RiskApprovedFlatten (plain objects, subclasses);
        2. a missing token, or one that isn't this module's;
        3. an approval this engine object never issued (judged by identity: copies, one built
           with the token by anyone else, another engine's), or one any of whose fields is not
           the very object issued;
        4. a RiskApprovedOrder at or after its `expires_at`;
        5. an id the SpentApprovals store already holds (reused);
        6. a BUY while `state.halted`; and a BUY that would be submission number
           `submission_halt_at` this session, which halts trading. A SELL is never refused here;
           it is counted, and halts trading the same way when it reaches the number.

        Then the id is recorded in the store (`spend` returning False is case 5) and a
        RiskApprovedOrder counts as one raw submission (C3). Cancels and flattens never count.
        """
        raise NotImplementedError


def verdict_record(verdict: RiskVerdict) -> dict[str, Any]:
    """The verdict as plain data for the DecisionRecord (ARCHITECTURE §4.3), ready for canonical
    JSON. Never the token.

    - approved: `{"verdict": "approved", "approval_id", "intent", "quantity", "limit_price",
      "notional", "issued_at", "expires_at", "flagged"}`;
    - pending: `{"verdict": "pending", "pending_id", "intent", "quantity", "limit_price",
      "notional", "issued_at", "expires_at"}`;
    - veto: `{"verdict": "veto", "intent", "check", "reason", "halts"}`, `check` as its value.

    `intent` is the OrderIntent itself (canonical JSON dumps a model); times stay datetimes.
    """
    raise NotImplementedError
