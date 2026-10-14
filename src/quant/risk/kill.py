"""The kill executor (spec §10, ARCHITECTURE §8.3).

One implementation, shared by the engine, quant-control and the CLI (`quant kill --direct`),
with a lock so only one copy runs at a time. In regular hours it:

1. halts: `RiskEngine.halt` and the HALTED flag in control.db (`KillStore.write_halted`);
2. cancels every working order, stops included (each through a RiskApprovedCancel);
3. closes the position with limit orders (each through a RiskApprovedFlatten) that start at the
   ±`collar_bp` collar and widen by `kill.widen_bp` after every `kill.requote_s` unfilled, up to
   `kill.max_collar_bp` (spec §10, your call of 2026-10-13: only kill-switch exits widen);
4. verifies on a broker snapshot that the account is flat with no open orders;
5. alerts (`KillAlerts.send`).

Outside regular hours it halts, cancels the working LIMIT orders, keeps the broker-side stop,
and queues the exit for the next open (09:30 ET), where the engine runs the executor again
(ARCHITECTURE §5's 09:30:05 job).

Every broker call goes through the BrokerPort, which spends each approval through the same
RiskEngine before it sends (I4). The executor never calls `RiskEngine.check`: kill exits are
risk-reducing and no collar checks them. "Now" is the executor's Clock; waiting is
`clock.sleep_until`, so a SimClock runs the whole sequence instantly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.core.orders import BrokerSnapshot, Quote
from quant.risk.broker import BrokerPort
from quant.risk.engine import RiskEngine
from quant.risk.limits import KillConfig, RiskConfig
from quant.risk.triggers import KillTrigger


class KillStatus(StrEnum):
    """How one run of the executor ended."""

    FLAT = "flat"  # verified: the broker holds no position and no open orders
    QUEUED = "queued"  # outside regular hours: LIMIT orders cancelled, stop kept, exit queued
    INCOMPLETE = "incomplete"  # the deadline or the session's close came before FLAT
    BUSY = "busy"  # another copy holds the lock; this run did nothing at all


@dataclass(frozen=True)
class KillRequest:
    """One request to kill.

    - `trigger`: why (KillTrigger.MANUAL for the CLI, the dashboard and Telegram);
    - `source`: who asked, in plain words: "cli", "dashboard", "telegram", "engine" or
      "control";
    - `reason`: what happened, in plain words (for example "drawdown 10.2% from the peak").
    """

    trigger: KillTrigger
    source: str
    reason: str


@dataclass(frozen=True)
class KillExit:
    """One closing order the executor sent: its client order id, its limit price, how far from
    the quote it sat (`collar_bp`), and when it was sent."""

    client_order_id: str
    limit_price: float
    collar_bp: float
    sent_at: datetime


@dataclass(frozen=True)
class KillReport:
    """What one run did. The executor sends it to `KillAlerts.send` and returns it.

    - `request`, `status`;
    - `halt_reason`: the reason given to `RiskEngine.halt` and `KillStore.write_halted`:
      `f"kill switch ({trigger}, {source}): {reason}"`; empty for BUSY;
    - `started_at`, `finished_at`: the clock when the run began and ended;
    - `regular_hours`: `in_regular_hours(started_at)`, which picks the path;
    - `cancelled`: the client order id of every cancel sent, in order, repeats included;
    - `kept`: the stops kept outside regular hours, by client order id;
    - `exits`: every closing order sent, in order;
    - `queued_exit_at`: when the queued exit runs (QUEUED, or INCOMPLETE at the close), else
      None;
    - `final`: the last broker snapshot read, or None when none was read;
    - `errors`: every error recorded and survived, in order, as `f"{type name}: {message}"`.
    """

    request: KillRequest
    status: KillStatus
    halt_reason: str
    started_at: datetime
    finished_at: datetime
    regular_hours: bool
    cancelled: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    exits: tuple[KillExit, ...] = ()
    queued_exit_at: datetime | None = None
    final: BrokerSnapshot | None = None
    errors: tuple[str, ...] = ()


class KillLock(Protocol):
    """The lock that lets only one copy of the executor run (ARCHITECTURE §8.3)."""

    def acquire(self) -> bool:
        """Take the lock without waiting: True when taken, False when another copy holds it."""
        ...

    def release(self) -> None:
        """Let go of a lock this caller took."""
        ...


class FileKillLock:
    """A KillLock across processes: an exclusive, non-blocking `fcntl.flock` on a file, so the
    engine, quant-control and the CLI exclude one another on the same machine. The kernel drops
    the lock if its holder dies, so a crashed copy never blocks the next."""

    def __init__(self, path: Path) -> None:
        """A lock on `path` (for example var/kill.lock). Nothing is opened until `acquire`."""
        raise NotImplementedError("B9b: FileKillLock")

    def acquire(self) -> bool:
        """Open `path` (creating it with mode 0o600 if missing; its directory must exist) and
        try `flock(LOCK_EX | LOCK_NB)`. True when taken. False, closing the file, when any other
        holder has it: another process, or another FileKillLock object on the same path in this
        process. RiskError when this object already holds it, or the file can't be opened."""
        raise NotImplementedError("B9b: FileKillLock.acquire")

    def release(self) -> None:
        """Unlock and close the file; the file stays. RiskError when this object doesn't hold
        the lock."""
        raise NotImplementedError("B9b: FileKillLock.release")


class MemoryKillLock:
    """A KillLock inside one process, for the backtest and tests: share one object between the
    executors it should exclude."""

    def __init__(self) -> None:
        raise NotImplementedError("B9b: MemoryKillLock")

    @property
    def held(self) -> bool:
        """True while some caller holds the lock."""
        raise NotImplementedError("B9b: MemoryKillLock.held")

    def acquire(self) -> bool:
        """True when taken; False while held, by anyone."""
        raise NotImplementedError("B9b: MemoryKillLock.acquire")

    def release(self) -> None:
        """Let go. RiskError when the lock isn't held."""
        raise NotImplementedError("B9b: MemoryKillLock.release")


class KillStore(Protocol):
    """Where the executor records the halt and a queued exit, durably (control.db; B14 and B15
    implement it). The engine reads both at startup (ARCHITECTURE §9.3, step 3)."""

    def write_halted(self, reason: str, at: datetime) -> None:
        """Set HALTED with `reason` at `at`, durably, before returning."""
        ...

    def queue_exit(self, run_at: datetime, reason: str) -> None:
        """Queue the kill's exit to run at `run_at` (a session's open), durably, before
        returning. The engine runs `KillExecutor.execute` again at or after it."""
        ...


class KillAlerts(Protocol):
    """Where the executor's alert goes (the alert outbox, ARCHITECTURE §12)."""

    def send(self, report: KillReport) -> None:
        """Queue an alert describing `report`. It may raise; the executor logs that and goes on."""
        ...


def in_regular_hours(now: datetime, calendar: TradingCalendar) -> bool:
    """True when `now` (UTC) lies in a session's regular hours, `session.open <= now <
    session.close`, for the session on `now`'s New York date. Early closes follow the calendar.
    ClockError for a naive `now`."""
    raise NotImplementedError("B9b: in_regular_hours")


def next_open(now: datetime, calendar: TradingCalendar) -> datetime:
    """The earliest session open at or after `now` (UTC): today's open before it, else the next
    session's. This is when an exit queued outside regular hours runs (ARCHITECTURE §8.3).
    ClockError for a naive `now`; CalendarError past the calendar's end."""
    raise NotImplementedError("B9b: next_open")


def kill_collar_bp(attempt: int, *, collar_bp: float, kill: KillConfig) -> float:
    """How far from the quote attempt `attempt` (0 for the first) sits, in basis points:
    `min(collar_bp + attempt * kill.widen_bp, kill.max_collar_bp)`. So with 10, 10 and 100:
    10, 20, 30 ... 100, 100. ValueError for a negative `attempt`."""
    raise NotImplementedError("B9b: kill_collar_bp")


def kill_limit_price(
    quote: Quote, position: int, attempt: int, *, collar_bp: float, kill: KillConfig
) -> float:
    """The limit price of kill exit `attempt` closing `position` shares, with
    `w = kill_collar_bp(attempt, ...)`, computed in decimal values (`Decimal(repr(x))`):

    - long (`position > 0`), a SELL: `quote.bid * (1 - w / 10_000)`, rounded *down* to the cent;
    - short (`position < 0`), a BUY: `quote.ask * (1 + w / 10_000)`, rounded *up* to the cent.

    Rounding always makes the order more marketable, never less. So a bid of 100.00 gives
    99.90, 99.80, 99.70 ... 99.00. The quote is used however old it is: a kill never waits for
    a fresh one. ValueError when `position` is 0 or `attempt` is negative."""
    raise NotImplementedError("B9b: kill_limit_price")


class KillExecutor:
    """Runs the kill switch (module docstring). One executor serves one engine and one broker;
    the lock is what keeps copies in other processes out."""

    def __init__(
        self,
        engine: RiskEngine,
        broker: BrokerPort,
        config: RiskConfig,
        *,
        clock: Clock,
        lock: KillLock,
        store: KillStore,
        alerts: KillAlerts,
        calendar: TradingCalendar | None = None,
    ) -> None:
        """An executor that halts `engine`, trades through `broker` (whose approvals `engine`
        issues and spends), with `config`'s collar and kill settings (the engine's own limits),
        judging time by `clock` and the hours by `calendar` (default `TradingCalendar.xnys()`).
        """
        raise NotImplementedError("B9b: KillExecutor")

    def execute(self, request: KillRequest) -> KillReport:
        """Run the kill switch for `request` and return the report.

        **Lock.** `lock.acquire()` first. When it returns False the run is BUSY: it returns a
        KillReport with status BUSY, an empty `halt_reason` and nothing else done (no halt, no
        broker call, no alert). Otherwise the lock is released however `execute` ends.

        **1. Halt**, before any broker call: `engine.halt(halt_reason)`, then
        `store.write_halted(halt_reason, now)`. A QuantError or OSError from either is recorded
        in `errors` and the run goes on: reducing risk never waits on a write.

        Then `regular_hours = in_regular_hours(now, calendar)` picks the path. Throughout,
        `deadline = started_at + kill.deadline_s`. Waiting means `clock.sleep_until(min(now +
        kill.poll_s, deadline))` and then a fresh `broker.snapshot()`. A BrokerError from any
        broker call is recorded in `errors` and the step is retried after the next wait. When
        `now >= deadline` before the run is done, it stops where it is, leaving any working
        exit at the broker: INCOMPLETE, and the caller runs the executor again (ARCHITECTURE
        §16: the kill keeps retrying). ApprovalError from the broker's spend is a bug and
        propagates.

        *Cancelling* an order means `engine.approve_cancel(client_order_id, reason=halt_reason)`
        and `broker.cancel(approval)`; *cancelling until gone* means cancelling, then waiting
        until no snapshot shows the orders being cancelled, cancelling again (with a new
        approval) any still shown `kill.requote_s` after its last cancel was sent.

        **Regular hours.**

        2. Read a snapshot and cancel until gone every open order, stops included (ARCHITECTURE
           §8.2: open sell orders reserve the shares an exit needs). Orders that appear while
           waiting are cancelled too.
        3. For attempt = 0, 1, 2 ...: when the latest snapshot shows `position == 0` and no
           open orders, the run is FLAT (step 4). Otherwise read `broker.quote()`, take
           `limit = kill_limit_price(quote, snapshot.position, attempt, collar_bp, kill)`, get
           `engine.approve_flatten(reason=halt_reason)`, send `broker.flatten(approval,
           limit_price=limit)` and record a KillExit. Wait until a snapshot shows the account
           flat with no open orders (FLAT), or until `kill.requote_s` has passed since the exit
           was sent; then cancel until gone every open order (the unfilled exit's remainder
           among them) and go on with the next attempt, so after a partial fill the next exit
           closes only what is left.
        4. **Verify.** FLAT only on a snapshot read after the last exit or cancel was sent that
           shows `position == 0` and no open orders; it becomes `final`.
           If the session closes (`now >= session.close`) first, the run stops: it cancels
           until gone the working exit (once, without waiting past the deadline), queues the
           exit for `next_open(now)` (`store.queue_exit`) and is INCOMPLETE with
           `queued_exit_at` set. Re-placing a protective stop over the night is execution's
           (B13), not the executor's.

        **Outside regular hours.**

        2. Read a snapshot. With `position == 0`, cancel until gone every open order, stops
           included: FLAT once a snapshot shows none.
        3. Otherwise cancel until gone every open LIMIT order (entries and exits) and keep every
           STOP order (`kept`, by client order id), send no exit, and
           `store.queue_exit(next_open(now), halt_reason)`: QUEUED, with `queued_exit_at` set,
           once a snapshot shows no LIMIT order. A QuantError or OSError from `queue_exit` is
           recorded in `errors`, and the run is still QUEUED.

        **5. Alert.** `alerts.send(report)` with the finished report, for every status but BUSY.
        An exception from it is logged with structlog and swallowed; the same report is
        returned either way.

        Every step is logged with structlog.
        """
        raise NotImplementedError("B9b: KillExecutor.execute")
