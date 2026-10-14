"""The kill executor (spec §10, ARCHITECTURE §8.3).

One implementation, shared by the engine, quant-control and the CLI (`quant kill --direct`),
with a lock so only one copy runs at a time. In regular hours it:

1. halts: `RiskEngine.halt`, `RiskEngine.revoke_orders` and the HALTED flag in control.db
   (`KillStore.write_halted`);
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

import fcntl
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import Enum, StrEnum
from pathlib import Path
from typing import Final, Protocol, TypeVar

import structlog

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import Clock, ensure_utc
from quant.core.errors import RiskError
from quant.core.orders import BrokerSnapshot, OpenOrder, OrderKind, Quote
from quant.risk.broker import BrokerPort
from quant.risk.engine import RiskEngine
from quant.risk.limits import KillConfig, RiskConfig
from quant.risk.triggers import KillTrigger

log = structlog.get_logger()


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
class KillStop:
    """One protective stop the executor left at the broker when it couldn't finish: its client
    order id, the shares it covers, its trigger price, and when it was placed."""

    client_order_id: str
    quantity: int
    stop_price: float
    placed_at: datetime


@dataclass(frozen=True)
class KillReport:
    """What one run did. The executor sends it to `KillAlerts.send` and returns it.

    - `request`, `status`;
    - `halt_reason`: the reason this kill gave `RiskEngine.halt`:
      `f"kill switch ({trigger}, {source}): {reason}"`; empty for BUSY. `KillStore.write_halted`
      gets the reason in force, which is an earlier halt's when trading was already halted;
    - `started_at`, `finished_at`: the clock when the run began and ended;
    - `regular_hours`: `in_regular_hours(started_at)`, which picks the path;
    - `cancelled`: the client order id of every cancel sent, in order, repeats included;
    - `kept`: the stops kept outside regular hours, by client order id;
    - `exits`: every closing order sent, in order;
    - `queued_exit_at`: when the queued exit runs (QUEUED, or INCOMPLETE at the close), else
      None;
    - `final`: the last broker snapshot read, or None when none was read;
    - `stops`: the protective stops placed before the run ended INCOMPLETE with a position
      open (KillExecutor.execute, step 6);
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
    stops: tuple[KillStop, ...] = ()


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
    the lock if its holder dies, so a crashed copy never blocks the next. Threads that share one
    object exclude one another too (the minute monitor's thread and a command's)."""

    def __init__(self, path: Path) -> None:
        """A lock on `path` (for example var/kill.lock). Nothing is opened until `acquire`."""
        self._path = Path(path)
        self._fd: int | None = None
        self._owner: int | None = None  # the thread that holds it
        self._guard = threading.Lock()

    def acquire(self) -> bool:
        """Open `path` (creating it with mode 0o600 if missing; its directory must exist) and
        try `flock(LOCK_EX | LOCK_NB)`. True when taken. False, closing the file, when any other
        holder has it: another process, another FileKillLock object on the same path in this
        process, or another thread holding this object. RiskError when the calling thread
        already holds it (it isn't re-entrant), or the file can't be opened or locked."""
        with self._guard:
            if self._fd is not None:
                if self._owner == threading.get_ident():
                    raise RiskError(f"this kill lock already holds {self._path}")
                return False
            try:
                fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
            except OSError as exc:
                raise RiskError(f"can't open the kill lock {self._path}: {exc}") from exc
            try:
                os.fchmod(fd, 0o600)  # whatever the umask made of it
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return False
            except OSError as exc:
                os.close(fd)
                raise RiskError(f"can't lock {self._path}: {exc}") from exc
            self._fd = fd
            self._owner = threading.get_ident()
            return True

    def release(self) -> None:
        """Unlock and close the file; the file stays. RiskError when this object doesn't hold
        the lock."""
        with self._guard:
            if self._fd is None:
                raise RiskError(f"this kill lock doesn't hold {self._path}")
            fd, self._fd, self._owner = self._fd, None, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


class MemoryKillLock:
    """A KillLock inside one process, for the backtest and tests: share one object between the
    executors it should exclude."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def held(self) -> bool:
        """True while some caller holds the lock."""
        return self._lock.locked()

    def acquire(self) -> bool:
        """True when taken; False while held, by anyone."""
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        """Let go. RiskError when the lock isn't held."""
        try:
            self._lock.release()
        except RuntimeError as exc:
            raise RiskError("the kill lock isn't held") from exc


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
    now = ensure_utc(now)
    day = now.astimezone(NEW_YORK).date()
    if not calendar.is_session(day):
        return False
    session = calendar.session(day)
    return session.open <= now < session.close


def next_open(now: datetime, calendar: TradingCalendar) -> datetime:
    """The earliest session open at or after `now` (UTC): today's open before it, else the next
    session's. This is when an exit queued outside regular hours runs (ARCHITECTURE §8.3).
    ClockError for a naive `now`; CalendarError past the calendar's end."""
    now = ensure_utc(now)
    day = now.astimezone(NEW_YORK).date()
    if calendar.is_session(day) and now <= calendar.session(day).open:
        return calendar.session(day).open
    return calendar.session(calendar.next_session(day)).open


def kill_collar_bp(attempt: int, *, collar_bp: float, kill: KillConfig) -> float:
    """How far from the quote attempt `attempt` (0 for the first) sits, in basis points:
    `min(collar_bp + attempt * kill.widen_bp, kill.max_collar_bp)`. So with 10, 10 and 100:
    10, 20, 30 ... 100, 100. ValueError for a negative `attempt`."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError(f"attempt must be a whole number from 0, not {attempt!r}")
    width = _exact(collar_bp) + attempt * _exact(kill.widen_bp)
    return float(min(width, _exact(kill.max_collar_bp)))


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
    if position == 0:
        raise ValueError("a kill exit needs a position to close")
    width = _exact(kill_collar_bp(attempt, collar_bp=collar_bp, kill=kill)) / 10_000
    if position > 0:
        price = (_exact(quote.bid) * (1 - width)).quantize(_CENT, rounding=ROUND_FLOOR)
    else:
        price = (_exact(quote.ask) * (1 + width)).quantize(_CENT, rounding=ROUND_CEILING)
    return float(price)


def kill_stop_price(quote: Quote, *, kill: KillConfig) -> float:
    """The trigger price of the protective stop a kill leaves: `quote.bid * (1 -
    kill.stop_distance_bp / 10_000)` in decimal values, rounded *down* to the cent, so it always
    sits below the bid. A bid of 100.00 with 100 bp gives 99.00."""
    raise NotImplementedError("B9c: kill_stop_price")


class KillExecutor:
    """Runs the kill switch (module docstring). One executor serves one engine and one broker;
    the lock is what keeps other copies out, in this process and in others."""

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
        self._engine = engine
        self._broker = broker
        self._config = config
        self._clock = clock
        self._lock = lock
        self._store = store
        self._alerts = alerts
        self._calendar = TradingCalendar.xnys() if calendar is None else calendar

    def execute(self, request: KillRequest) -> KillReport:
        """Run the kill switch for `request` and return the report.

        **A kill never dies halfway.** Every step is attempted on its own: an exception of any
        type from a step (the engine and its `on_change`, the store, the broker, the broker's
        spend) is recorded in `errors` and the run goes on, a broker step being retried after
        the next wait. An exception outside any step ends the run INCOMPLETE, recorded the same
        way. The alert is always attempted, and the report always returned. A failing
        `on_change` means the engine issues no approval, so under a lasting failure the run
        sends nothing and ends INCOMPLETE at the deadline, still halted and alerted.

        **Lock.** `lock.acquire()` first. When it returns False the run is BUSY: it returns a
        KillReport with status BUSY, an empty `halt_reason` and nothing else done (no halt, no
        broker call, no alert). When it raises (say var/ is missing), the error is recorded and
        the kill runs without the lock: halting and flattening matter more than excluding a
        copy, and a second copy's exit is refused while the first's reserves the shares.
        Otherwise the lock is released however `execute` ends.

        **1. Halt**, before any broker call: `engine.halt(halt_reason)`, then
        `engine.revoke_orders(halt_reason)` (every outstanding order approval, SELLs included:
        a SELL approved against shares the stop has since sold would open a short), then
        `store.write_halted(reason, started_at)` with the reason in force, which is the
        engine's first one (`state.halt_reason`, C10) when trading was already halted.

        Then `regular_hours = in_regular_hours(now, calendar)` picks the path. Throughout,
        `deadline = started_at + kill.deadline_s`. Waiting means `clock.sleep_until(min(now +
        kill.poll_s, deadline))` and then a fresh `broker.snapshot()`. When `now >= deadline`
        before the run is done, it stops where it is, leaving any working exit at the broker:
        INCOMPLETE, and the caller runs the executor again (ARCHITECTURE §16: the kill keeps
        retrying).

        *Cancelling* an order means `engine.approve_cancel(client_order_id, reason=halt_reason)`
        and `broker.cancel(approval)`; *cancelling until gone* means cancelling, then waiting
        until no snapshot shows the orders being cancelled, cancelling again (with a new
        approval) any still shown `kill.requote_s` after its last cancel was sent.

        **Regular hours.**

        2. Read a snapshot and cancel until gone every open order, stops included (ARCHITECTURE
           §8.2: open sell orders reserve the shares an exit needs). Orders that appear while
           waiting are cancelled too.
        3. For attempt = 0, 1, 2 ...: when the latest snapshot shows `position == 0` and no
           open orders, the run is FLAT (step 4). Otherwise read `broker.quote()`. If the
           deadline or the close has come by the time the quote arrives, no exit is sent.
           Else take `limit = kill_limit_price(quote, snapshot.position, attempt, collar_bp,
           kill)`, get `engine.approve_flatten(reason=halt_reason)`, send
           `broker.flatten(approval, limit_price=limit)` and record a KillExit. Wait until a
           snapshot shows the account flat with no open orders (FLAT), or until
           `kill.requote_s` has passed since the exit was sent; then cancel until gone every
           open order (the unfilled exit's remainder among them) and go on with the next
           attempt, so after a partial fill the next exit closes only what is left. A failed
           quote or exit keeps the attempt number.
        4. **Verify.** FLAT only on a snapshot read after the last exit or cancel was sent that
           shows `position == 0` and no open orders; it becomes `final`.
           If the session closes (`now >= session.close`) first, the run stops: it cancels
           until gone the working exit (once, without waiting past the deadline), queues the
           exit and is INCOMPLETE with `queued_exit_at` set. Re-placing a protective stop over
           the night is execution's (B13), not the executor's.

        **Outside regular hours.**

        2. Read a snapshot. With `position == 0`, cancel until gone every open order, stops
           included, but only while the account stays flat: once a snapshot shows a position
           (an entry filled before its cancel landed), its stops are kept. FLAT once a snapshot
           shows no position and no order.
        3. Otherwise cancel until gone every open LIMIT order (entries and exits) and keep every
           STOP order (`kept`, by client order id), send no exit, and queue the exit: QUEUED,
           with `queued_exit_at` set, once a snapshot shows no LIMIT order.

        *Queueing the exit* is `store.queue_exit(next_open(started_at), halt_reason)`: the
        first open at or after the moment the kill started, so a run that crosses 09:30 queues
        that morning's open, which has then passed and runs at once. A failed write is
        recorded, and the run's status stands.

        **5. Alert.** `alerts.send(report)` with the finished report, for every status but BUSY.
        An exception from it is logged with structlog and swallowed; the same report is
        returned either way.

        Every step is logged with structlog.
        """
        started = self._clock.now()
        errors: list[str] = []
        try:
            locked = self._lock.acquire()
        except Exception as exc:  # a kill never waits on its lock's file (docstring)
            errors.append(_error(exc))
            log.error("kill_lock_failed", error=str(exc), error_type=type(exc).__name__)
            locked = None
        if locked is False:
            log.warning("kill_busy", trigger=request.trigger.value, source=request.source)
            return KillReport(
                request=request,
                status=KillStatus.BUSY,
                halt_reason="",
                started_at=started,
                finished_at=started,
                regular_hours=in_regular_hours(started, self._calendar),
            )
        try:
            report = _Run(self, request, started, errors).execute()
            try:
                self._alerts.send(report)
            except Exception:  # an alert never stops a kill (ARCHITECTURE §8.3)
                log.exception("kill_alert_failed", status=report.status.value)
            return report
        finally:
            if locked:
                try:
                    self._lock.release()
                except Exception:
                    log.exception("kill_lock_release_failed")


class _Failed(Enum):
    """What `_Run.attempt` returns for a step that raised."""

    FAILED = "failed"


_T = TypeVar("_T")


class _Run:
    """One run of the executor, after it has taken the lock: what it did so far, for the
    report, and the waiting and cancelling every path shares (KillExecutor.execute)."""

    def __init__(
        self, executor: KillExecutor, request: KillRequest, started: datetime, errors: list[str]
    ) -> None:
        self.engine = executor._engine
        self.broker = executor._broker
        self.store = executor._store
        self.clock = executor._clock
        self.calendar = executor._calendar
        self.collar_bp = executor._config.collar_bp
        self.kill = executor._config.kill
        self.request = request
        self.started = started
        self.reason = f"kill switch ({request.trigger}, {request.source}): {request.reason}"
        self.deadline = started + timedelta(seconds=self.kill.deadline_s)
        self.regular_hours = in_regular_hours(started, self.calendar)
        # In regular hours the session's close stops the exits (step 4); None while it doesn't.
        self.close: datetime | None = None
        if self.regular_hours:
            self.close = self.calendar.session(started.astimezone(NEW_YORK).date()).close
        self.cancelled: list[str] = []
        self.kept: tuple[str, ...] = ()
        self.exits: list[KillExit] = []
        self.queued_exit_at: datetime | None = None
        self.final: BrokerSnapshot | None = None
        self.errors = errors

    def execute(self) -> KillReport:
        log.error(
            "kill_started",
            trigger=self.request.trigger.value,
            source=self.request.source,
            reason=self.request.reason,
            regular_hours=self.regular_hours,
        )
        self.halt()
        try:
            status = self.regular() if self.regular_hours else self.outside()
        except Exception as exc:  # outside any step: the run ends, and still alerts
            self.record(exc, "run")
            status = KillStatus.INCOMPLETE
        report = KillReport(
            request=self.request,
            status=status,
            halt_reason=self.reason,
            started_at=self.started,
            finished_at=self.clock.now(),
            regular_hours=self.regular_hours,
            cancelled=tuple(self.cancelled),
            kept=self.kept,
            exits=tuple(self.exits),
            queued_exit_at=self.queued_exit_at,
            final=self.final,
            errors=tuple(self.errors),
        )
        log.error(
            "kill_finished",
            status=status.value,
            exits=len(self.exits),
            cancelled=len(self.cancelled),
            errors=len(self.errors),
            queued_exit_at=self.queued_exit_at,
        )
        return report

    # --- 1. halt ---------------------------------------------------------------------------------

    def halt(self) -> None:
        """Halt the engine, revoke its order approvals, then write HALTED durably with the
        reason in force; a failure of any is recorded only."""
        self.attempt("engine.halt", lambda: self.engine.halt(self.reason))
        self.attempt("engine.revoke_orders", lambda: self.engine.revoke_orders(self.reason))
        state = self.engine.state
        reason = (state.halt_reason if state.halted else None) or self.reason
        self.attempt("store.write_halted", lambda: self.store.write_halted(reason, self.started))

    # --- the two paths ---------------------------------------------------------------------------

    def regular(self) -> KillStatus:
        """Regular hours: cancel everything, then exit with ever wider limits until flat."""
        snapshot = self.snapshot()
        attempt = 0
        while True:
            if snapshot is not None and _flat(snapshot):
                return KillStatus.FLAT
            if self.stopping():
                return self.stopped()
            if snapshot is None:
                snapshot = self.wait()
            elif snapshot.open_orders:
                snapshot = self.clear(snapshot, _every)
                if snapshot is None:
                    return self.stopped()
            else:
                sent = self.send_exit(snapshot.position, attempt)
                if sent is None:
                    snapshot = self.wait()
                else:
                    attempt += 1
                    snapshot = self.watch_exit(sent)

    def outside(self) -> KillStatus:
        """Outside regular hours: keep the stop, drop the LIMIT orders, queue the exit."""
        snapshot = self.snapshot()
        while snapshot is None:
            if self.stopping():
                return KillStatus.INCOMPLETE
            snapshot = self.wait()
        if snapshot.position == 0:
            cleared = self.clear(snapshot, _every_while_flat)
            if cleared is None:
                return KillStatus.INCOMPLETE
            if _flat(cleared):
                return KillStatus.FLAT
            snapshot = cleared  # a position appeared meanwhile: keep its stop, queue its exit
        cleared = self.clear(snapshot, _limit)
        last = cleared if cleared is not None else self.final
        if last is not None:
            self.kept = tuple(o.client_order_id for o in last.open_orders if not _limit(last, o))
            log.warning("kill_stops_kept", kept=list(self.kept))
        self.queue_exit()
        return KillStatus.QUEUED if cleared is not None else KillStatus.INCOMPLETE

    def stopped(self) -> KillStatus:
        """The deadline or the session's close came before FLAT. At the close the working exit
        is cancelled once more and the exit is queued for the next open (step 4)."""
        if self.close is None or self.clock.now() < self.close:
            log.error("kill_deadline_passed", exits=len(self.exits))
            return KillStatus.INCOMPLETE
        self.close = None  # from here on, only the deadline stops a wait
        log.error("kill_session_closed", exits=len(self.exits))
        snapshot = self.snapshot()
        while snapshot is None and not self.stopping():
            snapshot = self.wait()
        cleared = self.clear(snapshot, _limit) if snapshot is not None else None
        if cleared is not None and _flat(cleared):
            return KillStatus.FLAT
        self.queue_exit()
        return KillStatus.INCOMPLETE

    # --- the steps -------------------------------------------------------------------------------

    def send_exit(self, position: int, attempt: int) -> datetime | None:
        """Send kill exit `attempt` for `position` shares; when it was sent, or None when a
        step failed (recorded, and retried after the next wait) or the run must stop by the
        time the quote arrives."""
        quote = self.attempt("broker.quote", self.broker.quote)
        if isinstance(quote, _Failed):
            return None
        if self.stopping():  # the quote took us past the deadline or the close
            log.error("kill_exit_not_sent", reason="the deadline or the close came first")
            return None
        limit = self.attempt(
            "kill_limit_price",
            lambda: kill_limit_price(
                quote, position, attempt, collar_bp=self.collar_bp, kill=self.kill
            ),
        )
        if isinstance(limit, _Failed):
            return None
        approval = self.attempt(
            "engine.approve_flatten", lambda: self.engine.approve_flatten(reason=self.reason)
        )
        if isinstance(approval, _Failed):
            return None
        order_id = self.attempt(
            "broker.flatten", lambda: self.broker.flatten(approval, limit_price=limit)
        )
        if isinstance(order_id, _Failed):
            return None
        sent = self.clock.now()
        width = kill_collar_bp(attempt, collar_bp=self.collar_bp, kill=self.kill)
        self.exits.append(KillExit(order_id, limit, width, sent))
        log.warning(
            "kill_exit_sent",
            client_order_id=order_id,
            position=position,
            limit_price=limit,
            collar_bp=width,
            attempt=attempt,
        )
        return sent

    def watch_exit(self, sent: datetime) -> BrokerSnapshot | None:
        """Wait until a snapshot shows the account flat and clean, `requote_s` has passed since
        the exit went out, or the run must stop; the last snapshot read after `sent`, if any."""
        snapshot: BrokerSnapshot | None = None
        while not self.stopping():
            snapshot = self.wait()
            if snapshot is not None and _flat(snapshot):
                break
            if (self.clock.now() - sent).total_seconds() >= self.kill.requote_s:
                break
        return snapshot

    def clear(
        self,
        snapshot: BrokerSnapshot | None,
        select: Callable[[BrokerSnapshot, OpenOrder], bool],
    ) -> BrokerSnapshot | None:
        """Cancel until gone every open order `select` picks on the latest snapshot, those that
        appear meanwhile included: the first snapshot that shows none, or None when the run
        must stop first."""
        sent: dict[str, datetime] = {}
        while True:
            if snapshot is not None:
                targets = [o.client_order_id for o in snapshot.open_orders if select(snapshot, o)]
                if not targets:
                    return snapshot
                now = self.clock.now()
                for order_id in dict.fromkeys(targets):
                    last = sent.get(order_id)
                    due = last is None or (now - last).total_seconds() >= self.kill.requote_s
                    if due and self.cancel(order_id):
                        sent[order_id] = now
            if self.stopping():
                return None
            snapshot = self.wait()

    def cancel(self, order_id: str) -> bool:
        """Cancel `order_id` through a new approval; False when a step failed (recorded)."""
        approval = self.attempt(
            "engine.approve_cancel",
            lambda: self.engine.approve_cancel(order_id, reason=self.reason),
        )
        if isinstance(approval, _Failed):
            return False
        if isinstance(self.attempt("broker.cancel", lambda: self.broker.cancel(approval)), _Failed):
            return False
        self.cancelled.append(order_id)
        log.info("kill_order_cancelled", client_order_id=order_id)
        return True

    def queue_exit(self) -> None:
        """Queue the exit for the first open at or after the start, durably; a failed write is
        recorded only."""
        run_at = next_open(self.started, self.calendar)
        self.queued_exit_at = run_at
        self.attempt("store.queue_exit", lambda: self.store.queue_exit(run_at, self.reason))
        log.warning("kill_exit_queued", run_at=run_at)

    # --- waiting and reading ---------------------------------------------------------------------

    def stopping(self) -> bool:
        """True once the deadline, or in regular hours the session's close, has come."""
        now = self.clock.now()
        return now >= self.deadline or (self.close is not None and now >= self.close)

    def wait(self) -> BrokerSnapshot | None:
        """Sleep one poll, never past the deadline, then read a snapshot."""
        now = self.clock.now()
        until = min(now + timedelta(seconds=self.kill.poll_s), self.deadline)
        if until > now:
            self.clock.sleep_until(until)
        return self.snapshot()

    def snapshot(self) -> BrokerSnapshot | None:
        """A fresh broker snapshot, which becomes `final`; None when the broker failed."""
        snapshot = self.attempt("broker.snapshot", self.broker.snapshot)
        if isinstance(snapshot, _Failed):
            return None
        self.final = snapshot
        return snapshot

    def attempt(self, step: str, call: Callable[[], _T]) -> _T | _Failed:
        """Run one step: what it returns, or FAILED with its exception recorded (the policy in
        KillExecutor.execute)."""
        try:
            return call()
        except Exception as exc:
            self.record(exc, step)
            return _Failed.FAILED

    def record(self, exc: Exception, step: str) -> None:
        self.errors.append(_error(exc))
        log.warning("kill_step_failed", step=step, error=str(exc), error_type=type(exc).__name__)


_CENT: Final = Decimal("0.01")


def _exact(value: float) -> Decimal:
    """The decimal value `value` prints as (as quant.risk.engine compares limits)."""
    return Decimal(repr(value))


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _flat(snapshot: BrokerSnapshot) -> bool:
    return snapshot.position == 0 and not snapshot.open_orders


def _every(snapshot: BrokerSnapshot, order: OpenOrder) -> bool:
    return True


def _limit(snapshot: BrokerSnapshot, order: OpenOrder) -> bool:
    return order.kind is OrderKind.LIMIT


def _every_while_flat(snapshot: BrokerSnapshot, order: OpenOrder) -> bool:
    """Every order while the account is flat; once a position shows, only LIMIT orders, so a
    stop over shares that just arrived is kept."""
    return snapshot.position == 0 or _limit(snapshot, order)
