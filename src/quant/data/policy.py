"""Data segments and purpose-tagged reads (A2, C1, ARCHITECTURE §6.3).

| Segment | Span | May be read to |
|---|---|---|
| FIT | 2016-01 to 2018-12 | operate, score, calibrate, report |
| DEV | 2019-01 to 2024-12 | operate, report; score and calibrate only with a trial ticket that covers the read (C11) |
| LOCKBOX | 2025-01 to the seal | operate only (features, the filter, refits: C1) |
| FORWARD | after the seal | operate, calibrate and report; never score |

Calibration and Kelly edge estimates use the fit-window backtest, then DEV walk-forward history,
then forward data, and skip the lockbox (ARCHITECTURE §6.3). On DEV they come from a trial's
walk-forward, and Kelly's shadow returns are strategy P&L, so they need its ticket. Gates score
only FIT and DEV. Only the go-live lockbox test may score the lockbox (B19); v1 has no way to.
Days before the data starts can't be read for any purpose but operating, since no segment holds
them.

Tickets are checked by the trial ledger the policy is bound to (B10, K5). Only the ledger holds
the private token a TrialTicket is built with, and only it knows which tickets it issued, so the
policy asks it (`LedgerPort.verify_ticket`) rather than trusting anything shaped like a ticket.
A policy with no ledger can't score or calibrate DEV at all, which keeps DEV dark by default. A
policy bound to a ledger also records there every read it refuses, so a refused attempt leaves a
ledger entry (A2, ARCHITECTURE §6.3).

This module never imports the ledger (quant.research.ledger imports it), so the data layer stays
below research.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import ClassVar, Final, Protocol, final

import structlog

from quant.core.errors import ConfigError, LockboxError, PolicyError


class Segment(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    LOCKBOX = "LOCKBOX"
    FORWARD = "FORWARD"


class Purpose(StrEnum):
    """Why a read happens. Every read through the view names one."""

    OPERATE = "operate"  # features, the filter and refits
    SCORE = "score"  # strategy P&L, trades and gate results
    CALIBRATE = "calibrate"  # calibration and Kelly edge estimates
    REPORT = "report"  # parity replays and reports


FIT_START: Final = date(2016, 1, 1)
DEV_START: Final = date(2019, 1, 1)
LOCKBOX_START: Final = date(2025, 1, 1)


@final
@dataclass(frozen=True)
class Segments:
    """Where each segment starts and ends.

    The spec fixes where FIT, DEV and the lockbox start (SPEC §2 splits, A2, C1), so they are
    class constants: no caller can move them, and Segments can't be subclassed (TypeError). A
    later lockbox would make 2025 DEV, ticketed and scored; a later DEV would make its first
    days FIT, scored without a ticket. Only the seal, `lockbox_end`, is configuration.
    """

    fit_start: ClassVar[date] = FIT_START
    dev_start: ClassVar[date] = DEV_START
    lockbox_start: ClassVar[date] = LOCKBOX_START
    lockbox_end: date | None = None  # the seal: the day before the first paper decision

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("Segments can't be subclassed: the spec fixes where they start (A2, C1)")

    def __post_init__(self) -> None:
        if self.lockbox_end is not None and self.lockbox_end < self.lockbox_start:
            raise ConfigError(f"data segments out of order: {self}")

    def spans(self) -> dict[Segment, tuple[date, date]]:
        """Each segment's first and last day. An unsealed lockbox runs to date.max."""
        spans = {
            Segment.FIT: (self.fit_start, self.dev_start - timedelta(days=1)),
            Segment.DEV: (self.dev_start, self.lockbox_start - timedelta(days=1)),
            Segment.LOCKBOX: (self.lockbox_start, self.lockbox_end or date.max),
        }
        if self.lockbox_end is not None:
            spans[Segment.FORWARD] = (self.lockbox_end + timedelta(days=1), date.max)
        return spans

    def segment_of(self, day: date) -> Segment:
        for segment, (first, last) in self.spans().items():
            if first <= day <= last:
                return segment
        raise PolicyError(f"{day} is before the data starts ({self.fit_start})")

    def overlapping(self, first: date, last: date) -> set[Segment]:
        return {s for s, (lo, hi) in self.spans().items() if lo <= last and first <= hi}


class ScoringTicket(Protocol):
    """What the policy reads from a `TrialTicket`: the trial, and the days it may score.

    Only the trial ledger issues tickets (B10, K5). A ticket's shape proves nothing: the policy
    reads these fields only from a ticket its ledger has verified.
    """

    @property
    def trial_id(self) -> str: ...

    @property
    def first_day(self) -> date: ...

    @property
    def last_day(self) -> date: ...


@dataclass(frozen=True)
class Refusal:
    """A read the policy refused, as its ledger records it (A2, ARCHITECTURE §6.3).

    - `purpose`: the purpose the read named, as text (`str(purpose)`), or "ticket" for a ticket
      request the ledger refused;
    - `first`, `last`: the days the read asked for, as passed to `DataPolicy.check`;
    - `error`, `reason`: the name of the error raised (PolicyError, LockboxError, TicketError)
      and its message;
    - `ticket`: None when no ticket was shown; otherwise the shown object's type
      (`type(ticket).__qualname__`), followed by a space and its `trial_id` when that is a str,
      as in "TrialTicket T3" or "SimpleNamespace T1".
    """

    purpose: str
    first: date
    last: date
    error: str
    reason: str
    ticket: str | None


class LedgerPort(Protocol):
    """What the policy asks of the trial ledger it is bound to: `quant.research.ledger.TrialLedger`.

    Tests of the segment rules may bind a stand-in; the ledger's own checks are B10's tests.
    """

    def verify_ticket(self, ticket: object) -> ScoringTicket:
        """`ticket`, if this ledger issued it and it is unchanged; otherwise raise TicketError."""
        ...

    def record_refusal(self, refusal: Refusal) -> None:
        """Append a refusal entry."""
        ...


@dataclass(frozen=True)
class DataPolicy:
    """The segments, and the trial ledger that vouches for tickets (None: no ticket is good).

    A ledger that names its own segments (`TrialLedger.segments`) must name these, or the policy
    refuses to be built (ConfigError): the ledger issues tickets by the same segments the policy
    checks reads by.
    """

    segments: Segments = Segments()
    ledger: LedgerPort | None = None

    def __post_init__(self) -> None:
        if type(self.segments) is not Segments:
            raise ConfigError(f"a policy needs Segments, not {type(self.segments).__qualname__}")
        bound = getattr(self.ledger, "segments", None)
        if bound is not None and bound != self.segments:
            raise ConfigError(f"the policy's {self.segments} differ from its ledger's {bound}")

    def check(
        self, purpose: Purpose, first: date, last: date, *, ticket: ScoringTicket | None = None
    ) -> None:
        """Raise unless `purpose` may read the days from `first` to `last`.

        The segment rules are the module's table. A score or calibrate read that touches DEV
        needs a ticket for the DEV days it reads (C11, A2, K5), checked in this order:

        1. no ticket: PolicyError;
        2. no ledger: PolicyError, since no ticket can be vouched for;
        3. `self.ledger.verify_ticket(ticket)` raises TicketError for a forged, foreign or
           changed ticket, and the policy lets it propagate;
        4. the verified ticket must cover the DEV days read: PolicyError otherwise.

        Every refusal, by any rule and for any purpose, is recorded in `self.ledger` when there
        is one (`record_refusal`, with a `Refusal` built as its docstring says) before the error
        is raised; the error raised is the refusal's own, so `except PolicyError` (or
        LockboxError, or TicketError) still catches it. If recording fails (a broken chain, a
        locked file), the refusal's error is raised all the same, with a note saying why the
        ledger couldn't record it, and the failure is logged. A read that is allowed records
        nothing, and a ticket shown for a read outside DEV is not checked.
        """
        try:
            self._check(purpose, first, last, ticket=ticket)
        except PolicyError as exc:
            if self.ledger is not None:
                shown = None
                if ticket is not None:
                    shown = type(ticket).__qualname__
                    trial = getattr(ticket, "trial_id", None)
                    if isinstance(trial, str):
                        shown += f" {trial}"
                refusal = Refusal(str(purpose), first, last, type(exc).__name__, str(exc), shown)
                try:
                    self.ledger.record_refusal(refusal)
                except Exception as failure:  # the read stays refused, by its own error
                    reason = f"{type(failure).__name__}: {failure}"
                    exc.add_note(f"the ledger could not record this refusal: {reason}")
                    structlog.get_logger().error(
                        "refusal_not_recorded",
                        purpose=refusal.purpose,
                        first=first.isoformat(),
                        last=last.isoformat(),
                        error=refusal.error,
                        failure=reason,
                    )
            raise

    def _check(
        self, purpose: Purpose, first: date, last: date, *, ticket: ScoringTicket | None
    ) -> None:
        try:
            purpose = Purpose(purpose)  # the string "score" names Purpose.SCORE
        except ValueError:
            raise PolicyError(
                f"{purpose!r} is no purpose the policy knows; it may read nothing"
            ) from None
        if last < first:
            raise PolicyError(f"empty span {first} to {last}")
        if purpose is Purpose.OPERATE:
            return  # market data may feed features, the filter and refits anywhere (C1)
        if first < self.segments.fit_start:
            raise PolicyError(
                f"a {purpose} read of {first} to {last} starts before the data "
                f"({self.segments.fit_start})"
            )
        touched = self.segments.overlapping(first, last)
        if Segment.LOCKBOX in touched:
            seal = self.segments.lockbox_end or "the seal"
            raise LockboxError(
                f"a {purpose} read of {first} to {last} touches the lockbox "
                f"({self.segments.lockbox_start} to {seal}); only the go-live test may (A2, C1)"
            )
        if purpose is Purpose.SCORE and Segment.FORWARD in touched:
            raise PolicyError(
                f"a score read of {first} to {last} touches forward data, which is for "
                "calibration, parity and reports only"
            )
        if purpose in (Purpose.SCORE, Purpose.CALIBRATE) and Segment.DEV in touched:
            dev_first, dev_last = self.segments.spans()[Segment.DEV]
            need = (max(first, dev_first), min(last, dev_last))
            if ticket is None:
                raise PolicyError(
                    f"a {purpose} read of {need[0]} to {need[1]} in DEV needs a trial ticket "
                    "from the ledger (C11, A2)"
                )
            if self.ledger is None:
                raise PolicyError("DEV needs a policy bound to the ticket's trial ledger")
            ticket = self.ledger.verify_ticket(ticket)
            if not ticket.first_day <= need[0] <= need[1] <= ticket.last_day:
                raise PolicyError(
                    f"trial {ticket.trial_id} may score {ticket.first_day} to "
                    f"{ticket.last_day}, not {need[0]} to {need[1]}"
                )
