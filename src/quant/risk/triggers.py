"""What fires the kill switch (spec §10, ARCHITECTURE §8.3 and §16, C2).

The kill switch fires on a command (the CLI, the dashboard, or `/kill` on Telegram from your chat
id) or automatically:

- max drawdown: `RiskEngine.mark` returns `MarkOutcome.kill` (the minute monitor relays it);
- a reconciliation mismatch (quant.risk.reconcile, run by the minute monitor and check 2);
- repeated order errors: `order_errors_fire`;
- a position with no covering broker-side stop for `stop_missing_s` (the minute monitor);
- an engine health failure while a position is open (C2): `health_verdict`.

C2 defines a health failure as a failure of the engine itself: a lost heartbeat, a crash loop,
or a rule-6 flatten still incomplete after 10 minutes. A single late bar is rule 6's job (go
flat, then wait for two healthy bars) and fires nothing here: a late bar shows up only as the
rule-6 flatten it starts, which fires only once it has been incomplete for
`flatten_incomplete_s`.

Everything here is pure: it judges what it is given at the `now` it is given. Watching the
heartbeat and counting starts is quant-control's (B15); counting order errors is the order
layer's (B13).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import structlog

from quant.core.clock import ensure_utc
from quant.risk.limits import KillConfig

log = structlog.get_logger()


class KillTrigger(StrEnum):
    """Why the kill switch fired. The kill executor puts it in the halt reason and the alert."""

    MANUAL = "manual"  # the CLI, the dashboard or Telegram /kill
    DRAWDOWN = "drawdown"  # max drawdown (MarkOutcome.kill)
    RECONCILIATION = "reconciliation"  # broker and engine disagree
    ORDER_ERRORS = "order_errors"  # `order_errors` broker errors within the window
    STOP_MISSING = "stop_missing"  # a position without a covering stop for `stop_missing_s`
    HEARTBEAT_LOST = "heartbeat_lost"  # C2
    CRASH_LOOP = "crash_loop"  # C2
    FLATTEN_INCOMPLETE = "flatten_incomplete"  # C2: rule 6's flatten still incomplete


HEALTH_TRIGGERS: tuple[KillTrigger, ...] = (
    KillTrigger.HEARTBEAT_LOST,
    KillTrigger.CRASH_LOOP,
    KillTrigger.FLATTEN_INCOMPLETE,
)
"""C2's engine health failures, in the order `health_verdict` reports them."""


@dataclass(frozen=True)
class EngineHealth:
    """What quant-control knows about the engine at one moment (C2). Times are UTC.

    - `position`: SPY shares held, from the broker (quant-control reads the broker itself, so
      this holds while the engine is down);
    - `last_heartbeat`: the engine's latest heartbeat;
    - `starts`: when the engine (re)started, any order, any age; old ones are ignored;
    - `flatten_since`: when rule 6 began taking the position flat, while that flatten is still
      incomplete; None when rule 6 isn't flattening, or its flatten has completed.
    """

    position: int
    last_heartbeat: datetime
    starts: tuple[datetime, ...] = ()
    flatten_since: datetime | None = None


@dataclass(frozen=True)
class HealthVerdict:
    """C2's judgment of one EngineHealth.

    - `failures`: the health failures that hold, in HEALTH_TRIGGERS order;
    - `kill`: True when there is a failure *and* a position is open (`position != 0`): the kill
      switch fires. While flat a failure only alerts (B15).
    """

    failures: tuple[KillTrigger, ...]
    kill: bool


def health_verdict(health: EngineHealth, *, now: datetime, config: KillConfig) -> HealthVerdict:
    """Judge `health` at `now` (C2). Each failure is inclusive at its threshold:

    - HEARTBEAT_LOST when `now - health.last_heartbeat >= heartbeat_lost_s`;
    - CRASH_LOOP when at least `crash_loop_starts` of `health.starts` lie in
      `[now - crash_loop_window_s, now]`;
    - FLATTEN_INCOMPLETE when `health.flatten_since` is set and
      `now - health.flatten_since >= flatten_incomplete_s`.

    A late bar fires nothing by itself (module docstring). ClockError for a naive `now` or a
    naive time in `health`; times after `now` count as at `now` (a heartbeat from the future is
    fresh, a start from the future is in the window)."""
    now = ensure_utc(now)
    failures: list[KillTrigger] = []
    if _elapsed(now, health.last_heartbeat) >= config.heartbeat_lost_s:
        failures.append(KillTrigger.HEARTBEAT_LOST)
    if _count_within(health.starts, now, config.crash_loop_window_s) >= config.crash_loop_starts:
        failures.append(KillTrigger.CRASH_LOOP)
    if (
        health.flatten_since is not None
        and _elapsed(now, health.flatten_since) >= config.flatten_incomplete_s
    ):
        failures.append(KillTrigger.FLATTEN_INCOMPLETE)
    verdict = HealthVerdict(tuple(failures), bool(failures) and health.position != 0)
    if failures:
        log.warning(
            "kill_health_failure",
            failures=[f.value for f in failures],
            position=health.position,
            kill=verdict.kill,
        )
    return verdict


def order_errors_fire(errors: Sequence[datetime], *, now: datetime, config: KillConfig) -> bool:
    """True when at least `config.order_errors` of the broker order errors at `errors` lie in
    `[now - order_error_window_s, now]` (ARCHITECTURE §8.3 and §16: 5 errors within 10 minutes
    fire the kill, which keeps retrying while the broker-side stop protects the position).
    `errors` may be in any order and include old ones, which are ignored; an error after `now`
    counts as at `now`. ClockError for a naive time."""
    now = ensure_utc(now)
    return _count_within(errors, now, config.order_error_window_s) >= config.order_errors


def _elapsed(now: datetime, then: datetime) -> float:
    """Seconds from `then` to `now`; a `then` after `now` counts as at `now` (0)."""
    return max((now - ensure_utc(then)).total_seconds(), 0.0)


def _count_within(times: Sequence[datetime], now: datetime, window_s: float) -> int:
    """How many of `times` lie in `[now - window_s, now]`, a time after `now` counting as at
    `now`. Every time is checked for a timezone, old ones included."""
    return sum(1 for t in times if _elapsed(now, t) <= window_s)
