"""B9b: C2's health failures and the order-error count fire the kill switch; a single late bar
doesn't (spec §10, ARCHITECTURE §0.2 C2, §8.3 and §16).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant.risk.triggers import (
    EngineHealth,
    HealthVerdict,
    KillTrigger,
    health_verdict,
    order_errors_fire,
)
from tests.support.risk import config

NOW = datetime(2018, 3, 6, 16, 0, tzinfo=UTC)
KILL = config().kill


def ago(seconds: float) -> datetime:
    return NOW - timedelta(seconds=seconds)


def judge(**fields: object) -> HealthVerdict:
    fields.setdefault("position", 300)
    fields.setdefault("last_heartbeat", ago(5))
    return health_verdict(EngineHealth(**fields), now=NOW, config=KILL)  # type: ignore[arg-type]


@pytest.mark.spec("C2", "SPEC§10")
def test_a_healthy_engine_fires_nothing() -> None:
    assert judge(starts=(ago(3_600),)) == HealthVerdict((), False)


@pytest.mark.spec("C2", "SPEC§10")
def test_a_single_late_bar_does_not_fire_the_kill() -> None:
    # The bar due at tau is late, so rule 6 has started taking the position flat a minute ago.
    # That is rule 6's job (go flat, wait for two healthy bars): no health failure, no kill.
    verdict = judge(flatten_since=ago(60), starts=(ago(7_200),))
    assert verdict == HealthVerdict((), False)
    # Only a rule-6 flatten still incomplete after 10 minutes is the engine failing.
    assert judge(flatten_since=ago(599.999)) == HealthVerdict((), False)
    assert judge(flatten_since=ago(600)) == HealthVerdict((KillTrigger.FLATTEN_INCOMPLETE,), True)


@pytest.mark.spec("C2", "SPEC§10")
def test_a_lost_heartbeat_fires_the_kill_while_a_position_is_open() -> None:
    assert judge(last_heartbeat=ago(299.999)) == HealthVerdict((), False)
    assert judge(last_heartbeat=ago(300)) == HealthVerdict((KillTrigger.HEARTBEAT_LOST,), True)
    assert judge(last_heartbeat=NOW + timedelta(seconds=1)) == HealthVerdict((), False)


@pytest.mark.spec("C2", "SPEC§10")
def test_a_crash_loop_fires_the_kill_while_a_position_is_open() -> None:
    loop = (ago(600), ago(300), NOW)
    assert judge(starts=loop) == HealthVerdict((KillTrigger.CRASH_LOOP,), True)
    assert judge(starts=(NOW, ago(300), ago(600))).kill  # any order
    assert judge(starts=(ago(600.001), ago(300), NOW)) == HealthVerdict((), False)
    assert judge(starts=(ago(300), NOW)) == HealthVerdict((), False)  # one restart isn't a loop


@pytest.mark.spec("C2", "SPEC§10")
def test_while_flat_a_health_failure_alerts_but_does_not_kill() -> None:
    verdict = judge(position=0, last_heartbeat=ago(3_600), starts=(ago(600), ago(300), NOW))
    assert verdict.failures == (KillTrigger.HEARTBEAT_LOST, KillTrigger.CRASH_LOOP)
    assert not verdict.kill


@pytest.mark.spec("C2")
def test_every_failure_is_reported_in_order() -> None:
    verdict = judge(
        last_heartbeat=ago(900), starts=(ago(500), ago(400), ago(10)), flatten_since=ago(700)
    )
    assert verdict == HealthVerdict(
        (KillTrigger.HEARTBEAT_LOST, KillTrigger.CRASH_LOOP, KillTrigger.FLATTEN_INCOMPLETE), True
    )


@pytest.mark.spec("ARCH§8.3", "ARCH§16", "SPEC§10")
def test_five_order_errors_within_10_minutes_fire_the_kill() -> None:
    five = [ago(600), ago(400), ago(300), ago(1), NOW]
    assert order_errors_fire(five, now=NOW, config=KILL)
    assert order_errors_fire(list(reversed(five)), now=NOW, config=KILL)
    assert not order_errors_fire(five[1:], now=NOW, config=KILL)
    assert not order_errors_fire([ago(600.001), *five[1:]], now=NOW, config=KILL)
    assert order_errors_fire([ago(5_000), ago(4_000), *five], now=NOW, config=KILL)
    assert not order_errors_fire([], now=NOW, config=KILL)


@pytest.mark.spec("C2", "ARCH§8.3", "ARCH§16")
def test_starts_and_order_errors_after_now_are_ignored() -> None:
    # B9b review R6: an event stamped after `now` hasn't happened yet at `now`.
    later = [NOW + timedelta(minutes=k) for k in (1, 2, 3, 4, 5)]
    assert judge(starts=tuple(later[:3])) == HealthVerdict((), False)
    assert judge(starts=(ago(300), NOW, later[0])) == HealthVerdict((), False)
    assert not order_errors_fire(later, now=NOW, config=KILL)
    assert not order_errors_fire([ago(60), ago(30), NOW, *later[:2]], now=NOW, config=KILL)
