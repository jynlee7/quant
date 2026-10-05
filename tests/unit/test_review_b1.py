"""B1 review: the gaps the code review found in time, canonical JSON and logs (BUILD_LOG, B1)."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
import structlog

from quant.core import logs
from quant.core.calendar import TradingCalendar
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.clock import SimClock
from quant.core.errors import CalendarError, CanonicalJSONError, ClockError, ConfigError
from quant.core.ids import decision_id
from quant.core.schedule import Timing, decision_schedule
from quant_check import trace


def red(failure: type[BaseException]) -> pytest.MarkDecorator:
    """Red until the review fix lands; `failure` is how each test fails before it (P2)."""
    return pytest.mark.xfail(strict=True, reason="red:B1", raises=failure)


DID_NOT_RAISE = pytest.fail.Exception


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    out = io.StringIO()
    yield out
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "quant_redacting", False):
            root.removeHandler(handler)


@red(DID_NOT_RAISE)
@pytest.mark.spec("I1", "D2")
def test_a_decision_delay_that_would_see_the_next_bar_is_refused() -> None:
    # The shortest gap between two bar closes is the 30-minute CLOSE bar, so the next bar becomes
    # available feed_delay + 30 minutes after this one closes. tau must not reach it.
    Timing(feed_delay=timedelta(minutes=15), decision_delay=timedelta(minutes=45))
    for too_late in (46, 50, 90):
        with pytest.raises(ConfigError):
            Timing(feed_delay=timedelta(minutes=15), decision_delay=timedelta(minutes=too_late))


@red(DID_NOT_RAISE)
@pytest.mark.spec("ARCH§4.3", "ARCH§14")
@pytest.mark.parametrize("text", ["1e400", '{"x": -1e400}', "[1, 2e999]"])
def test_loading_a_number_that_overflows_to_infinity_is_refused(text: str) -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_loads(text)


@red(TypeError)
@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_zero_dimensional_arrays_and_bad_bytes_raise_canonical_errors() -> None:
    assert canonical_dumps({"x": np.array(1.5)}) == b'{"x":1.5}'
    for bad in (np.array(np.nan), np.array(1.5, dtype=np.float32)):
        with pytest.raises(CanonicalJSONError):
            canonical_dumps(bad)
    with pytest.raises(CanonicalJSONError):
        canonical_loads(b"\xff")


@red(ValueError)
@pytest.mark.spec("ARCH§4.3")
def test_a_decision_id_off_the_minute_is_a_clock_error() -> None:
    with pytest.raises(ClockError):
        decision_id(datetime(2019, 7, 2, 14, 46, 30, tzinfo=UTC))


@red(CalendarError)
@pytest.mark.spec("D1", "ARCH§5")
def test_the_calendar_reaches_well_past_this_year() -> None:
    cal = TradingCalendar.xnys()
    assert cal.next_session(date(2027, 12, 31)) == date(2028, 1, 3)
    assert decision_schedule(cal, date(2035, 12, 31))[-1].executes_at.date() == date(2036, 1, 2)


@red(AssertionError)
@pytest.mark.spec("I6")
@pytest.mark.parametrize(
    "name",
    [
        "apiSecret",
        "secretKey",
        "accessToken",
        "refresh_tokens",
        "api_keys",
        "private_key",
        "passphrase",
        "clientSecret",
    ],
)
def test_more_secret_field_names_are_recognized(name: str) -> None:
    assert logs.is_secret_field(name)


@pytest.mark.spec("I6")
@pytest.mark.parametrize(
    "name", ["trading_mode", "telegram_chat_id", "symbol", "event", "keys_seen"]
)
def test_ordinary_field_names_are_not_secret(name: str) -> None:
    assert not logs.is_secret_field(name)


@pytest.mark.spec("I6")
@pytest.mark.parametrize("name", ["api_key", "X-Api-Key", "alpaca_secret_key", "password"])
def test_the_first_secret_field_names_stay_recognized(name: str) -> None:
    assert logs.is_secret_field(name)


@red(NotImplementedError)
@pytest.mark.spec("I6")
def test_a_secret_registered_after_configuration_is_redacted(stream: io.StringIO) -> None:
    logs.configure_logging(stream=stream, secrets=[])
    secret = "late-" + "loaded-" + "secret-0042"
    logs.register_secret(secret)
    structlog.get_logger().info("settings loaded", detail=f"value {secret}")
    logging.getLogger("lib").warning("object repr %r", {"k": secret})
    assert secret not in stream.getvalue()


@red(TypeError)
@pytest.mark.spec("K6", "ARCH§14")
def test_log_timestamps_come_from_the_clock_port(stream: io.StringIO) -> None:
    tau = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)
    logs.configure_logging(stream=stream, clock=SimClock(tau))
    structlog.get_logger().info("decision")
    logging.getLogger("lib").info("library line")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert all('"timestamp": "2019-07-02T14:46:00Z"' in line for line in lines)


@red(NotImplementedError)
@pytest.mark.spec("P2")
def test_section_ids_sort_by_number() -> None:
    ids = ["ARCH§14", "ARCH§3", "ARCH§4.3", "K10", "K2", "ARCH§4"]
    assert sorted(ids, key=trace.id_sort_key) == [
        "ARCH§3",
        "ARCH§4",
        "ARCH§4.3",
        "ARCH§14",
        "K2",
        "K10",
    ]
