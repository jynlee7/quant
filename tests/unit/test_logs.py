"""B1: secrets never reach a log line (I6), whether they arrive as a known value, a field named
like a secret, or a string shaped like a key."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest
import structlog

from quant.core.logs import REDACTED, configure_logging

pytestmark = pytest.mark.xfail(strict=True, reason="red:B1", raises=NotImplementedError)


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    out = io.StringIO()
    yield out
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "quant_redacting", False):
            root.removeHandler(handler)


@pytest.mark.spec("I6")
def test_a_known_secret_is_redacted_everywhere(
    stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "s3cr3t-" + "v4lu3-" + "0123456789abcdef"
    monkeypatch.setenv("ALPACA_SECRET_KEY", secret)
    configure_logging(stream=stream)
    log = structlog.get_logger("test")
    log.info("connecting", note=f"using {secret}", nested={"deep": [secret]})
    try:
        raise RuntimeError(f"auth failed for {secret}")
    except RuntimeError:
        log.exception("broker call failed")
    logging.getLogger("httpx").warning("request carried %s", secret)
    try:
        raise ValueError(secret)
    except ValueError:
        logging.getLogger("alpaca").exception("library error")
    text = stream.getvalue()
    assert secret not in text
    assert text.count(REDACTED) >= 5
    assert "connecting" in text
    assert "library error" in text


@pytest.mark.spec("I6")
def test_fields_named_like_secrets_are_redacted_whatever_their_value(stream: io.StringIO) -> None:
    configure_logging(stream=stream, secrets=[])
    structlog.get_logger().info(
        "config",
        api_key="value-one",
        alpaca_secret_key="value-two",
        telegram_bot_token="value-three",
        password="value-four",
        authorization="Bearer value-five",
        trading_mode="paper",
    )
    text = stream.getvalue()
    for value in ("value-one", "value-two", "value-three", "value-four", "value-five"):
        assert value not in text
    assert "paper" in text


@pytest.mark.spec("I6")
def test_key_shaped_strings_are_redacted_even_when_unknown(stream: io.StringIO) -> None:
    configure_logging(stream=stream, secrets=[])
    alpaca_id = "P" + "K" + "Z9" * 9
    anthropic = "sk-" + "ant-" + "api03-" + "x7" * 30
    token = "123456789" + ":" + "AA" + "b4" * 17
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    structlog.get_logger().info("seen", a=alpaca_id, b=anthropic, c=url)
    logging.getLogger("httpx").info("HTTP Request: GET %s", url)
    text = stream.getvalue()
    for value in (alpaca_id, anthropic, token):
        assert value not in text
    assert "api.telegram.org" in text
