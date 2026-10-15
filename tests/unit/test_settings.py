"""B6: the settings loader reads the trading account's keys, registers them as secrets before
anything can log (I6), and refuses a missing key or any mode but paper (D1, I5)."""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from quant.core.errors import ConfigError
from quant.core.logs import REDACTED, configure_logging
from quant.core.orders import TradingMode
from quant.core.settings import load_settings

RED = pytest.mark.xfail(strict=True, reason="red:B6", raises=NotImplementedError)

# Unique to this file, and not shaped like an Alpaca key, so only registration can redact them.
KEY_ID = "settings-key-id-7f3a9c0b"
SECRET = "settings-secret-5e1d8b2a4c6f"
BOT_TOKEN = "settings-bot-token-91c4e7aa"


def write_env(directory: Path, text: str, *, mode: int = 0o600) -> Path:
    path = directory / ".env"
    path.write_text(text)
    os.chmod(path, mode)
    return path


def env_text(**overrides: str | None) -> str:
    values: dict[str, str | None] = {
        "ALPACA_API_KEY": KEY_ID,
        "ALPACA_SECRET_KEY": SECRET,
        "TRADING_MODE": "paper",
        "TELEGRAM_BOT_TOKEN": BOT_TOKEN,
        "TELEGRAM_CHAT_ID": "12345",
    }
    values.update(overrides)
    lines = ["# trading account", ""]
    lines += [f"{name}={value}" for name, value in values.items() if value is not None]
    return "\n".join(lines) + "\n"


@pytest.fixture
def stream(monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    """Logging configured before the keys are read, with no keys in the environment, so only
    the loader's registration can redact them."""
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "TRADING_MODE"):
        monkeypatch.delenv(name, raising=False)
    out = io.StringIO()
    configure_logging(stream=out)
    yield out
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "quant_redacting", False):
            root.removeHandler(handler)


@pytest.mark.spec("I6", "D1")
@RED
def test_settings_are_read_from_the_env_file(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, env_text()), environ={})
    assert settings.alpaca.key_id == KEY_ID
    assert settings.alpaca.secret_key == SECRET
    assert settings.trading_mode is TradingMode.PAPER
    assert settings.telegram_bot_token == BOT_TOKEN
    assert settings.telegram_chat_id == "12345"
    shown = repr(settings)
    assert KEY_ID not in shown
    assert SECRET not in shown
    assert BOT_TOKEN not in shown


@pytest.mark.spec("I6")
@RED
def test_quoted_values_and_comments_are_read(tmp_path: Path) -> None:
    text = (
        f"# keys\nALPACA_API_KEY=\"{KEY_ID}\"\nALPACA_SECRET_KEY='{SECRET}'\n\nTRADING_MODE=paper\n"
    )
    settings = load_settings(write_env(tmp_path, text), environ={})
    assert settings.alpaca.key_id == KEY_ID
    assert settings.alpaca.secret_key == SECRET
    assert settings.telegram_bot_token is None


@pytest.mark.spec("I6")
@RED
def test_keys_read_from_the_file_are_redacted_in_logs(tmp_path: Path, stream: io.StringIO) -> None:
    settings = load_settings(write_env(tmp_path, env_text()), environ={})
    log = structlog.get_logger("test")
    log.info("connecting", note=f"id {settings.alpaca.key_id} secret {settings.alpaca.secret_key}")
    log.info("telegram", note=f"bot{settings.telegram_bot_token}/sendMessage")
    logging.getLogger("urllib3").warning("GET /v2/stocks/bars?key=%s", settings.alpaca.secret_key)
    text = stream.getvalue()
    assert KEY_ID not in text
    assert SECRET not in text
    assert BOT_TOKEN not in text
    assert text.count(REDACTED) >= 4
    assert "connecting" in text


@pytest.mark.spec("I6", "ARCH§2")
@RED
def test_keys_injected_by_the_environment_are_registered_too(stream: io.StringIO) -> None:
    environ = {"ALPACA_API_KEY": KEY_ID + "-env", "ALPACA_SECRET_KEY": SECRET + "-env"}
    settings = load_settings(None, environ={**environ, "TRADING_MODE": "paper"})
    structlog.get_logger("test").info("loaded", note=settings.alpaca.secret_key)
    text = stream.getvalue()
    assert SECRET + "-env" not in text
    assert REDACTED in text


@pytest.mark.spec("I6", "D1")
@pytest.mark.parametrize("missing", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TRADING_MODE"])
@RED
def test_a_missing_or_empty_variable_is_refused_by_name(tmp_path: Path, missing: str) -> None:
    for text in (env_text(**{missing: None}), env_text(**{missing: ""})):
        with pytest.raises(ConfigError, match=missing) as caught:
            load_settings(write_env(tmp_path, text), environ={})
        assert SECRET not in str(caught.value)
        assert KEY_ID not in str(caught.value)


@pytest.mark.spec("D1", "I5")
@pytest.mark.parametrize("mode", ["live", "LIVE", "Paper", "paper-live"])
@RED
def test_any_trading_mode_but_paper_is_refused(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ConfigError, match="TRADING_MODE"):
        load_settings(write_env(tmp_path, env_text(TRADING_MODE=mode)), environ={})


@pytest.mark.spec("I6")
@RED
def test_a_malformed_line_is_refused_by_number_without_echoing_it(tmp_path: Path) -> None:
    text = f"ALPACA_API_KEY={KEY_ID}\n{SECRET}\nTRADING_MODE=paper\n"
    with pytest.raises(ConfigError, match="line 2") as caught:
        load_settings(write_env(tmp_path, text), environ={})
    assert SECRET not in str(caught.value)


@pytest.mark.spec("I6", "ARCH§2")
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
@RED
def test_an_env_file_other_users_can_read_is_refused(tmp_path: Path, mode: int) -> None:
    with pytest.raises(ConfigError, match="600"):
        load_settings(write_env(tmp_path, env_text(), mode=mode), environ={})


@pytest.mark.spec("I6")
@RED
def test_a_missing_env_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"\.env"):
        load_settings(tmp_path / ".env", environ={})
