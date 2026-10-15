"""The settings loader: the trading account's keys and mode, read from `.env` (I6, D1).

On the Mac the keys live in the repo's `.env`, gitignored and mode 600. On the VPS systemd
injects them from `/etc/quant/engine.env` as environment variables (ARCHITECTURE §2). Either way
this module is the one place that reads them, and it hands every secret to
`quant.core.logs.register_secret` before it returns, so no log line written afterwards can carry
one, whenever logging was configured.

v1 trades paper only: a `TRADING_MODE` other than `paper` is refused (D1, I5), and so is a
missing or empty key. Errors name the variable or the line number, never a value, since the
value may be a key.

It lives in `quant.core` because the engine, control, the CLI and the data client all need it,
and core is the one package every one of them may import (K1–K4).
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from quant.core.errors import ConfigError
from quant.core.logs import register_secret
from quant.core.orders import TradingMode

DEFAULT_ENV_FILE: Final = Path(".env")

# The variables the trading account needs (.env.example). The Telegram pair is optional until
# B15 builds control.
REQUIRED: Final = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TRADING_MODE")
OPTIONAL: Final = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


@dataclass(frozen=True)
class AlpacaKeys:
    """One Alpaca account's key pair. Neither value appears in repr."""

    key_id: str = field(repr=False)
    secret_key: str = field(repr=False)


@dataclass(frozen=True)
class Settings:
    alpaca: AlpacaKeys
    trading_mode: TradingMode
    telegram_bot_token: str | None = field(default=None, repr=False)
    telegram_chat_id: str | None = None


def load_settings(
    path: Path | None = DEFAULT_ENV_FILE, *, environ: Mapping[str, str] | None = None
) -> Settings:
    """Read the settings from the dotenv file at `path`, or, with `path` None, from `environ`
    (the process environment by default), as systemd injects them.

    The file holds `NAME=value` lines; blank lines and `#` comments are skipped, and a value may
    be wrapped in single or double quotes. It must be readable by its owner only (mode 600).

    Every secret read is registered with `register_secret` before this returns, and before
    anything here can log. Raises ConfigError for a missing file, a file other users can read, a
    malformed line (by its number), a missing or empty required variable (by its name), or a
    `TRADING_MODE` other than `paper`.
    """
    values: dict[str, str] = {}
    if path is None:
        values.update(os.environ if environ is None else environ)
    else:
        try:
            with path.open() as stream:
                if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) != 0o600:
                    raise ConfigError("settings file must have mode 600")
                for number, line in enumerate(stream, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    name, sep, value = line.partition("=")
                    name, value = name.strip(), value.strip()
                    if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                        raise ConfigError(f"malformed settings line {number}")
                    if value.startswith(("'", '"')):
                        if len(value) < 2 or value[-1] != value[0]:
                            raise ConfigError(f"malformed settings line {number}")
                        value = value[1:-1]
                    register_secret(value)
                    values[name] = value
        except OSError:
            raise ConfigError("cannot read settings .env file") from None
    for name in (*REQUIRED, *OPTIONAL):
        if values.get(name):
            register_secret(values[name])
    for name in REQUIRED:
        if not values.get(name):
            raise ConfigError(f"missing or empty {name}")
    if values["TRADING_MODE"] != "paper":
        raise ConfigError("TRADING_MODE must be paper")
    return Settings(
        AlpacaKeys(values["ALPACA_API_KEY"], values["ALPACA_SECRET_KEY"]),
        TradingMode.PAPER,
        values.get("TELEGRAM_BOT_TOKEN") or None,
        values.get("TELEGRAM_CHAT_ID") or None,
    )
