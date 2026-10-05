"""Structured logs that never carry a secret (I6).

Every line passes a redactor that replaces known secret values, fields named like secrets, and
strings shaped like keys. Known values come from the environment when logging is configured and
from register_secret afterwards, so a key read from `.env` later is still redacted.
Standard-library loggers, such as httpx's, which prints request URLs, go through the same
redactor. A last pass scrubs the rendered line itself, which catches secrets inside tracebacks
and in objects rendered by repr; at a few lines per bar, scrubbing twice costs nothing that
matters. Timestamps come from the Clock port (K6), so a replay's logs carry simulated time.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import structlog

from quant.core.clock import Clock, WallClock

REDACTED = "[REDACTED]"

# Environment variables whose values are secrets (spec §10, .env.example).
SECRET_ENV_VARS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY")

# Field names that hold secrets. A name is split into words (snake_case, kebab-case and camelCase
# alike), plurals are made singular, and it counts as secret if a word is one of _SECRET_WORDS or
# the words run together contain one of _SECRET_PARTS. Counts such as `max_tokens` are redacted
# too: hiding a number costs less than leaking a token.
_SECRET_WORDS = frozenset({"auth", "bearer", "cookie", "signature", "token"})
_SECRET_PARTS = (
    "apikey",
    "authorization",
    "credential",
    "passphrase",
    "passw",
    "privatekey",
    "secret",
)

# Strings shaped like the keys this project holds, redacted even when the value isn't known.
_KEY_SHAPES = (
    re.compile(r"\b(?:PK|AK)[A-Z0-9]{18}\b"),  # Alpaca key ids
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),  # Anthropic keys
    re.compile(
        r"(?<!\d)\d{6,12}:AA[A-Za-z0-9_\-]{30,}"
    ),  # Telegram bot tokens, also in /bot<token>/
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"),  # bearer credentials
)

# Known values shorter than this are left alone: they would redact ordinary words.
_MIN_SECRET_LENGTH = 8


# Values registered after logging was configured, such as keys read from `.env`.
_registered: set[str] = set()


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def is_secret_field(name: str) -> bool:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    words = [_singular(w) for w in re.split(r"[_\-.\s]+", spaced.lower()) if w]
    joined = "".join(words)
    return any(w in _SECRET_WORDS for w in words) or any(part in joined for part in _SECRET_PARTS)


def register_secret(value: str) -> None:
    """Redact `value` from every log line from now on, such as a key read from `.env`."""
    if len(value) >= _MIN_SECRET_LENGTH:
        _registered.add(value)


class Redactor:
    """A structlog processor that removes secrets from event dicts and rendered lines."""

    def __init__(self, secrets: Iterable[str]) -> None:
        self._own = frozenset(s for s in secrets if len(s) >= _MIN_SECRET_LENGTH)

    def scrub(self, text: str) -> str:
        # Longest first, so a secret containing another is replaced whole.
        for secret in sorted(self._own | _registered, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        for shape in _KEY_SHAPES:
            text = shape.sub(REDACTED, text)
        return text

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, Mapping):
            return {
                key: REDACTED
                if isinstance(key, str) and is_secret_field(key)
                else self._clean(item)
                for key, item in value.items()
            }
        if type(value) in (list, tuple):
            return type(value)(self._clean(item) for item in value)
        return value

    def __call__(self, logger: Any, method_name: str, event: Any) -> Any:
        if isinstance(event, str):  # the rendered line, on the last pass
            return self.scrub(event)
        return {
            key: REDACTED
            if is_secret_field(key) and not key.startswith("_")
            else self._clean(value)
            for key, value in event.items()
        }


def _stamp_from(clock: Clock) -> Callable[[Any, str, Any], Any]:
    def stamp(logger: Any, method_name: str, event: Any) -> Any:
        event["timestamp"] = clock.now().isoformat().replace("+00:00", "Z")
        return event

    return stamp


class _RedactingHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """The one handler configure_logging installs, so a second call replaces it."""

    quant_redacting = True


def configure_logging(
    *,
    level: str = "INFO",
    stream: io.TextIOBase | None = None,
    secrets: Iterable[str] = (),
    clock: Clock | None = None,
) -> Redactor:
    """Send structlog and standard-library logs, as JSON lines, through one redactor.

    Known secrets are `secrets`, the values of SECRET_ENV_VARS in the environment, and anything
    passed to register_secret, now or later. Timestamps come from `clock` (the wall clock by
    default; the simulated clock in a backtest or replay).
    """
    known = [*secrets, *(value for name in SECRET_ENV_VARS if (value := os.environ.get(name)))]
    redactor = Redactor(known)
    timestamper = _stamp_from(clock or WallClock())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redactor,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            timestamper,
            structlog.processors.format_exc_info,
            redactor,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(sort_keys=True),
            redactor,
        ],
    )
    handler = _RedactingHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, "quant_redacting", False)]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)
    return redactor
