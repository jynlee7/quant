"""Structured logs that never carry a secret (I6).

Every line passes a redactor that replaces known secret values (read from the environment),
fields named like secrets, and strings shaped like keys. Standard-library loggers, such as
httpx's, which prints request URLs, go through the same redactor. A last pass scrubs the
rendered line itself, which catches secrets inside tracebacks and objects rendered by repr.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

import structlog

REDACTED = "[REDACTED]"

# Environment variables whose values are secrets (spec §10, .env.example).
SECRET_ENV_VARS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY")

# Field names that hold secrets, matched word by word so `max_tokens` isn't caught by `token`.
_SECRET_WORDS = frozenset(
    {
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
        "token",
    }
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


def is_secret_field(name: str) -> bool:
    words = [w for w in re.split(r"[_\-.\s]+", name.lower()) if w]
    return any(w in _SECRET_WORDS for w in words) or ("api", "key") in pairwise(words)


class Redactor:
    """A structlog processor that removes secrets from event dicts and rendered lines."""

    def __init__(self, secrets: Iterable[str]) -> None:
        unique = {s for s in secrets if len(s) >= _MIN_SECRET_LENGTH}
        self._secrets = sorted(unique, key=len, reverse=True)

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
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


class _RedactingHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """The one handler configure_logging installs, so a second call replaces it."""

    quant_redacting = True


def configure_logging(
    *, level: str = "INFO", stream: io.TextIOBase | None = None, secrets: Iterable[str] = ()
) -> Redactor:
    """Send structlog and standard-library logs, as JSON lines, through one redactor.

    Known secrets are `secrets` plus the values of SECRET_ENV_VARS in the environment.
    """
    known = [*secrets, *(value for name in SECRET_ENV_VARS if (value := os.environ.get(name)))]
    redactor = Redactor(known)
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

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
