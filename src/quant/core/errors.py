"""The project's error types. Each names the kind of failure, so callers can respond to it."""

from __future__ import annotations


class QuantError(Exception):
    """Base for every error the project raises on purpose."""


class ConfigError(QuantError):
    """A configuration value is missing, malformed or inconsistent."""


class ClockError(QuantError):
    """Time misused: a naive datetime, or simulated time moved backwards."""


class CalendarError(QuantError):
    """A date outside the calendar, a non-session, or a session of an unknown shape."""


class CanonicalJSONError(QuantError, ValueError):
    """A value with no canonical JSON form: NaN, infinity, a naive datetime, a non-string key."""


class DataError(QuantError):
    """Market data is missing, late, invalid or inconsistent."""


class ModelError(QuantError):
    """A model can't produce a valid output: NaN features, a bad fit, a failed provenance check."""


class PolicyError(QuantError):
    """A read whose purpose the data segment doesn't allow (ARCHITECTURE §6.3)."""


class LockboxError(PolicyError):
    """Scoring or reporting on the lockbox before the go-live test (A2, C1)."""


class TicketError(PolicyError):
    """A trial ticket the ledger can't vouch for: not a TrialTicket, built without the ledger's
    token, never issued by this ledger (copies and other ledgers' tickets), or changed since it
    was issued (K5, A2)."""


class LedgerError(QuantError):
    """The trial ledger can't do what was asked: a broken chain, a run settled twice, a hash
    that isn't SHA-256 hex, or no ledger at the path given (A2)."""
