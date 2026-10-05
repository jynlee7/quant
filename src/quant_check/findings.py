"""One failed check, printed as `FAIL <rule> <location> <message>`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Finding:
    """A broken rule at a location. `rule` is a spec, architecture or plan id, or a tool name."""

    rule: str
    location: str
    message: str

    def render(self) -> str:
        return f"FAIL {self.rule} {self.location} {self.message}"
