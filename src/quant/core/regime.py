"""The regime labels (spec §4). Switching, sizing, playbooks and L1's schemas all name them, so
they live in core, which every layer may import."""

from __future__ import annotations

from enum import StrEnum


class Label(StrEnum):
    """The four labels, declared in the order spec §4's rules are checked, so `tuple(Label)` is
    (CRASH, STRESS, CALM_UP, CHOP). That order also breaks ties wherever one label must be
    picked from several equally likely ones: the most cautious wins."""

    CRASH = "CRASH"
    STRESS = "STRESS"
    CALM_UP = "CALM_UP"
    CHOP = "CHOP"
