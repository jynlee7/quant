"""B4: auto-labels from v and d, and label probabilities (spec §4)."""

from __future__ import annotations

import math

import numpy as np
import pydantic
import pytest

from quant.core.errors import ModelError
from quant.core.regime import Label
from tests.support.hmm import label_table

N = 252 * 7  # bars a year: the annualization the labels use


def _below(x: float) -> float:
    return float(np.nextafter(x, -math.inf))


def _above(x: float) -> float:
    return float(np.nextafter(x, math.inf))


# (v, d, label) at and around every boundary of spec §4's table, r1 thresholds.
BOUNDARIES = [
    # CRASH: v ≥ 1.75 and d < 0
    (1.75, _below(0.0), Label.CRASH),
    (1.75, -3.0, Label.CRASH),
    (_below(1.75), _below(0.0), Label.STRESS),  # not CRASH, but v ≥ 1.25
    (1.75, 0.0, Label.STRESS),  # d < 0 is strict
    (2.5, 0.8, Label.STRESS),  # high volatility while rising is STRESS, not CALM_UP
    # STRESS: v ≥ 1.25, or d ≤ -0.5
    (1.25, 0.0, Label.STRESS),
    (1.25, 0.9, Label.STRESS),
    (_below(1.25), 0.0, Label.CHOP),
    (1.0, -0.5, Label.STRESS),
    (1.0, _above(-0.5), Label.CHOP),
    (0.3, -2.0, Label.STRESS),
    # CALM_UP: v < 1.25 and d ≥ 0.5
    (_below(1.25), 0.5, Label.CALM_UP),
    (0.5, 0.5, Label.CALM_UP),
    (0.5, _below(0.5), Label.CHOP),
    (1.25, 0.5, Label.STRESS),
    # CHOP: everything else
    (1.0, 0.0, Label.CHOP),
    (0.0, 0.0, Label.CHOP),
]


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize(("v", "d", "label"), BOUNDARIES)
def test_the_label_rules_hold_at_their_exact_boundaries(v: float, d: float, label: Label) -> None:
    from quant.hmm.labels import classify, r1_thresholds

    assert classify(v, d, r1_thresholds()) is label


@pytest.mark.spec("SPEC§4")
def test_the_thresholds_come_from_config() -> None:
    from quant.hmm.labels import LabelThresholds, classify, r1_thresholds

    r1 = r1_thresholds()
    assert (r1.crash_v, r1.stress_v, r1.stress_d, r1.calm_d) == (1.75, 1.25, -0.5, 0.5)
    wide = LabelThresholds(crash_v=2.0, stress_v=1.5, stress_d=-1.0, calm_d=1.0)
    assert classify(1.75, -0.1, wide) is Label.STRESS
    assert classify(1.25, 0.0, wide) is Label.CHOP
    assert classify(1.25, 1.0, wide) is Label.CALM_UP
    assert classify(1.0, -0.5, wide) is Label.CHOP
    assert classify(2.0, -0.1, wide) is Label.CRASH

    for bad in (
        {"crash_v": 1.25, "stress_v": 1.25, "stress_d": -0.5, "calm_d": 0.5},
        {"crash_v": 1.0, "stress_v": 1.25, "stress_d": -0.5, "calm_d": 0.5},
        {"crash_v": 1.75, "stress_v": 0.0, "stress_d": -0.5, "calm_d": 0.5},
        {"crash_v": 1.75, "stress_v": 1.25, "stress_d": 0.0, "calm_d": 0.5},
        {"crash_v": 1.75, "stress_v": 1.25, "stress_d": -0.5, "calm_d": 0.0},
    ):
        with pytest.raises(pydantic.ValidationError):
            LabelThresholds(**bad)
    for v, d in ((math.nan, 0.0), (1.0, math.inf), (-0.1, 0.0)):
        with pytest.raises(ModelError):
            classify(v, d, r1)


@pytest.mark.spec("SPEC§4")
def test_v_and_d_come_from_returns_weighted_by_filtered_probability() -> None:
    from quant.hmm.labels import state_stats

    rng = np.random.default_rng(7)
    n, k = 3000, 3
    returns = rng.normal(0.0002, 0.004, size=n)
    raw = rng.random((n, k)) + np.array([0.0, 0.5, 1.0])
    probs = raw / raw.sum(axis=1, keepdims=True)

    v, d = state_stats(returns, probs)
    assert v.shape == d.shape == (k,)
    assert v.dtype == d.dtype == np.float64
    sigma_all = returns.std()
    for s in range(k):
        mu = np.average(returns, weights=probs[:, s])
        sigma = math.sqrt(np.average((returns - mu) ** 2, weights=probs[:, s]))
        assert v[s] == pytest.approx((sigma * math.sqrt(N)) / (sigma_all * math.sqrt(N)), rel=1e-12)
        assert d[s] == pytest.approx((mu * N) / (sigma * math.sqrt(N)), rel=1e-10)


@pytest.mark.spec("SPEC§4")
def test_a_fit_window_labels_its_states() -> None:
    """Four states with known returns, one-hot probabilities: each gets its rule's label."""
    from quant.hmm.labels import label_states, r1_thresholds, state_stats

    rng = np.random.default_rng(8)

    def block(m: int, mu: float, sigma: float) -> np.ndarray:
        z = rng.standard_normal(m)
        return mu + sigma * (z - z.mean()) / z.std()  # exactly mean mu and std sigma

    calm = block(3000, 0.0005, 0.001)  # low vol, d = 21
    chop = block(3000, 0.0, 0.0015)  # low vol, no drift
    stress = block(1500, 0.0, 0.004)  # v ≈ 1.38
    crash = block(500, -0.001, 0.008)  # v ≈ 2.77, d = -5.25
    blocks = [calm, chop, stress, crash]
    returns = np.concatenate(blocks)
    probs = np.zeros((len(returns), 4))
    start = 0
    for s, b in enumerate(blocks):
        probs[start : start + len(b), s] = 1.0
        start += len(b)

    v, d = state_stats(returns, probs)
    np.testing.assert_allclose(v[0], calm.std() / returns.std(), rtol=1e-12)
    np.testing.assert_allclose(d[3], math.sqrt(N) * crash.mean() / crash.std(), rtol=1e-10)

    table = label_states(returns, probs, r1_thresholds())
    assert [s.label for s in table.states] == [Label.CALM_UP, Label.CHOP, Label.STRESS, Label.CRASH]
    assert [s.v for s in table.states] == [float(x) for x in v]
    assert [s.d for s in table.states] == [float(x) for x in d]
    assert table.thresholds == r1_thresholds()


@pytest.mark.spec("SPEC§4")
def test_state_stats_refuses_what_it_cant_measure() -> None:
    from quant.hmm.labels import state_stats

    returns = np.linspace(-0.01, 0.01, 10)
    probs = np.tile([0.5, 0.5], (10, 1))
    never = np.tile([1.0, 0.0], (10, 1))  # state 1 has no weight
    for r, p in (
        (returns, never),
        (returns[:9], probs),
        (np.full(10, 0.25), probs),  # sigma_all = 0 exactly
        (np.where(returns > 0, np.nan, returns), probs),
        (returns, -probs),
        (returns[:1], probs[:1]),
    ):
        with pytest.raises(ModelError):
            state_stats(r, p)


@pytest.mark.spec("SPEC§4")
def test_the_probabilities_of_states_that_share_a_label_add_up() -> None:
    from quant.hmm.labels import label_probs

    table = label_table([Label.CHOP, Label.CALM_UP, Label.CHOP, Label.STRESS])
    assert table.labels == (Label.STRESS, Label.CALM_UP, Label.CHOP)
    assert table.states_of(Label.CHOP) == (0, 2)
    assert table.states_of(Label.CRASH) == ()
    assert table.label_of(3) is Label.STRESS

    p = np.array([0.1, 0.25, 0.4, 0.25])
    probs = label_probs(p, table)
    assert list(probs) == [Label.STRESS, Label.CALM_UP, Label.CHOP]
    assert probs[Label.CHOP] == 0.1 + 0.4
    assert probs[Label.CALM_UP] == 0.25
    assert probs[Label.STRESS] == 0.25
    assert Label.CRASH not in probs  # no crash-like state, no CRASH label
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ModelError):
        label_probs(np.array([0.5, 0.5]), table)


@pytest.mark.spec("SPEC§4", "SPEC§6")
def test_entropy_is_normalized_by_the_models_labels() -> None:
    from quant.hmm.labels import label_entropy, label_probs, top_label

    three = label_table([Label.CHOP, Label.CALM_UP, Label.CHOP, Label.STRESS])
    even = label_probs(np.array([1 / 6, 1 / 3, 1 / 6, 1 / 3]), three)
    assert label_entropy(even, three) == pytest.approx(1.0, abs=1e-12)
    sure = label_probs(np.array([0.0, 1.0, 0.0, 0.0]), three)
    assert label_entropy(sure, three) == 0.0
    some = label_probs(np.array([0.1, 0.6, 0.1, 0.2]), three)
    expected = -(0.2 * math.log(0.2) + 0.6 * math.log(0.6) + 0.2 * math.log(0.2)) / math.log(3)
    assert label_entropy(some, three) == pytest.approx(expected, abs=1e-12)

    one = label_table([Label.CHOP, Label.CHOP])
    assert label_entropy(label_probs(np.array([0.5, 0.5]), one), one) == 0.0

    # Ties go to the most cautious label, in rule order.
    assert top_label({Label.CALM_UP: 0.5, Label.CHOP: 0.5}) is Label.CALM_UP
    assert top_label({Label.CHOP: 0.4, Label.CRASH: 0.4, Label.STRESS: 0.2}) is Label.CRASH
    assert top_label({Label.CHOP: 0.7, Label.CRASH: 0.3}) is Label.CHOP
    with pytest.raises(ModelError):
        top_label({})


@pytest.mark.spec("SPEC§4", "SPEC§6")
def test_the_expected_remaining_duration() -> None:
    from quant.hmm.labels import remaining_duration

    a = np.array(
        [
            [0.90, 0.06, 0.04, 0.00],
            [0.05, 0.80, 0.05, 0.10],
            [0.10, 0.10, 0.75, 0.05],
            [0.20, 0.00, 0.10, 0.70],
        ]
    )
    table = label_table([Label.CHOP, Label.CALM_UP, Label.CHOP, Label.STRESS])
    p = np.array([0.3, 0.2, 0.4, 0.1])

    # One state: spec §4's expected duration, 1 / (1 - a_kk), whatever p is.
    assert remaining_duration(p, a, table, Label.CALM_UP) == pytest.approx(5.0, rel=1e-12)
    assert remaining_duration(p, a, table, Label.STRESS) == pytest.approx(1 / 0.3, rel=1e-12)

    # Two states: from the filtered mix within them, until the chain first leaves both.
    s = [0, 2]
    w = p[s] / p[s].sum()
    expected = float(w @ np.linalg.inv(np.eye(2) - a[np.ix_(s, s)]) @ np.ones(2))
    assert remaining_duration(p, a, table, Label.CHOP) == pytest.approx(expected, rel=1e-12)

    one = label_table([Label.CHOP] * 4)
    assert remaining_duration(p, a, one, Label.CHOP) is None
    with pytest.raises(ModelError):
        remaining_duration(p, a, table, Label.CRASH)
    closed = a.copy()
    closed[1] = [0.0, 1.0, 0.0, 0.0]  # CALM_UP can never be left
    with pytest.raises(ModelError):
        remaining_duration(p, closed, table, Label.CALM_UP)
