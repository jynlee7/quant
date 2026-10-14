"""B4: selection's rules and the sanity rejects on hand-built numbers (spec §4)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from structlog.testing import capture_logs

from quant.core.errors import ClockError, ModelError
from tests.support.hmm import (
    Floats,
    columns_of,
    identity_scaler,
    params_of,
    sample,
    three_full,
)
from tests.support.train import (
    r1_candidates,
    ref_bic,
    ref_bootstrap,
    ref_n_params,
    ref_sanity,
    ref_se,
)


def _candidate(k: int, covariance_type: str) -> Any:
    from quant.hmm.select import Candidate

    return Candidate(n_states=k, covariance_type=covariance_type)  # type: ignore[arg-type]


# --- Free parameters, simplicity and BIC ---------------------------------------------------


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize(("k", "covariance_type"), r1_candidates())
def test_free_parameters_count_pi_a_means_and_covariances(
    k: int, covariance_type: str, d: int
) -> None:
    from quant.hmm.select import n_free_params

    expected = ref_n_params(k, d, covariance_type)
    assert n_free_params(k, d, covariance_type) == expected  # type: ignore[arg-type]
    assert _candidate(k, covariance_type).n_free_params(d) == expected


@pytest.mark.spec("SPEC§4")
def test_free_parameters_of_three_states_over_two_features() -> None:
    from quant.hmm.select import n_free_params

    # π 2 + A 6 + means 6, then covariances 6 (diag) or 9 (full)
    assert n_free_params(3, 2, "diag") == 20
    assert n_free_params(3, 2, "full") == 23
    with pytest.raises(ModelError):
        n_free_params(0, 2, "diag")


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize("d", [1, 2, 5])
def test_simplest_means_fewest_parameters_then_fewer_states_then_diag(d: int) -> None:
    from quant.hmm.select import simplicity_key

    grid = [_candidate(k, c) for k, c in r1_candidates()]
    ordered = sorted(grid, key=lambda c: simplicity_key(c, d))
    expected = sorted(
        r1_candidates(), key=lambda kc: (ref_n_params(kc[0], d, kc[1]), kc[0], kc[1] == "full")
    )
    assert [(c.n_states, c.covariance_type) for c in ordered] == expected
    if d == 1:  # diag and full have the same count with one feature: diag is simpler
        assert simplicity_key(_candidate(3, "diag"), 1) < simplicity_key(_candidate(3, "full"), 1)


@pytest.mark.spec("SPEC§4")
def test_bic_is_minus_two_loglik_plus_p_log_n() -> None:
    from quant.hmm.select import bic

    assert bic(-46596.25, 23, 20000) == ref_bic(-46596.25, 23, 20000)
    assert bic(10.5, 3, 7) == -21.0 + 3 * math.log(7)
    with pytest.raises(ModelError):
        bic(float("nan"), 3, 7)
    with pytest.raises(ModelError):
        bic(-1.0, 3, 0)


# --- Restarts: the winner, ties, and the instability warning -------------------------------


@pytest.mark.spec("SPEC§4", "ARCH§7.1")
def test_restart_ties_break_to_the_lowest_index() -> None:
    from quant.hmm.select import choose_restart

    choice = choose_restart((-5.0, -3.0, -3.0, -4.0), candidate=_candidate(3, "full"))
    assert choice.best == 1
    assert choose_restart((-2.0, -2.0, -2.0), candidate=_candidate(2, "diag")).best == 0
    logliks = (-10.0, -9.5, -9.75, -9.5, -11.0)
    assert choose_restart(logliks, candidate=_candidate(2, "diag")).best == int(np.argmax(logliks))


@pytest.mark.spec("SPEC§4")
def test_the_instability_warning_fires_when_fewer_than_three_restarts_are_near_the_best() -> None:
    from quant.hmm.select import choose_restart

    # 0.1% of |-1000| is exactly 1: -1001 is within (inclusive), a hair further is not.
    beyond = float(np.nextafter(-1001.0, -math.inf))
    with capture_logs() as logs:
        choice = choose_restart(
            (-1000.0, -1001.0, beyond, -1200.0), candidate=_candidate(4, "full")
        )
    assert (choice.best, choice.near_best, choice.stable) == (0, 2, False)
    warnings = [e for e in logs if e["event"] == "hmm optimum unstable"]
    assert len(warnings) == 1
    event = warnings[0]
    assert event["log_level"] == "warning"
    assert (event["n_states"], event["covariance_type"]) == (4, "full")
    assert (event["near_best"], event["restarts"], event["best_loglik"]) == (2, 4, -1000.0)


@pytest.mark.spec("SPEC§4")
def test_three_restarts_near_the_best_is_stable_and_silent() -> None:
    from quant.hmm.select import choose_restart

    with capture_logs() as logs:
        choice = choose_restart(
            (-1001.0, -1000.0, -1000.5, -1500.0), candidate=_candidate(3, "diag")
        )
    assert (choice.best, choice.near_best, choice.stable) == (1, 3, True)
    assert not [e for e in logs if e["event"] == "hmm optimum unstable"]


@pytest.mark.spec("SPEC§4")
def test_a_single_restart_is_unstable() -> None:
    from quant.hmm.select import choose_restart

    with capture_logs() as logs:
        choice = choose_restart((-5.0,), candidate=_candidate(2, "full"))
    assert (choice.best, choice.near_best, choice.stable) == (0, 1, False)
    assert [e["event"] for e in logs] == ["hmm optimum unstable"]


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize("logliks", [(), (-1.0, float("nan")), (float("inf"),)])
def test_choose_restart_refuses_empty_or_non_finite(logliks: tuple[float, ...]) -> None:
    from quant.hmm.select import choose_restart

    with pytest.raises(ModelError):
        choose_restart(logliks, candidate=_candidate(2, "diag"))


# --- The validation block -------------------------------------------------------------------


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize(
    ("fit_end", "months", "start"),
    [
        (datetime(2019, 1, 1, tzinfo=UTC), 12, datetime(2018, 1, 1, tzinfo=UTC)),
        (datetime(2020, 3, 31, 12, tzinfo=UTC), 1, datetime(2020, 2, 29, 12, tzinfo=UTC)),
        (datetime(2024, 2, 29, 20, tzinfo=UTC), 12, datetime(2023, 2, 28, 20, tzinfo=UTC)),
        (datetime(2018, 7, 15, 3, 30, tzinfo=UTC), 18, datetime(2017, 1, 15, 3, 30, tzinfo=UTC)),
    ],
)
def test_the_validation_block_is_the_last_months_of_the_window(
    fit_end: datetime, months: int, start: datetime
) -> None:
    from quant.hmm.select import validation_start

    assert validation_start(fit_end, months) == start


@pytest.mark.spec("SPEC§4")
def test_validation_start_refuses_naive_times_and_empty_blocks() -> None:
    from quant.hmm.select import validation_start

    with pytest.raises(ClockError):
        validation_start(datetime(2019, 1, 1), 12)  # noqa: DTZ001
    with pytest.raises(ModelError):
        validation_start(datetime(2019, 1, 1, tzinfo=UTC), 0)


def _session_ids(n: int, per: int = 7) -> Any:
    return (np.arange(n, dtype=np.int64) // per + 726000).astype(np.int64)


@pytest.mark.spec("SPEC§4")
def test_validation_continues_the_training_pass_and_bootstraps_its_mean() -> None:
    from quant.hmm.filter import forward_pass
    from quant.hmm.select import bootstrap_se, validation_score

    truth = three_full()
    _, x = sample(truth, 700, 5)
    params, scaler, columns = params_of(truth), identity_scaler(truth.d), columns_of(x)
    sessions = _session_ids(700)[490:]
    score = validation_score(
        params, scaler, x[:490], x[490:], sessions, columns=columns, draws=200, seed=11
    )
    whole = forward_pass(params, scaler, x, columns=columns).loglik
    assert np.array_equal(score.loglik, whole[490:])  # bit for bit: one pass continued
    fresh = forward_pass(params, scaler, x[490:], columns=columns).loglik
    assert score.loglik[0] != fresh[0]  # not restarted from π
    assert score.mean == float(np.mean(score.loglik))
    assert score.se == bootstrap_se(score.loglik, sessions, draws=200, seed=11)


# --- The session-block bootstrap ------------------------------------------------------------


def _uneven_sessions() -> tuple[Floats, Any]:
    rng = np.random.default_rng(3)
    sizes = rng.integers(3, 8, size=60)  # half days and missing bars make sessions uneven
    sessions = np.repeat(np.arange(60, dtype=np.int64) + 730000, sizes).astype(np.int64)
    return rng.normal(-1.4, 0.9, size=len(sessions)), sessions


@pytest.mark.spec("SPEC§4")
def test_the_bootstrap_resamples_whole_sessions() -> None:
    from quant.hmm.select import bootstrap_means, bootstrap_se

    loglik, sessions = _uneven_sessions()
    means = bootstrap_means(loglik, sessions, draws=1000, seed=20160104)
    expected = ref_bootstrap(loglik, sessions, 1000, 20160104)
    assert means.shape == (1000,)
    np.testing.assert_allclose(means, expected, rtol=1e-12, atol=0)
    se = bootstrap_se(loglik, sessions, draws=1000, seed=20160104)
    assert se == float(np.std(means, ddof=1))
    assert math.isclose(se, ref_se(expected), rel_tol=1e-10)
    again = bootstrap_means(loglik, sessions, draws=1000, seed=20160104)
    assert again.tobytes() == means.tobytes()


@pytest.mark.spec("SPEC§4")
def test_the_bootstrap_ignores_order_inside_a_session() -> None:
    from quant.hmm.select import bootstrap_means, bootstrap_se

    loglik, sessions = _uneven_sessions()
    shuffled = loglik.copy()
    for sid in np.unique(sessions):
        idx = np.flatnonzero(sessions == sid)
        shuffled[idx] = shuffled[idx][::-1]
    np.testing.assert_allclose(
        bootstrap_means(shuffled, sessions, draws=300, seed=1),
        bootstrap_means(loglik, sessions, draws=300, seed=1),
        rtol=1e-12,
    )
    # Every session with the same mean and size: every draw has that mean.
    flat = np.tile(np.array([-1.0, -2.0, -3.0, -2.0]), 50)
    blocks = np.repeat(np.arange(50, dtype=np.int64), 4).astype(np.int64)
    assert bootstrap_se(flat, blocks, draws=100, seed=1) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.spec("SPEC§4")
def test_the_bootstrap_refuses_bad_blocks() -> None:
    from quant.hmm.select import bootstrap_means

    loglik, sessions = _uneven_sessions()
    with pytest.raises(ModelError):
        bootstrap_means(loglik, sessions[::-1].copy(), draws=10, seed=1)  # decreasing
    with pytest.raises(ModelError):
        bootstrap_means(loglik, np.zeros(len(loglik), dtype=np.int64), draws=10, seed=1)
    with pytest.raises(ModelError):
        bootstrap_means(loglik[:-1], sessions, draws=10, seed=1)
    with pytest.raises(ModelError):
        bootstrap_means(loglik, sessions, draws=1, seed=1)


# --- Sanity rejects ------------------------------------------------------------------------


def _good() -> dict[str, Floats]:
    truth = three_full()
    return {
        "startprob": truth.startprob.copy(),
        "transmat": truth.transmat.copy(),
        "means": truth.means.copy(),
        "covars": truth.covars.copy(),
    }


def _params(parts: dict[str, Floats]) -> Any:
    from quant.hmm.model import GaussianParams

    return GaussianParams(
        "full", parts["startprob"], parts["transmat"], parts["means"], parts["covars"]
    )


def _one_hot(counts: tuple[int, ...]) -> Floats:
    """State probabilities with n = sum(counts) rows, counts[k] of them all on state k."""
    states = np.repeat(np.arange(len(counts)), counts)
    return np.eye(len(counts), dtype=np.float64)[states]


def _check(parts: dict[str, Floats], probs: Floats) -> Any:
    from quant.hmm.select import sanity_check

    report = sanity_check(_params(parts), probs)
    expected = ref_sanity(parts["transmat"], parts["covars"], probs)
    failing = {name: [k for k, bad in enumerate(flags) if bad] for name, flags in expected.items()}
    named = {
        name: [
            k
            for k in range(3)
            if any(r.startswith(f"state {k}:") and name in r for r in report.reasons)
        ]
        for name in ("occupancy", "duration", "condition")
    }
    assert named == failing
    assert report.passed == (not any(failing.values()))
    return report


@pytest.mark.spec("SPEC§4")
def test_a_good_model_passes_every_sanity_check() -> None:
    parts = _good()
    report = _check(parts, _one_hot((40, 35, 25)))
    assert report.passed
    assert report.reasons == ()
    np.testing.assert_array_equal(report.occupancy, [0.40, 0.35, 0.25])
    np.testing.assert_allclose(report.durations, 1 / (1 - np.diag(parts["transmat"])), rtol=1e-15)
    np.testing.assert_allclose(
        report.condition, [np.linalg.cond(c) for c in parts["covars"]], rtol=1e-12
    )


@pytest.mark.spec("SPEC§4")
def test_occupancy_under_two_percent_is_rejected() -> None:
    report = _check(_good(), _one_hot((60, 39, 1)))  # state 2: 1%
    assert not report.passed
    assert len(report.reasons) == 1
    assert report.reasons[0].startswith("state 2:")
    assert _check(_good(), _one_hot((49, 49, 2))).passed  # exactly 2% passes


@pytest.mark.spec("SPEC§4")
def test_an_expected_duration_under_two_bars_is_rejected() -> None:
    parts = _good()
    parts["transmat"][0] = [0.49, 0.31, 0.20]  # 1 / (1 - 0.49) < 2
    report = _check(parts, _one_hot((40, 35, 25)))
    assert not report.passed
    assert report.reasons[0].startswith("state 0:")
    parts["transmat"][0] = [0.50, 0.30, 0.20]  # exactly 2 bars passes
    assert _check(parts, _one_hot((40, 35, 25))).passed


@pytest.mark.spec("SPEC§4")
def test_an_absorbing_state_has_infinite_duration_and_passes() -> None:
    parts = _good()
    parts["transmat"][2] = [0.0, 0.0, 1.0]
    report = _check(parts, _one_hot((40, 35, 25)))
    assert report.passed
    assert math.isinf(report.durations[2])


@pytest.mark.spec("SPEC§4")
def test_a_condition_number_over_a_million_is_rejected() -> None:
    parts = _good()
    parts["covars"][1] = np.diag([float(np.nextafter(1e6, math.inf)), 1.0])
    report = _check(parts, _one_hot((40, 35, 25)))
    assert not report.passed
    assert report.reasons[0].startswith("state 1:")
    assert report.condition[1] > 1e6
    parts["covars"][1] = np.diag([1e6, 1.0])  # exactly 1e6 passes
    assert _check(parts, _one_hot((40, 35, 25))).passed


@pytest.mark.spec("SPEC§4")
def test_every_failing_check_is_reported_in_state_order() -> None:
    parts = _good()
    parts["transmat"][2] = [0.30, 0.30, 0.40]
    parts["covars"][2] = np.diag([1.0, 1e-7])
    parts["transmat"][1] = [0.10, 0.45, 0.45]
    report = _check(parts, _one_hot((90, 9, 1)))
    assert [r.split(":")[0] for r in report.reasons] == ["state 1", "state 2", "state 2", "state 2"]
    assert ["occupancy" in report.reasons[1], "duration" in report.reasons[2]] == [True, True]
    assert "condition" in report.reasons[3]


@pytest.mark.spec("SPEC§4")
def test_sanity_check_refuses_probabilities_of_the_wrong_shape() -> None:
    from quant.hmm.select import sanity_check

    with pytest.raises(ModelError):
        sanity_check(_params(_good()), _one_hot((5, 5)))
    with pytest.raises(ModelError):
        sanity_check(_params(_good()), np.empty((0, 3)))


# --- The selection rule --------------------------------------------------------------------


def _score(k: int, c: str, bic: float, mean: float, se: float, passed: bool = True) -> Any:
    from quant.hmm.select import CandidateScore, SanityReport, ValidationScore

    reasons = () if passed else ("state 0: occupancy 0.01 < 0.02",)
    return CandidateScore(
        candidate=_candidate(k, c),
        n_params=ref_n_params(k, 2, c),
        loglik=-bic / 2,
        bic=bic,
        validation=ValidationScore(loglik=np.array([mean, mean]), mean=mean, se=se),
        sanity=SanityReport(np.ones(k) / k, np.full(k, 10.0), np.ones(k), reasons),
    )


def _pairs(candidates: Any) -> list[tuple[int, str]]:
    return [(c.n_states, c.covariance_type) for c in candidates]


@pytest.mark.spec("SPEC§4")
def test_the_pick_is_the_simplest_candidate_close_on_both_scores() -> None:
    from quant.hmm.select import select

    scores = (
        _score(2, "diag", 1000.0, -2.0, 0.125),  # close on BIC only (exactly 10 points)
        _score(3, "diag", 1005.0, -1.5, 0.125),  # close on neither
        _score(3, "full", 1000.0, -1.25, 0.125),  # close on both, each exactly at the edge
        _score(4, "full", 990.0, -1.0, 0.25),  # best on both
    )
    result = select(scores, 2)
    assert _pairs([result.best_bic, result.best_validation]) == [(4, "full"), (4, "full")]
    assert result.se == 0.25
    assert _pairs(result.close) == [(3, "full"), (4, "full")]
    assert _pairs([result.chosen]) == [(3, "full")]
    assert result.rule == "close_on_both"
    assert result.scores == scores


@pytest.mark.spec("SPEC§4")
def test_with_none_close_on_both_the_pick_is_the_simpler_winner() -> None:
    from quant.hmm.select import select

    scores = (
        _score(2, "diag", 900.0, -3.0, 0.25),  # BIC winner
        _score(3, "full", 950.0, -2.0, 0.25),
        _score(4, "full", 1000.0, -1.0, 0.25),  # validation winner
    )
    result = select(scores, 2)
    assert _pairs([result.best_bic, result.best_validation]) == [(2, "diag"), (4, "full")]
    assert result.close == ()
    assert _pairs([result.chosen]) == [(2, "diag")]
    assert result.rule == "simpler_winner"
    flipped = (
        _score(2, "diag", 1000.0, -1.0, 0.25),  # validation winner, and simpler
        _score(4, "full", 900.0, -3.0, 0.25),  # BIC winner
    )
    assert _pairs([select(flipped, 2).chosen]) == [(2, "diag")]


@pytest.mark.spec("SPEC§4")
def test_close_is_measured_with_the_best_candidates_standard_error() -> None:
    from quant.hmm.select import select

    scores = (
        _score(2, "diag", 995.0, -1.5, 1.0),  # a wide SE of its own doesn't make it close
        _score(3, "full", 990.0, -1.0, 0.25),
    )
    result = select(scores, 2)
    assert result.se == 0.25
    assert _pairs(result.close) == [(3, "full")]
    assert _pairs([result.chosen]) == [(3, "full")]


@pytest.mark.spec("SPEC§4")
def test_ties_for_best_go_to_the_simplest() -> None:
    from quant.hmm.select import select

    scores = (
        _score(4, "full", 990.0, -1.0, 0.25),
        _score(3, "diag", 990.0, -1.0, 0.25),
        _score(3, "full", 990.0, -1.0, 0.25),
    )
    result = select(scores, 2)
    assert _pairs([result.best_bic, result.best_validation]) == [(3, "diag"), (3, "diag")]
    assert _pairs(result.close) == [(3, "diag"), (3, "full"), (4, "full")]


@pytest.mark.spec("SPEC§4")
def test_candidates_that_fail_a_sanity_check_take_no_part() -> None:
    from quant.hmm.select import select

    scores = (
        _score(2, "diag", 1200.0, -3.0, 0.25),
        _score(3, "full", 1000.0, -1.25, 0.25),
        _score(4, "full", 500.0, 0.0, 0.25, passed=False),  # best on both, but rejected
    )
    result = select(scores, 2)
    assert _pairs([result.best_bic, result.best_validation, result.chosen]) == [(3, "full")] * 3
    assert result.rule == "close_on_both"
    assert (4, "full") not in _pairs(result.close)


@pytest.mark.spec("SPEC§4")
def test_selection_fails_when_every_candidate_is_rejected_or_none_given() -> None:
    from quant.hmm.select import select

    with pytest.raises(ModelError):
        select((_score(2, "diag", 1.0, -1.0, 0.1, passed=False),), 2)
    with pytest.raises(ModelError):
        select((), 2)
    with pytest.raises(ModelError):
        select((_score(2, "diag", 1.0, -1.0, 0.1), _score(2, "diag", 2.0, -1.0, 0.1)), 2)
