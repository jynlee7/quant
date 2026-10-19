---
# Release manifest (ARCHITECTURE §9.2). Code reads only this front matter; the prose below is
# for the owner. r1's values are PLAN §5's, fixed before any data was looked at (P7).
release: r1
version: 1
created_at: 2026-10-19T00:00:00Z
features:                     # quant.features.menu.r1_features(): F1-F5 (spec §3, D3, P8)
  slot_sessions: 60
  ret: {}
  rvol: {bars: 35}
  range: {}
  volume: {}
  trend: {fast: 35, slow: 175}
hmm:                          # quant.hmm.train.r1_train_config() (spec §4)
  base_seed: 20160104
  n_restarts: 50
  candidates:
    - {n_states: 2, covariance_type: diag}
    - {n_states: 2, covariance_type: full}
    - {n_states: 3, covariance_type: diag}
    - {n_states: 3, covariance_type: full}
    - {n_states: 4, covariance_type: diag}
    - {n_states: 4, covariance_type: full}
    - {n_states: 5, covariance_type: diag}
    - {n_states: 5, covariance_type: full}
  n_iter: 1000
  tol: 0.01
  min_covar: 0.001
  bootstrap_draws: 1000
  validation_months: 12
label_thresholds:             # quant.hmm.labels.r1_thresholds() (spec §4)
  crash_v: 1.75
  stress_v: 1.25
  stress_d: -0.5
  calm_d: 0.5
playbooks:                    # each label to its own file
  CALM_UP: playbooks/CALM_UP.md
  CHOP: playbooks/CHOP.md
  STRESS: playbooks/STRESS.md
  CRASH: playbooks/CRASH.md
gate_report: null             # none until CP2, where r1 becomes trial #1 (C11)
---

# Strategy r1

r1 is the seed release. It trades SPY on hourly bars, long or flat only, and lets a Gaussian
HMM decide which of four playbooks is in charge.

- **Features:** the five transforms of spec §3: the slot-normalized return, realized
  volatility over 35 bars, the range and volume against their slot medians, and the trend
  ln(EMA35 / EMA175).
- **Model:** selected on the 2016-2018 fit window from 2 to 5 states, diagonal or full
  covariance, 50 restarts each, then refit monthly.
- **Playbooks:** CALM_UP follows the trend, CHOP buys dips back to the mean, STRESS and CRASH
  stand aside.

Every number above was fixed in PLAN §5 before any data was looked at. Changing one makes a
new release, which has to pass the gates.
