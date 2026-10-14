# SPEC — SPY Regime Trader

**Status:** APPROVED v1.6 (Gate 1 of 3) · 2026-10-05. v1.0 was approved as written: defaults D1–D13 confirmed, amendments A1–A4 accepted, and D11 stays a per-trade hit rate. v1.1 folds in the architecture's clarifications C1–C10, and v1.2 folds in the plan's clarifications C11–C17, all accepted 2026-10-05. v1.3 spells out how B3 reads §3: the one feature stream, how slot histories count sessions, and the exact statistics (2026-10-06). v1.4 adds Jev, TypeSafe AI's typed-decision model, as a shadow advisor that is logged next to the HMM, scored only on forward paper bars and never affects a decision: layer L1s and invariant I7 (§1), §12a, and a known risk (§16). You approved it on 2026-10-13. v1.5 is held for B9's re-base of the peak on resume, which lands with B9's merge. v1.6 spells out how B4 reads §4–§6: instability measured on the standardized rows, candidates whose restarts all fail set aside, inclusive sanity edges, the drift baseline every model records, and the filter's log-likelihood, missing bars, absorbing labels and entropy (2026-10-14). Each one is tagged where it lands.
**Process:** brainstorm ✅ → spec ✅ → architecture ✅ → plan ✅ → **test-first build** ▶ → review → ship
The spec, the architecture and the [plan](PLAN.md) are approved, and the build started 2026-10-05. Progress is in [BUILD_LOG.md](BUILD_LOG.md).

---

## 0. What you approved

A paper-trading system for **SPY on 1-hour bars through Alpaca paper**. A Gaussian HMM reads the market regime on every candle. Deterministic code picks a playbook for each regime, sizes the position and enforces hard risk limits. Every night Opus 5.5 proposes improvements, and none ships until it re-clears fixed out-of-sample gates.

Two outcomes count as success:

1. A configuration clears every gate out of sample, paper trading matches the backtest, and the go-live checklist comes back clean. Going live then becomes your decision.
2. Nothing clears the gates, so the system stays in paper. This is the gates doing their job, not the project failing.

Over long periods, buy-and-hold SPY has had a Sharpe ratio below 1.0. Gate G1 asks a long/flat system for more than 1.5 after costs, so expect outcome 2 the first time through.

### 0.1 Defaults for your placeholders (confirmed 2026-10-05)

| ID | Placeholder | Proposed default |
|---|---|---|
| D1 | venue · asset · timeframe | Alpaca **paper** · SPY · 1-hour bars, regular trading hours only |
| D2 | data feed | Free plan. Its real-time stock data is IEX-only (about 2.5% of volume). The full consolidated (SIP) feed is free once it is 15+ minutes old. To train and trade on the same feed, every decision is made **16 minutes after its bar closes, using SIP data**, in the backtest and live alike. Algo Trader Plus ($99/mo) removes the delay; switching to it means one config change and a gate re-run. |
| D3 | [one trend feature] | `ln(EMA35 / EMA175)` of the close: roughly a 1-week trend against a 5-week trend |
| D4 | [3 years] | First model fit on 2016-01 → 2018-12, then refit every month on an expanding window |
| D5 | [N] cooldown | 7 bars (one session) |
| D6 | [tolerance] | Drift thresholds in §5 |
| D7 | [$ size] manual approval | **$25,000** order notional. In live trading the order blocks until you approve it and expires at the end of the bar. In paper it is flagged and alerted but not blocked, so paper stays comparable with the backtest; a test covers the blocking path. Orders that reduce risk never wait for approval. |
| D8 | playbooks | CALM_UP trend-following · CHOP mean reversion · STRESS **stands aside** to start (Opus can propose reduced-size participation through the gates) · CRASH flat |
| D9 | direction | **Long or flat only.** No shorts and no leverage above 1×. |
| D10 | "beats both baselines" | Higher out-of-sample Sharpe **and** lower max drawdown than each baseline, after costs. A regime filter almost never beats buy-and-hold on raw return during a bull decade. |
| D11 | hit rate | Share of closed trades with net P&L > 0, with at least 100 trades. Trend-following usually wins fewer than half its trades and relies on bigger winners, so this gate leans on the CHOP book. Confirmed per trade, not per day. |
| D12 | shipping nightly changes | Once the gates pass, you approve with one command. Auto-promotion while in paper is a one-line switch. |
| D13 | risk limits | Table in §10 |

### 0.2 Amendments to your rules (accepted 2026-10-05)

| ID | Amendment | Why |
|---|---|---|
| A1 | **Fast to cut risk, slow to add it.** A switch into a lower-risk playbook (one with a lower hard cap, such as → STRESS or → CRASH) needs one bar at ≥ 0.70 and ignores the cooldown. A switch into a higher-risk playbook keeps your 3 bars plus cooldown. | With 3-bar confirmation and a 7-bar cooldown, CALM_UP can stay the active playbook for up to 10 bars (about 1.5 sessions) into a crash. Rules 4 and 5 shrink size, but they don't change the playbook. |
| A2 | **Overfitting guard for the nightly loop.** A trial ledger counts every configuration ever tested. New gate G6 requires a Deflated Sharpe Ratio ≥ 0.95 given that count. A lockbox period (2025-01 → today) is hidden from development and from the nightly loop. It is opened only at the go-live gate, at most once per candidate, and every opening is recorded in the ledger. | Testing tweaks against the same history every night slowly turns out-of-sample data into in-sample data. |
| A3 | **Overnight exposure cap of 0.5× equity.** | Hourly stops and the daily loss limit can't act on an overnight gap. On 2020-03-16 the S&P 500 fell more than 7% at the open and tripped the market-wide circuit breaker. |
| A4 | **A defined ground truth for calibration.** Hidden states are never observed. Per-state Brier scores are therefore computed against the ex-post smoothed label, which is used only in evaluation and never for decisions. An outcome check backs this up: the HMM's one-bar-ahead predictive interval must cover actual returns at its stated rate. | Without a ground-truth label, a per-state Brier score has nothing to score against. |

### 0.3 Your keys

When we reach setup, you will add your keys to `.env` yourself. I'll create `.env.example` with the variable names and make sure `.env` is gitignored. **Please don't paste keys into the chat.** The transcript is stored on disk, so anything pasted there ends up in a log.

---

## 1. Layers: the core invariant

| Layer | Runs | May | May never |
|---|---|---|---|
| **L1 Opus 5.5** | Nightly, as a separate process and Unix user, with only `ANTHROPIC_API_KEY` in its environment | Propose features (from a tested menu), playbooks, strategy.md edits and HMM configs; audit fit reports; review sessions | Hold broker keys; place or cancel orders; read or write risk limits; grade or promote its own proposals; run inside the trading loop |
| **L2 HMM** | Every candle | Emit filtered state probabilities, next-bar state probabilities, entropy and predictive log-likelihood | See a future candle; decide or size anything |
| **L1s Jev shadow** (v1.4) | Every decision bar, after the decision is recorded, as a separate process under L1's Unix user, with only `TYPESAFE_API_KEY` in its environment | Answer three fixed questions about the next bar (§12a); have its answers logged and scored against the HMM's | Hold broker keys; place or cancel orders; read risk limits or the databases; affect any decision, size or order; be called on a backtest, a replay, DEV or the lockbox |
| **L3 Code** | Every candle, at all times | Own thresholds, switching, sizing, risk vetoes, every order, the kill switch, and promotions through the gates | Be overridden by L1 or L2 output |

Each invariant is enforced by a test that fails the build:

| ID | Invariant | Enforced by |
|---|---|---|
| I1 | No decision depends on a candle that is not yet available at decision time | T-NL1 truncation test (§6) |
| I2 | The decision path never calls a smoother or Viterbi (`predict`, `predict_proba`, `decode`, `score_samples`, or our own `eval.smoother`) | Static AST scan of the decision-path modules |
| I3 | Nothing in `llm/` or the playbook prose can be imported by the risk engine | Import-graph test |
| I4 | Every order passes the risk engine | The broker adapter accepts only a `RiskApprovedOrder`, and only the risk engine can construct one |
| I5 | The live endpoint is unreachable unless the go-live gate passes | Adapter constructor test |
| I6 | Secrets never appear in code, logs, the database or LLM prompts | Secret scan plus a test of the redacting log filter |
| I7 | No decision-path module, and neither `research` nor `harness`, imports `quant.shadow` or reads `shadow.db` (v1.4) | Import-graph test (ARCHITECTURE K7) plus a scan for the `shadow.db` path |

---

## 2. Market, venue and data

- **Instrument:** SPY. **Venue:** Alpaca paper (`paper-api.alpaca.markets`). Long or flat only.
- **Bars:** built in-house from SIP 1-minute bars, regular hours only, anchored at 09:30 ET. A full session has 7 bars: six 60-minute bars and a 15:30–16:00 bar. Half-days follow the exchange calendar.
- **Timestamp contract:** every bar stores `t_open`, `t_close` and `available_at = t_close + feed_delay` (15 minutes on the free plan). A decision at time τ may only use rows with `available_at < τ`, where τ = `t_close` + 16 minutes.
- **Closing bar:** the decision for the 15:30–16:00 bar executes at the next session's open, and only if it reduces exposure.
- **Execution reference:** the live IEX quote (free and real-time) sets the order's price collar. It is never used as a feature.
- **Prices:** raw, unadjusted. Back-adjusted series rewrite past prices using later corporate actions, which is a subtle form of lookahead. The backtest credits dividends on positions held through an ex-dividend date.
- **History:** everything available from 2016-01-01, about 2,700 sessions and 19,000 hourly bars.
- **Splits:** development out-of-sample is 2019-01-01 → 2024-12-31, which covers the COVID crash, the 2022 bear market and the 2023–24 rally. The lockbox is 2025-01-01 → present, which includes the April 2025 selloff (A2). It closes the day before the first paper decision, and that date goes into config and never moves. Bars inside the lockbox may feed features, the filter and refits, but no strategy P&L, trade or calibration result inside it is computed before the go-live lockbox test. Data after the close is *forward*: paper trades it and the nightly loop reviews it (C1).
- **Stale data:** if the bar expected at τ is missing or more than 5 minutes late, rule 6 applies and the system goes flat. A decision missed because the engine was down also counts as stale: on restart, the missed bars are replayed without trading and count as unhealthy (C6).

## 3. Features (the state engine)

Five numeric features per bar, computed only from rows with `available_at < τ`:

| # | Name | Definition | Why this form |
|---|---|---|---|
| F1 | ret | `r_t / σ_slot`, where `r_t = ln(C_t / C_{t-1})` and `σ_slot` is the RMS of returns in the same time slot over the prior 60 sessions | Opening bars carry the overnight gap and are naturally wilder. Scaling by slot stops the HMM from learning the clock as a regime. |
| F2 | rvol | `ln sqrt(Σ r²)` over the last 35 bars | 35 bars is 5 sessions, so every slot gets equal weight |
| F3 | range | `ln((H_t − L_t) / C_{t-1})` minus the log median of the same slot over the prior 60 sessions | Range relative to what is normal for that slot |
| F4 | volume | `ln(V_t / median volume in the same slot over the prior 60 sessions)` | Volume ratio relative to what is normal for that slot |
| F5 | trend | `ln(EMA35(C) / EMA175(C))` | D3 |

- **One stream (B3):** one stream computes the features over every hourly bar from 2016-01, and fits and live trading share it. A fit cuts its window from those rows, so both start from the same EMA seed.
- **Slot histories (B3):** sessions are numbered in the order the stream reads them, and a half day counts as one. "The same slot over the prior 60 sessions" means that slot's bars in sessions k−60 to k−1, for a bar in session k, or k−N to k−1 when the menu sets N sessions (PLAN §5). A half day has no 12:30, 13:30 or 14:30 bar, so for the N sessions after one, those slots hold N−1 values.
- **Statistics (B3):** σ_slot is the RMS, `sqrt(mean r²)`, with no mean removed. The median of an even count is the mean of the two middle values, and F3 takes the log of the median, not the median of the logs. Each EMA uses `α = 2/(n+1)` for its span n and starts at the first close.
- **Standardization:** the z-score parameters are fit on each model's training window at refit time, frozen, and applied forward until the next refit. They are never recomputed with data from after the refit date. They are the mean and the population standard deviation (ddof 0) of the warm rows whose `available_at` lies in [start, end) (B3).
- **Warm-up:** the first 60 sessions are dropped, whatever N is (B3). A missing or NaN feature at τ counts as a model error, so rule 6 applies.
- **Determinism:** features are pure float64 functions, so identical inputs give bit-identical outputs.
- **Feature changes:** Opus can only choose from a menu of pre-built, tested transforms with bounded parameters. A brand-new feature is a code change and goes through the normal build process: failing test first, then review, then your approval.

## 4. HMM

- `hmmlearn.GaussianHMM` on Python 3.12, because hmmlearn 0.3.3 ships no wheels for your default Python 3.14.
- **Candidates:** K ∈ {2, 3, 4, 5} × covariance ∈ {diag, full}.
- **Restarts:** 50 per candidate, with seeds `base_seed + i` and `base_seed` fixed in config. The run with the highest training log-likelihood wins. A warning fires if fewer than 3 restarts land within 0.1% of the best, which signals an unstable optimum. The 0.1% is taken of the best log-likelihood of the *standardized* rows, so the features' units can't change it (v1.6).
- **Selection:** each candidate gets two scores: BIC on the fit window, and mean one-step-ahead predictive log-likelihood on a validation block. The validation block is the last 12 months of the fit window, with the model fit on the rest. "Close" means within 10 BIC points, or within one standard error of the best validation log-likelihood (session-block bootstrap). The pick is the simplest candidate that is close to the best on both scores. If no candidate is, the pick is the simpler of the two winners. The chosen candidate is then refit on the full window. A candidate whose restarts all fail, on either fit, is set aside like a rejected one and listed in the report; selection fails only when no candidate is left (v1.6).
- **Sanity rejects:** a candidate is rejected if any state has under 2% occupancy, an expected duration under 2 bars, or a covariance condition number above 10⁶. Each edge is inclusive and compared on the decimal values of the numbers: exactly 2%, exactly 2 bars or exactly 10⁶ passes (v1.6).
- **Report for every model version:** the transition matrix. For each state: annualized mean return and volatility, expected duration `1 / (1 − a_kk)` in bars and in sessions, occupancy and label. Plus the BIC and validation log-likelihood table for every candidate.
- **Auto-labels:** the thresholds are fixed in config before any out-of-sample run. Define `v = σ_state / σ_all` and `d = μ_state / σ_state`, both annualized and computed from raw returns weighted by filtered probability over the fit window. Rules are checked in order:

  | Order | Label | Rule |
  |---|---|---|
  | 1 | CRASH | v ≥ 1.75 and d < 0 |
  | 2 | STRESS | v ≥ 1.25, or d ≤ −0.5 |
  | 3 | CALM_UP | v < 1.25 and d ≥ 0.5 |
  | 4 | CHOP | everything else |

  Several hidden states can share a label. A label's probability is the sum of its states' probabilities. Switching, sizing and the dashboard all work on labels. A model with no crash-like state simply has no CRASH label.

## 5. Model lifecycle: refit and drift

- **Refit:** every 30 calendar days on an expanding window. K and the covariance type stay fixed; re-selecting K is a gated change, not a routine refit. The previous model's parameters seed one of the restarts.
- **Label matching:** the Hungarian algorithm pairs old and new states by the symmetric KL divergence between their Gaussians. Each new state inherits its match's label.
- **Drift:** any one of these freezes new entries and sends an alert. Open positions keep their exits and stops.
  - a matched transition probability moves by more than 0.10;
  - a matched state's expected duration changes by more than 50%;
  - a matched state's mean moves more than 0.5 units on any feature, measured in the previous model's standardized units;
  - a matched state's rule-derived label differs from the label it inherited;
  - in live trading, the 35-bar rolling mean of the predictive log-likelihood falls below the 1st percentile of its distribution on the validation block. That distribution is the means of every full 35-bar window of the block's log-likelihoods under the validation fit (§4). Every model records its own, a refit included, which fits its validation block for this alone. Both sides are densities of the raw rows, so the two models' scalers need no reconciling (v1.6).
- **Failed refit:** a refit that fails a drift check is stored but not activated. The current model keeps running, with entries frozen (C5).
- **Unfreeze:** either the next refit passes every check, or you acknowledge the drift manually (logged). The backtest simulates only the first path.
- **Versioning:** every model, playbook and strategy.md is an immutable version behind a `current` pointer. Rollback is one command. Refits, promotions and rollbacks take effect at the next session open, never mid-session. An urgent rollback is the kill switch, then the rollback, then a resume (C7).

## 6. Filtering without lookahead

- **Our own forward filter**, in log space and normalized: `p_t ∝ b(x_t) ⊙ (p_{t−1} A)`. Emission log-densities are computed with scipy from the model's means and covariances, not from hmmlearn internals.
- **Outputs on every bar:** filtered state and label probabilities; next-bar probabilities `q_t = p_t A`; normalized label entropy `H_t ∈ [0, 1]`; the predictive log-likelihood `log p(x_t | x_{<t})`; and the expected remaining duration of the active label.
- **Clarifications (v1.6):**
  1. The predictive log-likelihood is the density of the raw row, so it includes the scaler's Jacobian, −Σ log std.
  2. A missing or NaN bar is absent from the filter: no transition is taken for it, and rule 6 handles the decision.
  3. A model with a reachable label it can never leave, other than a label covering every state, is a model error when its filter is built.
  4. Entropy is normalized by ln(the number of labels in the model).
- **Model swap:** a new model's filter is initialized by running it over its own fit window up to the swap time, so it only ever sees past data.
- **What hmmlearn does:** `predict_proba` and `score_samples` return *smoothed* posteriors, and `predict` and `decode` return Viterbi paths. All four are banned from the decision path (I2). Smoothed posteriors exist only in `eval/`, for calibration and for the nightly review of wrong state calls. EM training uses forward-backward internally, but only on the fit window; that is estimation on past data, not a decision.

| Test | What it proves |
|---|---|
| T-NL1 truncation invariance | For 200 random cut times τ, every decision at or before τ is identical whether computed on (a) data up to τ, (b) the full history, or (c) data up to τ followed by garbage (NaN, ±1e9, shuffled bars). It covers features, the scaler, the filter, refits, switching, sizing, risk and orders. |
| T-NL2 model provenance | Every decision records its model version, and that model's fit window ends before the decision time. |
| T-NL3 filter ≠ smoother | On synthetic data, our filter differs from hmmlearn's posteriors at interior points and matches them at the final point. This shows we are not calling the smoother. |

## 7. Playbooks

- There is one file per label: `playbooks/CALM_UP.md`, `CHOP.md`, `STRESS.md` and `CRASH.md`. Code reads only the YAML front matter, which is validated against a strict schema that rejects unknown keys. The prose body is for you.
- **Strategy primitives** (deterministic, tested, long or flat):
  - `trend_following`: enter when the fast EMA is above the slow EMA and the close confirms; exit on the cross back.
  - `mean_reversion`: enter when the close's z-score against an N-bar mean is ≤ −entry_z; exit when it is ≥ −exit_z.
  - `flat`: hold no position.
- **Required fields:** strategy, bounded params, entry, exit, `stop_loss` (ATR multiple, mandatory), `take_profit` (ATR multiple or none), `max_size` (fraction of equity, clamped to the label's hard cap), version, created_at, and `invalidation`. Invalidation conditions are typed and come from a fixed menu: `label_prob_below`, `close_below_ema`, `bars_in_trade_exceeds` and `trade_drawdown_exceeds`.
- **Invalidation** exits the position and blocks new entries under that playbook until a fresh takeover re-confirms the label.
- **CRASH** is flat in code. Its hard cap is 0, so no playbook can trade it.
- Opus writes and revises playbooks. The harness validates the schema, runs the gates and promotes. Opus never grades its own work.

## 8. Switching engine

Parameters: takeover at 0.70 · persistence of 3 bars · cooldown of 7 bars · early de-risk at 0.25, which caps the size multiplier at 0.5 · uncertainty gap of 0.15, which sets size to 0.

**Cold start:** every run starts flat with no active playbook. That applies to each backtest segment, the paper start and the lockbox test, so the first takeover always needs the full 3 bars (C4).

Every bar is evaluated in this order:

1. **Health (rule 6):** a model error, NaN features or stale data sets the target position to 0 and blocks entries. Normal operation resumes after 2 consecutive healthy bars.
2. **Takeover (rules 1–3):** a challenger label takes over once it has been the top label with P ≥ 0.70 for 3 consecutive bars and the cooldown has expired. Under A1, de-risking switches need only 1 bar and skip the cooldown.
3. **Uncertainty (rule 5):** if the top two label probabilities are within 0.15 of each other, the size multiplier is 0.
4. **Early de-risk (rule 4):** if next-bar P(STRESS) + P(CRASH) > 0.25, the size multiplier is at most 0.5.
5. **Log:** every switch, veto and multiplier is logged with all label probabilities, next-bar probabilities, entropy, the rule IDs and a reason.

Rules 4–6 act on the bar that triggers them. They never wait for hysteresis.

## 9. Sizing

```
target = min(cap[label], playbook.max_size, 0.25·f*) × P̃(active) × (1 − H_t) × m_rule4 × m_rule5
```

- **f\*** (the Kelly fraction) is max(0, μ_LCB) / σ². It uses the playbook's own past per-bar returns within that label: the fit-window backtest for the first model, then the accumulated walk-forward history. μ_LCB is the mean minus one standard error. With no demonstrated edge, the size is 0.
  - These are **shadow returns**: the per-bar return the active playbook's signal would have earned at 1.0×, net of modeled costs. They are counted on bars where its label is active and the signal is in a position, whether or not a position was actually taken. With fewer than 100 such bars, f\* = 0. Actual fills would lock a playbook at size zero after one weak stretch, because a playbook sized at zero earns no new history (C12).
- **P̃(active)** is the filtered probability of the active label after the calibration check below.
- **The size is 0** in CRASH, when uncertain, while halted, while drift-frozen (for new entries), and whenever the playbook has no entry signal.
- **Deadband:** the position is resized only when the gap between target and current is at least 10% of equity. Reductions forced by rules 4–6 or by risk limits always execute.
- **Calibration** (A4) must pass before P̃ can scale size above a floor:
  - per label: Brier score, Brier skill against the base rate, a 10-bin reliability curve, and ECE, all measured against the ex-post smoothed label;
  - outcome check: how often actual returns fall inside the one-bar-ahead predictive interval. It takes the central 90% interval for F1 over the latest 1,000 out-of-sample bars and passes when coverage is between 87.6% and 92.4%, the 99% binomial band. A failure caps every label at 25% of its cap (C13);
  - on your fills: the same metrics restricted to bars with a paper fill. They count once a label has at least 30 fills; until then, the backtest calibration stands in and is flagged as such;
  - pass means ECE ≤ 0.05 and Brier skill > 0. A failing label is recalibrated with isotonic regression fit on past out-of-sample data only. If it still fails, it is sized at no more than 25% of its cap;
  - a verdict needs at least 1,000 out-of-sample bars, about 7 months, and until then every label counts as failing. The 35 bars before the as-of time are left out, because their smoothed labels haven't settled (C13).

## 10. Risk engine: hard limits in code

The defaults assume the $100k paper account. They live in `config/risk.yaml`, which the L1 process can neither read nor write. Changing it requires a restart and a git commit, and it triggers an alert.

| Limit | Default |
|---|---|
| Max position | 1.0× equity gross; no shorts |
| Label caps | CALM_UP 1.0× · CHOP 0.5× · STRESS 0.25× · CRASH 0 · uncertain 0 |
| Overnight exposure (A3) | 0.5× equity, enforced by the last decision that can fill before the close (around 15:46 ET) |
| Daily loss | 2% of start-of-day equity: flatten, and no entries until the next session |
| Max drawdown | 10% from peak equity: kill switch. The gate allows 15%; this leaves headroom. |
| Single order | At most 0.5× equity, so a full-size entry scales in over two bars |
| Manual approval | D7 |
| Order rate | More than 20 order intents in a day: halt and alert. An intent is one rebalance, however many re-price attempts it takes. A separate runaway guard halts at 60 raw broker submissions in a day (C3). |
| Order types | Entries and discretionary exits are marketable limit orders with a ±10 bp collar around the live quote, never bare market orders. Protective stops are broker-side stop-market orders, so they always fill. Kill-switch exits widen the collar if they are still unfilled after 30 s. |
| Broker-side stops | Every open position carries a stop order at the broker, so protection survives an engine crash or a network loss. The backtest simulates these stops on 1-minute bars. |
| Reconciliation | Broker and internal positions must match before every order. A mismatch halts trading and sends an alert. |

- **Between bars:** a monitor checks daily loss and drawdown every minute on the freshest price: the IEX real-time trade when live, the SIP 1-minute close in the backtest. IEX prices feed only the order collar and this monitor, never a feature (C8).
- **Single choke point:** every order goes through `RiskEngine.check(order, state)`. It returns a `RiskApprovedOrder`, the only type the broker adapter accepts.
- **Kill switch:** it can be triggered from the CLI, the dashboard button, or `/kill` on Telegram (from your chat ID only). It also fires automatically on max drawdown, a reconciliation mismatch, repeated order errors, or a health failure while a position is open. A health failure here means the engine itself failing: a lost heartbeat, a crash loop, or a rule-6 flatten still incomplete after 10 minutes. A single late bar is rule 6's job and doesn't halt trading (C2). It cancels all orders, closes all positions, persists `HALTED` and alerts you. Resuming requires the local CLI and a typed confirmation. **Remote channels can only reduce risk.**
- **Paper lock:** the broker adapter refuses to connect to a live URL unless `TRADING_MODE=live` is set *and* the go-live checklist (§15) validates. v1 ships with live trading locked.
- **Keys:** stored only in `.env`, which is gitignored and `chmod 600`. For live trading, keys must have trading scope only, with withdrawals confirmed impossible. The system never asks for or enters passwords or 2FA codes.
- **Data is data:** headlines and feed contents reach L1 only inside delimited data blocks. L1 has no tools, and its output is schema-validated.

## 11. Backtest and gates

- **One codepath:** the backtester drives the live decision code with a simulated clock and a simulated broker.
- **Fills:** at the open of the first SIP 1-minute bar at or after τ, plus slippage. This is the same 16-minute delay as live.
- **Costs:** $0 commission; SEC and FINRA TAF fees on sells; 2 bp of slippage per side, with sensitivity runs at 5 bp.
- **Walk-forward:** monthly expanding-window refits across the development period. The lockbox stays untouched (A2).
- **Trials (A2):** before trial #1, nothing computes strategy P&L on the development period; the build uses synthetic data and the fit window. The seed release r1 becomes trial #1. After that, a scored development run of a candidate is a new trial whenever its decision records differ from every earlier scored run, including when they differ only because the code changed. Re-running the same configuration on the same code reuses its trial. Baselines and cost-sensitivity runs are yardsticks, not candidates, so they are not trials. A bug fix chosen after seeing development results is a look at the data too, so the re-run that follows it counts. Counting it costs little, because the Deflated Sharpe penalty grows slowly with the trial count (C11).
- **Reports**, for the full system and for each label: return, Sharpe, Sortino, max drawdown, Calmar, hit rate, number of trades, average win and loss, exposure, turnover, costs paid, time in label, equity curve and switch log.
- **Trades:** a trade runs from flat to flat. Scale-ins, resizes and partial exits inside it belong to that trade, and its net P&L includes every fill and fee. G3's hit rate and its 100-trade minimum count trades this way (C15).
- **After the 10% drawdown kill,** the backtest stays flat for the rest of the run, as live trading would until you resume, and the report marks the kill. Simulating an automatic resume would invent a decision you haven't made (C16).
- **Baselines** use the same costs and period: buy-and-hold SPY, walk-forward trend-following and walk-forward mean reversion. Each static family re-picks its parameters every month from a fixed grid, by the best Sharpe on its training window, and holds 1.0× when in position. G5 requires beating each of the three on both Sharpe and max drawdown. Beating both static families is at least as strict as beating whichever turns out better, and it removes the question of when "better" gets decided (C14).

| Gate | Requirement (development out-of-sample, after costs) |
|---|---|
| G1 | Sharpe (daily returns, annualized) > 1.5 |
| G2 | Max drawdown < 15% |
| G3 | Hit rate > 55% with at least 100 closed trades |
| G4 | Newey–West t-stat of the mean daily return > 2.0 |
| G5 | Beats buy-and-hold and both walk-forward static strategies (D10, C14) |
| G6 | Deflated Sharpe ≥ 0.95 given the trial ledger (A2) |

The winner is written to `strategy.md` along with its config hash, model version, playbook versions and full gate report.

## 12. Nightly loop (L1)

- Runs at 18:00 ET on trading days, after the daily report.
- **Inputs, passed as data:** the day's decisions, fills and switches; wrong state calls (the filtered label at decision time compared with the ex-post smoothed label); the daily report; the current playbooks, strategy.md and feature config; the gate history; and the trial ledger.
- **Outputs**, as schema-validated JSON (anything else is discarded):
  - one rule per losing trade, appended to `journal/losses.md`;
  - at most 3 change proposals (features from the menu, playbook parameters, or strategy.md), each with a hypothesis and a diff. To keep each night's changes small, no parameter moves more than ±20% unless you approve a bigger move. A proposal that exceeds the limit is parked before testing, so it spends no trial, until you allow it with one command (C10).
- **Harness:** each proposal adds an entry to the trial ledger, then runs a full walk-forward on the development period and gates G1–G6. If it passes and beats the current champion, it is queued for your approval (D12). Beating the champion means a higher out-of-sample Sharpe after costs on the development period; a champion that fails any gate is beaten by any candidate that passes all six (C9). Once approved, it is promoted as a new version that can be rolled back.
- **Model:** `claude-opus-5-5` via the Anthropic API. With the server-side refusal fallback on, another model may answer on a night Opus declines, and each proposal records which model wrote it (AD6).

## 12a. Shadow advisor (Jev) (v1.4)

Jev is TypeSafe AI's typed-decision model. It takes a text state and named, typed questions, and returns typed answers in one pass. Here it runs as a shadow: its answers are logged next to the HMM's and scored, and nothing on the way to a decision, a size or an order reads them (I7). Its vendor claims are unchecked (§16), so B20's spike checks them first.

- **When:** every decision bar of paper trading (the FORWARD segment), after the engine has recorded the decision. Backtests, replays, DEV and the lockbox never call Jev. A cloud model has probably seen that market history in training, so a score there would quietly use the future, and its answers may not repeat run to run.
- **Questions:** one call per bar, with three questions.

  | Name | Type | Question |
  |---|---|---|
  | `next_label` | `Choice` of CALM_UP, CHOP, STRESS, CRASH | The next bar's regime label |
  | `next_up` | `Noul`, a probability in [0, 1] | The next bar's return (F1) is above 0 |
  | `next_stress` | `Noul` | The next bar is STRESS or CRASH |

- **The bundle** holds only features F1–F5 at τ, the filtered and next-bar label probabilities, the entropy, the active playbook and the position fraction. It holds no dates or times, no ticker and no secrets, and a secret scan must pass before it is written.
- **Answers** are checked against a strict schema. Extra keys, a `Noul` outside [0, 1], a label outside the four, or an answer over the size cap make the answer `missing`.
- **Late and missing:** an answer that arrives at or after the next bar's `available_at` is `late`. It is stored and never scored. An outage, a timeout or a rejected answer is `missing`. Neither is an engine health event; the daily report counts both.
- **Scoring** runs each evening in `eval/`, the only place smoothing runs:
  - the Brier score and log-loss of `next_up` and `next_stress` against what happened. Log-loss clips probabilities to [1e-6, 1 − 1e-6];
  - the HMM's forecasts of the same events, scored the same way on the same bars. P(next F1 > 0) comes from its Gaussian mixture weighted by the next-bar state probabilities, and P(next bar is STRESS or CRASH) from the next-bar label probabilities;
  - `next_label` accuracy. The outcome of `next_label` and `next_stress` is the A4 smoothed label;
  - one line in the daily report, with the counts of late and missing answers.

  A bar is scored once, on the evening its outcome is known, and its score is never revised.
- **Promotion test**, fixed now, before any data exists. Jev may be **proposed** as an input only when all three hold:
  1. the test runs at one of its two looks, fixed now: at 500 scored forward bars, and once more at 1,000 only if the first look fails. It never runs at any other time, and after a second failure Jev is never proposed;
  2. a one-sided Diebold–Mariano test on the per-bar `next_stress` Brier differences (Jev minus HMM), over the bars where both are scored, with a Newey–West standard error, finds Jev's Brier score lower at p < 0.025. That is 0.05 split across the two looks (Bonferroni), so repeated looks can't manufacture a pass;
  3. Jev's late-or-missing rate is under 2% of the bars it was asked about.

  Passing allows only a proposal. Adopting it is a separate approval from you, with a spec change. An adopted input could only cut risk: it could lower the size multiplier, never raise it and never open a position. Its responses would be recorded as decision inputs, so replays stay exact.

## 13. Dashboard, alerts and reports

- **Dashboard:** a local web app. It is read-only apart from the kill and approve buttons. It shows:
  - filtered label probabilities over time, the current label and its expected duration, next-bar label probabilities and entropy;
  - the current model's transition matrix and state table;
  - the active playbook;
  - the latest decision with its rule trace;
  - position, P&L and equity;
  - risk-limit usage, drift, health and calibration status, and pending approvals.

  Updates arrive by server-sent events on every bar, plus P&L every minute. The app binds to 127.0.0.1; on the VPS you reach it through an SSH tunnel or Tailscale.
- **Telegram:** an alert for every fill, error, state switch, drift warning, kill-switch event and approval request. It accepts inbound `/kill` and `/status` from your chat ID only.
- **Daily report** at 16:20 ET: current state, time in each state (today and over 30 days), trades, P&L per state, win rate, largest loss, calibration score per label, drift status and the champion's gate status.

## 14. Deployment

- Develop and paper-trade on this Mac first, then move to an Ubuntu LTS VPS.
- Python 3.12 via uv. Three systemd units: `quant-engine` (Restart=always, with a watchdog heartbeat), `quant-dashboard` (Restart=always) and `quant-nightly.timer` (runs as the `quant-llm` user). Plus chrony for time sync, a nightly SQLite backup and log rotation. ARCHITECTURE §2 refines this list: a third service, `quant-control`, handles Telegram and keeps a kill path working when the engine is down (AD1), and the evening, harness, backup and integration-test work runs as scheduled jobs.
- **On restart:** the engine reconciles with the broker, rebuilds its filter state deterministically from stored bars, and resumes. Any mismatch halts trading and sends an alert.
- The step-by-step deployment walkthrough is part of the ship phase.

## 15. Go-live gate

`go_live/checklist.yaml` must validate before the code will construct a live broker client. Each item is machine-checked where possible, and you sign it:

1. **Paper matches the backtest:** replaying the paper period through the backtester gives 100% decision parity; the code is deterministic, so any mismatch is a bug. Realized slippage is no worse than modeled. The P&L gap is at most the larger of 10% of absolute paper P&L and 0.5% of equity. The evidence covers at least 3 months and 100 fills. Only decisions made on the VPS from a clean, tagged commit count, starting from the last change to decision-path code (`data`, `features`, `hmm`, `decide`, `risk`, `execution`, `engine`). A promotion doesn't restart the count; a code change does, because the parity replay must reproduce every decision with one codebase (C17).
2. **Filtered probabilities only:** T-NL1–3 and I2 pass on the release commit.
3. **Kill switch fired in testing:** a paper integration test opens a position, fires the kill switch, and verifies within 60 s that the account is flat, has no open orders and is `HALTED`. The test must have run within the last 7 days.
4. **No hard limit delegated to a model:** I3 and I4 pass, and `risk.yaml` is unchanged and owned by the engine user.
5. **Labels are stable:** no unresolved label flips across walk-forward and paper refits.
6. **What breaks it:** stress replays of March 2020, 2022, the April 2025 lockbox period, a synthetic −10% overnight gap, a 30-minute data outage and a broker API outage. In each, the worst-case loss stays inside the 10% drawdown kill.
7. **Lockbox:** the candidate clears G1–G3 and G5 on 2025-01 → today. G4 and the 100-trade minimum are waived because the period is shorter.

The checklist ends with a section titled **WHAT COULD BLOW UP THIS ACCOUNT?** Any answer that is not clean means no go-live.

## 16. Known risks so far

| Risk | Mitigation in this spec |
|---|---|
| Overnight and weekend gaps skip past hourly stops and the daily loss limit | A3 overnight cap, label caps, broker-side stops |
| A regime breaks faster than hysteresis can react | A1; rules 4–6 act immediately |
| The nightly loop overfits the backtest | A2: trial ledger, Deflated Sharpe, lockbox |
| Silent lookahead from smoothed posteriors, back-adjusted prices, a bar-timestamp off-by-one, or a scaler fit on all data | T-NL1–3, I2, raw prices, the `available_at` contract |
| Backtest and live use different data feeds | D2: one feed for both |
| The engine crashes, or the broker or data feed goes down, with a position open | Broker-side stops; stale data means flat; caps bound the damage |
| Alpaca paper fills are simulated, with no market impact or queue position | Paper validates logic and plumbing, not execution quality, so slippage assumptions stay conservative |
| Manual approvals in live trading make fills depend on how fast you respond | Factored into go-live item 1 |
| Pattern-day-trader rules on small live accounts | Check before going live |
| You loosen limits under pressure | Changing `risk.yaml` needs a restart and a commit, and sends an alert |
| Keys leak | `.env` only, log redaction, and keys are never pasted into chat |
| Jev's vendor claims (calibration, latency, determinism, data retention) come from secondary blogs and haven't been checked (v1.4) | Shadow only (I7); B20's spike checks the claims first and records them in VERIFY.md; promotion needs its own forward evidence and your approval (§12a) |

## 17. Build process: how "done" is defined

- **Phases:** spec → architecture → plan, each needing your approval, then test-first build → review → ship.
- **Every module:** a failing test is committed first. Then the implementation goes in until the test is green and the full suite is green. Then it is reviewed against this spec's IDs (D, A, I, T, G) and committed. Any module can be rolled back with `git revert`.
- **Runtime rollback:** models, playbooks and strategy.md are versioned, and one command repoints `current`.
- **Harness:** AgenKit is not installed. It is a paid, closed-source package that would sit next to your broker keys. These same gates run directly in Claude Code instead.
