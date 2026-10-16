# BUILD LOG: SPY Regime Trader

One entry per finished step (PLAN §2, rule 7): what was built, the tests added, run times, deviations and open items. Newest first. Each step is a branch `bNN-name` with a red commit and a green commit, merged to `main` and tagged `bNN-done`.

## B7 · Model lifecycle: refit, drift, calibration (done 2026-10-15, tag `b07-done`)

Built on `b07-lifecycle` in two parts, each with its own red, green and review:

- **B7a:** the dividend supplement, refit, label matching, drift, ops.db's pointers and freeze, and the live drift monitor.
- **B7b:** calibration, its artifact, and the refit and calibration layers in the truncation harness.

Claude agents wrote the reds and B7b's review fixes. Codex wrote B7a's green, its review fixes and B7b's green; its sandbox can't write `.git`, so the orchestrator committed them. A Claude agent stood in for Codex once it reached its usage limit.

**Built**

- **The dividend supplement** (your call at CP1):
  - `config/dividend_supplement.json` holds State Street's three SPY rows the feed misses (2016-03-18, 2018-06-15, 2026-09-18), with the source URL, retrieval date and the spreadsheet's SHA-256. The loader checks its schema strictly, and a test pins its rows.
  - The backfill stores them after the feed under the same stamp. They never revise a stored dividend, under NOW or HISTORY: a differing row is reported as a conflict.
  - **Result:** `var/market.db` holds 43 SPY dividends for 2016–2026, each one version, each known at the end of its ex-date.
- **`store.ops` (ops.db):** append-only pointers (model, calibration, release), refit events and freeze events.
  - Every table refuses UPDATE, DELETE and REPLACE.
  - Stamps never go back.
  - Activation and rollback take effect at the next session open (`TradingCalendar.next_open`), never mid-session (C7).
  - `frozen(t)` replays the events in order. A drifted refit freezes entries from its recording; the next passing refit unfreezes them at its open; a manual acknowledgement unfreezes them at once and is logged (C5).
- **`jobs.refit`:**
  - **Cutoff:** 17:00 New York.
  - **Cadence:** due 30 New York days after the later of the active model's last data day and the latest refit's cutoff, whatever that refit's outcome.
  - **Window:** expanding from the model's fit start, read through the view at the cutoff with the whole-session check that `first_model` has (`jobs._rows`, shared). K and the covariance type stay fixed.
  - **Seeding:** restart 0 of the full-window fit starts from the previous model, converted through raw units into the new scaler's space.
  - **Outcomes:** ACTIVATED (both pointers at the next open), FROZEN (stored, not activated), or REJECTED (nothing stored, a warning). A ModelError from training, matching or drift is REJECTED.
- **`hmm.match`:** Hungarian matching on symmetric KL in raw units. Labels are inherited, with the rule-derived ones kept beside them.
- **`hmm.drift`:**
  - **The four refit rules:** a transition probability moving more than 0.10, a duration changing more than 50%, a mean moving more than 0.5 in the previous model's units, and a derived label differing from the inherited one. They compare on decimal values, so they fire just past their thresholds and not at them.
  - **The live monitor:** full 35-bar windows, `fsum` means, firing strictly below the validation threshold, and reset by a model swap.
- **Calibration:**
  - **Metrics** (`eval.calibration`): Brier score, Brier skill, 10-bin reliability and ECE against the hard smoothed label, under the model active at each bar.
  - **The outcome check:** the F1 90% one-bar-ahead interval.
  - **Recalibration:** isotonic, judged out of sample.
  - **The sample rules (C13):** listed under Deviations.
  - **The artifact:** `quant.core.calibration`, which sizing reads as plain data (K3). It is `calibration.json` plus `knots.npz`, content-addressed, and its pass flags are re-derived from its numbers on load.
  - **`jobs.calibrate`:** builds the calibration from ops.db's model spans. It skips the lockbox and reads DEV only with a trial ticket (ARCH §6.3). A refit that activates builds its calibration as of its cutoff.
- **The truncation harness** (B5's open item): `RefitLayer` and `CalibrationLayer` are in the default registry. Their provenance is the governing model's data cutoff and the calibration's as-of time, both before τ (T-NL1, T-NL2). Its memo is per store, so the truncated and garbage runs recompute.

**Tests:** 1,339 fast tests, 328 more than at B6. They include:

- a permuted model matched back exactly;
- each drift rule's edge;
- activation across weekends and holidays;
- append-only history;
- calibrated and miscalibrated synthetic forecasters;
- C13's counts;
- the artifact's tamper and as-of refusals;
- planted leaks in the refit and calibration layers.

The full tier adds the monthly refits through 2018 on real FIT data, from a model selected on 2016–2017 (K = 5, full covariance), on a copy of `market.db` with everything known from 2019 on deleted.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | 105 s (1,339 tests) |
| Real-bar 200 cuts (full) | 336 s (125 s at B6: the lifecycle layers) |
| Synthetic 200 cuts (full) | 99 s |
| Monthly refits through 2018 (full) | 219 s |
| Backfill rerun | 3 s |

**Review:** two rounds, one per sub-step. Each round used the code-review skill and Codex, or a Claude stand-in. Each finding got a red test, then a fix.

- **B7a** (8cbb717, dbf7c1e):
  - a supplement row under NOW revised the feed's dividend;
  - `INSERT OR REPLACE` rewrote ops history;
  - a refit ran again the same evening, every evening after a FROZEN or REJECTED one, or on day 29;
  - `refit_data` dropped the whole-session check;
  - a ModelError after training escaped;
  - a "future rows" test that passed on the pointer check alone.
- **B7b** (092240b, e3a0bf0):
  - **(high)** every refit past 2018 would have failed, because the calibration read DEV without a ticket and the lockbox;
  - each superseded span's last 35 bars were scored with unsettled targets;
  - no gap before the isotonic check's held-out half;
  - the harness memo let the truncated runs reuse the full run's results;
  - the calibration layer saw only probabilities of 0 and 1;
  - a series' interval level went unchecked;
  - verdict flags weren't checked on load;
  - the forward pass ran twice.
- **A red-test defect, found by the green:** `garbage_store` accepted only a decision instant, which a 17:00 cutoff isn't. It now takes the latest decision at or before the cut. No assertion changed.
- **A test oracle changed by finding D:** `test_calibration_metrics.py`'s reference isotonic fit is now on the older half short of its last 35 bars.

Not fixed, with reasons:

- **A supplement row stored before a later feed row with a real declaration date.** History can't revise a stored row, so the feed's row is reported as a conflict on every run. The stand-in is conservative: it is known later, never earlier. If it happens, the row comes out of the supplement, with a gated store repair.
- **A freeze recorded after an ACTIVATED refit but before its open wins.** This can't happen now. No refit can rerun within 30 days, and the live monitor fires only on bars, so it can't fire between an evening and the next open.
- **A release that changes the feature config, or a pruned artifact,** makes every later refit's calibration raise. Release lineage is B14's: re-selection and promotions are gated, and artifacts are never pruned.
- **Outcome pairs are consecutive rows (filter time).** A dropped bar misscores one pair. The real data has three incomplete bars, all in DEV.
- **Duplicated code:** `symmetric_kl` factorizes each covariance K times, which is negligible for K ≤ 6. The three standardized-space conversions stay separate, because the decimal edges and the seed equality rest on their exact expressions. `save_calibration` and the array hashing repeat `hmm.model`'s; a shared `quant.core` helper can come later.

**Deviations** (documents updated in this merge)

- **ARCHITECTURE v1.10, §6.2:** missing dividends are filled from the committed supplement.
- **ARCHITECTURE v1.11, §7.4:** where SPEC §9 and C13 are silent, these readings apply:
  - **Counts:** C13's 1,000 bars are counted after the newest 35 are left out, so a verdict needs 1,035. The last 35 bars of *every* model span are left out, not only the newest 35.
  - **Recalibration:** a failing label's isotonic map is fit on the older half, short of its last 35 bars, and judged on the newer half.
  - **The outcome check:** it uses the latest min(1,000, n − 1) pairs, the newest 35 included, since realized F1 needs no settling.
  - **Edges:** ECE ≤ 0.05 and skill > 0, on decimal values.
  - **Reads:** the calibration skips the lockbox and reads DEV with the trial ticket.
  - **The artifact type** lives in `quant.core.calibration`.
- **PLAN v1.4:** B7's drift-edge test fires just past each threshold, not at it (SPEC §5 says "more than").
- **CLAUDE.md** cites ARCHITECTURE v1.11 and PLAN v1.4.
- **New artifact hashes:** every model artifact now records `info["restarts"]["seeded"]`, so new artifacts hash differently. No test pins a hash.

**Closed from B6:**

- the CP1 questions: the supplement was approved and built, and the second paper account exists (VERIFY §2);
- the keys: you regenerated them on 2026-10-15.

**Open items**

- **You, at this report:** confirm the readings above, especially leaving out the last 35 bars of every span. With monthly refits that drops about a quarter of the bars, so a verdict takes longer to reach.
- **B11:** the walk-forward passes the trial ticket to `refit(ticket=)`.
- **B13:**
  - `.env.test` has no `TRADING_MODE`, so `load_settings` refuses it as is;
  - the engine wires `record_freeze(LIVE_DRIFT)` from the live monitor.
- **B14:** live refits after the seal need the release's trial ticket for DEV calibration history; the fills source's 30-fill rule; release lineage across feature configs.
- **B8/B14:** the filter layer still runs the fixture model, so filtering under the governing model after a swap is untested in the harness.
- **The fast-tier budget:** 105 of 120 s. B7 moved B4's two slowest training tests and one lifecycle test to `full`. The next candidates are B4's remaining training tests and the calibration job tests.
- **The full tier** now takes much longer, mostly the real-bar 200 cuts at 336 s.

## B6 · Real data and the first model (done 2026-10-15, tag `b06-done`)

You put the Alpaca paper keys in on 2026-10-15, and B6 was built the same day on `b06-data`, in two parts:

- **B6a:** the code.
- **B6b:** the real runs, the first model and CP1.

Claude agents wrote the reds, the probe and the review fixes. Codex wrote both greens; B6b's was committed by the orchestrator after Codex hit its usage limit.

**Built**

- **`core.settings`:** loads `.env` and calls `register_secret` on each key before anything can log (B1's open item, I6).
  - It refuses a missing variable, `TRADING_MODE` other than `paper`, and a file other users can read (mode 600). A bad line is refused by its number, without echoing it.
  - With no path it reads the environment, as systemd injects it.
- **`data.alpaca_data`:** the market-data client, on alpaca-py's data API only. `alpaca.trading` stays in the execution adapter (K4).
  - **Requests:** SIP minutes ask for `feed=sip` and `adjustment=raw` explicitly. A bars request never ends inside the feed delay, because the free plan answers 403 there.
  - **Pagination, timeouts and limits:** it pages with `page_token`. Every call has a 5 s connect and 30 s read timeout, and its requests are spaced under the rate limit.
  - **429s:** waits are on the Clock. It honours Retry-After (as seconds or an HTTP date), then X-RateLimit-Reset, then backs off up to 60 s, within a 120 s budget. alpaca-py's own retry is switched off through its public `retry_exception_codes`.
  - **Other reads:** the latest IEX quote and trade, the calendar, and corporate actions from the v1 feed and the trading host's announcements.
- **`data.actions`:** dividends and splits.
  - **Declaration dates:** the feed never has one, and announcements start only at 2020-09. A missing declaration date, or one after the ex-date, falls back to the ex-date, so a dividend is never known early.
  - **Splits:** a split, or a stock dividend recorded as one, halts trading through `ActionSync.halt`.
  - **Changes:** an action that changed under HISTORY is reported in `ActionSync.conflicts`, and the others still store.
- **`data.backfill` and `quant data backfill`:** resumable from 2016-01-01.
  - **Resume:** market.db migration 2 adds `backfilled_sessions`, and a resume skips only recorded sessions. A partial session is fetched again; where history can't revise it, it is completed as NOW at the end of the run and reported.
  - **Calls and stamps:** one ingest call per session (B2's open item), using HISTORY only before live trading first reads the store.
  - **End:** it never goes past the last session whose close plus the feed delay has passed, whatever `--end` says.
- **`data.quality` and `quant data quality`:** missing sessions, minute gaps, every missing or incomplete hourly slot, move flags and half days. The run fails while a gap is unexplained. `config/data_gaps.json` holds the explanations.
- **`BarStore`:** a public `high_water`, plus `minute_times` and `has_minutes`, which carry no prices. The raw `latest_minutes` read from B6a's green was removed in review, so prices are read only through a view.
- **`PointInTimeView.flags`:** flags follow the minute rules: `t < τ`, `available_at < τ`, and the latest version known before as_of. The store's own flag read ignored availability.
- **`jobs.first_model` and `quant model select` (D4, CP1):**
  - **The read:** FIT only. `fit_window` refuses any span past 2018-12-31 before anything is read. A second guard refuses any bar on or after DEV's start, and any session missing bars at fit_end is refused, not dropped.
  - **Training:** it reads through the view at `as_of = fit_end`, starts the feature stream at the start of FIT, and runs B4's r1 selection.
  - **Outputs:** the artifact goes in `var/artifacts/models/<hash>/` (`.npz` without pickles, plus JSON), and the report in canonical JSON and Markdown. Provenance holds the commit (worktrees included), the lockfile hash and a dirty flag.
- **`spikes/alpaca_probe.py` and `docs/VERIFY.md`:** the ARCH §18 checks (below).
- **The truncation harness (B5's open items):**
  - its world can be built from a market.db, and the full tier runs 200 cuts on real FIT bars beside the synthetic 200;
  - a new `ActionsLayer` reads minutes, dividends, splits and flags;
  - the world raises flags, and the garbage now covers dividends, splits, flags and a minute revision known exactly at the cut.

**The real data** (`var/market.db`, gitignored):

- **The backfill:** 2,711 sessions, 2016-01-04 → 2026-10-14, in 14.5 minutes, with 0 skipped, 0 empty, no NOW sessions and no action conflicts.
- **Minutes:** 1,053,406 of 1,053,510 expected, with 21 half days and 0 move flags.
- **Nine gaps, all explained:**
  - four March 2020 market-wide circuit-breaker halts (14 min each);
  - five holes in Alpaca's SIP history, where paginated trades show none in the window: 2016-02-02 (9 min), 2019-08-12 (30 min), 2021-05-05 (5 min) and 2023-06-05 (4 min).
- **Incomplete hourly bars:** three, all in DEV. FIT's bars are all complete.

**What the probe found** (`docs/VERIFY.md`):

| Check | Result |
|---|---|
| SIP delay (D2) | holds: live, 19 new minutes were served 14.1–14.2 minutes after they ended, all 40 polls returned 200, and none changed after it appeared |
| Historical SIP | complete: 390 minutes, or 210 on half days, from 2016 to last week |
| Rate limit | 200 a minute on both hosts; the backfill is about 2,711 requests |
| Dividends | the feed misses 2016-03-18, 2018-06-15 and 2026-09-18, and has no declaration dates; announcements start at 2020-09, and five are declared after their ex-date |
| Paper dividends | Alpaca's docs say paper doesn't simulate them, so parity must add them |
| A second paper account | your question |

**The first model** (CP1):

- **Result:** K = 5 with diagonal covariance, on 4,840 FIT rows, selected in about 56 s, with the best validation mean of the 8 candidates. All 8 passed the sanity checks.
- **States:**
  - STRESS: −61% a year, 12% volatility, 0.7 sessions;
  - CRASH: −29%, 23%, 6.5 sessions;
  - three CALM_UP states: +36%/6%, +21%/11% and +19%/6%.
- **Report:** in `var/artifacts/reports/`, rerun at the merge commit so its provenance names it.

**Tests:** 116 new fast tests (1,011 fast in all), plus the real-bar and synthetic 200-cut full tests.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | 98 s (1,011 tests) |
| Full-tier truncation (49 tests) | 228 s: real-bar 200 cuts 125 s, synthetic 200 cuts 67 s |
| Backfill | 14.5 min |
| First-model selection | about 56 s |

**Review:** two rounds, one per sub-step, each with the code-review skill and Codex, or a Claude stand-in when Codex was at its limit. Each finding got a red test, then a fix.

- **B6a** (3f47ae4, c64349d), 13 findings and the probe's 403:
  - a resume that skipped partial sessions;
  - an explicit `--end` past the available data;
  - a settings test that passed without registration;
  - fully missing hours not listed;
  - no HTTP timeout;
  - a public raw minute read;
  - 429s that gave up before the reset;
  - refetched actions that broke every later run;
  - stock dividends;
  - a bare KeyError;
  - alpaca-py's private retry;
  - full reads on resume;
  - the declaration-date doc note.
- **B6b** (d46509f, da17257), 18 findings:
  - leaks the harness couldn't see: flags and splits revisions, and minutes known exactly at as_of;
  - the synthetic 200 cuts dropped;
  - features started at the span rather than at FIT;
  - the fixture model reading at a late knowledge time and duplicating the training rows;
  - CLI store errors;
  - worktree provenance;
  - FIT-only resting on one check;
  - silently dropped late sessions;
  - garbage read through the view under test;
  - the fit window's rendering.

Not fixed, with reasons:

- **Half of B6b's S10:** `_model_rows` was already cached.
- **FIT-only through the ledger:** the fit reads with the operate purpose, as ARCH §6.3 says. The policy never refuses an operate read, so a refused span is the job's own PolicyError, and nothing goes to the ledger.
- **Two planted leaks the harness can't see, and needn't:** dropping the minute's own `known_at` in the flags EXISTS changes nothing, because the version subquery enforces it. Dropping `raised = 1` is a correctness bug, not lookahead, and a unit test covers it.

The fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations** (documents updated in this merge)

- **ARCHITECTURE v1.9:**
  - §6.1: how the backfill ends, resumes and completes partial sessions as NOW; the 403; timeouts and 429 waits; stock dividends halting.
  - §6.2: the ex-date standing in for missing declaration dates; announcements from 2020-09; a late declaration counting as missing; the three dividends the feed misses; live dividend sync reading the announcements; refetched actions; the flags rule.
- **CLAUDE.md and PLAN's "builds on" line** say ARCHITECTURE v1.9.
- **The B5 fixture model's fit ends** are now New York midnights, so its hash changed. No test pins it.

**Open items**

- **You, at CP1:**
  - how to fill the FIT dividends the feed misses (2016-03-18, 2018-06-15). The probe proposes a small committed file from State Street's SPY history, checked by you;
  - whether your login can open a second paper account (AD3, by B13).
- **B11:**
  - credit a dividend on its ex-date, though it is known only at the end of that day;
  - treat the three incomplete DEV bars as rule 6 does live.
- **B13 and B14:**
  - live dividend sync reads the announcements;
  - the engine wires `ActionSync.halt`;
  - a live bars request ends at the bar's close, never at now;
  - the engine logs each bar's serve lag, because D2's margin of about 1.4 minutes rests on one 20-minute sample.
- **The fast-tier budget:** 98 of 120 s. B4's HMM training tests (about 32 s) are the first candidates to move to `full`.
- **Keys:** they were pasted into the chat. Regenerate them in the Alpaca dashboard and edit `.env` by hand (SPEC §0.3).

## B5 · The truncation harness (done 2026-10-14, tag `b05-done`)

B5 began on `b05-truncation` after B9 merged, as you asked on 2026-10-14. A Claude agent wrote the red commit and the review fixes, and Codex wrote the green.

**Built**

- **Where it lives:** `tests/support/truncation.py` holds the T-NL1 harness and T-NL2's provenance check. It is test code, because `quant.harness` is the research harness (ARCH §3). `src/` is unchanged.
- **Layers:** a `Registry` holds them, and later steps add theirs (PLAN §2 rule 4).
  - **view:** the bars a `PointInTimeView` returns, and its digest.
  - **features:** `FeatureStream` through the view. It records raw vectors before the model activates, and after that scaled vectors with the scaler's provenance.
  - **filter:** `ForwardFilter` and the labels, with the model's provenance.
  - **risk:** B9's open item "T-NL1 through the risk verdicts". Each decision gets a seeded `OrderIntent`, priced from the view's last close, through a fresh `RiskEngine` for each run. The record leaves out the token and the µs ids.
- **Runs:** for each cut τ, every record up to τ is compared by its canonical SHA-256 across three runs: (a) the store truncated to τ, (b) the full history, and (c) data to τ followed by garbage.
  - **Truncation:** later rows are deleted from every table that has a `known_at` column.
  - **Garbage:** NaN and ±1e9 bars and minutes, shuffled bars, and two bars that have closed but aren't available yet (one available exactly at the cut, one just after it). Also two revisions: one known exactly at the cut and one five minutes after it. The raw garbage is written straight into SQLite, so `check_minutes` can't screen it out.
- **T-NL2:** every record that uses a fitted artifact (the scaler, the model) names it and its fit end, and the check requires `fit_end < τ` strictly.
- **Strictness:**
  - Records are immutable canonical bytes.
  - A layer that raises fails the check, even when every run raises the same error.
  - A check where some layer compared nothing past its cold start fails as vacuous.
  - A failure names the first layer where runs differ.
- **Cuts:** drawn with an explicit seed, always including five edge cuts: the first warm bar, the model's first decision, the bar after a DST change, a half day's CLOSE bar and a session's first bar. The fast tier runs 50 cuts and the full tier 200.
- **The synthetic world:** minutes from 2016-06-01 to 2016-12-30, all within FIT, so DEV stays dark. The fixture model is K = 2 diag, fit on rows up to 2016-11-01 with `train_candidate(workers=1)`.

**The six planted leaks** (PLAN §4 B5), each caught and named by layer:

| # | Leak | Caught in |
|---|---|---|
| 1 | A centered moving-average feature | features |
| 2 | A scaler fit on the full history | features |
| 3 | The filter swapped for hmmlearn's `predict_proba`, and separately for `eval.smoother` | filter |
| 4 | A view that uses ≤ instead of < | view |
| 5 | Availability taken from `t_close` | view |
| 6 | A model whose fit window ends after τ (T-NL2) | filter (provenance) |

**Tests:** 26 fast and 1 full in `tests/invariants/test_truncation.py`, for 895 fast in all. They cover:
- clean code at 50 and 200 cuts;
- the six leaks;
- cut drawing, truncation and garbage;
- T-NL2's strict boundary;
- slicing the full run hiding no leak;
- one test for each review finding.

**Run times:**

| Measure | Time |
|---|---|
| B5 fast tests | 27.5 s (the 50-cut clean test is 13.5 s) |
| B5 full tests | 81.3 s (the 200-cut clean test is 54 s) |
| Fast gate | 89.3 s (895 tests) |

The fast gate now uses about 89 s of its 120 s budget. A plain replay of the harness took about 81 s, so the green shares record caches across runs, keyed on exact canonical bytes. B6–B8 need to watch their fast-tier time.

**Review:** one round, with the code-review skill and Codex. A Claude stand-in reviewer was lost to a session limit. 12 findings, all fixed as red tests (bd4c07b) and then fixes (065b7d9):

1. The clean features layer scaled decisions with a scaler fit after them, and feature records carried no provenance.
2. Two runs that raised the same error counted as agreeing, and the later layers were skipped.
3. No row was known exactly at τ, so `known_at <= as_of` went unseen.
4. `slice_full=False` let the smoother pass.
5. A full-history scaler built at setup passed (same root as 1).
6. Truncation left dividends, splits and calendar rows known after τ.
7. Garbage went into the hourly bars only, so the minutes now get it too.
8. Leak 3 didn't follow PLAN's `predict_proba` wording, so it's now planted both ways.
9. The caches keyed on Python equality, for example `0.0 == -0.0`.
10. `compared > 0` was checked only at 50 cuts, and it counted cold-start records.
11. Runs shared mutable records, so a mutating leak passed with 716 comparisons (Codex).
12. The feature-state hash read a private field instead of `snapshot()`. The fast path is now kept only for `FeatureStream`'s own snapshot, guarded by an equality test at every bar (Codex).

The fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations:** none from the docs. Where the harness lives and its synthetic world are design calls the docs leave open.

**Open items**

- **B6:**
  - run the full tier's 200 cuts on real FIT bars (PLAN P4, ARCH §13), since it is synthetic until the backfill;
  - add garbage for dividends, splits and flags, and a layer that reads the view's `minutes()` and actions.
- **B7:** register refits and calibration as layers, with their as-of times in T-NL2.
- **B8:** replace the per-layer records with DecisionRecords and state hashes (ARCH §13), and register `decide()`.
- **Private internals:** the harness uses the store's private connection to back up and to write raw garbage, and `FeatureStream._state` for the fast state hash. A guard test covers the hash, but a store refactor would need the harness updated.

## B9 · Risk engine and choke points (done 2026-10-14, tag `b09-done`)

B9 was built in three parts on `b09-risk`, beside B4, and merged before B5–B8 (you agreed on 2026-10-13; PLAN v1.3):

- **B9a:** the approved types, `RiskEngine.check`, `risk.yaml` and the counters.
- **B9b:** the broker port, reconciliation, the minute monitor, the C2 triggers and the kill executor.
- **B9c:** your calls of 2026-10-14.

Claude agents wrote the reds and the fixes. Codex wrote B9a's green, and a Claude agent wrote B9b's while Codex was at its limit.

**Built**

- **`core.orders`:** what B8's `decide()` emits and what the checks read: `OrderIntent`, `Side`, `Quote`, `OpenOrder`, `BrokerSnapshot`, `InternalBook` and `TradingMode`. They live in core because K1 lets `risk` import only `core` and `store`. The labels are B4's `core.regime.Label`.
- **The four approved types** (K5, I4): `RiskApprovedOrder`, `RiskApprovedCancel`, `RiskApprovedFlatten` and, from B9c, `RiskApprovedStop`.
  - Each is built only in `risk.engine` with its private token, registered when issued, and spendable once through the `SpentApprovals` hook.
  - Ids are `"{O|C|F|P|S}{n}-{µs}"`, and the counter is saved in the risk state.
  - The intent's canonical hash is taken at issue and checked at spend.
  - The K5 scan lists all four, and a planted copy of each is caught.
- **`RiskEngine.check`:** an INTENT precondition, then ARCHITECTURE §8.1's seven checks in order. The first failing check decides.
  - **The precondition:** the engine looks τ up in the schedule, enforces the bar window, and needs a started session.
  - **The price:** the caller proposes the limit price and check 6 bounds it. Exposure is valued at that price.
  - **Reservations:** an approval is outstanding until it is spent, released, revoked or expired. Outstanding BUYs count in the gross, label and overnight caps, and outstanding SELLs and stops reserve their shares. A spent order or flatten counts as in flight until a snapshot taken after the spend can show it.
  - **Revocation:** halts, a drift freeze, rule 6 and the daily-loss lock revoke BUYs. A snapshot short of the reserved shares revokes SELLs, a spent flatten revokes every order, and a SELL that takes a working or in-flight stop's shares revokes the stop approvals.
  - **Exact limits:** `mark`, the caps, the single-order cap and the collar compare decimal values, so the exact 2%, 10%, a cap or 10 bp lands on the inclusive side.
  - **PendingApproval (D7):** raised only when live above $25,000, and re-checked against everything outstanding.
- **C3's counters:** every intent checked after the session starts counts, vetoed or not. The 21st halts, and is vetoed only if it is a BUY. The 60th submission is sent, BUY or SELL, and then trading halts.
  - Every halt is signalled: `RiskApprovedOrder.halts`, `Veto.halts`, and `SubmissionLimitError` for a BUY past the limit.
  - A later halt keeps the first halt's reason.
- **Saving:** `RiskEngine(on_change=...)` gets every new RiskState before the call returns.
  - BUYs need a successful save.
  - Cancels, flattens, stops and SELLs still go out when a save or a spent-id write fails, kept as `RiskEngine.unsaved` and retried on the next call (your call 3).
  - Exception: while a halt is held only in memory, a SELL needs its save too, so a restart never finds more sent than it counted.
- **`resume(equity)`:** clears the halt and `killed`, and re-bases peak equity to the equity at the resume (your call of 2026-10-13). It keeps the counts and the daily-loss lock, refuses when nothing is halted, and returns `ResumeOutcome` for B14 to log.
- **`risk.limits`:** loads `config/risk.yaml`, refuses a file whose bytes differ from the committed version, and freezes the result. It holds SPEC §10's limits, plus the kill settings below.
- **`risk.broker.BrokerPort`:** `snapshot`, `quote`, `submit`, `cancel`, `flatten` and `place_stop`. Each changing call spends its approval before it sends. The port lives in `risk`, because K1 forbids `risk → execution`; B13's Alpaca adapter implements it in `execution` by shape.
- **`risk.reconcile`:** compares broker positions and orders with the engine's book: SHORT, POSITION, MISSING_AT_BROKER, UNKNOWN_TO_ENGINE, ORDER_DIFFERS or FOREIGN, else STALE or MATCH. Check 2 calls it and halts exactly on MISMATCH.
- **`risk.monitor.MinuteMonitor` (C8):** marks daily loss and drawdown each minute, on a fresh price and a fresh snapshot only, summing equity in decimals.
  - A non-finite cash or a bad price is a `data_fault`, with no mark and no kill. An equity ≤ 0 fires DRAWDOWN.
  - STOP_MISSING fires after 5 minutes without a broker-side stop. A working exit for the whole position counts as cover for at most 5 minutes in total, with no gap between exits extending it (your call 4).
- **`risk.triggers` (C2):** a lost heartbeat, a crash loop, a rule-6 flatten incomplete after 10 minutes, and 5 order errors in 10 minutes. A single late bar fires nothing, and starts or errors stamped after now aren't counted.
- **`risk.kill.KillExecutor`:** in regular hours it takes the lock, halts, writes the halt to control.db, revokes every order approval, cancels every open order, then flattens through fresh `RiskApprovedFlatten`s.
  - **The exit:** the collar widens 10 bp every 30 s, up to 100 bp, until the account is flat or the 300 s deadline or the close.
  - **Before an INCOMPLETE end:** it gets one 30 s grace to let its cancels land, then places a `RiskApprovedStop` 100 bp under the bid for the shares no working sell covers (`free_for_stop`).
  - **Outside regular hours:** it keeps the existing stop, queues the exit for `next_open`, and places a stop under any shares no stop covers (your call 2).
  - **Failures:** any exception in a step is recorded and the run goes on. The report and the alert are always attempted, and a failed save is reported once per run.
  - **The lock:** `FileKillLock` (flock, thread-aware) or `MemoryKillLock`. A lock that can't be taken is recorded and the kill runs unlocked; a second copy's exit is refused while the first's reserves the shares.

**New `risk.yaml` values the docs don't fix** (approved with the kill settings on 2026-10-14):

| Setting | Value |
|---|---|
| `kill.widen_bp` / `max_collar_bp` | 10 / 100 |
| `poll_s` / `deadline_s` | 1 / 300 |
| `heartbeat_lost_s` | 300 |
| `crash_loop_starts` | 3 in 600 s |
| `monitor_price_max_age_s` | 120 |
| `kill.stop_distance_bp` | 100 |

From the docs: `requote_s` 30, 5 order errors in 600 s, `flatten_incomplete_s` 600 and `stop_missing_s` 300.

**Tests:** 317 new, for 869 fast and 877 in all. Highlights:

- K5 and I4: runtime forgeries (a stolen token, a copy, a subclass, a reused, expired or revoked id) are refused for all four types. The static scan catches a planted copy.
- Each check vetoes at its boundary and passes just inside it.
- The counters halt at 21 and 60, every halt is signalled, and a resume re-bases the peak.
- Reconciliation flags every kind of mismatch, and check 2 agrees with it.
- The monitor and triggers fire at their exact thresholds, a single late bar fires nothing, and a gap between exits doesn't extend the 5 minutes.
- The kill executor halts, cancels, flattens, verifies and alerts in order. It widens the collar, and outside hours it keeps the stop and queues the exit (evening, pre-open, weekend across DST, half day). It closes only the remainder after a partial fill, survives broker, store, alert and save failures, and leaves a stop under shares it couldn't sell.
- **Three hypothesis properties:**
  - no sequence of intents and fills goes short or breaches a cap;
  - no kill goes short or sells at above the bid, with partial fills, a stop that fired first, and evening kills;
  - no stop approval or SELL lets the position go short.

  Throwaway mutants (never cancelling, an exit priced at the ask, a broker that oversells by one share, a no-op revoke) each fail them.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | about 63 s (869 tests in 62 s) |
| Full gate on the branch before the merge | 45.5 s (678 tests) |
| Kill property test | about 0.7 s |

**Review:** six rounds, each with the code-review skill and, where it was available, Codex, run in parallel. Each candidate was probed, then fixed as a red test and a green commit.

- **B9a, round 1** (20ced5d, 9a8db28): unreserved approvals, spends that didn't re-check, intent tampering, and inexact limit edges.
- **B9a, round 2** (665cbee, e85faef):
  - R6: intents vetoed before check 5 weren't counted;
  - C6: silent halts;
  - C7: unsaved state;
  - C8: no resume;
  - C9: 59 BUYs, not 60;
  - C10: a later halt overwrote the first reason;
  - C11: an unbounded re-price;
  - lists that only grew.
- **Your calls of 2026-10-13** (12915fa, 806c5c2): discretionary exits keep the ±10 bp collar and only kill exits widen; a resume re-bases the peak.
- **B9b** (b744497, c3ac1a2):
  - R1: no revoke after a kill;
  - R2/C1/C2: one failed step ended the kill;
  - R3: the queued exit's time;
  - R4: an exit after the deadline;
  - R5: inexact equity;
  - R6: future events counted;
  - R7/C6: a property that couldn't see shorts;
  - C4: data faults;
  - C7 and C8: locks;
  - C9: the first reason in control.db;
  - C10: stops cancelled outside hours.
- **B9c's own fixes** (25793e4, 7bb5475): the full tier's stop property found two ways to go short, a SELL taking a working or in-flight stop's shares.
- **B9c review** (10d3743, 89e882d):
  - R1/C4: a stop sized on a snapshot from before a flatten;
  - R2/C1: a pending-cancel exit counted as cover at the deadline;
  - C2: no stop on an outside-hours kill;
  - C3/C9: protect asked for more than was free;
  - R3: the spent store's failures blocked the bypass;
  - R4/C5: a halt lost through an unsaved SELL;
  - R5: the stop-missing clock;
  - C6: stop approvals couldn't be released;
  - C7: unreported save failures;
  - C10: one revoke helper.

Not fixed, with reasons:

- **`RiskContext` stays a dataclass, not a StrictModel.** A StrictModel would turn a bad context from a recorded Veto into an exception in the caller.
- **`committed_bytes` keeps its own git-variable scrub.** `quant` doesn't import the gate tooling, and its scrub is a superset of `scrubbed_env`.
- **B9c review C8 is not a defect.** A stop re-placed after a SELL also goes through `approve_stop`, which counts the stops already approved. In the reviewer's case it grants 40, not 60, so no share is covered twice. A guard test keeps this, and it relies on B13 placing every stop through `approve_stop`.

The B9c review's fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations** (documents updated in this merge)

- **SPEC v1.5, §10:**
  - a resume re-bases peak equity (2026-10-13);
  - a kill that can't finish leaves a stop under the unsold shares;
  - an outside-hours kill places one under shares no stop covers;
  - a working exit for the whole position stands in for the stop for at most 5 minutes (2026-10-14).
- **ARCHITECTURE v1.7:**
  - §3: the broker port is in `risk/`;
  - §3 and §4.4: K5 names four types;
  - §8.1: the precondition, reservations, counting, halts, saving and resume;
  - §8.2: the discretionary collar;
  - §8.3: the kill executor, the kill settings and how the stop is sized;
  - §16: the stop-missing row.
- **PLAN v1.3:** B9 merges before B5–B8, and every step merges through a pull request (§2 rule 7, your request of 2026-10-14).
- **CLAUDE.md:** the K5 line names `RiskApprovedStop`. Rule 7 says how a step merges through a PR, and the doc versions are SPEC v1.6, ARCHITECTURE v1.8 and PLAN v1.3.
- **The judgment calls of B9a and B9b** are recorded in ARCHITECTURE §8.1 and §8.3 (the price check, the INTENT precondition, inclusive thresholds, marketable rounding of kill exits, a BUSY lock, health kills only with a position). None changes an approved decision.

**Open items**

- **B13:**
  - give `SpentApprovals` an ops.db table, written in one transaction with the submission count, or rebuild the count from it, since SELLs sent unsaved before any halt lose their counts in a crash;
  - call `release` on any approval it drops, and spend the old approval before re-checking a re-quote;
  - handle `SubmissionLimitError` apart from forgeries;
  - place every protective stop through `approve_stop` and `place_stop`.
- **B14:**
  - persist the RiskState through `on_change` with EngineState, and alert when `halted` turns on and on `unsaved`, which now also means spent ids not written ("risk state or spent ids not saved");
  - run a queued exit whose time has passed at once, and run `execute` again after INCOMPLETE;
  - wire the monitor's flatten, replace_stop and kill outputs, and treat `data_fault` as unhealthy (rule 6);
  - don't act on `replace_stop` while a kill runs;
  - `quant resume` does a typed confirmation and a clean reconciliation, then calls `resume` and logs the `ResumeOutcome`.
- **B5 and after:** T-NL1 through the risk verdicts.
- **B8:** emit `core.orders.OrderIntent` as the engine reads it (whole shares, a stop below the limit on every BUY, never `reduce_only` on a BUY).
- **Kill timing:** a kill can run up to `deadline_s` + 30 s, because the grace reuses `requote_s`. On the QUEUED path, `protect()` can wait until the deadline while another approval reserves the bare shares.

## B4 · HMM: train, select, filter, label (done 2026-10-14, tag `b04-done`)

B4 was built in two halves on `b04-hmm`, in the worktree beside B9: B4a (the artifact, the filter, labels and the smoother) and B4b (training, selection and the sanity rejects). Claude agents wrote the reds and the review fixes, Codex wrote B4a's green, and B4b's green is Codex's work committed by the orchestrator after Codex hit its usage limit.

**Built**

- `hmm.model`: `HmmModel`, `GaussianParams`, `ScalerParams` and `TrainingRecord`.
  - The artifact is a hand-written, byte-identical `params.npz`, loaded with `allow_pickle=False`, plus canonical `meta.json`. The hash covers the canonical meta and each array's dtype, shape and SHA-256, and −inf is stored as None.
  - Models are immutable: training is re-parsed from canonical JSON, `TrainingRecord.info` is frozen all the way down, and `freeze_arrays` backs arrays with immutable bytes.
  - `save_model` fsyncs each file, the staged directory and the root, holds a flock for the whole save, and clears what a crash left. A best restart other than the argmax of `restart_logliks` (ties to the lowest index) is refused.
  - `ScalerParams.from_scaler` converts B3's Scaler with its STD_RTOL check. Every entry point takes `columns=` and refuses a mismatch.
- `hmm.filter`: our forward pass, in log space.
  - Densities come from each covariance's inverse Cholesky (`GaussianParams.whitening`), and emissions are centered on the bar's largest before the prior is added, so an outlier can't round the prior away.
  - l_t carries the scaler's Jacobian, so it is the density of the raw row (SPEC §6).
  - A missing or NaN row is absent, with no transition. `forward_pass(..., previous=)` continues a pass bit for bit. The state snapshots and restores.
  - `FilterOutput.top_label` and `top_label_remaining` are the most probable label. `ForwardFilter.remaining(label)` gives any label's duration for the switching engine's active label.
- `hmm.labels`: the rule table from v and d, label probabilities, entropy normalized by ln of the model's label count, and remaining duration w·(I−A_SS)⁻¹·1 over the states that leave the label. `Durations` refuses, when the filter is built, a model whose chain can reach from π a state that can never leave its label. A volatility within rounding residue of 0 is a ModelError.
- `eval.smoother`: our forward-backward, for evaluation only. It reuses the forward pass's emissions and centers them the same way.
- `hmm.select`, pure rules with no hmmlearn import:
  - free-parameter counts and BIC;
  - "simplest" means fewest free parameters, then fewer states, then diag before full;
  - restart choice, with ties to the lowest index and the 0.1% instability band taken of the standardized rows' log-likelihood;
  - the validation start and score;
  - the session-block bootstrap SE;
  - the sanity rejects, inclusive and compared as decimals after `math.fsum`;
  - the close rules and the simpler-winner fallback;
  - `drift_threshold`, the linear 1st percentile of the validation block's 35-bar rolling means.
- `hmm.train`: the only hmmlearn import, and it calls only `fit` (I2).
  - Restarts run in a spawn pool. Workers get the four BLAS thread variables set to 1 and get the rows once, through the initializer.
  - Seeds are `base_seed + i` within numpy's range. Only ModelError or ValueError counts as a failed restart, and a candidate whose restarts all fail is set aside.
  - The log-likelihood comes from our forward pass, and covariances are symmetrized exactly.
  - `select_and_train` and `train_candidate` both fit the validation block, so every model records its validation distribution and drift baseline. `meta.json` names the platform and numpy's and scipy's BLAS and LAPACK builds.
  - r1: `base_seed` 20160104, 50 restarts, K ∈ {2,3,4,5} × {diag, full}, n_iter 1000, tol 1e-2, min_covar 1e-3.
- `core.regime.Label`, the regime labels.
- `threadpoolctl` was added to the dependencies.

**Tests:** 199 new (196 fast, 3 full), for 552 fast and 560 in all. Highlights:

- **Recovery:** on 20,000 bars from a known 3-state HMM, training recovers the parameters up to a relabeling, for full and for diag covariances. Selection picks K = 3 and the generator's covariance type.
- **Equivalence:** the filter matches loop references, and a continued pass matches one pass. The smoother isn't the filter (`test_filter_not_smoother`).
- **Determinism:** the same seeds give the same hash, twice and pool against serial. Another seed gives other parameters, which a spy on `random_state` confirms.
- **Rules at their edges:** each sanity reject alone and at its exact threshold, the close rules at their edges, restart ties, and the instability warning at exactly 0.1%.
- **Leaks:** a spy checks byte for byte that the validation fit's scaler and EM rows exclude the validation block.
- **I2:** training calls only `fit`, checked at runtime. An unpickling payload with a side effect never runs.
- **Full tier:** `test_r1_selection_on_twenty_thousand_bars` runs r1's full 50-restart, 8-candidate selection.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | about 61 s (552 tests in 59 s) |
| Full gate | 1,107 s (560 tests), with the B9 agent loading the machine |
| One full r1 selection (50 restarts × 8 candidates, 20,000 bars) | about 516 s (Codex's measurement) |
| Warming the filter over 20,000 bars (K = 5, d = 5) | 0.78 s, down from 2.83 s |

**Review** (code-review skill and a Codex review in parallel, after each half's green). Each candidate was probed before a test was written.

**B4a, round 1** (81da83d red, 42bf030 green):

- R1: the prior was added before the emissions were centered, so a 1e9 row erased it.
- R3: the smoother lost transition evidence under an outlier.
- R2/C1: a label's duration solve went singular on an unreachable absorbing state, and a reachable closed label wasn't refused when the filter was built.
- C2: a covariance that passed validation crashed scipy with a bare LinAlgError.
- C3: `forward_pass` couldn't continue from earlier rows, which biased validation scores.
- C4: l_t lacked the scaler's Jacobian.
- R5: constant returns gave v = 337 from summation residue.
- C12: warming over 20,000 bars took 2.8 s.

**B4a, round 2** (61b4e4a red, 73695c6 green):

- R4: a model's info and arrays could change after hashing.
- C5: the scaler skipped STD_RTOL, and rows went in without their columns.
- C7: `active_label` became `top_label`, so it can't be mistaken for the engine's.
- C9: the unpickling test couldn't fail.
- C10: saves weren't durable.
- C11: nothing checked that the best restart was the argmax.

**B4b** (0b4c471 red, b5b680f green):

- R1/C7: one candidate whose restarts all failed aborted selection.
- R2/C4: `threadpool_limits(1)` can't limit Accelerate, so the workers now set the variables.
- R3: occupancy at exactly 2% was rejected by float residue.
- R4: the seed tests couldn't see the seeds actually passed to `fit`.
- C1: no test caught a validation leak.
- C2, parts 1 and 2: the drift baseline wasn't derivable from meta.json, and refits recorded none.
- C3: feature units moved the instability band.
- C5: the platform and BLAS weren't recorded.
- C6: out-of-range seeds and bugs silently dropped restarts.
- C8: a zero-mass state is now an explicit sanity reject.
- C9: validation-fit warnings were indistinguishable from full-window warnings.
- C10: the rows were pickled once per task.

Not fixed, with reasons:

- **B4a C8. A bad row is absent from the filter, with no transition**, not a predict-only step. This matches how `compute_features` drops NaN rows and skips incomplete bars, so warm-up and live agree, and rule 6 handles the decision. SPEC v1.6 states it.
- **B4a C6. Entropy is normalized by ln of the model's label count.** SPEC v1.6 states it.
- **B4a C5, in part.** `Scaler` and `ScalerParams` stay separate types: the Scaler carries a window and a row count the artifact doesn't store, and it raises FeatureError.
- **B4b C2, part 3, is not a defect.** l_t is the raw-row density on both sides, Jacobian included, so the baseline needs no rescaling to the live scaler. ARCHITECTURE §7.5 says so.
- **B4b C8 raises rather than returning a failing report.** A zero-mass state can't be labeled, and a sanity reject keeps the current model (ARCHITECTURE §7.3). The TrainResult docstring says so.

B4b's fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations** (documents updated in this merge)

- SPEC v1.6 (v1.5 is held for B9's merge), §4–§6, with no approved decision changed:
  - the instability band is measured on the standardized rows;
  - candidates whose restarts all fail are set aside;
  - the sanity edges are inclusive;
  - every model records its drift baseline (the 35-bar rolling-mean percentile);
  - l_t is the raw-row density, a missing bar is absent from the filter, and a reachable label the chain can never leave is a model error at build;
  - entropy is normalized by ln of the label count.
- ARCHITECTURE v1.8 (v1.7 is held for B9's merge), §7.1, §7.2, §7.5 and §14:
  - the workers set single-threaded BLAS themselves;
  - only ModelError and ValueError count as a failed restart;
  - candidates can be set aside;
  - meta.json records the drift baseline, the platform and the BLAS builds.
- `choose_restart(..., jacobian=, fit=)` and `train_candidate` fitting its validation block (so its window needs more than 12 months) differ from the red stubs. No document names those signatures.

**Open items**

- **B9's merge** adopts `core.regime.Label` in place of its own `core.orders.Label`, and takes SPEC v1.5 and ARCHITECTURE v1.7 as reserved. (Done in B9's merge.)
- **B5:** T-NL1 covers B3's features and scaler, and B4's filter (CLAUDE.md rule 4).
- **B7:** live drift computes rolling means as training does (full windows, each mean an fsum). ARCHITECTURE §7.1's "one restart starts from the previous model" is B7's.
- **Cost:** a refit now runs 2 × 50 restarts, and one full selection takes about 9 minutes.
- **BLAS:** the serial path uses whatever thread settings the caller had before importing numpy (ARCHITECTURE §14). The engine and harness must set them.
- **Fast tier:** about 60 s of the 120 s budget. The B4 files take the largest share.

## Docs · Jev shadow advisor (approved 2026-10-13)

You approved the plan on 2026-10-11 and the doc changes on 2026-10-13 ("approve Jev shadow"). Jev, TypeSafe AI's typed-decision model, joins as a shadow advisor: it is logged next to the HMM, scored only on forward paper bars, and never affects a decision or an order. No code changed.

- **SPEC v1.3 → v1.4:** layer L1s and invariant I7 (§1), the new §12a (the three questions, the bundle, forward-only scoring, late and missing, and the promotion test, fixed now), and a known risk for Jev's unchecked vendor claims (§16).
- **ARCHITECTURE v1.5 → v1.6:** AD7, the `shadow/` package and rule K7 (§3), the `quant-shadow` and ingest jobs and the shadow secret (§2), `shadow.db` (§9.1), the spool, outbox and ingest (§11.1), the I7 and K7 tests (§13), the outage row (§16), §17 narrowed to "any LLM that can affect a decision", and the spike's questions (§18).
- **PLAN v1.1 → v1.2:** step B20 (§1 and §4), after B19, size M. It depends on B14, B16 and B17.
- **CLAUDE.md** now names SPEC v1.4, ARCHITECTURE v1.6 and PLAN v1.2.
- **New ids:** I7, K7, AD7, SPEC§12a, and step B20.
- **Your calls on 2026-10-13 ("approve"):** the promotion test has two looks, fixed now: at 500 scored bars, and at 1,000 only if the first fails, each at p < 0.025 (Bonferroni). After a second failure Jev is never proposed. The ingest stamp is `received_at`, not `asked_at`, because it is taken when the answer is read.
- **Tests:** two gate tests used `K7` as their example of an id that doesn't exist. Now that K7 is defined, they use `ZZ999`, which is too long to collide with a future doc id. Neither test is weaker: each still plants an unknown id and expects the gate to catch it.

**Open items**

- The spike (`spikes/jev_probe.py`) waits for your TypeSafe key in `.env`.
- `.importlinter` is unchanged. B20's red commit adds K7 and puts `quant.shadow` in K1–K4, since a contract on a missing module would break the gate.
- quant-check's trace reads only numbered sections, so B20 teaches it lettered ones such as `SPEC§12a` before a test can name that id.

## B3 · Features and the scaler (done 2026-10-06, tag `b03-done`)

**Built**

- `features.stream`: `FeatureStream` computes F1–F5 bar by bar from a small fixed-size state: each slot's values over the last N sessions with their session numbers, the last B returns, two EMAs and the previous close.
  - The state snapshots to canonical JSON (`FeatureState`) and restores bit for bit, only under its own config. `restore` refuses a state that can't be one: negative ranges or volumes, a session that isn't an ISO date, a session count that no slot's numbers match, or histories out of step with their numbers.
  - `update` refuses a bad bar with FeatureError and leaves the state as it was: a price that isn't finite and positive, prices out of order, a bad volume, a bar out of order, or prices too far apart for float64. An incomplete bar is missing data and is skipped.
  - `compute_features` featurizes a fit's bars with the same stream, keeps the warm rows in a frozen `FeatureMatrix`, and drops a warm row holding a NaN, with a "feature row dropped" warning. A fit streams from 2016-01, as live trading does, and cuts its window with `FeatureMatrix.window` (SPEC v1.3).
  - `features_for` gives a decision its vector through the point-in-time view, with purpose operate. A missing decision bar, a warm-up row or a NaN raises FeatureError, a model error, so rule 6 applies.
- `features.scaler`: `Scaler.fit` takes the mean and the ddof 0 std of the rows whose `available_at` lies in [start, end), and nothing else. The scaler is frozen, refuses a constant column (std ≤ `STD_RTOL` × |mean|, also on load), and `transform` refuses columns from another config and any result that isn't finite.
- `features.menu`: `FeatureConfig`, a StrictModel with PLAN §5's bounds. F1 and F2 are required, F3–F5 are optional, and the trend's fast span must be below its slow one. `config_hash` hashes canonical JSON, and `r1_features()` is r1's config.
- `core.errors`: FeatureError, a ModelError.
- `tests/support/features.py`: the slow reference, written as plain loops from SPEC §3 and used only by tests.

A Claude agent wrote the stubs and tests (10d3191), and a Codex agent wrote the implementation (15c2cc9).

**Tests:** 65 new (64 fast, 1 full), for 356 fast and 361 in all. Highlights:

- The stream matches the reference within 1e-12 on every bar, warm-up included, for r1, every lower bound, and F1 and F2 at their upper bounds. Two runs agree bit for bit, and a restored snapshot continues bit for bit.
- I1: changing any bar after t leaves every feature at t unchanged (hypothesis), through the stream and through `compute_features`, and a planted batch featurizer that reads the last bar fails the check. `compute_features(bars[:k])` gives the same rows as the full list. A decision reads only through the view, and a revision known after `as_of` can't reach it.
- Slots: the CLOSE slot is compared only with CLOSE bars, a half day's included. For the N sessions after a half day, the 12:30, 13:30 and 14:30 histories hold N−1 values.
- The warm-up drops 60 sessions. A bad bar is refused with the state unchanged, an overflow too, and a NaN or missing feature at decision time is a model error.
- The scaler reads only its window, is frozen and applies forward. It refuses constant columns (0.1, 1e-20, −3.3 and 123456.789), and small values that really vary still fit.
- The menu refuses unknown transforms and parameters, out-of-bounds values, fast ≥ slow and configs without F1 or F2. The config hash ignores key order and changes with every parameter.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | about 16 s (356 tests in 14 s) |
| Full gate | 34 s |
| Ten years of features (17,535 synthetic bars, r1, 17,115 warm rows) | 0.28 s; the target is 5 s |
| One `update` | 13 µs |
| One snapshot | 0.2 ms |

**Review** (code-review skill and a Codex review in parallel, after the green). Each candidate was probed against the code before a test was written, and the candidates were fixed in two rounds.

**Round 1**, fixed as red tests (f722e12) and a green commit (06a409e):

- **R1/C4.** A slot's history held its last N bars, not its values in the prior N sessions as SPEC §3 says. A half day has no 12:30, 13:30 or 14:30 bar, so those histories reached back past N sessions: F1 at 2016-11-28 12:30 ET was −0.17990, against −0.17841 by the SPEC. The stream now numbers the sessions it reads and keeps each value's session number (`FeatureState.slot_numbers`), and the reference follows the SPEC.
- **C3.** The lookahead property drove the stream bar by bar, so it couldn't fail: a planted batch featurizer that reads the last bar passed it. It now also checks `compute_features`, and new tests compare `compute_features(bars[:k])` and `features_for` with the prefix and the full batch.

**Round 2**, fixed as red tests (6d59500) and a green commit (bce0792), in `tests/unit/test_review2_b3.py`:

- **R2.** Finite prices too far apart for float64 (1e-300, then 1e300) gave a NaN return that the state kept, so the snapshot had no canonical JSON. The range ratio and V / median overflowed the same way. `update` now computes every value before it changes the state, and refuses with FeatureError, state unchanged, a return that rounds to infinity or 0, an overflowing range ratio, an EMA that isn't finite and positive, or any infinite row value.
- **R3/C1.** A constant 0.1 column fit with a std of about 1e-17, because float64 can't hold 0.1. The scaler refuses a column whose std is at most `STD_RTOL` (1e-9) × |mean|. The check is in `__post_init__`, so a loaded scaler gets it too.
- **R4.** A z-score that overflowed raised RuntimeWarning. `transform` runs under `np.errstate` and raises FeatureError.
- **C5.** Int prices and volumes put ints in the snapshot, and a JSON round trip changed its canonical form. They become float64 before they are stored.
- **C7.** One flat bar (H = L) in a fit window raised FeatureError and aborted the fit. `compute_features` drops the row and logs it; at decision time the row is still a model error (SPEC §3). The test that expected an all-flat batch to raise now expects zero rows, and a scaler fit on them to raise.
- **C8.** `restore` accepted negative ranges or volumes, a last session that isn't an ISO date, an inconsistent session count, and histories out of step with their numbers. It refuses all four.
- **C9.** `transform` couldn't refuse a vector from another config of the same width. It now takes the column names, which must match, and only a float64 ndarray (deviations).
- **C10.** One helper, `freeze_arrays`, freezes the arrays of FeatureMatrix and Scaler.

Not fixed, with reasons:

- **C2. The warm-up stays 60 sessions whatever N is**, not max(60, N). SPEC §3 and PLAN B3 both say 60 sessions are dropped, and max(60, N) would change an approved decision for N > 60 while making no difference to r1. Fits and live trading agree all the same, because both stream from 2016-01 and share the EMA seed. The docstrings say so, and SPEC v1.3 states it.
- **C6, in part.** The wedge it describes, a stored complete bar the stream refuses, can't happen: `check_minutes` rejects bad bars at ingest. The late-completed bar is real: a live stream that skipped an incomplete 10:30 bar diverged from the fit once the missing minute arrived (F2 −3.4591 live, against −3.5027 from the fit). The fix is a rebuild policy, which belongs to the engine (open items).

The fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations** (documents updated in this merge)

- SPEC v1.3, §3, written down from how B3 reads it, with no approved decision changed:
  - one stream over every hourly bar from 2016-01, shared by fits and live trading, and a fit cuts its window from those rows;
  - sessions are numbered in the order they are read, a half day counts as one, and "the prior N sessions" means sessions k−N to k−1, so after a half day the 12:30, 13:30 and 14:30 slots hold N−1 values;
  - σ_slot is the RMS with no mean removed, an even count's median averages the two middle values, F3 is the log of the median, and each EMA uses α = 2/(n+1) and starts at the first close;
  - the scaler takes the mean and the ddof 0 std of the warm rows with `available_at` in [start, end), and the warm-up drops 60 sessions whatever N is.
- `Scaler.transform(values, *, columns)`: the red stub took `values` alone. The column names are required so that a vector from another config is refused (C9). No document names the signature; callers from B4 on pass the matrix's `columns`.

**Open items**

- **B14 or B5: the late-completed bar** (C6). The live stream skips an incomplete bar, and if its missing minute arrives later the stream no longer matches a fit over the same bars. It needs a rebuild policy: re-stream from history, or detect the late bar and apply rule 6.
- **B5:** add the features and scaler layer to the T-NL1 truncation harness (CLAUDE.md rule 4). B3 touches the decision path, but the harness doesn't exist yet.
- **B4** branched from main before this merge, so it must bring main in to use the features and the scaler.

## B10 · The trial ledger and TrialTicket (done 2026-10-06, tag `b10-done`)

**Built**

- `research.ledger` (`research.db`): the append-only trial ledger, on the `store.sqlite` connect and migrate pattern.
  - Each entry (`run`, `result`, `refusal`, `gate_config`) holds its kind, its time from the ledger's Clock, a canonical JSON body, and the hash of the entry before it. A single `head` row, written in the same transaction as every append, names the last entry, so a row deleted from the end is caught too.
  - Every read (`entries`, `trial_count`, `gate_config_hash`, `verify_ticket`) takes the rows and the head in one snapshot and raises LedgerError unless the chain verifies.
  - Appends run under the write lock, so two ledger objects or two processes chain correctly. An append checks only what is new since the chain the ledger object last verified or wrote, and refuses a tail rewritten since then.
  - A run is placed twice (C11). When it opens, by its configuration hash, code hash and days: a match reuses that run's trial, and anything else opens the next one. When it settles, by the hash of its decision records: records equal to an earlier settled run's join that run's trial, and a code change that moved no decision leaves its own trial empty. A run that never settles counts until a run with the same config, code and days settles, then goes where that run went.
  - `record_gate_config` appends gates.yaml as loaded whenever its canonical hash changes.
- `TrialTicket` (K5): built only by `open_run`, with the module's private token, and only after the run's entry is committed. A ticket is good only with the ledger object that issued it. `verify_ticket` refuses lookalikes, subclasses, copies, tickets from another ledger object on the same file, a ticket changed since it was issued (judged by identity), and one whose run entry no longer stands in a chain that verifies. No ticket can reach the lockbox (P9). Yardsticks score under the ticket of the run they measure and never settle, so they add no trial.
- `data.policy`:
  - DataPolicy is bound to a ledger through a `LedgerPort` protocol, so research imports data and never the reverse. A score or calibrate read of DEV needs a ticket that ledger verifies; an unbound policy refuses every DEV ticket, which keeps DEV dark by default.
  - A bound policy records every read it refuses as a `Refusal`. A refusal the ledger can't record still raises its own error, with a note, and logs `refusal_not_recorded`.
  - The starts of FIT, DEV and the lockbox are constants; only the seal (`lockbox_end`) is configured. A policy refuses a ledger that holds different Segments.
- `quant ledger verify` (typer, a new `quant` console script) exits 0 with the entry count and head, 1 with each problem, and 2 when there is no ledger. `verify_file` opens a closed ledger as immutable, so it creates no -wal or -shm files and works on a read-only mount.
- `core.errors`: TicketError (a PolicyError) and LedgerError.
- `tests/support/ledger.py`. B2's tests that scored DEV with a structural ticket now bind a vouching stand-in (`vouched_policy()`), so they still test the segment rules alone.

**Tests:** 78 new, all fast, for 292 fast and 296 in all. Highlights:

- K5: nine planted forgeries are refused for score and calibrate. Static scans already covered TrialTicket from B0, so the new tests forge at runtime: a stolen token, a copy, a subclass, a ticket from another ledger, and a field swapped for a subclass whose `==` vouches for anything.
- C11: an identical rerun reuses its trial, different records add one, a code change that moves no decision adds none, a rerun whose decisions changed is a new trial, and a crashed run follows its identical rerun.
- A2: scoring DEV without a ticket raises and leaves a ledger entry, and so does a refused lockbox read. No ticket reaches the lockbox, and no ledger or policy can be built with a later one.
- The chain: ten tamperings fail `verify` and the CLI. Two spawned processes settle 10 runs each at once (40 entries, chain valid), two ledger objects take turns, a failed append leaves no trace, and a ticket dies with its run entry.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | about 13 s (292 tests in 12 s) |
| Full gate | 27 s |
| Append one entry | 0.1 ms |
| Open and settle a run | 0.35 ms on a new ledger, 32 ms at 2,000 entries (placing a run reads the whole chain) |
| `trial_count` at 2,000 entries | 30 ms |
| `verify_file` over 2,001 entries | 26 ms |

**Review, round 1** (code-review skill and a Codex review in parallel, after the green). Each candidate was probed against the code before a test was written. Six were fixed as red tests (626db35) and a green commit (05a10c5):

- **R1.** A caller could move where DEV and the lockbox start, so a ledger with a later lockbox issued a ticket for 2025. The starts are now class constants that no caller or subclass can move. DEV's start is fixed too, since a later one would make DEV days FIT, scored without a ticket.
- **C6.** The ledger and its policy kept separate Segments. `TrialLedger.segments` is public, and a policy refuses a ledger with different ones, or with anything but Segments itself.
- **R2.** `verify_ticket` trusted a field's own `==`, so a ticket recorded as January could score all of DEV. It now compares each field with the issued object by identity. Issuance had the same hole (a date subclass in a RunSpec), so `open_run` takes only exact dates and strings.
- **R3.** A ticket outlived its run entry. `verify_ticket` now requires the entry to stand, as issued, in a chain that verifies, and rechecks the chain only when SQLite's `data_version` or `total_changes` has moved. Settling after a wipe raises TicketError, not KeyError.
- **C1.** The counts read the chain without checking it. `entries()`, `trial_count()` and `gate_config_hash()` read one snapshot and raise LedgerError unless it verifies; an unknown kind does too.
- **R4.** A crashed run stayed a trial of its own after its identical rerun joined an earlier trial. It now goes where the latest settled run with the same config, code and days went, so the crash case counts 1 (C11: a rerun reuses its trial).

Not fixed, with reasons:

- **C3. The chain has no anchor outside research.db**, so a rewrite of the whole file passes `verify`, and a full wipe makes `trial_count()` 0. Only the quant user writes research.db (ARCHITECTURE §2, §9.1), and an anchor kept on the same box could be rewritten by the same user. A2 guards against an honest but tempted loop, and L1 runs as quant-llm with no access to `var/`. The backups keep earlier copies, which v1 doesn't compare. The docstring, which claimed a deletion from the end is always caught, is corrected, and ARCHITECTURE v1.5 says so.
- **C2. The policy can't see which configuration is scored**, so a ticket for one config could score another. Only the walk-forward knows what it runs, so B11 enforces it (open items).
- **C4. An unbound DataPolicy refuses but can't record.** It already refuses DEV score and calibrate, and lockbox reads other than operate. A mandatory ledger would make the engine and evening jobs write research.db, which ARCHITECTURE §2 and §9.1 reserve for the harness and the research CLI. Narrowing §6.3's "a refused read leaves a ledger entry" needs the user (open items).

**Review, round 2** (the rest of round 1's findings, each confirmed by a probe). Fixed as red tests (4578e66) and a green commit (2064a6a):

- **C5.** When the ledger couldn't record a refusal, its LedgerError replaced the refusal, so a refused lockbox read escaped `except PolicyError`. A refusal now always raises its own error, with a note when it went unrecorded. Round 1's R3 test now asks for exactly TicketError.
- **C7, in part.** `verify_file` left -shm and -wal files beside a closed ledger and failed in a directory it couldn't write. It now opens a closed ledger as immutable, and one in use read-only through its WAL. A known gap, in the docstring: a checkpoint during an immutable check can report a problem that a rerun doesn't.
- **C8.** Every append rehashed the whole chain. Each ledger object now keeps the chain it last verified or wrote and checks only from that entry on. The trade-off: an append doesn't notice an edit before that entry, but `verify()` and every read still check all rows, and `verify_ticket` rechecks the whole chain after any write.
- **C9.** An append went on over a tail rewritten since it last saw the head, which `verify` can't catch. It now refuses. New tests cover concurrent processes, two ledger objects, a failed append and tail tampers, and the CLI test checks the entry count it prints.
- **C10.** A ticket's `issued_at` came from a second reading of the clock. It is now its run entry's `at`.

Not fixed, with reasons:

- **C7, the rest.** `quant ledger verify` defaults to `var/research.db`, relative to the working directory, and nothing resolves `var/` yet (ARCHITECTURE §2: the repo's `var/` on the Mac, `/var/lib/quant` on the VPS). That is one rule for every database, not the ledger's, so it is left to the step that first runs a service (open items).

Round 2's fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations** (documents updated in this merge)

- ARCHITECTURE v1.5:
  - §4.4: the walk-forward refuses a TrialTicket whose `config_hash` or `code_hash` differs from the release and code it runs, and runs yardsticks only inside the measured run's own walk-forward (C2, built in B11).
  - §6.3: the starts of FIT, DEV and the lockbox are constants in code, and the seal is the only configured boundary (R1). The docs never said otherwise; C1 already puts only the closing date in config.
  - §10: the chain has no anchor outside research.db, so a full rewrite passes `ledger verify`, which v1 accepts because only the quant user writes the file (C3).

**Open items**

- **B11:** binds tickets to config and code (C2). The walk-forward refuses a ticket whose `config_hash` or `code_hash` differs from the release and code it runs, and runs yardsticks only inside the measured run's walk-forward, under its ticket. B11 also builds the `config_hash` that RunSpec names.
- **B14 or B18:** one rule that resolves `var/` for every database, such as a `QUANT_VAR` setting, so `quant ledger verify` and the services find research.db wherever they run (C7).
- **For the user (C4):** ARCHITECTURE §6.3 says a refused read leaves a ledger entry. Processes that don't write research.db (the engine, the evening jobs) refuse without a record. Proposed wording: "A refused read is recorded in the ledger when its DataPolicy is bound to research.db (harness, research CLI); processes that don't write research.db refuse it without a record." This narrows an approved decision, so §6.3 is unchanged until you approve it.
- **B2's finding 6 is closed:** the policy no longer trusts anything shaped like a ticket. Only its bound ledger can vouch for one, by the private token and by identity.

## B2 · Bars and the point-in-time store (done 2026-10-05, tag `b02-done`)

**Built**

- `store.sqlite`, the pattern the other three databases reuse (AD2):
  - `connect` opens a database in WAL mode with foreign keys on and explicit transactions. Two connections switching a new file to WAL at once make SQLite refuse one straight away, so the switch is retried.
  - `migrate` applies numbered migrations in one transaction under the write lock, re-reads the schema version once it holds the lock, and refuses a database written by newer code.
  - `transaction()` commits inside its guard and rolls back only while a transaction is still open, so SQLite's own error comes through.
- `data.market_db` (`market.db`): minute and hourly bars, dividends and splits with their declaration dates, a calendar cache and data flags. Every row is versioned and none is overwritten.
  - Minutes enter only through `ingest_minutes`. In one `BEGIN IMMEDIATE` transaction it validates them, stores them, rebuilds the hourly bars they fall in, and recomputes the move flags of the sessions it touches. It reads no other session.
  - Every write says how it is known. **NOW** is stamped from the store's Clock while it holds the write lock. The store keeps the last stamp it used, on rows or a live view, and each new stamp comes after it even if the clock steps back, so `known_at` orders exactly with commits. **HISTORY** stamps each row at its own `available_at`, and a dividend or split at the end of its declaration day in New York. It only adds rows, and only rows known after the store's last stamp.
  - A revision is known NOW, strictly after the version it replaces.
  - Move flags are versioned. A revision that clears one appends a cleared version, and `flags(symbol, as_of=…)` reads them point in time.
- `data.view`: `PointInTimeView(τ, as_of)` returns rows with `available_at < τ`, each in the latest version known before `as_of`.
  - Every view names its `as_of`: τ in the backtest, the recorded value in a replay, and `NOW` live, stamped like a write, so a replay sees exactly what live trading saw.
  - Every read names its purpose, and the policy checks it over exactly the days the read can reach.
  - The view hashes each row it hands out, for cache keys.
- `data.policy`: the FIT, DEV, LOCKBOX and FORWARD segments, and four purposes. Only operate may read the lockbox or the days before 2016.

  | Purpose | Covers | May read |
  |---|---|---|
  | operate | features, filter, refits | anything (C1) |
  | score | strategy P&L, trades, gate results | FIT, and DEV with a covering trial ticket (C11) |
  | calibrate | calibration, Kelly edge estimates | FIT, DEV with a ticket, and FORWARD |
  | report | parity, reports | anything but the lockbox |

  An unknown purpose may read nothing.
- Supporting modules:
  - `data.bars`: the hourly builder, and `map_minutes`, which validation, the builder and the store share;
  - `data.validate`: checks on minutes, and the 5% move flag within a session;
  - `data.synthetic`: seeded minutes on the real calendar;
  - `data.frames`: minute columns and hourly bars;
  - `core.epoch`: exact microsecond times.

**Tests:** 61 new (60 fast, 1 full), for 214 fast and 218 in all. Highlights:

- Bars: a full day builds seven bars and a half day four, each from exactly its minutes. A missing last minute leaves its bar incomplete. Every kind of bad minute is refused.
- I1: a row available exactly at τ is invisible. A revision is invisible until it is known. A view at τ1 is contained in the view at τ2 (hypothesis). The cache key changes only when a row that was read changes.
- Replay parity:
  - an ingest that waited for the lock is stamped when it took the lock;
  - live views and ingests keep their order when the clock steps back;
  - history can't add a row a live view may have read, while a stamp that stamped nothing doesn't block it;
  - a view must name its `as_of`.
- Flags: a jump against a stored minute is flagged, and a settle that moves a jump raises the new flag and records the cleared one. An ingest reads only the sessions it touches.
- Races: two writers revising one minute both land, and two processes migrating a new file both succeed. A WAL switch refused during recovery is retried, and an error with no SQLite code comes through unchanged. Each race test fails when its fix is removed.
- DataPolicy: no purpose but operate reads the lockbox. DEV needs a covering ticket to score or calibrate. Forward data can be calibrated but not scored. A read is checked over exactly the days it can reach. A purpose named by a string is checked like the enum, and an unknown purpose is refused.

**Run times:**

| Measure | Time |
|---|---|
| Fast gate | about 12 s (214 tests in 11 s) |
| Full gate | 25 s |
| Validate and build ten years of synthetic minutes | 1.6 s |
| Store ten years | 1.6 s |
| Re-ingest ten years | 1.7 s |
| One-day settle revision | 2 ms |
| Read the last 300 bars | 2.4 ms |

**Review, round 1** (code-review skill, high, after the green). Six findings, fixed as red tests (f77952a) and a green commit (92c5613):

1. A read with no purpose fell back to operate and skipped the lockbox check.
2. Writes stamped `known_at = available_at` by default, so a late live bar was backdated and a replay saw more than live trading did.
3. A revision could claim to be history, or to be known before the version it replaces.
4. Two writers raced between reading the latest version and writing the next, and two processes raced to migrate a new file.
5. A big move across a missing minute wasn't flagged.
6. Scoring DEV needed no trial ticket (C11), and days before the data passed the policy.

Writing the fixes surfaced five more behaviors, tested red first in c952348:

- a bad ingest stores nothing;
- flags are recorded;
- history can't backdate a minute that live trading missed;
- an action revision is known after the version it replaces;
- a view scores DEV only with a covering ticket.

**Review, round 2** (code-review skill, high, over `main...b02-bars`). Ten candidates, each checked against the code with a probe before any test was written. All ten held. Nine were fixed as red tests (030c7c9) and a green commit (e4433ac):

1. `known_at` was the caller's fetch time. A writer that fetched, then waited for the lock while a live view read, committed rows known before that view's `as_of`: live trading saw 0 bars and a replay at the same `as_of` saw 1. The store now stamps NOW itself, under the write lock and after its last stamp, and so does a live view. History may not add rows at or before that stamp. Moving the stamp before the lock, or dropping the high-water mark, makes its test fail.
2. Move flags compared minutes only within one ingest. A jump across two live ingests went unflagged, and a settle that moved a jump left the old flag and missed the new one. Flags are now recomputed over whole stored sessions and versioned.
3. A backfill could stamp dividends and splits only at fetch time, so a 2019 backtest missed a 2019 dividend. History now stamps them at the end of their declaration day.
4. No purpose let calibration and Kelly read FORWARD while refusing the lockbox, as ARCHITECTURE §6.3 requires. Calibrate is now a purpose of its own.
5. `view.minutes()` checked the days from `start` through τ, so a bounded read of FIT minutes at a lockbox τ raised LockboxError. It now checks through `min(end, τ)`, end exclusive.
7. `hourly(last=-1)` returned every bar. A negative count is now refused.
8. ROLLBACK after SQLite's own rollback replaced the real IntegrityError with OperationalError. Checking it also showed that a failed COMMIT left the connection inside a transaction. Both are fixed.
9. Cleanup: the hourly build ran twice per ingest, and two modules repeated the session lookup. One helper now maps minutes to bars, and the store builds bars once, from the overlay.
10. Docs: ARCHITECTURE v1.4, PLAN v1.1, and notes here.

Not fixed:

- **6. `ScoringTicket` is structural.** Any object shaped like a ticket unlocks DEV scoring and calibration. B10 issues the ledger's tickets with a runtime token, and the policy verifies that token then (K5). Until then nothing scores DEV, which stays dark until trial #1 at CP2 (C11).

**Review, round 3** (code-review skill, high, over `main...b02-bars` after round 2). Ten candidates, each probed against the code first. Six held and were fixed as red tests (29b78df) and a green commit (550b5f7), along with part of #10:

1. `view(τ)` quietly took `as_of = τ`, so a live reader that forgot NOW would decide on the bar before and stamp nothing. `as_of` now has no default, just as `purpose` has none.
4. An hourly read at a midnight τ was checked through τ's own day, which the read can't reach. Reading 2024 at 2025-01-01 00:00 New York raised LockboxError.
5. An ingest read, rebuilt and re-flagged every session between the first and last it touched, while holding the write lock. It now works only on runs of consecutive touched sessions.
6. A purpose passed as the string "score" skipped the forward refusal, because the policy compared by identity. The policy now converts it to the enum.
8. A NOW stamp raised history's floor even when it stamped nothing: an unchanged re-ingest, or a live view refused for coming before τ. Only a stamp that is used counts now.
9. The WAL switch compared extended error codes with SQLITE_BUSY, so a switch refused during WAL recovery wasn't retried.
10. `MinuteFrame.with_close` was used only by tests, so it moved to `tests/support`.

Not fixed, with reasons:

- **2. `ScoringTicket` is structural.** This is round 2's #6, left to B10.
- **3. Report reads DEV without a ticket.** A purpose is the caller's declaration. Operate must read DEV without a ticket anyway, because features, the filter and refits run over it, so a ticket on report wouldn't close the gap. Strategy P&L is computed only under score, and B10 and B11 make the walk-forward refuse to score DEV without a ticket.
- **7. No BUILD_LOG notes for the doc versions.** This entry is that note.
- **10, in part.** `validate_minutes` stays as the module's one-call check: it shares `check_minutes` and the flag code with ingest, so the two can't drift. `MinuteFrame.at` stays as the frame's accessor for a minute's start time.

**Review, round 4** (code-review skill, high, over round 3's diff `e4433ac...550b5f7`). Eight candidates, each checked against the code. Two held and were fixed as red tests (ebd7816) and a green commit (eb28f4c):

- An unknown purpose escaped the policy as ValueError. It now raises PolicyError, as any refused read does.
- The WAL switch read `sqlite_errorcode` from errors that may lack it, which hid the real error behind an AttributeError.

A third was a design point, taken as a refactor in the same green: each NOW writer had to record its stamp by hand. `_Stamping` now marks a stamp used when it hands it out, and the write records it as it commits. Tests fail if the stamp is never recorded, and also if it always is.

Not fixed, with reasons:

- **`flags()` defaults to the flags that stand now.** It is an operational read on the store, for alerts and reports. Flags never feed a decision, and decisions read only through the view.
- **`hourly()` checks the policy after its query.** For the last n bars, or all of them, the first day read is the first bar returned. Nothing is returned or hashed before the check.
- **Spans cost one query per table for each run of consecutive sessions.** Backfills and daily settles are one run. Only an ingest of scattered sessions pays more, and no planned job does that.
- **CLAUDE.md names PLAN v1.0.** It names v1.1: the change landed in e4433ac, the base of that diff.
- **Round 3's unfixed findings were missing from BUILD_LOG.** They are recorded in this entry.

Round 4's fixes weren't reviewed again. Each has its test, and the full gate passes.

**Deviations** (documents updated in this merge)

- ARCHITECTURE v1.3: the view's knowledge time is `as_of`, not τ (§4.2, §4.3, §6.2). Live ingest happens at τ, so the live view reads just after it, and the DecisionRecord keeps `as_of`.
- ARCHITECTURE v1.4: `known_at` is stamped at commit, under the write lock, and so is a live view's `as_of`, after the store's last stamp. History only adds rows known after that stamp, and actions are history from the end of their declaration day (§4.2, §6.2). Move flags compare against stored minutes, and cleared flags are kept (§6.1). The policy has four purposes, and calibrate reads FIT, DEV with a ticket, and FORWARD (§6.3).

  Calibrate implements §6.3 as written ("Kelly edge estimates and calibration skip the lockbox span: they use DEV walk-forward history, then FORWARD"), which the score purpose couldn't express. It changes no approved decision. On DEV it needs the trial's ticket, as C11 already requires for strategy P&L.
- PLAN v1.1: B2 builds `PointInTimeView(τ, as_of)`.
- CLAUDE.md: a trap line on NOW, HISTORY and live views.

**Open items**

- **B6:** the backfill ingests each session's minutes in one call, and uses HISTORY only before live trading first reads the store. After that, late data is NOW. It also confirms that the corporate-actions feed gives declaration dates (ARCHITECTURE §18).
- **B14:** the engine's store gets a WallClock. Live and restart ingests are NOW, never HISTORY. Each decision reads `view(τ, as_of=NOW)` and records `as_of`. Raised and cleared flags become alerts.
- **B10:** the ledger's runtime token makes tickets verifiable (finding 6).
- **B5:** T-NL1 covers the view.
- **B11:** as history, a dividend is known at the end of its declaration day. One declared on its own ex-date reaches the backtest a day late, so dividend crediting must look back to the ex-date.
- **Snapshots:** a view's reads don't share one snapshot. Stamping makes that safe for anything written NOW. A history load running alongside a backtest could land between two of its reads, so backfills run alone (B6).

## B1 · Time, calendar and canonical data (done 2026-10-05, tag `b01-done`)

**Built**

- `core.clock`: the Clock port, with `WallClock` and `SimClock`. The simulated clock moves only forward and refuses naive datetimes. It is the only module that reads the wall clock (K6).
- `core.calendar`: XNYS sessions from exchange_calendars, built once and cached, from 2015-12 to 2040. Bars are anchored at 09:30: hourly bars, then the final 30-minute bar as the CLOSE slot, on full and half days alike. Any other session shape raises `CalendarError`.
- `core.schedule`: `Timing` (feed delay 15 min, decision delay 16 min). The decision delay must exceed the feed delay, and may not exceed it by more than the 30-minute CLOSE bar, or a decision would see the next bar (I1). `decision_schedule()` covers:
  - tau for each bar;
  - the last decision that can fill, which applies the overnight cap (A3);
  - reduce-only decisions at or after the close, which execute at the next session's open.
- `core.canonical`: sorted keys, shortest round-trip floats, UTC times written with Z, numpy scalars and float64 arrays. It refuses NaN and infinity in both directions, naive datetimes, float32, non-string keys, duplicate keys and bad UTF-8. `sha256_hex` hashes the canonical bytes.
- `core.ids` (`decision_id` from tau), `core.model` (`StrictModel`: no unknown keys, no coercion, no NaN, frozen), and `core.errors` (the `QuantError` hierarchy).
- `core.logs`: one redactor for structlog and stdlib loggers. It covers known values (environment plus `register_secret`), field names that look secret (camelCase, plurals), key-shaped strings (Alpaca, Anthropic, Telegram, bearer), and a last pass over the rendered line. Timestamps come from the Clock port.
- Trace ids now include doc sections (`SPEC§2`, `ARCH§4.3`, `PLAN§5`) for requirements written as prose, in natural order.

**Tests:** 55 new (154 fast in total). Highlights:

- All 2,765 sessions from 2016 to 2026, including the 23 half days, build a valid schedule.
- The named half days (2019-07-03, 2019-11-29, 2019-12-24) and both daylight-saving changes behave as specified.
- A hypothesis property: each bar is available before its decision, and the next bar is not.
- Canonical JSON round-trips and hashes the same whatever the key order.
- Log redaction works through structlog, stdlib loggers and tracebacks.

**Run times:** fast gate about 11 s (154 tests in 9 s).

**Review** (code-review skill, high): 10 findings, 9 fixed, as red tests first and then fixes:

1. A decision delay could reach the next bar.
2. `1e400` loaded as infinity.
3. A 0-d array raised TypeError.
4. Bad UTF-8 raised UnicodeDecodeError.
5. `decision_id` raised a plain ValueError.
6. The calendar ended in 2027.
7. Secret field names in camelCase, plural, `private_key` and `passphrase` were missed.
8. Secrets learned after configuration weren't redacted.
9. Log timestamps used the wall clock.

The cut cosmetic item, trace sort order, was fixed too.

Not fixed, with reasons:

- **Double scrubbing of values** (event dict and rendered line). This is deliberate defense in depth: the rendered-line pass catches reprs and tracebacks the first pass can't see, and at a few lines per bar the cost is negligible.
- **`canonical` repeats the naive-datetime check** that `clock.ensure_utc` has. It is kept so the error type stays `CanonicalJSONError`.

**Deviations:** none from the docs. Section ids extend the trace rule in PLAN §2, and CLAUDE.md documents them.

**Open items:** the settings loader (B6) must call `register_secret` for every key it reads from `.env`.

## B0 · Bootstrap and structural guards (done 2026-10-05, tag `b00-done`)

**Built**

- Git repo, with approved docs as the base commit on `main`. uv project pinned to Python 3.12.14, with the AD5 stack locked in `uv.lock` (plus uvicorn, jinja2 and pyyaml, which FastAPI and the YAML configs need). gitleaks 8.30.1 is installed from Homebrew.
- The package skeleton from ARCHITECTURE §3. The modules the rules name (`core.clock`, `hmm.train`/`filter`/`labels`, `eval.smoother`, `risk.engine`, `research.ledger`, `execution.alpaca_broker`) exist as placeholders that name the step that builds them.
- `quant-check` (`src/quant_check/`). Its stages:
  - python pin, ruff, ruff format, mypy (strict everywhere);
  - **imports**: import-linter contracts K1–K4 and P11, indirect chains included. A completeness check means a new subpackage must be placed in K1, K2 and K4. AST checks cover what import-linter can't see: `alpaca.trading` outside the adapter, and siblings in `quant.execution` importing the adapter.
  - **K5, K6, I2**: AST scans that resolve import aliases. K5 also catches copies of an instance annotated with a guarded type.
  - **I6**: gitleaks over a snapshot of the committable files, so the gitignored `.env` files are never read, and over the whole history.
  - **P2**: red marks may exist only on their own step's branch, never after `bNN-done`, and none at merge.
  - **hook**: both hooks installed.
  - **tests**: fast or full tier.
  - **trace**: every claimed id has a green test.
- pytest plugin:
  - tiers;
  - an offline fast tier (sockets, datagrams and DNS refused);
  - hypothesis profiles (fast is derandomized);
  - strict red marks that must name their expected failure;
  - spec ids required on every test and checked against the docs;
  - JSON outputs for the gate;
  - the fast-tier budget (120 s), checked in both tiers.
- `.githooks/pre-commit` runs the fast gate and refuses unstaged or untracked files, so the gate checks exactly what gets committed. `.githooks/pre-merge-commit` runs the full gate on every `git merge --no-ff`.
- `CLAUDE.md`, `.env.example`, `.gitignore`, `.gitleaks.toml` (default rules plus Alpaca key formats), `spikes/README.md`, `tests/steps.toml`.

**Tests:** 102 (99 fast, 3 full). Every guard is shown catching a planted violation in a copy of the repo (P3): K1–K6, I2, I6, P11, red marks that pass, fail for the wrong reason, are malformed, zero-padded, or outlive their step, missing or unknown spec ids, trace claims, the offline tier, tier selection, the fast-tier budget, a commit blocked by the hook, and a merge blocked by the merge hook. Clean code passes every stage.

**Run times:** fast gate about 10 s (99 fast tests in 8 s). Full gate 16 s.

**Review** (code-review skill, high): 10 findings, all fixed in 85944c7 (red) and the review-green commit after it:

1. K5 missed copies of an approved instance (`model_copy`, `dataclasses.replace`, `copy.copy`).
2. K4 let siblings in `quant.execution` import the adapter.
3. A broken contract with no import chains produced no finding.
4. Merges skipped the gate. Fixed with the pre-merge-commit hook, and the trace check now includes the step being merged.
5. The fast-tier budget was only a note.
6. K6 missed `process_time`, `thread_time`, `clock_gettime`, `os.times`, `uuid1` and event-loop time.
7. I2 let `eval/` import hmmlearn.
8. The offline guard missed DNS lookups and UDP.
9. Zero-padded red marks (`red:B03`) broke the audit. Steps are now unpadded, and the plugin enforces it.
10. The red-mark pattern was defined twice.

**Deviations** (documents updated in this merge)

- ARCHITECTURE v1.2: the I2 scan treats `decode` as Viterbi only when it is called with hmmlearn's arguments, so `bytes.decode("utf-8")` passes. A merge hook runs the full gate.
- `numpy` is held below 2.5, because hmmlearn 0.3.3 sets `ndarray.shape`, which NumPy 2.5 deprecates. This is in pyproject.toml and the CLAUDE.md traps.

**Open items**

- The static scans catch the usual ways of writing a violation, not deliberate evasion (`getattr` with variables, `importlib`). The runtime defenses come later: the approved-order token and one-time ids (B9), the ledger's ticket token (B10), and T-NL3 (B4).
- The offline guard covers this Python process only, not subprocesses.
