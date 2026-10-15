# PLAN — SPY Regime Trader

**Status:** APPROVED v1.3 (Gate 3 of 3) · 2026-10-05. Approved as written: decisions P1–P11 confirmed and clarifications C11–C17 accepted. C11–C17 are now part of SPEC.md v1.2. v1.1 names the view's knowledge time in B2, `PointInTimeView(τ, as_of)`, as ARCHITECTURE §6.2 does. v1.2 adds step B20, Jev as a shadow advisor (SPEC v1.4 §12a, ARCHITECTURE v1.6), which you approved on 2026-10-13 (§0, §0.3, §1, §4). v1.3 records two calls of yours: B9 may merge before B5–B8, which you decided on 2026-10-13 (§4 B9), and every step merges through a pull request on the private GitHub repo, decided on 2026-10-14 (§2 rule 7).
**Builds on:** [SPEC.md](SPEC.md) v1.6 (v1.1 plus this plan's C11–C17, B3's reading of §3, the Jev shadow advisor, B9's resume and kill calls, and B4's reading of §4–§6) and [ARCHITECTURE.md](ARCHITECTURE.md) v1.9 (v1.0 aligned with this plan, notes from B0, B2, B4, B6, B9 and B10, and the Jev shadow advisor), approved 2026-10-05 and amended since; the shadow advisor was approved 2026-10-13
**Process:** brainstorm ✅ → spec ✅ → architecture ✅ → plan ✅ → **test-first build** ▶ → review → ship
The build started 2026-10-05 with B0. Progress is in [BUILD_LOG.md](BUILD_LOG.md).

---

## 0. What you approved

This plan turns the architecture into an ordered build: 19 steps (B0–B18) in seven milestones, plus one step (B19) that waits until it's needed and one add-on (B20, the Jev shadow advisor, v1.2). For every step it lists the tests written first, the spec and architecture IDs they enforce, and what "done" means. It also fixes the seed strategy, r1, before anyone looks at the data, and it names the five checkpoints where I stop for you.

Four ideas carry the plan:

1. **Guarantees before features.** The point-in-time view and the truncation harness (B2–B5) exist before any switching, sizing or P&L code. Every later step that touches the decision path extends that harness to cover itself.
2. **Every guard is proven against a planted violation.** Each import rule, scan, choke point and no-lookahead test is shown catching a deliberate violation. A guard that has never failed proves nothing.
3. **The development period stays dark until r1 is frozen.** Building and debugging use synthetic data and the 2016–2018 fit window. The first P&L on 2019–2024 is r1's run as trial #1, at checkpoint CP2.
4. **Your time goes to what only you can do:** keys, accounts, one sudo script, the lockbox seal and five checkpoints.

**On approval (2026-10-05),** this plan became APPROVED v1.0, C11–C17 went into SPEC.md as v1.2, ARCHITECTURE.md became v1.1 where this plan changes it (C11, C14, P4, P11), and B0 started.

### 0.1 Plan decisions (confirmed 2026-10-05)

| ID | Decision | Proposed default | Why |
|---|---|---|---|
| P1 | Build order | The 19 steps in §1. They follow ARCHITECTURE §19 with four changes: a bootstrap (B0–B1) comes first; real market data arrives at B6, after the core is proven on synthetic data; the trial ledger (B10) comes before the backtest (B11); and the go-live gate becomes the deferred B19 (P9). | The guarantees that are hardest to retrofit (no lookahead, one choke point, an honest trial count) are proven before any strategy logic depends on them. With the ledger first, no development-period score can exist without a ledger entry. |
| P2 | How "failing test first" works | Each step's tests are committed first, alongside stubs that raise `NotImplementedError`, with a strict expected-failure mark that names the step: `xfail(strict=True, reason="red:B3")`. The mark also names the failure it expects (normally the stubs' `NotImplementedError`), so a test that fails for another reason, such as a typo, still breaks the build. The green commit removes the marks. `quant-check` fails if a marked test passes, or if a mark outlives its step. | Spec §17 commits failing tests first, and ARCHITECTURE §13 runs the whole suite before every commit. Strict marks satisfy both, and the commit hook itself proves that each test failed before the code existed. |
| P3 | Planted violations | Every guard gets a test that plants a violation and expects the guard to catch it: K1–K6, the I2 scan, T-NL1–3, I4, I5, I6 and the ledger chain. | See idea 2. |
| P4 | Test data and speed | **Fast tier** (every commit, under 2 minutes, offline): synthetic bars from a seeded regime-switching model, laid on the real NYSE calendar; T-NL1 at 50 cut times. **Full tier** (before each merge, and nightly): real bars from `var/market.db`, T-NL1 at 200 cut times, golden and stress tests. The golden window is 2018-01 → 2018-04, inside FIT and including the February 2018 volatility shock. Its decisions come from a fixture model fit on 2016–2017, so T-NL2 holds. ARCHITECTURE §13's early-2020 window is added after CP2. The golden data (about 2 MB compressed) is committed, so the repo should stay private. | Commits stay fast and need no network, and the expensive checks still run before anything reaches `main`. The golden window moves because early 2020 is in DEV, which stays dark until trial #1 (C11). |
| P5 | Git | A local repo with one branch per step (`b03-features`), merged to `main` and tagged (`b03-done`) when the step is done. No remote until the ship phase. If you want off-machine backup sooner, say so and I'll create a private GitHub repo with `gh`. | Tags make every step a rollback point. Pushing code anywhere is your call. |
| P6 | When I stop for you | I work through the steps without stopping, except at: hard checkpoints CP2–CP5 (reply "approve CPn"); anything that needs your keys, accounts, sudo or money; an irreversible act; or any change to an approved decision. CP1 is a report and doesn't wait for you. Every step also gets an independent code review (the code-review skill), and its findings are fixed before the merge. | Your time goes to decisions only you can make. |
| P7 | Seed release r1 | The values in §5, fixed now, before any data is looked at. r1 becomes trial #1 at CP2. | Choosing them after seeing DEV results would turn trial #1 into a disguised fit. |
| P8 | Feature menu v1 | F1–F5 with the bounded parameters in §5. F1 and F2 are always on. There are no other transforms in v1; adding one is a normal build step. | A small menu means a small search space. That keeps down the nightly loop's room to overfit, and its trial count. |
| P9 | Go-live machinery | Deferred to B19. The checklist validator, the lockbox test and the April 2025 replay are built when a candidate first clears G1–G6 on DEV, or when you ask. Until then the live client can't be constructed (I5), and no code exists that could open the lockbox. Parity replays, stress tests and the weekly kill test are built anyway (B11, B13–B15). | Spec §0 expects the first pass to stay in paper. Deferring also leaves the most sensitive data path unbuilt until something needs it. |
| P10 | Paper start | On this Mac at CP3: after the live engine (B14) and the kill paths (B15) work, and before the dashboard and the nightly loop. The lockbox is sealed the day before the first decision. The Mac period is a shakedown; go-live evidence starts on the VPS (C17). | Real fills and real outages teach the most, and they start while the rest is being built. |
| P11 | Spikes | Throwaway checks of vendor behavior, such as the Alpaca probe in B6, live in `spikes/`. They are exempt from red/green, can't be imported by `src/` (import-linter), and record their findings in `docs/VERIFY.md`. | Checking how a vendor behaves isn't production code, and it shouldn't pretend to be. |

### 0.2 Spec clarifications (accepted 2026-10-05)

Writing the plan surfaced seven more questions that the code has to answer. They continue the numbering from the architecture's C1–C10, and they are now part of SPEC.md v1.2.

| ID | Open question | Proposed answer |
|---|---|---|
| C11 | What counts as a trial (A2), and how is DEV protected during the build? | Before trial #1, nothing computes strategy P&L on DEV; the build uses synthetic data and FIT. r1 becomes trial #1 at CP2. After that, a scored DEV run of a candidate is a new trial whenever its decision records differ from every earlier scored run, including when they differ only because the code changed. Re-running the same configuration on the same code reuses its trial. Baselines and cost-sensitivity runs are yardsticks, not candidates, so they are not trials. A bug fix chosen after seeing DEV results is also a look at the data. Counting it costs little, because the Deflated Sharpe penalty grows slowly with the trial count. |
| C12 | Kelly sizing needs "the playbook's own past per-bar returns." Actual fills, or the signal's? | Shadow returns: the per-bar return the active playbook's signal would have earned at 1.0×, net of modeled costs. They are counted on bars where its label is active and the signal is in a position, whether or not a position was actually taken. With fewer than 100 such bars, f* = 0. Actual fills would lock a playbook at size zero after one weak stretch, because a playbook sized at zero earns no new history. |
| C13 | How much data does a calibration verdict need (A4)? | At least 1,000 out-of-sample bars, about 7 months. Until then every label counts as failing, so each is sized at no more than 25% of its cap (spec §9). The 35 bars before the as-of time are left out, because their smoothed labels haven't settled. The outcome check takes the central 90% one-bar-ahead interval for F1 over the latest 1,000 out-of-sample bars. It passes when coverage is between 87.6% and 92.4% (the 99% binomial band). A failure caps every label at 25% of its cap. |
| C14 | "The best static strategy": best when, and among what? | G5 checks three baselines, and the system must beat each one on both Sharpe and max drawdown: buy-and-hold, walk-forward trend-following and walk-forward mean reversion. Each static family re-picks its parameters every month from the fixed grid in §5, by the best Sharpe on its training window. It holds 1.0× when in position and pays the same costs. Beating both families is at least as strict as beating whichever turns out better, and it removes the question of when "better" gets decided. |
| C15 | What is one trade, for G3's hit rate and its 100-trade minimum? | A trade runs from flat to flat. Scale-ins, resizes and partial exits inside it belong to that trade, and its net P&L includes every fill and fee. |
| C16 | What does the backtest do after the 10% drawdown kill? | It stays flat for the rest of the run, as live trading would until you resume, and the report marks the kill. Simulating an automatic resume would invent a decision you haven't made. |
| C17 | What counts toward go-live item 1's 3 months and 100 fills? | Decisions made on the VPS from a clean, tagged commit, counted from the last change to decision-path code (`data`, `features`, `hmm`, `decide`, `risk`, `execution`, `engine`). A promotion doesn't restart the count; a code change does. The parity replay must reproduce every decision with one codebase, and mixing code versions inside the window would need a separate replay per version. |

### 0.3 What I need from you, and when

| When | What | Time |
|---|---|---|
| 2026-10-05 ✅ | Approve this plan (P1–P11, C11–C17): approved as written | — |
| By B6 | Alpaca paper keys for your trading account in `.env`. Add them yourself; don't paste them into the chat. | 5 min |
| By B13 | A second Alpaca paper account for tests, with its keys in `.env.test` (AD3). Until CP3 the trading account can stand in, because it has no track record yet. | 5 min |
| By B15 | A Telegram bot made with @BotFather, its token in `.env`, and one message sent to it; `quant control whoami` then finds your chat id. Later, one test `/kill` from your phone against the test account. | 10 min |
| CP2 | Read the first honest look and reply | 10 min |
| CP3 | Agree to seal the lockbox and start paper. The trading account must be flat with no open orders; resetting the paper account does that. Keep the Mac awake and on power during market hours. | 5 min |
| B17 | Read and run one sudo script, which creates `_quantllm`, `/opt/quant-llm` and the L1 spool folders. Then put your Anthropic API key in `/opt/quant-llm/.env.llm` yourself. | 15 min |
| By B20 | A TypeSafe AI API key, in `.env` for the spike and then in `/opt/quant-llm/.env.shadow` for the job. Add it yourself; don't paste it into the chat. Then run the updated sudo script once more, which adds the shadow spool folders (v1.2). | 5 min |
| CP4, CP5 | Read the report and reply | 10 min each |
| Ship phase | A VPS: Ubuntu 24.04, 4 vCPU, 8 GB of RAM | later |

If something isn't ready when I reach the step that needs it, I continue with the steps that don't and come back.

---

## 1. The build at a glance

| Step | Builds | Proves | Size | Needs you |
|---|---|---|---|---|
| **M1 Foundations** | | | | |
| B0 | Repo, toolchain, `quant-check`, commit hook, CLAUDE.md | K1–K6 and the I2 and secret scans catch planted violations | S | |
| B1 | Clock, calendar, decision schedule, canonical JSON, redacting logs | Decision times are right on every session from 2016 to 2026 | S | |
| **M2 Data and no lookahead** | | | | |
| B2 | Bar store, bar builder, point-in-time view, DataPolicy | A row is never visible before its `available_at` | M | |
| B3 | Features F1–F5, scaler, feature menu | Streaming matches a reference; no future rows used | M | |
| B4 | HMM training, selection, filter, labels, smoother | T-NL3; a known synthetic HMM is recovered | L | |
| B5 | Truncation harness (T-NL1, T-NL2) | It catches six planted leaks | M | |
| B6 | Alpaca data, backfill, probe, first model | D2's timing holds; first model report | M | keys · **CP1** |
| **M3 Decisions and risk** | | | | |
| B7 | Refit, label matching, drift, calibration | C5, C7, A4; T-NL1 covers refits | M | |
| B8 | Playbooks, switching, sizing, `decide()`, release r1 | A1 and rules 1–6; never short, never over a cap | L | |
| B9 | Risk engine, approved-order types, kill executor | I4; every limit vetoes at its boundary | M | |
| **M4 Honest backtest** | | | | |
| B10 | Trial ledger and `TrialTicket` | A2: no DEV score without a ledger entry | S | |
| B11 | SimBroker, walk-forward, costs, reports | One codepath; deterministic; exact fills and fees | L | |
| B12 | Gates G1–G6, baselines, gate report | r1 runs as trial #1 | M | **CP2** |
| **M5 Paper trading** | | | | |
| B13 | Alpaca broker adapter, order manager | I4 and I5 at the real boundary; the kill test | L | test account |
| B14 | Live engine, scheduler, recovery, parity, backups | A restart never double-sends or trades late | L | |
| B15 | quant-control: Telegram, kill paths, alerts | The kill switch works with the engine down | M | Telegram · **CP3** |
| **M6 Operations** | | | | |
| B16 | Dashboard and daily report | Local only, CSRF-protected, read-only | M | |
| B17 | Nightly loop: bundle, L1, nightly harness, promotion | Hostile proposals are rejected; L1 is isolated | L | sudo, API key · **CP4** |
| **M7 Deployment** | | | | |
| B18 | systemd units, hardening, `adopt-account` | Backups restore bit for bit | M | **CP5** |
| **Deferred** | | | | |
| B19 | Go-live gate: checklist, lockbox test | (P9) | M | when a candidate clears G1–G6 |
| **Add-ons** | | | | |
| B20 | Jev shadow advisor: spool, quant-shadow, ingest, evening scoring (v1.2) | I7 and K7; a late answer is never scored; nothing on the decision path reads Jev | M | TypeSafe key |

**Sizes:** S is a small module set and M a focused subsystem. L covers several subsystems and is split into sub-steps (B4a, B4b, …), each with its own red and green commits.

## 2. Rules for every step

1. **Branch:** `bNN-name`, from `main`.
2. **Red:** stubs plus the step's tests, marked `red:BNN` (P2). Every test names the spec IDs it enforces, for example `@pytest.mark.spec("A1")`.
3. **Green:** the implementation goes in and the marks come off. The fast tier passes on every commit, and the full tier passes before the merge.
4. **Truncation harness:** a step that touches the decision path adds its layer to T-NL1 before it counts as done.
5. **Review:** an independent code review of the step's diff, checked against the step's IDs. Every finding is either fixed or written down in the build log with a reason.
6. **No silent deviations:** if the code must differ from SPEC, ARCHITECTURE or this plan, the document changes in the same merge, with a minor version and a build-log note. If the change touches an approved decision, I stop and ask first.
7. **Done:** merge to `main`, tag `bNN-done`, and add an entry to `docs/BUILD_LOG.md`: what was built, tests added, run times, deviations and open items. The merge goes through a pull request on the private GitHub repo: push the branch and open the PR, merge locally with `--no-ff` so the merge hook runs the full gate, then push `main` and the tag. GitHub records the PR as merged (v1.3).
8. **Resuming:** every session starts from CLAUDE.md, BUILD_LOG.md and `git status`, then continues the open step. Nothing depends on chat memory.

`quant-check --trace` prints a matrix of spec IDs against tests. A step can't merge while an ID it claims to cover has no test.

## 3. Checkpoints

| ID | After | You see | Waits for you |
|---|---|---|---|
| CP1 | B6 | The first model on 2016–2018: the chosen K and covariance type, the transition matrix, the state table and labels, and the BIC and validation table. Also the probe results (`docs/VERIFY.md`) and a data-quality report. | No. Say "stop" if something looks wrong. |
| CP2 | B12 | The first honest look: r1 as trial #1 on 2019–2024, with G1–G6, the three baselines, per-label results, the March 2020 and 2022 windows, and the 5 bp cost run. | Yes |
| CP3 | B15 | A readiness list: integration tests green, the kill test passed on the test account, recovery drills passed. Then the lockbox seal, the model chain refit up to the seal (C1 allows refits), and the first paper decision. | Yes, because sealing can't be undone |
| CP4 | B17 | The first full night: the bundle, Opus's proposals, how the nightly harness graded them, and the ledger entries. | Yes |
| CP5 | B18 | Build complete: the trace matrix, test counts and timings, and open items. | Yes, then review and ship |

If r1 fails the gates at CP2, as spec §0 expects, the build continues as planned. Paper trading then tests the plumbing with r1, and the nightly loop takes over the search through the gates. If you'd rather change direction, CP2 is the cheapest place to do it.

## 4. The steps

### M1 — Foundations

#### B0 · Bootstrap and structural guards (S)

- **Builds:**
  - `git init`, and the uv project with the package skeleton from ARCHITECTURE §3, pinned to Python 3.12.14 (uv has already installed it on this Mac);
  - the AD5 dependencies, locked in `uv.lock`, and gitleaks from Homebrew;
  - `quant-check`: ruff, mypy, import-linter contracts K1–K4, AST scans for K5, K6 and I2, gitleaks, the red-mark audit, the test tiers and `--trace`;
  - the commit hook, `.gitignore`, `.env.example`, `spikes/`, `docs/BUILD_LOG.md`, and a CLAUDE.md carrying §2's rules.

  Services will call the project's own Python directly, never `uv run`, because your uv lives in `/opt/anaconda3`.
- **Red tests first:** each guard runs on a copy of the repo with one planted violation, and must fail and name its rule:
  - `risk` importing `llm` (K1);
  - `llm` importing `execution` (K2);
  - `decide` importing `hmmlearn` or `eval` (K3);
  - `dashboard` importing the Alpaca adapter (K4);
  - `RiskApprovedOrder` built outside `risk.engine` (K5);
  - `datetime.now()` in `features` (K6);
  - `.predict_proba(` in `hmm.filter` (I2);
  - a fake API key in a tracked file (I6);
  - a red test that passes, and a red mark left over from a finished step.
- **Covers:** K1–K6; the static parts of I2, I3, I4 and I6; spec §17.
- **Done when:** `quant-check` is green on the skeleton, every planted violation fails it, and the hook blocks a bad commit.

#### B1 · Time, calendar and canonical data (S)

- **Builds:**
  - the Clock port, with wall and simulated clocks;
  - the XNYS calendar with early closes, and the session map: seven bars, with the final 30-minute bar as the CLOSE slot on full and half days alike;
  - decision times: τ = bar close + 16 min, the last decision that can fill, and reduce-only decisions that execute at the next open;
  - canonical JSON with SHA-256 hashing, ids and error types, and strict pydantic models that reject unknown keys;
  - structlog with a redacting filter.
- **Red tests first:**
  - the decision schedule on a normal day, on half days (2019-07-03, 2019-11-29, 2019-12-24) and across daylight-saving changes;
  - schedules for every session from 2016 to 2026 build without error;
  - canonical JSON round-trips any value (hypothesis), rejects NaN and hashes the same whatever the key order;
  - a log line containing a secret comes out redacted (I6).
- **Covers:** I1 (the time contract), I6 (logs), K6, spec §2, ARCHITECTURE §5.
- **Done when:** green.

### M2 — Data and no lookahead

#### B2 · Bars and the point-in-time store (M)

- **Builds:**
  - `market.db` with migrations: minute and hourly bars stored as versions with `available_at` and `known_at`, plus dividends, splits and a calendar cache;
  - the hourly bar builder;
  - validation: OHLC order, duplicates, session bounds, and a flag on any one-minute move over 5%;
  - `PointInTimeView(τ, as_of)` (v1.1: ARCHITECTURE §6.2);
  - `DataPolicy`, with the FIT, DEV, LOCKBOX and FORWARD segments and purpose-tagged queries;
  - cache keys built from the bytes read through the view.
- **Red tests first:**
  - a full synthetic day builds seven correct bars, and a half day builds four;
  - a missing last minute leaves its bar incomplete;
  - a row with `available_at` equal to τ is invisible, because the inequality is strict;
  - a bar that has closed but isn't yet available is invisible;
  - a revision made after τ is invisible at τ and visible afterwards;
  - for any τ1 < τ2, the view at τ1 is contained in the view at τ2;
  - DataPolicy refuses to score LOCKBOX;
  - changing an invisible row leaves the cache key unchanged, and changing a visible one changes it.
- **Covers:** I1 (the data contract), spec §2, A2 (where the lockbox is enforced), C1, AD2 (the first SQLite file and the migration pattern the others reuse), ARCHITECTURE §6.
- **Done when:** green, and ten years of synthetic minutes build into hourly bars in under a minute.

#### B3 · Features and the scaler (M)

- **Builds:**
  - streaming F1–F5 with per-slot history, plus slow reference versions used only by tests;
  - a scaler fit on a window and then frozen, and the 60-session warm-up;
  - NaN reported as a model error;
  - the feature menu (§5) and the feature-config hash.
- **Red tests first:**
  - streaming matches the reference within 1e-12, and two runs agree bit for bit;
  - changing any bar after t leaves every feature at t unchanged;
  - the scaler ignores data after its window;
  - the CLOSE slot is normalized against CLOSE history only, half days included;
  - the warm-up drops 60 sessions;
  - the menu rejects unknown transforms, out-of-bounds parameters and configs without F1 or F2.
- **Covers:** spec §3, D3, P8, I1 (features).
- **Done when:** green, and ten years of features compute in under 5 seconds.

#### B4 · HMM: train, select, filter, label (L)

- **Builds:**
  - `hmm.train`, the only module that imports hmmlearn. It calls only `fit`, runs the 50 restarts in a process pool with single-threaded BLAS, and warns when the optimum is unstable;
  - selection by BIC and validation log-likelihood, with a 1,000-draw session-block bootstrap, and the sanity rejects;
  - model artifacts: `params.npz` and `meta.json`, never pickles;
  - our log-space filter, emitting every output in spec §6;
  - labels from v and d, and `eval.smoother`.
- **Red tests first:**
  - on 20,000 bars drawn from a known 3-state HMM, training recovers the parameters (up to a relabeling), and selection picks K = 3 and the generator's covariance type;
  - the filter matches a naive probability-space reference within 1e-10, and stays finite over 100,000 bars;
  - T-NL3: our filter differs from hmmlearn's posteriors at interior points and matches them at the last point;
  - next-bar probabilities equal p·A, and entropy lies in [0, 1];
  - deliberately bad models trip each sanity reject;
  - the label rules hold at their exact boundaries (v = 1.25 and 1.75, d = 0.5 and −0.5), and probabilities of states that share a label add up;
  - artifacts round-trip bit for bit and load with `allow_pickle=False`, and the same seeds give the same artifact hash.
- **Covers:** spec §4 and §6, D4, T-NL3, I2.
- **Done when:** green, with the time for a full K × covariance selection measured and logged.

#### B5 · The truncation harness (M)

- **Builds:** the T-NL1 harness, plus T-NL2's provenance check. For each cut time τ, it compares every record up to τ across three runs:
  - data up to τ;
  - the full history;
  - data up to τ followed by garbage: NaN, ±1e9, shuffled bars, late revisions, and bars that have closed but aren't yet available.

  Later steps register their layers with it.
- **Red tests first:** six planted leaks, each of which must be caught:
  - a centered moving-average feature;
  - a scaler fit on the full history;
  - the filter swapped for hmmlearn's `predict_proba`;
  - a view that uses ≤ instead of <;
  - availability taken from `t_close` instead of `available_at`;
  - a model whose fit window ends after τ (T-NL2).
- **Covers:** I1, I2, T-NL1, T-NL2.
- **Done when:** every planted leak fails, clean code passes, and the fast-tier run takes under 60 seconds.

#### B6 · Real data and the first model (M) · needs your keys

- **Builds:**
  - the Alpaca market-data client: historical SIP minutes requested raw, the latest IEX quote and trade, the calendar and corporate actions;
  - a resumable backfill from 2016-01-01 to today;
  - `spikes/alpaca_probe.py`, which checks the items in ARCHITECTURE §18;
  - a data-quality report;
  - selection on FIT, which produces the first model and its report.

  Downloading lockbox-period bars is allowed (C1); nothing in this step computes a strategy result.
- **Red tests first:**
  - the client against recorded responses: pagination, a 429 with retry-after, empty pages, a missing minute, and requests that explicitly ask for SIP and raw prices;
  - a backfill that is interrupted and resumed gives identical rows.
- **Covers:** D1, D2, D4, spec §2 and §4, ARCHITECTURE §6.1 and §18.
- **Done when:** `market.db` holds 2016 to today, every gap in the quality report is explained, and the probe results are in `docs/VERIFY.md` → **CP1**.
- **If the probe finds that bars arrive later than close + 15 min,** the delay is one config value, applied to the backtest and live trading alike. I'll propose the new value at CP1 rather than work around it.

### M3 — Decisions and risk

#### B7 · Model lifecycle: refit, drift, calibration (M)

- **Builds:**
  - the monthly refit job: expanding window, fixed K and covariance type, the previous model seeding one restart, and a sanity reject that keeps the current model;
  - Hungarian label matching on symmetric KL;
  - the five drift checks, with C5's store-without-activating and entry freeze;
  - activation at the next open (C7), through the pointer history;
  - calibration: Brier score, Brier skill, 10-bin reliability and ECE against smoothed labels; the outcome check; isotonic recalibration on past out-of-sample data; C13's sample rules; artifacts with an as-of time;
  - the live drift monitor.
- **Red tests first:**
  - a model with its states permuted is matched back exactly;
  - each drift rule fires at its threshold and not just below it;
  - a failed refit is stored but not activated, and freezes entries, and activation waits for the next open;
  - a well-calibrated synthetic forecaster passes; a miscalibrated one fails, and isotonic recalibration fixes it;
  - fewer than 1,000 out-of-sample bars count as failing, and the newest 35 bars are excluded;
  - a calibration artifact's as-of time precedes every use (T-NL2).
- **Covers:** spec §5 and §9 (calibration), D6, A4, C5, C7, C13, T-NL1 and T-NL2 over refits.
- **Done when:** green, and the nightly full tier covers monthly refits through 2018 on real data, starting from a model fit on 2016–2017.

#### B8 · Decisions: playbooks, switching, sizing (L)

- **Builds:**
  - the playbook schema: strict, bounded, with a typed invalidation menu;
  - the `trend_following`, `mean_reversion` and `flat` primitives, with ATR stops;
  - health (rule 6 with its two-bar recovery), switching (rules 1–5 and A1) and invalidation blocks;
  - sizing per spec §9: C12's Kelly, the calibration floor, the deadband and the caps;
  - intents: at most 0.5× per order, the overnight cap on the last decision that can fill, and reduce-only closing-bar decisions;
  - EngineState, DecisionRecord and the pure `decide()`;
  - release r1: `strategy.md` plus four playbooks with the §5 values.
- **Red tests first:**
  - switching scenario tables, one per rule: a 3-bar takeover held back by the cooldown, an A1 de-risk switch after one bar that ignores the cooldown, and rules 4–6 acting on the bar that triggers them;
  - property tests: never short; never over a cap, or over 0.5× in one order; label probabilities sum to 1; size is 0 in CRASH and when uncertain, halted or frozen;
  - the schema rejects unknown keys and out-of-bounds values;
  - invalidation blocks entries until a fresh takeover;
  - Kelly gives 0 without an edge or with fewer than 100 bars;
  - the overnight cap applies at 15:46, and at 12:46 on half days, and the 16:16 decision can only reduce;
  - the deadband holds except for forced reductions;
  - `decide()` runs with file, network and clock access disabled, and gives the same record hash every time.
- **Covers:** spec §7–9, D5, D8, D9, A1, A3, C4, C12, P7, I1 (T-NL1 extended through sizing).
- **Done when:** green, and r1's files validate against the schema.

#### B9 · Risk engine and choke points (M)

- **Order (v1.3):** B9 depends only on `core` and `store`, so it was built beside B4 and merged before B5–B8. You agreed on 2026-10-13. B8's sizing and B13's execution call into the engine as specified here.
- **Builds:**
  - `RiskApprovedOrder`, `RiskApprovedCancel` and `RiskApprovedFlatten`, each guarded by a private token, with every issued id registered and spendable once;
  - `RiskEngine.check`, running ARCHITECTURE §8.1's seven checks in order, and the risk-reducing clamp;
  - the `risk.yaml` loader, frozen after loading and compared with the committed version;
  - C3's counters, the daily-loss lock and the drawdown trigger;
  - the reconciliation comparator and the minute-monitor logic (C8);
  - the kill executor, with its lock and its outside-hours variant, and the broker port.
- **Red tests first:**
  - forged approvals are rejected (plain objects, subclasses, objects built without the token, ids that were never issued), and so are reused and expired ones;
  - each check vetoes at its boundary;
  - risk-reducing intents pass while HALTED, and are clamped so they never go short;
  - PendingApproval appears only when live and above $25,000;
  - 21 intents in a day halt trading, and so do 60 raw submissions;
  - the loader refuses a `risk.yaml` that differs from the committed version;
  - against a fake broker, the kill executor halts, cancels, flattens, verifies and alerts; outside hours it keeps the stop and queues the exit;
  - property: no sequence of intents and fills produces a short or breaches a cap.
- **Covers:** spec §10, D7, D13, I4, C2 (trigger logic), C3, C8, ARCHITECTURE §4.4, §8.1 and §8.3.
- **Done when:** green, with T-NL1 extended through the risk verdicts.

### M4 — Honest backtest

#### B10 · Trial ledger (S)

- **Builds:**
  - `research.db` and the append-only ledger, in which each row hashes the one before it;
  - `TrialTicket`, issued only after its trial is appended;
  - C11's trial identity, and records of gate-config changes;
  - `quant ledger verify`.

  In v1, no lockbox ticket can be issued at all (P9).
- **Red tests first:**
  - K5 for `TrialTicket`;
  - tampering with any row fails `verify`;
  - scoring DEV without a ticket raises an error and still leaves a ledger entry;
  - an identical re-run reuses its trial, and a run with different records adds one;
  - asking for a lockbox ticket fails.
- **Covers:** A2, C11, K5.
- **Done when:** green.

#### B11 · Backtest (L)

- **Builds:**
  - the engine loop on a simulated clock. This is the same loop code that runs live;
  - SimBroker: fills on the first SIP minute at or after τ, plus slippage; stops on minute bars, filled at the open when a gap jumps them; next-open exits; fees from `costs.yaml` by effective date; dividends on the ex-date;
  - the minute monitor on minute closes;
  - the walk-forward: the first model, then monthly refits from a cold start, with the refit cache;
  - the fit-window run that seeds Kelly's shadow returns (spec §9, C12);
  - the spec §11 reports and C16's kill handling;
  - the golden test, and synthetic stress tests: a −10% overnight gap, a 30-minute data outage and a broker outage.
- **Red tests first:**
  - fills, fees and dividends are computed by hand for small scenarios, including a gapped stop that fills at the open;
  - the 16:16 decision executes at the next open;
  - two runs give identical record hashes;
  - T-NL1 covers the whole chain, from features to orders;
  - refits run at the same simulated times as in live trading;
  - the cache hits and misses when it should.
- **Covers:** spec §11, I1 end to end, C4, C8, C12, C16.
- **Done when:** green, and a walk-forward through 2018, from a model fit on 2016–2017, runs end to end and is timed. DEV stays dark (C11).

#### B12 · Gates, baselines and the first honest look (M)

- **Builds:**
  - G1–G6, and buy-and-hold with dividends;
  - the two walk-forward static families (C14);
  - the gate report and the `strategy.md` writer.

  Then the CP2 run: register r1 as trial #1, then run the DEV walk-forward, the March 2020 and 2022 windows, the 5 bp cost run and the early-2020 golden window.
- **Red tests first:**
  - Sharpe, max drawdown and Newey–West against statsmodels and hand-computed vectors;
  - the Deflated Sharpe against the worked example in Bailey and López de Prado (2014);
  - hit-rate counting on constructed trades, in which a scale-in belongs to its trade (C15);
  - G5 fails if any baseline wins on Sharpe or on drawdown;
  - the gates refuse to run without a ticket.
- **Covers:** spec §11, D10, D11, G1–G6, A2, C9, C14, C15.
- **Done when:** green → **CP2**.

### M5 — Paper trading

#### B13 · Alpaca broker adapter and order manager (L) · needs the test account

- **Builds:**
  - the Alpaca trading adapter with the paper lock (I5). In v1, `golive.verify()` always fails, because B19 is deferred. `submit` accepts only approved types;
  - the order manager:
    - marketable limit orders, collared on the IEX quote and re-quoted after 30 s;
    - three attempts for an entry; exits widen until they fill;
    - one GTC protective stop after every fill, cancelled before any sale;
    - partial fills, and idempotent `client_order_id`s;
  - the trade-update stream, with REST polling as backup;
  - reconciliation snapshots;
  - the integration suite on the test account.
- **Red tests first:**
  - the I5 constructor matrix: mode × URL × checklist state;
  - recorded-response tests for every order path, including 4xx and 5xx errors and timeouts: a retry reuses its `client_order_id` and never double-submits;
  - a stop is present after every fill, and an exit cancels the stop first;
  - a reconciliation mismatch halts trading;
  - on the test account: place, replace and cancel orders; the stop's lifecycle; and the kill test, which opens a position, kills it, and verifies flat, no open orders and HALTED within 60 s.
- **Covers:** spec §10 (order types, broker-side stops, reconciliation), I4, I5, AD3, go-live item 3.
- **Done when:** green, including the integration suite. Integration orders are fixed, tiny plumbing orders rather than strategy decisions, so C1 doesn't apply to them.

#### B14 · Live engine (L)

- **Builds:**
  - the live loop and scheduler at ARCHITECTURE §5's times, with the calendar cross-check and the clock-skew check;
  - ingest at τ, with retries until τ + 5 min;
  - write-ahead persistence (`ops.db` with `synchronous=FULL`);
  - the startup and recovery sequence from ARCHITECTURE §9.3;
  - the live minute monitor and a daily parity replay;
  - launchd agents for the engine and the 17:00 evening job (calibration and refit), with `caffeinate`;
  - the 02:00 backup and log rotation;
  - the CLI: `status`, `kill --direct`, `resume` and `lockbox seal`.
- **Red tests first:**
  - a crash injected at each persistence step never causes a double send or a late send;
  - missed bars replay without trading and count as unhealthy (C6);
  - a filter-rebuild mismatch halts trading;
  - a bar still missing at τ + 5 min takes the position flat;
  - a heartbeat gap, such as the Mac sleeping, triggers rule 6;
  - a session recorded against a fake Alpaca replays with identical hashes;
  - `resume` needs the typed confirmation and a clean reconciliation.
- **Covers:** spec §2 (stale data) and §14 (restart), I1 on the live path, C6, C7, C8, AD2 (`ops.db` durability and backups), ARCHITECTURE §5, §9 and §16.
- **Done when:** green. Nothing runs the strategy on live data before the seal (C1).

#### B15 · quant-control: Telegram, kill paths, alerts (M) · needs a Telegram bot

- **Builds:**
  - the quant-control service, with Telegram long polling: `/kill` and `/status` from your chat id only;
  - alert delivery from the outbox: retries, de-duplication, rate limiting, and a health warning once an alert has gone unsent for 5 minutes;
  - heartbeat watching, with C2's kill;
  - `quant control whoami`, and the launchd agent;
  - the weekly integration job, including the kill test.
- **Red tests first:**
  - messages from other chat ids are ignored and logged;
  - `/kill` flattens the test account with the engine stopped;
  - a lost heartbeat while in a position fires the kill switch; while flat, it sends an alert only;
  - alerts queue through a Telegram outage and drain afterwards;
  - no remote channel can resume or approve.
- **Covers:** spec §10 (kill switch) and §13 (Telegram), AD1, C2, ARCHITECTURE §8.3–8.4 and §12.
- **Done when:** green, and a `/kill` from your phone flattens the test account within 60 s → **CP3**: seal the lockbox, then start paper trading with r1 on the trading account.

### M6 — Operations

#### B16 · Dashboard and daily report (M)

- **Builds:**
  - the FastAPI app on 127.0.0.1, with a Host-header check, CSRF tokens and read-only database connections;
  - the pages listed in spec §13, with SSE on every bar plus P&L every minute, and vendored uPlot charts;
  - kill and approve buttons; approving requires typing the proposal id;
  - the 16:20 daily report, stored as an artifact, with a Telegram summary.
- **Red tests first:**
  - a foreign Host header is rejected, and so is a POST without a CSRF token;
  - a write through the dashboard's database handle fails;
  - an approval with the wrong id is refused;
  - the kill button's command gets executed;
  - the report's numbers match `ops.db` on a fixture day;
  - K4 holds.
- **Covers:** spec §13, D7 (approval UI), ARCHITECTURE §8.4 and §12.
- **Done when:** green, plus a browser smoke check with Playwright.

#### B17 · The nightly loop (L) · needs sudo once and your Anthropic key

- **Builds:**
  - the evening bundle, per ARCHITECTURE §11: a secret scan, a 200 KB cap, and no lockbox results or risk limits;
  - the L1 job, run as `_quantllm` from its own install in `/opt/quant-llm`:
    - one streaming request to `claude-opus-5-5`, with no tools and a JSON-schema structured output;
    - `effort: "high"` set explicitly, since Opus 5.5 defaults to `medium`;
    - the server-side refusal fallback on (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`), with the serving model recorded from `response.model`;
    - a refusal, timeout or invalid output produces only an error record;
  - the nightly harness:
    - strict re-validation: the path whitelist, the bounds, C10's parking, and at most 3 proposals;
    - the losses journal;
    - for each proposal: a `TrialTicket`, the DEV walk-forward, the gates and the champion test (C9);
    - the approval queue and a Telegram alert;
    - the 09:00 deadline, resuming the next night;
  - promotion at the next open, and rollback. `harness.auto_promote_in_paper` stays false (D12);
  - the sudo setup script for `_quantllm`, the spool folders and the LaunchDaemon.
- **Red tests first:**
  - hostile proposals are rejected: unknown keys, path traversal, out-of-bounds values, more than 3 proposals, and instructions hidden in data;
  - a move beyond ±20% is parked and spends no trial;
  - the bundle scan finds no secrets, lockbox results or limits;
  - the L1 environment holds only `ANTHROPIC_API_KEY`;
  - as `_quantllm`, reading the repo, `.env`, `config/` or `var/` fails;
  - a refusal yields no proposals, and a fallback answer records its model;
  - nothing is promoted without approval, and rollback restores the previous pointers.
- **Covers:** spec §1 (L1) and §12, D12, A2, AD4, AD6, C7, C9, C10, I3, I6.
- **Done when:** green, plus one real night end to end: bundle → Opus → proposals → nightly harness → gates → approval queue → **CP4**.

### M7 — Deployment

#### B18 · Deployment kit (M)

- **Builds:**
  - systemd units and timers on New York time, hardened as in ARCHITECTURE §15;
  - the `quant` and `quant-llm` users without login shells, and the `/etc/quant/*.env` layout;
  - notes for chrony, ufw and unattended upgrades;
  - `quant adopt-account`.

  The step-by-step runbook belongs to the ship phase.
- **Red tests first:**
  - a backup restored into a scratch folder passes `ledger verify` and replays to identical record hashes;
  - `adopt-account` refuses unless the account is flat and reconciled.
- **Covers:** spec §14, ARCHITECTURE §9.4 and §15.
- **Done when:** green → **CP5**. The units are linted with `systemd-analyze verify` on the VPS during the ship phase.

### Deferred

#### B19 · Go-live gate (M)

- **Trigger:** a candidate first clears G1–G6 on DEV, or you ask for it.
- **Builds:**
  - the `go_live/checklist.yaml` schema and validator;
  - the one-shot lockbox test, recorded in the ledger;
  - the April 2025 replay;
  - the parity report over C17's evidence window;
  - the "WHAT COULD BLOW UP THIS ACCOUNT?" review.

### Add-ons

#### B20 · Jev shadow (M)

Added in v1.2. Jev answers three questions about the next bar, and its answers are logged and scored next to the HMM's. It never affects a decision (SPEC §12a, ARCHITECTURE §11.1).

- **Needs you:** a TypeSafe AI API key, and one more run of the sudo setup script for the shadow spool folders (§0.3).
- **Depends on:** B14 (decision records and the live loop), B16 (the daily report) and B17 (`_quantllm`, the spool, outbox and ingest pattern).
- **Spike first:** `spikes/jev_probe.py` answers ARCHITECTURE §18's Jev questions and records them in VERIFY.md. It sends synthetic bundles only, never DEV or lockbox bars.
- **Builds:**
  - the shadow bundle schema in `core`, and the engine's spool write after the DecisionRecord is persisted, only in paper trading inside FORWARD;
  - `quant.shadow`:
    - the TypeSafe client, on plain `httpx` if TypeSafe documents an HTTP API (AD5);
    - the `quant-shadow` job, run as `_quantllm` with only `TYPESAFE_API_KEY`, which goes through `register_secret`;
    - the strict answer schema and the ingest into `var/shadow.db`, with `late` and `missing`;
  - scoring in `eval/`: Brier score and log-loss for `next_up` and `next_stress`, the HMM's forecasts of the same events, `next_label` accuracy against the A4 label, and the Diebold–Mariano test;
  - one line in the daily report, with the late and missing counts;
  - import rules: contract K7, `quant.shadow` in K1–K4, a contract that keeps `research` and the harness from importing it, and the `shadow.db` scan;
  - quant-check's trace learns lettered sections, so tests can name `SPEC§12a`;
  - the launchd units, and the shadow spool folders in the sudo setup script.
- **Red tests first:**
  - I7 and K7 catch planted violations in a copy of the repo: `decide` importing `quant.shadow`, `shadow` importing `risk`, and `research` naming `shadow.db`;
  - the answer schema (hypothesis) rejects a `Noul` outside [0, 1], an unknown `Choice`, extra keys and an oversized outbox file;
  - a late answer is never scored, and an outage or an invalid answer is `missing` and never a health event;
  - golden Brier, log-loss and Diebold–Mariano values on fixed fixtures;
  - the bundle holds no secret, date or ticker, and backtests and replays write no bundle;
  - the shadow job's environment holds only `TYPESAFE_API_KEY`;
  - the fast tier stays offline, with a fake Jev client fed from recorded fixtures. A `jev` marker runs the real API by hand.
- **No T-NL1 layer:** B20 stays off the decision path, and I7 proves it.
- **Covers:** spec §1 (L1s), §12a and §16; I6, I7; AD4, AD5, AD7; K7.
- **Done when:** green, plus one real paper session end to end: bundle → Jev → outbox → ingest → evening score → daily report line. The promotion test (SPEC §12a) comes much later, after 500 scored bars, and passing it only allows a proposal.

## 5. Seed release r1 and the feature menu

These values are fixed before any data is looked at (P7). Anything not listed keeps its value from the spec.

| Part | r1 value | Bounds the nightly loop may move within |
|---|---|---|
| Features | F1–F5 as in spec §3: rvol over 35 bars; trend EMA 35 / 175; slot history of 60 sessions | rvol 14–70 bars; trend fast 14–70 and slow 70–350, with fast < slow; slot history 20–120 sessions; F1 and F2 always on (P8) |
| HMM | Selected on FIT by spec §4 over K ∈ {2, 3, 4, 5} × {diag, full}; 50 restarts; `base_seed` 20160104 | Changing K or the covariance type is a gated HMM-config change (spec §5) |
| CALM_UP | `trend_following`. Entry: EMA35 > EMA175 and the close is above EMA35. Exit: EMA35 < EMA175. Stop at 4 × ATR(35); no take-profit; `max_size` 1.0. Invalidated if P(CALM_UP) < 0.30. | EMAs as for features; stop 2–8 × ATR; take-profit none, or 2–10 × ATR; `max_size` from 0 to the cap; invalidation thresholds within the menu |
| CHOP | `mean_reversion`, with z = (close − 35-bar mean) / 35-bar s.d. Entry: z ≤ −1.5. Exit: z ≥ 0. Stop at 3 × ATR(35); no take-profit; `max_size` 0.5. Invalidated after 21 bars in a trade, or if P(CHOP) < 0.30. | N 14–70; entry_z 0.5–3.0; exit_z −1.0 to 1.0 and below entry_z; stop 2–8 × ATR; bars in trade 7–70 |
| STRESS | `flat`: stands aside (D8) | Either primitive, at `max_size` ≤ 0.25 |
| CRASH | `flat`; the cap is 0 in code | Fixed |
| Static baselines (C14) | Trend-following: fast ∈ {14, 21, 35, 50, 70} × slow ∈ {105, 175, 245, 350}, stop at 4 × ATR(35). Mean reversion: N ∈ {14, 21, 35, 70} × entry_z ∈ {1.0, 1.5, 2.0, 2.5}, exit_z 0, stop at 3 × ATR(35). Both hold 1.0× when in position. | Fixed, because they are the yardstick |
| Switching, sizing, risk, costs | The values in spec §8–§11, kept in `engine.yaml`, `risk.yaml` and `costs.yaml` | Never: these are L3 configuration |

ATR(35) is the mean true range over the last 35 bars. The opening bar's true range includes the overnight gap.

## 6. Risks to the plan

| Risk | Effect | Mitigation |
|---|---|---|
| HMM fits run slower than estimated | Walk-forwards take hours, and the nightly harness misses its 09:00 deadline | Timed at B4 and B11. Fits are cached by their inputs and run in a process pool on all 8 cores; playbook-only proposals reuse the champion's fits. |
| On the free plan, SIP minute bars arrive later than close + 15 min | The 16-minute decision misses its bar, so rule 6 fires every bar | The B6 probe checks this before anything depends on it. The delay is one config value, shared by the backtest and live trading. |
| Alpaca won't open a second paper account | Integration tests share the trading account | AD3's fallback: marked test windows that the track record excludes |
| DEV results leak into development choices | Gate results look better than they are | C11: DEV stays dark until trial #1, r1 is fixed in §5, and every differing DEV run counts as a trial |
| The fast tier slows down | Slower commits, and a temptation to skip it | Two tiers (P4). A step isn't done while the fast tier takes more than 2 minutes. |
| A session is cleared mid-step | The thread is lost | Each step has a branch with a red/green trail, and CLAUDE.md, BUILD_LOG.md and the tags say where things stand |
| The Mac sleeps or loses power during market hours | Decisions are missed | Rule 6 goes flat (C6); `caffeinate` runs in the engine's launchd agent; go-live evidence counts only on the VPS (C17) |
| r1 fails the gates, which is likely | No live candidate | Spec §0 expects this. CP2 is the cheapest place to change direction. Otherwise paper trading runs r1, and the nightly loop searches through the gates. |

## 7. After the build

- **Review:** a full review against every ID in SPEC v1.2, ARCHITECTURE v1.0 and this plan, starting from the trace matrix. A security review covers key handling, L1's isolation and the dashboard.
- **Ship:** the VPS runbook, the move with `quant adopt-account`, and the start of the evidence clock (C17).
- **Then:** paper trading and the nightly loop run on their own. Going live needs B19, a candidate that clears every gate, and your signature.
