# ARCHITECTURE — SPY Regime Trader

**Status:** APPROVED v1.7 (Gate 2 of 3) · 2026-10-05. v1.0 was approved as written: decisions AD1–AD6 confirmed and clarifications C1–C10 accepted, and C1–C10 are now part of SPEC.md. v1.1 aligns this document with the approved plan: clarifications C11 and C14, the test tiers (P4) and spikes (P11). v1.2 records how B0 built the gate (§13), v1.3 how B2 resolves the view's knowledge time (§6.2), and v1.4 how B2 stamps that time, flags moves against stored minutes, and lets calibration read forward data (§6.1–§6.3). v1.5 records how B10's tickets bind the walk-forward, that the segment starts are fixed in code, and what the ledger's chain can't catch (§4.4, §6.3, §10). v1.7 records how B9 built the risk engine: the checks' precondition, how approvals hold exposure until they are spent, how intents and submissions are counted, how halts are signalled, persisted and resumed, that a resume re-bases the peak, and that discretionary exits keep the ±10 bp collar while only kill-switch exits widen (§4.4, §8.1–§8.3; your calls of 2026-10-13). It also records that the broker port lives in `risk/`, and how B9b built the kill executor, reconciliation and the minute monitor (§3, §8.3). Still v1.7, it records your calls of 2026-10-14: the kill settings approved as proposed, a fourth approved type `RiskApprovedStop` that a kill unable to finish leaves under the bare shares, risk-reducing approvals issued even when the risk state can't be saved, and a working exit counting as the stop for at most 5 minutes (§3, §4.4, §8.1, §8.3, §16). Each change is tagged where it lands.
**Builds on:** [SPEC.md](SPEC.md) v1.2, which is v1.0 plus this document's C1–C10 and the plan's C11–C17
**Process:** brainstorm ✅ → spec ✅ → architecture ✅ → plan ✅ → **test-first build** ▶ → review → ship
The [plan](PLAN.md) was approved 2026-10-05 and the build has started. Progress is in [BUILD_LOG.md](BUILD_LOG.md).

---

## 0. What you approved

This document turns the approved spec into a buildable design. It covers:

- which processes run, under which Unix users and with which secrets;
- how the code splits into packages, and which imports are forbidden;
- how one decision function serves live trading, the backtest and the go-live replay;
- where every piece of state lives;
- which test enforces each spec invariant.

Four ideas carry the design:

1. **One pure decision function.** `decide(state, view, release) → (state′, record, order intents)` covers everything from features to sizing. Live trading, the backtest and the replay differ only in what they plug into it: the clock, the bars and the broker.
2. **Point-in-time data.** Every read of market data goes through a view. The view returns only rows that were available at decision time τ, in the version that was known at τ.
3. **Typed choke points.** Only the risk engine can create an order the broker adapter accepts. Only the trial ledger can issue the ticket a gate run needs. A live broker client can be constructed only after the go-live checklist validates.
4. **Isolation by Unix user, not by convention.** L1 runs as a user that can't read broker keys, risk limits or any database. It receives a data bundle and returns JSON, and deterministic code re-checks that JSON before anything uses it.

### 0.1 Architecture decisions (confirmed 2026-10-05)

| ID | Decision | Proposed default | Why |
|---|---|---|---|
| AD1 | Services | Three long-running services (engine, control, dashboard) and five scheduled jobs (§2). New compared with spec §14: **quant-control**, a small process that handles Telegram and provides a kill path that still works when the engine is down. | A kill switch that lives only inside the engine fails exactly when the engine is the problem. |
| AD2 | Storage | SQLite only, in four files, plus an immutable artifact store on disk. No database server, message queue or Docker. | The system makes about seven decisions a day. SQLite in WAL mode is durable and easy to inspect, and it backs up with one call. |
| AD3 | Integration-test account | A **second Alpaca paper account**, used only by integration tests. That includes the weekly kill-switch test for go-live item 3. | The kill test opens positions and force-closes them. On the trading account it would pollute the paper track record. If Alpaca won't give you a second account, tests run on the main one inside a marked window that the track record excludes. |
| AD4 | L1 on this Mac | A hidden role user, `_quantllm`, created once with `sudo`, runs L1 under launchd. It gets its own group, not `staff`. Your home folder is mode 750 with group `staff`, so this user can't read the repo, `.env` or `risk.yaml`. (Checked: macOS role accounts aren't members of `staff`.) | The spec's separate-user rule holds from day one. The alternative is no nightly loop until the VPS. |
| AD5 | Stack | Python 3.12 via uv (not installed yet; uv fetches it) · alpaca-py · hmmlearn 0.3.3 · numpy, scipy · exchange_calendars · pydantic 2 · statsmodels (Newey–West) · scikit-learn (isotonic regression only) · FastAPI with SSE, HTMX and uPlot (vendored, no Node build) · typer · structlog · httpx · anthropic · pytest, hypothesis · import-linter · ruff · mypy · gitleaks | Mainstream, open source and pinned by `uv.lock`. Nothing closed-source runs near the keys. |
| AD6 | The L1 request | One streaming request a night to `claude-opus-5-5`, with effort `high`, JSON-schema structured output and no tools. Anthropic's server-side refusal fallback is **on**: if Opus declines, another model answers, and the proposal records which model wrote it. Cost is about $0.30–0.75 a night. | Every proposal faces the same gates, whichever model wrote it. With the fallback off, a night when Opus declines produces no proposals. |

### 0.2 Spec clarifications (accepted 2026-10-05)

The spec left these questions open, and the code has to answer each one. They are now part of SPEC.md v1.1.

| ID | Open question | Proposed answer |
|---|---|---|
| C1 | The lockbox runs "2025-01 → present", but the engine can't trade without reading 2025–26 bars. Features look back 60 sessions, the filter runs from 2016 and refits use an expanding window. | The lockbox closes the day before the first paper decision. That date goes into config and never moves. Bars inside the lockbox may feed features, the filter and refits. **No strategy P&L, trade or calibration result inside the lockbox is computed before the go-live lockbox test.** Data after the close is "forward": paper trades it and the nightly loop reviews it (§6.3). |
| C2 | Which "health failure while a position is open" fires the kill switch? | A failure of the engine itself: a lost heartbeat, a crash loop, or a rule-6 flatten still incomplete after 10 minutes. A single late bar is rule 6's job (go flat, wait for two healthy bars) and doesn't halt trading. |
| C3 | What counts toward the limit of 20 orders a day? | Order intents: one per rebalance, however many re-price attempts it takes. A separate runaway guard halts at 60 raw broker submissions in a day. |
| C4 | What state does a run start from? | Flat, with no active playbook. Every backtest segment, the paper start and the lockbox test start cold, so the first takeover needs the full 3 bars. |
| C5 | What runs after a refit fails a drift check? | The failed refit is stored but not activated. The current model keeps running with entries frozen until the next refit passes or you acknowledge the drift. |
| C6 | What does a missed decision count as, when the engine was down at τ? | Stale data. On restart, the missed bars are replayed without trading and count as unhealthy. Rule 6 then applies: go flat, then wait for two healthy bars. |
| C7 | When do refits, promotions and rollbacks take effect? | At the next session open, never mid-session. An urgent rollback is the kill switch, then the rollback, then a resume. |
| C8 | Daily loss and max drawdown can be breached between bars. Which price watches them? | A monitor runs every minute on the freshest price: the IEX real-time trade when live, the SIP 1-minute close in the backtest. IEX prices feed only the order collar and this monitor, never a feature. |
| C9 | What does "beats the current champion" mean? | A higher out-of-sample Sharpe ratio after costs on the development period. A champion that fails any gate is beaten by any candidate that passes all six. |
| C10 | What happens to a proposal that moves a parameter by more than ±20%? | It is parked before testing, so no trial is spent, until you allow it with one command. |

---

## 1. The system at a glance

```
            ┌───────────────────────────── user: quant ──────────────────────────────┐
 Alpaca ◀──▶│ quant-engine     every bar: ingest → decide → risk → orders            │
 data and   │                  every minute: P&L, daily loss, drawdown, stops        │
 broker ◀──▶│ quant-control    Telegram /kill /status · kill executor · alerts       │──▶ Telegram
            │ quant-dashboard  127.0.0.1 · read-only except kill and approve         │
            │ quant-evening    report · calibration · refit · L1 bundle              │
            │ quant-harness    validate · ledger · walk-forward · gates · queue      │
            └────────────┬─────────────────────────────────────────▲─────────────────┘
                  bundle │ data only, no secrets         proposals │ JSON only
            ┌────────────▼────────────── user: quant-llm ──────────┴─────────────────┐
            │ quant-nightly (L1)  reads its inbox, writes its outbox, nothing else   │──▶ Anthropic API
            └────────────────────────────────────────────────────────────────────────┘
```

Processes communicate only through SQLite tables and the two L1 spool directories. The only listening socket is the dashboard, on 127.0.0.1.

## 2. Processes, users and secrets

| Unit | Runs | User | Secrets it receives | Writes | Network |
|---|---|---|---|---|---|
| `quant-engine` | always; restarted on exit; 90 s watchdog | quant | trading-account Alpaca keys | market.db, ops.db, alerts and approval requests in control.db | Alpaca data and broker |
| `quant-control` | always; restarted on exit | quant | trading-account Alpaca keys (used only to cancel and close), Telegram token | control.db | Alpaca broker, Telegram |
| `quant-dashboard` | always; restarted on exit | quant | none | commands and approvals in control.db | 127.0.0.1 only |
| `quant-evening` | 16:20 and 17:00 ET on trading days | quant | none | ops.db, artifacts, L1 inbox, alerts in control.db | none |
| `quant-nightly` | 18:00 ET on trading days | quant-llm | Anthropic key only | L1 outbox | Anthropic API |
| `quant-harness` | from 18:30 ET until done; stops at 09:00 | quant | none | research.db, artifacts, `journal/losses.md`, approval queue and alerts in control.db | none |
| `quant-backup` | 02:00 ET daily | quant | none | backups | optional copy off the box |
| `quant-itest` | Wednesdays 12:00 ET, and on demand | quant | test-account Alpaca keys | test results in ops.db | Alpaca broker (test account) |

On the Mac, "quant" is your own login and "quant-llm" is `_quantllm`. The engine fetches all market data, so the evening jobs and the harness need no keys at all.

| What | Mac | VPS | Readable by |
|---|---|---|---|
| Code | this repo; L1 runs from its own install in `/opt/quant-llm` | `/opt/quant/app` (git checkout) and `/opt/quant/venv` | both users, except `config/` |
| Engine secrets | `.env` in the repo, gitignored, mode 600 | `/etc/quant/engine.env`, root, 600, injected by systemd | engine and control only |
| L1 secret | `/opt/quant-llm/.env.llm`, 600 | `/etc/quant/llm.env`, root, 600 | L1 only |
| Test-account secrets | `.env.test`, 600 | `/etc/quant/test.env`, root, 600 | itest only |
| `config/` (`risk.yaml`, `gates.yaml`, …) | in the repo, committed | in the checkout, 750 root:quant | quant only, never L1 |
| Databases, artifacts, journal | `var/` in the repo, gitignored | `/var/lib/quant`, 700 | quant only |
| L1 inbox and outbox | `/Users/Shared/quant-l1/` | `/var/lib/quant-l1/` | inbox: quant writes, L1 reads · outbox: L1 writes, quant reads |

## 3. Code layout and structural rules

```
quant/
  pyproject.toml · uv.lock · .python-version (3.12) · .importlinter · .gitleaks.toml · .env.example
  config/     risk.yaml · gates.yaml · engine.yaml · costs.yaml       (committed; L3 only)
  strategy.md · playbooks/{CALM_UP,CHOP,STRESS,CRASH}.md             (seed release r1, from spec D3/D8)
  go_live/    checklist.yaml
  deploy/     systemd/ · launchd/ · scripts/
  docs/       SPEC.md · ARCHITECTURE.md · PLAN.md · BUILD_LOG.md · VERIFY.md
  spikes/     throwaway vendor checks; never imported by src/ (P11)
  .githooks/  the commit hook, which runs quant-check
  var/        *.db · artifacts/ · journal/losses.md · backups/       (gitignored runtime data)
  src/quant/
    core/       types, ids, errors, canonical JSON, Clock, trading calendar, schemas shared with L1
    data/       Alpaca market data, bar builder, point-in-time bar store, DataPolicy
    features/   streaming transforms, the feature menu, scaler
    hmm/        train (the only hmmlearn import), select, filter (ours), labels, matching, drift
    decide/     health, switching, playbook schema and primitives, sizing, edge estimates for Kelly
    risk/       RiskEngine, approved-order types, limits, kill executor, reconciliation, minute monitor,
                broker port (v1.7: K1 keeps risk from importing execution)
    execution/  Alpaca adapter with paper lock (implements risk's broker port), order manager, simulated broker
    engine/     live loop, scheduler, startup and recovery, state persistence
    store/      SQLite schemas and migrations, artifact store, pointers
    jobs/       evening pipeline: report, calibration, refit, L1 bundle
    eval/       smoother, calibration metrics, wrong-state calls   (the only place smoothing runs)
    backtest/   simulated clock, historical feed, costs, dividends, walk-forward, reports
    research/   gates G1–G6, baselines, deflated Sharpe, trial ledger
    harness/    proposal intake and validation, trial runs, approval queue, promotion
    llm/        L1: bundle reader, prompt, Anthropic client, output schema
    control/    Telegram, kill service, alert sender
    dashboard/  FastAPI app, SSE, templates, vendored JS
    golive/     checklist validation, parity replay, lockbox test
    cli/        the `quant` command
  src/quant_check/  the build gate (`quant-check`); a dev tool outside the `quant` package
  tests/      unit · property · invariants · golden · sim · integration
```

Each rule below is a test in the build gate. import-linter checks K1–K4, and small AST scans check K5 and K6.

| ID | Rule | Enforces |
|---|---|---|
| K1 | `risk` imports only `core` and `store` | I3: nothing from `llm`, the harness or playbook prose can reach the risk engine |
| K2 | `llm` imports only `core` | L1's code can't reach broker, risk, data or engine code |
| K3 | The decision path (`data`, `features`, `hmm.filter`, `hmm.labels`, `decide`, `risk`, `execution`, `engine`) never imports `hmmlearn` or `eval` | I2, together with the banned-method scan in §13 |
| K4 | Only `engine`, `control` and `cli` import the Alpaca broker adapter | Research, the dashboard and L1 can't trade |
| K5 | `RiskApprovedOrder`, `RiskApprovedCancel`, `RiskApprovedFlatten` and `RiskApprovedStop` are constructed only in `risk.engine`, and `TrialTicket` only in `research.ledger` | I4 and A2 (scan plus a runtime token, §4.4) |
| K6 | `datetime.now` and `time.time` appear only in `core.clock` | I1: "now" always comes from the Clock port |

## 4. The decision core

### 4.1 One function, three drivers

`decide(state, view, release) → (state′, DecisionRecord, [OrderIntent])` is pure: no I/O, no clock, no randomness. Everything around it is an adapter behind a port.

| Port | Live | Backtest | Replay (go-live parity) |
|---|---|---|---|
| Clock | wall clock, UTC | simulated | simulated |
| Bars | Alpaca SIP into the bar store | stored history, `available_at = t_close + 15 min` | bars exactly as the live engine saw them |
| Quotes (collar, minute monitor) | Alpaca IEX latest trade and quote | SIP 1-minute open and close | recorded |
| Broker | Alpaca paper; live only behind the checklist | SimBroker | recorded fills and rejections |
| Store | SQLite files | in-memory SQLite | in-memory SQLite |
| Alerts | control.db outbox | discarded | discarded |

### 4.2 Every bar, in order

| Step | Module | Pure | What happens |
|---|---|---|---|
| 1 Ingest | data | no | Fetch the bar's SIP minute bars, build the hourly bar, store it with `known_at` stamped at commit (§6.2) |
| 2 View | data | yes | `PointInTimeView(τ, as_of)`: rows with `available_at < τ`, each in the latest version known before `as_of` (§6.2) |
| 3 Features | features | yes | Streaming update: one new float64 row |
| 4 Filter | hmm.filter | yes | Log-space forward step, with every output listed in spec §6 |
| 5 Labels | hmm.labels | yes | State probabilities summed per label, entropy, expected remaining duration |
| 6 Health | decide | yes | Rule 6 inputs: data status, NaN features, model provenance, drift freeze |
| 7 Switching | decide | yes | Rules 1–5 plus A1 |
| 8 Signal | decide | yes | The active playbook's primitive: entry, exit, stop level and invalidation |
| 9 Size | decide | yes | Spec §9 formula, calibration floor and deadband |
| 10 Intents | decide | yes | Target minus current position, as at most one intent of up to 0.5× equity; the overnight cap applies on the last decision that can fill |
| 11 Risk | risk | yes, given a fresh broker snapshot | `RiskApprovedOrder`, `Veto` or `PendingApproval` |
| 12 Persist | engine | no | DecisionRecord, new state and intents written and synced to disk **before** anything is sent |
| 13 Execute | execution | no | Collared limit order, protective stop, re-price policy |
| 14 Publish | engine | no | Dashboard row and alerts |

Steps 2–11 are the pure core. Features, the filter and every accumulator are streaming, so each bar updates a small fixed-size state. Fitting a model on a window runs the same streaming feature code over that window. That leaves exactly one implementation of every feature, and live trading and the backtest can't drift apart.

### 4.3 Engine state and decision records

- **EngineState** holds the filter's log-probabilities; the switching state (active label, challenger and its count, cooldown, health streak, invalidation blocks); the trade state (entry, stop, bars in trade, trade drawdown); the risk state (start-of-day and peak equity, daily-loss lock, intents today, drift freeze); and the edge accumulators for Kelly.
- State is serialized as canonical JSON (sorted keys, shortest round-trip float64 text, NaN forbidden) and hashed with SHA-256.
- **A DecisionRecord** holds τ and the view's `as_of` (§6.2); the ids and hashes of the bars used; the release, model and calibration ids; the code commit; the features; filtered and next-bar probabilities, entropy and predictive log-likelihood; the rule trace with rule ids and reasons; the target; the intents; the risk verdicts; and the state hashes before and after. Replays compare records hash for hash.

### 4.4 Typed choke points

- **`RiskApprovedOrder`** is a frozen type whose constructor demands a private token held by `risk.engine`. It carries the intent, an approval id and an expiry at the end of the bar. The Alpaca adapter's `submit` accepts nothing else, refuses expired approvals, and spends each approval id once (persisted, so a restart can't reuse one). Cancels and kill-switch closes use `RiskApprovedCancel` and `RiskApprovedFlatten`, which the risk engine always issues because they only reduce risk. Protective stops use `RiskApprovedStop` (v1.7, your call of 2026-10-14): the engine issues one, halted or not, only for a stop-market SELL below the current bid over shares no other SELL covers, so a stop never makes a short, and it expires with the broker snapshot behind it. The adapter has no other method that changes anything at the broker. (I4) The adapter spends an approval through `RiskEngine.spend` right before it sends. That refuses copies, forgeries, changed approvals, and approvals that have expired, been spent, been released or been revoked. Until it is spent, an approval holds its exposure or its shares (B9, §8.1).
- **`TrialTicket`** is issued only by the trial ledger, after it has appended the trial. The walk-forward refuses to score the development period without one. (A2) It also refuses a ticket whose `config_hash` or `code_hash` differs from the release and code it runs, and it runs yardsticks (baselines and cost-sensitivity runs) only inside the measured run's own walk-forward, under that run's ticket (B10, built in B11).
- **Live broker client:** the constructor requires `TRADING_MODE=live` *and* `golive.verify()` to pass. v1 ships with the checklist unsigned, so the constructor always refuses. (I5)

## 5. Time and schedule

- UTC everywhere inside the code; New York time only at the edges (schedules and reports).
- The trading calendar is exchange_calendars' XNYS (holidays and early closes), cross-checked daily against Alpaca's calendar. A disagreement sends an alert, and the engine uses the more conservative answer.
- `core.clock` is the only source of "now" (K6). Skew against Alpaca's clock is checked at start and every hour: over 2 s sends a warning, and over 30 s blocks entries.
- Every scheduled job checks the trading calendar and a `job_runs` table before doing anything. A double fire or a missed fire is therefore harmless, and launchd's local-time schedules only need to fire often enough.

| ET | What | Who |
|---|---|---|
| 09:25 | Preflight: clock, reconciliation, health; queue any exit carried from the 16:16 decision | engine |
| 09:30:05 | Submit the queued next-open exit (risk-reducing only, re-checked by the risk engine) | engine |
| 10:46 · 11:46 · 12:46 · 13:46 · 14:46 | Decisions for bars 1–5 (τ = bar close + 16 min) | engine |
| 15:46 | Decision for bar 6, the last that can fill today, so the 0.5× overnight cap (A3) applies | engine |
| 16:16 | Decision for the 15:30–16:00 bar; it can only reduce exposure and executes at the next open | engine |
| every minute, 09:30–16:00 | Equity mark, daily loss, drawdown, stop presence, reconciliation, P&L for the dashboard (C8) | engine |
| 16:20 | Daily report | evening |
| 16:45 | Settle: re-fetch the day's minute bars, record revisions; fetch corporate actions | engine |
| 17:00 | Calibration update; refit if due, with drift checks; L1 bundle | evening |
| 18:00 | L1 run | nightly |
| 18:30 until done, deadline 09:00 | Validation, ledger, walk-forwards, gates, approval queue | harness |
| 02:00 | Backups and log rotation | backup |
| Wednesday 12:00 | Integration tests on the test account, including the kill switch | itest |

On half days the calendar shifts everything earlier. With a 13:00 close, decisions run at 10:46, 11:46 and 12:46 (the last that can fill), plus 13:16, which can only reduce and executes at the next open.

## 6. Data

### 6.1 Bars

- **Source:** Alpaca SIP 1-minute bars, raw (unadjusted), regular hours only. A one-time backfill from 2016-01-01 covers about 1 million minute bars; it is resumable and respects rate limits.
- **Bar builder:** hourly bars anchored at 09:30, as in spec §2. Slots are numbered by start time, and the final 30-minute bar of any session, full or half day, maps to a single CLOSE slot. This matters for the slot-normalized features F1, F3 and F4.
- **Completeness:** a bar is complete when its last minute is present. The engine retries from τ until τ + 5 minutes, then applies rule 6.
- **Validation:** OHLC ordering, no duplicate minutes, session bounds. A one-minute move over 5% is used but flagged with an alert. Each ingest recomputes the flags of the sessions it touches against the stored minutes, so a jump across two ingests is caught, and a revision that clears a flag adds a cleared version rather than deleting it (B2). A split in the corporate-actions feed halts trading, since raw prices would break. SPY has never split.

### 6.2 Point-in-time store

- Every minute and hourly bar row carries `available_at` (modeled as close + 15 minutes) and `known_at` (when we actually had it). Revisions become new versions and never overwrite.
- `known_at` is when the row was committed: the store stamps it from the Clock while it holds `market.db`'s write lock, not when the data was fetched (B2).
- Bulk-loaded history is stamped `known_at = available_at`, and a backfilled dividend or split at the end of its declaration day in New York. History only adds rows, and only rows known after the store's last stamp: once a live view has read the store, a late row is known now (B2).
- `PointInTimeView(τ, as_of)` returns, for each bar with `available_at < τ`, the latest version with `known_at < as_of`. In the backtest `as_of` is τ. Live, the engine ingests the bar at τ (§16), so it reads the view just after, and `as_of` is that moment, stamped like a write under the write lock; the DecisionRecord keeps it. Stamps never go back, even if the clock does, so a row is known before a live view's `as_of` exactly when it was committed before the view (B2). `as_of` is never earlier than τ. Live decisions see what they saw. Backtests see the single historical version. Replays see exactly what live trading saw (B2).
- Caches (features, model fits) are keyed by hashes of the bytes actually read through the view. A leak of future data would change the key rather than hide behind a cache hit.

### 6.3 Data segments (C1)

| Segment | Span | Engine (operates) | Harness, research, gates | L1 bundle | Go-live test |
|---|---|---|---|---|---|
| FIT | 2016-01 → 2018-12 | ✓ | ✓ | results only | ✓ |
| DEV | 2019-01 → 2024-12 | ✓ | ✓ (the only out-of-sample period gates may score) | results only | ✓ |
| LOCKBOX | 2025-01 → the day before paper starts | market data only: features, filter, refits | ✗ | ✗ | ✓, once per candidate, recorded in the ledger |
| FORWARD | from paper start | ✓ | parity and reports only | ✓ | ✓ |

- `DataPolicy` makes every query name its purpose. Scoring anything inside LOCKBOX without a lockbox ticket raises an error and leaves a ledger entry.
- The purposes (B2): **operate** (features, the filter, refits) reads anything; **score** (strategy P&L, trades, gate results) reads FIT, and DEV with a trial ticket; **calibrate** (calibration and Kelly edge estimates) reads FIT, DEV with a trial ticket, and FORWARD; **report** (parity and reports) reads anything but the lockbox. Only operate may read the lockbox.
- Kelly edge estimates and calibration skip the lockbox span: they use DEV walk-forward history, then FORWARD. On DEV that history is a trial's walk-forward, and Kelly's shadow returns are strategy P&L, so calibrate needs the trial's ticket there (C11).
- DEV stays dark during the build as well: nothing computes strategy P&L on DEV before trial #1, which is the seed release r1, run at checkpoint CP2 (C11).
- `quant lockbox seal` writes the closing date into `gates.yaml` when paper trading starts. It is committed like `risk.yaml`, and the engine won't trade paper until it is. The starts of FIT, DEV and the lockbox are constants in code; the seal is the only segment boundary that is configured (B10).
- At go-live, item 7 scores 2025-01 → today as the spec says. The report shows the sealed part and the forward part separately, because only the sealed part was truly hidden.

## 7. Models

### 7.1 Training and selection

- `hmm.train` is the only module that imports hmmlearn, and it calls only `fit`. All densities and probabilities used afterwards come from our own numpy/scipy code reading the stored parameters.
- The 50 restarts run in parallel processes, each single-threaded, with seeds `base_seed + i`. One restart starts from the previous model's parameters. The winner is the highest training log-likelihood, with ties broken by seed index.
- Selection over K × covariance (spec §4) runs only for the first model and for a gated HMM-config change.

### 7.2 Model artifacts

Each model is an immutable directory, `artifacts/models/<sha256>/`:

- `params.npz`: start probabilities, transition matrix, means, full covariances, scaler;
- `meta.json`: fit window, data hash, feature-config hash, HMM config, seeds and per-restart log-likelihoods, the label table with v and d, sanity checks, the validation block's predictive log-likelihood distribution, the code commit, the lockfile hash, and library and BLAS versions.

No pickles: they are unsafe to load and break across library versions.

### 7.3 Monthly refit (evening job)

1. Fit on the expanding window, using bars available at 17:00.
2. A sanity reject keeps the current model and sends an alert.
3. Label matching uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on symmetric KL divergence.
4. A failed drift check stores the new model without activating it and freezes entries (C5).
5. On a pass, a new calibration artifact is built and the pointer moves, effective at the next open (C7).

The backtest calls this same job at the same simulated times.

### 7.4 Calibration

- `eval` builds a calibration artifact at each refit, and nightly for the report. It contains per-label isotonic knots, pass or fail, the source (backtest or fills) and an as-of time.
- Sizing consumes the artifact as plain data and never imports `eval` (K3).
- Provenance is checked like T-NL2: an artifact's as-of time precedes every decision that uses it.

### 7.5 Live drift

The engine tracks the 35-bar rolling mean of the predictive log-likelihood. When it falls below the 1st percentile stored in the model's `meta.json`, entries freeze and you get an alert.

## 8. Orders, risk and the kill switch

### 8.1 Risk engine

`RiskEngine.check(intent, context)` runs these checks, cheapest first:

1. **HALTED:** only risk-reducing intents pass.
2. **Reconciliation:** a broker snapshot no older than 5 s must match the internal position and open orders. A mismatch halts trading and sends an alert, and is never auto-corrected.
3. **Limits:** long or flat only; gross exposure ≤ 1.0×; the label cap; a single order ≤ 0.5× equity; 0.5× overnight on the last decision that can fill.
4. **Locks:** the daily-loss lock, a drift freeze and rule 6 all veto entries.
5. **Rate:** more than 20 intents in a day halts trading (C3).
6. **Price:** the collar must sit within ±10 bp of a quote no older than 5 s, and every entry must carry a stop.
7. **Approval (D7):** when live, a risk-increasing order above $25,000 notional returns `PendingApproval`, which expires at the end of the bar. In paper it is approved, flagged and alerted.

Risk-reducing intents skip the caps in 3, the locks in 4 and the approval in 7. They are clamped so they can never flip the position short.

Limits load from `config/risk.yaml` at start and are frozen. The engine refuses to start if that file differs from the committed version, and any change on disk sends an alert.

**How B9 built it (v1.7).** These rules come from building and reviewing the engine; none changes an approved limit.

- **Before check 1:** the engine looks τ up in the decision schedule itself. An intent must be a scheduled decision, checked inside its execution window, with its execution session started. The window runs from the decision's execution time to the end of that bar, so the closing bar's intent executes in the next session's first bar. Anything else is vetoed and never sent late.
- **Prices:** the caller proposes the limit, check 6 checks it, and every exposure is valued at it. Working BUY limits count as held.
- **SELLs:** in a long-or-flat system a SELL is exactly a risk-reducing intent. The clamp subtracts working SELL limits and SELLs already approved, but not the stop, because execution cancels the stop before it sells (§8.2). The ±10 bp collar binds SELLs too, as spec §10 says for discretionary exits: they re-quote around each fresh bid and stay within 10 bp of it (§8.2). Only the kill switch's exits widen, through `RiskApprovedFlatten`, which no collar checks (§8.3). You confirmed this on 2026-10-13.
- **Counting intents (C3):** every intent the engine checks after a session has started counts once for its decision and leg, whether it is vetoed or not. Re-quotes don't count again. The 21st intent halts trading, whichever check decides it. A BUY is vetoed; a SELL passes.
- **Counting submissions (C3):** a submission is counted when the adapter spends the approval. C3 halts *at* 60, so the 60th submission goes out, BUY or SELL, and trading halts. After a resume, a BUY at or past 60 is refused with `SubmissionLimitError` and halts trading again. A SELL is never refused.
- **Outstanding approvals:** an approval is risk until it is sent. Unspent BUYs count in the gross, label and overnight caps, and unspent SELLs reserve their shares. A spent order keeps counting until a broker snapshot taken after the spend can show it.
- **Revocation:** re-checking the same decision and leg replaces the earlier approval. These revoke every unspent BUY:
  - HALTED;
  - a drift freeze or rule 6 in the context;
  - the daily-loss lock;
  - any halt.

  A snapshot holding fewer shares than the SELLs reserve (the stop fired) revokes the SELLs. A spent flatten revokes every unspent order.
- **Pending approvals (D7):** a PendingApproval reserves nothing. Approving it re-runs checks 1–6 against everything outstanding by then. It is vetoed if the price has risen more than the collar above the price you were asked about. The single-order cap applies to each approval that can be spent.
- **Spending:**
  - The adapter calls `RiskEngine.spend` right before it sends. It takes the intent's canonical hash at issue and checks it at spend, so a changed intent is refused.
  - It refuses a BUY once trading is halted, killed or locked.
  - Cancels and flattens never expire and never count as submissions.
  - Execution (B13) calls `release` on any approval it drops. On a re-quote it spends the old approval before it re-checks.
- **Ids:** approval ids are `{O|C|F|S|P}{n}-{µs}`, where the counter n is kept in the risk state. The microseconds keep them unique even after a restart from a state saved before a failed save. The engine lets go of an order or pending approval when it expires, and of a cancel or flatten once it is spent.
- **Exact limits:** `mark`, the caps, the single-order cap and the collar compare the decimal values the floats print as. So exactly 2%, 10%, a cap or 10 bp lands on the inclusive side. D7's $25,000 still compares floats.
- **Halts are signalled where they happen:** `Veto.halts`, `RiskApprovedOrder.halts` (a SELL past the intent limit), `MarkOutcome.kill`, `SubmissionLimitError`, and the published state. A later halt keeps the first halt's reason.
- **Persisting (C7):** each call that changes the risk state hands the new state to the engine's `on_change` before it returns or raises. The live engine (B14) persists it there, so `spend` saves the submission before anything is sent. If `on_change` fails, the change stays in memory, the engine keeps the error as `RiskEngine.unsaved` and logs it, and every later call hands `on_change` the latest state again until a save succeeds. Risk-reducing approvals still go out (your call of 2026-10-14): cancel, flatten and stop approvals, and SELL approvals, are issued and spent. Everything else raises, so a BUY is never approved or spent on a state that wasn't saved. The kill executor puts the error in its report, so its alert carries it; B14 alerts on `unsaved` otherwise. Outstanding approvals live only in the engine object, so a restart starts with none.
- **Resume:** `RiskEngine.resume(equity)`, behind `quant resume` (§8.3), clears HALTED, its reason and the drawdown trigger, and re-bases the peak to `equity`, so the next 10% drawdown is measured from there. This is your call of 2026-10-13: after a drawdown kill the account is flat, so equity can't recover to the old peak. The resume keeps the session's counts and the daily-loss lock. It hands the new state to `on_change` like every other change, and returns a `ResumeOutcome` with the old and new peak, which B14 logs and alerts. It refuses before the first session, on an equity that isn't finite and positive, and when trading isn't halted, since a resume with nothing to clear would only lower the peak. The typed confirmation and the clean reconciliation are B14's.
- **`risk.yaml`** holds the kill switch's settings under `kill`, and `monitor_price_max_age_s` (B9b, §8.3).
- **Check 2** calls `quant.risk.reconcile.reconcile`, the comparison the minute monitor runs too, so the two can't disagree.
- **Stops (B9c, your call of 2026-10-14):** `RiskEngine.approve_stop(snapshot, quote, quantity, stop_price)` issues a `RiskApprovedStop` for one broker-side stop-market SELL, halted or not. It refuses unless the snapshot and the quote are fresh (5 s), the quote isn't crossed, the stop price is below the bid, and the shares are free of every other SELL: the working SELL orders on the snapshot (limits and stops), the SELLs and stops spent since it was taken, and the SELL and stop approvals outstanding. An outstanding stop approval reserves its shares like a SELL, is revoked with the SELLs, and expires 5 s after its snapshot was taken. A spent stop is in flight for the next stop approval but not for the SELL clamp, which still leaves working stops to execution and the broker (§8.2). So a SELL approved for shares a working or in-flight stop covers revokes every outstanding stop approval, since each counted that stop as covering them. Stops never count as submissions. `BrokerPort.place_stop` sends it.

### 8.2 Execution

- **Entries and exits:** marketable limit orders, buying at the IEX ask + 10 bp or selling at the bid − 10 bp. If an order is unfilled after 30 s, it is cancelled and re-quoted. An entry gets up to 3 attempts and is then dropped for this bar. A discretionary exit keeps re-quoting at the bid − 10 bp every 30 s until it fills. Only the kill switch's exits widen, after 30 s (§8.3).
- **Protective stop:** after every fill, one broker-side GTC stop-market order covers the whole position, and it is replaced in place when the size changes. Alpaca reserves the shares behind open sell orders, so an exit cancels the stop first and then sells. After a partial fill, a new stop covers the remainder.
- **Idempotency:** `client_order_id` = instance id + decision id + leg + attempt. A retry after a timeout can't double-submit, and an order from an unknown instance fails reconciliation.
- **Fills** arrive on Alpaca's trade-update stream, with REST polling as backup.
- **Write-ahead:** intents are persisted before submission (§4.2, step 12). On restart, an intent with no broker order is expired and never sent late.

### 8.3 Kill switch

- **Triggers:** the CLI, the dashboard, Telegram `/kill` from your chat id, or automatic. Automatic triggers are max drawdown, a reconciliation mismatch, 5 order errors within 10 minutes, and engine health (C2).
- **Executor:** one implementation, shared by the engine, quant-control and the CLI, with a lock so only one copy runs. It:
  1. writes HALTED;
  2. cancels all orders, stops included;
  3. sells with collared limits, widening after 30 s;
  4. verifies the account is flat with no open orders;
  5. sends an alert.

  Outside regular hours, it cancels entries, keeps the stop and queues the exit for 09:30. A run that can't finish leaves a broker-side stop under the shares no working sell covers (v1.7, B9c).
- **Paths:** quant-control runs the executor when the engine can't, and also watches the engine's heartbeat. `quant kill --direct` is the break-glass path from a shell.
- **Resume:** `quant resume`, run on the machine itself, with a typed confirmation and a clean reconciliation. It re-bases peak equity to the current equity, logs and alerts the old and new peak, and the next 10% drawdown is measured from the new peak (§8.1, v1.7).

**How B9b built it (v1.7).**

- **The broker port** is `quant.risk.broker.BrokerPort`, not part of `execution/` as §3 first listed it. K1 lets `risk` import only `core` and `store`, and the kill executor and reconciliation live in `risk`. `execution` may import `risk`, so B13's Alpaca adapter and the simulated broker implement the port there, and K4 still limits who imports the adapter. Each changing call spends its approval through the engine before it sends.
- **Reconciliation** (`quant.risk.reconcile`) compares a broker snapshot with the engine's book and changes neither. It reports a short, a position difference, and missing, unknown, differing or foreign orders, by order id. A snapshot older than 5 s, or from the future, is STALE: it vetoes an order but halts nothing.
- **The minute monitor** (`quant.risk.monitor`, C8) marks equity as cash + position × the freshest price, only on a price no older than `monitor_price_max_age_s` (120 s) and a fresh snapshot. It returns what the caller must do: flatten for the daily-loss lock, re-place a missing stop, or run the kill for a drawdown, a mismatch, or a stop missing for 5 minutes. Equity is summed in the decimal values the floats print, as `mark` compares them, so exactly 2% or 10% lands on the inclusive side. An equity at or below 0 fires the drawdown kill. A cash that isn't a finite number, or a price that isn't finite and positive, is a data fault: nothing is marked, nothing fires, and the caller treats the engine as unhealthy (rule 6) and alerts. Scheduling and acting are B14's.
- **Health triggers (C2)** fire the kill only with a position open: a heartbeat lost for 300 s, 3 starts within 600 s, or a rule-6 flatten still incomplete after 600 s. A single late bar fires nothing. While flat, a failure only alerts. A start or an order error stamped after `now` hasn't happened yet and isn't counted, and a heartbeat or flatten start after `now` fires nothing, so a clock running ahead never brings a kill forward.
- **The executor** (`quant.risk.kill`) takes the lock without waiting: a second copy, in another process or on another thread, returns BUSY and does nothing. It halts the engine, revokes every outstanding order approval (SELLs too, since one approved against shares the stop has since sold would open a short), and writes HALTED before any broker call. control.db gets the reason in force, which is an earlier halt's when trading was already halted, so it agrees with the engine's first reason. Each cancel and each exit uses a new approval. Exits start at the bid − 10 bp (the ask + 10 bp for a short) and widen 10 bp after every 30 s unfilled, up to 100 bp, rounded toward marketable. After a partial fill the next exit closes only the remainder. Broker errors are retried after a 1 s poll until a 300 s deadline, which ends the run INCOMPLETE. The deadline and the close are checked again when a quote arrives, so no exit goes out after either. If the session closes first, it cancels the working exit and queues the exit for the next open. Outside regular hours it cancels every LIMIT order (entries and exits), keeps the stop and queues the exit; a flat account drops its orders, but once a position shows (an entry filled before its cancel landed) its stop is kept. An exit is queued for the first open at or after the moment the kill started, so a run that crosses 09:30 queues that morning's open; B14 runs a queued exit whose time has passed at once.
- **A kill never dies halfway (B9b review).** Every step is attempted on its own. An exception of any type from the engine and its `on_change`, the store, the broker or the broker's spend is recorded in the report and the run goes on, a broker step being retried after the next poll. An exception outside any step ends the run INCOMPLETE. The alert is always attempted and the report always returned. A lock that can't be taken (say `var/` is missing) is recorded, and the kill runs without it: halting and flattening matter more than excluding a copy, and a second copy's exit is refused while the first's reserves the shares. Under a lasting `on_change` failure the engine still issues the kill's cancel, flatten and stop approvals (§8.1, your call of 2026-10-14), so the kill goes flat, and the report and its alert record the failed save once.
- **New settings in `risk.yaml`** that the earlier docs don't fix: `widen_bp` 10, `max_collar_bp` 100, `poll_s` 1, `deadline_s` 300, `heartbeat_lost_s` 300, `crash_loop_starts` 3 within 600 s, and `monitor_price_max_age_s` 120. You approved them as proposed on 2026-10-14. Every threshold is inclusive.
- **A kill that can't finish leaves a stop (B9c, your call of 2026-10-14).** Before a run ends INCOMPLETE (the deadline, the close, or outside regular hours) it reads a fresh snapshot and places one stop-market SELL, through a `RiskApprovedStop`, under the shares no working sell order covers, and records it in the report. At the deadline it first cancels each working exit once, without waiting, so the stop usually covers the whole position. A cancel that hasn't landed leaves its exit working, and the stop then covers only the rest, so the two can't sell a share twice. The engine has no ATR stop rule (the playbook's ATR multiple reaches it only as an entry's `stop_price`), so the distance comes from `risk.yaml`: `kill.stop_distance_bp` 100, 100 bp below the bid, rounded down to the cent, the same as the widest kill exit. That value is a proposal, not yet approved. Runs that end FLAT or QUEUED place none.

### 8.4 Who may do what

| Action | CLI | Dashboard | Telegram |
|---|---|---|---|
| Status | ✓ | ✓ | ✓ |
| Kill | ✓ | ✓ | ✓ |
| Approve a promotion | ✓ | ✓, typing the proposal id | ✗ |
| Approve a live order (D7) | ✓ | ✓ | ✗ |
| Acknowledge drift, roll back, resume | ✓ | ✗ | ✗ |

The dashboard binds to 127.0.0.1, rejects any other Host header (which blocks DNS rebinding), and requires a CSRF token on every POST.

## 9. State and storage

### 9.1 Files

| File | Holds | Written by |
|---|---|---|
| `market.db` | Minute and hourly bars (point-in-time), revisions, dividends, splits, calendar cache | engine; `quant data backfill` |
| `ops.db` | Decision records, state snapshots, orders, fills, equity marks, risk, health and drift events, pointers, job runs, integration-test runs | engine, evening jobs, itest |
| `control.db` | HALTED flag, commands, approvals (promotions and live orders), alert outbox | dashboard, control, CLI, engine |
| `research.db` | Trial ledger, proposals, gate runs, walk-forward results, fit-cache index | harness, research CLI |
| `artifacts/` | Models, releases, calibrations, reports; immutable and named by SHA-256 | evening jobs, harness |

All four databases run in WAL mode. `ops.db` and `control.db` use `synchronous=FULL`, so a decision or a halt that has been recorded survives a power cut.

### 9.2 Releases and pointers

- A **release** is `strategy.md`, whose front matter is the manifest (feature config, HMM config, label → playbook map, versions and gate report), plus the four playbooks. Releases are immutable.
- The `current` pointers for release, model and calibration keep an append-only history in `ops.db`. A rollback adds a row pointing at an older version, effective at the next open.
- The seed release r1 is the `strategy.md` and `playbooks/` in the repo, built from the spec's defaults: features F1–F5 (§3, D3), the HMM search (§4) and the playbooks (D8).
- `engine.yaml` (switching thresholds, timing, collar, deadband) and `costs.yaml` are L3 config. They change only through a reviewed commit, and their hashes go into every DecisionRecord.

### 9.3 Engine startup and recovery

1. Config check: `risk.yaml` and `gates.yaml` must match the committed versions, and any change since the last start sends an alert.
2. Take the single-instance lock.
3. Execute pending kill commands before anything else.
4. Broker sync: record fills that happened while the engine was down, then reconcile. A mismatch halts trading.
5. Load the current release, model and calibration, and check their provenance.
6. Rebuild the filter from stored bars and compare it bit for bit with the last DecisionRecord. A mismatch halts trading.
7. Replay missed bars without trading (C6).
8. Start the loops and report ready to the service manager.

### 9.4 Backups

Every night at 02:00, SQLite's online backup copies all four databases and rsync copies the artifacts into `backups/<date>/`, keeping 30 days. On the VPS, your Mac can optionally pull a copy.

## 10. Backtest and research

- **SimBroker:** fills at the first SIP minute bar at or after τ, with 2 bp slippage (5 bp for sensitivity runs). Stops are simulated on minute bars, and a gap through a stop fills at the open. SEC and FINRA TAF fees come from `costs.yaml`, with effective dates. Dividends are credited on the ex-date. The minute monitor runs on minute closes (C8).
- **Walk-forward:** FIT trains the first model, then monthly refits run through DEV exactly as they would live: same job, same times, cold start (C4).
- **Speed:** the streaming decision loop covers 2019–2024 in seconds. Refits dominate the cost: about 73 refits × 50 restarts. Fits are cached by feature config, HMM config, data hash, window end, previous model and training-code version. A playbook-only proposal reuses the champion's models and finishes in minutes. A feature or HMM change refits everything, which should take on the order of an hour on 4 cores; the build will measure it.
- **Gates:** G1–G6 as in spec §11. The Deflated Sharpe uses the ledger's trial count and the variance of the Sharpe ratios across trials. The Newey–West lag is ⌊4(n/100)^(2/9)⌋.
- **Baselines:** buy-and-hold SPY, walk-forward trend-following and walk-forward mean reversion. Each static family re-picks its parameters monthly from a fixed grid by the best Sharpe on its training window, and G5 requires beating all three (C14).
- **Trial ledger:** append-only, and each row hashes the one before it. `quant ledger verify` re-checks the chain. Trials, lockbox openings and any gate-config change are all recorded. The chain has no anchor outside `research.db`, so a rewrite of the whole file (every row re-chained with its head, or both tables emptied) passes `ledger verify`. v1 accepts this because only the quant user writes `research.db` (§2, §9.1); the nightly backups keep earlier copies, which v1 doesn't compare (B10).

## 11. Nightly loop

1. **17:00, evening job (quant):** writes `inbox/<date>/bundle.json`. The bundle contains:
   - today's decisions, fills and switches;
   - wrong state calls;
   - the daily report;
   - the current release files;
   - the feature menu with its bounds, plus the playbook and proposal schemas;
   - gate history and a ledger summary (DEV results only);
   - the latest loss rules.

   It contains no lockbox results, no risk limits and no secrets; a secret scan must pass before the file is written. It is capped at about 200 KB, roughly 50,000 tokens, by dropping the oldest loss rules and ledger detail first.
2. **18:00, nightly job (quant-llm):** reads only the bundle and sends one request (AD6). The bundle sits inside delimited data blocks, and the system prompt says data never carries instructions. The job writes `outbox/<date>/proposals.json`, capped at 1 MB. On a refusal, a timeout or invalid output, it writes only an error record.
3. **18:30, harness (quant):**
   - Re-validates everything with strict models: unknown keys are rejected, paths are whitelisted, values must sit inside the menu bounds, the ±20% rule applies (C10), and there are at most 3 proposals. The API's schema checks only structure, so all bounds are checked here.
   - Appends one rule per losing trade to `journal/losses.md`.
   - For each surviving proposal: takes a `TrialTicket`, runs the DEV walk-forward, checks G1–G6, then asks whether it beats the champion (C9).
   - A winner joins the approval queue, and you get a Telegram alert.
4. **You approve** in the CLI or dashboard. If the release changes the feature or HMM config, the evening job first fits its operational model. The engine activates it at the next open (C7).
5. `harness.auto_promote_in_paper` is the one-line D12 switch. It defaults to false.

Each proposed change is a structured edit, `{path, from, to}`, against the release manifest or a playbook's front matter or prose. The harness can check and apply it exactly, which a free-text diff would not allow.

**Cost:** about 30–60k input tokens and 10–25k output tokens, thinking included, at $4 and $20 per million: $0.30–0.75 a night, under $20 a month. Prompt caching wouldn't help, because a cache entry lives at most an hour and the request runs once a night.

## 12. Dashboard, alerts and reports

- **Dashboard:** FastAPI serves server-rendered pages updated with HTMX, uPlot draws the charts, and SSE pushes changes. It reads all four databases read-only, polls `ops.db` every second, and writes only commands and approvals to `control.db`. It shows everything listed in spec §13.
- **Alerts:** any process writes to the `control.db` outbox, and quant-control delivers to Telegram with retries, de-duplication and rate limiting. An alert still unsent after 5 minutes shows as a health warning on the dashboard.
- **Daily report:** built from `ops.db` at 16:20, stored as an artifact, with a summary sent to Telegram.

## 13. Testing

| Spec ID | Test | How |
|---|---|---|
| I1, T-NL1 | Truncation invariance | Run the engine loop with SimBroker for cut times × 3 variants, then compare DecisionRecords and state hashes. The fast tier runs 50 cut times on synthetic bars at every commit; the full tier runs 200 cut times on real bars before each merge and nightly (P4). |
| I2 | No smoother on the decision path | An AST scan fails the build on `predict`, `predict_proba`, `decode`, `score_samples` or `eval.smoother` anywhere outside `eval/`, plus rule K3. `decode` counts when it is called the way hmmlearn's Viterbi is, with an array; `bytes.decode("utf-8")` is allowed. hmmlearn itself is imported only by `hmm.train`, `eval/` included (B0). |
| I3 | Isolation | Rules K1 and K2 |
| I4 | One choke point | Rule K5's scan, the runtime token, and adapter tests that reject plain objects, subclasses, and reused or expired approvals |
| I5 | Paper lock | Adapter-constructor tests across every combination of mode, URL and checklist state |
| I6 | No secrets leak | gitleaks, plus redaction tests over logs, database rows and the L1 bundle |
| T-NL2 | Provenance | Every DecisionRecord's model fit window and calibration as-of time precede τ |
| T-NL3 | Filter ≠ smoother | On synthetic HMM data, interior points differ from hmmlearn's posteriors and the final point matches |
| A2 | Lockbox | DataPolicy refuses to score LOCKBOX without a ticket, and the ledger chain verifies |
| (none) | Determinism | Two backtest runs give identical record hashes, and so does a replay of recorded live inputs |

The test suite has six other layers:

- **Unit tests.**
- **Property tests** with hypothesis: the system never goes short, never exceeds a cap, probabilities always sum to 1, and de-risking switches ignore the cooldown.
- **Golden tests:** a checked-in decision log for 2018-01 → 2018-04, inside FIT and including the February 2018 volatility shock, decided by a fixture model fit on 2016–2017. Any change to it must be explained in review. An early-2020 window is added after CP2, because DEV stays dark until trial #1 (P4, C11).
- **Simulation tests:** fills, stops, fees and dividends.
- **Integration tests** on the test account, run weekly and on demand.
- **Stress replays:** March 2020, 2022, a synthetic −10% overnight gap, a 30-minute data outage and a broker outage, using fault-injecting fakes. The April 2025 replay is inside the lockbox, so it runs only at go-live.

**Build gate:** `uv run quant-check` runs ruff, mypy (strict for `core`, `risk`, `decide` and `execution`), import-linter, the test suites listed above and gitleaks. A git hook runs the fast gate before every commit, and a second hook runs the full gate on every `git merge --no-ff` into `main` (B0).

## 14. Determinism

- `uv.lock` pins every dependency with hashes, and the Python 3.12 patch version is pinned too.
- The engine, backtest and harness set single-threaded BLAS (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` all set to 1) and `PYTHONHASHSEED=0` before importing numpy. Parallelism happens only across separate processes.
- All arithmetic is float64, hashing uses canonical JSON, and every seed is explicit.
- Every artifact and DecisionRecord records the code commit, the lockfile hash, the platform and the BLAS build.
- The paper track record counts toward go-live only for decisions made from a clean, tagged commit.
- Moving from the Mac to the VPS copies the artifacts rather than re-fitting. Parity is checked on the machine that made the decisions.

## 15. Deployment

**This Mac (development and first paper trading):**

- launchd agents under your user run the engine, control, dashboard, evening jobs, harness and backup. A LaunchDaemon with `UserName=_quantllm` runs L1.
- This Mac is a laptop, so it must stay awake and on power through market hours (`caffeinate`). Sleep counts as an engine outage, so rule 6 takes the position flat (C6). For that reason, the go-live evidence is collected on the VPS.

**VPS:**

- Ubuntu 24.04 LTS, 4 vCPU, 8 GB of RAM and an 80 GB SSD.
- Users `quant` and `quant-llm`, neither with a login shell.
- systemd units as in §2, hardened with NoNewPrivileges, ProtectSystem=strict, ProtectHome, PrivateTmp and ReadWritePaths per unit. The harness runs at low CPU weight so it can't starve the engine.
- Timers run on New York time.
- chrony keeps the clock in sync, and ufw allows only SSH or Tailscale. Security updates install unattended.

**Migration:** stop the Mac engine while flat, copy `var/`, start the VPS engine, and run `quant adopt-account` once. The step-by-step runbook is part of the ship phase.

## 16. Failure modes

| Failure | Detected by | Response |
|---|---|---|
| A bar is late or missing at τ | Ingest retries until τ + 5 min | Rule 6: target 0 and entries blocked until two healthy bars |
| The engine crashes | launchd or systemd | Restart, then the §9.3 sequence; missed bars count as stale (C6) |
| The engine hangs | 90 s watchdog | Killed and restarted |
| A crash loop with a position open | quant-control watches the heartbeat | Kill switch (C2) |
| Broker API errors | Order layer | Retries reuse the same `client_order_id`. After 5 errors in 10 minutes the kill switch fires and keeps retrying, while the broker-side stop protects the position. |
| The network is down | Data and broker adapters | Rule 6 and the broker-side stop; alerts queue until the network returns |
| Broker and internal positions differ | Reconciliation before every order and every minute | Halt and alert; never auto-corrected |
| A position has no live stop | Minute monitor | Re-place the stop; if it is still missing after 5 minutes, kill switch. A risk-approved exit working for the whole position counts as the stop and pauses that clock, but if it is still unfilled 5 minutes after exits began covering, the kill switch fires and its exits widen (C5, v1.7, your call of 2026-10-14). |
| A database write fails | Persist step | No record means no order (write-ahead); repeated failures are a health failure, so the position goes flat |
| The clock drifts | chrony; Alpaca clock check | Over 2 s a warning; over 30 s entries are blocked |
| `risk.yaml` or `gates.yaml` is edited | Startup check; hash check every minute | Uncommitted: the engine won't start. Any change sends an alert and takes effect only on restart. |
| Bars are revised after use | 16:45 settle | Stored as a new version; past decisions stand; listed in the parity report |
| SPY splits | Corporate-actions feed | Halt and alert |
| A refit fails sanity or drift | Refit job | Keep the current model; drift also freezes entries (C5); alert |
| L1 refuses, times out or returns invalid JSON | Nightly job; harness validation | No proposals that night; logged and alerted |
| The harness is still running at 09:00 | Deadline | Stops and resumes the next night |
| The Mac sleeps in market hours | Heartbeat gap | Treated as an engine outage (C6) |
| A market-wide trading halt | No new minute bars | Rule 6; orders can't fill anyway, and the broker-side stop stays in place |

## 17. Left out on purpose

- Docker, Postgres, Redis and message brokers. One machine and about seven decisions a day don't need them.
- Streaming bars for decisions. The engine polls at τ, which is exactly what the backtest models.
- Short and leveraged code paths. Position targets are non-negative types.
- Extended-hours orders.
- Any LLM at runtime. L1's output is data that the harness checks.
- A multi-asset abstraction beyond a `symbol` setting.

## 18. To verify during setup

- Whether Alpaca paper accounts credit dividends. The parity report reconciles dividends explicitly either way.
- Whether your Alpaca login can open a second paper account (AD3).
- That SIP minute bars on your plan are served 15 minutes after the minute, so the 16-minute decision delay (D2) holds.
- Alpaca's rate limits against the 1-million-bar backfill.

## 19. The plan

The plan (Gate 3, approved 2026-10-05) is [PLAN.md](PLAN.md). It started from this outline, which orders the build so that the riskiest guarantees are proven first:

1. bar store and point-in-time view;
2. features;
3. filter, with T-NL3;
4. the T-NL1 harness;
5. switching and sizing;
6. risk engine, with I4;
7. SimBroker and the backtest;
8. gates and the ledger;
9. Alpaca adapters and the paper engine;
10. control and the dashboard;
11. L1 and the harness;
12. deployment.

Each step will list its failing tests first, the spec IDs it covers, and its definition of done.
