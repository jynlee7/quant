# BUILD LOG: SPY Regime Trader

One entry per finished step (PLAN §2, rule 7): what was built, the tests added, run times, deviations and open items. Newest first. Each step is a branch `bNN-name` with a red commit and a green commit, merged to `main` and tagged `bNN-done`.

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
