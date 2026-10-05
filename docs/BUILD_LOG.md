# BUILD LOG: SPY Regime Trader

One entry per finished step (PLAN §2, rule 7): what was built, the tests added, run times, deviations and open items. Newest first. Each step is a branch `bNN-name` with a red commit and a green commit, merged to `main` and tagged `bNN-done`.

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
