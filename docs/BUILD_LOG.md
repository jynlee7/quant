# BUILD LOG: SPY Regime Trader

One entry per finished step (PLAN §2, rule 7): what was built, the tests added, run times, deviations and open items. Newest first. Each step is a branch `bNN-name` with a red commit and a green commit, merged to `main` and tagged `bNN-done`.

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
