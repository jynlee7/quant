# CLAUDE.md: SPY Regime Trader

This is a paper-trading system for SPY on 1-hour bars through Alpaca paper. A Gaussian HMM reads the regime, deterministic code decides and sizes, and hard risk limits sit in code. The approved design is in [docs/SPEC.md](docs/SPEC.md) v1.2, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) v1.4 and [docs/PLAN.md](docs/PLAN.md) v1.1. Build progress is in [docs/BUILD_LOG.md](docs/BUILD_LOG.md).

## Start every session here

1. Read this file, docs/BUILD_LOG.md and `git status`.
2. Continue the open step: the checked-out `bNN-*` branch, or the next step in PLAN §4.

Nothing depends on chat memory.

## Commands

| Command | What it does |
|---|---|
| `.venv/bin/quant-check` | The fast gate. The commit hook runs it before every commit. |
| `.venv/bin/quant-check --full` | Before a merge: full-tier tests, no red marks left, every claimed id covered |
| `.venv/bin/quant-check --trace` | The matrix of spec ids against the tests that name them |
| `.venv/bin/quant-check --only K5,K6` | Named stages only. Stages: python, ruff, format, mypy, imports, K5, K6, I2, I6, P2, hook, tests, trace |
| `.venv/bin/python -m pytest` | Fast tier directly; add `--tier full` or `--tier integration` |
| `uv sync` / `uv lock` | Install from uv.lock; re-lock after editing pyproject.toml |

Hooks and services call `.venv/bin/...` directly, never `uv run`, because uv lives in `/opt/anaconda3`. If the hook is missing, `git config core.hooksPath .githooks` reinstalls it, and quant-check fails until it's there.

## Rules for every step (PLAN §2)

1. **Branch** `bNN-name` from `main`.
2. **Red commit:** stubs that raise `NotImplementedError`, plus the step's tests. Each test is marked `@pytest.mark.xfail(strict=True, reason="red:B3", raises=NotImplementedError)` and names its ids with `@pytest.mark.spec(...)`. Steps are unpadded (B3, never B03). Add the step's claimed ids to `tests/steps.toml`.
3. **Green commit:** the implementation goes in and the marks come off. The fast tier passes on every commit; the full tier (`--full`) passes before the merge.
4. A step that touches the decision path adds its layer to the T-NL1 truncation harness before it is done.
5. **Review:** an independent code review of the step's diff (the code-review skill), checked against the step's ids. Fix every finding, or write it down in BUILD_LOG.md with a reason.
6. **No silent deviations:** if code must differ from SPEC, ARCHITECTURE or PLAN, change the doc in the same merge, with a minor version and a BUILD_LOG note. Changing an approved decision needs the user first.
7. **Done:** `git merge --no-ff` to `main`, tag `bNN-done`, and add a BUILD_LOG.md entry: built, tests, run times, deviations, open items.

## When to stop for the user (P6)

- Checkpoints CP2–CP5 (they reply "approve CPn"). CP1 is a report and doesn't wait.
- Anything that needs their keys, accounts, sudo or money.
- An irreversible act, or a change to an approved decision.

If a step is blocked on them, continue with the steps that aren't, and come back.

## Conventions

- UTC inside the code, New York time only at the edges. "Now" comes only from `quant.core.clock` (K6).
- float64 arithmetic, canonical JSON (sorted keys, NaN forbidden) with SHA-256 hashes, explicit seeds.
- No pickles. Artifacts are `.npz` loaded with `allow_pickle=False`, plus JSON.
- Any secret read from a file (`.env`, `.env.test`) goes through `quant.core.logs.register_secret` before anything can log, so the redactor knows it (I6).
- Log with `structlog.get_logger()`. Timestamps come from the Clock passed to `configure_logging`, so backtests and replays log simulated time.
- Only `quant.hmm.train` imports hmmlearn, and it calls only `fit`. `predict`, `predict_proba`, `decode` and `score_samples` never appear outside `eval/` (I2).
- `RiskApprovedOrder`, `RiskApprovedCancel` and `RiskApprovedFlatten` are built only in `quant.risk.engine`, and `TrialTicket` only in `quant.research.ledger` (K5).
- The import rules (K1–K4, P11) live in `.importlinter`. A new subpackage must be placed in K1, K2 and K4, or quant-check fails.
- Every test names the ids it enforces, for example `@pytest.mark.spec("A1")`. Ids are the first cells of table rows in the three docs (A1, K5, C11), or numbered sections for requirements written as prose (`SPEC§2`, `ARCH§4.3`, `PLAN§5`).
- The fast tier is offline and must stay under 2 minutes. Mark slow tests `full` and tests that need the Alpaca test account `integration`.
- Spikes (`spikes/`) are throwaway vendor checks, exempt from red/green. `src/` never imports them, and they record findings in `docs/VERIFY.md`.

## Traps

- **DEV stays dark** (2019–2024) until trial #1, which is r1 at CP2 (C11). Nothing computes strategy P&L on DEV before then, so build and debug on synthetic data and FIT (2016–2018).
- **The lockbox** (2025-01 to the seal) is never scored before the go-live test (C1, A2).
- **market.db stamps time itself.** A write is `known_at=NOW` (the store's Clock, under the write lock) or `HISTORY` (each row at its own available_at). History only adds whole sessions, and only before live trading reads the store; after that, late data is NOW. A live view is `view(tau, as_of=NOW)`.
- **Keys** live only in `.env` and `.env.test` (gitignored, chmod 600), and the user adds them. Never print, log or paste them.
- **numpy is held below 2.5:** hmmlearn 0.3.3 sets `ndarray.shape`, which NumPy 2.5 deprecates.
- **The hook refuses unstaged changes and untracked files.** The gate checks the working tree, so the working tree must equal the commit.
- **Git hooks export `GIT_INDEX_FILE` and friends.** quant_check scrubs them (`repo.scrubbed_env`) before running git anywhere else; test helpers must do the same.
