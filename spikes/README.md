# spikes/

Throwaway checks of how vendors behave, such as the Alpaca probe in B6 (PLAN P11).

- They are exempt from red/green.
- `src/` can never import them: import-linter contract P11 fails the build if it does.
- Each spike records what it found in `docs/VERIFY.md`, which is where the build reads the answer from.
