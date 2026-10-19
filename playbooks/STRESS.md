---
# STRESS: stand aside (D8, PLAN §5). Code reads only this front matter.
label: STRESS
strategy: flat
params: {}
stop_loss: null               # flat holds nothing, so it has no stop
take_profit: null
max_size: 0.0
version: 1
created_at: 2026-10-19T00:00:00Z
invalidation: []
---

# STRESS: stand aside

When volatility is high or the trend is falling, hold nothing. The nightly loop may later
propose taking part at a reduced size (at most 0.25 of equity), through the gates.
