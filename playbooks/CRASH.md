---
# CRASH: flat (D8, PLAN §5). Its hard cap is 0 in code, so no other strategy is allowed.
label: CRASH
strategy: flat
params: {}
stop_loss: null               # flat holds nothing, so it has no stop
take_profit: null
max_size: 0.0
version: 1
created_at: 2026-10-19T00:00:00Z
invalidation: []
---

# CRASH: flat

In a crash, hold nothing. This never changes: CRASH's hard cap is 0 in code, and the schema
refuses any strategy but flat for it.
