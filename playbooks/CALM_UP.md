---
# CALM_UP: trend-following (D8, PLAN §5). Code reads only this front matter.
label: CALM_UP
strategy: trend_following
params:
  fast: 35                    # enter when EMA35 > EMA175 and the close is above EMA35
  slow: 175                   # exit when EMA35 < EMA175
  atr_bars: 35                # ATR(35) sizes the stop
stop_loss: 4.0                # 4 x ATR(35) below the entry close, fixed at entry
take_profit: null             # none
max_size: 1.0                 # of equity, clamped to the label's hard cap
version: 1
created_at: 2026-10-19T00:00:00Z
invalidation:
  - {kind: label_prob_below, threshold: 0.30}
---

# CALM_UP: follow the trend

When the market is calm and rising, ride it. Buy when the one-week average (EMA35) is above
the five-week average (EMA175) and the price is above the one-week average. Sell when the
one-week average drops back below the five-week one.

Every entry carries a stop four average true ranges below the entry price. It never moves.
There is no profit target: the trend exit takes profits.

The trade is abandoned if the model's confidence in CALM_UP falls below 30%.
