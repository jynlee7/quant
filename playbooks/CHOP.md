---
# CHOP: mean reversion (D8, PLAN §5). Code reads only this front matter.
label: CHOP
strategy: mean_reversion
params:
  n: 35                       # z = (close - 35-bar mean) / 35-bar s.d.
  entry_z: 1.5                # enter at z <= -1.5
  exit_z: 0.0                 # exit at z >= 0
  atr_bars: 35                # ATR(35) sizes the stop
stop_loss: 3.0                # 3 x ATR(35) below the entry close, fixed at entry
take_profit: null             # none
max_size: 0.5                 # of equity, clamped to the label's hard cap
version: 1
created_at: 2026-10-19T00:00:00Z
invalidation:
  - {kind: bars_in_trade_exceeds, bars: 21}
  - {kind: label_prob_below, threshold: 0.30}
---

# CHOP: buy the dip back to the mean

When the market goes sideways, prices that stretch too far tend to come back. Buy when the
close is 1.5 standard deviations below its 35-bar average, and sell when it is back at the
average.

Every entry carries a stop three average true ranges below the entry price. It never moves.

The trade is abandoned after 21 bars (three sessions) in it, or if the model's confidence in
CHOP falls below 30%.
