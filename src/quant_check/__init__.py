"""The build gate, `quant-check` (ARCHITECTURE §13, PLAN §2).

It runs every structural guard (K1–K6, I2, I6), the red-mark audit (P2), the test tiers
(P4) and the trace matrix. It is a dev tool, and the `quant` package never imports it (P11).
"""
