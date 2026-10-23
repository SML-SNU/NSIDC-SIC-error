# The manuscript's own output, kept for the record

These are the files the manuscript's numbers were read off, and they are **not**
what the code in this repository produces. The difference is small and confined
to one baseline, but it is real, so both sets are shipped rather than one of
them quietly reconciled.

## What happened

The experiment was run twice, five weeks apart:

| run | date | code revision |
|---|---|---|
| `..._wc50` (total error) | 2026-10-08 | before the `sicreg` arm existed |
| `..._bc_wc50` (bias-removed error) | 2026-10-22 | current |

Between the two, the `sicreg` arm — σ resolved by region *and* concentration —
was added, and with it a change in how the region × season σ is built. The
manuscript's table therefore mixes two revisions: its first three rows come from
October 8, its bias-removed row from October 22.

`results/` one level up holds a rerun of all four configurations under the
single revision published here, so that the code and its stated output agree.

## How much moves

Comparing the manuscript's table against the current code, over all 16 cells of
each product's table:

| row | what changes |
|---|---|
| `Constant 15 %` | nothing — identical to the last digit |
| `σ(season, SIC)` | ≤ 0.05 RMSE, ≤ 0.02 CRPS, ≤ 0.14 coverage |
| `σ(season, SIC)`, bias-removed | nothing — this row already came from the current code |
| `σ(season, region)` | up to 1.1 points of coverage |

So only the region × season baseline moves visibly, and the largest move is
0.35× the spread already quoted on that number. The two largest:

|  | published | current code |
|---|---|---|
| NT, 15–85 %, coverage | 87.13 % | 86.07 % |
| BT, 15–85 %, coverage | 81.44 % | 80.33 % |

Both move away from the concentration-resolved result (96.8 %), so the
comparison the paper draws is marginally wider under the current code, not
narrower.

`none` and `const` reproduce bit for bit in every run, which is what localises
the change to the σ construction rather than to the data, the split, the
network or the scoring.
