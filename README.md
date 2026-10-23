# NSIDC-SIC-error — MLP experiment

Code and data for the machine-learning experiment in

> Lee et al. (2026), *SSMIS sea-ice concentration observation error characterised
> against downscaled Landsat-8 sea-ice concentration.*

The paper characterises the observation error of two passive-microwave
sea-ice-concentration (SIC) products — NASA Team (NSIDC-0051 v2.0, "NT") and
Bootstrap (NSIDC-0079 v4.0, "BT") — against a downscaled Landsat-8 SIC
reference, and fits it as a function of SIC, season and region for assimilation
into NEMO5/SI3.

This repository holds the Discussion experiment only: **does a
concentration-dependent observation error actually buy anything over a constant
one?** A small neural network maps the SSMIS observation onto the Landsat-8
reference, and the prescribed observation error enters as the width of the
network's predictive distribution. Four observation-error configurations are
compared on identical splits.

The error characterisation itself (binning, the cubic fit, the regional and
seasonal tables) is not in this repository; its results are shipped here as the
lookup tables the experiment reads.

## What the experiment does

The observation error σ enters in two places and **never as an input feature**:

1. as the per-sample weight in the loss, `w = 1/σ²`, so a matchup the product is
   known to get wrong counts for less during training;
2. as the predictive width that is *scored* — the 95 % coverage and the CRPS ask
   whether the declared σ matches the error the network actually makes.

σ must not be a feature: it is a deterministic function of the SIC that is
already an input, so as a feature it adds nothing and all configurations
collapse onto the same model.

Five configurations, differing only in σ. The manuscript's table reports the
four that matter for the product (`const`, `regseason`, `sic`, and `sic` with
the bias removed); `none` and `sicreg` are diagnostics.

| arm | σ |
|---|---|
| `none` | none — plain least squares |
| `const` | 15 % everywhere |
| `regseason` | one value per region × season |
| `sic` | σ(SIC \| season) — the cubic curve, **this study** |
| `sicreg` | σ(SIC \| season, region) |

`none` and `const` are the same model by construction — a uniform weight is a
scalar multiple of the loss — and come out numerically identical. That is the
built-in check that the weighting is implemented correctly; they differ only in
the σ that is scored.

Run with `--bc`, the σ arms are given the **bias-removed** error instead: each
5 % SIC bin's own mean difference is subtracted before the curve is fitted. The
`none` and `const` baselines are deliberately left on the total error, so any
change between the two runs is the bias removal and nothing else.

Three things the code is careful about, each of which changes the answer:

- **Splitting is by blocks of consecutive observation days**, not by random
  cell. SIC fields are strongly correlated in space and time, and a random
  split puts near-copies of training cells into the test set. Blocks are dealt
  out within each season, because Freezing is 12 % of the record and clustered
  in Feb–Mar: dealing from one pile left the test split with 1.2 % Freezing.
- **σ is refitted on the training split only.** The published lookup was fitted
  on everything, so prescribing it would leak the test set. `--lookup` uses the
  published tables instead, for comparison.
- **`w = 1/σ²` is brutally uneven**: consolidated ice (σ ≈ 2.4 %) outweighs the
  marginal ice zone (σ ≈ 32 %) by ~180× per sample. Pooled scores can therefore
  improve while the marginal ice zone gets worse, so every metric is reported
  per 5 % SIC bin as well as pooled. `--wclip 50` caps the weights at their
  median, which is what the paper reports.

## Files

| File | Role |
|---|---|
| `mlp_experiment.py` | the experiment: load, split, refit σ, train, score, plot |
| `make_results_table.py` | builds the manuscript's performance table from the 20-split output |
| `mlp_schematic.py` | the network diagram |
| `export_matchups.py` | provenance: how `data/matchups_*.csv.gz` was produced from the author's intermediate files |
| `figstyle.py`, `seasons.py` | shared figure style and the season definition |

| Data | Contents |
|---|---|
| `data/matchups_NT.csv.gz` | 55,243 SSMIS(NT)/Landsat-8 matchups, 31,096 of them at the paper's coverage cut |
| `data/matchups_BT.csv.gz` | the same cells for Bootstrap |
| `data/final_obserror_nd4-nd10-nd16*.csv` | published region × season error, read by the `regseason` arm |
| `data/obs_error_poly3_lookup*.csv` | published σ(SIC) curves, read only by `--lookup` |

`results/` and `figures/` hold the output of the four runs above, so that a
rerun can be checked against them. A rerun writes to `output/` and overwrites
neither.

`results/as_published/` holds the manuscript's own output, which for one of the
four configurations was produced by an earlier revision of this code. Only the
region × season baseline moves, by up to 1.1 points of coverage; the
concentration-resolved rows the paper is about are unchanged or move in the
second decimal. `results/as_published/README.md` gives the numbers.

### The matchup table

One row per 25 km NSIDC polar-stereographic grid cell per day on which both a
Landsat-8 and an SSMIS SIC exist, 2020-02 to 2022-09, 12 Arctic sub-regions.

| column | meaning |
|---|---|
| `date` | YYYYMMDD of the Landsat acquisition |
| `region` | one of the 12 sub-regions |
| `lat`, `lon` | cell centre, degrees |
| `ssmis` | SSMIS SIC, % |
| `landsat` | Landsat-8 SIC block-averaged onto the same cell, % |
| `n_distinct` | distinct native 6.25 km Landsat pixels in the cell, 1–16 |

`n_distinct` is the coverage measure the paper cuts on; `n_distinct >= 10`
(≥ 10 of 16 pixels) is the headline cut and gives 31,096 matchups. Cut on this
and never on a count of observations, which exceeds 16 where overlapping scenes
re-image the same ground without adding coverage.

NT and BT are given the identical cells and the identical Landsat reference —
only the microwave product compared against changes.

Two caveats that belong with the data:

- **Landsat-8 is optical, so the polar night is unobserved.** The seasons are
  not evenly sampled: Freezing (Nov–Mar) is 98 % Feb–Mar with no December data
  at all, and Interim (Apr/May/Oct) is 97 % Apr–May. A "Freezing" number is
  really a late-winter one.
- **Bootstrap has an intrinsic low-concentration cutoff near 13–15 %** from its
  weather filter, which writes a valid 0 % rather than a gap. BT is therefore
  under-sampled between 0 and 15 %, and this is in the source algorithm, not in
  this processing.

## Running it

```bash
conda env create -f environment.yml
conda activate nsidc-sic-error

python mlp_experiment.py 10 --wclip 50              # one split, ~1 min
python mlp_experiment.py 10 --wclip 50 --seeds 20   # the paper's run, ~8 min
```

The four runs the paper reports:

```bash
python mlp_experiment.py 10 --wclip 50 --seeds 20
python mlp_experiment.py 10 --wclip 50 --seeds 20 --bc
OBSERR_TAG=_BT python mlp_experiment.py 10 --wclip 50 --seeds 20
OBSERR_TAG=_BT python mlp_experiment.py 10 --wclip 50 --seeds 20 --bc
python make_results_table.py                        # the manuscript's table
```

`OBSERR_TAG=_BT` switches the whole chain to Bootstrap; it is the only
difference between the NT and BT numbers.

A single split can be lucky — the test split is ~8 blocks of 7 days, so one
unusual week moves the numbers. `--seeds 20` repeats the split, the σ refit and
the network initialisation together, and the spread across the 20 is the honest
uncertainty on every number quoted.

### Reproducibility

Everything that is random follows from `SEED = 20260909`: the split, the σ refit
and the network initialisation. The network is NumPy only, so there is no
framework non-determinism and no GPU. The shipped matchup table stores 9
significant digits in the original row order, so reading it is an exact
round-trip of the author's netCDF input — a rerun of the commands above
reproduces `results/` to the last digit.

### Environment

`numpy`, `pandas`, `scipy`, `matplotlib`. `netCDF4` is needed only by
`export_matchups.py` and by the author's netCDF input path.

The network is 4 inputs → 32 tanh → 32 tanh → 1 linear, implemented in NumPy
alone — no deep-learning framework, nothing to install, no GPU, and nothing that
changes answers between library versions. Inputs are the SSMIS SIC, the sine
and cosine of the day of year, and the latitude.

## Data sources

The matchup table is derived from, and subject to the terms of:

- **NSIDC-0051**, *Sea Ice Concentrations from Nimbus-7 SMMR and DMSP
  SSM/I-SSMIS Passive Microwave Data, Version 2* — https://nsidc.org/data/nsidc-0051
- **NSIDC-0079**, *Bootstrap Sea Ice Concentrations from Nimbus-7 SMMR and DMSP
  SSM/I-SSMIS, Version 4* — https://nsidc.org/data/nsidc-0079
- **Landsat-8 OLI/TIRS** Collection 2, USGS — https://earthexplorer.usgs.gov

Please cite those products as well as this paper if you use the table.

## Licence

Code: MIT, see `LICENSE`. The derived matchup table is released under the same
terms; the underlying NSIDC and USGS products carry their own.
