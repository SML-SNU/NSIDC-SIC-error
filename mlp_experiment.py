#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the SIC-dependent observation error buy anything over a constant 15 %?

A small supervised experiment for the Discussion.  The model maps the SSMIS
observation onto the Landsat truth; the observation error enters as the width of
the predictive distribution, i.e. through the Gaussian NLL

    L = sum_i  w_i (y_i - mu_i)^2 ,   w_i = 1 / sigma_i^2   (normalised, see below)

and is also the sigma that is scored.  Three arms, differing ONLY in sigma:

    none     no error information            w = 1          (plain MSE)
    const    sigma = 15 % everywhere         w = 1          (after normalisation)
    sic      sigma = sigma(SIC, season)      w = 1/sigma^2   <- this study

WHY sigma MUST NOT BE AN INPUT FEATURE
--------------------------------------
sigma is a deterministic function of the SSMIS SIC that is already an input, so
as a feature it carries no information: a constant is uninformative and
poly3(SIC) is redundant.  Fed as a feature the three arms converge to the same
model and the experiment is a guaranteed null.  Entering through the loss (and
through the scored predictive width) is what makes them genuinely differ.

WHY 'none' AND 'const' ARE THE SAME MODEL
-----------------------------------------
A uniform weight is a scalar multiple of the loss, so it leaves the optimum
untouched -- the two arms are identical by construction, up to the learning
rate.  Weights are normalised to mean 1 to remove even that difference, so the
two arms coming out numerically identical is the built-in check that the
weighting is implemented correctly.  They differ only in the sigma that is
SCORED (pooled training RMSE vs 15 %), which shows up in the calibration
metrics, not in RMSE.

THE TRAP THIS SCRIPT DELIBERATELY EXPOSES
-----------------------------------------
w = 1/sigma^2 is brutally uneven: consolidated ice (sigma ~ 2.4 %) outweighs the
MIZ (sigma ~ 32 %) by ~180x per sample.  Measured on the nd>=8 matchups the MIZ
is 31 % of the sample but under 1 % of the loss weight, while the 95-100 % bin
alone takes ~90 %.  Pooled RMSE can therefore improve while the MIZ gets worse,
so every metric here is reported per SIC bin as well as pooled, and --wclip caps
the weights at a percentile if you want to see the mitigated version.

NO CIRCULARITY
--------------
sigma(SIC, season) is refitted on the TRAINING split only (same recipe as
the paper's recipe: 5 % bins, sqrt(N)-weighted cubic, thin bins included so
the cubic cannot extrapolate).  The published lookup in data/ was fitted on
everything, so prescribing it would leak the test set; --lookup does that
deliberately, for comparison.

Splitting is by BLOCKS OF CONSECUTIVE DAYS, not by random cell: SIC fields are
strongly autocorrelated in space and time, and a random split leaks near-copies
of training cells into the test set.

Inputs
------
  data/matchups_<PROD>.csv.gz              the matchups (see export_matchups.py)
  data/final_obserror_nd4-nd10-nd16*.csv   the published region x season table,
                                           read by the 'regseason' arm
  data/obs_error_poly3_lookup*.csv         the published sigma(SIC) curves, read
                                           only with --lookup

Outputs
-------
  output/ml_experiment_metrics_<cut><TAG>.csv  arm x stratum -> N, RMSE, bias,
                                        CRPS, 95 % coverage, NLL (one split)
  output/ml_experiment_seeds_<cut><TAG>.csv    the same, every split, with
                                        --seeds (this is what the paper's table
                                        is built from)
  figures/fig34_ml_experiment<TAG>.png  (a) the interval each arm declares
                                        (b) RMSE by SIC bin
                                        (c) 95 % coverage by SIC bin
                                        (d) CRPS by SIC bin
  figures/fig35_ml_scatter<TAG>.png     predicted vs Landsat-8
  figures/fig36_ml_crps<TAG>.png        CRPS and coverage by SIC bin

Usage
-----
    conda activate python_js            # or: conda env create -f environment.yml
    python mlp_experiment.py                  # n_distinct >= 8, one split
    python mlp_experiment.py 10               # n_distinct >= 10 (the paper's cut)
    python mlp_experiment.py =16              # n_distinct == 16
    python mlp_experiment.py --xgb            # add an XGBoost arm set
    python mlp_experiment.py --wclip 95       # cap weights at p95
    python mlp_experiment.py --seeds 20       # repeat over 20 splits
    python mlp_experiment.py 10 --lookup      # sigma from the published CSVs

The two runs the manuscript reports (~25 min each on one core):

    python mlp_experiment.py 10 --wclip 50 --seeds 20
    python mlp_experiment.py 10 --wclip 50 --seeds 20 --bc

and the same two for Bootstrap, selected by one environment variable:

    OBSERR_TAG=_BT python mlp_experiment.py 10 --wclip 50 --seeds 20
    OBSERR_TAG=_BT python mlp_experiment.py 10 --wclip 50 --seeds 20 --bc

@author: kimjs
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
from scipy.stats import norm

import figstyle as FS
from seasons import SEASON_ORDER, season_arr

FS.apply()

#=======================================================
# Configuration
#=======================================================
HERE = os.path.dirname(os.path.abspath(__file__))

TAG = os.environ.get('OBSERR_TAG', '')   # '' for NT; '_BT' for Bootstrap
PROD = 'BT' if TAG == '_BT' else 'NT'

# Shipped inputs (matchup table + the published region x season and poly-3
# lookups) and the directory this script writes into.  Both overridable: the
# author's runs read the intermediate netCDF tree instead, which is why the
# netCDF loader below is kept.
diff_root = os.environ.get('OBSERR_DIFF_ROOT', os.path.join(HERE, 'data'))
out_root = os.environ.get('OBSERR_OUT_ROOT', os.path.join(HERE, 'output'))

# The flat matchup table shipped with this repository (see export_matchups.py).
# When it exists it is used in preference to the netCDF tree; the two give
# identical numbers, the CSV being an exact round-trip of the same float32
# values in the same row order.
MATCHUP_CSV = os.environ.get(
    'OBSERR_MATCHUPS', os.path.join(diff_root, 'matchups_%s.csv.gz' % PROD))

# A rerun writes its figures beside its tables, so the reference PNGs in
# figures/ are not silently overwritten and can be compared against.
FS.FIGDIR = os.environ.get('OBSERR_FIGDIR', os.path.join(out_root, 'figures'))

REGIONS = ['Barents_Sea', 'Baffin_Bay_Labrador_Seas', 'Beaufort_Sea', 'Bering_Sea',
           'Canadian_Archipelago', 'Central_Arctic', 'Chukchi_Sea', 'East_Greenland_Sea',
           'East_Siberian_Sea', 'Hudson_Bay', 'Kara_Sea', 'Laptev_Sea']

ND_MODE = '8'            # coverage cut: 'k' -> n_distinct>=k ; '=k' -> ==k
SEED = 20260909

# Split by blocks of consecutive days so that autocorrelated neighbours cannot
# straddle the train/test boundary.
BLOCK_DAYS = 7
FRAC = (0.8, 0.1, 0.1)   # train / val / test

# sigma(SIC) refit on the training split (the paper's fitting recipe)
SIG_EDGES = np.arange(0, 105, 5.0)
FIT_MIN_N = 5            # thin bins DO enter the fit, sqrt(N)-weighted
SIG_FLOOR = 0.5          # %, keeps 1/sigma^2 finite
MIN_STRAT = 30           # min matchups for a region x season its own value

# --lookup: take sigma from the PUBLISHED tables instead of refitting it on the
# training split.  This is what a user of the product actually gets, but the
# tables were fitted on every matchup including the test partition, so the
# calibration numbers carry a little test-set leakage.  Run both ways and
# compare: the refit is the clean estimate, the lookup the operational one.
USE_LOOKUP = False

# --bc: prescribe the BIAS-CORRECTED error instead of the total RMSD.  The
# network maps SSMIS onto Landsat, so any reproducible offset that depends on
# the SSMIS value is exactly what it learns away; its residual is close to
# bias-free, and declaring an interval that still contains the offset makes the
# model sound less certain than it is.  The correction follows
# the paper's bias-removal: each 5 % SSMIS-SIC bin of the TRAINING split loses
# its own mean difference, but only where that bin holds >= BC_MIN_N matchups.
# The 'none' and 'const' baselines are untouched.
BIAS_CORR = False
BC_MIN_N = 30


def _table(stem):
    """Published table under diff_root, with or without the product suffix.

    The error-characterisation code (not in this repository) encodes the product
    in the directory and does not always repeat it in the file name, so accept
    either spelling rather than depend on which writer produced the table.
    """
    for name in ('%s%s.csv' % (stem, TAG), '%s.csv' % stem):
        f = os.path.join(diff_root, name)
        if os.path.isfile(f):
            return f
    return os.path.join(diff_root, '%s%s.csv' % (stem, TAG))   # report the miss


LOOK_POLY = _table('obs_error_poly3_lookup')
LOOK_RS = _table('final_obserror_nd4-nd10-nd16')

CONST_SIGMA = 15.0       # the constant-error arm

# MLP
HID = (32, 32)
EPOCHS = 300
BATCH = 512
LR = 3e-3
PATIENCE = 25

# A 2x2 design over what the error is resolved by:
#            season only              season and region
#   no SIC   (subsumed by 'const')    'regseason'   <- one RMSE per stratum (fig11)
#   by SIC   'sic'                    'sicreg'
# 'regseason' is the test that matters for the product: is resolving the error
# by region and season a SUBSTITUTE for resolving it by concentration?
ARMS = ['none', 'const', 'regseason', 'sic', 'sicreg']
ARM_LABEL = {'none': 'no error info', 'const': 'constant 15 %',
             'regseason': '$\\sigma$(season, region)',
             'sic': '$\\sigma$(SIC | season)',
             'sicreg': '$\\sigma$(SIC | season, region)'}
ARM_COLOR = {'none': FS.OI['black'], 'const': FS.OI['green'],
             'regseason': FS.OI['vermillion'], 'sic': FS.OI['blue'],
             'sicreg': FS.OI['purple']}

WCLIP = None             # percentile cap on weights, or None
NSEEDS = 1               # >1 repeats the whole cycle over that many splits

# Ice regimes for the aggregated tables (same cuts as figstyle.REGIME_COLOR).
REGIMES = {'open water <15': lambda c: c < 15,
           'MIZ 15-85': lambda c: (c >= 15) & (c < 85),
           'consolidated >=85': lambda c: c >= 85}


#=======================================================
# Data
#=======================================================
def cut_mask(nd, mode):
    """Boolean mask for a coverage spec: '=k' -> ==k, else 'k' -> >=k."""
    if mode.startswith('='):
        return nd == int(mode[1:])
    return nd >= int(mode)


def cut_label_suffix(mode):
    """(figure label, filename suffix) for a coverage spec."""
    if mode.startswith('='):
        k = int(mode[1:])
        return 'n$_\\mathrm{distinct}$=%d' % k, 'nd%02deq' % k
    k = int(mode)
    return 'n$_\\mathrm{distinct}$$\\geq$%d' % k, 'nd%02d' % k


def _derive(d):
    """Month, day-of-year and season from the YYYYMMDD date column."""
    ds = d['date'].astype(str)
    d['month'] = np.array([int(x[4:6]) for x in ds], dtype=np.int8)
    d['doy'] = np.array([pd.Timestamp('%s-%s-%s' % (x[:4], x[4:6], x[6:8])).dayofyear
                         for x in ds], dtype=np.int16)
    d['season'] = season_arr(d['month'])
    return d


def load():
    """All matchups passing the coverage cut, with date/season/geometry.

    Reads the shipped flat table when it is present, otherwise the author's
    per-region netCDF tree.  Both paths return the same arrays in the same
    order, so either reproduces the published numbers.
    """
    if os.path.isfile(MATCHUP_CSV):
        return load_csv(MATCHUP_CSV)
    return load_netcdf()


def load_csv(f):
    """Matchups from the flat table written by export_matchups.py."""
    t = pd.read_csv(f)
    t = t[cut_mask(t.n_distinct.values, ND_MODE)]
    ri = {r: i for i, r in enumerate(REGIONS)}
    unknown = set(t.region) - set(ri)
    if unknown:
        raise SystemExit('unknown region(s) in %s: %s' % (f, sorted(unknown)))
    # float32 to match the netCDF path exactly -- the table stores 9 significant
    # digits, so the cast is lossless and the two loaders agree bit for bit.
    d = dict(truth=t.landsat.values.astype(np.float32),
             obs=t.ssmis.values.astype(np.float32),
             lat=t.lat.values.astype(np.float32),
             lon=t.lon.values.astype(np.float32),
             date=t.date.values.astype(np.int32),
             reg=np.array([ri[r] for r in t.region], dtype=np.int8))
    return _derive(d)


def load_netcdf():
    """Matchups from the author's per-region diff tree (not distributed)."""
    acc = {k: [] for k in ('truth', 'obs', 'lat', 'lon', 'date', 'reg')}
    for ri, region in enumerate(REGIONS):
        for f in sorted(glob.glob(os.path.join(diff_root, region,
                                               'SIC_diff_ssmis_landsat_*.nc'))):
            nc = Dataset(f)
            if nc.dimensions['ndata'].size == 0:
                nc.close(); continue
            nd = np.asarray(nc.variables['n_distinct'][:])
            m = cut_mask(nd, ND_MODE)
            if not m.any():
                nc.close(); continue
            acc['truth'].append(np.asarray(nc.variables['landsat'][:])[m])
            acc['obs'].append(np.asarray(nc.variables['ssmis'][:])[m])
            acc['lat'].append(np.asarray(nc.variables['lat'][:])[m])
            acc['lon'].append(np.asarray(nc.variables['lon'][:])[m])
            nc.close()
            date = int(os.path.basename(f).rsplit('_', 1)[-1][:8])
            acc['date'].append(np.full(int(m.sum()), date, dtype=np.int32))
            acc['reg'].append(np.full(int(m.sum()), ri, dtype=np.int8))
    if not acc['truth']:
        raise SystemExit('no matchups under %s -- is OBSERR_DIFF_ROOT right, and '
                         'is data/matchups_%s.csv.gz missing?' % (diff_root, PROD))
    return _derive({k: np.concatenate(v) for k, v in acc.items()})


def split_by_date_block(dates, seasons, rng):
    """train/val/test masks, blocked by date and stratified by season.

    Whole blocks of consecutive OBSERVATION dates (not calendar days -- the
    record has gaps) stay together, so autocorrelated neighbours cannot straddle
    the train/test boundary.  The blocks are then dealt out WITHIN each season,
    because Freezing is only 12 % of the record and clustered in Feb-Mar: dealing
    all blocks from one pile left the test split with 1.2 % Freezing, too few to
    report a seasonal result.  A block that straddles a season boundary is
    assigned to whichever season holds most of its matchups.
    """
    uniq = np.unique(dates)
    order = {v: i for i, v in enumerate(uniq)}
    blk_of_date = {v: order[v] // BLOCK_DAYS for v in uniq}
    blk = np.array([blk_of_date[v] for v in dates])

    dominant = {}
    for b in np.unique(blk):
        s, c = np.unique(seasons[blk == b], return_counts=True)
        dominant[int(b)] = s[int(np.argmax(c))]

    which = {}
    for s in SEASON_ORDER:
        grp = np.array([b for b in sorted(dominant) if dominant[b] == s])
        if grp.size == 0:
            continue
        rng.shuffle(grp)
        n1 = int(round(FRAC[0] * grp.size))
        n2 = n1 + int(round(FRAC[1] * grp.size))
        if grp.size >= 3:                  # never leave val or test empty
            n1 = min(n1, grp.size - 2)
            n2 = min(max(n2, n1 + 1), grp.size - 1)
        for i, b in enumerate(grp):
            which[int(b)] = 'train' if i < n1 else 'val' if i < n2 else 'test'
    lab = np.array([which[int(b)] for b in blk])
    return {k: lab == k for k in ('train', 'val', 'test')}


#=======================================================
# sigma(SIC) refit on the training split
#=======================================================
def _clamped_poly(coeffs, lo, hi):
    """Cubic held constant outside the SIC range it was actually fitted over.

    The fit is to 5 % bin CENTRES, so its support ends at 2.5 and 97.5.  Beyond
    that it extrapolates hard: at SIC = 100 all three seasonal fits fall to the
    0.5 % floor, while the measured error of the 95-100 % bin is 2.0-3.3 %.  Two
    thirds of consolidated matchups sit at exactly SIC = 100, so the unclamped
    curve hands them a sigma three times too small and the declared interval
    collapses (95 % coverage fell to 66 % in that bin before clamping).
    """
    p = np.poly1d(coeffs)
    return lambda c: p(np.clip(c, lo, hi))


def _cut_matches(label, mode):
    """Does a CSV cut label denote the coverage cut `mode`?

    Labels differ between tables ('nd>=10', 'n_distinct>=10', ...), so match on
    the number and the >= / == sense rather than the exact string.
    """
    import re
    mm = re.search(r'(>=|==|=)\s*(\d+)', label.replace(' ', ''))
    if not mm:
        return False
    eq = mm.group(1) in ('==', '=')
    k = int(mm.group(2))
    want_eq = mode.startswith('=')
    want_k = int(mode.lstrip('='))
    # n>=16 and n==16 are the same set; treat them as equal
    if k == 16 and want_k == 16:
        return True
    return (eq == want_eq) and (k == want_k)


def lookup_sigma_sic(mode):
    """{season: callable} read from the published poly-3 lookup CSV."""
    t = pd.read_csv(LOOK_POLY)
    t = t[[_cut_matches(c, mode) for c in t.cut]]
    if t.empty:
        raise SystemExit('no cut matching %r in %s' % (mode, LOOK_POLY))
    out = {}
    for s in SEASON_ORDER:
        q = t[t.season == s].sort_values('sic')
        if q.empty:
            out[s] = None
            continue
        xs, ys = q.sic.values.astype(float), q.obs_error.values.astype(float)
        out[s] = (lambda x, xs=xs, ys=ys: np.interp(np.clip(x, xs[0], xs[-1]), xs, ys))
    return out


def lookup_sigma_regseason(mode, pooled):
    """{(region, season): RMSE} read from the published region x season table."""
    t = pd.read_csv(LOOK_RS)
    t = t[[_cut_matches(c, mode) for c in t.cut]]
    if t.empty:
        raise SystemExit('no cut matching %r in %s' % (mode, LOOK_RS))
    allrow = {s: float(t[(t.region == 'ALL') & (t.season == s)].rmsd.iloc[0])
              for s in SEASON_ORDER
              if not t[(t.region == 'ALL') & (t.season == s)].empty}
    out, own = {}, 0
    for ri, rg in enumerate(REGIONS):
        for s in SEASON_ORDER:
            q = t[(t.region == rg) & (t.season == s)]
            if not q.empty and np.isfinite(q.rmsd.iloc[0]):
                out[(ri, s)] = float(q.rmsd.iloc[0]); own += 1
            else:
                out[(ri, s)] = allrow.get(s, pooled)
    return out, own


def bias_correct(obs, err, seasons):
    """Residual with each well-populated 5 % bin's own mean difference removed.

    Per season, as in the paper's bias-removed tables.  Bins below BC_MIN_N are left
    alone: a mean over a handful of matchups is not an estimate of the offset
    there, and subtracting it would simply move those points onto zero.
    """
    out = err.copy()
    nfix = 0
    for ss in SEASON_ORDER:
        sm = seasons == ss
        if not sm.any():
            continue
        idx = np.digitize(obs[sm], SIG_EDGES[1:-1])
        sub = out[sm]
        for b in range(len(SIG_EDGES) - 1):
            q = idx == b
            if q.sum() >= BC_MIN_N:
                sub[q] -= sub[q].mean()
                nfix += 1
        out[sm] = sub
    return out, nfix


def _fit_one(obs, err):
    """Cubic of RMSE vs SSMIS SIC for one stratum -> (callable|None, points).

    Returns None when the stratum cannot support a cubic (fewer than four 5 %
    bins holding FIT_MIN_N matchups); the caller then falls back to a coarser
    stratification rather than fitting noise.
    """
    idx = np.digitize(obs, SIG_EDGES[1:-1])
    xs, ys, ws = [], [], []
    for b in range(len(SIG_EDGES) - 1):
        mm = idx == b
        n = int(mm.sum())
        if n < FIT_MIN_N:
            continue
        xs.append(0.5 * (SIG_EDGES[b] + SIG_EDGES[b + 1]))
        ys.append(np.sqrt(np.mean(err[mm] ** 2)))
        ws.append(np.sqrt(n))
    pts = (np.array(xs), np.array(ys))
    if len(xs) < 4:
        return None, pts
    return _clamped_poly(np.polyfit(np.array(xs), np.array(ys), 3,
                                    w=np.array(ws)), min(xs), max(xs)), pts


def fit_sigma(obs, err, seasons):
    """{season: cubic}, plus the binned points so the figure can show evidence."""
    fits, pts = {}, {}
    for s in SEASON_ORDER:
        m = seasons == s
        if m.sum() < 50:
            fits[s], pts[s] = None, None
            continue
        fits[s], pts[s] = _fit_one(obs[m], err[m])
    return fits, pts


def fit_sigma_regseason(err, seasons, regions, pooled):
    """One scalar RMSE per (region, season) -- the stratification of fig11.

    No concentration dependence at all.  This is the arm that asks whether
    resolving the observation error by region and season can stand in for
    resolving it by concentration.  Strata with fewer than MIN_STRAT matchups
    fall back to the season value, and seasons that thin fall back to pooled.
    """
    sfall = {}
    for s in SEASON_ORDER:
        m = seasons == s
        sfall[s] = float(np.sqrt(np.mean(err[m] ** 2))) if m.sum() >= MIN_STRAT else pooled
    out, own = {}, 0
    for ri in range(len(REGIONS)):
        for s in SEASON_ORDER:
            m = (regions == ri) & (seasons == s)
            if m.sum() >= MIN_STRAT:
                out[(ri, s)] = float(np.sqrt(np.mean(err[m] ** 2)))
                own += 1
            else:
                out[(ri, s)] = sfall[s]
    return out, own


def fit_sigma_region(obs, err, seasons, regions, season_fits):
    """{(region, season): cubic}, falling back to the season curve when a
    region-season is too thin to fit on its own.

    Region x season is 36 strata against ~24 k training matchups, and they are
    very unevenly filled, so most of the value is in the well-sampled ones; the
    fallback keeps the sparse ones from contributing fitted noise.
    """
    fits, own = {}, 0
    for ri in range(len(REGIONS)):
        for s in SEASON_ORDER:
            m = (regions == ri) & (seasons == s)
            f = None
            if m.sum() >= 50:
                f, _ = _fit_one(obs[m], err[m])
            if f is None:
                fits[(ri, s)] = season_fits.get(s)
            else:
                fits[(ri, s)] = f
                own += 1
    return fits, own


def sigma_of(arm, obs, seasons, fits, pooled_rmse, regions=None, rfits=None,
             rsvals=None):
    """Predictive sigma (%) for each sample under one arm."""
    if arm == 'none':
        return np.full(obs.size, pooled_rmse)
    if arm == 'const':
        return np.full(obs.size, CONST_SIGMA)
    out = np.full(obs.size, pooled_rmse)
    if arm == 'regseason':
        for ri in range(len(REGIONS)):
            for s in SEASON_ORDER:
                m = (regions == ri) & (seasons == s)
                v = rsvals.get((ri, s))
                if m.any() and v is not None:
                    out[m] = v
    elif arm == 'sicreg':
        for ri in range(len(REGIONS)):
            for s in SEASON_ORDER:
                m = (regions == ri) & (seasons == s)
                f = rfits.get((ri, s))
                if m.any() and f is not None:
                    out[m] = f(np.clip(obs[m], 0, 100))
    else:
        for s in SEASON_ORDER:
            m = seasons == s
            if not m.any() or fits.get(s) is None:
                continue
            out[m] = fits[s](np.clip(obs[m], 0, 100))
    return np.maximum(out, SIG_FLOOR)


def weights_of(arm, sigma):
    """Loss weights, normalised to mean 1 so a constant sigma == no weighting."""
    if arm in ('none', 'const'):
        return np.ones(sigma.size)
    w = 1.0 / sigma ** 2
    if WCLIP is not None:
        w = np.minimum(w, np.percentile(w, WCLIP))
    return w / w.mean()


#=======================================================
# Minimal MLP with weighted MSE (no torch in this env)
#=======================================================
def _init(D, rng):
    sizes = [D] + list(HID) + [1]
    return [[rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])),
             np.zeros(sizes[i + 1])] for i in range(len(sizes) - 1)]


def _forward(P, X):
    A = [X]
    h = X
    for i, (W, b) in enumerate(P):
        z = h @ W + b
        h = np.tanh(z) if i < len(P) - 1 else z
        A.append(h)
    return A


def _backward(P, A, y, w):
    n = y.size
    g = 2.0 * w[:, None] * (A[-1] - y[:, None]) / n
    grads = [None] * len(P)
    for i in range(len(P) - 1, -1, -1):
        grads[i] = [A[i].T @ g, g.sum(0)]
        if i > 0:
            g = (g @ P[i][0].T) * (1.0 - A[i] ** 2)
    return grads


def train_mlp(Xtr, ytr, wtr, Xva, yva, wva, rng):
    """Weighted-MSE MLP; early stopping on the arm's OWN weighted val loss."""
    P = _init(Xtr.shape[1], rng)
    M = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    V = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best, best_P, bad = np.inf, [[W.copy(), b.copy()] for W, b in P], 0
    n = ytr.size
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            j = perm[s:s + BATCH]
            A = _forward(P, Xtr[j])
            G = _backward(P, A, ytr[j], wtr[j])
            t += 1
            for i in range(len(P)):
                for k in range(2):
                    M[i][k] = b1 * M[i][k] + (1 - b1) * G[i][k]
                    V[i][k] = b2 * V[i][k] + (1 - b2) * G[i][k] ** 2
                    mh = M[i][k] / (1 - b1 ** t)
                    vh = V[i][k] / (1 - b2 ** t)
                    P[i][k] -= LR * mh / (np.sqrt(vh) + eps)
        pv = _forward(P, Xva)[-1].ravel()
        vl = float(np.mean(wva * (pv - yva) ** 2))
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_P = [[W.copy(), b.copy()] for W, b in P]
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_P


def predict(P, X):
    return _forward(P, X)[-1].ravel()


#=======================================================
# Metrics
#=======================================================
def crps_gaussian(y, mu, sig):
    z = (y - mu) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def score(y, mu, sig):
    e = mu - y
    return dict(n=int(y.size),
                rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)),
                crps=float(np.mean(crps_gaussian(y, mu, sig))),
                cov95=float(np.mean(np.abs(e) <= 1.96 * sig) * 100.0),
                nll=float(np.mean(0.5 * (e / sig) ** 2 + np.log(sig))))


def features(d, sel):
    """Same feature set for every arm -- sigma is NEVER a feature (see docstring)."""
    doy = d['doy'][sel].astype(float)
    return np.column_stack([d['obs'][sel] / 100.0,
                            np.sin(2 * np.pi * doy / 365.25),
                            np.cos(2 * np.pi * doy / 365.25),
                            (d['lat'][sel] - 70.0) / 20.0])


#=======================================================
# Experiment
#=======================================================
def experiment(d, seed, verbose=True):
    """One complete split -> sigma fit -> train -> score cycle at a given seed.

    The split, the sigma(SIC) refit and the network initialisation all follow
    from `seed`, so repeating over seeds resamples everything that is random
    while leaving the data and the arms fixed.
    """
    rng = np.random.default_rng(seed)
    sp = split_by_date_block(d['date'], d['season'], rng)
    if verbose:
        for k in ('train', 'val', 'test'):
            comp = ' '.join('%s %4.1f%%' % (s[:4], 100.0 * (d['season'][sp[k]] == s).mean())
                            for s in SEASON_ORDER)
            print('  %-5s %7d cells  %4d days   %s' % (k, sp[k].sum(),
                                                       np.unique(d['date'][sp[k]]).size, comp))
        print('  %-5s %7d cells             %s'
              % ('all', d['truth'].size,
                 ' '.join('%s %4.1f%%' % (s[:4], 100.0 * (d['season'] == s).mean())
                          for s in SEASON_ORDER)))

    err_tr = d['obs'][sp['train']] - d['truth'][sp['train']]
    pooled = float(np.sqrt(np.mean(err_tr ** 2)))
    # the arms under test may be given the bias-corrected error; 'none' and
    # 'const' always keep the total, so the baselines do not move between runs
    err_sig, nfix = (bias_correct(d['obs'][sp['train']], err_tr,
                                  d['season'][sp['train']])
                     if BIAS_CORR else (err_tr, 0))
    pooled_sig = float(np.sqrt(np.mean(err_sig ** 2)))
    fits, sigpts = fit_sigma(d['obs'][sp['train']], err_sig, d['season'][sp['train']])
    rfits, nown = fit_sigma_region(d['obs'][sp['train']], err_sig,
                                   d['season'][sp['train']], d['reg'][sp['train']],
                                   fits)
    rsvals, nrs = fit_sigma_regseason(err_sig, d['season'][sp['train']],
                                      d['reg'][sp['train']], pooled_sig)
    if USE_LOOKUP:
        fits = lookup_sigma_sic(ND_MODE)
        rsvals, nrs = lookup_sigma_regseason(ND_MODE, pooled)
    if verbose:
        ntot = len(REGIONS) * len(SEASON_ORDER)
        print('  sigma source: %s' % ('PUBLISHED lookup tables' if USE_LOOKUP
                                      else 'refitted on the training split'))
        print('  region x season strata: %d of %d have their own scalar, '
              '%d their own SIC curve' % (nrs, ntot, nown))
    if verbose:
        print('  pooled training RMSE = %.2f %%   (constant arm uses %.1f %%)'
              % (pooled, CONST_SIGMA))
        if BIAS_CORR:
            print('  bias-corrected sigma: %d bins corrected, pooled %.2f %% '
                  '(was %.2f %%)' % (nfix, pooled_sig, pooled))

    X = {k: features(d, sp[k]) for k in sp}
    Y = {k: d['truth'][sp[k]].astype(float) for k in sp}
    mx, sx = X['train'].mean(0), X['train'].std(0) + 1e-9
    my, sy = Y['train'].mean(), Y['train'].std()
    Xs = {k: (v - mx) / sx for k, v in X.items()}
    Ys = {k: (v - my) / sy for k, v in Y.items()}

    sig = {k: {a: sigma_of(a, d['obs'][sp[k]], d['season'][sp[k]], fits,
                           pooled if a in ('none', 'const') else pooled_sig,
                           d['reg'][sp[k]], rfits, rsvals)
               for a in ARMS} for k in sp}

    rows, pred = [], {}
    for arm in ARMS:
        wtr = weights_of(arm, sig['train'][arm])
        wva = weights_of(arm, sig['val'][arm])
        P = train_mlp(Xs['train'], Ys['train'], wtr, Xs['val'], Ys['val'], wva,
                      np.random.default_rng(seed))
        pred[arm] = {k: np.clip(predict(P, Xs[k]) * sy + my, 0.0, 100.0)
                     for k in ('val', 'test')}
        mu = pred[arm]['test']
        s = sig['test'][arm]
        rows.append(dict(arm=arm, model='MLP', stratum='all', value='all',
                         **score(Y['test'], mu, s)))
        if verbose:
            print('  %-6s  RMSE %6.2f  bias %+6.2f  CRPS %6.2f  cov95 %5.1f %%'
                  % (arm, rows[-1]['rmse'], rows[-1]['bias'],
                     rows[-1]['crps'], rows[-1]['cov95']))

        idx = np.digitize(d['obs'][sp['test']], SIG_EDGES[1:-1])
        for b in range(len(SIG_EDGES) - 1):
            m = idx == b
            if m.sum() < 30:
                continue
            rows.append(dict(arm=arm, model='MLP', stratum='sic_bin',
                             value='%g-%g' % (SIG_EDGES[b], SIG_EDGES[b + 1]),
                             **score(Y['test'][m], mu[m], s[m])))
        for ss in SEASON_ORDER:
            m = d['season'][sp['test']] == ss
            if m.sum() < 30:
                continue
            rows.append(dict(arm=arm, model='MLP', stratum='season', value=ss,
                             **score(Y['test'][m], mu[m], s[m])))
        for nm, m in REGIMES.items():
            mm = m(d['obs'][sp['test']])
            if mm.sum() < 30:
                continue
            rows.append(dict(arm=arm, model='MLP', stratum='regime', value=nm,
                             **score(Y['test'][mm], mu[mm], s[mm])))

    if '--xgb' in sys.argv:
        rows += run_xgb(d, sp, sig, X, Y)

    same = np.allclose(pred['none']['test'], pred['const']['test'], atol=1e-6)
    if verbose:
        print("  check: 'none' and 'const' predictions identical -> %s" % same)
        if not same:
            print('    (they must be: a uniform weight cannot move the optimum)')
    for r in rows:
        r['seed'] = seed
    return dict(rows=rows, pred=pred, sig=sig, fits=fits, sigpts=sigpts,
                pooled=pooled, sp=sp, Y=Y, same=same)


def run(mode):
    global ND_MODE
    ND_MODE = mode
    cutlabel, suffix = cut_label_suffix(mode)
    if BIAS_CORR:
        suffix += '_bc'
    if WCLIP is not None:                 # keep clipped runs from overwriting the raw one
        suffix += '_wc%02d' % WCLIP

    os.makedirs(out_root, exist_ok=True)
    d = load()
    print('matchups (%s): %d   dates %d' % (cutlabel.replace('$', ''),
                                            d['truth'].size, np.unique(d['date']).size))
    r = experiment(d, SEED)
    rows = r['rows']

    out = os.path.join(out_root, 'ml_experiment_metrics_%s%s.csv' % (suffix, TAG))
    pd.DataFrame(rows).to_csv(out, index=False, float_format='%.4f')
    print('  wrote %s' % out)

    # Repeat first so the skill panels can show the spread over splits rather
    # than one draw; the sigma / interval / scatter panels stay on the first
    # split because they illustrate a single fit, not a distribution.
    seeds = repeat(d, suffix) if NSEEDS > 1 else None
    plot(d, r['sp'], r['sig'], rows, cutlabel, suffix, r['pred'], seeds)
    plot_scatter(r['pred'], r['Y'], suffix)
    plot_crps(d, r['sp'], r['sig'], r['pred'], r['Y'], suffix)
    return rows


def repeat(d, suffix):
    """Re-run the whole cycle over NSEEDS splits and report mean +/- SD.

    A single split can be lucky: with 81 blocks dealt into 8:1:1 the test set is
    only ~8 blocks, so one unusual week moves the numbers.  Repeating resamples
    the split, the sigma refit and the network init together, and the spread
    across seeds is the honest uncertainty on every number quoted.
    """
    allrows = []
    for i in range(NSEEDS):
        sd = SEED + i
        print('  seed %d/%d (%d) ...' % (i + 1, NSEEDS, sd), end='', flush=True)
        r = experiment(d, sd, verbose=False)
        allrows += r['rows']
        a = [x for x in r['rows'] if x['stratum'] == 'all']
        g = {x['arm']: x for x in a}
        print('  CRPS const %.2f -> sic %.2f   cov95 %.1f -> %.1f%s'
              % (g['const']['crps'], g['sic']['crps'],
                 g['const']['cov95'], g['sic']['cov95'],
                 '' if r['same'] else '   [none!=const WARNING]'))
    df = pd.DataFrame(allrows)
    out = os.path.join(out_root, 'ml_experiment_seeds_%s%s.csv' % (suffix, TAG))
    df.to_csv(out, index=False, float_format='%.4f')
    print('  wrote %s  (%d seeds)' % (out, NSEEDS))
    _ = df

    print('\n  mean +/- SD over %d splits (test set)' % NSEEDS)
    print('  %-18s %-6s %14s %14s %14s' % ('stratum', 'arm', 'RMSE', 'CRPS', 'cov95 %'))
    for stratum, val in [('all', 'all')] + [('regime', k) for k in REGIMES] \
            + [('season', k) for k in SEASON_ORDER]:
        sub = df[(df.stratum == stratum) & (df.value == val)]
        if sub.empty:
            continue
        for arm in ARMS:
            q = sub[sub.arm == arm]
            if q.empty:
                continue
            print('  %-18s %-6s %6.2f +/-%-5.2f %6.2f +/-%-5.2f %6.1f +/-%-5.1f'
                  % (val if arm == ARMS[0] else '', arm,
                     q.rmse.mean(), q.rmse.std(), q.crps.mean(), q.crps.std(),
                     q.cov95.mean(), q.cov95.std()))
    return df


def run_xgb(d, sp, sig, X, Y):
    """Same three arms through XGBoost's sample_weight, as a robustness check."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print('  xgboost not available, skipping')
        return []
    rows = []
    for arm in ARMS:
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                         early_stopping_rounds=30, verbosity=0)
        m.fit(X['train'], Y['train'], sample_weight=weights_of(arm, sig['train'][arm]),
              eval_set=[(X['val'], Y['val'])],
              sample_weight_eval_set=[weights_of(arm, sig['val'][arm])], verbose=False)
        mu = np.clip(m.predict(X['test']), 0.0, 100.0)
        rows.append(dict(arm=arm, model='XGB', stratum='all', value='all',
                         **score(Y['test'], mu, sig['test'][arm])))
        print('  XGB %-6s RMSE %6.2f  CRPS %6.2f  cov95 %5.1f %%'
              % (arm, rows[-1]['rmse'], rows[-1]['crps'], rows[-1]['cov95']))
    return rows


#=======================================================
# Figure
#=======================================================
def plot(d, sp, sig, rows, cutlabel, suffix, pred, seeds=None):
    df = pd.DataFrame([r for r in rows if r['model'] == 'MLP'])
    bins = df[df.stratum == 'sic_bin']
    centres = {v: 0.5 * (float(v.split('-')[0]) + float(v.split('-')[1]))
               for v in bins.value.unique()}
    tot = {r['arm']: r for r in rows if r['model'] == 'MLP' and r['stratum'] == 'all'}

    def annot(ax, key, fmt, x=0.03, y=0.97):
        for i, arm in enumerate(('const', 'regseason', 'sic')):
            if seeds is not None:
                q = seeds[(seeds.arm == arm) & (seeds.stratum == 'all')][key]
                txt = (fmt % q.mean()) + ' $\\pm$%.2f' % q.std()
            elif arm in tot:
                txt = fmt % tot[arm][key]
            else:
                continue
            ax.text(x, y - 0.08 * i, txt, transform=ax.transAxes,
                    color=ARM_COLOR[arm], fontsize=6.5, va='top', ha='left')

    def by_bin(arm, key):
        """(centre, mean, sd) per SIC bin across seeds, or the single split."""
        if seeds is None:
            sub = bins[bins.arm == arm].copy()
            if sub.empty:
                return None
            sub['c'] = sub.value.map(centres)
            sub = sub.sort_values('c')
            return sub.c.values, sub[key].values, None
        q = seeds[(seeds.arm == arm) & (seeds.stratum == 'sic_bin')]
        if q.empty:
            return None
        g = q.groupby('value')[key].agg(['mean', 'std', 'count'])
        g = g[g['count'] >= max(2, NSEEDS // 2)]      # bin must survive most splits
        if g.empty:
            return None
        g = g.assign(c=[0.5 * (float(v.split('-')[0]) + float(v.split('-')[1]))
                        for v in g.index]).sort_values('c')
        return g.c.values, g['mean'].values, g['std'].values

    # The sigma curves themselves are not drawn here -- they are the product,
    # shown in their own figure; this one is about what the specifications DO.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    a2, a3 = axes[0]
    a4, a5 = axes[1]

    # (a) the interval each arm actually DECLARES, drawn against the real spread
    # of the truth.  This is what panel (d) then scores: a constant sigma is one
    # fixed-width band everywhere, far too narrow in the MIZ and so wide in
    # consolidated ice that it runs off the physical 0-100 % range.
    obs_t, truth_t = d['obs'][sp['test']], d['truth'][sp['test']]
    a2.plot(obs_t, truth_t, '.', ms=1.0, alpha=0.18, color='0.45', mec='none',
            zorder=1, label='Landsat truth')
    ib = np.digitize(obs_t, SIG_EDGES[1:-1])
    cc = 0.5 * (SIG_EDGES[:-1] + SIG_EDGES[1:])
    for arm in ('const', 'regseason', 'sic'):
        xs, lo, hi = [], [], []
        for b in range(len(SIG_EDGES) - 1):
            m = ib == b
            if m.sum() < 30:
                continue
            mu_b, sg_b = pred[arm]['test'][m].mean(), sig['test'][arm][m].mean()
            xs.append(cc[b]); lo.append(mu_b - 1.96 * sg_b); hi.append(mu_b + 1.96 * sg_b)
        a2.fill_between(xs, lo, hi, color=ARM_COLOR[arm], alpha=0.18, lw=0, zorder=2)
        a2.plot(xs, hi, color=ARM_COLOR[arm], lw=1.0, zorder=3, label=ARM_LABEL[arm])
        a2.plot(xs, lo, color=ARM_COLOR[arm], lw=1.0, zorder=3)
    for yy in (0, 100):
        a2.axhline(yy, color='0.3', lw=0.6, ls=':', zorder=0)
    a2.set_ylim(-48, 148)
    a2.set_xlabel('SSMIS SIC (%)'); a2.set_ylabel('SIC (%)')
    a2.legend(loc='lower right', fontsize=5.5, markerscale=6)
    a2.text(0.03, 0.97, 'declared 95 % interval\n($\\mu\\pm$1.96$\\sigma$)',
            transform=a2.transAxes, fontsize=6.5, va='top', color='0.35')

    # (b)-(d) skill of the resulting model, by SIC bin.  'none' is drawn thick
    # underneath and 'const' dashed on top: they coincide exactly, by construction.
    style = {'none': dict(lw=2.6, alpha=0.45, marker=None),
             'const': dict(lw=1.1, ls='--', marker='o', ms=2.5),
             'regseason': dict(lw=1.2, ls=':', marker='s', ms=2.5),
             'sic': dict(lw=1.3, marker='o', ms=2.8),
             'sicreg': dict(lw=1.0, ls='-.', marker='^', ms=2.2, alpha=0.75)}
    for key, ax, lab in (('rmse', a3, 'RMSE vs Landsat (%)'),
                         ('cov95', a4, '95 % interval coverage (%)'),
                         ('crps', a5, 'CRPS (%)')):
        for arm in ARMS:
            got = by_bin(arm, key)
            if got is None:
                continue
            c, mu_b, sd_b = got
            if sd_b is not None and arm != 'none':   # 'none' coincides with const
                ax.fill_between(c, mu_b - sd_b, mu_b + sd_b, color=ARM_COLOR[arm],
                                alpha=0.15, lw=0, zorder=1)
            ax.plot(c, mu_b, color=ARM_COLOR[arm], label=ARM_LABEL[arm], zorder=2,
                    **style[arm])
        ax.set_xlabel('SSMIS SIC (%)'); ax.set_ylabel(lab)

    a4.axvspan(15, 85, color='0.92', zorder=0)
    a4.axhline(95, color='0.35', lw=0.9, ls='--')
    a4.text(50, 101, 'MIZ', ha='center', fontsize=6.5, color='0.35')
    if seeds is not None:
        a5.text(0.97, 0.97, 'mean $\\pm$1 SD\nover %d splits' % NSEEDS,
                transform=a5.transAxes, ha='right', va='top', fontsize=6, color='0.35')
    a3.legend(loc='upper left', fontsize=5.5)
    annot(a3, 'rmse', 'pooled %.2f', x=0.58)   # upper-left is taken by the legend
    annot(a4, 'cov95', 'pooled %.1f %%', y=0.30)
    annot(a5, 'crps', 'pooled %.2f')

    for ax, L in zip((a2, a3, a4, a5), 'abcd'):
        FS.panel(ax, L)
    fig.tight_layout()
    FS.save(fig, 'fig34_ml_experiment_%s%s' % (suffix, TAG))


def plot_scatter(pred, Y, suffix):
    """1:1 density scatter of prediction against Landsat truth, val and test.

    Every arm is shown, but their mu is the same fit to within the reweighting,
    so these panels are a check that the model itself is sane (no saturation, no
    collapse to the mean) rather than a comparison of the error sets -- that is
    what the calibration panels of fig34 are for.  Seeing the three side by side
    is the point: they are indistinguishable, which is the result.
    """
    # Every arm that was actually run, in the order they are discussed, so the
    # region-and-season control sits between the constant and the curve.
    arms = [a for a in ('const', 'regseason', 'sic') if a in pred]
    splits = ('val', 'test')
    edges = np.linspace(0, 100, 51)
    H = {(sl, arm): np.histogram2d(Y[sl], pred[arm][sl], bins=[edges, edges])[0]
         for sl in splits for arm in arms}
    vmax = max(h.max() for h in H.values())

    fig, axes = plt.subplots(2, len(arms), figsize=(3.1 * len(arms), 6.0),
                             sharex=True, sharey=True, squeeze=False)
    for i, sl in enumerate(splits):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            pcm = ax.pcolormesh(edges, edges, np.ma.masked_less(H[(sl, arm)].T, 1),
                                cmap=FS.CMAP_CNT, norm=LogNorm(vmin=1, vmax=vmax))
            ax.plot([0, 100], [0, 100], color='0.25', lw=0.9, ls='--', zorder=3)
            y, mu = Y[sl], pred[arm][sl]
            e = mu - y
            ax.text(0.04, 0.96, 'N %d\nRMSE %.2f\nbias %+.2f\nR %.3f'
                    % (y.size, np.sqrt(np.mean(e ** 2)), e.mean(),
                       np.corrcoef(y, mu)[0, 1]),
                    transform=ax.transAxes, va='top', ha='left', fontsize=6.5)
            ax.text(0.96, 0.06, ARM_LABEL[arm], transform=ax.transAxes,
                    va='bottom', ha='right', fontsize=6.5, color=ARM_COLOR[arm])
            ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
            if i == len(splits) - 1:
                ax.set_xlabel('Landsat SIC (truth, %)')
            if j == 0:
                ax.set_ylabel('%s set\npredicted SIC (%%)'
                              % ('validation' if sl == 'val' else 'test'))
    for ax, L in zip(axes.ravel(), 'abcdef'):
        FS.panel(ax, L)
    fig.colorbar(pcm, ax=axes, shrink=0.8, pad=0.02).set_label('data pairs per cell')
    FS.save(fig, 'fig35_ml_scatter_%s%s' % (suffix, TAG))


def plot_crps(d, sp, sig, pred, Y, suffix):
    """What CRPS actually measures, drawn.

    Top row: one representative test cell per regime.  The truth is the step
    function that jumps from 0 to 1 at the Landsat value; each arm predicts a
    Gaussian CDF.  CRPS is the integral of the SQUARED gap between them, so the
    shaded area is the region being scored.
    Bottom row: the same comparison in aggregate -- the empirical CDF of the
    model error (reality) against the N(0, sigma) each arm declares.  mu is
    effectively identical between arms, so there is one reality and two claims.
    """
    arms = ('const', 'regseason', 'sic')
    obs_t, y = d['obs'][sp['test']], Y['test']
    cases = [('MIZ  (SSMIS 45-55 %)', (obs_t >= 45) & (obs_t < 55)),
             ('consolidated  (SSMIS $\\geq$ 95 %)', obs_t >= 95)]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for j, (lab, m) in enumerate(cases):
        # representative cell: the one whose error is the median error of the regime
        e_all = pred['sic']['test'][m] - y[m]
        k = int(np.argmin(np.abs(np.abs(e_all) - np.median(np.abs(e_all)))))
        idx = int(np.where(m)[0][k])
        yy = y[idx]

        ax = axes[0, j]
        x = np.linspace(-45, 145, 761)
        step = (x >= yy).astype(float)
        for arm in arms:
            mu, sg = pred[arm]['test'][idx], sig['test'][arm][idx]
            F = norm.cdf((x - mu) / sg)
            ax.fill_between(x, F, step, color=ARM_COLOR[arm], alpha=0.16, lw=0)
            ax.plot(x, F, color=ARM_COLOR[arm], lw=1.3,
                    label='%s  ($\\sigma$=%.1f, CRPS %.1f)'
                          % (ARM_LABEL[arm], sg, crps_gaussian(yy, mu, sg)))
        ax.plot(x, step, color='0.15', lw=1.5, label='truth: step at %.0f %%' % yy)
        ax.set_xlim(-45, 145); ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel('SIC (%)'); ax.set_ylabel('cumulative probability')
        ax.legend(loc='upper left', fontsize=5.5)
        ax.text(0.97, 0.05, lab, transform=ax.transAxes, ha='right', fontsize=6.5,
                color='0.35')

        ax = axes[1, j]
        emp = np.sort(pred['sic']['test'][m] - y[m])
        lim = float(np.percentile(np.abs(emp), 99)) * 1.35 + 1
        xe = np.linspace(-lim, lim, 401)
        ax.plot(emp, np.arange(1, emp.size + 1) / emp.size, color='0.15', lw=1.5,
                label='empirical error (reality, N=%d)' % emp.size)
        for arm in arms:
            sgm = float(sig['test'][arm][m].mean())
            ax.plot(xe, norm.cdf(xe / sgm), color=ARM_COLOR[arm], lw=1.2, ls='--',
                    label='declared N(0, %.1f)' % sgm)
        ax.set_xlim(-lim, lim); ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel('model error, prediction $-$ truth (%)')
        ax.set_ylabel('cumulative probability')
        ax.legend(loc='upper left', fontsize=5.5)

    for ax, L in zip(axes.ravel(), 'abcd'):
        FS.panel(ax, L)
    fig.tight_layout()
    FS.save(fig, 'fig36_ml_crps_%s%s' % (suffix, TAG))


#=======================================================
def main():
    global WCLIP, NSEEDS, USE_LOOKUP, ARMS, BIAS_CORR
    argv = sys.argv[1:]
    if '--bc' in argv:
        BIAS_CORR = True
        argv = [a for a in argv if a != '--bc']
        print('sigma is the BIAS-CORRECTED error (bins with N >= %d); '
              "the 'none' and 'const' baselines keep the total RMSD" % BC_MIN_N)
    if '--lookup' in argv:
        USE_LOOKUP = True
        argv = [a for a in argv if a != '--lookup']
        # no published table resolves the SIC curve by region, so that arm
        # cannot be run from the lookup and is dropped here
        ARMS = [a for a in ARMS if a != 'sicreg']
        print('sigma taken from the published lookup tables '
              '(drops the sicreg arm; carries test-set leakage)')
    for flag in ('--wclip', '--seeds'):          # drop the VALUE too, or it
        if flag in argv:                         # would read as a cut spec
            i = argv.index(flag)
            v = float(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
            if flag == '--wclip':
                WCLIP = v
                print('weights capped at p%.0f' % WCLIP)
            else:
                NSEEDS = int(v)
                print('repeating over %d splits' % NSEEDS)
    specs = [a for a in argv if a.lstrip('=').isdigit()] or [ND_MODE]
    for m in specs:
        print('\n=== coverage cut %s ===' % m)
        run(m)


if __name__ == '__main__':
    main()
