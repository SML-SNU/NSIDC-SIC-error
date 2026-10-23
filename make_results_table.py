#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The manuscript's MLP performance table, with the bias-removed observation
error added as a fourth row in each block.

The Discussion experiment (mlp_experiment.py) is run twice at the same
coverage cut, the same 20 date-block splits and the same weight cap:

    python mlp_experiment.py 10 --wclip 50          -> ..._wc50
    python mlp_experiment.py 10 --wclip 50 --bc     -> ..._bc_wc50

The second run prescribes sigma from the **bias-removed** RMSD (each 5 % SSMIS
SIC bin's own offset taken out where the bin holds >= 30 matchups).  The
`none` and `const` arms are deliberately identical between the two runs -- they
never read the lookup -- which is the control: any change in the sigma arms is
the bias removal and nothing else.

This script only tabulates; it runs no model.  Mean +/- SD across the 20
splits, which is what the table in the manuscript reports.

Outputs (output/)
-----------------
  ml_results_table_nd<k><TAG>.csv   the table, long form, and the same thing
                                    printed to stdout

Usage
-----
    conda activate python_js
    python make_results_table.py            # NT and BT, n_distinct >= 10
    python make_results_table.py 10 BT      # one product only
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Where mlp_experiment.py writes, then where this repository's reference copies
# live.  Searched in that order, so a fresh run is tabulated if one exists and
# the shipped numbers are reproduced if not.
SEARCH = [os.environ.get('OBSERR_OUT_ROOT', os.path.join(HERE, 'output')),
          os.path.join(HERE, 'results')]
TAG = {'NT': '', 'BT': '_BT'}
K_DEFAULT = 10

# (row label, arm, which run).  Order follows the manuscript table; the
# bias-removed row is appended so the three published rows keep their place.
ROWS = [('Constant 15 %',                   'const',     'plain'),
        ('σ(season, region)',          'regseason', 'plain'),
        ('σ(season, SIC)',             'sic',       'plain'),
        ('σ(season, SIC), bias-removed', 'sic',     'bc')]

# (row label in the table, stratum column, value)
STRATA = [('SIC < 15 %',          'regime', 'open water <15'),
          ('15 % < SIC < 85 %',   'regime', 'MIZ 15-85'),
          ('SIC ≥ 85 %',     'regime', 'consolidated >=85'),
          ('All data',            'all',    'all')]


def seeds(alg, k, run):
    name = ('ml_experiment_seeds_nd%02d%s_wc50%s.csv'
            % (k, '_bc' if run == 'bc' else '', TAG[alg]))
    for root in SEARCH:
        f = os.path.join(root, name)
        if os.path.isfile(f):
            print('  reading %s' % f)
            return pd.read_csv(f)
    raise SystemExit('%s found in none of %s -- run\n    %spython '
                     'mlp_experiment.py %d --wclip 50%s --seeds 20'
                     % (name, SEARCH, 'OBSERR_TAG=_BT ' if alg == 'BT' else '',
                        k, ' --bc' if run == 'bc' else ''))


def build(alg, k):
    d = {r: seeds(alg, k, r) for r in ('plain', 'bc')}
    out = []
    for slabel, stratum, value in STRATA:
        for rlabel, arm, run in ROWS:
            q = d[run]
            q = q[(q.stratum == stratum) & (q.value == value) & (q.arm == arm)]
            if q.empty:
                raise SystemExit('no rows for %s / %s / %s' % (run, arm, value))
            row = dict(alg=alg, sic_range=slabel, obs_error=rlabel,
                       n=int(round(q.n.mean())), n_splits=len(q))
            for m, col in (('rmse', 'rmse'), ('crps', 'crps'), ('cov95', 'cov95')):
                # SD across splits, the spread the manuscript's +/- reports
                row[m] = float(q[col].mean())
                row['%s_sd' % m] = float(q[col].std(ddof=1))
            out.append(row)
    return pd.DataFrame(out)


def show(t, alg, k):
    print('\n=== %s, n_distinct >= %d, 20 date-block splits, weight cap p50 ==='
          % (alg, k))
    print('%-26s %-32s %14s %14s %14s'
          % ('SIC range (N)', 'Observation error', 'RMSE (%)', 'CRPS (%)',
             'Coverage (%)'))
    last = None
    for _, r in t.iterrows():
        head = '%s (%s)' % (r.sic_range, format(r.n, ',')) \
            if r.sic_range != last else ''
        last = r.sic_range
        print('%-26s %-32s %6.2f +/-%-6.2f %6.2f +/-%-6.2f %6.2f +/-%-6.2f'
              % (head, r.obs_error, r.rmse, r.rmse_sd, r.crps, r.crps_sd,
                 r.cov95, r.cov95_sd))


def main():
    k, algs = K_DEFAULT, ['NT', 'BT']
    for a in sys.argv[1:]:
        if a.isdigit():
            k = int(a)
        elif a.upper() in TAG:
            algs = [a.upper()]
    os.makedirs(SEARCH[0], exist_ok=True)
    for alg in algs:
        t = build(alg, k)
        show(t, alg, k)
        f = os.path.join(SEARCH[0], 'ml_results_table_nd%02d%s.csv'
                         % (k, TAG[alg]))
        t.round(4).to_csv(f, index=False)
        print('  wrote %s' % f)


if __name__ == '__main__':
    main()
