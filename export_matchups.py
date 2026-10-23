#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the SSMIS/Landsat-8 matchups from the author's per-region netCDF tree
into the single flat table that this repository ships.

This script is provenance, not part of the published workflow: it needs the
intermediate `diff` tree, which is not distributed (46 MB per product, and the
Landsat granules behind it are far larger).  `data/matchups_NT.csv.gz` and
`data/matchups_BT.csv.gz` are its output, and `mlp_experiment.py` reads those,
so nobody else needs to run this.  It is here so that the provenance of the
shipped table is auditable rather than asserted.

One row per 25 km grid cell per day on which a downscaled Landsat-8 SIC and an
SSMIS SIC both exist.  No coverage cut is applied -- `n_distinct` is carried so
that any cut in the paper can be reproduced, the headline one being
`n_distinct >= 10`.

Columns
-------
  date        YYYYMMDD, the Landsat acquisition day
  region      one of the 12 Arctic sub-regions
  lat, lon    cell centre, degrees
  ssmis       SSMIS SIC, %  (NSIDC-0051 v2.0 for NT, NSIDC-0079 v4.0 for BT)
  landsat     Landsat-8 SIC block-averaged onto the same cell, %
  n_distinct  distinct native 6.25 km Landsat pixels in the cell, 1..16

Written with 9 significant digits so the float32 values round-trip exactly and a
rerun reproduces the shipped reference numbers bit for bit.

Usage
-----
    conda activate python_js
    python export_matchups.py            # both products
    python export_matchups.py NT         # one product
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data')

# The author's intermediate tree, written by 02_diff_SSMIS_LANDSAT.py.
ROOT = {'NT': '/home/kimjs/DATA/SAT/LANDSAT/SIC/diff',
        'BT': '/home/kimjs/DATA/SAT/LANDSAT/SIC/diff_BT'}

# Only these directories are read.  The tree also holds a `daily/` directory
# that repeats every cell in a combined-per-day file; globbing the tree instead
# of the region list silently doubles the sample.
REGIONS = ['Barents_Sea', 'Baffin_Bay_Labrador_Seas', 'Beaufort_Sea', 'Bering_Sea',
           'Canadian_Archipelago', 'Central_Arctic', 'Chukchi_Sea', 'East_Greenland_Sea',
           'East_Siberian_Sea', 'Hudson_Bay', 'Kara_Sea', 'Laptev_Sea']

VARS = ['lat', 'lon', 'ssmis', 'landsat', 'n_distinct']


def export(prod):
    rows = []
    for region in REGIONS:
        pat = os.path.join(ROOT[prod], region, 'SIC_diff_ssmis_landsat_*.nc')
        for f in sorted(glob.glob(pat)):
            nc = Dataset(f)
            n = nc.dimensions['ndata'].size
            if n == 0:
                nc.close()
                continue
            d = {v: np.asarray(nc.variables[v][:]) for v in VARS}
            nc.close()
            d['date'] = np.full(n, int(os.path.basename(f).rsplit('_', 1)[-1][:8]),
                                dtype=np.int32)
            d['region'] = np.full(n, region)
            rows.append(pd.DataFrame(d))

    t = pd.concat(rows, ignore_index=True)
    t = t[['date', 'region', 'lat', 'lon', 'ssmis', 'landsat', 'n_distinct']]
    # NOT sorted.  The loop order above -- regions in REGIONS order, dates
    # ascending within a region -- is the order the netCDF loader produced, and
    # the experiment shuffles a seeded permutation of row indices.  Re-sorting
    # here would reproduce every statistic but not the shipped numbers.

    f = os.path.join(OUT, 'matchups_%s.csv.gz' % prod)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(f, index=False, float_format='%.9g', compression='gzip')
    print('%s: %d matchups (%d at n_distinct >= 10) -> %s'
          % (prod, len(t), int((t.n_distinct >= 10).sum()), f))


def main():
    prods = [a.upper() for a in sys.argv[1:] if a.upper() in ROOT] or ['NT', 'BT']
    for p in prods:
        export(p)


if __name__ == '__main__':
    main()
