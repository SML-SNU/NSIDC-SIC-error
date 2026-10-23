#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The manuscript's season definition, in one place.

    Freezing   Nov Dec Jan Feb Mar
    Melting    Jun Jul Aug Sep
    Interim    Apr May Oct

Every analysis script imports from here rather than carrying its own copy, so a
change to the convention cannot leave one script disagreeing with another.

A caveat that belongs with the definition
-----------------------------------------
Landsat-8 is optical, so the polar night cannot be observed and the seasons are
NOT evenly sampled by the matchups:

    Freezing   98 % of the data are Feb-Mar (Dec has none at all)
    Interim    97 % are Apr-May (Oct contributes ~2 %)
    Melting    reasonably spread: Jun 42 %, Jul 33 %, Aug 20 %, Sep 5 %

So a "Freezing" statistic is really a late-winter one.  `composition()` returns
the actual month breakdown so any table or caption can state it.
"""

import numpy as np

SEASONS = [('Freezing', (11, 12, 1, 2, 3)),
           ('Melting', (6, 7, 8, 9)),
           ('Interim', (4, 5, 10))]
SEASON_ORDER = [s for s, _ in SEASONS]
SEASON_MONTHS = dict(SEASONS)

# Months in which matchups actually exist (Feb-Oct); used for month axes.
MONTHS_WITH_DATA = list(range(2, 11))
MON_ABBR = ['', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
MON_NAME = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Short label including the months that carry the data -- use in captions.
SEASON_LABEL = {
    'Freezing': 'Freezing (Nov-Mar; data 98 % Feb-Mar)',
    'Melting': 'Melting (Jun-Sep)',
    'Interim': 'Interim (Apr, May, Oct; data 97 % Apr-May)',
}


def season_of(month):
    """Month number -> season name, or None."""
    for name, mm in SEASONS:
        if month in mm:
            return name
    return None


def season_arr(month):
    """Vectorised: array of month numbers -> array of season names."""
    out = np.empty(np.size(month), dtype='<U8')
    out[:] = ''
    for name, mm in SEASONS:
        out[np.isin(month, mm)] = name
    return out


def composition(month, mask=None):
    """{season: [(month_number, count, pct_of_season), ...]} for the given data."""
    month = np.asarray(month)
    if mask is not None:
        month = month[mask]
    sa = season_arr(month)
    out = {}
    for name, mm in SEASONS:
        tot = int((sa == name).sum())
        out[name] = [(x, int((month == x).sum()),
                      100.0 * int((month == x).sum()) / max(tot, 1)) for x in mm]
    return out
