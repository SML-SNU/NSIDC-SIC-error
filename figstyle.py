#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared figure style for the manuscript.

Every paper figure imports this so type sizes, colours and output conventions
stay identical across figures -- mismatched fonts between panels is the single
most common reason a figure set looks unprofessional.

Conventions
-----------
* No figure titles.  Journals put that text in the caption; a title inside the
  axes duplicates it and wastes vertical space.  Panels get (a), (b), (c) tags.
* Type: 8 pt labels, 7 pt ticks, 7 pt legends.  Sized for a figure reproduced at
  its natural width in a two-column layout without further shrinking.
* Colours are colour-blind safe (Okabe-Ito) and FIXED per season / ice regime,
  so the same colour means the same thing in every figure.
* Saved as 300 dpi PNG only (author preference; see save()).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figures land in `figures/` beside this module.  Overridable, because the
# manuscript build writes into the paper's own tree.
FIGDIR = os.environ.get('OBSERR_FIGDIR',
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'figures'))

# Okabe-Ito colour-blind-safe palette
OI = dict(black='#000000', orange='#E69F00', skyblue='#56B4E9', green='#009E73',
          yellow='#F0E442', blue='#0072B2', vermillion='#D55E00', purple='#CC79A7')

# Manuscript seasons (see seasons.py). Fixed so a colour means the same thing
# in every figure.
SEASON_COLOR = {'Freezing': OI['blue'], 'Melting': OI['vermillion'],
                'Interim': OI['green']}
# Ice regimes are ORDERED by concentration, so the palette is an ordered viridis
# ramp rather than four categorical hues.  The previous Okabe-Ito assignment put
# open water at L*=69.8 and pack ice at L*=70.6 -- a lightness difference of 0.8,
# i.e. the same grey once printed in black and white.  This ramp keeps a minimum
# lightness gap of 18 and a worst-case colour-blind separation of dE=23.
REGIME_COLOR = {'open water <15': '#472a7a', 'MIZ 15-85': '#287d8e',
                'pack 85-95': '#3bbb75', 'consolidated >=95': '#d2e21b'}

CMAP_SIC = 'viridis'
CMAP_DIFF = 'RdBu_r'
CMAP_CNT = 'plasma'


def apply():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.2,
        'lines.markersize': 3.5,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,      # embed as TrueType, not Type-3: editable + accepted
        'ps.fonttype': 42,
    })


def panel(ax, letter, x=-0.02, y=1.04, weight='bold'):
    """(a), (b), ... tag in the axes' upper-left, outside the frame."""
    ax.text(x, y, '(%s)' % letter, transform=ax.transAxes,
            fontsize=9, fontweight=weight, va='bottom', ha='right')


def save(fig, name):
    """Save the figure as PNG only (user preference: no PDFs)."""
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, '%s.png' % name)
    fig.savefig(p)
    plt.close(fig)
    print('  saved %s.png' % name)
    return [p]
