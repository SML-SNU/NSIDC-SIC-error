#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schematic of the network used in the Discussion experiment (33_).

The figure exists to make one point that a parameter count cannot: the
prescribed observation error sigma does NOT enter the network.  It enters the
loss, as a per-sample weight, and the predictive distribution that is scored.
Every arm of the experiment shares this architecture exactly; they differ only
in the sigma that sets those weights.

Output (to figures/)
--------------------
  fig45_mlp_schematic

Usage
-----
    conda activate python_js
    python mlp_schematic.py

@author: kimjs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

import figstyle as FS

FS.apply()

#=======================================================
IN_LABELS = ['SSMIS SIC', r'sin(2$\pi$d/365.25)', r'cos(2$\pi$d/365.25)',
             'latitude']
SHOW_HID = 5             # hidden nodes drawn; the rest stand behind an ellipsis
HID_N = 32
XCOL = [0.0, 1.0, 2.0, 3.0]
NODE_R = 0.055
EDGE = dict(color='0.75', lw=0.4, zorder=1)


def column_y(n, span=1.5):
    return np.linspace(span / 2, -span / 2, n) if n > 1 else np.array([0.0])


def draw_nodes(ax, x, ys, fc, ec):
    for y in ys:
        ax.add_patch(Circle((x, y), NODE_R, fc=fc, ec=ec, lw=0.8, zorder=3))


def connect(ax, x0, y0s, x1, y1s):
    for a in y0s:
        for b in y1s:
            ax.plot([x0 + NODE_R, x1 - NODE_R], [a, b], **EDGE)


def main():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    y_in = column_y(len(IN_LABELS), 1.3)
    y_h = column_y(SHOW_HID, 1.6)
    y_out = column_y(1)

    connect(ax, XCOL[0], y_in, XCOL[1], y_h)
    connect(ax, XCOL[1], y_h, XCOL[2], y_h)
    connect(ax, XCOL[2], y_h, XCOL[3], y_out)

    draw_nodes(ax, XCOL[0], y_in, 'white', FS.OI['blue'])
    draw_nodes(ax, XCOL[1], y_h, 'white', '0.35')
    draw_nodes(ax, XCOL[2], y_h, 'white', '0.35')
    draw_nodes(ax, XCOL[3], y_out, 'white', FS.OI['vermillion'])

    for x in (XCOL[1], XCOL[2]):                    # the nodes not drawn
        ax.text(x, -1.02, r'$\vdots$', ha='center', va='center', fontsize=11,
                color='0.35')

    for y, lab in zip(y_in, IN_LABELS):
        ax.text(XCOL[0] - 0.13, y, lab, ha='right', va='center', fontsize=7.5)
    ax.text(XCOL[3] + 0.13, 0, r'$\mu$   predicted SIC', ha='left', va='center',
            fontsize=8, color=FS.OI['vermillion'])

    heads = [(XCOL[0], 'input\n4'), (XCOL[1], 'hidden\n%d  tanh' % HID_N),
             (XCOL[2], 'hidden\n%d  tanh' % HID_N), (XCOL[3], 'output\n1  linear')]
    for x, lab in heads:
        ax.text(x, 1.12, lab, ha='center', va='bottom', fontsize=8,
                fontweight='bold', linespacing=1.4)

    for x0, x1, n in ((XCOL[0], XCOL[1], 160), (XCOL[1], XCOL[2], 1056),
                      (XCOL[2], XCOL[3], 33)):
        ax.text((x0 + x1) / 2, -1.30, '%s parameters' % '{:,}'.format(n),
                ha='center', va='top', fontsize=7, color='0.4')
    ax.text(1.5, -1.52, 'total 1,249 learned parameters (connection weights and biases)',
            ha='center', va='top', fontsize=7.5, color='0.25')

    # --- the loss, and where sigma enters -------------------------------
    box = FancyBboxPatch((1.02, -2.62), 1.96, 0.52, boxstyle='round,pad=0.03',
                         fc='#f4f4f4', ec='0.6', lw=0.7, zorder=2)
    ax.add_patch(box)
    ax.text(2.0, -2.36, r'loss  $L=\sum_i w_i\,(y_i-\mu_i)^2$ ,'
                        r'    $w_i=\sigma_i^{-2}$',
            ha='center', va='center', fontsize=8.5)

    ax.add_patch(FancyArrowPatch((XCOL[3], -NODE_R - 0.03), (2.6, -2.08),
                                 arrowstyle='-|>', mutation_scale=9,
                                 color=FS.OI['vermillion'], lw=0.9, zorder=4,
                                 connectionstyle='arc3,rad=-0.25'))
    ax.text(3.05, -2.36, r'$y$  Landsat-8', ha='left', va='center', fontsize=8)

    ax.add_patch(FancyArrowPatch((0.30, -2.36), (1.0, -2.36),
                                 arrowstyle='-|>', mutation_scale=9,
                                 color=FS.OI['blue'], lw=1.0, zorder=4))
    ax.text(0.26, -2.36, r'$\sigma$(SIC, season)', ha='right', va='center',
            fontsize=8.5, color=FS.OI['blue'])
    ax.text(0.26, -2.60, 'prescribed, not learned', ha='right', va='center',
            fontsize=6.8, color=FS.OI['blue'])

    # The point of the figure: sigma does not reach the input column.  The note
    # sits to the left of the inputs rather than above them, so it cannot cover
    # the column headings.
    ax.annotate('', xy=(-0.78, 0.0), xytext=(-1.28, 0.0),
                arrowprops=dict(arrowstyle='-|>', color='0.6', lw=0.9, ls=':',
                                shrinkA=0, shrinkB=0))
    ax.text(-1.03, 0.0, r'$\times$', ha='center', va='center', fontsize=15,
            color=FS.OI['vermillion'], fontweight='bold', zorder=5)
    ax.text(-1.33, 0.0, r'$\sigma$ is never' '\n' 'an input: it is a' '\n'
            'function of the SIC' '\n' 'that is already one',
            ha='right', va='center', fontsize=7, color='0.4', linespacing=1.5,
            bbox=dict(fc='white', ec='0.8', lw=0.5, pad=3))

    ax.set_xlim(-2.75, 3.95)
    ax.set_ylim(-2.95, 1.55)
    ax.axis('off')
    fig.tight_layout()
    FS.save(fig, 'fig45_mlp_schematic')


if __name__ == '__main__':
    main()
