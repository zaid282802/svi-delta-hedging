import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

# color palette — matches handout_v2.tex
NAVY      = '#1B2A4A'
TEAL      = '#2E8B8B'
GOLD      = '#D4A843'
POS_GREEN = '#2E7D32'
NEG_RED   = '#C62828'
LIGHT_GRAY = '#F5F5F5'
WHITE     = '#FFFFFF'
TEXT_DARK = '#1C2833'
TEXT_SEC  = '#5D6D7E'
GRID_CLR  = '#E5E8ED'
BORDER_CLR = '#D5D8DC'

SANS = 'DejaVu Sans'
MONO = 'DejaVu Sans Mono'


def apply_style(ax, title=None):
    ax.set_facecolor(WHITE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER_CLR)
    ax.spines['bottom'].set_color(BORDER_CLR)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    ax.grid(True, color=GRID_CLR, linewidth=0.4, zorder=0)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', color=NAVY,
                     fontfamily=SANS, pad=6, loc='left')


# ── CHART 1: Horse Race ──
def chart_horse_race():
    labels = ['Parkinson RV', 'Yang-Zhang RV', 'SVI Surface IV',
              'Flat BSM IV', 'CC Realized Vol']
    gains  = [-43.2, -18.4, -9.4, 0.0, 5.8]
    colors = [NEG_RED, NEG_RED, NEG_RED, BORDER_CLR, POS_GREEN]
    pvals  = [None, None, '($p$<0.001)', None, '($p$=0.008)']

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    fig.patch.set_alpha(0)
    apply_style(ax, title='Gain vs. Flat BSM Benchmark')

    bars = ax.barh(labels, gains, height=0.55, color=colors, zorder=3,
                   edgecolor='none')
    ax.axvline(0, color=BORDER_CLR, linewidth=0.7, zorder=2)
    ax.set_xlabel('Gain (%)', fontsize=8, color=TEXT_SEC, fontfamily=SANS)
    ax.tick_params(axis='y', labelsize=8.5)

    for bar, val, pv in zip(bars, gains, pvals):
        sign = '+' if val > 0 else ''
        txt = f'{sign}{val}%'
        if pv:
            txt += f'  {pv}'
        xpos = bar.get_width()
        ha = 'left' if val >= 0 else 'right'
        offset = 0.7 if val >= 0 else -0.7
        ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
                txt, va='center', ha=ha, fontsize=8, fontweight='bold',
                fontfamily=MONO, color=TEXT_DARK)

    ax.set_xlim(-55, 20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'horse_race.png'), dpi=300,
                bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('  horse_race.png')


# ── CHART 2: Regime Breakdown ──
def chart_regime_breakdown():
    # 3 rows: BSM Std, SVI Gain, CC Gain
    bsm_std = [18.36, 18.20, 20.64, 64.42]
    svi_gain = [-8.8, 0.8, 3.7, -2.3]
    cc_gain  = [-7.1, -15.3, -12.4, -4.0]

    col_labels = ['Low', 'Normal', 'High', 'Crisis']
    row_labels = ['BSM Std ($)', 'SVI Gain (%)', 'CC Gain (%)']

    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor(WHITE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    ax.set_title('SVI & CC Gain by VIX Regime', fontsize=9,
                 fontweight='bold', color=NAVY, fontfamily=SANS, pad=18, loc='left')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('auto')
    ax.invert_yaxis()

    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=8, fontfamily=SANS, color=TEXT_DARK,
                       fontweight='bold')
    ax.xaxis.tick_top()
    ax.set_yticks(range(3))
    ax.set_yticklabels(row_labels, fontsize=7.5, fontfamily=SANS, color=TEXT_DARK)

    # draw cells
    for j in range(4):
        # row 0: BSM Std — gray background
        v0 = bsm_std[j]
        rect = mpatches.FancyBboxPatch((j - 0.42, -0.38), 0.84, 0.76,
                boxstyle='round,pad=0.03', facecolor=LIGHT_GRAY,
                edgecolor=BORDER_CLR, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(j, 0, f'${v0:.1f}', ha='center', va='center',
                fontsize=8, fontweight='bold', fontfamily=MONO, color=TEXT_DARK)

        # row 1: SVI Gain — green if positive, red if negative
        v1 = svi_gain[j]
        c1 = '#D4E6D4' if v1 > 0 else '#F6D4D4'
        tc1 = POS_GREEN if v1 > 0 else NEG_RED
        rect = mpatches.FancyBboxPatch((j - 0.42, 0.62), 0.84, 0.76,
                boxstyle='round,pad=0.03', facecolor=c1,
                edgecolor=BORDER_CLR, linewidth=0.5)
        ax.add_patch(rect)
        sign1 = '+' if v1 > 0 else ''
        ax.text(j, 1, f'{sign1}{v1}%', ha='center', va='center',
                fontsize=8, fontweight='bold', fontfamily=MONO, color=tc1)

        # row 2: CC Gain — green if positive, red if negative
        v2 = cc_gain[j]
        c2 = '#D4E6D4' if v2 > 0 else '#F6D4D4'
        tc2 = POS_GREEN if v2 > 0 else NEG_RED
        rect = mpatches.FancyBboxPatch((j - 0.42, 1.62), 0.84, 0.76,
                boxstyle='round,pad=0.03', facecolor=c2,
                edgecolor=BORDER_CLR, linewidth=0.5)
        ax.add_patch(rect)
        sign2 = '+' if v2 > 0 else ''
        ax.text(j, 2, f'{sign2}{v2}%', ha='center', va='center',
                fontsize=8, fontweight='bold', fontfamily=MONO, color=tc2)

    # footnote
    ax.text(0.0, 1.0, 'SVI helps only in Normal/High VIX regimes',
            transform=ax.transAxes, fontsize=6, fontstyle='italic',
            color=TEXT_SEC, fontfamily=SANS, va='top')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'regime_breakdown.png'), dpi=300,
                bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('  regime_breakdown.png')


# ── CHART 3: Moneyness × DTE ──
def chart_moneyness_dte():
    # verified data from parquet
    grid = [
        [('SVI', 23.0),  ('SVI', 25.6),  ('YZ', 29.8)],
        [('BSM', 24.4),  ('YZ',  24.0),  ('YZ', 23.4)],
        [('BSM', 42.3),  ('CC',  25.5),  ('CC', 23.0)],
    ]
    row_labels = ['OTM Call', 'ATM', 'OTM Put']
    col_labels = ['Short', 'Medium', 'Long']
    color_map = {
        'SVI': TEAL,
        'CC':  POS_GREEN,
        'BSM': NAVY,
        'YZ':  GOLD,
    }

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    cell_w, cell_h = 0.95, 0.72
    x0, y0 = 0.7, 0.12

    for i, row in enumerate(grid):
        for j, (winner, rmse) in enumerate(row):
            x = x0 + j * cell_w
            y = y0 + (2 - i) * cell_h
            rect = mpatches.FancyBboxPatch(
                (x, y), cell_w - 0.04, cell_h - 0.04,
                boxstyle='round,pad=0.03',
                facecolor=color_map[winner], edgecolor=BORDER_CLR,
                linewidth=0.5, zorder=2)
            ax.add_patch(rect)
            ax.text(x + (cell_w - 0.04) / 2, y + (cell_h - 0.04) * 0.62,
                    winner, ha='center', va='center',
                    fontsize=9, fontweight='bold', fontfamily=SANS, color='white')
            ax.text(x + (cell_w - 0.04) / 2, y + (cell_h - 0.04) * 0.28,
                    f'${rmse:.1f}', ha='center', va='center',
                    fontsize=7.5, fontfamily=MONO, color='white')

    # row labels
    for i, lab in enumerate(row_labels):
        y = y0 + (2 - i) * cell_h + (cell_h - 0.04) / 2
        ax.text(x0 - 0.06, y, lab, ha='right', va='center',
                fontsize=8.5, fontfamily=SANS, color=TEXT_DARK, fontweight='bold')

    # column labels — centered over each cell column
    for j, lab in enumerate(col_labels):
        x = x0 + j * cell_w + (cell_w - 0.04) / 2
        y = y0 + 3 * cell_h - 0.04 + 0.10
        ax.text(x, y, lab, ha='center', va='bottom',
                fontsize=8, fontfamily=SANS, color=TEXT_DARK, fontweight='bold')

    ax.set_title('Best Vol Input by Moneyness \u00d7 DTE', fontsize=9,
                 fontweight='bold', color=NAVY, fontfamily=SANS, pad=6, loc='left')
    ax.text(0.0, -0.02, 'Values = RMSE ($). Lowest wins each cell.',
            transform=ax.transAxes, fontsize=6, fontstyle='italic',
            color=TEXT_SEC, fontfamily=SANS)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'moneyness_dte.png'), dpi=300,
                bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('  moneyness_dte.png')


# ── CHART 4: P&L Attribution ──
def chart_pnl_attribution():
    regimes   = ['Low', 'Normal', 'High', 'Crisis']
    discrete  = [-38, -51, -59, -58]
    vol_mis   = [-20, -10,   6,  -4]
    residual  = [158, 162, 153, 162]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    fig.patch.set_alpha(0)
    apply_style(ax, title='Variance Contribution by Regime')

    y = np.arange(len(regimes))
    h = 0.5

    # discrete rebalancing — Navy
    ax.barh(y, discrete, height=h, color=NAVY, label='Discrete Rebal.',
            zorder=3, edgecolor='none')

    # vol misspecification — stacked from discrete end — NegRed
    vm_left = list(discrete)
    ax.barh(y, vol_mis, height=h, left=vm_left, color=NEG_RED,
            label='Vol Misspec.', zorder=3, edgecolor='none')

    # residual — starts at 0, goes positive — Teal
    ax.barh(y, residual, height=h, color=TEAL,
            label='Higher-Order Resid.', zorder=3, edgecolor='none')

    ax.axvline(0, color=TEXT_DARK, linewidth=0.5, zorder=4)

    for i, r in enumerate(residual):
        ax.text(r + 2, i, f'{r}%', va='center', ha='left',
                fontsize=8, fontweight='bold', fontfamily=MONO, color=TEXT_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(regimes, fontsize=8.5, fontfamily=SANS)
    ax.set_xlabel('Variance Contribution (%)', fontsize=7.5,
                  color=TEXT_SEC, fontfamily=SANS)
    ax.set_xlim(-85, 185)

    # legend BELOW the chart
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=6.5, frameon=False, handlelength=1.0,
              columnspacing=1.0, prop={'family': SANS})

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    # footnote below legend — use fig coords to avoid overlap
    fig.text(0.12, 0.02, 'Components sum > 100% due to negative covariances',
             fontsize=5.5, fontstyle='italic', color=TEXT_SEC, fontfamily=SANS)
    fig.savefig(os.path.join(OUT, 'pnl_attribution.png'), dpi=300,
                bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('  pnl_attribution.png')


if __name__ == '__main__':
    print('generating handout charts...')
    chart_horse_race()
    chart_regime_breakdown()
    chart_moneyness_dte()
    chart_pnl_attribution()
    print('done.')
