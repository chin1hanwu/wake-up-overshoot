"""Shared style for the letter's figures: final print size, Times text, two hues (E, I).

Palette checked with the dataviz validator (light surface, CVD dE >= 24). References
(exact Bayes, CT filter, true rate, asymptotic laws) are black or gray and are told apart
by line style and direct labels, so every figure also reads in grayscale.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E = '#d55e00'      # explicit update
I = '#1f6fbf'      # implicit update
REF = '#000000'    # exact Bayes / CT filter / true rate
GRAY = '#6b6b6b'   # asymptotic laws and guide lines
BAND = '#ebebe6'   # transition region
GRID = '#dcdcd7'

FULL = 6.14        # 0.86 * \textwidth of IEEEtran journal (516 pt), inches
COL = 3.45         # \columnwidth (252 pt), inches

STYLE_E = dict(color=E, marker='o', ms=3.2, lw=1.0, mec=E, mfc=E)
STYLE_I = dict(color=I, marker='^', ms=3.6, lw=1.0, mec=I, mfc='white', mew=0.8)


def setup():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })


def panel(ax, tag, title):
    """Panel tag (A), (B) and a short title, left-aligned above the axes."""
    ax.set_title(r'(%s) %s' % (tag, title), loc='left', fontsize=8, pad=3)


def save(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=300)
