"""Fig. 3: per-cell median excess loss of E and I against the frozen crossing index kappa."""
import csv, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, encoding='utf8')))
fs.setup()
fig, ax = plt.subplots(figsize=(fs.COL, 2.3))
ax.axvspan(-math.log(10), math.log(10), color=fs.BAND, lw=0, zorder=0)
style = {'E': dict(color=fs.E, marker='o', s=11, lw=0, label='E (explicit)', zorder=3),
         'I': dict(facecolors='white', edgecolors=fs.I, marker='^', s=13, linewidths=0.7,
                   label='I (implicit)', zorder=4)}
for m in ['E', 'I']:
    r = [x for x in rows if x['method'] == m]
    ax.scatter([float(x['kappa']) for x in r], [float(x['median_log10']) for x in r], **style[m])
ax.set_xscale('symlog', linthresh=3, linscale=0.8)
ax.set_xticks([-10, 0, 10, 1e2, 1e3, 1e4, 1e5])
ax.set_xticklabels([r'$-10$', r'$0$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
ax.set_yscale('log')
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels([r'$10^{1}$', r'$10^{10}$', r'$10^{100}$', r'$10^{1000}$', r'$10^{10000}$'])
ax.minorticks_off()
ax.set_ylim(0.8, 1.2e5)
ax.set_xlabel(r'crossing index $\kappa$ (frozen before data)')
ax.set_ylabel('median excess loss (nats)')
tr = ax.get_xaxis_transform()
ax.text(math.log(10) + 0.6, 0.985, 'crossing', ha='left', va='top', fontsize=7, color=fs.GRAY, transform=tr)
ax.text(0, 0.62, 'transition', ha='center', va='center', rotation=90, fontsize=7, color=fs.GRAY, transform=tr)
ax.text(0.015, 0.985, 'no crossing', ha='left', va='top', fontsize=7, color=fs.GRAY, transform=ax.transAxes)
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.42), handletextpad=0.2, scatterpoints=1)
fig.tight_layout(pad=0.3)
fs.save(fig, out)
