"""Figs. 1 and 2 of the letter, redrawn at print size from the theory outputs (read-only).

  python make_fig_theory.py <theory_round2_results/outputs> <figures dir>

Same data and selections as theory_round2_results/verify/make_figures.py; the plotted
values are also written to fig1_data.csv and fig2_data.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
import matplotlib.pyplot as plt
import matplotlib.ticker

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
fs.setup()

# Fig. 1 -------------------------------------------------------------------------------
d = pd.read_csv(src / 'p1_discrete_asymptotics.csv')
d = d[d['case'] == 'baseline']
x = pd.read_csv(src / 'p5_ct_spectral_asymptotics.csv')
x = x[np.isclose(x.qc, .2) & np.isclose(x.rate0, 20) & np.isclose(x.V0, .05)].sort_values('t')
qc = 0.2
fig, ax = plt.subplots(1, 2, figsize=(fs.FULL, 2.1))
rows = []
for m, st in [('E', fs.STYLE_E), ('I', fs.STYLE_I)]:
    z = d[d.method == m].sort_values('n')
    ns = np.geomspace(z.n.min(), z.n.max(), 150)
    law = z.a_log.iloc[0] * np.log(ns) + z.reference_b_est.iloc[0]
    ax[0].plot(ns, law, color=st['color'], lw=1.0, zorder=2)
    ax[0].plot(z.n, z.V - z.q * z.n / 3, ls='none', **{k: v for k, v in st.items() if k != 'lw'}, zorder=3)
    rows += [dict(panel='A', method=m, n=n, V_minus_qn3=v) for n, v in zip(z.n, z.V - z.q * z.n / 3)]
ax[0].set_xscale('log')
ax[0].set_xlabel(r'zero-count bins $n$')
ax[0].set_ylabel(r'$V_n-qn/3$')
ax[0].text(2.2e2, 0.0768, 'I', color=fs.I, fontsize=8, ha='center')
ax[0].text(2.2e2, 0.0638, 'E', color=fs.E, fontsize=8, ha='center')
ax[0].plot([], [], color='k', lw=1.0, label='second-order law')
ax[0].plot([], [], ls='none', marker='o', ms=3.2, color='k', label='exact recursion')
ax[0].legend(loc='center left', bbox_to_anchor=(0.0, 0.5))
fs.panel(ax[0], 'A', 'variance after $n$ empty bins')

ts = np.geomspace(20, 10000, 180)
ax[1].plot(x.t, x.hazard, color=fs.REF, lw=1.0, marker='o', ms=3.0, label='exact CT Bayes mean', zorder=3)
ax[1].plot(ts, .5 / ts, color=fs.REF, lw=0.9, ls='--', label=r'$1/(2t)$')
ax[1].plot(ts, 6 / (qc * ts ** 2), color=fs.GRAY, lw=1.0, ls='-.', label=r'Gaussian median $6/(q_ct^2)$')
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel(r'silence length $t$ (s)')
ax[1].set_ylabel('predicted rate (/s)')
ax[1].legend(loc='lower left')
fs.panel(ax[1], 'B', 'predicted rate during silence')
rows += [dict(panel='B', t=t, exact_ct_mean=h) for t, h in zip(x.t, x.hazard)]
fig.tight_layout(pad=0.3, w_pad=1.2)
fs.save(fig, str(out / 'fig1_silence.pdf'))
pd.DataFrame(rows).to_csv(out / 'fig1_data.csv', index=False)
plt.close(fig)

# Fig. 2 -------------------------------------------------------------------------------
s = pd.read_csv(src / 'p3_scan.csv')
s = s[(s.L_requested == 60) & (s.Lambda == 1000) & (s.dt <= .05)].sort_values('dt')
c = pd.read_csv(src / 'p3_ct_convergence.csv').sort_values('dt')
unit, cross = s.unit_exponent_dt.iloc[0] * 1000, s.ct_frozen_onset_dt.iloc[0] * 1000
fig, ax = plt.subplots(1, 2, figsize=(fs.FULL, 2.1))
ax[0].plot(s.dt * 1000, s.leading_log10_post, color=fs.GRAY, lw=1.0, ls='--', label='leading silence law', zorder=2)
ax[0].plot(s.dt * 1000, s.mean_input_log10_post, ls='none', **{k: v for k, v in fs.STYLE_E.items() if k != 'lw'},
           label='E, exact recursion', zorder=3)
ax[0].axhline(3, color=fs.REF, lw=0.7)
ax[0].text(0.11, 6, r'true rate $\Lambda=1000$/s', fontsize=7, va='bottom')
for v, lab in [(unit, 'unit exponent\n%.2f ms' % unit), (cross, 'crossing\n%.2f ms' % cross)]:
    ax[0].axvline(v, color=fs.GRAY, lw=0.7, ls=':')
    ax[0].text(v * 1.12, 62, lab, fontsize=7, va='top', ha='left')
ax[0].set_xscale('log')
ax[0].set_xlabel(r'bin width $\Delta$ (ms)')
ax[0].set_ylabel(r'$\log_{10}$ post-wake rate (/s)')
ax[0].set_ylim(-8, 92)
ax[0].legend(loc='upper left')
fs.panel(ax[0], 'A', 'first update, mean-count input')

ax[1].plot(c.dt * 1000, np.log10(c.discrete_peak_rate), **fs.STYLE_E, label='E, same 50 event times', zorder=3)
ax[1].axhline(np.log10(c.ct_peak_rate.iloc[0]), color=fs.REF, lw=1.0, label='CT filter', zorder=2)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'bin width $\Delta$ (ms)')
ax[1].set_ylabel(r'$\log_{10}$ peak rate over 50 ms (/s)')
ax[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax[1].legend(loc='upper left')
fs.panel(ax[1], 'B', 'fixed event path')
fig.tight_layout(pad=0.3, w_pad=1.2)
fs.save(fig, str(out / 'fig2_binning.pdf'))
pd.concat([s.assign(panel='A')[['panel', 'dt', 'mean_input_log10_post', 'leading_log10_post', 'unit_exponent_dt', 'ct_frozen_onset_dt']],
           c.assign(panel='B')[['panel', 'dt', 'discrete_peak_rate', 'ct_peak_rate']]]).to_csv(out / 'fig2_data.csv', index=False)
plt.close(fig)
print('unit %.4f ms, crossing %.4f ms, CT peak %.1f/s' % (unit, cross, c.ct_peak_rate.iloc[0]))
