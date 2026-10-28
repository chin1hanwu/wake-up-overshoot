"""Three source-backed research figures; all plotted tables ship in outputs/."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import silence,explicit_step,implicit_step

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';FIG=ROOT/'figures'
COL={'E':'#b45309','I':'#2563eb','X':'#0f766e','gray':'#64748b'}

def finish(fig,name):
    fig.tight_layout(pad=1.2)
    fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'lines.linewidth':1.8,'pdf.fonttype':42})
    d=pd.read_csv(OUT/'p1_discrete_asymptotics.csv');d=d[d['case']=='baseline']
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for method in ['E','I']:
        z=d[d.method==method]
        ax[0].semilogx(z.n,z.V-z.q*z.n/3,'o',color=COL[method],label=method+' recurrence')
        ns=np.geomspace(z.n.min(),z.n.max(),150)
        ax[0].semilogx(ns,z.a_log.iloc[0]*np.log(ns)+z.reference_b_est.iloc[0],color=COL[method],label=method+' second order')
    ax[0].set(xlabel='Zero-count bins, n',ylabel=r'$V_n-qn/3$',title='A  Opposite logarithmic corrections')
    ax[0].legend(fontsize=8,ncol=2)
    x=pd.read_csv(OUT/'p5_ct_spectral_asymptotics.csv')
    x=x[np.isclose(x.qc,.2)&np.isclose(x.rate0,20)&np.isclose(x.V0,.05)].sort_values('t')
    ts=np.geomspace(20,10000,180)
    ax[1].loglog(x.t,x.hazard,'o-',color=COL['X'],label='Exact CT Bayes mean')
    ax[1].loglog(ts,.5/ts,'--',color=COL['X'],label=r'$1/(2t)$')
    ax[1].loglog(ts,6/(.2*ts**2),'--',color=COL['E'],label=r'Gaussian median: $6/(q_ct^2)$')
    ax[1].set(xlabel='Silence duration, t (s)',ylabel='Rate (events/s)',title='B  Distinct long-silence readouts')
    ax[1].legend(fontsize=8)
    finish(fig,'figure1_silence')

    d=pd.read_csv(OUT/'p3_scan.csv');d=d[(d.L_requested==60)&(d.Lambda==1000)&(d.dt<=.05)].sort_values('dt')
    c=pd.read_csv(OUT/'p3_ct_convergence.csv').sort_values('dt')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].semilogx(d.dt*1000,d.mean_input_log10_post,'o-',color=COL['E'],label='Mean-count proxy, exact recurrence')
    ax[0].semilogx(d.dt*1000,d.leading_log10_post,'--',color=COL['gray'],label='Leading silence law')
    ax[0].axhline(3,color='black',lw=1,label='True rate: 1000/s')
    ax[0].axvline(.25,color=COL['gray'],ls=':',label='Unit exponent: 0.25 ms')
    ax[0].axvline(d.ct_frozen_onset_dt.iloc[0]*1000,color=COL['I'],ls=':',label='Frozen crossing: 2.88 ms')
    ax[0].set(xlabel='Bin width (ms)',ylabel=r'$\log_{10}$ post-update rate',title='A  Unit exponent is not rate crossing')
    ax[0].legend(fontsize=7.6)
    ax[1].semilogx(c.dt*1000,np.log10(c.discrete_peak_rate),'o-',color=COL['E'],label='E: identical 50 event times')
    ax[1].axhline(np.log10(c.ct_peak_rate.iloc[0]),color=COL['X'],label='CT: event-aligned ODE')
    ax[1].set(xlabel='Bin width (ms)',ylabel=r'$\log_{10}$ peak rate over 50 ms',title='B  Fixed-path convergence to CT')
    ax[1].legend(fontsize=8)
    finish(fig,'figure2_binning')

    rows=[]
    for method,step in [('E',explicit_step),('I',implicit_step)]:
        th,v=silence(1200,.01,.05,method=method)
        for k in range(1,221):
            th,v=step(th,v,50,.01,.05)
            rows.append(dict(method=method,wake_bin=k,rate=math.exp(th),log10_rate=th/math.log(10)))
    r=pd.DataFrame(rows);r.to_csv(OUT/'figure3_gaussian_recovery.csv',index=False)
    x=pd.read_csv(OUT/'p5_wake_recovery.csv');x=x[x.trace=='constant50']
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for method in ['E','I']:
        z=r[r.method==method]
        ax[0].plot(z.wake_bin*.05,z.log10_rate,color=COL[method],label=method+' median readout')
    ax[0].axhline(3,color='black',lw=1,ls='--')
    ax[0].set(xlabel='Time after wake boundary (s)',ylabel=r'$\log_{10}$ post-update rate',title='A  Explicit relaxation after C = 50')
    ax[0].legend(fontsize=8)
    z=r[(r.method=='I')&(r.wake_bin<=8)];xx=x[x.wake_bin<=8]
    ax[1].plot(z.wake_bin,z.rate,'o-',color=COL['I'],label='I: Gaussian posterior mode')
    ax[1].plot(xx.wake_bin,xx.mean_rate,'s-',color=COL['X'],label='X: exact posterior mean')
    ax[1].axhline(1000,color='black',lw=1,ls='--',label='True rate / count rate')
    ax[1].axhline(1000/1.5,color=COL['gray'],lw=1,ls=':')
    ax[1].set(xlabel='Completed wake updates (50 ms each)',ylabel='Rate (events/s)',ylim=(0,1100),title='B  Exact Bayes needs several bins')
    ax[1].legend(fontsize=8)
    finish(fig,'figure3_recovery')
    print('make_figures: 3 figures saved as PNG and vector PDF.')

if __name__=='__main__':main()
