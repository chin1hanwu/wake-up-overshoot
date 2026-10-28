"""81-cell bin-width scan with fixed qc; event-coupled convergence to CT."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import poisson
from core import explicit_step,silence,ct_flow,ct_events

OUT=Path(__file__).resolve().parents[1]/'outputs'
DT=[.0001,.0003,.001,.003,.01,.03,.05,.1,1.]
QC=.2

def event_counts(events,dt,T=1.):
    edges=np.r_[np.arange(0,T,dt),T]
    return np.diff(edges),np.histogram(events,edges)[0]

def run_events(theta,V,events,dt,T=1.):
    widths,cs=event_counts(events,dt,T)
    peaks=[];logmass=[];linear=0.
    Lam=len(events)/T # only used below as a record, not scoring
    for h,C in zip(widths,cs):
        logmass.append(theta+math.log(h))
        theta,V=explicit_step(theta,V,int(C),QC*h,h)
        peaks.append(theta)
    return theta,V,max(peaks),float(logsumexp(logmass))

def main():
    OUT.mkdir(exist_ok=True)
    states={};cts={}
    for L in [5.,20.,60.]:
        cts[L]=ct_flow(math.log(20),.05,L,QC)
        for dt in DT:
            n=int(round(L/dt))
            states[(L,dt)]=(*silence(n,QC*dt,dt),n)
    # Same event times across widths within each (L,Lambda) cell.
    rng=np.random.default_rng(73493);events={}
    for L in [5.,20.,60.]:
        for Lam in [10.,100.,1000.]:
            events[(L,Lam)]=np.sort(rng.uniform(0,1,rng.poisson(Lam)))
    rows=[]
    for L in [5.,20.,60.]:
        thct,W=cts[L];lct=math.exp(thct)
        for Lam in [10.,100.,1000.]:
            hgap=math.log(Lam/lct)
            exponent_scale=1/(W*Lam)
            denom=W*(Lam-lct-hgap*lct)
            onset=hgap/denom if hgap>0 and denom>0 else math.nan
            for dt in DT:
                th,V,n=states[(L,dt)];Le=n*dt;m=Lam*dt;v=V+QC*dt;s=math.exp(th)*dt
                g=v/(1+v*s);d=g*s
                te,_=explicit_step(th,V,m,QC*dt,dt)
                leading=math.log(6/(QC*Le*Le))+QC*Le*Lam*dt/3
                p95=th-d+g*poisson.ppf(.95,m)
                z95=th-d+g*poisson.ppf(1-.05*(-math.expm1(-m)),m)
                cutoff=int(math.floor((math.log(Lam)-th+d)/g)+1)
                prob=poisson.sf(cutoff-1,m) if cutoff>0 else 1.
                tend,vend,pmax,lmass=run_events(th,V,events[(L,Lam)],dt,1.)
                rows.append(dict(L_requested=L,L_effective=Le,dt=dt,q=QC*dt,Lambda=Lam,n=n,V_prior=v,rate_prior=math.exp(th),gain=g,mean_C=m,mean_exponent=g*m,leading_exponent=QC*Le*Lam*dt/3,mean_input_log10_post=te/math.log(10),leading_log10_post=leading/math.log(10),leading_log_error=te-leading,mean_input_log_ratio=te-math.log(Lam),q95_log10_post=p95/math.log(10),positive_q95_log10_post=z95/math.log(10),P_onebin_overshoot=prob,unit_exponent_dt=exponent_scale,ct_frozen_onset_dt=onset,approx_onset_dt=3*math.log(QC*Le*Le*Lam/6)/(QC*Le*Lam),shared_path_count=len(events[(L,Lam)]),shared_path_1s_peak_log10=pmax/math.log(10),shared_path_1s_log10_integrated_prediction=lmass/math.log(10)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'p3_scan.csv',index=False)
    assert len(df)==81
    df.groupby('L_requested').agg(cells=('dt','size'),median_abs_leading_log_error=('leading_log_error',lambda x:np.median(abs(x))),max_abs_leading_log_error=('leading_log_error',lambda x:np.max(abs(x)))).to_csv(OUT/'p3_scan_summary.csv')

    # Preserve the exact RNG order in sanity2.py. These are not coupled paths.
    rng=np.random.default_rng(1);rows=[]
    for dt in [.05,.01,.001,.0001]:
        th,V,n=states[(60.,dt)];peak=-math.inf;cs=rng.poisson(1000*dt,round(.05/dt))
        for C in cs:
            th,V=explicit_step(th,V,C,QC*dt,dt);peak=max(peak,th)
        rows.append(dict(dt=dt,count_total=int(cs.sum()),first_C=int(cs[0]),peak_log10=peak/math.log(10),final_log10=th/math.log(10)))
    pd.DataFrame(rows).to_csv(OUT/'p3_F3_reference.csv',index=False)

    # Reproduce CT event sample from sanity.py, then improve its ODE accuracy.
    rng=np.random.default_rng(0)
    for _ in range(2000):
        rng.uniform(-10,8);rng.uniform(-3,1.5);rng.integers(0,200);rng.uniform(-3,-.5)
    ev=np.sort(rng.uniform(0,.05,50))
    th0,W0=cts[60.]
    tc,wc,_,jump_states=ct_events(th0,W0,ev,.05,QC)
    ctpeak=float(jump_states[:,0].max())
    np.savetxt(OUT/'p3_ct_events.csv',ev,delimiter=',',header='event_time_s',comments='')
    rows=[]
    for dt in [.01,.003,.001,.0003,.0001,.00003,.00001]:
        n=round(60/dt)
        th,V=silence(n,QC*dt,dt)
        te,ve,peak,lmass=run_events(th,V,ev,dt,.05)
        rows.append(dict(dt=dt,ct_peak_rate=math.exp(ctpeak),ct_end_rate=math.exp(tc),discrete_peak_rate=math.exp(peak),discrete_end_rate=math.exp(te),end_theta_error=te-tc,end_V_error=ve-wc,peak_log_error=peak-ctpeak))
    pd.DataFrame(rows).to_csv(OUT/'p3_ct_convergence.csv',index=False)
    print('p3_binning: 81 cells, reference RNG reproduction, and coupled CT convergence saved.')

if __name__=='__main__':main()
