"""Exact wake identity, Poisson quantiles/moments, and cooling-loss constants."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.special import logsumexp
from core import explicit_step,silence,predictive_loss

OUT=Path(__file__).resolve().parents[1]/'outputs'
OUT.mkdir(exist_ok=True)

def main():
    rows=[]
    for q in [.0001,.01,.1]:
        dt=.05
        for n in [10,100,1200,10000]:
            th,V=silence(n,q,dt);v=V+q;s=math.exp(th)*dt
            g=v/(1+v*s);d=g*s
            for C in [0,1,5,50,200]:
                actual,_=explicit_step(th,V,C,q,dt)
                main=math.log(6/(q*n*n))+q*n*C/3
                exact=math.log(s)+g*C-d
                remainder=-v*s*(v*C+1)/(1+v*s)
                bound=abs(math.log(s*q*n*n/6))+C*abs(v-q*n/3)+v*s*(v*C+1)/(1+v*s)
                error=actual+math.log(dt)-main
                assert abs(exact-(actual+math.log(dt)))<2e-11
                assert abs(error)<=bound+2e-11
                rows.append(dict(q=q,dt=dt,n=n,C=C,V_prior=v,s_prior=s,gain=g,log_count_actual=exact,leading=main,error=error,certified_error_bound=bound,state_exp_remainder=remainder,finite_state_identity_error=actual+math.log(dt)-math.log(s)-v*C-remainder))
    pd.DataFrame(rows).to_csv(OUT/'p2_wake_identity.csv',index=False)

    rows=[]
    for q,dt,n,Lambda in [(.01,.05,1200,1000),(.0002,.001,60000,1000),(.0002,.001,60000,20),(.01,.05,1200,1)]:
        th,V=silence(n,q,dt);s=math.exp(th)*dt;v=V+q;g=v/(1+v*s);d=g*s;m=Lambda*dt
        logA=th-d
        logmean=logA+m*math.expm1(g)
        # Independent log-domain summation: dominant counts are near m*exp(g),
        # far outside ordinary Monte Carlo samples in the explosive case.
        tilted=m*math.exp(g)
        ks=np.arange(0,int(poisson.ppf(1-1e-13,tilted))+2)
        summed=float(logsumexp(poisson.logpmf(ks,m)+logA+g*ks))
        assert abs(summed-logmean)<2e-9
        logmedian=logA+g*poisson.ppf(.5,m)
        logq95=logA+g*poisson.ppf(.95,m)
        zmed=poisson.ppf(1-.5*(-math.expm1(-m)),m)
        zq95=poisson.ppf(1-.05*(-math.expm1(-m)),m)
        threshold=(math.log(Lambda)-logA)/g
        k=int(math.floor(threshold)+1)
        prob=float(poisson.sf(k-1,m)) if k>0 else 1.
        probs=poisson.pmf(np.arange(0,int(poisson.ppf(1-1e-14,m))+1),m)
        eglog=float(np.dot(probs,logA+g*np.arange(len(probs))))
        assert abs(eglog-(logA+g*m))<1e-8
        rows.append(dict(q=q,dt=dt,L=n*dt,Lambda=Lambda,m=m,V_prior=v,gain=g,log10_median_rate=logmedian/math.log(10),log10_mean_rate=logmean/math.log(10),moment_logsum_error=summed-logmean,log10_q95_rate=logq95/math.log(10),conditional_median_C=zmed,conditional_q95_C=zq95,log10_first_nonzero_median_rate=(logA+g*zmed)/math.log(10),log10_first_nonzero_q95_rate=(logA+g*zq95)/math.log(10),P_rate_above_Lambda=prob,P_rate_above_Lambda_given_positive=prob/(-math.expm1(-m)) if k>=1 else 1.,next_bin_mean_mass_log10=(logmean+math.log(dt))/math.log(10)))
    pd.DataFrame(rows).to_csv(OUT/'p2_poisson_readouts.csv',index=False)

    rows=[]
    q=.01;dt=.05;Lam=1000.;m=Lam*dt
    th,V=silence(1200,q,dt)
    for C in [5,10,20,37,50]:
        tw,Vw=explicit_step(th,V,C,q,dt);peak=math.exp(tw)*dt
        total=0.;mass=0.;t=tw;v=Vw;rec=None
        for j in range(1200):
            total+=predictive_loss(t,m,Lam,dt);mass+=math.exp(t)*dt
            t,v=explicit_step(t,v,m,q,dt)
            if rec is None and abs(t-math.log(Lam))<=math.log(1.5):rec=j+1
        M=10000.;eta=(1/q+m)/M
        D=math.log(peak/M)
        rows.append(dict(wake_C=C,log10_peak_rate=tw/math.log(10),cooling_bins=rec,leading_log_overshoot=tw-math.log(Lam),loss_60s=total,mass_60s=mass,loss_over_peak_mass=total/peak,mass_over_peak_mass=mass/peak,geometric_constant=1/(-math.expm1(-1)),certified_threshold_M=M,certified_eta=eta,N_lower=math.ceil(max(0,D)),N_upper=math.ceil(max(0,D)/(1-eta))))
    pd.DataFrame(rows).to_csv(OUT/'p2_cooling.csv',index=False)

    # Exact reproduction of the reference post_wake.py, preserving RNG order.
    rng=np.random.default_rng(3);rows=[]
    for dt in [.05,.01,.001]:
        q=.2*dt;th,V=silence(round(60/dt),q,dt);loss=0.;rec=None;logs=[];counts=[]
        for k in range(round(60/dt)):
            C=rng.poisson(1000*dt);counts.append(C)
            loss+=predictive_loss(th,C,1000,dt)
            th,V=explicit_step(th,V,C,q,dt);logs.append(th)
            if rec is None and k>0 and abs(th-math.log(1000))<math.log(1.5):rec=k
        peak=max(logs)
        rows.append(dict(dt=dt,first_C=counts[0],peak_log10=peak/math.log(10),reference_recovery_index=rec,reference_recovery_seconds=rec*dt if rec is not None else math.nan,completed_bin_recovery_seconds=(rec+1)*dt if rec is not None else math.nan,prequential_loss_60s=loss,loss_over_peak_mass=loss/(math.exp(peak)*dt),first_five_log_steps=';'.join(f'{x:.8g}' for x in np.diff(logs)[:5])))
    pd.DataFrame(rows).to_csv(OUT/'p2_F4_reference.csv',index=False)
    print('p2_wake: exact identities/bounds passed; Poisson and cooling outputs saved.')

if __name__=='__main__':main()
