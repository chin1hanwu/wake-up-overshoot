#!/usr/bin/env python3
"""Round 2 P4: implicit-update bounds, asymptotics, and counterexamples.

Run: python verify/p4_implicit.py
No FFT, downloads, repository changes, or fitted constants are used.
The Monte Carlo section is a numerical verification of a fixed-bin theorem,
not a claim of Bayes optimality. All wake data are external constant-rate
Poisson stress inputs, not trajectories of the internal Brownian state model.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import wrightomega

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def implicit(theta, v, c, q, dt):
    """Stable posterior mode; v is the previous POSTERIOR variance."""
    s = v + q
    z = math.log(s * dt) + theta + s * c
    if z < -700:
        w = math.exp(z) if z > -745 else 0.0
        newtheta = theta + s * c - w
    else:
        w = float(wrightomega(z))
        # This form avoids subtracting nearly equal, very large s*c and W.
        newtheta = math.log(w) - math.log(s * dt)
    return newtheta, s / (1.0 + w)


def explicit(theta, v, c, q, dt):
    s = v + q
    mu = math.exp(theta) * dt
    vp = s / (1.0 + s * mu)
    return theta + vp * (c - mu), vp


def silent(kind, q, dt, n, rate=20.0, v=0.05):
    theta = math.log(rate)
    fun = explicit if kind == "E" else implicit
    for _ in range(n):
        theta, v = fun(theta, v, 0.0, q, dt)
    return theta, v


def bisect_mode(theta, s, c, dt, eps=1e-12):
    if c == 0:
        lo, hi = theta - s * math.exp(theta) * dt, theta
    else:
        target = math.log(c / dt)
        lo, hi = min(theta, target), max(theta, target)
    original_width = hi - lo
    bound = max(0, math.ceil(math.log2(original_width / (2 * eps)))) if original_width > 2*eps else 0
    count = 0
    while hi - lo > 2*eps:
        mid = 0.5 * (lo + hi)
        score = mid - theta + s * (dt * math.exp(mid) - c)
        if score > 0:
            hi = mid
        else:
            lo = mid
        count += 1
    return 0.5 * (lo + hi), count, bound, original_width


def f6():
    q, dt, n, c = .01, .05, 1200, 50
    theta, v = silent("E", q, dt, n)
    s = v + q
    th_i, _ = implicit(theta, v, c, q, dt)
    th_e, _ = explicit(theta, v, c, q, dt)
    th_b, it_b, bound, width = bisect_mode(theta, s, c, dt)
    b, seq = theta, []
    for it in range(1, 2001):
        mu = dt * math.exp(b)
        nb = b + (-(b-theta)/s+c-mu)/(1/s+mu)
        seq.append(nb)
        if abs(nb-b) < 1e-10:
            b = nb
            break
        b = nb
    assert abs(b-th_i) < 1e-10
    assert abs(th_b-th_i) < 2e-12
    assert it_b <= bound + 1
    own_th, own_v = silent("I", q, dt, n)
    own_wake, _ = implicit(own_th, own_v, c, q, dt)
    row = dict(prior_rate=math.exp(theta), prior_variance=s,
               explicit_log10_rate=th_e/math.log(10), implicit_rate=math.exp(th_i),
               implicit_own_silence_rate=math.exp(own_wake),
               undamped_iterations=it, undamped_max_log10_rate=max(seq)/math.log(10),
               bisection_iterations=it_b, bisection_bound=bound,
               bisection_width=width, bisection_abs_error=abs(th_b-th_i))
    write_csv("p4_f6.csv", [row])
    return row


def wake_bounds():
    rows = []
    for q in [.0001, .01, .1]:
        for n in [10, 100, 1200, 10000]:
            dt = .05
            theta, v = silent("I", q, dt, n)
            s, mu = v+q, dt*math.exp(theta)
            for c in [0, 1, 5, 50]:
                th, vp = implicit(theta, v, c, q, dt)
                rate = math.exp(th)
                upper = max(math.exp(theta), c/dt)
                assert rate <= upper * (1+2e-14)
                h = math.log(c/mu) if c else float("nan")
                k = s*c
                if c and 0 < h < k:
                    approx_ratio = 1-h/k
                    err_bound_ratio = -math.log1p(-h/k)/k
                    ratio = rate/(c/dt)
                    err = ratio-approx_ratio
                    assert err >= -5e-14
                    assert err <= err_bound_ratio+5e-14
                    theta_approx = math.log(c/dt)+math.log1p(-h/k)
                    theta_err_bound = -math.log1p(-h/k)/(k-h)
                    assert abs(th-theta_approx) <= theta_err_bound+5e-14
                else:
                    ratio = rate/(c/dt) if c else float("nan")
                    approx_ratio=err_bound_ratio=err=theta_approx=theta_err_bound=float("nan")
                rows.append(dict(q=q, dt=dt, n=n, C=c, prior_rate=math.exp(theta),
                    s=s, h=h, k=k, posterior_rate=rate, no_crossing_upper=upper,
                    ratio=ratio, leading_ratio=approx_ratio, ratio_error=err,
                    ratio_error_bound=err_bound_ratio, theta_approx=theta_approx,
                    theta_error_bound=theta_err_bound))
    write_csv("p4_wake_bounds.csv", rows)
    rows2=[]
    for n in [100, 1000, 1000000]:
        a=2/n
        A=-math.log(a)
        for multiple in [.1, .5, 1, 2, 10]:
            k=multiple*A
            # C=1, s=k, mu=a/k: genuine integer count at the transition.
            dt=.05
            theta=math.log(a/k/dt)
            th, _ = implicit(theta, k, 1, 0, dt)
            ratio=math.exp(th)*dt
            rows2.append(dict(n=n, a=a, k=k, k_over_log_n_half=multiple,
                              mode_over_count_rate=ratio,
                              uncorrected_task_approx=1-A/k,
                              exact_w_over_k=float(wrightomega(k+math.log(a)))/k))
    write_csv("p4_transition.csv", rows2)


def solver_grid():
    rows=[]
    for theta in [-40., math.log(.008), 16.]:
        for s in [1e-6, 4.075, 1000.]:
            for dt in [1e-12,.05,1.]:
                for c in [0,1,50]:
                    th,_=implicit(theta,s,c,0,dt)
                    root,it,bound,width=bisect_mode(theta,s,c,dt)
                    err=abs(root-th)
                    assert err < 3e-12
                    assert it <= bound+1
                    rows.append(dict(theta=theta,s=s,dt=dt,C=c,mode=th,
                        bisection_mode=root,abs_error=err,iterations=it,
                        iteration_bound=bound,initial_width=width,
                        lambert_log_argument=math.log(s*dt)+theta+s*c))
    write_csv("p4_solver_grid.csv",rows)


def mean_recovery():
    rows=[]
    for dt in [.001, .05, 1.0]:
        qc=.2
        q=qc*dt
        for L in [5, 60, 600]:
            theta0,v0=silent("I",q,dt,round(L/dt))
            for lam in [1,1000]:
                m=lam*dt
                D0=math.log(lam)-theta0
                d=math.log(1.5)
                vstar=(-q+math.sqrt(q*q+4*q/m))/2
                vmin=min(v0,vstar)
                smin=vmin+q
                max_steps=math.ceil(max(0,D0-d)/(smin*m*(1-math.exp(-d))))+1
                theta,v=theta0,v0
                steps=0
                if D0 > d:
                    for steps in range(1, max_steps+1):
                        theta,v=implicit(theta,v,m,q,dt)
                        assert theta <= math.log(lam)+1e-12
                        if math.log(lam)-theta <= d:
                            break
                    else:
                        raise AssertionError("mean-input recovery bound violated")
                else:
                    steps=0
                rows.append(dict(dt=dt,L=L,Lambda=lam,mean_count=m,initial_log_deficit=D0,
                    recovered_bins=steps,recovered_seconds=steps*dt,
                    rigorous_bin_upper=max_steps,rigorous_seconds_upper=max_steps*dt,
                    variance_lower=vmin))
    write_csv("p4_mean_recovery.csv",rows)


def ct_state(L,qc=.2):
    def rhs(t,y):
        theta,W=y
        lam=math.exp(theta)
        return [-W*lam,qc-W*W*lam]
    sol=solve_ivp(rhs,[0,L],[math.log(20),.05],rtol=2e-12,atol=2e-13)
    assert sol.success
    return tuple(sol.y[:,-1])


def ct_counterexample():
    rows=[]
    qc=.2
    a=2*qc/3
    lam_true=1000
    def critical_log_rate(L):
        theta,W=ct_state(L,qc)
        la=math.exp(theta+W)
        B=(W*W*la-a)/la**3
        return (math.log(a)-math.log(B))/3
    boundary=brentq(lambda L:critical_log_rate(L)-math.log(6*lam_true),360,600)
    for L in [60,120,180,240,360,boundary*.999,boundary,boundary*1.001,600,1000]:
        theta,W=ct_state(L,qc)
        rate=math.exp(theta)
        logA=theta+W
        lamA=math.exp(logA)
        dt=.05*math.exp(-2*W)
        thI,_=implicit(theta,W,1,qc*dt,dt)
        x=W*W*lamA
        row=dict(L=L,qc=qc,silent_rate=rate,silent_W=W,ct_first_event_rate=lamA,
            ct_first_event_ratio=lamA/lam_true,diagonal_dt=dt,
            ct_initialized_implicit_rate=math.exp(thI),
            I_over_CT=math.exp(thI-logA),critical_rate=float("nan"),
            descent_time_exact=float("nan"),descent_time_upper=float("nan"),
            integrated_rate_exact=float("nan"),integrated_rate_lower=float("nan"),
            integrated_kl_exact=float("nan"),kl_lower_when_critical_ge_6Lambda=float("nan"),
            ode_mass_relative_error=float("nan"),ode_final_rate_relative_error=float("nan"))
        if x > a:
            B=(x-a)/lamA**3
            lc=(a/B)**(1/3)
            R=lamA/lc
            if R>1:
                j=quad(lambda u:1/math.sqrt(1+u**3),1/R,1,epsabs=1e-12)[0]
                k=quad(lambda u:u/math.sqrt(1+u**3),1/R,1,epsabs=1e-12)[0]
                jlog=quad(lambda u:u/math.sqrt(1+u**3)*math.log(lc/(lam_true*u)),1/R,1,epsabs=1e-12)[0]
                mass=math.sqrt(lc/a)*j
                tau=k/math.sqrt(a*lc)
                kl=mass-lam_true*tau-lam_true*jlog/math.sqrt(a*lc)
                tlower=1/math.sqrt(2*B)*(1/lc-1/lamA)
                tupper=.5/math.sqrt(B)*(1/lc**2-1/lamA**2)
                def rhs(t,y):
                    lth,w,haz=y
                    la=math.exp(lth)
                    return [-w*la,qc-w*w*la,la]
                ode=solve_ivp(rhs,[0,tau],[logA,W,0],rtol=1e-10,atol=1e-11)
                assert ode.success
                rel=abs(ode.y[2,-1]/mass-1)
                raterr=abs(math.exp(ode.y[0,-1])/lc-1)
                assert rel < 1e-6
                assert raterr < 1e-6
                assert mass >= tlower*(1-1e-10)
                assert tau <= tupper*(1+1e-10)
                kl_lower=.5*tlower if lc >= 6*lam_true else float("nan")
                if math.isfinite(kl_lower):
                    assert kl >= kl_lower*(1-1e-10)
                row.update(critical_rate=lc,descent_time_exact=tau,descent_time_upper=tupper,
                    integrated_rate_exact=mass,integrated_rate_lower=tlower,
                    integrated_kl_exact=kl,kl_lower_when_critical_ge_6Lambda=kl_lower,
                    ode_mass_relative_error=rel,ode_final_rate_relative_error=raterr)
        rows.append(row)
    write_csv("p4_ct_counterexample.csv",rows)


def fixed_bin_risk():
    """Common random numbers, expected (conditional KL) risk estimator."""
    rng=np.random.default_rng(421017)
    rows=[]
    for lam,H,reps in [(10.,40,40000),(1000.,20,10000)]:
        dt=.05
        qc=.2
        q=qc*dt
        m=lam*dt
        counts=rng.poisson(m,(H,reps))
        coefficient=2*m*(-math.expm1(-H*m))/(-math.expm1(-m))
        for L in [20,60,200,1000,5000,20000]:
            theta,v=silent("I",q,dt,round(L/dt))
            th=np.full(reps,theta)
            vv=np.full(reps,v)
            risk=np.zeros(reps)
            peak=np.exp(th)
            for c in counts:
                rate=np.exp(th)
                risk+=dt*(rate-lam+lam*(math.log(lam)-th))
                s=vv+q
                z=np.log(s*dt)+th+s*c
                w=wrightomega(z)
                th=np.log(w)-np.log(s*dt)
                vv=s/(1+w)
                peak=np.maximum(peak,np.exp(th))
            bound=np.maximum(math.exp(theta),counts.max(axis=0)/dt)
            assert np.all(peak <= bound*(1+1e-12))
            rows.append(dict(L=L,dt=dt,Lambda=lam,H=H,reps=reps,
                risk_mean=float(risk.mean()),risk_mc_se=float(risk.std(ddof=1)/math.sqrt(reps)),
                logarithmic_coefficient=coefficient,
                log_n=math.log(L/dt),
                risk_minus_leading=float(risk.mean()-coefficient*math.log(L/dt)),
                peak_mean=float(peak.mean()),peak_count_bound_mean=float(bound.mean())))
    write_csv("p4_fixed_bin_risk.csv",rows)


def main():
    summary=f6()
    wake_bounds()
    solver_grid()
    mean_recovery()
    ct_counterexample()
    fixed_bin_risk()
    summary["status"]="passed"
    summary["notes"]=["C=Lambda*dt recovery is a deterministic mean-input proxy.",
        "CT-initialized diagonal single-step test is not a simulation of exponentially many silence bins.",
        "Fixed-bin asymptotic constants need not be uniform in dt."]
    (OUT/"p4_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
