#!/usr/bin/env python3
"""Direct-convolution exact Bayes checks for theory round 2, P5.

No FFT, probability floor, rate clipping, or replacement Gaussian is used.
All positive-count likelihood updates are performed in log space.  The finite
grid and transition-kernel truncations are explicitly diagnosed; these are
numerical approximations to the exact recursive Bayesian law.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.special import k0

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class GridBayes:
    def __init__(self, q=.01, dt=.05, rate0=20., V0=.05,
                 lo=-40., hi=16., h=.005, nsigma=12.):
        self.q, self.dt, self.h = float(q), float(dt), float(h)
        self.grid = np.arange(round((hi-lo)/h)+1, dtype=float)*h+lo
        self.rate = np.exp(self.grid)
        half = int(math.ceil(nsigma*math.sqrt(q)/h))
        offsets = np.arange(-half, half+1)*h
        self.kernel = np.exp(-offsets**2/(2*q))
        self.kernel /= self.kernel.sum()
        self.p = np.exp(-(self.grid-math.log(rate0))**2/(2*V0))
        self.p /= self.p.sum()
        self.zero_likelihood = np.exp(-dt*self.rate)
        self.log_survival = 0.
        self.max_boundary_loss = 0.
        self.boundary_loss_sum = 0.

    def predict(self):
        # numpy.convolve uses direct convolution, not an FFT dispatcher.
        pred = np.convolve(self.p, self.kernel, mode="same")
        norm = float(pred.sum())
        loss = max(0., 1.-norm)
        self.max_boundary_loss = max(self.max_boundary_loss, loss)
        self.boundary_loss_sum += loss
        pred /= norm
        return pred

    def update_from(self, pred, count):
        if count == 0:
            post = pred*self.zero_likelihood
            norm = float(post.sum())
            self.log_survival += math.log(norm)
            post /= norm
        else:
            # Do not add a machine-epsilon floor: it invents a false tail.
            logprior = np.full_like(pred, -np.inf)
            pos = pred > 0
            logprior[pos] = np.log(pred[pos])
            lw = logprior+count*self.grid-self.dt*self.rate
            post = np.exp(lw-np.max(lw))
            post /= post.sum()
        self.p = post

    def step(self, count):
        pred = self.predict()
        prediction_mean = float(pred @ self.rate)
        self.update_from(pred, count)
        return prediction_mean

    def stats(self, p=None):
        p = self.p if p is None else p
        mean = float(p @ self.rate)
        mean_theta = float(p @ self.grid)
        var_theta = float(p @ ((self.grid-mean_theta)**2))
        j = int(np.searchsorted(np.cumsum(p), .5))
        return dict(mean_rate=mean, median_rate=float(self.rate[j]),
                    theta_mode_rate=float(self.rate[int(np.argmax(p))]),
                    theta_mean=mean_theta, theta_variance=var_theta,
                    mass_below_minus35=float(p[self.grid < -35].sum()))


def stationary_variance(q, lam, dt):
    # V^2+qV=q/(lambda dt).
    return 2*q/(lam*dt)/(math.sqrt(q*q+4*q/(lam*dt))+q)


def silence(grid, n, checkpoints=()):
    rows = []
    checkpoint_set = set(checkpoints)
    previous = math.inf
    max_increase = -math.inf
    for k in range(n):
        pred = grid.predict()
        m = float(pred @ grid.rate)
        if math.isfinite(previous):
            max_increase = max(max_increase, m-previous)
        previous = m
        if k in checkpoint_set:
            s = grid.stats(pred)
            rows.append(dict(n_zeros=k, time=k*grid.dt, **s,
                             t_times_mean=k*grid.dt*m,
                             ratio_to_gaussian_6=(m*grid.q*grid.dt*k*k/6
                                                  if k else math.nan),
                             log_survival=grid.log_survival,
                             max_boundary_loss=grid.max_boundary_loss,
                             boundary_loss_sum=grid.boundary_loss_sum))
        grid.update_from(pred, 0)
    if n in checkpoint_set:
        pred = grid.predict()
        m = float(pred @ grid.rate)
        if math.isfinite(previous):
            max_increase = max(max_increase, m-previous)
        rows.append(dict(n_zeros=n, time=n*grid.dt, **grid.stats(pred),
                         t_times_mean=n*grid.dt*m,
                         ratio_to_gaussian_6=m*grid.q*grid.dt*n*n/6,
                         log_survival=grid.log_survival,
                         max_boundary_loss=grid.max_boundary_loss,
                         boundary_loss_sum=grid.boundary_loss_sum))
    return rows, max_increase


def ct_survival_hazard(t, qc=.2, rate0=20., V0=.05, gh_order=64):
    """Exact CT survival/hazard via the Matsumoto--Yor spectral integral.

    K_{i nu}(z)=integral exp(-z cosh u) cos(nu u) du allows the Gaussian
    spectral integral to be done analytically.  Integrate the resulting real
    expression in u. For the supplied t>=20 cases the integrand is well
    conditioned. This is independent of the finite-bin grid approximation.
    """
    s = qc*t/8
    if s < .2:
        raise ValueError('The real CT quadrature is cancellation-prone for qc*t/8 < .2; use a different representation or arbitrary precision.')
    a = math.pi/2
    gh_x, gh_w = hermgauss(gh_order)
    xs = math.log(rate0)+math.sqrt(2*V0)*gh_x
    gh_w = gh_w/math.sqrt(math.pi)
    I = J = A0 = M2 = 0.
    for x, weight in zip(xs, gh_w):
        z = math.sqrt(8*math.exp(x)/qc)
        # Factor e^{-z} to keep each integral away from underflow.
        umax = math.acosh(1+80/z)
        def f(u):
            return math.exp(-z*(math.cosh(u)-1)-u*u/(4*s))*math.cos(a*u/(2*s))
        def df(u):
            phase = a*u/(2*s)
            return math.exp(-z*(math.cosh(u)-1)-u*u/(4*s))*(
                u*u/(4*s*s)*math.cos(phase)+a*u/(2*s*s)*math.sin(phase))
        scale = float(weight)*math.exp(-z)
        I += scale*quad(f, 0., umax, epsabs=1e-13, epsrel=1e-12)[0]
        J += scale*quad(df, 0., umax, epsabs=1e-13, epsrel=1e-12)[0]
        A0 += float(weight)*float(k0(z))
        M2 += scale*quad(lambda u:u*u*math.exp(-z*(math.cosh(u)-1)),
                          0., umax, epsabs=1e-13, epsrel=1e-12)[0]
    survival = math.exp(a*a/(4*s))*I/math.sqrt(math.pi*s)
    hazard = qc/8*(1/(2*s)+a*a/(4*s*s)-J/I)
    B = math.pi**2/(2*qc)-2*M2/(qc*A0)
    lead_survival = math.sqrt(8/(math.pi*qc*t))*A0
    return dict(t=t, qc=qc, rate0=rate0, V0=V0, survival=survival,
                survival_leading=lead_survival, hazard=hazard,
                t_hazard=t*hazard, hazard_leading=.5/t,
                B=B, hazard_second=.5/t+B/(t*t),
                gaussian_silence_median_leading=6/(qc*t*t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Skip the n=20000 grid-domain diagnostic.")
    args = ap.parse_args()
    start = time.perf_counter()
    notes = {}

    configs = [(.005, -40., 12.), (.0025, -40., 12.),
               (.005, -40., 8.), (.005, -40., 16.), (.005, -60., 12.)]
    convergence = []
    base_p = None
    for h, lo, nsigma in configs:
        g = GridBayes(h=h, lo=lo, nsigma=nsigma)
        rows, inc = silence(g, 1200, [0, 100, 300, 1000, 1200])
        prior_stats = g.stats(g.predict())
        g.step(50)
        st = g.stats()
        nextmean = float(g.predict() @ g.rate)
        convergence.append(dict(h=h, lo=lo, hi=16., kernel_sigmas=nsigma,
                                prior_mean=prior_stats['mean_rate'],
                                prior_median=prior_stats['median_rate'],
                                posterior_mean=st['mean_rate'],
                                posterior_mode=st['theta_mode_rate'],
                                posterior_median=st['median_rate'],
                                next_predictive_mean=nextmean,
                                max_silence_predictive_increase=inc,
                                max_boundary_loss=g.max_boundary_loss))
        if (h, lo, nsigma) == configs[0]:
            write_csv("p5_exact_silence_f2.csv", rows)
            base_p = GridBayes()
            silence(base_p, 1200)
    write_csv("p5_grid_convergence.csv", convergence)
    assert base_p is not None
    notes['F2_posterior_mean'] = convergence[0]['posterior_mean']
    notes['F2_next_predictive_mean'] = convergence[0]['next_predictive_mean']
    notes['grid_halving_relative_error'] = abs(
        convergence[1]['posterior_mean']/convergence[0]['posterior_mean']-1)

    # The exact pre-wake prior has a much steeper right tail than the
    # moment-matched/filter Gaussian.  Quantify its score at posterior modes.
    tailrows=[]
    prewake=base_p.predict()
    saved=base_p.p.copy()
    for count in (1,10,50,100):
        base_p.update_from(prewake,count)
        j=int(np.argmax(base_p.p))
        score=(math.log(prewake[j+1])-math.log(prewake[j-1]))/(2*base_p.h)
        tailrows.append(dict(count=count,posterior_theta_mode=base_p.grid[j],
                             posterior_mode_rate=base_p.rate[j],
                             prior_logdensity_score_at_mode=score,
                             likelihood_score_at_mode=count-.05*base_p.rate[j],
                             log_prior_at_mode=math.log(prewake[j]),
                             posterior_mean=base_p.stats()['mean_rate'],
                             mean_exceeds_true_rate_1000=base_p.stats()['mean_rate']>1000))
    base_p.p=saved
    write_csv('p5_exact_tail_score.csv',tailrows)
    # A deliberately extreme count checks when a conventional Gaussian
    # transition truncation becomes inaccurate after likelihood reweighting.
    tail_kernel_rows=[]
    for nsigma in (12.,16.,24.,32.):
        g=GridBayes(nsigma=nsigma)
        silence(g,1200)
        g.step(500)
        tail_kernel_rows.append(dict(kernel_sigmas=nsigma,count=500,
                                     posterior_mean=g.stats()['mean_rate'],
                                     posterior_mode=g.stats()['theta_mode_rate'],
                                     finite_grid_hi=16.,h=.005))
    write_csv('p5_tail_kernel_boundary.csv',tail_kernel_rows)

    # Counts C=50 each bin give a deterministic, fully specified recovery trace.
    recovery_rows = []
    initial = base_p.p.copy()
    rng = np.random.default_rng(3)
    for label, counts in [('constant50',np.full(30,50)),
                          ('poisson_seed3',rng.poisson(50,30))]:
        base_p.p = initial.copy()
        first = None
        for j, count in enumerate(counts, start=1):
            premean = base_p.step(int(count))
            st = base_p.stats()
            within = (1000/1.5 <= st['mean_rate'] <= 1500)
            if first is None and within:
                first = j
            recovery_rows.append(dict(trace=label, wake_bin=j, count=int(count),
                                      predictive_mean=premean, **st,
                                      within_factor_1p5=within))
        notes[f'{label}_first_posterior_mean_in_band_bin'] = first
    write_csv('p5_wake_recovery.csv', recovery_rows)

    # Same initial condition as exact_wake.cpp, explicitly separated from F2.
    refs = []
    for q in (.001,.01):
        v0 = stationary_variance(q, 1000., .05)
        g = GridBayes(q=q, rate0=1000., V0=v0, h=.004, nsigma=8.)
        silence(g, 1200)
        g.step(50)
        refs.append(dict(q=q, rate0=1000., V0=v0, L=60., count=50,
                         posterior_mean=g.stats()['mean_rate'],
                         next_predictive_mean=float(g.predict() @ g.rate)))
    write_csv('p5_cpp_initialization_reference.csv', refs)

    # Counterexample to universal monotonic decrease, and boundary q -> 0.
    monotonic_rows=[]
    for rate0 in (1e-6,.01,20.):
        g=GridBayes(rate0=rate0)
        pred=g.predict(); m0=float(pred @ g.rate)
        zero_mean=float((pred*g.rate*g.zero_likelihood).sum()/(pred*g.zero_likelihood).sum())
        g.update_from(pred,0)
        m1=float(g.predict() @ g.rate)
        monotonic_rows.append(dict(rate0=rate0, prediction0=m0,
                                  posterior_after_zero=zero_mean,
                                  prediction1=m1, ratio=m1/m0,
                                  exact_identity=math.exp(.01/2)*zero_mean,
                                  decreases=m1<=m0))
    write_csv('p5_monotonicity_counterexample.csv',monotonic_rows)

    # Critical initial moment condition for the CT identity m'=qc*m/2-Var(lambda).
    boundary_rows=[]
    qc,V0=.2,.05
    critical_rate=qc/(2*math.exp(V0/2)*math.expm1(V0))
    for multiplier in (.9,1.,1.1):
        rate0=critical_rate*multiplier
        for dt in (.001,.0005):
            g=GridBayes(q=qc*dt,dt=dt,rate0=rate0,V0=V0,h=.001)
            m0=float(g.p @ g.rate)
            v0=float(g.p @ ((g.rate-m0)**2))
            g.step(0)
            m1=float(g.p @ g.rate)
            boundary_rows.append(dict(critical_rate=critical_rate,
                                      multiplier=multiplier,dt=dt,
                                      theoretical_derivative=qc*m0/2-v0,
                                      finite_step_derivative=(m1-m0)/dt))
    write_csv('p5_ct_moment_boundary.csv',boundary_rows)

    # Exact CT spectral calculation and its independently derived asymptotics.
    ctrows=[]
    for qc in (.002,.2,2.):
        times=(1000.,10000.,100000.,1000000.) if qc==.002 else (20.,60.,100.,500.,1000.,10000.)
        for t in times:
            ctrows.append(ct_survival_hazard(t,qc=qc))
    write_csv('p5_ct_spectral_asymptotics.csv',ctrows)
    spectralchecks=[]
    for t in (20.,60.,1000.):
        r=ct_survival_hazard(t)
        eps=t*1e-5
        sp=ct_survival_hazard(t+eps)['survival']
        sm=ct_survival_hazard(t-eps)['survival']
        fd=-(math.log(sp)-math.log(sm))/(2*eps)
        gh32=ct_survival_hazard(t,gh_order=32)['hazard']
        spectralchecks.append(dict(t=t,hazard=r['hazard'],
                                    centered_logsurvival_difference=fd,
                                    relative_fd_error=abs(fd/r['hazard']-1),
                                    gh32_hazard=gh32,
                                    relative_gh_error=abs(gh32/r['hazard']-1)))
    write_csv('p5_ct_spectral_checks.csv',spectralchecks)

    if not args.quick:
        longrows=[]
        for lo in (-40.,-80.):
            g=GridBayes(lo=lo)
            rows, inc=silence(g,20000,[1200,3000,6000,10000,20000])
            for row in rows:
                row['grid_lo']=lo
                row['h']=.005
                row['kernel_sigmas']=12.
            longrows.extend(rows)
        write_csv('p5_long_silence_grid.csv',longrows)

    # Small-bin comparison to CT at t=60, keeping qc=.2 fixed.
    dtrows=[]
    for dt in (.05,.02,.01):
        g=GridBayes(q=.2*dt,dt=dt)
        n=int(round(60/dt)); silence(g,n)
        dtrows.append(dict(dt=dt,q=.2*dt,n=n,
                           predictive_mean=float(g.predict() @ g.rate),
                           posterior_mean=g.stats()['mean_rate'],
                           ct_exact_hazard=ct_survival_hazard(60)['hazard']))
    write_csv('p5_dt_ct_check.csv',dtrows)

    notes['elapsed_seconds']=time.perf_counter()-start
    notes['method']='direct numpy.convolve; log-space positive-count update; no FFT or epsilon floor'
    notes['limitations']='Finite-grid/finite-kernel results; CT asymptotics are not asserted as fixed-bin constants.'
    (OUT/'p5_run_summary.json').write_text(json.dumps(notes,indent=2)+'\n')
    print(json.dumps(notes,indent=2),flush=True)


if __name__=='__main__':
    main()
