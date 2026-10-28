#!/usr/bin/env python3
"""P1/P6 verification: exact E/I zero-count recursions and CT invariant.

No clipping and no FFT.  Lambert W is evaluated by SciPy's principal branch.
The finite-N constants in the CSV are estimates, not certified intervals.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def silence(method, q, dt, rate0, v0, max_n, snapshots):
    v, s = float(v0), float(rate0 * dt)
    rows = []
    min_log_mean = math.log(rate0) + v / 2
    min_n = 0
    turns = []
    prev_sign = None
    a = (-1 if method == "E" else 1) * 2 * q / 15
    for n in range(1, max_n + 1):
        old_v, old_s = v, s
        prior = v + q
        r = prior * s
        if method == "E":
            d = r / (1 + r)
            v = prior / (1 + r)
        else:
            d = float(lambertw(r).real)
            v = prior / (1 + d)
        s *= math.exp(-d)
        log_mean = math.log(s / dt) + v / 2
        # Algebraic increment is preferable to subtracting two huge log means.
        change = -d + (v - old_v) / 2
        if log_mean < min_log_mean:
            min_log_mean, min_n = log_mean, n
        sign = change > 0
        if prev_sign is not None and sign != prev_sign:
            turns.append(n)
        prev_sign = sign
        if n in snapshots:
            x = q / v
            y = s * v * v / q
            k = -38 / 45 if method == "E" else -22 / 45
            rows.append(dict(
                method=method, q=q, dt=dt, rate0=rate0, V0=v0, n=n,
                V=v, rate=s / dt, a_log=a,
                b_est=v - q*n/3 - a*math.log(n),
                leading_V_ratio=v/(q*n/3),
                leading_rate_ratio=s/(6/(q*n*n)),
                y=y, y_minus_two_thirds_over_x=(y-2/3)/x,
                predicted_y_coefficient=k,
                log_mean=log_mean,
                mean=math.exp(log_mean) if log_mean < 700 else "overflow_not_evaluated",
                exact_mean_log_increment=change,
                algebraic_turn_test=(old_v+q)*old_s*(old_v+2) < q
                    if method == "E" else 2*d*d+(old_v+2)*d < q,
            ))
            assert sign == rows[-1]["algebraic_turn_test"]
    b_ref = v - q*max_n/3 - a*math.log(max_n)
    gamma = -19/5 if method == "E" else -11/5
    for row in rows:
        n = row["n"]
        vp = q*n/3+a*math.log(n)+b_ref
        lp = math.log(6/(q*dt*n*n)) + (
            -6*a/q*math.log(n)-6*b_ref/q+gamma)/n
        row.update(
            reference_N=max_n, reference_b_est=b_ref,
            V_second_order=vp, V_second_order_abs_error=abs(vp-row["V"]),
            rate_second_order_log_error=lp-math.log(row["rate"]),
        )
    return rows, dict(method=method, q=q, dt=dt, rate0=rate0, V0=v0,
                     min_n=min_n, min_time=min_n*dt,
                     leading_turn_n=12/q, min_log_mean=min_log_mean,
                     sign_change_bins=";".join(map(str, turns)),
                     reference_N=max_n, reference_b_est=b_ref)


def ct_cases():
    rows = []
    qc = .2
    for name, rate0, v0 in [("baseline",20,.05), ("zero_initial_variance",20,0),
                             ("exact_self_similar",20,math.sqrt(2*qc/(3*20))),
                             ("large_initial_variance",20,2)]:
        k = (v0*v0-2*qc/(3*rate0))/(rate0*rate0)
        c = 2*qc/3
        integral = quad(lambda z: z**1.5 / math.sqrt(max(c+k*z**3, 1e-300)),
                        0, rate0, epsabs=1e-12, epsrel=1e-12)[0]
        b = v0-k*integral
        ts = np.array([1.,5.,20.,60.,100.,1000.,10000.])
        def rhs(t, state):
            theta, w = state
            rate = math.exp(theta)
            return [-w*rate, qc-w*w*rate]
        sol = solve_ivp(rhs, (0,float(ts[-1])), [math.log(rate0),v0],
                        t_eval=ts, rtol=2e-12, atol=2e-13, method="DOP853")
        assert sol.success
        for t,theta,w in zip(ts,sol.y[0],sol.y[1]):
            rate = math.exp(theta)
            eta = abs(k*rate**3/c)
            bound = 2*abs(k)/(5*math.sqrt(c*(1-eta)))*rate**2.5 if eta<1 else math.nan
            inv_rel = abs(w*w-(c/rate+k*rate*rate))/max(1,w*w)
            rows.append(dict(case=name,qc=qc,rate0=rate0,W0=v0,t=t,W=w,rate=rate,
                             K=k,B=b,time_shift=3*b/qc,
                             W_shift_approx=qc*t/3+b,
                             W_shift_abs_error=abs(w-(qc*t/3+b)),
                             rigorous_analytic_tail_bound=bound,
                             eta=eta,
                             rate_shift_approx=6/(qc*(t+3*b/qc)**2),
                             invariant_relative_residual=inv_rel,
                             numerical_tolerance_note="ODE residual can exceed tiny analytic tail"))
        assert max(r["invariant_relative_residual"] for r in rows if r["case"]==name)<1e-8
    return rows


def cap_and_ou():
    rows=[]
    for q,dt in [(.01,.05),(.0002,.001)]:
        for rate in [.1,1,20,1000]:
            s=rate*dt
            astar=(q+math.sqrt(q*q+4*q/s))/2
            vstar=astar-q
            residual=abs(vstar-astar/(1+astar*s))
            rows.append(dict(kind="frozen_prior_cap",q=q,dt=dt,rate=rate,
                             rho="",constant=astar,residual=residual))
    for rho in [0,.9,.999]:
        q=.01; v=.05; bound=max(v,q/(1-rho*rho))
        max_v=v
        for _ in range(100000):
            # No information is the largest possible variance trajectory.
            v=rho*rho*v+q
            max_v=max(max_v,v)
        rows.append(dict(kind="OU_no_information_bound",q=q,dt="",rate="",
                         rho=rho,constant=bound,residual=max(0,max_v-bound)))
        assert max_v<=bound+1e-10
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-n",type=int,default=1000000)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    snaps={n for n in [100,1200,10000,20000,100000,500000,args.max_n] if n<=args.max_n}
    cases=[("baseline",.01,.05,20,.05),
           ("small_q",.0001,.05,20,.05),
           ("large_q",.1,.05,20,.05),
           ("zero_initial_variance",.01,.05,20,0),
           ("low_initial_rate",.01,.05,.01,.05),
           ("nonuniform_small_q",1e-8,.05,20,.05)]
    rows=[]; turns=[]
    for case,q,dt,rate,v0 in cases:
        for method in ["E","I"]:
            local,turn=silence(method,q,dt,rate,v0,args.max_n,snaps)
            for row in local: row["case"]=case
            turn["case"]=case
            rows.extend(local);turns.append(turn)
    write_csv("p1_discrete_asymptotics.csv",rows)
    write_csv("p1_mean_turning_points.csv",turns)
    write_csv("p1_ct_invariant.csv",ct_cases())
    write_csv("p1_cap_ou.csv",cap_and_ou())
    notes={"max_n":args.max_n,"cases":len(cases),"methods":["E","I"],
           "claim":"b_est is a finite-N estimate; no q-uniform remainder is claimed",
           "boundary_case":"q=1e-8 deliberately tests pre-asymptotic nonuniformity",
           "checks":"exact algebraic mean-turn conditions and CT invariant passed"}
    (OUT/"p1_run.json").write_text(json.dumps(notes,indent=2)+"\n")
    print(json.dumps(notes))


if __name__=="__main__": main()
