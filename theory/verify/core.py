"""Unclipped scalar filters. Units: dt seconds, qc variance/second, q=qc*dt.

The stable formulas are algebraically identical to the task's recurrences.
The primary readout is exp(theta); prediction precedes scoring and updating.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import wrightomega

def explicit_step(theta, V, count, q, dt):
    v=V+q
    log_r=math.log(v)+theta+math.log(dt)
    if log_r>=0:
        er=math.exp(-log_r)
        gain=v*er/(1+er)
        decrement=1/(1+er)
    else:
        er=math.exp(log_r)
        gain=v/(1+er)
        decrement=er/(1+er)
    return theta+gain*count-decrement,gain

def implicit_step(theta, V, count, q, dt):
    v=V+q
    log_z=math.log(v*dt)+theta+v*count
    if log_z<-700:
        w=math.exp(log_z) if log_z>-745 else 0.
        t=theta+v*count-w
        return t,v/(1+w)
    w=float(wrightomega(log_z))
    t=math.log(w)-math.log(v*dt)
    return t,v/(1+w)

def silence(n,q,dt,theta=math.log(20.),V=.05,method='E'):
    step=explicit_step if method=='E' else implicit_step
    for _ in range(int(n)):
        theta,V=step(theta,V,0,q,dt)
    return theta,V

def ct_flow(theta,W,duration,qc=.2,rtol=2e-10,atol=1e-12):
    if duration<=0:return theta,W
    def rhs(t,y):
        lam=math.exp(y[0]);v=y[1]
        return [-v*lam,qc-v*v*lam]
    sol=solve_ivp(rhs,(0,duration),(theta,W),rtol=rtol,atol=atol,method='DOP853')
    if not sol.success:raise RuntimeError(sol.message)
    return float(sol.y[0,-1]),float(sol.y[1,-1])

def ct_events(theta,W,event_times,horizon,qc=.2):
    ts=[];states=[];t0=0.
    for t in event_times:
        theta,W=ct_flow(theta,W,float(t)-t0,qc)
        theta+=W
        ts.append(float(t));states.append((theta,W));t0=float(t)
    theta,W=ct_flow(theta,W,horizon-t0,qc)
    return theta,W,np.asarray(ts),np.asarray(states).reshape(-1,2)

def predictive_loss(theta,count,Lambda,dt):
    """Realized oracle log score minus plug-in Poisson score (can be negative)."""
    log_mu=theta+math.log(dt);m=Lambda*dt
    mu=math.exp(log_mu) if log_mu<709 else math.inf
    return mu-m-count*(theta-math.log(Lambda))
