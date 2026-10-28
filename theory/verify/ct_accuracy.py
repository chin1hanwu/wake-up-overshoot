"""Check the Euler time-step used by the supplied CT reference."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from core import ct_flow,ct_events

OUT=Path(__file__).resolve().parents[1]/'outputs'
def main():
    ev=np.loadtxt(OUT/'p3_ct_events.csv',delimiter=',',skiprows=1)
    qc=.2;L=60.;T=.05
    th0,W0=ct_flow(math.log(20),.05,L,qc)
    end,w,_,states=ct_events(th0,W0,ev,T,qc)
    truepeak=math.exp(float(states[:,0].max()))
    rows=[]
    for h in [1e-4,5e-5,1e-5,2e-6,5e-7]:
        # Reference uses Euler also during silence. This has negligible effect
        # relative to the event timing/ODE error after wake; reproduce h=1e-4.
        if h==1e-4:
            th,W=math.log(20),.05;t=0.
            while t<L:
                lam=math.exp(th);th-=W*lam*h;W+=(qc-W*W*lam)*h;t+=h
        else:th,W=th0,W0
        tt=0.;si=0;peak=math.exp(th)
        while tt<T:
            while si<len(ev) and ev[si]<=tt+h:
                th+=W;si+=1
            lam=math.exp(th);peak=max(peak,lam)
            th-=W*lam*h;W+=(qc-W*W*lam)*h;tt+=h
        rows.append(dict(euler_h=h,euler_peak=peak,adaptive_peak=truepeak,relative_peak_error=peak/truepeak-1,euler_end=math.exp(th),adaptive_end=math.exp(end),relative_end_error=math.exp(th-end)-1))
    pd.DataFrame(rows).to_csv(OUT/'p3_ct_reference_accuracy.csv',index=False)
    print('ct_accuracy: reference Euler bias and convergence saved.')
if __name__=='__main__':main()
