"""Fixed-feature multivariate projection identity, including zero-diffusion edge."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng=np.random.default_rng(51);rows=[]
    for d in [1,2,8]:
        for rank in [0,1,d]:
            for _ in range(10):
                A=rng.normal(size=(d,d));P=A@A.T+.01*np.eye(d)
                x=rng.normal(size=d);B=rng.normal(size=(d,rank));Q=B@B.T*.01
                prior=P+Q;h=x@prior@x;rate=np.exp(rng.normal());dt=.05;C=int(rng.poisson(5))
                cov=np.linalg.inv(np.linalg.inv(prior)+rate*dt*np.outer(x,x))
                projected=float(x@cov@x);scalar=h/(1+h*rate*dt)
                delta=cov@x*(C-rate*dt)
                err=abs(projected-scalar);meanerr=abs(x@delta-scalar*(C-rate*dt))
                assert err<1e-9 and meanerr<1e-8
                rows.append(dict(d=d,rank_Q=rank,q_projected=x@Q@x,cov_abs_error=err,mean_abs_error=meanerr))
    out=Path(__file__).resolve().parents[1]/'outputs'
    pd.DataFrame(rows).to_csv(out/'p7_projection.csv',index=False)
    print('p7_projection: matrix/scalar identities passed, including Q=0 boundary.')
if __name__=='__main__':main()
