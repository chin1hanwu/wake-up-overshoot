"""只用 E 零计数递推冻结理论区域，不使用任何实验数据。"""
import math
import numpy as np
from config import HERE,sw_cells,DT,QC,atomic_json
from engine import run,stationary

def main():
    rows=[]
    for cell in sw_cells():
        for qc in QC:
            for dt in DT:
                q=qc*dt;V0=float(stationary(q,cell['rate'],dt));n=round(cell['L']/dt)
                res,_=run(np.zeros(n,dtype=np.int32),dt,[dict(method='E',qc=qc)],theta=math.log(cell['rate']),V=V0,folder='regions_io',label='无数据理论区域公式')
                th=res[0]['theta'];v=res[0]['V']+q;s=dt*math.exp(th);g=v/(1+v*s);d=g*s
                kappa=g*cell['rate']*dt-math.log(cell['rate'])+th-d
                region='越率区' if kappa>=math.log(10) else '不越区' if kappa<=-math.log(10) else '过渡区'
                rows.append(dict(cell=cell['id'],L=cell['L'],rate=cell['rate'],qc=qc,dt=dt,n_silence=n,initial_V=V0,theta_minus=th,lambda_minus=math.exp(th),v=v,g=g,d=d,kappa=kappa,region=region,H1=region=='越率区',H2=dt in [.001,.01] and qc*cell['L']/3>=8))
    atomic_json(HERE/'theory_regions.json',dict(rule='kappa>=log10 / <=-log10 / 过渡',counts={r:sum(x['region']==r for x in rows) for r in ['越率区','不越区','过渡区']},H1_count=sum(r['H1'] for r in rows),H2_count=sum(r['H2'] for r in rows),rows=rows))
    print('公式区域冻结：',len(rows),'格')
if __name__=='__main__':main()
