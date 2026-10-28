"""完整实验入口：合成流（SW 200 格 + CTRL）与 Binance 测试日；可断点恢复。"""
import argparse
import json
import math
import os
import time
from config import HERE,WORKERS,N_SW,N_CTRL,DT,QC,sw_cells,atomic_json,seed_id

def synthetic_job(args):
    cell,rep,h=args
    from generator import sw,ctrl,bin_events,oracle_means
    from engine import run,run_grid,stationary
    from config import gaussian_configs,x_covered,x_spec
    out=HERE/'runs'/h/'synthetic';out.mkdir(parents=True,exist_ok=True)
    seed=seed_id(20 if cell is None else cell['index'],rep,'test')
    resource=json.loads((HERE/'x_coverage.json').read_text(encoding='utf-8'))
    name='CTRL' if cell is None else cell['id'];dst=out/f'{name}_{seed}.json'
    if dst.exists():
        saved=json.loads(dst.read_text(encoding='utf-8'))
        if saved['seed']!=seed:raise ValueError('恢复检查失败')
        expected={(dt,q,m) for dt in DT for q in QC for m in ['E','I','E_cap','I_cap']}
        expected|={(dt,q,'X') for dt in DT for q in QC if (cell is None and dt in resource['x_ctrl_dt']) or (cell is not None and x_covered(cell,dt,q))}
        actual={(r['dt'],r['qc'],r['method']) for r in saved['records']}
        if actual!=expected or len(saved['records'])!=len(expected):raise ValueError('恢复摘要不完整')
        return saved['records']
    data=ctrl(rep,'test') if cell is None else sw(cell,rep,'test');rows=[]
    for dt in DT:
        c=bin_events(data,dt);truth=oracle_means(data,dt)
        for qc in QC:
            rate=1. if cell is None else cell['rate'];v=float(stationary(qc*dt,rate,dt))
            wake=-1 if cell is None else round((300+cell['L'])/dt)
            start=round((300 if cell is None else 60)/dt)
            res,meta=run(c,dt,gaussian_configs(qc),theta=math.log(rate),V=v,truth=truth,start=start,wake=wake,maxtruth=1000*math.exp(.5) if cell is None else rate,folder=f'runs/{h}/io',label=f'合成 {name} {seed}')
            for r in res:
                conf=r['config'];vp=r['V']+qc*dt;tp=r['theta']
                if conf['method'].endswith('_cap'):vp=min(vp,conf.get('cap',1.))
                r['next_predictive_lognormal_mean_log_rate']=tp+vp/2
                rows.append(dict(cell=name,seed=seed,dt=dt,qc=qc,method=r['config']['method'],**{k:v for k,v in r.items() if k!='config'},meta=meta))
            if (cell is not None and x_covered(cell,dt,qc)) or (cell is None and dt in resource['x_ctrl_dt']):
                x,xmeta=run_grid(c,dt,qc*dt,math.log(rate),v,**x_spec(qc,dt),start=start,folder=f'runs/{h}/io')
                from statistics import from_float
                rows.append(dict(cell=name,seed=seed,dt=dt,qc=qc,method='X',loss=from_float(res[0]['oracle_score']-x['score']),grid=x,meta=xmeta))
    atomic_json(dst,dict(seed=seed,records=rows))
    return rows

def real_run(h):
    import numpy as np
    from binance_loader import load,counts
    from config import SYMBOLS,TEST_DAYS
    from engine import run
    from statistics import as_float,sign_test,stratified,holm
    frozen=json.loads((HERE/'frozen_params.json').read_text(encoding='utf-8'));rows=[]
    for symbol in SYMBOLS:
        for day in TEST_DAYS:
            p=HERE/'runs'/h/'real'/f'{symbol}_{day}.json'
            if p.exists():
                saved=json.loads(p.read_text(encoding='utf-8'))
                expected={(dt,m) for dt in [.01,.05,.25,1.] for m in ['E','I','E_cap','I_cap','OU_I']}
                if {(r['dt'],r['method']) for r in saved['records']}!=expected or len(saved['records'])!=20:raise ValueError('真实恢复缺失或重复')
                rows.extend(saved['records']);continue
            a,source=load(symbol,day,'test');block=[]
            for dt in [.01,.05,.25,1.]:
                C=counts(a,dt);start=round(3600/dt);mask=np.zeros(len(C),dtype=bool)
                wake=a[1:][np.diff(a)>=20000]
                for ms in wake:
                    lo=int(ms//round(dt*1000));hi=min(len(C),int(math.ceil((ms+5000)/(dt*1000))))
                    mask[lo:hi]=True
                mask[:start]=False
                dailymax=int(counts(a,1.).max())
                for cfg in [r for r in frozen['selected'] if r['dt']==dt]:
                    res,meta=run(C,dt,[cfg],theta=frozen['initial_theta'],V=frozen['initial_V'],start=start,keep=True,folder=f'runs/{h}/real_io',label=f'真实 {symbol} {day} {dt} {cfg["method"]}')
                    r=res[0];trace=r.pop('trace')
                    # 真实数据没有 oracle：这里报告负预测对数得分的贡献。
                    from scipy.special import logsumexp,gammaln
                    from statistics import signed_sum,from_float
                    lm=trace[mask,9];linear=C[mask]*lm-gammaln(C[mask]+1)
                    masslog=float(logsumexp(lm)) if len(lm) else -math.inf
                    contribution=signed_sum(([dict(sign=1,log10=masslog/math.log(10))] if math.isfinite(masslog) else [])+[from_float(-float(linear.sum()))])
                    over=int(np.count_nonzero(trace[start:,0]>math.log(100*dailymax))) if dailymax>0 else 0
                    block.append(dict(symbol=symbol,day=day,dt=dt,method=cfg['method'],score=r['score'],silence_wake_5s_negative_logscore=contribution,large_rate_bins=over,daily_max_1s_count=dailymax,meta=meta))
                    del trace
            atomic_json(p,dict(source=source,records=block));rows.extend(block)
    table={(r['symbol'],r['day'],r['dt'],r['method']):r for r in rows}
    expected={(s,d,dt,m) for s in SYMBOLS for d in TEST_DAYS for dt in [.01,.05,.25,1.] for m in ['E','I','E_cap','I_cap','OU_I']}
    if set(table)!=expected or len(rows)!=len(expected):raise ValueError('真实摘要缺失或重复')
    units=[(s,d) for s in SYMBOLS for d in TEST_DAYS]
    def values(dt,m):return [table[s,d,dt,m]['score'] for s,d in units]
    # 得分大者好；R1/R2 调换符号检验两端以保持备择为 a<b。
    r1=sign_test(values(1.,'E'),values(1.,'I'));r2=sign_test(values(.25,'E'),values(.25,'I'))
    I=np.array([as_float(x) for x in values(1.,'I')]);cap=np.array([as_float(x) for x in values(1.,'I_cap')])
    r3=stratified([I-cap-.005*np.abs(I)])
    adjusted=holm([r1['p'],r2['p'],r3['p']])
    decisions={name:dict(p_holm=p,confirmed=p<.05,verdict='成立' if p<.05 else '不成立') for name,p in zip(['R1','R2','R3'],adjusted)}
    return dict(records=rows,R1=r1,R2=r2,R3=r3,p_holm=adjusted,decisions=decisions,alpha=.05,R3_definition=f'score_I-score_Icap-0.005*abs(score_I)<0；{len(units)}个配对日单位')

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=WORKERS);p.add_argument('--run',default='reproduction');p.add_argument('--synthetic-only',action='store_true');args=p.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    if args.workers<1:raise ValueError('并发数无效')
    h=args.run;out=HERE/'runs'/h;out.mkdir(parents=True,exist_ok=True);lock=out/'active.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),run=h,workers=args.workers),f)
    tasks=[(cell,rep,h) for cell in sw_cells() for rep in range(N_SW)]+[(None,rep,h) for rep in range(N_CTRL)]
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(synthetic_job,tasks,chunksize=1):records.extend(block)
    write_results(h,records,real=not args.synthetic_only)
    os.replace(lock,out/f'completed_{os.getpid()}_{time.time_ns()}.lock')
    print(str(out/'results.json'))

def write_results(h,records,real=True):
    """汇总合成结果、运行真实数据并写出 results.json。"""
    from statistics import analyze_synthetic
    regions=json.loads((HERE/'theory_regions.json').read_text(encoding='utf-8'))['rows']
    synthetic=analyze_synthetic([r for r in records if r['method']!='X'],regions,N_SW,N_CTRL)
    synthetic['X_descriptive']=[r for r in records if r['method']=='X']
    from statistics import descriptives,signed_sum
    ctrl_groups={}
    for r in records:
        if r['cell']=='CTRL':ctrl_groups.setdefault((r['dt'],r['qc'],r['method']),[]).append(r['loss'])
    synthetic['CTRL_descriptive']=[dict(dt=dt,qc=q,method=m,**descriptives(v)) for (dt,q,m),v in sorted(ctrl_groups.items())]
    matched={(r['cell'],r['seed'],r['dt'],r['qc'],r['method']):r['loss'] for r in records if r['method']!='X'}
    x_differences={}
    for r in synthetic['X_descriptive']:
        for m in ['E','I','E_cap','I_cap']:
            other=matched[r['cell'],r['seed'],r['dt'],r['qc'],m]
            difference=signed_sum([r['loss'],dict(sign=-other['sign'],log10=other['log10'])])
            x_differences.setdefault((r['cell'],r['dt'],r['qc'],m),[]).append(difference)
    synthetic['X_minus_Gaussian_descriptive']=[dict(cell=c,dt=dt,qc=q,method=m,**descriptives(v)) for (c,dt,q,m),v in sorted(x_differences.items())]
    real=real_run(h) if real else None
    atomic_json(HERE/'runs'/h/'results.json',dict(synthetic=synthetic,real=real))
if __name__=='__main__':main()
