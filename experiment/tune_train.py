"""穷举训练日网格，按合约×日总对数得分选参；块级原子检查点。"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
from config import HERE,TRAIN_DAYS,SYMBOLS,TUNE_Q,CAPS,KAPPAS,METHODS,WORKERS,atomic_json,digest
from statistics import signed_sum,compare

def candidates(mu):
    out=[]
    for m in METHODS:
        for q in TUNE_Q:
            for cap in CAPS if m.endswith('_cap') else [1.]:
                for kap in KAPPAS if m=='OU_I' else [0.]:
                    out.append(dict(method=m,qc=q,cap=cap,kappa=kap,mu=mu))
    return out

def binding():
    names=['engine.exe','engine.cpp','engine.py','tune_train.py','binance_loader.py','config.py','statistics.py','training_manifest.json']
    return hashlib.sha256(json.dumps({n:digest(HERE/n) for n in names},sort_keys=True).encode()).hexdigest()

def job(args):
    symbol,day,dt,plan=args
    from binance_loader import counts,load
    from engine import run
    manifest=json.loads((HERE/'training_manifest.json').read_text(encoding='utf-8'))
    r=next(x for x in manifest['files'] if x['symbol']==symbol and x['day']==day)
    # 重新校验原训练 zip，缓存还需独立哈希。
    a,_=load(symbol,day,expected_hash=r['sha256'])
    cfg=candidates(manifest['mu']);label=f'{symbol}_{day}_{round(dt*1000)}ms'
    dst=HERE/'tuning'/f'{label}.json'
    if dst.exists():
        saved=json.loads(dst.read_text(encoding='utf-8'))
        if saved['plan_hash']!=plan:raise ValueError('调参检查点绑定改变')
        return saved
    c=counts(a,dt)
    try:
        result,meta=run(c,dt,cfg,theta=manifest['mu'],V=1.,start=round(3600/dt),folder='tuning/io',label='训练日穷举 '+label)
    except Exception:
        atomic_json(HERE/'tuning/failures'/f'{label}_{time.time_ns()}.json',dict(plan_hash=plan,symbol=symbol,day=day,dt=dt,traceback=traceback.format_exc(),candidates=len(cfg)))
        raise
    rows=[]
    for conf,res in zip(cfg,result):
        sc=res['score']
        rows.append(dict(config=conf,extended_score=sc))
    saved=dict(plan_hash=plan,symbol=symbol,day=day,dt=dt,source_sha256=r['sha256'],rows=rows,meta=meta,n_bins=len(c),warmup_bins=round(3600/dt))
    atomic_json(dst,saved)
    return saved

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=WORKERS);parser.add_argument('--dt',type=float,nargs='+',default=[1.,.25,.05,.01]);parser.add_argument('--symbols',nargs='+',choices=SYMBOLS,default=SYMBOLS);args=parser.parse_args()
    if args.workers<1:raise ValueError('并发数无效')
    if len(set(args.dt))!=len(args.dt) or any(x not in [.01,.05,.25,1.] for x in args.dt) or len(set(args.symbols))!=len(args.symbols):raise ValueError('训练批次参数无效')
    folder=HERE/'tuning';folder.mkdir(exist_ok=True);lock=folder/'active.lock'
    # 不自动删除锁；完成重命名留存。
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),purpose='只读训练日全网格调参',workers=args.workers),f)
    plan=binding();manifest=json.loads((HERE/'training_manifest.json').read_text(encoding='utf-8'));began=time.perf_counter()
    atomic_json(folder/'plan.json',dict(plan_hash=plan,candidates=candidates(manifest['mu']),dt=[.01,.05,.25,1.],initial_theta=manifest['mu'],initial_V=1.,warmup_seconds=3600,objective='20个合约×日评分之和最大',tie_break='按方法内 q_c、cap、kappa 升序取首个',workers=args.workers))
    for dt in args.dt:
        tasks=[(s,d,dt,plan) for s in args.symbols for d in TRAIN_DAYS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(job,x) for x in tasks]):
                r=future.result();print(json.dumps(dict(day=r['day'],symbol=r['symbol'],dt=dt,seconds=r['meta']['seconds'])),flush=True)
    rows=[];cfg=candidates(manifest['mu']);boundary=[]
    for dt in [.01,.05,.25,1.]:
        checkpoints=[]
        for s in SYMBOLS:
            for d in TRAIN_DAYS:
                p=folder/f'{s}_{d}_{round(dt*1000)}ms.json'
                if not p.exists():continue
                r=json.loads(p.read_text(encoding='utf-8'))
                if r['plan_hash']!=plan:raise ValueError('检查点绑定不同')
                checkpoints.append(r)
        if len(checkpoints)!=20:continue
        for m in METHODS:
            ids=[j for j,c in enumerate(cfg) if c['method']==m]
            scores=[signed_sum([r['rows'][j]['extended_score'] for r in checkpoints]) for j in ids]
            chosen=0
            for j in range(1,len(ids)):
                if compare(scores[j],scores[chosen])>0:chosen=j
            best=ids[chosen];c=cfg[best]
            edges=[]
            if c['qc'] in [TUNE_Q[0],TUNE_Q[-1]]:edges.append('q_c')
            if m.endswith('_cap') and c['cap'] in [CAPS[0],CAPS[-1]]:edges.append('V_max')
            if m=='OU_I' and c['kappa'] in [KAPPAS[0],KAPPAS[-1]]:edges.append('kappa')
            rows.append(dict(dt=dt,**c,total_train_score=scores[chosen],grid_boundary=edges))
            if edges:boundary.append(dict(dt=dt,method=m,fields=edges))
    if len(rows)==20:
        # 写到 tuning/ 下，便于与随仓库发布的 frozen_params.json 比较。
        atomic_json(folder/'frozen_params.json',dict(mu=manifest['mu'],contract_log_rates=manifest['contract_log_rates'],training_days=TRAIN_DAYS,grid=dict(qc=TUNE_Q,cap=CAPS,kappa=KAPPAS),candidate_counts={m:sum(c['method']==m for c in cfg) for m in METHODS},boundary_choices=boundary,initial_theta=manifest['mu'],initial_V=1.,selected=rows))
    os.replace(lock,folder/f'completed_{os.getpid()}_{time.time_ns()}.lock')
    print(json.dumps(dict(frozen=len(rows)==20,selected=len(rows))),flush=True)
if __name__=='__main__':main()
