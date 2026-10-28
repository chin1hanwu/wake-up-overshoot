"""连续时间事件生成：SW 静默-唤醒流与 CTRL 对照流。"""
import math
import numpy as np
from config import guard_phase,seed_id

def sw(cell,rep=0,phase='gate'):
    guard_phase(phase);seed=seed_id(cell['index'],rep,phase);rng=np.random.default_rng(seed)
    rate=cell['rate'];L=cell['L'];T=420+L
    pre=np.sort(rng.uniform(0,300,rng.poisson(300*rate)))
    post=np.sort(rng.uniform(300+L,T,rng.poisson(120*rate)))
    return dict(events=np.concatenate([pre,post]),duration=T,seed=seed,kind='SW',L=L,rate=rate)

def base_log(t):
    return .5*math.log(1000)*(1-np.cos(np.pi*np.asarray(t)/150))

def ctrl(rep=0,phase='gate'):
    guard_phase(phase);seed=seed_id(20,rep,phase);rng=np.random.default_rng(seed)
    T=1800.;times=[0.];levels=[0.];t=0.;lev=0.
    while True:
        t+=rng.exponential(12.5)
        if t>=T:break
        lev=float(rng.choice([-.5,.5])) if lev==0 else 0.
        times.append(t);levels.append(lev)
    times=np.array(times+[T]);levels=np.array(levels)
    # 各跳变区间用全局严格上界 thinning，事件时刻连续。
    pieces=[];bound=1000*math.exp(.5)
    for a,b,level in zip(times[:-1],times[1:],levels):
        n=rng.poisson(bound*(b-a));candidate=rng.uniform(a,b,n)
        accept=rng.uniform(0,bound,n)<np.exp(base_log(candidate)+level)
        pieces.append(candidate[accept])
    return dict(events=np.sort(np.concatenate(pieces)),duration=T,seed=seed,kind='CTRL',jump_times=times,jump_levels=levels)

def bin_events(data,dt):
    n=round(data['duration']/dt)
    index=np.floor(data['events']/dt).astype(np.int64)
    if (index<0).any() or (index>=n).any():raise ValueError('事件时间越界')
    return np.bincount(index,minlength=n).astype(np.int32)

def oracle_means(data,dt):
    n=round(data['duration']/dt)
    if data['kind']=='SW':
        a=np.full(n,data['rate']*dt);a[round(300/dt):round((300+data['L'])/dt)]=0.;return a
    # 五点 Gauss–Legendre；含跳变的箱单独按跳变分段积分。
    nodes,weights=np.polynomial.legendre.leggauss(5);left=np.arange(n)*dt
    mean=np.zeros(n)
    for z,w in zip(nodes,weights):
        ts=left+(z+1)*dt/2;ix=np.searchsorted(data['jump_times'],ts,side='right')-1
        mean+=dt*w/2*np.exp(base_log(ts)+data['jump_levels'][ix])
    for k in sorted(set(np.floor(data['jump_times'][1:-1]/dt).astype(int))):
        a=k*dt;b=a+dt
        breaks=[a]+[float(t) for t in data['jump_times'] if a<t<b]+[b];s=0.
        for aa,bb in zip(breaks[:-1],breaks[1:]):
            mid=(aa+bb)/2;lev=data['jump_levels'][np.searchsorted(data['jump_times'],mid,side='right')-1]
            s+=(bb-aa)/2*float(weights@np.exp(base_log(mid+nodes*(bb-aa)/2)+lev))
        mean[k]=s
    return mean
