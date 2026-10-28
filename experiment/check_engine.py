"""C++ 核与独立 NumPy/SciPy 参考实现逐箱比对（只用 gate 种子，不读真实数据）。"""
import math
import numpy as np
from config import sw_cells,gaussian_configs
from engine import run,run_grid
from generator import sw,ctrl,bin_events,oracle_means
import reference

ATOL=1e-12;RTOL=1e-13

def close(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return False
    if not (np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b))):return False
    if np.isnan(a).any() or np.isnan(b).any():return False
    f=np.isfinite(a)&np.isfinite(b)
    return bool(np.all(np.abs(a[f]-b[f])<=ATOL+RTOL*np.abs(b[f])))

def main():
    results=[]
    for label,data,dt in [('SW',sw(sw_cells()[11]),.05),('CTRL',ctrl(),.05)]:
        C=bin_events(data,dt);truth=oracle_means(data,dt)
        slices=[(0,256)] if label=='CTRL' else [(0,256),(round(299/dt),round(362/dt))]
        for a,b in slices:
            c=C[a:b];tr=truth[a:b];theta=math.log(data.get('rate',20.));V=.05
            cfg=gaussian_configs(.2)+[dict(method='OU_I',qc=.2,mu=1.,kappa=.01)]
            rr,_=run(c,dt,cfg,theta=theta,V=V,truth=tr,keep=True,folder='check_io')
            for conf,res in zip(cfg,rr):
                ref=reference.run(c,dt,conf,theta,V,tr)
                ok=all(close(res['trace'][:,ix],ref['trace'][:,ix]) for ix in [0,1,2,3,4,5,6])
                ok=ok and res['loss']['sign']==ref['loss']['sign'] and abs(res['loss']['log10']-ref['loss']['log10'])<=1e-12
                results.append((f'{label}[{a}:{b}] {conf["method"]}',ok))
            gridargs=dict(dt=dt,q=.2*dt,theta=theta,V=V,lo=-150.,hi=16.,h=.025,nsigma=12.)
            gx,_=run_grid(c,keep=True,folder='check_io',**gridargs)
            g=reference.Grid(**gridargs);trace=np.array([g.step(int(x)) for x in c])
            results.append((f'{label}[{a}:{b}] X',all(close(gx['trace'][:,ix],trace[:,ix]) for ix in range(7))))
    # 极端计数：theta>700 时仍保持状态与扩展数损失。
    c=np.array([2000,0,0,1,0],dtype=np.int32);conf=dict(method='E',qc=.2)
    rr,_=run(c,.05,[conf],theta=-5.,V=10.,truth=np.ones(5)*50,keep=True,folder='check_io')
    ref=reference.run(c,.05,conf,-5.,10.,np.ones(5)*50)
    ok=abs(rr[0]['loss']['log10']-ref['loss']['log10'])<=1e-12 and rr[0]['loss']['sign']==ref['loss']['sign'] and np.max(rr[0]['trace'][:,0])>700
    results.append(('overflow fixture E',ok and close(rr[0]['trace'][:,[0,1,4,5,8]],ref['trace'][:,[0,1,4,5,8]])))
    for name,ok in results:print(('PASS ' if ok else 'FAIL ')+name)
    if not all(ok for _,ok in results):raise SystemExit(1)
    print(f'all {len(results)} checks passed')
if __name__=='__main__':main()
