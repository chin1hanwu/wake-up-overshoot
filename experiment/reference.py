"""独立 NumPy/SciPy 参考核：Wright omega、直接 numpy.convolve。"""
import math
import numpy as np
from scipy.special import wrightomega,gammaln,logsumexp

def step(theta,V,count,dt,cfg):
    q=cfg['qc']*dt;v=V+q;tp=theta
    if cfg['method']=='OU_I':
        rho=math.exp(-cfg['kappa']*dt);tp=cfg['mu']+rho*(theta-cfg['mu']);v=rho*rho*V+q
    if cfg['method'].endswith('_cap'):v=min(v,cfg.get('cap',1.))
    if cfg['method'].startswith('E'):
        lr=math.log(v)+tp+math.log(dt)
        if lr>=0:e=math.exp(-lr);gain=v*e/(1+e);d=1/(1+e)
        else:e=math.exp(lr);gain=v/(1+e);d=e/(1+e)
        return tp+gain*count-d,gain,tp,v
    z=math.log(v*dt)+tp+v*count;w=float(wrightomega(z))
    t=tp+v*count-w if z < -2 else math.log(w)-math.log(v*dt)
    return t,v/(1+w),tp,v

def signed_difference(loga,logb):
    if loga==logb:return dict(sign=0,log10=0.)
    hi=max(loga,logb);lo=min(loga,logb)
    return dict(sign=1 if loga>logb else -1,log10=(hi+math.log(-math.expm1(lo-hi)))/math.log(10))

def run(counts,dt,cfg,theta=0.,V=1.,truth=None,start=0):
    n=len(counts);trace=np.empty((n,10));ld=math.log(dt)
    for k,c in enumerate(counts):
        theta,V,tp,v=step(theta,V,int(c),dt,cfg);lm=tp+ld
        linear=c*lm-math.lgamma(int(c)+1)
        with np.errstate(over='ignore'):rate=float(np.exp(tp));mass=float(np.exp(lm))
        sc=signed_difference(math.log(linear) if linear>0 else -math.inf,lm) if linear>0 else dict(sign=-1,log10=float(np.logaddexp(lm,math.log(-linear) if linear<0 else -math.inf))/math.log(10))
        trace[k]=tp,v,rate,linear-mass,theta,V,tp+v/2,sc['sign'],sc['log10'],lm
    # 独立的向量 logsumexp 聚合，与 C++ 在线累积器不同。
    linear=np.asarray(counts[start:])*trace[start:,9]-gammaln(np.asarray(counts[start:])+1)
    f=float(linear.sum());masslog=float(logsumexp(trace[start:,9]))
    if f<=0:score=dict(sign=-1,log10=float(np.logaddexp(masslog,math.log(-f) if f else -math.inf))/math.log(10))
    else:score=signed_difference(math.log(f),masslog)
    oracle=0.
    if truth is not None:
        tt=np.asarray(truth[start:]);cc=np.asarray(counts[start:]);ll=np.zeros(len(tt));pos=cc>0
        if (tt[pos]<=0).any():raise ValueError('真零有计数')
        ll[pos]=cc[pos]*np.log(tt[pos]);oracle=float(np.sum(ll-tt-gammaln(cc+1)))
    finite=oracle-f
    loss=dict(sign=1,log10=float(np.logaddexp(masslog,math.log(finite) if finite>0 else -math.inf))/math.log(10)) if finite>=0 else signed_difference(masslog,math.log(-finite))
    if np.max(trace[start:,9])<600:
        # 独立扩展精度向量聚合，避免两个普通double大和相减。
        mass=np.exp(trace[start:,9].astype(np.longdouble)).sum(dtype=np.longdouble)
        ff=linear.sum(dtype=np.longdouble)
        if truth is not None:
            oo=(ll-tt-gammaln(cc+1)).sum(dtype=np.longdouble)
        else:oo=np.longdouble(0)
        def encode(v):return dict(sign=1 if v>0 else -1 if v<0 else 0,log10=float(np.log10(abs(v))) if v else 0.)
        score=encode(ff-mass);loss=encode(oo-ff+mass)
    return dict(trace=trace,score=score,loss=loss,theta=theta,V=V)

class Grid:
    def __init__(self,dt,q,theta,V,lo=-150.,hi=16.,h=.02,nsigma=12.):
        self.dt=dt;self.x=lo+np.arange(round((hi-lo)/h)+1)*h;self.rate=np.exp(self.x)
        half=math.ceil(nsigma*math.sqrt(q)/h);a=np.arange(-half,half+1)*h
        self.kernel=np.exp(-a*a/(2*q));self.kernel/=self.kernel.sum()
        self.p=np.exp(-(self.x-theta)**2/(2*V));self.p/=self.p.sum();self.lo=lo
    def step(self,count):
        pred=np.convolve(self.p,self.kernel,mode='same');pred/=pred.sum()
        mean=float(pred@self.rate);tm=float(pred@self.x);tv=float(pred@((self.x-tm)**2))
        lp=np.full(len(pred),-np.inf);pos=pred>0;lp[pos]=np.log(pred[pos])
        lw=lp+count*(self.x+math.log(self.dt))-self.dt*self.rate-math.lgamma(count+1)
        score=float(logsumexp(lw));self.p=np.exp(lw-score)
        pt=float(self.p@self.x);pv=float(self.p@((self.x-pt)**2));pm=float(self.p@self.rate)
        return np.array([mean,score,tm,tv,pm,pt,pv,float(pred[self.x<self.lo+5].sum())])
