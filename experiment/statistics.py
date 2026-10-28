"""扩展损失全序、精确符号检验、IUT、Holm 与配对 t 检验。"""
import math
import numpy as np
from scipy.stats import binomtest,t
from scipy.special import logsumexp

def validate(x):
    if x['sign'] not in [-1,0,1] or not math.isfinite(x['log10']):raise ValueError('扩展数格式无效')
    if x['sign']==0 and x['log10']!=0:raise ValueError('零的规范表示错误')
    return x

def compare(a,b):
    validate(a);validate(b)
    if a['sign']!=b['sign']:return 1 if a['sign']>b['sign'] else -1
    if a['sign']==0 or a['log10']==b['log10']:return 0
    return a['sign']*(1 if a['log10']>b['log10'] else -1)

def signed_sum(values):
    values=[validate(v) for v in values]
    pos=[v['log10']*math.log(10) for v in values if v['sign']==1]
    neg=[v['log10']*math.log(10) for v in values if v['sign']==-1]
    a=float(logsumexp(pos)) if pos else -math.inf;b=float(logsumexp(neg)) if neg else -math.inf
    if a==b:return dict(sign=0,log10=0.)
    return dict(sign=1 if a>b else -1,log10=(max(a,b)+math.log(-math.expm1(min(a,b)-max(a,b))))/math.log(10))

def as_float(x):
    validate(x)
    if x['log10']>308:raise ValueError('该检验要求有限普通损失')
    return x['sign']*10.**x['log10'] if x['sign'] else 0.

def from_float(x):
    if not math.isfinite(x):raise ValueError('非有限输入')
    return dict(sign=1 if x>0 else -1 if x<0 else 0,log10=math.log10(abs(x)) if x else 0.)

def sign_test(a,b):
    if len(a)!=len(b) or not a:raise ValueError('配对缺失')
    signs=[compare(x,y) for x,y in zip(a,b)];wins=signs.count(-1);losses=signs.count(1);ties=signs.count(0);n=wins+losses
    return dict(p=float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.,wins=wins,losses=losses,ties=ties,n=n)

def iut(parts):
    if not parts:raise ValueError('空 IUT')
    return max(float(p['p']) for p in parts)

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('p 值无效')
    order=np.argsort(p,kind='stable');out=np.ones(len(p));last=0.
    for rank,j in enumerate(order):last=max(last,(len(p)-rank)*p[j]);out[j]=min(1.,last)
    return out.tolist()

def region(pvalues,domain,alpha=.05/3):
    if len(pvalues)!=len(domain):raise ValueError('区域维度不匹配')
    p=holm(pvalues);passed=[x<alpha for x in p]
    return dict(verdict='全区域成立' if passed and all(passed) else '部分格成立' if any(passed) else '不成立',confirmed=[x for x,b in zip(domain,passed) if b],domain=domain,p_raw=pvalues,p_holm=p,alpha=alpha)

def stratified(groups):
    groups=[np.asarray(x,dtype=float) for x in groups]
    if not groups or any(len(x)<2 or not np.isfinite(x).all() for x in groups):raise ValueError('样本不足或非有限')
    H=len(groups);mean=float(np.mean([x.mean() for x in groups]));terms=[float(x.var(ddof=1)/len(x)/H**2) for x in groups];variance=sum(terms)
    if variance==0:return dict(mean=mean,se=0.,df=None,p=0. if mean<0 else 1.)
    df=variance**2/sum(v*v/(len(x)-1) for x,v in zip(groups,terms));se=math.sqrt(variance)
    return dict(mean=mean,se=se,df=df,p=float(t.cdf(mean/se,df)))

def noninferiority(a,b,margin=.02):
    a=np.asarray(a);b=np.asarray(b)
    test=stratified([a-(1+margin)*b]);positive=stratified([-b])
    return dict(p=iut([test,positive]),contrast=test,positive_reference=positive)

def descriptives(values):
    from functools import cmp_to_key
    values=sorted([validate(x) for x in values],key=cmp_to_key(compare));n=len(values)
    # 最近秩分位数不将极大损失转换为有限 double；四分位差也保留扩展表示。
    quant=lambda p:values[max(0,math.ceil(n*p)-1)]
    q1,med,q3=quant(.25),quant(.5),quant(.75)
    neg=dict(sign=-q1['sign'],log10=q1['log10'])
    mean=signed_sum(values)
    if mean['sign']:mean=dict(sign=mean['sign'],log10=mean['log10']-math.log10(n))
    return dict(n=n,median=med,q1=q1,q3=q3,iqr=signed_sum([q3,neg]),mean_descriptive=mean,catastrophe_fraction=sum(compare(v,dict(sign=1,log10=3.))>0 for v in values)/n)

def analyze_synthetic(records,regions,n_sw=64,n_ctrl=32):
    from config import sw_cells,DT,QC,BASE
    lookup={}
    expected=set()
    for c in sw_cells():
        for dt in DT:
            for q in QC:
                for i in range(n_sw):
                    seed=BASE['test']+1000*c['index']+i
                    for m in ['E','I','E_cap','I_cap']:expected.add((c['id'],dt,q,seed,m))
    for dt in DT:
        for q in QC:
            for i in range(n_ctrl):
                for m in ['E','I','E_cap','I_cap']:expected.add(('CTRL',dt,q,BASE['test']+20000+i,m))
    for r in records:
        key=(r['cell'],r['dt'],r['qc'],r['seed'],r['method'])
        if key in lookup:raise ValueError('重复摘要')
        validate(r['loss']);lookup[key]=r['loss']
    if set(lookup)!=expected:raise ValueError('摘要缺失、额外项或种子不匹配')
    def vals(row,m):
        cell=row['cell'];index=20 if cell=='CTRL' else int(cell[2:]);n=n_ctrl if cell=='CTRL' else n_sw
        return [lookup[cell,row['dt'],row['qc'],BASE['test']+1000*index+i,m] for i in range(n)]
    H1=[];H2=[];H3=[];descriptive=[]
    for row in regions:
        if row['region']=='越率区':H1.append(dict(cell=row,**sign_test(vals(row,'I'),vals(row,'E'))))
        if row['H2']:
            parts=[sign_test(vals(row,a),vals(row,b)) for a,b in [('I_cap','I'),('E_cap','E')]]
            H2.append(dict(cell=row,p=iut(parts),parts=parts))
        descriptive.append(dict(cell=row,methods={m:descriptives(vals(row,m)) for m in ['E','I','E_cap','I_cap']}))
    for dt in DT:
        for q in QC:
            row=dict(cell='CTRL',dt=dt,qc=q)
            parts=[noninferiority([as_float(v) for v in vals(row,a)],[as_float(v) for v in vals(row,b)]) for a,b in [('I','E'),('I_cap','I')]]
            H3.append(dict(cell=row,p=iut(parts),parts=parts))
    out={h:dict(**region([r['p'] for r in rows],[r['cell'] for r in rows]),tests=rows) for h,rows in [('H1',H1),('H2',H2),('H3',H3)]}
    out['descriptive']=descriptive
    return out
