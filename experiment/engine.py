"""本目录 C++ 核的二进制接口；所有进程等待退出并记录身份。"""
import os
from pathlib import Path
import struct
import subprocess
import time
import uuid
import threading
import numpy as np
import psutil
from config import HERE,COMPILER,METHODS,atomic_json

def stationary(q,rate,dt):
    return 2*q/(rate*dt)/(np.sqrt(q*q+4*q/(rate*dt))+q)

def _paths(folder):
    folder=(HERE/folder).resolve()
    if not folder.is_relative_to(HERE):raise PermissionError('核临时文件超出工作目录')
    folder.mkdir(parents=True,exist_ok=True)
    tag=uuid.uuid4().hex;inp=folder/(tag+'.in');out=folder/(tag+'.out')
    if any(p.exists() for p in [inp,out,out.with_suffix('.process.json')]):
        raise FileExistsError('核调用标识冲突；不覆盖既有文件')
    return inp,out

def _read_output(out,size):
    # 按字节核对，避免 fromfile 忽略尾部不足一个 double 的损坏。
    if out.stat().st_size!=size*8:raise ValueError('计算核输出长度不符')
    a=np.fromfile(out,dtype='<f8')
    if a.size!=size:raise ValueError('计算核输出截断')
    return a

def _validate_trace(trace,gaussian):
    # 分块验证以免为全天诊断再分配一个同尺寸数组；合法率溢出仍保留。
    for i in range(0,len(trace),65536):
        block=trace[i:i+65536]
        if gaussian:
            cols=[0,1,4,5,6,7,8,9]
            valid=np.isfinite(block[:,cols]).all() and not np.isnan(block).any()
            valid=valid and not np.isneginf(block[:,2]).any() and not np.isposinf(block[:,3]).any()
            valid=valid and (block[:,[1,5]]>=0).all() and np.isin(block[:,7],[-1,0,1]).all()
            valid=valid and (block[block[:,7]==0,8]==0).all()
        else:valid=np.isfinite(block).all()
        if not valid:raise ValueError('计算核逐箱输出完整性校验失败')

def _finish(inp,out,meta):
    # 仅清理本次调用创建且完整读回、校验成功的两个文件；失败路径不进入此函数。
    meta['io']=dict(input_bytes=inp.stat().st_size,output_bytes=out.stat().st_size,
                    validated=True,temp_files_removed=False)
    meta['io']['temporary_peak_bytes']=meta['io']['input_bytes']+meta['io']['output_bytes']
    atomic_json(out.with_suffix('.process.json'),meta)
    inp.unlink();out.unlink()
    meta['io']['temp_files_removed']=True
    atomic_json(out.with_suffix('.process.json'),meta)

def invoke(inp,out,label):
    env=dict(os.environ,PATH=str(COMPILER.parent)+os.pathsep+os.environ.get('PATH',''))
    began=time.perf_counter()
    p=subprocess.Popen([str(HERE/'engine.exe'),str(inp),str(out)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    peak={'child_rss':0,'child_private':0,'combined_rss':0,'combined_private':0}
    done=threading.Event()
    parent=psutil.Process()
    def sample():
        while not done.is_set():
            try:
                a=psutil.Process(p.pid).memory_info();b=parent.memory_info()
                for key,val in [('child_rss',a.rss),('child_private',getattr(a,'private',a.vms)),('combined_rss',a.rss+b.rss),('combined_private',getattr(a,'private',a.vms)+getattr(b,'private',b.vms))]:peak[key]=max(peak[key],val)
            except psutil.Error: pass
            done.wait(.01)
    t=threading.Thread(target=sample,daemon=True);t.start()
    stdout,stderr=p.communicate();done.set();t.join()
    meta=dict(pid=p.pid,purpose=label,seconds=time.perf_counter()-began,returncode=p.returncode,stderr=stderr.decode('utf-8',errors='replace'),stdout=stdout.decode('utf-8',errors='replace'),peak=peak,status='exited')
    atomic_json(out.with_suffix('.process.json'),meta)
    if p.returncode or stdout or stderr: raise RuntimeError('计算核失败：'+str(out.with_suffix('.process.json')))
    return meta

def run(counts,dt,configs,theta=0.,V=1.,truth=None,start=0,keep=False,wake=-1,maxtruth=0.,folder='scratch',label='Gaussian 单流'):
    counts=np.asarray(counts,dtype='<i4');n=len(counts)
    truth=np.full(n,-1.,dtype='<f8') if truth is None else np.asarray(truth,dtype='<f8')
    if truth.shape!=(n,):raise ValueError('真值维数错误')
    inp,out=_paths(folder)
    with inp.open('xb') as f:
        f.write(struct.pack('<6i4d',202610211,n,len(configs),start,int(keep),wake,dt,theta,V,maxtruth))
        f.write(counts.tobytes());f.write(truth.tobytes())
        for c in configs:f.write(struct.pack('<i4d',METHODS.index(c['method']),c['qc'],c.get('cap',1.),c.get('mu',0.),c.get('kappa',0.)))
    meta=invoke(inp,out,label)
    width=12+(n*10 if keep else 0);a=_read_output(out,len(configs)*width)
    result=[]
    for c,row in zip(configs,a.reshape(len(configs),width)):
        if not np.isfinite(row[:11]).all() or row[9]<0:raise ValueError('计算核摘要完整性校验失败')
        for j in [0,2]:
            if row[j] not in [-1,0,1] or (row[j]==0 and row[j+1]!=0):raise ValueError('计算核扩展数无效')
        if keep:_validate_trace(row[12:].reshape(n,10),True)
        item=dict(config=c,score={'sign':int(row[0]),'log10':float(row[1])},loss={'sign':int(row[2]),'log10':float(row[3])},max_log_rate=float(row[4]),first_wake_log_rate=float(row[5]),recovery_seconds=None if row[6]<0 else float(row[6]),clipped_score=float(row[7]),theta=float(row[8]),V=float(row[9]),oracle_score=float(row[10]))
        if keep:item['trace']=row[12:].reshape(n,10).copy()
        result.append(item)
    _finish(inp,out,meta)
    return result,meta

def run_grid(counts,dt,q,theta,V,lo=-150.,hi=16.,h=.02,nsigma=12.,start=0,keep=False,folder='scratch'):
    counts=np.asarray(counts,dtype='<i4');inp,out=_paths(folder)
    with inp.open('xb') as f:
        f.write(struct.pack('<4i8d',202610212,len(counts),start,int(keep),dt,q,theta,V,lo,hi,h,nsigma));f.write(counts.tobytes())
    meta=invoke(inp,out,'X 直接卷积')
    a=_read_output(out,8+(len(counts)*8 if keep else 0))
    if not np.isfinite(a[:8]).all() or a[6]!=round(a[6]) or a[7]!=round(a[7]) or a[6]<3 or not 0<=a[7]<a[6]:raise ValueError('X 摘要完整性校验失败')
    if keep:_validate_trace(a[8:].reshape(len(counts),8),False)
    item=dict(score=float(a[0]),posterior_mean=float(a[1]),posterior_variance=float(a[2]),max_lower5_mass=float(a[3]),max_boundary_loss=float(a[4]),boundary_loss_sum=float(a[5]),grid_points=int(a[6]),kernel_halfwidth=int(a[7]),lo=lo,hi=hi,h=h,kernel_sigmas=nsigma)
    if keep:item['trace']=a[8:].reshape(len(counts),8).copy()
    _finish(inp,out,meta)
    return item,meta
