"""Binance aggTrades 日文件读取：合约与数据日白名单、结构检查、可选 CHECKSUM 校验。"""
import csv
import io
from pathlib import Path
from datetime import datetime,timezone
import zipfile
import numpy as np
from config import RAW,HERE,TRAIN_DAYS,TEST_DAYS,SYMBOLS,digest,atomic_json

def authorize(symbol,day,phase):
    if symbol not in SYMBOLS:raise ValueError('合约不在冻结名单')
    if day in TEST_DAYS:
        if phase!='test':raise PermissionError('测试日须以 test 阶段读取')
    elif day not in TRAIN_DAYS or phase!='train':raise PermissionError('数据日或阶段不在白名单')
    return RAW/f'{symbol}-aggTrades-{day}.zip'

def load(symbol,day,phase='train',expected_hash=None):
    path=authorize(symbol,day,phase)
    h=digest(path)
    official=path.with_name(path.name+'.CHECKSUM')
    if official.exists() and official.read_text(encoding='utf-8').split()[0]!=h:raise ValueError('Binance CHECKSUM 校验失败')
    if expected_hash is not None and h!=expected_hash:raise ValueError('与训练清单记录的文件不同')
    start=round(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
    stamps=[];last_id=None;header=False
    with zipfile.ZipFile(path) as z:
        members=z.namelist()
        if len(members)!=1 or not members[0].endswith('.csv'):raise ValueError('zip 结构异常')
        with z.open(members[0]) as raw:
            rows=csv.reader(io.TextIOWrapper(raw,encoding='utf-8-sig',newline=''))
            for j,row in enumerate(rows):
                if j==0:
                    try:int(row[0]);int(row[5])
                    except ValueError:header=True;continue
                if len(row)!=7:raise ValueError('列数异常')
                agg=int(row[0]);ms=int(row[5])-start
                if not 0<=ms<86400000:raise ValueError('时间戳不在 UTC 数据日内')
                if last_id is not None and agg<=last_id:raise ValueError('aggTrade ID 重复或乱序')
                if stamps and ms<stamps[-1]:raise ValueError('时间乱序')
                stamps.append(ms);last_id=agg
    a=np.asarray(stamps,dtype=np.int64)
    return a,dict(symbol=symbol,day=day,file=path.name,sha256=h,events=len(a),header=header,unit='transact_time 毫秒；每个 aggTrade 一个事件')

def counts(stamps,dt):
    ms=round(dt*1000)
    if dt not in [.01,.05,.25,1.]:raise ValueError('真实数据箱宽不在冻结集合')
    return np.bincount(stamps//ms,minlength=86400000//ms).astype(np.int32)

def prepare():
    rows=[];means={s:[] for s in SYMBOLS};folder=HERE/'train_cache';folder.mkdir(exist_ok=True)
    for s in SYMBOLS:
        for d in TRAIN_DAYS:
            a,r=load(s,d);rows.append(r);means[s].append(len(a)/86400.)
            # 只缓存事件毫秒时刻，不复制或分发原始 CSV。
            p=folder/f'{s}_{d}.npy'
            with p.open('wb') as f:np.save(f,a,allow_pickle=False)
    import math
    contract_log_rates={s:math.log(sum(v)/len(v)) for s,v in means.items()}
    mu=sum(contract_log_rates.values())/len(contract_log_rates)
    manifest=dict(files=rows,mu=mu,contract_log_rates=contract_log_rates)
    atomic_json(HERE/'training_manifest.json',manifest)
    return manifest
if __name__=='__main__':m=prepare();print('训练日完成校验：',len(m['files']),'文件')
