"""冻结配置：箱宽、合约、数据日、方法、调参网格与种子。"""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil

HERE=Path(__file__).resolve().parent
# 编译器与 Binance 原始数据目录可由环境变量指定。
COMPILER=Path(os.environ.get('GXX') or shutil.which('g++') or 'g++')
RAW=Path(os.environ.get('BINANCE_RAW',HERE.parent/'data'/'raw'))
DT=[.001,.01,.05,.25,1.]
QC=[.02,.2]
LENGTHS=[5,20,60,300,1200]
RATES=[1,10,100,1000]
TRAIN_DAYS=['2026-10-10','2026-10-11']
TEST_DAYS=['2026-10-17','2026-10-18']
SYMBOLS=[s+'USDT' for s in ['SNDK','MU','TSLA','NVDA','SPY','SPX','QQQ','MSTR','COIN','AAPL']]
METHODS=['E','I','E_cap','I_cap','OU_I']
TUNE_Q=[10.**(-6+.5*j) for j in range(17)]
CAPS=[.0625,.125,.25,.5,1.,2.]
KAPPAS=[1e-4,1e-3,1e-2,1e-1,1.]
N_SW=64
N_CTRL=32
WORKERS=6
MEMORY_LIMIT=8_000_000_000
WALL_LIMIT=21600
BASE={'gate':80_000_000,'test':82_000_000}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE): raise PermissionError('写入超出工作目录')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    raw=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with temp.open('wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def guard_phase(phase):
    if phase not in ('gate','test'): raise ValueError('阶段无效')

def seed_id(index,rep,phase):
    guard_phase(phase)
    if not 0<=index<=20 or not 0<=rep<64: raise ValueError('种子索引越界')
    return BASE[phase]+1000*index+rep

def sw_cells():
    return [dict(id=f'SW{i}',L=L,rate=rate,index=i) for i,(L,rate) in enumerate((L,r) for L in LENGTHS for r in RATES)]

def gaussian_configs(qc,cap=1.):
    return [dict(method=m,qc=qc,cap=cap,mu=0.,kappa=0.) for m in METHODS[:4]]

def x_covered(cell,dt,qc):
    plan=json.loads((HERE/'x_coverage.json').read_text(encoding='utf-8'))
    return [cell['L'],cell['rate'],qc,dt] in plan['x_sw_cells']

def x_spec(qc,dt):
    q=qc*dt
    return dict(lo=-200.,hi=16.,h=min(.025,math.sqrt(q)/8),nsigma=32.)
