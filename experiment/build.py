"""编译 C++ 计算核：g++ -O2 -std=c++17 -static -Wall -Wextra，零诊断。"""
import os
import subprocess
from config import HERE,COMPILER,atomic_json

def environment():
    return dict(os.environ,PATH=str(COMPILER.parent)+os.pathsep+os.environ.get('PATH',''))
def build():
    kwargs=dict(cwd=HERE,capture_output=True,text=True,env=environment(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    v=subprocess.run([str(COMPILER),'--version'],check=True,**kwargs)
    command=[str(COMPILER),'-O2','-std=c++17','-static','-Wall','-Wextra',str(HERE/'engine.cpp'),'-o',str(HERE/'engine.exe')]
    p=subprocess.run(command,**kwargs)
    info=dict(command=command,compiler_version=v.stdout,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    atomic_json(HERE/'build_info.json',info)
    if p.returncode or p.stdout or p.stderr: raise RuntimeError('编译诊断非空或编译失败')
    return info
if __name__=='__main__': build();print('构建完成，零诊断')
