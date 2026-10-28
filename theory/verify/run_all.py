#!/usr/bin/env python3
"""Run every independent verification entry point with recorded outputs."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=["p1_silence.py","p2_wake.py","p3_binning.py","ct_accuracy.py",
         "p4_implicit.py","p5_bayes.py","p7_projection.py","make_figures.py"]
def main():
    out=ROOT/"outputs";out.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
               MPLCONFIGDIR="/tmp/theory_round2_mpl_cache")
    os.environ["MPLCONFIGDIR"]=env["MPLCONFIGDIR"]
    records=[];start=time.perf_counter()
    with (out/"run_all.log").open("w",encoding="utf8") as log:
        for script in SCRIPTS:
            before=time.perf_counter()
            print("Running "+script,flush=True)
            p=subprocess.run([sys.executable,str(ROOT/"verify"/script)],cwd=ROOT,
                             env=env,text=True,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
            log.write("\n### "+script+"\n"+p.stdout);log.flush()
            records.append(dict(script=script,exit_code=p.returncode,
                                seconds=time.perf_counter()-before))
            if p.returncode:
                print(p.stdout,flush=True)
                break
    versions={m:importlib.import_module(m).__version__
              for m in ["numpy","scipy","pandas","matplotlib"]}
    result=dict(success=len(records)==len(SCRIPTS) and all(r["exit_code"]==0 for r in records),
                python=sys.version,versions=versions,seconds=time.perf_counter()-start,
                scripts=records,csv_outputs=len(list(out.glob("*.csv"))))
    (out/"run_all_status.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)
    if not result["success"]:sys.exit(1)
if __name__=="__main__":main()
