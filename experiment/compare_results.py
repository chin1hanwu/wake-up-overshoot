"""Compare a rerun's results.json with the published one, ignoring per-call process metadata."""
import json
import sys

def clean(o):
    if isinstance(o,dict):return {k:clean(v) for k,v in o.items() if k not in ('meta','source_sha256')}
    if isinstance(o,list):return [clean(v) for v in o]
    return o

def main():
    new,old=[clean(json.load(open(p,encoding='utf-8'))) for p in sys.argv[1:3]]
    parts=['synthetic'] if new.get('real') is None else ['synthetic','real']
    ok=True
    for part in parts:
        same=new[part]==old[part];ok&=same
        print(part,'identical' if same else 'DIFFERENT')
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
