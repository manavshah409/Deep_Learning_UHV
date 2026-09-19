"""Run exactly two independent pilots, recording actual process exit codes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/accuracy_stage_b'
record=OUT/'process_status.json'
if record.exists(): raise SystemExit('Process record already exists; refusing to overwrite')
status={'started_utc':datetime.now(timezone.utc).isoformat(),'processes':[]}
record.write_text(json.dumps(status,indent=2)+'\n')
for method in ['unweighted','weighted']:
    run_id=f'E3_stageB_{method}_3ep_seed42_v1'
    log=ROOT/'runs'/f'{run_id}.log'
    with log.open('x') as stream:
        command=[sys.executable,'-m','src.experiments.stage_b','--sampling',method,'--run-id',run_id]
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        stream.flush();os.fsync(stream.fileno())
    status['processes'].append(dict(run_id=run_id,returncode=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat()))
    record.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status['processes'][-1]),flush=True)
    if result.returncode: raise SystemExit(result.returncode)
