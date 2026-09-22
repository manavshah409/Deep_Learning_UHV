"""Detached, one-shot E3 launcher and read-only status. No shell execution."""
import argparse
import csv
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='E3_fasterrcnn_unweighted_20ep_seed42_v1'


def utc():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def locked(path):
    if not path.exists():return False
    with path.open('r') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    return False


def launch(root,run_id,command,commit):
    if not run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    if (root/'runs'/run_id).exists():raise FileExistsError('Scientific run already exists; inspect it before any resume')
    folder=root/'runs/launches'/run_id
    folder.parent.mkdir(parents=True,exist_ok=True)
    folder.mkdir()  # Atomic reservation, never overwrite even a finished launch.
    lock=(folder/'launcher.lock').open('x+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(folder/'request.json',dict(run_id=run_id,command=command,commit=commit,requested_utc=utc()))
    try:
        with (folder/'supervisor.log').open('x') as output:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--supervise',str(folder),'--lock-fd',str(lock.fileno())],
                cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,pass_fds=(lock.fileno(),))
        atomic(folder/'supervisor_pid.json',dict(pid=p.pid,launched_utc=utc()))
    finally:lock.close()  # Child keeps the inherited flock for its full lifetime.
    return dict(status='dispatched',supervisor_pid=p.pid,launch_directory=str(folder.relative_to(root)))


def supervise(folder,fd):
    request=read(folder/'request.json');code=None
    try:
        with (folder/'stdout_stderr.log').open('x') as output:
            process=subprocess.Popen(request['command'],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,close_fds=True)
            atomic(folder/'pid.json',dict(pid=process.pid,supervisor_pid=os.getpid(),launch_utc=utc(),commit=request['commit']))
            code=process.wait()  # Preserve actual command status, including nonzero exits.
            output.flush();os.fsync(output.fileno())
        atomic(folder/'exit.json',dict(exit_code=code,finished_utc=utc(),command=request['command']))
    except BaseException as exc:
        atomic(folder/'supervisor_failure.json',dict(error=str(exc),utc=utc(),known_exit_code=code))
        raise
    finally:os.close(fd)


def status(root,run_id):
    run=root/'runs'/run_id;folder=root/'runs/launches'/run_id
    state=read(run/'run_state.json');pid=read(folder/'pid.json');exit_info=read(folder/'exit.json')
    run_active=locked(run/'.run.lock');supervisor_active=locked(folder/'launcher.lock')
    active=run_active or supervisor_active
    complete=(run/'COMPLETE.json').exists()
    phase='active' if active else 'completed' if complete and exit_info.get('exit_code',0)==0 else 'failed' if (run/'FAILURE.json').exists() or exit_info.get('exit_code',0)!=0 or (folder/'supervisor_failure.json').exists() else 'stopped/incomplete' if folder.exists() else 'not launched'
    entries=[]
    if (run/'epoch_timing.csv').exists():
        with (run/'epoch_timing.csv').open() as f:entries=list(csv.DictReader(f))
    last=entries[-1] if entries else {}
    completed=state.get('last_completed_epoch',0)
    mean=sum(float(r['total_epoch_seconds']) for r in entries)/len(entries) if entries else 14.86*3600/20
    return dict(status=phase,pid=read(run/'.run.lock').get('pid',pid.get('pid')),launch_utc=pid.get('launch_utc'),
        last_completed_epoch=completed,latest_loss=last.get('total_training_loss'),AP50=last.get('validation_map50'),AP50_95=last.get('validation_map50_95'),
        best_epoch=state.get('best_epoch'),best_AP50_95=state.get('best_metric'),latest_epoch_seconds=last.get('total_epoch_seconds'),
        cumulative_active_seconds=state.get('cumulative_active_seconds',0),estimated_remaining_hours=max(0,20-completed)*mean/3600,
        estimate_note='At last epoch boundary; before first epoch uses Stage B projection',exit_code=exit_info.get('exit_code'))


def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',action='store_true');mode.add_argument('--status',action='store_true');mode.add_argument('--supervise',type=Path)
    p.add_argument('--lock-fd',type=int);a=p.parse_args()
    if a.supervise:supervise(a.supervise,a.lock_fd);return
    if a.status:print(json.dumps(status(ROOT,RUN_ID),indent=2));return
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='master':raise RuntimeError('Expected master')
    # Ignore unrelated untracked Phase 3 work, but require frozen tracked E3 source.
    relevant=['src','scripts','tests/test_epoch_resume.py','tests/test_e3_launcher.py','configs/accuracy']
    if subprocess.check_output(['git','diff','HEAD','--name-only','--',*relevant],cwd=ROOT,text=True).strip():raise RuntimeError('Uncommitted E3 source changes')
    subprocess.check_call(['git','ls-files','--error-unmatch','scripts/launch_e3.py','src/training/epoch_resume.py','src/experiments/stage_b.py'],cwd=ROOT,stdout=subprocess.DEVNULL)
    if shutil.disk_usage(ROOT).free<15_000_000_000:raise RuntimeError('Less than 15 GB free')
    if "Now drawing from 'AC Power'" not in subprocess.check_output(['pmset','-g','batt'],text=True):raise RuntimeError('AC power required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    command=['/usr/bin/caffeinate','-i','-s',sys.executable,'-m','src.experiments.stage_b','--sampling','unweighted','--run-id',RUN_ID,'--allow-full-training']
    print(json.dumps(launch(ROOT,RUN_ID,command,commit)))
if __name__=='__main__':main()
