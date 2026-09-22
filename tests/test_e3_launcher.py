import importlib.util
import json
from pathlib import Path
import sys
import time
import pytest

spec=importlib.util.spec_from_file_location('launcher',Path(__file__).resolve().parents[1]/'scripts/launch_e3.py')
l=importlib.util.module_from_spec(spec);spec.loader.exec_module(l)

@pytest.mark.parametrize('exit_code',[0,7])
def test_detached_exit_and_duplicate_refusal(tmp_path,exit_code):
    name=f'synthetic_{exit_code}'
    command=[sys.executable,'-c',f'import time,sys;print("harmless",flush=True);time.sleep(.4);sys.exit({exit_code})']
    if Path('/usr/bin/caffeinate').exists():command=['/usr/bin/caffeinate','-i','-s',*command]
    result=l.launch(tmp_path,name,command,'synthetic-test')
    folder=tmp_path/result['launch_directory']
    assert result['supervisor_pid']>0 and not (tmp_path/'runs'/name).exists()
    assert l.locked(folder/'launcher.lock')
    with pytest.raises(FileExistsError):l.launch(tmp_path,name,command,'synthetic-test')
    deadline=time.monotonic()+10
    while not (folder/'exit.json').exists() and time.monotonic()<deadline:time.sleep(.05)
    assert json.loads((folder/'exit.json').read_text())['exit_code']==exit_code
    assert 'harmless' in (folder/'stdout_stderr.log').read_text()
    assert l.status(tmp_path,name)['exit_code']==exit_code

def test_existing_scientific_run_refused(tmp_path):
    (tmp_path/'runs/x').mkdir(parents=True)
    with pytest.raises(FileExistsError):l.launch(tmp_path,'x',[sys.executable,'-c','pass'],'test')


def test_detached_resume_same_directory_and_active_refusal(tmp_path):
    run=tmp_path/'runs/recovery';run.mkdir(parents=True)
    command=[sys.executable,'-c','import time;time.sleep(.3)']
    first=l.launch(tmp_path,'recovery',command,'repair',resume=True)
    with pytest.raises(RuntimeError,match='active'):l.launch(tmp_path,'recovery',command,'repair',resume=True)
    folder=tmp_path/first['launch_directory'];deadline=time.monotonic()+10
    while not (folder/'exit.json').exists() and time.monotonic()<deadline:time.sleep(.05)
    assert l.status(tmp_path,'recovery')['exit_code']==0
    assert run.is_dir() and not (run/'COMPLETE.json').exists()
    assert (folder/'request.json').exists()
    (run/'COMPLETE.json').write_text('{}')
    with pytest.raises(ValueError,match='incomplete'):l.launch(tmp_path,'recovery',command,'repair',resume=True)


def test_status_does_not_mix_stale_state_and_new_csv(tmp_path):
    import csv
    run=tmp_path/'runs/stale';run.mkdir(parents=True)
    (run/'run_state.json').write_text(json.dumps(dict(last_completed_epoch=5,best_epoch=4,best_metric=.3)))
    with (run/'epoch_timing.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','total_epoch_seconds','cumulative_seconds','total_training_loss','validation_map50','validation_map50_95'])
        w.writeheader();w.writerow(dict(epoch=6,total_epoch_seconds=10,cumulative_seconds=60,total_training_loss=.4,validation_map50=.5,validation_map50_95=.4))
    result=l.status(tmp_path,'stale')
    assert result['last_completed_epoch']==6 and result['state_metadata_stale'] and result['best_epoch'] is None
    assert result['cumulative_active_seconds']==60
