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
