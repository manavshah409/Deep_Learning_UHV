"""Versioned, fail-closed epoch-boundary recovery. Epoch files are immutable.

Canonical checkpoint + SHA receipt is the recovery source. CSV/state/aliases are
recoverable projections. OS advisory locking, not an unverified PID, owns a run.
"""
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from src.training.epoch_timing import FIELDS

FORMAT_VERSION = 2
REPAIR_POLICY = Path(__file__).resolve().parents[2] / 'configs/accuracy/resume_source_compatibility.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def active_clock():
    # Darwin uptime excludes system sleep; calendar time is recorded separately.
    if hasattr(time, 'CLOCK_UPTIME_RAW'):
        return time.clock_gettime(time.CLOCK_UPTIME_RAW)
    return time.monotonic()


CLOCK_POLICY = 'CLOCK_UPTIME_RAW (excludes sleep)' if hasattr(time, 'CLOCK_UPTIME_RAW') else 'monotonic (platform suspend semantics)'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path); fsync_directory(path.parent)


def capture_rng(device):
    state = np.random.get_state()
    result = dict(python=random.getstate(), numpy=dict(kind=state[0], keys=state[1].tolist(), position=state[2], has_gauss=state[3], cached_gaussian=state[4]), torch_cpu=torch.get_rng_state())
    get = getattr(torch.mps, 'get_rng_state', None)
    put = getattr(torch.mps, 'set_rng_state', None)
    result['mps'] = dict(get_api=callable(get), set_api=callable(put), saved=False, reason='CPU experiment')
    if device == 'mps':
        if callable(get) and callable(put):
            result['mps'].update(saved=True, reason=None, state=get().cpu())
        else:
            result['mps']['reason'] = 'Installed Torch does not expose both MPS RNG APIs; bitwise recovery unavailable'
    return result


def restore_rng(value):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(value['torch_cpu'])
    if value['mps']['saved']:
        setter = getattr(torch.mps, 'set_rng_state', None)
        if not callable(setter) or not torch.backends.mps.is_available():
            raise ValueError('Saved MPS RNG state cannot be restored on this runtime')
        setter(value['mps']['state'])


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def text_row(row):
    return {key: str(row[key]) for key in FIELDS}


class RunLock:
    def __init__(self, directory, session_id):
        self.path = Path(directory) / '.run.lock'
        self.session_id = session_id
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open('a+')
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close(); self.stream = None
            raise RuntimeError('Run is actively locked by another process') from None
        # The acquired kernel lock proves any previous PID record is unowned.
        # This avoids PID-reuse errors and automatically releases after SIGKILL.
        self.stream.seek(0); self.stream.truncate()
        json.dump(dict(pid=os.getpid(), session_id=self.session_id, started_utc=utc()), self.stream)
        self.stream.flush(); os.fsync(self.stream.fileno())
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close(); self.stream = None


class EpochRun:
    def __init__(self, directory, configuration, resume=False):
        self.path = Path(directory)
        self.configuration = configuration
        self.config_hash = canonical_hash(configuration)
        self.execution_sources = configuration.get('source_sha256', {})
        self.code_repair = None
        self.resume_mode = resume
        self.session_id = uuid.uuid4().hex
        self.epoch = 0; self.history = []; self.cumulative = 0.
        self.best_epoch = None; self.best_metric = None
        self.checkpoint = None

    def __enter__(self):
        if self.resume_mode:
            if not self.path.is_dir(): raise FileNotFoundError('Resume requires an existing run')
        else:
            self.path.mkdir(exist_ok=False)
        self.lock = RunLock(self.path, self.session_id)
        self.lock.__enter__()
        try:
            if (self.path / 'COMPLETE.json').exists(): raise ValueError('Completed runs cannot be resumed')
            if self.resume_mode:
                original = json.loads((self.path / 'config.json').read_text())
                if canonical_hash(original) != self.config_hash:
                    if not self._approved_code_repair(original): raise ValueError('Scientific configuration/provenance mismatch')
                    self.configuration = original
                    self.config_hash = canonical_hash(original)
            else:
                atomic_json(self.path / 'config.json', self.configuration)
                atomic_json(self.path / 'provenance.json', dict(format_version=FORMAT_VERSION, config_hash=self.config_hash, run_id=self.path.name, run_uuid=uuid.uuid4().hex, created_utc=utc()))
            provenance = json.loads((self.path / 'provenance.json').read_text())
            if provenance['format_version'] != FORMAT_VERSION or provenance['config_hash'] != self.config_hash:
                raise ValueError('Checkpoint format/configuration provenance mismatch')
            if provenance['run_id'] != self.path.name: raise ValueError('Scientific run identity mismatch')
            self.run_uuid = provenance['run_uuid']
            self.created_utc = provenance['created_utc']
            for name in ['sessions', 'failures', 'partial_attempts']:
                (self.path / name).mkdir(exist_ok=True)
            self.started_utc = utc(); self.started_wall = time.time(); self.started_active = active_clock()
            self.session = dict(session_id=self.session_id, pid=os.getpid(), resumed=self.resume_mode,
                                started_utc=self.started_utc, active_clock=CLOCK_POLICY, status='running',
                                execution_source_sha256=self.execution_sources, approved_code_repair=self.code_repair)
            atomic_json(self.path / 'sessions' / f'{self.session_id}.json', self.session)
            return self
        except BaseException:
            self.lock.__exit__(None, None, None)
            raise

    def _approved_code_repair(self, original):
        # Narrow, reviewed code-only exception. Never rewrite frozen config or old checkpoints.
        current = self.configuration
        if {k:v for k,v in original.items() if k!='source_sha256'} != {k:v for k,v in current.items() if k!='source_sha256'}:
            return False
        if not REPAIR_POLICY.exists(): return False
        for repair in json.loads(REPAIR_POLICY.read_text())['repairs']:
            if (repair['run_id']==self.path.name and repair['original_config_hash']==canonical_hash(original)
                    and repair['original_source_sha256']==original.get('source_sha256')
                    and repair['approved_execution_source_sha256']==current.get('source_sha256')):
                self.code_repair=repair['id']
                return True
        return False

    def _payload(self, model, optimizer, epoch, row, metric, scheduler):
        rates = self.configuration['lr_by_epoch']
        history = self.history + ([row] if row is not None else [])
        improved = metric is not None and (self.best_metric is None or metric > self.best_metric)
        return dict(execution_source_sha256=self.execution_sources, approved_code_repair=self.code_repair, format_version=FORMAT_VERSION, run_uuid=self.run_uuid, run_id=self.path.name, configuration=self.configuration, config_hash=self.config_hash,
            schedule_hash=canonical_hash(self.configuration['schedule']), schedule_definition=self.configuration['schedule'],
            model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None, completed_epoch=epoch, next_epoch=epoch+1,
            current_lr=float(optimizer.param_groups[0]['lr']), next_lr=rates[epoch] if epoch<len(rates) else None,
            cumulative_active_seconds=row['cumulative_seconds'] if row else 0.,
            best_epoch=epoch if improved else self.best_epoch, best_metric=metric if improved else self.best_metric,
            timing_row=row, timing_history=history, rng=capture_rng(self.configuration['device']),
            sampler=dict(policy='local Random(seed+epoch-1); no persistent generator', seed=self.configuration['seed'], next_epoch=epoch+1, next_seed=self.configuration['seed']+epoch),
            session_id=self.session_id)

    def _verify_payload(self, payload):
        if payload['format_version'] != FORMAT_VERSION: raise ValueError('Unsupported checkpoint format')
        if payload['run_uuid'] != self.run_uuid or payload['run_id'] != self.path.name: raise ValueError('Checkpoint belongs to a different scientific run')
        if payload['config_hash'] != self.config_hash or canonical_hash(payload['configuration']) != self.config_hash:
            raise ValueError('Checkpoint configuration/provenance mismatch')
        if payload['schedule_hash'] != canonical_hash(self.configuration['schedule']) or payload['schedule_definition'] != self.configuration['schedule']:
            raise ValueError('LR schedule mismatch')
        e = payload['completed_epoch']; history = payload['timing_history']; rates = self.configuration['lr_by_epoch']
        if e < 0 or e > len(rates) or payload['next_epoch'] != e+1: raise ValueError('Invalid epoch position')
        if len(history) != e or [r['epoch'] for r in history] != list(range(1,e+1)): raise ValueError('Invalid checkpoint timing history')
        if not finite_tree([payload['model'], payload['optimizer'], history, payload['best_metric']]): raise ValueError('Non-finite checkpoint tensor/value')
        if e and (payload['timing_row'] != history[-1] or payload['current_lr'] != rates[e-1]): raise ValueError('Epoch timing/LR mismatch')
        if payload['next_lr'] != (rates[e] if e<len(rates) else None): raise ValueError('Next LR mismatch')
        if any(g['lr'] != payload['current_lr'] for g in payload['optimizer']['param_groups']): raise ValueError('Optimizer LR mismatch')
        if payload['cumulative_active_seconds'] != (history[-1]['cumulative_seconds'] if history else 0.): raise ValueError('Cumulative time mismatch')
        if e and (payload['best_epoch'] not in range(1,e+1) or payload['best_metric'] is None): raise ValueError('Invalid best checkpoint metadata')
        cumulative = 0.
        for r in history:
            if set(r) != set(FIELDS) or r['total_epoch_seconds'] < 0: raise ValueError('Invalid timing row')
            cumulative += r['total_epoch_seconds']
            if not math.isclose(cumulative,r['cumulative_seconds'],rel_tol=1e-10,abs_tol=1e-9): raise ValueError('Inconsistent cumulative timing')
        sampler = payload['sampler']
        if sampler['seed'] != self.configuration['seed'] or sampler['next_epoch'] != e+1 or sampler['next_seed'] != sampler['seed']+e:
            raise ValueError('Sampler position mismatch')

    def _write_checkpoint(self, payload):
        e = payload['completed_epoch']
        path = self.path / ('initial.pth' if e==0 else f'epoch_{e:03d}.pth')
        if path.exists() or path.with_suffix('.sha256.json').exists(): raise FileExistsError(path)
        temp = path.with_name(path.name + '.' + self.session_id + '.tmp')
        with temp.open('xb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        loaded = torch.load(temp, map_location='cpu', weights_only=True)
        self._verify_payload(loaded)
        checksum = digest(temp)
        os.replace(temp, path); fsync_directory(self.path)
        if digest(path) != checksum: raise ValueError('Checkpoint changed during publication')
        atomic_json(path.with_suffix('.sha256.json'), dict(sha256=checksum, completed_epoch=e, bytes=path.stat().st_size, format_version=FORMAT_VERSION))
        return path

    def _read_checkpoint(self, path):
        receipt = json.loads(path.with_suffix('.sha256.json').read_text())
        if receipt['format_version'] != FORMAT_VERSION or digest(path) != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
            raise ValueError('Checkpoint hash/size/format mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        self._verify_payload(payload)
        if payload['completed_epoch'] != receipt['completed_epoch']: raise ValueError('Checkpoint receipt epoch mismatch')
        expected = 'initial.pth' if payload['completed_epoch']==0 else f"epoch_{payload['completed_epoch']:03d}.pth"
        if path.name != expected: raise ValueError('Checkpoint filename/epoch mismatch')
        return payload

    def _csv_rows(self):
        path = self.path / 'epoch_timing.csv'
        if not path.exists(): return []
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS: raise ValueError('Timing CSV schema mismatch')
            rows = list(reader)
        if any(set(r)!=set(FIELDS) or None in r.values() for r in rows): raise ValueError('Partial timing CSV row')
        if [int(r['epoch']) for r in rows] != list(range(1,len(rows)+1)): raise ValueError('Duplicate/missing timing CSV epochs')
        return rows

    def _append_row(self, row):
        path = self.path / 'epoch_timing.csv'
        rows = self._csv_rows()
        if len(rows) != row['epoch']-1: raise ValueError('Refusing duplicate or out-of-order timing row')
        new = not path.exists()
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new: writer.writeheader()
            writer.writerow(row); stream.flush(); os.fsync(stream.fileno())
        fsync_directory(self.path)

    def _reconcile(self, payload):
        rows = self._csv_rows(); e = payload['completed_epoch']
        if len(rows)>e: raise ValueError('Timing CSV ahead of checkpoint')
        if e-len(rows)>1: raise ValueError('Checkpoint more than one epoch ahead of CSV')
        if rows != [text_row(r) for r in payload['timing_history'][:len(rows)]]: raise ValueError('Timing data mismatch')
        if e==len(rows)+1: self._append_row(payload['timing_row'])

    def _alias(self, canonical, name):
        # rename(old,new) is a no-op when both names already refer to one inode.
        # Skip that case; use a fresh temporary name for every other publication.
        target = self.path/name
        if target.exists() and os.path.samefile(canonical,target): return
        temp = self.path / (name + '.' + self.session_id + '.' + uuid.uuid4().hex + '.tmp')
        os.link(canonical,temp)
        if digest(temp)!=digest(canonical): raise ValueError('Checkpoint alias hash mismatch')
        os.replace(temp,target)
        # POSIX permits a same-inode rename to leave its source name in place.
        if temp.exists(): temp.unlink()  # Only this invocation's new temporary link.
        fsync_directory(self.path)

    def _publish(self, payload, path):
        self.epoch = payload['completed_epoch']; self.history = payload['timing_history']
        self.cumulative = payload['cumulative_active_seconds']; self.best_epoch = payload['best_epoch']; self.best_metric = payload['best_metric']
        self.checkpoint = path
        self._alias(path, 'last.pth')
        if self.best_epoch is not None:
            best = self.path/f'epoch_{self.best_epoch:03d}.pth'
            self._read_checkpoint(best)
            self._alias(best, 'best.pth')
        atomic_json(self.path/'run_state.json', dict(status='running', session_id=self.session_id, pid=os.getpid(),
            format_version=FORMAT_VERSION, config_hash=self.config_hash, last_completed_epoch=self.epoch, next_epoch=self.epoch+1,
            last_checkpoint=path.name, last_sha256=digest(path), best_epoch=self.best_epoch, best_metric=self.best_metric,
            best_sha256=digest(self.path/'best.pth') if self.best_epoch else None, cumulative_active_seconds=self.cumulative,
            current_lr=payload['current_lr'], next_lr=payload['next_lr'], updated_utc=utc(), created_utc=self.created_utc))

    def _archive_partial(self):
        candidates = list(self.path.glob('*.tmp'))
        for pattern in ['epoch_*.pth','epoch_*.sha256.json','sampling_epoch_*.json','losses_epoch_*.json','metrics_epoch_*.json']:
            for path in self.path.glob(pattern):
                try: e = int(path.name.split('_')[-1].split('.')[0])
                except ValueError: continue
                if e>self.epoch: candidates.append(path)
        if candidates:
            dest = self.path/'partial_attempts'/self.session_id; dest.mkdir()
            for path in set(candidates): os.replace(path,dest/path.name)
            fsync_directory(dest); fsync_directory(self.path)

    def initialize(self, model, optimizer, scheduler=None):
        if self.resume_mode: raise ValueError('Use restore for an existing run')
        payload = self._payload(model,optimizer,0,None,None,scheduler)
        path = self._write_checkpoint(payload); self._publish(payload,path)

    def restore(self, model, optimizer, scheduler=None):
        if not self.resume_mode: raise ValueError('Use initialize for a new run')
        receipts = sorted(self.path.glob('epoch_*.sha256.json'))
        path = receipts[-1].with_name(receipts[-1].name.replace('.sha256.json','.pth')) if receipts else self.path/'initial.pth'
        payload = self._read_checkpoint(path)
        if (scheduler is None) != (payload['scheduler'] is None): raise ValueError('Scheduler object mismatch')
        # Refuse all history conflicts before altering the model or repairing projections.
        self._reconcile(payload)
        model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
        if scheduler is not None: scheduler.load_state_dict(payload['scheduler'])
        restore_rng(payload['rng'])
        self._publish(payload,path); self._archive_partial()
        return payload

    def commit(self, model, optimizer, row, metric, scheduler=None):
        if row['epoch'] != self.epoch+1: raise ValueError('Nonsequential epoch commit')
        if not math.isfinite(metric): raise ValueError('Non-finite selection metric')
        payload = self._payload(model,optimizer,row['epoch'],row,metric,scheduler)
        path = self._write_checkpoint(payload)
        self._reconcile(payload)
        self._publish(payload,path)
        return dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size)

    def complete(self):
        if self.epoch != len(self.configuration['lr_by_epoch']): raise ValueError('Cannot complete unfinished training')
        atomic_json(self.path/'COMPLETE.json',dict(completed_epochs=self.epoch,exit_status=0,completed_utc=utc(),session_id=self.session_id,config_hash=self.config_hash))
        state=json.loads((self.path/'run_state.json').read_text());state['status']='complete'
        atomic_json(self.path/'run_state.json',state)

    def __exit__(self, kind, error, tb):
        try:
            self.session.update(ended_utc=utc(), wall_seconds=time.time()-self.started_wall,
                session_uptime_seconds=active_clock()-self.started_active, last_completed_epoch=self.epoch,
                cumulative_completed_epoch_seconds=self.cumulative, status='failed' if error else 'stopped')
            if (self.path/'COMPLETE.json').exists(): self.session['status']='complete'
            if error:
                import traceback
                failure=dict(session_id=self.session_id,error=str(error),traceback=''.join(traceback.format_exception(kind,error,tb)),last_published_epoch=self.epoch,failed_utc=utc())
                atomic_json(self.path/'failures'/f'{self.session_id}.json',failure)
                atomic_json(self.path/'FAILURE.json',failure)
            atomic_json(self.path/'sessions'/f'{self.session_id}.json',self.session)
            state_path=self.path/'run_state.json'
            if state_path.exists():
                state=json.loads(state_path.read_text());state['status']=self.session['status'];state['updated_utc']=utc()
                atomic_json(state_path,state)
        finally:
            self.lock.__exit__(kind,error,tb)
