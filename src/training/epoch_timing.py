"""Reusable durable completed-epoch log; incomplete epochs are never appended."""
import csv
import os
from pathlib import Path

FIELDS=['epoch','start_utc','end_utc','training_seconds','validation_seconds','total_epoch_seconds','cumulative_seconds','learning_rate','total_training_loss','classification_loss','box_regression_loss','rpn_objectness_loss','rpn_box_loss','validation_precision','validation_recall','validation_map50','validation_map50_95','checkpoint_saved','device','batch_size','resize_policy']


class EpochLogger:
    def __init__(self,path):
        self.path=Path(path)
        if self.path.exists():raise FileExistsError(self.path)
        self.last_epoch=0
        with self.path.open('x',newline='') as f:
            csv.DictWriter(f,fieldnames=FIELDS).writeheader();f.flush();os.fsync(f.fileno())
    def append(self,row):
        if set(row)!=set(FIELDS) or row['epoch']!=self.last_epoch+1:raise ValueError('Incomplete schema or nonsequential epoch')
        if not row['checkpoint_saved']:raise ValueError('Epoch checkpoint not saved')
        with self.path.open('a',newline='') as f:
            csv.DictWriter(f,fieldnames=FIELDS).writerow(row);f.flush();os.fsync(f.fileno())
        self.last_epoch=row['epoch']
