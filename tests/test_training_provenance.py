import json
from types import SimpleNamespace
from src.training.train_baseline import record_checkpoint


def test_best_epoch_tracks_saved_checkpoint_including_ties(tmp_path):
    info = {"status": "running"}
    report = tmp_path / "run.json"
    record_checkpoint(
        SimpleNamespace(best_fitness=0.25, fitness=0.25, epoch=0), info, report
    )
    record_checkpoint(
        SimpleNamespace(best_fitness=0.25, fitness=0.20, epoch=1), info, report
    )
    assert info["best_epoch"] == 1
    record_checkpoint(
        SimpleNamespace(best_fitness=0.25, fitness=0.25, epoch=2), info, report
    )
    assert json.loads(report.read_text())["best_epoch"] == 3
    assert info["last_saved_epoch"] == 3
