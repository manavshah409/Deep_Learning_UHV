import pytest
from src.data.common import paths, require_directory


def test_project_relative_paths(tmp_path, monkeypatch):
    config = tmp_path / "configs"
    config.mkdir()
    f = config / "paths.yaml"
    f.write_text("raw: data/raw\n")
    monkeypatch.chdir("/")
    assert paths(f)["raw"] == tmp_path / "data/raw"


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_directory(tmp_path / "missing")


def test_existing_directory(tmp_path):
    assert require_directory(tmp_path) == tmp_path
