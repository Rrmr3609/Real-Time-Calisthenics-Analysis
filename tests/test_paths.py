from utils import paths


def test_create_project_directories_creates_only_runtime_outputs(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    log_dir = project_root / "experiments" / "logs"
    output_dir = project_root / "experiments" / "outputs"
    unrelated_cwd = tmp_path / "working-directory"
    unrelated_cwd.mkdir()

    monkeypatch.setattr(paths, "LOG_DIR", log_dir)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.chdir(unrelated_cwd)

    paths.create_project_directories()
    paths.create_project_directories()

    assert log_dir.is_dir()
    assert output_dir.is_dir()
    assert not (project_root / "data" / "raw").exists()
    assert not (project_root / "data" / "annotations").exists()
    assert not (project_root / "results" / "figures").exists()
    assert not (project_root / "results" / "tables").exists()
