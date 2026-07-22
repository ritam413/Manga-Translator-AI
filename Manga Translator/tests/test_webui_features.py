"""
Unit tests for the new WebUI features:
- Project manifest backups, list-backups, and restores
- Training endpoints (count, clear, train-detection)
- CLI subcommand 'finetune'
"""

import os
import sys
import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure webui/backend is at the front of import path for relative imports inside backend
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir.parent / "webui" / "backend"
sys.path.insert(0, str(backend_dir))

# Dynamically import backend main.py to avoid name collision with root main.py
import importlib.util
spec = importlib.util.spec_from_file_location("webui_backend_main", str(backend_dir / "main.py"))
webui_backend_main = importlib.util.module_from_spec(spec)
sys.modules["webui_backend_main"] = webui_backend_main
spec.loader.exec_module(webui_backend_main)

app = webui_backend_main.app
from fastapi.testclient import TestClient

client = TestClient(app)


# ── Tests: CLI Subcommand 'finetune' ──────────────────────────────────────────

@patch("subprocess.run")
def test_cli_finetune_subcommand(mock_run, tmp_path: Path) -> None:
    # Setup mock files
    corrections_file = tmp_path / "user_corrections.jsonl"
    corrections_file.write_text(
        json.dumps({
            "manga_id": "test_manga",
            "page": "page_001.png",
            "region_index": 0,
            "box": {"x_min": 10, "y_min": 10, "x_max": 100, "y_max": 100},
            "raw_text": "テスト",
            "translated_text": "Test",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        }) + "\n",
        encoding="utf-8"
    )

    base_dataset_dir = tmp_path / "training_data"
    base_dataset_dir.mkdir(parents=True, exist_ok=True)
    base_dataset_file = base_dataset_dir / "dataset_v2.jsonl"
    base_dataset_file.write_text(
        json.dumps({
            "manga_id": "test_manga",
            "page": "page_001.png",
            "region_index": 0,
            "box": {"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 90},
            "jp_ocr_text": "テスト",
            "font_size": 10.0,
            "padding": 2.0,
            "line_spacing": 1.0,
        }) + "\n",
        encoding="utf-8"
    )

    # Mock the argv
    test_args = [
        "main.py",
        "finetune",
        "--use-corrections", str(corrections_file),
        "--model-version", "v2"
    ]

    mock_run.return_value = MagicMock(returncode=0)

    # Patch paths inside cmd_finetune to point to our tmp_path layout
    from main import cmd_finetune
    
    mock_args = MagicMock()
    mock_args.use_corrections = str(corrections_file)
    mock_args.model_version = "v2"

    resolve_target = "pathlib.WindowsPath.resolve" if os.name == "nt" else "pathlib.PosixPath.resolve"
    with patch(resolve_target) as mock_resolve:
        mock_resolve.return_value = tmp_path / "main.py"  # Path(__file__).resolve()
        with patch("scripts.train_layout_model.main") as mock_train_main:
            mock_train_main.return_value = 0
            
            # Run the command function directly
            exit_code = cmd_finetune(mock_args)
            assert exit_code == 0

            # Combined dataset must have been created and merged
            combined_file = tmp_path / "training_data" / "combined_dataset_v2.jsonl"
            assert combined_file.exists()
            
            # Read merged content to verify override
            lines = combined_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            merged_row = json.loads(lines[0])
            # Should have the updated coordinates and font size from user correction
            assert merged_row["box"]["x_max"] == 100
            assert merged_row["font_size"] == 12.0


# ── Tests: Backend API Routers ────────────────────────────────────────────────

@patch("routers.project.ENGINE_DIR")
def test_api_list_backups(mock_engine_dir, tmp_path: Path) -> None:
    mock_engine_dir.return_value = tmp_path
    
    # Create fake backups in project dir
    proj_dir = tmp_path / "test_project"
    proj_dir.mkdir(parents=True, exist_ok=True)
    
    backup_1 = proj_dir / "project_manifest.backup_20260703_120000.json"
    backup_2 = proj_dir / "project_manifest.backup_20260703_130000.json"
    backup_1.touch()
    backup_2.touch()
    
    # Call list-backups endpoint
    with patch("routers.project.Path") as mock_path:
        mock_path.return_value = proj_dir
        response = client.get(f"/api/project/list-backups?project_dir={proj_dir}")
        assert response.status_code == 200
        data = response.json()
        
        # Should be sorted newest first (130000 before 120000)
        assert len(data) == 2
        assert "130000" in data[0]["filename"]
        assert "120000" in data[1]["filename"]


@patch("routers.project.ENGINE_DIR")
def test_api_restore_backup(mock_engine_dir, tmp_path: Path) -> None:
    proj_dir = tmp_path / "test_project"
    proj_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_file = proj_dir / "project_manifest.json"
    manifest_file.write_text("current_manifest_content")
    
    backup_file = proj_dir / "project_manifest.backup_20260703_120000.json"
    backup_file.write_text("backup_manifest_content")
    
    response = client.post("/api/project/restore-backup", json={
        "project_dir": str(proj_dir),
        "backup_filename": backup_file.name
    })
    
    assert response.status_code == 200
    assert response.json()["success"] is True
    # The manifest file should have been overwritten with the backup content
    assert manifest_file.read_text() == "backup_manifest_content"


@patch("routers.training.ENGINE_DIR")
def test_api_correction_count_and_clear(mock_engine_dir, tmp_path: Path) -> None:
    # Setup fake corrections file
    corr_dir = tmp_path / "training_data"
    corr_dir.mkdir(parents=True, exist_ok=True)
    corrections_file = corr_dir / "user_corrections.jsonl"
    
    # Initially no corrections file exists
    with patch("routers.training.ENGINE_DIR", tmp_path):
        response = client.get(f"/api/training/correction-count?project_dir={tmp_path}")
        assert response.status_code == 200
        assert response.json()["count"] == 0

        # Create corrections file with 3 entries
        corrections_file.write_text("line1\nline2\nline3\n")
        response = client.get(f"/api/training/correction-count?project_dir={tmp_path}")
        assert response.status_code == 200
        assert response.json()["count"] == 3

        # Clear corrections
        response = client.post("/api/training/clear-corrections", json={"project_dir": str(tmp_path)})
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify it was deleted
        assert not corrections_file.exists()
