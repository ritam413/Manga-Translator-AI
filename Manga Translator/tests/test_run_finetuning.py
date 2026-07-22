"""Unit tests for the master finetuning orchestrator script."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_finetuning import main


@patch("scripts.run_finetuning.subprocess.run")
def test_run_finetuning_success(mock_run) -> None:
    # Set up mock returns
    mock_run.return_value = MagicMock(returncode=0)

    test_args = [
        "run_finetuning.py",
        "--model-version",
        "v2",
        "--raw-dir",
        "my_raw",
        "--eng-dir",
        "my_eng",
        "--reset-checkpoint",
        "--reset-checkpoint-reset",
        "--checkpoint-interval",
        "20",
        "--iou-threshold",
        "0.4",
        "--no-gpu",
        "--no-manga-ocr",
        "--min-confidence",
        "0.6",
        "--tune",
        "--vocab-threshold",
        "0.35",
    ]

    with patch.object(sys, "argv", test_args):
        exit_code = main()
        assert exit_code == 0

    # Ensure subprocess.run was called 4 times (alignment -> extraction -> layout training -> TM database)
    assert mock_run.call_count == 4

    # Get calls and verify command structures
    calls = [c[0][0] for c in mock_run.call_args_list]

    # Step 1: Alignment command
    assert "align_image_resolutions.py" in calls[0][1]
    assert "--raw-dir" in calls[0]
    assert "my_raw" in calls[0]
    assert "--eng-dir" in calls[0]
    assert "my_eng" in calls[0]

    # Step 2: Extractor command
    assert "dataset_extractor.py" in calls[1][1]
    assert "--model-version" in calls[1]
    assert "v2" in calls[1]
    assert "--raw-dir" in calls[1]
    assert "my_raw" in calls[1]
    assert "--eng-dir" in calls[1]
    assert "my_eng" in calls[1]
    assert "--reset-checkpoint" in calls[1]
    assert "--reset-checkpoint-reset" in calls[1]
    assert "--checkpoint-interval" in calls[1]
    assert "20" in calls[1]
    assert "--iou-threshold" in calls[1]
    assert "0.4" in calls[1]
    assert "--no-gpu" in calls[1]
    assert "--no-manga-ocr" in calls[1]

    # Step 3: Layout training command
    assert "train_layout_model.py" in calls[2][1]
    assert "--model-version" in calls[2]
    assert "v2" in calls[2]
    assert "--min-confidence" in calls[2]
    assert "0.6" in calls[2]
    assert "--tune" in calls[2]

    # Step 4: TM database command
    assert "build_translation_memory.py" in calls[3][1]
    assert "--model-version" in calls[3]
    assert "v2" in calls[3]
    assert "--vocab-threshold" in calls[3]
    assert "0.35" in calls[3]


@patch("scripts.run_finetuning.subprocess.run")
def test_run_finetuning_failure(mock_run) -> None:
    import subprocess
    # Simulate a failure in step 1
    mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="align_image_resolutions.py")

    test_args = [
        "run_finetuning.py",
        "--model-version",
        "v2",
    ]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    # Should have stopped after step 1
    assert mock_run.call_count == 1
