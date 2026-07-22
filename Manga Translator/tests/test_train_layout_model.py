"""Unit tests for the layout model training script."""

from __future__ import annotations
import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from scripts.train_layout_model import (
    load_dataset,
    train_model,
    main,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_dataset_success(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 20.0},
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.9,
            "jp_box": {"x_min": 10.0, "y_min": 10.0, "x_max": 30.0, "y_max": 30.0},
            "jp_text_detected": "いう",
            "font_size": 16.0,
            "padding": 8.0,
            "line_spacing": 1.2,
        },
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)

    assert X.shape == (2, 6)
    assert y.shape == (2, 3)

    # Row 1 features: w=10, h=20, area=200, aspect_ratio=0.5, char_count=1, char_density=1/200=0.005
    assert np.allclose(X[0], [10.0, 20.0, 200.0, 0.5, 1.0, 0.005])
    assert np.allclose(y[0], [12.0, 4.0, 1.0])

    # Row 2 features: w=20, h=20, area=400, aspect_ratio=1.0, char_count=2, char_density=2/400=0.005
    assert np.allclose(X[1], [20.0, 20.0, 400.0, 1.0, 2.0, 0.005])
    assert np.allclose(y[1], [16.0, 8.0, 1.2])


def test_load_dataset_confidence_filtering(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": 12,
            "padding": 4,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.2,
            "jp_box": {"x_min": 10, "y_min": 10, "x_max": 30, "y_max": 30},
            "jp_ocr_text": "い",
            "font_size": 16,
            "padding": 8,
            "line_spacing": 1.2,
        },
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)
    assert X.shape == (1, 6)
    assert np.allclose(X[0], [10.0, 20.0, 200.0, 0.5, 1.0, 0.005])


def test_load_dataset_aspect_ratio_and_density_guards(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 10.0, "y_min": 10.0, "x_max": 10.0, "y_max": 10.0},
            "jp_ocr_text": "テスト",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        }
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)
    assert X.shape == (1, 6)
    # w=0, h=0, area=0, aspect_ratio=0, char_count=3, char_density=0
    assert np.allclose(X[0], [0.0, 0.0, 0.0, 0.0, 3.0, 0.0])


def test_load_dataset_clamping_ranges(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
            "jp_ocr_text": "a",
            "font_size": 120.0,  # Clamped to 72.0
            "padding": -5.0,     # Clamped to 0.0
            "line_spacing": 4.0, # Clamped to 2.5
        },
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
            "jp_ocr_text": "b",
            "font_size": 2.0,     # Clamped to 8.0
            "padding": 60.0,     # Clamped to 50.0
            "line_spacing": 0.2, # Clamped to 0.5
        },
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)
    assert np.allclose(y[0], [72.0, 0.0, 2.5])
    assert np.allclose(y[1], [8.0, 50.0, 0.5])


def test_load_dataset_empty_raises(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.2,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
            "jp_ocr_text": "a",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        }
    ]
    _write_jsonl(dataset_file, rows)

    with pytest.raises(ValueError, match="No valid training samples remaining"):
        load_dataset(dataset_file, min_confidence=0.5)


def test_load_dataset_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("does_not_exist_file.jsonl"), min_confidence=0.5)


def test_train_and_predict_flow(tmp_path: Path) -> None:
    X = np.random.rand(5, 6)
    y = np.random.rand(5, 3)

    model = train_model(X, y, tune=False)
    assert isinstance(model, RandomForestRegressor)

    pred = model.predict(X[:1])
    assert pred.shape == (1, 3)

    # Test serialization
    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    assert model_file.exists()
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)

    loaded_pred = loaded_model.predict(X[:1])
    assert np.allclose(pred, loaded_pred)


def test_tuning_cv_flow() -> None:
    X = np.random.rand(12, 6)
    y = np.random.rand(12, 3)

    # With tune=True and 12 samples, it should run RandomizedSearchCV
    model = train_model(X, y, tune=True)
    # The returned model is the best estimator fitted on the entire training set
    assert isinstance(model, RandomForestRegressor)


def test_tuning_fallback_on_small_data() -> None:
    X = np.random.rand(5, 6)
    y = np.random.rand(5, 3)

    # With tune=True but only 5 samples, it should fallback to standard fit
    model = train_model(X, y, tune=True)
    assert isinstance(model, RandomForestRegressor)


def test_cli_main_success(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    # Create 12 valid rows to allow train/test split and CV tuning
    rows = []
    for i in range(12):
        rows.append({
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        })
    _write_jsonl(dataset_file, rows)

    model_out = tmp_path / "models" / "layout_model.pkl"

    test_args = [
        "train_layout_model.py",
        "--dataset-file",
        str(dataset_file),
        "--model-out",
        str(model_out),
        "--min-confidence",
        "0.5",
        "--tune",
    ]

    with patch.object(sys, "argv", test_args):
        exit_code = main()
        assert exit_code == 0

    assert model_out.exists()


def test_cli_main_failure(tmp_path: Path) -> None:
    model_out = tmp_path / "models" / "layout_model.pkl"
    test_args = [
        "train_layout_model.py",
        "--dataset-file",
        "non_existent.jsonl",
        "--model-out",
        str(model_out),
    ]

    with patch.object(sys, "argv", test_args):
        exit_code = main()
        assert exit_code == 1

    assert not model_out.exists()


def test_load_dataset_negative_dimensions(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 10.0, "y_min": 10.0, "x_max": 5.0, "y_max": 20.0},  # w = -5
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 10.0, "y_min": 10.0, "x_max": 20.0, "y_max": 5.0},  # h = -5
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 20.0},   # valid
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)
    assert X.shape == (1, 6)
    assert np.allclose(X[0], [10.0, 20.0, 200.0, 0.5, 1.0, 0.005])


def test_load_dataset_nan_inf_values(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": float("nan"), "y_max": 20},  # NaN in feature
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": float("inf"),  # Inf in target
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)
    assert X.shape == (1, 6)
    assert np.allclose(X[0], [10.0, 20.0, 200.0, 0.5, 1.0, 0.005])


def test_load_dataset_malformed_confidence(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    rows = [
        {
            "typesetting_confidence": float("nan"),  # Non-finite confidence
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": float("inf"),  # Non-finite confidence
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
        {
            "typesetting_confidence": 0.8,
            "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
            "jp_ocr_text": "あ",
            "font_size": 12.0,
            "padding": 4.0,
            "line_spacing": 1.0,
        },
    ]
    _write_jsonl(dataset_file, rows)

    X, y = load_dataset(dataset_file, min_confidence=0.5)
    assert X.shape == (1, 6)
