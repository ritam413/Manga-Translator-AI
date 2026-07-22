#!/usr/bin/env python3
"""Train an optimized Random Forest Regressor to predict manga typesetting parameters."""

from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_layout_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train layout optimization Random Forest model with validation and tuning."
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Model version target ('v1' or 'v2')",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to the dataset.jsonl file",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Path to save the trained layout model (.pkl)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum typesetting confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV hyperparameter optimization",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path, min_confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Load JSONL dataset, apply edge case guards, clip target ranges, and return X, y."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    X_list = []
    y_list = []
    skipped_confidence = 0
    skipped_invalid = 0
    total_rows = 0

    # Count total lines for progress bar
    total_lines = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    from core.progress import MangaProgress
    mp = MangaProgress()
    mp.__enter__()
    task_id = mp.add_task("Loading dataset", total=total_lines)

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                mp.advance(task_id)
                line_str = line.strip()
                if not line_str:
                    continue
                total_rows += 1
                try:
                    row = json.loads(line_str)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d is not valid JSON. Skipping. Error: %s", line_num, exc)
                    continue

                # 1. Confidence filtering
                confidence = row.get("typesetting_confidence", 1.0)
                if confidence is None:
                    confidence = 1.0
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = 1.0

                if not np.isfinite(confidence) or confidence < min_confidence:
                    skipped_confidence += 1
                    continue

                # 2. Geometry extraction & division-by-zero guards
                jp_box = row.get("jp_box")
                if not jp_box or not isinstance(jp_box, dict):
                    skipped_invalid += 1
                    continue

                try:
                    x_min = float(jp_box["x_min"])
                    y_min = float(jp_box["y_min"])
                    x_max = float(jp_box["x_max"])
                    y_max = float(jp_box["y_max"])
                except (KeyError, ValueError, TypeError):
                    skipped_invalid += 1
                    continue

                w = x_max - x_min
                h = y_max - y_min

                # Geometry check: speech bubble boxes must be non-negative in dimensions
                if w < 0 or h < 0:
                    skipped_invalid += 1
                    continue

                area = w * h

                # Guard aspect ratio division-by-zero
                aspect_ratio = w / h if h > 0 else 0.0

                # 3. Japanese character count & density
                jp_text = row.get("jp_ocr_text")
                if jp_text is None:
                    jp_text = row.get("jp_text_detected", "")

                if not isinstance(jp_text, str):
                    jp_text = str(jp_text)

                char_count = len(jp_text.strip())

                # Guard density division-by-zero
                char_density = char_count / area if area > 0 else 0.0

                # 4. Target extraction & range boundary clipping
                try:
                    font_size = float(row["font_size"])
                    padding = float(row["padding"])
                    line_spacing = float(row["line_spacing"])
                except (KeyError, ValueError, TypeError):
                    skipped_invalid += 1
                    continue

                # Check if any computed feature or parsed target is NaN or Inf
                feats = [w, h, area, aspect_ratio, float(char_count), char_density]
                targs = [font_size, padding, line_spacing]
                if not all(np.isfinite(feats)) or not all(np.isfinite(targs)):
                    skipped_invalid += 1
                    continue

                # Target bounds clamping to remove outliers
                font_size = np.clip(font_size, 8.0, 72.0)
                padding = np.clip(padding, 0.0, 50.0)
                line_spacing = np.clip(line_spacing, 0.5, 2.5)

                # Features: [w, h, area, aspect_ratio, char_count, char_density]
                X_list.append([w, h, area, aspect_ratio, char_count, char_density])
                y_list.append([font_size, padding, line_spacing])
    finally:
        mp.__exit__(None, None, None)

    logger.info("Total rows scanned: %d", total_rows)
    logger.info("Skipped due to low confidence (< %s): %d", min_confidence, skipped_confidence)
    logger.info("Skipped due to invalid fields/geometry: %d", skipped_invalid)
    logger.info("Valid training samples: %d", len(X_list))

    if not X_list:
        raise ValueError("No valid training samples remaining after filtering.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = False
) -> Any:
    """Initialize and fit RandomForestRegressor with optional RandomizedSearchCV tuning."""
    from sklearn.ensemble import RandomForestRegressor

    if tune and len(X_train) >= 10:
        from sklearn.model_selection import RandomizedSearchCV
        logger.info("Starting hyperparameter tuning via RandomizedSearchCV (5-fold CV)...")

        param_dist = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }

        base_rf = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=15,
            cv=5,
            random_state=42,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )
        search.fit(X_train, y_train)

        logger.info("Best hyperparameters found: %s", search.best_params_)
        model = search.best_estimator_
    else:
        if tune:
            logger.warning("Fewer than 10 training samples. Disabling hyperparameter tuning.")
        logger.info("Training default RandomForestRegressor with 100 estimators...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    return model


def main() -> int:
    args = parse_args()
    model_version = args.model_version

    dataset_file = args.dataset_file
    if dataset_file is None:
        dataset_file = f"training_data/dataset_{model_version}.jsonl"

    model_out = args.model_out
    if model_out is None:
        model_out = f"assets/models/layout_model_{model_version}.pkl"

    dataset_path = Path(dataset_file)
    model_path = Path(model_out)

    try:
        X, y = load_dataset(dataset_path, args.min_confidence)
    except Exception as exc:
        logger.error("Dataset load failed: %s", exc)
        return 1

    # 1. Split training and test data if dataset is large enough
    if len(X) >= 10:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Split dataset: Train size = %d, Test size = %d", len(X_train), len(X_test))
    else:
        logger.warning("Dataset too small (%d samples) for train/test split. Training on all data.", len(X))
        X_train, X_test, y_train, y_test = X, None, y, None

    # 2. Train model
    try:
        model = train_model(X_train, y_train, tune=args.tune)
    except Exception as exc:
        logger.error("Model training failed: %s", exc)
        return 1

    # 3. Validation and Metrics Reporting
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Train Metrics
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred, multioutput="raw_values")
    mse_train = mean_squared_error(y_train, y_train_pred, multioutput="raw_values")
    r2_train = model.score(X_train, y_train)

    logger.info("--- Training Metrics ---")
    logger.info("Train R^2 score: %.4f", r2_train)
    logger.info("Train MAE [font_size, padding, line_spacing]: %s", np.round(mae_train, 4))
    logger.info("Train MSE [font_size, padding, line_spacing]: %s", np.round(mse_train, 4))

    # Test Metrics
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_test_pred, multioutput="raw_values")
        mse_test = mean_squared_error(y_test, y_test_pred, multioutput="raw_values")
        r2_test = model.score(X_test, y_test)

        logger.info("--- Test/Validation Metrics ---")
        logger.info("Test R^2 score: %.4f", r2_test)
        logger.info("Test MAE [font_size, padding, line_spacing]: %s", np.round(mae_test, 4))
        logger.info("Test MSE [font_size, padding, line_spacing]: %s", np.round(mse_test, 4))

    # Feature Importance Reporting
    feature_names = ["width", "height", "area", "aspect_ratio", "char_count", "char_density"]
    importances = model.feature_importances_
    logger.info("--- Feature Importances ---")
    for name, importance in zip(feature_names, importances):
        logger.info("  %s: %.4f", name, importance)

    # 4. Serialize Model
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model successfully saved to: %s", model_path)
    except Exception as exc:
        logger.error("Failed to save model: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
