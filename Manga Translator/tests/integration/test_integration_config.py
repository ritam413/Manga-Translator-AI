import os
import pytest
import tempfile
from pydantic import ValidationError
from cli.config.models import CliConfig, load_config, save_config
from cli.worker_factory import (
    build_ocr_worker,
    build_translation_worker,
    build_inpainter,
    build_typesetter,
)


def test_finetuning_settings_defaults():
    """Verify default values of FinetuningSettings are populated correctly."""
    config = CliConfig()
    assert config.finetuning.enabled is True
    assert config.finetuning.translation_memory_path == "assets/models/translation_memory.json"
    assert config.finetuning.layout_model_path == "assets/models/layout_model.pkl"
    assert config.finetuning.fuzzy_match_threshold == 0.8
    assert config.finetuning.context_sentence_limit == 5
    assert config.finetuning.use_contour_inpaint is True
    assert config.finetuning.layout_model_min_confidence == 0.5


@pytest.mark.parametrize(
    "key,val,expected",
    [
        ("finetuning.enabled", "false", False),
        ("finetuning.fuzzy_match_threshold", "0.75", 0.75),
        ("finetuning.context_sentence_limit", "3", 3),
        ("finetuning.use_contour_inpaint", "false", False),
        ("finetuning.layout_model_min_confidence", "0.4", 0.4),
    ],
)
def test_finetuning_settings_set_value(key, val, expected):
    """Verify dot notation set_value type coercion and updates."""
    config = CliConfig()
    new_config = config.set_value(key, val)
    assert new_config.get_value(key) == expected


@pytest.mark.parametrize(
    "key,val",
    [
        ("finetuning.fuzzy_match_threshold", "1.5"),
        ("finetuning.fuzzy_match_threshold", "-0.1"),
        ("finetuning.context_sentence_limit", "0"),
        ("finetuning.context_sentence_limit", "25"),
        ("finetuning.layout_model_min_confidence", "1.1"),
        ("finetuning.layout_model_min_confidence", "-0.5"),
    ],
)
def test_finetuning_settings_validation_errors(key, val):
    """Verify Pydantic validation rejects invalid ranges."""
    config = CliConfig()
    with pytest.raises((ValidationError, ValueError)):
        config.set_value(key, val)


def test_toml_roundtrip_save_load():
    """Verify configuration can be successfully saved to and loaded from TOML."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, ".manga-translator", "config.toml")
        
        config = CliConfig()
        config = config.set_value("finetuning.fuzzy_match_threshold", "0.65")
        config = config.set_value("finetuning.enabled", "false")
        config.save(config_path)
        
        loaded = CliConfig.from_toml(config_path)
        assert loaded.finetuning.enabled is False
        assert loaded.finetuning.fuzzy_match_threshold == 0.65


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("use_contour", [True, False])
def test_worker_factory_propagation_inpainter(enabled, use_contour):
    """Verify build_inpainter propagates settings appropriately."""
    config = CliConfig()
    config.finetuning.enabled = enabled
    config.finetuning.use_contour_inpaint = use_contour
    config.finetuning.layout_model_min_confidence = 0.6
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "input"), exist_ok=True)
        worker = build_inpainter(tmp_dir, config)
        assert worker._use_contour_inpaint == (enabled and use_contour)
        assert worker._layout_model_min_confidence == 0.6


@pytest.mark.parametrize("enabled", [True, False])
def test_worker_factory_propagation_typesetter(enabled, synthetic_layout_model_path):
    """Verify build_typesetter propagates layout model settings appropriately."""
    config = CliConfig()
    config.finetuning.enabled = enabled
    config.finetuning.layout_model_path = synthetic_layout_model_path
    config.finetuning.layout_model_min_confidence = 0.7
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "input"), exist_ok=True)
        worker = build_typesetter(tmp_dir, config)
        
        if enabled:
            assert worker._layout_model is not None
        else:
            assert worker._layout_model is None
            
        assert worker._layout_model_min_confidence == 0.7


@pytest.mark.parametrize("enabled", [True, False])
def test_worker_factory_propagation_ocr(enabled, synthetic_tm_db):
    """Verify build_ocr_worker propagates spelling correction settings."""
    config = CliConfig()
    config.finetuning.enabled = enabled
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "input"), exist_ok=True)
        # Mock load_translation_memory to return synthetic_tm_db
        import cli.worker_factory
        from unittest.mock import patch
        
        with patch("finetuning.pipeline_integration.load_translation_memory", return_value=synthetic_tm_db):
            worker = build_ocr_worker(tmp_dir, config)
            assert worker._enable_tm_correction == enabled
            if enabled:
                assert worker._tm_db is not None
            else:
                assert worker._tm_db is None


@pytest.mark.parametrize("enabled", [True, False])
def test_worker_factory_propagation_translator(enabled, synthetic_tm_db):
    """Verify build_translation_worker propagates Translation Memory settings."""
    config = CliConfig()
    config.finetuning.enabled = enabled
    config.finetuning.fuzzy_match_threshold = 0.85
    config.finetuning.context_sentence_limit = 4
    
    from unittest.mock import patch
    with patch("finetuning.pipeline_integration.load_translation_memory", return_value=synthetic_tm_db):
        worker = build_translation_worker(config, project_dir=".")
        
        if enabled:
            assert worker._tm_db is not None
            assert worker._fuzzy_threshold == 0.85
            assert worker._context_limit == 4
        else:
            assert worker._tm_db is None
            assert worker._fuzzy_threshold == 0.8
            assert worker._context_limit == 5
