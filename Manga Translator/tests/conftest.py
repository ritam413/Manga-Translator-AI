import os
import pickle
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from finetuning.translation_memory import TranslationMemoryDB, TranslationMemoryEntry


@pytest.fixture
def synthetic_tm_db():
    """Returns a TranslationMemoryDB pre-populated with synthetic entries and vocabularies."""
    db = TranslationMemoryDB()
    
    # Pre-populate entries
    entries_m1 = [
        TranslationMemoryEntry("こんにちは", "Hello", 1, "manga1::0001::00", 0),
        TranslationMemoryEntry("お元気ですか", "How are you?", 1, "manga1::0001::01", 1),
        TranslationMemoryEntry("ルフィ！", "Luffy!", 2, "manga1::0002::00", 0),
        TranslationMemoryEntry("俺は海賊王になる男だ", "I will be the Pirate King!", 2, "manga1::0002::01", 1),
        TranslationMemoryEntry("ゾロ", "Zoro", 3, "manga1::0003::00", 0),
    ]
    
    entries_m2 = [
        TranslationMemoryEntry("こんにちは", "Hi there", 1, "manga2::0001::00", 0),
        TranslationMemoryEntry("ナミ", "Nami", 1, "manga2::0001::01", 1),
    ]
    
    for entry in entries_m1:
        db.add_entry("manga1", entry)
        
    for entry in entries_m2:
        db.add_entry("manga2", entry)
        
    # Build vocabulary mappings
    db.manga_vocab = {
        "manga1": {
            "ルフィ": "luffy",
            "ゾロ": "zoro",
            "こんにちは": "hello",
            "海賊王": "pirate_king",
        },
        "manga2": {
            "ナミ": "nami",
            "こんにちは": "hi_there",
        }
    }
    
    return db


@pytest.fixture
def synthetic_layout_model_path():
    """Fits a tiny RandomForestRegressor and yields its temporary pickle path."""
    from sklearn.ensemble import RandomForestRegressor
    
    # 5 samples, 6 features: [w, h, area, aspect_ratio, char_count, char_density]
    X = np.array([
        [100.0, 100.0, 10000.0, 1.0, 5.0, 0.0005],
        [200.0, 100.0, 20000.0, 2.0, 10.0, 0.0005],
        [50.0, 150.0, 7500.0, 0.33, 3.0, 0.0004],
        [300.0, 300.0, 90000.0, 1.0, 20.0, 0.0002],
        [150.0, 50.0, 7500.0, 3.0, 8.0, 0.001],
    ], dtype=np.float32)
    
    # 3 targets: [font_size, padding, line_spacing]
    y = np.array([
        [14.0, 4.0, 1.2],
        [18.0, 6.0, 1.4],
        [12.0, 3.0, 1.0],
        [24.0, 10.0, 1.8],
        [16.0, 5.0, 1.3],
    ], dtype=np.float32)
    
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X, y)
    
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    
    with open(path, "wb") as f:
        pickle.dump(model, f)
        
    yield path
    
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture
def make_manifest_with_story():
    """Returns a helper factory to build a ProjectManifest populated with story metadata."""
    def _factory(project_name="manga1", manga_id="manga1", pages_count=3, regions_per_page=2):
        pages = {}
        for p_idx in range(1, pages_count + 1):
            filename = f"page_{p_idx:03d}.png"
            regions = []
            for r_idx in range(regions_per_page):
                box = BoundingBox(10, 10 + r_idx * 100, 150, 90 + r_idx * 100)
                regions.append(TextRegion(
                    id=f"r{r_idx:02d}",
                    box=box,
                    raw_text=f"Raw text {p_idx}-{r_idx}",
                    translated_text=f"Translated text {p_idx}-{r_idx}",
                    ocr_confidence=0.9,
                    region_index=r_idx,
                ))
            pages[filename] = PageManifest(
                image_filename=filename,
                width=800,
                height=1200,
                text_regions=regions,
                phase_a_completed=False,
                phase_b_completed=False,
                page_number=p_idx,
            )
        return ProjectManifest(
            project_name=project_name,
            pages=pages,
            source_language="ja",
            target_language="en",
            manga_id=manga_id,
        )
    return _factory
