import json
import os
import pytest
import tempfile
from pathlib import Path

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from core.manifest import ManifestEngine
from finetuning.pipeline_integration import build_story_position, derive_page_order


def test_manifest_fields_backward_compatibility():
    """Verify that deserializing old manifests works correctly and sets defaults."""
    old_json = """{
        "project_name": "Old Manga",
        "pages": {
            "001.png": {
                "image_filename": "001.png",
                "width": 800,
                "height": 1200,
                "text_regions": [
                    {
                        "id": "r00",
                        "box": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 80},
                        "raw_text": "こんにちは"
                    }
                ]
            }
        }
    }"""
    data = json.loads(old_json)
    manifest = ProjectManifest.from_dict(data)
    
    assert manifest.manga_id == ""
    page = manifest.pages["001.png"]
    assert page.page_number == 0
    assert page.text_regions[0].region_index == 0


def test_manifest_serialization_roundtrip(make_manifest_with_story):
    """Verify that serialization preserves new fields (manga_id, page_number, region_index)."""
    manifest = make_manifest_with_story("Test project", "test_manga_id", pages_count=2, regions_per_page=1)
    serialized = manifest.to_dict()
    
    assert serialized["manga_id"] == "test_manga_id"
    assert serialized["pages"]["page_001.png"]["page_number"] == 1
    assert serialized["pages"]["page_001.png"]["text_regions"][0]["region_index"] == 0
    
    deserialized = ProjectManifest.from_dict(serialized)
    assert deserialized.manga_id == "test_manga_id"
    assert deserialized.pages["page_001.png"].page_number == 1
    assert deserialized.pages["page_002.png"].text_regions[0].region_index == 0


@pytest.mark.parametrize(
    "filenames,expected_numbers",
    [
        (
            ["001.png", "002.png", "003.png"],
            {"001.png": 1, "002.png": 2, "003.png": 3}
        ),
        (
            ["page_10.png", "page_02.png", "page_1.png"],
            {"page_1.png": 1, "page_02.png": 2, "page_10.png": 3}
        ),
        (
            ["1.jpg", "10.jpg", "2.jpg"],
            {"1.jpg": 1, "2.jpg": 2, "10.jpg": 3}
        ),
        (
            ["ch1_001.png", "ch1_002.png", "ch2_001.png"],
            {"ch1_001.png": 1, "ch1_002.png": 2, "ch2_001.png": 3}
        ),
    ]
)
def test_derive_page_order_natural_sort(filenames, expected_numbers):
    """Verify natural page ordering derivation from filenames."""
    pages = {fn: PageManifest(fn, 800, 1200) for fn in filenames}
    manifest = ProjectManifest("Project", pages)
    order = derive_page_order(manifest)
    assert order == expected_numbers


@pytest.mark.parametrize(
    "manga_id,page_number,region_index,expected",
    [
        ("manga1", 1, 0, "manga1::0001::00"),
        ("manga_project", 25, 9, "manga_project::0025::09"),
        ("onepiece", 123, 15, "onepiece::0123::15"),
    ]
)
def test_build_story_position(manga_id, page_number, region_index, expected):
    """Verify story position string construction format."""
    assert build_story_position(manga_id, page_number, region_index) == expected


def test_manifest_engine_sync_disk_assigns_page_numbers():
    """Verify that ManifestEngine.sync_with_disk automatically computes page numbers."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir)
        
        # Create some files in out-of-order natural names
        img = Image.new("RGB", (1, 1), "white")
        img.save(os.path.join(input_dir, "page_10.png"))
        img.save(os.path.join(input_dir, "page_02.png"))
        img.save(os.path.join(input_dir, "page_1.png"))
        
        # Load engine
        engine = ManifestEngine(tmp_dir)
        manifest = engine.load_or_create(project_name="test_project")
        
        # Sync
        engine.sync_with_disk()
        
        assert manifest.manga_id == "test_project"
        assert manifest.pages["page_1.png"].page_number == 1
        assert manifest.pages["page_02.png"].page_number == 2
        assert manifest.pages["page_10.png"].page_number == 3
