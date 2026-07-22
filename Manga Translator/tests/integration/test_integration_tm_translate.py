import pytest
from unittest.mock import patch, MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.translator import TranslationWorker


@pytest.mark.parametrize(
    "manga_id,jp_text,expected_translation,expected_source",
    [
        ("manga1", "こんにちは", "Hello", "tm_exact"),                 # Exact local match
        ("manga1", "こん にち は", "Hello", "tm_exact"),               # Normalized spaces match
        ("manga1", "こんにちは！", "Hello", "tm_fuzzy"),               # Fuzzy match with punctuation
        ("manga1", "こんにちは世界", None, "mt"),                       # No match -> MT
        ("manga1", "ナミ", "Nami", "tm_exact"),                         # Exact global fallback match (manga2 -> manga1)
    ]
)
@patch("workers.translator.GoogleTranslator")
def test_translation_worker_tm_lookup(mock_google_cls, synthetic_tm_db, manga_id, jp_text, expected_translation, expected_source):
    """Verify exact and fuzzy translation memory bypass paths."""
    mock_google = MagicMock()
    mock_google_cls.return_value = mock_google
    mock_google.translate.return_value = "Google translation"

    worker = TranslationWorker(prefer_offline=False, tm_db=synthetic_tm_db, fuzzy_threshold=0.8)
    worker.load()
    
    region = TextRegion(id="r00", box=BoundingBox(10, 10, 50, 50), raw_text=jp_text)
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region])
    manifest = ProjectManifest(project_name="Test", pages={"page01.png": page}, manga_id=manga_id)
    
    updated_page = worker.process_page(page, manifest)
    updated_region = updated_page.text_regions[0]
    
    if expected_translation:
        assert updated_region.translated_text == expected_translation
        assert getattr(updated_region, "translation_source", "") == expected_source
        # Verify MT backend not called
        mock_google.translate.assert_not_called()
    else:
        assert updated_region.translated_text == "Google translation"
        assert getattr(updated_region, "translation_source", "") == "mt"
        mock_google.translate.assert_called_once_with(jp_text)


@pytest.mark.parametrize("similarity_threshold,expected_translation,expected_source", [
    (0.80, "Hello", "tm_fuzzy"),  # fuzzy match is ~0.83 similarity, accepts at 0.80
    (0.95, "Google translation", "mt"),  # rejects at 0.95, falls back to MT
])
@patch("workers.translator.GoogleTranslator")
def test_translation_worker_fuzzy_thresholds(mock_google_cls, synthetic_tm_db, similarity_threshold, expected_translation, expected_source):
    """Verify that the fuzzy match threshold is respected."""
    mock_google = MagicMock()
    mock_google_cls.return_value = mock_google
    mock_google.translate.return_value = "Google translation"

    worker = TranslationWorker(prefer_offline=False, tm_db=synthetic_tm_db, fuzzy_threshold=similarity_threshold)
    worker.load()
    
    # "こんにちは！" has ~0.83 similarity to "こんにちは"
    region = TextRegion(id="r00", box=BoundingBox(10, 10, 50, 50), raw_text="こんにちは！")
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region])
    manifest = ProjectManifest(project_name="Test", pages={"page01.png": page}, manga_id="manga1")
    
    updated_page = worker.process_page(page, manifest)
    updated_region = updated_page.text_regions[0]
    
    assert updated_region.translated_text == expected_translation
    assert getattr(updated_region, "translation_source", "") == expected_source


@patch("workers.translator.GoogleTranslator")
def test_translation_worker_context_prepend_google(mock_google_cls, synthetic_tm_db):
    """Verify preceding context sentences are prepended for MT translation."""
    mock_google = MagicMock()
    mock_google_cls.return_value = mock_google
    
    # Mock Google to return translated block containing target separator
    mock_google.translate.return_value = "Hello.\nHow are you?.\n---\nActually doing good"
    
    worker = TranslationWorker(prefer_offline=False, tm_db=synthetic_tm_db, context_limit=5)
    worker.load()
    
    # Create page with chronological regions (region index 2, context should be preceding regions)
    # The synthetic DB has:
    #   r0: manga1::0001::00 -> "こんにちは"
    #   r1: manga1::0001::01 -> "お元気ですか"
    # Let's run a translation for current region manga1::0001::02
    region = TextRegion(id="r02", box=BoundingBox(10, 10, 50, 50), raw_text="実際には元気です", region_index=2)
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region], page_number=1)
    manifest = ProjectManifest(project_name="Test", pages={"page01.png": page}, manga_id="manga1")
    
    # Let's run it
    updated_page = worker.process_page(page, manifest)
    updated_region = updated_page.text_regions[0]
    
    # Preceding context JP sentences: "こんにちは。お元気ですか"
    # Separator is \n---\n
    expected_query = "こんにちは。お元気ですか\n---\n実際には元気です"
    mock_google.translate.assert_called_once_with(expected_query)
    
    # verify only target translation is extracted
    assert updated_region.translated_text == "Actually doing good"


@patch("workers.translator.GoogleTranslator")
def test_translation_worker_context_prepend_delimiter_corrupted_fallback(mock_google_cls, synthetic_tm_db):
    """Verify fallback to sentence tokenization when separator delimiter is missing."""
    mock_google = MagicMock()
    mock_google_cls.return_value = mock_google
    
    # Delimiter missing, but returns multiple sentences
    mock_google.translate.return_value = "Hello. How are you? Actually doing good."
    
    worker = TranslationWorker(prefer_offline=False, tm_db=synthetic_tm_db, context_limit=5)
    worker.load()
    
    region = TextRegion(id="r02", box=BoundingBox(10, 10, 50, 50), raw_text="実際には元気です", region_index=2)
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region], page_number=1)
    manifest = ProjectManifest(project_name="Test", pages={"page01.png": page}, manga_id="manga1")
    
    updated_page = worker.process_page(page, manifest)
    updated_region = updated_page.text_regions[0]
    
    # NLTK/regex sent_tokenize fallback should take the last sentence
    assert updated_region.translated_text == "Actually doing good."


@patch("workers.translator.GoogleTranslator")
def test_translation_worker_fallback_chain(mock_google_cls):
    """Verify that translator falls back to needs_review when all engines fail."""
    mock_google = MagicMock()
    mock_google_cls.return_value = mock_google
    mock_google.translate.side_effect = Exception("Network down")
    
    worker = TranslationWorker(prefer_offline=False, tm_db=None)
    worker.load()
    
    region = TextRegion(id="r00", box=BoundingBox(10, 10, 50, 50), raw_text="お元気ですか")
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region])
    manifest = ProjectManifest(project_name="Test", pages={"page01.png": page}, manga_id="manga1")
    
    updated_page = worker.process_page(page, manifest)
    updated_region = updated_page.text_regions[0]
    
    # Falls back to raw_text and sets needs_review = True
    assert updated_region.translated_text == "お元気ですか"
    assert updated_region.needs_review is True
