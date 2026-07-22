import os
import shutil
import unittest
import tempfile
from PIL import Image

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from core.manifest import ManifestEngine

class TestManifestEngine(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to act as project root
        self.test_dir = tempfile.mkdtemp()
        self.input_subdir = "test_input"
        self.input_dir = os.path.join(self.test_dir, self.input_subdir)
        
        # Initialize engine
        self.engine = ManifestEngine(
            project_dir=self.test_dir,
            input_subdir=self.input_subdir,
            manifest_filename="test_manifest.json"
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, filename: str, width: int, height: int):
        os.makedirs(self.input_dir, exist_ok=True)
        filepath = os.path.join(self.input_dir, filename)
        img = Image.new("RGB", (width, height), color="white")
        img.save(filepath)
        return filepath

    def test_load_or_create_new(self):
        manifest = self.engine.load_or_create(
            project_name="Test Project",
            source_lang="ja",
            target_lang="en"
        )
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.project_name, "Test Project")
        self.assertEqual(manifest.source_language, "ja")
        self.assertEqual(manifest.target_language, "en")
        self.assertEqual(len(manifest.pages), 0)
        self.assertTrue(os.path.exists(self.input_dir))

    def test_sync_with_disk_and_dimensions(self):
        self.engine.load_or_create()
        
        # Create a few dummy images
        self._create_dummy_image("page1.png", 600, 800)
        self._create_dummy_image("page2.jpg", 1024, 768)
        with open(os.path.join(self.input_dir, "ignored.txt"), "w") as f:
            f.write("This is not an image.")

        added, missing = self.engine.sync_with_disk()
        
        self.assertEqual(set(added), {"page1.png", "page2.jpg"})
        self.assertEqual(len(missing), 0)
        
        # Verify page details in manifest
        page1 = self.engine.get_page("page1.png")
        self.assertIsNotNone(page1)
        self.assertEqual(page1.width, 600)
        self.assertEqual(page1.height, 800)
        
        page2 = self.engine.get_page("page2.jpg")
        self.assertIsNotNone(page2)
        self.assertEqual(page2.width, 1024)
        self.assertEqual(page2.height, 768)

    def test_save_and_backup(self):
        self.engine.load_or_create()
        self._create_dummy_image("page1.png", 600, 800)
        self.engine.sync_with_disk()
        
        # Save manifest
        self.engine.save_manifest(backup=True)
        
        # Verify manifest file exists and contains correct content
        self.assertTrue(os.path.exists(self.engine.manifest_path))
        loaded_manifest = ProjectManifest.load_from_file(self.engine.manifest_path)
        self.assertIn("page1.png", loaded_manifest.pages)
        
        # Modify and save again to verify backup
        self.engine.manifest.project_name = "Updated Project Name"
        self.engine.save_manifest(backup=True)
        
        backup_path = self.engine.manifest_path + ".bak"
        self.assertTrue(os.path.exists(backup_path))
        
        # Check that backup contains the original project name
        backup_manifest = ProjectManifest.load_from_file(backup_path)
        self.assertEqual(backup_manifest.project_name, "Manga Project")
        
        # Check that main manifest has the updated name
        self.assertEqual(loaded_manifest.load_from_file(self.engine.manifest_path).project_name, "Updated Project Name")

    def test_incremental_sync_with_deletions(self):
        self.engine.load_or_create()
        
        # Step 1: Add two pages
        p1_path = self._create_dummy_image("page1.png", 600, 800)
        p2_path = self._create_dummy_image("page2.png", 500, 700)
        self.engine.sync_with_disk()
        
        # Add a text region to page1
        box = BoundingBox(10, 10, 100, 100)
        region = TextRegion(id="reg1", box=box, raw_text="Hello")
        self.engine.add_text_region("page1.png", region)
        
        # Step 2: Delete page1.png from disk
        os.remove(p1_path)
        
        # Sync again
        added, missing = self.engine.sync_with_disk()
        
        # Verify result
        self.assertEqual(len(added), 0)
        self.assertEqual(missing, ["page1.png"])
        
        # The page must still exist in manifest, but regions should have needs_review = True
        page1 = self.engine.get_page("page1.png")
        self.assertIsNotNone(page1)
        self.assertTrue(page1.text_regions[0].needs_review)

    def test_text_region_crud(self):
        self.engine.load_or_create()
        self._create_dummy_image("page1.png", 600, 800)
        self.engine.sync_with_disk()
        
        # Create text region
        box = BoundingBox(10, 10, 100, 100)
        region = TextRegion(id="region_a", box=box, raw_text="こんにちは")
        
        # Add region
        self.engine.add_text_region("page1.png", region)
        page = self.engine.get_page("page1.png")
        self.assertEqual(len(page.text_regions), 1)
        self.assertEqual(page.text_regions[0].raw_text, "こんにちは")
        
        # Duplicate region ID check
        with self.assertRaises(ValueError):
            self.engine.add_text_region("page1.png", region)
            
        # Update region
        updated_region = TextRegion(id="region_a", box=box, raw_text="こんにちは", translated_text="Hello")
        self.engine.update_text_region("page1.png", "region_a", updated_region)
        page = self.engine.get_page("page1.png")
        self.assertEqual(page.text_regions[0].translated_text, "Hello")
        
        # Delete region
        self.engine.delete_text_region("page1.png", "region_a")
        page = self.engine.get_page("page1.png")
        self.assertEqual(len(page.text_regions), 0)
        
        # Delete non-existent region check
        with self.assertRaises(KeyError):
            self.engine.delete_text_region("page1.png", "non_existent")

if __name__ == "__main__":
    unittest.main()
