import unittest
from unittest.mock import MagicMock, patch
from core.progress import MangaProgress


class TestMangaProgress(unittest.TestCase):
    def test_init_and_context_manager(self):
        with patch('core.progress.RichProgress') as MockRichProgress:
            mock_progress_instance = MockRichProgress.return_value
            
            with MangaProgress(transient=True) as mp:
                self.assertIsNotNone(mp)
                
            mock_progress_instance.__enter__.assert_called_once()
            mock_progress_instance.__exit__.assert_called_once()

    def test_add_task_and_update(self):
        with patch('core.progress.RichProgress') as MockRichProgress:
            mock_progress_instance = MockRichProgress.return_value
            mock_progress_instance.add_task.return_value = 1
            
            mp = MangaProgress()
            task_id = mp.add_task("test task", total=10)
            self.assertEqual(task_id, 1)
            mock_progress_instance.add_task.assert_called_once_with("test task", total=10, visible=True)
            
            mp.update(1, completed=5)
            mock_progress_instance.update.assert_called_once_with(1, completed=5)
            
            mp.advance(1, 2)
            mock_progress_instance.advance.assert_called_once_with(1, 2)

    def test_track(self):
        with patch('core.progress.RichProgress') as MockRichProgress:
            mock_progress_instance = MockRichProgress.return_value
            mock_progress_instance.add_task.return_value = 1
            
            mp = MangaProgress()
            items = [1, 2, 3]
            tracked = list(mp.track(items, description="tracking"))
            
            self.assertEqual(tracked, items)
            mock_progress_instance.add_task.assert_called_once_with("tracking", total=3, visible=True)
            self.assertEqual(mock_progress_instance.advance.call_count, 3)

    def test_print(self):
        with patch('core.progress.RichProgress') as MockRichProgress:
            mock_progress_instance = MockRichProgress.return_value
            mp = MangaProgress()
            mp.print("hello world")
            mock_progress_instance.console.print.assert_called_once_with("hello world")
