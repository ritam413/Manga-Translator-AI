import numpy as np
import cv2
import unittest
from quality.bubble_shape import BubbleShape, detect_bubble_contour, classify_bubble_shape
from core.models import BoundingBox

class TestBubbleShape(unittest.TestCase):
    def test_bubble_shape_enum(self):
        self.assertEqual(BubbleShape.RECTANGLE, "RECTANGLE")
        self.assertEqual(BubbleShape.OVAL, "OVAL")
        self.assertEqual(BubbleShape.SPIKY, "SPIKY")
        self.assertEqual(BubbleShape.CLOUD, "CLOUD")
        self.assertEqual(BubbleShape.UNKNOWN, "UNKNOWN")

    def test_detect_bubble_contour(self):
        # Create a synthetic image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a white rectangle in the center (speech bubble)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        
        bbox = BoundingBox(30, 30, 70, 70)
        mask = detect_bubble_contour(img, bbox, padding=5)
        
        # Crop goes from x_min-5=25 to x_max+5=75 (width 50) and y_min-5=25 to y_max+5=75 (height 50)
        # Plus clamping might change it, but it should be exactly 50x50
        self.assertEqual(mask.shape, (50, 50))

    def test_classify_rectangle(self):
        # Create a clean rectangle mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (15, 15), (85, 85), 255, -1)
        
        shape = classify_bubble_shape(mask)
        self.assertEqual(shape, BubbleShape.RECTANGLE)

    def test_classify_oval(self):
        # Create a clean oval/circle mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 40, 255, -1)
        
        shape = classify_bubble_shape(mask)
        self.assertEqual(shape, BubbleShape.OVAL)

    def test_classify_spiky(self):
        # Create a spiky/star mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Draw a star
        pts = []
        center = (100, 100)
        for i in range(20):
            r = 80 if i % 2 == 0 else 30
            angle = i * np.pi / 10
            pts.append([int(center[0] + r * np.cos(angle)), int(center[1] + r * np.sin(angle))])
        
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        shape = classify_bubble_shape(mask)
        self.assertEqual(shape, BubbleShape.SPIKY)

    def test_classify_cloud(self):
        # Create a cloud mask by drawing overlapping circles
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 45, 255, -1)
        cv2.circle(mask, (70, 100), 35, 255, -1)
        cv2.circle(mask, (130, 100), 35, 255, -1)
        cv2.circle(mask, (100, 70), 35, 255, -1)
        cv2.circle(mask, (100, 130), 35, 255, -1)
        
        shape = classify_bubble_shape(mask)
        self.assertEqual(shape, BubbleShape.CLOUD)

    def test_classify_empty_or_small(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        shape = classify_bubble_shape(mask)
        self.assertEqual(shape, BubbleShape.UNKNOWN)
