# Manga Translator — Finetuning Library Documentation Reference

This document compiles the API references and usage guidelines for new libraries introduced specifically for the ML extraction and model training upgrades. 

It supplements the base `library_documentation.md` (which covers MangaOCR, EasyOCR, ArgosTranslate, OpenCV, and Pillow).

---

## 📌 New Library Version Matrix (Proposed)

| Library | Version | Primary Use Case |
| :--- | :--- | :--- |
| **`scikit-learn`** | `1.4.1.post1` | Training the Layout Optimization Random Forest model (Phase 7). CPU-based, lightweight. |
| **`nltk`** | `3.8.1` | Tokenization and statistical word matching for the Translation Memory database (Phase 8). |
| **`numpy`** | (Existing) | Used heavily for bounding box IoU (Intersection over Union) calculations and image differencing. |
| **`opencv-python-headless`** | (Existing) | Used for generating inpaint masks (Phase 5). |

---

## 🖥️ VRAM Footprint & Hardware Resource Allocation

Since the system targets an **NVIDIA GTX 1050ti with 4GB VRAM**, preserving GPU memory for deep learning models is critical.

*   **`scikit-learn` (CPU-based)**: Training and inference for the layout optimization Random Forest model are fully CPU-bound. There is **0 MB VRAM footprint**.
*   **`nltk` (CPU-based)**: Text tokenization and Translation Memory matching are fully CPU-bound. There is **0 MB VRAM footprint**.
*   **`manga-ocr` (GPU/CPU-based)**: Transcription model occupies **~400 MB VRAM** when loaded on GPU. Combined with EasyOCR (~2 GB VRAM), the total footprint reaches **~2.4 GB VRAM**, well under the 4 GB hardware limit. Models are loaded once per batch and unloaded at checkpoint boundaries to prevent VRAM leakage. Falls back to CPU (0 MB VRAM) in `--no-gpu` or `--no-manga-ocr` modes.
*   **GPU Impact**: Co-loading is carefully managed at batch boundaries. Running concurrently is safe, but memory is freed immediately after batch completion.

---

## 🚀 Detailed API References & Usage

### 1. Scikit-Learn (`scikit-learn==1.4.1.post1`)
Used for training lightweight regressors. Because we are constrained by a GTX 1050ti and want fast iteration, a Random Forest Regressor is ideal for predicting font sizes and margins based on bubble dimensions.

#### 💡 Training a Layout Model
```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

# Features: [BubbleWidth, BubbleHeight, JapaneseCharCount]
X_train = np.array([
    [150, 300, 15],
    [200, 200, 8]
])

# Targets: [EnglishFontSize, PaddingMargin]
y_train = np.array([
    [18, 5],
    [24, 8]
])

# Initialize and train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('assets/models/layout_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict new bubble
prediction = model.predict([[160, 280, 12]])
print("Predicted [FontSize, Margin]:", prediction)
```

---

### 2. Geometric Bounding Box Matching (Using Existing Numpy)
In Phase 3, we must match Japanese text boxes to English text boxes on the same page. We use Intersection over Union (IoU).

#### 💡 IoU Calculation
```python
import numpy as np

def calculate_iou(boxA, boxB):
    # box format: [x_min, y_min, x_max, y_max]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
```

---

### 3. Image Differencing for Mask Extraction (Using Existing OpenCV)
In Phase 5, we generate the exact shape of the inpaint mask by comparing the raw image to the translated image, using a per-region crop instead of the full page.

#### 💡 Mask Generation
```python
import cv2
import numpy as np
from finetuning.typesetting_features import get_text_ink_mask

# Load raw Japanese and finished English pages
img_raw = cv2.imread('raw/manga1/1.jpg', cv2.IMREAD_GRAYSCALE)
img_eng = cv2.imread('eng/manga1/1.jpg', cv2.IMREAD_GRAYSCALE)

# We use per-region crops to isolate the text bubble.
raw_crop = img_raw[y_min:y_max, x_min:x_max]
eng_crop = img_eng[y_min:y_max, x_min:x_max]

# Absolute difference highlights where pixels changed (where Japanese text was erased)
diff = cv2.absdiff(raw_crop, eng_crop)

# Threshold to create a clean binary mask
_, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Subtract English ink using Otsu binarization to avoid keeping EN glyphs in the mask
en_ink, _, crop_origin = get_text_ink_mask(img_eng, en_box)
# (Dilate en_ink and subtract from diff_mask...)
erase_mask = diff_mask & ~en_ink_dilated

# Morphological opening/closing to clean noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean_mask = cv2.morphologyEx(erase_mask, cv2.MORPH_OPEN, kernel, iterations=1)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Save mask crop for training Phase 10 models
cv2.imwrite('training_data/masks/manga1_1_region0.png', clean_mask)

---

### 4. Text Normalization and Ordering (Using Stdlib and PageOrderTracker)
In Phase 6, we normalize full-width alphanumeric ASCII characters to half-width for consistent downstream Translation Memory mapping. We also track sequential page numbers.

#### 💡 Text Normalization (NFKC)
```python
import unicodedata

text = " ＡＢＣ１２３ "
# Strip and normalize full-width -> half-width
normalized = unicodedata.normalize("NFKC", text.strip())
print(normalized)  # Output: "ABC123"
```

#### 💡 PageOrderTracker Usage
```python
from pathlib import Path
from finetuning.page_order import PageOrderTracker

# Pairs list: (raw_path, eng_path, manga_name, img_name)
pairs = [
    (Path("raw/m1/page_02.png"), Path("eng/m1/page_02.png"), "m1", "page_02.png"),
    (Path("raw/m1/001.png"), Path("eng/m1/001.png"), "m1", "001.png"),
]

tracker = PageOrderTracker(pairs)
page_num = tracker.get_page_number("m1", "page_02.png")  # 2
story_pos = tracker.get_story_position("m1", "page_02.png", 0)  # "m1::0002::00"
```
```

---

### 5. Translation Memory and Word Association (Using NLTK and Stdlib Regex)
In Phase 8, we build a local Translation Memory (TM) Database. We use `nltk.metrics.edit_distance` to compute string similarity and custom Jaccard metrics on token runs for terminology alignment.

#### 💡 Fuzzy Matching with edit_distance
```python
from nltk.metrics import edit_distance

dist = edit_distance("こんにちは世界", "こんいちは世界")
max_len = max(len("こんにちは世界"), len("こんいちは世界"))
similarity = 1.0 - (dist / max_len)
# similarity = 1.0 - (1 / 7) = ~0.857
```

#### 💡 Statistical Terminology Co-occurrence (Jaccard Association)
We isolate Kanji and Katakana runs from Japanese text, and lowercase English words (ignoring common stop words):
```python
import re

# katakana and kanji regex
jp_regex = re.compile(r'[\u30a0-\u30ff\u30fc]+|[\u4e00-\u9fff]+')
en_regex = re.compile(r'[a-zA-Z]+')

jp_tokens = jp_regex.findall("ルフィが走る") # -> ['ルフィ', '走']
en_tokens = en_regex.findall("Luffy runs")   # -> ['Luffy', 'runs']
```
Jaccard score is then computed to map proper nouns and character names dynamically based on bubble context:
$$J(j, e) = \frac{\text{co-occurrences}}{\text{count}_j + \text{count}_e - \text{co-occurrences}}$$
