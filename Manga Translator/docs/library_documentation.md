# Manga Translator Pipeline — Library Documentation Reference

This document compiles the API references, initialization patterns, and usage guidelines for the pinned library versions specified in `requirements.txt` of the Manga Translator pipeline.

---

## 📌 Library Version Matrix

| Library | Pinned Version | Primary Use Case |
| :--- | :--- | :--- |
| **`onnxruntime-gpu`** | `1.17.1` | Hardware acceleration for AI/deep-learning model inference. |
| **`manga-ocr`** | `0.1.11` | OCR specialized in Japanese manga text bubbles (vertical/horizontal). |
| **`easyocr`** | `1.7.1` | General OCR & text localization (useful for bounding boxes & coordinates). |
| **`deep-translator`** | `1.11.4` | Online translation wrapper (Google Translate, MyMemory) for low-latency translation. |
| **`argostranslate`** | `1.11.0` | Offline neural machine translation (Japanese to English) using OpenNMT. |
| **`opencv-python-headless`** | `4.8.1.78` | Headless image pre-processing, binarization, contour detection, and text bubble extraction. |
| **`pillow`** | `10.2.0` | High-level image manipulation, cropping, resizing, and drawing typeset text. |

---

## 🚀 Detailed API References & Usage

### 1. ONNX Runtime GPU (`onnxruntime-gpu==1.17.1`)
ONNX Runtime is used to execute deep learning models with GPU acceleration (CUDA). Pinned at version `1.17.1`.

> [!IMPORTANT]
> Since we are running in a virtual environment on Windows, you must ensure that matching CUDA (e.g., CUDA 11.8 or 12.x depending on the build) and cuDNN libraries are present in your system's PATH. If `onnxruntime` fails to locate the CUDA DLLs, it will automatically fall back to CPU inference.

#### 💡 Initialization and Session Configuration
```python
import onnxruntime as ort

# Check available hardware execution providers
providers = ort.get_available_providers()
print("Available execution providers:", providers)
# Expected output: ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Configure CUDA Execution Provider specific options
cuda_options = {
    'device_id': 0,                            # Set GPU index
    'arena_extend_strategy': 'kNextPowerOfTwo', # Optimize VRAM growth
    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,   # Optional: limit VRAM usage (e.g., 4GB)
    'cudnn_conv_algo_search': 'EXHAUSTIVE',     # Exhaustive search for best conv algorithms
    'do_copy_in_default_stream': True          # Thread-safety/performance recommendation
}

# Define fallback order: CUDA first, CPU as backup
session_providers = [
    ('CUDAExecutionProvider', cuda_options),
    'CPUExecutionProvider'
]

# Set general Session Options
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load ONNX model session
session = ort.InferenceSession("model.onnx", sess_options=sess_options, providers=session_providers)
```

---

### 2. Manga OCR (`manga-ocr==0.1.11`)
Specifically tuned for reading Japanese text in manga. Pinned at `0.1.11`. It uses PyTorch under the hood and automatically downloads model weights (`~444MB`) on its first run.

#### 💡 Python Usage
```python
from manga_ocr import MangaOcr
from PIL import Image

# 1. Initialize the OCR engine
# The model will download to standard huggingface cache folder upon first instantiation.
# It automatically runs on CUDA if PyTorch finds a GPU, otherwise falls back to CPU.
mocr = MangaOcr()

# 2. Open an image using PIL
img = Image.open('manga_bubble.png')

# 3. Perform OCR
# The __call__ method accepts either a file path string or a PIL.Image object.
text = mocr(img)

print("Recognized Text:", text)
```

> [!TIP]
> `manga-ocr` performs best on cropped text bubble panels or individual text columns rather than the entire raw page. Feed it bounding box crops extracted during text detection for maximum transcription accuracy.

---

### 3. EasyOCR (`easyocr==1.7.1`)
An OCR library built on PyTorch that performs both text detection (finding bounding boxes) and recognition. Pinned at `1.7.1`.

#### 💡 Bounding Box Detection and Recognition
```python
import easyocr
import cv2

# Initialize Reader
# Provide list of language codes. For Japanese manga OCR cross-checking, use ['ja', 'en'].
# gpu=True triggers CUDA acceleration.
reader = easyocr.Reader(['ja', 'en'], gpu=True)

# Path to the image file, numpy array (OpenCV), or raw image bytes
img_path = 'manga_page.png'

# Perform OCR
# Results contains a list of tuples: (bounding_box_coordinates, text, confidence_score)
results = reader.readtext(img_path)

for (bbox, text, prob) in results:
    # bbox format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    (top_left, top_right, bottom_right, bottom_left) = bbox
    print(f"Coordinates: {top_left} -> {bottom_right}")
    print(f"Detected: {text} | Confidence: {prob:.4f}")
```

#### 💡 Text Grouping (Paragraph Mode)
If you want EasyOCR to merge adjacent lines that form a paragraph/bubble automatically:
```python
# Setting paragraph=True attempts to cluster contiguous bounding boxes
paragraphs = reader.readtext('manga_page.png', paragraph=True)
for (bbox, text) in paragraphs:
    print(f"Merged Paragraph Box: {bbox} | Content: {text}")
```

---

### 4. Deep Translator (`deep-translator==1.11.4`)
A tool to interact with free/paid translation services. Pinned at `1.11.4`.

#### 💡 Google Translate & MyMemory Translations
```python
from deep_translator import GoogleTranslator, MyMemoryTranslator

# Translate a single string from Japanese ('ja') to English ('en')
translated = GoogleTranslator(source='ja', target='en').translate("お前はもう死んでいる。")
print(translated) # Output: You are already dead.

# Translate a list of text strings in batch
texts = ["こんにちは", "さようなら"]
translations = GoogleTranslator(source='ja', target='en').translate_sentences(texts)
print(translations) # Output: ['Hello', 'Goodbye']

# Using MyMemory API as a fallback translator
mymemory_translated = MyMemoryTranslator(source='ja', target='en').translate("お元気ですか？")
print(mymemory_translated) # Output: How are you?
```

---

### 5. Argos Translate (`argostranslate==1.11.0`)
A local, offline translation library using OpenNMT underneath. Pinned at `1.11.0`.

> [!WARNING]
> Since Argos Translate works offline, you must explicitly download and install the required translation packages (e.g., Japanese to English) before performing translations in code.

#### 💡 Downloading and Translating Offline
```python
import argostranslate.package
import argostranslate.translate

# Target language pair codes
from_code = "ja"
to_code = "en"

# 1. Update the translation packages index
argostranslate.package.update_package_index()

# 2. Query available packages
available_packages = argostranslate.package.get_available_packages()

# 3. Find the Japanese-to-English package
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, 
        available_packages
    ), None
)

if package_to_install:
    # 4. Download and install locally
    print(f"Installing offline model {from_code} -> {to_code}...")
    argostranslate.package.install_from_path(package_to_install.download())
else:
    print("Language pack not found.")

# 5. Perform offline translation
translated_text = argostranslate.translate.translate("お名前は何ですか？", from_code, to_code)
print("Offline Translation Result:", translated_text)
```

---

### 6. OpenCV Headless (`opencv-python-headless==4.8.1.78`)
Headless version of OpenCV containing all the core image processing functionality but excluding high-level GUI modules (like `cv2.imshow` or `cv2.waitKey`), which are prone to crashes on servers/non-interactive terminals. Pinned at `4.8.1.78`.

#### 💡 Key Operations for Manga Processing
```python
import cv2
import numpy as np

# 1. Load an image in Grayscale (crucial for OCR pre-processing)
img_gray = cv2.imread('manga_page.png', cv2.IMREAD_GRAYSCALE)

# 2. Binarization (Otsu thresholding to separate text bubbles from artwork)
_, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Morphological Operations to merge text letters into coherent blocks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(img_thresh, kernel, iterations=2)

# 4. Find Contours representing potential text bubbles
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter based on size criteria (avoid small noise and full-page panels)
    if 20 < w < 800 and 20 < h < 800:
        print(f"Found candidate text bubble at: X={x}, Y={y}, Width={w}, Height={h}")
        # Crop region for OCR
        crop = img_gray[y:y+h, x:x+w]
        # Save or pass to OCR engine
```

---

### 7. Pillow (`pillow==10.2.0`)
Pillow provides simple operations for cropping, drawing text, and manipulating images before/after typesetting. Pinned at `10.2.0`.

#### 💡 Image Manipulation and Drawing
```python
from PIL import Image, ImageDraw, ImageFont

# 1. Open and query properties
img = Image.open('manga_page.png')
w, h = img.size
print(f"Resolution: {w}x{h} | Mode: {img.mode}")

# 2. High-quality cropping of text boxes
# Box format: (left, top, right, bottom)
crop_area = (120, 200, 280, 450)
bubble_crop = img.crop(crop_area)

# 3. Drawing text (Typesetting step)
# Draw bounding box outline or fill it with white background (inpainting baseline)
draw = ImageDraw.Draw(img)
draw.rectangle(crop_area, fill="white")  # Inpaint target region

# 4. Drawing text with custom fonts
try:
    font = ImageFont.truetype("arial.ttf", 18)
except IOError:
    font = ImageFont.load_default()

# Wrap and draw translated text inside the typeset box
text_position = (130, 210)
draw.text(text_position, "Here is the translated text.", fill="black", font=font)

# 5. Save output
img.save('manga_translated_output.png')
```

---

## ⚡ VRAM Optimization & Double-Stage Pipeline Constraints

When building stages using these libraries:
1. **Model Cache Pre-checks:** Check if local caches or CUDA dependencies are met in initialization functions before calling main execution loops.
2. **Resource Release:** Avoid keeping heavy readers/models (`EasyOCR`, `MangaOcr`, `onnxruntime` sessions) active simultaneously if VRAM is constrained. Explicitly instantiate and clean them up (using PyTorch `torch.cuda.empty_cache()` if needed) after processing.
