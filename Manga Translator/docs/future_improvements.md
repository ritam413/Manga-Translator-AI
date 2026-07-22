# Future Improvements & Extensions

This document outlines proposed future enhancements, optimization opportunities, and edge-case handling for the Manga Translator pipeline and its Machine Learning Finetuning tools.

---

## 🚀 Proposed Improvements

### 1. Automatic GPU-to-CPU Fallback (VRAM Out-Of-Memory Recovery)

#### Context & Problem
The dataset extractor ([dataset_extractor.py](file:///e:/u%20torrentz/Manga%20Translator/scripts/dataset_extractor.py)) and the pipeline workers run heavy PyTorch-based deep learning models (`EasyOCR` and `MangaOCR`) on the GPU by default. On hardware with limited VRAM (such as the target 4GB NVIDIA GTX 1050ti), running batch processes can lead to sudden CUDA Out of Memory (OOM) exceptions. 

Currently, when a CUDA OOM occurs:
1. PyTorch throws a `RuntimeError` or `torch.cuda.OutOfMemoryError`.
2. The extractor catches the generic exception, logs it, skips the current image pair, and attempts the next pair using the GPU again.
3. Because the GPU memory is not cleared and the script does not switch to CPU mode, subsequent image pairs continue to fail.

#### Proposed Solution
Implement a robust, dynamic recovery mechanism to switch to CPU execution automatically upon encountering VRAM exhaustion.

1. **Catch CUDA OOM Exceptions:**
   Wrap the reader loading and inference calls in a try-except block specifically catching PyTorch CUDA OOM errors:
   ```python
   try:
       # EasyOCR/MangaOCR operations
   except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
       if "out of memory" in str(e).lower():
           # Trigger fallback
   ```

2. **Clear CUDA Cache & Release Resources:**
   Explicitly garbage collect the loaded model instances and call `torch.cuda.empty_cache()` to free up GPU memory.

3. **Dynamic CPU Fallback:**
   Change the run flag (`self.gpu = False`), reload the models (`EasyOCR` and `MangaOCR`) on CPU, and retry processing the failed image pair.

4. **Logged Warning:**
   Log a clear warning so the user knows the GPU was exhausted and a CPU fallback was activated.

### 2. Deterministic Tone Scoring Engine

#### Context & Problem
Currently, the TypesetterWorker applies a single font to all translated text, with only basic formatting (ALL CAPS, white stroke outline). Manga heavily relies on typography to convey tone (shouting, thinking, whispering, narration). Previous if/elif proposals based solely on text patterns create brittle rules where real dialogue triggers multiple branches simultaneously.

#### Proposed Architecture (Resolved V2)
Implement a zero-ML, zero-VRAM, pure-Python deterministic scoring engine that evaluates multiple signals (bubble shape, punctuation, case, geometry) and uses a weighted voting system to assign the optimal font variant.

1. **Bubble Shape Classifier (`quality/bubble_shape.py`)**
   - Implements `detect_bubble_contour(page_image, bbox, padding=15)` to dynamically crop and threshold the inpainted page image (detecting the bubble contour).
   - Classifies bubble shapes into categories: `RECTANGLE`, `OVAL`, `SPIKY`, `CLOUD`, or `UNKNOWN` using contour math.

2. **Tone Scoring Engine (`quality/tone_engine.py`)**
   - Evaluates weighted signals: bubble shape, punctuation patterns, text case ratios, and bounding box geometry.
   - Assigns tone using `argmax` selection (`SHOUT`, `THOUGHT`, `WHISPER`, `NARRATOR`, `STANDARD`).
   - Handles `color_for_speaker` mapping logic, defaulting to fallback if `speaker_id` is empty.

3. **Smart Word Wrap**
   - Replaces fixed word-count wrapping with a width-aware greedy-fit + rebalance algorithm in Stage 10.

4. **Integration into Stage 10**
   - Wires the predicted tone into `TypesetterWorker` to dynamically select fonts via `FONT_MAP`.
   - Optional feature flag `tone_engine_enabled` to toggle shape-based scoring per-project.
   - Serializes `debug_tone_reasons` back to the `TextRegion` manifest. Note: `speaker_id` requires prerequisite item #6 to be built first.
