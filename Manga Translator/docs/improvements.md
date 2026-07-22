# Manga Translator — Future Improvements & Roadmap

This document outlines the planned improvements for the Manga Translator pipeline based on current priorities. No major dataset re-extraction (`datasetv2.jsonl`) is required for these upgrades. Where machine learning is used, it will rely on the existing extracted data to train lightweight models.

## 🚀 High Priority (Immediate Next Steps)

### 1. Typography, Multiple Fonts & Tone-Based Fonts (Topmost Priority)
*   **Goal:** Move away from a single font (Bangers) and dynamically assign tone-based fonts according to the context or shape of the speech bubble (e.g., shouting, inner thoughts, standard speech).
*   **Implementation:** 
    *   Integrate a small set of standard manga fonts (e.g., Wild Words for standard speech, Anime Ace Italic for thoughts, Impact for shouting).
    *   Use bounding box heuristics (e.g., aspect ratio, size, jaggedness if contour detection is added) or a lightweight classifier to automatically pick the best font for the tone of the dialogue. 
    *   **No dataset re-extraction needed.**

### 2. Custom Text Styling & Colors
*   **Goal:** Allow specific characters or types of speech to be rendered in custom colors (e.g., pink text for the main character).
*   **Implementation:** 
    *   Update `TextRegion` in `core/models.py` to support `text_color` and `stroke_color` attributes.
    *   Update `TypesetterWorker` (Stage 10) to read these colors and apply them via Pillow's `ImageDraw`.
    *   Colors can be assigned manually by the user in the manifest or via heuristics.
    *   **No dataset re-extraction or model retraining needed.**

### 3. Smart Word Wrapping (Bubble Fit Heuristic)
*   **Goal:** Improve how text wraps inside the rectangular bounding box so it looks more natural.
*   **Implementation:** 
    *   Update `TypesetterWorker` to use a dynamic word count heuristic per line.
    *   *Rule:* Depending on the width of the text box, keep at least 3 words and at max 5 words per line if the box is wide enough.
    *   **No ML retraining needed.**

## 🧠 Medium Priority (Lightweight ML & Heuristics)

### 4. SFX & Garbage Text Filter (Removing "oo")
*   **Goal:** Automatically detect and ignore background sound effects (SFX) and garbage OCR reads (like "oooo").
*   **Implementation:** 
    *   Train a small, lightweight CPU-based classifier (like `scikit-learn` Random Forest or SVM) to classify regions as `Speech` vs `SFX/Garbage`.
    *   We will **reuse the existing `datasetv2.jsonl`** to train this model by looking at features like text length, aspect ratio, and OCR confidence. 
    *   **Requires training a new `.pkl` model, but NO dataset re-extraction.**

### 5. Panel Detection & Strict Reading Order
*   **Goal:** Ensure text is read strictly panel-by-panel (right-to-left, top-to-bottom) so the Translation Memory context is perfectly ordered.
*   **Implementation:** 
    *   Use OpenCV contour detection (pure math/heuristics, no ML) to find the borders of manga panels.
    *   Group the detected EasyOCR text boxes into their respective panels before sorting them.
    *   **No dataset re-extraction or model retraining needed.**

### 6. Speaker Identification (Free / No-API approach)
*   **Goal:** Identify who is speaking without relying on heavy local LLMs (like Ollama) or paid APIs.
*   **Implementation Options:** 
    *   *Option A:* Use free-tier APIs if acceptable (e.g., Google Gemini Free Tier or Groq Free Tier).
    *   *Option B (Zero-Cost/Zero-LLM):* Enhance the Translation Memory (Phase 8) to use NLP dictionary lookups. For example, if the Japanese text contains specific pronouns ("ore", "watashi") or catchphrases mapped to a character in a local JSON dictionary, automatically assign that character's color.

## 🖥️ Long-Term Enhancements

### 7. Interactive Web Dashboard
*   **Goal:** Build a GUI for the user to visually inspect and edit the `project_manifest.json` before Phase B (Inpainting/Typesetting) runs.
*   **Implementation:** 
    *   To be built using **FastAPI** (backend) and **React / NextJS** (frontend).
    *   Will allow users to click bubbles, fix OCR typos, manually change text colors, and re-assign speaker tags visually.
    *   *Note: Scheduled for development only after the High and Medium priority features are complete.*

### 8. SFX Translation & Warping (Optional / Experimental)
*   **Goal:** Translate background SFX ("ドドド") and warp the English text ("RUMBLE") to match the curve of the original art.
*   **Status:** Marked as **Optional**. Will only be explored if performance allows, as OpenCV warping can be computationally heavy.
