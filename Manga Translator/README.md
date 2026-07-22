# Manga Translator Pipeline & Model Finetuning Suite

Welcome to the **Manga Translator** repository. This project consists of two primary modules:
1. **Model Finetuning Suite**: Tools to extract training datasets from paired manga pages (raw Japanese and human-translated English), train a layout prediction model, and build a local Translation Memory (TM) database.
2. **Main Translation Pipeline**: An end-to-end multi-stage pipeline that translates Japanese manga pages to English.

---

## 🛠️ Environment Setup

Ensure you are using the virtual environment configured for the project:
```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

Install dependencies if needed:
- Core Pipeline: `pip install -r requirements.txt`
- Finetuning Suite: `pip install -r requirements-finetuning.txt`

---

## 📈 Module 1: How to Finetune the ML Models

Finetuning consists of three sequential steps. The source images must be placed under `training_data/raw/` (Japanese) and `training_data/en/` (English) using either a flat layout (e.g. `training_data/raw/1.jpg` ↔ `training_data/en/1.jpg`) or a nested directory per manga (e.g., `training_data/raw/MangaName/1.jpg` ↔ `training_data/en/MangaName/1.jpg`).

### Prerequisite: Renaming, Format Conversion & Resolution Alignment
Before running the dataset extraction, ensure page filenames match exactly, extensions are uniform, and resolutions are aligned. The following utility scripts are provided to prepare your training data:

1. **Page Renumbering & Format Conversion**:
   Use [renumber_images.ps1](file:///e:/u%20torrentz/Manga%20Translator/scripts/renumber_images.ps1) to renumber images sequentially starting from `1` and automatically convert them to the standard lossless **PNG** format. This ensures that raw and English files have matching filenames and extensions (preventing matching issues where raw has `.jpg` and English has `.webp`).
   
   Run the following commands in PowerShell:
   ```powershell
   # Renumber and convert raw Japanese images to PNG
   .\scripts\renumber_images.ps1 -Directory "training_data\raw" -Extensions @(".jpg", ".jpeg", ".png", ".webp")

   # Renumber and convert English translated images to PNG
   .\scripts\renumber_images.ps1 -Directory "training_data\en" -Extensions @(".jpg", ".jpeg", ".png", ".webp")
   ```
   *Note: If your images are located inside subdirectories under `raw/` and `en/` (nested layout), the script will automatically recurse into all subdirectories and process each folder.*

2. **Resolution Alignment**:
   To prevent dimension failures during dataset extraction due to scaling mismatches (e.g., high-resolution raw pages vs scaled-down English releases), run the alignment script [align_image_resolutions.py](file:///e:/u%20torrentz/Manga%20Translator/scripts/align_image_resolutions.py). This will resize English translated pages to match their corresponding raw Japanese page dimensions:
   ```powershell
   python scripts/align_image_resolutions.py --raw-dir training_data/raw --eng-dir training_data/en
   ```

---

### Step 1: Extract Features & Generate Inpaint Masks
Run the dataset extractor script [dataset_extractor.py](file:///e:/u%20torrentz/Manga%20Translator/scripts/dataset_extractor.py) to detect text bubbles, match overlapping boxes, extract typesetting metrics (font size, margins, spacing), and crop inpainting masks:

```powershell
python scripts/dataset_extractor.py --raw-dir training_data/raw --eng-dir training_data/en --checkpoint-interval 50 --reset-checkpoint
```
- **GPU mode (default)**: Uses GPU for text detection.
- **CPU fallback**: Add `--no-gpu` if GPU runs out of VRAM.
- **Outputs**:
  - `training_data/dataset.jsonl` (contains matched bubble features and typesetting targets).
  - `training_data/masks/` (contains per-bubble inpaint mask images).

---

### Step 2: Train the Layout Optimization Model
Train a Random Forest Regressor using [train_layout_model.py](file:///e:/u%20torrentz/Manga%20Translator/scripts/train_layout_model.py) to predict typesetting parameters (font size, padding, line spacing) from the Japanese bubble geometry and text density:

```powershell
python scripts/train_layout_model.py --dataset-file training_data/dataset.jsonl --model-out assets/models/layout_model.pkl --tune
```
- `--tune`: Optional flag to run RandomizedSearchCV hyperparameter optimization.
- **Output**: Serialized model at `assets/models/layout_model.pkl`.

---

### Step 3: Build the Translation Memory Database
Parse the extracted dataset using [build_translation_memory.py](file:///e:/u%20torrentz/Manga%20Translator/scripts/build_translation_memory.py) to compile a local, fast Translation Memory (TM) database:

```powershell
python scripts/build_translation_memory.py --dataset-file training_data/dataset.jsonl --output-file assets/models/translation_memory.json --vocab-threshold 0.3
```
- **Output**: Compiled Translation Memory database at `assets/models/translation_memory.json`.

---

## 🚀 Module 2: How to Run the Translation Pipeline

The main translation pipeline is fully automated using [main.py](file:///e:/u%20torrentz/Manga%20Translator/main.py) and is executed in a **Plan-Execute** architecture.

### Prerequisite: Directory Layout
Place the manga pages to translate inside an input directory within your project:
```
project_folder/
└── input/            # Place raw Japanese manga page images here
```

---

### Step 1: Plan the Execution
Initialize or update the project manifest (`project_manifest.json`) by scanning the input directory:
```powershell
python main.py plan --project-dir "/path/to/project_folder"
```

---

### Step 2: Execute the Pipeline Stages

To translate the planned pages, you can execute all stages or target specific steps.

#### Option A: Run the Entire Pipeline (Recommended)
This runs all 5 translation stages sequentially:
```powershell
python main.py execute all --project-dir "/path/to/project_folder"
```

#### Option B: Run a Single Stage
Run a specific stage by passing its name:
```powershell
python main.py execute --stage ocr --project-dir "/path/to/project_folder"
```

#### Option C: Run a Range of Stages
Execute a range of stages (e.g. from translation to typesetting):
```powershell
python main.py execute --from-stage translate --to-stage typeset --project-dir "/path/to/project_folder"
```

---

### 📋 Overview of Pipeline Stages

You can list all stages using:
```powershell
python main.py stages
```

| Stage ID | Stage Name | Description |
| :--- | :--- | :--- |
| **6** | `detect-text` | Detects text bounding boxes via EasyOCR |
| **7** | `ocr` | Transcribes Japanese text via MangaOCR |
| **8** | `translate` | Translates transcribed Japanese to English via ArgosTranslate (offline) or Google Translator (online) |
| **9** | `inpaint` | Erases original Japanese text from bubbles using OpenCV |
| **10** | `typeset` | Renders styled, wrapped English text on the inpainted pages |

---

### ⚙️ Useful Execution Options
Customize the translation execution using the following arguments in the `execute` subcommand:
- `--force`: Force a re-run of all pages even if they are marked completed in the manifest.
- `--limit N`: Limit processing to the first `N` pages (numerically sorted).
- `--pages filename1.png filename2.png`: Process only the specified pages.
- `--input-subdir dirname`: Change the input subdirectory name (defaults to `input`).
- `--model-version {v1,v2}`: Model version to use (v1: single-word/column-based, v2: merged-bubble). Defaults to v2 or prompts for selection.
- `--max-words-per-line N`: Maximum number of words per typeset line (overrides config setting).

---

## 🖥️ Module 3: How to Run the Web UI Dashboard & Editor

You can run the web-based visual dashboard and editor to manage projects, edit speech bubbles, and view side-by-side translations.

The Web UI consists of three services:
1. **Engine API** (Port 8001): Exposes REST APIs for the translation pipeline and model training.
2. **Backend API** (Port 8000): FastAPI bridge gateway.
3. **Frontend** (Port 3000): Next.js interactive web app.

### Option A: Automatic Launch (Recommended)
You can launch all three services automatically using the provided PowerShell script:
```powershell
# Run the start script from the webui directory
cd "e:\u torrentz\webui"
.\start.ps1
```

### Option B: Manual Launch
Alternatively, you can start the services manually in separate terminals:

1. **Start the Engine API** (in `Manga Translator` folder):
   ```powershell
   cd "e:\u torrentz\Manga Translator"
   .\venv\Scripts\python.exe engine_api.py
   ```

2. **Start the Backend API** (in `webui/backend` folder):
   ```powershell
   cd "e:\u torrentz\webui\backend"
   .\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
   ```

3. **Start the Frontend Next.js app** (in `webui/frontend` folder):
   ```powershell
   cd "e:\u torrentz\webui\frontend"
   npm run dev
   ```

Once all three are running, open your browser and go to:
- Dashboard: http://localhost:3000
- Backend docs: http://localhost:8000/docs
- Engine health check: http://localhost:8001/api/engine/health
