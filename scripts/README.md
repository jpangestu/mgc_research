# Data Preparation and Preprocessing Scripts

This folder contains the data engineering pipeline scripts used to analyze, filter, and prepare Mel-spectrogram features from both the **GTZAN** and **MTG-Jamendo** datasets. 

Together, these scripts standardize separate music datasets into a uniform, shape-stabilized format ready for PyTorch model training.

---

## Script Index

### 1. `check_jamendo_gtzan_match.py`
A diagnostic script that analyzes the raw, multi-labeled MTG-Jamendo metadata (`autotagging_genre.tsv`) to assess overlap with the 10 classic GTZAN genres.
* **Purpose:** Inspects class sizes and distributions before copying data.
* **How to run:**
  ```bash
  # Run with local defaults:
  python scripts/check_jamendo_gtzan_match.py

  # Run with custom repository location:
  python scripts/check_jamendo_gtzan_match.py --repo_dir /path/to/mtg-jamendo-dataset
  ```
* **Key Concept:** Distinguishes between **Single-Label Tracks** (songs representing exactly *one* target genre, ideal for clean training) and **Multi-Label Matches** (songs blending multiple target genres).

### 2. `extract_jamendo_gtzan_subset.py`
The main data engineering script that extracts a clean, balanced, single-labeled subset of Mel-spectrogram arrays from the massive MTG-Jamendo dataset.
* **Purpose:** Prepares the Jamendo counterpart for genre classification.
* **How to run:**
  ```bash
  # Run with local defaults:
  python scripts/extract_jamendo_gtzan_subset.py

  # Run with customized paths:
  python scripts/extract_jamendo_gtzan_subset.py \
      --repo_dir /path/to/mtg-jamendo-dataset \
      --src_dir /path/to/mtg-jamendo \
      --dest_dir /path/to/mtg-jamendo_gtzan_subset \
      --csv_file data/processed/jamendo_gtzan_single_label.csv \
      --max_cap 361
  ```
* **Pipeline Steps:**
  1. Parses raw Jamendo metadata and filters tags (e.g., normalizes "hip hop" or "rap" to `hiphop`).
  2. Enforces **strictly single-labeled** tracks matching only one GTZAN genre.
  3. Enforces **strict class balancing** by capping all classes at exactly **361** tracks (governed by the `disco` bottleneck).
  4. Saves an index tracker file to `data/processed/jamendo_gtzan_single_label.csv`.
  5. Isolates and copies the selected `.npy` arrays to `D:\dev\dataset\mtg-jamendo_gtzan_subset`.

### 3. `generate_gtzan_melspecs.py`
Converts raw audio tracks from the GTZAN dataset into log-scaled Mel-spectrogram matrices that match the exact array layout of the MTG-Jamendo subset.
* **Purpose:** Converts raw wave audio (`.wav`, `.au`) to standardized pre-processed feature matrices.
* **How to run:**
  ```bash
  # Run with local defaults:
  python scripts/generate_gtzan_melspecs.py

  # Run with customized paths:
  python scripts/generate_gtzan_melspecs.py \
      --src_dir /path/to/genres_original \
      --dest_dir /path/to/gtzan_melspecs \
      --csv_file data/processed/gtzan_single_label.csv
  ```
* **Key Specifications:**
  * Downsamples audio to **16kHz**.
  * Extracts **96 Mel bands**.
  * Constrains the time dimension to exactly **1,366 frames** using structural padding/trimming.
  * Saves `.npy` arrays and writes a clean tracker index to `data/processed/gtzan_single_label.csv`.

### 4. `convert_pth_to_onnx.py`
Converts trained PyTorch model checkpoints (`.pth`) to the optimized, self-contained ONNX format.
* **Purpose:** Exports trained classifier weights into a portable format for production deployment or cross-platform runtime inference.
* **How to run:**
  ```bash
  # Auto-detect defaults (looks for best_genre_model_jamendo.pth/best_genre_model_gtzan.pth in models/):
  python scripts/convert_pth_to_onnx.py

  # Convert a specific model checkpoint:
  python scripts/convert_pth_to_onnx.py --model_path models/best_genre_model_jamendo.pth
  ```
* **Key Steps:**
  1. Instantiates a new `GenreClassifierCNN` with the target number of output classes (default is 10).
  2. Loads weights from the selected `.pth` checkpoint.
  3. Traces the model network with a dummy input of shape `[1, 1, 96, 1366]` using PyTorch's internal exporter.
  4. Bundles the model parameters and architecture into a single, self-contained `.onnx` binary file under the `models/` directory (e.g. `mgc_model_jamendo.onnx` or `mgc_model_gtzan.onnx`), automatically cleaning up auxiliary external weights files.
