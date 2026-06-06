# Music Genre Classification Research

This repository contains a modular deep learning pipeline built in PyTorch to train and evaluate a Convolutional Neural Network (CNN) for music genre classification. 

The project unifies and standardizes two separate audio datasets—**GTZAN** and a subset of **MTG-Jamendo**—into identical log-scaled Mel-spectrogram features, enabling cross-dataset generalization experiments.

---

## Project Structure

```text
mgc_research/
│
├── data/
│   └── processed/          # Contains CSV index mapping track IDs to labels and relative file paths
│
├── models/                 # Destination directory for saving trained model
│
├── notebooks/
│   └── colab_training.ipynb	# Google Colab setup for automated Kaggle dataset download and GPU training
│
├── scripts/
│   ├── check_jamendo_gtzan_match.py		# Checks metadata overlap between MTG-Jamendo and the 10 GTZAN genres
│   ├── extract_jamendo_gtzan_subset.py	    # Filters, single-label validates, and balance-caps the Jamendo subset
│   ├── generate_gtzan_melspecs.py			# Converts GTZAN raw wave audio to standardized log-Mel spectrogram arrays
│   ├── cross_evaluate.py					# Evaluates a trained model checkpoint on the target opposite dataset
│   ├── convert_pth_to_onnx.py        # Python script to convert PyTorch .pth checkpoints to ONNX format
│   └── README.md							# Usage instructions and parameter descriptions for scripts
│
├── src/                    # Core modular PyTorch components imported by training and evaluation scripts
│   ├── __init__.py         # Exposes package-level modules for direct importing
│   ├── dataset.py          # Custom PyTorch Dataset class for lazy-loading Mel-spectrogram .npy files
│   ├── models.py           # Convolutional Neural Network (CNN) classifier model definition
│   └── training.py         # PyTorch training epoch, validation loop, and accuracy metric helpers
│
├── .gitignore              # Excludes heavy local dataset arrays and system caches while tracking CSV indices
├── main.py                 # Orchestrates model training pipelines on datasets via command-line arguments
└── README.md               # Main project documentation covering setup, execution, and structure (this file)
```

---

## Features

* **Unified Input Specification:** Standardizes raw wave audio and processed matrices to exactly **96 Mel bands** and **1,366 time frames** (16kHz sampling rate).
* **Strict Class Balancing:** Implements a class-balancing mechanism that caps extracted data at **361 tracks per genre** to avoid class bias.
* **Separation of Concerns:** Keep model definitions and data loaders isolated in `src/` so they can be easily imported like a library.
* **Google Colab Optimization:** Integrated with `kagglehub` for anonymous, zero-credential dataset downloads.
* **ONNX Export Support:** Convert trained PyTorch checkpoints to self-contained ONNX models for deployment and cross-platform inference.

---

## Getting Started

### 1. Requirements & Environment Setup
To run the code locally, set up a virtual environment (e.g., Anaconda/Miniconda) and install dependencies:

```bash
# Create and activate conda environment
conda create -n mgc_research python=3.11 -y
conda activate mgc_research

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Configure Local Settings (Optional)
If using VS Code/Antigravity and your Conda path is not in the system `PATH`, you can set the local Python path in `.vscode/settings.json`:
```json
{
    "python.condaPath": "C:\\Users\\<Username>\\miniconda3\\Scripts\\conda.exe",
    "python.useEnvironmentsExtension": false
}
```

---

## How to Run

### 1. Data Preparation (Local Machine)
If you have raw datasets locally and want to extract the Mel-spectrogram features:

* **GTZAN Mel-spectrogram Extraction:**
  ```bash
  python scripts/generate_gtzan_melspecs.py
  ```
* **MTG-Jamendo Extraction & Balancing:**
  ```bash
  python scripts/extract_jamendo_gtzan_subset.py
  ```

### 2. Model Training (Local CLI)
To train the CNN model locally on the extracted data:
```bash
python main.py --epochs 15 --batch_size 32 --lr 0.001
```
Run `python main.py --help` to view all configurable parameters (such as `save_dir`, `num_workers`, etc.).

### 3. Model Training (Google Colab GPU)
For faster GPU training, upload this folder to Google Drive and open:
`notebooks/colab_training.ipynb`

This notebook will download the preprocessed Mel-spectrogram datasets from Kaggle using `kagglehub` (no API key required), and launch training with GPU acceleration.

### 4. Cross-Dataset Generalization Testing
To evaluate how well a model trained on one dataset generalizes to the other:
```bash
# Evaluate a Jamendo-trained model on the GTZAN dataset
python scripts/cross_evaluate.py \
    --model_path "models/best_genre_model_jamendo.pth" \
    --csv_file "data/processed/gtzan_single_label.csv" \
    --base_dir "path/to/gtzan_melspecs"
```

### 5. Exporting Models to ONNX
To convert trained PyTorch checkpoints into ONNX format for deployment or cross-platform inference:

```bash
# Auto-detects and exports default checkpoints in the models/ directory:
python scripts/convert_pth_to_onnx.py

# Export a specific checkpoint file:
python scripts/convert_pth_to_onnx.py --model_path models/best_genre_model_jamendo.pth
```
This script loads the checkpoint weights, instantiates the model structure, runs a dummy input trace, and outputs self-contained `.onnx` files (e.g., `mgc_model_jamendo.onnx` and `mgc_model_gtzan.onnx`) inside the `models/` directory.
