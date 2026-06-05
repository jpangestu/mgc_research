import os
import csv
import sys
import numpy as np
import librosa
import argparse
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Hyperparameters to explicitly match the MTG-Jamendo baseline shape
SAMPLE_RATE = 16000
N_MELS = 96
TARGET_FRAMES = 1366
HOP_LENGTH = 351  # Math: 480,000 samples / 351 hop length ≈ 1366 frames

GTZAN_GENRES = {"blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"}

def parse_args():
    parser = argparse.ArgumentParser(description="Convert GTZAN raw audio tracks to standardized Mel-spectrogram arrays.")
    parser.add_argument(
        "--src_dir", 
        type=str, 
        default=r"D:\dev\dataset\gtzan_source\genres_original", 
        help="Path to the directory containing raw GTZAN genres."
    )
    parser.add_argument(
        "--dest_dir", 
        type=str, 
        default=r"D:\dev\dataset\gtzan_melspecs", 
        help="Path to save the generated NumPy Mel-spectrogram matrices."
    )
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data", "processed", "gtzan_single_label.csv"), 
        help="Path to save the generated CSV index mapping."
    )
    return parser.parse_args()

def compute_jamendo_matched_melspec(audio_path):
    """Loads audio and processes it to match the exact MTG-Jamendo array layout."""
    # 1. Load audio and downsample to 16kHz to optimize tensor calculation
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=30.0)
    
    # 2. Compute Mel-spectrogram
    mel_raw = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=2048, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS,
        fmin=0.0,
        fmax=None
    )
    
    # 3. Convert power matrix to decibels (Log-scaling matching human perception)
    mel_db = librosa.power_to_db(mel_raw, ref=1.0)
    
    # 4. Strict structural padding/trimming constraint to safeguard PyTorch/TF shapes
    if mel_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_db.shape[1] > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]
        
    return mel_db

def main():
    args = parse_args()

    print(f"[-] Scanning for raw audio inside: {args.src_dir}")
    if not os.path.exists(args.src_dir):
        print(f"[!] Error: Source directory does not exist. Check step 1 extraction.")
        sys.exit(1)

    # Walk the directory to collect all valid audio files
    audio_tasks = []
    for root, _, files in os.walk(args.src_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".au")):
                # Determine the genre from the folder name or the filename prefix
                # Kaggle GTZAN structures usually have folders like 'genres_original/blues/'
                folder_name = os.path.basename(root).lower().strip()
                
                if folder_name in GTZAN_GENRES:
                    genre = folder_name
                else:
                    # Fallback to filename prefix check (e.g., 'blues.00000.wav')
                    genre = file.split('.')[0].lower().strip()
                    
                if genre in GTZAN_GENRES:
                    audio_tasks.append({
                        "src_path": os.path.join(root, file),
                        "filename": file,
                        "genre": genre
                    })

    total_files = len(audio_tasks)
    print(f"[+] Found {total_files} audio files matching target genres.")
    
    if total_files == 0:
        print("[!] Error: No valid audio files identified. Verify your extraction folder.")
        sys.exit(1)

    # Create directories
    os.makedirs(args.dest_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)

    csv_records = []
    success_count = 0

    print("[-] Converting audio files to standardized NumPy matrices...")
    for task in tqdm(audio_tasks, desc="Processing tracks"):
        try:
            # Run the conversion pipeline
            melspec_matrix = compute_jamendo_matched_melspec(task["src_path"])
            
            # Formulate cross-compatible relative paths and IDs
            track_id = os.path.splitext(task["filename"])[0]
            rel_npy_path = os.path.join(task["genre"], f"{track_id}.npy")
            full_dest_path = os.path.join(args.dest_dir, rel_npy_path)
            
            # Ensure target genre subfolder exists inside destination
            os.makedirs(os.path.dirname(full_dest_path), exist_ok=True)
            
            # Save raw matrix array directly to disk
            np.save(full_dest_path, melspec_matrix)
            
            # CRITICAL DESIGN STEP: Map headers to match the Jamendo CSV layout exactly
            csv_records.append({
                "track_id": track_id,
                "rel_path": rel_npy_path.replace("\\", "/"), # Standardize slash direction
                "label": task["genre"]
            })
            success_count += 1
            
        except Exception as e:
            # Prevent one bad audio file from crashing an entire multi-hour pipeline run
            print(f"\n[!] Skipping corrupt track {task['filename']}: {str(e)}")
            continue

    # Write the uniform index tracking file
    print(f"[-] Saving structured CSV tracking index to {args.csv_file}...")
    with open(args.csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "rel_path", "label"])
        writer.writeheader()
        writer.writerows(csv_records)

    print("\n[=] Processing Pipeline Complete [=]")
    print(f" -> Successfully converted: {success_count} / {total_files} files.")
    print(f" -> Array shape output: ({N_MELS}, {TARGET_FRAMES})")
    print(f" -> Features location: {args.dest_dir}")
    print(f" -> Mapping file location: {args.csv_file}")

if __name__ == "__main__":
    main()