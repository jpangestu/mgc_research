import os
import sys
import csv
import shutil
from tqdm import tqdm
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ==========================================
# DYNAMIC PATH PARSING
# ==========================================

GTZAN_GENRES = {
    "blues", "classical", "country", "disco", "hiphop", 
    "jazz", "metal", "pop", "reggae", "rock"
}

def clean_tag_string(tag):
    """Strips prefixes and standardizes variations like hip-hop/hip hop"""
    if "---" in tag:
        tag = tag.split("---")[1]
    tag = tag.lower().strip()
    if tag in ["hip hop", "hip-hop", "rap"]:
        return "hiphop"
    return tag

def parse_args():
    parser = argparse.ArgumentParser(description="Extract a balanced GTZAN-aligned subset from MTG-Jamendo dataset.")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=r"D:\dev\projects\mtg-jamendo-dataset",
        help="Path to the local mtg-jamendo-dataset repository directory."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default=r"D:\dev\dataset\mtg-jamendo",
        help="Path to the directory containing source MTG-Jamendo Mel-spectrogram files."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default=r"D:\dev\dataset\mtg-jamendo_gtzan_subset",
        help="Path to the directory where the isolated subset should be saved."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed", "jamendo_gtzan_single_label.csv"),
        help="Path to save the generated CSV tracking index mapping."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to the autotagging genre TSV metadata file (defaults to data/autotagging_genre.tsv inside --repo_dir)."
    )
    parser.add_argument(
        "--max_cap",
        type=int,
        default=361,
        help="Maximum track cap limit per genre to enforce balancing."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Defer importing commons until after parsing args
    repo_scripts_dir = os.path.join(args.repo_dir, "scripts")
    if repo_scripts_dir not in sys.path:
        sys.path.append(repo_scripts_dir)
        
    try:
        # pyrefly: ignore [missing-import]
        import commons
    except ImportError:
        print(f"[!] Error: Could not import 'commons' module from {repo_scripts_dir}.")
        print("Please make sure the repository directory is correct and accessible.")
        sys.exit(1)

    metadata_file = args.metadata_file
    if metadata_file is None:
        metadata_file = os.path.join(args.repo_dir, "data", "autotagging_genre.tsv")

    print(f"[-] Parsing metadata from: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"[!] Error: File not found at {metadata_file}")
        sys.exit(1)
        
    tracks, _, _ = commons.read_file(metadata_file)
    filtered_tracks = []
    
    # Initialize dictionary to keep track of counts per genre
    genre_counts = {genre: 0 for genre in GTZAN_GENRES}
    
    print(f"[-] Filtering and balancing data (Cap limit: {args.max_cap} per genre)...")
    for track_id, data in tracks.items():
        raw_tags = data.get("genre", data.get("tags", []))
        extracted_genres = {clean_tag_string(t) for t in raw_tags}
        matched = extracted_genres.intersection(GTZAN_GENRES)
        
        if len(matched) == 1:
            single_label = list(matched)[0]
            
            # CRITICAL BALANCING STEP: Check if this genre pool is full
            if genre_counts[single_label] >= args.max_cap:
                continue
                
            rel_path = data["path"].replace(".mp3", ".npy")
            
            filtered_tracks.append({
                "track_id": str(track_id),
                "rel_path": rel_path,
                "label": single_label
            })
            
            # Increment pool count for this genre
            genre_counts[single_label] += 1
            
    total_found = len(filtered_tracks)
    print(f"[+] Successfully gathered {total_found} total balanced tracks.")
    
    # Print individual yields to verify balance
    print("\n--- Final Balanced Distribution ---")
    for genre, count in genre_counts.items():
        print(f" -> {genre.ljust(12)}: {count} tracks")
    print("-----------------------------------\n")

    if total_found == 0:
        print("[!] Error: No tracks matched specifications.")
        sys.exit(1)

    # Make target directories
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    os.makedirs(args.dest_dir, exist_ok=True)

    print(f"[-] Saving balanced tracking index to {args.csv_file}...")
    with open(args.csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "rel_path", "label"])
        writer.writeheader()
        writer.writerows(filtered_tracks)

    # Extracting and copying specific files
    print(f"[-] Harvesting files to {args.dest_dir}...")
    success_count = 0
    
    for track in tqdm(filtered_tracks, desc="Isolating matrices"):
        src_file = os.path.join(args.src_dir, track["rel_path"])
        dest_file = os.path.join(args.dest_dir, track["rel_path"])
        
        if os.path.exists(src_file):
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copy(src_file, dest_file)
            success_count += 1

    print("\n[=] Operation Summary [=]")
    print(f" -> Successfully isolated: {success_count} / {total_found} balanced .npy files.")
    print(f" -> Disk Space consumed: ~{(success_count * 4.1) / 1024:.2f} GB")
    print(f" -> Harvested destination: {args.dest_dir}")
    print("[+] Complete.")

if __name__ == "__main__":
    main()
