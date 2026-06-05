import os
import sys

# ==========================================
# DYNAMIC PATH PARSING
# ==========================================

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze metadata overlap between MTG-Jamendo and GTZAN genres.")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=r"D:\dev\projects\mtg-jamendo-dataset",
        help="Path to the local mtg-jamendo-dataset repository directory."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to the autotagging genre TSV metadata file (defaults to data/autotagging_genre.tsv inside --repo_dir)."
    )
    return parser.parse_args()

# Target configuration
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

def main():
    args = parse_args()

    # Defer importing commons until after parsing args
    repo_scripts_dir = os.path.join(args.repo_dir, "scripts")
    if repo_scripts_dir not in sys.path:
        sys.path.append(repo_scripts_dir)
        
    try:
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
    
    # Initialize trackers
    total_tracks = len(tracks)
    genre_counts_single = {genre: 0 for genre in GTZAN_GENRES}
    genre_counts_multi = {genre: 0 for genre in GTZAN_GENRES}
    
    print("[-] Analyzing metadata overlap with GTZAN genres...")
    for track_id, data in tracks.items():
        raw_tags = data.get("genre", data.get("tags", []))
        extracted_genres = {clean_tag_string(t) for t in raw_tags}
        matched = extracted_genres.intersection(GTZAN_GENRES)
        
        # Single-label matches (strictly one GTZAN genre)
        if len(matched) == 1:
            genre = list(matched)[0]
            genre_counts_single[genre] += 1
            
        # Multi-label matches (matches one or more of the GTZAN genres)
        for genre in matched:
            genre_counts_multi[genre] += 1
            
    print(f"\n[+] Analyzed {total_tracks} total MTG-Jamendo tracks.\n")
    
    print("===================================================================")
    print("           MTG-JAMENDO & GTZAN GENRE OVERLAP REPORT")
    print("===================================================================")
    print(f"{'Genre':<15} | {'Single-Label Tracks':<20} | {'Total Matches (incl. Multi-label)':<30}")
    print("-" * 71)
    
    total_single_sum = 0
    total_multi_sum = 0
    for genre in sorted(GTZAN_GENRES):
        single_cnt = genre_counts_single[genre]
        multi_cnt = genre_counts_multi[genre]
        total_single_sum += single_cnt
        total_multi_sum += multi_cnt
        print(f"{genre:<15} | {single_cnt:<20} | {multi_cnt:<30}")
        
    print("-" * 71)
    print(f"{'TOTAL SUM':<15} | {total_single_sum:<20} | {total_multi_sum:<30}")
    print("===================================================================\n")
    print("[Note] Single-Label Tracks are highly recommended for clean training.")

if __name__ == "__main__":
    main()
