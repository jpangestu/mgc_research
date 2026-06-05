import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from src import MusicGenreDataset, GenreClassifierCNN, train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train GenreClassifierCNN on preprocessed Mel-spectrogram arrays.")
    
    # Dataset locations
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default=r"data\processed\jamendo_gtzan_single_label.csv", 
        help="Path to the index CSV file mapping tracks to genre labels."
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default=r"D:\dev\dataset\mtg-jamendo_gtzan_subset", 
        help="Path to the root directory containing the preprocessed Mel-spectrogram (.npy) directories."
    )
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation loaders.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer.")
    
    # Execution options
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the trained model weights.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader multi-threading.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Ratio of training data (remaining goes to validation).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Device selection (CUDA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] Running on target device: {device}")
    
    # 2. Verify paths
    if not os.path.exists(args.csv_file):
        print(f"[!] Error: The CSV index file was not found at '{args.csv_file}'. Ensure preprocessing is run first.")
        return
    if not os.path.exists(args.base_dir):
        print(f"[!] Error: The dataset directory was not found at '{args.base_dir}'. Ensure files have been harvested.")
        return

    # 3. Instantiate Dataset
    print(f"[-] Loading dataset index from {args.csv_file}...")
    full_dataset = MusicGenreDataset(csv_file=args.csv_file, base_dir=args.base_dir)
    
    # 4. Train-Validation Split
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"[+] Loaded {len(full_dataset)} total tracks.")
    print(f" -> Training split size:   {train_size} tracks")
    print(f" -> Validation split size: {val_size} tracks")
    
    # 5. DataLoaders
    # Note: pin_memory=True allows faster GPU memory copying.
    pin_memory = True if device.type == "cuda" else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory
    )
    
    # 6. Instantiate Model
    num_classes = len(full_dataset.label_to_idx)
    model = GenreClassifierCNN(num_classes=num_classes).to(device)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Initialized GenreClassifierCNN with {num_classes} classes.")
    print(f" -> Total Trainable Parameters: {total_params:,}")
    
    # 7. Start Training Loop
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
