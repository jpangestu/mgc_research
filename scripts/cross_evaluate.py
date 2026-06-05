import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root to python path to resolve src imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src import MusicGenreDataset, GenreClassifierCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-evaluate a trained GenreClassifierCNN checkpoint on a target dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model weights (.pth).")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the target dataset index CSV.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to the target dataset directory containing .npy files.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DataLoader.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker threads for the DataLoader.")
    return parser.parse_args()

def evaluate_cross(model, dataloader, device, label_to_idx):
    model.eval()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
            
            for label, pred in zip(labels, predicted):
                l_val = label.item()
                class_total[l_val] += 1
                if l_val == pred.item():
                    class_correct[l_val] += 1
                    
    overall_accuracy = total_correct / total_samples
    
    print("\n==================================================")
    print("             CROSS DATASET EVALUATION")
    print("==================================================")
    print(f"Target Dataset:  {dataloader.dataset.df['label'].count()} tracks")
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%\n")
    print(f"{'Genre':<15} | {'Accuracy':<10} | {'Correct/Total':<15}")
    print("-" * 46)
    
    for i in range(num_classes):
        genre_name = idx_to_label[i]
        correct = class_correct[i]
        total = class_total[i]
        acc = (correct / total * 100) if total > 0 else 0.0
        print(f"{genre_name:<15} | {acc:<9.2f}% | {correct:>4}/{total:<5}")
    print("==================================================\n")

def main():
    args = parse_args()
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] Running evaluation on device: {device}")
    
    # 2. Verify files exist
    if not os.path.exists(args.model_path):
        print(f"[!] Error: Model checkpoint file not found at: '{args.model_path}'")
        sys.exit(1)
    if not os.path.exists(args.csv_file):
        print(f"[!] Error: Target CSV index file not found at: '{args.csv_file}'")
        sys.exit(1)
    if not os.path.exists(args.base_dir):
        print(f"[!] Error: Target dataset directory not found at: '{args.base_dir}'")
        sys.exit(1)
        
    # 3. Instantiate target dataset and loader
    print(f"[-] Loading target dataset index from: {args.csv_file}")
    dataset = MusicGenreDataset(csv_file=args.csv_file, base_dir=args.base_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 4. Instantiate model and load weights
    num_classes = len(dataset.label_to_idx)
    model = GenreClassifierCNN(num_classes=num_classes)
    
    print(f"[-] Loading model weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # 5. Run evaluation
    evaluate_cross(model, dataloader, device, dataset.label_to_idx)

if __name__ == "__main__":
    main()
