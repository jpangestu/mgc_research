import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class MusicGenreDataset(Dataset):
    def __init__(self, csv_file, base_dir, label_to_idx=None):
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        
        if label_to_idx is None:
            unique_genres = sorted(self.df['label'].unique())
            self.label_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
        else:
            self.label_to_idx = label_to_idx
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_dir, row['rel_path'])
        
        # Lazy load raw matrix
        matrix = np.load(file_path, mmap_mode='r')
        x = torch.tensor(matrix, dtype=torch.float32)
        
        # Shape Stabilization (Force time dimension to exactly 1366 frames)
        target_time = 1366
        current_time = x.size(1)
        
        if current_time > target_time:
            x = x[:, :target_time]
        elif current_time < target_time:
            pad_amount = target_time - current_time
            x = F.pad(x, (0, pad_amount), mode='constant', value=0)
            
        # FIX: Clone the tensor to break the read-only memory-map storage lock.
        # This allocates fresh, resizable RAM so the DataLoader can stack batches safely.
        x = x.clone()
            
        y = self.label_to_idx[row['label']]
        return x, y

if __name__ == "__main__":
    CSV_PATH = r"data\processed\jamendo_gtzan_single_label.csv"
    JAMENDO_BASE_DIR = r"D:\dev\dataset\mtg-jamendo_gtzan_subset"
    BATCH_SIZE = 16
    
    if os.path.exists(CSV_PATH) and os.path.exists(JAMENDO_BASE_DIR):
        print("[-] Running local dataset sanity check...")
        full_dataset = MusicGenreDataset(csv_file=CSV_PATH, base_dir=JAMENDO_BASE_DIR)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        
        x, y = next(iter(train_loader))
        print(f"[+] Loader check successful! Batch shape: {x.shape}, Labels shape: {y.shape}")
        print("[+] Dataset memory locks released. Tensors are ready for batching.")
    else:
        print("[!] Local sanity check skipped: CSV index or dataset folder not found.")