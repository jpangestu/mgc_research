import os
import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Runs a single training epoch and returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training Batches", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + Backward + Optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation/test set and returns loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation Batches", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, epochs, lr, device, save_dir, dataset_name="genre"):
    """Orchestrates the entire multi-epoch training process and saves the best model."""
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_model_path = os.path.join(save_dir, f"best_genre_model_{dataset_name}.pth")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"[-] Training initiated on device: {device}")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[+] Validation loss improved. Saved checkpoint to {best_model_path}")
            
    print("\n[=] Training Complete [=]")
    print(f" -> Best validation loss achieved: {best_val_loss:.4f}")
    print(f" -> Model weights saved to: {best_model_path}")
