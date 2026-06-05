import torch
import torch.nn as nn

class GenreClassifierCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GenreClassifierCNN, self).__init__()
        
        # Block 1: Input (Batch, 1, 96, 1366)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (Batch, 32, 48, 683)
        )
        
        # Block 2:
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (Batch, 64, 24, 341)
        )
        
        # Block 3:
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (Batch, 128, 12, 170)
        )
        
        # Block 4:
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (Batch, 256, 6, 85)
        )
        
        # Performance Optimization: Adaptive Pooling
        # Instead of flattening a massive 256 x 6 x 85 matrix (130,560 features),
        # we force the spatial dimensions down to a clean 2x2 grid.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2)) # Output: (Batch, 256, 2, 2)
        
        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4), # Prevents overfitting on the capped training data
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Ensure input tensor has a channel dimension: (Batch, 96, 1366) -> (Batch, 1, 96, 1366)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits

# =====================================================================
# LOCAL SHAPE VERIFICATION SANITY CHECK
# =====================================================================
if __name__ == "__main__":
    print("[-] Initializing model baseline...")
    model = GenreClassifierCNN(num_classes=10)
    
    # Simulate a single batch of 4 mel-spectrogram arrays matching our exact shape
    fake_input = torch.randn(4, 96, 1366)
    
    print(f"[-] Passing tensor through the network with input shape: {fake_input.shape}")
    predictions = model(fake_input)
    
    print("[+] Forward pass successful!")
    print(f" -> Output shape (Batch, Classes): {predictions.shape}")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Total Trainable Parameters: {total_params:,} (~{total_params * 4 / (1024**2):.2f} MB RAM cost)")