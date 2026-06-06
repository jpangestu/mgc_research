import os
import torch
import sys
import argparse

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src import GenreClassifierCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Export trained PyTorch model checkpoints to ONNX format.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Provide a direct path to a specific PyTorch checkpoint .pth file."
    )
    return parser.parse_args()

def export_model(model_path, num_classes=10):
    if not os.path.exists(model_path):
        print(f"[!] Error: Model checkpoint file not found at: {model_path}")
        return False

    print(f"\n[-] Loading weights from: {model_path}")
    model = GenreClassifierCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Create dummy input matching the shape the model expects: [batch_size, channels, n_mels, frames]
    dummy_input = torch.randn(1, 1, 96, 1366)

    # Determine dataset name for output file naming
    filename_lower = os.path.basename(model_path).lower()
    if "jamendo" in filename_lower:
        dataset_name = "jamendo"
    elif "gtzan" in filename_lower:
        dataset_name = "gtzan"
    else:
        dataset_name = "custom"

    # Export to ONNX format
    onnx_filename = f"mgc_model_{dataset_name}.onnx"
    onnx_path = os.path.join(PROJECT_ROOT, "models", onnx_filename)
    print(f"[-] Exporting model to ONNX format (target: {onnx_filename})...")

    # Ensure the target models/ directory exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"[+] Model exported successfully to {onnx_path}")

    # Re-embed weights to make it a single, self-contained file for ONNX Runtime Web
    print(f"[-] Packaging model weights into a single file...")
    try:
        import onnx
        model_proto = onnx.load(onnx_path)
        # Save without external data to bundle everything
        onnx.save(model_proto, onnx_path)
        
        # Clean up the external data file if it exists
        data_path = onnx_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
        print(f"[+] Packaged successfully. Removed external data file: {data_path}")
    except Exception as e:
        print(f"[!] Warning: Could not package model into a single file: {str(e)}")
    
    return True

def main():
    # Reconfigure console streams to UTF-8 to support emojis printed by torch.onnx on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()

    # Determine models to export
    if args.model_path is not None:
        export_model(args.model_path)
    else:
        # Auto-detect default checkpoints in models/
        jamendo_path = os.path.join(PROJECT_ROOT, "models", "best_genre_model_jamendo.pth")
        gtzan_path = os.path.join(PROJECT_ROOT, "models", "best_genre_model_gtzan.pth")

        exported_any = False
        if os.path.exists(jamendo_path):
            if export_model(jamendo_path):
                exported_any = True
        
        if os.path.exists(gtzan_path):
            if export_model(gtzan_path):
                exported_any = True
                
        if not exported_any:
            print(f"[!] Error: No checkpoints found in models/ directory.")
            print(f"    Looked for: {jamendo_path} and {gtzan_path}")
            sys.exit(1)

if __name__ == "__main__":
    main()
