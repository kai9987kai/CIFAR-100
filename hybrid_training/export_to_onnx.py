import torch
import torch.onnx
import onnx
from src.student import StudentModel
import warnings
warnings.filterwarnings('ignore')

def export_student_to_onnx():
    """
    Export the trained student model to ONNX format for browser deployment.
    """
    print("Loading student model...")
    
    # Load model
    model = StudentModel(num_classes=10)
    
    # Load weights if available
    try:
        model.load_state_dict(torch.load("student_final.pth", map_location="cpu"))
        print("✓ Loaded trained weights from student_final.pth")
    except Exception as e:
        print(f"⚠ Could not load weights ({e}). Exporting untrained model for demo purposes.")
    
    model.eval()
    
    # Create dummy input (CIFAR-100 resized to 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    output_path = "demo/cifar100_model.onnx"
    print(f"\nExporting to {output_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # Increased opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"✓ Model exported successfully to {output_path}")
    
    # Merge external data into a single file for browser compatibility
    try:
        print("\nMerging external data into a single file...")
        model_onnx = onnx.load(output_path, load_external_data=True)
        single_file_path = "demo/cifar100_model.onnx" # Overwrite with single file
        
        # We need to temporarily move the split files to avoid conflicts if we save to same name
        # Actually, saving to a different name first is safer
        tmp_path = "demo/cifar100_model_full.onnx"
        onnx.save_model(model_onnx, tmp_path, save_as_external_data=False)
        
        # Cleanup and rename
        import os
        if os.path.exists(output_path): os.remove(output_path)
        if os.path.exists(output_path + ".data"): os.remove(output_path + ".data")
        os.rename(tmp_path, output_path)
        
        print(f"✓ Success! Single-file model created at {output_path}")
        print(f"  New file size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"⚠ Could not merge external data: {e}")
        print("  The model might still work if hosted with the .data file.")

    print(f"\nSummary:")
    print(f"  Input shape: [batch, 3, 224, 224]")
    print(f"  Output shape: [batch, 10]")
    
if __name__ == "__main__":
    export_student_to_onnx()
