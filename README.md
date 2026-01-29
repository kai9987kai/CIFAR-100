# CIFAR-100 Browser Demo


## Features

- 🌐 **Runs entirely in the browser** - No server required!
- 📦 **ONNX.js** - Uses ONNX Runtime for JavaScript
- 🎨 **Modern UI** - Beautiful, responsive interface
- 🚀 **100 Classes** - Trained using hybrid NPU+GPU+CPU training

## Files

- `index.html` - Standalone demo page (open directly in browser)
- `cifar100_model.pt` - TorchScript model (needs conversion to ONNX)

## How to Use

### Option 1: Quick Test (Without Trained Model)

1. Open `index.html` in your web browser
2. The page will show a note that the model needs to be exported
3. You can still see the beautiful UI!

### Option 2: With ONNX Model (Recommended)

**Step 1: Export the model to ONNX**

From the parent directory (`hybrid_training/`), run:

```bash
# First, install onnxscript if not already installed
pip install git+https://github.com/microsoft/onnxscript.git

# Then export the model
python export_to_onnx.py
```

This will create `demo/cifar100_model.onnx`.

**Step 2: Open the demo**

Simply open `demo/index.html` in any modern web browser (Chrome, Firefox, Edge, Safari).

**Step 3: Classify images**

1. Click or drag-and-drop an image onto the upload area
2. Click "Classify Image"
3. See the predicted class and confidence score!

## CIFAR-100 Classes

The model can recognize 100 different categories including:
- Animals: apple, bear, beaver, bee, camel, cattle, etc.
- Vehicles: bicycle, bus, motorcycle, pickup_truck, train, etc.
- Objects: bottle, chair, clock, lamp, table, etc.
- Nature: cloud, forest, mountain, sea, etc.

## Technical Details

- **Model**: ResNet18 (Hybrid-trained on NPU+GPU+CPU)
- **Input**: 224x224 RGB images
- **Output**: 100 class probabilities
- **Framework**: ONNX.js (v1.14+)
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Troubleshooting

**Q: "Model not found" error?**
- Make sure you've run `python export_to_onnx.py` from the parent directory
- Check that `cifar100_model.onnx` exists in the `demo/` folder

**Q: Not loading in browser?**
- Some browsers block local file access. Use a local web server:
  ```bash
  cd demo
  python -m http.server 8000
  ```
  Then open `http://localhost:8000`

**Q: Classifications seem random?**
- The model may not have been fully trained yet
- CIFAR-100 is a challenging dataset with 100 fine-grained classes
- Try images that clearly represent one of the CIFAR-100 categories

## Credits

Built using:
- PyTorch
- ONNX
- ONNX Runtime Web
- Hybrid training across NPU/GPU/CPU
