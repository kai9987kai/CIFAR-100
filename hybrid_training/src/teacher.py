import torch
import torchvision.models as models
import openvino as ov
import os
import numpy as np

class TeacherModel:
    def __init__(self, model_name="resnet18_cifar100", device="NPU"):
        self.model_path = f"{model_name}_teacher.onnx"
        self.device = device
        self.core = ov.Core()
        
        # Check if NPU is available, else fallback
        available_devices = self.core.available_devices
        print(f"[Teacher] Available OpenVINO devices: {available_devices}")
        if device not in available_devices:
            print(f"[Teacher] WARNING: Device '{device}' not found. Falling back to 'CPU'.")
            self.device = "CPU"
        else:
            print(f"[Teacher] using device: {self.device}")

        # Export if not exists
        if not os.path.exists(self.model_path):
            print(f"[Teacher] Exporting {model_name} to ONNX...")
            model = models.resnet18(pretrained=True)
            # Adapt to 100 classes (CIFAR-100)
            model.fc = torch.nn.Linear(model.fc.in_features, 100)
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, dummy_input, self.model_path, 
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        # Load model
        print(f"[Teacher] Loading model to {self.device}...")
        self.compiled_model = self.core.compile_model(self.model_path, self.device)
        self.infer_request = self.compiled_model.create_infer_request()
        print("[Teacher] Model loaded successfully.")

    def predict(self, images):
        """
        Runs inference on the NPU/Backend.
        Args:
            images (torch.Tensor or np.ndarray): Batch of images (B, C, H, W).
        Returns:
            np.ndarray: Logits (B, N_Classes).
        """
        # Ensure input is numpy
        if isinstance(images, torch.Tensor):
            input_data = images.cpu().numpy()
        else:
            input_data = images
            
        # Run inference
        results = self.infer_request.infer([input_data])
        # OpenVINO returns a dict or list, get the first output
        output_tensor = list(results.values())[0]
        return output_tensor
