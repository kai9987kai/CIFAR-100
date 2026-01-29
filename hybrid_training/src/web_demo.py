import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms
from student import StudentModel, get_student_device
import io
import numpy as np

app = Flask(__name__)

# Load Student Model
print("Loading student model...")
device_student = get_student_device()
student = StudentModel(num_classes=10)
# Load trained weights if available
student_weights = "student_final.pth"
if os.path.exists(student_weights):
    try:
        student.load_state_dict(torch.load(student_weights, map_location=device_student))
        print("Loaded student weights.")
    except RuntimeError as e:
        print(f"WARNING: Could not load weights ({e}). Using random weights.")
else:
    print("WARNING: student_final.pth not found. Using random weights.")
student.to(device_student)
student.eval()

# CIFAR-100 class names (simplified)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        
        # Student Inference
        tensor_student = tensor.to(device_student)
        with torch.no_grad():
            outputs_student = student(tensor_student)
            probabilities = torch.nn.functional.softmax(outputs_student, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_idx = int(predicted.item())
        predicted_class = CIFAR10_CLASSES[predicted_idx] if predicted_idx < len(CIFAR10_CLASSES) else f"Class_{predicted_idx}"
        
        return jsonify({
            'class_id': predicted_idx,
            'class_name': predicted_class,
            'confidence': float(confidence.item() * 100),
            'message': 'Success'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
