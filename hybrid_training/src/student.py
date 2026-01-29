import torch
import torch.nn as nn
import torchvision.models as models

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        # Using a smaller model (ResNet18, same as teacher for simplicity, but initialized randomly)
        # In a real scenario, this might be a smaller model like MobileNet or ResNet18-Student
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_student_device():
    if torch.cuda.is_available():
        return "cuda"
    # Fallback to MPS for Mac (though user is Windows) or CPU
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
