import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import torch.nn.functional as F

from dataset import get_dataloader
from teacher import TeacherModel
from student import StudentModel, get_student_device

def distillation_loss(student_logits, labels, teacher_logits, alpha=0.5, temperature=2.0):
    """
    Combines CrossEntropy (Student vs Labels) and KLDiv (Student vs Teacher).
    """
    # Hard loss (Student correctness)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft loss (Mimic teacher)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_log_softmax = F.log_softmax(student_logits / temperature, dim=1)
    
    # KLDivLoss expects log_probs as input, validation target (not log) 
    # and reduction usually is 'batchmean' for mathematical correctness
    distill_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    return alpha * hard_loss + (1 - alpha) * distill_loss

def main():
    print("=== Hybrid CPU/GPU/NPU Training Demo ===")
    
    # 1. Setup Student (GPU)
    device_student = get_student_device()
    print(f"[Main] Student training on: {device_student.upper()}")
    student = StudentModel(num_classes=100).to(device_student)
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    # 2. Setup Teacher (NPU)
    # Note: We rely on the TeacherModel class to find NPU or fallback
    print(f"[Main] Initializing Teacher on NPU...")
    teacher = TeacherModel()
    
    # 3. Setup Data (CPU)
    dataloader = get_dataloader(batch_size=32)
    
    print("\nStarting Training Loop...")
    print("Detailed flow: CPU (Data) -> NPU (Teacher Infer) -> GPU (Student Train)")
    
    epochs = 1
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # images is CPU tensor
            
            # --- NPU STEP ---
            # Run Teacher Inference on NPU
            # Note: teacher.predict handles numpy conversion internally if needed
            # We do this BEFORE moving images to GPU to simulate NPU offloading from host memory
            teacher_logits_np = teacher.predict(images)
            teacher_logits = torch.from_numpy(teacher_logits_np).to(device_student)
            
            # --- GPU STEP ---
            # Move data to GPU for student
            images_gpu = images.to(device_student)
            labels_gpu = labels.to(device_student)
            
            optimizer.zero_grad()
            student_logits = student(images_gpu)
            
            # Calculate Hybrid Loss
            loss = distillation_loss(student_logits, labels_gpu, teacher_logits)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
                
        print(f"Epoch {epoch+1} finished. Avg Loss: {running_loss / len(dataloader)}")

    print("\n=== Training Complete ===")
    print("If you check Task Manager, you should have seen simultaneous activity.")
    # Prompt to save model
    torch.save(student.state_dict(), "student_final.pth")
    print("Saved student model to student_final.pth")

if __name__ == "__main__":
    main()
