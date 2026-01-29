import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']
        # 'label' is the 10 classes target for CIFAR-10.
        label = item['label']

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloader(batch_size=32, num_workers=2):
    """
    Creates a CPU-bound dataloader for CIFAR-100 using Hugging Face Datasets.
    """
    print(f"[CPU] Initializing Dataset (CIFAR-100 via Hugging Face) and DataLoader...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize for standard models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-100
    # This caches automatically in ~/.cache/huggingface/datasets
    print("[CPU] Downloading/Loading CIFAR-100...")
    dataset_hf = load_dataset("cifar10", split="train")
    
    # Wrap it
    dataset = HFDataset(dataset_hf, transform=transform)

    # Pin memory speeds up transfer to GPU later
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=True)
    
    print(f"[CPU] DataLoader ready. Dataset size: {len(dataset)}")
    return loader
