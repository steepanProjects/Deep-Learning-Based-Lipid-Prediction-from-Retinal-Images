"""PyTorch Dataset and DataLoader setup."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

class RetinalLipidDataset(Dataset):
    """Custom dataset for retinal images and lipid values."""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load image
        img_path = os.path.join(config.IMAGES_DIR, row['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get lipid values
        labels = torch.tensor([
            row['total_cholesterol'],
            row['ldl'],
            row['hdl'],
            row['triglycerides']
        ], dtype=torch.float32)
        
        return image, labels

def get_transforms(train=True):
    """Get image transforms for training or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders():
    """Create train, validation, and test dataloaders."""
    # Load labels
    df = pd.read_csv(config.LABELS_FILE)
    
    # Split dataset
    train_df = df.iloc[:config.TRAIN_SIZE].reset_index(drop=True)
    val_df = df.iloc[config.TRAIN_SIZE:config.TRAIN_SIZE + config.VAL_SIZE].reset_index(drop=True)
    test_df = df.iloc[config.TRAIN_SIZE + config.VAL_SIZE:].reset_index(drop=True)
    
    # Create datasets
    train_dataset = RetinalLipidDataset(train_df, transform=get_transforms(train=True))
    val_dataset = RetinalLipidDataset(val_df, transform=get_transforms(train=False))
    test_dataset = RetinalLipidDataset(test_df, transform=get_transforms(train=False))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[0]}")
