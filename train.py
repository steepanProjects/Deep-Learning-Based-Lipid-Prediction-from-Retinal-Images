"""Training script for the lipid prediction model."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import config
from model import create_model
from dataset import create_dataloaders
from utils import (save_checkpoint, save_training_history, 
                   plot_training_history, AverageMeter)

def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
    
    for images, labels in pbar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    return losses.avg

def validate(model, dataloader, criterion, epoch):
    """Validate the model."""
    model.eval()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            if config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    return losses.avg

def train_model():
    """Main training function."""
    print("\n" + "="*70)
    print("TRAINING LIPID PREDICTION MODEL")
    print("="*70)
    
    # Create directories
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders()
    
    # Create model
    print("\nInitializing model...")
    model = create_model()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, epoch)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config.MODEL_PATH)
            print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
        
        print("-"*70)
    
    # Save training history
    save_training_history(history, f'{config.LOGS_DIR}/training_history.json')
    plot_training_history(history, f'{config.PLOTS_DIR}/training_history.png')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.MODEL_PATH}")

if __name__ == '__main__':
    train_model()
