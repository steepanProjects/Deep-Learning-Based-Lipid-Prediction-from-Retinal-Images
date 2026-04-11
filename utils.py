"""Utility functions for training and evaluation."""

import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def save_training_history(history, filepath):
    """Save training history to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved: {filepath}")

def plot_training_history(history, save_path):
    """Plot training and validation loss."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved: {save_path}")

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def plot_predictions(y_true, y_pred, lipid_names, save_path):
    """Plot predicted vs actual values for each lipid type."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, lipid_name in enumerate(lipid_names):
        ax = axes[i]
        
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate metrics
        metrics = calculate_metrics(true_vals, pred_vals)
        
        ax.set_xlabel('Actual Values (mg/dL)', fontsize=11)
        ax.set_ylabel('Predicted Values (mg/dL)', fontsize=11)
        ax.set_title(f'{lipid_name}\nMAE: {metrics["mae"]:.2f}, R²: {metrics["r2"]:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Predictions plot saved: {save_path}")

def print_metrics_table(metrics_dict, lipid_names):
    """Print metrics in a formatted table."""
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    print(f"{'Lipid Type':<25} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-"*70)
    
    for lipid_name in lipid_names:
        metrics = metrics_dict[lipid_name]
        print(f"{lipid_name:<25} {metrics['mae']:<12.2f} {metrics['rmse']:<12.2f} {metrics['r2']:<12.3f}")
    
    print("="*70)

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
