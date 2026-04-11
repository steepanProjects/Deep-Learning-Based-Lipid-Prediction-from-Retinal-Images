"""Evaluation script for the trained model."""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import config
from model import create_model
from dataset import create_dataloaders
from utils import calculate_metrics, plot_predictions, print_metrics_table

LIPID_NAMES = ['Total Cholesterol', 'LDL', 'HDL', 'Triglycerides']

def evaluate_model():
    """Evaluate the trained model on test set."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    # Create directories
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders()
    
    # Load model
    print("Loading trained model...")
    model = create_model()
    checkpoint = torch.load(config.MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {config.MODEL_PATH}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(config.DEVICE)
            
            if config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate results
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal test samples: {len(y_true)}")
    
    # Calculate metrics for each lipid type
    metrics_dict = {}
    for i, lipid_name in enumerate(LIPID_NAMES):
        metrics = calculate_metrics(y_true[:, i], y_pred[:, i])
        metrics_dict[lipid_name] = metrics
    
    # Print metrics table
    print_metrics_table(metrics_dict, LIPID_NAMES)
    
    # Plot predictions
    plot_predictions(y_true, y_pred, LIPID_NAMES, 
                    f'{config.PLOTS_DIR}/predictions.png')
    
    # Show sample predictions
    print("\nSample Predictions (first 5 test samples):")
    print("="*70)
    for i in range(min(5, len(y_true))):
        print(f"\nSample {i+1}:")
        for j, lipid_name in enumerate(LIPID_NAMES):
            print(f"  {lipid_name:<20} Actual: {y_true[i, j]:>6.1f}  Predicted: {y_pred[i, j]:>6.1f}  Error: {abs(y_true[i, j] - y_pred[i, j]):>5.1f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    
    return metrics_dict, y_true, y_pred

if __name__ == '__main__':
    evaluate_model()
