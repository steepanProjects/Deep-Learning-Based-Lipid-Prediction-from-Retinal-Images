"""Comprehensive visualization of model results."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import cv2
from tqdm import tqdm
from torch.cuda.amp import autocast
import config
from model import create_model
from dataset import create_dataloaders
from utils import calculate_metrics
import pandas as pd

LIPID_NAMES = ['Total Cholesterol', 'LDL', 'HDL', 'Triglycerides']
LIPID_UNITS = 'mg/dL'

def load_model_and_data():
    """Load trained model and test data."""
    print("Loading model and data...")
    
    # Load model
    model = create_model()
    checkpoint = torch.load(config.MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    _, _, test_loader = create_dataloaders()
    
    # Get predictions
    all_predictions = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Getting predictions'):
            images_gpu = images.to(config.DEVICE)
            
            if config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images_gpu)
            else:
                outputs = model(images_gpu)
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_images.append(images.cpu().numpy())
    
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    images_array = np.concatenate(all_images, axis=0)
    
    return y_true, y_pred, images_array

def plot_comprehensive_results(y_true, y_pred, images_array):
    """Create comprehensive visualization dashboard."""
    print("\nCreating comprehensive visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. Scatter plots for each lipid type (2x2 grid)
    for i, lipid_name in enumerate(LIPID_NAMES):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Scatter with color gradient
        scatter = ax.scatter(true_vals, pred_vals, 
                           c=np.abs(true_vals - pred_vals),
                           cmap='RdYlGn_r', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate metrics
        metrics = calculate_metrics(true_vals, pred_vals)
        
        ax.set_xlabel(f'Actual {lipid_name} ({LIPID_UNITS})', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Predicted {lipid_name} ({LIPID_UNITS})', fontsize=10, fontweight='bold')
        ax.set_title(f'{lipid_name}\nMAE: {metrics["mae"]:.2f} | RMSE: {metrics["rmse"]:.2f} | R²: {metrics["r2"]:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prediction Error', fontsize=8)
    
    # 2. Error distribution histograms (row 3)
    for i, lipid_name in enumerate(LIPID_NAMES):
        ax = fig.add_subplot(gs[2, i])
        
        errors = y_pred[:, i] - y_true[:, i]
        
        ax.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(np.mean(errors), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        
        ax.set_xlabel(f'Prediction Error ({LIPID_UNITS})', fontsize=9, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax.set_title(f'{lipid_name} Error Distribution\nStd: {np.std(errors):.2f}', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Sample predictions with images (row 4)
    sample_indices = np.random.choice(len(y_true), 4, replace=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[3, idx])
        
        # Denormalize and display image
        img = images_array[sample_idx].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Add text with predictions
        text = f"Sample {sample_idx}\n"
        for i, lipid_name in enumerate(LIPID_NAMES):
            actual = y_true[sample_idx, i]
            predicted = y_pred[sample_idx, i]
            error = abs(actual - predicted)
            text += f"{lipid_name[:2]}: {actual:.0f}→{predicted:.0f} (±{error:.0f})\n"
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Comprehensive Model Evaluation Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f'{config.PLOTS_DIR}/comprehensive_results.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/comprehensive_results.png")
    plt.close()

def plot_correlation_matrix(y_true, y_pred):
    """Plot correlation matrix between actual and predicted values."""
    print("Creating correlation matrix...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Actual correlations
    corr_actual = np.corrcoef(y_true.T)
    sns.heatmap(corr_actual, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=LIPID_NAMES, yticklabels=LIPID_NAMES,
                vmin=-1, vmax=1, center=0, ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Actual Lipid Correlations', fontsize=12, fontweight='bold')
    
    # Predicted correlations
    corr_pred = np.corrcoef(y_pred.T)
    sns.heatmap(corr_pred, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=LIPID_NAMES, yticklabels=LIPID_NAMES,
                vmin=-1, vmax=1, center=0, ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Predicted Lipid Correlations', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/correlation_matrix.png")
    plt.close()

def plot_error_analysis(y_true, y_pred):
    """Detailed error analysis plots."""
    print("Creating error analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error vs Actual Value
    ax = axes[0, 0]
    for i, lipid_name in enumerate(LIPID_NAMES):
        errors = y_pred[:, i] - y_true[:, i]
        ax.scatter(y_true[:, i], errors, alpha=0.5, s=20, label=lipid_name)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual Value (mg/dL)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Error (mg/dL)', fontsize=11, fontweight='bold')
    ax.set_title('Error vs Actual Value', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Absolute Error by Lipid Type
    ax = axes[0, 1]
    abs_errors = [np.abs(y_pred[:, i] - y_true[:, i]) for i in range(4)]
    bp = ax.boxplot(abs_errors, labels=LIPID_NAMES, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']):
        patch.set_facecolor(color)
    ax.set_ylabel('Absolute Error (mg/dL)', fontsize=11, fontweight='bold')
    ax.set_title('Absolute Error Distribution by Lipid Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Percentage Error
    ax = axes[1, 0]
    for i, lipid_name in enumerate(LIPID_NAMES):
        pct_errors = 100 * np.abs(y_pred[:, i] - y_true[:, i]) / y_true[:, i]
        ax.hist(pct_errors, bins=30, alpha=0.6, label=lipid_name)
    ax.set_xlabel('Absolute Percentage Error (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Error Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_data = []
    for i, lipid_name in enumerate(LIPID_NAMES):
        errors = y_pred[:, i] - y_true[:, i]
        abs_errors = np.abs(errors)
        pct_errors = 100 * abs_errors / y_true[:, i]
        
        stats_data.append([
            lipid_name,
            f"{np.mean(abs_errors):.2f}",
            f"{np.median(abs_errors):.2f}",
            f"{np.std(errors):.2f}",
            f"{np.mean(pct_errors):.1f}%"
        ])
    
    table = ax.table(cellText=stats_data,
                    colLabels=['Lipid', 'Mean AE', 'Median AE', 'Std Dev', 'Mean %E'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, 5):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Error Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/error_analysis.png")
    plt.close()

def plot_best_worst_predictions(y_true, y_pred, images_array):
    """Show best and worst predictions with images."""
    print("Creating best/worst predictions visualization...")
    
    # Calculate overall error for each sample
    overall_errors = np.mean(np.abs(y_pred - y_true), axis=1)
    
    # Get best and worst
    best_indices = np.argsort(overall_errors)[:6]
    worst_indices = np.argsort(overall_errors)[-6:]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    
    # Best predictions
    for idx, sample_idx in enumerate(best_indices):
        ax = axes[0, idx]
        
        # Display image
        img = images_array[sample_idx].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Add border (green for best)
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
            spine.set_visible(True)
        
        # Add text
        avg_error = overall_errors[sample_idx]
        text = f"Avg Error: {avg_error:.1f}\n"
        for i in range(4):
            text += f"{LIPID_NAMES[i][:2]}: {y_true[sample_idx, i]:.0f}→{y_pred[sample_idx, i]:.0f}\n"
        
        ax.text(0.5, -0.05, text, transform=ax.transAxes,
               fontsize=7, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Worst predictions
    for idx, sample_idx in enumerate(worst_indices):
        ax = axes[1, idx]
        
        # Display image
        img = images_array[sample_idx].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Add border (red for worst)
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
            spine.set_visible(True)
        
        # Add text
        avg_error = overall_errors[sample_idx]
        text = f"Avg Error: {avg_error:.1f}\n"
        for i in range(4):
            text += f"{LIPID_NAMES[i][:2]}: {y_true[sample_idx, i]:.0f}→{y_pred[sample_idx, i]:.0f}\n"
        
        ax.text(0.5, -0.05, text, transform=ax.transAxes,
               fontsize=7, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    axes[0, 0].text(-0.3, 0.5, 'BEST\nPREDICTIONS', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                   color='green')
    axes[1, 0].text(-0.3, 0.5, 'WORST\nPREDICTIONS', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                   color='red')
    
    plt.suptitle('Best vs Worst Predictions (TC→LDL→HDL→Trig)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/best_worst_predictions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/best_worst_predictions.png")
    plt.close()

def plot_lipid_ranges(y_true, y_pred):
    """Show performance across different lipid value ranges."""
    print("Creating lipid range analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, lipid_name in enumerate(LIPID_NAMES):
        ax = axes[i]
        
        # Define ranges
        if lipid_name == 'Total Cholesterol':
            ranges = [(120, 200, 'Desirable'), (200, 240, 'Borderline'), (240, 300, 'High')]
        elif lipid_name == 'LDL':
            ranges = [(50, 100, 'Optimal'), (100, 130, 'Near Optimal'), (130, 200, 'High')]
        elif lipid_name == 'HDL':
            ranges = [(30, 40, 'Low'), (40, 60, 'Normal'), (60, 90, 'High')]
        else:  # Triglycerides
            ranges = [(50, 150, 'Normal'), (150, 200, 'Borderline'), (200, 250, 'High')]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for low, high, name in ranges:
            mask = (y_true[:, i] >= low) & (y_true[:, i] < high)
            if np.sum(mask) > 0:
                mae = np.mean(np.abs(y_pred[mask, i] - y_true[mask, i]))
                range_names.append(name)
                range_maes.append(mae)
                range_counts.append(np.sum(mask))
        
        # Bar plot
        bars = ax.bar(range_names, range_maes, color=['green', 'yellow', 'red'][:len(range_names)], 
                     alpha=0.7, edgecolor='black')
        
        # Add count labels
        for bar, count in zip(bars, range_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Mean Absolute Error (mg/dL)', fontsize=10, fontweight='bold')
        ax.set_title(f'{lipid_name} - Performance by Clinical Range', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/lipid_range_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/lipid_range_analysis.png")
    plt.close()

def visualize_all_results():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Load data
    y_true, y_pred, images_array = load_model_and_data()
    
    # Generate all plots
    plot_comprehensive_results(y_true, y_pred, images_array)
    plot_correlation_matrix(y_true, y_pred)
    plot_error_analysis(y_true, y_pred)
    plot_best_worst_predictions(y_true, y_pred, images_array)
    plot_lipid_ranges(y_true, y_pred)
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print(f"\nGenerated plots in '{config.PLOTS_DIR}/':")
    print("  1. comprehensive_results.png - Main dashboard")
    print("  2. correlation_matrix.png - Lipid correlations")
    print("  3. error_analysis.png - Detailed error analysis")
    print("  4. best_worst_predictions.png - Best/worst cases")
    print("  5. lipid_range_analysis.png - Performance by clinical ranges")
    print("="*70 + "\n")

if __name__ == '__main__':
    visualize_all_results()
