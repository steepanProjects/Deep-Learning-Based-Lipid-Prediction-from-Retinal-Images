"""Visualize sample synthetic retinal images."""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from data_generator import create_retinal_image, generate_lipid_values

def visualize_samples():
    """Generate and display sample images with different lipid profiles."""
    
    # Create sample scenarios
    scenarios = [
        {
            'name': 'Healthy Profile',
            'values': {
                'total_cholesterol': 180.0,
                'ldl': 100.0,
                'hdl': 60.0,
                'triglycerides': 100.0
            }
        },
        {
            'name': 'Borderline High',
            'values': {
                'total_cholesterol': 220.0,
                'ldl': 140.0,
                'hdl': 45.0,
                'triglycerides': 175.0
            }
        },
        {
            'name': 'High Risk',
            'values': {
                'total_cholesterol': 260.0,
                'ldl': 180.0,
                'hdl': 35.0,
                'triglycerides': 225.0
            }
        },
        {
            'name': 'Very High Risk',
            'values': {
                'total_cholesterol': 290.0,
                'ldl': 195.0,
                'hdl': 32.0,
                'triglycerides': 245.0
            }
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        # Generate image
        img = create_retinal_image(seed=1000 + idx, lipid_values=scenario['values'])
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        
        # Title with lipid values
        title = f"{scenario['name']}\n"
        title += f"Total Chol: {scenario['values']['total_cholesterol']:.0f} | "
        title += f"LDL: {scenario['values']['ldl']:.0f}\n"
        title += f"HDL: {scenario['values']['hdl']:.0f} | "
        title += f"Trig: {scenario['values']['triglycerides']:.0f}"
        
        axes[idx].set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    plt.suptitle('Realistic Synthetic Retinal Fundus Images\nwith Different Lipid Profiles', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/sample_retinal_images.png', dpi=150, bbox_inches='tight')
    print("Sample images saved to: plots/sample_retinal_images.png")
    plt.show()

def show_features():
    """Show detailed features of a single image."""
    print("\nGenerating detailed sample image...")
    
    lipid_values = {
        'total_cholesterol': 240.0,
        'ldl': 160.0,
        'hdl': 40.0,
        'triglycerides': 200.0
    }
    
    img = create_retinal_image(seed=5000, lipid_values=lipid_values)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb)
    ax.axis('off')
    
    title = "Realistic Retinal Fundus Image Features:\n"
    title += "• Optic disc with central cup\n"
    title += "• Realistic blood vessel network (arteries & veins)\n"
    title += "• Macula and fovea\n"
    title += "• Hard exudates (lipid deposits)\n"
    title += "• Microaneurysms\n"
    title += "• Natural vignetting and lighting\n\n"
    title += f"Lipid Profile: TC={lipid_values['total_cholesterol']}, "
    title += f"LDL={lipid_values['ldl']}, HDL={lipid_values['hdl']}, "
    title += f"Trig={lipid_values['triglycerides']}"
    
    plt.title(title, fontsize=11, loc='left', pad=15)
    plt.tight_layout()
    plt.savefig('plots/detailed_retinal_image.png', dpi=150, bbox_inches='tight')
    print("Detailed image saved to: plots/detailed_retinal_image.png")
    plt.show()

if __name__ == '__main__':
    print("="*70)
    print("VISUALIZING REALISTIC SYNTHETIC RETINAL IMAGES")
    print("="*70)
    
    visualize_samples()
    show_features()
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
