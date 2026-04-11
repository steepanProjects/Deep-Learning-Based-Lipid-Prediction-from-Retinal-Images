"""Main pipeline script - runs the complete deep learning pipeline."""

import os
import sys
import torch
import config

def check_system():
    """Check system configuration."""
    print("="*70)
    print("SYSTEM CONFIGURATION")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"Device: {config.DEVICE}")
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    print("="*70 + "\n")

def run_pipeline():
    """Run the complete pipeline."""
    print("\n" + "="*70)
    print("DEEP LEARNING PIPELINE FOR LIPID PREDICTION")
    print("="*70 + "\n")
    
    # Check system
    check_system()
    
    # Step 1: Generate synthetic dataset
    print("STEP 1: Generating Synthetic Dataset")
    print("-"*70)
    from data_generator import generate_dataset
    
    if not os.path.exists(config.LABELS_FILE):
        generate_dataset()
    else:
        print(f"Dataset already exists at {config.LABELS_FILE}")
        print("Skipping generation. Delete the file to regenerate.\n")
    
    # Step 2: Train model
    print("\nSTEP 2: Training Model")
    print("-"*70)
    from train import train_model
    
    if not os.path.exists(config.MODEL_PATH):
        train_model()
    else:
        print(f"Model already exists at {config.MODEL_PATH}")
        response = input("Retrain? (y/n): ")
        if response.lower() == 'y':
            train_model()
        else:
            print("Skipping training.\n")
    
    # Step 3: Evaluate model
    print("\nSTEP 3: Evaluating Model")
    print("-"*70)
    from evaluate import evaluate_model
    evaluate_model()
    
    # Step 4: Generate comprehensive visualizations
    print("\nSTEP 4: Generating Comprehensive Visualizations")
    print("-"*70)
    from visualize_results import visualize_all_results
    visualize_all_results()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Model: {config.MODEL_PATH}")
    print(f"  - Training plots: {config.PLOTS_DIR}/training_history.png")
    print(f"  - Prediction plots: {config.PLOTS_DIR}/predictions.png")
    print(f"  - Comprehensive dashboard: {config.PLOTS_DIR}/comprehensive_results.png")
    print(f"  - Error analysis: {config.PLOTS_DIR}/error_analysis.png")
    print(f"  - Best/worst cases: {config.PLOTS_DIR}/best_worst_predictions.png")
    print(f"  - Correlation matrix: {config.PLOTS_DIR}/correlation_matrix.png")
    print(f"  - Clinical range analysis: {config.PLOTS_DIR}/lipid_range_analysis.png")
    print(f"  - Training logs: {config.LOGS_DIR}/training_history.json")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
