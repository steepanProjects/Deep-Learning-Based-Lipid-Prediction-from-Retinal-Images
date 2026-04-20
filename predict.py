"""Predict lipid values from a single retinal image."""

import torch
from PIL import Image
from torchvision import transforms
import argparse
import sys
import config
from model import LipidPredictionModel


def load_model(model_path='model.pth'):
    """Load the trained model."""
    model = LipidPredictionModel(
        model_name=config.MODEL_NAME,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    # Handle both checkpoint dict and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(config.DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess an image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(model, image_tensor):
    """Run inference on the image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(config.DEVICE)
        predictions = model(image_tensor)
    return predictions.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='Predict lipid values from retinal image')
    parser.add_argument('image_path', type=str, help='Path to the retinal image')
    parser.add_argument('--model', type=str, default='model.pth', 
                       help='Path to the trained model (default: model.pth)')
    args = parser.parse_args()
    
    # Check if image exists
    try:
        image_tensor = preprocess_image(args.image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(args.model)
        print(f"Model loaded from: {args.model}")
        print(f"Using device: {config.DEVICE}\n")
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Predict
    predictions = predict(model, image_tensor)
    
    # Display results
    print("=" * 50)
    print("LIPID PREDICTION RESULTS")
    print("=" * 50)
    print(f"Total Cholesterol: {predictions[0]:.2f} mg/dL")
    print(f"LDL Cholesterol:   {predictions[1]:.2f} mg/dL")
    print(f"HDL Cholesterol:   {predictions[2]:.2f} mg/dL")
    print(f"Triglycerides:     {predictions[3]:.2f} mg/dL")
    print("=" * 50)
    
    # Health interpretation
    print("\nHealth Interpretation:")
    if predictions[0] < 200:
        print("✓ Total Cholesterol: Desirable")
    elif predictions[0] < 240:
        print("⚠ Total Cholesterol: Borderline high")
    else:
        print("✗ Total Cholesterol: High")
    
    if predictions[1] < 100:
        print("✓ LDL: Optimal")
    elif predictions[1] < 130:
        print("✓ LDL: Near optimal")
    elif predictions[1] < 160:
        print("⚠ LDL: Borderline high")
    else:
        print("✗ LDL: High")
    
    if predictions[2] >= 60:
        print("✓ HDL: Good (protective)")
    elif predictions[2] >= 40:
        print("⚠ HDL: Acceptable")
    else:
        print("✗ HDL: Low (risk factor)")
    
    if predictions[3] < 150:
        print("✓ Triglycerides: Normal")
    elif predictions[3] < 200:
        print("⚠ Triglycerides: Borderline high")
    else:
        print("✗ Triglycerides: High")


if __name__ == '__main__':
    main()
