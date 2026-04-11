"""Model architecture with transfer learning."""

import torch
import torch.nn as nn
from torchvision import models
import config

class LipidPredictionModel(nn.Module):
    """Transfer learning model for lipid prediction from retinal images."""
    
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(LipidPredictionModel, self).__init__()
        
        if model_name == 'efficientnet_b0':
            # EfficientNet-B0 (memory efficient)
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'resnet18':
            # ResNet18 (lightweight fallback)
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_OUTPUTS)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

def create_model():
    """Create and initialize the model."""
    model = LipidPredictionModel(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED
    )
    model = model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {config.MODEL_NAME}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {config.DEVICE}")
    
    return model

if __name__ == '__main__':
    model = create_model()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    output = model(dummy_input)
    print(f"\nTest output shape: {output.shape}")
    print(f"Sample output: {output[0]}")
