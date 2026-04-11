"""Configuration file for the deep learning pipeline."""

import torch

# System Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = True
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on Windows
PIN_MEMORY = True

# Dataset Configuration
IMAGE_SIZE = 224
DATASET_SIZE = 15000
TRAIN_SIZE = 12000
VAL_SIZE = 2000
TEST_SIZE = 1000

# Data paths
DATA_DIR = 'data'
IMAGES_DIR = f'{DATA_DIR}/images'
LABELS_FILE = f'{DATA_DIR}/labels.csv'
MODEL_PATH = 'model.pth'
PLOTS_DIR = 'plots'
LOGS_DIR = 'logs'

# Lipid value ranges (mg/dL)
LIPID_RANGES = {
    'total_cholesterol': (120, 300),
    'ldl': (50, 200),
    'hdl': (30, 90),
    'triglycerides': (50, 250)
}

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
DROPOUT_RATE = 0.3

# Model Configuration
MODEL_NAME = 'efficientnet_b0'  # or 'resnet18'
PRETRAINED = True
NUM_OUTPUTS = 4  # 4 lipid values
