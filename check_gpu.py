"""Quick script to check GPU availability."""

import torch

print("="*70)
print("GPU CHECK")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU memory: {total_memory:.2f} GB")
    
    # Test tensor on GPU
    try:
        x = torch.randn(100, 100).cuda()
        print(f"\n✓ Successfully created tensor on GPU!")
        print(f"Tensor device: {x.device}")
    except Exception as e:
        print(f"\n✗ Error creating tensor on GPU: {e}")
else:
    print("\n✗ CUDA is NOT available!")
    print("\nPossible reasons:")
    print("1. PyTorch CPU-only version is installed")
    print("2. NVIDIA drivers not installed")
    print("3. CUDA toolkit not installed")
    print("\nTo fix:")
    print("Run: install_pytorch_gpu.bat")

print("="*70)
