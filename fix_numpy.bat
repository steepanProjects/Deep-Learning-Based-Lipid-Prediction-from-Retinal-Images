@echo off
echo Fixing NumPy compatibility issue...
echo.

echo Uninstalling NumPy 2.x...
pip uninstall -y numpy

echo.
echo Installing NumPy 1.26.4 (compatible with PyTorch)...
pip install "numpy<2.0" "numpy>=1.24.0"

echo.
echo Verifying installation...
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo Fix complete!
pause
