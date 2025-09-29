#!/bin/bash

set -e  

echo "Building GPU-Accelerated BEV Transformation System"
echo "=================================================="

if ! command -v nvcc &> /dev/null; then
    echo "ERROR CUDA compiler (nvcc) not found. Install CUDA toolkit."
    exit 1
fi

echo " CUDA compiler found: $(nvcc --version | head -n1)"

if ! python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Please install PyTorch first."
    exit 1
fi

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" 2>/dev/null; then
    echo "ERROR: PyTorch CUDA support not available. Please install PyTorch with CUDA support."
    exit 1
fi

echo "PyTorch with CUDA support found"

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Building CUDA extension..."
python3 setup.py build_ext --inplace

echo "Testing installation..."
python3 -c "
try:
    import bev_cuda
    print('CUDA extension imported successfully')
except ImportError as e:
    print(f'Failed to import CUDA extension: {e}')
    exit(1)
"


echo "Running basic tests..."
python3 -c "
import torch
import bev_cuda
import numpy as np

# Test LiDAR voxelization
points = torch.randn(10000, 4).cuda()
result = bev_cuda.lidar_voxelization(points, 0.1, 50.0, True, True)
print(f'LiDAR voxelization test passed. Output shape: {result.shape}')

# Test camera projection
image = torch.randint(0, 255, (480, 640, 3)).cuda()
intrinsics = torch.tensor([[721.5, 0, 320.0], [0, 721.5, 240.0], [0, 0, 1.0]]).cuda()
extrinsics = torch.eye(4).cuda()
result = bev_cuda.camera_to_bev_projection(image, intrinsics, extrinsics, 0.1, 50.0, 1)
print(f'Camera projection test passed. Output shape: {result.shape}')

print('All tests passed!')
"


# Run benchmark
echo "Running performance benchmark..."
python3 -c "
from benchmarks.benchmark import run_benchmark
results = run_benchmark('benchmark_results')
print('Benchmark completed')
"
