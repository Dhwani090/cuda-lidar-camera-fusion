# GPU-Accelerated Bird's-Eye-View (BEV) Transformation

A high-performance CUDA implementation for LiDAR + Camera fusion in autonomous vehicles, featuring real-time BEV transformation with speedup over CPU implementations.

## Key Features

- **LiDAR Voxelization**: GPU-accelerated point cloud voxelization with atomic operations
- **Camera-to-BEV Projection**: Real-time perspective transformation from camera frames to BEV grid
- **Feature Fusion**: Advanced fusion methods (attention, cross-modal, weighted)
- **PyTorch Integration**: Seamless CUDA extension wrapper for deep learning pipelines
- **Real-time Visualization**: Live BEV map updates with performance monitoring
- **Performance Benchmarking**: Comprehensive comparison against CPU OpenCV baseline


## Project Structure

```
CUDA_CV/
├── kernels/                    # CUDA kernel implementations
│   ├── lidar_voxelization.cu  # LiDAR point cloud voxelization
│   ├── camera_bev_projection.cu # Camera-to-BEV projection
│   └── feature_fusion.cu     # Multi-modal feature fusion
├── python/                    # Python bindings and PyTorch integration
│   ├── bev_cuda_extension.cpp # PyTorch CUDA extension
│   └── bev_api.py            # High-level Python API
├── benchmarks/               # Performance testing and comparison
│   └── benchmark.py          # Comprehensive benchmarking suite
├── examples/                 # Usage examples and demos
│   └── kitti_example.py     # KITTI dataset integration
├── setup.py                  # Build configuration
├── build.sh                  # Automated build script
└── demo.py                   # Interactive demo
```

## Core Components

### 1. LiDAR Voxelization Kernel
- **Input**: N×4 point cloud (x, y, z, intensity)
- **Output**: 200×200×32 occupancy grid
- **Features**: Atomic operations, height filtering, intensity max-pooling

### 2. Camera Projection Kernel
- **Input**: H×W×3 RGB image + calibration matrices
- **Output**: 200×200×3 color grid + depth map
- **Features**: Bilinear interpolation, multi-camera fusion

### 3. Feature Fusion
- **Methods**: Simple concatenation, weighted fusion, attention-based, cross-modal
- **Output**: Fused BEV feature tensor
- **Features**: Dynamic weighting, spatial enhancement

### 4. Performance Monitoring
- **Real-time**: Performance monitoring and metrics

## Quick Start

### Prerequisites
- **CUDA 11.0+ with compatible GPU (REQUIRED - no CPU fallback)**
- PyTorch with CUDA support
- Python 3.7+

### Installation
```bash
git clone <repository-url>
cd CUDA_CV

./build.sh
```

### Basic Usage
```python
import torch
from python.bev_api import BEVTransformer, FusionMethod

transformer = BEVTransformer(device='cuda')

bev_features = transformer.transform(
    lidar_points, camera_image, camera_intrinsics, camera_extrinsics,
    fusion_method=FusionMethod.ATTENTION_FUSION
)

print(f"BEV features shape: {bev_features.shape}")
```

## Dataset Support

### KITTI Dataset
```python
from examples.kitti_example import KITTIDataset
dataset = KITTIDataset('/path/to/kitti', sequence='00')

for i in range(len(dataset)):
    frame_data = dataset[i]
```

## Benchmarking

### Run Performance Benchmark
```bash
python -c "from benchmarks.benchmark import run_benchmark; run_benchmark()"
```

### Custom Benchmarking
```python
from benchmarks.benchmark import BEVBenchmark

benchmark = BEVBenchmark(device='cuda', num_iterations=100)
results = benchmark.run_comprehensive_benchmark()
benchmark.generate_report(results, 'my_results')
```

## Use Cases

- **Autonomous Vehicles**: Real-time perception and mapping
- **Robotics**: Navigation and scene understanding
- **Sensor Fusion**: Multi-modal data integration
- **Research**: 3D scene analysis and reconstruction

## Extensions & Future Work

- **Dynamic Mapping**: Temporal fusion and occupancy grid updates
- **Semantic Fusion**: CNN feature projection to BEV space
- **Path Planning**: GPU-accelerated A* on BEV occupancy grids
- **Multi-Sensor**: LiDAR + Radar + Camera fusion
- **Neural Networks**: End-to-end BEV transformation learning

## Performance Optimization

### Memory Optimization
- Efficient GPU memory management
- Streaming data processing
- Batch processing support

### Computational Optimization
- Optimized CUDA kernels
- Memory coalescing
- Shared memory usage


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KITTI dataset for autonomous driving research
- PyTorch team for CUDA extension framework

---

