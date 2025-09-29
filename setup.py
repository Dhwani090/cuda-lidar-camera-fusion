from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
import os

def get_cuda_paths():
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    cuda_include = os.path.join(cuda_home, 'include')
    cuda_lib = os.path.join(cuda_home, 'lib64')
    
    if not os.path.exists(cuda_include):
        raise RuntimeError(f"CUDA include directory not found: {cuda_include}")
    
    return cuda_include, cuda_lib

def get_torch_cuda_paths():
    torch_cuda_include = torch.utils.cpp_extension.CUDA_HOME
    if torch_cuda_include is None:
        raise RuntimeError("PyTorch CUDA not found. Please install PyTorch with CUDA support.")
    
    torch_cuda_lib = os.path.join(torch_cuda_include, 'lib64')
    return torch_cuda_include, torch_cuda_lib

def setup_cuda_extension():
    cuda_include, cuda_lib = get_cuda_paths()
    torch_cuda_include, torch_cuda_lib = get_torch_cuda_paths()
    
    ext = Pybind11Extension(
        'bev_cuda',
        sources=[
            'python/bev_cuda_extension.cpp',
            'kernels/lidar_voxelization.cu',
            'kernels/camera_bev_projection.cu',
            'kernels/feature_fusion.cu',
        ],
        include_dirs=[
            'kernels',
            cuda_include,
            torch_cuda_include,
            pybind11.get_include(),
        ],
        library_dirs=[
            cuda_lib,
            torch_cuda_lib,
        ],
        libraries=['cudart', 'cublas', 'curand'],
        language='c++',
        cxx_std=14,
    )
    
    return ext

setup(
    name='bev_cuda',
    version='1.0.0',
    description='GPU-accelerated Bird\'s-Eye-View transformation for LiDAR + Camera fusion',
    author='CUDA CV Team',
    ext_modules=[setup_cuda_extension()],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
        'pybind11>=2.6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'open3d>=0.13.0',
            'plotly>=5.0',
        ],
    },
)
