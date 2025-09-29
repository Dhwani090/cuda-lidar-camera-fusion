"""
Performance benchmarking system for BEV transformation
"""

import time
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import json
import os

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    method: str
    operation: str
    execution_time: float
    memory_usage: float
    throughput: float
    accuracy: Optional[float] = None
    parameters: Optional[Dict] = None

class BEVBenchmark:
    """
    Comprehensive benchmarking system for BEV transformation
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 num_warmup: int = 10,
                 num_iterations: int = 100):
        """
        Initialize benchmark system
        
        Args:
            device: Device to use for benchmarking
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
        """
        self.device = device
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.results = []
        
        try:
            import bev_cuda
            self.cuda_available = True
        except ImportError:
            self.cuda_available = False
            print("Warning: CUDA extension not available.")
    
    
    def benchmark_lidar_voxelization(self, data: Dict) -> BenchmarkResult:
        """Benchmark LiDAR voxelization"""
        points = data['lidar_points']
        
        for _ in range(self.num_warmup):
            _ = bev_cuda.lidar_voxelization(points, 0.1, 50.0, True, True)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.num_iterations):
            result = bev_cuda.lidar_voxelization(points, 0.1, 50.0, True, True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        execution_time = (end_time - start_time) / self.num_iterations
        throughput = points.size(0) / execution_time
        
        return BenchmarkResult(
            method='CUDA',
            operation='lidar_voxelization',
            execution_time=execution_time * 1000,  
            memory_usage=0.0,  # TODO: Implement memory measurement
            throughput=throughput,
            parameters={'num_points': points.size(0), 'voxel_size': 0.1}
        )
    
    def benchmark_camera_projection(self, data: Dict) -> BenchmarkResult:
        """Benchmark camera-to-BEV projection"""
        image = data['camera_image']
        intrinsics = data['camera_intrinsics']
        extrinsics = data['camera_extrinsics']
        
        for _ in range(self.num_warmup):
            _ = bev_cuda.camera_to_bev_projection(image, intrinsics, extrinsics, 0.1, 50.0, 1)        

        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.num_iterations):
            result = bev_cuda.camera_to_bev_projection(image, intrinsics, extrinsics, 0.1, 50.0, 1)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        execution_time = (end_time - start_time) / self.num_iterations
        throughput = image.numel() / execution_time
        
        return BenchmarkResult(
            method='CUDA',
            operation='camera_projection',
            execution_time=execution_time * 1000,  
            memory_usage=0.0,
            throughput=throughput,
            parameters={'image_size': image.shape, 'voxel_size': 0.1}
        )
    
    def benchmark_feature_fusion(self, data: Dict) -> BenchmarkResult:
        """Benchmark feature fusion"""

        lidar_occupancy = torch.randn(200, 200, 32).to(self.device)
        lidar_intensity = torch.randn(200, 200, 32).to(self.device)
        camera_color = torch.randn(200, 200, 3).to(self.device)
        camera_depth = torch.randn(200, 200).to(self.device)

        for _ in range(self.num_warmup):
            _ = bev_cuda.fuse_bev_features(lidar_occupancy, lidar_intensity, 
                                        camera_color, camera_depth, 2, 0.6, 0.4, True)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.num_iterations):
            result = bev_cuda.fuse_bev_features(lidar_occupancy, lidar_intensity, 
                                              camera_color, camera_depth, 2, 0.6, 0.4, True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        execution_time = (end_time - start_time) / self.num_iterations
        throughput = lidar_occupancy.numel() / execution_time
        
        return BenchmarkResult(
            method='CUDA',
            operation='feature_fusion',
            execution_time=execution_time * 1000,  # Convert to ms
            memory_usage=0.0,
            throughput=throughput,
            parameters={'bev_size': lidar_occupancy.shape, 'fusion_method': 2}
        )
    
    def benchmark_complete_pipeline(self, data: Dict) -> BenchmarkResult:
        """Benchmark complete BEV transformation pipeline"""
        points = data['lidar_points']
        image = data['camera_image']
        intrinsics = data['camera_intrinsics']
        extrinsics = data['camera_extrinsics']
        
        for _ in range(self.num_warmup):
            _ = bev_cuda.bev_transformation_pipeline(points, image, intrinsics, extrinsics, 
                                                   0.1, 50.0, 2, 0.6, 0.4)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.num_iterations):
            result = bev_cuda.bev_transformation_pipeline(points, image, intrinsics, extrinsics, 
                                                        0.1, 50.0, 2, 0.6, 0.4)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        execution_time = (end_time - start_time) / self.num_iterations
        throughput = points.size(0) / execution_time
        
        return BenchmarkResult(
            method='CUDA',
            operation='complete_pipeline',
            execution_time=execution_time * 1000,  
            memory_usage=0.0,
            throughput=throughput,
            parameters={'num_points': points.size(0), 'image_size': image.shape}
        )
    
    
    
    def run_comprehensive_benchmark(self, data: Dict) -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite"""
        print("Running comprehensive BEV transformation benchmark...")

        results = []
        results.append(self.benchmark_lidar_voxelization(data))
        results.append(self.benchmark_camera_projection(data))
        results.append(self.benchmark_feature_fusion(data))
        results.append(self.benchmark_complete_pipeline(data))
        
        self.results.extend(results)
        
        return results


def print_benchmark_results(results: List[BenchmarkResult]):
    print("\n" + "=" * 60)
    print("BEV TRANSFORMATION BENCHMARK RESULTS")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result.operation.upper()}:")
        print(f"  Method: {result.method}")
        print(f"  Execution Time: {result.execution_time:.2f} ms")
        print(f"  Throughput: {result.throughput:.0f} ops/sec")
        if result.parameters:
            print(f"  Parameters: {result.parameters}")
    
    print("\n" + "=" * 60)


def run_benchmark(data: Dict):
    """Run complete benchmark suite"""
    benchmark = BEVBenchmark()
    results = benchmark.run_comprehensive_benchmark(data)
    print_benchmark_results(results)
    return results
