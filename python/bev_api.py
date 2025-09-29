
import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from enum import Enum

try:
    import bev_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not available.")

class FusionMethod(Enum):
    SIMPLE_CONCATENATION = 0
    WEIGHTED_FUSION = 1
    ATTENTION_FUSION = 2

class BEVTransformer:
    """
    High-level API for Bird's-Eye-View transformation
    """
    
    def __init__(self, 
                 voxel_size: float = 0.1,
                 max_range: float = 50.0,
                 device: str = 'cuda'):
        """
        Initialize BEV transformer
        
        Args:
            voxel_size: Size of each voxel in meters
            max_range: Maximum range for LiDAR points in meters
            device: Device to use ('cuda' or 'cpu')
        """
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.device = device
        
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available.")
    
    def transform_lidar(self, 
                        points: Union[torch.Tensor, np.ndarray],
                        return_intensity: bool = True,
                        return_height: bool = True) -> torch.Tensor:
        """
        Transform LiDAR point cloud to BEV occupancy grid
        
        Args:
            points: Point cloud as Nx4 array (x, y, z, intensity)
            return_intensity: Whether to return intensity features
            return_height: Whether to return height features
            
        Returns:
            BEV occupancy grid tensor
        """
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        if points.device.type != self.device:
            points = points.to(self.device)
        
        if CUDA_AVAILABLE and self.device == 'cuda':
            return bev_cuda.lidar_voxelization(
                points, self.voxel_size, self.max_range, 
                return_intensity, return_height
            )
        else:
            raise RuntimeError("CUDA not available.")
    
    def transform_camera(self,
                        image: Union[torch.Tensor, np.ndarray],
                        intrinsics: Union[torch.Tensor, np.ndarray],
                        extrinsics: Union[torch.Tensor, np.ndarray],
                        interpolation_method: int = 1) -> torch.Tensor:
        """
        Transform camera image to BEV color grid
        
        Args:
            image: Camera image as HxWx3 RGB array
            intrinsics: Camera intrinsic matrix 3x3
            extrinsics: Camera extrinsic matrix 4x4
            interpolation_method: 0=nearest, 1=bilinear
            
        Returns:
            BEV color and depth tensors
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics).float()
        if isinstance(extrinsics, np.ndarray):
            extrinsics = torch.from_numpy(extrinsics).float()
        
        image = image.to(self.device)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)
        
        if CUDA_AVAILABLE and self.device == 'cuda':
            return bev_cuda.camera_to_bev_projection(
                image, intrinsics, extrinsics, 
                self.voxel_size, self.max_range, interpolation_method
            )
        else:
            raise RuntimeError("CUDA not available.")
    
    def fuse_features(self,
                     lidar_occupancy: torch.Tensor,
                     lidar_intensity: torch.Tensor,
                     camera_color: torch.Tensor,
                     camera_depth: torch.Tensor,
                     fusion_method: FusionMethod = FusionMethod.ATTENTION_FUSION,
                     lidar_weight: float = 0.6,
                     camera_weight: float = 0.4,
                     normalize_features: bool = True) -> torch.Tensor:
        """
        Fuse LiDAR and camera features
        
        Args:
            lidar_occupancy: LiDAR occupancy grid
            lidar_intensity: LiDAR intensity grid
            camera_color: Camera color grid
            camera_depth: Camera depth grid
            fusion_method: Fusion method to use
            lidar_weight: Weight for LiDAR features
            camera_weight: Weight for camera features
            normalize_features: Whether to normalize features
            
        Returns:
            Fused feature tensor
        """
        if CUDA_AVAILABLE and self.device == 'cuda':
            return bev_cuda.fuse_bev_features(
                lidar_occupancy, lidar_intensity, camera_color, camera_depth,
                fusion_method.value, lidar_weight, camera_weight, normalize_features
            )
        else:
            raise RuntimeError("CUDA not available.")
    
    def transform(self,
                  lidar_points: Union[torch.Tensor, np.ndarray],
                  camera_image: Union[torch.Tensor, np.ndarray],
                  camera_intrinsics: Union[torch.Tensor, np.ndarray],
                  camera_extrinsics: Union[torch.Tensor, np.ndarray],
                  fusion_method: FusionMethod = FusionMethod.ATTENTION_FUSION,
                  lidar_weight: float = 0.6,
                  camera_weight: float = 0.4) -> torch.Tensor:
        """
        Complete BEV transformation pipeline
        
        Args:
            lidar_points: LiDAR point cloud Nx4
            camera_image: Camera image HxWx3
            camera_intrinsics: Camera intrinsics 3x3
            camera_extrinsics: Camera extrinsics 4x4
            fusion_method: Fusion method
            lidar_weight: LiDAR feature weight
            camera_weight: Camera feature weight
            
        Returns:
            Fused BEV feature tensor
        """
        if CUDA_AVAILABLE and self.device == 'cuda':
            return bev_cuda.bev_transformation_pipeline(
                lidar_points, camera_image, camera_intrinsics, camera_extrinsics,
                self.voxel_size, self.max_range, fusion_method.value,
                lidar_weight, camera_weight
            )
        else:
            raise RuntimeError("CUDA not available.")
    

# Utility functions
def create_camera_intrinsics(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """Create camera intrinsic matrix"""
    intrinsics = torch.zeros(3, 3)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    intrinsics[2, 2] = 1.0
    return intrinsics

def create_camera_extrinsics(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Create camera extrinsic matrix"""
    extrinsics = torch.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation
    return extrinsics

def load_kitti_calibration(calib_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load KITTI calibration data"""
    # This is a simplified version - in practice, you'd parse the actual KITTI calib file
    intrinsics = create_camera_intrinsics(721.5, 721.5, 609.5, 172.8)
    extrinsics = torch.eye(4)  # Identity matrix for now
    return intrinsics, extrinsics
