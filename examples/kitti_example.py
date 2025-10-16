import os
import sys
import numpy as np
import torch
import cv2
from typing import Tuple, Optional, Dict, List
import struct
from pathlib import Path

class KITTIDataset:
    def __init__(self, data_root: str, sequence: str = '00'):
        self.data_root = Path(data_root)
        self.sequence = sequence
        
        self.sequence_path = self.data_root / 'sequences' / sequence
        self.calib_path = self.data_root / 'calib' / f'{sequence}.txt'
        
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"ERROR: Sequence path not found: {self.sequence_path}")
        if not self.calib_path.exists():
            raise FileNotFoundError(f"ERROR: Calibration file not found: {self.calib_path}")
        
        self.calib_data = self._load_calibration()
        
        self.lidar_files = sorted(list((self.sequence_path / 'velodyne').glob('*.bin')))
        self.image_files = sorted(list((self.sequence_path / 'image_2').glob('*.png')))
        
        if len(self.lidar_files) != len(self.image_files):
            print(f"WARNING: LIDAR ({len(self.lidar_files)}) and image ({len(self.image_files)}) files are different sizes")
        
        self.num_frames = min(len(self.lidar_files), len(self.image_files))
        print(f"Loaded KITTI sequence {sequence} with {self.num_frames} frames")
    
    def _load_calibration(self) -> Dict:
        calib_data = {}
        
        with open(self.calib_path, 'r') as f:
            for line in f:
                if line.startswith('P2:'):
                    values = [float(x) for x in line.split()[1:]]
                    calib_data['P2'] = np.array(values).reshape(3, 4)
                elif line.startswith('Tr_velo_to_cam:'):
                    values = [float(x) for x in line.split()[1:]]
                    calib_data['Tr_velo_to_cam'] = np.array(values).reshape(3, 4)
                elif line.startswith('Tr_imu_to_velo:'):
                    values = [float(x) for x in line.split()[1:]]
                    calib_data['Tr_imu_to_velo'] = np.array(values).reshape(3, 4)
        
        return calib_data
    
    def load_lidar_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx >= len(self.lidar_files):
            raise IndexError(f"ERROR: Frame index {frame_idx} out of range")
        
        lidar_file = self.lidar_files[frame_idx]
        
        with open(lidar_file, 'rb') as f:
            data = f.read()
        
        points = struct.unpack('f' * (len(data) // 4), data)
        points = np.array(points).reshape(-1, 4)
        
        return points.astype(np.float32)
    
    def load_camera_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx >= len(self.image_files):
            raise IndexError(f"ERROR: Frame index {frame_idx} out of range")
        
        image_file = self.image_files[frame_idx]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image.astype(np.uint8)
    
    def get_camera_intrinsics(self) -> np.ndarray:
        if 'P2' not in self.calib_data:
            raise ValueError("ERROR: Camera intrinsics not found")
        
        P2 = self.calib_data['P2']
        intrinsics = P2[:3, :3]
        
        return intrinsics.astype(np.float32)
    
    def get_camera_extrinsics(self) -> np.ndarray:
        if 'Tr_velo_to_cam' not in self.calib_data:
            raise ValueError("ERROR: Camera extrinsics not found")
        
        Tr_velo_to_cam = self.calib_data['Tr_velo_to_cam']
        extrinsics = np.eye(4)
        extrinsics[:3, :] = Tr_velo_to_cam
        
        return extrinsics.astype(np.float32)
    
    def get_frame(self, frame_idx: int) -> Dict:
        lidar_points = self.load_lidar_frame(frame_idx)
        camera_image = self.load_camera_frame(frame_idx)
        intrinsics = self.get_camera_intrinsics()
        extrinsics = self.get_camera_extrinsics()
        
        return {
            'lidar_points': lidar_points,
            'camera_image': camera_image,
            'camera_intrinsics': intrinsics,
            'camera_extrinsics': extrinsics,
            'frame_idx': frame_idx
        }
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        return self.get_frame(idx)

class BEVExample:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from python.bev_api import BEVTransformer, FusionMethod
            self.bev_transformer = BEVTransformer(device=device)
            self.api_available = True
        except ImportError:
            print("Warning: BEV API not available")
            self.api_available = False
    
    def run_kitti_example(self, data_root: str, sequence: str = '00', num_frames: int = 10):
        print(f"Running BEV transformation on KITTI sequence {sequence}")
        
        dataset = KITTIDataset(data_root, sequence)
        
        for i in range(min(num_frames, len(dataset))):
            print(f"Processing frame {i+1}/{num_frames}")
            
            frame_data = dataset.get_frame(i)
            
            lidar_points = torch.from_numpy(frame_data['lidar_points']).to(self.device)
            camera_image = torch.from_numpy(frame_data['camera_image']).to(self.device)
            camera_intrinsics = torch.from_numpy(frame_data['camera_intrinsics']).to(self.device)
            camera_extrinsics = torch.from_numpy(frame_data['camera_extrinsics']).to(self.device)
            
            if self.api_available:
                bev_features = self.bev_transformer.transform(
                    lidar_points, camera_image, camera_intrinsics, camera_extrinsics,
                    2, 0.6, 0.4
                )
            else:
                bev_features = self._fallback_transform(
                    lidar_points, camera_image, camera_intrinsics, camera_extrinsics
                )
            
            print(f"Frame {i+1} processed. BEV features shape: {bev_features.shape}")
    
    def run_benchmark_example(self):
        print("Running performance benchmark:")
        
        try:
            from benchmarks.benchmark import run_benchmark
            results = run_benchmark('benchmark_results')
            print("Benchmark completed. Results saved to 'benchmark_results' directory.")
        except ImportError:
            print("Warning: Benchmark module not available.")
    
    def _fallback_transform(self, lidar_points: torch.Tensor, camera_image: torch.Tensor,
                          camera_intrinsics: torch.Tensor, camera_extrinsics: torch.Tensor) -> torch.Tensor:
        BEV_WIDTH = BEV_HEIGHT = 200
        BEV_DEPTH = 32
        
        points_np = lidar_points.cpu().numpy()
        occupancy_grid = np.zeros((BEV_WIDTH, BEV_HEIGHT, BEV_DEPTH), dtype=np.float32)
        
        for point in points_np[:10000]:
            x, y, z, intensity = point
            voxel_x = int((x + 25) / 0.1)
            voxel_y = int((y + 25) / 0.1)
            voxel_z = int((z + 2.5) / 0.1)
            
            if 0 <= voxel_x < BEV_WIDTH and 0 <= voxel_y < BEV_HEIGHT and 0 <= voxel_z < BEV_DEPTH:
                occupancy_grid[voxel_x, voxel_y, voxel_z] += 1.0
        
        image_np = camera_image.cpu().numpy()
        color_grid = np.zeros((BEV_WIDTH, BEV_HEIGHT, 3), dtype=np.float32)
        
        for x in range(0, BEV_WIDTH, 10):
            for y in range(0, BEV_HEIGHT, 10):
                world_x = (x - BEV_WIDTH/2) * 0.1
                world_y = (y - BEV_HEIGHT/2) * 0.1
                world_z = 0.0
                
                img_u = int(320 + world_x * 10)
                img_v = int(240 + world_y * 10)
                
                if 0 <= img_u < image_np.shape[1] and 0 <= img_v < image_np.shape[0]:
                    color_grid[x, y] = image_np[img_v, img_u] / 255.0
        
        occupancy_tensor = torch.from_numpy(occupancy_grid).to(self.device)
        color_tensor = torch.from_numpy(color_grid).to(self.device)
        
        occupancy_flat = occupancy_tensor.view(BEV_WIDTH, BEV_HEIGHT, -1)
        color_flat = color_tensor.view(BEV_WIDTH, BEV_HEIGHT, -1)
        
        fused_features = torch.cat([occupancy_flat, color_flat], dim=-1)
        
        return fused_features

def main():
    print("GPU-Accelerated BEV Transformation Example")
    print("=========================================")
    
    example = BEVExample(device='cuda')
    
    print("\n Running performance benchmark:")
    example.run_benchmark_example()
    
    print("\nCOMPLETED")

if __name__ == "__main__":
    main()