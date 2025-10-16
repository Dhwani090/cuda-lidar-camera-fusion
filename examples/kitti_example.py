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
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from python.bev_api import BEVTransformer, FusionMethod
        self.bev_transformer = BEVTransformer(device=device)
    
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
            
            bev_features = self.bev_transformer.transform(
                lidar_points, camera_image, camera_intrinsics, camera_extrinsics,
                2, 0.6, 0.4
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

def main():
    print("GPU-Accelerated BEV Transformation Example")
    print("=========================================")
    
    example = BEVExample(device='cuda')
    
    print("\n Running performance benchmark:")
    example.run_benchmark_example()
    
    print("\nCOMPLETED")

if __name__ == "__main__":
    main()