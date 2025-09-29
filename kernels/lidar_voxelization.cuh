#ifndef LIDAR_VOXELIZATION_CUH
#define LIDAR_VOXELIZATION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

// Constants for BEV grid dimensions
#define BEV_WIDTH 200
#define BEV_HEIGHT 200
#define BEV_DEPTH 32
#define MAX_POINTS_PER_VOXEL 10

// Point cloud structure: [x, y, z, intensity]
struct PointCloud {
    float* points;  // N x 4 array (x, y, z, intensity)
    int num_points;
};

// BEV grid structure
struct BEVGrid {
    float* occupancy;     // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* intensity;     // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* max_height;    // BEV_WIDTH x BEV_HEIGHT
    int* point_count;     // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
};

// Voxelization parameters
struct VoxelizationParams {
    float voxel_size_x;      // Size of each voxel in X direction (meters)
    float voxel_size_y;      // Size of each voxel in Y direction (meters)
    float voxel_size_z;      // Size of each voxel in Z direction (meters)
    float min_x, max_x;      // Bounding box in X direction
    float min_y, max_y;      // Bounding box in Y direction
    float min_z, max_z;      // Bounding box in Z direction
    float max_range;         // Maximum range for points (meters)
};

/**
 * CUDA kernel for LiDAR point cloud voxelization
 * Each thread processes one point and maps it to the appropriate voxel
 * Uses atomic operations to handle multiple points per voxel
 */
__global__ void lidar_voxelization_kernel(
    const float* points,           // Input: N x 4 (x, y, z, intensity)
    int num_points,
    float* occupancy_grid,        // Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* intensity_grid,        // Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* max_height_grid,       // Output: BEV_WIDTH x BEV_HEIGHT
    int* point_count_grid,        // Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    VoxelizationParams params
);

/**
 * CUDA kernel for max pooling intensity values within each voxel
 * Reduces noise and provides cleaner intensity maps
 */
__global__ void intensity_max_pooling_kernel(
    float* intensity_grid,        // Input/Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    int* point_count_grid,        // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    int kernel_size
);

/**
 * CUDA kernel for height-based occupancy filtering
 * Removes ground points and focuses on elevated objects
 */
__global__ void height_filtering_kernel(
    float* occupancy_grid,        // Input/Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* max_height_grid,       // Input: BEV_WIDTH x BEV_HEIGHT
    float ground_height_threshold // Points below this height are filtered
);

/**
 * Host function to launch LiDAR voxelization
 * Manages memory allocation and kernel launches
 */
cudaError_t lidar_voxelization(
    const PointCloud& point_cloud,
    BEVGrid& bev_grid,
    const VoxelizationParams& params,
    cudaStream_t stream = 0
);

/**
 * Utility function to initialize voxelization parameters
 */
void init_voxelization_params(
    VoxelizationParams& params,
    float voxel_size = 0.1f,      // 10cm voxels
    float max_range = 50.0f       // 50m range
);

/**
 * Memory management functions
 */
cudaError_t allocate_bev_grid(BEVGrid& grid);
cudaError_t free_bev_grid(BEVGrid& grid);
cudaError_t reset_bev_grid(BEVGrid& grid);

#endif // LIDAR_VOXELIZATION_CUH
