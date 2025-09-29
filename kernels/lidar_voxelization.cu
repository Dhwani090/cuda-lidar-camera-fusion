#include "lidar_voxelization.cuh"
#include <math.h>
#include <stdio.h>

__device__ __forceinline__ bool is_point_valid(
    float x, float y, float z, const VoxelizationParams& params
) {
    return (x >= params.min_x && x <= params.max_x && y >= params.min_y && y <= params.max_y && z >= params.min_z && z <= params.max_z && sqrtf(x*x + y*y + z*z) <= params.max_range);
}

__device__ __forceinline__ void world_to_voxel(
    float x, float y, float z,
    int& voxel_x, int& voxel_y, int& voxel_z,
    const VoxelizationParams& params
) {
    voxel_x = (int)((x - params.min_x) / params.voxel_size_x);
    voxel_y = (int)((y - params.min_y) / params.voxel_size_y);
    voxel_z = (int)((z - params.min_z) / params.voxel_size_z);
    
    voxel_x = max(0, min(voxel_x, BEV_WIDTH - 1));
    voxel_y = max(0, min(voxel_y, BEV_HEIGHT - 1));
    voxel_z = max(0, min(voxel_z, BEV_DEPTH - 1));
}

// 3D voxel coordinates to linear index
__device__ __forceinline__ int get_voxel_index(int x, int y, int z) {
    return z * BEV_WIDTH * BEV_HEIGHT + y * BEV_WIDTH + x;
}

/**
 * Transforms raw LIDAR point cloud data to structured grids
 * Grids are 3d or 2d but stored in 1D arrays for efficiency
 * atomic operations are used to handle multiple points per voxel
 * Majority of voxels are empty
 * Each thread gets assigned and processes one point from the point cloud
 */
__global__ void lidar_voxelization_kernel(
    const float* points,          // Input: N x 4 (x, y, z, intensity)
    int num_points,
    float* occupancy_grid,        // Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* intensity_grid,        // Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* max_height_grid,       // Output: BEV_WIDTH x BEV_HEIGHT
    int* point_count_grid,        // Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    VoxelizationParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;

    float x = points[idx * 4 + 0];
    float y = points[idx * 4 + 1];
    float z = points[idx * 4 + 2];
    float intensity = points[idx * 4 + 3];
    
    if (!is_point_valid(x, y, z, params)) return;
    
    int voxel_x, voxel_y, voxel_z;
    world_to_voxel(x, y, z, voxel_x, voxel_y, voxel_z, params);

    int voxel_idx = get_voxel_index(voxel_x, voxel_y, voxel_z);
    
    int bev_idx = voxel_y * BEV_WIDTH + voxel_x;
    
    atomicAdd(&point_count_grid[voxel_idx], 1);
    atomicAdd(&occupancy_grid[voxel_idx], 1.0f);
    atomicMax((int*)&intensity_grid[voxel_idx], __float_as_int(intensity));
    atomicMax((int*)&max_height_grid[bev_idx], __float_as_int(z));
}

/**
 * Updates intensity grid with intensity of point with most points
 * Voxels with more points are more reliable
 * Smoother intensity grid is better for neural networks
 */
__global__ void intensity_max_pooling_kernel(
    float* intensity_grid,        // Input/Output: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    int* point_count_grid,        // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT || z >= BEV_DEPTH) return;
    
    int half_kernel = kernel_size / 2;
    float max_intensity = 0.0f;
    int max_count = 0;
    

    for (int dx = -half_kernel; dx <= half_kernel; dx++) {
        for (int dy = -half_kernel; dy <= half_kernel; dy++) {
            int x1 = x + dx;
            int y1 = y + dy;
            
            if (x1 >= 0 && x1 < BEV_WIDTH && y1 >= 0 && y1 < BEV_HEIGHT) {
                int idx = get_voxel_index(x1, y1, z);
                if (point_count_grid[idx] > max_count) {
                    max_count = point_count_grid[idx];
                    max_intensity = intensity_grid[idx];
                }
            }
        }
    }
    
    if (max_count > point_count_grid[get_voxel_index(x, y, z)]) {
        intensity_grid[get_voxel_index(x, y, z)] = max_intensity;
    }
}

/**
 * Height filtering kernel to remove ground points
 */
__global__ void height_filtering_kernel(
    float* occupancy_grid,        
    float* max_height_grid,      
    float ground_height_threshold // Points below this height are filtered
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT || z >= BEV_DEPTH) return;
    
    int bev_idx = y * BEV_WIDTH + x;
    int voxel_idx = get_voxel_index(x, y, z);
    

    float voxel_height = z * 0.1f; 
    if (voxel_height < ground_height_threshold) {
        occupancy_grid[voxel_idx] *= 0.1f; 
    }
}

/**
 * Host function to launch LiDAR voxelization
 */
cudaError_t lidar_voxelization(
    const PointCloud& point_cloud,
    BEVGrid& bev_grid,
    const VoxelizationParams& params,
    cudaStream_t stream
) {

    dim3 block_size(256);
    dim3 grid_size((point_cloud.num_points + block_size.x - 1) / block_size.x);

    lidar_voxelization_kernel<<<grid_size, block_size, 0, stream>>>(
        point_cloud.points,
        point_cloud.num_points,
        bev_grid.occupancy,
        bev_grid.intensity,
        bev_grid.max_height,
        bev_grid.point_count,
        params
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    dim3 pooling_block(8, 8, 4);
    dim3 pooling_grid(
        (BEV_WIDTH + pooling_block.x - 1) / pooling_block.x,
        (BEV_HEIGHT + pooling_block.y - 1) / pooling_block.y,
        (BEV_DEPTH + pooling_block.z - 1) / pooling_block.z
    );
    
    intensity_max_pooling_kernel<<<pooling_grid, pooling_block, 0, stream>>>(
        bev_grid.intensity,
        bev_grid.point_count,
        3 //kernel size (3x3)
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    height_filtering_kernel<<<pooling_grid, pooling_block, 0, stream>>>(
        bev_grid.occupancy,
        bev_grid.max_height,
        0.2f  //ground threshold = 20cm
    );
    
    return cudaGetLastError();
}

// Voxelization parameters initialization
void init_voxelization_params(
    VoxelizationParams& params,
    float voxel_size,
    float max_range
) {
    params.voxel_size_x = voxel_size;
    params.voxel_size_y = voxel_size;
    params.voxel_size_z = voxel_size;
    params.max_range = max_range;
    
    
    params.min_x = -max_range / 2.0f;
    params.max_x = max_range / 2.0f;
    params.min_y = -max_range / 2.0f;
    params.max_y = max_range / 2.0f;
    params.min_z = -2.0f;  // 2m below ground
    params.max_z = 3.0f;   // 3m above ground
}

//Allocate memory for BEV grid
cudaError_t allocate_bev_grid(BEVGrid& grid) {
    size_t grid_size = BEV_WIDTH * BEV_HEIGHT * BEV_DEPTH * sizeof(float);
    size_t bev_size = BEV_WIDTH * BEV_HEIGHT * sizeof(float);
    size_t count_size = BEV_WIDTH * BEV_HEIGHT * BEV_DEPTH * sizeof(int);
    
    cudaError_t err;
    
    err = cudaMalloc(&grid.occupancy, grid_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&grid.intensity, grid_size);
    if (err != cudaSuccess) {
        cudaFree(grid.occupancy);
        return err;
    }
    
    err = cudaMalloc(&grid.max_height, bev_size);
    if (err != cudaSuccess) {
        cudaFree(grid.occupancy);
        cudaFree(grid.intensity);
        return err;
    }
    
    err = cudaMalloc(&grid.point_count, count_size);
    if (err != cudaSuccess) {
        cudaFree(grid.occupancy);
        cudaFree(grid.intensity);
        cudaFree(grid.max_height);
        return err;
    }
    
    return cudaSuccess;
}

// Free memory for BEV grid
cudaError_t free_bev_grid(BEVGrid& grid) {
    cudaError_t err1 = cudaFree(grid.occupancy);
    cudaError_t err2 = cudaFree(grid.intensity);
    cudaError_t err3 = cudaFree(grid.max_height);
    cudaError_t err4 = cudaFree(grid.point_count);
    
    return (err1 != cudaSuccess) ? err1 : 
           (err2 != cudaSuccess) ? err2 :
           (err3 != cudaSuccess) ? err3 : err4;
}

//BEV grid reset to zero
cudaError_t reset_bev_grid(BEVGrid& grid) {
    size_t grid_size = BEV_WIDTH * BEV_HEIGHT * BEV_DEPTH * sizeof(float);
    size_t bev_size = BEV_WIDTH * BEV_HEIGHT * sizeof(float);
    size_t count_size = BEV_WIDTH * BEV_HEIGHT * BEV_DEPTH * sizeof(int);
    
    cudaError_t err;
    
    err = cudaMemset(grid.occupancy, 0, grid_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(grid.intensity, 0, grid_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(grid.max_height, 0, bev_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(grid.point_count, 0, count_size);
    
    return err;
}
