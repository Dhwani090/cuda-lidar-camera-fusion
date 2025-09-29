#include "feature_fusion.cuh"
#include <math.h>
#include <stdio.h>

//linear index for 3D features
__device__ __forceinline__ int get_feature_3d_index(int x, int y, int z, int channels) {
    return z * BEV_WIDTH * BEV_HEIGHT * channels + y * BEV_WIDTH * channels + x * channels;
}

//linear index for 2D features
__device__ __forceinline__ int get_feature_2d_index(int x, int y, int channels) {
    return y * BEV_WIDTH * channels + x * channels;
}

//sigmoid activation
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

//softmax activation 
__device__ __forceinline__ float softmax(float x, float sum) {
    return expf(x) / sum;
}

//fabsf calculate absolute value of float
//attention weight based on depth and strength of features
__device__ __forceinline__ float compute_attention_weight(
    float lidar_feat, float camera_feat, float depth
) {
    float feature_strength = fabsf(lidar_feat) + fabsf(camera_feat);
    float depth_weight = 1.0f / (1.0f + depth * 0.1f); 
    return sigmoid(feature_strength * depth_weight);
}

//Combines LiDAR and camera features
//flattens 3d Lidar features into 1d vector for fusion
//simple concatenation with no weights
__global__ void simple_fusion_kernel(
    const float* lidar_occupancy,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const float* lidar_intensity,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const unsigned char* camera_color, // Input: BEV_WIDTH x BEV_HEIGHT x 3
    const float* camera_depth,        // Input: BEV_WIDTH x BEV_HEIGHT
    float* fused_features,            // Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    FusionParams params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT) return;
    
    int output_idx = get_feature_2d_index(x, y, feature_dim);
    int depth_idx = y * BEV_WIDTH + x;
    int color_idx = get_feature_2d_index(x, y, 3);
    
    int feature_offset = 0;
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        int lidar_idx = get_feature_3d_index(x, y, z, 1);
        fused_features[output_idx + feature_offset] = lidar_occupancy[lidar_idx];
        feature_offset++;
    }
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        int lidar_idx = get_feature_3d_index(x, y, z, 1);
        fused_features[output_idx + feature_offset] = lidar_intensity[lidar_idx];
        feature_offset++;
    }
    
    for (int i = 0; i < 3; i++) {
        fused_features[output_idx + feature_offset] = camera_color[color_idx + i] / 255.0f;
        feature_offset++;
    }
    fused_features[output_idx + feature_offset] = camera_depth[depth_idx];
    feature_offset++;
    
    //zero padding
    while (feature_offset < feature_dim) {
        fused_features[output_idx + feature_offset] = 0.0f;
        feature_offset++;
    }
}

//Combines LiDAR and camera features through weighted fusion
__global__ void weighted_fusion_kernel(
    const float* lidar_occupancy,     
    const float* lidar_intensity,     
    const unsigned char* camera_color, 
    const float* camera_depth,        
    float* fused_features,            
    int feature_dim,
    FusionParams params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT) return;
    
    int output_idx = get_feature_2d_index(x, y, feature_dim);
    int depth_idx = y * BEV_WIDTH + x;
    int color_idx = get_feature_2d_index(x, y, 3);
    float lidar_weight = params.lidar_weight;
    float camera_weight = params.camera_weight;
    int feature_offset = 0;
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        int lidar_idx = get_feature_3d_index(x, y, z, 1);
        fused_features[output_idx + feature_offset] = lidar_occupancy[lidar_idx] * lidar_weight;
        feature_offset++;
    }
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        int lidar_idx = get_feature_3d_index(x, y, z, 1);
        fused_features[output_idx + feature_offset] = lidar_intensity[lidar_idx] * lidar_weight;
        feature_offset++;
    }
    for (int i = 0; i < 3; i++) {
        fused_features[output_idx + feature_offset] = (camera_color[color_idx + i] / 255.0f) * camera_weight;
        feature_offset++;
    }
    fused_features[output_idx + feature_offset] = camera_depth[depth_idx] * camera_weight;
    feature_offset++;

    while (feature_offset < feature_dim) {
        fused_features[output_idx + feature_offset] = 0.0f;
        feature_offset++;
    }
}

//computes average of LiDAR and camera features for each BEV cell
//then computes attention weights based on the average features and depth
//weights are higher based on strength of data quality
__global__ void attention_fusion_kernel(
    const float* lidar_occupancy,     
    const float* lidar_intensity,     
    const unsigned char* camera_color, 
    const float* camera_depth,        
    float* fused_features,            
    int feature_dim,
    FusionParams params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT) return;
    
    int output_idx = get_feature_2d_index(x, y, feature_dim);
    int depth_idx = y * BEV_WIDTH + x;
    int color_idx = get_feature_2d_index(x, y, 3);
    
    float depth = camera_depth[depth_idx];
    float avg_occupancy = 0.0f;
    float avg_intensity = 0.0f;
    int lidar_idx = 0;
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        lidar_idx = get_feature_3d_index(x, y, z, 1);
        avg_occupancy += lidar_occupancy[lidar_idx];
        avg_intensity += lidar_intensity[lidar_idx];
    }
    avg_occupancy /= BEV_DEPTH;
    avg_intensity /= BEV_DEPTH;
    
    float camera_strength = (camera_color[color_idx] + camera_color[color_idx + 1] + camera_color[color_idx + 2]) / (3.0f * 255.0f);
    float lidar_attention = compute_attention_weight(avg_occupancy, avg_intensity, depth);
    float camera_attention = compute_attention_weight(camera_strength, 0.0f, depth);
    
    float total_attention = lidar_attention + camera_attention;
    if (total_attention > 0.0f) {
        lidar_attention /= total_attention;
        camera_attention /= total_attention;
    }
    int feature_offset = 0;
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        int lidar_idx = get_feature_3d_index(x, y, z, 1);
        fused_features[output_idx + feature_offset] = lidar_occupancy[lidar_idx] * lidar_attention;
        feature_offset++;
    }
    
    for (int z = 0; z < BEV_DEPTH; z++) {
        int lidar_idx = get_feature_3d_index(x, y, z, 1);
        fused_features[output_idx + feature_offset] = lidar_intensity[lidar_idx] * lidar_attention;
        feature_offset++;
    }
    for (int i = 0; i < 3; i++) {
        fused_features[output_idx + feature_offset] = (camera_color[color_idx + i] / 255.0f) * camera_attention;
        feature_offset++;
    } 
    fused_features[output_idx + feature_offset] = camera_depth[depth_idx] * camera_attention;
    feature_offset++;
    
    while (feature_offset < feature_dim) {
        fused_features[output_idx + feature_offset] = 0.0f;
        feature_offset++;
    }
}


/**
 * Feature normalization kernel
 */
__global__ void normalize_features_kernel(
    float* features,                  // Input/Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    float min_val, float max_val
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT) return;
    
    int idx = get_feature_2d_index(x, y, feature_dim);
    
    for (int c = 0; c < feature_dim; c++) {
        float val = features[idx + c];
        val = (val - min_val) / (max_val - min_val);
        val = fmaxf(0.0f, fminf(1.0f, val));
        features[idx + c] = val;
    }
}

//replaces each feature with the average of its neighbors 
//smoothens out the features
__global__ void enhance_features_kernel(
    float* features,                  // Input/Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT) return;
    
    int half_kernel = kernel_size / 2;
    int idx = get_feature_2d_index(x, y, feature_dim);
    
    for (int c = 0; c < feature_dim; c++) {
        float sum = 0.0f;
        int count = 0;
        
        for (int dx = -half_kernel; dx <= half_kernel; dx++) {
            for (int dy = -half_kernel; dy <= half_kernel; dy++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < BEV_WIDTH && ny >= 0 && ny < BEV_HEIGHT) {
                    int nidx = get_feature_2d_index(nx, ny, feature_dim);
                    sum += features[nidx + c];
                    count++;
                }
            }
        }
        
        if (count > 0) {
            features[idx + c] = sum / count;
        }
    }
}

/**
 * Host function to launch feature fusion
 */
cudaError_t fuse_bev_features(
    const BEVGrid& lidar_grid,
    const unsigned char* camera_color_grid,
    const float* camera_depth_grid,
    FusedBEVFeatures& fused_features,
    const FusionParams& params,
    cudaStream_t stream
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (BEV_WIDTH + block_size.x - 1) / block_size.x,
        (BEV_HEIGHT + block_size.y - 1) / block_size.y
    );
    
    switch (params.method) {
        case SIMPLE_CONCATENATION:
            simple_fusion_kernel<<<grid_size, block_size, 0, stream>>>(
                lidar_grid.occupancy,
                lidar_grid.intensity,
                camera_color_grid,
                camera_depth_grid,
                fused_features.fused_features,
                fused_features.feature_dim,
                params
            );
            break;
            
        case WEIGHTED_FUSION:
            weighted_fusion_kernel<<<grid_size, block_size, 0, stream>>>(
                lidar_grid.occupancy,
                lidar_grid.intensity,
                camera_color_grid,
                camera_depth_grid,
                fused_features.fused_features,
                fused_features.feature_dim,
                params
            );
            break;
            
        case ATTENTION_FUSION:
            attention_fusion_kernel<<<grid_size, block_size, 0, stream>>>(
                lidar_grid.occupancy,
                lidar_grid.intensity,
                camera_color_grid,
                camera_depth_grid,
                fused_features.fused_features,
                fused_features.feature_dim,
                params
            );
            break;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
 
    if (params.normalize_features) {
        normalize_features_kernel<<<grid_size, block_size, 0, stream>>>(
            fused_features.fused_features,
            fused_features.feature_dim,
            0.0f, 1.0f
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

//Memory management functions
cudaError_t allocate_fused_features(FusedBEVFeatures& features, int feature_dim) {
    features.feature_dim = feature_dim;
    
    size_t grid_size = BEV_WIDTH * BEV_HEIGHT * BEV_DEPTH * sizeof(float);
    size_t bev_size = BEV_WIDTH * BEV_HEIGHT * sizeof(float);
    size_t color_size = BEV_WIDTH * BEV_HEIGHT * 3 * sizeof(unsigned char);
    size_t fused_size = BEV_WIDTH * BEV_HEIGHT * feature_dim * sizeof(float);
    
    cudaError_t err;
    
    err = cudaMalloc(&features.occupancy_features, grid_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&features.intensity_features, grid_size);
    if (err != cudaSuccess) {
        cudaFree(features.occupancy_features);
        return err;
    }
    
    err = cudaMalloc(&features.color_features, color_size);
    if (err != cudaSuccess) {
        cudaFree(features.occupancy_features);
        cudaFree(features.intensity_features);
        return err;
    }
    
    err = cudaMalloc(&features.depth_features, bev_size);
    if (err != cudaSuccess) {
        cudaFree(features.occupancy_features);
        cudaFree(features.intensity_features);
        cudaFree(features.color_features);
        return err;
    }
    
    err = cudaMalloc(&features.fused_features, fused_size);
    if (err != cudaSuccess) {
        cudaFree(features.occupancy_features);
        cudaFree(features.intensity_features);
        cudaFree(features.color_features);
        cudaFree(features.depth_features);
        return err;
    }
    
    return cudaSuccess;
}

cudaError_t free_fused_features(FusedBEVFeatures& features) {
    cudaError_t err1 = cudaFree(features.occupancy_features);
    cudaError_t err2 = cudaFree(features.intensity_features);
    cudaError_t err3 = cudaFree(features.color_features);
    cudaError_t err4 = cudaFree(features.depth_features);
    cudaError_t err5 = cudaFree(features.fused_features);
    
    return (err1 != cudaSuccess) ? err1 : 
           (err2 != cudaSuccess) ? err2 :
           (err3 != cudaSuccess) ? err3 :
           (err4 != cudaSuccess) ? err4 : err5;
}

cudaError_t reset_fused_features(FusedBEVFeatures& features) {
    size_t fused_size = BEV_WIDTH * BEV_HEIGHT * features.feature_dim * sizeof(float);
    return cudaMemset(features.fused_features, 0, fused_size);
}
