#ifndef FEATURE_FUSION_CUH
#define FEATURE_FUSION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "lidar_voxelization.cuh"
#include "camera_bev_projection.cuh"

// Feature fusion methods
enum FusionMethod {
    SIMPLE_CONCATENATION = 0,
    WEIGHTED_FUSION = 1,
    ATTENTION_FUSION = 2
};

// Fused BEV feature structure
struct FusedBEVFeatures {
    float* occupancy_features;    // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* intensity_features;    // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    float* color_features;        // BEV_WIDTH x BEV_HEIGHT x 3
    float* depth_features;        // BEV_WIDTH x BEV_HEIGHT
    float* fused_features;        // BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim;              // Dimension of fused features
};

// Fusion parameters
struct FusionParams {
    FusionMethod method;
    float lidar_weight;           // Weight for LiDAR features
    float camera_weight;          // Weight for camera features
    float occupancy_threshold;    // Threshold for occupancy
    float intensity_threshold;    // Threshold for intensity
    bool normalize_features;      // Whether to normalize features
    bool use_depth_guidance;      // Whether to use depth for fusion
};

/**
 * CUDA kernel for simple feature concatenation
 * Concatenates LiDAR and camera features along channel dimension
 */
__global__ void simple_fusion_kernel(
    const float* lidar_occupancy,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const float* lidar_intensity,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const unsigned char* camera_color, // Input: BEV_WIDTH x BEV_HEIGHT x 3
    const float* camera_depth,        // Input: BEV_WIDTH x BEV_HEIGHT
    float* fused_features,            // Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    FusionParams params
);

/**
 * CUDA kernel for weighted feature fusion
 * Combines features using learned or fixed weights
 */
__global__ void weighted_fusion_kernel(
    const float* lidar_occupancy,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const float* lidar_intensity,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const unsigned char* camera_color, // Input: BEV_WIDTH x BEV_HEIGHT x 3
    const float* camera_depth,        // Input: BEV_WIDTH x BEV_HEIGHT
    float* fused_features,            // Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    FusionParams params
);

/**
 * CUDA kernel for attention-based fusion
 * Uses attention mechanism to dynamically weight features
 */
__global__ void attention_fusion_kernel(
    const float* lidar_occupancy,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const float* lidar_intensity,     // Input: BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    const unsigned char* camera_color, // Input: BEV_WIDTH x BEV_HEIGHT x 3
    const float* camera_depth,        // Input: BEV_WIDTH x BEV_HEIGHT
    float* fused_features,            // Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    FusionParams params
);


/**
 * CUDA kernel for feature normalization
 * Normalizes features to [0, 1] range
 */
__global__ void normalize_features_kernel(
    float* features,                  // Input/Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    float min_val, float max_val
);

/**
 * CUDA kernel for feature enhancement
 * Applies spatial filtering and enhancement
 */
__global__ void enhance_features_kernel(
    float* features,                  // Input/Output: BEV_WIDTH x BEV_HEIGHT x feature_dim
    int feature_dim,
    int kernel_size
);

/**
 * Host function to launch feature fusion
 */
cudaError_t fuse_bev_features(
    const BEVGrid& lidar_grid,
    const unsigned char* camera_color_grid,
    const float* camera_depth_grid,
    FusedBEVFeatures& fused_features,
    const FusionParams& params,
    cudaStream_t stream = 0
);

/**
 * Host function to create multi-scale features
 * Creates features at different resolutions
 */
cudaError_t create_multiscale_features(
    const FusedBEVFeatures& base_features,
    FusedBEVFeatures& multiscale_features,
    int num_scales,
    cudaStream_t stream = 0
);

/**
 * Utility functions for feature processing
 */
__device__ __forceinline__ float sigmoid(float x);
__device__ __forceinline__ float softmax(float x, float sum);
__device__ __forceinline__ float compute_attention_weight(
    float lidar_feat, float camera_feat, float depth
);

/**
 * Memory management functions
 */
cudaError_t allocate_fused_features(FusedBEVFeatures& features, int feature_dim);
cudaError_t free_fused_features(FusedBEVFeatures& features);
cudaError_t reset_fused_features(FusedBEVFeatures& features);

/**
 * Feature extraction utilities
 */
cudaError_t extract_lidar_features(
    const BEVGrid& lidar_grid,
    float* features,
    int feature_dim,
    cudaStream_t stream = 0
);

cudaError_t extract_camera_features(
    const unsigned char* camera_color,
    const float* camera_depth,
    float* features,
    int feature_dim,
    cudaStream_t stream = 0
);


#endif // FEATURE_FUSION_CUH
