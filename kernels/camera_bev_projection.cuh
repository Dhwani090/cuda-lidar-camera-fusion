#ifndef CAMERA_BEV_PROJECTION_CUH
#define CAMERA_BEV_PROJECTION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Camera calibration parameters
struct CameraIntrinsics {
    float fx, fy;           // Focal lengths
    float cx, cy;           // Principal point
    float k1, k2, k3;       // Distortion coefficients
    float p1, p2;           // Tangential distortion
};

// Camera extrinsics (transformation from camera to world)
struct CameraExtrinsics {
    float R[9];             // 3x3 rotation matrix (row-major)
    float t[3];             // 3x1 translation vector
};

// Camera parameters
struct CameraParams {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    int image_width;
    int image_height;
};

// BEV grid parameters
struct BEVParams {
    float voxel_size_x;      // Size of each voxel in X direction (meters)
    float voxel_size_y;      // Size of each voxel in Y direction (meters)
    float min_x, max_x;      // Bounding box in X direction
    float min_y, max_y;      // Bounding box in Y direction
    float ground_height;     // Height of ground plane
};

// Color interpolation methods
enum InterpolationMethod {
    NEAREST_NEIGHBOR = 0,
    BILINEAR = 1,
    BICUBIC = 2
};

/**
 * CUDA kernel for camera-to-BEV projection
 * Each thread processes one BEV cell and projects it to camera image
 */
__global__ void camera_to_bev_projection_kernel(
    const unsigned char* camera_image,    // Input: H x W x 3 RGB image
    unsigned char* bev_color_grid,       // Output: BEV_WIDTH x BEV_HEIGHT x 3
    float* bev_depth_grid,               // Output: BEV_WIDTH x BEV_HEIGHT
    CameraParams camera_params,
    BEVParams bev_params,
    InterpolationMethod interp_method
);


/**
 * CUDA kernel for semantic segmentation projection
 * Projects CNN features from camera to BEV space
 */
__global__ void semantic_projection_kernel(
    const float* camera_features,        // Input: H x W x C feature map
    float* bev_feature_grid,            // Output: BEV_WIDTH x BEV_HEIGHT x C
    int feature_channels,
    CameraParams camera_params,
    BEVParams bev_params,
    InterpolationMethod interp_method
);

/**
 * CUDA kernel for depth-aware color blending
 * Uses depth information to blend colors from multiple cameras
 */
__global__ void depth_aware_blending_kernel(
    const unsigned char* camera_images,  // Input: N_cameras x H x W x 3
    const float* camera_depths,          // Input: N_cameras x H x W
    unsigned char* bev_color_grid,       // Output: BEV_WIDTH x BEV_HEIGHT x 3
    CameraParams* camera_params_array,   // Input: N_cameras camera parameters
    int num_cameras,
    BEVParams bev_params
);

/**
 * Host function to launch camera-to-BEV projection
 */
cudaError_t camera_to_bev_projection(
    const unsigned char* camera_image,
    unsigned char* bev_color_grid,
    float* bev_depth_grid,
    const CameraParams& camera_params,
    const BEVParams& bev_params,
    InterpolationMethod interp_method = BILINEAR,
    cudaStream_t stream = 0
);


/**
 * Utility functions for coordinate transformations
 */
__device__ __forceinline__ void world_to_camera(
    float world_x, float world_y, float world_z,
    float& cam_x, float& cam_y, float& cam_z,
    const CameraExtrinsics& extrinsics
);

__device__ __forceinline__ void camera_to_image(
    float cam_x, float cam_y, float cam_z,
    float& img_u, float& img_v,
    const CameraIntrinsics& intrinsics
);

__device__ __forceinline__ void image_to_camera(
    float img_u, float img_v, float depth,
    float& cam_x, float& cam_y, float& cam_z,
    const CameraIntrinsics& intrinsics
);

__device__ __forceinline__ void camera_to_world(
    float cam_x, float cam_y, float cam_z,
    float& world_x, float& world_y, float& world_z,
    const CameraExtrinsics& extrinsics
);

/**
 * Interpolation functions
 */
__device__ __forceinline__ unsigned char bilinear_interpolate(
    const unsigned char* image,
    float u, float v,
    int width, int height, int channels
);

__device__ __forceinline__ float bilinear_interpolate_float(
    const float* image,
    float u, float v,
    int width, int height, int channels
);

/**
 * Memory management functions
 */
cudaError_t allocate_bev_color_grid(unsigned char** grid);
cudaError_t allocate_bev_depth_grid(float** grid);
cudaError_t free_bev_color_grid(unsigned char* grid);
cudaError_t free_bev_depth_grid(float* grid);

#endif // CAMERA_BEV_PROJECTION_CUH
