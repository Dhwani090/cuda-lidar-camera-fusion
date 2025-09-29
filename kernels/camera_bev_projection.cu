#include "camera_bev_projection.cuh"
#include <math.h>
#include <stdio.h>

__device__ __forceinline__ bool is_image_coord_valid(float u, float v, int width, int height) {
    return (u >= 0.0f && u < width && v >= 0.0f && v < height);
}

__device__ __forceinline__ int get_image_index(int u, int v, int width, int channels) {
    return (v * width + u) * channels;
}

// Helper function to get linear index for BEV grid
__device__ __forceinline__ int get_bev_index(int x, int y, int channels) {
    return (y * BEV_WIDTH + x) * channels;
}

//World coordinates to camera coordinates
__device__ __forceinline__ void world_to_camera(
    float world_x, float world_y, float world_z,
    float& cam_x, float& cam_y, float& cam_z,
    const CameraExtrinsics& extrinsics
) {
    // Apply rotation and translation: cam = R * world + t
    cam_x = extrinsics.R[0] * world_x + extrinsics.R[1] * world_y + extrinsics.R[2] * world_z + extrinsics.t[0];
    cam_y = extrinsics.R[3] * world_x + extrinsics.R[4] * world_y + extrinsics.R[5] * world_z + extrinsics.t[1];
    cam_z = extrinsics.R[6] * world_x + extrinsics.R[7] * world_y + extrinsics.R[8] * world_z + extrinsics.t[2];
}

//Camera coords to Image coords
__device__ __forceinline__ void camera_to_image(
    float cam_x, float cam_y, float cam_z,
    float& img_u, float& img_v,
    const CameraIntrinsics& intrinsics
) {
    // Project to image plane: u = fx * x/z + cx, v = fy * y/z + cy
    img_u = intrinsics.fx * cam_x / cam_z + intrinsics.cx;
    img_v = intrinsics.fy * cam_y / cam_z + intrinsics.cy;
}

//Image coords to camera coords
__device__ __forceinline__ void image_to_camera(
    float img_u, float img_v, float depth,
    float& cam_x, float& cam_y, float& cam_z,
    const CameraIntrinsics& intrinsics
) {
    // Back-project from image plane: x = (u - cx) * z / fx, y = (v - cy) * z / fy
    cam_x = (img_u - intrinsics.cx) * depth / intrinsics.fx;
    cam_y = (img_v - intrinsics.cy) * depth / intrinsics.fy;
    cam_z = depth;
}

//Camera to world coords
__device__ __forceinline__ void camera_to_world(
    float cam_x, float cam_y, float cam_z,
    float& world_x, float& world_y, float& world_z,
    const CameraExtrinsics& extrinsics
) {
    // Apply inverse transformation: world = R^T * (cam - t)
    float temp_x = cam_x - extrinsics.t[0];
    float temp_y = cam_y - extrinsics.t[1];
    float temp_z = cam_z - extrinsics.t[2];
    
    world_x = extrinsics.R[0] * temp_x + extrinsics.R[3] * temp_y + extrinsics.R[6] * temp_z;
    world_y = extrinsics.R[1] * temp_x + extrinsics.R[4] * temp_y + extrinsics.R[7] * temp_z;
    world_z = extrinsics.R[2] * temp_x + extrinsics.R[5] * temp_y + extrinsics.R[8] * temp_z;
}

//Bilinear interpolation for color values
//Bilinear interpolation takes weighted average of 4 nearest pixels based on distance from the target pixel
__device__ __forceinline__ unsigned char bilinear_interpolate(
    const unsigned char* image,
    float u, float v,
    int width, int height, int channels
) {
    int u0 = (int)u;
    int v0 = (int)v;

    u0 = max(0, min(u0, width - 1));
    v0 = max(0, min(v0, height - 1));
    int u1 = max(0, min(u0+1, width - 1));
    int v1 = max(0, min(v0+1, height - 1));
    
    float du = u - u0;
    float dv = v - v0;
    
    unsigned char p00 = image[get_image_index(u0, v0, width, channels)];
    unsigned char p01 = image[get_image_index(u0, v1, width, channels)];
    unsigned char p10 = image[get_image_index(u1, v0, width, channels)];
    unsigned char p11 = image[get_image_index(u1, v1, width, channels)];
    
    float result = (1.0f - du) * (1.0f - dv) * p00 +
                   (1.0f - du) * dv * p01 +
                   du * (1.0f - dv) * p10 +
                   du * dv * p11;
    
    return (unsigned char)result;
}

//Bilinear interpolation for float values
__device__ __forceinline__ float bilinear_interpolate_float(
    const float* image,
    float u, float v,
    int width, int height, int channels
) {
    int u0 = (int)u;
    int v0 = (int)v;
    
    u0 = max(0, min(u0, width - 1));
    v0 = max(0, min(v0, height - 1));
    int u1 = max(0, min(u0+1, width - 1));
    int v1 = max(0, min(v0+1, height - 1));
    
    float du = u - u0;
    float dv = v - v0;
    
    float p00 = image[get_image_index(u0, v0, width, channels)];
    float p01 = image[get_image_index(u0, v1, width, channels)];
    float p10 = image[get_image_index(u1, v0, width, channels)];
    float p11 = image[get_image_index(u1, v1, width, channels)];
    
    return (1.0f - du) * (1.0f - dv) * p00 +
           (1.0f - du) * dv * p01 +
           du * (1.0f - dv) * p10 +
           du * dv * p11;
}


//Camera to BEV grid projection
__global__ void camera_to_bev_projection_kernel(
    const unsigned char* camera_image,    // Input: H x W x 3 RGB image
    unsigned char* bev_color_grid,       // Output: BEV_WIDTH x BEV_HEIGHT x 3
    float* bev_depth_grid,               // Output: BEV_WIDTH x BEV_HEIGHT
    CameraParams camera_params,
    BEVParams bev_params,
    InterpolationMethod interp_method
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= BEV_WIDTH || y >= BEV_HEIGHT) return;
    
    float world_x = bev_params.min_x + x * bev_params.voxel_size_x;
    float world_y = bev_params.min_y + y * bev_params.voxel_size_y;
    float world_z = bev_params.ground_height;

    float cam_x, cam_y, cam_z;
    world_to_camera(world_x, world_y, world_z, cam_x, cam_y, cam_z, camera_params.extrinsics);
    if (cam_z <= 0.0f) return;
    
    float img_u, img_v;
    camera_to_image(cam_x, cam_y, cam_z, img_u, img_v, camera_params.intrinsics);
    
    if (!is_image_coord_valid(img_u, img_v, camera_params.image_width, camera_params.image_height)) return;
        
    int bev_idx = get_bev_index(x, y, 3);
    int depth_idx = y * BEV_WIDTH + x;
    
    if (interp_method == BILINEAR) {
        bev_color_grid[bev_idx + 0] = bilinear_interpolate(camera_image, img_u, img_v, 
                                                           camera_params.image_width, 
                                                           camera_params.image_height, 3);
        bev_color_grid[bev_idx + 1] = bilinear_interpolate(camera_image, img_u, img_v, 
                                                           camera_params.image_width, 
                                                           camera_params.image_height, 3);
        bev_color_grid[bev_idx + 2] = bilinear_interpolate(camera_image, img_u, img_v, 
                                                           camera_params.image_width, 
                                                           camera_params.image_height, 3);
    } else {
        int u = (int)(img_u + 0.5f);
        int v = (int)(img_v + 0.5f);
        u = max(0, min(u, camera_params.image_width - 1));
        v = max(0, min(v, camera_params.image_height - 1));
        
        int img_idx = get_image_index(u, v, camera_params.image_width, 3);
        bev_color_grid[bev_idx + 0] = camera_image[img_idx + 0];
        bev_color_grid[bev_idx + 1] = camera_image[img_idx + 1];
        bev_color_grid[bev_idx + 2] = camera_image[img_idx + 2];
    }
    
    bev_depth_grid[depth_idx] = cam_z;
}


/**
 * Host function to launch camera-to-BEV projection
 */
cudaError_t camera_to_bev_projection(
    const unsigned char* camera_image,
    unsigned char* bev_color_grid,
    float* bev_depth_grid,
    const CameraParams& camera_params,
    const BEVParams& bev_params,
    InterpolationMethod interp_method,
    cudaStream_t stream
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (BEV_WIDTH + block_size.x - 1) / block_size.x,
        (BEV_HEIGHT + block_size.y - 1) / block_size.y
    );
    
    camera_to_bev_projection_kernel<<<grid_size, block_size, 0, stream>>>(
        camera_image,
        bev_color_grid,
        bev_depth_grid,
        camera_params,
        bev_params,
        interp_method
    );
    
    return cudaGetLastError();
}

//Memory management
cudaError_t allocate_bev_color_grid(unsigned char** grid) {
    size_t size = BEV_WIDTH * BEV_HEIGHT * 3 * sizeof(unsigned char);
    return cudaMalloc(grid, size);
}

cudaError_t allocate_bev_depth_grid(float** grid) {
    size_t size = BEV_WIDTH * BEV_HEIGHT * sizeof(float);
    return cudaMalloc(grid, size);
}

cudaError_t free_bev_color_grid(unsigned char* grid) {
    return cudaFree(grid);
}

cudaError_t free_bev_depth_grid(float* grid) {
    return cudaFree(grid);
}
