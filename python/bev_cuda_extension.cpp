#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

#include "lidar_voxelization.cuh"
#include "camera_bev_projection.cuh"
#include "feature_fusion.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int BEV_WIDTH = 200;
constexpr int BEV_HEIGHT = 200;
constexpr int BEV_DEPTH = 32;


torch::Tensor lidar_voxelization_torch(
    torch::Tensor points, float voxel_size = 0.1f, float max_range = 50.0f, bool return_intensity = true, bool return_height = true
) {
    CHECK_INPUT(points);
    TORCH_CHECK(points.size(1) == 4, "Points must have 4 dimensions (x, y, z, intensity)");
    
    int num_points = points.size(0);
    auto device = points.device();

    VoxelizationParams params;
    init_voxelization_params(params, voxel_size, max_range);
    
    BEVGrid bev_grid;
    cudaError_t err = allocate_bev_grid(bev_grid);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate BEV grid memory");
    }
    
    err = reset_bev_grid(bev_grid);
    if (err != cudaSuccess) {
        free_bev_grid(bev_grid);
        throw std::runtime_error("Failed to reset BEV grid");
    }
    
    PointCloud point_cloud;
    point_cloud.points = points.data_ptr<float>();
    point_cloud.num_points = num_points;
    
    err = lidar_voxelization(point_cloud, bev_grid, params);
    if (err != cudaSuccess) {
        free_bev_grid(bev_grid);
        throw std::runtime_error("LiDAR voxelization kernel failed");
    }

    std::vector<torch::Tensor> outputs;
    
    auto occupancy_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto occupancy = torch::from_blob(bev_grid.occupancy, {BEV_WIDTH, BEV_HEIGHT, BEV_DEPTH}, occupancy_options).clone();
    outputs.push_back(occupancy);
    
    if (return_intensity) {
        auto intensity = torch::from_blob(bev_grid.intensity, {BEV_WIDTH, BEV_HEIGHT, BEV_DEPTH}, occupancy_options).clone();
        outputs.push_back(intensity);
    }
    
    if (return_height) {
        auto max_height = torch::from_blob(bev_grid.max_height, {BEV_WIDTH, BEV_HEIGHT}, occupancy_options).clone();
        outputs.push_back(max_height);
    }
    
    free_bev_grid(bev_grid);
    
    if (outputs.size() == 1) {
        return outputs[0];
    } else {
        return torch::stack(outputs);
    }
}

/**
 * PyTorch wrapper for camera-to-BEV projection
 */
torch::Tensor camera_to_bev_projection_torch(
    torch::Tensor camera_image,     // H x W x 3 RGB image
    torch::Tensor camera_intrinsics, // 3 x 3 intrinsic matrix
    torch::Tensor camera_extrinsics, // 4 x 4 extrinsic matrix
    float voxel_size = 0.1f,
    float max_range = 50.0f,
    int interpolation_method = 1    // 0=nearest, 1=bilinear
) {
    CHECK_INPUT(camera_image);
    CHECK_INPUT(camera_intrinsics);
    CHECK_INPUT(camera_extrinsics);
    
    TORCH_CHECK(camera_image.size(2) == 3, "Camera image must have 3 channels (RGB)");
    TORCH_CHECK(camera_intrinsics.size(0) == 3 && camera_intrinsics.size(1) == 3, 
                "Camera intrinsics must be 3x3 matrix");
    TORCH_CHECK(camera_extrinsics.size(0) == 4 && camera_extrinsics.size(1) == 4, 
                "Camera extrinsics must be 4x4 matrix");
    
    auto device = camera_image.device();
    int image_height = camera_image.size(0);
    int image_width = camera_image.size(1);
    
    CameraParams camera_params;
    camera_params.image_width = image_width;
    camera_params.image_height = image_height;
    
    auto intrinsics_data = camera_intrinsics.data_ptr<float>();
    camera_params.intrinsics.fx = intrinsics_data[0];
    camera_params.intrinsics.fy = intrinsics_data[4];
    camera_params.intrinsics.cx = intrinsics_data[2];
    camera_params.intrinsics.cy = intrinsics_data[5];
    camera_params.intrinsics.k1 = 0.0f; 
    camera_params.intrinsics.k2 = 0.0f;
    camera_params.intrinsics.k3 = 0.0f;
    camera_params.intrinsics.p1 = 0.0f;
    camera_params.intrinsics.p2 = 0.0f;
    
    auto extrinsics_data = camera_extrinsics.data_ptr<float>();
    for (int i = 0; i < 9; i++) {
        camera_params.extrinsics.R[i] = extrinsics_data[i];
    }
    camera_params.extrinsics.t[0] = extrinsics_data[3];
    camera_params.extrinsics.t[1] = extrinsics_data[7];
    camera_params.extrinsics.t[2] = extrinsics_data[11];
    
    BEVParams bev_params;
    bev_params.voxel_size_x = voxel_size;
    bev_params.voxel_size_y = voxel_size;
    bev_params.min_x = -max_range / 2.0f;
    bev_params.max_x = max_range / 2.0f;
    bev_params.min_y = -max_range / 2.0f;
    bev_params.max_y = max_range / 2.0f;
    bev_params.ground_height = 0.0f;
    
    unsigned char* bev_color_grid;
    float* bev_depth_grid;
    
    cudaError_t err = allocate_bev_color_grid(&bev_color_grid);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate BEV color grid");
    }
    
    err = allocate_bev_depth_grid(&bev_depth_grid);
    if (err != cudaSuccess) {
        free_bev_color_grid(bev_color_grid);
        throw std::runtime_error("Failed to allocate BEV depth grid");
    }
    
    InterpolationMethod interp_method = (interpolation_method == 0) ? NEAREST_NEIGHBOR : BILINEAR;
    err = camera_to_bev_projection(
        camera_image.data_ptr<unsigned char>(),
        bev_color_grid,
        bev_depth_grid,
        camera_params,
        bev_params,
        interp_method
    );
    
    if (err != cudaSuccess) {
        free_bev_color_grid(bev_color_grid);
        free_bev_depth_grid(bev_depth_grid);
        throw std::runtime_error("Camera-to-BEV projection failed");
    }
    

    auto color_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    auto depth_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto bev_color = torch::from_blob(bev_color_grid, {BEV_WIDTH, BEV_HEIGHT, 3}, color_options).clone();
    auto bev_depth = torch::from_blob(bev_depth_grid, {BEV_WIDTH, BEV_HEIGHT}, depth_options).clone();
    
    free_bev_color_grid(bev_color_grid);
    free_bev_depth_grid(bev_depth_grid);
    
    return torch::stack({bev_color.to(torch::kFloat32), bev_depth});
}

/**
 * PyTorch wrapper for feature fusion
 */
torch::Tensor fuse_bev_features_torch(
    torch::Tensor lidar_occupancy,     // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    torch::Tensor lidar_intensity,      // BEV_WIDTH x BEV_HEIGHT x BEV_DEPTH
    torch::Tensor camera_color,        // BEV_WIDTH x BEV_HEIGHT x 3
    torch::Tensor camera_depth,        // BEV_WIDTH x BEV_HEIGHT
    int fusion_method = 1,              // 0=simple, 1=weighted, 2=attention
    float lidar_weight = 0.5f,
    float camera_weight = 0.5f,
    bool normalize_features = true
) {
    CHECK_INPUT(lidar_occupancy);
    CHECK_INPUT(lidar_intensity);
    CHECK_INPUT(camera_color);
    CHECK_INPUT(camera_depth);
    
    auto device = lidar_occupancy.device();
    
    FusionParams params;
    params.method = static_cast<FusionMethod>(fusion_method);
    params.lidar_weight = lidar_weight;
    params.camera_weight = camera_weight;
    params.occupancy_threshold = 0.1f;
    params.intensity_threshold = 0.1f;
    params.normalize_features = normalize_features;
    params.use_depth_guidance = true;
    

    int feature_dim = BEV_DEPTH * 2 + 4; // occupancy + intensity + color + depth
    
    FusedBEVFeatures fused_features;
    cudaError_t err = allocate_fused_features(fused_features, feature_dim);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate fused features");
    }
    
    BEVGrid lidar_grid;
    lidar_grid.occupancy = lidar_occupancy.data_ptr<float>();
    lidar_grid.intensity = lidar_intensity.data_ptr<float>();
    lidar_grid.max_height = nullptr; 
    lidar_grid.point_count = nullptr; 
    
    err = fuse_bev_features(
        lidar_grid,
        camera_color.data_ptr<unsigned char>(),
        camera_depth.data_ptr<float>(),
        fused_features,
        params
    );
    
    if (err != cudaSuccess) {
        free_fused_features(fused_features);
        throw std::runtime_error("Feature fusion failed");
    }
    
    auto output_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto output = torch::from_blob(fused_features.fused_features, 
                                   {BEV_WIDTH, BEV_HEIGHT, feature_dim}, 
                                   output_options).clone();
    
    free_fused_features(fused_features);
    
    return output;
}

/**
 * Complete BEV transformation pipeline
 */
torch::Tensor bev_transformation_pipeline(
    torch::Tensor lidar_points,        // N x 4 (x, y, z, intensity)
    torch::Tensor camera_image,        // H x W x 3 RGB image
    torch::Tensor camera_intrinsics,   // 3 x 3 intrinsic matrix
    torch::Tensor camera_extrinsics,   // 4 x 4 extrinsic matrix
    float voxel_size = 0.1f,
    float max_range = 50.0f,
    int fusion_method = 2,              // Use attention fusion by default
    float lidar_weight = 0.6f,
    float camera_weight = 0.4f
) {
    auto lidar_outputs = lidar_voxelization_torch(lidar_points, voxel_size, max_range, true, true);
    
    torch::Tensor lidar_occupancy, lidar_intensity, lidar_height;
    if (lidar_outputs.dim() == 3) {
        lidar_occupancy = lidar_outputs;
        lidar_intensity = torch::zeros_like(lidar_occupancy);
        lidar_height = torch::zeros({BEV_WIDTH, BEV_HEIGHT}, lidar_occupancy.options());
    } else {
        lidar_occupancy = lidar_outputs[0];
        lidar_intensity = lidar_outputs[1];
        lidar_height = lidar_outputs[2];
    }
    
    auto camera_outputs = camera_to_bev_projection_torch(
        camera_image, camera_intrinsics, camera_extrinsics, voxel_size, max_range
    );
    
    auto camera_color = camera_outputs[0];
    auto camera_depth = camera_outputs[1];
    
    auto fused_features = fuse_bev_features_torch(
        lidar_occupancy, lidar_intensity, camera_color, camera_depth,
        fusion_method, lidar_weight, camera_weight
    );
    
    return fused_features;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lidar_voxelization", &lidar_voxelization_torch, "LiDAR point cloud voxelization");
    m.def("camera_to_bev_projection", &camera_to_bev_projection_torch, "Camera to BEV projection");
    m.def("fuse_bev_features", &fuse_bev_features_torch, "Fuse LiDAR and camera features");
    m.def("bev_transformation_pipeline", &bev_transformation_pipeline, "Complete BEV transformation pipeline");
}
