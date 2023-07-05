#include <cuda_runtime_api.h>
#include <optix.h>

#pragma once
struct RayGenData {
};

struct HitGroupData {
};

struct MissData {
};

struct float5 {
    float x, y, z, phi, psi;
};

struct Params {
    // Handle to the initialized acceleration structure
    OptixTraversableHandle handle;
    
    OptixAabb* grid;
    float t_near;
    float t_far;
    
    float* transform_matrix; // 4x4 matrix
    // store color (R,G,B) and density for each point 
    float* output;
    // store coordinates and view direction for each point
    float5* input_points;

    // start points is [W, H, num_boxes_in_grid]
    // same with end points
    float3* start_points;
    float3* end_points;
    float3 delta;
    float3 min_point;
    float3 max_point;
    unsigned int width, height, depth;
};