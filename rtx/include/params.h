#include <cuda_runtime_api.h>
#include <optix.h>

#pragma once
struct RayGenData {
};

struct HitGroupData {
};

struct MissData {
};

struct Params {
    // Handle to the initialized acceleration structure
    OptixTraversableHandle handle;
    float* look_at; // 4x4 matrix

    // start/end points is [W, H, num_boxes_in_grid]
    // W is width of the image
    // H is height of the image
    float3* start_points; // store the start points of each ray
    float3* end_points; // store the end points of each ray
    int* num_hits; // store the number of intersections with AABBs per ray (<= num_primitives per ray)
    float3 delta;
    float3 min_point;
    float3 max_point;
    unsigned int width, height;
};