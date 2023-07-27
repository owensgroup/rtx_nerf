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
    int num_primitives;
    OptixAabb *aabb;
    // start/end points is [W, H, num_boxes_in_grid]
    // W is width of the image
    // H is height of the image
    float3* start_points; // store the start points of each ray
    float3* end_points; // store the end points of each ray
    // store the number of intersections with AABBs per ray (<= num_primitives per ray)
    // we'll use this to gather the intersections for each ray
    int* num_hits;
    float focal_length;
    float aspect_ratio;
    //int total_num_hits; // total number of intersections
    float3 delta;
    float3 min_point;
    float3 max_point;
    unsigned int width, height;
};
