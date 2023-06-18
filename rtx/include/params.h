#include <cuda_runtime.h>
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
    OptixTraversableHandle handle;
    unsigned int grid_x, grid_y, grid_z;
    float t_near;
    float t_far;
    
    
    // store color (R,G,B) and density for each point 
    float4* output_buffer;
    // store coordinates and view direction for each point
    float5* input_points;
};