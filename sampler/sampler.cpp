#include "sampler.h"

void launchUniformSampler(
    float3* d_start_points,
    float3* d_end_points,
    int* d_num_hits,
    int samples_per_intersection, 
    unsigned int width, 
    unsigned int height,
    int num_primitives,
    cudaStream_t& stream) {
        
    }