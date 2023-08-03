#include <cuda_runtime_api.h>
#include <optix.h>

#define NUM_SAMPLES_PER_SEGMENT 32
enum SAMPLING_TYPE {
    SAMPLING_REGULAR,
    SAMPLING_STRATIFIED_JITTERING,
    SAMPLING_UNIFORM,
};

struct float5 {
    float x;
    float y;
    float z;
    float phi;
    float psi;
};

void launchUniformSampler(
    float3* d_start_points,
    float3* d_end_points,
    int* d_num_hits,
    float5* d_hit_data,
    int samples_per_intersection, 
    unsigned int width, 
    unsigned int height,
    int num_primitives,
    cudaStream_t& stream,
    int& num_points);