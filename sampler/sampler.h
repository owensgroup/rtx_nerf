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
    float theta;
    float phi;
};

void launchSampler(
    float3* d_start_points,
    float3* d_end_points,
    float2* d_view_dirs,
    float5* d_sampled_points,
    unsigned int width, 
    unsigned int height,
    int grid_res,
    int* d_num_hits,
    int* d_indices,
    SAMPLING_TYPE sample_type, 
    cudaStream_t& stream);