#include <cuda_runtime_api.h>
#include <optix.h>
#include <cuda_fp16.h>

void launch_volrender_cuda(
    float* network_inputs,
    float* network_outputs,
    int* num_hits,
    int* indices,
    float* ray_hit,
    int batch_size,
    int num_samples_per_hit,
    float* pixels);

void launch_volrender_backward_cuda(
    float* loss_values,
    __half* loss_gradients,
    float* sampled_points_radiance,
    float* t_hit,
    int* num_hits,
    int* indices,
    int width,
    int height,
    int num_samples_per_hit,
    __half* radiance_gradients
);