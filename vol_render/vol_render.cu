#include "vol_render.h"
// A CUDA kernel which given a list of samples per ray and the t-values and densities
// at those sample points, computes a volume rendering of the given rays using those
// samples.
//
// INPUT -- LENGTH
// network_outputs -- NUM_RAY_HITS * NUM_SAMPLES_PER_RAY * 4
// network_inputs -- NUM_RAY_HITS * NUM_SAMPLES_PER_RAY * 5 (x,y,z,theta,phi)
// NUM_RAYS -- numBlocks * threadsPerBlock
// NUM_SAMPLES_PER_RAY -- 32
// 
// T_SAMPLES -- NUM_RAYS * NUM_SAMPLES_PER_RAY
// 
//
// OUTPUT -- SIZE
// PIXELS -- NUM_RAYS
//
//
__global__ void volrender_cuda(
    float* network_inputs,
    float* network_outputs,
    int* num_hits,
    int* indices,
    float* ray_hit,
    int width,
    int height,
    int num_samples_per_hit,
    float* pixels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) {
        return;
    }

    int ray_idx = x + y * width;
    int start_index = indices[ray_idx];
    int num_ray_hits = num_hits[ray_idx];
    float transmittance = 0.0f;
    float t_initial = 0.0f;
    float t = 0.0f;
    float3 accum_color;
    accum_color.x = 0.0f;
    accum_color.y = 0.0f;
    accum_color.z = 0.0f;
    for(int j = 0; j < num_ray_hits; j++) {
        #pragma unroll
        for(int i = 0; i < num_samples_per_hit; i++) {
            float3 color;
            float sigma;

            color.x = network_outputs[(start_index + j) * num_samples_per_hit * 4 + i * 4];
            color.y = network_outputs[(start_index + j) * num_samples_per_hit * 4 + i * 4 + 1];
            color.z = network_outputs[(start_index + j) * num_samples_per_hit * 4 + i * 4 + 2];
            sigma = network_outputs[(start_index + j) * num_samples_per_hit * 4 + i * 4 + 3];
            
            t = ray_hit[(start_index + j) * num_samples_per_hit + i];
            float delta = abs(t - t_initial); // FIXME T should not be zero after the first hit
            t_initial = t;
            

            transmittance += delta * sigma;
            color.x = exp(-transmittance) * (1 - exp(-delta * sigma)) * color.x;
            color.y = exp(-transmittance) * (1 - exp(-delta * sigma)) * color.y;
            color.z = exp(-transmittance) * (1 - exp(-delta * sigma)) * color.z;

            accum_color.x += color.x;
            accum_color.y += color.y;
            accum_color.z += color.z;
        }
    }
    pixels[ray_idx*3] = accum_color.x;
    pixels[ray_idx*3+1] = accum_color.y;
    pixels[ray_idx*3+2] = accum_color.z;
}

void launch_volrender_cuda(
    float* network_inputs,
    float* network_outputs,
    int* num_hits,
    int* indices,
    float* ray_hit,
    int width,
    int height,
    int num_samples_per_hit,
    float* pixels) {
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    volrender_cuda<<<numBlocks, threadsPerBlock>>>(
        network_inputs,
        network_outputs,
        num_hits,
        indices,
        ray_hit,
        width,
        height,
        num_samples_per_hit,
        pixels);
}