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
    int batch_size,
    int num_samples_per_hit,
    float* pixels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(x >= batch_size) {
        return;
    }

    int ray_idx = x;
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

__global__ void volrender_backward_cuda(
    float* loss_values,
    __half* loss_gradients,
    float* sampled_points_radiance,
    float* t_hit,
    int* num_hits,
    int* indices,
    int batch_size,
    int num_samples_per_hit,
    __half* radiance_gradients
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(x >= batch_size) {
        return;
    }

    int ray_idx = x;
    int start_index = indices[ray_idx];
    int num_ray_hits = num_hits[ray_idx];
    float transmittance = 0.0f;

    float t_initial = 0.0f;
    float t = 0.0f;

    for(int j = 0; j < num_ray_hits; j++) {
        #pragma unroll
        for(int i = 0; i < num_samples_per_hit; i++) {
            float3 color;
            float sigma;
            float3 loss_gradient;
            float density_gradient = 0.0f;
            float3 color_gradient;
            int radiance_idx = ((start_index + j) * num_samples_per_hit + i) * 4;
            color.x = sampled_points_radiance[radiance_idx];
            color.y = sampled_points_radiance[radiance_idx + 1];
            color.z = sampled_points_radiance[radiance_idx + 2];
            sigma = sampled_points_radiance[radiance_idx + 3];

            t = t_hit[(start_index + j) * num_samples_per_hit + i];
            float delta = abs(t - t_initial); // FIXME T should not be zero after the first hit
            t_initial = t;

            transmittance = delta * sigma;


            loss_gradient.x = __half2float(loss_gradients[ray_idx * 3]);
            loss_gradient.y = __half2float(loss_gradients[ray_idx * 3 + 1]);
            loss_gradient.z = __half2float(loss_gradients[ray_idx * 3 + 2]);
            
            // compute gradient for density
            // we compute the gradient for the density for each color channel and sum them
            density_gradient += loss_gradient.x * transmittance * color.x * delta * exp(-sigma * delta);
            density_gradient += loss_gradient.y * transmittance * color.y * delta * exp(-sigma * delta);
            density_gradient += loss_gradient.z * transmittance * color.z * delta * exp(-sigma * delta);
            

            // compute gradient for color
            color_gradient.x = loss_gradient.x * transmittance * (1 - exp(-delta * sigma));
            color_gradient.y = loss_gradient.y * transmittance * (1 - exp(-delta * sigma));
            color_gradient.z = loss_gradient.z * transmittance * (1 - exp(-delta * sigma));
            radiance_gradients[radiance_idx] = __float2half(color_gradient.x);
            radiance_gradients[radiance_idx + 1] = __float2half(color_gradient.y);
            radiance_gradients[radiance_idx + 2] = __float2half(color_gradient.z);
            radiance_gradients[radiance_idx + 3] = __float2half(density_gradient);
        }
    }

}
void launch_volrender_cuda(
    float* network_inputs,
    float* network_outputs,
    int* num_hits,
    int* indices,
    float* ray_hit,
    int batch_size,
    int num_samples_per_hit,
    float* pixels) {
        int block_size = 1024;
        int num_blocks = (batch_size + block_size - 1) / block_size;
        volrender_cuda<<<num_blocks, block_size>>>(
            network_inputs,
            network_outputs,
            num_hits,
            indices,
            ray_hit,
            batch_size,
            num_samples_per_hit,
            pixels);
}

void launch_volrender_backward_cuda(
    float* loss_values,
    __half* loss_gradients,
    float* sampled_points_radiance,
    float* t_hit,
    int* num_hits,
    int* indices,
    int batch_size,
    int num_samples_per_hit,
    __half* radiance_gradients
) {
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    volrender_backward_cuda<<<numBlocks, threadsPerBlock>>>(
        loss_values,
        loss_gradients,
        sampled_points_radiance,
        t_hit,
        num_hits,
        indices,
        batch_size,
        num_samples_per_hit,
        radiance_gradients
    );
}