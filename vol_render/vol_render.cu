// A CUDA kernel which given a list of samples per ray and the t-values and densities
// at those sample points, computes a volume rendering of the given rays using those
// samples.
//
// INPUT -- LENGTH
// network_outputs -- NUM_RAY_HITS * NUM_SAMPLES_PER_RAY * 4
// network_inputs -- NUM_RAY_HITS * NUM_SAMPLES_PER_RAY * 5
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
    float* network_outputs,
    float* network_inputs,
    int* num_hits,
    int* indices,
    int width,
    int height,
    int num_samples_per_ray,
    float3* pixels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) {
        return;
    }

    int ray_idx = x + y * width;
    int start_index = indices[ray_idx];
    int num_ray_hits = num_hits[ray_idx];
    for(int j = 0; j < num_ray_hits; j++) {
        float transmittance = 0.0f;
        float t_initial = 0.0f;
        float3 accum_color;
        accum_color.x = 0.0f;
        accum_color.y = 0.0f;
        accum_color.z = 0.0f;
    }
    float transmittance = 0.0f;
    float t_initial = 0.0f;
    float3 accum_color;
    accum_color.x = 0.0f;
    accum_color.y = 0.0f;
    accum_color.z = 0.0f;
    #pragma unroll
    for (int i = 0; i < num_samples_per_ray; i++) {
        float3 color = colors[global_thread_idx * i + i];
        float sigma = densities[global_thread_idx * i + i];

        float t_final = t_samples[global_thread_idx * i + i];
        float delta = t_final - t_initial;
        t_initial = t_final;

        transmittance += delta * sigma;

        color.x = exp(-transmittance) * (1 - exp(-delta * sigma)) * color.x;
        color.y = exp(-transmittance) * (1 - exp(-delta * sigma)) * color.y;
        color.z = exp(-transmittance) * (1 - exp(-delta * sigma)) * color.z;

        accum_color.x += color.x;
        accum_color.y += color.y;
        accum_color.z += color.z;
    }

    pixels[global_thread_idx] = accum_color;
}