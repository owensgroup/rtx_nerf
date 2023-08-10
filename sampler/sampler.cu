#include "sampler.h"
#include <thrust/random.h>
#define BLOCK_SIZE 32
// Samples are returned from 0.0 to 1.0, where 0.0 is the same as start_points[0] and
// 1.0 is the same as the last end_point



// each thread computes the samples for one ray
// this should be launched with one thread per ray in an image
// each thread will read from num_hits
// each thread will compute samples for num_hits points
// start_points and end_points are the same size [width, height, grid_res * 3]

__global__ void generate_samples(
    float3* start_points,
    float3* end_points,
    float2* view_dirs,
    int width,
    int height,
    int grid_res,
    int* num_hits,
    int* indices,
    SAMPLING_TYPE sample_type,
    float* samples,
    thrust::minstd_rand rng) 
{
    // Get index for this ray
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    // get the viewing direction for this ray
    int start_index = indices[y * width + x];
    float2 view_dir = view_dirs[y * width + x];
    float theta = view_dir.x;
    float phi = view_dir.y;

    if(x < width && y < height) {
        for(int j = 0; j < num_hits[y * width + x]; j++) {
            // grab the start and end points for this segment
            // start and end points have size [width, height, grid_res * 3]
            // find the start and end points for this thread
            int idx = (y * width * grid_res * 3) + (x * grid_res * 3) + j;
            float3 origin = start_points[idx];
            float3 finish = end_points[idx];
            float3 direction;
            direction.x = finish.x - origin.x;
            direction.y = finish.y - origin.y;
            direction.z = finish.z - origin.z;
            
            float t_initial = 0.0f;
            float t_final = 1.0f / NUM_SAMPLES_PER_SEGMENT;

            #pragma unroll
            for (int i = 0; i < NUM_SAMPLES_PER_SEGMENT; i++) {
                if (sample_type == SAMPLING_REGULAR) {
                    float t = t_initial;
                    float3 sample = origin;
                    sample.x = t * direction.x + origin.x;
                    sample.y = t * direction.y + origin.y;
                    sample.z = t * direction.z + origin.z;
                    printf("Thread %d, Writing to sample index %d\n", y*width+x, ((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5);
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5] = sample.x;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 1] = sample.y;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 2] = sample.z;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 3] = theta;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 4] = phi;
                    t_initial += 1.0f / NUM_SAMPLES_PER_SEGMENT;
                } else if (sample_type == SAMPLING_UNIFORM) {
                    thrust::uniform_real_distribution<float> dist(0,1);
                    float t = dist(rng);

                    float3 sample = origin;
                    sample.x = t * direction.x + origin.x;
                    sample.y = t * direction.y + origin.y;
                    sample.z = t * direction.z + origin.z;
                    
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5] = sample.x;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 1] = sample.y;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 2] = sample.z;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 3] = theta;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 4] = phi;
                } else if (sample_type == SAMPLING_STRATIFIED_JITTERING) {
                    thrust::uniform_real_distribution<float> dist(t_initial, t_final);
                    float t = dist(rng);

                    float3 sample = origin;
                    sample.x = t * direction.x + origin.x;
                    sample.y = t * direction.y + origin.y;
                    sample.z = t * direction.z + origin.z;

                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5] = sample.x;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 1] = sample.y;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 2] = sample.z;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 3] = theta;
                    samples[((start_index + j) * NUM_SAMPLES_PER_SEGMENT + i) * 5 + 4] = phi;            

                    t_initial = t_final;
                    t_final += 1.0f / NUM_SAMPLES_PER_SEGMENT;
                }
            }
    
        }
    }
}

void launchSampler(
    float3* d_start_points,
    float3* d_end_points,
    float2* d_view_dirs,
    float* d_sampled_points,
    unsigned int width, 
    unsigned int height,
    int grid_res,
    int* d_num_hits,
    int* d_indices,
    SAMPLING_TYPE sample_type, 
    cudaStream_t& stream) {
        thrust::minstd_rand rng;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        generate_samples<<<grid, block,0,stream>>>(
            d_start_points,
            d_end_points,
            d_view_dirs,
            width,
            height,
            grid_res,
            d_num_hits,
            d_indices,
            sample_type,
            d_sampled_points,
            rng
        );
    }