#include "sampler.h"
#include <thrust/random.h>
// Samples are returned from 0.0 to 1.0, where 0.0 is the same as start_points[0] and
// 1.0 is the same as the last end_point


// this should be launched with one thread per ray in an image
// each thread will read from num_hits
// each thread will compute samples for num_hits points
// start_points and end_points are the same size


__global__ void generate_samples(
    float3* start_points,
    float3* end_points,
    int* num_hits,
    int* indices,
    int num_segments,
    SAMPLING_TYPE sample_type,
    float3* samples,
    thrust::minstd_rand rng) 
{
    // Get index of this segment
    int global_thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    float3 origin = start_points[global_thread_idx];
    float3 finish = end_points[global_thread_idx];
    int start_index = indices[global_thread_idx];
    int end_index = start_index + num_hits[global_thread_idx];
    float3 direction;
    direction.x = finish.x - origin.x;
    direction.y = finish.y - origin.y;
    direction.z = finish.z - origin.z;
    
    float t_initial = 0.0f;
    float t_final = 1.0f / NUM_SAMPLES_PER_SEGMENT;
    for(int j = start_index; j < end_index; j++) {
        #pragma unroll
        for (int i = 0; i < NUM_SAMPLES_PER_SEGMENT; i++) {
            if (sample_type == SAMPLING_REGULAR) {
                float t = t_initial;
                float3 sample = origin;
                sample.x = t * direction.x + origin.x;
                sample.y = t * direction.y + origin.y;
                sample.z = t * direction.z + origin.z;
                samples[global_thread_idx * NUM_SAMPLES_PER_SEGMENT + i + j] = sample;
                t_initial += 1.0f / NUM_SAMPLES_PER_SEGMENT;
            } else if (sample_type == SAMPLING_UNIFORM) {
                thrust::uniform_real_distribution<float> dist(0,1);
                float t = dist(rng);

                float3 sample = origin;
                sample.x = t * direction.x + origin.x;
                sample.y = t * direction.y + origin.y;
                sample.z = t * direction.z + origin.z;
                samples[global_thread_idx * NUM_SAMPLES_PER_SEGMENT +i+j] = sample;
            } else if (sample_type == SAMPLING_STRATIFIED_JITTERING) {
                thrust::uniform_real_distribution<float> dist(t_initial, t_final);
                float t = dist(rng);

                float3 sample = origin;
                sample.x = t * direction.x + origin.x;
                sample.y = t * direction.y + origin.y;
                sample.z = t * direction.z + origin.z;
                samples[global_thread_idx * NUM_SAMPLES_PER_SEGMENT +i+j] = sample;            
    
                t_initial = t_final;
                t_final += 1.0f / NUM_SAMPLES_PER_SEGMENT;
            }
        }
    
    }
}

void launchSampler(
    float3* d_start_points,
    float3* d_end_points,
    float5* d_hit_data,
    int samples_per_intersection, 
    unsigned int width, 
    unsigned int height,
    int num_primitives,
    cudaStream_t& stream,
    int& num_points) {
        
    }