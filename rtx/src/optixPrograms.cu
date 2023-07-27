/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 //include printf
#include <stdio.h>
#include <optix.h>

#include "params.h"
#include <optix.h>

#include <thrust/random.h>

//#define DBG_RAY (26*params.width + 50)
#define DBG_RAY 1

extern "C" static __constant__ Params params;

extern "C" __global__ void __raygen__ray_march() {
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int ray_idx = launch_index.x + launch_index.y * params.width;
  const float3 &delta = params.delta;
  const float3 &min_point = params.min_point;
  const float3 &max_point = params.max_point;

  // generate ray origin
  // TODO: Apply transform matrix to ray origin
  // look_at is the 4x4 transform matrix
  float* look_at = params.look_at;

  
  float u = (2 * (launch_index.x + 0.5) / params.width - 1) * params.aspect_ratio;
  float v = (2 * (launch_index.y + 0.5) / params.height - 1);
  
  float xd, yd, zd;

  
  xd = look_at[0] * u + look_at[1] * v + look_at[2] * -1.0f * params.focal_length;
  yd = look_at[4] * u + look_at[5] * v + look_at[6] * -1.0f * params.focal_length;
  zd = look_at[8] * u + look_at[9] * v + look_at[10] * -1.0f * params.focal_length;
  // normalize ray direction
  float norm = sqrt(xd * xd + yd * yd + zd * zd);
  xd /= norm;
  yd /= norm;
  zd /= norm;
  float3 ray_direction = make_float3(xd, yd, zd);
  float3 ray_origin = make_float3(look_at[3], look_at[7], look_at[11]);
  float xo = ray_origin.x / 10;
  float yo = ray_origin.y / 10;
  float zo = ray_origin.z / 10;
  float tmin = 0.0f;
  float tmax = (max_point.z - min_point.z) + 100.0;
  float ray_time = 0.0f;
  // Visibility mask is used to mask out objects from rays
  // for each part of the scene, a mask is assigned and when
  // the ray intersects a bitwise and is performed 
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  //unsigned int rayFlags = OPTIX_RAY_FLAG_NONE;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;
  unsigned int hit = 1;
  unsigned int o_x = __float_as_uint(xo);
  unsigned int o_y = __float_as_uint(yo);
  unsigned int o_z = __float_as_uint(zo);



  while (hit) {
    xo = __uint_as_float(o_x); 
    yo = __uint_as_float(o_y);
    zo = __uint_as_float(o_z);
    //if (ray_idx == DBG_RAY) {
    //  printf("Tracing ray (%d, %d)\n"
    //      " ray origin: %f, %f, %f\n"
    //      " ray dir: %f, %f, %f\n",
    //      launch_index.x, launch_index.y,
    //      xo, yo, zo,
    //      xd, yd, zd);
    //}
    ray_origin = make_float3(xo, yo, zo);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
               visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
               hit, o_x, o_y, o_z);
  }
}

extern "C" __global__ void __anyhit__ray_march() {
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int ray_idx = launch_index.x + launch_index.y * params.width;
  // For every intersection, we update a scalar value
  //float val = __uint_as_float(optixGetPayload_0());
  //val += 0.2; // can be the scalar value associated with a triangle.
  //optixSetPayload_0(__float_as_uint(val));
  //optixIgnoreIntersection();
  //if (ray_idx ==  DBG_RAY) {
  //  float t = optixGetRayTmax();
  //  printf("ray (%i, %i) AH  t %f\n", launch_index.x, launch_index.y, t);
  //}
}

extern "C" __global__ void __intersection__ray_march() {
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int ray_idx = launch_index.x + launch_index.y * params.width;
  float3 ray_origen = optixGetWorldRayOrigin();
  float3 ray_direction = optixGetWorldRayDirection();
  uint primitiveIndex = optixGetPrimitiveIndex();

  OptixAabb b = params.aabb[primitiveIndex];
  float tmin = -INFINITY;
  float tmax = INFINITY;

  float tx1 = (b.minX - ray_origen.x) / ray_direction.x;
  float tx2 = (b.maxX - ray_origen.x) / ray_direction.x;
  tmin = max(tmin, min(tx1, tx2));
  tmax = min(tmax, max(tx1, tx2));

  float ty1 = (b.minY - ray_origen.y) / ray_direction.y;
  float ty2 = (b.maxY - ray_origen.y) / ray_direction.y;
  tmin = max(tmin, min(ty1, ty2));
  tmax = min(tmax, max(ty1, ty2));

  float tz1 = (b.minZ - ray_origen.z) / ray_direction.z;
  float tz2 = (b.maxZ - ray_origen.z) / ray_direction.z;
  tmin = max(tmin, min(tz1, tz2));
  tmax = min(tmax, max(tz1, tz2));

  //if (ray_idx ==  DBG_RAY) {
  //  printf("ray (%i, %i) IS\n"
  //      "   tmin %f,  tmax %f\n",
  //      launch_index.x, launch_index.y, tmin, tmax);
  //}

  if (tmax > tmin) {
    if (tmin < 0 && tmax > 1e-6)
      tmin = 0;
    optixReportIntersection(tmin, 0);
  }
}

extern "C" __global__ void __miss__ray_march() {
  optixSetPayload_0(0);
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int ray_idx = launch_index.x + launch_index.y * params.width;
  //if (ray_idx ==  DBG_RAY) {
  //  printf("ray (%i, %i) MISS\n", launch_index.x, launch_index.y);
  //}
}

extern "C" __global__ void __closesthit__ray_march() {
  // For every closest hit, grab the entry and exit point for the ray
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int ray_idx = launch_index.x + launch_index.y * params.width;
  // get the optixAabb that we currenty intersected with
  uint primitiveIndex = optixGetPrimitiveIndex();
  //float t_min = optixGetRayTmin();
  float t_hit = optixGetRayTmax(); // t_max returns smallest reported hitT

  // compute the ray entry point from t_min
  float3 ray_direction = optixGetWorldRayDirection();
  float3 ray_origin = optixGetWorldRayOrigin();
  
  float s_x = ray_origin.x + t_hit * ray_direction.x;
  float s_y = ray_origin.y + t_hit * ray_direction.y;
  float s_z = ray_origin.z + t_hit * ray_direction.z;
  float3 start = make_float3(s_x, s_y, s_z);

  // compute the ray exit point from t_max
  OptixAabb aabb = params.aabb[primitiveIndex];
  float plane_x = ray_direction.x < 0 ? aabb.minX : aabb.maxX;
  float plane_y = ray_direction.y < 0 ? aabb.minY : aabb.maxY;
  float plane_z = ray_direction.z < 0 ? aabb.minZ : aabb.maxZ;
  float t_x = (plane_x - ray_origin.x) / ray_direction.x;
  float t_y = (plane_y - ray_origin.y) / ray_direction.y;
  float t_z = (plane_z - ray_origin.z) / ray_direction.z;
  float t_e = min(min(t_x, t_y), t_z);

  float e_x = ray_origin.x + t_e * ray_direction.x;
  float e_y = ray_origin.y + t_e * ray_direction.y;
  float e_z = ray_origin.z + t_e * ray_direction.z;
  //if (ray_idx == DBG_RAY) {
  //  printf("ray (%i, %i) CH\n"
  //      "  tmax: %f\n"
  //      "  entry point: %f, %f, %f\n"
  //      "  exit point: %f, %f, %f\n"
  //      "  p_x %f   p_y %f   p_z %f\n"
  //      "  t: %f   %f   %f\n"
  //      "  num_hits: %i\n",
  //      launch_index.x, launch_index.y,
  //      t_hit,
  //      s_x, s_y, s_z,
  //      e_x, e_y, e_z,
  //      plane_x, plane_y, plane_z,
  //      t_x, t_y, t_z,
  //      params.num_hits[ray_idx]);
  //}
  float3 end = make_float3(e_x, e_y, e_z);


  // store the entry and exit points of this ray in this AABB in param buffers
  // entry_points is an array with dimension [H, W, numPrimitives]
  // exit_points is an array with dimension [H, W, numPrimitives]
  unsigned int idx = params.num_hits[ray_idx] + ray_idx * params.num_primitives;
  params.start_points[idx] = start;
  params.end_points[idx] = end;

  // update the number of intersections for this ray
  params.num_hits[ray_idx] += 1;

  // Set payload to exit points
  // this will be used in raygen to launch a new ray and find the next AABB
  optixSetPayload_1(__float_as_uint(e_x));
  optixSetPayload_2(__float_as_uint(e_y));
  optixSetPayload_3(__float_as_uint(e_z));
}

extern "C" __global__ void __raygen__ray_sample() {
  const uint3 launch_index = optixGetLaunchIndex();
  const float3 &delta = params.delta;
  const float3 &min_point = params.min_point;
  //const float* transform_matrix = params.transform_matrix;
  float xo = min_point.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = min_point.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = -1 * (min_point.z + delta.z * launch_index.z);
  float3 ray_origin = make_float3(xo, yo, zo);

  float3 ray_direction = make_float3(0.0, 0.0, 1.0);

  float tmin = 0.0f;
  float tmax = delta.z + 1;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;

  // Save exit points of each AABB in optixPayload (setoptixpayload)
  // relaunch optix trace replace ray origins with exit points from optix payload
  // may need another payload to know when to stop launching rays
  
  // tmax should be 
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
             visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex);
}

// extern "C" __global__ void __anyhit__ray_sample() {}
// extern "C" __global__ void __miss__ray_sample() {}

extern "C" __global__ void __closesthit__ray_sample() {
  const uint3 launch_index = optixGetLaunchIndex();
  // For every closest hit, we update a scalar value in Global memory
  unsigned int idx = launch_index.x + launch_index.y * params.width;
  //float *output = params.output;

  // get t max value of ray for a hit
  // float t_current = optixGetRayTmax();
  // float3 ray_direction = optixGetWorldRayDirection();
  // float3 ray_origin = optixGetWorldRayOrigin();
  // float3 hit_point = ray_origin + t_current * ray_direction;

  // from the hitpoint march through till the exit point of the current AABB
  // store the sampled points in a buffer in Params
  // determine start point and end point
  // compute the end point given bounding box coordinates
  
  // compute the start and end points
  // store them in a buffer in Params
  // get the optixAabb that we currenty intersected with
  // use the bounding box coordinates in order to compute the end point
  // store the start and end points of this ray in this AABB in param buffers
  // uint primitiveIndex = optixGetPrimitiveIndex();
  // OptixAabb aabb = Params.aabb[primitiveIndex];
  // float3 min_point = make_float3(aabb.minX, aabb.minY, aabb.minZ);
  // float3 max_point = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

  // compute the end point
  // this should store sampled points in a buffer in Params
  // atomicAdd(output + idx,
  //           0.2f); // can be the scalar value associated with a triangle.
}

// Samples are returned from 0.0 to 1.0, where 0.0 is the same as start_points[0] and
// 1.0 is the same as the last end_point


#define NUM_SAMPLES_PER_SEGMENT 32
enum SAMPLING_TYPE {
    SAMPLING_REGULAR,
    SAMPLING_STRATIFIED_JITTERING,
    SAMPLING_UNIFORM,
};
extern "C" __global__ void generate_samples(float3* start_points, float3* end_points,
                                            int num_segments,
                                            SAMPLING_TYPE sample_type,
                                            float3* samples,
                                            thrust::minstd_rand rng) 
{
    // Get index of this segment
    int global_thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    float3 origin = start_points[global_thread_idx];
    float3 finish = end_points[global_thread_idx];
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
            samples[global_thread_idx * NUM_SAMPLES_PER_SEGMENT + i] = sample;
            t_initial += 1.0f / NUM_SAMPLES_PER_SEGMENT;
        } else if (sample_type == SAMPLING_UNIFORM) {
            thrust::uniform_real_distribution<float> dist(0,1);
            float t = dist(rng);

            float3 sample = origin;
            sample.x = t * direction.x + origin.x;
            sample.y = t * direction.y + origin.y;
            sample.z = t * direction.z + origin.z;
            samples[global_thread_idx * NUM_SAMPLES_PER_SEGMENT +i] = sample;
        } else if (sample_type == SAMPLING_STRATIFIED_JITTERING) {
            thrust::uniform_real_distribution<float> dist(t_initial, t_final);
            float t = dist(rng);

            float3 sample = origin;
            sample.x = t * direction.x + origin.x;
            sample.y = t * direction.y + origin.y;
            sample.z = t * direction.z + origin.z;
            samples[global_thread_idx * NUM_SAMPLES_PER_SEGMENT +i] = sample;            
 
            t_initial = t_final;
            t_final += 1.0f / NUM_SAMPLES_PER_SEGMENT;
        }
    }
}

// A CUDA kernel which given a list of samples per ray and the t-values and densities
// at those sample points, computes a volume rendering of the given rays using those
// samples.
//
// INPUT -- LENGTH
// NUM_RAYS -- numBlocks * threadsPerBlock
// NUM_SAMPLES_PER_RAY -- 32
// COLORS -- NUM_RAYS * NUM_SAMPLES_PER_RAY
// T_SAMPLES -- NUM_RAYS * NUM_SAMPLES_PER_RAY
// DENSITIES -- NUM_RAYS * NUM_SAMPLES_PER_RAY
//
// OUTPUT -- SIZE
// PIXELS -- NUM_RAYS
//
//
extern "C" __global__ void volrender_cuda(float3* colors, float* t_samples,
                                          int num_samples_per_ray,
                                          float* densities,
                                          float3* pixels) {
    int global_thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

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