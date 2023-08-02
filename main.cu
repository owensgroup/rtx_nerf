#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <math.h>
#include "stdio.h"
#include "sampler.h"

#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "optix.h"
#include "optix_types.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "tiny-cuda-nn/common.h"
#include "tiny-cuda-nn/gpu_matrix.h"
#include <json/json.h>
#include "rtx/include/params.h"
#include "rtx/include/rtxFunctions.h"

#include "data_loader.h"

// Configure the model
nlohmann::json config = {
	{"loss", {
		{"otype", "L2"}
	}},
	{"optimizer", {
		{"otype", "Adam"},
		{"learning_rate", 1e-3},
	}},
	{"encoding", {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", 2.0},
	}},
	{"network", {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", 2},
	}},
};



__global__ void printFloats(float* gpuPointer, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        printf("Value at index %d: %f\n", tid, gpuPointer[tid]);
    }
}
__global__ void print_batch(float* batch, int batch_size, int image_size) {
    //printf("HELLO???\n");
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid < batch_size) {
    //     printf("Batch %d\n", tid);
    //     for (int i = 0; i < image_size; ++i) {
    //         printf("%f ", batch[tid * image_size + i]);
    //     }
    //     printf("\n");
    // }
}

__global__ void gatherIntersections(
    float3* d_start_points, 
    float3* d_end_points, 
    int* d_num_hits, 
    float3* d_intersect_start,
    float3* d_intersect_end,
    int width, int height, int grid_size)
{
    // Calculate the index of the pixel this thread should process.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Calculate the base index for this pixel in the d_start_points and d_end_points arrays.
        int base_index = (y * width + x) * grid_size;

        // Find the number of grid cells hit by the ray from this pixel.
        int num_hits = d_num_hits[y * width + x];

        // For each hit, gather the entry and exit points.
        for (int i = 0; i < num_hits; ++i)
        {
            float3 start_point = d_start_points[base_index + i];
            float3 end_point = d_end_points[base_index + i];

            // Store the intersection points.
            d_intersect_start[2 * (base_index + i)] = start_point;
            d_intersect_end[2 * (base_index + i)] = end_point;
        }
    }
}
 
// Creates a grid of Axis-aligned bounding boxes with specified resolution
// Bounding box coordinates are specified in normalized coordinates from -1 to 1
// TODO: make this a CUDA kernel
std::vector<OptixAabb> make_grid(int resolution) {
    std::vector<OptixAabb> grid;
    float box_length = 2.0f/ (float)resolution;
    for(int x = 0; x < resolution; x++) {
        for(int y = 0; y < resolution; y++) {
            for(int z = 0; z < resolution; z++) {
                OptixAabb aabb;
                aabb.minX = -1.0f + (float)x * box_length;
                aabb.maxX = -1.0f + x * box_length + box_length;
                aabb.minY = -1.0f + y * box_length;
                aabb.maxY = -1.0f + y * box_length + box_length;
                aabb.minZ = -1.0f + z * box_length;
                aabb.maxZ = -1.0f + z * box_length + box_length;
                grid.push_back(aabb);
                //std::printf("aabb (%.2f %.2f %.2f) (%.2f %.2f %.2f)\n",
                //        aabb.minX, aabb.minY, aabb.minZ, aabb.maxX, aabb.maxY, aabb.maxZ);
            }
        }
    }
    return grid;
}
//auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);

#define EPOCHS 10
#define BATCH_SIZE tcnn::batch_size_granularity
#define DATASET_SIZE 1000

RTXDataHolder *rtx_dataholder;

__global__ void print_intersections(float3* start, float3* end, int* num_hits, int num_prim) {
    printf("Intersections\n");
    for (int i = 0; i < 10; ++i) {
        printf("ray (%i): %i hits\n", i, num_hits[i]); // origin = (%.2f, %.2f, %.2f)\n  ",
        for (int j = 0; j < num_hits[i]; ++j) {
            float3 s = start[i*num_prim + j];
            float3 e = end[i*num_prim + j];
            printf("   (%.2f %.2f %.2f) (%.2f %.2f %.2f)\n", s.x, s.y, s.z, e.x, e.y, e.z);
        }
    }
}



int main() {
    // load data from files
    // TODO: take images and poses from json and load into DataLoader
    int num_epochs = EPOCHS;
    std::cout << "---------------------- Loading Data ----------------------\n";
    // Loads the Training, validation, and test sets from the synthetic lego scene
    std::vector<ImageDataset> datasets = load_data(SceneType::SYNTHETIC, SyntheticName::LEGO);
    auto train_set = datasets[0];
    unsigned int width = train_set.image_width;
    unsigned int height = train_set.image_height;
    unsigned int channels = train_set.image_channels;
    float training_focal = train_set.focal;
    float aspect_ratio = (float)width / (float)height;
    float focal_length = 1.0f / tan(0.5f * training_focal);
    size_t image_size = width * height * channels;
    // get training dataset from datasets
    std::vector<float*> training_images = datasets[0].images;
    std::vector<float*> training_poses = datasets[0].poses;
    std::cout << "---------------------- Data Loaded ----------------------\n\n\n";
    // Initialize our Optix Program Groups and Pipeline
    // We also build our initial dense acceleration structure of AABBs

    std::cout << "---------------------- Initializing Optix ----------------------\n";
    cudaStream_t inference;
    cudaStream_t training;
    CUDA_CHECK(cudaStreamCreate(&inference));
    CUDA_CHECK(cudaStreamCreate(&training));
    std::string ptx_filename = BUILD_DIR "bin/ptx/optixPrograms.ptx";

    rtx_dataholder = new RTXDataHolder();
    std::cout << "Initializing Context \n";
    rtx_dataholder->initContext();
    std::cout << "Reading PTX file and creating modules \n";
    rtx_dataholder->createModule(ptx_filename);
    std::cout << "Creating Optix Program Groups \n";
    rtx_dataholder->createProgramGroups();
    std::cout << "Linking Pipeline \n";
    rtx_dataholder->linkPipeline(false);
    std::cout << "Building Shader Binding Table (SBT) \n";
    rtx_dataholder->buildSBT();
    
    // Build our initial dense acceleration structure
    int grid_resolution = 8;
    std::cout << "Building Acceleration Structure \n";
    std::vector<OptixAabb> grid = make_grid(grid_resolution);
    int num_primitives = grid.size();
    
    OptixAabb* d_aabb = rtx_dataholder->initAccelerationStructure(grid);
    std::cout << "Done Building Acceleration Structure \n";
    std::cout << "---------------------- Done Initializing Optix ----------------------\n\n\n";

    std::cout << "Allocating Buffers on GPU" << std::endl;
    float* d_image, *d_look_at;
    CUDA_CHECK(cudaMalloc((void **)&d_image, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_look_at, 16 * sizeof(float)));
    std::cout << "Image Buffers Allocated on GPU" << std::endl;
    // Allocate buffers to hold outputs from ray intersection tests
    // start and end points are equal to # of AABBs in AS per ray [width * height * num_primitives]
    float3 *d_start_points;
    float3 *d_end_points;
    int *d_num_hits;
            
    CUDA_CHECK(cudaMalloc((void **)&d_start_points, width * height * 3 * grid_resolution * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_end_points, width * height * 3 * grid_resolution * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_num_hits, width * height * sizeof(int)));
    std::cout << "Ray Intersection Buffers Allocated on GPU" << std::endl;

    Params *d_param;
    CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
    std::cout << "Params Buffer Allocated on GPU" << std::endl;


    // We train our neural network for a specific amount of epochs
    for (int j = 0; j < num_epochs; ++j) {
        std::printf("Started training loop epoch %d\n", j);
        
        // Loop through each set of images and poses in our training dataset
        for(int i = 0; i < training_images.size(); i++) {
            float* image = training_images[i];
            float* look_at = training_poses[i];

            // transfer image and look_at to GPU
            CUDA_CHECK(cudaMemcpyAsync(d_image, image, image_size * sizeof(float), cudaMemcpyHostToDevice, inference));
            CUDA_CHECK(cudaMemcpyAsync(d_look_at, look_at, 16 * sizeof(float), cudaMemcpyHostToDevice, inference));

            // Memset ray intersection buffers
            CUDA_CHECK(cudaMemset(d_start_points, -2, width * height * 3 * grid_resolution * sizeof(float3)));
            CUDA_CHECK(cudaMemset(d_end_points, -2, width * height * 3 * grid_resolution * sizeof(float3)));
            CUDA_CHECK(cudaMemset(d_num_hits, 0, width * height * sizeof(int)));

            // Algorithmic parameters and data pointers used in GPU program
            Params params;
            // params.transform_matrix = transform_matrix;
            float d =  2.0f / grid_resolution;
            params.delta = make_float3(d, d, d);
            params.min_point = make_float3(-1, -1, -1);
            params.max_point = make_float3(1, 1, 1);
            params.intersection_arr_size = 3 * grid_resolution;
            params.width = width;
            params.height = height;
            params.focal_length = focal_length;
            params.aspect_ratio = aspect_ratio;
            params.handle = rtx_dataholder->gas_handle;
            params.aabb = d_aabb;
            params.start_points = d_start_points;
            params.end_points = d_end_points;
            params.num_hits = d_num_hits;
            params.num_primitives = num_primitives;
            params.look_at = d_look_at;

            
            CUDA_CHECK(cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));
            const OptixShaderBindingTable &sbt_ray_march = rtx_dataholder->sbt_ray_march;
            std::cout << "Launching Ray Tracer in Ray Marching Mode (" << width*height << " rays)\n";
            OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline_ray_march, inference,
                                    reinterpret_cast<CUdeviceptr>(d_param),
                                    sizeof(Params), &sbt_ray_march, width, height, 1));
            CUDA_CHECK(cudaStreamSynchronize(inference));

            // CUDA Launch Sampling Kernel given entry and exit points from this perspective
            d_start_points = params.start_points;
            d_end_points = params.end_points;
            d_num_hits = params.num_hits;

            print_intersections<<<1,1>>>(d_start_points, d_end_points, d_num_hits, 3 * grid_resolution);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::cout << "Launching Sampling Kernel \n";
            //each point stores a location xyz and a viewing direction phi and psi
            float5* d_sampled_points;
            int num_points;
            int samples_per_intersect = 32;
            std::cout << "Finding number of points to sample \n";
            int* h_num_hits = (int*)malloc(sizeof(int) * width * height);

            cudaMemcpy(h_num_hits, d_num_hits, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
            int num_hits = 0;
            for (int i = 0; i < width * height; ++i) {
                num_hits += h_num_hits[i];
            }
            printf("num_hits: %d\n", num_hits);
            printf("sampled_points: %d\n", samples_per_intersect * num_hits);
            int size_samples = num_hits * samples_per_intersect * sizeof(float5);
            printf("ALLOCATING %d bytes for samples (shouldn't be zero) \n", size_samples);

            CUDA_CHECK(cudaMalloc((void**)&d_sampled_points, size_samples));

            launchUniformSampler(
                d_start_points,
                d_end_points,
                d_num_hits,
                d_sampled_points,
                samples_per_intersect,
                width, height,
                num_primitives, 
                inference,
                num_points);

            // tcnn inference on point buffer from sampling kernels
            
            // Optix Launch Volume Rendering kernel

            // tcnn compute loss and backpropagate

	    break;
        }
        break;
    }
    return 0;
}
