#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>

#include "stdio.h"


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
// #include "transform_loader.h"

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
// Creates a grid of Axis-aligned bounding boxes with specified resolution
// Bounding box coordinates are specified in normalized coordinates from -1 to 1
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




int main() {
    // load data from files
    // TODO: take images and poses from json and load into DataLoader
    int num_epochs = EPOCHS;
    // Loads the Training, validation, and test sets from the synthetic lego scene
    std::vector<ImageDataset> datasets = load_data(SceneType::SYNTHETIC, SceneName::LEGO);
    unsigned int width = datasets[0].image_width;
    unsigned int height = datasets[0].image_height;
    unsigned int channels = datasets[0].image_channels;
    size_t image_size = width * height * channels;
    // get training dataset from datasets
    std::vector<float*> training_images = datasets[0].images;
    std::vector<float*> training_poses = datasets[0].poses;
    
    // Initialize our Optix Program Groups and Pipeline
    // We also build our initial dense acceleration structure of AABBs
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string ptx_filename = BUILD_DIR "/ptx/optixPrograms.ptx";

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
    
    rtx_dataholder->initAccelerationStructure(grid);
    std::cout << "Done Building Acceleration Structure \n";
    // We train our neural network for a specific amount of epochs
    for (int j = 0; j < num_epochs; ++j) {
        std::printf("Started training loop epoch %d\n", j);
        
        // Loop through each set of images and poses in our training dataset
        for(int i = 0; i < training_images.size(); i++) {
            float* image = training_images[i];
            float* look_at = training_poses[i];

            float* d_image, *d_look_at;
            // allocate memory for image and look_at on GPU
            CUDA_CHECK(cudaMalloc((void **)&d_image, image_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void **)&d_look_at, 16 * sizeof(float)));

            // transfer image and look_at to GPU
            CUDA_CHECK(cudaMemcpyAsync(d_image, image, image_size * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_look_at, look_at, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));

            std::cout << "Allocating Buffers for Ray Intersection Tests \n";
            // Allocate buffers to hold outputs from ray intersection tests
            // start and end points are equal to # of AABBs in AS per ray [width * height * num_primitives]
            float3 *d_start_points;
            float3 *d_end_points;
            int *d_num_hits;
            CUDA_CHECK(cudaMalloc(
                (void **)&d_start_points, width * height * num_primitives * sizeof(float3)));
            CUDA_CHECK(cudaMalloc(
                (void **)&d_end_points, width * height * num_primitives * sizeof(float3)));
            CUDA_CHECK(cudaMalloc((void **)&d_num_hits, width * height * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_start_points, -2, width * height * num_primitives * sizeof(float3)));
            CUDA_CHECK(cudaMemset(d_end_points, -2, width * height * num_primitives * sizeof(float3)));
            CUDA_CHECK(cudaMemset(d_num_hits, 0, width * height * sizeof(int)));

            // Algorithmic parameters and data pointers used in GPU program
            Params params;
            // params.transform_matrix = transform_matrix;
            float d =  2.0f / grid_resolution;
            params.delta = make_float3(d, d, d);
            params.min_point = make_float3(-1, -1, -1);
            params.max_point = make_float3(1, 1, 1);
            params.width = width;
            params.height = height;
            params.handle = rtx_dataholder->gas_handle;
            params.start_points = d_start_points;
            params.end_points = d_end_points;
            params.num_hits = d_num_hits;
            params.num_primitives = num_primitives;
            Params *d_param;
            CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
            CUDA_CHECK(cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));
            const OptixShaderBindingTable &sbt_ray_march = rtx_dataholder->sbt_ray_march;
            std::cout << "Launching Ray Tracer in Ray Marching Mode \n";
            OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline_ray_march, stream,
                                    reinterpret_cast<CUdeviceptr>(d_param),
                                    sizeof(Params), &sbt_ray_march, width, height, 1));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // CUDA Launch Sampling Kernel given entry and exit points from this perspective
            d_start_points = params.start_points;
            d_end_points = params.end_points;
            d_num_hits = params.num_hits;
            std::cout << "Launching Sampling Kernel \n";
            int num_points = 0;
            // tcnn inference on point buffer from sampling kernels
            
            // Optix Launch Volume Rendering kernel

            // tcnn compute loss and backpropagate


        }
    }
    return 0;
}
