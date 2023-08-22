#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
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
#include "tiny-cuda-nn/config.h"
#include "tiny-cuda-nn/reduce_sum.h"
#include <json/json.h>
#include "rtx/include/params.h"
#include "rtx/include/rtxFunctions.h"

#include "data_loader.h"
#include "vol_render.h"
// Configure the model
nlohmann::json config = {
	{"loss", {
		{"otype", "L2"}
	}},
    // adam optimizer decays from 5e-4 to 5e-5
	{"optimizer", {
		{"otype", "Adam"},
		{"learning_rate", 1e-3},
        {"beta1", 0.9},
        {"beta2", 0.999},
        {"epsilon", 1e-8}
	}},
	{"encoding", {
        {"otype", "Composite"},
        {"nested", {
            {
                {"n_dims_to_encode", 3}, // Spatial dims
                {"otype", "Frequency"},
                {"n_frequencies", 10}
            },
            {
                {"n_dims_to_encode", 2}, // Non-linear appearance dims.
                {"otype", "Frequency"},
                {"n_bins", 4}
            }
        }}
    }},
	{"network", {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "Sigmoid"},
		{"n_neurons", 128},
		{"n_hidden_layers", 8}
	}}
};

template<typename T>
void printGPUMatrix(
    const tcnn::GPUMatrix<T>& matrix,
    int n_rows, int n_cols) {
    // Get the dimensions of the matrix
    uint32_t rows = matrix.rows();
    uint32_t cols = matrix.cols();

    // Allocate host memory to store the matrix data
    T* hostData = new T[rows * cols];

    // Copy the matrix data from GPU to host
    cudaMemcpy(hostData, matrix.data(), sizeof(T) * rows * cols, cudaMemcpyDeviceToHost);

    // Print the matrix values
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            std::cout << hostData[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free the host memory
    delete[] hostData;
}

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

void printGPUMem() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t usedMem = totalMem - freeMem;
    std::cout << "GPU Memory Usage: " << usedMem / 1024 / 1024 << " MB" << std::endl;
}

//auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);

#define EPOCHS 10
#define BATCH_SIZE tcnn::BATCH_SIZE_GRANULARITY*160
#define DATASET_SIZE 1000

RTXDataHolder *rtx_dataholder;

__global__ void print_intersections(float3* start, float3* end, int* num_hits, int num_prim) {
    printf("Intersections\n");
    for (int i = 0; i < 100; ++i) {
        printf("ray (%i): %i hits\n", i, num_hits[i]); // origin = (%.2f, %.2f, %.2f)\n  ",
        for (int j = 0; j < num_hits[i]; ++j) {
            float3 s = start[i*num_prim + j];
            float3 e = end[i*num_prim + j];
            printf("   (%.2f %.2f %.2f) (%.2f %.2f %.2f)\n", s.x, s.y, s.z, e.x, e.y, e.z);
        }
    }
}

__global__ void convertHalfToFloat(__half* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] = __half2float(input[tid]);
    }
}

__global__ void floatToHalf(float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void print_int_arr(int* arr, int size) {
    // Print the first 10 and last 10 elements in the buffer
    printf("First 10 elements:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    printf("Last 10 elements:\n");
    for (int i = size - 10; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}



__global__ void print_float_arr(float* arr, int size) {
    printf("Printing float array\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

__global__ void print_float2_arr(float2* arr, int width, int height) {
    printf("Printing float2 array\n");
    for (int i = 0; i < 10; ++i) {
        for(int j = 0; j < 10; ++j) {
            printf("%f %f ", arr[i * width + j].x, arr[i * width + j].y);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_float5_arr(float* arr, int size) {
    printf("Printing first 32 points \n");
    for(int i = 0; i < 64; ++i) {
        printf("%f %f %f %f %f\n", arr[i*5], arr[i*5+1], arr[i*5+2], arr[i*5+3], arr[i*5+4]);
    }
    printf("\n");
    printf("Printing last 32 points \n");
    for(int i = size-64; i < size; ++i) {
        printf("%f %f %f %f %f\n", arr[i*5], arr[i*5+1], arr[i*5+2], arr[i*5+3], arr[i*5+4]);
    }
    printf("\n");
}

__global__ void print_float3_arr(float3* arr, int size) {
    printf("Printing float3 array\n");
    printf("Printing first 32 points \n");
    for(int i = 0; i < 32; ++i) {
        printf("%f %f %f\n", arr[i].x, arr[i].y, arr[i].z);
    }
    printf("\n");
    printf("Printing last 32 points \n");
    for(int i = size-32; i < size; ++i) {
        printf("%f %f %f\n", arr[i].x, arr[i].y, arr[i].z);
    }
    printf("\n");
}

__global__ void print_float4_arr(float* arr, int size) {
    printf("Printing float4 array\n");
    printf("Printing first 32 points \n");
    for(int i = 0; i < 32; ++i) {
        printf("%f %f %f %f\n", arr[i*4], arr[i*4+1], arr[i*4+2], arr[i*4+3]);
    }
    printf("\n");
    printf("Printing last 32 points \n");
    for(int i = size-32; i < size; ++i) {
        printf("%f %f %f %f\n", arr[i*4], arr[i*4+1], arr[i*4+2], arr[i*4+3]);
    }
    printf("\n");
}

struct RayPayload {
    int num_hits;
    float3 origin;
    float2 view_dir;
    float* t_start;
    float* t_end;
    float3* start_points;
    float3* end_points;
    float3 pixel_color_gt;
};

int main() {
    // load data from files
    // TODO: take images and poses from json and load into DataLoader
    int n_input_dims = 5;
    int n_output_dims = 4;
    int batch_size = BATCH_SIZE;
    auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);
    model.optimizer->allocate(model.network);
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
    cudaStream_t inference_stream;
    cudaStream_t training_stream;
    CUDA_CHECK(cudaStreamCreate(&inference_stream));
    CUDA_CHECK(cudaStreamCreate(&training_stream));
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

    // first generate rays for each pixel

    // Allocate buffers to hold outputs from ray intersection tests
    // start and end points are equal to # of AABBs in AS per ray [width * height * num_primitives]
    float3 *d_start_points;
    float3 *d_end_points;
    float3 *d_ray_origins;
    int *d_num_hits;
    float2 *d_view_dir;
    float* d_pixels;
    float* d_temp_out;
    float* d_t_start;
    float* d_t_end;
    tcnn::network_precision_t* d_pixels_half;

    CUDA_CHECK(cudaMalloc((void**)&d_ray_origins, width * height * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**)&d_t_start, width * height * 3 * grid_resolution * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_t_end, width * height * 3 * grid_resolution * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_start_points, width * height * 3 * grid_resolution * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_end_points, width * height * 3 * grid_resolution * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_num_hits, width * height * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_view_dir, width * height * sizeof(float2)));
    std::cout << "Ray Intersection Buffers Allocated on GPU" << std::endl;

    CUDA_CHECK(cudaMalloc((void**)&d_pixels, batch_size * sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc((void **)&d_temp_out, batch_size * n_output_dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_pixels_half, batch_size * sizeof(tcnn::network_precision_t) * 3));
    

    

    Params *d_param;
    CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
    std::cout << "Params Buffer Allocated on GPU" << std::endl;

    float3* h_origin;
    float2* h_view_dir;
    int* h_num_hits;
    float* h_t_start;
    float* h_t_end;
    float3* h_start_points;
    float3* h_end_points;

    h_origin = (float3*)malloc(width * height * sizeof(float3));
    h_view_dir = (float2*)malloc(width * height * sizeof(float2));
    h_num_hits = (int*)malloc(width * height * sizeof(int));
    h_t_start = (float*)malloc(width * height * 3 * grid_resolution * sizeof(float));
    h_t_end = (float*)malloc(width * height * 3 * grid_resolution * sizeof(float));
    h_start_points = (float3*)malloc(width * height * 3 * grid_resolution * sizeof(float3));
    h_end_points = (float3*)malloc(width * height * 3 * grid_resolution * sizeof(float3));
    std::vector<RayPayload> ray_payloads;
    // Loop through training data and build dataset
    // dataset consists of ray_payloads and ground truth pixel colors
    // ray_payloads: (origin, dir, num_hits, t_start, t_end)
    for(int i = 0; i < training_images.size(); i++) {
        float* image = training_images[i];
        float* look_at = training_poses[i];
        // transfer image and look_at to GPU
        

        CUDA_CHECK(cudaMemcpyAsync(d_image, image, image_size * sizeof(float), cudaMemcpyHostToDevice, inference_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_look_at, look_at, 16 * sizeof(float), cudaMemcpyHostToDevice, inference_stream));

        // Memset ray intersection buffers
        CUDA_CHECK(cudaMemsetAsync(d_start_points, -2, width * height * 3 * grid_resolution * sizeof(float3)));
        CUDA_CHECK(cudaMemsetAsync(d_end_points, -2, width * height * 3 * grid_resolution * sizeof(float3)));
        CUDA_CHECK(cudaMemsetAsync(d_t_start, -2, width * height * 3 * grid_resolution * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_t_end, -2, width * height * 3 * grid_resolution * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_view_dir, -2, width * height * sizeof(float2)));
        CUDA_CHECK(cudaMemsetAsync(d_ray_origins, -2, width * height * sizeof(float3))); 
        CUDA_CHECK(cudaMemsetAsync(d_num_hits, 0, width * height * sizeof(int)));

        Params params;
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
        params.t_start = d_t_start;
        params.t_end = d_t_end;
        params.num_hits = d_num_hits;
        params.num_primitives = num_primitives;
        params.look_at = d_look_at;
        params.viewing_direction = d_view_dir;
        params.ray_origins = d_ray_origins;

        CUDA_CHECK(cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));
        const OptixShaderBindingTable &sbt_ray_march = rtx_dataholder->sbt_ray_march;
        std::cout << "Launching Ray Tracer in Ray Marching Mode (" << width*height << " rays)\n";
        OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline_ray_march, inference_stream,
                                reinterpret_cast<CUdeviceptr>(d_param),
                                sizeof(Params), &sbt_ray_march, width, height, 1));
        CUDA_CHECK(cudaStreamSynchronize(inference_stream));
        d_start_points = params.start_points;
        d_end_points = params.end_points;
        d_t_start = params.t_start;
        d_t_end = params.t_end;
        d_num_hits = params.num_hits;
        d_ray_origins = params.ray_origins;

        CUDA_CHECK(cudaMemcpy(h_origin, d_ray_origins, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_view_dir, d_view_dir, width * height * sizeof(float2), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_num_hits, d_num_hits, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_t_start, d_t_start, width * height * 3 * grid_resolution * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_t_end, d_t_end, width * height * 3 * grid_resolution * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_start_points, d_start_points, width * height * 3 * grid_resolution * sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end_points, d_end_points, width * height * 3 * grid_resolution * sizeof(float3), cudaMemcpyDeviceToHost));

        for(int i = 0; i < width * height; i++) {
            RayPayload payload;
            payload.origin = h_origin[i];
            payload.view_dir = h_view_dir[i];
            payload.num_hits = h_num_hits[i];
            payload.t_start = (float*)malloc(h_num_hits[i] * sizeof(float));
            payload.t_end = (float*)malloc(h_num_hits[i] * sizeof(float));
            payload.start_points = (float3*)malloc(h_num_hits[i] * sizeof(float3));
            payload.end_points = (float3*)malloc(h_num_hits[i] * sizeof(float3));
            for(int j = 0; j < payload.num_hits; j++) {
                payload.t_start[j] = h_t_start[i * 3 * grid_resolution + j];
                payload.t_end[j] = h_t_end[i * 3 * grid_resolution + j];
                payload.start_points[j] = h_start_points[i * 3 * grid_resolution + j];
                payload.end_points[j] = h_end_points[i * 3 * grid_resolution + j];
            }
            payload.pixel_color_gt = make_float3(image[i * 3], image[i * 3 + 1], image[i * 3 + 2]);
            ray_payloads.push_back(payload);
        }
    }
    free(h_origin);
    free(h_view_dir);
    free(h_num_hits);
    free(h_t_start);
    free(h_t_end);
    free(h_start_points);
    free(h_end_points);

    cudaFree(d_ray_origins);
    cudaFree(d_t_start);
    cudaFree(d_t_end);
    cudaFree(d_start_points);
    cudaFree(d_end_points);
    cudaFree(d_num_hits);
    cudaFree(d_view_dir);
    std::cout << "---------------------- Done Generating Rays ----------------------\n\n\n";

    // Print 10 random payloads from ray_payloads
    // std::cout << "Random Ray Payloads:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     int random_index = rand() % ray_payloads.size();
    //     RayPayload random_payload = ray_payloads[random_index];
    //     std::cout << "Payload " << i+1 << ":" << std::endl;
    //     std::cout << "Origin: (" << random_payload.origin.x << ", " << random_payload.origin.y << ", " << random_payload.origin.z << ")" << std::endl;
    //     std::cout << "View Direction: (" << random_payload.view_dir.x << ", " << random_payload.view_dir.y << ")" << std::endl;
    //     std::cout << "Number of Hits: " << random_payload.num_hits << std::endl;
    //     std::cout << "T Start: ";
    //     for (int j = 0; j < random_payload.num_hits; j++) {
    //         std::cout << random_payload.t_start[j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "T End: ";
    //     for (int j = 0; j < random_payload.num_hits; j++) {
    //         std::cout << random_payload.t_end[j] << " ";
    //     }

    //     // print start and end points
    //     std::cout << std::endl;
    //     std::cout << "Start Points: ";
    //     for (int j = 0; j < random_payload.num_hits; j++) {
    //         std::cout << "(" << random_payload.start_points[j].x << ", " << random_payload.start_points[j].y << ", " << random_payload.start_points[j].z << ") ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "End Points: ";
    //     for (int j = 0; j < random_payload.num_hits; j++) {
    //         std::cout << "(" << random_payload.end_points[j].x << ", " << random_payload.end_points[j].y << ", " << random_payload.end_points[j].z << ") ";
    //     }

    //     std::cout << std::endl;
    //     // print ground truth pixel color
    //     std::cout << "Ground Truth Pixel Color: (" << random_payload.pixel_color_gt.x << ", " << random_payload.pixel_color_gt.y << ", " << random_payload.pixel_color_gt.z << ")" << std::endl;
    //     std::cout << std::endl << std::endl;
    // }
    

    int* h_batch_num_hits = (int*)malloc(batch_size * sizeof(int));
    float3* h_gt_pixels = (float3*)malloc(batch_size * sizeof(float3));
    h_view_dir = (float2*)malloc(batch_size * sizeof(float2));

    float* d_gt_pixels;
    
    int* d_batch_num_hits;
    CUDA_CHECK(cudaMalloc((void **)&d_gt_pixels, batch_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_num_hits, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_view_dir, batch_size * sizeof(float2)));

    int* d_batch_hit_inds;
    // We train our neural network for a specific amount of epochs
    for (int j = 0; j < num_epochs; ++j) {
        std::printf("Started training loop epoch %d\n", j);
        // shuffle ray payloads
        std::random_shuffle(ray_payloads.begin(), ray_payloads.end());
        // Loop through each set of images and poses in our training dataset
        for(int i = 0; i < ray_payloads.size(); i+=batch_size) {
            // get batch_size ray payloads from ray_payloads
            // std::cout << "Getting batch of ray payloads \n";
            std::vector<RayPayload> batch_ray_payloads(ray_payloads.begin() + i, ray_payloads.begin() + i + batch_size);
            // store num_hits in ray payloads in h_batch_num_hits

            // std::cout << "Going from AOS to SOA \n";
            for(int k = 0; k < batch_size; k++) {
                h_batch_num_hits[k] = batch_ray_payloads[k].num_hits;
                h_gt_pixels[k] = batch_ray_payloads[k].pixel_color_gt;
                h_view_dir[k] = batch_ray_payloads[k].view_dir;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_num_hits, h_batch_num_hits, batch_size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_gt_pixels, h_gt_pixels, batch_size * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_view_dir, h_view_dir, batch_size * sizeof(float2), cudaMemcpyHostToDevice));
            // turn d_batch_num_hits into a thrust device pointer
            thrust::device_ptr<int> dev_ptr_num_hits(d_batch_num_hits);
            int num_points = thrust::reduce(dev_ptr_num_hits, dev_ptr_num_hits + batch_size);
            std::cout << "num_points: " << num_points << std::endl;
            thrust::device_vector<int> d_hit_indsV(batch_size);
            thrust::exclusive_scan(dev_ptr_num_hits, dev_ptr_num_hits + batch_size, d_hit_indsV.begin());
            d_batch_num_hits = dev_ptr_num_hits.get();
            d_batch_hit_inds = thrust::raw_pointer_cast(d_hit_indsV.data());
            // print d_batch_num_hits and d_batch_hit_inds
            // std::cout << "Printing d_batch_num_hits and d_batch_hit_inds \n";
            // print_int_arr<<<1,1>>>(d_batch_num_hits, batch_size);
            // CUDA_CHECK(cudaDeviceSynchronize());
            // print_int_arr<<<1,1>>>(d_batch_hit_inds, batch_size);
            // CUDA_CHECK(cudaDeviceSynchronize());

            //free both
            float3* h_start_points = (float3*)malloc(num_points * sizeof(float3));
            float3* h_end_points = (float3*)malloc(num_points * sizeof(float3));
            // float* h_t_end = (float*)malloc(num_points * sizeof(float));

            std::cout << "Filling in start_points, end_points, and t_end \n";
            // fill in start_points, end_points, and t_end
            int offset = 0;
            for(int k = 0; k < batch_size; k++) {
                for(int l = 0; l < batch_ray_payloads[k].num_hits; l++) {
                    h_start_points[offset + l] = batch_ray_payloads[k].start_points[l];
                    h_end_points[offset + l] = batch_ray_payloads[k].end_points[l];
                    // h_t_end[offset + l] = batch_ray_payloads[k].t_end[l];
                }
                offset += batch_ray_payloads[k].num_hits;
            }

            std::cout << "Allocating GPU Buffers for Sampling \n";
            float3* d_start_points;
            float3* d_end_points;
            // float* d_t_end;
            //cudafree both
            CUDA_CHECK(cudaMalloc((void **)&d_start_points, num_points * sizeof(float3)));
            CUDA_CHECK(cudaMalloc((void **)&d_end_points, num_points * sizeof(float3)));
            // CUDA_CHECK(cudaMalloc((void **)&d_t_end, num_points * sizeof(float)));

            std::cout << "Copying start_points, end_points, and t_end to GPU \n";
            CUDA_CHECK(cudaMemcpyAsync(d_start_points, h_start_points, num_points * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_end_points, h_end_points, num_points * sizeof(float3), cudaMemcpyHostToDevice));
            // CUDA_CHECK(cudaMemcpyAsync(d_t_end, h_t_end, num_points * sizeof(float), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            int samples_per_intersect = 32;
            printf("num_hits_cu: %d\n", num_points);
            int num_sampled_points = samples_per_intersect * num_points;
            printf("sampled_points: %d\n", num_sampled_points);
            num_sampled_points = (num_sampled_points / 256) * 256 + 256;
            printf("upsampled_points: %d\n", num_sampled_points);
            float* d_sampled_points;
            float* d_sampled_points_radiance;
            float* d_t_vals;
            unsigned int size_input = num_sampled_points * sizeof(float) * 5;
            unsigned int size_output = num_sampled_points * sizeof(float) * 4;
            printf("ALLOCATING %d bytes for samples (shouldn't be zero) \n", size_input);
            printf("ALLOCATING %d bytes for radiance (shouldn't be zero) \n", size_output);
            // cudafree all of these
            CUDA_CHECK(cudaMalloc((void**)&d_sampled_points, size_input));
            CUDA_CHECK(cudaMalloc((void**)&d_sampled_points_radiance,
                        size_output));
            CUDA_CHECK(cudaMalloc((void**)&d_t_vals, sizeof(float) * num_sampled_points));
            

            // std::cout << "Printing start_points and end_points \n";
            // print_float3_arr<<<1,1>>>(d_start_points, num_points);
            // CUDA_CHECK(cudaDeviceSynchronize());
            // print_float3_arr<<<1,1>>>(d_end_points, num_points);
            // CUDA_CHECK(cudaDeviceSynchronize());

            // std::cout << "Launching Sampling Kernel \n";
            launchSampler(
                d_start_points,
                d_end_points,
                d_view_dir,
                d_t_vals,
                d_sampled_points,
                batch_size, grid_resolution,
                d_batch_num_hits, d_batch_hit_inds,
                SAMPLING_REGULAR, inference_stream);
            
            
            uint32_t padded_output_width = model.network->padded_output_width();
            tcnn::GPUMatrix<float> input_batch(n_input_dims, num_sampled_points);
            tcnn::GPUMatrix<tcnn::network_precision_t> output_fwd(padded_output_width, num_sampled_points);
            
            // printGPUMem();
            // printf("Launching Forward Pass\n");
            auto ctx = model.network->forward(inference_stream, input_batch, &output_fwd, true, true);
            // printf("Done Forward Pass\n");
            tcnn::GPUMatrix<tcnn::network_precision_t> output_slice = output_fwd.slice_rows(0, n_output_dims);
            
            int num_el = output_slice.n_elements();
            int blockSize1 = 1024;
            int numBlocks1 = (num_el + blockSize1 - 1) / blockSize1;
            convertHalfToFloat<<<numBlocks1,blockSize1>>>(output_slice.data(), d_sampled_points_radiance, num_el);
            // print radiance buffer values
            // printf("Printing radiance buffer values\n");
            // print_float4_arr<<<1,1>>>(d_sampled_points_radiance, num_sampled_points);
            // CUDA_CHECK(cudaDeviceSynchronize());
            
            // Launch Volume Rendering kernel
            // printf("Launching Volume Rendering Kernel\n");
            
            // TODO: inference stream
            launch_volrender_cuda(
                d_sampled_points,
                d_sampled_points_radiance,
                d_batch_num_hits,
                d_batch_hit_inds,
                d_t_vals,
                batch_size,
                samples_per_intersect,
                d_pixels
            );
            // printf("Done Volume Rendering Kernel\n");
            // print pixel buffer values
            // printf("Printing pixel buffer values\n");
            // print_float_arr<<<1,1>>>(d_pixels, batch_size);
            // CUDA_CHECK(cudaDeviceSynchronize());
            int blockSize2 = 1024;
            int numBlocks2 = (batch_size + blockSize2 - 1) / blockSize2;
            floatToHalf<<<numBlocks2, blockSize2>>>(d_pixels, d_pixels_half, batch_size);
            tcnn::GPUMatrix<tcnn::network_precision_t> predicted_image(d_pixels_half, batch_size, channels);
            tcnn::GPUMatrix<float> target_image(d_gt_pixels, batch_size, channels);
            tcnn::GPUMatrix<float> values(batch_size, channels);
            tcnn::GPUMatrix<tcnn::network_precision_t> gradients(batch_size, channels);
            model.loss->evaluate(1.0f, predicted_image, target_image, values, gradients);
            float batch_loss = tcnn::reduce_sum(values.data(), values.n_elements(), inference_stream);
            std::cout << "Batch Loss: " << batch_loss << std::endl;
            
            
            tcnn::network_precision_t* d_loss_mlp;
            CUDA_CHECK(cudaMalloc((void**)&d_loss_mlp, sizeof(tcnn::network_precision_t) * 16 * num_sampled_points));
            
            launch_volrender_backward_cuda(
                values.data(),
                gradients.data(),
                d_sampled_points_radiance,
                d_t_vals,
                d_batch_num_hits,
                d_batch_hit_inds,
                batch_size,
                samples_per_intersect,
                d_loss_mlp
            );
            // printf("Done Volume Rendering Backward Kernel\n");
            tcnn::GPUMatrix<tcnn::network_precision_t> loss_mlp(d_loss_mlp, 16, num_sampled_points);
            model.network->backward(inference_stream, *ctx, input_batch, output_fwd, loss_mlp);
            printGPUMem();
            // free buffers
            cudaFree(d_sampled_points);
            cudaFree(d_sampled_points_radiance);
            cudaFree(d_t_vals);
            cudaFree(d_start_points);
            cudaFree(d_end_points);
            cudaFree(d_loss_mlp);
            free(h_start_points);
            free(h_end_points);
            // std::cout << "Done freeing buffers \n";
            printGPUMem();
        }
        break;
    }
    return 0;
}
