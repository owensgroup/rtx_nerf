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
//auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);

#define EPOCHS 10
#define BATCH_SIZE tcnn::batch_size_granularity*2048
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

__global__ void print_int_arr(int* arr, int width, int height) {
    printf("Printing int array\n");
    for (int i = 0; i < 10; ++i) {
        for(int j = 0; j < 10; ++j) {
            printf("%d ", arr[i * width + j]);
        }
        printf("\n");
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
    for(int i = 0; i < 32; ++i) {
        printf("%f %f %f %f %f\n", arr[i*5], arr[i*5+1], arr[i*5+2], arr[i*5+3], arr[i*5+4]);
    }
    printf("\n");
    printf("Printing last 32 points \n");
    for(int i = size-32; i < size; ++i) {
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

int main() {
    // load data from files
    // TODO: take images and poses from json and load into DataLoader
    int n_input_dims = 5;
    int n_output_dims = 4;
    int batch_size = BATCH_SIZE;
    auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);
    int num_epochs = EPOCHS;
    std::cout << "---------------------- Loading Data ----------------------\n";
    // Loads the Training, validation, and test sets from the synthetic lego scene
    std::vector<ImageDataset> datasets = load_data(SceneType::SYNTHETIC, SyntheticName::LEGO);
    auto train_set = datasets[0];
    unsigned int width = train_set.image_width;
    unsigned int height = train_set.image_height;
    unsigned int channels = train_set.image_channels;
    printf("Channels: %i\n", channels);
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
    // Allocate buffers to hold outputs from ray intersection tests
    // start and end points are equal to # of AABBs in AS per ray [width * height * num_primitives]
    float3 *d_start_points;
    float3 *d_end_points;
    int *d_num_hits;
    float2 *d_view_dir;
    float* d_pixels;
    float* d_temp_out;
    tcnn::network_precision_t* d_pixels_half;
    CUDA_CHECK(cudaMalloc((void**)&d_pixels, width * height * sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc((void **)&d_start_points, width * height * 3 * grid_resolution * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_end_points, width * height * 3 * grid_resolution * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void **)&d_num_hits, width * height * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_view_dir, width * height * sizeof(float2)));
    CUDA_CHECK(cudaMalloc((void **)&d_temp_out, batch_size * n_output_dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_pixels_half, width * height * sizeof(tcnn::network_precision_t) * 3));
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
            CUDA_CHECK(cudaMemcpyAsync(d_image, image, image_size * sizeof(float), cudaMemcpyHostToDevice, inference_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_look_at, look_at, 16 * sizeof(float), cudaMemcpyHostToDevice, inference_stream));

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
            params.viewing_direction = d_view_dir;

            
            CUDA_CHECK(cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));
            const OptixShaderBindingTable &sbt_ray_march = rtx_dataholder->sbt_ray_march;
            std::cout << "Launching Ray Tracer in Ray Marching Mode (" << width*height << " rays)\n";
            OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline_ray_march, inference_stream,
                                    reinterpret_cast<CUdeviceptr>(d_param),
                                    sizeof(Params), &sbt_ray_march, width, height, 1));
            CUDA_CHECK(cudaStreamSynchronize(inference_stream));

            // CUDA Launch Sampling Kernel given entry and exit points from this perspective
            d_start_points = params.start_points;
            d_end_points = params.end_points;
            d_num_hits = params.num_hits;

            // print_intersections<<<1,1>>>(d_start_points, d_end_points, d_num_hits, 3 * grid_resolution);
            // CUDA_CHECK(cudaDeviceSynchronize());

            std::cout << "Launching Sampling Kernel \n";
            //each point stores a location xyz and a viewing direction phi and psi
            
            int num_points;
            int samples_per_intersect = 32;
            // std::cout << "Print Num Hits \n";
            // print_int_arr<<<1,1>>>(d_num_hits, width, height);
            // CUDA_CHECK(cudaDeviceSynchronize());

            // std::cout << "Print Viewdirs \n";
            // print_float2_arr<<<1,1>>>(d_view_dir, width, height);
            // CUDA_CHECK(cudaDeviceSynchronize());

            thrust::device_ptr<int> dev_ptr_num_hits = thrust::device_pointer_cast(d_num_hits);
            num_points = thrust::reduce(dev_ptr_num_hits, dev_ptr_num_hits + width * height);
            thrust::device_vector<int> d_hit_indsV(width * height);
            // exclusive scan on dev_ptr_num_hits
            thrust::exclusive_scan(dev_ptr_num_hits, dev_ptr_num_hits + width * height, d_hit_indsV.begin());

            // convert dev_ptr_num_hits back to device int pointer
            d_num_hits = dev_ptr_num_hits.get();
            int *d_hit_inds = thrust::raw_pointer_cast(d_hit_indsV.data());
            // std::cout << "Print Num Hits post scan \n";
            // print_int_arr<<<1,1>>>(d_hit_inds, width, height);
            CUDA_CHECK(cudaDeviceSynchronize());


            printf("num_hits_cu: %d\n", num_points);
            int num_sampled_points = samples_per_intersect * num_points;
            printf("sampled_points: %d\n", num_sampled_points);
            num_sampled_points = (num_sampled_points / batch_size) * batch_size + batch_size;
            printf("upsampled_points: %d\n", num_sampled_points);
            float* d_sampled_points;
            float* d_sampled_points_radiance;
            float* d_t_vals;
            unsigned int size_input = num_sampled_points * sizeof(float) * 5;
            unsigned int size_output = num_sampled_points * sizeof(float) * 4;
            printf("ALLOCATING %d bytes for samples (shouldn't be zero) \n", size_input);
            printf("ALLOCATING %d bytes for radiance (shouldn't be zero) \n", size_output);
            CUDA_CHECK(cudaMalloc((void**)&d_sampled_points, size_input));
            CUDA_CHECK(cudaMalloc((void**)&d_sampled_points_radiance,
                        size_output));
            CUDA_CHECK(cudaMalloc((void**)&d_t_vals, sizeof(float) * num_sampled_points));

            launchSampler(
                d_start_points,
                d_end_points,
                d_view_dir,
                d_t_vals,
                d_sampled_points,
                width, height, grid_resolution,
                d_num_hits, d_hit_inds, 
                SAMPLING_REGULAR, inference_stream
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            // print_float5_arr<<<1,1>>>(d_sampled_points, num_sampled_points);
            // CUDA_CHECK(cudaDeviceSynchronize());
            // print_float3_arr<<<1,1>>>(d_start_points, width * height * grid_resolution * 3);
            // CUDA_CHECK(cudaDeviceSynchronize());
            // print_float3_arr<<<1,1>>>(d_end_points, width * height * grid_resolution * 3);
            // CUDA_CHECK(cudaDeviceSynchronize());
            
            uint32_t padded_output_width = model.network->padded_output_width();
            tcnn::GPUMatrix<float> input_batch(n_input_dims, batch_size);
            tcnn::GPUMatrix<tcnn::network_precision_t> output_fwd(padded_output_width, batch_size);
            int num_iters = 0;
            for(int i = 0; i < num_sampled_points; i+=batch_size) {
                num_iters++;
                unsigned int offset = i * n_input_dims;
                CUDA_CHECK(cudaMemcpy(
                    input_batch.data(),
                    d_sampled_points + offset,
                    batch_size * n_input_dims * sizeof(float),
                    cudaMemcpyDeviceToDevice));
                
                auto ctx = model.network->forward(inference_stream, input_batch, &output_fwd, true);
                tcnn::GPUMatrix<tcnn::network_precision_t> output_slice = output_fwd.slice_rows(0, n_output_dims);
                // convert output_fwd to float
                int num_el = output_fwd.n_elements();
                int blockSize = 256;
                int numBlocks = (num_el + blockSize - 1) / blockSize;
                convertHalfToFloat<<<numBlocks,blockSize>>>(output_slice.data(), d_temp_out, batch_size * n_output_dims);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(
                    d_sampled_points_radiance + i * 4,
                    d_temp_out,
                    batch_size * n_output_dims * sizeof(float),
                    cudaMemcpyDeviceToDevice));
            }
            //print radiance buffer values
            printf("Printing radiance buffer values\n");
            print_float4_arr<<<1,1>>>(d_sampled_points_radiance, num_sampled_points);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Launch Volume Rendering kernel
            printf("Launching Volume Rendering Kernel\n");
            

            launch_volrender_cuda(
                d_sampled_points,
                d_sampled_points_radiance,
                d_num_hits,
                d_hit_inds,
                d_t_vals,
                width,
                height,
                samples_per_intersect,
                d_pixels
            );
            
            
            // initialize host pixels
            float* h_pixels = new float[width * height * 3];
            // copy pixels to host
            CUDA_CHECK(cudaMemcpy(
                h_pixels,
                d_pixels,
                width * height * sizeof(float) * 3,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // save pixels to png file with stb
            stbi_write_png("output.png", width, height, 3, h_pixels, width);
            size_t bufferSize = width * height * channels;
            int blockSize = 256;
            int numBlocks = (bufferSize + blockSize - 1) / blockSize;
            floatToHalf<<<numBlocks, blockSize>>>(d_pixels, d_pixels_half, bufferSize);
            cudaDeviceSynchronize();
            tcnn::GPUMatrix<tcnn::network_precision_t> predicted_image(d_pixels_half, width * height, channels);

            // tcnn compute loss and backpropagate
            tcnn::GPUMatrix<float> target_image(d_image, width * height, channels);
            tcnn::GPUMatrix<float> values(width * height, channels);
            tcnn::GPUMatrix<tcnn::network_precision_t> gradients(width * height, channels);
            model.loss->evaluate(1.0f, predicted_image, target_image, values, gradients);
            float image_loss = tcnn::reduce_sum(values.data(), values.n_elements(), inference_stream);
            std::cout << "Image Loss: " << image_loss << std::endl;

            
	    break;
        }
        break;
    }
    return 0;
}
