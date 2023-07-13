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

// #include "data_loader.h"
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

struct ImageDataset {
  std::vector<float*> images;
  std::vector<float*> poses;
  float focal;
};

ImageDataset load_images_json(std::string basename, std::string s) {
    std::ifstream file(basename + "/transforms_" + s + ".json");
    if (!file.is_open()) {
        std::cerr << "Failed to open transform JSON file: " << basename + "/transforms_train.json" << std::endl;
        exit(1);
    }

    std::vector<float*> images;
    std::vector<float*> poses;

    Json::Value root;
    file >> root;
    float camera_angle_x = root["camera_angle_x"].asFloat();
    std::cout << "Camera Angle X: " << camera_angle_x << std::endl;

    const Json::Value frames = root["frames"];
    for (const auto& frame : frames) {
        std::string file_path = frame["file_path"].asString();
        float rotation = frame["rotation"].asFloat();
        const Json::Value transform_matrix = frame["transform_matrix"];

        std::cout << "File Path: " << file_path << std::endl;
        std::cout << "Rotation: " << rotation << std::endl;

        std::printf("basename = %s\n", basename.c_str());
        // Load the image using the file path
        file_path = basename + "/" + file_path + ".png";
        //cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
        int width, height, channels_in_file;
        int desired_channels = 4;
        std::printf("Loading image from %s\n", file_path.c_str());
        float* image = stbi_loadf(file_path.c_str(), &width, &height, 
                                            &channels_in_file, 
                                            desired_channels);
        float* pose = new float[16];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                pose[i*4 + j] = transform_matrix[i][j].asFloat();
            }
        }

        // Check if the image was loaded successfully
        if (image == nullptr) {
            std::cerr << "Failed to load the image." << std::endl;
            ImageDataset dataset;
            return dataset;
        }

        // Add the image to the vector of training images
        images.push_back(image);
        poses.push_back(pose);
    }

    float focal = .5 * 800 / std::tan(.5 * camera_angle_x);
    ImageDataset dataset;
    dataset.images = images;
    dataset.poses = poses;
    dataset.focal = focal;
    return dataset;
}

std::vector<ImageDataset> load_synthetic_data(std::string directory) {
    // Load training JSON in directory root
    std::vector<std::string> strings = { "train", "val", "test" };
    std::vector<ImageDataset> datasets;
    for (const auto& string : strings) {
        std::string basename = std::string("/home/tsaluru/opt_nerf/data/nerf_synthetic/lego/");
        ImageDataset dataset = load_images_json(basename, string);
        datasets.push_back(dataset);
        break;
    }

    return datasets;
}

enum class SceneType { LLFF, SYNTHETIC };
enum class SceneName { LEGO, FERN };

std::vector<ImageDataset> load_data(SceneType type, SceneName name) {
    std::string directory;
    std::string filename;
    switch (name) {
        case SceneName::LEGO:
            filename = "lego/";
            break;
        case SceneName::FERN:
            filename = "fern/";
            break;
    }

    switch (type) {
        case SceneType::LLFF:
            directory = "/data/nerf_llff_data/" + filename;
            break;
        case SceneType::SYNTHETIC:
            directory = "/data/nerf_synthetic/" + filename;
            return load_synthetic_data(directory);
            break;
    }
  return std::vector<ImageDataset>();
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
uint32_t width = 800u;
uint32_t height = 600u;
uint32_t depth = 1;



int main() {
    // // load data from files
    // // TODO: take images and poses from json and load into DataLoader
    // int num_epochs = EPOCHS;
    // int batch_size = BATCH_SIZE;
    // size_t image_size = 800*800*4;

    // std::vector<ImageDataset> datasets = load_data(SceneType::SYNTHETIC, SceneName::LEGO);
    // std::printf("BATCH SIZE GRANULARITY %d \n", tcnn::batch_size_granularity);
    // std::printf("IMAGES IN TRAINING DATASET: %d\n", datasets[0].images.size());
    // // calculate the number of training iterations in an epoch given the batch size
    // int num_batches = datasets[0].images.size() / batch_size + 1;
    // // calculate the total number of training iterations
    // // get batch from dataloaders
    // // get training dataset from datasets
    // std::vector<float*> images = datasets[0].images;
    // std::vector<float*> poses = datasets[0].poses;
    
    // auto training_set = datasets[0];
    
    // for (int j = 0; j < num_epochs; ++j) {
    //     std::printf("Start training loop epoch %d\n", j);

    //     // instantiate number of streams according to the number of batches
    //     std::vector<cudaStream_t> streams(num_batches);
    //     for (auto& stream : streams) {
    //         cudaStreamCreate(&stream);
    //     }

    //     // instantiate batch and predicted output matrices
    //     tcnn::GPUMatrix<float> batch(batch_size, image_size);
    //     tcnn::GPUMatrix<float> predicted_output(batch_size, image_size);
    //     tcnn::GPUMatrix<float> poses(batch_size, 16);
        
    //     float* batch_dev = batch.data();
    //     float* predicted_output_dev = predicted_output.data();
    //     float* poses_dev = poses.data();

    //     int current_batch_size = 0;
    //     int current_batch = 0;
    //     for (int i = 0; i < images.size(); ++i) {
    //         //std::printf("Adding image %d to batch %d\n", i, current_batch);
    //         //std::memcpy(batch_host.data() + current_batch_size * image_size, images[i], image_size * sizeof(float));
    //         cudaMemcpyAsync(batch_dev + current_batch_size * image_size, images[i], image_size * sizeof(float), cudaMemcpyHostToDevice, streams[current_batch]);
    //         current_batch_size += 1;
    //         if (current_batch_size >= batch_size) {
    //             current_batch_size = 0;
    //             //printf("Launching kernel for batch %d current_batch\n", current_batch);
    //             print_batch<<<1, 1, 0, streams[current_batch]>>>(batch_dev, batch_size, image_size);
    //             //model.trainer->training_step(predicted_output, batch, &loss);
    //             current_batch ++; 
    //         }
    //     }
    //     if (current_batch_size > 0) {
    //         std::printf("Launching kernel for batch %d current_batch\n", current_batch);
    //         print_batch<<<1, 1, 0, streams[current_batch]>>>(batch_dev, batch_size, image_size);
    //         cudaStreamSynchronize(streams[current_batch]);
    //         current_batch++;
    //     }
    // }
    // std::vector<OptixAabb> grid = make_grid(initial_grid_resolution);
    // float transform_matrix[] = {
    //     -0.9999021887779236,0.004192245192825794,-0.013345719315111637,-0.05379832163453102,
    //     -0.013988681137561798, -0.2996590733528137, 0.95394366979599, 3.845470428466797,
    //     -4.656612873077393e-10, 0.9540371894836426, 0.29968830943107605, 1.2080823183059692,
    //     0.0, 0.0, 0.0, 1.0};
    
    std::string ptx_filename = BUILD_DIR "/ptx/optixPrograms.ptx";
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

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
    
    // /home/tsaluru/optix/rtx_compute_samples/build/resources/cow.obj
    std::string obj_file = "wavelet.txt";
    std::cout << "Building Acceleration Structure \n";
    std::vector<float3> vertices;
    std::vector<uint3> triangles;
    OptixAabb aabb_box = rtx_dataholder->buildAccelerationStructure(obj_file, vertices, triangles);
    std::cout << "Done Building Acceleration Structure \n";
    // Allocating GPU buffers for Params
    // calculate delta
    float3 delta = make_float3((aabb_box.maxX - aabb_box.minX) / width,
                                (aabb_box.maxY - aabb_box.minY) / height,
                                (aabb_box.maxZ - aabb_box.minZ) / depth);
    float3 min_point = make_float3(aabb_box.minX, aabb_box.minY, aabb_box.minZ);
    float3 max_point = make_float3(aabb_box.maxX, aabb_box.maxY, aabb_box.maxZ);
   
    float *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_output, width * height * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, width * height * sizeof(float)));

    // float3 *d_start_points;
    // float3 *d_end_points;
    // CUDA_CHECK(cudaMalloc((void **)&d_start_points, image_width * image_height * width * depth * height * sizeof(float3)));
    // CUDA_CHECK(cudaMalloc((void **)&d_end_points, image_width * image_height * width * depth * height * sizeof(float3)));
    // CUDA_CHECK(cudaMemset(d_start_points, -1, image_width * image_height * width * depth * height * sizeof(float3)));
    // CUDA_CHECK(cudaMemset(d_end_points, -1, image_width * image_height * width * depth * height * sizeof(float3)));

    // int numPrimitives = grid.size();
    // OptixAabb* d_aabbBuffer;
    // cudaMalloc(&d_aabbBuffer, sizeof(OptixAabb) * numPrimitives);
    // cudaMemcpy(d_aabbBuffer, grid.data(), sizeof(OptixAabb) * numPrimitives,
    //   cudaMemcpyHostToDevice);
    
    // Algorithmic parameters and data pointers used in GPU program
    Params params;
    // params.transform_matrix = transform_matrix;
    params.min_point = min_point;
    params.max_point = max_point;
    params.delta = delta;
    params.handle = rtx_dataholder->gas_handle;
    params.width = width;
    params.height = height;
    params.depth = depth;
    params.output = d_output;
    // params.start_points = d_start_points;
    // params.end_points = d_end_points;

    Params *d_param;
    CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
    CUDA_CHECK(
        cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));

    const OptixShaderBindingTable &sbt_ray_march = rtx_dataholder->sbt_ray_march;
    std::cout << "Launching Ray Tracer in Ray Marching Mode \n";
    OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline_ray_march, stream,
                            reinterpret_cast<CUdeviceptr>(d_param),
                            sizeof(Params), &sbt_ray_march, width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemset(d_output, 0, width * height * sizeof(float)));
    const OptixShaderBindingTable &sbt_ray_sample =
        rtx_dataholder->sbt_ray_sample;
    std::cout << "Launching Ray Tracer in Ray Sample Mode \n";
    OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline_ray_sample, stream,
                            reinterpret_cast<CUdeviceptr>(d_param),
                            sizeof(Params), &sbt_ray_march, width, height,
                            depth));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "Writing output to file \n";
    float *h_output = new float[width * height];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, width * height * sizeof(float),
                           cudaMemcpyDeviceToHost));
                        
    stbi_write_png("test.png", width, height, 1, h_output, width);
    std::cout << "Cleaning up ... \n";
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_param));
    delete rtx_dataholder;
    

    
    return 0;
}
