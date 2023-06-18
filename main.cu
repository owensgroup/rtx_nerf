#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>

#include "stdio.h"


#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "optix.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "tiny-cuda-nn/common.h"
#include "tiny-cuda-nn/gpu_matrix.h"
#include <json/json.h>
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


//auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);

#define EPOCHS 10
#define BATCH_SIZE tcnn::batch_size_granularity
#define DATASET_SIZE 1000
int main() {
    // load data from files
    // TODO: take images and poses from json and load into DataLoader
    int num_epochs = EPOCHS;
    int batch_size = BATCH_SIZE;
    size_t image_size = 800*800*4;

    std::vector<ImageDataset> datasets = load_data(SceneType::SYNTHETIC, SceneName::LEGO);
    std::printf("BATCH SIZE GRANULARITY %d \n", tcnn::batch_size_granularity);
    std::printf("IMAGES IN TRAINING DATASET: %d\n", datasets[0].images.size());
    // calculate the number of training iterations in an epoch given the batch size
    int num_batches = datasets[0].images.size() / batch_size + 1;
    // calculate the total number of training iterations
    // get batch from dataloaders
    // get training dataset from datasets
    std::vector<float*> images = datasets[0].images;
    std::vector<float*> poses = datasets[0].poses;
    
    auto training_set = datasets[0];
    
    for (int j = 0; j < num_epochs; ++j) {
        std::printf("Start training loop epoch %d\n", j);

        // instantiate number of streams according to the number of batches
        std::vector<cudaStream_t> streams(num_batches);
        for (auto& stream : streams) {
            cudaStreamCreate(&stream);
        }

        // instantiate batch and predicted output matrices
        tcnn::GPUMatrix<float> batch(batch_size, image_size);
        tcnn::GPUMatrix<float> predicted_output(batch_size, image_size);
        tcnn::GPUMatrix<float> poses(batch_size, 16);
        
        float* batch_dev = batch.data();
        float* predicted_output_dev = predicted_output.data();
        float* poses_dev = poses.data();

        int current_batch_size = 0;
        int current_batch = 0;
        for (int i = 0; i < images.size(); ++i) {
            //std::printf("Adding image %d to batch %d\n", i, current_batch);
            //std::memcpy(batch_host.data() + current_batch_size * image_size, images[i], image_size * sizeof(float));
            cudaMemcpyAsync(batch_dev + current_batch_size * image_size, images[i], image_size * sizeof(float), cudaMemcpyHostToDevice, streams[current_batch]);
            current_batch_size += 1;
            if (current_batch_size >= batch_size) {
                current_batch_size = 0;
                //printf("Launching kernel for batch %d current_batch\n", current_batch);
                print_batch<<<1, 1, 0, streams[current_batch]>>>(batch_dev, batch_size, image_size);
                //model.trainer->training_step(predicted_output, batch, &loss);
                current_batch ++; 
            }
        }
        if (current_batch_size > 0) {
            std::printf("Launching kernel for batch %d current_batch\n", current_batch);
            print_batch<<<1, 1, 0, streams[current_batch]>>>(batch_dev, batch_size, image_size);
            cudaStreamSynchronize(streams[current_batch]);
            current_batch++;
        }
    }

   
    
    return 0;
}
