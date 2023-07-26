#include "data_loader.h"
#include "stb_image.h"
#include <iostream>
#include <json/json.h>
#include <fstream>
#include <cmath>

class ProgressBar {
    const int barWidth = 70;

public:
    void update(double progress) {
        std::cout << "\n";  // Reset to the start of the line
        
        // Print the loading message
        std::cout << "Loading image [" + std::to_string(int(progress * 100.0)) + "/100]" + "\n";

        // Print the progress bar
        std::cout << "[";
        int position = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < position) std::cout << "=";
            else if (i == position) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %";
        std::cout.flush();

        // Move the cursor up 2 lines for the next update
        std::cout << "\033[2A";  // ANSI escape code to move up 2 lines
    }
};

ImageDataset load_images_json(std::string basename, std::string s) {
    std::ifstream file(basename + "/transforms_" + s + ".json");
    if (!file.is_open()) {
        std::cerr << "Failed to open transform JSON file: " << basename + "/transforms_train.json" << std::endl;
        exit(1);
    }

    std::vector<float*> images;
    std::vector<float*> poses;

    std::cout << "Loading JSON..." << std::endl;
    Json::Value root;
    file >> root;
    float camera_angle_x = root["camera_angle_x"].asFloat();
    //std::cout << "Camera Angle X: " << camera_angle_x << std::endl;
    int width, height, channels_in_file;
    const Json::Value frames = root["frames"];
    ProgressBar bar;
    double i = 0;
    std::cout << "Loading images..." << std::endl;
    for (const auto& frame : frames) {
        std::string file_path = frame["file_path"].asString();
        float rotation = frame["rotation"].asFloat();
        const Json::Value transform_matrix = frame["transform_matrix"];
        bar.update(i);
        i += .01;
        // std::cout << "File Path: " << file_path << std::endl;
        // std::cout << "Rotation: " << rotation << std::endl;

        //std::printf("basename = %s\n", basename.c_str());
        // Load the image using the file path
        file_path = basename + "/" + file_path + ".png";
        //cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
        int desired_channels = 4;
        //std::printf("Loading image from %s\n", file_path.c_str());
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
    std::cout << "\033[2B" << std::endl;
    float focal = .5 * 800 / std::tan(.5 * camera_angle_x);
    ImageDataset dataset;
    dataset.images = images;
    dataset.poses = poses;
    dataset.focal = focal;
    dataset.image_width = width;
    dataset.image_height = height;
    dataset.image_channels = channels_in_file;
    return dataset;
}

std::vector<ImageDataset> load_synthetic_data(std::string directory) {
    // Load training JSON in directory root
    std::vector<std::string> strings = { "train", "val", "test" };
    std::vector<ImageDataset> datasets;
    for (const auto& string : strings) {
        ImageDataset dataset = load_images_json(directory, string);
        datasets.push_back(dataset);
        break;
    }

    return datasets;
}

std::vector<ImageDataset> load_data(SceneType type, SyntheticName name) {
    std::string directory;
    std::string filename;
    switch (name) {
        case SyntheticName::CHAIR:
            filename = "chair/";
            break;
        case SyntheticName::DRUMS:
            filename = "drums/";
            break;
        case SyntheticName::FICUS:
            filename = "ficus/";
            break;
        case SyntheticName::HOTDOG:
            filename = "hotdog/";
            break;
        case SyntheticName::LEGO:
            filename = "lego/";
            break;
        case SyntheticName::MATERIALS:
            filename = "fern/";
            break;
        case SyntheticName::MIC:
            filename = "mic/";
            break;
        case SyntheticName::SHIP:
            filename = "ship/";
            break;
    }

    switch (type) {
        case SceneType::LLFF:
            directory = "./data/nerf_llff_data/" + filename;
            break;
        case SceneType::SYNTHETIC:
            directory = "./data/nerf_synthetic/" + filename;
            return load_synthetic_data(directory);
            break;
    }
  return std::vector<ImageDataset>();
}