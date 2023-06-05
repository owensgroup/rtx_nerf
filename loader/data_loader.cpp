#include "data_loader.h"
#include "transform.h"
#include <sstream>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <numeric>

std::vector<DataLoader> load_data(SceneType type, SceneName name) {
    std::string directory;
    std::string name;
    switch (name) {
        case SceneName::LEGO:
            name = "lego/";
            break;
        case SceneName::FERN:
            name = "fern/";
            break;
    }

    switch (type) {
        case SceneType::LLFF:
            directory = "/data/nerf_llff_data/" + name;
            break;
        case SceneType::SYNTHETIC:
            directory = "/data/nerf_synthetic/" + name;
            load_synthetic_data(directory);
            break;
    }
}


std::map<std::string, DataLoader> load_synthetic_data(std::string path) {
    // Load training JSON in directory root
    std::vector<std::string> strings = { "train", "val", "test" };
    std::map<std::string, DataLoader> loaders;
    std::vector<DataLoader> loaders;
    for (const auto& string : strings) {
        std::string filename = directory + "/transforms_" + name + ".json";
        DataLoader loader = loader_from_json(filename);
        loaders[string] = loader;
    }

    return loaders;
}

DataLoader loader_from_json(std::string file_path) {
    std::ifstream_file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open transform JSON file: " << filename << std::endl;
        exit(1);
    }

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> poses;

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

        // Create a cv::Mat to store the transform matrix
        cv::Mat transform(4, 4, CV_32FC1);
        
        // Fill the transform matrix with the values from the JSON
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                transform.at<float>(i, j) = transform_matrix[i][j].asFloat();
            }
        }

        // Load the image using the file path
        filepath = directory + "/" + file_path;
        cv::Mat image = cv::imread("image.png", cv::IMREAD_COLOR);

        // Check if the image was loaded successfully
        if (image.empty()) {
            std::cerr << "Failed to load the image." << std::endl;
            return 1;
        }

        // Convert image to float32 range from 0 to 1
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F, 1.0 / 255.0);

        // Add the image to the vector of training images
        training_images.push_back(image);
        poses.push_back(transform);
    }

    float focal = .5 * image.cols / std::tan(.5 * camera_angle_x);
    int height = image.rows;
    int width = image.cols;
    int batch_size = 64;

    return DataLoader(images, poses, focal, height, width, batch_size, num_workers);
    
}

// DataLoader constructor
DataLoader::DataLoader(std::vector<cv::Mat> images, std::vector<cv::Mat> poses, float focal, int height, int width, int batch_size, int num_workers) {
    this->images = images;
    this->poses = poses;
    this->focal = focal;
    this->height = height;
    this->width = width;
    this->batch_size = batch_size;
    this->num_workers = num_workers;
    this->order = std::vector<int>(images.size());
    std::iota(this->order.begin(), this->order.end(), 0);
    this->curr_index = 0;
    this->num_images = images.size();
}

std::vector<cv::Mat> DataLoader::get_batch() {

}


std::vector<std::vector<Ray>> get_rays(const cv::Mat& image, float focalLength, const cv::Mat& cameraToWorld)
{
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    std::vector<std::vector<Ray>> rays(imageHeight, std::vector<Ray>(imageWidth));

    float aspectRatio = static_cast<float>(imageWidth) / imageHeight;
    float fovY = 2.0f * std::atan(0.5f * static_cast<float>(imageHeight) / focalLength);
    float fovX = 2.0f * std::atan(0.5f * aspectRatio * std::tan(0.5f * fovY));

    float centerX = (imageWidth - 1) / 2.0f;
    float centerY = (imageHeight - 1) / 2.0f;

    for (int y = 0; y < imageHeight; ++y)
    {
        for (int x = 0; x < imageWidth; ++x)
        {
            float normalizedX = (x - centerX) / centerX;
            float normalizedY = (y - centerY) / centerY;

            float directionX = normalizedX * std::tan(0.5f * fovX);
            float directionY = -normalizedY * std::tan(0.5f * fovY);
            float directionZ = -1.0f;
            cv::Vec3f direction(directionX, directionY, directionZ);
            direction = cameraToWorld(cv::Range(0, 3), cv::Range(0, 3)) * direction;
            direction = direction / cv::norm(direction);

            cv::Vec3f origin(0.0f, 0.0f, 0.0f);
            origin = cameraToWorld(cv::Range(0, 3), cv::Range(3, 4)) + origin;

            rays[y][x] = Ray(origin, direction);
        }
    }

    return rays;
}