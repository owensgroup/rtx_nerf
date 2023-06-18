#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


enum class SceneType { LLFF, SYNTHETIC };
enum class SceneName { LEGO, FERN };

std::vector<DataLoader> load_data(SceneType type, SceneName name);
std::map<std::string, DataLoader> load_synthetic_data(std::string path)
std::vector<DataLoader> load_llff_data(std::string path);
DataLoader loader_from_json(std::string file_path);






class DataLoader {
public:
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> poses;
    float focal_length;
    int height;
    int width;
    int batch_size;
    int num_workers;
    DataLoader(std::vector<cv::Mat> images, std::vector<cv::Mat> poses, 
        float focal_length, int height, int width, int batch_size, int num_workers);
    
    // gets the next batch for this epoch
    std::vector<cv::Mat> getBatch();
    bool hasNextBatch();
    
private:
    std::vector<int> order;
    int curr_index;
    int num_images;
    // shuffle gets a random order of indices to batch data
    shuffle();

};

struct Ray
{
    cv::Vec3f origin;
    cv::Vec3f direction;

    Ray(const cv::Vec3f& origin, const cv::Vec3f& direction)
        : origin(origin), direction(direction)
    {
    }
};

std::vector<std::vector<Ray>> get_rays(const cv::Mat& image, float focalLength, const cv::Mat& cameraToWorld);
#endif

// cv::Mat trans_t(float t) {
//     cv::Mat mat = cv::Mat::eye(4, 4, CV_32F);
//     mat.at<float>(2, 3) = t;
//     return mat;
// }

// cv::Mat rot_phi(float phi) {
//     cv::Mat mat = cv::Mat::eye(4, 4, CV_32F);
//     mat.at<float>(1, 1) = std::cos(phi);
//     mat.at<float>(1, 2) = -std::sin(phi);
//     mat.at<float>(2, 1) = std::sin(phi);
//     mat.at<float>(2, 2) = std::cos(phi);
//     return mat;
// }

// cv::Mat rot_theta(float theta) {
//     cv::Mat mat = cv::Mat::eye(4, 4, CV_32F);
//     mat.at<float>(0, 0) = std::cos(theta);
//     mat.at<float>(0, 2) = -std::sin(theta);
//     mat.at<float>(2, 0) = std::sin(theta);
//     mat.at<float>(2, 2) = std::cos(theta);
//     return mat;
// }

// cv::Mat pose_spherical(float theta, float phi, float radius) {
//     cv::Mat c2w = trans_t(radius);
//     c2w = rot_phi(phi / 180.0f * CV_PI) * c2w;
//     c2w = rot_theta(theta / 180.0f * CV_PI) * c2w;
//     cv::Mat flip = (cv::Mat_<float>(4, 4) << -1, 0, 0, 0,
//                                             0, 0, 1, 0,
//                                             0, 1, 0, 0,
//                                             0, 0, 0, 1);
//     c2w = flip * c2w;
//     return c2w;
// }