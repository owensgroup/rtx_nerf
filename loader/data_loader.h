#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>



enum class SceneType { LLFF, SYNTHETIC };
enum class SceneName { LEGO, FERN };

struct ImageDataset {
  std::vector<float*> images;
  std::vector<float*> poses;
  float focal;
  unsigned int image_width;
  unsigned int image_height;
  unsigned int image_channels;
};

ImageDataset load_images_json(std::string basename, std::string s);
std::vector<ImageDataset> load_synthetic_data(std::string directory);
std::vector<ImageDataset> load_data(SceneType type, SceneName name);

#endif