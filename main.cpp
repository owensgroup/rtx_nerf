#include <iostream>
#include <string>
#include <vector>


#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "optix.h"

#include "tiny-cuda-nn/common.h"
#include "tiny-cuda-nn/gpu_matrix.h"
#include <opencv2/opencv.hpp>
#include <json/json.h>
// #include "data_loader.h"
// #include "transform_loader.h"

int main() {
    

    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    optixInit();

    tcnn::GPUMatrix<float> training_batch_inputs(1024, 1024);

    return 0;
}
