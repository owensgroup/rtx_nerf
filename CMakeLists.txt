cmake_minimum_required(VERSION 3.12)
include(FetchContent)
project(optix_nerf LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find the JsonCpp package
FetchContent_Declare(
  JSONCPP
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
  GIT_TAG        6aba23f4a8628d599a9ef7fa4811c4ff6e4070e2 # 1.9.3
)

FetchContent_Declare(
  OpenCV
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG        725e440d278aca07d35a5e8963ef990572b07316 # 1.9.3
)
FetchContent_MakeAvailable(JSONCPP) # Brings in jsoncpp_lib
FetchContent_MakeAvailable(OpenCV)

set(CMAKE_CUDA_ARCHITECTURES 75)

get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
message("${JSON_INC_PATH}")
find_package(OptiX REQUIRED)

add_compile_definitions(TCNN_MIN_GPU_ARCH=75)

include_directories(${JSON_INC_PATH}
                    ${CMAKE_SOURCE_DIR}/loader
                    ${OptiX_INCLUDE}
                    ${CMAKE_BINARY_DIR}
                    ${OpenCV_SOURCE_DIR}/include
                    ${OpenCV_SOURCE_DIR}/modules/core/include/
                    ${OpenCV_SOURCE_DIR}/modules/calib3d/include/
                    ${OpenCV_SOURCE_DIR}/modules/features2d/include/
                    ${OpenCV_SOURCE_DIR}/modules/flann/include/
                    ${OpenCV_SOURCE_DIR}/modules/gapi/include/
                    ${OpenCV_SOURCE_DIR}/modules/imgcodecs/include/
                    ${OpenCV_SOURCE_DIR}/modules/imgproc/include/
                    ${OpenCV_SOURCE_DIR}/modules/ml/include/
                    ${OpenCV_SOURCE_DIR}/modules/objdetect/include/
                    ${OpenCV_SOURCE_DIR}/modules/photo/include/
                    ${OpenCV_SOURCE_DIR}/modules/stitching/include/
                    ${OpenCV_SOURCE_DIR}/modules/video/include/
                    ${OpenCV_SOURCE_DIR}/modules/videoio/include/
                    ${OpenCV_SOURCE_DIR}/modules/dnn/include/
                    ${OpenCV_SOURCE_DIR}/modules/highgui/include/
                    ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/include
                    ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/dependencies/
                    ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/dependencies/fmt/include)


# Add your source files
set(SOURCES
    main.cpp
    loader/data_loader.cpp
    loader/transform_loader.cpp
)

# Create the executable
add_executable(optx_nerf ${SOURCES})

set_target_properties(optx_nerf PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(optx_nerf PROPERTIES CUDA_ARCHITECTURES 75)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

# Link against the JsonCpp library
target_link_libraries(optx_nerf jsoncpp_lib ${OptiX_LIBRARY} 
            ${CUDA_LIBRARIES}
            ${OpenCV_LIBS} 
            ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/build/libtiny-cuda-nn.a
            ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/build/dependencies/fmt/libfmt.a)
