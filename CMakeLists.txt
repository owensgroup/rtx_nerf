cmake_minimum_required(VERSION 3.12)
include(FetchContent)
project(optx_nerf LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# Find the JsonCpp package
FetchContent_Declare(
  JSONCPP
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
  GIT_TAG        6aba23f4a8628d599a9ef7fa4811c4ff6e4070e2 # 1.9.3
)

# FetchContent_Declare(
#   OpenCV
#   GIT_REPOSITORY https://github.com/opencv/opencv.git
#   GIT_TAG        725e440d278aca07d35a5e8963ef990572b07316 # 1.9.3
# )
FetchContent_MakeAvailable(JSONCPP) # Brings in jsoncpp_lib
# FetchContent_MakeAvailable(OpenCV)

set(CMAKE_CUDA_ARCHITECTURES 70)

get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
message("${JSON_INC_PATH}")
find_package(OptiX7 REQUIRED)


add_compile_definitions(TCNN_MIN_GPU_ARCH=70)

include_directories(${JSON_INC_PATH}
                    ${CMAKE_SOURCE_DIR}/loader
                    ${CMAKE_SOURCE_DIR}/rtx/include
                    ${CMAKE_SOURCE_DIR}/common/
		    ${OptiX7_INCLUDE_DIRS}
                    ${CMAKE_BINARY_DIR}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/include
                    ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/dependencies/
                    ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/dependencies/fmt/include)
include(PTXUtilities)

# Add your source files
set(SOURCES
    main.cu
    rtx/src/rtxFunctions.cpp
    loader/data_loader.cpp
    common/common.cpp
)

# Create the executable
add_executable(optx_nerf ${SOURCES})

set_target_properties(optx_nerf PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(optx_nerf PROPERTIES CUDA_ARCHITECTURES 70)
#set_source_files_properties(test.cu PROPERTIES LANGUAGE CUDA)

message("${OpenCV_LIBS}")
# Link against the JsonCpp library
target_link_libraries(optx_nerf jsoncpp_lib cuda ${OptiX_LIBRARY} 
            ${CUDA_LIBRARIES}
            ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/build/libtiny-cuda-nn.a
            ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/build/dependencies/fmt/libfmt.a
            ${CMAKE_DL_LIBS})

add_ptx_targets(optx_nerf optixPrograms)
