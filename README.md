# rtx_nerf
An implementation of NeRF acceleration using RTX cores to compute ray-grid intersections

# Setup
## Requirements
You will need:
- a Nvidia GPU with compute capability>7.0.
- Optix 7.7
- CUDA

## Instructions
In order to setup the repository, first download Optix. Then from the root run CMake
```
mkdir build && cd build
cmake ../ -DOPTIX_HOME=[/path/to/optix/]
make
```

In order to build in debug mode
```
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DOPTIX_HOME=[/path/to/optix/]
make
```
