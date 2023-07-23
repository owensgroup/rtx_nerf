# rtx_nerf
An implementation of NeRF acceleration using RTX cores to compute ray-grid intersections

# Setup
## Requirements
You will need:
- a Nvidia GPU with compute capability>7.0 and driver version >=530.41.
- Optix 7.7
- CUDA
- CMake > 3.18

## Initial build and dependencies
In order to setup the repository, first download Optix. Then clone
the repo. You'l first need to setup submodules.
```
git submodule init
git submodule update --init --recursive
```

If your Linux doesn't already have them installed install the following:
```
sudo apt-get install build-essential
```

We then need to build tiny-cuda-nn (this takes time)
```
cd lib/tiny-cuda-nn
mkdir build && cd build
cmake ..
make -j
```

Now that the tiny-cuda-nn static library is built we can compile 
the rtx_nerf project.

Navigate to the project root, and run the following:
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
## Running the executables
First download the data from the [NeRF website](https://www.matthewtancik.com/nerf). 
Then create a data folder and extract the data.
