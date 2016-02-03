cmake_minimum_required(VERSION 2.8)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

project(stereo-sgm)

add_subdirectory(src)

add_subdirectory(sample/image)
