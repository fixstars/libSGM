cmake_minimum_required(VERSION 3.0)
project(sgm)

find_package(CUDA REQUIRED)

include_directories(../include)

file(GLOB SOURCES "*.cu" "*.cpp")

set(CUDA_NVCC_FLAGS
    -std=c++11
    -arch=sm_61
    -lineinfo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

cuda_add_library(sgm ${SOURCES})
