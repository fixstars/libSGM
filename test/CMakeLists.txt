cmake_minimum_required(VERSION 3.0)
project(sgm-test)

find_package(CUDA REQUIRED)

add_subdirectory(googletest)

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(../include)
include_directories(../src)

file(GLOB TEST_SOURCES "*.cpp" "*.cu")

set(CUDA_NVCC_FLAGS
    -std=c++11
    -arch=sm_61
    -lineinfo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

cuda_add_executable(sgm-test ${TEST_SOURCES})
target_link_libraries(sgm-test sgm gtest)
