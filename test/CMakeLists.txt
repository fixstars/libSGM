cmake_minimum_required(VERSION 3.1)
project(sgm-test)

find_package(CUDA REQUIRED)

if (MSVC)
	option(gtest_force_shared_crt "Force Gmock to use standard compiler flags" ON)
endif()

add_subdirectory(googletest)

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(../include)
include_directories(../src)

file(GLOB TEST_SOURCES "*.cpp" "*.cu")

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo")

cuda_add_executable(sgm-test ${TEST_SOURCES})
target_link_libraries(sgm-test sgm gtest)
