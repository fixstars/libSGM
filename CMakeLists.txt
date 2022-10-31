cmake_minimum_required(VERSION 3.17)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()

option(ENABLE_ZED_DEMO      "Build a Demo using ZED Camera" OFF)
option(ENABLE_SAMPLES       "Build samples" OFF)
option(ENABLE_TESTS         "Test library" OFF)
option(LIBSGM_SHARED        "Build a shared library" OFF)
option(BUILD_OPENCV_WRAPPER "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" OFF)

project(libSGM VERSION 2.7.0)

configure_file(${PROJECT_SOURCE_DIR}/include/libsgm_config.h.in
               ${PROJECT_SOURCE_DIR}/include/libsgm_config.h
)

add_subdirectory(src)

if(ENABLE_SAMPLES)
	add_subdirectory(sample)
endif()

if(ENABLE_TESTS)
	add_subdirectory(test)
endif()
