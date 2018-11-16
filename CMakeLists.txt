cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

option(ENABLE_ZED_DEMO "Build a Demo using ZED Camera" OFF)
option(ENABLE_SAMPLES  "Build samples" OFF)
option(ENABLE_TESTS    "Test library" OFF)
option(LIBSGM_SHARED   "Build a shared library" OFF)
option(WITH_OPENCV     "Make library compatible with cv::cuda::GpuMat of OpenCV" OFF)

if(WIN32)
  set(ZED_SDK_LIB "C:\\Program Files (x86)\\ZED SDK\\lib\\sl_zed64.lib" CACHE STRING "ZED SDK library(sl_zed**.llb) path.")
  set(ZED_SDK_INCLUDE_DIR "C:\\Program Files (x86)\\ZED SDK\\include" CACHE STRING "ZED SDK include path.")
else()
  set(ZED_SDK_LIB "/usr/local/zed/lib/libsl_zed.so" CACHE STRING "ZED SDK library(sl_zed**.llb) path.")
  set(ZED_SDK_INCLUDE_DIR "/usr/local/zed/include" CACHE STRING "ZED SDK include path.")
endif()

project(libSGM VERSION 2.2.0)

if(WITH_OPENCV)
	find_package(OpenCV REQUIRED core)
	include_directories(${OpenCV_INCLUDE_DIRS})
endif()

configure_file(${CMAKE_SOURCE_DIR}/include/libsgm_config.h.in
               ${CMAKE_SOURCE_DIR}/include/libsgm_config.h
)

add_subdirectory(src)

if(ENABLE_SAMPLES)
    add_subdirectory(sample/image)
    add_subdirectory(sample/movie)
    add_subdirectory(sample/reprojection)
    add_subdirectory(sample/benchmark)
    if(WITH_OPENCV)
        add_subdirectory(sample/image_cv_gpumat)
    endif()
endif()

if(ENABLE_TESTS)
	add_subdirectory(test)
endif()

if(ENABLE_ZED_DEMO)
	add_subdirectory(sample/zed)
endif()
