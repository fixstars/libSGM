cmake_minimum_required(VERSION 2.8)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

option(ENABLE_ZED_DEMO "Build a Demo using ZED Camera" OFF)

if(WIN32)
  set(ZED_SDK_LIB "C:\\Program Files (x86)\\ZED SDK\\lib\\sl_zed64.lib" CACHE STRING "ZED SDK library(sl_zed**.llb) path.")
  set(ZED_SDK_INCLUDE_DIR "C:\\Program Files (x86)\\ZED SDK\\include" CACHE STRING "ZED SDK include path.")
else()
  set(ZED_SDK_LIB "/usr/local/zed/lib/libsl_zed.so" CACHE STRING "ZED SDK library(sl_zed**.llb) path.")
  set(ZED_SDK_INCLUDE_DIR "/usr/local/zed/include" CACHE STRING "ZED SDK include path.")
endif()

project(libSGM)

add_subdirectory(src)

add_subdirectory(sample/image)
add_subdirectory(sample/movie)

if(ENABLE_ZED_DEMO)
	add_subdirectory(sample/zed)
endif()
