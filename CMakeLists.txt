cmake_minimum_required(VERSION 2.8)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

project(libSGM)

add_subdirectory(src)

option(ENABLE_ZED_DEMO "Build a Demo using ZED Camera" OFF)

add_subdirectory(sample/image)
add_subdirectory(sample/movie)

if(ENABLE_ZED_DEMO)
	if (NOT DEFINED ZED_SDK_LIB)
		set(ZED_SDK_LIB "C:\\Program Files (x86)\\ZED SDK\\lib\\sl_zed64.lib")
	endif()
	if (NOT DEFINED ZED_SDK_INCLUDE_DIR)
		set(ZED_SDK_INCLUDE_DIR "C:\\Program Files (x86)\\ZED SDK\\include")
	endif()

	add_subdirectory(sample/zed)
endif()