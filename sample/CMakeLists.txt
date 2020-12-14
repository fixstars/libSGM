cmake_minimum_required(VERSION 3.18)
project(libsgm_samples)

###############################  CMake Modules  #############################################
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
###############################  CMake Modules  #############################################

###############################     Modules     #############################################
include(HandlePDB)
include(HandleDLL)
###############################     Modules     #############################################

add_subdirectory(image)
add_subdirectory(movie)
add_subdirectory(reprojection)
add_subdirectory(benchmark)

if(BUILD_OPENCV_WRAPPER)
    add_subdirectory(image_cv_gpumat)
endif()

if(ENABLE_ZED_DEMO)
	add_subdirectory(zed)
endif()