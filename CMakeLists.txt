cmake_minimum_required(VERSION 3.18)

option(ENABLE_ZED_DEMO      "Build a Demo using ZED Camera" OFF)
option(ENABLE_SAMPLES       "Build samples" OFF)
option(ENABLE_TESTS         "Test library" OFF)
option(LIBSGM_SHARED        "Build a shared library" OFF)
option(BUILD_OPENCV_WRAPPER "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" OFF)
option(USE_HIP              "Build the GPU kernels with ROCm/HIP instead of CUDA" OFF)

# For HIP the architecture is handled by enable_language(HIP) in src/, which
# auto-detects the host GPU; -DCMAKE_HIP_ARCHITECTURES overrides it.
if(NOT USE_HIP)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "52;61;72;75;86")
    endif()
endif()

project(libSGM VERSION 3.1.0)

configure_file(
${PROJECT_SOURCE_DIR}/include/libsgm_config.h.in
${PROJECT_SOURCE_DIR}/include/libsgm_config.h
)

add_subdirectory(src)

if(ENABLE_SAMPLES)
	add_subdirectory(sample)
endif()

if(ENABLE_TESTS)
	add_subdirectory(test)
endif()
