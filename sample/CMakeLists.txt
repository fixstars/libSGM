cmake_minimum_required(VERSION 3.17)

project(samples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# required packages
find_package(OpenCV REQUIRED)

# sample image
add_executable(stereosgm_image stereosgm_image.cpp)
target_include_directories(stereosgm_image PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereosgm_image sgm ${OpenCV_LIBS})

# sample movie
add_executable(stereosgm_movie stereosgm_movie.cpp)
target_include_directories(stereosgm_movie PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereosgm_movie sgm ${OpenCV_LIBS})

# sample benchmark
add_executable(stereosgm_benchmark stereosgm_benchmark.cpp)
target_include_directories(stereosgm_benchmark PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereosgm_benchmark sgm ${OpenCV_LIBS})

# sample reprojection
add_executable(stereosgm_reprojection stereosgm_reprojection.cpp)
target_include_directories(stereosgm_reprojection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereosgm_reprojection sgm ${OpenCV_LIBS})

# sample image with cv::GpuMat
if(BUILD_OPENCV_WRAPPER)
	add_executable(stereosgm_image_cv_gpumat stereosgm_image_cv_gpumat.cpp)
	target_include_directories(stereosgm_image_cv_gpumat PRIVATE ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(stereosgm_image_cv_gpumat sgm ${OpenCV_LIBS})
endif()

# sample ZED camera
if(ENABLE_ZED_DEMO)
	if(WIN32)
		set(ZED_SDK_LIB "C:\\Program Files (x86)\\ZED SDK\\lib\\sl_zed64.lib" CACHE STRING "ZED SDK library(sl_zed**.llb) path.")
		set(ZED_SDK_INCLUDE_DIR "C:\\Program Files (x86)\\ZED SDK\\include" CACHE STRING "ZED SDK include path.")
	else()
		set(ZED_SDK_LIB "/usr/local/zed/lib/libsl_zed.so" CACHE STRING "ZED SDK library(sl_zed**.llb) path.")
		set(ZED_SDK_INCLUDE_DIR "/usr/local/zed/include" CACHE STRING "ZED SDK include path.")
	endif()

	find_package(ZED 3 REQUIRED)
	string(REGEX REPLACE [[; +]] [[;]] CUDA_NPP_LIBRARIES_ZED "${CUDA_NPP_LIBRARIES_ZED}")

	add_executable(stereosgm_zed stereosgm_zed.cpp)
	target_include_directories(stereosgm_zed PRIVATE ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS})
	target_link_directories(stereosgm_zed PRIVATE ${ZED_LIBRARY_DIR})
	target_link_libraries(stereosgm_zed sgm ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_NPP_LIBRARIES_ZED})
endif()
