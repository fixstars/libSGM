cmake_minimum_required(VERSION 3.1)

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../../include)

cuda_add_executable(stereo_test stereosgm_image.cpp)
target_link_libraries(stereo_test sgm ${CUDA_LIBRARIES} ${OpenCV_LIBS})
