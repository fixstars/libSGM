cmake_minimum_required(VERSION 3.1)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED core) # dummy
if(OpenCV_VERSION VERSION_LESS 3.0)
	find_package(OpenCV REQUIRED core highgui imgproc contrib)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../../include)

cuda_add_executable(stereosgm_reprojection stereosgm_reprojection.cpp)
target_link_libraries(stereosgm_reprojection sgm ${CUDA_LIBRARIES} ${OpenCV_LIBS})