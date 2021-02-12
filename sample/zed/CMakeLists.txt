cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../movie/cmake)

message(STATUS ${CMAKE_MODULE_PATH})

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

include_directories(../../include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

if(NOT WIN32)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()


CUDA_ADD_EXECUTABLE(zed_demo
	zed_demo.cpp
)

string(REGEX REPLACE [[; +]] [[;]] CUDA_NPP_LIBRARIES_ZED "${CUDA_NPP_LIBRARIES_ZED}")

TARGET_LINK_LIBRARIES(zed_demo 
	sgm 
	${CUDA_LIBRARIES} ${CUDA_NPP_LIBRARIES_ZED}
	${OpenCV_LIBS} 
	${ZED_LIBRARIES}
)
