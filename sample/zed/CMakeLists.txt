cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../movie/cmake)

message(STATUS ${CMAKE_MODULE_PATH})

find_package(ZED 2 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

find_package(OpenCV REQUIRED core) # dummy
if(OpenCV_VERSION VERSION_LESS 3.0) 
	find_package(OpenCV REQUIRED core highgui imgproc)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

include_directories(../../include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

if(NOT WIN32)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()


CUDA_ADD_EXECUTABLE(zed_demo
	zed_demo.cpp
)


TARGET_LINK_LIBRARIES(zed_demo 
	sgm 
	${CUDA_LIBRARIES} ${CUDA_NPP_LIBRARIES_ZED}
	${OpenCV_LIBS} 
	${ZED_LIBRARIES}
)
