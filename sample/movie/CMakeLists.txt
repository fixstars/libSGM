cmake_minimum_required(VERSION 3.1)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

project(sgm_demo)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS ${CMAKE_MODULE_PATH})

FIND_PACKAGE(CUDA  REQUIRED)

find_package(OpenCV REQUIRED core) # dummy
if(OpenCV_VERSION VERSION_LESS 3.0) 
	find_package(OpenCV REQUIRED core highgui imgproc)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)

find_package(GLFW REQUIRED)

include_directories(../../include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(stereo_movie
stereosgm_movie.cpp
demo.cpp
renderer.cpp
cuda_gl.cu
)


TARGET_LINK_LIBRARIES(stereo_movie sgm  ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${GLEW_LIBRARIES} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
