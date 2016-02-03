cmake_minimum_required(VERSION 2.8)

project(sgm_demo)

FIND_PACKAGE(CUDA  REQUIRED)

find_package(OpenCV REQUIRED core) # dammy 
if(OpenCV_VERSION VERSION_LESS 3.0) 
	find_package(OpenCV REQUIRED core highgui imgproc)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

find_package(GLEW REQUIRED)

# TODO GLFW3 find_package #find_package(glfw REQUIRED)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	#SET(GLFW3_INCLUDE_DIRS "")
	SET(GLFW3_LIBRARY /usr/local/lib/libglfw3.a Xxf86vm X11 Xinerama Xi Xrandr Xcursor GL dl pthread)
else() # for Windows
	SET(GLFW3_INCLUDE_DIRS "C:\\libs\\opengl\\include") # fix path
	SET(GLFW3_LIBRARY "C:\\libs\\opengl\\lib\\glfw3.lib") # fix path
endif()

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
    #SET(CMAKE_BUILD_TYPE Debug CACHE STRING "CMake build type" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
	EXECUTE_PROCESS(COMMAND uname -a OUTPUT_VARIABLE SYSTEM_FULLNAME)
	if("${SYSTEM_FULLNAME}" MATCHES "tegra-ubuntu")
		SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_32 -O2") # for jetson tk1
	else()
		SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -O2")
	endif()
else() # for Windows
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50")
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4819\"") # disable warning c4819 (file encoding warning)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo") # debug info
endif()


include_directories(../../include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLFW3_INCLUDE_DIRS})


CUDA_ADD_EXECUTABLE(stereo_movie
stereosgm_movie.cpp
demo.cpp
renderer.cpp
cuda_gl.cu
)

TARGET_LINK_LIBRARIES(stereo_movie stereosgm  ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${OpenCV_LIBS} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})
