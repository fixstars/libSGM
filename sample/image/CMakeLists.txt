cmake_minimum_required(VERSION 3.1)

find_package(CUDA REQUIRED)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

# test apps
find_package(OpenCV REQUIRED core) # dummy
if(OpenCV_VERSION VERSION_LESS 3.0)
	find_package(OpenCV REQUIRED core highgui imgproc contrib)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
# end test apps

include_directories(../../include)

CUDA_ADD_EXECUTABLE(stereo_test stereosgm_image.cpp ${CUDA_SRC})


TARGET_LINK_LIBRARIES(stereo_test sgm ${CUDA_LIBRARIES} ${OpenCV_LIBS})
