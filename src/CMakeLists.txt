cmake_minimum_required(VERSION 3.1)

find_package(CUDA REQUIRED)

include_directories(../include)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

file(GLOB STEREOSRCS "*.cu" "*.cpp")

if(LIBSGM_SHARED)
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} SHARED)
	target_link_libraries(sgm ${CUDA_LIBRARIES})
	if(BUILD_OPENCV_WRAPPER)
		target_link_libraries(sgm ${OpenCV_LIBS})
	endif()
else()
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} STATIC)
endif()

install(
	TARGETS sgm
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(
	DIRECTORY ${CMAKE_SOURCE_DIR}/include
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
	FILES ${CMAKE_SOURCE_DIR}/FindLibSGM.cmake
	DESTINATION ${CMAKE_INSTALL_PREFIX}
)
