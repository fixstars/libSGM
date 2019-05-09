cmake_minimum_required(VERSION 3.1)

find_package(CUDA REQUIRED)

if(NOT DEFINED AUTO_DETECT_ARCH OR NOT "${AUTO_DETECT_ARCH}"
   OR ${CMAKE_VERSION} VERSION_LESS 3.7)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH}")
	message(STATUS "User specified variable CUDA_ARCH: ${CUDA_ARCH} is used.")
else()
	CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
	LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
	message(STATUS "Automatically generated flag \"${ARCH_FLAGS}\" is used.")
endif()

include_directories(../include)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()


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
