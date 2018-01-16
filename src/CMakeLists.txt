find_package(CUDA REQUIRED)

include_directories(../include)

find_package(CUDA REQUIRED)

if(CUDA_VERSION VERSION_LESS "9.0")
	set(NPP_LIBRARIES "${CUDA_nppi_LIBRARY}")
else()
	set(NPP_LIBRARIES "${CUDA_nppif_LIBRARY}")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

SET(STEREOSRCS
	census.cu
	matching_cost.cu
	scan_scost.cu
	winner_takes_all.cu
	median_filter.cu
	check_consistency.cu
	cuda_utils.cu
)

if(LIBSGM_SHARED)
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} SHARED)
	target_link_libraries(sgm ${CUDA_LIBRARIES} ${NPP_LIBRARIES})
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
