find_package(CUDA REQUIRED)

include_directories(../include)

find_package(CUDA REQUIRED)

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

if(BUILD_DYNAMIC_LIB)
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} SHARED)
	target_link_libraries(sgm ${CUDA_nppi_LIBRARY})
else()
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} STATIC)
endif()
