cmake_minimum_required(VERSION 3.18)

set(LIBSGM_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBSGM_INCLUDE_DIR ${LIBSGM_ROOT_DIR}/include)

# create project
set(PROJECT_NAME sgm)
if(USE_HIP)
	project(${PROJECT_NAME} LANGUAGES CXX)
	# enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects
	# the host GPU and errors on a no-GPU build host.
	enable_language(HIP)
else()
	project(${PROJECT_NAME} LANGUAGES CXX CUDA)
endif()

# dependent packages
if(USE_HIP)
	find_package(hip REQUIRED)
else()
	find_package(CUDAToolkit REQUIRED)
endif()

if(BUILD_OPENCV_WRAPPER)
	find_package(OpenCV REQUIRED core)
endif()

# library type
set(SGM_LIB_TYPE STATIC)
if(LIBSGM_SHARED)
	set(SGM_LIB_TYPE SHARED)
endif()

# target configuration
file(GLOB SRCS ./*.cpp ./*.cu ./*.h* ${LIBSGM_INCLUDE_DIR}/*.h*)
add_library(${PROJECT_NAME} ${SGM_LIB_TYPE})
target_sources(${PROJECT_NAME} PRIVATE ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE ${LIBSGM_INCLUDE_DIR} $<$<BOOL:${BUILD_OPENCV_WRAPPER}>:${OpenCV_INCLUDE_DIRS}>)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(USE_HIP)
	# Compile the .cu (and .cpp, which include hip/hip_runtime.h) with the HIP
	# toolchain; the device runtime is implicit, no CUDA::cudart link needed.
	file(GLOB CU_SRCS ./*.cu)
	set_source_files_properties(${CU_SRCS} PROPERTIES LANGUAGE HIP)
	set_target_properties(${PROJECT_NAME} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}" POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP)
	target_link_libraries(${PROJECT_NAME} PUBLIC hip::host $<$<BOOL:${BUILD_OPENCV_WRAPPER}>:${OpenCV_LIBS}>)
else()
	target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart $<$<BOOL:${BUILD_OPENCV_WRAPPER}>:${OpenCV_LIBS}>)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${LIBSGM_INCLUDE_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

install(
	TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(
	DIRECTORY ${LIBSGM_INCLUDE_DIR}
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
	FILES ${LIBSGM_ROOT_DIR}/FindLibSGM.cmake
	DESTINATION ${CMAKE_INSTALL_PREFIX}
)
