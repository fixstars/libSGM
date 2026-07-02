cmake_minimum_required(VERSION 3.18)

if(USE_HIP)
	project(sgm-test LANGUAGES CXX)
else()
	project(sgm-test LANGUAGES CXX CUDA)
endif()

set(LIBSGM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

# required packages
if(USE_HIP)
	find_package(hip REQUIRED)
else()
	find_package(CUDAToolkit REQUIRED)
endif()
find_package(OpenCV REQUIRED)

if (MSVC)
	option(gtest_force_shared_crt "Force Gmock to use standard compiler flags" ON)
endif()

add_subdirectory(googletest)

file(GLOB SRCS ./*.cpp ./*.cu ./*.h*)
add_executable(sgm-test ${SRCS})

target_compile_features(sgm-test PRIVATE cxx_std_17)
target_include_directories(sgm-test PRIVATE ${LIBSGM_SOURCE_DIR} ${gtest_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sgm-test sgm gtest ${OpenCV_LIBS})

target_compile_options(
	sgm-test PRIVATE
	$<$<CXX_COMPILER_ID:GCC>:-O3 -Wall>
	$<$<CXX_COMPILER_ID:Clang>:-O3 -Wall>
	$<$<CXX_COMPILER_ID:MSVC>:/wd4819>
	$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
