cmake_minimum_required(VERSION 3.18)
project(libsgm VERSION 2.7.0 LANGUAGES CXX CUDA)
# Set Libraries default visibility to hidden
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

###############################  CMake Modules  #############################################
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
###############################  CMake Modules  #############################################

###############################     Options       ###########################################
option(ENABLE_ZED_DEMO      "Build a Demo using ZED Camera" ON)
option(ENABLE_SAMPLES       "Build samples" ON)
option(ENABLE_TESTS         "Test library" OFF)
option(BUILD_OPENCV_WRAPPER "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" ON)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
###############################     Options       ###########################################

###############################     Modules     #############################################
include(HandlePDB)
include(HandleDLL)
###############################     Modules     #############################################

###############################     Package     #############################################
find_package(CUDAToolkit REQUIRED)

if(BUILD_OPENCV_WRAPPER)
    find_package(OpenCV COMPONENTS core REQUIRED)
endif()
###############################     Package     #############################################
###############################      SHARED       ###########################################
set(generated_headers "${CMAKE_CURRENT_BINARY_DIR}/generated_headers")
set(libsgm_export "${generated_headers}/libsgm_export.h")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/libsgm_config.h.in
               ${generated_headers}/libsgm_config.h)

add_library(${PROJECT_NAME})
add_library(${PROJECT_NAME}::sgm ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
               PRIVATE
               src/stereo_sgm.cpp
               src/libsgm_wrapper.cpp
               src/census_transform.cu
               src/correct_disparity_range.cu
               src/horizontal_path_aggregation.cu
               src/oblique_path_aggregation.cu
               src/sgm.cu
               src/winner_takes_all.cu
               src/check_consistency.cu
               src/cuda_utils.cu
               src/median_filter.cu
               src/path_aggregation.cu
               src/vertical_path_aggregation.cu)

target_include_directories(${PROJECT_NAME}
                           PUBLIC
                           $<BUILD_INTERFACE:${generated_headers}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart_static)

if(BUILD_OPENCV_WRAPPER)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME}
                       EXPORT_FILE_NAME ${libsgm_export}
                       EXPORT_MACRO_NAME  LIBSGM_API)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_PROPAGATE_HOST_FLAGS OFF)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_HOST_COMPILATION_CPP OFF)
###############################      SHARED       ###########################################
if(ENABLE_SAMPLES)
add_subdirectory(sample)
endif()
if(ENABLE_TESTS)
add_subdirectory(test)
endif()
###############################     INSTALL       ###########################################
include(GNUInstallDirs)

set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}-config-version.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}-config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "libsgm")

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_NAME}-config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Headers:
install (DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
# Export headers:
install (DIRECTORY "${generated_headers}/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

# install pdb-files
if(MSVC)
install(
      # this path is generated from HandlePDB.cmake
      DIRECTORY "${pdb_output_dir}/"
      DESTINATION ${pdb_dst})

install(
      # this path is generated from HandleDLL.cmake
      DIRECTORY "${dll_output_dir}/"
      DESTINATION ${binary_dst})

endif()
install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)
###############################     INSTALL       ###########################################