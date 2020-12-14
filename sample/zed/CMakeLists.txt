cmake_minimum_required(VERSION 3.18)
project(zed_demo)

find_package(CUDAToolkit REQUIRED)

find_package(OpenCV 3 REQUIRED)

find_package(ZED 3 REQUIRED)
if(ZED_FOUND)
    add_library(ZED INTERFACE)
    add_library(ZED::ZED ALIAS ZED)

    target_include_directories(ZED
                               INTERFACE
                               ${ZED_INCLUDE_DIRS}
                               ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(ZED
                          INTERFACE
                          ${OpenCV_LIBS}
                          ${ZED_LIBRARIES}
                          CUDA::cudart
                          CUDA::npps)

else()
    message("Install ZED SDK from https://www.stereolabs.com/developers/release/")
endif()

add_executable(${PROJECT_NAME} zed_demo.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE libsgm::sgm ZED::ZED)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
