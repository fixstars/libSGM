cmake_minimum_required(VERSION 3.18)
project(stereo_movie)

find_package(CUDA REQUIRED)

find_package(OpenCV 3 REQUIRED)

add_executable(${PROJECT_NAME} stereosgm_movie.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE libsgm::sgm ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)