find_package(pybind11 REQUIRED)

pybind11_add_module(pysgm MODULE "pysgm.cpp")
target_include_directories(pysgm PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pysgm PRIVATE sgm)

if(BUILD_OPENCV_WRAPPER)
    target_link_libraries(pysgm PRIVATE ${OpenCV_LIBS})
endif()

set_target_properties(pysgm
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pysgm"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pysgm"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pysgm"
        )
