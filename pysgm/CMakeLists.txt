find_package(Python COMPONENTS Development Interpreter)
find_package(pybind11 CONFIG)

pybind11_add_module(pysgm MODULE "pysgm.cpp")
target_include_directories(pysgm PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pysgm PRIVATE sgm)

if(BUILD_OPENCV_WRAPPER)
    target_link_libraries(pysgm PRIVATE ${OpenCV_LIBS})
endif()

install(
    TARGETS pysgm
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
)
