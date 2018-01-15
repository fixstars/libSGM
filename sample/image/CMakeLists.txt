find_package(CUDA REQUIRED)

# test apps
find_package(OpenCV REQUIRED core) # dammy
if(OpenCV_VERSION VERSION_LESS 3.0)
	find_package(OpenCV REQUIRED core highgui imgproc contrib)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
# end test apps

include_directories(../../include)

CUDA_ADD_EXECUTABLE(stereo_test stereosgm_image.cpp ${CUDA_SRC})

if(CUDA_VERSION VERSION_LESS "9.0")
	set(NPP_LIBRARIES "${CUDA_nppi_LIBRARY}")
else()
	set(NPP_LIBRARIES "${CUDA_nppif_LIBRARY}")
endif()

TARGET_LINK_LIBRARIES(stereo_test sgm ${CUDA_LIBRARIES} ${NPP_LIBRARIES} ${OpenCV_LIBS})
