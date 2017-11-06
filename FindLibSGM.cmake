###############################################################################
# Find LibSGM
#
# This sets the following variables:
# LIBSGM_FOUND - True if LIBSGM was found.
# LIBSGM_INCLUDE_DIRS - Directories containing the LIBSGM include files.
# LIBSGM_LIBRARY - Libraries needed to use LIBSGM.

# Find lib
set(LIBSGM_FOUND FALSE CACHE BOOL "" FORCE)
find_library(LIBSGM_LIBRARY
    NAMES sgm libsgm
    PATH_SUFFIXES lib/
)

# Find include
find_path(LIBSGM_INCLUDE_DIRS
    NAMES libsgm.h
    PATH_SUFFIXES include/
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibSGM DEFAULT_MSG LIBSGM_LIBRARY LIBSGM_INCLUDE_DIRS)

message(STATUS "(LIBSGM_FOUND : ${LIBSGM_FOUND} include: ${LIBSGM_INCLUDE_DIRS}, lib: ${LIBSGM_LIBRARY})")

mark_as_advanced(LIBSGM_FOUND)

if(LIBSGM_FOUND)
    set(LIBSGM_FOUND TRUE CACHE BOOL "" FORCE)
    set(LIBSGM_LIBRARIES ${LIBSGM_LIBRARY})
    message(STATUS "LibSGM found ( include: ${LIBSGM_INCLUDE_DIRS}, lib: ${LIBSGM_LIBRARY})")
endif()
