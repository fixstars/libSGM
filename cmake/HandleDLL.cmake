if(MSVC)
  set(dll_output_dir "${CMAKE_CURRENT_BINARY_DIR}/bin")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dll_output_dir}")

  # Introduce variables:
  # * CMAKE_INSTALL_LIBDIR
  # * CMAKE_INSTALL_BINDIR
  include(GNUInstallDirs)

  if(BUILD_SHARED_LIBS)
    set(binary_dst ${CMAKE_INSTALL_BINDIR})
  endif()
endif()