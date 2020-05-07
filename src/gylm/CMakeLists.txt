file(GLOB HEADERS *.hpp)
file(GLOB LOCAL_SOURCES *.cpp)

pybind11_add_module(_gylm ${LOCAL_SOURCES})
configure_file(GYLMRC.in GYLMRC @ONLY)

install(TARGETS _gylm LIBRARY DESTINATION ${LOCAL_INSTALL_DIR})
install(FILES __init__.py ${CMAKE_CURRENT_BINARY_DIR}/GYLMRC DESTINATION ${LOCAL_INSTALL_DIR})
install(FILES ${HEADERS} DESTINATION ${LOCAL_INSTALL_DIR}/include/gylm)

add_subdirectory(python)
