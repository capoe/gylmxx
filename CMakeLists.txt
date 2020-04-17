# CONFIGURE CMAKE
message("CMake version: ${CMAKE_VERSION}")
cmake_minimum_required(VERSION 2.8.3)

# PROJECT OPTIONS
project(gylmxx)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(LOCAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/gylm)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDBG")

# BUILD OPTIONS
enable_language(CXX)
#set(CMAKE_CXX_COMPILER "/usr/local/shared/intel/compilers_and_libraries_2018/linux/bin/intel64/icc")
message("C++ compiler: " ${CMAKE_CXX_COMPILER} " " ${CMAKE_CXX_COMPILER_ID})
option(BUILD_SHARED_LIBS "Build shared libs" ON)
if(${CMAKE_VERSION} VERSION_GREATER 3.1)
    message("Setting C++ standard 11 (CMake version > 3.1)")
    set(CMAKE_CXX_STANDARD 11)
else(${CMAKE_VERSION} VERSION_GREATER 3.1)
    message("Setting C++ standard 11 (CMake version <= 3.1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(${CMAKE_VERSION} VERSION_GREATER 3.1)

# LOCAL PACKAGES
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# FIND PACKAGES
find_package(PythonLibs 3.6)
include_directories(${PYTHON_INCLUDE_DIRS})

set(GYLMXX_LINK_LIBRARIES ${PYTHON_LIBRARIES})

# SUMMARIZE INCLUDES & LIBS
get_property(local_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
get_property(local_libs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY GYLMXX_LINK_LIBRARIES)

message(STATUS "Include directories: ")
foreach(dir ${local_dirs})
  message(STATUS " o ${dir}")
endforeach()
message(STATUS "Linked libraries:    ")
foreach(lib ${local_dirs})
  message(STATUS " o ${lib}")
endforeach()

# SUBDIRECTORIES
add_subdirectory(pybind11)
add_subdirectory(src/gylm)
add_subdirectory(tools)

