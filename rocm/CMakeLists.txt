cmake_minimum_required(VERSION 3.16)
project(mamba LANGUAGES CXX HIP)

set(CMAKE_HIP_ARCHITECTURES gfx1011)
set(CMAKE_CXX_STANDARD 20)

# Dependencies
include(FetchContent)

macro(fetch_dependency name repo tag)
  FetchContent_Declare(${name}
    GIT_REPOSITORY ${repo}
    GIT_TAG ${tag}
  )
  FetchContent_GetProperties(${name})
  if (NOT ${name}_POPULATED)
    message(STATUS "Populating ${name} dependencies")
    FetchContent_Populate(${name})
  endif()
endmacro()

fetch_dependency(pybind11 https://github.com/pybind/pybind11.git stable)
add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})

find_package(hip REQUIRED)

add_executable(test mamba.cpp)
target_link_libraries(test PRIVATE hip::device)
set_source_files_properties(mamba.cpp PROPERTIES LANGUAGE HIP)

pybind11_add_module(mamba mamba_python.cpp)
