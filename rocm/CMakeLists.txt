cmake_minimum_required(VERSION 3.16)
project(mamba LANGUAGES CXX HIP)

set(CMAKE_HIP_ARCHITECTURES gfx1011)
set(CMAKE_CXX_STANDARD 17)

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

set(PT "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch")
include_directories(${PT}/include ${PT}/include/torch/csrc/api/include)
link_directories(${PT}/lib)

add_executable(test mamba.cpp)
target_link_libraries(test PRIVATE c10 torch hip::device)
set_source_files_properties(mamba.cpp PROPERTIES LANGUAGE HIP)

pybind11_add_module(mamba_ssm mamba_python.cpp)
target_link_libraries(mamba_ssm PRIVATE c10 torch)
set_source_files_properties(mamba_python.cpp PROPERTIES LANGUAGE HIP)

add_executable(bug1 EXCLUDE_FROM_ALL bug1.hip)
target_link_libraries(bug1 PRIVATE hip::host)
