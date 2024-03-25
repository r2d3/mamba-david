cmake_minimum_required(VERSION 3.16)
project(mamba LANGUAGES CXX HIP)

# AMD XTX 7900: gfx1100
# AMD MI210: gfx908 ??
# AWS g4ad: gfx1011
# We could limit to one architecture for faster compile
set(CMAKE_HIP_ARCHITECTURES gfx1011;gfx1030;gfx1100)
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
find_package(PythonInterp REQUIRED)

# Find Python site-packages
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(PT "${PYTHON_SITE}/torch")
message(STATUS "Using torch library from: ${PT}")
include_directories(${PT}/include ${PT}/include/torch/csrc/api/include)
link_directories(${PT}/lib)

pybind11_add_module(mamba mamba_python.cpp)
target_link_libraries(mamba PRIVATE c10 c10_hip)
set_source_files_properties(mamba_python.cpp PROPERTIES LANGUAGE HIP)

pybind11_add_module(selective_scan_cuda selective_scan.cpp mamba.hip)
target_link_libraries(selective_scan_cuda PRIVATE c10 c10_hip torch torch_python)
set_source_files_properties(selective_scan.cpp PROPERTIES LANGUAGE HIP)
target_compile_definitions(selective_scan_cuda PRIVATE TORCH_EXTENSION_NAME=selective_scan_cuda)

add_executable(bug1 EXCLUDE_FROM_ALL bug1.hip)
target_link_libraries(bug1 PRIVATE hip::host)
