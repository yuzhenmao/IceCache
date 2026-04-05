cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(icecache_cpp LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(${CMAKE_SOURCE_DIR}/../../3rdparty/pybind11 ${CMAKE_BINARY_DIR}/pybind11)
file(GLOB CPP_SOURCES "src/*.cu")
pybind11_add_module(icecache_cpp MODULE ${PYTORCH_CPP_SOURCES} ${CPP_SOURCES})

target_include_directories(icecache_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/pybind11/include)
target_include_directories(icecache_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/flashinfer/include)
target_include_directories(icecache_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/cutlass/include)
target_include_directories(icecache_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/cutlass/tools/util/include)
target_compile_options(icecache_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_compile_definitions(icecache_cpp PRIVATE FLASHINFER_ENABLE_BF16)

set_target_properties(icecache_cpp
PROPERTIES CXX_STANDARD                        17
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       17
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_SOURCE_DIR}/../../3rdparty/raft/cpp/template/cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()
include(${CMAKE_SOURCE_DIR}/../../3rdparty/raft/cpp/template/cmake/thirdparty/get_raft.cmake)
           
target_link_libraries(icecache_cpp PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Python pybind11::module raft::raft)
