# SPDX-License-Identifier: Apache-2.0
#
# CMake build for libgfx906_kernels.so.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=$DTK_ROOT -DGX906_ARCH=gfx906
#   cmake --build build -j
#
# Requires a DTK/ROCm toolchain whose CMake package files provide `hip`.
# tools/build.sh is the dependency-free path used by CI and the test gate.
cmake_minimum_required(VERSION 3.20)
project(gfx906_kernels LANGUAGES CXX HIP)

set(GX906_ARCH "gfx906" CACHE STRING "AMDGPU target")
option(GX906_BUILD_TORCH_ADAPTER "Build the optional torch adapter" OFF)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "${GX906_ARCH}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB GX906_KERNEL_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.hip")

add_library(gfx906_kernels SHARED
  src/runtime.cpp
  src/capi/gfx906_capi.cpp
  ${GX906_KERNEL_SOURCES})

target_include_directories(gfx906_kernels
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor")

target_compile_definitions(gfx906_kernels PRIVATE USE_ROCM=1)

# gfx906 has no MFMA: keep the generated ISA in the GCN5.1 subset that the
# production vLLM build targets as well (same -O3, no fast-math).
# -mno-fma / -fno-gpu-rdc match the vLLM production compile line: with FMA
# contraction enabled the GDN state update differs by 1 ulp on a few elements.
target_compile_options(gfx906_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GX906_ARCH}>
  $<$<COMPILE_LANGUAGE:HIP>:-O3>
  $<$<COMPILE_LANGUAGE:HIP>:-mno-fma>
  $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>
  $<$<COMPILE_LANGUAGE:HIP>:-DHIP_ENABLE_WARP_SYNC_BUILTINS=1>
  $<$<COMPILE_LANGUAGE:HIP>:-DNDEBUG>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)

find_package(hip REQUIRED)
target_link_libraries(gfx906_kernels PRIVATE hip::device)

if(GX906_BUILD_TORCH_ADAPTER)
  find_package(Torch REQUIRED)
  add_library(gfx906_torch SHARED torch/gfx906_torch.cpp)
  target_include_directories(gfx906_torch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(gfx906_torch PRIVATE gfx906_kernels ${TORCH_LIBRARIES})
  set_target_properties(gfx906_torch PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
endif()
