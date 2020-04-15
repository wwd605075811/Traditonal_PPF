cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ppf)

find_package(PCL 1.9 REQUIRED)

#[[SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")]]

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -O0 -g --std=c++11")
set(CMAKE_CUDA_NVCC_FLAGS "${CMAKE_CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ftz=true;--prec-div=false;--prec-sqrt=false;-arch=sm_75;--expt-extended-lambda)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DTHRUST_DEBUG;-g)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant")

#set(CMAKE_LIBRARY_PATH "/usr/local/cuda-10.2/lib64")
find_package(CUDA QUIET REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

cuda_add_executable (main src/main.cpp include/PPFMatch.h src/PPFMatch.cpp include/PPF.h src/PPF.cpp
        include/HCluster.h src/HCluster.cpp src/Model.cu include/Model.h include/Kernel.h src/Kernel.cu
        src/Scene.cu include/Scene.h src/PointCloudPretreatment.cpp include/PointCloudPretreatment.h)
target_link_libraries (main ${PCL_LIBRARIES})
