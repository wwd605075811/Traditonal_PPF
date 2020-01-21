cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ppf)

find_package(PCL 1.9 REQUIRED)

#set(CMAKE_LIBRARY_PATH "/usr/local/cuda-10.2/lib64")
find_package(CUDA QUIET REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

cuda_add_executable (main src/main.cpp src/SurfaceMatching.h src/SurfaceMatching.cpp src/PPF.h src/PPF.cpp src/stdafx.h src/kernel.cu src/kernel.cuh)
target_link_libraries (main ${PCL_LIBRARIES})
