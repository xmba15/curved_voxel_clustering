cmake_minimum_required(VERSION 3.10)

project(curved_voxel_clustering)

set (CMAKE_CXX_STANDARD 17)

set(LIBRARY_NAME ${PROJECT_NAME})

find_package(PCL REQUIRED)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)
