cmake_minimum_required(VERSION 3.10)

add_executable(clustering_app
  ${CMAKE_CURRENT_LIST_DIR}/ClusteringApp.cpp
)

target_link_libraries(clustering_app
  PUBLIC
    ${LIBRARY_NAME}
)
