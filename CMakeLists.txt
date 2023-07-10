cmake_minimum_required(VERSION 3.10)
project(bev-demo VERSION 0.0.1)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/app)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lstdc++fs -g -O0 -O3 -s -DNDEBUG")

find_package(OpenCV REQUIRED)
set(ENV_DIR $ENV{BEV_DIR})

include_directories(
    /usr/local/include
    ${ENV_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

link_directories(
    ${ENV_DIR}/lib
    /usr/local/lib)


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src sources)
add_executable(bev-demo main.cpp ${sources})

target_link_libraries(bev-demo  pthread stdc++fs -lglog ${OpenCV_LIBS})
install(TARGETS bev-demo DESTINATION .)
