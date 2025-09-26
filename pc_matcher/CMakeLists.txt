cmake_minimum_required(VERSION 3.10.0)
project(pc_matcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)

add_definitions(${PCL_DEFINITIONS})

add_library(pc_matcher_lib SHARED
    src/optimizer.cpp
    src/matcher.cpp
)

link_libraries(
    pc_matcher_lib
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(CONFIG_FILE_ABSOLUTE_PATH ${CMAKE_SOURCE_DIR}/config/config.toml)

add_executable(main src/main.cpp)

target_compile_definitions(main PRIVATE 
    CONFIG_FILE_PATH="${CONFIG_FILE_ABSOLUTE_PATH}"
)