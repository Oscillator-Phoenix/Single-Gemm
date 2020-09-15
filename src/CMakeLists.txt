cmake_minimum_required(VERSION 2.8)

# set the project name
project(main)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 添加子目录
add_subdirectory(gemm)


# 指定生成目标
add_executable(${PROJECT_NAME} main.cpp)


# 包含头文件
target_include_directories(${PROJECT_NAME} PUBLIC gemm)
target_include_directories(${PROJECT_NAME} PUBLIC utils)


# 添加链接库
target_link_libraries(${PROJECT_NAME} gemm)

message("==============================================================")
message("======================== Message  ============================")
message("CMake build HPC lab 1")
message("project name: " ${PROJECT_NAME})
message("source path: " ${PROJECT_SOURCE_DIR})
message("build path: " ${PROJECT_BINARY_DIR})
message("C++ standard: C++" ${CMAKE_CXX_STANDARD})
message("==============================================================")
message("==============================================================")
