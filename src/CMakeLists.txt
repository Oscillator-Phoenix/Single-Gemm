cmake_minimum_required(VERSION 2.8)

# set the project name
project(main)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 添加子目录
add_subdirectory(gemm)
add_subdirectory(sparse)



# 指定生成目标
add_executable(${PROJECT_NAME} main.cpp)


# 包含头文件
target_include_directories(${PROJECT_NAME} PUBLIC gemm)
target_include_directories(${PROJECT_NAME} PUBLIC utils)
target_include_directories(${PROJECT_NAME} PUBLIC sparse)



# 添加链接库
target_link_libraries(${PROJECT_NAME} gemm)
target_link_libraries(${PROJECT_NAME} sparse)



# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
                       -O3
                       >)


# 显示 make 编译命令
# set(CMAKE_VERBOSE_MAKEFILE ON)


message("==============================================================")
message("======================== Message  ============================")
message("CMake build HPC lab 1")
message("project name: " ${PROJECT_NAME})
message("source path: " ${PROJECT_SOURCE_DIR})
message("build path: " ${PROJECT_BINARY_DIR})
message("C++ standard: C++" ${CMAKE_CXX_STANDARD})
message("==============================================================")
message("==============================================================")
